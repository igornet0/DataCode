//! Build runtime predicate constants and compile table filter expressions.

use crate::common::value::Value;
use crate::compiler::context::CompilationContext;
use crate::compiler::expr;
use crate::lexer::TokenKind;
use crate::parser::ast::{Expr, TableFilterPred};
use std::cell::RefCell;
use std::rc::Rc;

fn token_to_op_str(op: &TokenKind) -> &'static str {
    match op {
        TokenKind::Equal => "=",
        TokenKind::EqualEqual => "==",
        TokenKind::BangEqual => "!=",
        TokenKind::Less => "<",
        TokenKind::Greater => ">",
        TokenKind::LessEqual => "<=",
        TokenKind::GreaterEqual => ">=",
        _ => "==",
    }
}

fn build_pred_value(pred: &TableFilterPred, value_index: &mut usize) -> Value {
    match pred {
        TableFilterPred::Compare { column, op, .. } => {
            let vi = *value_index;
            *value_index += 1;
            Value::Array(Rc::new(RefCell::new(vec![
                Value::String("cmp".to_string()),
                Value::String(column.clone()),
                Value::String(token_to_op_str(op).to_string()),
                Value::Number(vi as f64),
            ])))
        }
        TableFilterPred::Membership {
            column,
            negate,
            ..
        } => {
            let vi = *value_index;
            *value_index += 1;
            Value::Array(Rc::new(RefCell::new(vec![
                Value::String("member".to_string()),
                Value::String(column.clone()),
                Value::Number(vi as f64),
                Value::Number(if *negate { 1.0 } else { 0.0 }),
            ])))
        }
        TableFilterPred::StringMatch {
            column,
            op,
            ..
        } => {
            let vi = *value_index;
            *value_index += 1;
            Value::Array(Rc::new(RefCell::new(vec![
                Value::String("str".to_string()),
                Value::String(column.clone()),
                Value::String(op.as_str().to_string()),
                Value::Number(vi as f64),
            ])))
        }
        TableFilterPred::And(l, r) => Value::Array(Rc::new(RefCell::new(vec![
            Value::String("and".to_string()),
            build_pred_value(l, value_index),
            build_pred_value(r, value_index),
        ]))),
        TableFilterPred::Or(l, r) => Value::Array(Rc::new(RefCell::new(vec![
            Value::String("or".to_string()),
            build_pred_value(l, value_index),
            build_pred_value(r, value_index),
        ]))),
    }
}

fn collect_value_exprs(pred: &TableFilterPred, out: &mut Vec<Box<Expr>>) {
    match pred {
        TableFilterPred::Compare { value, .. }
        | TableFilterPred::Membership { container: value, .. }
        | TableFilterPred::StringMatch { pattern: value, .. } => {
            out.push(value.clone())
        }
        TableFilterPred::And(l, r) | TableFilterPred::Or(l, r) => {
            collect_value_exprs(l, out);
            collect_value_exprs(r, out);
        }
    }
}

pub(crate) fn compile_table_filter(
    ctx: &mut CompilationContext,
    table: &Expr,
    predicate: &TableFilterPred,
    line: usize,
) -> Result<(), crate::common::error::LangError> {
    let mut vi = 0usize;
    let pred_value = build_pred_value(predicate, &mut vi);
    let pred_index = ctx.chunk.add_constant(pred_value);

    let mut value_exprs: Vec<Box<Expr>> = Vec::new();
    collect_value_exprs(predicate, &mut value_exprs);

    *ctx.current_line = line;
    expr::compile_expr(ctx, table)?;

    let n = value_exprs.len();
    for e in value_exprs {
        expr::compile_expr(ctx, &e)?;
    }
    if n > 0 {
        ctx.chunk.write_with_line(crate::bytecode::OpCode::MakeArray(n), line);
    } else {
        let empty = ctx.chunk.add_constant(Value::Array(Rc::new(RefCell::new(vec![]))));
        ctx.chunk.write_with_line(crate::bytecode::OpCode::Constant(empty), line);
    }

    ctx.chunk
        .write_with_line(crate::bytecode::OpCode::TableFilterPred(pred_index), line);
    Ok(())
}
