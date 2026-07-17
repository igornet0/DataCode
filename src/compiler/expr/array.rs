use crate::bytecode::OpCode;
use crate::common::error::LangError;
use crate::common::value::Value;
use crate::compiler::context::CompilationContext;
use crate::compiler::expr;
use crate::compiler::expr::object_literal;
/// Компиляция массивов, кортежей и индексации
use crate::parser::ast::{Expr, IndexExpr};

pub fn compile_array(ctx: &mut CompilationContext, expr: &Expr) -> Result<(), LangError> {
    match expr {
        Expr::ArrayLiteral { elements, line } => {
            *ctx.current_line = *line;
            // Компилируем каждый элемент массива
            for element in elements {
                expr::compile_expr(ctx, element)?;
            }
            // Создаем массив из элементов на стеке
            let arity = elements.len();
            ctx.chunk.write_with_line(OpCode::MakeArray(arity), *line);
            Ok(())
        }
        Expr::TupleLiteral { elements, line } => {
            *ctx.current_line = *line;
            // Компилируем каждый элемент кортежа
            for element in elements {
                expr::compile_expr(ctx, element)?;
            }
            // Создаем кортеж из элементов на стеке
            // Используем MakeArray, но в VM будем создавать Tuple
            let arity = elements.len();
            ctx.chunk.write_with_line(OpCode::MakeTuple(arity), *line);
            Ok(())
        }
        Expr::ObjectLiteral { pairs, line } => {
            object_literal::compile_object_literal(ctx, pairs, *line)
        }
        Expr::ArrayIndex { array, index, line } => {
            *ctx.current_line = *line;
            expr::compile_expr(ctx, array)?;
            match index {
                IndexExpr::Scalar(e) => {
                    expr::compile_expr(ctx, e)?;
                    ctx.chunk
                        .write_with_line(OpCode::ObjectIndexIntegral, *line);
                }
                IndexExpr::Slice {
                    start,
                    stop,
                    step,
                    line: sl,
                } => {
                    emit_slice_bound(ctx, start.as_deref(), *sl)?;
                    emit_slice_bound(ctx, stop.as_deref(), *sl)?;
                    emit_slice_bound(ctx, step.as_deref(), *sl)?;
                    ctx.chunk.write_with_line(OpCode::GetArraySlice, *sl);
                }
            }
            Ok(())
        }
        Expr::TableFilter {
            table, predicate, line,
        } => crate::compiler::expr::table_filter::compile_table_filter(ctx, table, predicate, *line),
        _ => Err(LangError::ParseError {
            message: "Expected ArrayLiteral, TupleLiteral, ObjectLiteral, ArrayIndex, or TableFilter expression".to_string(),
            line: expr.line(),
            file: None,
        }),
    }
}

pub(crate) fn emit_slice_bound(
    ctx: &mut CompilationContext,
    bound: Option<&Expr>,
    line: usize,
) -> Result<(), LangError> {
    if let Some(e) = bound {
        expr::compile_expr(ctx, e)?;
    } else {
        let idx = ctx.chunk.add_constant(Value::Null);
        ctx.chunk.write_with_line(OpCode::Constant(idx), line);
    }
    Ok(())
}
