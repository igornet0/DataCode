//! Extract table filter predicate trees from bracket expressions.

use crate::common::value::Value;
use crate::lexer::TokenKind;
use crate::parser::ast::{Arg, BinaryOpKind, Expr, StringMatchOp, TableFilterPred};

fn is_comparison_op(op: &BinaryOpKind) -> bool {
    match op {
        BinaryOpKind::Builtin(tk) => matches!(
            tk,
            TokenKind::Equal
                | TokenKind::EqualEqual
                | TokenKind::BangEqual
                | TokenKind::Less
                | TokenKind::Greater
                | TokenKind::LessEqual
                | TokenKind::GreaterEqual
        ),
        BinaryOpKind::Plugin { .. } => false,
    }
}

fn try_extract_string_match_call(expr: &Expr) -> Option<(StringMatchOp, Box<Expr>)> {
    let Expr::Call { name, args, .. } = expr else {
        return None;
    };
    if args.len() != 1 {
        return None;
    }
    let Arg::Positional(pattern) = &args[0] else {
        return None;
    };
    let op = StringMatchOp::from_call_name(name)?;
    Some((op, Box::new(pattern.clone())))
}

/// Parse `expr` as a table filter predicate tree (`col op val`, `and`, `or`).
pub fn try_extract_table_filter_pred(expr: &Expr) -> Option<TableFilterPred> {
    match expr {
        Expr::Binary {
            left,
            op: BinaryOpKind::Builtin(TokenKind::And),
            right,
            ..
        } => Some(TableFilterPred::And(
            Box::new(try_extract_table_filter_pred(left)?),
            Box::new(try_extract_table_filter_pred(right)?),
        )),
        Expr::Binary {
            left,
            op: BinaryOpKind::Builtin(TokenKind::Or),
            right,
            ..
        } => Some(TableFilterPred::Or(
            Box::new(try_extract_table_filter_pred(left)?),
            Box::new(try_extract_table_filter_pred(right)?),
        )),
        Expr::Binary {
            left,
            op: BinaryOpKind::Builtin(TokenKind::In),
            right,
            ..
        } => {
            if let Expr::Literal {
                value: Value::String(column),
                ..
            } = left.as_ref()
            {
                return Some(TableFilterPred::Membership {
                    column: column.clone(),
                    container: right.clone(),
                    negate: false,
                });
            }
            None
        }
        Expr::Unary {
            op: TokenKind::Bang,
            right,
            ..
        } => {
            if let Expr::Binary {
                left,
                op: BinaryOpKind::Builtin(TokenKind::In),
                right: container,
                ..
            } = right.as_ref()
            {
                if let Expr::Literal {
                    value: Value::String(column),
                    ..
                } = left.as_ref()
                {
                    return Some(TableFilterPred::Membership {
                        column: column.clone(),
                        container: container.clone(),
                        negate: true,
                    });
                }
            }
            None
        }
        Expr::Binary {
            left,
            op: BinaryOpKind::Builtin(TokenKind::Amp),
            right,
            ..
        } => {
            if let Expr::Literal {
                value: Value::String(column),
                ..
            } = left.as_ref()
            {
                let (op, pattern) = try_extract_string_match_call(right)?;
                return Some(TableFilterPred::StringMatch {
                    column: column.clone(),
                    op,
                    pattern,
                });
            }
            None
        }
        Expr::Binary {
            left,
            op,
            right,
            ..
        } if is_comparison_op(op) => {
            if let Expr::Literal {
                value: Value::String(column),
                ..
            } = left.as_ref()
            {
                if let BinaryOpKind::Builtin(op_tk) = op {
                    return Some(TableFilterPred::Compare {
                        column: column.clone(),
                        op: op_tk.clone(),
                        value: right.clone(),
                    });
                }
            }
            None
        }
        _ => None,
    }
}
