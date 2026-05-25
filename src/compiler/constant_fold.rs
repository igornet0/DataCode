use crate::common::error::LangError;
use crate::common::value::Value;
use crate::lexer::TokenKind;
/// Константное сворачивание (constant folding) - вычисление константных выражений во время компиляции
use crate::parser::ast::{BinaryOpKind, Expr, IfBranch};

/// Only finite IEEE pairs participate in arithmetic / ordered-compare folding.
fn finite_ieee_pair(l: &Value, r: &Value) -> Option<(f64, f64)> {
    let x = l.as_ieee_f64()?;
    let y = r.as_ieee_f64()?;
    (x.is_finite() && y.is_finite()).then_some((x, y))
}

fn numeric_string_concat(n: &Value) -> Option<String> {
    let x = n.as_ieee_f64()?;
    if x.is_nan() || x.is_infinite() {
        return None;
    }
    Some(x.to_string())
}

/// Оптимизация: вычисляет константные выражения во время компиляции
pub fn evaluate_constant_expr(expr: &Expr) -> Result<Option<Value>, LangError> {
    match expr {
        Expr::Literal { value, .. } => Ok(Some(value.clone())),
        Expr::ArrayLiteral { .. } => Ok(None),
        Expr::ObjectLiteral { .. } | Expr::DictComprehension { .. } | Expr::ListComprehension { .. } => Ok(None),
        Expr::TupleLiteral { .. } => Ok(None),
        Expr::Property { .. } => Ok(None),
        Expr::MethodCall { .. } => Ok(None),
        Expr::InterpolatedString { .. } => Ok(None),
        Expr::Binary {
            left, op, right, ..
        } => {
            let left_val = evaluate_constant_expr(left)?;
            let right_val = evaluate_constant_expr(right)?;

            if let (Some(l), Some(r)) = (left_val, right_val) {
                let op = match op {
                    BinaryOpKind::Builtin(t) => t,
                    BinaryOpKind::Plugin { .. } => return Ok(None),
                };
                match op {
                    TokenKind::Plus => {
                        if let Some((n1, n2)) = finite_ieee_pair(&l, &r) {
                            return Ok(Some(Value::Number(n1 + n2)));
                        }
                        if let (Value::String(s1), Value::String(s2)) = (&l, &r) {
                            return Ok(Some(Value::String(format!("{}{}", s1, s2))));
                        }
                        if let Value::String(s) = &l {
                            if let Some(ns) = numeric_string_concat(&r) {
                                return Ok(Some(Value::String(format!("{}{}", s, ns))));
                            }
                        }
                        if let Value::String(s) = &r {
                            if let Some(ns) = numeric_string_concat(&l) {
                                return Ok(Some(Value::String(format!("{}{}", ns, s))));
                            }
                        }
                        Ok(None)
                    }
                    TokenKind::Minus => {
                        if let Some((n1, n2)) = finite_ieee_pair(&l, &r) {
                            Ok(Some(Value::Number(n1 - n2)))
                        } else {
                            Ok(None)
                        }
                    }
                    TokenKind::Star => {
                        if let Some((n1, n2)) = finite_ieee_pair(&l, &r) {
                            return Ok(Some(Value::Number(n1 * n2)));
                        }
                        if let Value::String(s) = &l {
                            if let Some(ix) = r.as_finite_f64() {
                                let count = ix as i64;
                                return Ok(Some(if count <= 0 {
                                    Value::String(String::new())
                                } else {
                                    Value::String(s.repeat(count as usize))
                                }));
                            }
                        }
                        if let Value::String(s) = &r {
                            if let Some(ix) = l.as_finite_f64() {
                                let count = ix as i64;
                                return Ok(Some(if count <= 0 {
                                    Value::String(String::new())
                                } else {
                                    Value::String(s.repeat(count as usize))
                                }));
                            }
                        }
                        Ok(None)
                    }
                    TokenKind::Slash => {
                        if let Some((n1, n2)) = finite_ieee_pair(&l, &r) {
                            if n2 == 0.0 {
                                return Ok(None);
                            }
                            Ok(Some(Value::Number(n1 / n2)))
                        } else {
                            Ok(None)
                        }
                    }
                    TokenKind::SlashSlash => {
                        if let Some((n1, n2)) = finite_ieee_pair(&l, &r) {
                            if n2 == 0.0 {
                                return Ok(None);
                            }
                            Ok(Some(Value::Number((n1 / n2).floor())))
                        } else {
                            Ok(None)
                        }
                    }
                    TokenKind::EqualEqual => Ok(Some(Value::Bool(l == r))),
                    TokenKind::BangEqual => Ok(Some(Value::Bool(l != r))),
                    TokenKind::Greater => Ok(Some(if let Some((n1, n2)) =
                        finite_ieee_pair(&l, &r)
                    {
                        Value::Bool(n1 > n2)
                    } else {
                        return Ok(None);
                    })),
                    TokenKind::Less => Ok(Some(if let Some((n1, n2)) = finite_ieee_pair(&l, &r) {
                        Value::Bool(n1 < n2)
                    } else {
                        return Ok(None);
                    })),
                    TokenKind::GreaterEqual => Ok(Some(if let Some((n1, n2)) =
                        finite_ieee_pair(&l, &r)
                    {
                        Value::Bool(n1 >= n2)
                    } else {
                        return Ok(None);
                    })),
                    TokenKind::LessEqual => Ok(Some(if let Some((n1, n2)) =
                        finite_ieee_pair(&l, &r)
                    {
                        Value::Bool(n1 <= n2)
                    } else {
                        return Ok(None);
                    })),
                    _ => Ok(None),
                }
            } else {
                Ok(None)
            }
        }
        Expr::Unary { op, right, .. } => {
            let right_val = evaluate_constant_expr(right)?;
            if let Some(r) = right_val {
                match op {
                    TokenKind::Minus => Ok(Some(match r {
                        Value::Int(iv) => Value::Int(iv.neg()),
                        Value::Float(fv) => Value::Float(fv.neg()),
                        _ => return Ok(None),
                    })),
                    TokenKind::Bang => Ok(Some(Value::Bool(!r.is_truthy()))),
                    _ => Ok(None),
                }
            } else {
                Ok(None)
            }
        }
        Expr::If {
            condition,
            then_branch,
            else_branch,
            ..
        } => {
            let cond_val = evaluate_constant_expr(condition)?;
            if let Some(Value::Bool(b)) = cond_val {
                let branch = if b { then_branch } else { else_branch };
                match branch {
                    IfBranch::Expr(e) => evaluate_constant_expr(e),
                    IfBranch::Block(_) => Ok(None),
                }
            } else {
                Ok(None)
            }
        }
        _ => Ok(None),
    }
}
