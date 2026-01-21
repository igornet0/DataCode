/// Константное сворачивание (constant folding) - вычисление константных выражений во время компиляции

use crate::parser::ast::Expr;
use crate::common::error::LangError;
use crate::common::value::Value;
use crate::lexer::TokenKind;

/// Оптимизация: вычисляет константные выражения во время компиляции
pub fn evaluate_constant_expr(expr: &Expr) -> Result<Option<Value>, LangError> {
    match expr {
        Expr::Literal { value, .. } => Ok(Some(value.clone())),
        Expr::ArrayLiteral { .. } => Ok(None), // Не можем вычислить во время компиляции
        Expr::ObjectLiteral { .. } => Ok(None), // Не можем вычислить во время компиляции
        Expr::TupleLiteral { .. } => Ok(None), // Не можем вычислить во время компиляции
        Expr::Property { .. } => Ok(None), // Не можем вычислить во время компиляции
        Expr::MethodCall { .. } => Ok(None), // Не можем вычислить во время компиляции
        Expr::Binary { left, op, right, .. } => {
            // Пытаемся вычислить бинарное выражение, если оба операнда константы
            let left_val = evaluate_constant_expr(left)?;
            let right_val = evaluate_constant_expr(right)?;
            
            if let (Some(l), Some(r)) = (left_val, right_val) {
                match op {
                    TokenKind::Plus => {
                        match (l, r) {
                            (Value::Number(n1), Value::Number(n2)) => Ok(Some(Value::Number(n1 + n2))),
                            (Value::String(s1), Value::String(s2)) => Ok(Some(Value::String(format!("{}{}", s1, s2)))),
                            (Value::String(s), Value::Number(n)) => Ok(Some(Value::String(format!("{}{}", s, n)))),
                            (Value::Number(n), Value::String(s)) => Ok(Some(Value::String(format!("{}{}", n, s)))),
                            _ => Ok(None),
                        }
                    }
                    TokenKind::Minus => {
                        if let (Value::Number(n1), Value::Number(n2)) = (l, r) {
                            Ok(Some(Value::Number(n1 - n2)))
                        } else {
                            Ok(None)
                        }
                    }
                    TokenKind::Star => {
                        match (&l, &r) {
                            (Value::Number(n1), Value::Number(n2)) => Ok(Some(Value::Number(n1 * n2))),
                            (Value::String(s), Value::Number(n)) => {
                                let count = *n as i64;
                                if count <= 0 {
                                    Ok(Some(Value::String(String::new())))
                                } else {
                                    Ok(Some(Value::String(s.repeat(count as usize))))
                                }
                            }
                            (Value::Number(n), Value::String(s)) => {
                                let count = *n as i64;
                                if count <= 0 {
                                    Ok(Some(Value::String(String::new())))
                                } else {
                                    Ok(Some(Value::String(s.repeat(count as usize))))
                                }
                            }
                            _ => Ok(None),
                        }
                    }
                    TokenKind::Slash => {
                        if let (Value::Number(n1), Value::Number(n2)) = (l, r) {
                            if n2 == 0.0 {
                                // Don't constant-fold division by zero - let it be a runtime error
                                // so we can provide proper stack traces
                                return Ok(None);
                            }
                            Ok(Some(Value::Number(n1 / n2)))
                        } else {
                            Ok(None)
                        }
                    }
                    TokenKind::SlashSlash => {
                        if let (Value::Number(n1), Value::Number(n2)) = (l, r) {
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
                    TokenKind::Greater => {
                        if let (Value::Number(n1), Value::Number(n2)) = (l, r) {
                            Ok(Some(Value::Bool(n1 > n2)))
                        } else {
                            Ok(None)
                        }
                    }
                    TokenKind::Less => {
                        if let (Value::Number(n1), Value::Number(n2)) = (l, r) {
                            Ok(Some(Value::Bool(n1 < n2)))
                        } else {
                            Ok(None)
                        }
                    }
                    TokenKind::GreaterEqual => {
                        if let (Value::Number(n1), Value::Number(n2)) = (l, r) {
                            Ok(Some(Value::Bool(n1 >= n2)))
                        } else {
                            Ok(None)
                        }
                    }
                    TokenKind::LessEqual => {
                        if let (Value::Number(n1), Value::Number(n2)) = (l, r) {
                            Ok(Some(Value::Bool(n1 <= n2)))
                        } else {
                            Ok(None)
                        }
                    }
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
                    TokenKind::Minus => {
                        if let Value::Number(n) = r {
                            Ok(Some(Value::Number(-n)))
                        } else {
                            Ok(None)
                        }
                    }
                    TokenKind::Bang => {
                        Ok(Some(Value::Bool(!r.is_truthy())))
                    }
                    _ => Ok(None),
                }
            } else {
                Ok(None)
            }
        }
        _ => Ok(None), // Переменные, вызовы функций и присваивания не могут быть вычислены во время компиляции
    }
}


