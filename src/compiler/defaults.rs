//! Compile-time resolution of default parameter values (literals + module constants).

use std::collections::HashMap;

use crate::common::error::LangError;
use crate::common::numeric::IntValue;
use crate::common::value::Value;
use crate::compiler::constant_fold;
use crate::lexer::TokenKind;
use crate::parser::ast::{BinaryOpKind, Expr, IfBranch};

fn finite_ieee_pair(l: &Value, r: &Value) -> Option<(f64, f64)> {
    let x = l.as_ieee_f64()?;
    let y = r.as_ieee_f64()?;
    (x.is_finite() && y.is_finite()).then_some((x, y))
}

fn value_as_bitwise_int(v: &Value) -> Option<i64> {
    match v {
        Value::Int(IntValue::Finite(n)) => Some(*n),
        Value::Number(n) if n.is_finite() && n.fract() == 0.0 => {
            Some(crate::common::numeric::f64_trunc_to_i64_clamped(*n))
        }
        _ => None,
    }
}

fn int_pair(l: &Value, r: &Value) -> Option<(i64, i64)> {
    Some((value_as_bitwise_int(l)?, value_as_bitwise_int(r)?))
}

fn shift_amount(b: i64) -> Option<u32> {
    if b < 0 {
        return None;
    }
    Some((b as u32).min(63))
}

fn int_result(n: i64) -> Value {
    Value::Int(IntValue::Finite(n))
}

fn numeric_string_concat(n: &Value) -> Option<String> {
    let x = n.as_ieee_f64()?;
    if x.is_nan() || x.is_infinite() {
        return None;
    }
    Some(x.to_string())
}

/// Evaluate an expression using module-level compile-time bindings (and literals).
pub fn evaluate_compile_time_expr(
    expr: &Expr,
    bindings: &HashMap<String, Value>,
) -> Result<Option<Value>, LangError> {
    if let Ok(Some(v)) = constant_fold::evaluate_constant_expr(expr) {
        return Ok(Some(v));
    }

    match expr {
        Expr::Variable { name, .. } => Ok(bindings.get(name).cloned()),
        Expr::Binary {
            left, op, right, ..
        } => {
            let left_val = evaluate_compile_time_expr(left, bindings)?;
            let right_val = evaluate_compile_time_expr(right, bindings)?;
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
                    TokenKind::Greater => Ok(finite_ieee_pair(&l, &r).map(|(a, b)| Value::Bool(a > b))),
                    TokenKind::Less => Ok(finite_ieee_pair(&l, &r).map(|(a, b)| Value::Bool(a < b))),
                    TokenKind::GreaterEqual => {
                        Ok(finite_ieee_pair(&l, &r).map(|(a, b)| Value::Bool(a >= b)))
                    }
                    TokenKind::LessEqual => {
                        Ok(finite_ieee_pair(&l, &r).map(|(a, b)| Value::Bool(a <= b)))
                    }
                    TokenKind::Amp => Ok(int_pair(&l, &r).map(|(a, b)| int_result(a & b))),
                    TokenKind::Pipe => Ok(int_pair(&l, &r).map(|(a, b)| int_result(a | b))),
                    TokenKind::Caret => Ok(int_pair(&l, &r).map(|(a, b)| int_result(a ^ b))),
                    TokenKind::LessLess => Ok(int_pair(&l, &r).and_then(|(a, b)| {
                        shift_amount(b).map(|s| int_result(a.wrapping_shl(s)))
                    })),
                    TokenKind::GreaterGreater => Ok(int_pair(&l, &r).and_then(|(a, b)| {
                        shift_amount(b).map(|s| int_result(a >> s))
                    })),
                    _ => Ok(None),
                }
            } else {
                Ok(None)
            }
        }
        Expr::Unary { op, right, .. } => {
            let right_val = evaluate_compile_time_expr(right, bindings)?;
            if let Some(r) = right_val {
                match op {
                    TokenKind::Minus => Ok(Some(match r {
                        Value::Int(iv) => Value::Int(iv.neg()),
                        Value::Float(fv) => Value::Float(fv.neg()),
                        _ => return Ok(None),
                    })),
                    TokenKind::Bang => Ok(Some(Value::Bool(!r.is_truthy()))),
                    TokenKind::Tilde => Ok(value_as_bitwise_int(&r).map(|n| int_result(!n))),
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
            let cond_val = evaluate_compile_time_expr(condition, bindings)?;
            if let Some(Value::Bool(b)) = cond_val {
                let branch = if b { then_branch } else { else_branch };
                match branch {
                    IfBranch::Expr(e) => evaluate_compile_time_expr(e, bindings),
                    IfBranch::Block(_) => Ok(None),
                }
            } else {
                Ok(None)
            }
        }
        _ => Ok(None),
    }
}

fn check_no_param_refs(expr: &Expr, param_names: &[String], line: usize, file: Option<&str>) -> Result<(), LangError> {
    let mut names = Vec::new();
    collect_variable_refs(expr, &mut names);
    for name in names {
        if param_names.iter().any(|p| p == &name) {
            return Err(LangError::SemanticError {
                message: format!(
                    "Default value cannot reference parameter '{}' of the same signature",
                    name
                ),
                line,
                file: file.map(String::from),
            });
        }
    }
    Ok(())
}

fn collect_variable_refs(expr: &Expr, out: &mut Vec<String>) {
    match expr {
        Expr::Variable { name, .. } => out.push(name.clone()),
        Expr::Binary { left, right, .. } => {
            collect_variable_refs(left, out);
            collect_variable_refs(right, out);
        }
        Expr::Unary { right, .. } => collect_variable_refs(right, out),
        Expr::If {
            condition,
            then_branch,
            else_branch,
            ..
        } => {
            collect_variable_refs(condition, out);
            match then_branch {
                IfBranch::Expr(e) => collect_variable_refs(e, out),
                IfBranch::Block(stmts) => {
                    for s in stmts {
                        collect_stmt_variable_refs(s, out);
                    }
                }
            }
            match else_branch {
                IfBranch::Expr(e) => collect_variable_refs(e, out),
                IfBranch::Block(stmts) => {
                    for s in stmts {
                        collect_stmt_variable_refs(s, out);
                    }
                }
            }
        }
        _ => {}
    }
}

fn collect_stmt_variable_refs(stmt: &crate::parser::ast::Stmt, out: &mut Vec<String>) {
    use crate::parser::ast::Stmt;
    match stmt {
        Stmt::Expr { expr, .. } => collect_variable_refs(expr, out),
        Stmt::Let { value, .. } => collect_variable_refs(value, out),
        Stmt::Return { value, .. } => {
            if let Some(v) = value {
                collect_variable_refs(v, out);
            }
        }
        _ => {}
    }
}

/// Resolve a default parameter expression to a frozen [`Value`] (Python-style: fixed at definition compile time).
pub fn resolve_default_param_value(
    expr: &Expr,
    param_name: &str,
    param_names: &[String],
    bindings: &HashMap<String, Value>,
    file: Option<&str>,
) -> Result<Value, LangError> {
    check_no_param_refs(expr, param_names, expr.line(), file)?;

    match evaluate_compile_time_expr(expr, bindings)? {
        Some(v) => Ok(v),
        None => {
            let hint = if let Expr::Variable { name, .. } = expr {
                format!("; '{}' is not defined or not constant at this point", name)
            } else {
                String::new()
            };
            Err(LangError::SemanticError {
                message: format!(
                    "Default value for parameter '{}' must be a compile-time constant{}",
                    param_name, hint
                ),
                line: expr.line(),
                file: file.map(String::from),
            })
        }
    }
}

/// Update module-level compile-time binding after a top-level assignment.
pub fn update_compile_time_binding(
    bindings: &mut HashMap<String, Value>,
    name: &str,
    expr: &Expr,
) {
    match evaluate_compile_time_expr(expr, bindings) {
        Ok(Some(v)) => {
            bindings.insert(name.to_string(), v);
        }
        _ => {
            bindings.remove(name);
        }
    }
}
