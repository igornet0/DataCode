//! Shared compile-time helpers for integral-key dict/set/heap peephole opcodes.

use crate::bytecode::OpCode;
use crate::common::error::LangError;
use crate::common::numeric::{FloatValue, IntValue};
use crate::common::value::Value;
use crate::compiler::context::CompilationContext;
use crate::compiler::expr;
use crate::lexer::TokenKind;
use crate::parser::ast::{Arg, BinaryOpKind, Expr};

/// Keys that may use integral fast opcodes (`ObjectGetIntegral`, `InIntegral`, …).
pub fn expr_may_be_integral(expr: &Expr) -> bool {
    match expr {
        Expr::Literal {
            value: Value::Number(n),
            ..
        } => n.is_finite() && n.fract() == 0.0,
        Expr::Literal {
            value: Value::Int(_), ..
        } => true,
        Expr::Variable { .. } => true,
        Expr::Binary { .. } | Expr::Unary { .. } => true,
        _ => false,
    }
}

/// Emit a constant default for `dict.get(key, default)` without `Call float(inf)`.
pub fn try_compile_get_default_constant(
    ctx: &mut CompilationContext,
    arg: &Arg,
    line: usize,
) -> Result<bool, LangError> {
    let Arg::Positional(expr) = arg else {
        return Ok(false);
    };
    if let Expr::Call { name, args, .. } = expr {
        if name == "float" && args.len() == 1 {
            if let Arg::Positional(inner) = &args[0] {
                if let Some(v) = literal_number_or_inf(inner) {
                    let idx = ctx.chunk.add_constant(v);
                    ctx.chunk.write_with_line(OpCode::Constant(idx), line);
                    return Ok(true);
                }
            }
        }
    }
    if let Some(v) = literal_number_or_inf(expr) {
        let idx = ctx.chunk.add_constant(v);
        ctx.chunk.write_with_line(OpCode::Constant(idx), line);
        return Ok(true);
    }
    Ok(false)
}

fn literal_number_or_inf(expr: &Expr) -> Option<Value> {
    match expr {
        Expr::Literal {
            value: Value::Number(n),
            ..
        } if n.is_infinite() => Some(Value::Number(*n)),
        Expr::Literal {
            value: Value::Float(FloatValue::PosInfinity | FloatValue::NegInfinity),
            ..
        } => match expr {
            Expr::Literal { value, .. } => Some(value.clone()),
            _ => None,
        },
        Expr::Variable { name, .. } if name == "inf" => Some(Value::Number(f64::INFINITY)),
        _ => None,
    }
}

fn is_zero_expr(expr: &Expr) -> bool {
    matches!(
        expr,
        Expr::Literal {
            value: Value::Number(n),
            ..
        } if *n == 0.0
    ) || matches!(
        expr,
        Expr::Literal {
            value: Value::Int(IntValue::Finite(0)),
            ..
        }
    )
}

fn same_var_expr(a: &Expr, b: &Expr) -> bool {
    matches!(
        (a, b),
        (Expr::Variable { name: na, .. }, Expr::Variable { name: nb, .. }) if na == nb
    )
}

fn match_bounds_half(expr: &Expr) -> Option<(&Expr, &Expr)> {
    let Expr::Binary {
        left,
        op: BinaryOpKind::Builtin(TokenKind::And),
        right,
        ..
    } = expr
    else {
        return None;
    };
    let Expr::Binary {
        left: l0,
        op: BinaryOpKind::Builtin(TokenKind::LessEqual),
        right: nr1,
        ..
    } = left.as_ref()
    else {
        return None;
    };
    let Expr::Binary {
        left: nr2,
        op: BinaryOpKind::Builtin(TokenKind::Less),
        right: bound,
        ..
    } = right.as_ref()
    else {
        return None;
    };
    if !is_zero_expr(l0) || !same_var_expr(nr1, nr2) {
        return None;
    }
    Some((nr1.as_ref(), bound.as_ref()))
}

fn match_grid_bounds(expr: &Expr) -> Option<(&Expr, &Expr, &Expr, &Expr)> {
    let Expr::Binary {
        left,
        op: BinaryOpKind::Builtin(TokenKind::And),
        right,
        ..
    } = expr
    else {
        return None;
    };
    let (nr, rows) = match_bounds_half(left)?;
    let (nc, cols) = match_bounds_half(right)?;
    Some((nr, rows, nc, cols))
}

/// `!(0 <= nr < rows and 0 <= nc < cols)` → [`OpCode::InGridBoundsOut`] (out-of-bounds bool).
pub fn try_compile_negated_grid_bounds(
    ctx: &mut CompilationContext,
    inner: &Expr,
    line: usize,
) -> Result<bool, LangError> {
    let Some((nr, rows, nc, cols)) = match_grid_bounds(inner) else {
        return Ok(false);
    };
    expr::compile_expr(ctx, nr)?;
    expr::compile_expr(ctx, nc)?;
    expr::compile_expr(ctx, rows)?;
    expr::compile_expr(ctx, cols)?;
    ctx.chunk.write_with_line(OpCode::InGridBoundsOut, line);
    Ok(true)
}

fn match_dict_get_one_key<'a>(
    expr: &'a Expr,
) -> Option<(&'a Expr, &'a Expr)> {
    let Expr::MethodCall {
        object,
        method,
        args,
        ..
    } = expr
    else {
        return None;
    };
    if method != "get" || args.len() != 1 {
        return None;
    };
    let Arg::Positional(key) = &args[0] else {
        return None;
    };
    Some((object.as_ref(), key))
}

fn match_dict_get_inf_default(args: &[Arg]) -> bool {
    if args.len() != 2 {
        return false;
    }
    let Arg::Positional(default_expr) = &args[1] else {
        return false;
    };
    if let Expr::Call { name, args, .. } = default_expr {
        if name == "float" && args.len() == 1 {
            if let Arg::Positional(inf_expr) = &args[0] {
                if literal_number_or_inf(inf_expr).is_some() {
                    return true;
                }
            }
        }
    }
    literal_number_or_inf(default_expr).is_some()
}

/// `current_f != f_score.get(current)` → [`OpCode::FScoreStaleCheck`].
pub fn try_compile_f_score_stale_check(
    ctx: &mut CompilationContext,
    left: &Expr,
    right: &Expr,
    line: usize,
) -> Result<bool, LangError> {
    let Some((dict, key)) = match_dict_get_one_key(right) else {
        return Ok(false);
    };
    if !expr_may_be_integral(key) {
        return Ok(false);
    }
    expr::compile_expr(ctx, left)?;
    expr::compile_expr(ctx, dict)?;
    expr::compile_expr(ctx, key)?;
    ctx.chunk.write_with_line(OpCode::FScoreStaleCheck, line);
    Ok(true)
}

/// `tentative_g < g_score.get(neighbor, float(inf))` → [`OpCode::DictGetIntegralLt`].
pub fn try_compile_dict_get_integral_lt(
    ctx: &mut CompilationContext,
    left: &Expr,
    right: &Expr,
    line: usize,
) -> Result<bool, LangError> {
    let (dict, args) = match match_dict_get_call(right) {
        Some(v) => v,
        None => return Ok(false),
    };
    if args.is_empty() {
        return Ok(false);
    }
    let Arg::Positional(key) = &args[0] else {
        return Ok(false);
    };
    if !expr_may_be_integral(key) {
        return Ok(false);
    }
    if args.len() == 2 && !match_dict_get_inf_default(args) {
        return Ok(false);
    }
    if args.len() != 1 && args.len() != 2 {
        return Ok(false);
    }
    expr::compile_expr(ctx, left)?;
    expr::compile_expr(ctx, dict)?;
    expr::compile_expr(ctx, key)?;
    ctx.chunk.write_with_line(OpCode::DictGetIntegralLt, line);
    Ok(true)
}

/// `container[integral_key] + int_literal` → [`OpCode::DictIndexIntegralAddImm`] (array or dict).
pub fn try_compile_dict_index_integral_add(
    ctx: &mut CompilationContext,
    left: &Expr,
    right: &Expr,
    line: usize,
) -> Result<bool, LangError> {
    let addend = match literal_i64_expr(right) {
        Some(n) if (i8::MIN as i64..=i8::MAX as i64).contains(&n) => n as i8,
        _ => return Ok(false),
    };
    let Expr::ArrayIndex {
        array,
        index: crate::parser::ast::IndexExpr::Scalar(key),
        ..
    } = left
    else {
        return Ok(false);
    };
    if !expr_may_be_integral(key) {
        return Ok(false);
    }
    expr::compile_expr(ctx, array)?;
    expr::compile_expr(ctx, key)?;
    ctx.chunk
        .write_with_line(OpCode::DictIndexIntegralAddImm(addend), line);
    Ok(true)
}

fn literal_i64_expr(e: &Expr) -> Option<i64> {
    match e {
        Expr::Literal {
            value: Value::Number(n),
            ..
        } if n.is_finite() && n.fract() == 0.0 => Some(*n as i64),
        Expr::Literal {
            value: Value::Int(crate::common::numeric::IntValue::Finite(n)),
            ..
        } => Some(*n),
        Expr::Unary {
            op: TokenKind::Minus,
            right,
            ..
        } => literal_i64_expr(right).map(|n| -n),
        _ => None,
    }
}

/// `!member in container` → [`OpCode::NotInIntegral`].
pub fn try_compile_not_in_integral(
    ctx: &mut CompilationContext,
    right: &Expr,
    line: usize,
) -> Result<bool, LangError> {
    let Expr::Binary {
        left: member,
        op: BinaryOpKind::Builtin(TokenKind::In),
        right: container,
        ..
    } = right
    else {
        return Ok(false);
    };
    if !expr_may_be_integral(member) {
        return Ok(false);
    }
    expr::compile_expr(ctx, member)?;
    expr::compile_expr(ctx, container)?;
    ctx.chunk.write_with_line(OpCode::NotInIntegral, line);
    Ok(true)
}

fn match_dict_get_call<'a>(expr: &'a Expr) -> Option<(&'a Expr, &'a [Arg])> {
    match expr {
        Expr::MethodCall {
            object,
            method,
            args,
            ..
        } if method == "get" => Some((object.as_ref(), args.as_slice())),
        _ => None,
    }
}
