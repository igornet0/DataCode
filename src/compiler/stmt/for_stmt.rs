use crate::bytecode::OpCode;
use crate::common::error::LangError;
use crate::common::value::Value;
use crate::compiler::context::{CompilationContext, LoopContext};
use crate::compiler::expr;
use crate::compiler::stmt;
use crate::compiler::unpack;
use crate::parser::ast::Expr;
/// Компиляция for statements
use crate::parser::ast::{Arg, Stmt, UnpackPattern};

/// Если iterable — вызов range с числовыми литералами, возвращает (start, end, step).
fn try_range_literals(iterable: &Expr) -> Option<(i64, i64, i64)> {
    let Expr::Call { name, args, .. } = iterable else {
        return None;
    };
    if name != "range" {
        return None;
    }
    let positional: Vec<_> = args
        .iter()
        .filter_map(|a| match a {
            Arg::Positional(e) => Some(e),
            _ => None,
        })
        .collect();
    if positional.is_empty() || positional.len() > 3 {
        return None;
    }
    let to_i64 = |e: &Expr| -> Option<i64> {
        if let Expr::Literal {
            value: Value::Number(n),
            ..
        } = e
        {
            let x = *n;
            if x.fract() == 0.0 && x >= i64::MIN as f64 && x <= i64::MAX as f64 {
                return Some(x as i64);
            }
        }
        None
    };
    match positional.len() {
        1 => {
            let end = to_i64(positional[0])?;
            Some((0, end, 1))
        }
        2 => {
            let start = to_i64(positional[0])?;
            let end = to_i64(positional[1])?;
            Some((start, end, 1))
        }
        3 => {
            let start = to_i64(positional[0])?;
            let end = to_i64(positional[1])?;
            let step = to_i64(positional[2])?;
            if step == 0 {
                return None;
            }
            Some((start, end, step))
        }
        _ => None,
    }
}

pub fn compile_for(ctx: &mut CompilationContext, stmt: &Stmt) -> Result<(), LangError> {
    if let Stmt::For {
        pattern,
        iterable,
        body,
        line,
    } = stmt
    {
        *ctx.current_line = *line;

        // Same as Python: loop body bindings (including the iterator name) live in the
        // enclosing function scope — not a separate block that ends after the loop.

        // Быстрый путь: for i in range(start, end, step) с литеральными границами — без материализации диапазона
        let is_simple_case = pattern.len() == 1
            && matches!(
                pattern[0],
                UnpackPattern::Variable(_) | UnpackPattern::Wildcard
            );
        if let Some((start, end, step)) = try_range_literals(iterable) {
            if is_simple_case {
                let var_slot = match &pattern[0] {
                    UnpackPattern::Variable(var_name) => ctx.scope.declare_local(var_name),
                    UnpackPattern::Wildcard => ctx.scope.declare_local("__for_discard"),
                    _ => unreachable!(),
                };
                let start_const = ctx.chunk.add_constant(Value::Number(start as f64));
                let end_const = ctx.chunk.add_constant(Value::Number(end as f64));
                let step_const = ctx.chunk.add_constant(Value::Number(step as f64));
                let loop_start_label = ctx.labels.create_label();
                let loop_end_label = ctx.labels.create_label();
                ctx.labels
                    .mark_label(loop_start_label, ctx.chunk.code.len());
                let forrange_ip = ctx.chunk.code.len();
                ctx.chunk.write_with_line(
                    OpCode::ForRange(var_slot, start_const, end_const, step_const, 0),
                    *line,
                );
                ctx.labels
                    .register_for_range_end(forrange_ip, loop_end_label);
                let loop_context = LoopContext {
                    continue_label: loop_start_label,
                    break_label: loop_end_label,
                    is_for_range: true,
                };
                ctx.loop_contexts.push(loop_context);
                for s in body {
                    stmt::compile_stmt(ctx, s, true)?;
                }
                ctx.loop_contexts.pop();
                let forrangenext_ip = ctx.chunk.code.len();
                // When executor runs ForRangeNext, frame.ip is already forrangenext_ip + 1; we want frame.ip = forrange_ip.
                let back_offset = (forrangenext_ip as i32) + 1 - (forrange_ip as i32);
                ctx.chunk
                    .write_with_line(OpCode::ForRangeNext(back_offset), *line);
                ctx.labels.mark_label(loop_end_label, ctx.chunk.code.len());
                return Ok(());
            }
        }

        // Унифицированный путь: `for-in` через coerce + ForIterableNext (включая `enum(...)` и таблицы).
        expr::compile_expr(ctx, iterable)?;
        let array_local = ctx.scope.declare_local("__array_iter");
        ctx.chunk
            .write_with_line(OpCode::StoreLocal(array_local), *line);
        ctx.chunk
            .write_with_line(OpCode::CoerceForInIterable(array_local), *line);

        let is_simple_case = pattern.len() == 1
            && matches!(
                pattern[0],
                UnpackPattern::Variable(_) | UnpackPattern::Wildcard
            );
        let expected_count = if is_simple_case {
            0
        } else {
            unpack::count_unpack_variables(pattern)
        };

        let var_locals = if is_simple_case {
            match &pattern[0] {
                UnpackPattern::Variable(name) => {
                    vec![Some(ctx.scope.declare_local(name))]
                }
                UnpackPattern::Wildcard => vec![None],
                _ => unreachable!(),
            }
        } else {
            unpack::declare_unpack_pattern_variables(pattern, ctx.scope, *line)?
        };

        let loop_start_label = ctx.labels.create_label();
        let loop_end_label = ctx.labels.create_label();

        ctx.labels
            .mark_label(loop_start_label, ctx.chunk.code.len());
        ctx.chunk
            .write_with_line(OpCode::ForIterableNext(array_local), *line);
        ctx.labels
            .emit_jump(ctx.chunk, *ctx.current_line, true, loop_end_label)?;

        if is_simple_case {
            match &pattern[0] {
                UnpackPattern::Variable(_) => {
                    if let Some(local_index) = var_locals[0] {
                        ctx.chunk
                            .write_with_line(OpCode::StoreLocal(local_index), *line);
                    }
                }
                UnpackPattern::Wildcard => {
                    ctx.chunk.write_with_line(OpCode::Pop, *line);
                }
                _ => unreachable!(),
            }
        } else {
            unpack::compile_unpack_pattern(
                pattern,
                &var_locals,
                expected_count,
                ctx.chunk,
                ctx.scope,
                ctx.labels,
                *ctx.current_line,
                *line,
            )?;
        }

        let loop_context = LoopContext {
            continue_label: loop_start_label,
            break_label: loop_end_label,
            is_for_range: false,
        };
        ctx.loop_contexts.push(loop_context);
        for stmt in body {
            stmt::compile_stmt(ctx, stmt, true)?;
        }
        ctx.loop_contexts.pop();

        ctx.labels
            .emit_loop(ctx.chunk, *ctx.current_line, loop_start_label)?;
        ctx.labels.mark_label(loop_end_label, ctx.chunk.code.len());
        return Ok(());
    } else {
        Err(LangError::ParseError {
            message: "Expected For statement".to_string(),
            line: stmt.line(),
            file: None,
        })
    }
}
