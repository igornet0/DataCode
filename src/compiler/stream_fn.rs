//! Компиляция `stream fn`: `return` → yield, `ereturn` → завершение.
//! `ereturn expr` сохраняет значение в [`crate::common::value::GeneratorState::final_value`] (`GeneratorDoneWithFinal`), не в поток yield.
//! `x = return expr` / `x = ireturn` / `x = ireturn expr` → [`OpCode::YieldAwaitInput`]: yield RHS и ожидание значения из `.send()`; при следующем `.next()` вне `for-in` подставляется значение RHS yield (как `send(None)` в Python), если `.send()` не вызывали (см. [`crate::vm::generator::run_generator_resume`]).
//! Обычный `return expr;` (statement) → [`OpCode::Yield`].

use crate::bytecode::OpCode;
use crate::common::error::LangError;
use crate::common::value::Value;
use crate::compiler::context::CompilationContext;
use crate::compiler::expr;
use crate::compiler::stmt::let_stmt;
use crate::parser::ast::{Expr, Stmt};

fn compile_yield_await_assign(
    ctx: &mut CompilationContext,
    name: &str,
    rhs: Option<&Expr>,
    line: usize,
    yield_counter: &mut i32,
) -> Result<(), LangError> {
    let slot = if let Some(idx) = ctx.scope.resolve_local(name) {
        idx
    } else {
        ctx.scope.declare_local(name)
    };
    if let Some(e) = rhs {
        expr::compile_expr(ctx, e)?;
    } else {
        let c = ctx.chunk.add_constant(Value::Null);
        ctx.chunk.write_with_line(OpCode::Constant(c), line);
    }
    let y = *yield_counter;
    *yield_counter += 1;
    ctx.chunk
        .write_with_line(OpCode::YieldAwaitInput(y, slot), line);
    Ok(())
}

/// Компилирует тело stream fn (линейные stmt, `if`, `let`, `expr`, `return`, `ereturn`).
pub fn compile_stream_body(
    ctx: &mut CompilationContext,
    body: &[Stmt],
    line_hint: usize,
) -> Result<(), LangError> {
    let mut yield_counter: i32 = 0;
    for stmt in body {
        compile_stream_stmt(ctx, stmt, &mut yield_counter, line_hint)?;
    }
    // Неявный конец потока (после последнего yield)
    if !matches!(
        ctx.chunk.code.last(),
        Some(OpCode::GeneratorDone) | Some(OpCode::GeneratorDoneWithFinal)
    ) {
        ctx.chunk.write_with_line(OpCode::GeneratorDone, line_hint);
    }
    Ok(())
}

fn compile_stream_stmt(
    ctx: &mut CompilationContext,
    stmt: &Stmt,
    yield_counter: &mut i32,
    line_hint: usize,
) -> Result<(), LangError> {
    match stmt {
        Stmt::Let {
            name,
            value,
            is_global,
            line,
        } => {
            if *is_global {
                return Err(LangError::ParseError {
                    message: "global let is not supported inside stream fn yet".to_string(),
                    line: *line,
                    file: None,
                });
            }
            if let Expr::ExprReturn { value: rv, .. } | Expr::Ireturn { value: rv, .. } = value {
                compile_yield_await_assign(ctx, name, rv.as_deref(), *line, yield_counter)
            } else {
                let_stmt::compile_let(ctx, stmt, true)
            }
        }
        Stmt::Expr { expr, line } => {
            *ctx.current_line = *line;
            if let Expr::Assign {
                name,
                value,
                line: al,
            } = expr
            {
                if let Expr::ExprReturn { value: rv, .. } | Expr::Ireturn { value: rv, .. } =
                    value.as_ref()
                {
                    return compile_yield_await_assign(
                        ctx,
                        name,
                        rv.as_deref(),
                        *al,
                        yield_counter,
                    );
                }
            }
            if let Expr::ExprReturn { value, line: el } = expr {
                *ctx.current_line = *el;
                if let Some(e) = value {
                    expr::compile_expr(ctx, e)?;
                } else {
                    let c = ctx.chunk.add_constant(Value::Null);
                    ctx.chunk.write_with_line(OpCode::Constant(c), *el);
                }
                let y = *yield_counter;
                *yield_counter += 1;
                ctx.chunk.write_with_line(OpCode::Yield(y), *line);
                return Ok(());
            }
            if let Expr::Ireturn {
                value: rv,
                line: el,
            } = expr
            {
                return compile_yield_await_assign(
                    ctx,
                    "__ireturn_discard",
                    rv.as_deref(),
                    *el,
                    yield_counter,
                );
            }
            expr::compile_expr(ctx, expr)?;
            ctx.chunk.write_with_line(OpCode::Pop, *line);
            Ok(())
        }
        Stmt::Return { value, line } => {
            *ctx.current_line = *line;
            if let Some(e) = value {
                expr::compile_expr(ctx, e)?;
            } else {
                let c = ctx.chunk.add_constant(Value::Null);
                ctx.chunk.write_with_line(OpCode::Constant(c), *line);
            }
            let y = *yield_counter;
            *yield_counter += 1;
            ctx.chunk.write_with_line(OpCode::Yield(y), *line);
            Ok(())
        }
        Stmt::EReturn { value, line } => {
            *ctx.current_line = *line;
            if let Some(e) = value {
                expr::compile_expr(ctx, e)?;
                ctx.chunk
                    .write_with_line(OpCode::GeneratorDoneWithFinal, *line);
            } else {
                ctx.chunk.write_with_line(OpCode::GeneratorDone, *line);
            }
            Ok(())
        }
        Stmt::If {
            condition,
            then_branch,
            else_branch,
            line,
        } => {
            *ctx.current_line = *line;
            expr::compile_expr(ctx, condition)?;
            let else_label = ctx.labels.create_label();
            let end_label = ctx.labels.create_label();
            let target_else = if else_branch.is_some() {
                else_label
            } else {
                end_label
            };
            ctx.labels.emit_jump(ctx.chunk, *line, true, target_else)?;
            ctx.scope.begin_scope();
            for s in then_branch {
                compile_stream_stmt(ctx, s, yield_counter, line_hint)?;
            }
            ctx.scope.end_scope();
            ctx.labels.emit_jump(ctx.chunk, *line, false, end_label)?;
            if else_branch.is_some() {
                ctx.labels.mark_label(else_label, ctx.chunk.code.len());
                ctx.scope.begin_scope();
                for s in else_branch.as_ref().unwrap() {
                    compile_stream_stmt(ctx, s, yield_counter, line_hint)?;
                }
                ctx.scope.end_scope();
            }
            ctx.labels.mark_label(end_label, ctx.chunk.code.len());
            Ok(())
        }
        Stmt::While { line, .. }
        | Stmt::For { line, .. }
        | Stmt::Try { line, .. }
        | Stmt::Class { line, .. }
        | Stmt::Import { line, .. }
        | Stmt::Function { line, .. }
        | Stmt::StreamFunction { line, .. } => Err(LangError::ParseError {
            message: "this statement is not supported inside stream fn yet".to_string(),
            line: *line,
            file: None,
        }),
        Stmt::Break { line } | Stmt::Continue { line } => Err(LangError::ParseError {
            message: "break/continue inside stream fn is not supported yet".to_string(),
            line: *line,
            file: None,
        }),
        Stmt::Throw { line, .. } => Err(LangError::ParseError {
            message: "throw inside stream fn is not supported yet".to_string(),
            line: *line,
            file: None,
        }),
    }
}
