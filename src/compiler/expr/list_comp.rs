//! List comprehension: `[ e for x in it ... ]` → nested for-in + filters + push.

use crate::bytecode::OpCode;
use crate::common::error::LangError;
use crate::compiler::context::CompilationContext;
use crate::compiler::expr;
use crate::compiler::unpack;
use crate::parser::ast::{Arg, Expr, ListComprehensionClause, UnpackPattern};

pub fn compile_list_comprehension(
    ctx: &mut CompilationContext,
    elt: &Expr,
    clauses: &[ListComprehensionClause],
    line: usize,
) -> Result<(), LangError> {
    if clauses.is_empty() || !matches!(clauses[0], ListComprehensionClause::For { .. }) {
        return Err(LangError::ParseError {
            message: "List comprehension requires at least one `for` clause".to_string(),
            line,
            file: None,
        });
    }

    *ctx.current_line = line;
    ctx.scope.begin_scope();

    ctx.chunk.write_with_line(OpCode::MakeArray(0), line);
    let _result_slot = ctx.scope.declare_local("__list_comp_result");
    ctx.chunk
        .write_with_line(OpCode::StoreLocal(_result_slot), line);

    let mut iter_id: u32 = 0;
    let mut continue_stack: Vec<usize> = Vec::new();
    compile_clause_tail(
        ctx,
        clauses,
        0,
        elt,
        line,
        &mut iter_id,
        &mut continue_stack,
    )?;

    ctx.chunk
        .write_with_line(OpCode::LoadLocal(_result_slot), line);
    ctx.scope.end_scope();
    Ok(())
}

fn compile_clause_tail(
    ctx: &mut CompilationContext,
    clauses: &[ListComprehensionClause],
    idx: usize,
    elt: &Expr,
    line: usize,
    iter_id: &mut u32,
    continue_stack: &mut Vec<usize>,
) -> Result<(), LangError> {
    *ctx.current_line = line;
    if idx >= clauses.len() {
        let call = Expr::Call {
            name: "push".to_string(),
            args: vec![
                Arg::Positional(Expr::Variable {
                    name: "__list_comp_result".to_string(),
                    line,
                }),
                Arg::Positional(elt.clone()),
            ],
            line,
        };
        expr::compile_expr(ctx, &call)?;
        ctx.chunk.write_with_line(OpCode::Pop, line);
        return Ok(());
    }

    match &clauses[idx] {
        ListComprehensionClause::For {
            pattern,
            iterable,
        } => {
            expr::compile_expr(ctx, iterable.as_ref())?;
            let iter_name = format!("__list_comp_iter_{}", iter_id);
            *iter_id += 1;
            let iter_local = ctx.scope.declare_local(&iter_name);
            ctx.chunk
                .write_with_line(OpCode::StoreLocal(iter_local), line);
            ctx.chunk
                .write_with_line(OpCode::CoerceForInIterable(iter_local), line);

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
                unpack::declare_unpack_pattern_variables(pattern, ctx.scope, line)?
            };

            let loop_start_label = ctx.labels.create_label();
            let loop_end_label = ctx.labels.create_label();

            ctx.labels
                .mark_label(loop_start_label, ctx.chunk.code.len());
            ctx.chunk
                .write_with_line(OpCode::ForIterableNext(iter_local), line);
            ctx.labels
                .emit_jump(ctx.chunk, *ctx.current_line, true, loop_end_label)?;

            if is_simple_case {
                match &pattern[0] {
                    UnpackPattern::Variable(_) => {
                        if let Some(local_index) = var_locals[0] {
                            ctx.chunk
                                .write_with_line(OpCode::StoreLocal(local_index), line);
                        }
                    }
                    UnpackPattern::Wildcard => {
                        ctx.chunk.write_with_line(OpCode::Pop, line);
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
                    line,
                )?;
            }

            continue_stack.push(loop_start_label);
            compile_clause_tail(
                ctx,
                clauses,
                idx + 1,
                elt,
                line,
                iter_id,
                continue_stack,
            )?;
            continue_stack.pop();

            ctx.labels
                .emit_loop(ctx.chunk, *ctx.current_line, loop_start_label)?;
            ctx.labels.mark_label(loop_end_label, ctx.chunk.code.len());
            Ok(())
        }
        ListComprehensionClause::If { condition } => {
            let continue_label = *continue_stack.last().ok_or_else(|| LangError::ParseError {
                message: "Internal error: `if` in list comprehension without enclosing `for`"
                    .to_string(),
                line,
                file: None,
            })?;
            expr::compile_expr(ctx, condition.as_ref())?;
            ctx.labels
                .emit_jump(ctx.chunk, *ctx.current_line, true, continue_label)?;
            compile_clause_tail(
                ctx,
                clauses,
                idx + 1,
                elt,
                line,
                iter_id,
                continue_stack,
            )
        }
    }
}
