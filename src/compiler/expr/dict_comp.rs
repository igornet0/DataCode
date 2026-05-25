//! Dict comprehension: `{ k: v for x in it [if c] }` → stack pairs + MakeObjectDynamic.

use crate::bytecode::OpCode;
use crate::common::error::LangError;
use crate::common::value::Value;
use crate::compiler::context::CompilationContext;
use crate::compiler::expr;
use crate::parser::ast::Expr;

pub fn compile_dict_comprehension(
    ctx: &mut CompilationContext,
    key_expr: &Expr,
    value_expr: &Expr,
    loop_var: &str,
    iterable: &Expr,
    condition: Option<&Expr>,
    line: usize,
) -> Result<(), LangError> {
    *ctx.current_line = line;

    ctx.scope.begin_scope();

    let count_slot = ctx.scope.declare_local("__dict_comp_count");
    let zero = ctx.chunk.add_constant(Value::Number(0.0));
    ctx.chunk.write_with_line(OpCode::Constant(zero), line);
    ctx.chunk
        .write_with_line(OpCode::StoreLocal(count_slot), line);

    expr::compile_expr(ctx, iterable)?;
    let iter_local = ctx.scope.declare_local("__dict_comp_iter");
    ctx.chunk.write_with_line(OpCode::StoreLocal(iter_local), line);
    ctx.chunk
        .write_with_line(OpCode::CoerceForInIterable(iter_local), line);

    let loop_var_slot = ctx.scope.declare_local(loop_var);

    let loop_start = ctx.labels.create_label();
    let loop_end = ctx.labels.create_label();
    let skip_pair = ctx.labels.create_label();

    ctx.labels.mark_label(loop_start, ctx.chunk.code.len());
    ctx.chunk
        .write_with_line(OpCode::ForIterableNext(iter_local), line);
    ctx.labels
        .emit_jump(ctx.chunk, *ctx.current_line, true, loop_end)?;

    ctx.chunk
        .write_with_line(OpCode::StoreLocal(loop_var_slot), line);

    if let Some(cond) = condition {
        expr::compile_expr(ctx, cond)?;
        ctx.labels
            .emit_jump(ctx.chunk, *ctx.current_line, true, skip_pair)?;
    }

    expr::compile_expr(ctx, key_expr)?;
    expr::compile_expr(ctx, value_expr)?;
    ctx.chunk
        .write_with_line(OpCode::LoadLocal(count_slot), line);
    let one = ctx.chunk.add_constant(Value::Number(1.0));
    ctx.chunk.write_with_line(OpCode::Constant(one), line);
    ctx.chunk.write_with_line(OpCode::Add, line);
    ctx.chunk
        .write_with_line(OpCode::StoreLocal(count_slot), line);

    if condition.is_some() {
        ctx.labels.mark_label(skip_pair, ctx.chunk.code.len());
    }

    ctx.labels
        .emit_loop(ctx.chunk, *ctx.current_line, loop_start)?;
    ctx.labels.mark_label(loop_end, ctx.chunk.code.len());

    ctx.chunk
        .write_with_line(OpCode::LoadLocal(count_slot), line);
    ctx.chunk
        .write_with_line(OpCode::MakeObjectDynamic, line);

    ctx.scope.end_scope();
    Ok(())
}
