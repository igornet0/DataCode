//! Scope-aware compilation of `{ key: value }` object literals.

use crate::bytecode::OpCode;
use crate::common::error::LangError;
use crate::common::value::Value;
use crate::compiler::context::CompilationContext;
use crate::compiler::expr;
use crate::parser::ast::{Expr, ObjectLiteralKey, ObjectPair};

pub fn is_bound_object_key_name(ctx: &CompilationContext, name: &str) -> bool {
    ctx.scope.resolve_local(name).is_some() || ctx.known_bound_names.contains(name)
}

fn compile_static_key(
    ctx: &mut CompilationContext,
    key: &ObjectLiteralKey,
    line: usize,
) -> Result<(), LangError> {
    let key_val = match key {
        ObjectLiteralKey::Ident(s) | ObjectLiteralKey::String(s) => Value::String(s.clone()),
        ObjectLiteralKey::Number(n) => Value::Number(*n),
    };
    let key_index = ctx.chunk.add_constant(key_val);
    ctx.chunk.write_with_line(OpCode::Constant(key_index), line);
    Ok(())
}

fn compile_object_pair_key_expr(
    ctx: &mut CompilationContext,
    key: &Expr,
    line: usize,
) -> Result<(), LangError> {
    if let Expr::Variable { name, .. } = key {
        if !is_bound_object_key_name(ctx, name) {
            let key_index = ctx.chunk.add_constant(Value::String(name.clone()));
            ctx.chunk.write_with_line(OpCode::Constant(key_index), line);
            return Ok(());
        }
    }
    expr::compile_expr(ctx, key)
}

pub fn compile_object_literal(
    ctx: &mut CompilationContext,
    pairs: &[ObjectPair],
    line: usize,
) -> Result<(), LangError> {
    *ctx.current_line = line;
    let has_spread = pairs.iter().any(|p| matches!(p, ObjectPair::Spread(_)));
    if !has_spread {
        let n_pairs = pairs
            .iter()
            .filter(|p| {
                matches!(
                    p,
                    ObjectPair::KeyValue(_, _) | ObjectPair::KeyValueExpr(_, _)
                )
            })
            .count();
        for p in pairs.iter().rev() {
            match p {
                ObjectPair::KeyValue(key, value) => {
                    compile_static_key(ctx, key, line)?;
                    expr::compile_expr(ctx, value)?;
                }
                ObjectPair::KeyValueExpr(key, value) => {
                    compile_object_pair_key_expr(ctx, key, line)?;
                    expr::compile_expr(ctx, value)?;
                }
                ObjectPair::Spread(_) => {}
            }
        }
        ctx.chunk.write_with_line(OpCode::MakeObject(n_pairs), line);
    } else {
        let count_slot = ctx.scope.declare_local("__object_pair_count");
        let zero_index = ctx.chunk.add_constant(Value::Number(0.0));
        ctx.chunk.write_with_line(OpCode::Constant(zero_index), line);
        ctx.chunk
            .write_with_line(OpCode::StoreLocal(count_slot), line);
        for p in pairs {
            match p {
                ObjectPair::KeyValue(key, value) => {
                    compile_static_key(ctx, key, line)?;
                    expr::compile_expr(ctx, value)?;
                    ctx.chunk
                        .write_with_line(OpCode::LoadLocal(count_slot), line);
                    let one_index = ctx.chunk.add_constant(Value::Number(1.0));
                    ctx.chunk.write_with_line(OpCode::Constant(one_index), line);
                    ctx.chunk.write_with_line(OpCode::Add, line);
                    ctx.chunk
                        .write_with_line(OpCode::StoreLocal(count_slot), line);
                }
                ObjectPair::KeyValueExpr(key, value) => {
                    compile_object_pair_key_expr(ctx, key, line)?;
                    expr::compile_expr(ctx, value)?;
                    ctx.chunk
                        .write_with_line(OpCode::LoadLocal(count_slot), line);
                    let one_index = ctx.chunk.add_constant(Value::Number(1.0));
                    ctx.chunk.write_with_line(OpCode::Constant(one_index), line);
                    ctx.chunk.write_with_line(OpCode::Add, line);
                    ctx.chunk
                        .write_with_line(OpCode::StoreLocal(count_slot), line);
                }
                ObjectPair::Spread(expr) => {
                    expr::compile_expr(ctx, expr)?;
                    ctx.chunk
                        .write_with_line(OpCode::UnpackObject(count_slot), line);
                }
            }
        }
        ctx.chunk
            .write_with_line(OpCode::LoadLocal(count_slot), line);
        ctx.chunk.write_with_line(OpCode::MakeObjectDynamic, line);
    }
    Ok(())
}
