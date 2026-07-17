use crate::bytecode::OpCode;
use crate::common::error::LangError;
use crate::common::value::Value;
use crate::compiler::context::CompilationContext;
use crate::compiler::defaults;
use crate::compiler::expr;
use crate::compiler::stmt::for_stmt;
use crate::compiler::variable::VariableResolver;
use crate::parser::ast::{AssignTarget, Expr, Stmt};
/// Компиляция let statements

pub fn compile_let(
    ctx: &mut CompilationContext,
    stmt: &Stmt,
    pop_value: bool,
) -> Result<(), LangError> {
    if let Stmt::Let {
        name,
        value,
        is_global,
        line,
    } = stmt
    {
        *ctx.current_line = *line;

        // Проверяем, является ли value UnpackAssign (распаковка кортежа)
        if let Expr::UnpackAssign {
            targets,
            value: tuple_value,
            ..
        } = value
        {
            // Распаковка кортежа в let statement
            expr::compile_expr(ctx, &**tuple_value)?;

            let tuple_temp = ctx.scope.declare_local(&format!("__tuple_temp_{}", line));
            ctx.chunk
                .write_with_line(OpCode::StoreLocal(tuple_temp), *line);

            for (index, target) in targets.iter().enumerate() {
                let AssignTarget::Name(var_name) = target else {
                    return Err(LangError::ParseError {
                        message: "let unpack supports variable names only".to_string(),
                        line: *line,
                        file: None,
                    });
                };
                ctx.chunk
                    .write_with_line(OpCode::LoadLocal(tuple_temp), *line);
                let index_const = ctx.chunk.add_constant(Value::Number(index as f64));
                ctx.chunk
                    .write_with_line(OpCode::Constant(index_const), *line);
                ctx.chunk.write_with_line(OpCode::GetArrayElement, *line);

                if *is_global {
                    let global_index = if let Some(&idx) = ctx.scope.globals.get(var_name) {
                        idx
                    } else {
                        let idx = ctx.scope.globals.len();
                        ctx.scope.globals.insert(var_name.clone(), idx);
                        idx
                    };
                    ctx.chunk
                        .global_names
                        .insert(global_index, var_name.clone());
                    ctx.chunk
                        .explicit_global_names
                        .insert(global_index, var_name.clone());
                    ctx.scope.explicit_globals.insert(var_name.clone());
                    ctx.chunk
                        .write_with_line(OpCode::StoreGlobal(global_index), *line);
                } else if let Some(local_index) = ctx.scope.resolve_local(var_name) {
                    ctx.chunk
                        .write_with_line(OpCode::StoreLocal(local_index), *line);
                } else if ctx.current_function.is_some() {
                    let var_index = ctx.declare_local_for_binding(var_name);
                    ctx.chunk
                        .write_with_line(OpCode::StoreLocal(var_index), *line);
                } else if let Some(&global_index) = ctx.scope.globals.get(var_name) {
                    ctx.chunk
                        .global_names
                        .insert(global_index, var_name.clone());
                    ctx.chunk
                        .write_with_line(OpCode::StoreGlobal(global_index), *line);
                } else {
                    let global_index = ctx.scope.globals.len();
                    ctx.scope.globals.insert(var_name.clone(), global_index);
                    ctx.chunk
                        .global_names
                        .insert(global_index, var_name.clone());
                    ctx.chunk
                        .write_with_line(OpCode::StoreGlobal(global_index), *line);
                }
                ctx.record_bound_name(var_name);
            }

            if let Some(AssignTarget::Name(last_name)) = targets.last() {
                VariableResolver::resolve_and_load(ctx, last_name, *line)?;
            }
            // Когда pop_value (не последний statement или результат не нужен), снимаем значение со стека
            if pop_value {
                ctx.chunk.write_with_line(OpCode::Pop, *line);
            }
            return Ok(());
        }

        // Обычное присваивание
        // Best-effort tracking for `set()` locals to safely emit SetAddIntegral/SetDiscardIntegral.
        // (Used by A* and by opcode-emission regression tests.)
        let is_set_call = matches!(
            value,
            Expr::Call { name: callee, args, .. } if callee == "set" && args.is_empty()
        );
        if is_set_call {
            ctx.known_set_vars.insert(name.clone());
        } else {
            ctx.known_set_vars.remove(name);
        }
        // Best-effort: track constructor-like assignments (UpperCamelCase(...)) to avoid miscompiling `.add`.
        let is_ctor_like_call = matches!(
            value,
            Expr::Call { name: callee, .. }
                if callee.chars().next().is_some_and(|ch| ch.is_uppercase())
        );
        if is_ctor_like_call {
            ctx.known_class_instance_vars.insert(name.clone());
        } else {
            ctx.known_class_instance_vars.remove(name);
        }
        if let Some(pairs) = for_stmt::try_const_tuple_of_pairs(value) {
            ctx.known_const_pair_tuples.insert(name.clone(), pairs);
        } else {
            ctx.known_const_pair_tuples.remove(name);
        }
        if ctx.current_function.is_none() {
            defaults::update_compile_time_binding(ctx.compile_time_bindings, name, value);
        }
        expr::compile_expr(ctx, value)?;
        // Не клонируем автоматически - переменные должны разделять ссылки на массивы/таблицы/объекты
        VariableResolver::resolve_and_store(ctx, name, *is_global, *line)?;
        ctx.record_bound_name(name);

        Ok(())
    } else {
        Err(LangError::ParseError {
            message: "Expected Let statement".to_string(),
            line: stmt.line(),
            file: None,
        })
    }
}
