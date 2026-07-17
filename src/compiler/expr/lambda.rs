use crate::bytecode::function::CapturedVar;
/// Компиляция лямбда-выражений `fn(...) => expr`
use crate::bytecode::{Function, OpCode};
use crate::common::error::LangError;
use crate::common::value::Value;
use crate::compiler::closure;
use crate::compiler::context::CompilationContext;
use crate::compiler::defaults;
use crate::compiler::expr;
use crate::parser::ast::Expr;

pub fn compile_lambda(ctx: &mut CompilationContext, expr: &Expr) -> Result<(), LangError> {
    if let Expr::Lambda {
        params,
        return_type,
        body,
        line,
    } = expr
    {
        *ctx.current_line = *line;

        let function_index = ctx.functions.len();
        let name = format!("__lambda_{}_{}", line, function_index);
        let mut function = Function::new(name.clone(), params.len());

        let mut param_names = Vec::new();
        let mut param_types = Vec::new();
        let mut default_values = Vec::new();
        let signature_param_names: Vec<String> = params.iter().map(|p| p.name.clone()).collect();

        for param in params.iter() {
            param_names.push(param.name.clone());
            param_types.push(param.type_annotation.clone());
            if let Some(ref default_expr) = param.default_value {
                let constant_value = defaults::resolve_default_param_value(
                    default_expr,
                    &param.name,
                    &signature_param_names,
                    ctx.compile_time_bindings,
                    ctx.source_name,
                )?;
                default_values.push(Some(constant_value));
            } else {
                default_values.push(None);
            }
        }

        function.param_names = param_names.clone();
        function.param_types = param_types;
        function.return_type = return_type.clone();
        function.default_values = default_values;
        function.variadic_pos_index = params
            .iter()
            .position(|p| p.kind == crate::parser::ast::ParamKind::VariadicPositional);
        function.variadic_kw_index = params
            .iter()
            .position(|p| p.kind == crate::parser::ast::ParamKind::VariadicKeyword);
        function.arity = params.len();

        ctx.functions.push(function.clone());
        ctx.function_names.push(name.clone());

        let function_chunk_clone = function.chunk.clone();
        let saved_chunk = std::mem::replace(&mut *ctx.chunk, function_chunk_clone);
        let saved_exception_handlers = ctx.exception_handlers.clone();
        let saved_error_type_table = ctx.error_type_table.clone();
        let saved_function = ctx.current_function;
        let enclosing_function_index = saved_function.unwrap_or(usize::MAX);
        let saved_local_count = ctx.scope.local_count;

        let saved_label_counter = ctx.labels.label_counter;
        let saved_labels = ctx.labels.labels.clone();
        let saved_pending_jumps = ctx.labels.pending_jumps.clone();
        let saved_pending_for_range = ctx.labels.pending_for_range.clone();
        ctx.labels.label_counter = 0;
        ctx.labels.labels.clear();
        ctx.labels.pending_jumps.clear();
        ctx.labels.pending_for_range.clear();

        ctx.current_function = Some(function_index);
        ctx.scope.local_count = 0;
        ctx.exception_handlers.clear();
        ctx.error_type_table.clear();

        ctx.chunk.set_source_name(ctx.source_name);

        let parent_locals_snapshot: Vec<std::collections::HashMap<String, usize>> =
            ctx.scope.locals.iter().map(|s| s.clone()).collect();

        let ancestor_bindings = closure::flatten_parent_binding_names(&parent_locals_snapshot);
        closure::check_illegal_outer_assignments_in_lambda_expr(
            body.as_ref(),
            &ancestor_bindings,
            &param_names,
        )?;

        ctx.scope.begin_scope();

        let current_scope = ctx.scope.locals.last().cloned().unwrap_or_default();
        let captured_vars = closure::find_captured_variables_lambda(
            body.as_ref(),
            params,
            &parent_locals_snapshot,
            &current_scope,
        );

        let mut captured_vars_info = Vec::new();
        for var_name in &captured_vars {
            let local_slot_index = ctx.scope.declare_local(var_name);

            let mut parent_slot_index = None;
            let mut ancestor_depth = 0;
            for (depth, scope) in parent_locals_snapshot.iter().rev().enumerate() {
                if let Some(&slot_idx) = scope.get(var_name) {
                    parent_slot_index = Some(slot_idx);
                    ancestor_depth = depth;
                    break;
                }
            }

            if parent_slot_index.is_none() {
                return Err(LangError::ParseError {
                    message: format!(
                        "Captured variable '{}' not found in parent scopes",
                        var_name
                    ),
                    line: *line,
                    file: None,
                });
            }

            let parent_slot = parent_slot_index.unwrap();
            captured_vars_info.push(CapturedVar {
                name: var_name.clone(),
                parent_slot_index: parent_slot,
                local_slot_index,
                ancestor_depth,
                parent_function_index: enclosing_function_index,
            });
        }

        for param in params.iter() {
            ctx.scope.declare_local(&param.name);
            ctx.record_bound_name(&param.name);
        }

        expr::compile_expr(ctx, body.as_ref())?;
        ctx.chunk.write_with_line(OpCode::Return, *ctx.current_line);

        ctx.scope.end_scope();

        ctx.labels.stabilize_layout(&mut *ctx.chunk, *line)?;
        ctx.labels.finalize_jumps(&mut *ctx.chunk, *line)?;

        let function_chunk = std::mem::replace(&mut *ctx.chunk, saved_chunk);
        function.chunk = function_chunk;
        function.captured_vars = captured_vars_info;
        ctx.functions[function_index] = function;

        *ctx.exception_handlers = saved_exception_handlers;
        *ctx.error_type_table = saved_error_type_table;
        ctx.current_function = saved_function;
        ctx.scope.local_count = saved_local_count;

        ctx.labels.label_counter = saved_label_counter;
        ctx.labels.labels = saved_labels;
        ctx.labels.pending_jumps = saved_pending_jumps;
        ctx.labels.pending_for_range = saved_pending_for_range;

        let constant_index = ctx.chunk.add_constant(Value::Function(function_index));
        ctx.chunk
            .write_with_line(OpCode::Constant(constant_index), *line);
        Ok(())
    } else {
        Err(LangError::ParseError {
            message: "Expected Lambda expression".to_string(),
            line: expr.line(),
            file: None,
        })
    }
}
