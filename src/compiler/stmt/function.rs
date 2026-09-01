use crate::bytecode::{CapturedVar, OpCode};
use crate::common::error::LangError;
use crate::common::value::Value;
use crate::compiler::closure;
use crate::compiler::context::CompilationContext;
use crate::compiler::defaults;
use crate::compiler::stmt;
use crate::compiler::stream_fn;
/// Компиляция function statements
use crate::parser::ast::Stmt;

fn resolve_function_index_by_name(
    ctx: &mut CompilationContext,
    name: &str,
    line: usize,
) -> Result<usize, LangError> {
    let pass = ctx
        .function_compile_pass
        .entry(name.to_string())
        .or_insert(0);
    let function_index = ctx
        .function_names
        .iter()
        .enumerate()
        .filter(|(_, n)| *n == name)
        .map(|(i, _)| i)
        .nth(*pass)
        .ok_or_else(|| LangError::ParseError {
            message: format!("Function '{}' not found in forward declarations", name),
            line,
            file: None,
        })?;
    *pass += 1;
    Ok(function_index)
}

fn emit_function_value_to_slot(
    ctx: &mut CompilationContext,
    function_index: usize,
    slot: usize,
    line: usize,
) {
    let constant_index = ctx
        .chunk
        .add_constant(Value::Function(function_index));
    ctx.chunk
        .write_with_line(OpCode::Constant(constant_index), line);
    ctx.chunk
        .write_with_line(OpCode::StoreLocal(slot), line);
    ctx.local_fn_by_slot.insert(slot, function_index);
}

/// Resolve user-defined function metadata for a call site (respects nested local bindings).
pub fn user_function_info_for_call<'a>(
    ctx: &'a CompilationContext<'a>,
    name: &str,
) -> Option<(usize, &'a crate::bytecode::Function)> {
    if let Some(local_index) = ctx.scope.resolve_local(name) {
        if let Some(&function_index) = ctx.local_fn_by_slot.get(&local_index) {
            return Some((function_index, &ctx.functions[function_index]));
        }
    }
    ctx.function_names
        .iter()
        .position(|n| n == name)
        .map(|function_index| (function_index, &ctx.functions[function_index]))
}

pub fn compile_function(ctx: &mut CompilationContext, stmt: &Stmt) -> Result<(), LangError> {
    if let Stmt::Function {
        name,
        params,
        return_type,
        body,
        is_cached,
        route,
        ws_route,
        line,
    } = stmt
    {
        *ctx.current_line = *line;

        // Находим индекс функции (она уже объявлена в первом проходе; nth при совпадении имён)
        let function_index = resolve_function_index_by_name(ctx, name, *line)?;

        // Получаем функцию и обновляем количество параметров и флаг кэширования
        let mut function = ctx.functions[function_index].clone();
        function.arity = params.len();
        function.is_cached = *is_cached;
        if let Some((ref method, ref path)) = route {
            function.route_method = Some(method.clone());
            function.route_path = Some(path.clone());
        }
        if let Some(ref ws_type) = ws_route {
            function.ws_route_type = Some(ws_type.clone());
        }

        // Сохраняем имена параметров, типы и вычисляем значения по умолчанию
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

        // Если кэш включен, но еще не инициализирован, инициализируем его
        if *is_cached && function.cache.is_none() {
            use crate::bytecode::function::FnCache;
            use std::cell::RefCell;
            use std::rc::Rc;
            function.cache = Some(Rc::new(RefCell::new(FnCache::new())));
        }

        // ВАЖНО: Сохраняем сигнатуру функции ДО компиляции тела
        ctx.functions[function_index] = function.clone();

        // Сохраняем текущие локальные области видимости для доступа к переменным родительских функций
        let parent_locals_snapshot: Vec<std::collections::HashMap<String, usize>> =
            ctx.scope.locals.iter().map(|scope| scope.clone()).collect();
        closure::check_illegal_outer_assignments_in_function_body(
            body,
            &closure::ancestor_bindings_for_function_check(
                &parent_locals_snapshot,
                ctx.current_function.is_some(),
            ),
            &param_names,
        )?;

        // Компилируем тело функции в chunk функции
        let function_chunk_clone = function.chunk.clone();
        let saved_chunk = std::mem::replace(&mut *ctx.chunk, function_chunk_clone);
        let saved_exception_handlers = ctx.exception_handlers.clone();
        let saved_error_type_table = ctx.error_type_table.clone();
        let saved_function = ctx.current_function;
        let enclosing_function_index = saved_function.unwrap_or(usize::MAX);

        // Вложенная функция: резервируем локальный слот в родительской области
        let parent_fn_slot = if saved_function.is_some() {
            let slot = ctx.scope.declare_local(name);
            ctx.record_bound_name(name);
            Some(slot)
        } else {
            None
        };
        let saved_local_count = ctx.scope.local_count;
        let saved_method_object_temps = ctx.scope.snapshot_method_object_temps();

        // ВАЖНО: Сохраняем состояние меток перед компиляцией функции
        // и очищаем метки, чтобы предотвратить переиспользование меток между функциями
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
        ctx.scope.reset_method_object_temps();
        // Очищаем обработчики и таблицу типов ошибок для новой функции
        ctx.exception_handlers.clear();
        ctx.error_type_table.clear();

        // Начинаем новую область видимости для функции
        ctx.scope.begin_scope();

        // Находим переменные, которые используются в теле функции, но не объявлены в ней
        let current_scope = ctx.scope.locals.last().cloned().unwrap_or_default();
        let captured_vars = closure::find_captured_variables(
            body,
            &parent_locals_snapshot,
            &param_names,
            &current_scope,
        );

        // Создаем локальные слоты для захваченных переменных (перед параметрами)
        let mut captured_vars_info = Vec::new();

        for var_name in &captured_vars {
            let local_slot_index = ctx.scope.declare_local(var_name);

            // Находим slot index в родительской функции и глубину предка
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

        // Объявляем параметры как локальные переменные (после захваченных переменных)
        for param in params {
            ctx.scope.declare_local(&param.name);
            ctx.record_bound_name(&param.name);
        }

        // Имя функции в собственной области — для рекурсивных вызовов
        let self_slot = ctx.scope.declare_local(name);
        ctx.record_bound_name(name);
        emit_function_value_to_slot(ctx, function_index, self_slot, *line);

        // Компилируем тело функции
        for stmt in body {
            stmt::compile_stmt(ctx, stmt, true)?;
        }

        // Если функция не вернула значение явно, добавляем неявный return
        // Используем текущую строку (последнего statement) для лучшей диагностики ошибок
        if ctx.chunk.code.is_empty() || !matches!(ctx.chunk.code.last(), Some(OpCode::Return)) {
            ctx.chunk.write_with_line(OpCode::Return, *ctx.current_line);
        }

        // Заканчиваем область видимости функции
        ctx.scope.end_scope();

        // Эталонный алгоритм апгрейда jump-инструкций: стабилизация layout и финализация
        ctx.labels.stabilize_layout(&mut *ctx.chunk, *line)?;
        ctx.labels.finalize_jumps(&mut *ctx.chunk, *line)?;

        // Сохраняем скомпилированную функцию
        let function_chunk = std::mem::replace(&mut *ctx.chunk, saved_chunk);
        function.chunk = function_chunk;
        function.captured_vars = captured_vars_info;
        ctx.functions[function_index] = function.clone();

        // Восстанавливаем состояние компилятора
        *ctx.exception_handlers = saved_exception_handlers;
        *ctx.error_type_table = saved_error_type_table;
        ctx.current_function = saved_function;
        ctx.scope.local_count = saved_local_count;
        ctx.scope.restore_method_object_temps(saved_method_object_temps);

        // Восстанавливаем состояние меток после компиляции функции
        ctx.labels.label_counter = saved_label_counter;
        ctx.labels.labels = saved_labels;
        ctx.labels.pending_jumps = saved_pending_jumps;
        ctx.labels.pending_for_range = saved_pending_for_range;

        // Сохраняем функцию: вложенные — в локальный слот родителя, top-level — в глобал
        if let Some(slot) = parent_fn_slot {
            emit_function_value_to_slot(ctx, function_index, slot, *line);
        } else if name != "__main__" {
            let global_index = *ctx.scope.globals.get(name).unwrap();
            ctx.chunk.global_names.insert(global_index, name.clone());
            let constant_index = ctx.chunk.add_constant(Value::Function(function_index));
            ctx.chunk
                .write_with_line(OpCode::Constant(constant_index), *line);
            ctx.chunk
                .write_with_line(OpCode::StoreGlobal(global_index), *line);
        }

        Ok(())
    } else {
        Err(LangError::ParseError {
            message: "Expected Function statement".to_string(),
            line: stmt.line(),
            file: None,
        })
    }
}

pub fn compile_stream_function(ctx: &mut CompilationContext, stmt: &Stmt) -> Result<(), LangError> {
    if let Stmt::StreamFunction {
        name,
        params,
        return_type,
        body,
        is_cached,
        route,
        ws_route,
        line,
    } = stmt
    {
        if *is_cached {
            return Err(LangError::ParseError {
                message: "@cache is not supported on stream fn".to_string(),
                line: *line,
                file: None,
            });
        }
        *ctx.current_line = *line;

        let function_index = resolve_function_index_by_name(ctx, name, *line)?;

        let mut function = ctx.functions[function_index].clone();
        function.arity = params.len();
        function.is_cached = false;
        function.is_stream = true;
        if let Some((ref method, ref path)) = route {
            function.route_method = Some(method.clone());
            function.route_path = Some(path.clone());
        }
        if let Some(ref ws_type) = ws_route {
            function.ws_route_type = Some(ws_type.clone());
        }

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

        ctx.functions[function_index] = function.clone();

        let parent_locals_snapshot: Vec<std::collections::HashMap<String, usize>> =
            ctx.scope.locals.iter().map(|scope| scope.clone()).collect();

        let ancestor_bindings = closure::ancestor_bindings_for_function_check(
            &parent_locals_snapshot,
            ctx.current_function.is_some(),
        );
        closure::check_illegal_outer_assignments_in_function_body(
            body,
            &ancestor_bindings,
            &param_names,
        )?;

        let function_chunk_clone = function.chunk.clone();
        let saved_chunk = std::mem::replace(&mut *ctx.chunk, function_chunk_clone);
        let saved_exception_handlers = ctx.exception_handlers.clone();
        let saved_error_type_table = ctx.error_type_table.clone();
        let saved_function = ctx.current_function;
        let saved_local_count = ctx.scope.local_count;
        let saved_method_object_temps = ctx.scope.snapshot_method_object_temps();

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
        ctx.scope.reset_method_object_temps();
        ctx.exception_handlers.clear();
        ctx.error_type_table.clear();

        ctx.scope.begin_scope();

        let current_scope = ctx.scope.locals.last().cloned().unwrap_or_default();
        let parent_locals_for_capture: &[std::collections::HashMap<String, usize>] =
            if saved_function.is_some() {
                &parent_locals_snapshot
            } else {
                &[]
            };
        let captured_vars = closure::find_captured_variables(
            body,
            parent_locals_for_capture,
            &param_names,
            &current_scope,
        );
        if !captured_vars.is_empty() {
            return Err(LangError::ParseError {
                message: "stream fn with captured variables is not supported yet".to_string(),
                line: *line,
                file: None,
            });
        }

        for param in params {
            ctx.scope.declare_local(&param.name);
            ctx.record_bound_name(&param.name);
        }

        stream_fn::compile_stream_body(ctx, body, *line)?;

        ctx.scope.end_scope();

        ctx.labels.stabilize_layout(&mut *ctx.chunk, *line)?;
        ctx.labels.finalize_jumps(&mut *ctx.chunk, *line)?;

        let function_chunk = std::mem::replace(&mut *ctx.chunk, saved_chunk);
        function.chunk = function_chunk;
        function.captured_vars = Vec::new();
        ctx.functions[function_index] = function.clone();

        *ctx.exception_handlers = saved_exception_handlers;
        *ctx.error_type_table = saved_error_type_table;
        ctx.current_function = saved_function;
        ctx.scope.local_count = saved_local_count;
        ctx.scope.restore_method_object_temps(saved_method_object_temps);

        ctx.labels.label_counter = saved_label_counter;
        ctx.labels.labels = saved_labels;
        ctx.labels.pending_jumps = saved_pending_jumps;
        ctx.labels.pending_for_range = saved_pending_for_range;

        let global_index = *ctx.scope.globals.get(name).unwrap();
        ctx.chunk.global_names.insert(global_index, name.clone());

        if name != "__main__" {
            let constant_index = ctx.chunk.add_constant(Value::Function(function_index));
            ctx.chunk
                .write_with_line(OpCode::Constant(constant_index), *line);
            ctx.chunk
                .write_with_line(OpCode::StoreGlobal(global_index), *line);
        }

        Ok(())
    } else {
        Err(LangError::ParseError {
            message: "Expected StreamFunction statement".to_string(),
            line: stmt.line(),
            file: None,
        })
    }
}
