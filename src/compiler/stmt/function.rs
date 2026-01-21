/// Компиляция function statements

use crate::parser::ast::Stmt;
use crate::bytecode::{OpCode, CapturedVar};
use crate::common::error::LangError;
use crate::common::value::Value;
use crate::compiler::context::CompilationContext;
use crate::compiler::constant_fold;
use crate::compiler::closure;
use crate::compiler::stmt;

pub fn compile_function(ctx: &mut CompilationContext, stmt: &Stmt) -> Result<(), LangError> {
    if let Stmt::Function { name, params, body, is_cached, line } = stmt {
        *ctx.current_line = *line;
        
        // Находим индекс функции (она уже объявлена в первом проходе)
        let function_index = ctx.function_names.iter()
            .position(|n| n == name)
            .ok_or_else(|| LangError::ParseError {
                message: format!("Function '{}' not found in forward declarations", name),
                line: *line,
            })?;
        
        // Получаем функцию и обновляем количество параметров и флаг кэширования
        let mut function = ctx.functions[function_index].clone();
        function.arity = params.len();
        function.is_cached = *is_cached;
        
        // Сохраняем имена параметров и вычисляем значения по умолчанию
        let mut param_names = Vec::new();
        let mut default_values = Vec::new();
        
        for param in params.iter() {
            param_names.push(param.name.clone());
            
            // Вычисляем значение по умолчанию во время компиляции
            if let Some(ref default_expr) = param.default_value {
                match constant_fold::evaluate_constant_expr(default_expr) {
                    Ok(Some(constant_value)) => {
                        default_values.push(Some(constant_value));
                    }
                    Ok(None) => {
                        return Err(LangError::ParseError {
                            message: format!(
                                "Default value for parameter '{}' must be a constant expression",
                                param.name
                            ),
                            line: default_expr.line(),
                        });
                    }
                    Err(e) => {
                        return Err(e);
                    }
                }
            } else {
                default_values.push(None);
            }
        }
        
        function.param_names = param_names.clone();
        function.default_values = default_values;
        
        // Если кэш включен, но еще не инициализирован, инициализируем его
        if *is_cached && function.cache.is_none() {
            use std::rc::Rc;
            use std::cell::RefCell;
            use crate::bytecode::function::FnCache;
            function.cache = Some(Rc::new(RefCell::new(FnCache::new())));
        }
        
        // ВАЖНО: Сохраняем сигнатуру функции ДО компиляции тела
        ctx.functions[function_index] = function.clone();
        
        // Сохраняем текущие локальные области видимости для доступа к переменным родительских функций
        let parent_locals_snapshot: Vec<std::collections::HashMap<String, usize>> = ctx.scope.locals.iter()
            .map(|scope| scope.clone())
            .collect();
        
        // Компилируем тело функции в chunk функции
        let function_chunk_clone = function.chunk.clone();
        let saved_chunk = std::mem::replace(&mut *ctx.chunk, function_chunk_clone);
        let saved_exception_handlers = ctx.exception_handlers.clone();
        let saved_error_type_table = ctx.error_type_table.clone();
        let saved_function = ctx.current_function;
        let saved_local_count = ctx.scope.local_count;
        
        // ВАЖНО: Сохраняем состояние меток перед компиляцией функции
        // и очищаем метки, чтобы предотвратить переиспользование меток между функциями
        let saved_label_counter = ctx.labels.label_counter;
        let saved_labels = ctx.labels.labels.clone();
        let saved_pending_jumps = ctx.labels.pending_jumps.clone();
        ctx.labels.label_counter = 0;
        ctx.labels.labels.clear();
        ctx.labels.pending_jumps.clear();
        
        ctx.current_function = Some(function_index);
        ctx.scope.local_count = 0;
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
                    message: format!("Captured variable '{}' not found in parent scopes", var_name),
                    line: *line,
                });
            }
            
            let parent_slot = parent_slot_index.unwrap();
            
            captured_vars_info.push(CapturedVar {
                name: var_name.clone(),
                parent_slot_index: parent_slot,
                local_slot_index,
                ancestor_depth,
            });
        }
        
        // Объявляем параметры как локальные переменные (после захваченных переменных)
        for param in params {
            ctx.scope.declare_local(&param.name);
        }
        
        // Компилируем тело функции
        for stmt in body {
            stmt::compile_stmt(ctx, stmt, true)?;
        }
        
        // Если функция не вернула значение явно, добавляем неявный return
        // Используем текущую строку (последнего statement) для лучшей диагностики ошибок
        if ctx.chunk.code.is_empty() || 
           !matches!(ctx.chunk.code.last(), Some(OpCode::Return)) {
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
        
        // Восстанавливаем состояние меток после компиляции функции
        ctx.labels.label_counter = saved_label_counter;
        ctx.labels.labels = saved_labels;
        ctx.labels.pending_jumps = saved_pending_jumps;
        
        // Сохраняем функцию в глобальную таблицу (уже сделано в первом проходе)
        let global_index = *ctx.scope.globals.get(name).unwrap();
        
        // Сохраняем имя глобальной переменной для использования в JOIN
        ctx.chunk.global_names.insert(global_index, name.clone());
        
        // Сохраняем функцию как константу
        let constant_index = ctx.chunk.add_constant(Value::Function(function_index));
        
        // Сохраняем функцию в глобальную переменную
        ctx.chunk.write_with_line(OpCode::Constant(constant_index), *line);
        ctx.chunk.write_with_line(OpCode::StoreGlobal(global_index), *line);
        
        Ok(())
    } else {
        Err(LangError::ParseError {
            message: "Expected Function statement".to_string(),
            line: stmt.line(),
        })
    }
}

