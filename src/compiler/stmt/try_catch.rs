/// Компиляция try/catch/else statements

use crate::parser::ast::Stmt;
use crate::bytecode::OpCode;
use crate::common::error::LangError;
use crate::compiler::context::{CompilationContext, ExceptionHandler};
use crate::compiler::stmt;

pub fn compile_try_catch(ctx: &mut CompilationContext, stmt: &Stmt) -> Result<(), LangError> {
    if let Stmt::Try { try_block, catch_blocks, else_block, finally_block, line } = stmt {
        *ctx.current_line = *line;
        
        // Сохраняем текущую высоту стека
        let stack_height = ctx.chunk.code.len();
        
        // Создаем обработчик исключений
        let mut handler = ExceptionHandler {
            catch_ips: Vec::new(),
            error_types: Vec::new(),
            error_var_slots: Vec::new(),
            else_ip: None,
            finally_ip: None,
            stack_height,
        };
        
        // Создаем метку для finally блока (если есть)
        let finally_label = if finally_block.is_some() {
            Some(ctx.labels.create_label())
        } else {
            None
        };
        
        // Начинаем новую область видимости для try блока
        ctx.scope.begin_scope();
        
        // Генерируем BeginTry (пока без индекса обработчика, патчим позже)
        let begin_try_ip = ctx.chunk.code.len();
        ctx.chunk.write_with_line(OpCode::BeginTry(0), *line); // Временное значение
        
        // Компилируем try блок
        for stmt in try_block {
            stmt::compile_stmt(ctx, stmt, true)?;
        }
        
        // Генерируем EndTry
        ctx.chunk.write_with_line(OpCode::EndTry, *line);
        
        // Завершаем область видимости try блока
        ctx.scope.end_scope();
        
        // Создаем метки для try/catch блоков
        let after_try_label = ctx.labels.create_label();
        let after_catch_label = if else_block.is_some() {
            Some(ctx.labels.create_label())
        } else {
            None
        };
        
        // Если есть finally, переходим к нему после успешного try
        // Иначе переходим к метке после try (начало catch блоков)
        if let Some(ref finally_label) = finally_label {
            ctx.labels.emit_jump(ctx.chunk, *ctx.current_line, false, *finally_label)?;
        } else {
            ctx.labels.emit_jump(ctx.chunk, *ctx.current_line, false, after_try_label)?;
        }
        
        // Компилируем catch блоки
        for catch_block in catch_blocks {
            // Сохраняем IP начала catch блока
            let catch_ip = ctx.chunk.code.len();
            handler.catch_ips.push(catch_ip);
            
            // Определяем тип ошибки
            let error_type_index = if let Some(ref error_type_name) = catch_block.error_type {
                Some(ctx.get_error_type_index(error_type_name))
            } else {
                None // catch всех
            };
            handler.error_types.push(error_type_index);
            
            // Если есть переменная ошибки, создаем для неё слот
            let error_var_slot = if let Some(ref error_var) = catch_block.error_var {
                ctx.scope.begin_scope();
                let slot = ctx.scope.declare_local(error_var);
                Some(slot)
            } else {
                None
            };
            handler.error_var_slots.push(error_var_slot);
            
            // Генерируем Catch опкод
            ctx.chunk.write_with_line(
                OpCode::Catch(error_type_index),
                catch_block.line,
            );
            
            // Компилируем тело catch блока
            for stmt in &catch_block.body {
                stmt::compile_stmt(ctx, stmt, true)?;
            }
            
            // Генерируем EndCatch
            ctx.chunk.write_with_line(OpCode::EndCatch, catch_block.line);
            
            // Завершаем область видимости catch блока
            if error_var_slot.is_some() {
                ctx.scope.end_scope();
            }
            
            // Если есть finally, переходим к нему после catch
            // Иначе переходим к else или концу
            if let Some(ref finally_label) = finally_label {
                ctx.labels.emit_jump(ctx.chunk, *ctx.current_line, false, *finally_label)?;
            } else if let Some(ref after_catch_label) = after_catch_label {
                ctx.labels.emit_jump(ctx.chunk, *ctx.current_line, false, *after_catch_label)?;
            }
        }
        
        // Помечаем метку после try (начало catch блоков)
        ctx.labels.mark_label(after_try_label, ctx.chunk.code.len());
        
        // Компилируем else блок (если есть)
        if let Some(else_block) = else_block {
            let else_ip = ctx.chunk.code.len();
            handler.else_ip = Some(else_ip);
            
            ctx.scope.begin_scope();
            for stmt in else_block {
                stmt::compile_stmt(ctx, stmt, true)?;
            }
            ctx.scope.end_scope();
            
            // Если есть finally, переходим к нему после else
            if let Some(ref finally_label) = finally_label {
                ctx.labels.emit_jump(ctx.chunk, *ctx.current_line, false, *finally_label)?;
            }
            
            // Помечаем метку после catch блоков (если есть else и нет finally)
            if finally_label.is_none() {
                if let Some(ref after_catch_label) = after_catch_label {
                    ctx.labels.mark_label(*after_catch_label, ctx.chunk.code.len());
                }
            }
        }
        
        // Компилируем finally блок (если есть)
        if let Some(ref finally_block) = finally_block {
            let finally_ip = ctx.chunk.code.len();
            handler.finally_ip = Some(finally_ip);
            
            // Помечаем метку finally
            if let Some(ref finally_label) = finally_label {
                ctx.labels.mark_label(*finally_label, ctx.chunk.code.len());
            }
            
            ctx.scope.begin_scope();
            for stmt in finally_block {
                stmt::compile_stmt(ctx, stmt, true)?;
            }
            ctx.scope.end_scope();
        }
        
        // Добавляем обработчик в стек компилятора
        let handler_index = ctx.exception_handlers.len();
        ctx.exception_handlers.push(handler.clone());
        
        // Сохраняем обработчик в chunk
        let handler_info = crate::bytecode::ExceptionHandlerInfo {
            catch_ips: handler.catch_ips.clone(),
            error_types: handler.error_types.clone(),
            error_var_slots: handler.error_var_slots.clone(),
            else_ip: handler.else_ip,
            finally_ip: handler.finally_ip,
            stack_height: handler.stack_height,
        };
        ctx.chunk.exception_handlers.push(handler_info);
        
        // Копируем таблицу типов ошибок в chunk (если еще не скопирована)
        if ctx.chunk.error_type_table.is_empty() {
            ctx.chunk.error_type_table = ctx.error_type_table.clone();
        }
        
        // Патчим BeginTry с правильным индексом обработчика
        if let Some(OpCode::BeginTry(_)) = ctx.chunk.code.get_mut(begin_try_ip) {
            *ctx.chunk.code.get_mut(begin_try_ip).unwrap() = OpCode::BeginTry(handler_index);
        }
        
        // Генерируем PopExceptionHandler в конце
        ctx.chunk.write_with_line(OpCode::PopExceptionHandler, *line);
        
        Ok(())
    } else {
        Err(LangError::ParseError {
            message: "Expected Try statement".to_string(),
            line: stmt.line(),
        })
    }
}

