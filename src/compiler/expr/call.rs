/// Компиляция вызовов функций

use crate::parser::ast::{Expr, Arg};
use crate::bytecode::OpCode;
use crate::common::error::LangError;
use crate::common::value::Value;
use crate::compiler::context::CompilationContext;
use crate::compiler::expr;
use crate::compiler::args;

pub fn compile_call(ctx: &mut CompilationContext, expr: &Expr) -> Result<(), LangError> {
    if let Expr::Call { name, args: call_args, line } = expr {
        *ctx.current_line = *line;
        
        // Находим функцию для получения информации о параметрах
        let function_info = if let Some(function_index) = ctx.function_names.iter().position(|n| n == name) {
            Some((function_index, &ctx.functions[function_index]))
        } else {
            None
        };
        
        // Разрешаем аргументы: именованные -> позиционные, применяем значения по умолчанию
        let resolved_args = args::resolve_function_args(name, call_args, function_info, *line)?;
        
        // Специальная обработка для isinstance: преобразуем идентификаторы типов в строки
        let processed_args = if name == "isinstance" && resolved_args.len() >= 2 {
            let mut new_args = resolved_args.clone();
            if let Arg::Positional(Expr::Variable { name: type_name, .. }) = &resolved_args[1] {
                let type_names = vec!["int", "str", "bool", "array", "null", "num", "float"];
                if type_names.contains(&type_name.as_str()) {
                    new_args[1] = Arg::Positional(Expr::Literal {
                        value: Value::String(type_name.clone()),
                        line: match &resolved_args[1] {
                            Arg::Positional(e) => e.line(),
                            Arg::Named { value, .. } => value.line(),
                        },
                    });
                }
            }
            new_args
        } else {
            resolved_args
        };
        
        // Специальная обработка для функций, которые модифицируют первый аргумент in-place
        let in_place_functions = vec!["push", "reverse", "sort"];
        let should_assign_back = in_place_functions.contains(&name.as_str()) 
            && !processed_args.is_empty()
            && matches!(&processed_args[0], Arg::Positional(Expr::Variable { .. }));
        
        // Если нужно присвоить обратно, сохраняем имя переменной
        let var_name_to_assign = if should_assign_back {
            if let Arg::Positional(Expr::Variable { name: var_name, .. }) = &processed_args[0] {
                Some(var_name.clone())
            } else {
                None
            }
        } else {
            None
        };
        
        // Компилируем аргументы на стек
        for arg in &processed_args {
            match arg {
                Arg::Positional(expr) => {
                    expr::compile_expr(ctx, expr)?;
                }
                Arg::Named { value, .. } => {
                    // Именованные аргументы уже разрешены в позиционные
                    expr::compile_expr(ctx, value)?;
                }
            }
        }
        
        // Загружаем функцию: сначала проверяем локальные переменные,
        // затем пользовательские функции (они имеют приоритет над встроенными),
        // затем глобальные переменные (встроенные функции)
        if let Some(local_index) = ctx.scope.resolve_local(name) {
            // Локальная переменная содержит функцию
            ctx.chunk.write_with_line(OpCode::LoadLocal(local_index), *ctx.current_line);
        } else if let Some(function_index) = ctx.function_names.iter().position(|n| n == name) {
            // Пользовательская функция найдена - она имеет приоритет над встроенными
            let constant_index = ctx.chunk.add_constant(Value::Function(function_index));
            ctx.chunk.write_with_line(OpCode::Constant(constant_index), *ctx.current_line);
        } else if let Some(&global_index) = ctx.scope.globals.get(name) {
            // Глобальная переменная содержит функцию (встроенная функция)
            ctx.chunk.write_with_line(OpCode::LoadGlobal(global_index), *ctx.current_line);
        } else {
            // Функция не найдена - это может быть функция, импортированная через import *
            // Регистрируем её как глобальную переменную для разрешения во время выполнения
            let global_index = if let Some(&idx) = ctx.scope.globals.get(name) {
                idx
            } else {
                let idx = ctx.scope.globals.len();
                ctx.scope.globals.insert(name.clone(), idx);
                ctx.chunk.global_names.insert(idx, name.clone());
                idx
            };
            ctx.chunk.write_with_line(OpCode::LoadGlobal(global_index), *ctx.current_line);
        }
        
        // Вызываем функцию с количеством аргументов
        ctx.chunk.write_with_line(OpCode::Call(processed_args.len()), *line);
        
        // Если нужно присвоить результат обратно в переменную
        if let Some(var_name) = var_name_to_assign {
            // Определяем, глобальная или локальная переменная
            let is_local = !ctx.scope.locals.is_empty() && ctx.scope.resolve_local(&var_name).is_some();
            
            if is_local {
                // Локальная переменная
                if let Some(local_index) = ctx.scope.resolve_local(&var_name) {
                    ctx.chunk.write_with_line(OpCode::StoreLocal(local_index), *line);
                    ctx.chunk.write_with_line(OpCode::LoadLocal(local_index), *line);
                }
            } else {
                // Глобальная переменная
                let global_index = if let Some(&idx) = ctx.scope.globals.get(&var_name) {
                    idx
                } else {
                    let idx = ctx.scope.globals.len();
                    ctx.scope.globals.insert(var_name.clone(), idx);
                    idx
                };
                ctx.chunk.global_names.insert(global_index, var_name.clone());
                ctx.chunk.write_with_line(OpCode::StoreGlobal(global_index), *line);
                ctx.chunk.write_with_line(OpCode::LoadGlobal(global_index), *line);
            }
        }
        
        Ok(())
    } else {
        Err(LangError::ParseError {
            message: "Expected Call expression".to_string(),
            line: expr.line(),
        })
    }
}

