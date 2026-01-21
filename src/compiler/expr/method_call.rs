/// Компиляция вызовов методов

use crate::parser::ast::{Expr, Arg};
use crate::bytecode::OpCode;
use crate::common::error::LangError;
use crate::common::value::Value;
use crate::compiler::context::CompilationContext;
use crate::compiler::expr;
use crate::compiler::args;

pub fn compile_method_call(ctx: &mut CompilationContext, expr: &Expr) -> Result<(), LangError> {
    if let Expr::MethodCall { object, method, args: call_args, line } = expr {
        *ctx.current_line = *line;
        
        // Компилируем объект
        expr::compile_expr(ctx, object)?;
        
        // Специальная обработка для метода clone()
        if method == "clone" {
            return compile_clone_method(ctx, call_args, *line);
        }
        
        // Специальная обработка для метода suffixes
        if method == "suffixes" {
            return compile_suffixes_method(ctx, call_args, *line);
        }
        
        // Специальная обработка для JOIN методов
        if matches!(method.as_str(), "inner_join" | "left_join" | "right_join" | "full_join" | 
                   "cross_join" | "semi_join" | "anti_join" | "zip_join" | "asof_join" | 
                   "apply_join" | "join_on") {
            return compile_join_method(ctx, method, call_args, *line);
        }
        
        // Общий случай: метод может быть нативной функцией или обычной функцией в объекте
        compile_generic_method(ctx, method, call_args, *line)
    } else {
        Err(LangError::ParseError {
            message: "Expected MethodCall expression".to_string(),
            line: expr.line(),
        })
    }
}

fn compile_clone_method(
    ctx: &mut CompilationContext,
    args: &[Arg],
    line: usize,
) -> Result<(), LangError> {
    // Для clone() не нужны аргументы
    if !args.is_empty() {
        return Err(LangError::ParseError {
            message: "clone() method takes no arguments".to_string(),
            line,
        });
    }
    // Используем специальный opcode для клонирования
    ctx.chunk.write_with_line(OpCode::Clone, line);
    Ok(())
}

fn compile_suffixes_method(
    ctx: &mut CompilationContext,
    args: &[Arg],
    line: usize,
) -> Result<(), LangError> {
    // Метод suffixes для применения суффиксов к колонкам таблицы
    // Проверяем количество аргументов (должно быть 2)
    if args.len() != 2 {
        return Err(LangError::ParseError {
            message: format!("suffixes() method expects 2 arguments (left_suffix, right_suffix), got {}", args.len()),
            line,
        });
    }
    
    // Сохраняем объект (таблицу) во временную локальную переменную
    let temp_object_slot = ctx.scope.declare_local("__method_object");
    ctx.chunk.write_with_line(OpCode::StoreLocal(temp_object_slot), line);
    
    // Компилируем аргументы в нормальном порядке (left_suffix, right_suffix)
    for arg in args {
        match arg {
            Arg::Positional(expr) => expr::compile_expr(ctx, expr)?,
            Arg::Named { value, .. } => expr::compile_expr(ctx, value)?,
        }
    }
    
    // Переставляем аргументы для правильного порядка: table, left_suffix, right_suffix
    // Удаляем только аргументы со стека (объект уже сохранен в локальной переменной)
    for _ in 0..args.len() {
        ctx.chunk.write_with_line(OpCode::Pop, line);
    }
    
    // Загружаем в правильном порядке: table, left_suffix, right_suffix
    ctx.chunk.write_with_line(OpCode::LoadLocal(temp_object_slot), line);
    // Компилируем аргументы заново
    for arg in args {
        match arg {
            Arg::Positional(expr) => expr::compile_expr(ctx, expr)?,
            Arg::Named { value, .. } => expr::compile_expr(ctx, value)?,
        }
    }
    
    // Находим индекс функции table_suffixes и загружаем её на стек
    if let Some(&function_index) = ctx.scope.globals.get("table_suffixes") {
        ctx.chunk.write_with_line(OpCode::LoadGlobal(function_index), line);
        ctx.chunk.write_with_line(OpCode::Call(3), line);
        Ok(())
    } else {
        Err(LangError::ParseError {
            message: "Function 'table_suffixes' not found".to_string(),
            line,
        })
    }
}

fn compile_join_method(
    ctx: &mut CompilationContext,
    method: &str,
    args: &[Arg],
    line: usize,
) -> Result<(), LangError> {
    // JOIN методы для таблиц
    // Сохраняем объект во временную локальную переменную
    let temp_object_slot = ctx.scope.declare_local("__method_object");
    ctx.chunk.write_with_line(OpCode::StoreLocal(temp_object_slot), line);
    
    // Компилируем аргументы в нормальном порядке
    for arg in args {
        match arg {
            Arg::Positional(expr) => expr::compile_expr(ctx, expr)?,
            Arg::Named { value, .. } => expr::compile_expr(ctx, value)?,
        }
    }
    
    // Загружаем объект обратно (он должен быть первым аргументом)
    ctx.chunk.write_with_line(OpCode::LoadLocal(temp_object_slot), line);
    
    // Определяем имя функции для вызова
    let function_name = match method {
        "inner_join" => "inner_join",
        "left_join" => "left_join",
        "right_join" => "right_join",
        "full_join" => "full_join",
        "cross_join" => "cross_join",
        "semi_join" => "semi_join",
        "anti_join" => "anti_join",
        "zip_join" => "zip_join",
        "asof_join" => "asof_join",
        "apply_join" => "apply_join",
        "join_on" => "join_on",
        _ => unreachable!(),
    };
    
    // Находим индекс функции
    if let Some(&function_index) = ctx.scope.globals.get(function_name) {
        // Перекомпилируем аргументы в обратном порядке
        // Удаляем текущие аргументы со стека (кроме объекта)
        for _ in 0..args.len() {
            ctx.chunk.write_with_line(OpCode::Pop, line);
        }
        
        // Компилируем аргументы в обратном порядке
        for arg in args.iter().rev() {
            match arg {
                Arg::Positional(expr) => expr::compile_expr(ctx, expr)?,
                Arg::Named { value, .. } => expr::compile_expr(ctx, value)?,
            }
        }
        
        // Загружаем объект обратно
        ctx.chunk.write_with_line(OpCode::LoadLocal(temp_object_slot), line);
        
        // Загружаем функцию на стек
        ctx.chunk.write_with_line(OpCode::LoadGlobal(function_index), line);
        
        // Вызываем функцию с количеством аргументов (object + args)
        ctx.chunk.write_with_line(OpCode::Call(args.len() + 1), line);
        Ok(())
    } else {
        Err(LangError::ParseError {
            message: format!("Function '{}' not found", function_name),
            line,
        })
    }
}

fn compile_generic_method(
    ctx: &mut CompilationContext,
    method: &str,
    args: &[Arg],
    line: usize,
) -> Result<(), LangError> {
    // Сохраняем объект во временную переменную
    let temp_object_slot = ctx.scope.declare_local("__method_object");
    ctx.chunk.write_with_line(OpCode::StoreLocal(temp_object_slot), line);
    
    // Проверяем, является ли это методом объекта (например, axis.imshow)
    // или функцией модуля (например, ml.load_mnist)
    let is_axis_method = matches!(method, "imshow" | "set_title" | "axis");
    let is_nn_method = matches!(method, "device" | "get_device" | "save" | "train" | "train_sh");
    let is_layer_method = matches!(method, "freeze" | "unfreeze");
    
    if is_nn_method {
        compile_nn_method(ctx, method, args, temp_object_slot, line)
    } else if is_axis_method {
        compile_axis_method(ctx, method, args, temp_object_slot, line)
    } else if is_layer_method {
        compile_layer_method(ctx, method, args, temp_object_slot, line)
    } else {
        compile_module_method(ctx, method, args, temp_object_slot, line)
    }
}

fn compile_nn_method(
    ctx: &mut CompilationContext,
    method: &str,
    args: &[Arg],
    temp_object_slot: usize,
    line: usize,
) -> Result<(), LangError> {
    // Для методов device, get_device и save на NeuralNetwork, вызываем соответствующие нативные функции
    // Загружаем объект первым
    ctx.chunk.write_with_line(OpCode::LoadLocal(temp_object_slot), line);
    
    // Определяем имя функции в ml модуле
    let function_name = match method {
        "device" => "nn_set_device",
        "get_device" => "nn_get_device",
        "save" => "nn_save",
        "train" => "nn_train",
        "train_sh" => "nn_train_sh",
        _ => {
            return Err(LangError::ParseError {
                message: format!("Unknown NeuralNetwork method: {}", method),
                line,
            });
        }
    };
    
    // Определяем фактическое количество аргументов для Call инструкции
    let actual_arg_count = if method == "get_device" {
        if !args.is_empty() {
            return Err(LangError::ParseError {
                message: "get_device() takes no arguments".to_string(),
                line,
            });
        }
        0
    } else if method == "train" {
        // Разрешаем именованные аргументы для train метода
        let resolved_args = args::resolve_function_args("nn_train", args, None, line)?;
        
        // Компилируем разрешенные аргументы в правильном порядке
        for arg in &resolved_args {
            match arg {
                Arg::Positional(expr) => expr::compile_expr(ctx, expr)?,
                Arg::Named { value, .. } => expr::compile_expr(ctx, value)?,
            }
        }
        
        resolved_args.len()
    } else if method == "train_sh" {
        // Разрешаем именованные аргументы для train_sh метода
        let resolved_args = args::resolve_function_args("nn_train_sh", args, None, line)?;
        
        // Компилируем разрешенные аргументы в правильном порядке
        for arg in &resolved_args {
            match arg {
                Arg::Positional(expr) => expr::compile_expr(ctx, expr)?,
                Arg::Named { value, .. } => expr::compile_expr(ctx, value)?,
            }
        }
        
        resolved_args.len()
    } else {
        // Для device и save компилируем аргументы
        if args.len() != 1 {
            return Err(LangError::ParseError {
                message: format!("{}() takes exactly 1 argument", method),
                line,
            });
        }
        for arg in args {
            match arg {
                Arg::Positional(expr) => expr::compile_expr(ctx, expr)?,
                Arg::Named { value, .. } => expr::compile_expr(ctx, value)?,
            }
        }
        args.len()
    };
    
    // Загружаем функцию из ml модуля
    if let Some(&ml_index) = ctx.scope.globals.get("ml") {
        ctx.chunk.write_with_line(OpCode::LoadGlobal(ml_index), line);
        let method_name_index = ctx.chunk.add_constant(Value::String(function_name.to_string()));
        ctx.chunk.write_with_line(OpCode::Constant(method_name_index), line);
        ctx.chunk.write_with_line(OpCode::GetArrayElement, line);
        ctx.chunk.write_with_line(OpCode::Call(actual_arg_count + 1), line);
        Ok(())
    } else {
        Err(LangError::ParseError {
            message: "ml module not found".to_string(),
            line,
        })
    }
}

fn compile_axis_method(
    ctx: &mut CompilationContext,
    method: &str,
    args: &[Arg],
    temp_object_slot: usize,
    line: usize,
) -> Result<(), LangError> {
    // Для методов Axis компилируем аргументы сразу
    for arg in args {
        match arg {
            Arg::Positional(expr) => expr::compile_expr(ctx, expr)?,
            Arg::Named { value, .. } => expr::compile_expr(ctx, value)?,
        }
    }
    
    // Удаляем текущие аргументы со стека
    for _ in 0..args.len() {
        ctx.chunk.write_with_line(OpCode::Pop, line);
    }
    
    // Загружаем объект первым
    ctx.chunk.write_with_line(OpCode::LoadLocal(temp_object_slot), line);
    
    // Компилируем аргументы в нормальном порядке (они будут после объекта)
    for arg in args {
        match arg {
            Arg::Positional(expr) => expr::compile_expr(ctx, expr)?,
            Arg::Named { value, .. } => expr::compile_expr(ctx, value)?,
        }
    }
    
    // Получаем свойство объекта по имени метода
    ctx.chunk.write_with_line(OpCode::LoadLocal(temp_object_slot), line);
    let method_name_index = ctx.chunk.add_constant(Value::String(method.to_string()));
    ctx.chunk.write_with_line(OpCode::Constant(method_name_index), line);
    ctx.chunk.write_with_line(OpCode::GetArrayElement, line);
    
    // Вызываем метод
    ctx.chunk.write_with_line(OpCode::Call(args.len() + 1), line);
    Ok(())
}

fn compile_layer_method(
    ctx: &mut CompilationContext,
    method: &str,
    args: &[Arg],
    temp_object_slot: usize,
    line: usize,
) -> Result<(), LangError> {
    // Для методов Layer (freeze, unfreeze) компилируем аргументы сразу
    // Эти методы не принимают аргументов, кроме самого layer
    if !args.is_empty() {
        return Err(LangError::ParseError {
            message: format!("layer.{}() takes no arguments", method),
            line,
        });
    }
    
    // Загружаем объект первым
    ctx.chunk.write_with_line(OpCode::LoadLocal(temp_object_slot), line);
    
    // Получаем свойство объекта по имени метода
    ctx.chunk.write_with_line(OpCode::LoadLocal(temp_object_slot), line);
    let method_name_index = ctx.chunk.add_constant(Value::String(method.to_string()));
    ctx.chunk.write_with_line(OpCode::Constant(method_name_index), line);
    ctx.chunk.write_with_line(OpCode::GetArrayElement, line);
    
    // При вызове Call(1): pop 1 раз, reverse -> функция получает [object] в правильном порядке
    ctx.chunk.write_with_line(OpCode::Call(1), line);
    Ok(())
}

fn compile_module_method(
    ctx: &mut CompilationContext,
    method: &str,
    args: &[Arg],
    temp_object_slot: usize,
    line: usize,
) -> Result<(), LangError> {
    // Для функций модулей: получаем метод, не добавляем объект
    // Пытаемся разрешить именованные аргументы
    let resolved_args = match args::resolve_function_args(method, args, None, line) {
        Ok(resolved) => resolved,
        Err(e) => {
            // Проверяем, является ли это ошибкой "not supported"
            let error_msg = match &e {
                LangError::ParseError { message, .. } => message,
                _ => "",
            };
            
            if error_msg.contains("not supported") || error_msg.contains("Named arguments are not supported") {
                // Fallback: компилируем аргументы как есть
                args.iter().map(|a| match a {
                    Arg::Positional(e) => Arg::Positional(e.clone()),
                    Arg::Named { value, .. } => Arg::Positional(value.clone()),
                }).collect()
            } else {
                return Err(e);
            }
        }
    };
    
    // Компилируем разрешенные аргументы
    for arg in &resolved_args {
        match arg {
            Arg::Positional(expr) => expr::compile_expr(ctx, expr)?,
            Arg::Named { .. } => {
                return Err(LangError::ParseError {
                    message: format!("Unexpected named argument in resolved args for method '{}'", method),
                    line,
                });
            }
        }
    }
    
    // Загружаем объект обратно для получения метода
    ctx.chunk.write_with_line(OpCode::LoadLocal(temp_object_slot), line);
    
    // Получаем свойство объекта по имени метода
    let method_name_index = ctx.chunk.add_constant(Value::String(method.to_string()));
    ctx.chunk.write_with_line(OpCode::Constant(method_name_index), line);
    ctx.chunk.write_with_line(OpCode::GetArrayElement, line);
    
    // Вызываем функцию без объекта как первого аргумента
    ctx.chunk.write_with_line(OpCode::Call(resolved_args.len()), line);
    Ok(())
}

