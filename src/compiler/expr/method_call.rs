/// Компиляция вызовов методов

use crate::debug_println;
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
        
        // Проверяем, является ли это методом класса ДО сохранения объекта
        // Методы классов имеют формат ClassName::method_<method_name>
        let method_pattern = format!("::method_{}", method);
        let class_method_name: Option<String> = ctx.function_names.iter()
            .find(|name| name.ends_with(&method_pattern))
            .map(|s| s.clone());
        
        debug_println!("[DEBUG compile_method_call] Проверяем метод '{}', паттерн: '{}', найденные методы класса: {:?}", 
            method, method_pattern, 
            ctx.function_names.iter().filter(|n| n.ends_with(&method_pattern)).collect::<Vec<_>>());
        
        if class_method_name.is_some() {
            debug_println!("[DEBUG compile_method_call] Метод '{}' распознан как метод класса: '{}'", method, class_method_name.as_ref().unwrap());
            // Это метод класса - обрабатываем его отдельно
            let start_ip = ctx.chunk.code.len();
            debug_println!("[DEBUG compile_method_call] Начало компиляции вызова метода класса '{}' на строке {}, начальный IP: {}", method, *line, start_ip);
            
            // Сохраняем объект во временную переменную
            let temp_object_slot = ctx.scope.declare_local("__method_object");
            ctx.chunk.write_with_line(OpCode::StoreLocal(temp_object_slot), *line);
            debug_println!("[DEBUG compile_method_call] После StoreLocal(temp_object_slot), IP: {}", ctx.chunk.code.len());
            
            // Компилируем аргументы в нормальном порядке
            for arg in call_args {
                match arg {
                    Arg::Positional(expr) => expr::compile_expr(ctx, expr)?,
                    Arg::Named { value, .. } => expr::compile_expr(ctx, value)?,
                    Arg::UnpackObject(expr) => expr::compile_expr(ctx, expr)?,
                }
            }
            debug_println!("[DEBUG compile_method_call] После компиляции аргументов, IP: {}", ctx.chunk.code.len());
            
            // Удаляем аргументы со стека временно
            for _ in 0..call_args.len() {
                ctx.chunk.write_with_line(OpCode::Pop, *line);
            }
            debug_println!("[DEBUG compile_method_call] После Pop аргументов, IP: {}", ctx.chunk.code.len());
            
            // Загружаем объект (this) первым
            ctx.chunk.write_with_line(OpCode::LoadLocal(temp_object_slot), *line);
            debug_println!("[DEBUG compile_method_call] После LoadLocal(temp_object_slot), IP: {}", ctx.chunk.code.len());
            
            // Компилируем аргументы снова в нормальном порядке
            for arg in call_args {
                match arg {
                    Arg::Positional(expr) => expr::compile_expr(ctx, expr)?,
                    Arg::Named { value, .. } => expr::compile_expr(ctx, value)?,
                    Arg::UnpackObject(expr) => expr::compile_expr(ctx, expr)?,
                }
            }
            debug_println!("[DEBUG compile_method_call] После повторной компиляции аргументов, IP: {}", ctx.chunk.code.len());
            
            // Находим функцию метода
            if let Some(function_index) = ctx.function_names.iter().position(|n| n == class_method_name.as_ref().unwrap()) {
                let constant_index = ctx.chunk.add_constant(Value::Function(function_index));
                ctx.chunk.write_with_line(OpCode::Constant(constant_index), *line);
                debug_println!("[DEBUG compile_method_call] После Constant(Function({})), IP: {}", function_index, ctx.chunk.code.len());
                
                let call_ip = ctx.chunk.code.len();
                ctx.chunk.write_with_line(OpCode::Call(call_args.len() + 1), *line);
                debug_println!("[DEBUG compile_method_call] Сгенерирован Call({}) на IP {} для метода класса '{}'", call_args.len() + 1, call_ip, method);
                
                // Логируем все инструкции, которые были сгенерированы
                debug_println!("[DEBUG compile_method_call] Сгенерированные инструкции для метода класса '{}' (IP {} - {}):", method, start_ip, ctx.chunk.code.len());
                for i in start_ip..ctx.chunk.code.len() {
                    debug_println!("[DEBUG compile_method_call]   IP {}: {:?}", i, ctx.chunk.code.get(i));
                }
                
                return Ok(());
            }
        }
        
        // Общий случай: метод может быть нативной функцией или обычной функцией в объекте
        debug_println!("[DEBUG compile_method_call] Метод '{}' не распознан как метод класса, используем compile_generic_method", method);
        compile_generic_method(ctx, object, method, call_args, *line)
    } else {
        Err(LangError::ParseError {
            message: "Expected MethodCall expression".to_string(),
            line: expr.line(),
            file: None,
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
            file: None,
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
            file: None,
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
            Arg::UnpackObject(expr) => expr::compile_expr(ctx, expr)?,
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
            Arg::UnpackObject(expr) => expr::compile_expr(ctx, expr)?,
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
            file: None,
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
            Arg::UnpackObject(expr) => expr::compile_expr(ctx, expr)?,
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
                Arg::UnpackObject(expr) => expr::compile_expr(ctx, expr)?,
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
            file: None,
        })
    }
}

fn compile_generic_method(
    ctx: &mut CompilationContext,
    object: &Expr,
    method: &str,
    args: &[Arg],
    line: usize,
) -> Result<(), LangError> {
    // Сохраняем объект во временную переменную
    let temp_object_slot = ctx.scope.declare_local("__method_object");
    ctx.chunk.write_with_line(OpCode::StoreLocal(temp_object_slot), line);
    
    // Проверяем, является ли это методом объекта (например, axis.imshow)
    // или функцией модуля (например, ml.load_mnist)
    // ml.add/sum/... are module functions (no receiver); cluster.add/array.sum need receiver.
    let is_ml_receiver = matches!(object, Expr::Variable { name, .. } if name == "ml");
    let is_axis_method = matches!(method, "imshow" | "set_title" | "axis");
    let is_nn_method = matches!(method, "device" | "get_device" | "save" | "train" | "train_sh");
    let is_layer_method = matches!(method, "freeze" | "unfreeze");
    let is_string_method = matches!(method, "lower" | "upper" | "isupper" | "islower" | "trim" | "split" | "join" | "contains");
    // ml.sum/mean are module functions (no receiver); array.sum/average need receiver.
    let is_array_method = if is_ml_receiver {
        matches!(method, "push" | "pop" | "unique" | "reverse" | "sort" | "average" | "count" | "any" | "all")
    } else {
        matches!(method, "push" | "pop" | "unique" | "reverse" | "sort" | "sum" | "average" | "count" | "any" | "all")
    };
    let is_db_receiver_method = if is_ml_receiver {
        matches!(method, "get" | "names" | "connect" | "execute" | "query" | "run")
    } else {
        matches!(method, "add" | "get" | "names" | "connect" | "execute" | "query" | "run")
    };
    
    if is_db_receiver_method {
        compile_db_receiver_method(ctx, method, args, temp_object_slot, line)
    } else if is_nn_method {
        compile_nn_method(ctx, method, args, temp_object_slot, line)
    } else if is_axis_method {
        compile_axis_method(ctx, method, args, temp_object_slot, line)
    } else if is_layer_method {
        compile_layer_method(ctx, method, args, temp_object_slot, line)
    } else if is_string_method {
        compile_string_method(ctx, method, args, temp_object_slot, line)
    } else if is_array_method {
        compile_array_method(ctx, method, args, temp_object_slot, line)
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
                file: None,
            });
        }
    };
    
    // Определяем фактическое количество аргументов для Call инструкции
    let actual_arg_count = if method == "get_device" {
        if !args.is_empty() {
            return Err(LangError::ParseError {
                message: "get_device() takes no arguments".to_string(),
                line,
                file: None,
            });
        }
        0
    } else if method == "train" {
        // Разрешаем именованные аргументы для train метода
        let resolved_args = args::resolve_function_args("nn_train", args, None, line, ctx.source_name)?;
        // Пропускаем первый аргумент (nn): объект уже на стеке через LoadLocal(temp_object_slot)
        for arg in resolved_args.iter().skip(1) {
            match arg {
                Arg::Positional(expr) => expr::compile_expr(ctx, expr)?,
                Arg::Named { value, .. } => expr::compile_expr(ctx, value)?,
                Arg::UnpackObject(expr) => expr::compile_expr(ctx, expr)?,
            }
        }
        // Всего аргументов на стеке: 1 receiver + (resolved_args.len() - 1). Call(arity) принимает arity = это число; мы передаём actual_arg_count+1 в Call, значит actual_arg_count = resolved_args.len() - 1.
        resolved_args.len() - 1
    } else if method == "train_sh" {
        // Разрешаем именованные аргументы для train_sh метода
        let resolved_args = args::resolve_function_args("nn_train_sh", args, None, line, ctx.source_name)?;
        // Пропускаем первый аргумент (nn): объект уже на стеке через LoadLocal(temp_object_slot)
        for arg in resolved_args.iter().skip(1) {
            match arg {
                Arg::Positional(expr) => expr::compile_expr(ctx, expr)?,
                Arg::Named { value, .. } => expr::compile_expr(ctx, value)?,
                Arg::UnpackObject(expr) => expr::compile_expr(ctx, expr)?,
            }
        }
        resolved_args.len() - 1
    } else {
        // Для device и save компилируем аргументы
        if args.len() != 1 {
            return Err(LangError::ParseError {
                message: format!("{}() takes exactly 1 argument", method),
                line,
                file: None,
            });
        }
        for arg in args {
            match arg {
                Arg::Positional(expr) => expr::compile_expr(ctx, expr)?,
                Arg::Named { value, .. } => expr::compile_expr(ctx, value)?,
                Arg::UnpackObject(expr) => expr::compile_expr(ctx, expr)?,
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
            file: None,
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
            Arg::UnpackObject(expr) => expr::compile_expr(ctx, expr)?,
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
            Arg::UnpackObject(expr) => expr::compile_expr(ctx, expr)?,
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

/// Database engine/cluster methods (add, get, names, connect, execute, query, run) need receiver as first arg.
/// Stack before Call: [receiver, arg1, ..., method_fn]. VM pops (arity+1) and passes (receiver, arg1, ...) to native.
fn compile_db_receiver_method(
    ctx: &mut CompilationContext,
    method: &str,
    args: &[Arg],
    temp_object_slot: usize,
    line: usize,
) -> Result<(), LangError> {
    // Push receiver first, then args, then receiver again for GetArrayElement, then get method; Call(1 + n).
    ctx.chunk.write_with_line(OpCode::LoadLocal(temp_object_slot), line);
    for arg in args {
        match arg {
            Arg::Positional(expr) => expr::compile_expr(ctx, expr)?,
            Arg::Named { value, .. } => expr::compile_expr(ctx, value)?,
            Arg::UnpackObject(expr) => expr::compile_expr(ctx, expr)?,
        }
    }
    ctx.chunk.write_with_line(OpCode::LoadLocal(temp_object_slot), line);
    let method_name_index = ctx.chunk.add_constant(Value::String(method.to_string()));
    ctx.chunk.write_with_line(OpCode::Constant(method_name_index), line);
    ctx.chunk.write_with_line(OpCode::GetArrayElement, line);
    ctx.chunk.write_with_line(OpCode::Call(1 + args.len()), line);
    Ok(())
}

fn compile_array_method(
    ctx: &mut CompilationContext,
    method: &str,
    args: &[Arg],
    temp_object_slot: usize,
    line: usize,
) -> Result<(), LangError> {
    // Array methods (push, pop, unique, reverse, sort, sum, average, count, any, all) expect (array, ...args).
    // Call(arity) pops function then arity args; after reverse, args[0] = first pushed.
    // Push receiver first, then method args, then get method; stack: [receiver, arg1, ..., method_fn], Call(1 + n).
    ctx.chunk.write_with_line(OpCode::LoadLocal(temp_object_slot), line);
    for arg in args {
        match arg {
            Arg::Positional(expr) => expr::compile_expr(ctx, expr)?,
            Arg::Named { value, .. } => expr::compile_expr(ctx, value)?,
            Arg::UnpackObject(expr) => expr::compile_expr(ctx, expr)?,
        }
    }
    ctx.chunk.write_with_line(OpCode::LoadLocal(temp_object_slot), line);
    let method_name_index = ctx.chunk.add_constant(Value::String(method.to_string()));
    ctx.chunk.write_with_line(OpCode::Constant(method_name_index), line);
    ctx.chunk.write_with_line(OpCode::GetArrayElement, line);
    ctx.chunk.write_with_line(OpCode::Call(1 + args.len()), line);
    Ok(())
}

fn compile_string_method(
    ctx: &mut CompilationContext,
    method: &str,
    args: &[Arg],
    temp_object_slot: usize,
    line: usize,
) -> Result<(), LangError> {
    // String methods: native receives (receiver, ...args) except join which expects (array, delim).
    // For "".join(chars): receiver is delim, arg is array; native_join(array, delim).
    // So for join we push arg first then receiver; stack [arg, receiver, fn], Call(2) -> args after reverse = [receiver, arg] -> we need [arg, receiver], so push arg, receiver.
    let n = args.len();
    if method == "join" {
        if n != 1 {
            return Err(LangError::ParseError {
                message: "string.join() takes exactly 1 argument (array)".to_string(),
                line,
                file: None,
            });
        }
        for arg in args {
            match arg {
                Arg::Positional(expr) => expr::compile_expr(ctx, expr)?,
                Arg::Named { value, .. } => expr::compile_expr(ctx, value)?,
                Arg::UnpackObject(expr) => expr::compile_expr(ctx, expr)?,
            }
        }
        ctx.chunk.write_with_line(OpCode::LoadLocal(temp_object_slot), line);
    } else {
        if matches!(method, "lower" | "upper" | "isupper" | "islower" | "trim") && n != 0 {
            return Err(LangError::ParseError {
                message: format!("string.{}() takes no arguments", method),
                line,
                file: None,
            });
        }
        if matches!(method, "split" | "contains") && n != 1 {
            return Err(LangError::ParseError {
                message: format!("string.{}() takes exactly 1 argument", method),
                line,
                file: None,
            });
        }
        ctx.chunk.write_with_line(OpCode::LoadLocal(temp_object_slot), line);
        for arg in args {
            match arg {
                Arg::Positional(expr) => expr::compile_expr(ctx, expr)?,
                Arg::Named { value, .. } => expr::compile_expr(ctx, value)?,
                Arg::UnpackObject(expr) => expr::compile_expr(ctx, expr)?,
            }
        }
    }
    ctx.chunk.write_with_line(OpCode::LoadLocal(temp_object_slot), line);
    let method_name_index = ctx.chunk.add_constant(Value::String(method.to_string()));
    ctx.chunk.write_with_line(OpCode::Constant(method_name_index), line);
    ctx.chunk.write_with_line(OpCode::GetArrayElement, line);
    ctx.chunk.write_with_line(OpCode::Call(1 + n), line);
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
            file: None,
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
    let resolved_args = match args::resolve_function_args(method, args, None, line, ctx.source_name) {
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
                    Arg::UnpackObject(e) => Arg::Positional(e.clone()),
                }).collect()
            } else {
                return Err(e);
            }
        }
    };
    
    // ВАЖНО: Для методов модуля (ml, plot, settings_env) receiver НЕ передаётся в нативу.
    // VM для Call(arity) извлекает функцию (с вершины стека), затем arity аргументов.
    // Стек ДО Call: [arg_1, ..., arg_n, method_function]. Натив получает только (arg_1, ..., arg_n).
    //
    // Последовательность:
    // 1. Compile args — аргументы вызова. Стек: [arg_1, ..., arg_n]
    // 2. LoadLocal (object) + Constant(method_name) + GetArrayElement — получить метод и положить на вершину.
    //    Стек после: [arg_1, ..., arg_n, method_function]
    // 3. Call(n) — VM извлечёт method_function, затем arg_n..arg_1 и передаст нативу (arg_1, ..., arg_n).
    
    let start_ip = ctx.chunk.code.len();
    debug_println!("[DEBUG compile_module_method] Начало компиляции вызова метода '{}' на строке {}, начальный IP: {}", method, line, start_ip);
    
    // 1. Компилируем аргументы (всегда компилируем из resolved_args; если пусто — из исходных args)
    debug_println!("[DEBUG compile_module_method] resolved_args.len() = {}, args.len() = {}", resolved_args.len(), args.len());
    let args_to_compile = if resolved_args.is_empty() {
        debug_println!("[DEBUG compile_module_method] resolved_args пуст, используем исходные args");
        args.iter().map(|a| match a {
            Arg::Positional(e) => Arg::Positional(e.clone()),
            Arg::Named { value, .. } => Arg::Positional(value.clone()),
            Arg::UnpackObject(e) => Arg::Positional(e.clone()),
        }).collect::<Vec<_>>()
    } else {
        debug_println!("[DEBUG compile_module_method] используем resolved_args");
        resolved_args.clone()
    };
    debug_println!("[DEBUG compile_module_method] args_to_compile.len() = {}", args_to_compile.len());
    for arg in &args_to_compile {
        match arg {
            Arg::Positional(expr) => expr::compile_expr(ctx, expr)?,
            Arg::Named { value, .. } => expr::compile_expr(ctx, value)?,
            Arg::UnpackObject(expr) => expr::compile_expr(ctx, expr)?,
        }
    }
    debug_println!("[DEBUG compile_module_method] После компиляции аргументов, IP: {}, стек: [arg_1, ..., arg_n]", ctx.chunk.code.len());
    
    // 2. Получаем метод и кладём на вершину стека (GetArrayElement ожидает [container, index])
    ctx.chunk.write_with_line(OpCode::LoadLocal(temp_object_slot), line);
    let method_name_index = ctx.chunk.add_constant(Value::String(method.to_string()));
    ctx.chunk.write_with_line(OpCode::Constant(method_name_index), line);
    ctx.chunk.write_with_line(OpCode::GetArrayElement, line);
    debug_println!("[DEBUG compile_module_method] После GetArrayElement для метода '{}', IP: {}, стек: [arg_1, ..., arg_n, method_function]", method, ctx.chunk.code.len());
    
    // 3. Call(n): VM извлечёт function, затем arg_n..arg_1 → натив получит (arg_1, ..., arg_n), без receiver
    let call_arity = args_to_compile.len();
    let call_ip = ctx.chunk.code.len();
    ctx.chunk.write_with_line(OpCode::Call(call_arity), line);
    debug_println!("[DEBUG compile_module_method] Сгенерирован Call({}) на IP {} для метода '{}'", call_arity, call_ip, method);
    
    // ВАЖНО: После вызова метода Call должен извлечь функцию и аргументы со стека
    // и вызвать метод. Если Call не выполняется (например, из-за ошибки или раннего возврата),
    // функция может остаться на стеке. Это может вызвать проблемы в следующей итерации цикла.
    // Однако, мы не можем добавить Pop здесь, так как Call должен вернуть результат метода.
    // Вместо этого, мы полагаемся на то, что Call правильно обработает стек.
    
    // Логируем все инструкции, которые были сгенерированы
    debug_println!("[DEBUG compile_module_method] Сгенерированные инструкции для метода '{}' (IP {} - {}):", method, start_ip, ctx.chunk.code.len());
    for i in start_ip..ctx.chunk.code.len() {
        debug_println!("[DEBUG compile_module_method]   IP {}: {:?}", i, ctx.chunk.code.get(i));
    }
    
    Ok(())
}

