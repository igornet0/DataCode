/// Компиляция вызовов функций

use crate::debug_println;
use crate::parser::ast::{Expr, Arg};
use crate::bytecode::OpCode;
use crate::common::error::LangError;
use crate::common::value::Value;
use crate::compiler::context::CompilationContext;
use crate::compiler::expr;
use crate::compiler::args;
use crate::compiler::stmt::class::{MODEL_CONFIG_CLASS_LOAD_INDEX, CONSTRUCTING_CLASS_GLOBAL_NAME};

/// True if class name is "Settings" or has Settings as an ancestor (used for 1-arg call expansion).
fn is_in_settings_chain(name: &str, class_superclass: &std::collections::HashMap<String, String>) -> bool {
    if name == "Settings" {
        return true;
    }
    class_superclass
        .get(name)
        .map(|parent| is_in_settings_chain(parent, class_superclass))
        .unwrap_or(false)
}

pub fn compile_call(ctx: &mut CompilationContext, expr: &Expr) -> Result<(), LangError> {
    if let Expr::Call { name, args: call_args, line } = expr {
        *ctx.current_line = *line;
        
        // Проверяем, начинается ли имя с маленькой буквы
        // Функции, начинающиеся с маленькой буквы, не могут быть конструкторами классов
        let is_lowercase = name.chars().next().map(|c| c.is_lowercase()).unwrap_or(false);
        
        if is_lowercase {
            // Это обычная функция (нативная или пользовательская) - обрабатываем как обычный вызов
            debug_println!("[DEBUG compile_call] Функция '{}' начинается с маленькой буквы, обрабатываем как обычную функцию", name);
            // Пропускаем все проверки конструктора и переходим к обработке обычной функции
        } else {
            // Имя начинается с заглавной буквы - это может быть конструктор класса
            // Проверяем, является ли это вызовом конструктора класса
            // Конструкторы имеют формат ClassName::new_<arity>
            let constructor_name = format!("{}::new_{}", name, call_args.len());
            debug_println!("[DEBUG compile_call] Проверяем вызов '{}' с {} аргументами", name, call_args.len());
            debug_println!("[DEBUG compile_call] Ищем конструктор '{}'", constructor_name);

            // Settings() with no args: expand to Settings(__constructing_class__["model_config"]) so env file comes from model_config.
            if name == "Settings" && call_args.is_empty() && !ctx.function_names.iter().any(|n| n == "Settings::new_0") {
                let settings_slot = if let Some(&idx) = ctx.scope.globals.get("Settings") {
                    idx
                } else {
                    let idx = ctx.scope.globals.len();
                    ctx.scope.globals.insert("Settings".to_string(), idx);
                    idx
                };
                ctx.chunk.global_names.insert(MODEL_CONFIG_CLASS_LOAD_INDEX, CONSTRUCTING_CLASS_GLOBAL_NAME.to_string());
                ctx.chunk.write_with_line(OpCode::LoadGlobal(MODEL_CONFIG_CLASS_LOAD_INDEX), *line);
                let model_config_key = ctx.chunk.add_constant(Value::String("model_config".to_string()));
                ctx.chunk.write_with_line(OpCode::Constant(model_config_key), *line);
                ctx.chunk.write_with_line(OpCode::GetArrayElement, *line);
                ctx.chunk.global_names.insert(settings_slot, "Settings".to_string());
                ctx.chunk.write_with_line(OpCode::LoadGlobal(settings_slot), *line);
                ctx.chunk.write_with_line(OpCode::Call(1), *line);
                return Ok(());
            }

            // Settings subclass with 0 args (e.g. ProdSettings()): expand to (path="", required_keys, model_config) and Call(3).
            // Set __constructing_class__ = class before pushing args so required_keys and model_config are always loaded from the class we're instantiating (fixes cross-module call e.g. load_settings() -> DevSettings()).
            let arg_count = call_args.len();
            if arg_count == 0 && is_in_settings_chain(name, ctx.class_superclass) {
                let ctor_1_name = format!("{}::new_1", name);
                let class_global_index_opt = ctx.scope.globals.get(name).copied();
                let required_keys_value = ctx.class_required_keys_value.get(name).cloned();
                // Same-file: use compiler's required_keys; imported class: load from class["__required_keys"] at runtime.
                let emit_0_arg_settings = |ctx: &mut CompilationContext, line: usize, use_class_required_keys: bool| -> Result<(), LangError> {
                    let class_global_index = class_global_index_opt.expect("Settings class in globals");
                    let new_idx = ctx.scope.globals.len();
                    let constructing_class_idx = *ctx.scope.globals.entry(CONSTRUCTING_CLASS_GLOBAL_NAME.to_string()).or_insert(new_idx);
                    ctx.chunk.global_names.insert(constructing_class_idx, CONSTRUCTING_CLASS_GLOBAL_NAME.to_string());
                    ctx.chunk.write_with_line(OpCode::LoadGlobal(class_global_index), line);
                    ctx.chunk.write_with_line(OpCode::StoreGlobal(constructing_class_idx), line);
                    let path_empty = ctx.chunk.add_constant(Value::String(String::new()));
                    ctx.chunk.write_with_line(OpCode::Constant(path_empty), line);
                    if use_class_required_keys {
                        let req_const = ctx.chunk.add_constant(required_keys_value.as_ref().unwrap().clone());
                        ctx.chunk.write_with_line(OpCode::Constant(req_const), line);
                    } else {
                        ctx.chunk.write_with_line(OpCode::LoadGlobal(constructing_class_idx), line);
                    let req_key = ctx.chunk.add_constant(Value::String("__required_keys".to_string()));
                    ctx.chunk.write_with_line(OpCode::Constant(req_key), line);
                    ctx.chunk.write_with_line(OpCode::GetArrayElement, line);
                }
                ctx.chunk.write_with_line(OpCode::LoadGlobal(constructing_class_idx), line);
                let model_config_key = ctx.chunk.add_constant(Value::String("model_config".to_string()));
                ctx.chunk.write_with_line(OpCode::Constant(model_config_key), line);
                ctx.chunk.write_with_line(OpCode::GetArrayElement, line);
                let null_const = ctx.chunk.add_constant(Value::Null);
                ctx.chunk.write_with_line(OpCode::Constant(null_const), line);
                Ok(())
            };
            if let Some(_) = &required_keys_value {
                if let Some(function_index) = ctx.function_names.iter().position(|n| n == &ctor_1_name) {
                    emit_0_arg_settings(ctx, *line, true)?;
                    let constant_index = ctx.chunk.add_constant(Value::Function(function_index));
                    ctx.chunk.write_with_line(OpCode::Constant(constant_index), *line);
                    ctx.chunk.write_with_line(OpCode::Call(4), *line);
                        return Ok(());
                    }
                    if ctx.scope.globals.contains_key(name) && ctx.scope.globals.contains_key(&ctor_1_name) {
                        let &global_index = ctx.scope.globals.get(&ctor_1_name).unwrap();
                        emit_0_arg_settings(ctx, *line, true)?;
                        ctx.chunk.global_names.insert(global_index, ctor_1_name.clone());
                        ctx.chunk.write_with_line(OpCode::LoadGlobal(global_index), *line);
                        ctx.chunk.write_with_line(OpCode::Call(4), *line);
                        return Ok(());
                    }
                }
                // Imported Settings class: load required_keys from class["__required_keys"].
                if ctx.scope.globals.contains_key(name) && ctx.scope.globals.contains_key(&ctor_1_name) {
                    let &global_index = ctx.scope.globals.get(&ctor_1_name).unwrap();
                    emit_0_arg_settings(ctx, *line, false)?;
                    ctx.chunk.global_names.insert(global_index, ctor_1_name.clone());
                    ctx.chunk.write_with_line(OpCode::LoadGlobal(global_index), *line);
                    ctx.chunk.write_with_line(OpCode::Call(4), *line);
                    return Ok(());
                }
            }
        
            // Сначала проверяем в function_names (для конструкторов, определенных в текущем файле)
        if let Some(function_index) = ctx.function_names.iter().position(|n| n == &constructor_name) {
            // Это вызов конструктора
            debug_println!("[DEBUG compile_call] Найден конструктор '{}' с индексом функции {} в function_names", constructor_name, function_index);
            let arg_count = call_args.len();
            debug_println!("[DEBUG compile_call] Сохранено количество аргументов: {} для конструктора '{}' (function_names)", arg_count, constructor_name);
            // 0-arg Settings subclass: set __constructing_class__ = class so new_0's body can load required_keys/model_config from it.
            if arg_count == 0 && is_in_settings_chain(name, ctx.class_superclass) {
                if let Some(&class_global_index) = ctx.scope.globals.get(name) {
                    let new_idx = ctx.scope.globals.len();
                    let constructing_class_idx = *ctx.scope.globals.entry(CONSTRUCTING_CLASS_GLOBAL_NAME.to_string()).or_insert(new_idx);
                    ctx.chunk.global_names.insert(constructing_class_idx, CONSTRUCTING_CLASS_GLOBAL_NAME.to_string());
                    ctx.chunk.write_with_line(OpCode::LoadGlobal(class_global_index), *line);
                    ctx.chunk.write_with_line(OpCode::StoreGlobal(constructing_class_idx), *line);
                }
            }
            // Settings subclass with 1 arg: expand to (path, required_keys, model_config) and Call(3).
            if arg_count == 1 && is_in_settings_chain(name, ctx.class_superclass) {
                let required_keys_value = ctx.class_required_keys_value.get(name).cloned();
                if let Some(required_keys_value) = required_keys_value {
                    match &call_args[0] {
                        Arg::Positional(e) => expr::compile_expr(ctx, e)?,
                        Arg::Named { value, .. } => expr::compile_expr(ctx, value)?,
                        Arg::UnpackObject(e) => expr::compile_expr(ctx, e)?,
                    }
                    let req_const = ctx.chunk.add_constant(required_keys_value);
                    ctx.chunk.write_with_line(OpCode::Constant(req_const), *line);
                    let class_global_index = *ctx.scope.globals.get(name).expect("Settings class in globals");
                    ctx.chunk.global_names.insert(class_global_index, name.clone());
                    ctx.chunk.write_with_line(OpCode::LoadGlobal(class_global_index), *line);
                    let model_config_key = ctx.chunk.add_constant(Value::String("model_config".to_string()));
                    ctx.chunk.write_with_line(OpCode::Constant(model_config_key), *line);
                    ctx.chunk.write_with_line(OpCode::GetArrayElement, *line);
                    let null_const = ctx.chunk.add_constant(Value::Null);
                    ctx.chunk.write_with_line(OpCode::Constant(null_const), *line);
                    let constant_index = ctx.chunk.add_constant(Value::Function(function_index));
                    ctx.chunk.write_with_line(OpCode::Constant(constant_index), *line);
                    ctx.chunk.write_with_line(OpCode::Call(4), *line);
                    return Ok(());
                }
            }
            
            for (i, arg) in call_args.iter().enumerate() {
                match arg {
                    Arg::Positional(expr) => {
                        debug_println!("[DEBUG compile_call] Компилируем позиционный аргумент {} из {} (function_names)", i + 1, arg_count);
                        expr::compile_expr(ctx, expr)?;
                    }
                    Arg::Named { value, .. } => {
                        debug_println!("[DEBUG compile_call] Компилируем именованный аргумент {} из {} (function_names)", i + 1, arg_count);
                        expr::compile_expr(ctx, value)?;
                    }
                    Arg::UnpackObject(expr) => {
                        debug_println!("[DEBUG compile_call] Компилируем ** аргумент {} из {} (function_names)", i + 1, arg_count);
                        expr::compile_expr(ctx, expr)?;
                    }
                }
            }
            
            if ctx.abstract_classes.contains(name) {
                let class_global_index = *ctx.scope.globals.get(name).expect("abstract class must be in globals");
                ctx.chunk.write_with_line(OpCode::LoadGlobal(class_global_index), *line);
                ctx.chunk.write_with_line(OpCode::Call(arg_count), *line);
                return Ok(());
            }
            
            let constant_index = ctx.chunk.add_constant(Value::Function(function_index));
            let constant_ip = ctx.chunk.code.len();
            ctx.chunk.write_with_line(OpCode::Constant(constant_index), *line);
            let call_ip = ctx.chunk.code.len();
            ctx.chunk.write_with_line(OpCode::Call(arg_count), *line);
            debug_println!("[DEBUG compile_call] Сгенерирован OpCode::Constant({}) на IP {}, OpCode::Call({}) на IP {} для конструктора '{}' (function_names)", 
                constant_index, constant_ip, arg_count, call_ip, constructor_name);
            
            if let Some(OpCode::Call(recorded_arity)) = ctx.chunk.code.get(call_ip) {
                if *recorded_arity != arg_count {
                    debug_println!("[ERROR compile_call] КРИТИЧЕСКАЯ ОШИБКА: Записано Call({}), но ожидалось Call({})!", recorded_arity, arg_count);
                }
            }
            
            return Ok(());
        }
        
        // Если конструктор не найден в function_names, проверяем в глобальных переменных
        // Это нужно для конструкторов, импортированных из модулей.
        // Входим сюда только если конструктор уже зарегистрирован (класс скомпилирован);
        // иначе вызов Base(10) при обычной функции Base не должен создавать слот Base::new_1.
        if ctx.scope.globals.contains_key(name) && ctx.scope.globals.contains_key(&constructor_name) {
            let &global_index = ctx.scope.globals.get(&constructor_name).unwrap();
            debug_println!("[DEBUG compile_call] Найден конструктор '{}' в globals с индексом {}", constructor_name, global_index);
            let arg_count = call_args.len();
            debug_println!("[DEBUG compile_call] Сохранено количество аргументов: {} для конструктора '{}' (globals)", arg_count, constructor_name);
            // 0-arg Settings subclass: set __constructing_class__ = class so new_0's body can load required_keys/model_config from it.
            if arg_count == 0 && is_in_settings_chain(name, ctx.class_superclass) {
                if let Some(&class_global_index) = ctx.scope.globals.get(name) {
                    let new_idx = ctx.scope.globals.len();
                    let constructing_class_idx = *ctx.scope.globals.entry(CONSTRUCTING_CLASS_GLOBAL_NAME.to_string()).or_insert(new_idx);
                    ctx.chunk.global_names.insert(constructing_class_idx, CONSTRUCTING_CLASS_GLOBAL_NAME.to_string());
                    ctx.chunk.write_with_line(OpCode::LoadGlobal(class_global_index), *line);
                    ctx.chunk.write_with_line(OpCode::StoreGlobal(constructing_class_idx), *line);
                }
            }
            // Settings subclass with 1 arg: expand to (path, required_keys, model_config) and Call(3).
            if arg_count == 1 && is_in_settings_chain(name, ctx.class_superclass) {
                let required_keys_value = ctx.class_required_keys_value.get(name).cloned();
                if let Some(required_keys_value) = required_keys_value {
                    match &call_args[0] {
                        Arg::Positional(e) => expr::compile_expr(ctx, e)?,
                        Arg::Named { value, .. } => expr::compile_expr(ctx, value)?,
                        Arg::UnpackObject(e) => expr::compile_expr(ctx, e)?,
                    }
                    let req_const = ctx.chunk.add_constant(required_keys_value);
                    ctx.chunk.write_with_line(OpCode::Constant(req_const), *line);
                    let class_global_index = *ctx.scope.globals.get(name).expect("Settings class in globals");
                    ctx.chunk.global_names.insert(class_global_index, name.clone());
                    ctx.chunk.write_with_line(OpCode::LoadGlobal(class_global_index), *line);
                    let model_config_key = ctx.chunk.add_constant(Value::String("model_config".to_string()));
                    ctx.chunk.write_with_line(OpCode::Constant(model_config_key), *line);
                    ctx.chunk.write_with_line(OpCode::GetArrayElement, *line);
                    let null_const = ctx.chunk.add_constant(Value::Null);
                    ctx.chunk.write_with_line(OpCode::Constant(null_const), *line);
                    ctx.chunk.global_names.insert(global_index, constructor_name.clone());
                    ctx.chunk.write_with_line(OpCode::LoadGlobal(global_index), *line);
                    ctx.chunk.write_with_line(OpCode::Call(4), *line);
                    return Ok(());
                }
            }
            
            for (i, arg) in call_args.iter().enumerate() {
                match arg {
                    Arg::Positional(expr) => {
                        debug_println!("[DEBUG compile_call] Компилируем позиционный аргумент {} из {} (globals)", i + 1, arg_count);
                        expr::compile_expr(ctx, expr)?;
                    }
                    Arg::Named { value, .. } => {
                        debug_println!("[DEBUG compile_call] Компилируем именованный аргумент {} из {} (globals)", i + 1, arg_count);
                        expr::compile_expr(ctx, value)?;
                    }
                    Arg::UnpackObject(expr) => {
                        debug_println!("[DEBUG compile_call] Компилируем ** аргумент {} из {} (globals)", i + 1, arg_count);
                        expr::compile_expr(ctx, expr)?;
                    }
                }
            }
            
            if ctx.abstract_classes.contains(name) {
                let class_global_index = *ctx.scope.globals.get(name).unwrap();
                ctx.chunk.write_with_line(OpCode::LoadGlobal(class_global_index), *line);
                ctx.chunk.write_with_line(OpCode::Call(arg_count), *line);
                return Ok(());
            }
            // Нужно для update_chunk_indices при merge модуля (патч LoadGlobal по имени).
            ctx.chunk.global_names.insert(global_index, constructor_name.clone());
            let load_global_ip = ctx.chunk.code.len();
            ctx.chunk.write_with_line(OpCode::LoadGlobal(global_index), *line);
            let call_ip = ctx.chunk.code.len();
            ctx.chunk.write_with_line(OpCode::Call(arg_count), *line);
            debug_println!("[DEBUG compile_call] Сгенерирован OpCode::LoadGlobal({}) на IP {}, OpCode::Call({}) на IP {} для конструктора '{}' (globals)", 
                global_index, load_global_ip, arg_count, call_ip, constructor_name);
            
            if let Some(OpCode::Call(recorded_arity)) = ctx.chunk.code.get(call_ip) {
                if *recorded_arity != arg_count {
                    debug_println!("[ERROR compile_call] КРИТИЧЕСКАЯ ОШИБКА: Записано Call({}), но ожидалось Call({})!", recorded_arity, arg_count);
                }
            }
            
            return Ok(());
        }
        
        // Если конструктор не найден в globals, но класс найден, генерируем код для проверки во время выполнения.
        // Это нужно для конструкторов, импортированных из модулей через __lib__.dc.
        // Не входим сюда для обычной функции с заглавной буквы (например Base) — только для классов.
        let has_constructor = ctx.scope.globals.contains_key(&constructor_name)
            || ctx.function_names.iter().any(|n| n == &constructor_name);
        if ctx.scope.globals.contains_key(name) && has_constructor {
            debug_println!("[DEBUG compile_call] Класс '{}' найден в globals, генерируем код для проверки конструктора во время выполнения", name);
            let arg_count = call_args.len();
            debug_println!("[DEBUG compile_call] Сохранено количество аргументов: {} для конструктора '{}' (класс в globals)", arg_count, constructor_name);
            // 0-arg call: set __constructing_class__ = class so Settings subclass new_0 can load required_keys/model_config (class_superclass may be empty when class is imported).
            if arg_count == 0 {
                if let Some(&class_global_index) = ctx.scope.globals.get(name) {
                    let new_idx = ctx.scope.globals.len();
                    let constructing_class_idx = *ctx.scope.globals.entry(CONSTRUCTING_CLASS_GLOBAL_NAME.to_string()).or_insert(new_idx);
                    ctx.chunk.global_names.insert(constructing_class_idx, CONSTRUCTING_CLASS_GLOBAL_NAME.to_string());
                    ctx.chunk.write_with_line(OpCode::LoadGlobal(class_global_index), *line);
                    ctx.chunk.write_with_line(OpCode::StoreGlobal(constructing_class_idx), *line);
                }
            }
            for (i, arg) in call_args.iter().enumerate() {
                match arg {
                    Arg::Positional(expr) => {
                        debug_println!("[DEBUG compile_call] Компилируем позиционный аргумент {} из {} (класс в globals)", i + 1, arg_count);
                        expr::compile_expr(ctx, expr)?;
                    }
                    Arg::Named { value, .. } => {
                        debug_println!("[DEBUG compile_call] Компилируем именованный аргумент {} из {} (класс в globals)", i + 1, arg_count);
                        expr::compile_expr(ctx, value)?;
                    }
                    Arg::UnpackObject(expr) => {
                        debug_println!("[DEBUG compile_call] Компилируем ** аргумент {} из {} (класс в globals)", i + 1, arg_count);
                        expr::compile_expr(ctx, expr)?;
                    }
                }
            }
            
            if ctx.abstract_classes.contains(name) {
                let class_global_index = *ctx.scope.globals.get(name).unwrap();
                ctx.chunk.write_with_line(OpCode::LoadGlobal(class_global_index), *line);
                ctx.chunk.write_with_line(OpCode::Call(arg_count), *line);
                return Ok(());
            }
            
            let constructor_global_index = if let Some(&idx) = ctx.scope.globals.get(&constructor_name) {
                idx
            } else {
                let idx = ctx.scope.globals.len();
                ctx.scope.globals.insert(constructor_name.clone(), idx);
                idx
            };
            ctx.chunk.global_names.insert(constructor_global_index, constructor_name.clone());
            let load_global_ip = ctx.chunk.code.len();
            ctx.chunk.write_with_line(OpCode::LoadGlobal(constructor_global_index), *line);
            let call_ip = ctx.chunk.code.len();
            ctx.chunk.write_with_line(OpCode::Call(arg_count), *line);
            debug_println!("[DEBUG compile_call] Сгенерирован OpCode::LoadGlobal({}) на IP {}, OpCode::Call({}) на IP {} для конструктора '{}' (класс в globals)", 
                constructor_global_index, load_global_ip, arg_count, call_ip, constructor_name);
            
            if let Some(OpCode::Call(recorded_arity)) = ctx.chunk.code.get(call_ip) {
                if *recorded_arity != arg_count {
                    debug_println!("[ERROR compile_call] КРИТИЧЕСКАЯ ОШИБКА: Записано Call({}), но ожидалось Call({})!", recorded_arity, arg_count);
                }
            }
            
            return Ok(());
        }
        
        // Класс найден, но конструктор отсутствует — возможно, неявный конструктор пропущен
        // (суперкласс не принимает переданное число аргументов). Ошибка на строке вызова.
        if ctx.scope.globals.contains_key(name)
            && !ctx.scope.globals.contains_key(&constructor_name)
            && !ctx.function_names.iter().any(|n| n == &constructor_name)
        {
            // Только для пропущенного неявного конструктора наследника — compile-time ошибка.
            // Прямые вызовы с неверной арностью (Parent(20)) дадут runtime-ошибку, чтобы try-catch сработал.
            if let Some(superclass_name) = ctx.class_superclass.get(name) {
                return Err(LangError::ParseError {
                    message: format!("Class '{}' cannot accept {} argument(s)", superclass_name, call_args.len()),
                    line: *line,
                    file: None,
                });
            }
        }
        
        // Если класс не найден, но имя начинается с заглавной буквы (характерно для классов),
        // это может быть импортированный класс из __lib__.dc или другого модуля.
        // Не входим сюда, если имя уже в scope (например обычная функция Base) — тогда обрабатываем как вызов функции ниже.
        if name.chars().next().map(|c| c.is_uppercase()).unwrap_or(false) && !ctx.scope.globals.contains_key(name) {
            debug_println!("[DEBUG compile_call] Класс '{}' не найден в globals, но имя начинается с заглавной буквы - предполагаем, что это класс, импортированный из модуля", name);
            
            // Регистрируем класс в scope, если его там еще нет
            if !ctx.scope.globals.contains_key(name) {
                let class_global_index = ctx.scope.globals.len();
                ctx.scope.globals.insert(name.clone(), class_global_index);
                // ВАЖНО: Не добавляем класс в chunk.global_names здесь, чтобы избежать конфликта с конструктором
                // Класс будет добавлен в chunk.global_names только если он используется напрямую
                debug_println!("[DEBUG compile_call] Зарегистрирован класс '{}' в globals с индексом {} (не добавляем в chunk.global_names)", name, class_global_index);
            }
            
            // Регистрируем конструктор в scope
            let constructor_global_index = if let Some(&idx) = ctx.scope.globals.get(&constructor_name) {
                idx
            } else {
                let idx = ctx.scope.globals.len();
                ctx.scope.globals.insert(constructor_name.clone(), idx);
                // ВАЖНО: Добавляем конструктор в chunk.global_names с правильным именем
                ctx.chunk.global_names.insert(idx, constructor_name.clone());
                debug_println!("[DEBUG compile_call] Зарегистрирован конструктор '{}' в globals с индексом {} и в chunk.global_names", constructor_name, idx);
                idx
            };
            
            // Всегда записываем имя конструктора в chunk для update_chunk_indices при merge (безусловно).
            ctx.chunk.global_names.insert(constructor_global_index, constructor_name.clone());
            
            // Сохраняем количество аргументов до компиляции (на случай, если call_args будет перемещено)
            let arg_count = call_args.len();
            debug_println!("[DEBUG compile_call] Сохранено количество аргументов: {} для конструктора '{}'", arg_count, constructor_name);
            
            // Компилируем аргументы ПЕРЕД загрузкой конструктора
            let args_start_ip = ctx.chunk.code.len();
            debug_println!("[DEBUG compile_call] Начало компиляции аргументов, IP: {}", args_start_ip);
            for (i, arg) in call_args.iter().enumerate() {
                match arg {
                    Arg::Positional(expr) => {
                        debug_println!("[DEBUG compile_call] Компилируем позиционный аргумент {} из {}", i + 1, arg_count);
                        expr::compile_expr(ctx, expr)?;
                    }
                    Arg::Named { value, .. } => {
                        debug_println!("[DEBUG compile_call] Компилируем именованный аргумент {} из {}", i + 1, arg_count);
                        expr::compile_expr(ctx, value)?;
                    }
                    Arg::UnpackObject(expr) => {
                        debug_println!("[DEBUG compile_call] Компилируем ** аргумент {} из {}", i + 1, arg_count);
                        expr::compile_expr(ctx, expr)?;
                    }
                }
            }
            let args_end_ip = ctx.chunk.code.len();
            debug_println!("[DEBUG compile_call] Конец компиляции аргументов, IP: {}, сгенерировано инструкций: {}", args_end_ip, args_end_ip - args_start_ip);
            
            // Генерируем код для загрузки конструктора из глобальных переменных во время выполнения
            debug_println!("[DEBUG compile_call] Компилируем вызов конструктора '{}' с {} аргументами", constructor_name, arg_count);
            let load_global_ip = ctx.chunk.code.len();
            ctx.chunk.write_with_line(OpCode::LoadGlobal(constructor_global_index), *line);
            let call_ip = ctx.chunk.code.len();
            ctx.chunk.write_with_line(OpCode::Call(arg_count), *line);
            debug_println!("[DEBUG compile_call] Сгенерирован OpCode::LoadGlobal({}) на IP {}, OpCode::Call({}) на IP {} для конструктора '{}'", 
                constructor_global_index, load_global_ip, arg_count, call_ip, constructor_name);
            
            // Проверяем, что инструкция действительно записана правильно
            if let Some(OpCode::Call(recorded_arity)) = ctx.chunk.code.get(call_ip) {
                if *recorded_arity != arg_count {
                    debug_println!("[ERROR compile_call] КРИТИЧЕСКАЯ ОШИБКА: Записано Call({}), но ожидалось Call({})!", recorded_arity, arg_count);
                    // Показываем окружающие инструкции для отладки
                    let start = call_ip.saturating_sub(5);
                    let end = (call_ip + 5).min(ctx.chunk.code.len());
                    debug_println!("[ERROR compile_call] Окружающие инструкции (IP {} - {}):", start, end);
                    for i in start..end {
                        let marker = if i == call_ip { " <-- ТЕКУЩАЯ" } else { "" };
                        debug_println!("[ERROR compile_call]   IP {}: {:?}{}", i, ctx.chunk.code.get(i), marker);
                    }
                } else {
                    debug_println!("[DEBUG compile_call] Подтверждено: Call({}) записан правильно на IP {}", arg_count, call_ip);
                }
            } else {
                debug_println!("[ERROR compile_call] КРИТИЧЕСКАЯ ОШИБКА: На IP {} не найдена инструкция Call!", call_ip);
                // Показываем, что там на самом деле
                if let Some(opcode) = ctx.chunk.code.get(call_ip) {
                    debug_println!("[ERROR compile_call] На IP {} найдена инструкция: {:?}", call_ip, opcode);
                }
            }
            
            // Дополнительная проверка: убеждаемся, что Call инструкция не была перезаписана
            // Проверяем еще раз после небольшой задержки (если есть другие операции)
            let final_check_ip = ctx.chunk.code.len() - 1;
            if final_check_ip == call_ip {
                if let Some(OpCode::Call(final_arity)) = ctx.chunk.code.get(final_check_ip) {
                    if *final_arity != arg_count {
                        debug_println!("[ERROR compile_call] КРИТИЧЕСКАЯ ОШИБКА: Call инструкция была изменена! Ожидалось Call({}), но найдено Call({}) на IP {}", 
                            arg_count, final_arity, final_check_ip);
                    }
                }
            }
            
            return Ok(());
        }
        
        // Импортированный символ с заглавной буквы (from X import Config): не fallback в обычную функцию,
        // а вызов конструктора — слот Config::new_N заполнится при выполнении ImportFrom.
        // Только для файловых модулей: встроенные (settings_env, ml, plot, uuid) не экспортируют конструкторы в globals так же, оставляем старый путь (LoadGlobal(name)+Call).
        fn is_builtin_module(name: &str) -> bool {
            matches!(name, "ml" | "plot" | "settings_env" | "uuid" | "database_engine")
        }
        if name.chars().next().map(|c| c.is_uppercase()).unwrap_or(false)
            && ctx.imported_symbols.get(name).map_or(false, |m| !is_builtin_module(m))
        {
            // Регистрируем класс в scope, если его там еще нет (уже есть от import, но индекс нужен для консистентности)
            if !ctx.scope.globals.contains_key(name) {
                let class_global_index = ctx.scope.globals.len();
                ctx.scope.globals.insert(name.clone(), class_global_index);
            }
            // Регистрируем конструктор в scope и chunk.global_names (обязательно для update_chunk_indices при merge).
            let constructor_global_index = if let Some(&idx) = ctx.scope.globals.get(&constructor_name) {
                idx
            } else {
                let idx = ctx.scope.globals.len();
                ctx.scope.globals.insert(constructor_name.clone(), idx);
                idx
            };
            ctx.chunk.global_names.insert(constructor_global_index, constructor_name.clone());
            let arg_count = call_args.len();
            // 0-arg call: set __constructing_class__ = class so Settings subclass new_0 can load required_keys/model_config.
            if arg_count == 0 {
                if let Some(&class_global_index) = ctx.scope.globals.get(name) {
                    ctx.chunk.global_names.insert(class_global_index, name.clone());
                    let new_idx = ctx.scope.globals.len();
                    let constructing_class_idx = *ctx.scope.globals.entry(CONSTRUCTING_CLASS_GLOBAL_NAME.to_string()).or_insert(new_idx);
                    ctx.chunk.global_names.insert(constructing_class_idx, CONSTRUCTING_CLASS_GLOBAL_NAME.to_string());
                    ctx.chunk.write_with_line(OpCode::LoadGlobal(class_global_index), *line);
                    ctx.chunk.write_with_line(OpCode::StoreGlobal(constructing_class_idx), *line);
                }
            }
            for arg in call_args.iter() {
                match arg {
                    Arg::Positional(expr) => expr::compile_expr(ctx, expr)?,
                    Arg::Named { value, .. } => expr::compile_expr(ctx, value)?,
                    Arg::UnpackObject(expr) => expr::compile_expr(ctx, expr)?,
                }
            }
            ctx.chunk.write_with_line(OpCode::LoadGlobal(constructor_global_index), *line);
            ctx.chunk.write_with_line(OpCode::Call(arg_count), *line);
            return Ok(());
        }
        
        debug_println!("[DEBUG compile_call] Конструктор '{}' не найден ни в function_names, ни в globals, класс '{}' тоже не найден, проверяем как обычную функцию", constructor_name, name);
        }

        // If constructor not found by arity but we have named args: resolve via class_constructor (extends_table field-based constructor)
        if !is_lowercase {
            let has_named = call_args.iter().any(|a| matches!(a, Arg::Named { .. } | Arg::UnpackObject(_)));
            if has_named {
                let ctor_info = ctx.class_constructor.get(name).map(|(c, i)| (c.clone(), *i));
                if let Some((ctor_name, function_index)) = ctor_info {
                    let function = &ctx.functions[function_index];
                    let function_info = (function_index, function);
                    let resolved_args = args::resolve_function_args(name, call_args, Some(function_info), *line, ctx.source_name)?;
                    let arity = resolved_args.len();
                    for arg in &resolved_args {
                        match arg {
                            Arg::Positional(expr) => expr::compile_expr(ctx, expr)?,
                            Arg::Named { value, .. } => expr::compile_expr(ctx, value)?,
                            Arg::UnpackObject(expr) => expr::compile_expr(ctx, expr)?,
                        }
                    }
                    if let Some(&global_index) = ctx.scope.globals.get(&ctor_name) {
                        ctx.chunk.global_names.insert(global_index, ctor_name.clone());
                        ctx.chunk.write_with_line(OpCode::LoadGlobal(global_index), *line);
                        ctx.chunk.write_with_line(OpCode::Call(arity), *line);
                        return Ok(());
                    }
                    let constant_index = ctx.chunk.add_constant(Value::Function(function_index));
                    ctx.chunk.write_with_line(OpCode::Constant(constant_index), *line);
                    ctx.chunk.write_with_line(OpCode::Call(arity), *line);
                    return Ok(());
                }
            }
        }
        
        // Обработка обычной функции (для функций, начинающихся с маленькой буквы, или если конструктор не найден)
        // Находим функцию для получения информации о параметрах. Сначала проверяем function_names,
        // чтобы пользовательские функции с default-параметрами получали подстановку аргументов.
        let function_info = if let Some(function_index) = ctx.function_names.iter().position(|n| n == name) {
            debug_println!("[DEBUG compile_call] Найдена функция '{}' с индексом {}", name, function_index);
            Some((function_index, &ctx.functions[function_index]))
        } else if is_lowercase && ctx.scope.globals.get(name).is_some() {
            // Builtin — arity разрешается во время выполнения
            None
        } else {
            if !is_lowercase {
                debug_println!("[DEBUG compile_call] WARNING: Функция '{}' не найдена в function_names", name);
                debug_println!("[DEBUG compile_call] Доступные функции: {:?}", ctx.function_names.iter().take(20).collect::<Vec<_>>());
            }
            None
        };
        
        // Разрешаем аргументы: именованные -> позиционные, применяем значения по умолчанию
        let resolved_args = args::resolve_function_args(name, call_args, function_info, *line, ctx.source_name)?;
        
        // Специальная обработка для isinstance: преобразуем идентификаторы типов в строки
        let processed_args = if name == "isinstance" && resolved_args.len() >= 2 {
            let mut new_args = resolved_args.clone();
            if let Arg::Positional(Expr::Variable { name: type_name, .. }) = &resolved_args[1] {
                let type_names = vec!["int", "str", "bool", "array", "null", "num", "float", "table", "Table"];
                if type_names.contains(&type_name.as_str()) {
                    new_args[1] = Arg::Positional(Expr::Literal {
                        value: Value::String(type_name.clone()),
                        line: match &resolved_args[1] {
                            Arg::Positional(e) => e.line(),
                            Arg::Named { value, .. } => value.line(),
                            Arg::UnpackObject(e) => e.line(),
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
        
        // Вызов с единственным **obj: эмитим CallWithUnpack(1), ключи объекта проверяются в VM.
        let is_single_unpack = processed_args.len() == 1
            && matches!(&processed_args[0], Arg::UnpackObject(_));
        
        // Компилируем аргументы на стек
        for arg in &processed_args {
            match arg {
                Arg::Positional(expr) => {
                    expr::compile_expr(ctx, expr)?;
                }
                Arg::Named { value, .. } => {
                    expr::compile_expr(ctx, value)?;
                }
                Arg::UnpackObject(expr) => {
                    expr::compile_expr(ctx, expr)?;
                }
            }
        }
        
        // Загружаем функцию: локальные → function_names (user-defined) → globals (builtins) → globals / новый слот
        if let Some(local_index) = ctx.scope.resolve_local(name) {
            // Локальная переменная содержит функцию
            ctx.chunk.write_with_line(OpCode::LoadLocal(local_index), *ctx.current_line);
        } else if ctx.function_names.iter().any(|n| n == name) {
            // Пользовательская функция найдена. Если у неё есть глобальный слот (main, __main__ и т.д.),
            // используем LoadGlobal, чтобы брать значение из слота, установленного set_functions, и не
            // полагаться на константный пул (избегаем путаницы с argv или другими глобалами).
            if let Some(&global_index) = ctx.scope.globals.get(name) {
                ctx.chunk.global_names.insert(global_index, name.clone());
                ctx.chunk.write_with_line(OpCode::LoadGlobal(global_index), *ctx.current_line);
            } else {
                let function_index = ctx.function_names.iter().position(|n| n == name).unwrap();
                let constant_index = ctx.chunk.add_constant(Value::Function(function_index));
                ctx.chunk.write_with_line(OpCode::Constant(constant_index), *ctx.current_line);
            }
        } else if is_lowercase && ctx.scope.globals.get(name).is_some() {
            // Для имён с маленькой буквы без пользовательского переопределения — встроенные
            let &global_index = ctx.scope.globals.get(name).unwrap();
            ctx.chunk.global_names.insert(global_index, name.clone());
            ctx.chunk.write_with_line(OpCode::LoadGlobal(global_index), *ctx.current_line);
        } else if let Some(&global_index) = ctx.scope.globals.get(name) {
            // Глобальная переменная содержит функцию (встроенная или импорт)
            ctx.chunk.global_names.insert(global_index, name.clone());
            ctx.chunk.write_with_line(OpCode::LoadGlobal(global_index), *ctx.current_line);
        } else {
            // Функция не найдена — ошибка на этапе компиляции
            return Err(LangError::ParseError {
                message: format!("Undefined function: {}", name),
                line: *line,
                file: None,
            });
        }
        
        // Вызываем функцию: при единственном **obj — CallWithUnpack(1), иначе Call(n)
        if is_single_unpack {
            ctx.chunk.write_with_line(OpCode::CallWithUnpack(1), *line);
        } else {
            ctx.chunk.write_with_line(OpCode::Call(processed_args.len()), *line);
        }
        
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
            file: None,
        })
    }
}

