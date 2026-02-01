/// Компиляция вызовов функций

use crate::debug_println;
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
        
            // Сначала проверяем в function_names (для конструкторов, определенных в текущем файле)
        if let Some(function_index) = ctx.function_names.iter().position(|n| n == &constructor_name) {
            // Это вызов конструктора
            debug_println!("[DEBUG compile_call] Найден конструктор '{}' с индексом функции {} в function_names", constructor_name, function_index);
            // Сохраняем количество аргументов до компиляции
            let arg_count = call_args.len();
            debug_println!("[DEBUG compile_call] Сохранено количество аргументов: {} для конструктора '{}' (function_names)", arg_count, constructor_name);
            
            // Компилируем аргументы ПЕРЕД загрузкой конструктора
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
                }
            }
            
            // Вызываем конструктор
            let constant_index = ctx.chunk.add_constant(Value::Function(function_index));
            let constant_ip = ctx.chunk.code.len();
            ctx.chunk.write_with_line(OpCode::Constant(constant_index), *line);
            let call_ip = ctx.chunk.code.len();
            ctx.chunk.write_with_line(OpCode::Call(arg_count), *line);
            debug_println!("[DEBUG compile_call] Сгенерирован OpCode::Constant({}) на IP {}, OpCode::Call({}) на IP {} для конструктора '{}' (function_names)", 
                constant_index, constant_ip, arg_count, call_ip, constructor_name);
            
            // Проверяем, что инструкция действительно записана правильно
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
            // Сохраняем количество аргументов до компиляции
            let arg_count = call_args.len();
            debug_println!("[DEBUG compile_call] Сохранено количество аргументов: {} для конструктора '{}' (globals)", arg_count, constructor_name);
            
            // Компилируем аргументы ПЕРЕД загрузкой конструктора
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
                }
            }
            
            // Загружаем конструктор из глобальной переменной и вызываем его
            let load_global_ip = ctx.chunk.code.len();
            ctx.chunk.write_with_line(OpCode::LoadGlobal(global_index), *line);
            let call_ip = ctx.chunk.code.len();
            ctx.chunk.write_with_line(OpCode::Call(arg_count), *line);
            debug_println!("[DEBUG compile_call] Сгенерирован OpCode::LoadGlobal({}) на IP {}, OpCode::Call({}) на IP {} для конструктора '{}' (globals)", 
                global_index, load_global_ip, arg_count, call_ip, constructor_name);
            
            // Проверяем, что инструкция действительно записана правильно
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
            // Сохраняем количество аргументов до компиляции
            let arg_count = call_args.len();
            debug_println!("[DEBUG compile_call] Сохранено количество аргументов: {} для конструктора '{}' (класс в globals)", arg_count, constructor_name);
            
            // Компилируем аргументы ПЕРЕД загрузкой конструктора
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
                }
            }
            
            // Загружаем конструктор из глобальной переменной по имени конструктора
            // Если конструктор не найден в scope.globals, создаем новый индекс для него
            let constructor_global_index = if let Some(&idx) = ctx.scope.globals.get(&constructor_name) {
                idx
            } else {
                // Создаем новый индекс для конструктора (он будет установлен во время выполнения)
                let idx = ctx.scope.globals.len();
                ctx.scope.globals.insert(constructor_name.clone(), idx);
                ctx.chunk.global_names.insert(idx, constructor_name.clone());
                idx
            };
            
            let load_global_ip = ctx.chunk.code.len();
            ctx.chunk.write_with_line(OpCode::LoadGlobal(constructor_global_index), *line);
            let call_ip = ctx.chunk.code.len();
            ctx.chunk.write_with_line(OpCode::Call(arg_count), *line);
            debug_println!("[DEBUG compile_call] Сгенерирован OpCode::LoadGlobal({}) на IP {}, OpCode::Call({}) на IP {} для конструктора '{}' (класс в globals)", 
                constructor_global_index, load_global_ip, arg_count, call_ip, constructor_name);
            
            // Проверяем, что инструкция действительно записана правильно
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
            
            // ВАЖНО: Убеждаемся, что в chunk.global_names для constructor_global_index записано имя конструктора, а не класса
            if let Some(existing_name) = ctx.chunk.global_names.get(&constructor_global_index) {
                if existing_name != &constructor_name {
                    debug_println!("[DEBUG compile_call] ВНИМАНИЕ: В chunk.global_names для индекса {} записано имя '{}', но ожидается '{}'. Исправляем.", 
                        constructor_global_index, existing_name, constructor_name);
                    ctx.chunk.global_names.insert(constructor_global_index, constructor_name.clone());
                }
            }
            
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
            matches!(name, "ml" | "plot" | "settings_env" | "uuid")
        }
        if name.chars().next().map(|c| c.is_uppercase()).unwrap_or(false)
            && ctx.imported_symbols.get(name).map_or(false, |m| !is_builtin_module(m))
        {
            // Регистрируем класс в scope, если его там еще нет (уже есть от import, но индекс нужен для консистентности)
            if !ctx.scope.globals.contains_key(name) {
                let class_global_index = ctx.scope.globals.len();
                ctx.scope.globals.insert(name.clone(), class_global_index);
            }
            // Регистрируем конструктор в scope и chunk.global_names
            let constructor_global_index = if let Some(&idx) = ctx.scope.globals.get(&constructor_name) {
                idx
            } else {
                let idx = ctx.scope.globals.len();
                ctx.scope.globals.insert(constructor_name.clone(), idx);
                ctx.chunk.global_names.insert(idx, constructor_name.clone());
                idx
            };
            if let Some(existing_name) = ctx.chunk.global_names.get(&constructor_global_index) {
                if existing_name != &constructor_name {
                    ctx.chunk.global_names.insert(constructor_global_index, constructor_name.clone());
                }
            }
            let arg_count = call_args.len();
            for arg in call_args.iter() {
                match arg {
                    Arg::Positional(expr) => expr::compile_expr(ctx, expr)?,
                    Arg::Named { value, .. } => expr::compile_expr(ctx, value)?,
                }
            }
            ctx.chunk.write_with_line(OpCode::LoadGlobal(constructor_global_index), *line);
            ctx.chunk.write_with_line(OpCode::Call(arg_count), *line);
            return Ok(());
        }
        
        debug_println!("[DEBUG compile_call] Конструктор '{}' не найден ни в function_names, ни в globals, класс '{}' тоже не найден, проверяем как обычную функцию", constructor_name, name);
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
        
        // Загружаем функцию: локальные → для имён с маленькой буквы сначала globals (builtins) → function_names → globals / новый слот
        if let Some(local_index) = ctx.scope.resolve_local(name) {
            // Локальная переменная содержит функцию
            ctx.chunk.write_with_line(OpCode::LoadLocal(local_index), *ctx.current_line);
        } else if is_lowercase && ctx.scope.globals.get(name).is_some() {
            // Для имён с маленькой буквы сначала резолвим через globals (builtins)
            let &global_index = ctx.scope.globals.get(name).unwrap();
            ctx.chunk.global_names.insert(global_index, name.clone());
            ctx.chunk.write_with_line(OpCode::LoadGlobal(global_index), *ctx.current_line);
        } else if let Some(function_index) = ctx.function_names.iter().position(|n| n == name) {
            // Пользовательская функция найдена
            let constant_index = ctx.chunk.add_constant(Value::Function(function_index));
            ctx.chunk.write_with_line(OpCode::Constant(constant_index), *ctx.current_line);
        } else if let Some(&global_index) = ctx.scope.globals.get(name) {
            // Глобальная переменная содержит функцию (встроенная или импорт)
            ctx.chunk.global_names.insert(global_index, name.clone());
            ctx.chunk.write_with_line(OpCode::LoadGlobal(global_index), *ctx.current_line);
        } else {
            // Функция не найдена — регистрируем как глобальную для разрешения во время выполнения
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

