// Публичный API языка DataCode (новая архитектура Bytecode + VM)

pub mod abi;
pub mod common;
pub mod dpm;
pub mod lexer;
pub mod parser;
pub mod semantic;
pub mod bytecode;
pub mod compiler;
pub mod vm;

#[path = "lib/websocket/mod.rs"]
pub mod websocket;
#[path = "lib/sqlite_export/mod.rs"]
pub mod sqlite_export;
#[path = "lib/ml/mod.rs"]
pub mod ml;
#[path = "lib/plot/mod.rs"]
pub mod plot;
#[path = "lib/settings_env/mod.rs"]
pub mod settings_env;
#[path = "lib/uuid/mod.rs"]
pub mod uuid;
#[path = "lib/database/mod.rs"]
pub mod database;

// Публичный API для запуска интерпретатора
pub use common::{error::LangError, value::Value};
pub use bytecode::Chunk;
pub use vm::Vm;

pub fn run(source: &str) -> Result<Value, LangError> {
    run_with_existing_vm(source, None)
}

/// Выполняет код с возможностью использования существующего VM для глобальных переменных
pub fn run_with_existing_vm(source: &str, existing_vm: Option<&mut Vm>) -> Result<Value, LangError> {
    use lexer::Lexer;
    use parser::Parser;
    use semantic::resolver::Resolver;
    use compiler::Compiler;
    use vm::Vm;

    // 1. Лексический анализ
    let mut lexer = Lexer::new(source);
    let tokens = lexer.tokenize()?;

    // 2. Парсинг
    let mut parser = Parser::new(tokens);
    let ast = parser.parse()?;

    // 3. Семантический анализ
    let mut resolver = Resolver::new();
    resolver.resolve(&ast)?;

    // 4. Компиляция в байт-код
    let mut compiler = Compiler::new();
    let mut chunk = compiler.compile(&ast)?;
    let functions = compiler.get_functions();

    // 5. Выполнение на VM
    let vm = if let Some(existing) = existing_vm {
        // Используем существующий VM (глобальные переменные уже установлены)
        existing
    } else {
        // Создаем новый VM
        Box::leak(Box::new(Vm::new()))
    };
    
    vm.register_native_globals();
    vm.ensure_globals_from_chunk(&chunk);
    for f in &functions {
        vm.ensure_globals_from_chunk(&f.chunk);
    }
    vm.register_all_builtin_modules()?;
    vm.set_functions(functions, Some(&mut chunk));
    let result = vm.run(&chunk)?;

    Ok(result)
}

/// Выполняет код с возможностью передачи базового пути для импортов и существующего VM.
/// Когда передан base_path, он задаётся в thread-local и явно передаётся во внутренний VM,
/// чтобы разрешение модулей и load_env не зависело от порядка вызовов.
pub fn run_with_vm_and_path(source: &str, base_path: Option<&std::path::Path>, existing_vm: Option<&mut Vm>) -> Result<(Value, Vm), LangError> {
    use crate::vm::file_import;
    use std::path::PathBuf;

    let explicit_base = base_path.map(PathBuf::from);

    if let Some(ref path) = explicit_base {
        file_import::set_base_path(Some(path.clone()));
    }

    if let Some(vm) = existing_vm {
        run_with_vm_into_vm(source, vm)
    } else {
        run_with_vm_internal_with_args(source, None, None, explicit_base)
    }
}

/// Выполняет код в существующий VM
fn run_with_vm_into_vm(source: &str, vm: &mut Vm) -> Result<(Value, Vm), LangError> {
    use lexer::Lexer;
    use parser::Parser;
    use semantic::resolver::Resolver;
    use compiler::Compiler;

    // 1. Лексический анализ
    let mut lexer = Lexer::new(source);
    let tokens = lexer.tokenize()?;

    // 2. Парсинг
    let mut parser = Parser::new(tokens);
    let ast = parser.parse()?;

    // 3. Семантический анализ
    let mut resolver = Resolver::new();
    resolver.resolve(&ast)?;

    // 4. Компиляция в байт-код
    let mut compiler = Compiler::new();
    let chunk = compiler.compile(&ast)?;
    let functions = compiler.get_functions();

    // 5. Добавляем функции в существующий VM
    vm.add_functions(functions);
    
    // Обновляем индексы функций для новых функций
    // (это нужно для правильной работы вызовов функций)
    
    // 6. Выполнение на существующем VM
    let result = vm.run(&chunk)?;

    Ok((result, Vm::new())) // Возвращаем пустой VM, так как результат уже в существующем
}

/// Выполняет код и возвращает VM для доступа к глобальным переменным
pub fn run_with_vm(source: &str) -> Result<(Value, Vm), LangError> {
    run_with_vm_with_args(source, None)
}

/// Выполняет код с аргументами командной строки и возвращает VM
pub fn run_with_vm_with_args(source: &str, args: Option<Vec<String>>) -> Result<(Value, Vm), LangError> {
    run_with_vm_with_args_and_lib(source, args, None, None)
}

/// Выполняет код с заданным базовым путём (для разрешения локальных модулей и load_env).
/// Удобно для тестов и сценариев, когда код выполняется по строке, но нужен импорт из .dc модуля (например `from config import Config`).
pub fn run_with_base_path(source: &str, base_path: &std::path::Path) -> Result<Value, LangError> {
    run_with_vm_and_path(source, Some(base_path), None).map(|(v, _)| v)
}

/// Выполняет код с аргументами командной строки, путём к __lib__.dc и опциональным базовым путём скрипта.
/// base_path: при запуске файла из CLI передавать директорию скрипта, чтобы VM разрешал импорты и load_env относительно неё.
pub fn run_with_vm_with_args_and_lib(
    source: &str,
    args: Option<Vec<String>>,
    lib_path: Option<&std::path::Path>,
    base_path: Option<&std::path::Path>,
) -> Result<(Value, Vm), LangError> {
    run_with_vm_internal_with_args(
        source,
        args,
        lib_path,
        base_path.map(std::path::PathBuf::from),
    )
}

/// Извлекает глобальные переменные из VM как HashMap
pub fn extract_globals_from_vm(vm: &Vm) -> std::collections::HashMap<String, Value> {
    let mut globals_map = std::collections::HashMap::new();
    let globals = vm.get_globals();
    let global_names = vm.get_global_names();
    
    for (index, name) in global_names.iter() {
        if let Some(value) = globals.get(*index) {
            globals_map.insert(name.clone(), value.clone());
        }
    }
    
    globals_map
}

/// Собирает имена модулей из `from X import ...` в AST (для поиска lib в папках при старте).
fn import_module_names_from_ast(ast: &[parser::ast::Stmt]) -> Vec<String> {
    use parser::ast::{ImportStmt, Stmt};
    let mut names = Vec::new();
    for stmt in ast {
        if let Stmt::Import { import_stmt, .. } = stmt {
            if let ImportStmt::From { module, .. } = import_stmt {
                names.push(module.clone());
            }
        }
    }
    names
}

/// Внутренняя функция выполнения кода (используется через run_with_vm_internal_with_args)
#[allow(dead_code)]
fn run_with_vm_internal(source: &str) -> Result<(Value, Vm), LangError> {
    run_with_vm_internal_with_args(source, None, None, None)
}

/// Внутренняя функция выполнения кода с аргументами
/// 
/// # Параметры
/// * `source` - исходный код для выполнения
/// * `args` - аргументы командной строки
/// * `lib_path` - путь к __lib__.dc. Если Some, то мы выполняем __lib__.dc и не должны искать его автоматически.
///               Если None, то пытаемся автоматически найти __lib__.dc в базовом пути.
fn run_with_vm_internal_with_args(
    source: &str,
    args: Option<Vec<String>>,
    lib_path: Option<&std::path::Path>,
    explicit_base_path: Option<std::path::PathBuf>,
) -> Result<(Value, Vm), LangError> {
    use lexer::Lexer;
    use parser::Parser;
    use semantic::resolver::Resolver;
    use compiler::Compiler;
    use vm::Vm;
    use vm::file_import;
    use std::rc::Rc;
    use std::cell::RefCell;

    let base_for_lib = explicit_base_path.clone().or_else(file_import::get_base_path);

    // Парсинг до выбора lib, чтобы при отсутствии __lib__.dc в директории искать папки по импортам (from X import ...)
    let mut lexer = Lexer::new(source);
    let tokens = lexer.tokenize()?;
    let mut parser = Parser::new(tokens);
    let ast = parser.parse()?;

    // Определяем путь к __lib__.dc
    // Если lib_path указан, это означает, что мы хотим загрузить __lib__.dc для основного файла
    // Если lib_path не указан, пытаемся автоматически найти __lib__.dc в базовом пути
    // НО только если мы не выполняем уже __lib__.dc (проверяем флаг is_executing_lib)
    let lib_vm = if file_import::is_executing_lib() {
        // Мы уже выполняем __lib__.dc - не загружаем его снова
        debug_println!("[DEBUG run_with_vm_internal_with_args] Уже выполняем __lib__.dc, не загружаем снова");
        None
    } else if let Some(lib_path_val) = lib_path {
        // lib_path указан явно - загружаем __lib__.dc
        debug_println!("[DEBUG run_with_vm_internal_with_args] lib_path указан явно: {:?}, загружаем __lib__.dc", lib_path_val);
        Some(run_lib_file(lib_path_val)?)
    } else {
        // lib_path не указан - пытаемся автоматически найти __lib__.dc
        if let Some(base_path) = base_for_lib {
            // Ищем ближайший __lib__.dc, поднимаясь вверх по дереву каталогов
            debug_println!(
                "[DEBUG run_with_vm_internal_with_args] Ищем ближайший __lib__.dc, начиная с базового пути: {:?}",
                base_path
            );
            let potential_lib_path = file_import::find_nearest_lib(&base_path).or_else(|| {
                // Если в директории (и выше) нет __lib__.dc — ищем папки по импортам (from X import ...)
                let module_names = import_module_names_from_ast(&ast);
                if !module_names.is_empty() {
                    debug_println!(
                        "[DEBUG run_with_vm_internal_with_args] Ищем __lib__.dc в папках по импортам: {:?}",
                        module_names
                    );
                    file_import::find_lib_in_package_dirs(&base_path, &module_names)
                } else {
                    None
                }
            });
            if let Some(potential_lib_path) = potential_lib_path {
                debug_println!(
                    "[DEBUG run_with_vm_internal_with_args] Автоматически найден __lib__.dc: {:?}",
                    potential_lib_path
                );
                // Передаем найденный путь в run_lib_file
                // run_lib_file установит флаг is_executing_lib, чтобы предотвратить автоматический поиск
                Some(run_lib_file(&potential_lib_path)?)
            } else {
                debug_println!(
                    "[DEBUG run_with_vm_internal_with_args] __lib__.dc не найден ни в базовом пути, ни выше по дереву каталогов, ни в папках по импортам, начиная с: {:?}",
                    base_path
                );

                // Специальное правило: для сценариев в директории `test_src`
                // наличие __lib__.dc обязательно. Если базовый путь оканчивается
                // на `test_src` и библиотека не найдена, возвращаем ошибку.
                if base_path.ends_with("test_src") {
                    return Err(LangError::runtime_error(
                        format!(
                            "Для файлов в директории '{}' требуется __lib__.dc (файл не найден ни в самой директории, ни выше по дереву каталогов)",
                            base_path.display()
                        ),
                        0,
                    ));
                }

                None
            }
        } else {
            debug_println!("[DEBUG run_with_vm_internal_with_args] Базовый путь не установлен, не можем найти __lib__.dc");
            None
        }
    };

    // 3. Семантический анализ
    let mut resolver = Resolver::new();
    resolver.resolve(&ast)?;

    // 4. Компиляция в байт-код
    let mut compiler = Compiler::new();
    let chunk = compiler.compile(&ast)?;
    let functions = compiler.get_functions();
    // Отладка: главный chunk должен содержать "Config" в global_names (from config import Config)
    let config_in_chunk: Vec<usize> = chunk.global_names.iter().filter(|(_, n)| n.as_str() == "Config").map(|(i, _)| *i).collect();
    debug_println!("[DEBUG run_with_vm_internal_with_args] После компиляции: главный chunk global_names содержит 'Config': {} (индексы: {:?})", !config_in_chunk.is_empty(), config_in_chunk);

    // 5. Выполнение на VM
    let mut vm = Vm::new();
    // Сразу задаём base_path в VM, чтобы импорты и load_env разрешались детерминированно
    vm.set_base_path(explicit_base_path.clone().or_else(file_import::get_base_path));

    // Сначала регистрируем нативные функции (индексы 0-69)
    vm.register_native_globals();
    // Добавляем слоты для глобалов из chunk (в т.ч. имён модулей и импортов: Config из "from config import Config")
    // ДО регистрации встроенных модулей и ДО merge из __lib__.dc, чтобы merge перезаписывал существующий слот,
    // а не создавал второй слот с тем же именем (детерминизм: sentinel в конструкторах резолвится в один и тот же индекс).
    vm.ensure_globals_from_chunk(&chunk);
    for f in &functions {
        vm.ensure_globals_from_chunk(&f.chunk);
    }
    // Регистрируем встроенные модули (ml, plot, settings_env, uuid) — они заполняют слоты по имени
    vm.register_all_builtin_modules()?;
    
    // Объединяем глобальные переменные из __lib__.dc (один раз), затем патчим байткод функций lib пост-merge global_names.
    if let Some(ref lib_vm) = lib_vm {
        debug_println!("[DEBUG run_with_vm_internal_with_args] Объединяем глобальные переменные из __lib__.dc");
        let lib_function_count = lib_vm.get_functions().len();
        vm.merge_globals_from(lib_vm);
        // Патч чанков lib один раз, используя актуальный global_names после merge (Config в слоте 71 и т.д.).
        let global_names = vm.get_global_names().clone();
        for i in 0..lib_function_count {
            crate::vm::vm::Vm::update_chunk_indices_from_names(
                &mut vm.get_functions_mut()[i].chunk,
                &global_names,
                None, // cannot pass globals while borrowing vm mutably for chunk
            );
        }
        debug_println!("[DEBUG run_with_vm_internal_with_args] После объединения __lib__.dc, глобальных переменных: {}", vm.get_global_names().len());
        // Отладочный вывод: проверяем, что Data и Data::new_3 установлены правильно
        for (idx, name) in vm.get_global_names().iter() {
            if name.contains("Data") {
                let value = vm.get_globals().get(*idx);
                debug_println!("[DEBUG run_with_vm_internal_with_args] После merge: '{}' в globals[{}] = {:?}", name, idx,
                    match value {
                        Some(Value::Null) => "Null",
                        Some(Value::Function(_)) => "Function",
                        Some(Value::Object(_)) => "Object",
                        Some(_) => "Other",
                        None => "None",
                    });
            }
        }
    } else {
        debug_println!("[DEBUG run_with_vm_internal_with_args] __lib__.dc не загружен (lib_path не указан)");
    }
    
    // Устанавливаем аргументы командной строки как глобальную переменную argv
    // Делаем это ДО set_functions, чтобы индексы обновлялись правильно
    // Всегда устанавливаем argv, даже если аргументы не переданы (пустой массив)
    // Это нужно, потому что компилятор может генерировать код, который обращается к argv
    let script_args = args.unwrap_or_default();
    let argv: Vec<Value> = script_args.iter().map(|s| Value::String(s.clone())).collect();
    let argv_array = Value::Array(Rc::new(RefCell::new(argv)));
    let argv_index = vm.get_globals().len();
    vm.get_globals_mut().push(argv_array);
    vm.get_global_names_mut().insert(argv_index, "argv".to_string());
    
    // Добавляем слоты для глобальных переменных из главного chunk (в т.ч. __main__), которых ещё нет в VM
    vm.ensure_globals_from_chunk(&chunk);
    // Проверка: есть ли "Config" в global_names перед set_functions (для отладки sentinel/недетерминизма)
    let config_indices: Vec<usize> = vm.get_global_names().iter()
        .filter(|(_, n)| n.as_str() == "Config")
        .map(|(idx, _)| *idx)
        .collect();
    if let Some(&idx) = config_indices.iter().min() {
        debug_println!("[DEBUG run_with_vm_internal_with_args] После второго ensure: 'Config' в global_names под индексом {} (всего слотов с именем Config: {})", idx, config_indices.len());
    } else {
        debug_println!("[DEBUG run_with_vm_internal_with_args] После второго ensure: 'Config' НЕ найден в global_names");
    }
    // Обновляем индексы в главном chunk на основе реальных индексов в VM
    // Это нужно сделать ДО set_functions, чтобы все индексы были согласованы
    // Главный chunk может содержать ссылки на глобальные переменные (например, argv),
    // которые были установлены в VM после merge из __lib__.dc
    let mut chunk = chunk;
    vm.update_chunk_indices(&mut chunk);
    
    // Затем устанавливаем функции основного скрипта (передаём main chunk, чтобы all_pairs включал его global_names
    // и "Config" резолвился в слот 71 из ensure, а не создавался новый слот 74).
    vm.set_functions(functions, Some(&mut chunk));
    
    // Отладочный вывод: проверяем, что Data и Data::new_3 установлены правильно после set_functions
    for (idx, name) in vm.get_global_names().iter() {
        if name.contains("Data") {
            let value = vm.get_globals().get(*idx);
            debug_println!("[DEBUG run_with_vm_internal_with_args] После set_functions: '{}' в globals[{}] = {:?}", name, idx,
                match value {
                    Some(Value::Null) => "Null",
                    Some(Value::Function(_)) => "Function",
                    Some(Value::Object(_)) => "Object",
                    Some(_) => "Other",
                    None => "None",
                });
        }
    }
    
    // Восстанавливаем слоты встроенных нативов (0..BUILTIN_GLOBAL_NAMES.len()), чтобы пользовательские функции с тем же именем не перезаписывали их
    use crate::vm::globals;
    for (idx, &name) in globals::BUILTIN_GLOBAL_NAMES.iter().enumerate() {
        if vm.get_global_names().get(&idx).map(|s| s.as_str()) == Some(name) {
            vm.get_globals_mut()[idx] = Value::NativeFunction(idx);
        }
    }

    // Use explicit base path when provided (e.g. from run_with_vm_and_path), else thread-local
    vm.set_base_path(explicit_base_path.or_else(file_import::get_base_path));
    let result = vm.run(&chunk)?;

    Ok((result, vm))
}

/// Выполняет __lib__.dc файл и возвращает VM с глобальными переменными
pub fn run_lib_file(lib_path: &std::path::Path) -> Result<Vm, LangError> {
    use crate::vm::file_import;
    
    // Сохраняем текущий базовый путь
    let old_base_path = file_import::get_base_path();
    
    // Устанавливаем флаг, что мы выполняем __lib__.dc
    // Это предотвратит автоматический поиск __lib__.dc в рекурсивном вызове
    file_import::set_executing_lib(true);
    
    // Устанавливаем базовый путь для импортов в __lib__.dc
    // Базовый путь должен указывать на директорию, где находится __lib__.dc
    // чтобы модули (например, data.dc) могли быть найдены
    if let Some(base_path) = lib_path.parent() {
        file_import::set_base_path(Some(base_path.to_path_buf()));
    }
    
    let source = std::fs::read_to_string(lib_path).map_err(|e| {
        LangError::runtime_error(
            format!("Failed to read __lib__.dc: {}", e),
            0,
        )
    })?;
    
    // Выполняем __lib__.dc без argv (пустой массив)
    // Флаг is_executing_lib предотвратит автоматический поиск __lib__.dc
    let (_, vm) = run_with_vm_internal_with_args(&source, Some(Vec::new()), None, None)?;
    
    // Снимаем флаг выполнения __lib__.dc
    file_import::set_executing_lib(false);
    
    // Восстанавливаем старый базовый путь
    file_import::set_base_path(old_base_path);
    
    Ok(vm)
}

/// Компилирует код в байт-код без выполнения (для отладки)
pub fn compile(source: &str) -> Result<(Chunk, Vec<bytecode::Function>), LangError> {
    use lexer::Lexer;
    use parser::Parser;
    use semantic::resolver::Resolver;
    use compiler::Compiler;

    // 1. Лексический анализ
    let mut lexer = Lexer::new(source);
    let tokens = lexer.tokenize()?;

    // 2. Парсинг
    let mut parser = Parser::new(tokens);
    let ast = parser.parse()?;

    // 3. Семантический анализ
    let mut resolver = Resolver::new();
    resolver.resolve(&ast)?;

    // 4. Компиляция в байт-код
    let mut compiler = Compiler::new();
    let chunk = compiler.compile(&ast)?;
    let functions = compiler.get_functions();

    Ok((chunk, functions))
}

/// Выполняет код с включенным debug mode (выводит байт-код)
pub fn run_debug(source: &str) -> Result<Value, LangError> {
    use lexer::Lexer;
    use parser::Parser;
    use semantic::resolver::Resolver;
    use compiler::Compiler;
    use vm::Vm;

    // 1. Лексический анализ
    let mut lexer = Lexer::new(source);
    let tokens = lexer.tokenize()?;

    // 2. Парсинг
    let mut parser = Parser::new(tokens);
    let ast = parser.parse()?;

    // 3. Семантический анализ
    let mut resolver = Resolver::new();
    resolver.resolve(&ast)?;

    // 4. Компиляция в байт-код
    let mut compiler = Compiler::new();
    let chunk = compiler.compile(&ast)?;
    let functions = compiler.get_functions();

    // Debug: выводим байт-код
    println!("{}", chunk.disassemble("<main>"));
    for function in &functions {
        println!("{}", function.chunk.disassemble(&function.name));
    }

    // 5. Выполнение на VM
    let mut vm = Vm::new();
    vm.set_functions(functions, None);
    vm.register_native_globals();
    let result = vm.run(&chunk)?;

    Ok(result)
}

