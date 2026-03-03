// Публичный API языка DataCode (новая архитектура Bytecode + VM)

#[cfg(feature = "allocator_jemalloc")]
#[global_allocator]
static GLOBAL: jemallocator::Jemalloc = jemallocator::Jemalloc;

pub mod abi;
pub mod common;
pub mod dpm;
pub mod lexer;
pub mod parser;
pub mod semantic;
pub mod bytecode;
pub mod compiler;
pub mod vm;
pub mod infra;

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
#[path = "lib/database_engine/mod.rs"]
pub mod database_engine;

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

    // 5. Выполнение на VM (VM всегда имеет владельца; при отсутствии existing_vm используем локальный VM, без Box::leak)
    let mut local_vm = Vm::new();
    let vm: &mut Vm = existing_vm.unwrap_or(&mut local_vm);

    vm.register_native_globals();
    // Main chunk first, preserving indices (75, 76, …) so bytecode LoadGlobal matches VM slots.
    vm.ensure_globals_from_chunk_preserve_indices(&chunk);
    for f in &functions {
        vm.ensure_globals_from_chunk(&f.chunk);
    }
    vm.register_all_builtin_modules()?;
    // Sync Path B with Path A: update LoadGlobal/StoreGlobal indices to match vm.global_names
    // before set_functions, so create_all and other globals resolve correctly (fixes "no such table: users").
    vm.update_chunk_indices(&mut chunk);
    let mut functions = functions;
    for f in &mut functions {
        vm.update_chunk_indices(&mut f.chunk);
    }
    // set_functions patches main chunk and all function chunks via name_to_new_idx.
    vm.set_functions(functions, Some(&mut chunk), None);
    let result = vm.run(&chunk, None)?;

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
        run_with_vm_internal_with_args(source, None, None, explicit_base, None)
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
    let result = vm.run(&chunk, None)?;

    Ok((result, Vm::new())) // Возвращаем пустой VM, так как результат уже в существующем
}

/// Выполняет код и возвращает VM для доступа к глобальным переменным
pub fn run_with_vm(source: &str) -> Result<(Value, Vm), LangError> {
    run_with_vm_with_args(source, None)
}

/// Выполняет код с аргументами командной строки и возвращает VM
pub fn run_with_vm_with_args(source: &str, args: Option<Vec<String>>) -> Result<(Value, Vm), LangError> {
    run_with_vm_with_args_and_lib(source, args, None, None, None)
}

/// Выполняет код с заданным базовым путём (для разрешения локальных модулей и load_env).
/// Удобно для тестов и сценариев, когда код выполняется по строке, но нужен импорт из .dc модуля (например `from config import Config`).
pub fn run_with_base_path(source: &str, base_path: &std::path::Path) -> Result<Value, LangError> {
    run_with_vm_and_path(source, Some(base_path), None).map(|(v, _)| v)
}

/// Выполняет код с аргументами командной строки, путём к __lib__.dc и опциональным базовым путём скрипта.
/// base_path: при запуске файла из CLI передавать директорию скрипта, чтобы VM разрешал импорты и load_env относительно неё.
/// source_name: путь к исходному файлу для сообщений об ошибках (например, путь к .dc скрипту).
pub fn run_with_vm_with_args_and_lib(
    source: &str,
    args: Option<Vec<String>>,
    lib_path: Option<&std::path::Path>,
    base_path: Option<&std::path::Path>,
    source_name: Option<&std::path::Path>,
) -> Result<(Value, Vm), LangError> {
    run_with_vm_internal_with_args(
        source,
        args,
        lib_path,
        base_path.map(std::path::PathBuf::from),
        source_name.map(std::path::PathBuf::from),
    )
}

/// Извлекает глобальные переменные из VM как HashMap (globals are GlobalSlot)
pub fn extract_globals_from_vm(vm: &mut Vm) -> std::collections::HashMap<String, Value> {
    use crate::vm::store_convert::load_value;
    let mut globals_map = std::collections::HashMap::new();
    let globals = vm.get_globals();
    let global_names = vm.get_global_names();
    let to_extract: Vec<(usize, String)> = global_names
        .iter()
        .filter_map(|(index, name)| globals.get(*index).map(|_| (*index, name.clone())))
        .collect();
    for (_index, name) in to_extract {
        let value_id = vm.resolve_global_to_value_id(_index);
        let value = load_value(value_id, vm.value_store(), vm.heavy_store());
        globals_map.insert(name, value);
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

/// Извлекает литеральное значение по умолчанию из Expr (только Literal).
fn expr_default_to_value(expr: &parser::ast::Expr) -> Option<Value> {
    if let parser::ast::Expr::Literal { value, .. } = expr {
        Some(value.clone())
    } else {
        None
    }
}

/// Параметры `fn __main__(...)` для CLI: имена и опциональные значения по умолчанию (только литералы).
/// По ним распознаются опции `--имя=значение` и строится argv в порядке параметров.
pub fn get_main_entry_params(source: &str) -> Option<Vec<(String, Option<Value>)>> {
    use lexer::Lexer;
    use parser::Parser;
    use parser::ast::Stmt;
    let mut lexer = Lexer::new(source);
    let tokens = lexer.tokenize().ok()?;
    let mut parser = Parser::new(tokens);
    let ast = parser.parse().ok()?;
    for stmt in &ast {
        if let Stmt::Function { name, params, .. } = stmt {
            if name == "__main__" {
                return Some(
                    params
                        .iter()
                        .map(|p| {
                            let default = p
                                .default_value
                                .as_ref()
                                .and_then(expr_default_to_value);
                            (p.name.clone(), default)
                        })
                        .collect(),
                );
            }
        }
    }
    None
}

/// Remap function indices in an export map (e.g. from __lib__ or imported module) so they refer to caller VM's function table.
pub(crate) fn remap_function_indices_in_exports(exports: &mut std::collections::HashMap<String, Value>, start_idx: usize) {
    fn remap_value(v: &mut Value, start_idx: usize) {
        match v {
            Value::Function(i) => *i = start_idx + *i,
            Value::Object(rc) => {
                for (_, inner) in rc.borrow_mut().iter_mut() {
                    remap_value(inner, start_idx);
                }
            }
            _ => {}
        }
    }
    for v in exports.values_mut() {
        remap_value(v, start_idx);
    }
}

/// Replace Value::Function(local_index) with Value::ModuleFunction { module_id, local_index } in an export map.
/// Used by executor after merging a module so namespace is valid across cache hits.
pub(crate) fn replace_function_with_module_function_in_exports(
    exports: &mut std::collections::HashMap<String, Value>,
    module_id: usize,
) {
    fn replace_value(v: &mut Value, module_id: usize) {
        match v {
            Value::Function(local_index) => {
                *v = Value::ModuleFunction {
                    module_id,
                    local_index: *local_index,
                };
            }
            Value::Object(rc) => {
                for (_, inner) in rc.borrow_mut().iter_mut() {
                    replace_value(inner, module_id);
                }
            }
            _ => {}
        }
    }
    for v in exports.values_mut() {
        replace_value(v, module_id);
    }
}

/// Remap ModuleFunction { module_id, local_index } in an export map so submodule IDs refer to caller VM's registry.
/// Used after replace_function_with_module_function_in_exports when the loaded module had submodules (e.g. dev_config);
/// class objects from submodules carry ModuleFunction(old_id, local_index) valid in the loaded VM; this rewrites
/// old_id to new_id so get_module_function_index resolves in the caller.
pub(crate) fn remap_module_function_ids_in_exports(
    exports: &mut std::collections::HashMap<String, Value>,
    old_to_new: &std::collections::HashMap<usize, usize>,
) {
    fn remap_value(v: &mut Value, old_to_new: &std::collections::HashMap<usize, usize>) {
        match v {
            Value::ModuleFunction { module_id, local_index: _ } => {
                if let Some(&new_id) = old_to_new.get(module_id) {
                    *module_id = new_id;
                }
            }
            Value::Object(rc) => {
                for (_, inner) in rc.borrow_mut().iter_mut() {
                    remap_value(inner, old_to_new);
                }
            }
            _ => {}
        }
    }
    for v in exports.values_mut() {
        remap_value(v, old_to_new);
    }
}

const BUILTIN_NATIVE_COUNT: usize = 75;

/// Remap Value::Function(local_i) in chunk constants of merged module functions to global indices.
/// After extending VM's function array with a module's functions, instances created by that module's
/// code (e.g. Security with method get_code) would otherwise store Value::Function(local_index);
/// calling such a method from the main script would then dispatch to the wrong function.
pub(crate) fn remap_function_constants_in_chunks(
    functions: &mut [crate::bytecode::Function],
    start_idx: usize,
    module_function_count: usize,
) {
    for f in functions.iter_mut().skip(start_idx) {
        for c in &mut f.chunk.constants {
            if let Value::Function(local_i) = c {
                if *local_i < module_function_count {
                    *local_i = start_idx + *local_i;
                }
            }
        }
    }
}

/// Remap NativeFunction indices in an export map so they refer to caller VM's native table.
/// Caller must have already extended its natives with module_natives[BUILTIN_NATIVE_COUNT..].
/// For NativeFunction(i) with i >= BUILTIN_NATIVE_COUNT: new_idx = native_start + (i - BUILTIN_NATIVE_COUNT).
pub(crate) fn remap_native_indices_in_exports(
    exports: &mut std::collections::HashMap<String, Value>,
    native_start: usize,
) {
    fn remap_value(v: &mut Value, native_start: usize) {
        match v {
            Value::NativeFunction(i) if *i >= BUILTIN_NATIVE_COUNT => {
                *i = native_start + (*i - BUILTIN_NATIVE_COUNT);
            }
            Value::Object(rc) => {
                for (_, inner) in rc.borrow_mut().iter_mut() {
                    remap_value(inner, native_start);
                }
            }
            _ => {}
        }
    }
    for v in exports.values_mut() {
        remap_value(v, native_start);
    }
}

/// Внутренняя функция выполнения кода (используется через run_with_vm_internal_with_args)
#[allow(dead_code)]
fn run_with_vm_internal(source: &str) -> Result<(Value, Vm), LangError> {
    run_with_vm_internal_with_args(source, None, None, None, None)
}

/// Внутренняя функция выполнения кода с аргументами
/// 
/// # Параметры
/// * `source` - исходный код для выполнения
/// * `args` - аргументы командной строки
/// * `lib_path` - путь к __lib__.dc. Если Some, то мы выполняем __lib__.dc и не должны искать его автоматически.
///               Если None, то пытаемся автоматически найти __lib__.dc в базовом пути.
/// * `source_name` - путь к исходному файлу для сообщений об ошибках.
fn run_with_vm_internal_with_args(
    source: &str,
    args: Option<Vec<String>>,
    lib_path: Option<&std::path::Path>,
    explicit_base_path: Option<std::path::PathBuf>,
    source_name: Option<std::path::PathBuf>,
) -> Result<(Value, Vm), LangError> {
    use lexer::Lexer;
    use parser::Parser;
    use semantic::resolver::Resolver;
    use compiler::Compiler;
    use vm::Vm;
    use vm::file_import;
    use std::rc::Rc;
    use std::cell::RefCell;

    // Set base_path up front so run_lib_file's save/restore keeps it; then load_env in Settings constructor can resolve relative paths.
    if let Some(ref base) = explicit_base_path {
        file_import::set_base_path(Some(base.clone()));
    }
    let base_for_lib = explicit_base_path.clone().or_else(file_import::get_base_path);
    let source_name_str = source_name.as_deref().map(|p| p.to_string_lossy().into_owned());

    // Парсинг до выбора lib, чтобы при отсутствии __lib__.dc в директории искать папки по импортам (from X import ...)
    let mut lexer = Lexer::new_with_source_name(source, source_name_str.as_deref());
    let tokens = lexer.tokenize()?;
    let mut parser = Parser::new_with_source_name(tokens, source_name_str.as_deref());
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
            // Сначала ищем __lib__.dc в папках по импортам (from X import ...), затем поднимаемся вверх по дереву
            let module_names = import_module_names_from_ast(&ast);
            // Ищем __lib__.dc только в директории скрипта или в папках импортов (base_path/<module>/__lib__.dc).
            // Не поднимаемся вверх по дереву (find_nearest_lib), иначе при запуске из sandbox/web_api
            // подхватывается __lib__.dc из корня репо, добавляются функции в VM до импорта core.config,
            // и индексы конструкторов из модуля дают "Function index N out of bounds".
            let potential_lib_path = if !module_names.is_empty() {
                debug_println!(
                    "[DEBUG run_with_vm_internal_with_args] Ищем __lib__.dc в папках по импортам: {:?}",
                    module_names
                );
                file_import::find_lib_in_package_dirs(&base_path, &module_names)
            } else {
                None
            }.or_else(|| {
                let in_script_dir = base_path.join("__lib__.dc");
                if in_script_dir.exists() {
                    debug_println!(
                        "[DEBUG run_with_vm_internal_with_args] Найден __lib__.dc в директории скрипта: {:?}",
                        in_script_dir
                    );
                    Some(in_script_dir)
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
    let mut resolver = Resolver::new_with_source_name(source_name_str.as_deref());
    resolver.resolve(&ast)?;

    // 4. Компиляция в байт-код
    let mut compiler = Compiler::new_with_source_name(source_name_str.as_deref());
    let chunk = compiler.compile(&ast)?;
    let functions = compiler.get_functions();
    // Отладка: главный chunk должен содержать "Config" в global_names (from config import Config)
    let config_in_chunk: Vec<usize> = chunk.global_names.iter().filter(|(_, n)| n.as_str() == "Config").map(|(i, _)| *i).collect();
    debug_println!("[DEBUG run_with_vm_internal_with_args] После компиляции: главный chunk global_names содержит 'Config': {} (индексы: {:?})", !config_in_chunk.is_empty(), config_in_chunk);

    // 5. Выполнение на VM
    let mut vm = Vm::new();
    // Сразу задаём base_path и project_root в VM, чтобы импорты и load_env разрешались детерминированно
    let base = explicit_base_path.clone().or_else(file_import::get_base_path);
    vm.set_base_path(base.clone());
    vm.set_project_root(base);

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
    
    // Module isolation: register __lib__.dc as a module (no merge). Main must "from __lib__ import X" to use lib exports.
    if let Some(mut lib_vm) = lib_vm {
        debug_println!("[DEBUG run_with_vm_internal_with_args] Регистрируем __lib__.dc как модуль (без merge)");
        let start_idx = vm.add_functions_only(lib_vm.get_functions().clone());
        let lib_fn_count = lib_vm.get_functions().len();
        remap_function_constants_in_chunks(vm.get_functions_mut(), start_idx, lib_fn_count);
        let mut exports = crate::vm::file_import::export_globals_from_vm(&mut lib_vm);
        remap_function_indices_in_exports(&mut exports, start_idx);
        let lib_module_value = Value::Object(Rc::new(RefCell::new(exports)));
        {
            use crate::vm::module_object::ModuleObject;
            if let Value::Object(ref namespace_rc) = lib_module_value {
                let mod_obj = ModuleObject::from_namespace("__lib__".to_string(), namespace_rc.clone());
                vm.get_modules_mut().insert("__lib__".to_string(), Rc::new(RefCell::new(mod_obj)));
            }
        }
        // Set __lib__ slot in main VM so "from __lib__ import X" and "import __lib__" resolve.
        let lib_slot_idx = vm.get_global_names().iter()
            .find(|(_, n)| n.as_str() == "__lib__")
            .map(|(i, _)| *i);
        if let Some(idx) = lib_slot_idx {
            let id = vm.with_stores_mut(|store, heap| crate::vm::store_convert::store_value(lib_module_value.clone(), store, heap));
            if idx >= vm.get_globals().len() {
                vm.get_globals_mut().resize(idx + 1, crate::vm::global_slot::default_global_slot());
            }
            vm.get_globals_mut()[idx] = crate::vm::global_slot::GlobalSlot::Heap(id);
        } else {
            let id = vm.with_stores_mut(|store, heap| crate::vm::store_convert::store_value(lib_module_value, store, heap));
            let idx = vm.get_globals().len();
            vm.get_globals_mut().push(crate::vm::global_slot::GlobalSlot::Heap(id));
            vm.get_global_names_mut().insert(idx, "__lib__".to_string());
        }
        debug_println!("[DEBUG run_with_vm_internal_with_args] __lib__ зарегистрирован как модуль");
    } else {
        debug_println!("[DEBUG run_with_vm_internal_with_args] __lib__.dc не загружен (lib_path не указан)");
    }
    
    // Добавляем слоты для глобальных переменных из главного chunk (кроме "argv" — его слот создаём ниже).
    vm.ensure_globals_from_chunk(&chunk);
    // Единственный слот "argv": сохраняем индекс и пушим в конец; перед run пишем по этому индексу.
    let script_args = args.unwrap_or_default();
    let argv: Vec<Value> = script_args.iter().map(|s| Value::String(s.clone())).collect();
    let argv_array = Value::Array(Rc::new(RefCell::new(argv)));
    let argv_id = vm.with_stores_mut(|store, heap| crate::vm::store_convert::store_value(argv_array, store, heap));
    let argv_slot_index = vm.get_globals().len();
    vm.get_globals_mut().push(crate::vm::global_slot::GlobalSlot::Heap(argv_id));
    vm.get_global_names_mut().insert(argv_slot_index, "argv".to_string());
    vm.set_argv_slot_index(Some(argv_slot_index)); // set before update_chunk_indices so argv is forced to this slot; executor will see it when re-patching after ImportFrom
    // Проверка: есть ли "Config" в global_names перед set_functions (для отладки sentinel/недетерминизма)
    let config_indices: Vec<usize> = vm.get_global_names().iter()
        .filter(|(_, n)| n.as_str() == "Config")
        .map(|(idx, _)| *idx)
        .collect();
    if let Some(&idx) = config_indices.iter().min() {
        debug_println!("[DEBUG run_with_vm_internal_with_args] После ensure и установки argv: 'Config' в global_names под индексом {} (всего слотов с именем Config: {})", idx, config_indices.len());
    } else {
        debug_println!("[DEBUG run_with_vm_internal_with_args] После второго ensure: 'Config' НЕ найден в global_names");
    }
    // Сохраняем маппинг (индекс -> имя) главного chunk ДО патча, чтобы set_functions мог корректно
    // пропатчить байткод вложенных функций (main/__main__): компилятор присвоил load_settings=78,
    // после update_chunk_indices главный chunk уже имеет 78=get_settings; без этого снимка мы бы
    // ошибочно не меняли 78->79 для вызова load_settings(env).
    let main_old_idx_to_name: std::collections::HashMap<usize, String> = chunk
        .global_names
        .iter()
        .map(|(i, n)| (*i, n.clone()))
        .collect();
    let argv_old_indices: Vec<usize> = main_old_idx_to_name.iter()
        .filter(|(_, n)| n.as_str() == "argv")
        .map(|(i, _)| *i)
        .collect();
    // Явно не патчим argv в главном chunk до update_chunk_indices: и argv, и __main__ могут иметь один и тот же
    // компиляторский индекс (76), и замена всех LoadGlobal(76)->82 подменяет загрузку __main__ на argv.
    // update_chunk_indices сам мапит по имени (76=argv->82, 75=__main__->76), поэтому дополнительный патч не делаем.
    // Обновляем индексы в главном chunk на основе реальных индексов в VM
    let mut chunk = chunk;
    vm.update_chunk_indices(&mut chunk);
    // Патчим argv во всех function chunks: argv идёт в argv_slot_index, а то что было по argv_slot_index (e.g. load_settings) — в old_idx. Байткод через временный слот, затем синхронизируем global_names.
    // Важно: заменяем LoadGlobal(old_idx) на argv_slot_index только в тех чанках, где old_idx действительно означает "argv"
    // (иначе в функции __main__ загрузка "main" по индексу 76 могла бы быть ошибочно заменена на argv и вызов main(env) падал бы с "Can only call functions, got: Array").
    const SWAP_TEMP_SLOT: usize = 0xFFFF; // temporary to swap 79 and 85 in bytecode
    let mut functions = functions;
    // Обновляем индексы во всех function chunks, чтобы "argv" и остальные глобалы резолвились в правильные слоты (в т.ч. после base_path + __lib__).
    for f in &mut functions {
        vm.update_chunk_indices(&mut f.chunk);
    }
    for f in &mut functions {
        for (old_idx, name) in &main_old_idx_to_name {
            if name == "argv" {
                let this_chunk_uses_argv_at_old = f.chunk.global_names.get(old_idx).map(|n| n.as_str()) == Some("argv");
                if !this_chunk_uses_argv_at_old {
                    continue;
                }
                // Bytecode swap: 85 -> temp, 79 -> 85, temp -> 79.
                for opcode in &mut f.chunk.code {
                    if let crate::bytecode::OpCode::LoadGlobal(idx) = opcode {
                        if *idx == argv_slot_index {
                            *opcode = crate::bytecode::OpCode::LoadGlobal(SWAP_TEMP_SLOT);
                        } else if *idx == *old_idx {
                            *opcode = crate::bytecode::OpCode::LoadGlobal(argv_slot_index);
                        }
                    }
                }
                for opcode in &mut f.chunk.code {
                    if let crate::bytecode::OpCode::LoadGlobal(idx) = opcode {
                        if *idx == SWAP_TEMP_SLOT {
                            *opcode = crate::bytecode::OpCode::LoadGlobal(*old_idx);
                        }
                    }
                }
                // global_names: 85->"argv", and whatever was at 85 moves to old_idx.
                f.chunk.global_names.remove(old_idx);
                if let Some(prev_name) = f.chunk.global_names.remove(&argv_slot_index) {
                    f.chunk.global_names.insert(*old_idx, prev_name);
                }
                f.chunk.global_names.insert(argv_slot_index, "argv".to_string());
                f.chunk.global_names.remove(&SWAP_TEMP_SLOT);
                break;
            }
        }
    }
    // Затем устанавливаем функции основного скрипта (передаём main chunk и снимок имён до патча).
    vm.set_functions(functions, Some(&mut chunk), Some(main_old_idx_to_name));
    
    let debug_sf: Vec<(usize, String)> = vm.get_global_names().iter()
        .filter(|(_, n)| n.contains("Data"))
        .filter_map(|(i, n)| vm.get_globals().get(*i).map(|_| (*i, n.clone())))
        .collect();
    for (idx, name) in debug_sf {
        let id = vm.resolve_global_to_value_id(idx);
        let v = crate::vm::store_convert::load_value(id, vm.value_store(), vm.heavy_store());
        let type_str = match &v {
            Value::Null => "Null",
            Value::Function(_) | Value::ModuleFunction { .. } => "Function",
            Value::Object(_) => "Object",
            _ => "Other",
        };
        debug_println!("[DEBUG run_with_vm_internal_with_args] После set_functions: '{}' в globals[{}] = {:?}", name, idx, type_str);
    }
    use crate::vm::globals;
    for (idx, &name) in globals::BUILTIN_GLOBAL_NAMES.iter().enumerate() {
        if vm.get_global_names().get(&idx).map(|s| s.as_str()) == Some(name) {
            let id = vm.with_stores_mut(|store, heap| crate::vm::store_convert::store_value(Value::NativeFunction(idx), store, heap));
            vm.get_globals_mut()[idx] = crate::vm::global_slot::GlobalSlot::Heap(id);
        }
    }

    // Use explicit base path when provided (e.g. from run_with_vm_and_path), else thread-local
    let base = explicit_base_path.or_else(file_import::get_base_path);
    vm.set_base_path(base.clone());
    vm.set_project_root(base);
    // Не повторно патчим argv в главном chunk: после update_chunk_indices и set_functions индексы уже верны,
    // а замена всех LoadGlobal(old_idx) на argv_slot_index подменяет загрузку __main__ (если тот оказался по тому же индексу).
    // Guarantee main chunk has argv_slot_index -> "argv" so executor's update_chunk_indices_from_names (when argv_slot=Some) forces argv to this slot and does not remap 85 -> 79.
    chunk.global_names.insert(argv_slot_index, "argv".to_string());
    // Патчим главный chunk только когда в chunk по этому индексу значится "argv", чтобы не подменять загрузку __main__.
    for op in chunk.code.iter_mut() {
        if let crate::bytecode::OpCode::LoadGlobal(idx) = op {
            let name_is_argv = chunk.global_names.get(idx).map(|n| n.as_str()) == Some("argv");
            if name_is_argv && *idx != argv_slot_index {
                *idx = argv_slot_index;
            }
        }
    }
    // Записываем argv в слот по сохранённому индексу (resize если merge добавил слоты и индекс ещё в границах).
    if argv_slot_index >= vm.get_globals().len() {
        vm.get_globals_mut()
            .resize(argv_slot_index + 1, crate::vm::global_slot::default_global_slot());
    }
    vm.get_globals_mut()[argv_slot_index] = crate::vm::global_slot::GlobalSlot::Heap(argv_id);
    let result = vm.run(&chunk, Some((argv_slot_index, &argv_old_indices, Some(argv_id))))?;

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
    let (_, vm) = run_with_vm_internal_with_args(&source, Some(Vec::new()), None, None, None)?;
    
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
    vm.set_functions(functions, None, None);
    vm.register_native_globals();
    let result = vm.run(&chunk, None)?;

    Ok(result)
}

