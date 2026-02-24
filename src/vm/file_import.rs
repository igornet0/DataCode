// Модуль для загрузки локальных .dc файлов как модулей
// When VM is running, RunContext holds base_path/executing_lib/dpm_package_paths; we prefer it over thread_locals.

use crate::debug_println;
use crate::vm::run_context::RunContext;
use std::path::{Path, PathBuf};
use std::collections::HashMap;
use crate::common::{error::LangError, value::Value};
use crate::vm::Vm;

// Legacy thread-local storage (used when RunContext is not set, e.g. before run() or in tests).
thread_local! {
    static BASE_PATH: std::cell::RefCell<Option<PathBuf>> = std::cell::RefCell::new(None);
    static EXECUTING_LIB: std::cell::RefCell<bool> = std::cell::RefCell::new(false);
    static DPM_PACKAGE_PATHS: std::cell::RefCell<Vec<PathBuf>> = std::cell::RefCell::new(Vec::new());
}

/// Устанавливает базовый путь для текущего потока (updates RunContext when set, else legacy thread_local).
pub fn set_base_path(path: Option<PathBuf>) {
    RunContext::with_current_opt(|r| r.base_path = path.clone());
    BASE_PATH.with(|p| *p.borrow_mut() = path);
}

/// Получает базовый путь (from RunContext when set, else legacy thread_local).
pub fn get_base_path() -> Option<PathBuf> {
    RunContext::get_base_path().or_else(|| BASE_PATH.with(|p| p.borrow().clone()))
}

/// Устанавливает флаг выполнения __lib__.dc
pub fn set_executing_lib(executing: bool) {
    RunContext::with_current_opt(|r| r.executing_lib = executing);
    EXECUTING_LIB.with(|f| *f.borrow_mut() = executing);
}

/// Проверяет, выполняем ли мы __lib__.dc
pub fn is_executing_lib() -> bool {
    if RunContext::is_set() {
        RunContext::get_executing_lib()
    } else {
        EXECUTING_LIB.with(|f| *f.borrow())
    }
}

/// Устанавливает дополнительные пути поиска модулей (DPM packages).
pub fn set_dpm_package_paths(paths: Vec<PathBuf>) {
    RunContext::with_current_opt(|r| r.dpm_package_paths = paths.clone());
    DPM_PACKAGE_PATHS.with(|p| *p.borrow_mut() = paths);
}

/// Получает дополнительные пути поиска модулей (from RunContext when set, else legacy thread_local).
pub fn get_dpm_package_paths() -> Vec<PathBuf> {
    if RunContext::is_set() {
        RunContext::get_dpm_package_paths()
    } else {
        DPM_PACKAGE_PATHS.with(|p| p.borrow().clone())
    }
}

/// Пытается найти модуль в заданном корне: <root>/<module_name>.dc или <root>/<module_name>/__lib__.dc.
fn try_find_module_in(module_name: &str, root: &Path) -> Option<(PathBuf, PathBuf)> {
    let file_path = root.join(format!("{}.dc", module_name));
    if file_path.exists() {
        return Some((root.to_path_buf(), file_path));
    }
    let dir_path = root.join(module_name);
    let lib_path = dir_path.join("__lib__.dc");
    if dir_path.is_dir() && lib_path.exists() {
        return Some((dir_path, lib_path));
    }
    None
}

/// Загружает локальный .dc файл или пакет как модуль
///
/// Поддерживаются два варианта:
/// - Обычный файл: `<base_path>/<module_name>.dc`
/// - Пакет‑директория: `<base_path>/<module_name>/__lib__.dc`
///
/// В обоих случаях перед выполнением модуля временно устанавливается
/// BASE_PATH на директорию модуля, чтобы вложенные `import`/`from`
/// разрешались относительно самого модуля (для подпакетов).
///
/// # Аргументы
/// * `module_name` - имя модуля (без расширения .dc)
/// * `base_path` - базовый путь, относительно которого ищется модуль
///
/// # Возвращает
/// Объект Value::Object с экспортированными глобальными переменными и функциями
pub fn load_local_module(
    module_name: &str,
    base_path: &Path,
) -> Result<Value, LangError> {
    // 1. Ищем модуль: сначала base_path, затем DPM package paths
    let mut search_paths = vec![base_path.to_path_buf()];
    search_paths.extend(get_dpm_package_paths());
    let (module_dir, file_path) = search_paths
        .iter()
        .find_map(|root| try_find_module_in(module_name, root))
        .ok_or_else(|| {
            LangError::runtime_error(
                format!(
                    "Module '{}' not found. Searched in base path and DPM packages.",
                    module_name,
                ),
                0,
            )
        })?;

    // 2. Загрузить содержимое файла модуля
    let source = std::fs::read_to_string(&file_path).map_err(|e| {
        LangError::runtime_error(
            format!("Failed to read module file '{}': {}", file_path.display(), e),
            0,
        )
    })?;

    // 3. Временно устанавливаем BASE_PATH на директорию модуля,
    // чтобы вложенные импорты разрешались относительно нее
    let old_base_path = get_base_path();
    set_base_path(Some(module_dir.clone()));

    // 4. Скомпилировать и выполнить модуль (передаём module_dir в VM для вложенных импортов)
    let (_, mut vm) = compile_and_run_module(&source, Some(module_dir))?;

    // Восстанавливаем предыдущий BASE_PATH
    set_base_path(old_base_path);

    // 5. Экспортировать глобальные переменные в объект модуля
    let module_object = export_globals_from_vm(&mut vm);

    Ok(Value::Object(std::rc::Rc::new(std::cell::RefCell::new(module_object))))
}

/// Загружает подмодуль по составному имени (например `config.xyz`).
/// Разрешает каждый сегмент относительно предыдущего (папка пакета).
fn load_local_module_dotted_with_vm(
    module_name: &str,
    base_path: &Path,
) -> Result<(Value, Vm), LangError> {
    let parts: Vec<&str> = module_name.split('.').collect();
    if parts.is_empty() {
        return Err(LangError::runtime_error("Empty module name".to_string(), 0));
    }
    let mut search_paths = vec![base_path.to_path_buf()];
    search_paths.extend(get_dpm_package_paths());
    let mut current_base = base_path.to_path_buf();
    for i in 0..parts.len() - 1 {
        let (module_dir, _) = search_paths
            .iter()
            .find_map(|root| try_find_module_in(parts[i], root))
            .ok_or_else(|| {
                LangError::runtime_error(
                    format!(
                        "Module '{}' not found (package segment '{}')",
                        module_name, parts[i]
                    ),
                    0,
                )
            })?;
        current_base = module_dir;
        search_paths = vec![current_base.clone()];
    }
    load_local_module_with_vm_inner(parts[parts.len() - 1], &current_base)
}

/// Загружает локальный .dc файл или пакет как модуль и возвращает VM с функциями
///
/// Поддерживает составные имена: `config.xyz` загружает модуль `xyz` из пакета `config`.
///
/// # Аргументы
/// * `module_name` - имя модуля (без расширения .dc), может содержать точки для подмодулей
/// * `base_path` - базовый путь, относительно которого ищется файл
///
/// # Возвращает
/// Кортеж (объект модуля, VM с функциями)
pub fn load_local_module_with_vm(
    module_name: &str,
    base_path: &Path,
) -> Result<(Value, Vm), LangError> {
    if module_name.contains('.') {
        return load_local_module_dotted_with_vm(module_name, base_path);
    }
    load_local_module_with_vm_inner(module_name, base_path)
}

/// Внутренняя загрузка одного сегмента модуля (имя без точек).
fn load_local_module_with_vm_inner(
    module_name: &str,
    base_path: &Path,
) -> Result<(Value, Vm), LangError> {
    // 1. Ищем модуль: сначала base_path, затем DPM package paths
    let mut search_paths = vec![base_path.to_path_buf()];
    search_paths.extend(get_dpm_package_paths());
    let (module_dir, file_path) = search_paths
        .iter()
        .find_map(|root| try_find_module_in(module_name, root))
        .ok_or_else(|| {
            LangError::runtime_error(
                format!(
                    "Module '{}' not found. Searched in base path and DPM packages.",
                    module_name,
                ),
                0,
            )
        })?;

    // 2. Загрузить содержимое файла модуля
    let source = std::fs::read_to_string(&file_path).map_err(|e| {
        LangError::runtime_error(
            format!("Failed to read module file '{}': {}", file_path.display(), e),
            0,
        )
    })?;

    // 3. Временно устанавливаем BASE_PATH на директорию модуля
    let old_base_path = get_base_path();
    set_base_path(Some(module_dir.clone()));

    // 4. Скомпилировать и выполнить модуль (передаём module_dir в VM, чтобы run() не затирал thread-local в None)
    let (_, mut vm) = compile_and_run_module(&source, Some(module_dir))?;

    // Восстанавливаем предыдущий BASE_PATH
    set_base_path(old_base_path);

    // 5. Экспортировать глобальные переменные в объект модуля
    let module_object = export_globals_from_vm(&mut vm);

    Ok((
        Value::Object(std::rc::Rc::new(std::cell::RefCell::new(module_object))),
        vm,
    ))
}

/// Компилирует и выполняет код модуля.
/// module_base_path: директория модуля (например config/), чтобы VM разрешал вложенные импорты и не затирал thread-local в None в run().
fn compile_and_run_module(source: &str, module_base_path: Option<PathBuf>) -> Result<(Value, Vm), LangError> {
    use crate::lexer::Lexer;
    use crate::parser::Parser;
    use crate::semantic::resolver::Resolver;
    use crate::compiler::Compiler;
    use crate::vm::Vm;

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

    // 5. Выполнение на VM
    let mut vm = Vm::new();
    // Базовый путь модуля — чтобы при run() thread-local не затирался в None и вложенные импорты (from config import Config) работали
    vm.set_base_path(module_base_path.or_else(get_base_path));
    // Сначала регистрируем нативные глобалы и встроенные модули, чтобы set_functions
    // не сопоставлял имена модуля (например "Settings") с индексами 0..74 (print и т.д.)
    let max_global_index = chunk.global_names.keys().max().copied().unwrap_or(0);
    let needed_size = (max_global_index + 1).max(74);
    if vm.get_globals().len() < needed_size {
        vm.get_globals_mut().resize(needed_size, crate::vm::global_slot::default_global_slot());
    }
    vm.register_native_globals();
    vm.register_all_builtin_modules().map_err(|e| {
        LangError::runtime_error(format!("Failed to register built-in modules: {}", e), 0)
    })?;
    // Добавляем слоты по индексам из chunk (Config, DatabaseConfig и т.д.), чтобы sentinel резолвился и main chunk хранил по тем же индексам.
    vm.ensure_globals_from_chunk_preserve_indices(&chunk);
    for f in &functions {
        vm.ensure_globals_from_chunk(&f.chunk);
    }
    vm.set_functions(functions, None);
    let result = vm.run(&chunk)?;

    Ok((result, vm))
}

/// Извлекает все глобальные переменные из VM после выполнения модуля (globals are GlobalSlot)
fn export_globals_from_vm(vm: &mut Vm) -> HashMap<String, Value> {
    use crate::vm::store_convert::load_value;
    let mut exports = HashMap::new();
    let globals = vm.get_globals();
    let global_names = vm.get_global_names();
    let to_export: Vec<(usize, String)> = global_names
        .iter()
        .filter_map(|(index, name)| globals.get(*index).map(|_| (*index, name.clone())))
        .collect();
    debug_println!("[DEBUG export_globals_from_vm] Экспортируем {} глобальных переменных", to_export.len());
    for (index, name) in to_export {
        let value_id = vm.resolve_global_to_value_id(index);
        let value = load_value(value_id, vm.value_store(), vm.heavy_store());
            let value_type = match &value {
                Value::Object(_) => "Object",
                Value::Function(_) => "Function",
                Value::Null => "Null",
                _ => "Other",
            };
            debug_println!("[DEBUG export_globals_from_vm] Экспортируем: {} (index: {}, type: {})", name, index, value_type);
            exports.insert(name, value);
    }
    debug_println!("[DEBUG export_globals_from_vm] Всего экспортировано: {} переменных", exports.len());
    exports
}

/// Вспомогательная функция для получения базового пути из пути к файлу
pub fn get_base_path_from_file(file_path: &Path) -> Option<PathBuf> {
    file_path.parent().map(|p| p.to_path_buf())
}

/// Ищет ближайший `__lib__.dc`, начиная с указанной директории и поднимаясь
/// вверх по дереву каталогов до корня файловой системы.
///
/// Используется для автоматического поиска библиотечного файла для скрипта.
pub fn find_nearest_lib(start_dir: &Path) -> Option<PathBuf> {
    let mut current = Some(start_dir.to_path_buf());

    while let Some(dir) = current {
        let candidate = dir.join("__lib__.dc");
        if candidate.exists() {
            return Some(candidate);
        }
        current = dir.parent().map(|p| p.to_path_buf());
    }

    None
}

/// Если в директории (и выше) нет __lib__.dc, ищет папки-пакеты по именам модулей:
/// для каждого `module_name` проверяет `base_path/<module_name>/__lib__.dc`.
/// Возвращает первый найденный путь (для предзагрузки lib при старте).
pub fn find_lib_in_package_dirs(base_path: &Path, module_names: &[String]) -> Option<PathBuf> {
    for name in module_names {
        let candidate = base_path.join(name).join("__lib__.dc");
        if candidate.exists() {
            return Some(candidate);
        }
    }
    None
}
