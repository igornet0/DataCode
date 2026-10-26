// Модуль для загрузки локальных .dc файлов как модулей
// When VM is running, RunContext holds base_path/executing_lib/dpm_package_paths; we prefer it over thread_locals.

use crate::common::{error::LangError, value::Value};
use crate::debug_println;
use crate::vm::run_context::RunContext;
use crate::vm::Vm;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

// Legacy thread-local storage (used when RunContext is not set, e.g. before run() or in tests).
thread_local! {
    static BASE_PATH: std::cell::RefCell<Option<PathBuf>> = std::cell::RefCell::new(None);
    static EXECUTING_LIB: std::cell::RefCell<bool> = std::cell::RefCell::new(false);
    static DPM_PACKAGE_PATHS: std::cell::RefCell<Vec<PathBuf>> = std::cell::RefCell::new(Vec::new());
}

static NATIVE_LIB_OVERRIDE: OnceLock<Mutex<Option<PathBuf>>> = OnceLock::new();

fn native_lib_mutex() -> &'static Mutex<Option<PathBuf>> {
    NATIVE_LIB_OVERRIDE.get_or_init(|| Mutex::new(None))
}

/// Явный путь к `lib<name>.dylib` / `.so` / `.dll` из CLI `--lib` (общий для всех потоков, в т.ч. GUI).
pub fn set_native_lib_override(path: Option<PathBuf>) {
    if let Ok(mut g) = native_lib_mutex().lock() {
        *g = path;
    }
}

pub fn get_native_lib_override() -> Option<PathBuf> {
    native_lib_mutex().lock().ok().and_then(|g| g.clone())
}

/// Сбрасывает override при выходе из `execute_file` (CLI).
pub struct NativeLibOverrideGuard;

impl Drop for NativeLibOverrideGuard {
    fn drop(&mut self) {
        set_native_lib_override(None);
    }
}

/// Установить override и вернуть guard, который очистит при drop.
pub fn push_native_lib_override(path: Option<PathBuf>) -> NativeLibOverrideGuard {
    set_native_lib_override(path);
    NativeLibOverrideGuard
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

/// Пытается найти модуль в заданном корне: предпочитается пакет <root>/<module_name>/__lib__.dc,
/// иначе файл <root>/<module_name>.dc. Так "core.config" даёт core/config/__lib__.dc, а не core/config.dc.
pub(crate) fn try_find_module_in(module_name: &str, root: &Path) -> Option<(PathBuf, PathBuf)> {
    let dir_path = root.join(module_name);
    let lib_path = dir_path.join("__lib__.dc");
    if dir_path.is_dir() && lib_path.exists() {
        return Some((dir_path, lib_path));
    }
    let file_path = root.join(format!("{}.dc", module_name));
    if file_path.exists() {
        return Some((root.to_path_buf(), file_path));
    }
    None
}

/// Разрешает промежуточный сегмент dotted-импорта (не последний).
/// Пакет с __lib__.dc → директория пакета; namespace-папка без __lib__ → та же директория;
/// файл `<segment>.dc` без директории → ошибка (нельзя `foo.bar`, если foo — файл).
pub(crate) fn try_find_path_segment(
    segment: &str,
    full_module_name: &str,
    root: &Path,
) -> Result<PathBuf, LangError> {
    let dir_path = root.join(segment);
    let lib_path = dir_path.join("__lib__.dc");
    if dir_path.is_dir() && lib_path.exists() {
        return Ok(dir_path);
    }
    if dir_path.is_dir() {
        return Ok(dir_path);
    }
    let file_path = root.join(format!("{}.dc", segment));
    if file_path.exists() {
        return Err(LangError::runtime_error(
            format!(
                "Module '{}' not found: segment '{}' is a file ('{}'), not a package or directory; cannot import '{}.…'",
                full_module_name,
                segment,
                file_path.display(),
                segment
            ),
            0,
        ));
    }
    Err(LangError::runtime_error(
        format!(
            "Module '{}' not found (package segment '{}')",
            full_module_name, segment
        ),
        0,
    ))
}

/// Compiles source to bytecode (chunk + functions). Does not run.
/// Also returns import module names from AST for dependency graph.
/// source_name: path to source file for error messages (e.g. when loading a .dc module).
pub(crate) fn compile_module(
    source: &str,
    source_name: Option<&Path>,
) -> Result<
    (
        crate::bytecode::Chunk,
        Vec<crate::bytecode::Function>,
        Vec<String>,
    ),
    LangError,
> {
    use crate::compiler::Compiler;
    use crate::lexer::Lexer;
    use crate::parser::ast::import_module_names_from_stmts;
    use crate::parser::Parser;
    use crate::semantic::resolver::Resolver;

    let source_name_str = source_name.map(|p| p.to_string_lossy().into_owned());
    let base_for_native = source_name.and_then(|p| p.parent());
    let mut lexer = Lexer::new_with_source_name(source, source_name_str.as_deref());
    let tokens = lexer.tokenize()?;
    let operator_registry = crate::preload_operator_registry_for_parse(&tokens, base_for_native)?;
    let native_call_registry =
        crate::preload_native_call_registry_for_parse(&tokens, base_for_native)?;
    let mut parser = Parser::new_with_source_name_and_registry(
        tokens,
        source_name_str.as_deref(),
        operator_registry,
    );
    let mut ast = parser.parse()?;
    crate::compiler::array_map_onehot_fusion::inject_ml_onehots_import(&mut ast);
    let import_names = import_module_names_from_stmts(&ast);
    let mut resolver = Resolver::new_with_source_name(source_name_str.as_deref());
    resolver.resolve(&ast)?;
    let mut compiler = Compiler::for_module(
        source_name_str.as_deref(),
        Some(native_call_registry),
    );
    let chunk = compiler.compile(&ast)?;
    let functions = compiler.get_functions();
    Ok((chunk, functions, import_names))
}

/// Exports VM globals as a name -> Value map (for module namespace / __lib__ registration).
pub fn export_globals_from_vm(vm: &mut Vm) -> HashMap<String, Value> {
    use crate::vm::store_convert::load_value;
    let mut exports = HashMap::new();
    let globals = vm.get_globals();
    let global_names = vm.get_global_names();
    let mut to_export: Vec<(usize, String)> = global_names
        .iter()
        .filter_map(|(index, name)| globals.get(*index).map(|_| (*index, name.clone())))
        .collect();
    to_export.sort_by(|a, b| a.1.cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
    debug_println!(
        "[DEBUG export_globals_from_vm] Экспортируем {} глобальных переменных",
        to_export.len()
    );
    // Group by name so we can prefer non-null when the same name appears at multiple indices.
    let mut by_name: std::collections::HashMap<String, Vec<(usize, Value)>> =
        std::collections::HashMap::new();
    for (index, name) in to_export {
        let value_id = vm.resolve_global_to_value_id(index);
        let value = load_value(value_id, vm.value_store(), vm.heavy_store());
        let value_type = match &value {
            Value::Object(_) => "Object",
            Value::Function(_) | Value::ModuleFunction { .. } => "Function",
            Value::Null => "Null",
            _ => "Other",
        };
        debug_println!(
            "[DEBUG export_globals_from_vm] Экспортируем: {} (index: {}, type: {})",
            name,
            index,
            value_type
        );
        by_name.entry(name).or_default().push((index, value));
    }
    let mut by_name_vec: Vec<_> = by_name.into_iter().collect();
    by_name_vec.sort_by(|a, b| a.0.cmp(&b.0));

    for (name, mut entries) in by_name_vec {
        // Prefer non-null when same name at multiple indices (e.g. constructor at 92, Null at 84).
        // When both non-null and both Function, prefer larger function index (later definition).
        // Sort entries so choice is deterministic (no HashMap iteration order dependency).
        entries.sort_by(|a, b| {
            let a_ok = !matches!(a.1, Value::Null);
            let b_ok = !matches!(b.1, Value::Null);
            match (a_ok, b_ok) {
                (true, false) => std::cmp::Ordering::Greater,
                (false, true) => std::cmp::Ordering::Less,
                (true, true) => match (&a.1, &b.1) {
                    (Value::Function(ia), Value::Function(ib)) => ia.cmp(ib),
                    (
                        Value::ModuleFunction {
                            module_uid: ma,
                            local_index: la,
                        },
                        Value::ModuleFunction {
                            module_uid: mb,
                            local_index: lb,
                        },
                    ) => (ma, la).cmp(&(mb, lb)),
                    _ => a.0.cmp(&b.0),
                },
                _ => a.0.cmp(&b.0),
            }
        });
        let best_entry = entries.last().cloned();
        if let Some((_, v)) = best_entry {
            exports.insert(name.clone(), v.clone());
        }
    }
    debug_println!(
        "[DEBUG export_globals_from_vm] Всего экспортировано: {} переменных",
        exports.len()
    );
    exports
}

/// Вспомогательная функция для получения базового пути из пути к файлу
pub fn get_base_path_from_file(file_path: &Path) -> Option<PathBuf> {
    file_path.parent().map(|p| p.to_path_buf())
}

/// Максимальная глубина подъёма при поиске __lib__.dc (защита от долгого обхода на медленных ФС).
const FIND_NEAREST_LIB_MAX_DEPTH: u32 = 25;

/// Ищет ближайший `__lib__.dc`, начиная с указанной директории и поднимаясь
/// вверх по дереву каталогов. Ограничено [`FIND_NEAREST_LIB_MAX_DEPTH`] уровнями.
///
/// Используется для автоматического поиска библиотечного файла для скрипта.
pub fn find_nearest_lib(start_dir: &Path) -> Option<PathBuf> {
    let mut current = Some(start_dir.to_path_buf());
    let mut depth = 0u32;

    while let Some(dir) = current {
        if depth > FIND_NEAREST_LIB_MAX_DEPTH {
            return None;
        }
        let candidate = dir.join("__lib__.dc");
        if candidate.exists() {
            return Some(candidate);
        }
        current = dir.parent().map(|p| p.to_path_buf());
        depth += 1;
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
