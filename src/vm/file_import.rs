// Модуль для загрузки локальных .dc файлов как модулей
// When VM is running, RunContext holds base_path/executing_lib/dpm_package_paths; we prefer it over thread_locals.

use crate::debug_println;
use crate::vm::run_context::RunContext;
use crate::vm::module_cache::{self, CachedModule};
use std::path::{Path, PathBuf};
use std::collections::HashMap;
use std::sync::Arc;
use crate::common::{error::LangError, value::Value};
use crate::vm::Vm;

/// Restores RunContext on drop so caller's context (e.g. argv_value_id) is restored after loading a module.
struct RestoreRunContextGuard(Option<RunContext>);
impl RestoreRunContextGuard {
    /// Restore now and prevent restore on drop.
    fn restore_now(&mut self) {
        if let Some(ctx) = self.0.take() {
            RunContext::set_current(ctx);
        }
    }
}
impl Drop for RestoreRunContextGuard {
    fn drop(&mut self) {
        self.restore_now();
    }
}

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

/// Пытается найти модуль в заданном корне: предпочитается пакет <root>/<module_name>/__lib__.dc,
/// иначе файл <root>/<module_name>.dc. Так "core.config" даёт core/config/__lib__.dc, а не core/config.dc.
fn try_find_module_in(module_name: &str, root: &Path) -> Option<(PathBuf, PathBuf)> {
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

/// Загружает подмодуль по составному имени (например `core.config`).
/// Absolute import: first segment is resolved from project_root, not current base_path.
fn load_local_module_dotted_with_vm(
    module_name: &str,
    base_path: &Path,
    vm: &mut Vm,
) -> Result<LoadModuleResult, LangError> {
    let parts: Vec<&str> = module_name.split('.').collect();
    if parts.is_empty() {
        return Err(LangError::runtime_error("Empty module name".to_string(), 0));
    }
    // Absolute import: first segment resolves from project_root (if set), else base_path.
    let project_root_opt = vm.get_project_root();
    let root_for_first = project_root_opt
        .as_ref()
        .map(|p| p.as_path())
        .unwrap_or(base_path);
    let mut search_paths = vec![root_for_first.to_path_buf()];
    search_paths.extend(get_dpm_package_paths());
    let mut current_base = base_path.to_path_buf();
    for i in 0..parts.len() - 1 {
        let roots_for_segment: Vec<PathBuf> = if i == 0 {
            vec![root_for_first.to_path_buf()]
        } else {
            vec![current_base.clone()]
        };
        let mut full_search = roots_for_segment;
        full_search.extend(get_dpm_package_paths());
        let (module_dir, _) = full_search
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
    }
    let last_part = parts[parts.len() - 1];
    let result = load_local_module_with_vm_inner(last_part, &current_base, vm)?;
    // Register under full dotted name too so LoadGlobal in module context finds it (frame.module_name is "core.config").
    if last_part != module_name {
        let mut modules = vm.get_modules_mut();
        if let Some(rc) = modules.get(last_part).cloned() {
            modules.insert(module_name.to_string(), rc);
        }
    }
    Ok(result)
}

/// Result of loading a .dc module: (module namespace object, VM if first run else None).
/// When `Option<Vm>` is `None`, the module was already executed once this run; caller must not merge, only use the namespace.
pub type LoadModuleResult = (Value, Option<Vm>);

/// Загружает локальный .dc файл или пакет как модуль.
/// Uses vm's module_cache (compile) and executed_modules (run once per path).
///
/// Поддерживает составные имена: `config.xyz` загружает модуль `xyz` из пакета `config`.
///
/// # Возвращает
/// `(module_object, Some(module_vm))` on first load (caller merges); `(module_object, None)` when module was already executed this run.
pub fn load_local_module_with_vm(
    module_name: &str,
    base_path: &Path,
    vm: &mut Vm,
) -> Result<LoadModuleResult, LangError> {
    if module_name.contains('.') {
        return load_local_module_dotted_with_vm(module_name, base_path, vm);
    }
    load_local_module_with_vm_inner(module_name, base_path, vm)
}

/// Внутренняя загрузка одного сегмента модуля (имя без точек).
/// Uses module_cache (compile) and executed_modules (run once per canonical path).
fn load_local_module_with_vm_inner(
    module_name: &str,
    base_path: &Path,
    vm: &mut Vm,
) -> Result<LoadModuleResult, LangError> {
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

    let cache_key = module_cache::canonical_module_cache_key(&file_path);
    let base_path_before = get_base_path();

    // Save caller's RunContext before changing base_path so it can be restored after run_compiled_module
    // (module's run() overwrites thread-local RunContext; without restore, LoadGlobal(argv) would miss argv_value_id).
    // Also save RESTORED_SCRIPT_ARGV_AFTER_IMPORT: nested module run() may overwrite it with None; we must restore.
    let saved_restored_argv = RunContext::get_restored_script_argv_after_import();
    let mut _run_ctx_guard = RestoreRunContextGuard(RunContext::take_current());

    // Already executed this run: return saved namespace. The object was remapped when first loaded (in executor),
    // so do NOT add_functions_only or remap again — that would duplicate functions and double-remap indices.
    // If we have a cache hit but no stored functions (e.g. module was first loaded in another VM context), re-load below.
    {
        let module_object_opt = {
            let m = vm.get_executed_modules_mut();
            m.get(&cache_key).cloned()
        };
        let has_stored_fns = {
            let m = vm.get_executed_module_functions_mut();
            m.contains_key(&cache_key)
        };
        if let Some(module_object) = module_object_opt {
            if has_stored_fns {
                return Ok((module_object, None));
            }
            // Cache hit but no stored functions: object has indices from another VM. Fall through to re-load.
        }
    }

    let source = std::fs::read_to_string(&file_path).map_err(|e| {
        LangError::runtime_error(
            format!("Failed to read module file '{}': {}", file_path.display(), e),
            0,
        )
    })?;

    let source_mtime = std::fs::metadata(&file_path)
        .ok()
        .and_then(|m| m.modified().ok())
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_nanos() as u64);

    set_base_path(Some(module_dir.clone()));

    let dcb_path = crate::vm::dcb::dcb_cache_path(&cache_key);

    let mut module_vm = {
        let mut cache = vm.get_module_cache_mut();
        if let Some(cached) = cache.get(&cache_key) {
            let cached = cached.clone();
            let project_root = vm.get_project_root();
            drop(cache);
            run_compiled_module(&cached.chunk, &cached.functions, Some(module_dir.clone()), project_root)?
        } else if let Some(cached) = crate::vm::dcb::load_dcb_if_fresh(&dcb_path, &source, source_mtime) {
            cache.insert(cache_key.clone(), cached.clone());
            let project_root = vm.get_project_root();
            drop(cache);
            run_compiled_module(&cached.chunk, &cached.functions, Some(module_dir.clone()), project_root)?
        } else {
            let (chunk, functions, import_names) = compile_module(&source, Some(&file_path))?;
            let mut dep_paths = Vec::new();
            let mut search_paths = vec![module_dir.clone()];
            search_paths.extend(get_dpm_package_paths());
            for name in &import_names {
                if let Some((_, dep_file_path)) = search_paths.iter().find_map(|root| try_find_module_in(name, root)) {
                    dep_paths.push(module_cache::canonical_module_cache_key(&dep_file_path));
                }
            }
            drop(cache);
            vm.get_module_deps_mut().insert(cache_key.clone(), dep_paths);
            let mut cache = vm.get_module_cache_mut();
            let functions_arc = Arc::new(functions.clone());
            let compiled = CachedModule {
                chunk: chunk.clone(),
                functions: functions_arc.clone(),
            };
            if crate::vm::dcb::save_dcb(&dcb_path, &compiled, &source, source_mtime).is_ok() {
                // .dcb written; on next run we may load from disk
            }
            cache.insert(cache_key.clone(), compiled);
            let project_root = vm.get_project_root();
            drop(cache);
            run_compiled_module(&chunk, &functions, Some(module_dir.clone()), project_root)?
        }
    };
    // Restore caller's RunContext immediately so base_path and argv_value_id are correct before any further use.
    _run_ctx_guard.restore_now();
    RunContext::set_restored_script_argv_after_import(saved_restored_argv);
    set_base_path(base_path_before);
    let exports = export_globals_from_vm(&mut module_vm);
    let module_object = Value::Object(std::rc::Rc::new(std::cell::RefCell::new(exports)));

    // Do NOT remap here: executor will extend functions, register module in module_registry,
    // then convert Value::Function(local_index) -> Value::ModuleFunction { module_id, local_index } in this namespace.
    // That keeps indices stable across cache hits and different VM states.

    vm.get_executed_module_functions_mut().insert(cache_key.clone(), module_vm.get_functions().clone());
    vm.get_executed_modules_mut().insert(cache_key, module_object.clone());

    // Register in vm.modules for module isolation (name -> ModuleObject with shared namespace).
    if let Value::Object(ref namespace_rc) = module_object {
        use crate::vm::module_object::ModuleObject;
        let mod_obj = ModuleObject::from_namespace(module_name.to_string(), namespace_rc.clone());
        vm.get_modules_mut().insert(module_name.to_string(), std::rc::Rc::new(std::cell::RefCell::new(mod_obj)));
    }

    Ok((module_object, Some(module_vm)))
}

/// Compiles source to bytecode (chunk + functions). Does not run.
/// Also returns import module names from AST for dependency graph.
/// source_name: path to source file for error messages (e.g. when loading a .dc module).
fn compile_module(source: &str, source_name: Option<&Path>) -> Result<(crate::bytecode::Chunk, Vec<crate::bytecode::Function>, Vec<String>), LangError> {
    use crate::lexer::Lexer;
    use crate::parser::Parser;
    use crate::parser::ast::import_module_names_from_stmts;
    use crate::semantic::resolver::Resolver;
    use crate::compiler::Compiler;

    let source_name_str = source_name.map(|p| p.to_string_lossy().into_owned());
    let mut lexer = Lexer::new_with_source_name(source, source_name_str.as_deref());
    let tokens = lexer.tokenize()?;
    let mut parser = Parser::new_with_source_name(tokens, source_name_str.as_deref());
    let ast = parser.parse()?;
    let import_names = import_module_names_from_stmts(&ast);
    let mut resolver = Resolver::new_with_source_name(source_name_str.as_deref());
    resolver.resolve(&ast)?;
    let mut compiler = Compiler::new_with_source_name(source_name_str.as_deref());
    let chunk = compiler.compile(&ast)?;
    let functions = compiler.get_functions();
    Ok((chunk, functions, import_names))
}

/// Runs compiled module (chunk + functions) in a new VM and returns that VM. Caller exports globals.
/// project_root: inherited from parent VM for absolute imports; propagated to nested modules.
fn run_compiled_module(
    chunk: &crate::bytecode::Chunk,
    functions: &[crate::bytecode::Function],
    module_base_path: Option<PathBuf>,
    project_root: Option<PathBuf>,
) -> Result<Vm, LangError> {
    let mut vm = Vm::new();
    vm.set_base_path(module_base_path.or_else(get_base_path));
    vm.set_project_root(project_root);
    let max_global_index = chunk.global_names.keys().max().copied().unwrap_or(0);
    let needed_size = (max_global_index + 1).max(74);
    if vm.get_globals().len() < needed_size {
        vm.get_globals_mut().resize(needed_size, crate::vm::global_slot::default_global_slot());
    }
    vm.register_native_globals();
    vm.register_all_builtin_modules().map_err(|e| {
        LangError::runtime_error(format!("Failed to register built-in modules: {}", e), 0)
    })?;
    vm.ensure_globals_from_chunk_preserve_indices(chunk);
    vm.ensure_builtin_globals_high_indices();
    for f in functions {
        vm.ensure_globals_from_chunk(&f.chunk);
    }
    vm.ensure_exception_constructors();
    vm.set_functions(functions.to_vec(), None, None);
    vm.run(chunk, None)?;
    Ok(vm)
}

/// Compiles and runs module (used when no cache is available, e.g. load_local_module without vm).
fn compile_and_run_module(source: &str, module_base_path: Option<PathBuf>) -> Result<(Value, Vm), LangError> {
    let (chunk, functions, _import_names) = compile_module(source, None)?;
    let mut vm = run_compiled_module(&chunk, &functions, module_base_path, None)?;
    let module_object = export_globals_from_vm(&mut vm);
    Ok((
        Value::Object(std::rc::Rc::new(std::cell::RefCell::new(module_object))),
        vm,
    ))
}

/// Извлекает все глобальные переменные из VM после выполнения модуля (globals are GlobalSlot)
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
    debug_println!("[DEBUG export_globals_from_vm] Экспортируем {} глобальных переменных", to_export.len());
    // Group by name so we can prefer non-null when the same name appears at multiple indices.
    let mut by_name: std::collections::HashMap<String, Vec<(usize, Value)>> = std::collections::HashMap::new();
    for (index, name) in to_export {
        let value_id = vm.resolve_global_to_value_id(index);
        let value = load_value(value_id, vm.value_store(), vm.heavy_store());
        let value_type = match &value {
            Value::Object(_) => "Object",
            Value::Function(_) | Value::ModuleFunction { .. } => "Function",
            Value::Null => "Null",
            _ => "Other",
        };
        debug_println!("[DEBUG export_globals_from_vm] Экспортируем: {} (index: {}, type: {})", name, index, value_type);
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
                    (Value::ModuleFunction { module_id: ma, local_index: la }, Value::ModuleFunction { module_id: mb, local_index: lb }) => (ma, la).cmp(&(mb, lb)),
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
    debug_println!("[DEBUG export_globals_from_vm] Всего экспортировано: {} переменных", exports.len());
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
