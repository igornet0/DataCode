// Виртуальная машина

use crate::abi::NativeAbiFn;
use crate::debug_println;
use crate::bytecode::Chunk;
use crate::common::{error::LangError, table::Table, value::Value, value_store::{ValueStore, ValueCell, ValueId, NULL_VALUE_ID}, TaggedValue};
use crate::vm::store_convert::tagged_to_value_id;
use crate::vm::frame::CallFrame;
use crate::vm::natives;
use crate::vm::store_convert::{load_value, store_value_arena};
use crate::vm::heavy_store::HeavyStore;
use crate::vm::types::{ExplicitRelation, ExplicitPrimaryKey, VMStatus};
use crate::vm::exceptions::ExceptionHandler;
use crate::vm::global_slot::{self, GlobalSlot};
use crate::vm::globals;
use crate::vm::calls;
use crate::vm::executor;
use crate::vm::module_cache::CachedModule;
use crate::vm::module_object::ModuleObject;
use libloading::Library;
use std::cell::RefCell;
use std::collections::HashMap;
use std::path::PathBuf;
use std::rc::Rc;

pub type NativeFn = fn(&[Value]) -> Value;

/// Restores VM's ML context from thread-local on drop (any exit path from run()).
/// Uses raw pointer so run() can still call self.step() while guard is alive.
struct MlContextGuard(*mut Option<crate::ml::MlContext>);
impl Drop for MlContextGuard {
    fn drop(&mut self) {
        unsafe {
            *self.0 = crate::ml::MlContext::take_current();
        }
    }
}

/// Restores VM's Plot context from thread-local on drop (any exit path from run()).
struct PlotContextGuard(*mut Option<crate::plot::PlotContext>);
impl Drop for PlotContextGuard {
    fn drop(&mut self) {
        unsafe {
            *self.0 = crate::plot::PlotContext::take_current();
        }
    }
}

/// Restores VM base_path from RunContext on drop (any exit path from run()).
struct RunContextGuard(*mut Option<PathBuf>);
impl Drop for RunContextGuard {
    fn drop(&mut self) {
        if let Some(ctx) = crate::vm::run_context::RunContext::take_current() {
            unsafe {
                *self.0 = ctx.base_path;
            }
        }
    }
}

// Thread-local storage для хранения контекста VM во время вызова нативных функций
// Это позволяет нативным функциям вызывать пользовательские функции
thread_local! {
    pub(crate) static VM_CALL_CONTEXT: RefCell<Option<*mut Vm>> = RefCell::new(None);
}

/// Number of builtin global slots (0..BUILTIN_END). Indices >= this are module globals.
const BUILTIN_END: usize = 75;

/// Info for a merged module: used to resolve Value::ModuleFunction { module_id, local_index } at Call.
#[derive(Clone, Debug)]
pub struct ModuleInfo {
    pub name: String,
    pub function_offset: usize,
    pub function_count: usize,
}

pub struct Vm {
    /// Stack of TaggedValues (immediates + heap refs; no store lookup for numbers in hot path)
    stack: Vec<TaggedValue>,
    frames: Vec<CallFrame>,
    /// Builtin globals (indices 0..BUILTIN_END). Shared by all modules.
    builtins: Vec<GlobalSlot>,
    /// Module globals (indices >= BUILTIN_END). Used when no per-module isolation (legacy) or for __main__ before we switch fully.
    globals: Vec<GlobalSlot>,
    functions: Vec<crate::bytecode::Function>,
    natives: Vec<NativeFn>,
    exception_handlers: Vec<ExceptionHandler>,
    error_type_table: Vec<String>,
    global_names: std::collections::BTreeMap<usize, String>,
    explicit_global_names: std::collections::BTreeMap<usize, String>,
    explicit_relations: Vec<ExplicitRelation>,
    explicit_primary_keys: Vec<ExplicitPrimaryKey>,
    loaded_modules: std::collections::HashSet<String>,
    abi_natives: Vec<NativeAbiFn>,
    loaded_native_libraries: Vec<Library>,
    base_path: Option<PathBuf>,
    ml_context: Option<crate::ml::MlContext>,
    plot_context: Option<crate::plot::PlotContext>,
    value_store: ValueStore,
    /// Heavy values (Table, Tensor, etc.) indexed by ValueCell::Heavy(usize)
    heavy_store: HeavyStore,
    /// Reusable buffer for native call arguments (avoids allocating Vec on every CallNative).
    native_args_buffer: Vec<Value>,
    /// Reusable buffer for native arg ValueIds (avoids Vec::with_capacity on every CallNative).
    reusable_native_arg_ids: Vec<ValueId>,
    /// Reusable buffer for db engine method popped values (avoids Vec::new on every db native call).
    reusable_all_popped: Vec<Value>,
    /// Pending relations from relate() native (VM-owned; natives push here when VM_CALL_CONTEXT is set).
    pub(crate) pending_relations: Vec<(Rc<RefCell<Table>>, String, Rc<RefCell<Table>>, String)>,
    /// Pending primary keys from primary_key() native (VM-owned).
    pub(crate) pending_primary_keys: Vec<(Rc<RefCell<Table>>, String)>,
    /// Runtime module cache: canonical path -> compiled (chunk + functions).
    module_cache: RefCell<HashMap<PathBuf, CachedModule>>,
    /// Modules already executed once this run: canonical path -> saved namespace (module object). Re-import returns this without running again.
    executed_modules: RefCell<HashMap<PathBuf, Value>>,
    /// Functions from each executed module (canonical path -> list). On cache hit we add these to the caller and remap the object so indices stay valid.
    executed_module_functions: RefCell<HashMap<PathBuf, Vec<crate::bytecode::Function>>>,
    /// Dependency graph: canonical path -> list of canonical paths of imported modules (for topological order / invalidation).
    module_deps: RefCell<HashMap<PathBuf, Vec<PathBuf>>>,
    /// Cache of loaded modules by canonical name (e.g. "core.config") or path. Each module has its own namespace.
    modules: RefCell<HashMap<String, Rc<RefCell<ModuleObject>>>>,
    /// When set (e.g. from run_with_vm_internal_with_args), update_chunk_indices_from_names will always map "argv" to this slot,
    /// so ImportFrom re-patch does not remap LoadGlobal(argv) to load_settings (slot 79) after merge.
    argv_slot_index: Option<usize>,
    /// Registry of merged modules: module_id = index. Resolves ModuleFunction { module_id, local_index } -> functions[offset + local_index].
    module_registry: RefCell<Vec<ModuleInfo>>,
}

/// Preallocated capacities for hot-path Vecs to reduce resize in loop-heavy runs.
const DEFAULT_STACK_CAPACITY: usize = 4096;
const DEFAULT_FRAMES_CAPACITY: usize = 64;
const DEFAULT_GLOBALS_CAPACITY: usize = 256;
const DEFAULT_NATIVE_BUF_CAPACITY: usize = 32;

// Runtime state is set via thread_locals (VM_CALL_CONTEXT, relations, etc.) and RunContext (base_path, file_import) at run().

/// Run a closure with the current VM's value_store and heavy_store. Call only from native code (VM_CALL_CONTEXT must be set).
pub fn with_current_stores<R, F>(f: F) -> R
where
    F: FnOnce(&ValueStore, &HeavyStore) -> R,
{
    VM_CALL_CONTEXT.with(|ctx| {
        let ptr = (*ctx.borrow()).expect("with_current_stores: VM context not set");
        unsafe { (*ptr).with_stores(f) }
    })
}

impl Vm {
    /// Run a closure with references to value_store and heavy_store. Used by with_current_stores.
    pub(crate) fn with_stores<R, F>(&self, f: F) -> R
    where
        F: FnOnce(&ValueStore, &HeavyStore) -> R,
    {
        f(&self.value_store, &self.heavy_store)
    }

    pub fn new() -> Self {
        let mut vm = Self {
            stack: Vec::with_capacity(DEFAULT_STACK_CAPACITY),
            frames: Vec::with_capacity(DEFAULT_FRAMES_CAPACITY),
            builtins: Vec::with_capacity(BUILTIN_END),
            globals: Vec::with_capacity(DEFAULT_GLOBALS_CAPACITY),
            functions: Vec::new(),
            natives: Vec::new(),
            exception_handlers: Vec::new(),
            error_type_table: Vec::new(),
            global_names: std::collections::BTreeMap::new(),
            explicit_global_names: std::collections::BTreeMap::new(),
            explicit_relations: Vec::new(),
            explicit_primary_keys: Vec::new(),
            loaded_modules: std::collections::HashSet::new(),
            abi_natives: Vec::new(),
            loaded_native_libraries: Vec::new(),
            base_path: None,
            ml_context: Some(crate::ml::MlContext::new()),
            plot_context: Some(crate::plot::PlotContext::new()),
            value_store: ValueStore::new(),
            heavy_store: HeavyStore::new(),
            native_args_buffer: Vec::with_capacity(DEFAULT_NATIVE_BUF_CAPACITY),
            reusable_native_arg_ids: Vec::with_capacity(DEFAULT_NATIVE_BUF_CAPACITY),
            reusable_all_popped: Vec::with_capacity(DEFAULT_NATIVE_BUF_CAPACITY),
            pending_relations: Vec::new(),
            pending_primary_keys: Vec::new(),
            module_cache: RefCell::new(HashMap::new()),
            executed_modules: RefCell::new(HashMap::new()),
            executed_module_functions: RefCell::new(HashMap::new()),
            module_deps: RefCell::new(HashMap::new()),
            modules: RefCell::new(HashMap::new()),
            argv_slot_index: None,
            module_registry: RefCell::new(Vec::new()),
        };
        vm.register_natives();
        vm
    }

    /// Set base path for resolving relative paths (e.g. in settings_env). Used at start of run() to set thread-local.
    pub fn set_base_path(&mut self, path: Option<PathBuf>) {
        self.base_path = path;
    }

    /// Base path for resolving relative paths (imports, load_env). Used by executor when thread-local may not be set.
    pub fn get_base_path(&self) -> Option<PathBuf> {
        self.base_path.clone()
    }

    /// Set the slot index used for argv so update_chunk_indices_from_names (e.g. after ImportFrom) always maps "argv" to this slot.
    pub fn set_argv_slot_index(&mut self, slot: Option<usize>) {
        self.argv_slot_index = slot;
    }

    /// Get the argv slot index if set.
    pub fn get_argv_slot_index(&self) -> Option<usize> {
        self.argv_slot_index
    }

    /// Mutable borrow of the runtime module cache (canonical path -> CachedModule). Used by file_import.
    pub fn get_module_cache_mut(&self) -> std::cell::RefMut<'_, HashMap<PathBuf, CachedModule>> {
        self.module_cache.borrow_mut()
    }

    /// Mutable borrow of executed modules (canonical path -> module namespace). Used by file_import.
    pub fn get_executed_modules_mut(&self) -> std::cell::RefMut<'_, HashMap<PathBuf, Value>> {
        self.executed_modules.borrow_mut()
    }

    /// Mutable borrow of executed module functions (canonical path -> functions). Used by file_import on cache hit to add and remap.
    pub fn get_executed_module_functions_mut(&self) -> std::cell::RefMut<'_, HashMap<PathBuf, Vec<crate::bytecode::Function>>> {
        self.executed_module_functions.borrow_mut()
    }

    /// Mutable borrow of module dependency graph (path -> deps). Used by file_import to register edges after compile.
    pub fn get_module_deps_mut(&self) -> std::cell::RefMut<'_, HashMap<PathBuf, Vec<PathBuf>>> {
        self.module_deps.borrow_mut()
    }

    /// Immutable borrow of module registry (for executor to read loaded module's submodules).
    pub fn get_module_registry(&self) -> std::cell::Ref<'_, Vec<ModuleInfo>> {
        self.module_registry.borrow()
    }

    /// Mutable borrow of module registry (for executor to push ModuleInfo on first import).
    pub fn get_module_registry_mut(&self) -> std::cell::RefMut<'_, Vec<ModuleInfo>> {
        self.module_registry.borrow_mut()
    }

    /// Resolve module_id + local_index to real function index. Returns None if module_id or local_index out of range.
    pub fn get_module_function_index(&self, module_id: usize, local_index: usize) -> Option<usize> {
        let reg = self.module_registry.borrow();
        let info = reg.get(module_id)?;
        if local_index >= info.function_count {
            return None;
        }
        Some(info.function_offset + local_index)
    }

    /// Native index for ValueError::new_1 (constructor used by raise ValueError("...")).
    pub const VALUE_ERROR_NATIVE_INDEX: usize = 75;

    fn register_natives(&mut self) {
        // Регистрируем нативные функции
        // Порядок важен - индексы должны соответствовать register_native_globals
        self.natives.push(natives::native_print);      // 0
        self.natives.push(natives::native_len);        // 1
        self.natives.push(natives::native_range);      // 2
        self.natives.push(natives::native_int);        // 3
        self.natives.push(natives::native_float);        // 4
        self.natives.push(natives::native_bool);        // 5
        self.natives.push(natives::native_str);        // 6
        self.natives.push(natives::native_array);      // 7
        self.natives.push(natives::native_typeof);     // 8
        self.natives.push(natives::native_isinstance); // 9
        self.natives.push(natives::native_date);      // 10
        self.natives.push(natives::native_money);     // 11
        self.natives.push(natives::native_path);     // 12
        self.natives.push(natives::native_path_name);     // 13
        self.natives.push(natives::native_path_parent);   // 14
        self.natives.push(natives::native_path_exists);   // 15
        self.natives.push(natives::native_path_is_file); // 16
        self.natives.push(natives::native_path_is_dir);  // 17
        self.natives.push(natives::native_path_extension); // 18
        self.natives.push(natives::native_path_stem);    // 19
        self.natives.push(natives::native_path_len);     // 20
        // Математические функции
        self.natives.push(natives::native_abs);          // 21
        self.natives.push(natives::native_sqrt);         // 22
        self.natives.push(natives::native_pow);          // 23
        self.natives.push(natives::native_min);          // 24
        self.natives.push(natives::native_max);          // 25
        self.natives.push(natives::native_round);        // 26
        // Строковые функции
        self.natives.push(natives::native_upper);        // 27
        self.natives.push(natives::native_lower);        // 28
        self.natives.push(natives::native_trim);         // 29
        self.natives.push(natives::native_split);        // 30
        self.natives.push(natives::native_join);         // 31
        self.natives.push(natives::native_contains);     // 32
        self.natives.push(natives::native_isupper);      // 33
        self.natives.push(natives::native_islower);      // 34
        // Функции массивов
        self.natives.push(natives::native_push);         // 35
        self.natives.push(natives::native_pop);          // 34
        self.natives.push(natives::native_unique);       // 35
        self.natives.push(natives::native_reverse);      // 36
        self.natives.push(natives::native_sort);        // 37
        self.natives.push(natives::native_sum);          // 38
        self.natives.push(natives::native_average);     // 39
        self.natives.push(natives::native_count);        // 40
        self.natives.push(natives::native_any);          // 41
        self.natives.push(natives::native_all);          // 42
        // Функции для работы с таблицами
        self.natives.push(natives::native_table);        // 43
        self.natives.push(natives::native_read_file);    // 44
        self.natives.push(natives::native_table_info);   // 45
        self.natives.push(natives::native_table_head);   // 46
        self.natives.push(natives::native_table_tail);   // 47
        self.natives.push(natives::native_table_select); // 48
        self.natives.push(natives::native_table_sort);   // 49
        self.natives.push(natives::native_table_where);  // 50
        self.natives.push(natives::native_show_table);   // 51
        self.natives.push(natives::native_merge_tables); // 52
        self.natives.push(natives::native_now);          // 53
        self.natives.push(natives::native_getcwd);       // 54
        self.natives.push(natives::native_list_files);   // 55
        // JOIN операции
        self.natives.push(natives::native_inner_join);   // 56
        self.natives.push(natives::native_left_join);    // 57
        self.natives.push(natives::native_right_join);  // 58
        self.natives.push(natives::native_full_join);    // 59
        self.natives.push(natives::native_cross_join);   // 60
        self.natives.push(natives::native_semi_join);   // 61
        self.natives.push(natives::native_anti_join);   // 62
        self.natives.push(natives::native_zip_join);    // 63
        self.natives.push(natives::native_asof_join);   // 64
        self.natives.push(natives::native_apply_join);   // 65
        self.natives.push(natives::native_join_on);     // 66
        self.natives.push(natives::native_table_suffixes); // 67
        self.natives.push(natives::native_relate);      // 68
        self.natives.push(natives::native_primary_key); // 69
        self.natives.push(natives::native_enum);       // 70
        self.natives.push(natives::native_table_class); // 71 - Table (built-in class for inheritance)
        self.natives.push(natives::native_array_with_capacity); // 72
        // 73, 74: reserved for future builtins if BUILTIN_GLOBAL_NAMES grows
        while self.natives.len() < Self::VALUE_ERROR_NATIVE_INDEX {
            self.natives.push(natives::native_value_error_new); // placeholder so indices line up
        }
        self.natives.push(natives::native_value_error_new); // 75 - ValueError::new_1 for raise ValueError("...")
    }

    /// Добавляет в VM слоты для всех имён из chunk.global_names, которых ещё нет в VM.
    /// Индексы назначаются в детерминированном порядке по idx из chunk (sort_by_key), чтобы
    /// слот в VM совпадал с индексом в chunk и не требовался swap при патче (устраняет
    /// флакующие тесты при "from settings_env import load_env").
    /// Нужно вызывать перед update_chunk_indices, чтобы главный chunk мог ссылаться на __main__.
    pub fn ensure_globals_from_chunk(&mut self, chunk: &crate::bytecode::Chunk) {
        /// Compiler uses this for undefined variables; we must not allocate a slot (would mis-report "undefined" as null).
        const UNDEFINED_GLOBAL_SENTINEL: usize = usize::MAX;
        // Не добавляем слот для "argv" из chunk: argv устанавливается в lib.rs в один слот в конце,
        // иначе получается два слота (из chunk и наш), байткод мапится на min и может загрузить функцию.
        let mut entries: Vec<_> = chunk
            .global_names
            .iter()
            .filter(|(idx, _)| **idx != UNDEFINED_GLOBAL_SENTINEL)
            .filter(|(_, name)| name.as_str() != "argv")
            .filter(|(_, name)| !self.global_names.values().any(|n| n == name.as_str()))
            .map(|(idx, name)| (*idx, name.clone()))
            .collect();
        entries.sort_by_key(|(idx, _)| *idx);
        for (_old_idx, name) in entries {
            let new_idx = self.globals.len();
            self.globals.push(global_slot::default_global_slot());
            self.global_names.insert(new_idx, name.clone());
            debug_println!("[DEBUG ensure_globals_from_chunk] Добавлен слот для '{}' в globals[{}]", name, new_idx);
            if name == "Config" || name == "DatabaseConfig" {
                debug_println!("[DEBUG ensure_globals_from_chunk] Config/DatabaseConfig: '{}' -> слот {}", name, new_idx);
            }
        }
    }

    /// Как ensure_globals_from_chunk, но сохраняет индексы из chunk: слоты создаются по chunk.global_names (idx, name),
    /// с расширением globals при необходимости. Нужно для модулей (file_import), чтобы main chunk без патча писал по тем же индексам.
    /// Всегда привязываем (idx, name) из chunk к VM по этому индексу, чтобы StoreGlobal(idx) в main chunk записал в слот,
    /// который при экспорте будет иметь это имя (иначе конструктор может оказаться только по другому индексу и экспортироваться Null).
    pub fn ensure_globals_from_chunk_preserve_indices(&mut self, chunk: &crate::bytecode::Chunk) {
        /// Compiler uses this for undefined variables; we must not allocate a slot (would overflow on resize).
        const UNDEFINED_GLOBAL_SENTINEL: usize = usize::MAX;
        let mut entries: Vec<_> = chunk
            .global_names
            .iter()
            .filter(|(i, _)| **i != UNDEFINED_GLOBAL_SENTINEL)
            .map(|(i, n)| (*i, n.clone()))
            .collect();
        entries.sort_by_key(|(idx, _)| *idx);
        for (idx, name) in entries {
            if idx >= self.globals.len() {
                self.globals.resize(idx + 1, global_slot::default_global_slot());
            }
            self.global_names.insert(idx, name.clone());
            debug_println!("[DEBUG ensure_globals_from_chunk_preserve_indices] Добавлен слот для '{}' в globals[{}]", name, idx);
            if name == "Config" || name == "DatabaseConfig" {
                debug_println!("[DEBUG ensure_globals_from_chunk_preserve_indices] Config/DatabaseConfig: '{}' -> слот {}", name, idx);
            }
        }
        debug_println!(
            "[DEBUG ensure_globals_from_chunk_preserve_indices] после: global_names 75..80: {:?}",
            (75..80).filter_map(|i| self.global_names.get(&i).map(|n| (i, n.as_str()))).collect::<Vec<_>>()
        );
    }

    /// Fills global slots at index >= 75 whose name is a builtin (e.g. "str", "path").
    /// ensure_globals_from_chunk_preserve_indices adds (idx, name) from chunk and only resizes;
    /// register_native_globals only fills 0..75, so slots at 75+ stay null and LoadGlobal(idx) returns null → "Can only call functions".
    pub fn ensure_builtin_globals_high_indices(&mut self) {
        const BUILTIN_END: usize = 75;
        for (idx, name) in self.global_names.iter() {
            if *idx < BUILTIN_END {
                continue;
            }
            if let Some(builtin_index) = globals::builtin_global_index(name) {
                if *idx >= self.globals.len() {
                    continue;
                }
                let value_id = self.globals[*idx].resolve_to_value_id(&mut self.value_store);
                let current = load_value(value_id, &self.value_store, &self.heavy_store);
                if matches!(current, Value::Null) {
                    let id = self.value_store.allocate(ValueCell::NativeFunction(builtin_index));
                    self.globals[*idx] = GlobalSlot::Heap(id);
                    debug_println!("[DEBUG ensure_builtin_globals_high_indices] '{}' at globals[{}] -> builtin index {}", name, idx, builtin_index);
                }
            }
        }
    }

    /// Injects built-in exception constructors (e.g. ValueError::new_1) into slots that are still null after ensure_globals_from_chunk.
    /// Called from run_compiled_module so that raise ValueError("...") works in modules.
    pub fn ensure_exception_constructors(&mut self) {
        const NAME: &str = "ValueError::new_1";
        let idx = match self.global_names.iter().find(|(_, n)| n.as_str() == NAME).map(|(i, _)| *i) {
            Some(i) => i,
            None => return,
        };
        if idx >= self.globals.len() {
            return;
        }
        let value_id = self.globals[idx].resolve_to_value_id(&mut self.value_store);
        let current = load_value(value_id, &self.value_store, &self.heavy_store);
        if matches!(current, Value::Null) {
            let id = self.value_store.allocate(ValueCell::NativeFunction(Self::VALUE_ERROR_NATIVE_INDEX));
            self.globals[idx] = GlobalSlot::Heap(id);
            debug_println!("[DEBUG ensure_exception_constructors] Injected {} at globals[{}]", NAME, idx);
        }
    }

    /// Обновляет индексы глобальных переменных в chunk на основе реальных индексов в VM
    pub fn update_chunk_indices(&mut self, chunk: &mut crate::bytecode::Chunk) {
        Self::update_chunk_indices_from_names(
            chunk,
            &self.global_names,
            Some(self.globals.as_mut_slice()),
            Some(&mut self.value_store),
            Some(&self.heavy_store),
            self.argv_slot_index,
        );
    }

    /// Sentinel index for model_config class load in Settings subclass constructors (must match compiler).
    const MODEL_CONFIG_CLASS_LOAD_INDEX: usize = 0x0FFF_FFFF;

    /// Обновляет индексы глобалов в chunk по переданной карте имён.
    /// globals_for_verify: Option<&mut [GlobalSlot]>; при проверке слот материализуется через resolve_to_value_id + load_value (с кэшем).
    /// store/heap: когда None, проверка по слотам не выполняется (для вызовов без доступа к store/heap).
    /// argv_slot_index: когда Some(slot), имя "argv" всегда мапится на этот слот (после ImportFrom слот 79 может стать load_settings).
    pub fn update_chunk_indices_from_names(
        chunk: &mut crate::bytecode::Chunk,
        global_names: &std::collections::BTreeMap<usize, String>,
        mut globals_for_verify: Option<&mut [GlobalSlot]>,
        mut store: Option<&mut ValueStore>,
        heap: Option<&HeavyStore>,
        argv_slot_index: Option<usize>,
    ) {
        // Phase 1: Build old_idx → real_idx mapping (no bytecode changes).
        // Resolve real_idx by name deterministically (min index if multiple) for stability.
        // Iterate in deterministic order (by name then index) to avoid HashMap iteration order affecting outcome.
        let mut old_to_real: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
        let mut name_index_pairs: Vec<_> = chunk.global_names.iter().map(|(i, n)| (*i, n.clone())).collect();
        name_index_pairs.sort_by(|a, b| a.1.cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
        let can_prefer_non_null = globals_for_verify.is_some() && store.is_some() && heap.is_some();
        for (old_idx, name) in &name_index_pairs {
            // Always map "argv" to the designated slot when set (avoids ImportFrom re-patch mapping it to load_settings at 79).
            let real_idx: Option<usize> = if name == "argv" && argv_slot_index.is_some() {
                argv_slot_index
            } else {
                let matching_indices: Vec<usize> = global_names
                    .iter()
                    .filter(|(_, n)| *n == name)
                    .map(|(idx, _)| *idx)
                    .collect();
                // Prefer index whose slot is non-null when multiple indices share the same name (e.g. 84 Null, 92 Function).
                // For constructor names (::new_), prefer slot that holds Function so we don't pick argv (Array) at lower index.
                if matching_indices.is_empty() {
                    None
                } else if can_prefer_non_null {
                    const BUILTIN_COUNT: usize = 75;
                    if let (Some(globals), Some(store), Some(heap)) = (globals_for_verify.as_deref_mut(), store.as_deref_mut(), heap) {
                        let want_function = name.contains("::new_");
                        let non_null: Vec<usize> = matching_indices
                            .iter()
                            .filter(|&&idx| {
                                idx >= BUILTIN_COUNT
                                    && idx < globals.len()
                                    && !matches!(
                                        load_value(globals[idx].resolve_to_value_id(store), store, heap),
                                        Value::Null
                                    )
                            })
                            .copied()
                            .collect();
                        let preferred: Option<usize> = if want_function {
                            non_null
                                .iter()
                                .filter(|&&idx| {
                                    idx < globals.len()
                                        && matches!(
                                            load_value(globals[idx].resolve_to_value_id(store), store, heap),
                                            Value::Function(_) | Value::ModuleFunction { .. }
                                        )
                                })
                                .min()
                                .copied()
                        } else {
                            None
                        };
                        preferred
                            .or_else(|| non_null.into_iter().min())
                            .or_else(|| matching_indices.iter().filter(|&&i| i >= BUILTIN_COUNT).min().copied())
                            .or_else(|| matching_indices.iter().min().copied())
                    } else {
                        matching_indices.iter().min().copied()
                    }
                } else {
                    matching_indices.iter().min().copied()
                }
            };
            if let Some(r) = real_idx {
                // Never remap the argv slot to another index only when this chunk's name at old_idx is "argv".
                let r_final = if name == "argv" && argv_slot_index == Some(*old_idx) && r != *old_idx {
                    *old_idx
                } else {
                    r
                };
                if *old_idx != r_final {
                    // Avoid collision: if real_idx is already used in chunk for a different name that we won't remap
                    // (no match in global_names or match only at real_idx), don't overwrite it.
                    let would_overwrite = chunk.global_names.get(&r_final).map(|existing| {
                        existing != name
                            && global_names
                                .iter()
                                .filter(|(_, n)| *n == existing)
                                .map(|(idx, _)| *idx)
                                .all(|idx| idx == r_final)
                    }).unwrap_or(false);
                    if !would_overwrite {
                        old_to_real.insert(*old_idx, r_final);
                        debug_println!("[DEBUG update_chunk_indices] Маппинг '{}': {} -> {} (argv forced={})", name, old_idx, r_final, name == "argv" && argv_slot_index.is_some());
                    } else {
                        debug_println!("[DEBUG update_chunk_indices] Пропуск маппинга '{}': {} -> {} (слот {} занят другим именем)", name, old_idx, r_final, r_final);
                    }
                }
            } else {
                // Warn when no match: always for argv, for other names only when verbose constructor debug is on.
                if name == "argv" || crate::common::debug::verbose_constructor_debug() {
                    eprintln!("[update_chunk_indices] no match for name '{}' (old_idx {}), chunk LoadGlobal will not be patched", name, old_idx);
                }
            }
        }
        if crate::common::debug::verbose_constructor_debug() {
            for (old_idx, name) in &name_index_pairs {
                if let Some(&real_idx) = old_to_real.get(old_idx) {
                    eprintln!("[update_chunk_indices] '{}' old_idx {} -> real_idx {}", name, old_idx, real_idx);
                } else {
                    eprintln!("[update_chunk_indices] '{}' old_idx {} -> no match (not in caller global_names)", name, old_idx);
                }
            }
        }
        // Resolve compiler's UNDEFINED_GLOBAL_SENTINEL (usize::MAX) by name so merged functions see caller globals
        const UNDEFINED_GLOBAL_SENTINEL: usize = usize::MAX;
        if let Some(name) = chunk.global_names.get(&UNDEFINED_GLOBAL_SENTINEL) {
            let matching: Vec<usize> = global_names
                .iter()
                .filter(|(_, n)| *n == name)
                .map(|(idx, _)| *idx)
                .collect();
            if let Some(&real_idx) = matching.iter().min() {
                old_to_real.insert(UNDEFINED_GLOBAL_SENTINEL, real_idx);
                debug_println!("[DEBUG update_chunk_indices] Маппинг sentinel '{}': MAX -> {}", name, real_idx);
            }
        }
        // Always bind model_config sentinel to the class by name so Settings subclass constructors load the correct class for GetArrayElement(class, "model_config").
        let needs_sentinel = chunk.code.iter().any(|op| {
            matches!(op, crate::bytecode::OpCode::LoadGlobal(i) if *i == Self::MODEL_CONFIG_CLASS_LOAD_INDEX)
        });
        if needs_sentinel {
            if let Some(name) = chunk.global_names.get(&Self::MODEL_CONFIG_CLASS_LOAD_INDEX) {
                let matching_indices: Vec<usize> = global_names
                    .iter()
                    .filter(|(_, n)| *n == name)
                    .map(|(idx, _)| *idx)
                    .collect();
                // For __constructing_class__ use .min() so the same slot is used as in executor (global_index_by_name uses .min()).
                let real_idx = if name.as_str() == "__constructing_class__" {
                    matching_indices.into_iter().min()
                } else {
                    // Prefer the slot that holds the class Object (has __class_name, model_config), not a constructor (Function).
                    let mut class_slots = Vec::new();
                    if let (Some(globals), Some(store), Some(heap)) = (globals_for_verify.as_deref_mut(), store.as_deref_mut(), heap.as_deref()) {
                        for &idx in &matching_indices {
                            if idx < globals.len() {
                                let id = globals[idx].resolve_to_value_id(store);
                                let v = load_value(id, store, heap);
                                if let Value::Object(obj_rc) = &v {
                                    let obj = obj_rc.borrow();
                                    if obj.get("__class_name").is_some() && obj.get("model_config").is_some() {
                                        class_slots.push(idx);
                                    }
                                }
                            }
                        }
                    }
                    class_slots.into_iter().max().or_else(|| matching_indices.into_iter().max())
                };
                if let Some(real_idx) = real_idx {
                    old_to_real.insert(Self::MODEL_CONFIG_CLASS_LOAD_INDEX, real_idx);
                    debug_println!("[DEBUG update_chunk_indices] Маппинг model_config (sentinel) класс '{}': sentinel -> globals[{}]", name, real_idx);
                    if let (Some(globals), Some(store), Some(heap)) = (globals_for_verify.as_deref_mut(), store.as_deref_mut(), heap.as_deref()) {
                        if real_idx < globals.len() {
                            let id = globals[real_idx].resolve_to_value_id(store);
                            let v = load_value(id, store, heap);
                            let slot_type = match &v {
                                Value::Object(_) => "Object",
                                Value::Function(_) | Value::ModuleFunction { .. } => "Function",
                                Value::Null => "Null",
                                _ => "Other",
                            };
                            debug_println!("[DEBUG update_chunk_indices] sentinel '{}' -> globals[{}], значение в слоте: {}", name, real_idx, slot_type);
                            if let Value::Object(obj_rc) = &v {
                                let obj = obj_rc.borrow();
                                if let Some(Value::String(actual_name)) = obj.get("__class_name") {
                                    debug_println!("[DEBUG update_chunk_indices] globals[{}].__class_name = '{}'", real_idx, actual_name);
                                    if actual_name != name {
                                        debug_println!(
                                            "[DEBUG update_chunk_indices] WARNING: ожидался класс '{}', в слоте {} — класс '{}'",
                                            name, real_idx, actual_name
                                        );
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // Phase 2: Single pass over bytecode; use original idx from instructions only (no overwriting of already-patched).
        let mut updated_count = 0usize;
        for opcode in &mut chunk.code {
            match opcode {
                crate::bytecode::OpCode::LoadGlobal(idx) => {
                    if let Some(&real) = old_to_real.get(idx) {
                        *opcode = crate::bytecode::OpCode::LoadGlobal(real);
                        updated_count += 1;
                    }
                }
                crate::bytecode::OpCode::StoreGlobal(idx) => {
                    if let Some(&real) = old_to_real.get(idx) {
                        *opcode = crate::bytecode::OpCode::StoreGlobal(real);
                        updated_count += 1;
                    }
                }
                _ => {}
            }
        }
        if !old_to_real.is_empty() {
            debug_println!("[DEBUG update_chunk_indices] Обновлено {} инструкций по {} маппингам", updated_count, old_to_real.len());
        }

        // Phase 3: Update chunk.global_names. Collect (real_idx, name) first so we don't overwrite
        // or remove entries that are still needed when mappings overlap (e.g. 71->70, 72->71, 73->72).
        let to_insert: Vec<_> = old_to_real
            .iter()
            .filter_map(|(old_idx, real_idx)| {
                chunk.global_names.get(old_idx).map(|name| (*real_idx, name.clone()))
            })
            .collect();
        for old_idx in old_to_real.keys() {
            chunk.global_names.remove(old_idx);
        }
        for (real_idx, name) in to_insert {
            // Don't overwrite real_idx if it already has a different name we didn't remove (collision).
            if let Some(existing) = chunk.global_names.get(&real_idx) {
                if existing != &name && !old_to_real.contains_key(&real_idx) {
                    debug_println!("[DEBUG update_chunk_indices] Phase3: пропуск вставки (real_idx {} уже = '{}', не перезаписываем на '{}')", real_idx, existing, name);
                    continue;
                }
            }
            chunk.global_names.insert(real_idx, name);
        }
    }

    /// Находит индекс функции в списке функций VM по имени функции
    fn find_function_index_by_name(&self, function_name: &str) -> Option<usize> {
        self.functions.iter()
            .position(|f| f.name == function_name)
    }

    /// Устанавливает функции в VM. Если передан main_chunk, маппинг глобалов строится по нему
    /// (для run() главный chunk не входит в functions; для run_with_vm_internal_with_args можно передать None).
    /// Используется маппинг по имени (name -> new_idx), чтобы один и тот же old_idx в разных чанках
    /// с разными именами (например 72 = "config" в main и 72 = "DatabaseConfig" в конструкторе) патчился в разные слоты.
    /// main_old_idx_to_name: снимок main chunk global_names (idx -> name) до вызова update_chunk_indices,
    /// чтобы при патче байткода вложенных функций (main/__main__) правильно резолвить индексы по имени
    /// (иначе 78 в main() для load_settings не патчится в 79, т.к. main chunk уже имеет 78=get_settings).
    pub fn set_functions(
        &mut self,
        functions: Vec<crate::bytecode::Function>,
        main_chunk: Option<&mut crate::bytecode::Chunk>,
        main_old_idx_to_name: Option<std::collections::HashMap<usize, String>>,
    ) {
        let mut explicit_global_names_to_add = std::collections::HashMap::new();

        // First pass: collect all (old_idx, name) from main chunk + all function chunks (no overwrite by index).
        let mut all_pairs: Vec<(usize, String)> = Vec::new();
        if let Some(mc) = main_chunk.as_ref() {
            for (idx, name) in &mc.global_names {
                all_pairs.push((*idx, name.clone()));
            }
        }
        for function in &functions {
            for (idx, name) in &function.chunk.global_names {
                all_pairs.push((*idx, name.clone()));
            }
        }
        // Unique names in deterministic order (sort by name).
        let mut unique_names: Vec<String> = all_pairs.iter().map(|(_, n)| n.clone()).collect();
        unique_names.sort();
        unique_names.dedup();

        // Build name -> new_idx: each distinct name gets one slot in the VM.
        // Встроенные имена (sum, count, ...) всегда мапятся на канонический индекс 0..69.
        // Names that only appear as UNDEFINED_GLOBAL_SENTINEL (compiler's undefined-variable placeholder) are not given a slot so LoadGlobal(MAX) is not patched and runtime reports "Undefined variable".
        const UNDEFINED_GLOBAL_SENTINEL: usize = usize::MAX;
        let mut name_to_new_idx: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
        let mut constructor_slots_used: std::collections::HashSet<usize> = std::collections::HashSet::new();
        for name in &unique_names {
            let only_sentinel = all_pairs.iter()
                .filter(|(_, n)| n == name)
                .all(|(idx, _)| *idx == UNDEFINED_GLOBAL_SENTINEL);
            if only_sentinel {
                // Do not allocate a slot; bytecode will keep LoadGlobal(MAX) and runtime will error.
                continue;
            }
            let new_idx = if let Some(builtin_idx) = globals::builtin_global_index(name) {
                debug_println!("[DEBUG set_functions] Встроенное имя '{}' -> канонический индекс {}", name, builtin_idx);
                builtin_idx
            } else if name == "argv" && self.argv_slot_index.is_some() {
                // argv must always map to the slot we reserved (lib.rs pushes the array there); otherwise
                // min() could pick another slot and main chunk LoadGlobal could end up loading a function.
                let slot = self.argv_slot_index.unwrap();
                debug_println!("[DEBUG set_functions] argv -> слот {} (argv_slot_index)", slot);
                slot
            } else {
                let matching_indices: Vec<usize> = self.global_names.iter()
                    .filter(|(_, n)| *n == name)
                    .map(|(idx, _)| *idx)
                    .collect();
                let candidate = matching_indices.iter().min().copied();
                let is_constructor = name.contains("::new_");
                let new_idx = if is_constructor {
                    if let Some(real_idx) = candidate {
                        if constructor_slots_used.insert(real_idx) {
                            real_idx
                        } else {
                            let fresh = self.globals.len();
                            self.globals.push(global_slot::default_global_slot());
                            self.global_names.insert(fresh, name.clone());
                            constructor_slots_used.insert(fresh);
                            debug_println!("[DEBUG set_functions] Конструктор '{}' -> новый слот {} (конфликт)", name, fresh);
                            fresh
                        }
                    } else {
                        let fresh = self.globals.len();
                        self.globals.push(global_slot::default_global_slot());
                        self.global_names.insert(fresh, name.clone());
                        constructor_slots_used.insert(fresh);
                        debug_println!("[DEBUG set_functions] Конструктор '{}' не найден в VM, слот {}", name, fresh);
                        fresh
                    }
                } else if let Some(real_idx) = candidate {
                    debug_println!("[DEBUG set_functions] Найдена переменная '{}' в VM с индексом {} (детерминированный min)", name, real_idx);
                    if name == "Config" || name == "DatabaseConfig" {
                        debug_println!("[DEBUG set_functions] Config/DatabaseConfig: '{}' -> индекс {} (найден в self.global_names)", name, real_idx);
                    }
                    real_idx
                } else {
                    let new_idx = self.globals.len();
                    self.globals.push(global_slot::default_global_slot());
                    self.global_names.insert(new_idx, name.clone());
                    debug_println!("[DEBUG set_functions] Переменная '{}' не найдена в VM, создаем новый индекс {}", name, new_idx);
                    if name == "Config" || name == "DatabaseConfig" {
                        debug_println!("[DEBUG set_functions] Config/DatabaseConfig: '{}' -> индекс {} (создан новый слот)", name, new_idx);
                    }
                    new_idx
                };
                new_idx
            };
            name_to_new_idx.insert(name.clone(), new_idx);
        }

        if main_chunk.is_none() {
            if let Some(main_function) = functions.first() {
                for (idx, name) in &main_function.chunk.explicit_global_names {
                    explicit_global_names_to_add.insert(*idx, name.clone());
                }
            }
        }

        // main_old_idx_to_name used only for function chunks; main chunk is patched from current mc.global_names (after update_chunk_indices).
        let main_old_idx_to_name: std::collections::HashMap<usize, String> = main_old_idx_to_name
            .unwrap_or_else(|| {
                main_chunk
                    .as_ref()
                    .map(|mc| mc.global_names.iter().map(|(i, n)| (*i, n.clone())).collect())
                    .unwrap_or_default()
            });

        // Patch main chunk: используем текущие mc.global_names (после update_chunk_indices), а не снимок:
        // в снимке 75=main, а в chunk уже 75=__main__ (кали), и патч по снимку 75→76 подменяет вызов __main__ на main.
        if let Some(mc) = main_chunk {
            let old_to_new: std::collections::HashMap<usize, usize> = mc
                .global_names
                .iter()
                .filter_map(|(old_idx, name)| {
                    name_to_new_idx.get(name).and_then(|&new_idx| {
                        if *old_idx != new_idx {
                            Some((*old_idx, new_idx))
                        } else {
                            None
                        }
                    })
                })
                .collect();
            for opcode in &mut mc.code {
                match opcode {
                    crate::bytecode::OpCode::LoadGlobal(idx) => {
                        if let Some(&new_idx) = old_to_new.get(idx) {
                            *opcode = crate::bytecode::OpCode::LoadGlobal(new_idx);
                        }
                    }
                    crate::bytecode::OpCode::StoreGlobal(idx) => {
                        if let Some(&new_idx) = old_to_new.get(idx) {
                            *opcode = crate::bytecode::OpCode::StoreGlobal(new_idx);
                        }
                    }
                    _ => {}
                }
            }
            let to_insert: Vec<_> = old_to_new
                .iter()
                .filter_map(|(old_idx, new_idx)| {
                    mc.global_names.get(old_idx).map(|name| (*old_idx, *new_idx, name.clone()))
                })
                .collect();
            for (old_idx, _, _) in &to_insert {
                mc.global_names.remove(old_idx);
            }
            for (_, new_idx, name) in to_insert {
                mc.global_names.insert(new_idx, name);
            }
        }

        // Patch each function chunk: один проход с маппингом old_idx -> new_idx.
        // Sentinel (model_config) must resolve from VM's global_names so it points to the actual class slot (e.g. 71 from ensure/merge), not name_to_new_idx (which can allocate a new slot 74 when "Config" wasn't found yet).
        let existing_functions_count = self.functions.len();
        let mut updated_functions = functions;
        for function in &mut updated_functions {
            let mut old_to_new: std::collections::HashMap<usize, usize> = function
                .chunk
                .global_names
                .iter()
                .filter_map(|(old_idx, name)| {
                    let new_idx = if *old_idx == Self::MODEL_CONFIG_CLASS_LOAD_INDEX {
                        // Resolve sentinel by name from VM's current global_names. Use .min() so the same slot is used as in executor (global_index_by_name uses .min()).
                        let matching: Vec<usize> = self.global_names.iter()
                            .filter(|(_, n)| *n == name)
                            .map(|(idx, _)| *idx)
                            .collect();
                        if let Some(&real_idx) = matching.iter().min() {
                            debug_println!(
                                "[DEBUG set_functions] sentinel функция '{}' класс '{}' -> globals[{}] (из VM global_names)",
                                function.name, name, real_idx
                            );
                            Some(real_idx)
                        } else {
                            let fallback = name_to_new_idx.get(name).copied();
                            debug_println!(
                                "[DEBUG set_functions] WARNING: sentinel функция '{}' класс '{}' не найден в self.global_names, fallback name_to_new_idx -> {:?}",
                                function.name, name, fallback
                            );
                            fallback
                        }
                    } else {
                        name_to_new_idx.get(name).copied()
                    };
                    new_idx.and_then(|new_idx| {
                        if *old_idx != new_idx {
                            Some((*old_idx, new_idx))
                        } else {
                            None
                        }
                    })
                })
                .collect();
            // Resolve indices that appear in this function's bytecode but not in old_to_new.
            // Prefer this function's chunk.global_names over main_old_idx_to_name.
            for (_op_index, opcode) in function.chunk.code.iter().enumerate() {
                let idx = match opcode {
                    crate::bytecode::OpCode::LoadGlobal(i) | crate::bytecode::OpCode::StoreGlobal(i) => *i,
                    _ => continue,
                };
                if old_to_new.contains_key(&idx) {
                    continue;
                }
                let name = function.chunk.global_names.get(&idx)
                    .or_else(|| main_old_idx_to_name.get(&idx));
                if let Some(name) = name {
                    if let Some(&new_idx) = name_to_new_idx.get(name) {
                        if idx != new_idx {
                            old_to_new.insert(idx, new_idx);
                        }
                    }
                }
            }
            for opcode in &mut function.chunk.code {
                match opcode {
                    crate::bytecode::OpCode::LoadGlobal(idx) => {
                        if let Some(&new_idx) = old_to_new.get(idx) {
                            *opcode = crate::bytecode::OpCode::LoadGlobal(new_idx);
                        }
                    }
                    crate::bytecode::OpCode::StoreGlobal(idx) => {
                        if let Some(&new_idx) = old_to_new.get(idx) {
                            *opcode = crate::bytecode::OpCode::StoreGlobal(new_idx);
                        }
                    }
                    _ => {}
                }
            }
            // Collect (old_idx, new_idx, name) then remove all old, then insert all new.
            // HashMap iteration order is non-deterministic; with overlapping mappings (e.g. 78->79, 79->78)
            // processing (79,78) first would overwrite slot 78 before we read it and lose a name.
            let to_insert: Vec<_> = old_to_new
                .iter()
                .filter_map(|(old_idx, new_idx)| {
                    function.chunk.global_names.get(old_idx).map(|name| (*old_idx, *new_idx, name.clone()))
                })
                .collect();
            for (old_idx, _, _) in &to_insert {
                function.chunk.global_names.remove(old_idx);
            }
            for (_, new_idx, name) in to_insert {
                function.chunk.global_names.insert(new_idx, name);
            }
        }
        
        // Добавляем новые функции к существующим, а не заменяем их
        // Это важно, потому что функции из __lib__.dc уже были добавлены через merge_globals_from
        debug_println!("[DEBUG set_functions] Добавляем {} новых функций к существующим {} функциям", updated_functions.len(), existing_functions_count);
        
        // Отладочный вывод: показываем все Call инструкции в функциях после обновления
        for function in &updated_functions {
            debug_println!("[DEBUG set_functions] Проверка Call инструкций в функции '{}':", function.name);
            for (ip, opcode) in function.chunk.code.iter().enumerate() {
                if let crate::bytecode::OpCode::Call(arity) = opcode {
                    debug_println!("[DEBUG set_functions]   IP {}: Call({})", ip, arity);
                }
            }
        }
        
        // ВАЖНО: После добавления функций из основного скрипта, нужно обновить индексы функций
        // в глобальных переменных, потому что функции из __lib__.dc были добавлены первыми,
        // и их индексы могли измениться.
        
        // Собираем уникальные имена из chunk'ов новых функций. Итерация в детерминированном порядке (сортировка по имени),
        // чтобы слот для каждого имени всегда определялся через name_to_new_idx одинаково.
        let mut names_from_chunks: std::collections::HashSet<String> = std::collections::HashSet::new();
        for function in &updated_functions {
            for (_, name) in &function.chunk.global_names {
                names_from_chunks.insert(name.clone());
            }
        }
        let mut names_sorted: Vec<String> = names_from_chunks.into_iter().collect();
        names_sorted.sort();
        let chunk_global_names: Vec<(usize, String)> = names_sorted
            .iter()
            .filter_map(|name| name_to_new_idx.get(name).map(|&idx| (idx, name.clone())))
            .collect();
        
        self.functions.extend(updated_functions);
        
        // Обновляем индексы функций в глобальных переменных по имени и name_to_new_idx (детерминированный слот).
        debug_println!("[DEBUG set_functions] Начинаем обновление индексов функций в globals. Всего глобальных переменных в VM: {}, имён из chunk: {}", self.global_names.len(), chunk_global_names.len());
        
        for (real_global_idx, name) in &chunk_global_names {
            debug_println!("[DEBUG set_functions] Обрабатываем '{}' из chunk -> слот {}", name, real_global_idx);
            // Не трогаем слот argv — туда lib.rs кладёт массив аргументов скрипта
            if Some(*real_global_idx) == self.argv_slot_index {
                continue;
            }
            // Слот встроенного натива: пользовательская функция с тем же именем перекрывает встроенный
            if *real_global_idx < globals::BUILTIN_GLOBAL_NAMES.len() {
                if let Some(canonical) = globals::builtin_global_name(*real_global_idx) {
                    if canonical == name.as_str() {
                        if let Some(fn_idx) = self.find_function_index_by_name(name) {
                            self.globals[*real_global_idx] = GlobalSlot::Heap(self.value_store.allocate_arena(ValueCell::Function(fn_idx)));
                            debug_println!("[DEBUG set_functions] Перезапись встроенного натива '{}' в слоте {} пользовательской функцией", name, real_global_idx);
                        } else {
                            debug_println!("[DEBUG set_functions] Пропуск перезаписи встроенного натива '{}' в слоте {}", name, real_global_idx);
                        }
                        continue;
                    }
                }
            }
            if *real_global_idx < self.globals.len() {
                debug_println!("[DEBUG set_functions] Проверяем globals[{}] для '{}'", real_global_idx, name);
                let id = self.globals[*real_global_idx].resolve_to_value_id(&mut self.value_store);
                let old_fn_idx_opt = self.value_store.get(id)
                    .and_then(|c| if let ValueCell::Function(i) = c { Some(*i) } else { None });
                if let Some(old_fn_idx) = old_fn_idx_opt {
                    debug_println!("[DEBUG set_functions] Найдена функция '{}' в globals[{}] (из chunk) с индексом функции {}", name, real_global_idx, old_fn_idx);
                    if let Some(new_fn_idx) = self.find_function_index_by_name(name) {
                        if old_fn_idx != new_fn_idx {
                            debug_println!("[DEBUG set_functions] Обновляем индекс функции для '{}' в globals[{}]: {} -> {}", 
                                name, real_global_idx, old_fn_idx, new_fn_idx);
                            self.globals[*real_global_idx] = GlobalSlot::Heap(self.value_store.allocate_arena(ValueCell::Function(new_fn_idx)));
                        } else {
                            debug_println!("[DEBUG set_functions] Индекс функции для '{}' уже правильный: {}", name, new_fn_idx);
                        }
                    } else {
                        debug_println!("[DEBUG set_functions] WARNING: Функция '{}' не найдена в списке функций VM", name);
                    }
                } else {
                    if let Some(fn_idx) = self.find_function_index_by_name(name) {
                        self.globals[*real_global_idx] = GlobalSlot::Heap(self.value_store.allocate_arena(ValueCell::Function(fn_idx)));
                        debug_println!("[DEBUG set_functions] Установлена функция '{}' в globals[{}] = Value::Function({}) (слот был Null/другой)", name, real_global_idx, fn_idx);
                    } else {
                        debug_println!("[DEBUG set_functions] globals[{}] для '{}' не является функцией", real_global_idx, name);
                    }
                }
            } else {
                debug_println!("[DEBUG set_functions] WARNING: Индекс {} выходит за границы globals (всего: {})", real_global_idx, self.globals.len());
            }
        }
        
        // Также обрабатываем все функции, которые были только что добавлены
        // Это нужно для функций, которые не были добавлены в global_names (например, __main__)
        // Итерируемся по функциям, которые были добавлены в этом вызове set_functions
        let start_fn_idx = existing_functions_count;
        for new_fn_idx in start_fn_idx..self.functions.len() {
            let function_name = &self.functions[new_fn_idx].name;
            debug_println!("[DEBUG set_functions] Проверяем функцию '{}' с индексом {} (только что добавлена, compiler_idx: {})", 
                function_name, new_fn_idx, new_fn_idx - existing_functions_count);
            
            // Старый индекс функции в компиляторе равен (new_fn_idx - existing_functions_count)
            let compiler_fn_idx = new_fn_idx - existing_functions_count;
            
            // Ищем эту функцию в globals
            // Сначала проверяем, есть ли она в chunk's global_names
            let mut found_in_chunk = false;
            for function in &self.functions[start_fn_idx..] {
                if let Some(global_idx) = function.chunk.global_names.iter()
                    .find(|(_, name)| *name == function_name)
                    .map(|(idx, _)| *idx) {
                    // Нашли в chunk's global_names, используем mapped index
                    let real_global_idx = global_idx;
                    // Слот встроенного натива: пользовательская функция с тем же именем перекрывает встроенный
                    if real_global_idx < globals::BUILTIN_GLOBAL_NAMES.len() {
                        if let Some(canonical) = globals::builtin_global_name(real_global_idx) {
                            if canonical == function_name.as_str() {
                                self.globals[real_global_idx] = GlobalSlot::Heap(self.value_store.allocate_arena(ValueCell::Function(new_fn_idx)));
                                debug_println!("[DEBUG set_functions] Перезапись встроенного натива '{}' в слоте {} пользовательской функцией (второй цикл)", function_name, real_global_idx);
                                found_in_chunk = true;
                                break;
                            }
                        }
                    }
                    if real_global_idx < self.globals.len() {
                        let rid = self.globals[real_global_idx].resolve_to_value_id(&mut self.value_store);
                        let old_opt = self.value_store.get(rid)
                            .and_then(|c| if let ValueCell::Function(i) = c { Some(*i) } else { None });
                        if let Some(old_fn_idx) = old_opt {
                            if old_fn_idx != new_fn_idx {
                                debug_println!("[DEBUG set_functions] Найдена функция '{}' в globals[{}] через chunk global_names: {} -> {}", 
                                    function_name, real_global_idx, old_fn_idx, new_fn_idx);
                                self.globals[real_global_idx] = GlobalSlot::Heap(self.value_store.allocate_arena(ValueCell::Function(new_fn_idx)));
                                found_in_chunk = true;
                                break;
                            }
                        }
                    }
                }
            }
            
            if !found_in_chunk {
                debug_println!("[DEBUG set_functions] Ищем функцию '{}' во всех globals (compiler_idx: {})", function_name, compiler_fn_idx);
                for global_idx in 0..self.globals.len() {
                    let gid = self.globals[global_idx].resolve_to_value_id(&mut self.value_store);
                    let old_fn_idx_opt = self.value_store.get(gid)
                        .and_then(|c| if let ValueCell::Function(i) = c { Some(*i) } else { None });
                    if let Some(old_fn_idx) = old_fn_idx_opt {
                        // Проверяем, не обработали ли мы уже эту функцию через global_names
                        let already_processed = self.global_names.get(&global_idx)
                            .map(|name| name == function_name)
                            .unwrap_or(false);
                        
                        if !already_processed && old_fn_idx == compiler_fn_idx {
                            debug_println!("[DEBUG set_functions] Найден кандидат для '{}' в globals[{}] с old_fn_idx={}, compiler_idx={}", 
                                function_name, global_idx, old_fn_idx, compiler_fn_idx);
                        }
                        
                        if !already_processed {
                            // Проверяем три случая:
                            // 1. Старый индекс соответствует индексу функции в компиляторе
                            //    И функция с индексом old_fn_idx имеет другое имя (была перезаписана)
                            // 2. Старая функция с индексом old_fn_idx имеет то же имя, что и наша функция
                            let should_update = if old_fn_idx == compiler_fn_idx {
                                if old_fn_idx < self.functions.len() {
                                    let old_fn_name = &self.functions[old_fn_idx].name;
                                    let was_overwritten = old_fn_name != function_name;
                                    debug_println!("[DEBUG set_functions] Проверка для '{}': old_fn_idx={}, old_fn_name='{}', compiler_idx={}, was_overwritten={}", 
                                        function_name, old_fn_idx, old_fn_name, compiler_fn_idx, was_overwritten);
                                    was_overwritten
                                } else {
                                    true
                                }
                            } else if old_fn_idx < self.functions.len() {
                                self.functions[old_fn_idx].name == *function_name && old_fn_idx != new_fn_idx
                            } else {
                                false
                            };
                            
                            if should_update {
                                debug_println!("[DEBUG set_functions] Найдена функция '{}' в globals[{}] с индексом {} -> {} (compiler_idx: {}, old_fn_name: '{}')", 
                                    function_name, global_idx, old_fn_idx, new_fn_idx, compiler_fn_idx,
                                    if old_fn_idx < self.functions.len() { &self.functions[old_fn_idx].name } else { "OUT_OF_BOUNDS" });
                                self.globals[global_idx] = GlobalSlot::Heap(self.value_store.allocate_arena(ValueCell::Function(new_fn_idx)));
                                break;
                            }
                        }
                    }
                }
            }
        }
        
        for (global_idx, name) in self.global_names.clone().iter() {
            if Some(*global_idx) == self.argv_slot_index {
                continue;
            }
            if *global_idx < self.globals.len() {
                let gid = self.globals[*global_idx].resolve_to_value_id(&mut self.value_store);
                let old_fn_idx_opt = self.value_store.get(gid)
                    .and_then(|c| if let ValueCell::Function(i) = c { Some(*i) } else { None });
                if let Some(old_fn_idx) = old_fn_idx_opt {
                    debug_println!("[DEBUG set_functions] Найдена функция '{}' в globals[{}] с индексом функции {}", name, global_idx, old_fn_idx);
                    let old_fn_name_matches = if old_fn_idx < self.functions.len() {
                        let matches = self.functions[old_fn_idx].name == *name;
                        debug_println!("[DEBUG set_functions] Старый индекс функции {}: имя='{}', совпадает с глобальной переменной '{}': {}", 
                            old_fn_idx, self.functions[old_fn_idx].name, name, matches);
                        matches
                    } else {
                        debug_println!("[DEBUG set_functions] Старый индекс функции {} выходит за границы (всего функций: {})", 
                            old_fn_idx, self.functions.len());
                        false
                    };
                    if let Some(new_fn_idx) = self.find_function_index_by_name(name) {
                        debug_println!("[DEBUG set_functions] Найдена функция '{}' в VM с индексом {}", name, new_fn_idx);
                        if new_fn_idx < self.functions.len() && self.functions[new_fn_idx].name == *name {
                            if old_fn_idx != new_fn_idx {
                                debug_println!("[DEBUG set_functions] Обновляем индекс функции для '{}' в globals[{}]: {} -> {} (старое имя совпадает: {})", 
                                    name, global_idx, old_fn_idx, new_fn_idx, old_fn_name_matches);
                                self.globals[*global_idx] = GlobalSlot::Heap(self.value_store.allocate_arena(ValueCell::Function(new_fn_idx)));
                            } else {
                                debug_println!("[DEBUG set_functions] Индекс функции для '{}' уже правильный: {}", name, new_fn_idx);
                            }
                        } else {
                            debug_println!("[DEBUG set_functions] WARNING: Имя функции в новом индексе {} не совпадает с именем глобальной переменной '{}' (имя функции: '{}')", 
                                new_fn_idx, name, 
                                if new_fn_idx < self.functions.len() { &self.functions[new_fn_idx].name } else { "OUT OF BOUNDS" });
                        }
                    } else {
                        debug_println!("[DEBUG set_functions] WARNING: Функция '{}' не найдена в списке функций VM (старый индекс: {}, старое имя совпадает: {})", 
                            name, old_fn_idx, old_fn_name_matches);
                    }
                }
            }
        }
        debug_println!("[DEBUG set_functions] Завершено обновление индексов функций в globals");
        
        // Не перезаписываем слот argv (там лежит массив аргументов скрипта).
        let argv_slot = self.argv_slot_index;
        // В конце явно устанавливаем слот __main__ (точка входа из главного chunk), чтобы его не перезаписали другие циклы
        if let Some((&global_idx, _)) = self.global_names.iter().find(|(_, n)| *n == "__main__") {
            if let Some(fn_idx) = self.find_function_index_by_name("__main__") {
                if global_idx < self.globals.len() && Some(global_idx) != argv_slot {
                    self.globals[global_idx] = GlobalSlot::Heap(self.value_store.allocate_arena(ValueCell::Function(fn_idx)));
                    debug_println!("[DEBUG set_functions] Установлен слот __main__ в globals[{}] = Value::Function({})", global_idx, fn_idx);
                }
            }
        }
        // Явно устанавливаем слот main (вызывается из __main__), иначе при merge/порядке загрузки слот может остаться null
        if let Some((&global_idx, _)) = self.global_names.iter().find(|(_, n)| *n == "main") {
            if let Some(fn_idx) = self.find_function_index_by_name("main") {
                if global_idx < self.globals.len() && Some(global_idx) != argv_slot {
                    self.globals[global_idx] = GlobalSlot::Heap(self.value_store.allocate_arena(ValueCell::Function(fn_idx)));
                    debug_println!("[DEBUG set_functions] Установлен слот main в globals[{}] = Value::Function({})", global_idx, fn_idx);
                }
            }
        }

        for (idx, name) in &self.global_names {
            if name.contains("::new_") {
                if let Some(slot) = self.globals.get_mut(*idx) {
                    let id = slot.resolve_to_value_id(&mut self.value_store);
                    if let Some(ValueCell::Function(fn_idx)) = self.value_store.get(id) {
                        if *fn_idx < self.functions.len() {
                            let func = &self.functions[*fn_idx];
                            debug_println!("[DEBUG set_functions] Проверка: конструктор '{}' в globals[{}] имеет индекс функции {}, имя функции: '{}', arity: {}", 
                                name, idx, fn_idx, func.name, func.arity);
                        } else {
                            debug_println!("[DEBUG set_functions] ОШИБКА: конструктор '{}' в globals[{}] имеет индекс функции {} (выходит за границы, всего функций: {})", 
                                name, idx, fn_idx, self.functions.len());
                        }
                    }
                }
            }
        }
        
        // Добавляем explicit_global_names
        for (idx, name) in explicit_global_names_to_add {
            self.explicit_global_names.insert(idx, name);
        }
    }

    /// Добавляет функции к существующим функциям в VM
    /// Возвращает начальный индекс добавленных функций
    pub fn add_functions(&mut self, functions: Vec<crate::bytecode::Function>) -> usize {
        let start_index = self.functions.len();
        self.functions.extend(functions);
        // Обновляем имена глобальных переменных из новых функций
        for function in self.functions.iter().skip(start_index) {
            for (idx, name) in &function.chunk.global_names {
                self.global_names.insert(*idx, name.clone());
            }
            for (idx, name) in &function.chunk.explicit_global_names {
                self.explicit_global_names.insert(*idx, name.clone());
            }
        }
        start_index
    }

    /// Adds functions to the VM without merging their global_names (for module isolation: __lib__).
    /// Returns the start index so caller can remap Value::Function in exported namespace.
    pub fn add_functions_only(&mut self, functions: Vec<crate::bytecode::Function>) -> usize {
        let start_index = self.functions.len();
        self.functions.extend(functions);
        start_index
    }

    /// Adds functions from a loaded module and sets their module_name so LoadGlobal/StoreGlobal
    /// resolve from that module's namespace. Returns the start index.
    pub fn add_functions_from_module(
        &mut self,
        mut functions: Vec<crate::bytecode::Function>,
        module_name: String,
    ) -> usize {
        let start_index = self.functions.len();
        for f in &mut functions {
            f.module_name = Some(module_name.clone());
        }
        self.functions.extend(functions);
        start_index
    }
    
    /// Получает количество функций в VM
    pub fn functions_count(&self) -> usize {
        self.functions.len()
    }
    
    /// Получает функции VM (для использования при импорте модулей)
    pub fn get_functions(&self) -> &Vec<crate::bytecode::Function> {
        &self.functions
    }

    /// Мутабельный доступ к функциям VM (для обновления чанков после merge модуля)
    pub fn get_functions_mut(&mut self) -> &mut Vec<crate::bytecode::Function> {
        &mut self.functions
    }

    pub fn register_native_globals(&mut self) {
        globals::register_native_globals(&mut self.globals, &mut self.global_names, &mut self.value_store);
        self.builtins.resize(BUILTIN_END, global_slot::default_global_slot());
        for i in 0..BUILTIN_END.min(self.globals.len()) {
            self.builtins[i] = self.globals[i];
        }
    }

    /// Register all built-in modules (ml, plot, settings_env) so native indices are consistent
    /// across all VMs (main and sub-VMs used for module loading).
    pub fn register_all_builtin_modules(&mut self) -> Result<(), LangError> {
        use crate::vm::modules;
        modules::register_module("ml", &mut self.natives, &mut self.globals, &mut self.global_names, &mut self.value_store, &mut self.heavy_store)?;
        modules::register_module("plot", &mut self.natives, &mut self.globals, &mut self.global_names, &mut self.value_store, &mut self.heavy_store)?;
        modules::register_module("settings_env", &mut self.natives, &mut self.globals, &mut self.global_names, &mut self.value_store, &mut self.heavy_store)?;
        modules::register_module("uuid", &mut self.natives, &mut self.globals, &mut self.global_names, &mut self.value_store, &mut self.heavy_store)?;
        modules::register_module("database", &mut self.natives, &mut self.globals, &mut self.global_names, &mut self.value_store, &mut self.heavy_store)?;
        Ok(())
    }

    // Exception handling methods moved to exceptions.rs

    /// argv_patch: when Some((argv_slot_index, old_indices, argv_value_id)), the chunk is cloned and any LoadGlobal(old_idx)
    /// in the clone is replaced with LoadGlobal(argv_slot_index). If argv_value_id is Some(id), that value is written
    /// to globals[argv_slot_index] immediately before the execution loop so the argv slot is never overwritten by earlier code.
    pub fn run(&mut self, chunk: &Chunk, argv_patch: Option<(usize, &[usize], Option<ValueId>)>) -> Result<Value, LangError> {
        // Set RunContext (base_path, executing_lib, dpm_package_paths, smb_manager) so file_import and file_ops use one source
        let run_ctx = crate::vm::run_context::RunContext {
            base_path: self.base_path.clone(),
            executing_lib: false,
            dpm_package_paths: crate::vm::file_import::get_dpm_package_paths(),
            smb_manager: crate::vm::file_ops::get_smb_manager(),
        };
        crate::vm::run_context::RunContext::set_current(run_ctx);
        let _run_guard = RunContextGuard(&mut self.base_path as *mut Option<PathBuf>);
        // Keep file_import thread_locals in sync for code that sets them outside run()
        crate::vm::file_import::set_base_path(self.base_path.clone());

        // Set ML context thread-local so ML natives use VM-owned pool/cache without global Mutex.
        // If this VM has no ML context, clear any stale one from a previous run (e.g. from ml_api_tests) so settings_env/other tests don't see it.
        if let Some(ctx) = self.ml_context.take() {
            crate::ml::MlContext::set_current(ctx);
        } else {
            let _ = crate::ml::MlContext::take_current();
        }
        let _ml_guard = MlContextGuard(&mut self.ml_context as *mut _);

        // Set Plot context thread-local so plot natives use VM-owned state without global RefCells
        let plot_ctx = self.plot_context.take().unwrap_or_else(crate::plot::PlotContext::new);
        crate::plot::PlotContext::set_current(plot_ctx);
        let _plot_guard = PlotContextGuard(&mut self.plot_context as *mut _);

        // Заполняем имена глобальных переменных из chunk
        globals::merge_global_names(
            &mut self.global_names,
            &mut self.explicit_global_names,
            &chunk.global_names,
            &chunk.explicit_global_names,
        );

        // Settings subclasses load model_config from __constructing_class__ (set by executor before each constructor call).
        const CONSTRUCTING_CLASS_NAME: &str = "__constructing_class__";
        if !self.global_names.values().any(|n| n == CONSTRUCTING_CLASS_NAME) {
            let idx = self.globals.len();
            self.globals.push(global_slot::default_global_slot());
            self.global_names.insert(idx, CONSTRUCTING_CLASS_NAME.to_string());
        }

        #[cfg(feature = "profile")]
        crate::vm::profile::set();

        // Клонируем chunk и при необходимости патчим LoadGlobal(argv) на argv_slot_index,
        // чтобы после merge_global_names и возможного ImportFrom слот по старому индексу не оказался функцией (load_settings и т.д.).
        // Важно: заменяем только те LoadGlobal(old_idx), у которых в chunk по этому индексу имя "argv":
        // после set_functions в главном chunk индекс 76 может означать __main__ (кали), и замена всех 76→82 даёт "Can only call functions, got: Array".
        let mut chunk_to_run = chunk.clone();
        if let Some((argv_slot_index, old_indices, _)) = argv_patch.as_ref() {
            for &old_idx in *old_indices {
                if chunk_to_run.global_names.get(&old_idx).map(|n| n.as_str()) != Some("argv") {
                    continue;
                }
                for opcode in &mut chunk_to_run.code {
                    if let crate::bytecode::OpCode::LoadGlobal(idx) = opcode {
                        if *idx == old_idx {
                            *opcode = crate::bytecode::OpCode::LoadGlobal(*argv_slot_index);
                        }
                    }
                }
            }
        }

        // Создаем начальный frame (constants loaded into store here)
        let function = crate::bytecode::Function::new("<main>".to_string(), 0);
        let mut function = function;
        function.chunk = chunk_to_run;
        let frame = CallFrame::new(function, 0, &mut self.value_store, &mut self.heavy_store);
        self.frames.push(frame);

        // Записываем argv в слот в самый последний момент, чтобы никакой код (set_functions и т.д.) не мог перезаписать слот функцией.
        if let Some((argv_slot_index, _old_indices, argv_value_id)) = argv_patch {
            if let Some(argv_id) = argv_value_id {
                if argv_slot_index >= self.globals.len() {
                    self.globals
                        .resize(argv_slot_index + 1, global_slot::default_global_slot());
                }
                self.globals[argv_slot_index] = GlobalSlot::Heap(argv_id);
            }
        }

        loop {
            match self.step()? {
                VMStatus::Continue => {}
                VMStatus::Return(id) => {
                    #[cfg(feature = "profile")]
                    if let Some(stats) = crate::vm::profile::take() {
                        crate::vm::profile::print_stats(&stats);
                    }
                    return Ok(load_value(id, &self.value_store, &self.heavy_store));
                }
                VMStatus::FrameEnded => break,
            }
        }

        #[cfg(feature = "profile")]
        if let Some(stats) = crate::vm::profile::take() {
            crate::vm::profile::print_stats(&stats);
        }

        if !self.stack.is_empty() {
            let tv = self.stack.pop().unwrap();
            let id = tagged_to_value_id(tv, &mut self.value_store);
            Ok(load_value(id, &self.value_store, &self.heavy_store))
        } else {
            Ok(load_value(NULL_VALUE_ID, &self.value_store, &self.heavy_store))
        }
    }

    /// Выполнить один шаг VM - получить следующую инструкцию и выполнить её
    fn step(&mut self) -> Result<VMStatus, LangError> {
        let vm_ptr = self as *mut Vm;
        let (instruction, line) = {
            match executor::step(&mut self.frames)? {
                Some((inst, ln)) => (inst, ln),
                None => return Ok(VMStatus::FrameEnded),
            }
        };
        executor::execute_instruction(
            instruction,
            line,
            &mut self.stack,
            &mut self.frames,
            &mut self.globals,
            &mut self.global_names,
            &self.explicit_global_names,
            &mut self.functions,
            &mut self.natives,
            &mut self.exception_handlers,
            &mut self.error_type_table,
            &mut self.explicit_relations,
            &mut self.explicit_primary_keys,
            &mut self.loaded_modules,
            &mut self.abi_natives,
            &mut self.loaded_native_libraries,
            &mut self.value_store,
            &mut self.heavy_store,
            &mut self.native_args_buffer,
            &mut self.reusable_native_arg_ids,
            &mut self.reusable_all_popped,
            vm_ptr,
        )
    }

    // execute_instruction moved to executor.rs
    // Stack operations moved to stack.rs
    // Binary and unary operations moved to operations.rs

    // Module registration methods moved to modules.rs

    /// Builtins (indices 0..BUILTIN_END). Shared by all modules.
    pub fn get_builtins(&self) -> &[GlobalSlot] {
        &self.builtins[..]
    }
    pub fn get_builtins_mut(&mut self) -> &mut Vec<GlobalSlot> {
        &mut self.builtins
    }

    /// Module cache: name/path -> ModuleObject. Used for import and isolated module globals.
    pub fn get_modules_mut(&self) -> std::cell::RefMut<'_, HashMap<String, Rc<RefCell<ModuleObject>>>> {
        self.modules.borrow_mut()
    }
    pub fn get_modules(&self) -> std::cell::Ref<HashMap<String, Rc<RefCell<ModuleObject>>>> {
        self.modules.borrow()
    }

    /// Получить доступ к глобальным переменным (GlobalSlot; use resolve_to_value_id or store_convert::slot_to_value for Value)
    /// Combines builtins (0..BUILTIN_END) and module globals (BUILTIN_END+). Legacy: prefer per-module lookup.
    pub fn get_globals(&self) -> &[GlobalSlot] {
        &self.globals[..]
    }

    /// Получить мутабельный доступ к глобальным переменным
    pub fn get_globals_mut(&mut self) -> &mut Vec<GlobalSlot> {
        &mut self.globals
    }

    /// Resolve global slot at index to ValueId (Inline → materialize once and cache as Heap).
    pub fn resolve_global_to_value_id(&mut self, index: usize) -> ValueId {
        if index < self.globals.len() {
            self.globals[index].resolve_to_value_id(&mut self.value_store)
        } else {
            NULL_VALUE_ID
        }
    }

    /// ValueStore and HeavyStore for materializing Value from ValueId (e.g. at native boundaries)
    pub fn value_store(&self) -> &ValueStore {
        &self.value_store
    }
    pub fn heavy_store(&self) -> &HeavyStore {
        &self.heavy_store
    }
    pub fn value_store_mut(&mut self) -> &mut ValueStore {
        &mut self.value_store
    }
    pub fn heavy_store_mut(&mut self) -> &mut HeavyStore {
        &mut self.heavy_store
    }
    /// Call a function with both stores mutably (avoids double mutable borrow).
    pub fn with_stores_mut<F, R>(&mut self, f: F) -> R
    where
        F: FnOnce(&mut ValueStore, &mut HeavyStore) -> R,
    {
        f(&mut self.value_store, &mut self.heavy_store)
    }

    /// Take pending relations pushed by relate() native (VM-owned storage; replaces thread-local take_relations).
    pub fn take_pending_relations(&mut self) -> Vec<(Rc<RefCell<Table>>, String, Rc<RefCell<Table>>, String)> {
        std::mem::take(&mut self.pending_relations)
    }

    /// Take pending primary keys pushed by primary_key() native (VM-owned storage; replaces thread-local take_primary_keys).
    pub fn take_pending_primary_keys(&mut self) -> Vec<(Rc<RefCell<Table>>, String)> {
        std::mem::take(&mut self.pending_primary_keys)
    }

    /// Reset value_store and heavy_store and re-establish only function references in globals.
    /// Use when reusing the same VM for stateless runs (e.g. HTTP request handlers). Non-function
    /// globals (config, tables, etc.) are dropped; global state is not preserved between calls.
    pub fn reset_stores_and_globals_for_stateless(&mut self) {
        let mut function_globals: Vec<(usize, usize)> = Vec::new();
        for (idx, slot) in self.globals.iter_mut().enumerate() {
            let id = slot.resolve_to_value_id(&mut self.value_store);
            if let Some(ValueCell::Function(fn_idx)) = self.value_store.get(id) {
                function_globals.push((idx, *fn_idx));
            }
        }
        self.value_store.clear();
        self.heavy_store.clear();
        for i in 0..self.globals.len() {
            self.globals[i] = GlobalSlot::null();
        }
        for (global_idx, fn_idx) in function_globals {
            if global_idx < self.globals.len() {
                self.globals[global_idx] = GlobalSlot::Heap(self.value_store.allocate_arena(ValueCell::Function(fn_idx)));
            }
        }
    }

    /// Количество встроенных нативов (natives.len()). Индексы >= этого — ABI-нативы.
    pub fn builtin_natives_count(&self) -> usize {
        self.natives.len()
    }

    /// Срез нативных функций (для merge: копирование нативов модуля в main VM).
    pub fn get_natives(&self) -> &[NativeFn] {
        &self.natives[..]
    }

    /// ABI-нативы (из загруженных .so/.dylib). Индекс в Value::NativeFunction = builtin_natives_count + индекс в этом векторе.
    pub fn get_abi_natives(&self) -> &[NativeAbiFn] {
        &self.abi_natives
    }

    /// Зарезервировано для будущего API (другие крейты, тесты).
    #[allow(dead_code)]
    pub(crate) fn get_abi_natives_mut(&mut self) -> &mut Vec<NativeAbiFn> {
        &mut self.abi_natives
    }

    /// Зарезервировано для будущего API (другие крейты, тесты).
    #[allow(dead_code)]
    pub(crate) fn get_loaded_native_libraries_mut(&mut self) -> &mut Vec<Library> {
        &mut self.loaded_native_libraries
    }

    /// Получить доступ к именам глобальных переменных
    pub fn get_global_names(&self) -> &std::collections::BTreeMap<usize, String> {
        &self.global_names
    }

    /// Получить мутабельный доступ к именам глобальных переменных
    pub fn get_global_names_mut(&mut self) -> &mut std::collections::BTreeMap<usize, String> {
        &mut self.global_names
    }

    /// Получить доступ к именам переменных, явно объявленных с ключевым словом 'global'
    pub fn get_explicit_global_names(&self) -> &std::collections::BTreeMap<usize, String> {
        &self.explicit_global_names
    }

    /// Legacy: merges another VM's globals into this VM. Not used with module isolation (__lib__ is registered as a module instead).
    #[allow(dead_code)]
    pub fn merge_globals_from(&mut self, other: &Vm) {
        const BUILTIN_COUNT: usize = 75;
        // Чтобы NativeFunction(i) из модуля (i >= BUILTIN_COUNT) был валиден в self, добавляем нативы модуля.
        let other_natives = other.get_natives();
        if other_natives.len() > BUILTIN_COUNT {
            self.natives.extend_from_slice(&other_natives[BUILTIN_COUNT..]);
            debug_println!("[DEBUG merge_globals_from] Добавлено нативов из модуля: {}", other_natives.len() - BUILTIN_COUNT);
        }

        let other_globals = other.get_globals();
        let other_global_names = other.get_global_names();
        
        debug_println!("[DEBUG merge_globals_from] Объединяем глобальные переменные из другого VM");
        debug_println!("[DEBUG merge_globals_from] Функций в текущем VM: {}", self.functions.len());
        debug_println!("[DEBUG merge_globals_from] Функций в другом VM: {}", other.functions.len());
        debug_println!("[DEBUG merge_globals_from] Глобальных переменных в другом VM: {}", other_global_names.len());
        
        // Также объединяем функции из другого VM
        // ВАЖНО: Если функции из модулей уже были добавлены в VM во время выполнения,
        // start_function_index должен учитывать эти функции. Но merge происходит ДО выполнения кода,
        // поэтому функции из модулей еще не добавлены. Индексы функций будут обновлены позже при импорте модулей.
        let start_function_index = self.functions.len();
        self.functions.extend(other.functions.iter().cloned());
        debug_println!("[DEBUG merge_globals_from] Начальный индекс функций: {} (функций в текущем VM: {})", start_function_index, self.functions.len() - other.functions.len());
        debug_println!("[DEBUG merge_globals_from] Всего функций после объединения: {}", self.functions.len());
        
        // Итерируем в детерминированном порядке (по имени, затем по индексу). При дубликатах имени
        // предпочитаем значение-класс (Value::Object с __class_name), иначе — значение с большим индексом.
        let mut pairs: Vec<_> = other_global_names
            .iter()
            .map(|(i, n)| (*i, n.clone()))
            .collect();
        pairs.sort_by(|a, b| a.1.cmp(&b.1).then_with(|| a.0.cmp(&b.0)));

        let mut by_name: std::collections::HashMap<String, (usize, Value)> = std::collections::HashMap::new();
        for (index, name) in &pairs {
            if name == "argv" {
                continue;
            }
            if let Some(slot) = other_globals.get(*index) {
                let value = match slot {
                    GlobalSlot::Inline(tv) => crate::vm::store_convert::slot_to_value(*tv, other.value_store(), other.heavy_store()),
                    GlobalSlot::Heap(id) => load_value(*id, other.value_store(), other.heavy_store()),
                };
                let prefer = match by_name.get(name) {
                    Some((prev_index, existing)) => {
                        let existing_is_obj = matches!(existing, Value::Object(_));
                        let value_is_obj = matches!(&value, Value::Object(_));
                        if value_is_obj && !existing_is_obj {
                            true
                        } else if !value_is_obj && existing_is_obj {
                            false
                        } else {
                            *index >= *prev_index
                        }
                    }
                    None => true,
                };
                if prefer {
                    by_name.insert(name.clone(), (*index, value));
                }
            }
        }

        let mut names_sorted: Vec<_> = by_name.keys().cloned().collect();
        names_sorted.sort();
        for name in names_sorted {
            let (index, value) = by_name.get(&name).unwrap().clone();
            debug_println!("[DEBUG merge_globals_from] Объединяем '{}' (index: {})", name, index);
            let value_to_store = match value {
                    Value::Function(function_index) => {
                        // Обновляем индекс функции с учетом уже добавленных функций
                        let new_function_index = start_function_index + function_index;
                        debug_println!("[DEBUG merge_globals_from] Обновляем индекс функции для '{}': {} -> {}", name, function_index, new_function_index);
                        Value::Function(new_function_index)
                    }
                    Value::Object(obj_rc) => {
                        // Для объектов (классов) нужно обновить индексы функций и нативов внутри объекта
                        use std::rc::Rc;
                        use std::cell::RefCell;
                        let obj = obj_rc.borrow();
                        let mut new_obj = HashMap::new();
                        let other_natives = other.get_natives();
                        for (key, val) in obj.iter() {
                            let updated_val = match val {
                                Value::Function(function_index) => {
                                    Value::Function(start_function_index + *function_index)
                                }
                                Value::NativeFunction(i) => {
                                    if *i >= other_natives.len() {
                                        val.clone()
                                    } else {
                                        let fn_ptr = other_natives[*i] as *const ();
                                        let remapped = if *i < self.natives.len()
                                            && std::ptr::eq(self.natives[*i] as *const (), fn_ptr)
                                        {
                                            *i
                                        } else {
                                            self.natives[BUILTIN_COUNT..]
                                                .iter()
                                                .position(|&f| std::ptr::eq(f as *const (), fn_ptr))
                                                .map(|pos| BUILTIN_COUNT + pos)
                                                .unwrap_or_else(|| {
                                                    self.natives.push(other_natives[*i]);
                                                    self.natives.len() - 1
                                                })
                                        };
                                        Value::NativeFunction(remapped)
                                    }
                                }
                                _ => val.clone()
                            };
                            new_obj.insert(key.clone(), updated_val);
                        }
                        Value::Object(Rc::new(RefCell::new(new_obj)))
                    }
                    Value::NativeFunction(i) if i >= BUILTIN_COUNT => {
                        let other_natives = other.get_natives();
                        let idx = i;
                        if idx >= other_natives.len() {
                            value.clone()
                        } else {
                            let fn_ptr = other_natives[idx] as *const ();
                            let remapped = if idx < self.natives.len()
                                && std::ptr::eq(self.natives[idx] as *const (), fn_ptr)
                            {
                                idx
                            } else {
                                self.natives[BUILTIN_COUNT..]
                                    .iter()
                                    .position(|&f| std::ptr::eq(f as *const (), fn_ptr))
                                    .map(|pos| BUILTIN_COUNT + pos)
                                    .unwrap_or_else(|| {
                                        self.natives.push(other_natives[idx]);
                                        self.natives.len() - 1
                                    })
                            };
                            Value::NativeFunction(remapped)
                        }
                    }
                    _ => value.clone()
                };
                
                // Проверяем, существует ли уже такая переменная. Берём минимальный индекс при дубликатах,
                // чтобы всегда перезаписывать один и тот же слот (Config в 71, а не создавать второй в 78).
                let existing_indices: Vec<usize> = self.global_names.iter()
                    .filter(|(_, n)| n.as_str() == name.as_str())
                    .map(|(idx, _)| *idx)
                    .collect();
                if let Some(&existing_index) = existing_indices.iter().min() {
                    // Не перезаписывать слоты 0..71 (канонические нативы) значениями из модуля.
                    if existing_index < BUILTIN_COUNT {
                        debug_println!("[DEBUG merge_globals_from] Пропуск перезаписи '{}' (слот {} — канонический натив)", name, existing_index);
                        continue;
                    }
                    // Не перезаписывать слот, если текущее значение — нативная функция (например Config из settings_env).
                    // Иначе при "from config import Settings" класс Config из config.dc затёр бы Config из settings_env в dev_config.
                    if existing_index < self.globals.len() {
                        let existing_slot = &self.globals[existing_index];
                        let existing_val = match existing_slot {
                            GlobalSlot::Inline(tv) => crate::vm::store_convert::slot_to_value(*tv, &self.value_store, &self.heavy_store),
                            GlobalSlot::Heap(id) => load_value(*id, &self.value_store, &self.heavy_store),
                        };
                        if matches!(existing_val, Value::NativeFunction(_)) {
                            debug_println!("[DEBUG merge_globals_from] Пропуск перезаписи '{}' (текущее значение — нативная функция)", name);
                            continue;
                        }
                    }
                    // Не перезаписывать слот main встроенным модулем из other (ml, plot, settings_env, uuid):
                    // иначе затрутся экспорты модуля (Config, DatabaseConfig и т.д.), занявшие те же индексы.
                    if crate::vm::modules::is_known_module(name.as_str()) {
                        debug_println!("[DEBUG merge_globals_from] Пропуск перезаписи '{}' (встроенный модуль)", name);
                        continue;
                    }
                    if let Value::NativeFunction(i) = &value_to_store {
                        if *i < BUILTIN_COUNT {
                            if let Some(canonical) = globals::builtin_global_name(*i) {
                                if canonical != name.as_str() {
                                    debug_println!("[DEBUG merge_globals_from] Пропуск перезаписи '{}' на встроенный {:?} (индекс {})", name, canonical, i);
                                    continue;
                                }
                            }
                        }
                    }
                    // Перезаписываем существующую переменную
                    let val_type = match &value_to_store {
                        Value::Object(_) => "Object",
                        Value::Function(_) => "Function",
                        _ => "Other",
                    };
                    debug_println!("[DEBUG merge_globals_from] '{}' уже существует, перезаписываем globals[{}] ({})", name, existing_index, val_type);
                    if name == "Config" || name == "DatabaseConfig" {
                        debug_println!("[DEBUG merge_globals_from] Config/DatabaseConfig: '{}' -> слот {} ({})", name, existing_index, val_type);
                    }
                    let id = store_value_arena(value_to_store.clone(), &mut self.value_store, &mut self.heavy_store);
                    if existing_index < self.globals.len() {
                        self.globals[existing_index] = GlobalSlot::Heap(id);
                    } else {
                        self.globals.resize(existing_index + 1, global_slot::default_global_slot());
                        self.globals[existing_index] = GlobalSlot::Heap(id);
                    }
                } else {
                    if let Value::NativeFunction(i) = &value_to_store {
                        if *i < BUILTIN_COUNT {
                            if let Some(canonical) = globals::builtin_global_name(*i) {
                                if canonical != name.as_str() {
                                    debug_println!("[DEBUG merge_globals_from] Пропуск создания глобальной '{}' с встроенным {:?} (индекс {})", name, canonical, i);
                                    continue;
                                }
                            }
                        }
                    }
                    let new_index = self.globals.len();
                    let val_type = match &value_to_store {
                        Value::Object(_) => "Object",
                        Value::Function(_) => "Function",
                        _ => "Other",
                    };
                    self.globals.push(GlobalSlot::Heap(store_value_arena(value_to_store, &mut self.value_store, &mut self.heavy_store)));
                    self.global_names.insert(new_index, name.clone());
                    debug_println!("[DEBUG merge_globals_from] Создана новая глобальная переменная '{}' в globals[{}] ({})", name, new_index, val_type);
                    if name == "Config" || name == "DatabaseConfig" {
                        debug_println!("[DEBUG merge_globals_from] Config/DatabaseConfig: '{}' -> новый слот {} ({})", name, new_index, val_type);
                    }
                }
        }
        debug_println!("[DEBUG merge_globals_from] Объединение завершено. Всего глобальных переменных: {}", self.global_names.len());
    }

    /// After merging a module into the caller, re-establish __main__ and main in the caller's globals
    /// so they are not overwritten by the merge (caller's entry point slots).
    pub fn ensure_entry_point_slots(
        target_globals: &mut [GlobalSlot],
        target_global_names: &std::collections::BTreeMap<usize, String>,
        target_functions: &[crate::bytecode::Function],
        store: &mut ValueStore,
    ) {
        for name in ["__main__", "main"] {
            if let Some(&slot) = target_global_names.iter().find(|(_, n)| n.as_str() == name).map(|(idx, _)| idx) {
                if let Some(fn_idx) = target_functions.iter().position(|f| f.name == name) {
                    if slot < target_globals.len() {
                        target_globals[slot] = GlobalSlot::Heap(store.allocate_arena(ValueCell::Function(fn_idx)));
                        debug_println!("[DEBUG ensure_entry_point_slots] Установлен слот '{}' в globals[{}] = Function({})", name, slot, fn_idx);
                    }
                }
            }
        }
    }

    /// Legacy: merge module VM into caller's buffers. Not used with module isolation (ImportFrom only adds requested items).
    #[allow(dead_code)]
    pub fn merge_globals_from_into(
        other: &Vm,
        target_globals: &mut Vec<GlobalSlot>,
        target_global_names: &mut std::collections::BTreeMap<usize, String>,
        target_functions: &mut Vec<crate::bytecode::Function>,
        target_natives: &mut Vec<NativeFn>,
        store: &mut ValueStore,
        heap: &mut HeavyStore,
    ) {
        const BUILTIN_COUNT: usize = 75;
        let other_natives = other.get_natives();
        if other_natives.len() > BUILTIN_COUNT {
            target_natives.extend_from_slice(&other_natives[BUILTIN_COUNT..]);
            debug_println!("[DEBUG merge_globals_from_into] Добавлено нативов из модуля: {}", other_natives.len() - BUILTIN_COUNT);
        }
        let other_globals = other.get_globals();
        let other_global_names = other.get_global_names();
        debug_println!("[DEBUG merge_globals_from_into] Объединяем глобальные переменные из другого VM");
        let start_function_index = target_functions.len();
        target_functions.extend(other.get_functions().iter().cloned());
        debug_println!("[DEBUG merge_globals_from_into] start_function_index: {}, всего функций: {}", start_function_index, target_functions.len());

        let mut pairs: Vec<_> = other_global_names.iter().map(|(i, n)| (*i, n.clone())).collect();
        pairs.sort_by(|a, b| a.1.cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
        let mut by_name: std::collections::HashMap<String, (usize, Value)> = std::collections::HashMap::new();
        for (index, name) in &pairs {
            if name == "argv" {
                continue;
            }
            if let Some(slot) = other_globals.get(*index) {
                let value = match slot {
                    GlobalSlot::Inline(tv) => crate::vm::store_convert::slot_to_value(*tv, other.value_store(), other.heavy_store()),
                    GlobalSlot::Heap(id) => load_value(*id, other.value_store(), other.heavy_store()),
                };
                let prefer = match by_name.get(name) {
                    Some((prev_index, existing)) => {
                        let existing_is_null = matches!(existing, Value::Null);
                        let value_is_null = matches!(&value, Value::Null);
                        if value_is_null && !existing_is_null {
                            false
                        } else if !value_is_null && existing_is_null {
                            true
                        } else {
                            let existing_is_obj = matches!(existing, Value::Object(_));
                            let value_is_obj = matches!(&value, Value::Object(_));
                            if value_is_obj && !existing_is_obj {
                                true
                            } else if !value_is_obj && existing_is_obj {
                                false
                            } else {
                                *index >= *prev_index
                            }
                        }
                    }
                    None => true,
                };
                if prefer {
                    by_name.insert(name.clone(), (*index, value));
                }
            }
        }

        let mut names_sorted: Vec<_> = by_name.keys().cloned().collect();
        names_sorted.sort();
        for name in names_sorted {
            let (index, value) = by_name.get(&name).unwrap().clone();
            debug_println!("[DEBUG merge_globals_from_into] Объединяем '{}' (index: {})", name, index);
            let value_to_store = match value {
                Value::Function(function_index) => {
                    Value::Function(start_function_index + function_index)
                }
                Value::Object(obj_rc) => {
                    use std::rc::Rc;
                    use std::cell::RefCell;
                    let obj = obj_rc.borrow();
                    let mut new_obj = HashMap::new();
                    let other_natives_slice = other.get_natives();
                    for (key, val) in obj.iter() {
                        let updated_val = match val {
                            Value::Function(function_index) => Value::Function(start_function_index + *function_index),
                            Value::NativeFunction(i) => {
                                if *i >= other_natives_slice.len() {
                                    val.clone()
                                } else {
                                    let fn_ptr = other_natives_slice[*i] as *const ();
                                    let remapped = target_natives[BUILTIN_COUNT..]
                                        .iter()
                                        .position(|&f| std::ptr::eq(f as *const (), fn_ptr))
                                        .map(|pos| BUILTIN_COUNT + pos)
                                        .unwrap_or_else(|| {
                                            target_natives.push(other_natives_slice[*i]);
                                            target_natives.len() - 1
                                        });
                                    Value::NativeFunction(remapped)
                                }
                            }
                            _ => val.clone(),
                        };
                        new_obj.insert(key.clone(), updated_val);
                    }
                    Value::Object(Rc::new(RefCell::new(new_obj)))
                }
                Value::NativeFunction(i) if i >= BUILTIN_COUNT => {
                    let other_natives_slice = other.get_natives();
                    if i >= other_natives_slice.len() {
                        value.clone()
                    } else {
                        let fn_ptr = other_natives_slice[i] as *const ();
                        let remapped = target_natives[BUILTIN_COUNT..]
                            .iter()
                            .position(|&f| std::ptr::eq(f as *const (), fn_ptr))
                            .map(|pos| BUILTIN_COUNT + pos)
                            .unwrap_or_else(|| {
                                target_natives.push(other_natives_slice[i]);
                                target_natives.len() - 1
                            });
                        Value::NativeFunction(remapped)
                    }
                }
                _ => value.clone(),
            };

            let existing_indices: Vec<usize> = target_global_names
                .iter()
                .filter(|(_, n)| n.as_str() == name.as_str())
                .map(|(idx, _)| *idx)
                .collect();
            if let Some(&existing_index) = existing_indices.iter().min() {
                if existing_index < BUILTIN_COUNT {
                    continue;
                }
                if existing_index < target_globals.len() {
                    let existing_slot = &target_globals[existing_index];
                    let existing_val = match existing_slot {
                        GlobalSlot::Inline(tv) => crate::vm::store_convert::slot_to_value(*tv, store, heap),
                        GlobalSlot::Heap(id) => load_value(*id, store, heap),
                    };
                    if matches!(existing_val, Value::NativeFunction(_)) {
                        continue;
                    }
                }
                if crate::vm::modules::is_known_module(name.as_str()) {
                    continue;
                }
                if let Value::NativeFunction(i) = &value_to_store {
                    if *i < BUILTIN_COUNT {
                        if let Some(canonical) = globals::builtin_global_name(*i) {
                            if canonical != name.as_str() {
                                continue;
                            }
                        }
                    }
                }
                let id = store_value_arena(value_to_store.clone(), store, heap);
                if existing_index < target_globals.len() {
                    target_globals[existing_index] = GlobalSlot::Heap(id);
                } else {
                    target_globals.resize(existing_index + 1, global_slot::default_global_slot());
                    target_globals[existing_index] = GlobalSlot::Heap(id);
                }
                if name == "get_settings" || name == "load_settings" {
                    let fn_info = match &value_to_store {
                        Value::Function(fi) => format!("Function({})", fi),
                        _ => format!("{:?}", std::mem::discriminant(&value_to_store)),
                    };
                    debug_println!(
                        "[DEBUG merge_globals_from_into] merge '{}' -> caller slot {} value={}",
                        name, existing_index, fn_info
                    );
                }
            } else {
                // Canonicalization: do not create a new slot if this name already exists in target (any slot).
                // Duplicate name->slot entries cause wrong binding (e.g. get_settings at 74 and 78; bytecode may pick the wrong one).
                let any_existing: Option<usize> = target_global_names
                    .iter()
                    .find(|(_, n)| n.as_str() == name.as_str())
                    .map(|(i, _)| *i);
                if let Some(canonical_slot) = any_existing {
                    if canonical_slot >= BUILTIN_COUNT {
                        if canonical_slot >= target_globals.len() {
                            target_globals.resize(canonical_slot + 1, global_slot::default_global_slot());
                        }
                        let id = store_value_arena(value_to_store.clone(), store, heap);
                        target_globals[canonical_slot] = GlobalSlot::Heap(id);
                        if name == "get_settings" || name == "load_settings" {
                            let fn_info = match &value_to_store {
                                Value::Function(fi) => format!("Function({})", fi),
                                _ => "non-Fn".to_string(),
                            };
                            debug_println!(
                                "[DEBUG merge_globals_from_into] canonicalize '{}' -> existing slot {} value={}",
                                name, canonical_slot, fn_info
                            );
                        }
                    }
                    continue;
                }
                if let Value::NativeFunction(i) = &value_to_store {
                    if *i < BUILTIN_COUNT {
                        if let Some(canonical) = globals::builtin_global_name(*i) {
                            if canonical != name.as_str() {
                                continue;
                            }
                        }
                    }
                }
                let new_index = target_globals.len();
                target_globals.push(GlobalSlot::Heap(store_value_arena(value_to_store, store, heap)));
                target_global_names.insert(new_index, name.clone());
            }
        }
        debug_println!("[DEBUG merge_globals_from_into] Объединение завершено. Всего глобальных переменных: {}", target_global_names.len());
    }

    /// Merges exports from an already-executed module object into this VM's globals.
    /// Used when re-importing the same module (canonical path): no run(), just copy namespace.
    #[allow(dead_code)]
    pub fn merge_module_exports_into_globals(&mut self, module_object: &Value) {
        const BUILTIN_COUNT: usize = 75;
        let obj = match module_object {
            Value::Object(rc) => rc.borrow(),
            _ => return,
        };
        let mut names_sorted: Vec<String> = obj.keys().cloned().collect();
        names_sorted.sort();
        for name in names_sorted {
            if name == "__start_function_index" || name == "argv" {
                continue;
            }
            let value = match obj.get(&name) {
                Some(v) => v.clone(),
                None => continue,
            };
            let mut existing_indices: Vec<usize> = self.global_names
                .iter()
                .filter(|(_, n)| n.as_str() == name.as_str())
                .map(|(idx, _)| *idx)
                .collect();
            existing_indices.sort_unstable();
            if let Some(&first_index) = existing_indices.first() {
                if first_index < BUILTIN_COUNT {
                    continue;
                }
                if crate::vm::modules::is_known_module(name.as_str()) {
                    continue;
                }
                // Do not overwrite a slot that already holds a Function (from merge); keep the remapped function.
                if first_index < self.globals.len() {
                    let existing_slot = &self.globals[first_index];
                    let existing_val = match existing_slot {
                        GlobalSlot::Inline(tv) => crate::vm::store_convert::slot_to_value(*tv, &self.value_store, &self.heavy_store),
                        GlobalSlot::Heap(id) => load_value(*id, &self.value_store, &self.heavy_store),
                    };
                    if matches!(existing_val, Value::Function(_)) {
                        continue;
                    }
                }
                let id = store_value_arena(value, &mut self.value_store, &mut self.heavy_store);
                for &idx in &existing_indices {
                    if idx < self.globals.len() {
                        self.globals[idx] = GlobalSlot::Heap(id);
                    } else {
                        self.globals.resize(idx + 1, global_slot::default_global_slot());
                        self.globals[idx] = GlobalSlot::Heap(id);
                    }
                }
            } else {
                let value_to_store = match &value {
                    Value::Function(fn_idx) => {
                        if let Some(Value::Number(start)) = obj.get("__start_function_index") {
                            Value::Function((*start as usize) + fn_idx)
                        } else {
                            value.clone()
                        }
                    }
                    _ => value.clone(),
                };
                if let Value::NativeFunction(i) = &value_to_store {
                    if *i < BUILTIN_COUNT {
                        if let Some(canonical) = globals::builtin_global_name(*i) {
                            if canonical != name.as_str() {
                                continue;
                            }
                        }
                    }
                }
                let new_index = self.globals.len();
                self.globals.push(GlobalSlot::Heap(store_value_arena(value_to_store, &mut self.value_store, &mut self.heavy_store)));
                self.global_names.insert(new_index, name);
            }
        }
    }

    /// Remap Function indices in a value from module export (and Object inner) for merge into caller.
    /// ModuleFunction is passed through unchanged (resolved at Call via module_registry).
    fn remap_module_export_value(value: &Value, start_fn: usize) -> Value {
        match value {
            Value::Function(fn_idx) => Value::Function(start_fn + fn_idx),
            Value::ModuleFunction { module_id, local_index } => Value::ModuleFunction { module_id: *module_id, local_index: *local_index },
            Value::Object(obj_rc) => {
                let obj = obj_rc.borrow();
                let mut new_obj = HashMap::new();
                for (k, v) in obj.iter() {
                    let inner = match v {
                        Value::Function(i) => Value::Function(start_fn + i),
                        Value::ModuleFunction { module_id, local_index } => Value::ModuleFunction { module_id: *module_id, local_index: *local_index },
                        _ => v.clone(),
                    };
                    new_obj.insert(k.clone(), inner);
                }
                Value::Object(Rc::new(RefCell::new(new_obj)))
            }
            _ => value.clone(),
        }
    }

    /// Merge module object exports into caller's globals (used from executor to avoid double mutable borrow).
    #[allow(dead_code)]
    pub fn merge_module_exports_into_globals_into(
        module_object: &Value,
        target_globals: &mut Vec<GlobalSlot>,
        target_global_names: &mut std::collections::BTreeMap<usize, String>,
        store: &mut ValueStore,
        heap: &mut HeavyStore,
    ) {
        const BUILTIN_COUNT: usize = 75;
        let obj = match module_object {
            Value::Object(rc) => rc.borrow(),
            _ => return,
        };
        let mut names_sorted: Vec<String> = obj.keys().cloned().collect();
        names_sorted.sort();
        for name in names_sorted {
            if name == "__start_function_index" || name == "argv" {
                continue;
            }
            let value = match obj.get(&name) {
                Some(v) => v.clone(),
                None => continue,
            };
            let mut existing_indices: Vec<usize> = target_global_names
                .iter()
                .filter(|(_, n)| n.as_str() == name.as_str())
                .map(|(idx, _)| *idx)
                .collect();
            existing_indices.sort_unstable();
            if let Some(&first_index) = existing_indices.first() {
                if first_index < BUILTIN_COUNT {
                    continue;
                }
                if crate::vm::modules::is_known_module(name.as_str()) {
                    continue;
                }
                if first_index < target_globals.len() {
                    let existing_slot = &target_globals[first_index];
                    let existing_val = match existing_slot {
                        GlobalSlot::Inline(tv) => crate::vm::store_convert::slot_to_value(*tv, store, heap),
                        GlobalSlot::Heap(id) => load_value(*id, store, heap),
                    };
                    if matches!(existing_val, Value::Function(_) | Value::ModuleFunction { .. }) {
                        if name == "get_settings" || name == "load_settings" {
                            let fi = if let Value::Function(fi) = &existing_val { *fi } else { 0 };
                            debug_println!(
                                "[DEBUG merge_module_exports_into_globals_into] skip '{}' -> slot {} (already Function({}))",
                                name, first_index, fi
                            );
                        }
                        continue;
                    }
                    // Keep class Object from merge_globals_from_into (already remapped); do not overwrite with raw module_object.
                    if matches!(existing_val, Value::Object(_)) {
                        continue;
                    }
                    // Do not overwrite non-null (e.g. constructor from merge_globals_from_into) with Null from export.
                    if matches!(&value, Value::Null) && !matches!(existing_val, Value::Null) {
                        continue;
                    }
                }
                let start_fn = obj.get("__start_function_index").and_then(|v| if let Value::Number(s) = v { Some(*s as usize) } else { None }).unwrap_or(0);
                let value_to_store = Self::remap_module_export_value(&value, start_fn);
                if name == "get_settings" || name == "load_settings" {
                    let fn_info = match &value_to_store {
                        Value::Function(fi) => format!("Function({})", fi),
                        Value::ModuleFunction { module_id, local_index } => format!("ModuleFunction({},{})", module_id, local_index),
                        _ => "non-Fn".to_string(),
                    };
                    debug_println!(
                        "[DEBUG merge_module_exports_into_globals_into] write '{}' -> slots {:?} value={}",
                        name, existing_indices, fn_info
                    );
                }
                let id = store_value_arena(value_to_store, store, heap);
                for &idx in &existing_indices {
                    if idx < target_globals.len() {
                        target_globals[idx] = GlobalSlot::Heap(id);
                    } else {
                        target_globals.resize(idx + 1, global_slot::default_global_slot());
                        target_globals[idx] = GlobalSlot::Heap(id);
                    }
                }
            } else {
                let start_fn = obj.get("__start_function_index").and_then(|v| if let Value::Number(s) = v { Some(*s as usize) } else { None }).unwrap_or(0);
                let value_to_store = Self::remap_module_export_value(&value, start_fn);
                if let Value::NativeFunction(native_idx) = &value_to_store {
                    if *native_idx < BUILTIN_COUNT {
                        if let Some(canonical) = globals::builtin_global_name(*native_idx) {
                            if canonical != name.as_str() {
                                continue;
                            }
                        }
                    }
                }
                let new_index = target_globals.len();
                target_globals.push(GlobalSlot::Heap(store_value_arena(value_to_store, store, heap)));
                target_global_names.insert(new_index, name);
            }
        }
    }

    /// Добавить явную связь между колонками таблиц
    pub fn add_explicit_relation(&mut self, relation: ExplicitRelation) {
        self.explicit_relations.push(relation);
    }

    /// Получить все явные связи
    pub fn get_explicit_relations(&self) -> &Vec<ExplicitRelation> {
        &self.explicit_relations
    }

    /// Добавить явный первичный ключ таблицы
    pub fn add_explicit_primary_key(&mut self, primary_key: ExplicitPrimaryKey) {
        self.explicit_primary_keys.push(primary_key);
    }

    /// Получить явные первичные ключи таблиц
    pub fn get_explicit_primary_keys(&self) -> &Vec<ExplicitPrimaryKey> {
        &self.explicit_primary_keys
    }

    /// Вызвать пользовательскую функцию по индексу с заданными аргументами
    /// Используется нативными функциями для вызова пользовательских функций
    pub fn call_function_by_index(&mut self, function_index: usize, args: &[Value]) -> Result<Value, LangError> {
        // Setup function call (creates frame, handles cache, sets up captured variables)
        if let Some(cached_result) = calls::setup_function_call(
            function_index,
            args,
            &self.functions,
            &mut self.stack,
            &mut self.frames,
            &mut self.error_type_table,
            &mut self.value_store,
            &mut self.heavy_store,
        )? {
            return Ok(cached_result);
        }

        // Execute the function using step()
        let initial_stack_size = self.stack.len();
        let initial_frames_count = self.frames.len();
        
        loop {
            if self.frames.len() < initial_frames_count {
                break;
            }

            match self.step()? {
                VMStatus::Continue => {}
                VMStatus::Return(id) => {
                    return Ok(load_value(id, &self.value_store, &self.heavy_store));
                }
                VMStatus::FrameEnded => {
                    break;
                }
            }
        }

        if let Some(frame) = self.frames.last() {
            if self.stack.len() > frame.stack_start {
                let tv = self.stack.pop().unwrap_or(TaggedValue::null());
                let id = tagged_to_value_id(tv, &mut self.value_store);
                Ok(load_value(id, &self.value_store, &self.heavy_store))
            } else {
                Ok(load_value(NULL_VALUE_ID, &self.value_store, &self.heavy_store))
            }
        } else {
            if self.stack.len() > initial_stack_size {
                let tv = self.stack.pop().unwrap_or(TaggedValue::null());
                let id = tagged_to_value_id(tv, &mut self.value_store);
                Ok(load_value(id, &self.value_store, &self.heavy_store))
            } else {
                Ok(load_value(NULL_VALUE_ID, &self.value_store, &self.heavy_store))
            }
        }
    }
}

