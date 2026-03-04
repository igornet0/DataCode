// Виртуальная машина

use crate::abi::NativeAbiFn;
use crate::debug_println;
use crate::bytecode::Chunk;
use crate::common::{error::LangError, table::Table, value::Value, value_store::{ValueStore, ValueCell, ValueId, NULL_VALUE_ID}, TaggedValue};
use crate::vm::store_convert::tagged_to_value_id;
use crate::vm::frame::CallFrame;
use crate::vm::store_convert::load_value;
use crate::vm::heavy_store::HeavyStore;
use crate::vm::types::{ExplicitRelation, ExplicitPrimaryKey, ModuleInfo, VMStatus};
use crate::vm::exceptions::ExceptionHandler;
use crate::vm::global_slot::{self, GlobalSlot};
use crate::vm::globals;
use crate::vm::calls;
use crate::vm::executor;
use crate::vm::host::HostEntry;
use crate::vm::module_cache::CachedModule;
use crate::vm::module_object::ModuleObject;
use libloading::Library;
use std::cell::RefCell;
use std::collections::HashMap;
use std::path::PathBuf;
use std::rc::Rc;

/// Legacy function pointer type for native functions. Host layer uses HostEntry (Builtin/Extended).
pub type NativeFn = fn(&[Value]) -> Value;

/// Restores VM's current_argv_value_id on drop so nested run() (e.g. from natives) does not leave it cleared.
/// Used by vm::run::execute_run.
pub(crate) struct RestoreArgvIdGuard(pub(crate) *mut Vm, pub(crate) Option<ValueId>);
impl Drop for RestoreArgvIdGuard {
    fn drop(&mut self) {
        unsafe {
            (*self.0).set_current_argv_value_id(self.1);
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

pub struct Vm {
    /// Stack of TaggedValues (immediates + heap refs; no store lookup for numbers in hot path)
    stack: Vec<TaggedValue>,
    frames: Vec<CallFrame>,
    /// Builtin globals (indices 0..BUILTIN_END). Shared by all modules.
    builtins: Vec<GlobalSlot>,
    /// Module globals (indices >= BUILTIN_END). Used when no per-module isolation (legacy) or for __main__ before we switch fully.
    globals: Vec<GlobalSlot>,
    functions: Vec<crate::bytecode::Function>,
    natives: Vec<HostEntry>,
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
    /// Root directory of the project (entry script dir). Never overwritten; used for absolute imports.
    project_root: Option<PathBuf>,
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
    /// Runtime module cache: canonical path -> compiled (chunk + functions). Shared with child VMs so modules are singletons.
    module_cache: Rc<RefCell<HashMap<PathBuf, CachedModule>>>,
    /// Modules already executed once this run: canonical path -> saved namespace. Shared with child VMs so core.config etc. are singletons.
    executed_modules: Rc<RefCell<HashMap<PathBuf, Value>>>,
    /// Functions from each executed module. Shared with child VMs for cache hit remapping.
    executed_module_functions: Rc<RefCell<HashMap<PathBuf, Vec<crate::bytecode::Function>>>>,
    /// Dependency graph: canonical path -> list of canonical paths of imported modules. Shared with child VMs.
    module_deps: Rc<RefCell<HashMap<PathBuf, Vec<PathBuf>>>>,
    /// Cache of loaded modules by canonical name (e.g. "core.config") or path. Each module has its own namespace.
    modules: RefCell<HashMap<String, Rc<RefCell<ModuleObject>>>>,
    /// When set (e.g. from run_with_vm_internal_with_args), update_chunk_indices_from_names will always map "argv" to this slot,
    /// so ImportFrom re-patch does not remap LoadGlobal(argv) to load_settings (slot 79) after merge.
    argv_slot_index: Option<usize>,
    /// Indices that meant "argv" in the main chunk before patching; LoadGlobal(any of these) should load script argv.
    argv_old_indices: Option<Vec<usize>>,
    /// During run(), when argv_patch is set, this holds the canonical argv value id so LoadGlobal(argv_slot) always loads it
    /// even if the slot was overwritten by ImportFrom or merge.
    current_argv_value_id: Option<ValueId>,
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
            project_root: None,
            ml_context: Some(crate::ml::MlContext::new()),
            plot_context: Some(crate::plot::PlotContext::new()),
            value_store: ValueStore::new(),
            heavy_store: HeavyStore::new(),
            native_args_buffer: Vec::with_capacity(DEFAULT_NATIVE_BUF_CAPACITY),
            reusable_native_arg_ids: Vec::with_capacity(DEFAULT_NATIVE_BUF_CAPACITY),
            reusable_all_popped: Vec::with_capacity(DEFAULT_NATIVE_BUF_CAPACITY),
            pending_relations: Vec::new(),
            pending_primary_keys: Vec::new(),
            module_cache: Rc::new(RefCell::new(HashMap::new())),
            executed_modules: Rc::new(RefCell::new(HashMap::new())),
            executed_module_functions: Rc::new(RefCell::new(HashMap::new())),
            module_deps: Rc::new(RefCell::new(HashMap::new())),
            modules: RefCell::new(HashMap::new()),
            argv_slot_index: None,
            argv_old_indices: None,
            current_argv_value_id: None,
            module_registry: RefCell::new(Vec::new()),
        };
        vm.register_natives();
        vm
    }

    /// Creates a child VM that shares module_cache, executed_modules, executed_module_functions, and module_deps with the parent.
    /// Ensures module singletons: when engine imports core.config, it gets the same namespace as main (e.g. load_settings mutates shared state).
    pub(crate) fn new_child(parent: &Self) -> Self {
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
            project_root: parent.project_root.clone(),
            ml_context: Some(crate::ml::MlContext::new()),
            plot_context: Some(crate::plot::PlotContext::new()),
            value_store: ValueStore::new(),
            heavy_store: HeavyStore::new(),
            native_args_buffer: Vec::with_capacity(DEFAULT_NATIVE_BUF_CAPACITY),
            reusable_native_arg_ids: Vec::with_capacity(DEFAULT_NATIVE_BUF_CAPACITY),
            reusable_all_popped: Vec::with_capacity(DEFAULT_NATIVE_BUF_CAPACITY),
            pending_relations: Vec::new(),
            pending_primary_keys: Vec::new(),
            module_cache: parent.module_cache.clone(),
            executed_modules: parent.executed_modules.clone(),
            executed_module_functions: parent.executed_module_functions.clone(),
            module_deps: parent.module_deps.clone(),
            modules: RefCell::new(HashMap::new()),
            argv_slot_index: None,
            argv_old_indices: None,
            current_argv_value_id: None,
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

    /// Set project root for absolute imports. Set once at entry; never overwritten when loading nested modules.
    pub fn set_project_root(&mut self, path: Option<PathBuf>) {
        self.project_root = path;
    }

    /// Project root for absolute imports (e.g. from core.config). Used by module resolver.
    pub fn get_project_root(&self) -> Option<PathBuf> {
        self.project_root.clone()
    }

    /// Set the slot index used for argv so update_chunk_indices_from_names (e.g. after ImportFrom) always maps "argv" to this slot.
    pub fn set_argv_slot_index(&mut self, slot: Option<usize>) {
        self.argv_slot_index = slot;
    }

    /// Get the argv slot index if set.
    pub fn get_argv_slot_index(&self) -> Option<usize> {
        self.argv_slot_index
    }

    /// Set the bytecode indices that meant "argv" in the main chunk (so LoadGlobal(any) is treated as argv).
    pub(crate) fn set_argv_old_indices(&mut self, indices: Option<Vec<usize>>) {
        self.argv_old_indices = indices;
    }

    /// Get the bytecode indices that mean "argv" for this run.
    pub(crate) fn get_argv_old_indices(&self) -> Option<&[usize]> {
        self.argv_old_indices.as_deref()
    }

    /// Set the canonical argv value id for this run so LoadGlobal(argv_slot) always loads it.
    pub(crate) fn set_current_argv_value_id(&mut self, id: Option<ValueId>) {
        self.current_argv_value_id = id;
    }

    /// Get the current run's argv value id if set.
    pub(crate) fn get_current_argv_value_id(&self) -> Option<ValueId> {
        self.current_argv_value_id
    }

    // --- Run orchestration helpers (used by vm::run::execute_run) ---
    pub(crate) fn get_base_path_mut_ptr(&mut self) -> *mut Option<PathBuf> {
        &mut self.base_path as *mut _
    }
    pub(crate) fn take_ml_context(&mut self) -> Option<crate::ml::MlContext> {
        self.ml_context.take()
    }
    pub(crate) fn get_ml_context_mut_ptr(&mut self) -> *mut Option<crate::ml::MlContext> {
        &mut self.ml_context as *mut _
    }
    pub(crate) fn take_plot_context(&mut self) -> Option<crate::plot::PlotContext> {
        self.plot_context.take()
    }
    pub(crate) fn get_plot_context_mut_ptr(&mut self) -> *mut Option<crate::plot::PlotContext> {
        &mut self.plot_context as *mut _
    }
    pub(crate) fn merge_global_names_from_chunk(&mut self, chunk: &crate::bytecode::Chunk) {
        globals::merge_global_names(
            &mut self.global_names,
            &mut self.explicit_global_names,
            &chunk.global_names,
            &chunk.explicit_global_names,
        );
    }
    pub(crate) fn global_names_contains(&self, name: &str) -> bool {
        self.global_names.values().any(|n| n == name)
    }
    pub(crate) fn ensure_global_slot(&mut self, name: &str) {
        let idx = self.globals.len();
        self.globals.push(global_slot::default_global_slot());
        self.global_names.insert(idx, name.to_string());
    }
    pub(crate) fn ensure_globals_len(&mut self, len: usize) {
        if self.globals.len() < len {
            self.globals.resize(len, global_slot::default_global_slot());
        }
    }
    pub(crate) fn set_global_slot(&mut self, index: usize, slot: GlobalSlot) {
        self.ensure_globals_len(index + 1);
        self.globals[index] = slot;
    }
    pub(crate) fn push_frame(&mut self, frame: CallFrame) {
        self.frames.push(frame);
    }
    pub(crate) fn stack_is_empty(&self) -> bool {
        self.stack.is_empty()
    }
    pub(crate) fn stack_pop(&mut self) -> Option<TaggedValue> {
        self.stack.pop()
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
        crate::vm::native_registry::register_builtin_natives(&mut self.natives);
    }

    /// Добавляет в VM слоты для всех имён из chunk.global_names, которых ещё нет в VM.
    pub fn ensure_globals_from_chunk(&mut self, chunk: &crate::bytecode::Chunk) {
        crate::vm::module_system::linker::ensure_globals_from_chunk(&mut self.globals, &mut self.global_names, chunk);
    }

    /// Как ensure_globals_from_chunk, но сохраняет индексы из chunk.
    pub fn ensure_globals_from_chunk_preserve_indices(&mut self, chunk: &crate::bytecode::Chunk) {
        crate::vm::module_system::linker::ensure_globals_from_chunk_preserve_indices(&mut self.globals, &mut self.global_names, chunk);
    }

    /// Fills global slots at index >= 75 whose name is a builtin (e.g. "str", "path").
    /// ensure_globals_from_chunk_preserve_indices adds (idx, name) from chunk and only resizes;
    /// register_native_globals only fills 0..75, so slots at 75+ stay null and LoadGlobal(idx) returns null → "Can only call functions".
    pub fn ensure_builtin_globals_high_indices(&mut self) {
        crate::vm::module_system::linker::ensure_builtin_globals_high_indices(
            &mut self.globals,
            &self.global_names,
            &mut self.value_store,
            &self.heavy_store,
        );
    }

    /// Injects built-in exception constructors (e.g. ValueError::new_1) into slots that are still null after ensure_globals_from_chunk.
    /// Called from run_compiled_module so that raise ValueError("...") works in modules.
    pub fn ensure_exception_constructors(&mut self) {
        crate::vm::module_system::linker::ensure_exception_constructors(
            &mut self.globals,
            &self.global_names,
            &mut self.value_store,
            &self.heavy_store,
        );
    }

    /// Обновляет индексы глобальных переменных в chunk по VM.
    /// Перед патчем добавляет в VM.global_names все имена из chunk, которых там ещё нет,
    /// чтобы не получать "no match" и не оставлять LoadGlobal/StoreGlobal без ремаппинга.
    pub fn update_chunk_indices(&mut self, chunk: &mut crate::bytecode::Chunk) {
        const UNDEFINED_GLOBAL_SENTINEL: usize = usize::MAX;
        for (idx, name) in &chunk.global_names.clone() {
            if *idx == UNDEFINED_GLOBAL_SENTINEL || name.as_str() == "argv" {
                continue;
            }
            if !self.global_names.values().any(|n| n == name) {
                let new_idx = self.globals.len();
                self.globals.push(global_slot::default_global_slot());
                self.global_names.insert(new_idx, name.clone());
                debug_println!("[DEBUG update_chunk_indices] Добавлен слот для '{}' в globals[{}] (отсутствовал в caller)", name, new_idx);
            }
        }
        crate::vm::module_system::chunk_patcher::update_chunk_indices_from_names(
            chunk,
            &self.global_names,
            Some(self.globals.as_mut_slice()),
            Some(&mut self.value_store),
            Some(&self.heavy_store),
            self.argv_slot_index,
            true, // resolve sentinel (normal pre-run patch)
        );
    }

    /// Устанавливает функции в VM. Патчит LoadGlobal/StoreGlobal, обновляет globals.
    pub fn set_functions(
        &mut self,
        functions: Vec<crate::bytecode::Function>,
        main_chunk: Option<&mut crate::bytecode::Chunk>,
        main_old_idx_to_name: Option<std::collections::HashMap<usize, String>>,
    ) {
        crate::vm::module_system::linker::set_functions(
            &mut self.globals,
            &mut self.global_names,
            &mut self.functions,
            &mut self.value_store,
            &mut self.explicit_global_names,
            self.argv_slot_index,
            functions,
            main_chunk,
            main_old_idx_to_name,
        );
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
        modules::register_module("database_engine", &mut self.natives, &mut self.globals, &mut self.global_names, &mut self.value_store, &mut self.heavy_store)?;
        Ok(())
    }

    // Exception handling methods moved to exceptions.rs

    /// argv_patch: when Some((argv_slot_index, old_indices, argv_value_id)), the chunk is cloned and any LoadGlobal(old_idx)
    /// in the clone is replaced with LoadGlobal(argv_slot_index). If argv_value_id is Some(id), that value is written
    /// to globals[argv_slot_index] immediately before the execution loop so the argv slot is never overwritten by earlier code.
    pub fn run(&mut self, chunk: &Chunk, argv_patch: Option<(usize, &[usize], Option<ValueId>)>) -> Result<Value, LangError> {
        crate::vm::run::execute_run(self, chunk, argv_patch)
    }

    /// Выполнить один шаг VM - получить следующую инструкцию и выполнить её.
    /// Pub(crate) for vm::run::execute_run.
    pub(crate) fn step(&mut self) -> Result<VMStatus, LangError> {
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
    pub fn get_natives(&self) -> &[HostEntry] {
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
        crate::vm::module_system::linker::merge_globals_from(
            other.get_globals(),
            other.get_global_names(),
            other.get_natives(),
            other.get_functions(),
            other.value_store(),
            other.heavy_store(),
            &mut self.globals,
            &mut self.global_names,
            &mut self.functions,
            &mut self.natives,
            &mut self.value_store,
            &mut self.heavy_store,
        );
    }

    /// After merging a module into the caller, re-establish __main__ and main in the caller's globals.
    pub fn ensure_entry_point_slots(
        target_globals: &mut [GlobalSlot],
        target_global_names: &std::collections::BTreeMap<usize, String>,
        target_functions: &[crate::bytecode::Function],
        store: &mut ValueStore,
    ) {
        crate::vm::module_system::linker::ensure_entry_point_slots(target_globals, target_global_names, target_functions, store);
    }

    /// Legacy: merge module VM into caller's buffers. Not used with module isolation (ImportFrom only adds requested items).
    #[allow(dead_code)]
    pub fn merge_globals_from_into(
        other: &Vm,
        target_globals: &mut Vec<GlobalSlot>,
        target_global_names: &mut std::collections::BTreeMap<usize, String>,
        target_functions: &mut Vec<crate::bytecode::Function>,
        target_natives: &mut Vec<HostEntry>,
        store: &mut ValueStore,
        heap: &mut HeavyStore,
    ) {
        crate::vm::module_system::linker::merge_globals_from_into(
            other.get_globals(),
            other.get_global_names(),
            other.get_natives(),
            other.get_functions(),
            other.value_store(),
            other.heavy_store(),
            target_globals,
            target_global_names,
            target_functions,
            target_natives,
            store,
            heap,
        );
    }

    /// Merges exports from an already-executed module object into this VM's globals.
    #[allow(dead_code)]
    pub fn merge_module_exports_into_globals(&mut self, module_object: &Value) {
        crate::vm::module_system::linker::merge_module_exports_into_globals_into(
            module_object,
            &mut self.globals,
            &mut self.global_names,
            &mut self.value_store,
            &mut self.heavy_store,
        );
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
        crate::vm::module_system::linker::merge_module_exports_into_globals_into(
            module_object,
            target_globals,
            target_global_names,
            store,
            heap,
        );
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

