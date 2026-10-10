//! Program modules: every `.dc` module of a run lives in the importing VM.
//!
//! A module is loaded once per VM (keyed by canonical path): its functions are appended to the
//! VM's function table once (function indices in constants relocated), it gets its own global
//! table built straight from the compiler's indices, and its top-level code runs as an ordinary
//! frame of the same VM. Frames carry the module through `Function::module_id`; `Vm::step`
//! checks out the global table of the executing frame (O(1) swap), so `LoadGlobal` /
//! `StoreGlobal` and every by-name lookup see the module's own globals.
//!
//! Import opcodes do not run module code recursively: the first time they meet a module they
//! push its top-level frame and rewind their own `ip`, and when the module frame finishes the same
//! opcode runs again and binds names from the now initialized table. Exceptions, cycles and deep
//! import chains therefore go through the normal frame machinery.

use crate::bytecode::{Chunk, Function};
use crate::common::error::LangError;
use crate::common::value::{ObjectKind, Value};
use crate::common::value_store::{ValueCell, ValueId, ValueStore};
use crate::common::TaggedValue;
use crate::vm::frame::{CallFrame, CALL_FRAME_FUNCTION_INDEX_MAIN};
use crate::vm::global_slot::{default_global_slot, GlobalSlot};
use crate::vm::globals::{BUILTIN_GLOBAL_COUNT, BUILTIN_GLOBAL_NAMES};
use crate::vm::heavy_store::HeavyStore;
use crate::vm::store_convert::{object_map_upsert, store_value};
use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::sync::Arc;

/// Global table of one module (checked out into the VM while the module executes).
#[derive(Default)]
pub struct ModuleTable {
    pub globals: Vec<GlobalSlot>,
    pub global_names: BTreeMap<usize, String>,
    pub explicit_global_names: BTreeMap<usize, String>,
    pub loaded_modules: HashSet<String>,
}

pub enum InitState {
    NotStarted,
    /// Top-level frame pushed by the import at `importer_ip` of the frame at `importer_depth`.
    Running {
        importer_depth: usize,
        importer_ip: usize,
        stack_start: usize,
        saved_base: Option<PathBuf>,
    },
    Ready,
}

pub struct ProgramModule {
    pub name: String,
    pub path: PathBuf,
    /// Directory nested imports of this module resolve against.
    pub dir: PathBuf,
    pub table: ModuleTable,
    /// Top-level code (relocated), kept to re-run the module after a failed initialization.
    pub init: Option<Function>,
    /// First function of this module in the VM function table and how many there are.
    pub functions_start: usize,
    pub functions_len: usize,
    pub state: InitState,
    /// Store object bound by `import m` (fields = this module's global cells).
    pub namespace_object: Option<ValueId>,
}

impl ProgramModule {
    /// Entry 0: the main script (its table lives in the VM fields while it executes).
    pub fn main() -> Self {
        Self {
            name: "__main__".to_string(),
            path: PathBuf::new(),
            dir: PathBuf::new(),
            table: ModuleTable::default(),
            init: None,
            functions_start: 0,
            functions_len: 0,
            state: InitState::Ready,
            namespace_object: None,
        }
    }
}

pub(crate) enum DcImport {
    /// No `.dc` file / package with this name: other loaders (builtin, native plugin) apply.
    NotFound,
    /// Module top-level frame pushed and the import opcode rewound; continue execution.
    Pending,
    /// Module initialized (or partially, inside an import cycle): bind names.
    Ready(u32),
}

fn vm<'a>(vm_ptr: *mut crate::vm::vm::Vm) -> &'a mut crate::vm::vm::Vm {
    unsafe { &mut *vm_ptr }
}

/// Directory imports of the executing module resolve against.
fn importer_dir(vm_ptr: *mut crate::vm::vm::Vm) -> Option<PathBuf> {
    let v = vm(vm_ptr);
    if v.current_module != 0 {
        return Some(v.program_modules[v.current_module as usize].dir.clone());
    }
    v.get_base_path().or_else(crate::vm::file_import::get_base_path)
}

/// Locate `name` (`pkg`, `file`, `a.b.c`) as a `.dc` file or package: (module dir, source file).
fn resolve_module_file(
    name: &str,
    importer_dir: &Path,
    project_root: Option<&Path>,
) -> Option<(PathBuf, PathBuf)> {
    use crate::vm::file_import::{get_dpm_package_paths, try_find_module_in, try_find_path_segment};
    let parts: Vec<&str> = name.split('.').collect();
    if parts.len() == 1 {
        let mut roots = vec![importer_dir.to_path_buf()];
        roots.extend(get_dpm_package_paths());
        return roots.iter().find_map(|r| try_find_module_in(name, r));
    }
    // Absolute dotted import: first segment from project_root (if set), else the importer's dir.
    let root_for_first = project_root.unwrap_or(importer_dir).to_path_buf();
    let mut current = root_for_first.clone();
    for (i, segment) in parts[..parts.len() - 1].iter().enumerate() {
        let mut roots = vec![if i == 0 { root_for_first.clone() } else { current.clone() }];
        roots.extend(get_dpm_package_paths());
        current = roots
            .iter()
            .find_map(|r| try_find_path_segment(segment, name, r).ok())?;
    }
    let mut roots = vec![current];
    roots.extend(get_dpm_package_paths());
    roots
        .iter()
        .find_map(|r| try_find_module_in(parts[parts.len() - 1], r))
}

/// Bytecode of a module: in-memory cache, else fresh `.dcb`, else compile (and save both).
fn compiled_module(
    vm_ptr: *mut crate::vm::vm::Vm,
    key: &Path,
    file: &Path,
) -> Result<(Chunk, Arc<Vec<Function>>), LangError> {
    use crate::vm::module_cache::CachedModule;
    let valid = |chunk: &Chunk, functions: &[Function]| {
        chunk.constant_indices_in_bounds()
            && functions.iter().all(|f| f.chunk.constant_indices_in_bounds())
    };
    if let Some(cached) = vm(vm_ptr).get_module_cache_mut().get(key) {
        if valid(&cached.chunk, &cached.functions) {
            return Ok((cached.chunk.clone(), Arc::clone(&cached.functions)));
        }
    }
    let source = std::fs::read_to_string(file).map_err(|e| {
        LangError::runtime_error(
            format!("Failed to read module file '{}': {}", file.display(), e),
            0,
        )
    })?;
    let mtime = std::fs::metadata(file)
        .ok()
        .and_then(|m| m.modified().ok())
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_nanos() as u64);
    let dcb_path = crate::vm::dcb::dcb_cache_path(key);
    let compiled = match crate::vm::dcb::load_dcb_if_fresh(&dcb_path, &source, mtime)
        .filter(|c| valid(&c.chunk, &c.functions))
    {
        Some(c) => c,
        None => {
            let (chunk, functions, _imports) =
                crate::vm::file_import::compile_module(&source, Some(file))?;
            let c = CachedModule {
                chunk,
                functions: Arc::new(functions),
            };
            let _ = crate::vm::dcb::save_dcb(&dcb_path, &c, &source, mtime);
            c
        }
    };
    let out = (compiled.chunk.clone(), Arc::clone(&compiled.functions));
    vm(vm_ptr)
        .get_module_cache_mut()
        .insert(key.to_path_buf(), compiled);
    Ok(out)
}

/// Copy of `v` with every `Function(i)` shifted by `base` (module-local -> VM index). Containers
/// holding functions (class objects: `new_N`, methods, `__special_methods`) are rebuilt; sharing
/// and cycles are kept through `memo`. Containers without functions are shared as is.
fn relocate_value(v: &Value, base: usize, memo: &mut HashMap<*const (), Value>) -> Value {
    fn has_function(v: &Value, seen: &mut HashSet<*const ()>) -> bool {
        match v {
            Value::Function(_) => true,
            Value::Array(rc) | Value::Tuple(rc) => {
                seen.insert(Rc::as_ptr(rc) as *const ())
                    && rc.borrow().iter().any(|x| has_function(x, seen))
            }
            Value::Object(rc) => {
                if !seen.insert(Rc::as_ptr(rc) as *const ()) {
                    return false;
                }
                match &*rc.borrow() {
                    ObjectKind::Legacy(m) => m.values().any(|x| has_function(x, seen)),
                    ObjectKind::Inline(e) => e
                        .iter()
                        .any(|(k, x)| has_function(k, seen) || has_function(x, seen)),
                    ObjectKind::Bucket(_) => false,
                }
            }
            _ => false,
        }
    }
    match v {
        Value::Function(i) => Value::Function(i + base),
        Value::Array(_) | Value::Tuple(_) | Value::Object(_)
            if !has_function(v, &mut HashSet::new()) =>
        {
            v.clone()
        }
        Value::Array(rc) | Value::Tuple(rc) => {
            let key = Rc::as_ptr(rc) as *const ();
            if let Some(done) = memo.get(&key) {
                return done.clone();
            }
            let fresh = Rc::new(RefCell::new(Vec::new()));
            let out = match v {
                Value::Array(_) => Value::Array(Rc::clone(&fresh)),
                _ => Value::Tuple(Rc::clone(&fresh)),
            };
            memo.insert(key, out.clone());
            let items: Vec<Value> = rc.borrow().iter().map(|x| relocate_value(x, base, memo)).collect();
            *fresh.borrow_mut() = items;
            out
        }
        Value::Object(rc) => {
            let key = Rc::as_ptr(rc) as *const ();
            if let Some(done) = memo.get(&key) {
                return done.clone();
            }
            let fresh = Rc::new(RefCell::new(ObjectKind::Inline(Vec::new())));
            let out = Value::Object(Rc::clone(&fresh));
            memo.insert(key, out.clone());
            let kind = match &*rc.borrow() {
                ObjectKind::Legacy(m) => ObjectKind::Legacy(
                    m.iter()
                        .map(|(k, x)| (k.clone(), relocate_value(x, base, memo)))
                        .collect(),
                ),
                ObjectKind::Inline(e) => ObjectKind::Inline(
                    e.iter()
                        .map(|(k, x)| (relocate_value(k, base, memo), relocate_value(x, base, memo)))
                        .collect(),
                ),
                ObjectKind::Bucket(b) => ObjectKind::Bucket(b.clone()),
            };
            *fresh.borrow_mut() = kind;
            out
        }
        _ => v.clone(),
    }
}

fn relocate_chunk(chunk: &mut Chunk, base: usize, memo: &mut HashMap<*const (), Value>) {
    for c in &mut chunk.constants {
        *c = relocate_value(c, base, memo);
    }
}

fn relocate_function(f: &mut Function, base: usize, module_id: u32, memo: &mut HashMap<*const (), Value>) {
    relocate_chunk(&mut f.chunk, base, memo);
    for d in f.default_values.iter_mut().flatten() {
        *d = relocate_value(d, base, memo);
    }
    for c in &mut f.captured_vars {
        if c.parent_function_index != usize::MAX {
            c.parent_function_index += base;
        }
    }
    f.module_id = module_id;
    // Functions of program modules resolve globals through their own table, not by module name.
    f.module_name = None;
}

/// The compiler may give one name several indices (e.g. a dedicated slot for a `default_factory`
/// constructor besides the slot the constructor is stored in); they all mean the same global.
/// Rewrite every chunk of the module to one index per name (the smallest), once at load.
fn canonicalize_global_indices(init: &mut Chunk, functions: &mut [Function]) {
    use crate::bytecode::OpCode;
    let mut canon: HashMap<String, usize> = HashMap::new();
    for chunk in std::iter::once(&*init).chain(functions.iter().map(|f| &f.chunk)) {
        for (&i, n) in &chunk.global_names {
            if is_placeholder_index(i) {
                continue;
            }
            canon
                .entry(n.clone())
                .and_modify(|c| *c = (*c).min(i))
                .or_insert(i);
        }
    }
    let patch = |chunk: &mut Chunk| {
        let remap: HashMap<usize, usize> = chunk
            .global_names
            .iter()
            .filter(|(i, _)| !is_placeholder_index(**i))
            .filter_map(|(&i, n)| canon.get(n).filter(|&&c| c != i).map(|&c| (i, c)))
            .collect();
        if remap.is_empty() {
            return;
        }
        for op in &mut chunk.code {
            if let OpCode::LoadGlobal(i) | OpCode::StoreGlobal(i) = op {
                if let Some(&c) = remap.get(i) {
                    *i = c;
                }
            }
        }
        for (from, to) in &remap {
            if let Some(n) = chunk.global_names.remove(from) {
                chunk.global_names.insert(*to, n);
            }
            if let Some(n) = chunk.explicit_global_names.remove(from) {
                chunk.explicit_global_names.insert(*to, n);
            }
        }
    };
    patch(init);
    for f in functions.iter_mut() {
        patch(&mut f.chunk);
    }
}

/// Compile-time placeholder indices that are not table slots: undefined names (resolved by name at
/// run time) and the `model_config` class load (`MODEL_CONFIG_CLASS_LOAD_INDEX`, rewritten by
/// [`patch_placeholder_loads`]).
fn is_placeholder_index(idx: usize) -> bool {
    crate::bytecode::is_undefined_global_sentinel(idx)
        || idx == crate::vm::module_system::chunk_patcher::MODEL_CONFIG_CLASS_LOAD_INDEX
}

/// Point `LoadGlobal(MODEL_CONFIG_CLASS_LOAD_INDEX)` at the module table slot of the name the
/// compiler recorded for it (`__constructing_class__`).
fn patch_placeholder_loads(chunk: &mut Chunk, table: &ModuleTable) {
    use crate::bytecode::OpCode;
    let idx = crate::vm::module_system::chunk_patcher::MODEL_CONFIG_CLASS_LOAD_INDEX;
    let Some(name) = chunk.global_names.get(&idx).cloned() else {
        return;
    };
    let Some(real) = crate::vm::global_utils::global_index_by_name(&table.global_names, &name) else {
        return;
    };
    for op in &mut chunk.code {
        if let OpCode::LoadGlobal(i) | OpCode::StoreGlobal(i) = op {
            if *i == idx {
                *i = real;
            }
        }
    }
    // Keep the chunk's name map in step (constructor setup looks the slot up through it).
    chunk.global_names.remove(&idx);
    chunk.global_names.insert(real, name);
}

/// Global table of a new module: builtins, then the compiler's indices of the module (main chunk
/// and all its functions share one index space), plus the builtin modules the VM preregisters.
fn build_table(
    init: &Chunk,
    functions: &[Function],
    builtins: &[GlobalSlot],
    builtin_modules: &[(String, GlobalSlot)],
    argv: Option<ValueId>,
) -> ModuleTable {
    let mut names: BTreeMap<usize, String> = BTreeMap::new();
    let mut explicit: BTreeMap<usize, String> = BTreeMap::new();
    for (i, n) in BUILTIN_GLOBAL_NAMES.iter().enumerate() {
        names.insert(i, (*n).to_string());
    }
    for chunk in std::iter::once(init).chain(functions.iter().map(|f| &f.chunk)) {
        for (&idx, name) in &chunk.global_names {
            if idx >= BUILTIN_GLOBAL_COUNT && !is_placeholder_index(idx) {
                names.entry(idx).or_insert_with(|| name.clone());
            }
        }
        for (&idx, name) in &chunk.explicit_global_names {
            if !is_placeholder_index(idx) {
                explicit.entry(idx).or_insert_with(|| name.clone());
            }
        }
    }
    let size = names.keys().next_back().map(|m| m + 1).unwrap_or(0).max(BUILTIN_GLOBAL_COUNT);
    let mut globals = vec![default_global_slot(); size];
    for (i, slot) in builtins.iter().enumerate().take(BUILTIN_GLOBAL_COUNT) {
        globals[i] = *slot;
    }
    let wants_argv = names.values().any(|n| n == "argv");
    let mut ensure = |name: &str, slot: Option<GlobalSlot>, globals: &mut Vec<GlobalSlot>| {
        let idx = names
            .iter()
            .find(|(_, n)| n.as_str() == name)
            .map(|(i, _)| *i)
            .unwrap_or_else(|| {
                let i = globals.len();
                globals.push(default_global_slot());
                names.insert(i, name.to_string());
                i
            });
        if let Some(slot) = slot {
            globals[idx] = slot;
        }
    };
    for (name, slot) in builtin_modules {
        ensure(name, Some(*slot), &mut globals);
    }
    ensure("__constructing_class__", None, &mut globals);
    if let Some(argv) = argv {
        if wants_argv {
            ensure("argv", Some(GlobalSlot::Heap(argv)), &mut globals);
        }
    }
    ModuleTable {
        globals,
        global_names: names,
        explicit_global_names: explicit,
        loaded_modules: HashSet::new(),
    }
}

/// Runtime values every module table needs: built-in exception constructors (`ValueError(...)`).
fn finish_table(table: &mut ModuleTable, store: &mut ValueStore, heap: &HeavyStore) {
    crate::vm::module_system::linker::ensure_exception_constructors(
        &mut table.globals,
        &table.global_names,
        store,
        heap,
    );
}

/// `fn __main__` / `fn main` are not stored by the compiled code (the entry point is invoked only
/// for a script); bind them by name so `from m import __main__` works, as the old linker did.
fn fill_entry_point_slots(
    table: &mut ModuleTable,
    functions: &[Function],
    base: usize,
    store: &mut ValueStore,
) {
    for name in ["__main__", "main"] {
        let Some(slot) = crate::vm::global_utils::global_index_by_name(&table.global_names, name)
        else {
            continue;
        };
        if let Some(pos) = functions.iter().position(|f| f.name == name) {
            if slot < table.globals.len() {
                table.globals[slot] =
                    GlobalSlot::Heap(store.allocate_arena(ValueCell::Function(base + pos)));
            }
        }
    }
}

/// Builtin-module bindings of the main table (`plot`, `uuid`, ...), copied into module tables like
/// the per-module VMs used to register them.
fn builtin_module_slots(
    current_globals: &[GlobalSlot],
    current_names: &BTreeMap<usize, String>,
    vm_ptr: *mut crate::vm::vm::Vm,
) -> Vec<(String, GlobalSlot)> {
    let v = vm(vm_ptr);
    let (globals, names): (&[GlobalSlot], &BTreeMap<usize, String>) = if v.current_module == 0 {
        (current_globals, current_names)
    } else {
        let t = &v.program_modules[0].table;
        (&t.globals, &t.global_names)
    };
    names
        .iter()
        .filter(|(i, n)| **i < globals.len() && crate::vm::modules::is_known_module(n))
        .map(|(i, n)| (n.clone(), globals[*i]))
        .collect()
}

/// Load (first time) and register module `file`; returns its id.
fn instantiate(
    name: &str,
    dir: PathBuf,
    file: PathBuf,
    key: PathBuf,
    current_globals: &[GlobalSlot],
    current_names: &BTreeMap<usize, String>,
    store: &mut ValueStore,
    heap: &HeavyStore,
    vm_ptr: *mut crate::vm::vm::Vm,
) -> Result<u32, LangError> {
    let (chunk, functions) = compiled_module(vm_ptr, &key, &file)?;
    let builtin_modules = builtin_module_slots(current_globals, current_names, vm_ptr);
    let v = vm(vm_ptr);
    let id = v.program_modules.len() as u32;
    let base = v.get_functions().len();
    let mut memo = HashMap::new();
    let relocated: Vec<Function> = functions
        .iter()
        .map(|f| {
            let mut f = f.clone();
            relocate_function(&mut f, base, id, &mut memo);
            f
        })
        .collect();
    let mut init = Function::new(format!("<module {}>", name), 0);
    init.chunk = chunk;
    relocate_chunk(&mut init.chunk, base, &mut memo);
    init.module_id = id;
    let mut relocated = relocated;
    canonicalize_global_indices(&mut init.chunk, &mut relocated);
    let table = build_table(
        &init.chunk,
        &relocated,
        v.get_builtins(),
        &builtin_modules,
        v.get_current_argv_value_id(),
    );
    let mut table = table;
    finish_table(&mut table, store, heap);
    fill_entry_point_slots(&mut table, &relocated, base, store);
    patch_placeholder_loads(&mut init.chunk, &table);
    for f in &mut relocated {
        patch_placeholder_loads(&mut f.chunk, &table);
    }
    let functions_len = relocated.len();
    v.get_functions_mut().extend(relocated);
    v.program_modules.push(ProgramModule {
        name: name.to_string(),
        path: file,
        dir,
        table,
        init: Some(init),
        functions_start: base,
        functions_len,
        state: InitState::NotStarted,
        namespace_object: None,
    });
    v.module_by_path.insert(key, id);
    Ok(id)
}

fn start_init(
    id: u32,
    frames: &mut Vec<CallFrame>,
    stack: &mut Vec<TaggedValue>,
    store: &mut ValueStore,
    heap: &mut HeavyStore,
    vm_ptr: *mut crate::vm::vm::Vm,
) {
    let v = vm(vm_ptr);
    let module = &mut v.program_modules[id as usize];
    let init = module.init.clone().expect("program module without init code");
    let importer_depth = frames.len() - 1;
    let importer = frames.last_mut().unwrap();
    // Run the import opcode again once the module's top level has finished.
    importer.ip -= 1;
    let importer_ip = importer.ip;
    let stack_start = crate::vm::stack::logical_len(stack);
    let saved_base = crate::vm::file_import::get_base_path();
    crate::vm::file_import::set_base_path(Some(module.dir.clone()));
    module.state = InitState::Running {
        importer_depth,
        importer_ip,
        stack_start,
        saved_base,
    };
    let frame = CallFrame::new(init, CALL_FRAME_FUNCTION_INDEX_MAIN, stack_start, store, heap);
    frames.push(frame);
}

/// Handle `import name` / `from name import ...` for a `.dc` module (see module docs).
pub(crate) fn import_dc_module(
    name: &str,
    frames: &mut Vec<CallFrame>,
    stack: &mut Vec<TaggedValue>,
    current_globals: &[GlobalSlot],
    current_names: &BTreeMap<usize, String>,
    store: &mut ValueStore,
    heap: &mut HeavyStore,
    vm_ptr: *mut crate::vm::vm::Vm,
) -> Result<DcImport, LangError> {
    let Some(dir) = importer_dir(vm_ptr) else {
        return Ok(DcImport::NotFound);
    };
    let project_root = vm(vm_ptr).get_project_root();
    let Some((module_dir, file)) = resolve_module_file(name, &dir, project_root.as_deref()) else {
        return Ok(DcImport::NotFound);
    };
    let key = crate::vm::module_cache::canonical_module_cache_key(&file);
    let id = match vm(vm_ptr).module_by_path.get(&key).copied() {
        Some(id) => id,
        None => instantiate(
            name,
            module_dir,
            file,
            key,
            current_globals,
            current_names,
            store,
            heap,
            vm_ptr,
        )?,
    };
    let v = vm(vm_ptr);
    match &v.program_modules[id as usize].state {
        InitState::Ready => Ok(DcImport::Ready(id)),
        InitState::NotStarted => {
            start_init(id, frames, stack, store, heap, vm_ptr);
            Ok(DcImport::Pending)
        }
        InitState::Running {
            importer_depth,
            importer_ip,
            stack_start,
            ..
        } => {
            let (importer_depth, importer_ip, stack_start) =
                (*importer_depth, *importer_ip, *stack_start);
            let init_on_stack = frames
                .iter()
                .any(|f| f.function.module_id == id && f.function_index == CALL_FRAME_FUNCTION_INDEX_MAIN);
            if init_on_stack {
                // Import cycle: bind the partially initialized module, as Python does.
                return Ok(DcImport::Ready(id));
            }
            let resumed = frames.len() == importer_depth + 1
                && frames.last().map(|f| f.ip.wrapping_sub(1)) == Some(importer_ip);
            let module = &mut v.program_modules[id as usize];
            if resumed {
                // Top level finished: drop what its statements left on the stack.
                let InitState::Running { saved_base, .. } =
                    std::mem::replace(&mut module.state, InitState::Ready)
                else {
                    unreachable!()
                };
                crate::vm::file_import::set_base_path(saved_base);
                crate::vm::stack::truncate_to(stack, &mut v.stack_sp, stack_start);
                return Ok(DcImport::Ready(id));
            }
            // The previous initialization failed (an exception unwound its frame): start over.
            if let InitState::Running { saved_base, .. } =
                std::mem::replace(&mut module.state, InitState::NotStarted)
            {
                crate::vm::file_import::set_base_path(saved_base);
            }
            let fresh = {
                let builtin_modules = builtin_module_slots(current_globals, current_names, vm_ptr);
                let v = vm(vm_ptr);
                let module = &v.program_modules[id as usize];
                let (start, len) = (module.functions_start, module.functions_len);
                build_table(
                    &module.init.as_ref().unwrap().chunk,
                    &v.get_functions()[start..start + len],
                    v.get_builtins(),
                    &builtin_modules,
                    v.get_current_argv_value_id(),
                )
            };
            let mut fresh = fresh;
            finish_table(&mut fresh, store, heap);
            {
                let v = vm(vm_ptr);
                let module = &v.program_modules[id as usize];
                let (start, len) = (module.functions_start, module.functions_len);
                let functions: Vec<Function> = v.get_functions()[start..start + len].to_vec();
                fill_entry_point_slots(&mut fresh, &functions, start, store);
            }
            v.program_modules[id as usize].table = fresh;
            start_init(id, frames, stack, store, heap, vm_ptr);
            Ok(DcImport::Pending)
        }
    }
}

fn slot_id(slot: &mut GlobalSlot, store: &mut ValueStore) -> ValueId {
    slot.resolve_to_value_id(store)
}

/// Namespace object for `import m`: a store object whose fields are the module's global cells
/// (so `m.items` and `push(m.items, x)` share the module's array). Kept current by
/// [`publish_store`] when module code rebinds a global.
pub(crate) fn namespace_object(
    id: u32,
    store: &mut ValueStore,
    heap: &mut HeavyStore,
    vm_ptr: *mut crate::vm::vm::Vm,
) -> ValueId {
    let v = vm(vm_ptr);
    if let Some(obj) = v.program_modules[id as usize].namespace_object {
        return obj;
    }
    let module_name = v.program_modules[id as usize].name.clone();
    let mut omap = crate::common::object_map::ObjectMap::new();
    let entries: Vec<(String, ValueId)> = {
        let t = &mut v.program_modules[id as usize].table;
        let names: Vec<(usize, String)> = t
            .global_names
            .iter()
            .filter(|(i, n)| **i >= BUILTIN_GLOBAL_COUNT && **i < t.globals.len() && exportable(n))
            .map(|(i, n)| (*i, n.clone()))
            .collect();
        names
            .into_iter()
            .map(|(i, n)| (n, slot_id(&mut t.globals[i], store)))
            .collect()
    };
    for (name, vid) in entries {
        let key = Value::String(name);
        let kid = store_value(key.clone(), store, heap);
        object_map_upsert(&mut omap, store, heap, &key, kid, vid);
    }
    let marker = Value::String(crate::vm::module_object::MODULE_MARKER_KEY.to_string());
    let kid = store_value(marker.clone(), store, heap);
    let vid = store_value(Value::String(module_name), store, heap);
    object_map_upsert(&mut omap, store, heap, &marker, kid, vid);
    let obj = store.allocate(ValueCell::Object(omap));
    v.program_modules[id as usize].namespace_object = Some(obj);
    v.namespace_objects.insert(obj, id);
    obj
}

/// Names a module exposes through `m.x` / `from m import *`.
fn exportable(name: &str) -> bool {
    name != "__constructing_class__" && name != "argv"
}

/// StoreGlobal in a module frame rebound `name`: keep the module's namespace object field current.
pub(crate) fn publish_store(
    name: &str,
    slot: &mut GlobalSlot,
    store: &mut ValueStore,
    heap: &HeavyStore,
    vm_ptr: *mut crate::vm::vm::Vm,
) {
    let v = vm(vm_ptr);
    let Some(obj) = v.program_modules[v.current_module as usize].namespace_object else {
        return;
    };
    if !exportable(name) {
        return;
    }
    let vid = slot_id(slot, store);
    let key = Value::String(name.to_string());
    let kid = store_value(key.clone(), store, &mut HeavyStore::new());
    crate::vm::store_convert::object_map_upsert_in_place(store, obj, heap, &key, kid, vid);
}

/// Program module whose namespace object is `id` (receiver of `m.f(x)`).
pub(crate) fn namespace_module(id: ValueId, vm_ptr: *mut crate::vm::vm::Vm) -> Option<u32> {
    vm(vm_ptr).namespace_objects.get(&id).copied()
}

/// Write `slot` to every current-table index named `name` (creating one if missing).
pub(crate) fn bind_current(
    globals: &mut Vec<GlobalSlot>,
    global_names: &mut BTreeMap<usize, String>,
    name: &str,
    slot: GlobalSlot,
) {
    let indices = crate::vm::global_utils::global_indices_by_name(global_names, name);
    if indices.is_empty() {
        let idx = globals.len();
        globals.push(slot);
        global_names.insert(idx, name.to_string());
        return;
    }
    for idx in indices {
        if idx >= globals.len() {
            globals.resize(idx + 1, default_global_slot());
        }
        globals[idx] = slot;
    }
}

/// `from m import a, b:alias, *`: copy the module's current slots into the importer's table.
/// Class names also bring their `Class::...` constructor/method globals. Errors name what is missing.
pub(crate) fn bind_from(
    id: u32,
    module_name: &str,
    items: &[String],
    globals: &mut Vec<GlobalSlot>,
    global_names: &mut BTreeMap<usize, String>,
    vm_ptr: *mut crate::vm::vm::Vm,
) -> Result<(), String> {
    let v = vm(vm_ptr);
    let t = &v.program_modules[id as usize].table;
    let lookup = |name: &str| -> Option<GlobalSlot> {
        crate::vm::global_utils::global_index_by_name(&t.global_names, name)
            .filter(|i| *i < t.globals.len())
            .map(|i| t.globals[i])
    };
    let mut binds: Vec<(String, GlobalSlot)> = Vec::new();
    let add_with_members = |src: &str, dst: &str, binds: &mut Vec<(String, GlobalSlot)>| {
        let prefix = format!("{}::", src);
        for (i, n) in &t.global_names {
            if let Some(rest) = n.strip_prefix(&prefix) {
                if *i < t.globals.len() {
                    binds.push((format!("{}::{}", dst, rest), t.globals[*i]));
                }
            }
        }
    };
    for item in items {
        if item == "*" {
            for (i, n) in &t.global_names {
                if *i >= BUILTIN_GLOBAL_COUNT
                    && *i < t.globals.len()
                    && exportable(n)
                    && !n.starts_with("__")
                {
                    binds.push((n.clone(), t.globals[*i]));
                }
            }
            continue;
        }
        let (src, dst) = match item.split_once(':') {
            Some((s, d)) => (s, d),
            None => (item.as_str(), item.as_str()),
        };
        let Some(slot) = lookup(src).filter(|_| exportable(src)) else {
            let mut avail: Vec<&str> = t
                .global_names
                .iter()
                .filter(|(i, n)| **i >= BUILTIN_GLOBAL_COUNT && exportable(n) && !n.starts_with("__") && !n.contains("::"))
                .map(|(_, n)| n.as_str())
                .collect();
            avail.sort();
            avail.dedup();
            return Err(format!(
                "Module '{}' has no attribute '{}'. Available: {}",
                module_name,
                src,
                if avail.is_empty() { "(none)".to_string() } else { avail.join(", ") }
            ));
        };
        binds.push((dst.to_string(), slot));
        add_with_members(src, dst, &mut binds);
    }
    for (name, slot) in binds {
        bind_current(globals, global_names, &name, slot);
    }
    Ok(())
}

/// `vm.modules` entry collecting the classes defined by program modules. Class lookups by name
/// (instances built by natives, superclass chains, `metadata`) fall back to `vm.modules` exports,
/// so a class defined in one module is found from any other.
pub(crate) const PROGRAM_CLASSES_MODULE: &str = "__program_classes__";

/// StoreGlobal of a class object in a module frame: register it by name (see [`PROGRAM_CLASSES_MODULE`]).
pub(crate) fn register_class(
    name: &str,
    slot: &mut GlobalSlot,
    store: &mut ValueStore,
    heap: &HeavyStore,
    vm_ptr: *mut crate::vm::vm::Vm,
) {
    let GlobalSlot::Heap(id) = *slot else {
        return;
    };
    // Class objects live in the heavy store; anything else is not a class (cheap reject).
    if !matches!(store.get(id), Some(ValueCell::Heavy(_))) {
        return;
    }
    let value = crate::vm::store_convert::load_value(id, store, heap);
    let is_class = matches!(&value, Value::Object(rc)
        if rc.borrow().str_key_get("__class_name").is_some()
            && !rc.borrow().str_key_contains("__class"));
    if !is_class {
        return;
    }
    let v = vm(vm_ptr);
    let mut modules = v.get_modules_mut();
    let entry = modules.entry(PROGRAM_CLASSES_MODULE.to_string()).or_insert_with(|| {
        Rc::new(RefCell::new(crate::vm::module_object::ModuleObject::from_exports(
            PROGRAM_CLASSES_MODULE.to_string(),
            HashMap::new(),
        )))
    });
    entry.borrow().set_export(name, value);
}
