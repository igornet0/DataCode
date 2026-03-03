//! Import and ImportFrom opcode handlers.
//! Loads built-in, .dc file, and native modules; merges into caller's globals.

use crate::debug_println;
use crate::common::{error::LangError, value::Value, value_store::{ValueId, ValueStore}, TaggedValue};
use crate::vm::types::VMStatus;
use crate::vm::frame::CallFrame;
use crate::vm::exceptions::ExceptionHandler;
use crate::vm::modules;
use crate::vm::global_slot::{GlobalSlot, default_global_slot};
use crate::vm::store_convert::{store_value, load_value};
use crate::vm::heavy_store::HeavyStore;
use crate::vm::executor::{global_index_by_name, global_indices_by_name};
use std::rc::Rc;
use std::cell::RefCell;

/// Execute Import(module_index): load module by name and store in globals.
#[allow(clippy::too_many_arguments)]
pub fn handle_import(
    module_index: usize,
    line: usize,
    stack: &mut Vec<TaggedValue>,
    frames: &mut Vec<CallFrame>,
    globals: &mut Vec<GlobalSlot>,
    global_names: &mut std::collections::BTreeMap<usize, String>,
    _functions: &mut Vec<crate::bytecode::Function>,
    natives: &mut Vec<crate::vm::host::HostEntry>,
    exception_handlers: &mut Vec<ExceptionHandler>,
    loaded_modules: &mut std::collections::HashSet<String>,
    abi_natives: &mut Vec<crate::abi::NativeAbiFn>,
    loaded_native_libraries: &mut Vec<libloading::Library>,
    value_store: &mut ValueStore,
    heavy_store: &mut HeavyStore,
    vm_ptr: *mut crate::vm::vm::Vm,
) -> Result<VMStatus, LangError> {
    let frame = frames.last().unwrap();
    let module_name = match load_value(frame.constant_ids[module_index], value_store, heavy_store) {
        Value::String(name) => name,
        _ => {
            let error = ExceptionHandler::runtime_error(
                &frames,
                "Import expects module name as string".to_string(),
                line,
            );
            match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error, value_store, heavy_store) {
                Ok(()) => return Ok(VMStatus::Continue),
                Err(e) => return Err(e),
            }
        }
    };

    if loaded_modules.contains(&module_name) {
        return Ok(VMStatus::Continue);
    }
    if modules::is_known_module(&module_name) {
        modules::register_module(&module_name, natives, globals, global_names, value_store, heavy_store)?;
        loaded_modules.insert(module_name);
        return Ok(VMStatus::Continue);
    }
    let base_path = unsafe { (*vm_ptr).get_base_path() }.or_else(crate::vm::file_import::get_base_path);
    let load_dc_err = if let Some(ref base_path) = base_path {
        match crate::vm::file_import::load_local_module_with_vm(&module_name, base_path, unsafe { &mut *vm_ptr }) {
            Ok((module_object, opt_vm)) => {
                if let Some(module_vm) = opt_vm {
                    let start_function_index = unsafe {
                        let vm_ref = &mut *vm_ptr;
                        vm_ref.add_functions_from_module(
                            module_vm.get_functions().clone(),
                            module_name.clone(),
                        )
                    };
                    if let Value::Object(module_obj_rc) = &module_object {
                        let mut module_obj = module_obj_rc.borrow_mut();
                        module_obj.insert("__start_function_index".to_string(), Value::Number(start_function_index as f64));
                        for (_, v) in module_obj.iter_mut() {
                            if let Value::Function(i) = v {
                                *v = Value::Function(start_function_index + *i);
                            }
                        }
                    }
                }
                let id = store_value(module_object, value_store, heavy_store);
                if let Some(idx) = global_index_by_name(global_names, &module_name) {
                    if idx < globals.len() { globals[idx] = GlobalSlot::Heap(id); } else { globals.resize(idx + 1, default_global_slot()); globals[idx] = GlobalSlot::Heap(id); }
                } else {
                    let idx = globals.len();
                    globals.push(GlobalSlot::Heap(id));
                    global_names.insert(idx, module_name.clone());
                }
                loaded_modules.insert(module_name);
                return Ok(VMStatus::Continue);
            }
            Err(e) => Some(e),
        }
    } else {
        None
    };
    if let Ok(module_object) = crate::vm::native_loader::try_load_native_module(
        &module_name,
        base_path.as_deref(),
        natives.len(),
        abi_natives,
        loaded_native_libraries,
    ) {
        let module_value = Value::Object(Rc::new(RefCell::new(module_object)));
        let id = store_value(module_value, value_store, heavy_store);
        if let Some(idx) = global_index_by_name(global_names, &module_name) {
            if idx < globals.len() { globals[idx] = GlobalSlot::Heap(id); } else { globals.resize(idx + 1, default_global_slot()); globals[idx] = GlobalSlot::Heap(id); }
        } else {
            let idx = globals.len();
            globals.push(GlobalSlot::Heap(id));
            global_names.insert(idx, module_name.clone());
        }
        loaded_modules.insert(module_name);
        return Ok(VMStatus::Continue);
    }
    Err(load_dc_err.map_or_else(
        || LangError::runtime_error(
            format!("Module '{}' not found (built-in, .dc file, or native module)", module_name),
            line,
        ),
        |e| LangError::runtime_error_with_source(
            format!("Failed to load module '{}'", module_name),
            e,
        ),
    ))
}


/// Execute ImportFrom(module_index, items_index): load module and import selected items into globals.
#[allow(clippy::too_many_arguments)]
pub fn handle_import_from(
    module_index: usize,
    items_index: usize,
    line: usize,
    stack: &mut Vec<TaggedValue>,
    frames: &mut Vec<CallFrame>,
    globals: &mut Vec<GlobalSlot>,
    global_names: &mut std::collections::BTreeMap<usize, String>,
    functions: &mut Vec<crate::bytecode::Function>,
    natives: &mut Vec<crate::vm::host::HostEntry>,
    exception_handlers: &mut Vec<ExceptionHandler>,
    loaded_modules: &mut std::collections::HashSet<String>,
    abi_natives: &mut Vec<crate::abi::NativeAbiFn>,
    loaded_native_libraries: &mut Vec<libloading::Library>,
    value_store: &mut ValueStore,
    heavy_store: &mut HeavyStore,
    vm_ptr: *mut crate::vm::vm::Vm,
) -> Result<VMStatus, LangError> {
    let frame = frames.last().unwrap();
    let module_name = match load_value(frame.constant_ids[module_index], value_store, heavy_store) {
            Value::String(name) => name,
            _ => {
                let error = ExceptionHandler::runtime_error(
                    &frames,
                    "ImportFrom expects module name as string".to_string(),
                    line,
                );
                match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error, value_store, heavy_store) {
                    Ok(()) => return Ok(VMStatus::Continue),
                    Err(e) => return Err(e),
                }
            }
        };
        let items_array = match load_value(frame.constant_ids[items_index], value_store, heavy_store) {
            Value::Array(arr) => arr.borrow().clone(),
            _ => {
                let error = ExceptionHandler::runtime_error(
                &frames,
                    "ImportFrom expects items array".to_string(),
                    line,
                );
                match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error, value_store, heavy_store) {
                    Ok(()) => return Ok(VMStatus::Continue),
                    Err(e) => return Err(e),
                }
            }
        };
        // Bound names to import (for module isolation: only these are visible in caller)
        let imported_names: std::collections::HashSet<String> = items_array
            .iter()
            .filter_map(|v| {
                if let Value::String(s) = v {
                    if s == "*" {
                        Some("*".to_string())
                    } else if let Some((_, alias)) = s.split_once(':') {
                        Some(alias.to_string())
                    } else {
                        Some(s.clone())
                    }
                } else {
                    None
                }
            })
            .collect();
        let argv_slot_import = unsafe { (*vm_ptr).get_argv_slot_index() };
        // Preserve argv value id before any feed/per-item writes so we can restore the slot at the end (nested module run clears RunContext/SCRIPT_ARGV_VALUE_ID visibility).
        let saved_argv_value_id = argv_slot_import.and_then(|slot_idx| {
            if slot_idx < globals.len() {
                Some(globals[slot_idx].resolve_to_value_id(value_store))
            } else {
                unsafe { (*vm_ptr).get_current_argv_value_id() }
            }
        });
        
        // Register the module if not already loaded
        if !loaded_modules.contains(&module_name) {
            // Сначала попробуем зарегистрировать как встроенный модуль
            if modules::is_known_module(&module_name) {
                modules::register_module(&module_name, natives, globals, global_names, value_store, heavy_store)?;
                loaded_modules.insert(module_name.clone());
            } else {
                // Попробуем загрузить как локальный файл (VM base_path затем thread-local)
                use crate::vm::file_import;
                let base_path = unsafe { (*vm_ptr).get_base_path() }.or_else(file_import::get_base_path);
                if let Some(ref base_path) = base_path {
                    let functions_len_before = functions.len();
                    match file_import::load_local_module_with_vm(&module_name, base_path, unsafe { &mut *vm_ptr }) {
                        Ok((module_object, opt_vm)) => {
                            debug_println!("[DEBUG ImportFrom] Загружен модуль '{}'", module_name);
                            if let Some(ref module_vm) = opt_vm {
                                // Add module's functions so Function indices in module_object are valid when we copy requested items. Do not merge module global_names.
                                let start_idx = functions.len();
                                let mut new_fns: Vec<_> = module_vm.get_functions().iter().cloned().collect();
                                for f in &mut new_fns {
                                    f.module_name = Some(module_name.clone());
                                }
                                functions.extend(new_fns);
                                let module_function_count = module_vm.get_functions().len();
                                crate::remap_function_constants_in_chunks(functions, start_idx, module_function_count);
                                debug_println!("[DEBUG ImportFrom] Функций в модуле: {}, start_idx: {}", module_function_count, start_idx);
                                // Register submodules first, then the top-level module, so the top-level module_id never collides with
                                // module VM registry indices (0, 1, ...). That way replace_function_with_module_function_in_exports
                                // uses a distinct id and remap only rewrites submodule-origin ModuleFunctions.
                                let (module_id, submodule_old_to_new): (usize, std::collections::HashMap<usize, usize>) = {
                                    let sub_infos: Vec<_> = module_vm.get_module_registry().iter().cloned().collect();
                                    let mut reg = unsafe { (*vm_ptr).get_module_registry_mut() };
                                    let mut map = std::collections::HashMap::new();
                                    for (old_id, info) in sub_infos.iter().enumerate() {
                                        reg.push(crate::vm::types::ModuleInfo {
                                            name: info.name.clone(),
                                            function_offset: start_idx + info.function_offset,
                                            function_count: info.function_count,
                                        });
                                        map.insert(old_id, reg.len() - 1);
                                    }
                                    reg.push(crate::vm::types::ModuleInfo {
                                        name: module_name.clone(),
                                        function_offset: start_idx,
                                        function_count: module_vm.get_functions().len(),
                                    });
                                    let id = reg.len() - 1;
                                    (id, map)
                                };
                                // Convert namespace: Value::Function(local_index) -> Value::ModuleFunction { module_id, local_index }.
                                if let Value::Object(module_obj_rc) = &module_object {
                                    let mut module_obj = module_obj_rc.borrow_mut();
                                    crate::replace_function_with_module_function_in_exports(&mut *module_obj, module_id);
                                    // Remap ModuleFunction from submodule VM indices to caller VM registry (classes from dev_config/prod_config).
                                    crate::remap_module_function_ids_in_exports(&mut *module_obj, &submodule_old_to_new);
                                }
                                // Extend caller's natives with module's natives and remap NativeFunction in module object
                                // (fixes "Native function index 194 out of bounds" when merged code uses Config etc.).
                                const BUILTIN_NATIVE_COUNT: usize = 75;
                                let other_natives = module_vm.get_natives();
                                let native_start = natives.len();
                                if other_natives.len() > BUILTIN_NATIVE_COUNT {
                                    natives.extend_from_slice(&other_natives[BUILTIN_NATIVE_COUNT..]);
                                }
                                if let Value::Object(module_obj_rc) = &module_object {
                                    let mut module_obj = module_obj_rc.borrow_mut();
                                    crate::remap_native_indices_in_exports(&mut *module_obj, native_start);
                                }
                                // Merge submodules (e.g. config, prod_config loaded by core.config) into caller so __constructing_class__ lookup finds classes from them.
                                // Do not overwrite existing modules: caller may have runtime state (e.g. core.config.settings from load_settings).
                                {
                                    let mods = module_vm.get_modules();
                                    let mut caller_mods = unsafe { (*vm_ptr).get_modules_mut() };
                                    for (k, v) in mods.iter() {
                                        caller_mods.entry(k.clone()).or_insert_with(|| v.clone());
                                        debug_println!("[DEBUG ImportFrom] merged submodule '{}' into caller", k);
                                    }
                                }
                                // Feed caller globals with names that merged module chunks reference (from module object or caller's merged modules),
                                // so update_chunk_indices can map LoadGlobal to filled slots (fixes "Undefined variable: DatabaseConfig::new_1" in prod).
                                // Feed all names that merged functions reference and that have non-Null values (classes, functions).
                                // Skip internal module vars (e.g. "settings" = Null) so module state is not leaked.
                                let argv_slot_feed = unsafe { (*vm_ptr).get_argv_slot_index() };
                                {
                                    let value_for_name = |name: &str| -> Option<Value> {
                                        if let Value::Object(ref module_obj_rc) = &module_object {
                                            if let Some(v) = module_obj_rc.borrow().get(name) {
                                                return Some(v.clone());
                                            }
                                        }
                                        let caller_mods = unsafe { (*vm_ptr).get_modules() };
                                        for (_mod_name, rc) in caller_mods.iter() {
                                            if let Some(v) = rc.borrow().get_export(name) {
                                                return Some(v);
                                            }
                                        }
                                        None
                                    };
                                    let is_internal_module_var = |_name: &str, value: &Value| -> bool {
                                        matches!(value, Value::Null)
                                    };
                                    for i in start_idx..functions.len() {
                                        let chunk = &functions[i].chunk;
                                        for (_idx, name) in &chunk.global_names {
                                            if let Some(value) = value_for_name(name) {
                                                if is_internal_module_var(name, &value) {
                                                    continue;
                                                }
                                                let ok_to_feed = match &value {
                                                    Value::Function(_) | Value::ModuleFunction { .. } | Value::Object(_) => true,
                                                    _ => false,
                                                };
                                                if !ok_to_feed {
                                                    continue;
                                                }
                                                let existing: Vec<usize> = global_names
                                                    .iter()
                                                    .filter(|(_, n)| *n == name)
                                                    .map(|(idx, _)| *idx)
                                                    .collect();
                                                let slot_idx = if existing.is_empty() {
                                                    let idx = globals.len();
                                                    globals.resize(idx + 1, default_global_slot());
                                                    global_names.insert(idx, name.clone());
                                                    idx
                                                } else {
                                                    *existing.iter().min().unwrap()
                                                };
                                                let slot_empty = slot_idx >= globals.len()
                                                    || matches!(
                                                        load_value(globals[slot_idx].resolve_to_value_id(value_store), value_store, heavy_store),
                                                        Value::Null
                                                    );
                                                if slot_empty && Some(slot_idx) != argv_slot_feed {
                                                    if slot_idx >= globals.len() {
                                                        globals.resize(slot_idx + 1, default_global_slot());
                                                    }
                                                    let id = store_value(value.clone(), value_store, heavy_store);
                                                    globals[slot_idx] = GlobalSlot::Heap(id);
                                                }
                                            }
                                        }
                                    }
                                }
                                // Ensure all names from main + function chunks are in global_names
                                // so update_chunk_indices_from_names finds them (no "no match").
                                // Module isolation: skip only internal plain vars (e.g. "settings" = Null).
                                const UNDEFINED_GLOBAL_SENTINEL: usize = usize::MAX;
                                let is_internal_module_var = |name: &str| -> bool {
                                    if imported_names.contains(name) || imported_names.contains("*") {
                                        return false;
                                    }
                                    let v_opt = if let Value::Object(ref rc) = &module_object {
                                        rc.borrow().get(name).cloned()
                                    } else {
                                        None
                                    };
                                    v_opt.as_ref().map_or(false, |v| matches!(v, Value::Null))
                                };
                                let chunks_to_feed: Vec<_> = frames.first().map(|f| &f.function.chunk).into_iter()
                                    .chain(functions.iter().map(|f| &f.chunk))
                                    .collect();
                                for (chunk_index, chunk) in chunks_to_feed.iter().enumerate() {
                                    let is_merged_module_chunk = chunk_index >= 1 && (chunk_index - 1) >= start_idx;
                                    for (idx, name) in &chunk.global_names {
                                        if *idx == UNDEFINED_GLOBAL_SENTINEL || name.as_str() == "argv" {
                                            continue;
                                        }
                                        if is_merged_module_chunk && is_internal_module_var(name) {
                                            continue;
                                        }
                                        if !global_names.values().any(|n| n == name) {
                                            let new_idx = globals.len();
                                            globals.push(default_global_slot());
                                            global_names.insert(new_idx, name.clone());
                                            debug_println!("[DEBUG ImportFrom merge] Добавлен слот для '{}' в globals[{}]", name, new_idx);
                                        }
                                    }
                                }
                                for name in ["create_all", "__constructing_class__"] {
                                    if !global_names.values().any(|n| n == name)
                                        && chunks_to_feed.iter().any(|chunk| chunk.global_names.values().any(|n| n == name))
                                    {
                                        let new_idx = globals.len();
                                        globals.push(default_global_slot());
                                        global_names.insert(new_idx, name.to_string());
                                        debug_println!("[DEBUG ImportFrom merge] Добавлен fallback слот для '{}' в globals[{}]", name, new_idx);
                                    }
                                }
                                // Module isolation: do NOT merge module globals into caller. Only requested items are added below (per-item loop).
                                let global_names_snapshot = global_names.clone();
                                let argv_slot = unsafe { (*vm_ptr).get_argv_slot_index() };
                                if std::env::var("DATACODE_DEBUG").is_ok() {
                                    eprintln!("=== IMPORTFROM DEBUG START (merge) ===");
                                    if let Some(mf) = frames.first() {
                                        let chunk = &mf.function.chunk;
                                        eprintln!("[IMPORTFROM] Chunk global names (main):");
                                        for (idx, name) in &chunk.global_names {
                                            eprintln!("  chunk idx={} name={}", idx, name);
                                        }
                                    }
                                    eprintln!("[IMPORTFROM] caller_global_names (snapshot) before update:");
                                    for (idx, name) in global_names_snapshot.iter() {
                                        eprintln!("  caller idx={} name={}", idx, name);
                                    }
                                    if !global_names_snapshot.values().any(|n| n == "create_all") {
                                        eprintln!("[WARNING] create_all not found in caller_global_names BEFORE update");
                                    }
                                }
                                for i in 0..functions.len() {
                                    crate::vm::module_system::chunk_patcher::update_chunk_indices_from_names(
                                        &mut functions[i].chunk,
                                        &global_names_snapshot,
                                        Some(globals.as_mut_slice()),
                                        Some(value_store),
                                        Some(heavy_store),
                                        argv_slot,
                                        true, // merged module chunks: resolve sentinel
                                    );
                                }
                                if let Some(main_frame) = frames.first_mut() {
                                    crate::vm::module_system::chunk_patcher::update_chunk_indices_from_names(
                                        &mut main_frame.function.chunk,
                                        &global_names_snapshot,
                                        Some(globals.as_mut_slice()),
                                        Some(value_store),
                                        Some(heavy_store),
                                        argv_slot,
                                        false, // main chunk: do NOT resolve sentinel (module isolation)
                                    );
                                }
                                if std::env::var("DATACODE_DEBUG").is_ok() {
                                    if !global_names_snapshot.values().any(|n| n == "create_all") {
                                        eprintln!("[WARNING] create_all STILL MISSING AFTER update_chunk_indices_from_names");
                                    }
                                    eprintln!("=== IMPORTFROM DEBUG END (merge) ===");
                                }
                                crate::vm::module_system::linker::ensure_entry_point_slots(
                                    globals.as_mut_slice(),
                                    global_names,
                                    functions,
                                    value_store,
                                );
                            } else {
                                // Module already in cache: file_import added stored_fns and remapped the object.
                                // Feed caller globals so added functions' LoadGlobal find classes/functions (e.g. DevSettings for load_settings).
                                // Skip internal module vars (Null) to preserve isolation.
                                let start_idx = functions_len_before;
                                if functions.len() > start_idx {
                                    let argv_slot_feed = unsafe { (*vm_ptr).get_argv_slot_index() };
                                    let value_for_name = |name: &str| -> Option<Value> {
                                        if let Value::Object(ref module_obj_rc) = &module_object {
                                            if let Some(v) = module_obj_rc.borrow().get(name) {
                                                return Some(v.clone());
                                            }
                                        }
                                        let caller_mods = unsafe { (*vm_ptr).get_modules() };
                                        for (_mod_name, rc) in caller_mods.iter() {
                                            if let Some(v) = rc.borrow().get_export(name) {
                                                return Some(v);
                                            }
                                        }
                                        None
                                    };
                                    let is_internal_module_var = |_name: &str, value: &Value| -> bool {
                                        matches!(value, Value::Null)
                                    };
                                    for i in start_idx..functions.len() {
                                        let chunk = &functions[i].chunk;
                                        for (_idx, name) in &chunk.global_names {
                                            if let Some(value) = value_for_name(name) {
                                                if is_internal_module_var(name, &value) {
                                                    continue;
                                                }
                                                let ok_to_feed = match &value {
                                                    Value::Function(_) | Value::ModuleFunction { .. } | Value::Object(_) => true,
                                                    _ => false,
                                                };
                                                if !ok_to_feed {
                                                    continue;
                                                }
                                                let existing: Vec<usize> = global_names
                                                    .iter()
                                                    .filter(|(_, n)| *n == name)
                                                    .map(|(idx, _)| *idx)
                                                    .collect();
                                                let slot_idx = if existing.is_empty() {
                                                    let idx = globals.len();
                                                    globals.resize(idx + 1, default_global_slot());
                                                    global_names.insert(idx, name.clone());
                                                    idx
                                                } else {
                                                    *existing.iter().min().unwrap()
                                                };
                                                let slot_empty = slot_idx >= globals.len()
                                                    || matches!(
                                                        load_value(globals[slot_idx].resolve_to_value_id(value_store), value_store, heavy_store),
                                                        Value::Null
                                                    );
                                                if slot_empty && Some(slot_idx) != argv_slot_feed {
                                                    if slot_idx >= globals.len() {
                                                        globals.resize(slot_idx + 1, default_global_slot());
                                                    }
                                                    let id = store_value(value.clone(), value_store, heavy_store);
                                                    globals[slot_idx] = GlobalSlot::Heap(id);
                                                }
                                            }
                                        }
                                    }
                                    // Ensure all names from main + function chunks are in global_names.
                                    // For merged chunks: add slots for all names except internal (Null).
                                    let chunks_to_feed: Vec<_> = frames.first().map(|f| &f.function.chunk).into_iter()
                                        .chain(functions.iter().map(|f| &f.chunk))
                                        .collect();
                                    let cache_is_internal = |name: &str| -> bool {
                                        value_for_name(name).as_ref().map_or(true, |v| matches!(v, Value::Null))
                                    };
                                    for (chunk_index, chunk) in chunks_to_feed.iter().enumerate() {
                                        let is_merged_module_chunk = chunk_index >= 1 && (chunk_index - 1) >= start_idx;
                                        for (idx, name) in &chunk.global_names {
                                            if *idx == usize::MAX || name.as_str() == "argv" {
                                                continue;
                                            }
                                            if is_merged_module_chunk && cache_is_internal(name) {
                                                continue;
                                            }
                                            if !global_names.values().any(|n| n == name) {
                                                let new_idx = globals.len();
                                                globals.push(default_global_slot());
                                                global_names.insert(new_idx, name.clone());
                                                debug_println!("[DEBUG ImportFrom cache] Добавлен слот для '{}' в globals[{}]", name, new_idx);
                                            }
                                        }
                                    }
                                    for name in ["create_all", "__constructing_class__"] {
                                        if !global_names.values().any(|n| n == name)
                                            && chunks_to_feed.iter().any(|chunk| chunk.global_names.values().any(|n| n == name))
                                        {
                                            let new_idx = globals.len();
                                            globals.push(default_global_slot());
                                            global_names.insert(new_idx, name.to_string());
                                            debug_println!("[DEBUG ImportFrom cache] Добавлен fallback слот для '{}' в globals[{}]", name, new_idx);
                                        }
                                    }
                                    let global_names_snapshot = global_names.clone();
                                    let argv_slot = unsafe { (*vm_ptr).get_argv_slot_index() };
                                    if std::env::var("DATACODE_DEBUG").is_ok() {
                                        eprintln!("=== IMPORTFROM DEBUG START (cache hit) ===");
                                        if let Some(mf) = frames.first() {
                                            let chunk = &mf.function.chunk;
                                            eprintln!("[IMPORTFROM] Chunk global names (main):");
                                            for (idx, name) in &chunk.global_names {
                                                eprintln!("  chunk idx={} name={}", idx, name);
                                            }
                                        }
                                        eprintln!("[IMPORTFROM] caller_global_names (snapshot) before update:");
                                        for (idx, name) in global_names_snapshot.iter() {
                                            eprintln!("  caller idx={} name={}", idx, name);
                                        }
                                        if !global_names_snapshot.values().any(|n| n == "create_all") {
                                            eprintln!("[WARNING] create_all not found in caller_global_names BEFORE update");
                                        }
                                    }
                                    for i in 0..functions.len() {
                                        crate::vm::module_system::chunk_patcher::update_chunk_indices_from_names(
                                            &mut functions[i].chunk,
                                            &global_names_snapshot,
                                            Some(globals.as_mut_slice()),
                                            Some(value_store),
                                            Some(heavy_store),
                                            argv_slot,
                                            true,
                                        );
                                    }
                                    if let Some(main_frame) = frames.first_mut() {
                                        crate::vm::module_system::chunk_patcher::update_chunk_indices_from_names(
                                            &mut main_frame.function.chunk,
                                            &global_names_snapshot,
                                            Some(globals.as_mut_slice()),
                                            Some(value_store),
                                            Some(heavy_store),
                                            argv_slot,
                                            false, // main chunk: do NOT resolve sentinel
                                        );
                                    }
                                    if std::env::var("DATACODE_DEBUG").is_ok() {
                                        if !global_names_snapshot.values().any(|n| n == "create_all") {
                                            eprintln!("[WARNING] create_all STILL MISSING AFTER update_chunk_indices_from_names");
                                        }
                                        eprintln!("=== IMPORTFROM DEBUG END (cache hit) ===");
                                    }
                                    crate::vm::module_system::linker::ensure_entry_point_slots(
                                        globals.as_mut_slice(),
                                        global_names,
                                        functions,
                                        value_store,
                                    );
                                }
                            }

                            let module_id = store_value(module_object, value_store, heavy_store);
                            // Plan 2.3: do not overwrite a slot that already holds an export (class/constructor); never overwrite argv slot.
                            let existing_idx = global_index_by_name(global_names, &module_name);
                            let overwrite_ok = existing_idx.map(|idx| {
                                Some(idx) != argv_slot_import
                                    && (idx >= globals.len()
                                        || matches!(
                                            crate::vm::store_convert::load_value(globals[idx].resolve_to_value_id(value_store), value_store, heavy_store),
                                            Value::Null
                                        ))
                            }).unwrap_or(true);
                            if let Some(idx) = existing_idx {
                                if overwrite_ok && Some(idx) != argv_slot_import {
                                    if idx < globals.len() {
                                        globals[idx] = GlobalSlot::Heap(module_id);
                                    } else {
                                        globals.resize(idx + 1, default_global_slot());
                                        globals[idx] = GlobalSlot::Heap(module_id);
                                    }
                                } else if Some(idx) == argv_slot_import {
                                    // Keep argv slot; push module to new slot
                                    let new_idx = globals.len();
                                    globals.push(GlobalSlot::Heap(module_id));
                                    global_names.remove(&idx);
                                    global_names.insert(new_idx, module_name.clone());
                                } else if overwrite_ok == false {
                                    let new_idx = globals.len();
                                    globals.push(GlobalSlot::Heap(module_id));
                                    global_names.remove(&idx);
                                    global_names.insert(new_idx, module_name.clone());
                                }
                            } else {
                                let idx = globals.len();
                                globals.push(GlobalSlot::Heap(module_id));
                                global_names.insert(idx, module_name.clone());
                            }
                            loaded_modules.insert(module_name.clone());
                        }
                        Err(load_err) => {
                            match crate::vm::native_loader::try_load_native_module(
                                &module_name,
                                Some(&base_path),
                                natives.len(),
                                abi_natives,
                                loaded_native_libraries,
                            ) {
                                Ok(module_object) => {
                                    let module_value = Value::Object(Rc::new(RefCell::new(module_object)));
                                    let id = store_value(module_value, value_store, heavy_store);
                                    if let Some(idx) = global_index_by_name(global_names, &module_name) {
                                        if Some(idx) != argv_slot_import {
                                            if idx < globals.len() {
                                                globals[idx] = GlobalSlot::Heap(id);
                                            } else {
                                                globals.resize(idx + 1, default_global_slot());
                                                globals[idx] = GlobalSlot::Heap(id);
                                            }
                                        } else {
                                            let new_idx = globals.len();
                                            globals.push(GlobalSlot::Heap(id));
                                            global_names.remove(&idx);
                                            global_names.insert(new_idx, module_name.clone());
                                        }
                                    } else {
                                        let idx = globals.len();
                                        globals.push(GlobalSlot::Heap(id));
                                        global_names.insert(idx, module_name.clone());
                                    }
                                    loaded_modules.insert(module_name.clone());
                                }
                                Err(_) => {
                                    let error = ExceptionHandler::runtime_error_with_source(
                                        &frames,
                                        format!("Failed to load module '{}'", module_name),
                                        load_err,
                                    );
                                    match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error, value_store, heavy_store) {
                                        Ok(()) => return Ok(VMStatus::Continue),
                                        Err(e) => return Err(e),
                                    }
                                }
                            }
                        }
                    }
                } else {
                    // Базовый путь не установлен — локальные .dc модули недоступны
                    let builtins = modules::builtin_modules_list();
                    let error = ExceptionHandler::runtime_error(
                        &frames,
                        format!(
                            "Module '{}' not found. Built-in modules: {}. For local .dc modules (e.g. from config import Config), run a script file from CLI or use run_with_vm_with_args_and_lib(..., base_path).",
                            module_name, builtins
                        ),
                        line,
                    );
                    match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error, value_store, heavy_store) {
                        Ok(()) => return Ok(VMStatus::Continue),
                        Err(e) => return Err(e),
                    }
                }
            }
        }
        
        // Get the module object from globals (deterministic slot by name)
        let module_global_index = if let Some(idx) = global_index_by_name(global_names, &module_name) {
            idx
        } else {
            let error = ExceptionHandler::runtime_error(
                &frames,
                format!("Module {} not found in globals", module_name),
                line,
            );
            match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error, value_store, heavy_store) {
                Ok(()) => return Ok(VMStatus::Continue),
                Err(e) => return Err(e),
            }
        };
        
        if module_global_index >= globals.len() {
            globals.resize(module_global_index + 1, default_global_slot());
        }
        if module_global_index >= globals.len() {
            let error = ExceptionHandler::runtime_error(
                &frames,
                format!("Module {} global index {} out of bounds (globals.len() = {})", 
                    module_name, module_global_index, globals.len()),
                line,
            );
            match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error, value_store, heavy_store) {
                Ok(()) => return Ok(VMStatus::Continue),
                Err(e) => return Err(e),
            }
        }
        let module_id = globals[module_global_index].resolve_to_value_id(value_store);
        let module_value = load_value(module_id, value_store, heavy_store);
        let module_object_rc = match &module_value {
            Value::Object(map_rc) => map_rc.clone(),
            Value::Null => {
                let error = ExceptionHandler::runtime_error(
                    &frames,
                    format!("Module {} is Null - module registration may have failed", module_name),
                    line,
                );
                match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error, value_store, heavy_store) {
                    Ok(()) => return Ok(VMStatus::Continue),
                    Err(e) => return Err(e),
                }
            }
            _ => {
                let error = ExceptionHandler::runtime_error(
                    &frames,
                    format!("Module {} is not an object (found: {:?})", module_name, 
                        std::mem::discriminant(&module_value)),
                    line,
                );
                match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error, value_store, heavy_store) {
                    Ok(()) => return Ok(VMStatus::Continue),
                    Err(e) => return Err(e),
                }
            }
        };
        // Clone the HashMap to avoid borrowing issues - we can now mutate globals
        let module_object = module_object_rc.borrow().clone();
        
        // Collect named imports (no *, no alias) to process in deterministic (sorted) order,
        // so slot assignment does not depend on source order or HashMap iteration.
        let mut named_imports: Vec<(String, Value)> = Vec::new();
        for item_value in &items_array {
            if let Value::String(ref item_str) = item_value {
                if item_str != "*" && !item_str.contains(':') {
                    if let Some(value) = module_object.get(item_str) {
                        named_imports.push((item_str.clone(), value.clone()));
                    }
                }
            }
        }
        named_imports.sort_by(|a, b| a.0.cmp(&b.0));
        
        // Import items: first "*" and aliased in array order, then named in sorted order
        for item_value in items_array {
            match item_value {
                Value::String(item_str) => {
                    if item_str == "*" {
                        // Import all items (iterate in sorted order for determinism)
                        let mut indices_to_set: Vec<(usize, String, ValueId)> = Vec::new();
                        let mut max_index_needed = globals.len();
                        let mut new_indices = Vec::new();
                        let mut star_keys: Vec<_> = module_object.keys().cloned().collect();
                        star_keys.sort();
                        for key in star_keys {
                            let value = module_object.get(&key).unwrap();
                            let global_index = global_index_by_name(global_names, &key);
                            let global_index = match global_index {
                                Some(idx) => idx,
                                None => {
                                    let idx = globals.len() + new_indices.len();
                                    new_indices.push((idx, key.clone()));
                                    idx
                                }
                            };
                            max_index_needed = max_index_needed.max(global_index + 1);
                            let id = store_value(value.clone(), value_store, heavy_store);
                            indices_to_set.push((global_index, key.clone(), id));
                        }
                        if max_index_needed > globals.len() {
                            globals.resize(max_index_needed, default_global_slot());
                        }
                        for (idx, name) in new_indices {
                            global_names.insert(idx, name);
                        }
                        indices_to_set.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
                        for (global_index, _key, id) in indices_to_set {
                            if Some(global_index) != argv_slot_import {
                                globals[global_index] = GlobalSlot::Heap(id);
                            }
                        }
                    } else if item_str.contains(':') {
                        // Aliased import: "name:alias" (process in array order)
                        let parts: Vec<&str> = item_str.split(':').collect();
                        if parts.len() == 2 {
                            let name = parts[0];
                            let alias = parts[1];
                            
                            if let Some(value) = module_object.get(name) {
                                let id = store_value(value.clone(), value_store, heavy_store);
                                let global_index = if let Some(idx) = global_index_by_name(global_names, alias) {
                                    idx
                                } else {
                                    let idx = globals.len();
                                    globals.push(GlobalSlot::Heap(id));
                                    global_names.insert(idx, alias.to_string());
                                    idx
                                };
                                if global_index >= globals.len() {
                                    globals.resize(global_index + 1, default_global_slot());
                                }
                                if Some(global_index) != argv_slot_import {
                                    globals[global_index] = GlobalSlot::Heap(id);
                                }
                            } else {
                                let error = ExceptionHandler::runtime_error_with_type(
                    &frames,
                                    format!("Module '{}' has no attribute '{}'", module_name, name),
                                    line,
                                    crate::common::error::ErrorType::KeyError,
                                );
                                match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error, value_store, heavy_store) {
                                    Ok(()) => continue,
                                    Err(e) => return Err(e),
                                }
                            }
                        }
                    } else {
                        // Named import: processed below in sorted order (skip here)
                        continue;
                    }
                }
                _ => {
                    let error = ExceptionHandler::runtime_error(
                &frames,
                        "ImportFrom item must be a string".to_string(),
                        line,
                    );
                    match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error, value_store, heavy_store) {
                        Ok(()) => continue,
                        Err(e) => return Err(e),
                    }
                }
            }
        }
        // Process named imports in deterministic (sorted) order
        for (item_str, value) in named_imports {
            debug_println!("[DEBUG ImportFrom] Импортируем '{}' из модуля '{}'", item_str, module_name);
            debug_println!("[DEBUG ImportFrom] Доступные ключи в модуле: {:?}", module_object.keys().collect::<Vec<_>>());
            
            debug_println!("[DEBUG ImportFrom] Найден '{}' в модуле, тип: {:?}", item_str, match &value {
                                Value::Object(_) => "Object",
                                Value::Function(_) | Value::ModuleFunction { .. } => "Function",
                                Value::Null => "Null",
                                _ => "Other",
                            });
                            // ModuleFunction is stored as-is (resolved at Call). Legacy: remap Value::Function when __start_function_index present.
                            let value_to_store = match value {
                                Value::ModuleFunction { .. } => value.clone(),
                                Value::Function(fn_idx) => {
                                    if let Some(Value::Number(start)) = module_object.get("__start_function_index") {
                                        let start_u = *start as usize;
                                        if fn_idx >= start_u {
                                            value.clone()
                                        } else {
                                            Value::Function(start_u + fn_idx)
                                        }
                                    } else {
                                        value.clone()
                                    }
                                }
                                _ => value.clone(),
                            };
                            let is_function = matches!(value_to_store, Value::Function(_) | Value::ModuleFunction { .. });
                            let indices = global_indices_by_name(global_names, &item_str);
                            let indices_to_update: Vec<usize> = if indices.is_empty() {
                                vec![globals.len()]
                            } else {
                                indices
                            };
                            let has_merge = module_object.get("__start_function_index").is_some();
                            let id = store_value(value_to_store, value_store, heavy_store);
                            for &global_index in &indices_to_update {
                                // When we just merged (__start_function_index present), per-item is source of truth for
                                // Functions only: always write remapped function so explicitly imported names get the correct one.
                                // Do not overwrite existing Object (e.g. class from merge) so we keep the class and its model_config
                                // (required for Settings subclasses: load_env uses model_config.env_file from the class).
                                // When module came from cache (no __start_function_index), skip overwriting existing Function.
                                let skip_overwrite = if has_merge {
                                    if is_function {
                                        false
                                    } else if global_index < globals.len() {
                                        let cur_id = globals[global_index].resolve_to_value_id(value_store);
                                        matches!(load_value(cur_id, value_store, heavy_store), Value::Object(_))
                                    } else {
                                        false
                                    }
                                } else if global_index < globals.len() {
                                    let cur_id = globals[global_index].resolve_to_value_id(value_store);
                                    matches!(load_value(cur_id, value_store, heavy_store), Value::Function(_))
                                } else {
                                    false
                                };
                                if skip_overwrite {
                                    debug_println!("[DEBUG ImportFrom] Пропуск перезаписи '{}' в globals[{}] (в слоте уже функция из merge)", item_str, global_index);
                                } else if Some(global_index) == argv_slot_import {
                                    debug_println!("[DEBUG ImportFrom] Пропуск перезаписи слота argv (globals[{}])", global_index);
                                } else {
                                    if global_index >= globals.len() {
                                        globals.resize(global_index + 1, default_global_slot());
                                        global_names.insert(global_index, item_str.clone());
                                        debug_println!("[DEBUG ImportFrom] Создан новый глобальный индекс {} для '{}'", global_index, item_str);
                                    }
                                    globals[global_index] = GlobalSlot::Heap(id);
                                    debug_println!("[DEBUG ImportFrom] '{}' установлен в globals[{}]", item_str, global_index);
                                }
                            }
                            
                            // Если импортируется класс (объект с метаданными класса), также импортируем все конструкторы
                            // Конструкторы имеют формат ClassName::new_<arity>
                            if let Value::Object(class_obj_rc) = value {
                                let class_obj = class_obj_rc.borrow();
                                debug_println!("[DEBUG ImportFrom] Проверяем, является ли '{}' классом...", item_str);
                                // Проверяем, что это класс (имеет метаданные __class_name)
                                if class_obj.contains_key("__class_name") {
                                    debug_println!("[DEBUG ImportFrom] '{}' является классом! Импортируем конструкторы...", item_str);
                                    let start_function_index = if let Some(module_global_idx) = global_index_by_name(global_names, &module_name) {
                                        if module_global_idx < globals.len() {
                                            let mid = globals[module_global_idx].resolve_to_value_id(value_store);
                                            let mod_val = load_value(mid, value_store, heavy_store);
                                            if let Value::Object(module_obj_rc) = &mod_val {
                                                let module_obj = module_obj_rc.borrow();
                                                if let Some(Value::Number(idx)) = module_obj.get("__start_function_index") {
                                                    debug_println!("[DEBUG ImportFrom] Найден start_function_index={} для модуля '{}'", *idx, module_name);
                                                    *idx as usize
                                                } else {
                                                    debug_println!("[DEBUG ImportFrom] WARNING: start_function_index не найден в модуле '{}', используем 0", module_name);
                                                    0
                                                }
                                            } else {
                                                debug_println!("[DEBUG ImportFrom] WARNING: Модуль '{}' не является объектом", module_name);
                                                0
                                            }
                                        } else {
                                            debug_println!("[DEBUG ImportFrom] WARNING: Индекс модуля {} выходит за границы globals (len={})", module_global_idx, globals.len());
                                            0
                                        }
                                    } else {
                                        debug_println!("[DEBUG ImportFrom] WARNING: Модуль '{}' не найден в global_names", module_name);
                                        0
                                    };
                                    
                                    // Импортируем все конструкторы этого класса из модуля
                                    let constructor_prefix = format!("{}::new_", item_str);
                                    debug_println!("[DEBUG ImportFrom] Ищем конструкторы с префиксом '{}'", constructor_prefix);
                                    let mut found_constructors = 0;
                                    for (key, val) in module_object.iter() {
                                        if key.starts_with(&constructor_prefix) {
                                            found_constructors += 1;
                                            debug_println!("[DEBUG ImportFrom] Найден конструктор: {}", key);
                                            // Обновляем индекс функции в конструкторе
                                            let (updated_val, new_function_index) = match val {
                                                Value::ModuleFunction { .. } => (val.clone(), 0),
                                                Value::Function(function_index) => {
                                                    let new_index = start_function_index + *function_index;
                                                    debug_println!("[DEBUG ImportFrom] Обновляем индекс функции: {} -> {}", function_index, new_index);
                                                    (Value::Function(new_index), new_index)
                                                }
                                                _ => {
                                                    debug_println!("[DEBUG ImportFrom] WARNING: Конструктор {} не является функцией", key);
                                                    (val.clone(), 0)
                                                }
                                            };
                                            
                                            let updated_id = store_value(updated_val.clone(), value_store, heavy_store);
                                            let constructor_global_index = if let Some(idx) = global_index_by_name(global_names, key) {
                                                debug_println!("[DEBUG ImportFrom] Конструктор '{}' уже существует в globals с индексом {}, обновляем индекс функции", key, idx);
                                                if Some(idx) != argv_slot_import {
                                                    if idx < globals.len() {
                                                        let rid = globals[idx].resolve_to_value_id(value_store);
                                                        if let Value::Function(old_fn_idx) = &load_value(rid, value_store, heavy_store) {
                                                            debug_println!("[DEBUG ImportFrom] Старый индекс функции: {}, новый индекс функции: {} (функции из модуля добавлены в VM)", old_fn_idx, new_function_index);
                                                        }
                                                        globals[idx] = GlobalSlot::Heap(updated_id);
                                                    } else {
                                                        globals.resize(idx + 1, default_global_slot());
                                                        globals[idx] = GlobalSlot::Heap(updated_id);
                                                    }
                                                }
                                                idx
                                            } else {
                                                let idx = globals.len();
                                                globals.push(GlobalSlot::Heap(updated_id));
                                                global_names.insert(idx, key.clone());
                                                debug_println!("[DEBUG ImportFrom] Создан новый глобальный индекс {} для конструктора '{}'", idx, key);
                                                idx
                                            };
                                            if Some(constructor_global_index) != argv_slot_import {
                                                if constructor_global_index >= globals.len() {
                                                    globals.resize(constructor_global_index + 1, default_global_slot());
                                                }
                                                globals[constructor_global_index] = GlobalSlot::Heap(updated_id);
                                            }
                                            debug_println!("[DEBUG ImportFrom] Конструктор '{}' установлен в globals[{}] с индексом функции {}", key, constructor_global_index, new_function_index);
                                        }
                                    }
                                    debug_println!("[DEBUG ImportFrom] Всего найдено конструкторов: {}", found_constructors);
                                    // Методы класса живут только внутри объекта класса (getBalance, deposit и т.д.), не экспортируем их в globals при ImportFrom.
                                } else {
                                    debug_println!("[DEBUG ImportFrom] '{}' не является классом (нет ключа __class_name)", item_str);
                                }
                            }
        }
        
        debug_println!(
            "[DEBUG ImportFrom] после установки: global_names 75..80: {:?}",
            (75..80).filter_map(|i| global_names.get(&i).map(|n| (i, n.as_str()))).collect::<Vec<_>>()
        );
        // Ensure all names from main + function chunks are in global_names before update_chunk_indices,
        // so we don't get "no match" and LoadGlobal/StoreGlobal get correct remapping.
        // Module isolation: skip only internal plain vars (e.g. "settings" = Null).
        const UNDEFINED_GLOBAL_SENTINEL: usize = usize::MAX;
        let chunks_to_feed: Vec<_> = frames.first().map(|f| &f.function.chunk).into_iter()
            .chain(functions.iter().map(|f| &f.chunk))
            .collect();
        let is_internal_module_var = |name: &str| -> bool {
            if imported_names.contains(name) || imported_names.contains("*") {
                return false;
            }
            module_object.get(name).map_or(false, |v| matches!(v, Value::Null))
        };
        for (chunk_index, chunk) in chunks_to_feed.iter().enumerate() {
            let is_merged_module_chunk = chunk_index >= 1
                && functions.get(chunk_index - 1).and_then(|f| f.module_name.as_deref()) == Some(module_name.as_str());
            for (idx, name) in &chunk.global_names {
                if *idx == UNDEFINED_GLOBAL_SENTINEL || name.as_str() == "argv" {
                    continue;
                }
                if is_merged_module_chunk && is_internal_module_var(name) {
                    continue;
                }
                if !global_names.values().any(|n| n == name) {
                    let new_idx = globals.len();
                    globals.push(default_global_slot());
                    global_names.insert(new_idx, name.clone());
                    debug_println!("[DEBUG ImportFrom] Добавлен слот для '{}' в globals[{}] (отсутствовал в caller)", name, new_idx);
                }
            }
        }
        // Fallback: ensure create_all and __constructing_class__ are in global_names when any
        // chunk references them, so update_chunk_indices_from_names can map LoadGlobal correctly.
        for name in ["create_all", "__constructing_class__"] {
            if !global_names.values().any(|n| n == name)
                && chunks_to_feed.iter().any(|chunk| chunk.global_names.values().any(|n| n == name))
            {
                let new_idx = globals.len();
                globals.push(default_global_slot());
                global_names.insert(new_idx, name.to_string());
                debug_println!("[DEBUG ImportFrom] Добавлен fallback слот для '{}' в globals[{}]", name, new_idx);
            }
        }
        // After importing items (builtin or file), update main chunk's LoadGlobal/StoreGlobal
        // to the current global_names so subsequent instructions see the correct slots.
        let argv_slot = unsafe { (*vm_ptr).get_argv_slot_index() };
        if std::env::var("DATACODE_DEBUG").is_ok() {
            eprintln!("=== IMPORTFROM DEBUG START (per-item) ===");
            if let Some(mf) = frames.first() {
                let chunk = &mf.function.chunk;
                eprintln!("[IMPORTFROM] Chunk global names:");
                for (idx, name) in &chunk.global_names {
                    eprintln!("  chunk idx={} name={}", idx, name);
                }
            }
            eprintln!("[IMPORTFROM] caller_global_names before update:");
            for (idx, name) in global_names.iter() {
                eprintln!("  caller idx={} name={}", idx, name);
            }
            if !global_names.values().any(|n| n == "create_all") {
                eprintln!("[WARNING] create_all not found in caller_global_names BEFORE update");
            }
        }
        if let Some(main_frame) = frames.first_mut() {
            crate::vm::module_system::chunk_patcher::update_chunk_indices_from_names(
                &mut main_frame.function.chunk,
                global_names,
                Some(globals.as_mut_slice()),
                Some(value_store),
                Some(heavy_store),
                argv_slot,
                false, // main chunk: do NOT resolve sentinel (module isolation)
            );
        }
        if std::env::var("DATACODE_DEBUG").is_ok() {
            if !global_names.values().any(|n| n == "create_all") {
                eprintln!("[WARNING] create_all STILL MISSING AFTER update_chunk_indices_from_names");
            }
            eprintln!("=== IMPORTFROM DEBUG END (per-item) ===");
        }
        // Re-patch all function chunks with final global_names after per-item imports.
        // Otherwise constructors from the module (e.g. ProdSettings::new_0) keep LoadGlobal(sentinel)
        // mapped to pre-import indices and may load the wrong class (e.g. DevSettings instead of ProdSettings).
        for f in functions.iter_mut() {
            crate::vm::module_system::chunk_patcher::update_chunk_indices_from_names(
                &mut f.chunk,
                global_names,
                Some(globals.as_mut_slice()),
                Some(value_store),
                Some(heavy_store),
                argv_slot,
                true,
            );
        }
        // Re-establish argv slot so next LoadGlobal(argv) sees script args. Prefer SCRIPT_ARGV_VALUE_ID (survives nested module run), then VM, then saved at start of ImportFrom.
        let argv_id_to_restore = unsafe { (*vm_ptr).get_current_argv_value_id() }
            .or_else(|| crate::vm::run_context::RunContext::get_script_argv_value_id())
            .or(saved_argv_value_id)
            .or_else(crate::vm::run_context::RunContext::get_argv_value_id);
        if let Some(slot_idx) = argv_slot {
            if let Some(argv_id) = argv_id_to_restore {
                if slot_idx >= globals.len() {
                    globals.resize(slot_idx + 1, default_global_slot());
                }
                globals[slot_idx] = GlobalSlot::Heap(argv_id);
                crate::vm::run_context::RunContext::set_restored_script_argv_after_import(Some(argv_id));
            }
        }
        return Ok(VMStatus::Continue);
}
