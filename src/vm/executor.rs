// Opcode execution for VM (Stage 1: stack/globals as Vec<ValueId>; one borrow store per instruction)

use crate::debug_println;
use crate::bytecode::OpCode;
use crate::common::{error::LangError, value::Value, value_store::{ValueCell, ValueId, ValueStore, NULL_VALUE_ID}, TaggedValue};
use crate::common::table::Table;
use crate::vm::types::VMStatus;
use crate::vm::frame::CallFrame;
use crate::vm::exceptions::ExceptionHandler;
use crate::vm::operations;
use crate::vm::stack;
use crate::vm::modules;
use crate::vm::vm::VM_CALL_CONTEXT;
use crate::vm::global_slot::{GlobalSlot, default_global_slot};
use crate::vm::module_object::BUILTIN_END;
use crate::vm::store_convert::{store_value, store_value_arena, load_value, update_cell_if_mutable, tagged_to_value_id, tagged_to_value_id_arena, slot_to_value};
use crate::vm::heavy_store::HeavyStore;
use crate::ml::tensor::Tensor;
use crate::common::error::ErrorType;
use crate::vm::types::{ExplicitRelation, ExplicitPrimaryKey};
use std::rc::Rc;
use std::cell::RefCell;
use std::fmt::Write;

/// Deterministic global slot by name (min index when multiple; stable across HashMap iteration).
fn global_index_by_name(global_names: &std::collections::BTreeMap<usize, String>, name: &str) -> Option<usize> {
    global_names
        .iter()
        .filter(|(_, n)| n.as_str() == name)
        .map(|(idx, _)| *idx)
        .min()
}

/// All global slot indices for a name (for updating every binding of the same name on import).
fn global_indices_by_name(global_names: &std::collections::BTreeMap<usize, String>, name: &str) -> Vec<usize> {
    let mut indices: Vec<usize> = global_names
        .iter()
        .filter(|(_, n)| n.as_str() == name)
        .map(|(idx, _)| *idx)
        .collect();
    indices.sort_unstable();
    indices
}

/// Returns [class_name, superclass, ...] for VM protected access checks (uses load_value for Object).
/// Stops at cycle or missing __superclass to avoid infinite loop.
fn get_superclass_chain(
    globals: &mut [GlobalSlot],
    global_names: &std::collections::BTreeMap<usize, String>,
    class_name: &str,
    store: &mut ValueStore,
    heap: &HeavyStore,
) -> Vec<String> {
    use std::collections::HashSet;
    let mut chain = vec![class_name.to_string()];
    let mut seen = HashSet::new();
    seen.insert(class_name.to_string());
    let mut current = class_name.to_string();
    loop {
        let super_name_opt = global_names
            .iter()
            .find(|(_, name)| name.as_str() == current)
            .and_then(|(idx, _)| {
                if *idx < globals.len() {
                    let id = globals[*idx].resolve_to_value_id(store);
                    let v = load_value(id, store, heap);
                    if let Value::Object(rc) = &v {
                        let map = rc.borrow();
                        map.get("__superclass").cloned()
                    } else {
                        None
                    }
                } else {
                    None
                }
            });
        let super_name = match super_name_opt {
            Some(Value::String(s)) => s,
            _ => break,
        };
        if !seen.insert(super_name.clone()) {
            // Cycle in class hierarchy — stop to avoid infinite loop
            break;
        }
        chain.push(super_name.clone());
        current = super_name;
    }
    chain
}

/// Execute one step of the VM - get next instruction and execute it
pub fn step(
    frames: &mut Vec<CallFrame>,
) -> Result<Option<(OpCode, usize)>, LangError> {
    loop {
        let frame = match frames.last_mut() {
            Some(f) => f,
            None => return Ok(None),
        };

        if frame.ip >= frame.function.chunk.code.len() {
            // Frame exhausted (e.g. empty method body); pop and continue with caller
            frames.pop();
            continue;
        }

        let ip = frame.ip;
        let instruction = frame.function.chunk.code[ip].clone();
        let line = frame.function.chunk.get_line(ip);
        frame.ip += 1;

        return Ok(Some((instruction, line)));
    }
}

/// Execute a single instruction
/// Returns VMStatus indicating what to do next. vm_ptr used for VM_CALL_CONTEXT and module loading.
pub fn execute_instruction(
    instruction: OpCode,
    line: usize,
    stack: &mut Vec<TaggedValue>,
    frames: &mut Vec<CallFrame>,
    globals: &mut Vec<GlobalSlot>,
    global_names: &mut std::collections::BTreeMap<usize, String>,
    explicit_global_names: &std::collections::BTreeMap<usize, String>,
    functions: &mut Vec<crate::bytecode::Function>,
    natives: &mut Vec<fn(&[Value]) -> Value>,
    exception_handlers: &mut Vec<ExceptionHandler>,
    error_type_table: &mut Vec<String>,
    explicit_relations: &mut Vec<crate::vm::types::ExplicitRelation>,
    explicit_primary_keys: &mut Vec<crate::vm::types::ExplicitPrimaryKey>,
    loaded_modules: &mut std::collections::HashSet<String>,
    abi_natives: &mut Vec<crate::abi::NativeAbiFn>,
    loaded_native_libraries: &mut Vec<libloading::Library>,
    value_store: &mut crate::common::ValueStore,
    heavy_store: &mut crate::vm::heavy_store::HeavyStore,
    native_args_buffer: &mut Vec<Value>,
    reusable_native_arg_ids: &mut Vec<crate::common::value_store::ValueId>,
    reusable_all_popped: &mut Vec<Value>,
    vm_ptr: *mut crate::vm::vm::Vm,
) -> Result<VMStatus, LangError> {
    #[cfg(feature = "profile")]
    crate::vm::profile::record_opcode();
    #[cfg(feature = "profile")]
    crate::vm::profile::set_current_opcode(&instruction);

    /// Pop one TaggedValue and convert to ValueId (for opcodes that need store id).
    fn pop_to_value_id(
        stack: &mut Vec<TaggedValue>,
        frames: &mut Vec<CallFrame>,
        exception_handlers: &mut Vec<ExceptionHandler>,
        value_store: &mut ValueStore,
        heavy_store: &mut HeavyStore,
    ) -> Result<ValueId, LangError> {
        let tv = stack::pop(stack, frames, exception_handlers, value_store, heavy_store)?;
        Ok(tagged_to_value_id(tv, value_store))
    }
    let frame = frames.last_mut().unwrap();
    let current_ip = frame.ip - 1; // IP уже инкрементирован в step()

    // Логирование выполнения конструктора
    let is_constructor = frame.function.name.contains("::new_");
    
    if is_constructor && crate::common::debug::is_debug_enabled() {
        let is_return = matches!(instruction, OpCode::Return);
        debug_println!("[DEBUG executor constructor] '{}' IP {} line {}: {:?} (stack len {})",
            frame.function.name, current_ip, line, instruction, stack.len());
        if is_return && !stack.is_empty() {
            let return_tv = stack[stack.len() - 1];
            let return_id = tagged_to_value_id(return_tv, value_store);
            let return_value = load_value(return_id, value_store, heavy_store);
            let val_type = match &return_value {
                Value::Object(obj_rc) => {
                    let map = obj_rc.borrow();
                    let keys: Vec<String> = map.keys().cloned().collect();
                    format!("Object с ключами: {:?}", keys)
                },
                _ => format!("{:?}", return_value),
            };
            debug_println!("[DEBUG executor constructor] Возвращаемое значение: {}", val_type);
        }
    }
    
    match instruction {
                OpCode::Import(module_index) => {
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
                                        // Patch exported function indices so mymod.one() calls the right function in the main VM
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
                    return Err(load_dc_err.map_or_else(
                        || LangError::runtime_error(
                            format!("Module '{}' not found (built-in, .dc file, or native module)", module_name),
                            line,
                        ),
                        |e| LangError::runtime_error_with_source(
                            format!("Failed to load module '{}'", module_name),
                            e,
                        ),
                    ));
                }
                OpCode::ImportFrom(module_index, items_index) => {
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
                                            debug_println!("[DEBUG ImportFrom] Функций в модуле: {}, start_idx: {}", module_vm.get_functions().len(), start_idx);
                                            // Register submodules first, then the top-level module, so the top-level module_id never collides with
                                            // module VM registry indices (0, 1, ...). That way replace_function_with_module_function_in_exports
                                            // uses a distinct id and remap only rewrites submodule-origin ModuleFunctions.
                                            let (module_id, submodule_old_to_new): (usize, std::collections::HashMap<usize, usize>) = {
                                                let sub_infos: Vec<_> = module_vm.get_module_registry().iter().cloned().collect();
                                                let mut reg = unsafe { (*vm_ptr).get_module_registry_mut() };
                                                let mut map = std::collections::HashMap::new();
                                                for (old_id, info) in sub_infos.iter().enumerate() {
                                                    reg.push(crate::vm::vm::ModuleInfo {
                                                        name: info.name.clone(),
                                                        function_offset: start_idx + info.function_offset,
                                                        function_count: info.function_count,
                                                    });
                                                    map.insert(old_id, reg.len() - 1);
                                                }
                                                reg.push(crate::vm::vm::ModuleInfo {
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
                                            {
                                                let mods = module_vm.get_modules();
                                                let mut caller_mods = unsafe { (*vm_ptr).get_modules_mut() };
                                                for (k, v) in mods.iter() {
                                                    caller_mods.insert(k.clone(), v.clone());
                                                    debug_println!("[DEBUG ImportFrom] merged submodule '{}' into caller", k);
                                                }
                                            }
                                            // Feed caller globals with names that merged module chunks reference (from module object or caller's merged modules),
                                            // so update_chunk_indices can map LoadGlobal to filled slots (fixes "Undefined variable: DatabaseConfig::new_1" in prod).
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
                                                for i in start_idx..functions.len() {
                                                    let chunk = &functions[i].chunk;
                                                    for (_idx, name) in &chunk.global_names {
                                                        if let Some(value) = value_for_name(name) {
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
                                                            if slot_empty {
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
                                            // Module isolation: do NOT merge module globals into caller. Only requested items are added below (per-item loop).
                                            let global_names_snapshot = global_names.clone();
                                            let argv_slot = unsafe { (*vm_ptr).get_argv_slot_index() };
                                            for i in 0..functions.len() {
                                                crate::vm::vm::Vm::update_chunk_indices_from_names(
                                                    &mut functions[i].chunk,
                                                    &global_names_snapshot,
                                                    Some(globals.as_mut_slice()),
                                                    Some(value_store),
                                                    Some(heavy_store),
                                                    argv_slot,
                                                );
                                            }
                                            if let Some(main_frame) = frames.first_mut() {
                                                crate::vm::vm::Vm::update_chunk_indices_from_names(
                                                    &mut main_frame.function.chunk,
                                                    &global_names_snapshot,
                                                    Some(globals.as_mut_slice()),
                                                    Some(value_store),
                                                    Some(heavy_store),
                                                    argv_slot,
                                                );
                                            }
                                            crate::vm::vm::Vm::ensure_entry_point_slots(
                                                globals.as_mut_slice(),
                                                global_names,
                                                functions,
                                                value_store,
                                            );
                                        } else {
                                            // Module already in cache: file_import added stored_fns and remapped the object.
                                            // Feed caller globals and update chunk indices so added functions' LoadGlobal refer to caller slots (fixes "Function index N out of bounds").
                                            let start_idx = functions_len_before;
                                            if functions.len() > start_idx {
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
                                                for i in start_idx..functions.len() {
                                                    let chunk = &functions[i].chunk;
                                                    for (_idx, name) in &chunk.global_names {
                                                        if let Some(value) = value_for_name(name) {
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
                                                            if slot_empty {
                                                                if slot_idx >= globals.len() {
                                                                    globals.resize(slot_idx + 1, default_global_slot());
                                                                }
                                                                let id = store_value(value.clone(), value_store, heavy_store);
                                                                globals[slot_idx] = GlobalSlot::Heap(id);
                                                            }
                                                        }
                                                    }
                                                }
                                                let global_names_snapshot = global_names.clone();
                                                let argv_slot = unsafe { (*vm_ptr).get_argv_slot_index() };
                                                for i in 0..functions.len() {
                                                    crate::vm::vm::Vm::update_chunk_indices_from_names(
                                                        &mut functions[i].chunk,
                                                        &global_names_snapshot,
                                                        Some(globals.as_mut_slice()),
                                                        Some(value_store),
                                                        Some(heavy_store),
                                                        argv_slot,
                                                    );
                                                }
                                                if let Some(main_frame) = frames.first_mut() {
                                                    crate::vm::vm::Vm::update_chunk_indices_from_names(
                                                        &mut main_frame.function.chunk,
                                                        &global_names_snapshot,
                                                        Some(globals.as_mut_slice()),
                                                        Some(value_store),
                                                        Some(heavy_store),
                                                        argv_slot,
                                                    );
                                                }
                                                crate::vm::vm::Vm::ensure_entry_point_slots(
                                                    globals.as_mut_slice(),
                                                    global_names,
                                                    functions,
                                                    value_store,
                                                );
                                            }
                                        }

                                        let module_id = store_value(module_object, value_store, heavy_store);
                                        // Plan 2.3: do not overwrite a slot that already holds an export (class/constructor)
                                        let existing_idx = global_index_by_name(global_names, &module_name);
                                        let overwrite_ok = existing_idx.map(|idx| {
                                            idx >= globals.len()
                                                || matches!(
                                                    crate::vm::store_convert::load_value(globals[idx].resolve_to_value_id(value_store), value_store, heavy_store),
                                                    Value::Null
                                                )
                                        }).unwrap_or(true);
                                        if let Some(idx) = existing_idx {
                                            if overwrite_ok {
                                                if idx < globals.len() {
                                                    globals[idx] = GlobalSlot::Heap(module_id);
                                                } else {
                                                    globals.resize(idx + 1, default_global_slot());
                                                    globals[idx] = GlobalSlot::Heap(module_id);
                                                }
                                            } else {
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
                                                    if idx < globals.len() {
                                                        globals[idx] = GlobalSlot::Heap(id);
                                                    } else {
                                                        globals.resize(idx + 1, default_global_slot());
                                                        globals[idx] = GlobalSlot::Heap(id);
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
                                        globals[global_index] = GlobalSlot::Heap(id);
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
                                            globals[global_index] = GlobalSlot::Heap(id);
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
                                                            idx
                                                        } else {
                                                            let idx = globals.len();
                                                            globals.push(GlobalSlot::Heap(updated_id));
                                                            global_names.insert(idx, key.clone());
                                                            debug_println!("[DEBUG ImportFrom] Создан новый глобальный индекс {} для конструктора '{}'", idx, key);
                                                            idx
                                                        };
                                                        if constructor_global_index >= globals.len() {
                                                            globals.resize(constructor_global_index + 1, default_global_slot());
                                                        }
                                                        globals[constructor_global_index] = GlobalSlot::Heap(updated_id);
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
                    // After importing items (builtin or file), update main chunk's LoadGlobal/StoreGlobal
                    // to the current global_names so subsequent instructions see the correct slots.
                    let argv_slot = unsafe { (*vm_ptr).get_argv_slot_index() };
                    
                    if let Some(main_frame) = frames.first_mut() {
                        crate::vm::vm::Vm::update_chunk_indices_from_names(
                            &mut main_frame.function.chunk,
                            global_names,
                            Some(globals.as_mut_slice()),
                            Some(value_store),
                            Some(heavy_store),
                            argv_slot,
                        );
                    }
                    // Re-patch all function chunks with final global_names after per-item imports.
                    // Otherwise constructors from the module (e.g. ProdSettings::new_0) keep LoadGlobal(sentinel)
                    // mapped to pre-import indices and may load the wrong class (e.g. DevSettings instead of ProdSettings).
                    for f in functions.iter_mut() {
                        crate::vm::vm::Vm::update_chunk_indices_from_names(
                            &mut f.chunk,
                            global_names,
                            Some(globals.as_mut_slice()),
                            Some(value_store),
                            Some(heavy_store),
                            argv_slot,
                        );
                    }
                    return Ok(VMStatus::Continue);
                }
                OpCode::Constant(index) => {
                    let tv = frame.constant_tagged.get(index)
                        .and_then(|opt| *opt)
                        .unwrap_or_else(|| TaggedValue::from_heap(frame.constant_ids[index]));
                    stack::push(stack, tv);
                    return Ok(VMStatus::Continue);
                }
                OpCode::LoadLocal(index) => {
                    let current_ip = frame.ip - 1;
                    if frame.load_local_cache_ip == Some(current_ip) && frame.load_local_cache_slot == Some(index) {
                        if let Some(tv) = frame.load_local_cache_tagged {
                            stack::push(stack, tv);
                            return Ok(VMStatus::Continue);
                        }
                    }
                    let frame = frames.last_mut().unwrap();
                    if index >= frame.slots.len() {
                        frame.ensure_slot(index);
                    }
                    let tv = frame.slots[index];
                    stack::push(stack, tv);
                    {
                        let frame = frames.last_mut().unwrap();
                        frame.load_local_cache_ip = Some(current_ip);
                        frame.load_local_cache_slot = Some(index);
                        frame.load_local_cache_tagged = Some(tv);
                    }
                    return Ok(VMStatus::Continue);
                }
                OpCode::StoreLocal(index) => {
                    let tv = stack::pop(stack, frames, exception_handlers, value_store, heavy_store)?;
                    let frame = frames.last_mut().unwrap();
                    if frame.load_local_cache_slot == Some(index) {
                        frame.load_local_cache_slot = None;
                    }
                    if index >= frame.slots.len() {
                        frame.slots.resize(index + 1, TaggedValue::null());
                    }
                    frame.slots[index] = tv;
                    if cfg!(debug_assertions) {
                        let val = slot_to_value(tv, value_store, heavy_store);
                        if let Value::Object(obj_rc) = &val {
                            let _obj_ptr = Rc::as_ptr(obj_rc);
                            let f = frames.last().unwrap();
                            let is_constructor = f.function.name.contains("::new_");
                            let current_ip = f.ip - 1;
                            if is_constructor {
                                let map = obj_rc.borrow();
                                debug_println!("[DEBUG StoreLocal] constructor '{}' IP {} slot {}: Object ({} keys)", f.function.name, current_ip, index, map.len());
                            }
                        }
                    }
                    return Ok(VMStatus::Continue);
                }
                OpCode::LoadGlobal(index) => {
                    // __constructing_class__ must always come from caller's globals (set by executor before constructor call),
                    // not from module namespace — otherwise in a super() chain we'd load the current class (e.g. ConfigApp)
                    // instead of the leaf class (e.g. ProdSettings), and model_config would be null.
                    let is_constructing_class_load = frame.function.name.contains("::new_")
                        && frame.function.chunk.global_names.get(&index).map(|n| n.as_str()) == Some("__constructing_class__");
                    // Per-module isolation: resolve from frame's module namespace (and builtins) when frame has module_name.
                    if let Some(ref mod_name) = frame.module_name {
                        if is_constructing_class_load {
                            // Fall through to globals path so we use caller's __constructing_class__ (leaf class), not module export.
                        } else if index < BUILTIN_END {
                            let builtins = unsafe { (*vm_ptr).get_builtins() };
                            if index < builtins.len() {
                                let (inline_tv, heap_id_opt) = match &builtins[index] {
                                    GlobalSlot::Inline(tv) => (Some(*tv), None),
                                    GlobalSlot::Heap(id) => (None, Some(*id)),
                                };
                                if let Some(tv) = inline_tv {
                                    stack::push(stack, tv);
                                } else {
                                    let id = heap_id_opt.unwrap();
                                    stack::push_id(stack, id);
                                }
                                return Ok(VMStatus::Continue);
                            }
                        } else {
                            let module_rc = {
                                let modules = unsafe { (*vm_ptr).get_modules() };
                                modules.get(mod_name).cloned()
                            };
                            if let Some(rc) = module_rc {
                                let var_name = frame.function.chunk.global_names.get(&index).map(String::as_str);
                                let class_name = frame.function.name.split("::").next().unwrap_or("");
                                let value_opt = {
                                    let module = rc.borrow();
                                    if let Some(name) = var_name {
                                        module.get_export(name)
                                            .or_else(|| {
                                                if name == "__constructing_class__" && frame.function.name.contains("::new_") {
                                                    module.get_export(class_name)
                                                } else {
                                                    None
                                                }
                                            })
                                            .and_then(|v| {
                                                if name == "__constructing_class__" && frame.function.name.contains("::new_") && matches!(&v, Value::Null) {
                                                    module.get_export(class_name)
                                                } else {
                                                    Some(v)
                                                }
                                            })
                                    } else {
                                        None
                                    }
                                };
                                let value_opt = value_opt
                                    .and_then(|v| {
                                        if var_name == Some("__constructing_class__") && frame.function.name.contains("::new_") && matches!(&v, Value::Null) {
                                            None
                                        } else {
                                            Some(v)
                                        }
                                    })
                                    .or_else(|| {
                                        if var_name == Some("__constructing_class__") && frame.function.name.contains("::new_") {
                                            let modules = unsafe { (*vm_ptr).get_modules() };
                                            for try_name in [class_name, "config", "dev_config", "prod_config"] {
                                                if let Some(rc) = modules.get(try_name) {
                                                    if let Some(v) = rc.borrow().get_export(class_name) {
                                                        return Some(v);
                                                    }
                                                }
                                            }
                                            // Fallback: search all loaded modules for the class (e.g. dotted module "core.config" shares namespace with "config")
                                            for (_mod_key, rc) in modules.iter() {
                                                if let Some(v) = rc.borrow().get_export(class_name) {
                                                    return Some(v);
                                                }
                                            }
                                        }
                                        None
                                    });
                                if let Some(value) = value_opt {
                                    let id = store_value(value, value_store, heavy_store);
                                    stack::push_id(stack, id);
                                    return Ok(VMStatus::Continue);
                                }
                                if !(var_name == Some("__constructing_class__") && frame.function.name.contains("::new_")) {
                                    let err_name = var_name.unwrap_or("?").to_string();
                                    return Err(LangError::runtime_error(
                                        format!("Undefined variable: {}", err_name),
                                        line,
                                    ));
                                }
                            }
                        }
                    }

                    let mut effective_index = index;
                    if index >= globals.len() {
                        // Resolve by name from current chunk (e.g. merged module function with unpatched sentinel)
                        if let Some(var_name) = frame.function.chunk.global_names.get(&index) {
                            let real_idx_opt = global_names.iter()
                                .filter(|(_, n)| n.as_str() == var_name.as_str())
                                .map(|(i, _)| *i)
                                .min();
                            if let Some(ri) = real_idx_opt {
                                effective_index = ri;
                                if effective_index >= globals.len() {
                                    globals.resize(effective_index + 1, default_global_slot());
                                }
                            }
                        }
                    }
                    if effective_index >= globals.len() {
                        let error_message = if let Some(var_name) = global_names.get(&effective_index)
                            .or_else(|| frame.function.chunk.global_names.get(&index)) {
                            if modules::is_known_module(var_name) && !loaded_modules.contains(var_name) {
                                format!("Module {} not imported", var_name)
                            } else {
                                format!("Undefined variable: {}", var_name)
                            }
                        } else {
                            format!("Undefined variable")
                        };
                        let error = ExceptionHandler::runtime_error(&frames, error_message, line);
                        match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error, value_store, heavy_store) {
                            Ok(()) => stack::push_id(stack, NULL_VALUE_ID),
                            Err(e) => return Err(e),
                        }
                    } else {
                        let (inline_tv, heap_id_opt) = match &globals[effective_index] {
                            GlobalSlot::Inline(tv) => (Some(*tv), None),
                            GlobalSlot::Heap(id) => (None, Some(*id)),
                        };
                        if let Some(tv) = inline_tv {
                            stack::push(stack, tv);
                        } else {
                            let id = heap_id_opt.unwrap();
                            // Avoid full materialization: one store.get to decide id_to_push (O(1) instead of O(size)).
                            let cell = value_store.get(id);
                            let var_name_legacy = frame.function.chunk.global_names.get(&index).map(String::as_str);
                            // Do NOT fallback to module.get_export(class_name) for __constructing_class__: that would
                            // push the current class (e.g. ConfigApp) instead of the leaf class (e.g. ProdSettings)
                            // set by the executor, and model_config would be wrong/null.
                            let id_to_push = if cell.map(|c| matches!(c, ValueCell::Null)).unwrap_or(true)
                                && var_name_legacy == Some("__constructing_class__")
                                && frame.function.name.contains("::new_")
                            {
                                id
                            } else if cell.map(|c| matches!(c, ValueCell::Array(_))).unwrap_or(false) {
                                let ctor_name = frame.function.chunk.global_names.get(&index)
                                    .or_else(|| global_names.get(&effective_index));
                                if ctor_name.map(|n| n.contains("::new_")).unwrap_or(false) {
                                    let name = ctor_name.unwrap().clone();
                                    let candidate_indices: Vec<usize> = global_names.iter()
                                        .filter(|(_, n)| n.as_str() == name)
                                        .map(|(i, _)| *i)
                                        .collect();
                                    let found_id = candidate_indices.into_iter().find_map(|i| {
                                        if i >= globals.len() {
                                            return None;
                                        }
                                        let value_id = globals[i].resolve_to_value_id(value_store);
                                        let is_function = value_store
                                            .get(value_id)
                                            .map(|c| matches!(c, ValueCell::Function(_)))
                                            .unwrap_or(false);
                                        if is_function {
                                            Some(value_id)
                                        } else {
                                            None
                                        }
                                    });
                                    found_id.unwrap_or(id)
                                } else {
                                    id
                                }
                            } else {
                                id
                            };
                            if crate::common::debug::is_debug_enabled() {
                                if let Some(var_name) = global_names.get(&effective_index) {
                                    if var_name.contains("Data") || var_name.contains("range") || var_name.contains("print") || var_name.contains("::new_") {
                                        let value = load_value(id_to_push, value_store, heavy_store);
                                        let value_type_str = match &value {
                                            Value::Null => "Null".to_string(),
                                            Value::Function(fn_idx) => {
                                                if *fn_idx < functions.len() {
                                                    format!("Function({}, имя: '{}')", fn_idx, functions[*fn_idx].name)
                                                } else {
                                                    format!("Function({}, OUT OF BOUNDS!)", fn_idx)
                                                }
                                            },
                                            Value::Object(_) => "Object".to_string(),
                                            Value::NativeFunction(_) => "NativeFunction".to_string(),
                                            _ => "Other".to_string(),
                                        };
                                        debug_println!("[DEBUG LoadGlobal] Загружаем '{}' из globals[{}], значение: {}", var_name, index, value_type_str);
                                        if var_name.contains("::new_") {
                                            if let Value::Function(fn_idx) = &value {
                                                if *fn_idx < functions.len() {
                                                    let func = &functions[*fn_idx];
                                                    debug_println!("[DEBUG LoadGlobal] Конструктор '{}' имеет индекс функции {}, имя функции: '{}', arity: {}", var_name, fn_idx, func.name, func.arity);
                                                } else {
                                                    debug_println!("[DEBUG LoadGlobal] ОШИБКА: Конструктор '{}' имеет индекс функции {} (выходит за границы, всего функций: {})", var_name, fn_idx, functions.len());
                                                }
                                            }
                                        }
                                    }
                                    if id_to_push == NULL_VALUE_ID && (var_name.contains("Data") || var_name.contains("range") || var_name.contains("print")) {
                                        debug_println!("[DEBUG LoadGlobal] WARNING: '{}' в globals[{}] равен Null", var_name, index);
                                    }
                                }
                            }
                            stack::push_id(stack, id_to_push);
                        }
                    }
                    return Ok(VMStatus::Continue);
                }
                OpCode::StoreGlobal(index) => {
                    let tv = stack::pop(stack, frames, exception_handlers, value_store, heavy_store)?;
                    // Per-module isolation: write to frame's module namespace when frame has module_name.
                    let store_mod_name = frames.last().and_then(|f| f.module_name.clone());
                    let store_global_name = frames.last().and_then(|f| f.function.chunk.global_names.get(&index).cloned());
                    if let Some(ref mod_name) = store_mod_name {
                        if index >= BUILTIN_END {
                            let module_rc = {
                                let modules = unsafe { (*vm_ptr).get_modules() };
                                modules.get(mod_name).cloned()
                            };
                            if let Some(rc) = module_rc {
                                if let Some(name) = store_global_name {
                                    let value = load_value(tagged_to_value_id_arena(tv, value_store), value_store, heavy_store);
                                    rc.borrow().set_export(&name, value);
                                    return Ok(VMStatus::Continue);
                                }
                            }
                        }
                    }
                    // Do not allow script to overwrite the argv slot (host-only, read-only).
                    // Debug: see if StoreGlobal is ever executed for constructor new_0 (diagnose Null in export).
                    let name_at_idx = global_names.get(&index).map(|s| s.as_str());
                    if name_at_idx.map(|n| n.contains("::new_0")).unwrap_or(false) {
                        let value_type = if tv.is_heap() {
                            let vid = tagged_to_value_id_arena(tv, value_store);
                            match value_store.get(vid) {
                                Some(crate::common::value_store::ValueCell::Function(_)) => "Function",
                                Some(crate::common::value_store::ValueCell::Object(_)) => "Object",
                                Some(crate::common::value_store::ValueCell::Null) => "Null",
                                Some(crate::common::value_store::ValueCell::Array(_)) => "Array",
                                Some(_) => "Other",
                                None => "Unknown",
                            }
                        } else {
                            "Inline"
                        };
                        if crate::common::debug::verbose_constructor_debug() {
                            eprintln!("[StoreGlobal] idx={} name={:?} value_type={}", index, name_at_idx, value_type);
                        }
                    }
                    // Do not allow script to overwrite the argv slot (host-only, read-only).
                    if unsafe { (*vm_ptr).get_argv_slot_index() } == Some(index) {
                        return Ok(VMStatus::Continue);
                    }
                    // Inline path: primitives (number, bool, null, int) — no alloc, no get.
                    if !tv.is_heap() {
                        if index >= globals.len() {
                            globals.resize(index + 1, default_global_slot());
                        }
                        globals[index] = GlobalSlot::Inline(tv);
                        return Ok(VMStatus::Continue);
                    }
                    let value_id = tagged_to_value_id_arena(tv, value_store);
                    // Do not fast-path when value comes from constant pool (shared); copy so globals stay independent.
                    let from_constant = frames.last().map(|f| f.constant_ids.contains(&value_id)).unwrap_or(false);
                    // Fast path: types that need no metadata — just store the id, no load_value/store_value copy.
                    if !from_constant {
                        if let Some(cell) = value_store.get(value_id) {
                            let skip_full_path = match cell {
                                ValueCell::Array(_) | ValueCell::Tuple(_) | ValueCell::Function(_)
                                | ValueCell::ModuleFunction { .. } | ValueCell::NativeFunction(_) | ValueCell::String(_) | ValueCell::Number(_)
                                | ValueCell::Bool(_) | ValueCell::Null | ValueCell::Path(_) | ValueCell::Uuid(_, _)
                                | ValueCell::ColumnReference { .. } | ValueCell::Layer(_) | ValueCell::Window(_)
                                | ValueCell::Enumerate { .. } | ValueCell::Ellipsis => true,
                                ValueCell::Object(_) | ValueCell::Heavy(_) => false,
                            };
                            if skip_full_path {
                                if index >= globals.len() {
                                    globals.resize(index + 1, default_global_slot());
                                }
                                globals[index] = GlobalSlot::Heap(value_id);
                                return Ok(VMStatus::Continue);
                            }
                        }
                    }
                    let mut value = load_value(value_id, value_store, heavy_store);
                    if let Value::Table(table_rc) = &mut value {
                        if let Some(var_name) = global_names.get(&index) {
                            table_rc.borrow_mut().set_name(var_name.clone());
                        }
                        // Table is already in heap; no need to store_value again.
                        if index >= globals.len() {
                            globals.resize(index + 1, default_global_slot());
                        }
                        globals[index] = GlobalSlot::Heap(value_id);
                        return Ok(VMStatus::Continue);
                    }
                    let (super_name, class_name, class_meta_opt) = if let Value::Object(class_rc) = &value {
                        let class_obj = class_rc.borrow();
                        let super_name = class_obj.get("__superclass").and_then(|v| if let Value::String(s) = v { Some(s.clone()) } else { None });
                        let class_name = class_obj.get("__class_name").and_then(|v| if let Value::String(s) = v { Some(s.clone()) } else { None });
                        let class_meta_opt = class_obj.get("metadata").cloned();
                        (super_name, class_name, class_meta_opt)
                    } else {
                        (None, None, None)
                    };
                    if let Some(parent_name) = super_name {
                            let parent_global_idx = global_index_by_name(global_names, &parent_name);
                            if let Some(parent_idx) = parent_global_idx {
                                if parent_idx < globals.len() {
                                    let parent_id = globals[parent_idx].resolve_to_value_id(value_store);
                                    let parent_val = load_value(parent_id, value_store, heavy_store);
                                    if let Value::Object(parent_rc) = &parent_val {
                                        let meta_opt = parent_rc.borrow().get("metadata").cloned();
                                        if let Some(Value::Object(meta_rc)) = meta_opt {
                                            let (is_meta, tables_opt) = {
                                                let meta_ref = meta_rc.borrow();
                                                let is_m = meta_ref.get("__meta").and_then(|v| if let Value::Bool(b) = v { Some(*b) } else { None }).unwrap_or(false);
                                                let tables = meta_ref.get("tables").cloned();
                                                (is_m, tables)
                                            };
                                            if is_meta {
                                                if let Some(Value::Array(tables_rc)) = tables_opt {
                                                    tables_rc.borrow_mut().push(value.clone());
                                                }
                                                // Register ancestors in metadata.classes so run_create_all can resolve __superclass (class is fully built by now)
                                                let mut meta = meta_rc.borrow_mut();
                                                if !meta.contains_key("classes") {
                                                    meta.insert("classes".to_string(), Value::Object(Rc::new(RefCell::new(std::collections::HashMap::new()))));
                                                }
                                                let classes_rc_opt = meta.get("classes").cloned();
                                                drop(meta);
                                                if let Some(Value::Object(classes_rc)) = classes_rc_opt {
                                                    let mut classes = classes_rc.borrow_mut();
                                                    let mut current_name = parent_name.clone();
                                                    let mut seen_ancestors = std::collections::HashSet::new();
                                                    loop {
                                                        if !seen_ancestors.insert(current_name.clone()) {
                                                            break; // cycle in __superclass
                                                        }
                                                        let ancestor_idx = global_index_by_name(global_names, &current_name);
                                                        let (ancestor_has_meta, next_super) = if let Some(idx) = ancestor_idx {
                                                            if idx < globals.len() {
                                                                let aid = globals[idx].resolve_to_value_id(value_store);
                                                                let anc_val = load_value(aid, value_store, heavy_store);
                                                                if let Value::Object(ancestor_rc) = &anc_val {
                                                                    let a = ancestor_rc.borrow();
                                                                    let has_meta = a.contains_key("metadata");
                                                                    let next = a.get("__superclass").and_then(|v| if let Value::String(s) = v { Some(s.clone()) } else { None });
                                                                    (has_meta, next)
                                                                } else {
                                                                    (false, None)
                                                                }
                                                            } else {
                                                                (false, None)
                                                            }
                                                        } else {
                                                            (false, None)
                                                        };
                                                        if ancestor_has_meta {
                                                            if let Some(idx) = ancestor_idx {
                                                                if idx < globals.len() {
                                                                    let cid = globals[idx].resolve_to_value_id(value_store);
                                                                    classes.insert(current_name.clone(), load_value(cid, value_store, heavy_store));
                                                                }
                                                            }
                                                        }
                                                        if let Some(next) = next_super {
                                                            current_name = next;
                                                        } else {
                                                            break;
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                    }
                    // Register class in metadata.classes when it has metadata (so run_create_all can resolve it as parent later); use class_name/class_meta_opt from single borrow above
                    if let Some(name) = class_name {
                        if let Some(Value::Object(meta_rc)) = class_meta_opt {
                            let mut meta = meta_rc.borrow_mut();
                            if !meta.contains_key("classes") {
                                meta.insert("classes".to_string(), Value::Object(Rc::new(RefCell::new(std::collections::HashMap::new()))));
                            }
                            let classes_rc_opt = meta.get("classes").cloned();
                            drop(meta);
                            if let Some(Value::Object(classes_rc)) = classes_rc_opt {
                                classes_rc.borrow_mut().insert(name, value.clone());
                            }
                        }
                    }
                    let final_id = store_value_arena(value, value_store, heavy_store);
                    if index >= globals.len() {
                        globals.resize(index + 1, default_global_slot());
                    }
                    globals[index] = GlobalSlot::Heap(final_id);
                    return Ok(VMStatus::Continue);
                }
                OpCode::Add => {
                    let b_tv = stack::pop(stack, frames, exception_handlers, value_store, heavy_store)?;
                    let a_tv = stack::pop(stack, frames, exception_handlers, value_store, heavy_store)?;
                    {
                        let frame = frames.last_mut().unwrap();
                        let cache_hit = frame.add_cache_ip == Some(current_ip) && frame.add_cache_both_number;
                        if cache_hit && a_tv.is_number() && b_tv.is_number() {
                            stack::push(stack, TaggedValue::from_f64(a_tv.get_f64() + b_tv.get_f64()));
                            return Ok(VMStatus::Continue);
                        }
                        if cache_hit {
                            frame.add_cache_both_number = false;
                        }
                        if a_tv.is_number() && b_tv.is_number() {
                            frame.add_cache_ip = Some(current_ip);
                            frame.add_cache_both_number = true;
                            stack::push(stack, TaggedValue::from_f64(a_tv.get_f64() + b_tv.get_f64()));
                            return Ok(VMStatus::Continue);
                        }
                        frame.add_cache_both_number = false;
                    }
                    let a_id = tagged_to_value_id(a_tv, value_store);
                    let b_id = tagged_to_value_id(b_tv, value_store);
                    if let (Some(ValueCell::String(sid)), Some(ValueCell::Number(n))) =
                        (value_store.get(a_id), value_store.get(b_id))
                    {
                        if let Some(prefix) = value_store.get_string(*sid) {
                            let mut buf = String::with_capacity(prefix.len() + 24);
                            buf.push_str(prefix);
                            let _ = write!(buf, "{}", n);
                            let new_sid = value_store.intern_string(buf);
                            let result_id = value_store.allocate(ValueCell::String(new_sid));
                            stack::push_id(stack, result_id);
                            return Ok(VMStatus::Continue);
                        }
                    }
                    if let (Some(ValueCell::Number(n)), Some(ValueCell::String(sid))) =
                        (value_store.get(a_id), value_store.get(b_id))
                    {
                        if let Some(suffix) = value_store.get_string(*sid) {
                            let mut buf = String::with_capacity(24 + suffix.len());
                            let _ = write!(buf, "{}", n);
                            buf.push_str(suffix);
                            let new_sid = value_store.intern_string(buf);
                            let result_id = value_store.allocate(ValueCell::String(new_sid));
                            stack::push_id(stack, result_id);
                            return Ok(VMStatus::Continue);
                        }
                    }
                    let a = load_value(a_id, value_store, heavy_store);
                    let b = load_value(b_id, value_store, heavy_store);
                    let result = match (&a, &b) {
                        (Value::Number(n1), Value::Number(n2)) => Value::Number(n1 + n2),
                        _ => operations::binary_add(&a, &b, frames, stack, exception_handlers, value_store, heavy_store)?,
                    };
                    stack::push_id(stack, store_value(result, value_store, heavy_store));
                    return Ok(VMStatus::Continue);
                }
                OpCode::RegAdd(rd, r1, r2) => {
                    let frame = frames.last_mut().unwrap();
                    let (rd, r1, r2) = (rd as usize, r1 as usize, r2 as usize);
                    let n = 1 + rd.max(r1).max(r2);
                    if frame.regs.len() < n {
                        frame.regs.resize(n, TaggedValue::null());
                    }
                    let a = frame.regs[r1];
                    let b = frame.regs[r2];
                    if a.is_number() && b.is_number() {
                        frame.regs[rd] = TaggedValue::from_f64(a.get_f64() + b.get_f64());
                    }
                    return Ok(VMStatus::Continue);
                }
                OpCode::Sub => {
                    let b_tv = stack::pop(stack, frames, exception_handlers, value_store, heavy_store)?;
                    let a_tv = stack::pop(stack, frames, exception_handlers, value_store, heavy_store)?;
                    if a_tv.is_number() && b_tv.is_number() {
                        stack::push(stack, TaggedValue::from_f64(a_tv.get_f64() - b_tv.get_f64()));
                        return Ok(VMStatus::Continue);
                    }
                    {
                        let frame = frames.last_mut().unwrap();
                        if frame.sub_cache_ip == Some(current_ip) {
                            frame.sub_cache_both_number = false;
                        }
                    }
                    let a_id = tagged_to_value_id(a_tv, value_store);
                    let b_id = tagged_to_value_id(b_tv, value_store);
                    let a = load_value(a_id, value_store, heavy_store);
                    let b = load_value(b_id, value_store, heavy_store);
                    let result = match (&a, &b) {
                        (Value::Number(n1), Value::Number(n2)) => Value::Number(n1 - n2),
                        (Value::Null, Value::Number(n2)) => Value::Number(-n2),
                        (Value::Number(n1), Value::Null) => Value::Number(*n1),
                        _ => operations::binary_sub(&a, &b, frames, stack, exception_handlers, value_store, heavy_store)?,
                    };
                    stack::push_id(stack, store_value(result, value_store, heavy_store));
                    return Ok(VMStatus::Continue);
                }
                OpCode::Mul => {
                    let b_tv = stack::pop(stack, frames, exception_handlers, value_store, heavy_store)?;
                    let a_tv = stack::pop(stack, frames, exception_handlers, value_store, heavy_store)?;
                    if a_tv.is_number() && b_tv.is_number() {
                        stack::push(stack, TaggedValue::from_f64(a_tv.get_f64() * b_tv.get_f64()));
                        return Ok(VMStatus::Continue);
                    }
                    {
                        let frame = frames.last_mut().unwrap();
                        if frame.mul_cache_ip == Some(current_ip) {
                            frame.mul_cache_both_number = false;
                        }
                    }
                    let a_id = tagged_to_value_id(a_tv, value_store);
                    let b_id = tagged_to_value_id(b_tv, value_store);
                    let a = load_value(a_id, value_store, heavy_store);
                    let b = load_value(b_id, value_store, heavy_store);
                    let result = match (&a, &b) {
                        (Value::Number(n1), Value::Number(n2)) => Value::Number(n1 * n2),
                        _ => operations::binary_mul(&a, &b, frames, stack, exception_handlers, value_store, heavy_store)?,
                    };
                    stack::push_id(stack, store_value(result, value_store, heavy_store));
                    return Ok(VMStatus::Continue);
                }
                OpCode::Div => {
                    let b_tv = stack::pop(stack, frames, exception_handlers, value_store, heavy_store)?;
                    let a_tv = stack::pop(stack, frames, exception_handlers, value_store, heavy_store)?;
                    if a_tv.is_number() && b_tv.is_number() {
                        let n2 = b_tv.get_f64();
                        if n2 != 0.0 {
                            stack::push(stack, TaggedValue::from_f64(a_tv.get_f64() / n2));
                            return Ok(VMStatus::Continue);
                        }
                    }
                    {
                        let frame = frames.last_mut().unwrap();
                        if frame.div_cache_ip == Some(current_ip) {
                            frame.div_cache_both_number = false;
                        }
                    }
                    let a_id = tagged_to_value_id(a_tv, value_store);
                    let b_id = tagged_to_value_id(b_tv, value_store);
                    let a = load_value(a_id, value_store, heavy_store);
                    let b = load_value(b_id, value_store, heavy_store);
                    let result = match (&a, &b) {
                        (Value::Number(n1), Value::Number(n2)) if *n2 != 0.0 => Value::Number(n1 / n2),
                        (Value::Number(_), Value::Number(_)) | _ => operations::binary_div(&a, &b, frames, stack, exception_handlers, value_store, heavy_store)?,
                    };
                    stack::push_id(stack, store_value(result, value_store, heavy_store));
                    return Ok(VMStatus::Continue);
                }
                OpCode::IntDiv => {
                    let b_tv = stack::pop(stack, frames, exception_handlers, value_store, heavy_store)?;
                    let a_tv = stack::pop(stack, frames, exception_handlers, value_store, heavy_store)?;
                    if a_tv.is_number() && b_tv.is_number() {
                        let n2 = b_tv.get_f64();
                        if n2 != 0.0 {
                            stack::push(stack, TaggedValue::from_f64((a_tv.get_f64() / n2).floor()));
                            return Ok(VMStatus::Continue);
                        }
                    }
                    {
                        let frame = frames.last_mut().unwrap();
                        if frame.intdiv_cache_ip == Some(current_ip) {
                            frame.intdiv_cache_both_number = false;
                        }
                    }
                    let a_id = tagged_to_value_id(a_tv, value_store);
                    let b_id = tagged_to_value_id(b_tv, value_store);
                    let a = load_value(a_id, value_store, heavy_store);
                    let b = load_value(b_id, value_store, heavy_store);
                    let result = operations::binary_int_div(&a, &b, frames, stack, exception_handlers, value_store, heavy_store)?;
                    stack::push_id(stack, store_value(result, value_store, heavy_store));
                    return Ok(VMStatus::Continue);
                }
                OpCode::Mod => {
                    let b_tv = stack::pop(stack, frames, exception_handlers, value_store, heavy_store)?;
                    let a_tv = stack::pop(stack, frames, exception_handlers, value_store, heavy_store)?;
                    if a_tv.is_number() && b_tv.is_number() {
                        let n2 = b_tv.get_f64();
                        if n2 != 0.0 {
                            stack::push(stack, TaggedValue::from_f64(a_tv.get_f64() % n2));
                            return Ok(VMStatus::Continue);
                        }
                    }
                    {
                        let frame = frames.last_mut().unwrap();
                        if frame.mod_cache_ip == Some(current_ip) {
                            frame.mod_cache_both_number = false;
                        }
                    }
                    let a_id = tagged_to_value_id(a_tv, value_store);
                    let b_id = tagged_to_value_id(b_tv, value_store);
                    let a = load_value(a_id, value_store, heavy_store);
                    let b = load_value(b_id, value_store, heavy_store);
                    let result = operations::binary_mod(&a, &b, frames, stack, exception_handlers, value_store, heavy_store)?;
                    stack::push_id(stack, store_value(result, value_store, heavy_store));
                    return Ok(VMStatus::Continue);
                }
                OpCode::Pow => {
                    let b_tv = stack::pop(stack, frames, exception_handlers, value_store, heavy_store)?;
                    let a_tv = stack::pop(stack, frames, exception_handlers, value_store, heavy_store)?;
                    if a_tv.is_number() && b_tv.is_number() {
                        stack::push(stack, TaggedValue::from_f64(a_tv.get_f64().powf(b_tv.get_f64())));
                        return Ok(VMStatus::Continue);
                    }
                    let a_id = tagged_to_value_id(a_tv, value_store);
                    let b_id = tagged_to_value_id(b_tv, value_store);
                    let a = load_value(a_id, value_store, heavy_store);
                    let b = load_value(b_id, value_store, heavy_store);
                    let result = operations::binary_pow(&a, &b, frames, stack, exception_handlers, value_store, heavy_store)?;
                    stack::push_id(stack, store_value(result, value_store, heavy_store));
                }
                OpCode::Negate => {
                    let val_tv = stack::pop(stack, frames, exception_handlers, value_store, heavy_store)?;
                    if val_tv.is_number() {
                        stack::push(stack, TaggedValue::from_f64(-val_tv.get_f64()));
                        return Ok(VMStatus::Continue);
                    }
                    let val_id = tagged_to_value_id(val_tv, value_store);
                    let value = load_value(val_id, value_store, heavy_store);
                    let result = operations::unary_negate(&value, frames, stack, exception_handlers, value_store, heavy_store)?;
                    stack::push_id(stack, store_value(result, value_store, heavy_store));
                }
                OpCode::Not => {
                    let val_tv = stack::pop(stack, frames, exception_handlers, value_store, heavy_store)?;
                    if val_tv.is_bool() {
                        stack::push(stack, TaggedValue::from_bool(!val_tv.get_bool()));
                        return Ok(VMStatus::Continue);
                    }
                    let val_id = tagged_to_value_id(val_tv, value_store);
                    let value = load_value(val_id, value_store, heavy_store);
                    let result = operations::unary_not(&value);
                    stack::push_id(stack, store_value(result, value_store, heavy_store));
                }
                OpCode::Or => {
                    let b_id = pop_to_value_id(stack, frames, exception_handlers, value_store, heavy_store)?;
                    let a_id = pop_to_value_id(stack, frames, exception_handlers, value_store, heavy_store)?;
                    if let Some(ValueCell::Bool(a)) = value_store.get(a_id) {
                        if *a {
                            stack::push_id(stack, a_id);
                            return Ok(VMStatus::Continue);
                        }
                        stack::push_id(stack, b_id);
                        return Ok(VMStatus::Continue);
                    }
                    let a = load_value(a_id, value_store, heavy_store);
                    let b = load_value(b_id, value_store, heavy_store);
                    let result = operations::binary_or(&a, &b);
                    stack::push_id(stack, store_value(result, value_store, heavy_store));
                }
                OpCode::And => {
                    let b_id = pop_to_value_id(stack, frames, exception_handlers, value_store, heavy_store)?;
                    let a_id = pop_to_value_id(stack, frames, exception_handlers, value_store, heavy_store)?;
                    if let Some(ValueCell::Bool(a)) = value_store.get(a_id) {
                        if !*a {
                            stack::push_id(stack, a_id);
                            return Ok(VMStatus::Continue);
                        }
                        stack::push_id(stack, b_id);
                        return Ok(VMStatus::Continue);
                    }
                    let a = load_value(a_id, value_store, heavy_store);
                    let b = load_value(b_id, value_store, heavy_store);
                    let result = operations::binary_and(&a, &b);
                    stack::push_id(stack, store_value(result, value_store, heavy_store));
                }
                OpCode::Equal => {
                    let b_tv = stack::pop(stack, frames, exception_handlers, value_store, heavy_store)?;
                    let a_tv = stack::pop(stack, frames, exception_handlers, value_store, heavy_store)?;
                    if a_tv.is_number() && b_tv.is_number() {
                        stack::push(stack, TaggedValue::from_bool(a_tv.get_f64() == b_tv.get_f64()));
                        return Ok(VMStatus::Continue);
                    }
                    if a_tv.is_bool() && b_tv.is_bool() {
                        stack::push(stack, TaggedValue::from_bool(a_tv.get_bool() == b_tv.get_bool()));
                        return Ok(VMStatus::Continue);
                    }
                    if a_tv.is_null() && b_tv.is_null() {
                        stack::push(stack, TaggedValue::from_bool(true));
                        return Ok(VMStatus::Continue);
                    }
                    let a_id = tagged_to_value_id(a_tv, value_store);
                    let b_id = tagged_to_value_id(b_tv, value_store);
                    let a = load_value(a_id, value_store, heavy_store);
                    let b = load_value(b_id, value_store, heavy_store);
                    let result = match (&a, &b) {
                        (Value::Number(n1), Value::Number(n2)) => Value::Bool(n1 == n2),
                        _ => operations::binary_equal(&a, &b),
                    };
                    stack::push_id(stack, store_value(result, value_store, heavy_store));
                }
                OpCode::NotEqual => {
                    let b_tv = stack::pop(stack, frames, exception_handlers, value_store, heavy_store)?;
                    let a_tv = stack::pop(stack, frames, exception_handlers, value_store, heavy_store)?;
                    if a_tv.is_number() && b_tv.is_number() {
                        stack::push(stack, TaggedValue::from_bool(a_tv.get_f64() != b_tv.get_f64()));
                        return Ok(VMStatus::Continue);
                    }
                    if a_tv.is_bool() && b_tv.is_bool() {
                        stack::push(stack, TaggedValue::from_bool(a_tv.get_bool() != b_tv.get_bool()));
                        return Ok(VMStatus::Continue);
                    }
                    if a_tv.is_null() && b_tv.is_null() {
                        stack::push(stack, TaggedValue::from_bool(false));
                        return Ok(VMStatus::Continue);
                    }
                    let a_id = tagged_to_value_id(a_tv, value_store);
                    let b_id = tagged_to_value_id(b_tv, value_store);
                    let a = load_value(a_id, value_store, heavy_store);
                    let b = load_value(b_id, value_store, heavy_store);
                    let result = match (&a, &b) {
                        (Value::Number(n1), Value::Number(n2)) => Value::Bool(n1 != n2),
                        _ => operations::binary_not_equal(&a, &b),
                    };
                    stack::push_id(stack, store_value(result, value_store, heavy_store));
                }
                OpCode::Greater => {
                    let b_tv = stack::pop(stack, frames, exception_handlers, value_store, heavy_store)?;
                    let a_tv = stack::pop(stack, frames, exception_handlers, value_store, heavy_store)?;
                    if a_tv.is_number() && b_tv.is_number() {
                        stack::push(stack, TaggedValue::from_bool(a_tv.get_f64() > b_tv.get_f64()));
                        return Ok(VMStatus::Continue);
                    }
                    let a_id = tagged_to_value_id(a_tv, value_store);
                    let b_id = tagged_to_value_id(b_tv, value_store);
                    let a = load_value(a_id, value_store, heavy_store);
                    let b = load_value(b_id, value_store, heavy_store);
                    let result = match (&a, &b) {
                        (Value::Number(n1), Value::Number(n2)) => Value::Bool(n1 > n2),
                        _ => operations::binary_greater(&a, &b, frames, stack, exception_handlers, value_store, heavy_store)?,
                    };
                    stack::push_id(stack, store_value(result, value_store, heavy_store));
                    return Ok(VMStatus::Continue);
                }
                OpCode::Less => {
                    let b_tv = stack::pop(stack, frames, exception_handlers, value_store, heavy_store)?;
                    let a_tv = stack::pop(stack, frames, exception_handlers, value_store, heavy_store)?;
                    if a_tv.is_number() && b_tv.is_number() {
                        stack::push(stack, TaggedValue::from_bool(a_tv.get_f64() < b_tv.get_f64()));
                        return Ok(VMStatus::Continue);
                    }
                    let a_id = tagged_to_value_id(a_tv, value_store);
                    let b_id = tagged_to_value_id(b_tv, value_store);
                    let a = load_value(a_id, value_store, heavy_store);
                    let b = load_value(b_id, value_store, heavy_store);
                    let result = match (&a, &b) {
                        (Value::Number(n1), Value::Number(n2)) => Value::Bool(n1 < n2),
                        _ => operations::binary_less(&a, &b, frames, stack, exception_handlers, value_store, heavy_store)?,
                    };
                    stack::push_id(stack, store_value(result, value_store, heavy_store));
                    return Ok(VMStatus::Continue);
                }
                OpCode::GreaterEqual => {
                    let b_tv = stack::pop(stack, frames, exception_handlers, value_store, heavy_store)?;
                    let a_tv = stack::pop(stack, frames, exception_handlers, value_store, heavy_store)?;
                    if a_tv.is_number() && b_tv.is_number() {
                        stack::push(stack, TaggedValue::from_bool(a_tv.get_f64() >= b_tv.get_f64()));
                        return Ok(VMStatus::Continue);
                    }
                    let a_id = tagged_to_value_id(a_tv, value_store);
                    let b_id = tagged_to_value_id(b_tv, value_store);
                    let a = load_value(a_id, value_store, heavy_store);
                    let b = load_value(b_id, value_store, heavy_store);
                    let result = match (&a, &b) {
                        (Value::Number(n1), Value::Number(n2)) => Value::Bool(n1 >= n2),
                        _ => operations::binary_greater_equal(&a, &b, frames, stack, exception_handlers, value_store, heavy_store)?,
                    };
                    stack::push_id(stack, store_value(result, value_store, heavy_store));
                    return Ok(VMStatus::Continue);
                }
                OpCode::LessEqual => {
                    let b_tv = stack::pop(stack, frames, exception_handlers, value_store, heavy_store)?;
                    let a_tv = stack::pop(stack, frames, exception_handlers, value_store, heavy_store)?;
                    if a_tv.is_number() && b_tv.is_number() {
                        stack::push(stack, TaggedValue::from_bool(a_tv.get_f64() <= b_tv.get_f64()));
                        return Ok(VMStatus::Continue);
                    }
                    let a_id = tagged_to_value_id(a_tv, value_store);
                    let b_id = tagged_to_value_id(b_tv, value_store);
                    let a = load_value(a_id, value_store, heavy_store);
                    let b = load_value(b_id, value_store, heavy_store);
                    let result = match (&a, &b) {
                        (Value::Number(n1), Value::Number(n2)) => Value::Bool(n1 <= n2),
                        _ => operations::binary_less_equal(&a, &b, frames, stack, exception_handlers, value_store, heavy_store)?,
                    };
                    stack::push_id(stack, store_value(result, value_store, heavy_store));
                }
                OpCode::In => {
                    let array_id = pop_to_value_id(stack, frames, exception_handlers, value_store, heavy_store)?;
                    let value_id = pop_to_value_id(stack, frames, exception_handlers, value_store, heavy_store)?;
                    let array = load_value(array_id, value_store, heavy_store);
                    let value = load_value(value_id, value_store, heavy_store);
                    match array {
                        Value::Array(arr) => {
                            let arr_ref = arr.borrow();
                            let found = arr_ref.iter().any(|item| item == &value);
                            stack::push_id(stack, store_value(Value::Bool(found), value_store, heavy_store));
                        }
                        _ => {
                            let error = ExceptionHandler::runtime_error(
                                &frames,
                                "Right operand of 'in' operator must be an array".to_string(),
                                line,
                            );
                            match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error, value_store, heavy_store) {
                                Ok(()) => return Ok(VMStatus::Continue),
                                Err(e) => return Err(e),
                            }
                        }
                    }
                }
                OpCode::Jump8(offset) => {
                    frame.ip = (frame.ip as i32 + offset as i32) as usize;
                }
                OpCode::Jump16(offset) => {
                    frame.ip = (frame.ip as i32 + offset as i32) as usize;
                }
                OpCode::Jump32(offset) => {
                    frame.ip = (frame.ip as i64 + offset as i64) as usize;
                }
                OpCode::JumpIfFalse8(offset) => {
                    let cond_id = pop_to_value_id(stack, frames, exception_handlers, value_store, heavy_store)?;
                    let condition = load_value(cond_id, value_store, heavy_store);
                    let frame = frames.last_mut().unwrap();
                    if !condition.is_truthy() {
                        frame.ip = (frame.ip as i32 + offset as i32) as usize;
                    }
                }
                OpCode::JumpIfFalse16(offset) => {
                    let cond_id = pop_to_value_id(stack, frames, exception_handlers, value_store, heavy_store)?;
                    let condition = load_value(cond_id, value_store, heavy_store);
                    let frame = frames.last_mut().unwrap();
                    if !condition.is_truthy() {
                        frame.ip = (frame.ip as i32 + offset as i32) as usize;
                    }
                }
                OpCode::JumpIfFalse32(offset) => {
                    let cond_id = pop_to_value_id(stack, frames, exception_handlers, value_store, heavy_store)?;
                    let condition = load_value(cond_id, value_store, heavy_store);
                    let frame = frames.last_mut().unwrap();
                    if !condition.is_truthy() {
                        frame.ip = (frame.ip as i64 + offset as i64) as usize;
                    }
                }
                OpCode::JumpLabel(_) | OpCode::JumpIfFalseLabel(_) => {
                    return Err(crate::common::error::LangError::runtime_error(
                        "JumpLabel found in VM - compilation not finalized".to_string(),
                        frame.function.chunk.get_line(frame.ip),
                    ));
                }
                OpCode::ForRange(var_slot, start_const, end_const, step_const, end_offset) => {
                    let frame = frames.last_mut().unwrap();
                    let read_const = |store: &ValueStore, f: &CallFrame, idx: usize| -> i64 {
                        let id = f.constant_ids.get(idx).copied().unwrap_or(NULL_VALUE_ID);
                        store.get(id).and_then(|c| match c {
                            ValueCell::Number(n) => Some(*n as i64),
                            _ => None,
                        }).unwrap_or(0)
                    };
                    // Pop only if the top state belongs to this loop (same var_slot); otherwise it's an outer loop's state (nested ForRange).
                    let (current, end, step, _continued) = match frame.for_range_stack.last() {
                        Some((_, _, _, slot)) if *slot == var_slot => {
                            let s = frame.for_range_stack.pop().unwrap();
                            (s.0, s.1, s.2, true)
                        },
                        _ => {
                            let start = read_const(value_store, frame, start_const);
                            let end_val = read_const(value_store, frame, end_const);
                            let step_val = read_const(value_store, frame, step_const);
                            (start, end_val, step_val, false)
                        },
                    };
                    let done = if step > 0 { current >= end } else { current <= end };
                    if done {
                        frame.ip = (frame.ip as i32 + end_offset) as usize;
                    } else {
                        frame.for_range_stack.push((current, end, step, var_slot));
                        frame.ensure_slot(var_slot);
                        frame.slots[var_slot] = TaggedValue::from_f64(current as f64);
                        if frame.load_local_cache_slot == Some(var_slot) {
                            frame.load_local_cache_slot = None;
                        }
                    }
                }
                OpCode::ForRangeNext(back_offset) => {
                    let frame = frames.last_mut().unwrap();
                    let Some((cur, end, step, vslot)) = frame.for_range_stack.pop() else {
                        // Defensive: loop was exited (e.g. end_offset jump landed here or state was lost).
                        // Treat as loop end: do not jump back; ip already advanced in step(), so continue to next instruction.
                        return Ok(VMStatus::Continue);
                    };
                    let next_val = cur + step;
                    let done = if step > 0 { next_val >= end } else { next_val <= end };
                    if done {
                        // Exit loop: do not push state, do not jump back; next instruction is after the loop.
                        return Ok(VMStatus::Continue);
                    }
                    frame.for_range_stack.push((next_val, end, step, vslot));
                    frame.ip = (frame.ip as i32 - back_offset) as usize;
                }
                OpCode::PopForRange => {
                    let frame = frames.last_mut().unwrap();
                    let _ = frame.for_range_stack.pop();
                }
                OpCode::CallWithUnpack(unpack_arity) => {
                    // Один аргумент — объект для распаковки в kwargs; ключи должны совпадать с именами параметров функции.
                    let callee_tv = stack::pop(stack, frames, exception_handlers, value_store, heavy_store)?;
                    if unpack_arity != 1 {
                        let error = ExceptionHandler::runtime_error(
                            &frames,
                            format!("CallWithUnpack expects 1 argument (kwargs object), got {}", unpack_arity),
                            line,
                        );
                        match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error, value_store, heavy_store) {
                            Ok(()) => return Ok(VMStatus::Continue),
                            Err(e) => return Err(e),
                        }
                    }
                    let kwargs_tv = stack::pop(stack, frames, exception_handlers, value_store, heavy_store)?;
                    let kwargs_id = tagged_to_value_id(kwargs_tv, value_store);
                    let kwargs_val = load_value(kwargs_id, value_store, heavy_store);
                    let obj_map = match &kwargs_val {
                        Value::Object(rc) => rc.borrow().clone(),
                        _ => {
                            let error = ExceptionHandler::runtime_error(
                                &frames,
                                "** unpacking in call requires an object (dict), not a value of another type".to_string(),
                                line,
                            );
                            match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error, value_store, heavy_store) {
                                Ok(()) => return Ok(VMStatus::Continue),
                                Err(e) => return Err(e),
                            }
                        }
                    };
                    let callee_id = tagged_to_value_id(callee_tv, value_store);
                    let callee_val = load_value(callee_id, value_store, heavy_store);
                    let (_function_index, function) = match &callee_val {
                        Value::Function(i) if *i < functions.len() => (*i, functions[*i].clone()),
                        Value::ModuleFunction { module_id, local_index } => {
                            match unsafe { (*vm_ptr).get_module_function_index(*module_id, *local_index) } {
                                Some(real_idx) if real_idx < functions.len() => (real_idx, functions[real_idx].clone()),
                                _ => {
                                    let error = ExceptionHandler::runtime_error(
                                        &frames,
                                        "** unpacking is only supported for user-defined functions".to_string(),
                                        line,
                                    );
                                    match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error, value_store, heavy_store) {
                                        Ok(()) => return Ok(VMStatus::Continue),
                                        Err(e) => return Err(e),
                                    }
                                }
                            }
                        }
                        _ => {
                            let error = ExceptionHandler::runtime_error(
                                &frames,
                                "** unpacking is only supported for user-defined functions".to_string(),
                                line,
                            );
                            match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error, value_store, heavy_store) {
                                Ok(()) => return Ok(VMStatus::Continue),
                                Err(e) => return Err(e),
                            }
                        }
                    };
                    let param_names = &function.param_names;
                    let obj_keys: std::collections::HashSet<&String> = obj_map.keys().collect();
                    let param_set: std::collections::HashSet<&String> = param_names.iter().collect();
                    if obj_keys != param_set {
                        let expected: Vec<&str> = param_names.iter().map(|s| s.as_str()).collect();
                        let got: Vec<&str> = obj_map.keys().map(|s| s.as_str()).collect();
                        let error = ExceptionHandler::runtime_error(
                            &frames,
                            format!(
                                "Object keys must match function parameters. Expected keys: [{}], got keys: [{}]",
                                expected.join(", "),
                                got.join(", ")
                            ),
                            line,
                        );
                        match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error, value_store, heavy_store) {
                            Ok(()) => return Ok(VMStatus::Continue),
                            Err(e) => return Err(e),
                        }
                    }
                    let mut arg_tvs = Vec::with_capacity(param_names.len());
                    for p in param_names {
                        let v = obj_map.get(p).cloned().unwrap_or(Value::Null);
                        let id = store_value(v, value_store, heavy_store);
                        arg_tvs.push(TaggedValue::from_heap(id));
                    }
                    let stack_start = stack.len();
                    let mut new_frame = if function.is_cached {
                        CallFrame::new_with_cache(function.clone(), stack_start, arg_tvs.clone(), value_store, heavy_store)
                    } else {
                        CallFrame::new(function.clone(), stack_start, value_store, heavy_store)
                    };
                    if !function.chunk.error_type_table.is_empty() {
                        *error_type_table = function.chunk.error_type_table.clone();
                    }
                    if !frames.is_empty() && !function.captured_vars.is_empty() {
                        for captured_var in &function.captured_vars {
                            if captured_var.local_slot_index >= new_frame.slots.len() {
                                new_frame.slots.resize(captured_var.local_slot_index + 1, TaggedValue::null());
                            }
                            let ancestor_index = frames.len().saturating_sub(1 + captured_var.ancestor_depth);
                            if ancestor_index < frames.len() {
                                let ancestor_frame = &frames[ancestor_index];
                                if captured_var.parent_slot_index < ancestor_frame.slots.len() {
                                    new_frame.slots[captured_var.local_slot_index] = ancestor_frame.slots[captured_var.parent_slot_index];
                                } else {
                                    new_frame.slots[captured_var.local_slot_index] = TaggedValue::null();
                                }
                            } else {
                                new_frame.slots[captured_var.local_slot_index] = TaggedValue::null();
                            }
                        }
                    }
                    let param_start_index = function.captured_vars.len();
                    for (i, &arg_tv) in arg_tvs.iter().enumerate() {
                        let slot_index = param_start_index + i;
                        if slot_index >= new_frame.slots.len() {
                            new_frame.slots.resize(slot_index + 1, TaggedValue::null());
                        }
                        new_frame.slots[slot_index] = arg_tv;
                    }
                    frames.push(new_frame);
                    return Ok(VMStatus::Continue);
                }
                OpCode::Call(arity) => {
                    let callee_tv = stack::pop(stack, frames, exception_handlers, value_store, heavy_store)?;
                    let current_ip = frames.last().unwrap().ip - 1;
                    let mut function_index_opt: Option<usize> = None;
                    let mut constructing_class_opt: Option<Value> = None;
                    {
                        let frame = frames.last_mut().unwrap();
                        if frame.call_cache_ip == Some(current_ip) && frame.call_cache_is_user_function && callee_tv.is_heap() {
                            let id = callee_tv.get_heap_id();
                            match value_store.get(id) {
                                Some(ValueCell::Function(i)) if *i < functions.len() => {
                                    function_index_opt = Some(*i);
                                }
                                Some(ValueCell::ModuleFunction { module_id, local_index }) => {
                                    if let Some(real_idx) = unsafe { (*vm_ptr).get_module_function_index(*module_id, *local_index) } {
                                        function_index_opt = Some(real_idx);
                                    } else {
                                        frame.call_cache_is_user_function = false;
                                    }
                                }
                                _ => {
                                    frame.call_cache_is_user_function = false;
                                }
                            }
                        }
                    }
                    let mut actual_callee: Value = if function_index_opt.is_none() {
                        let frame = frames.last().unwrap();
                        debug_println!("[DEBUG executor OpCode::Call] Получен OpCode::Call({}) на строке {}, IP: {}", arity, line, current_ip);
                        if let Some(OpCode::Call(recorded_arity)) = frame.function.chunk.code.get(current_ip) {
                            if *recorded_arity != arity {
                                debug_println!("[ERROR executor OpCode::Call] КРИТИЧЕСКАЯ ОШИБКА: В байткоде на IP {} записано Call({}), но прочитано Call({})!",
                                    current_ip, recorded_arity, arity);
                            }
                        }
                        let function_value_id = tagged_to_value_id(callee_tv, value_store);
                        let function_value = load_value(function_value_id, value_store, heavy_store);
                        let function_type = match &function_value {
                            Value::Null => "Null",
                            Value::Function(_) => "Function",
                            Value::NativeFunction(_) => "NativeFunction",
                            _ => "Other",
                        };
                        debug_println!("[DEBUG executor OpCode::Call] Значение на стеке перед вызовом: тип = {}, значение = {:?}", function_type, function_value);
                        if matches!(&function_value, Value::Null) {
                            debug_println!("[DEBUG Call] Пытаемся вызвать Null с {} аргументами на строке {}", arity, line);
                        }
                        // If callee is a class Object (from import), resolve to constructor; if Object has __call__, use it (e.g. Settings)
                        let ac: Value = {
                        if let Value::Object(obj_rc) = &function_value {
                            let (class_name_opt, is_abstract, method_new, call_opt) = {
                                let obj_ref = obj_rc.borrow();
                                let class_name = obj_ref.get("__class_name").cloned();
                                let abstract_val = obj_ref.get("__abstract").cloned();
                                let method_key = format!("new_{}", arity);
                                let method_new = obj_ref.get(&method_key).cloned();
                                let call_ = obj_ref.get("__call__").cloned();
                                (class_name, abstract_val, method_new, call_)
                            };
                            if let Some(Value::String(ref class_name)) = class_name_opt {
                                if matches!(is_abstract.as_ref(), Some(Value::Bool(true))) {
                                    let error = ExceptionHandler::runtime_error(
                                        &frames,
                                        format!("Cannot instantiate abstract class '{}'", class_name),
                                        line,
                                    );
                                    match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error, value_store, heavy_store) {
                                        Ok(()) => return Ok(VMStatus::Continue),
                                        Err(e) => return Err(e),
                                    }
                                }
                                let constructor_name = format!("{}::new_{}", class_name, arity);
                                let constructor_value = global_names
                                    .iter()
                                    .find(|(_, n)| *n == &constructor_name)
                                    .and_then(|(idx, _)| {
                                        if *idx < globals.len() {
                                            let id = globals[*idx].resolve_to_value_id(value_store);
                                            Some(load_value(id, value_store, heavy_store))
                                        } else {
                                            None
                                        }
                                    });
                                if let Some(Value::Function(constructor_fn_idx)) = constructor_value.as_ref() {
                                    debug_println!("[DEBUG executor OpCode::Call] Class object '{}' resolved to constructor '{}'", class_name, constructor_name);
                                    constructing_class_opt = Some(function_value.clone());
                                    Value::Function(*constructor_fn_idx)
                                } else if let Some(Value::ModuleFunction { module_id, local_index }) = constructor_value.as_ref() {
                                    debug_println!("[DEBUG executor OpCode::Call] Class object '{}' resolved to module constructor '{}'", class_name, constructor_name);
                                    constructing_class_opt = Some(function_value.clone());
                                    Value::ModuleFunction { module_id: *module_id, local_index: *local_index }
                                } else if let Some(Value::Function(constructor_fn_idx)) = method_new {
                                    debug_println!("[DEBUG executor OpCode::Call] Class object '{}' resolved to constructor from class key '{}'", class_name, format!("new_{}", arity));
                                    constructing_class_opt = Some(function_value.clone());
                                    Value::Function(constructor_fn_idx)
                                } else if let Some(Value::ModuleFunction { module_id, local_index }) = method_new {
                                    debug_println!("[DEBUG executor OpCode::Call] Class object '{}' resolved to module constructor from class key '{}'", class_name, format!("new_{}", arity));
                                    constructing_class_opt = Some(function_value.clone());
                                    Value::ModuleFunction { module_id, local_index }
                                } else {
                                    function_value
                                }
                            } else if let Some(Value::Function(_)) | Some(Value::NativeFunction(_)) = call_opt.as_ref() {
                                call_opt.unwrap()
                            } else {
                                function_value
                            }
                        } else {
                            function_value
                        }
                        };
                        if let Value::Function(i) = &ac {
                            function_index_opt = Some(*i);
                            let fr = frames.last_mut().unwrap();
                            fr.call_cache_ip = Some(current_ip);
                            fr.call_cache_is_user_function = true;
                        } else if let Value::ModuleFunction { module_id, local_index } = &ac {
                            if let Some(real_idx) = unsafe { (*vm_ptr).get_module_function_index(*module_id, *local_index) } {
                                function_index_opt = Some(real_idx);
                            }
                            let fr = frames.last_mut().unwrap();
                            fr.call_cache_ip = Some(current_ip);
                            fr.call_cache_is_user_function = true;
                        } else {
                            let fr = frames.last_mut().unwrap();
                            fr.call_cache_ip = Some(current_ip);
                            fr.call_cache_is_user_function = false;
                        }
                        ac
                    } else {
                        Value::Null
                    };
                    // Runtime fallback: callee is Array but previous instruction was LoadGlobal(idx) for a constructor.
                    // 1) If chunk has name for idx, resolve by name. 2) Else (name=? from stale/cached chunk) resolve by arity.
                    if matches!(&actual_callee, Value::Array(_)) {
                        if let Some(frame) = frames.last() {
                            let prev_ip = frame.ip.saturating_sub(2);
                            let chunk = &frame.function.chunk;
                            let resolved = if let Some(crate::bytecode::OpCode::LoadGlobal(_idx)) = chunk.code.get(prev_ip) {
                                // Try by name first (chunk has global_names[idx] = "DevSettings::new_0" etc.)
                                let by_name = chunk.global_names.get(_idx)
                                    .filter(|n| n.contains("::new_"))
                                    .and_then(|name| {
                                        global_names
                                            .iter()
                                            .filter(|(_, n)| n.as_str() == name.as_str())
                                            .find_map(|(i, _)| {
                                                if *i < globals.len() {
                                                    let id = globals[*i].resolve_to_value_id(value_store);
                                                    let v = load_value(id, value_store, heavy_store);
                                                    if matches!(&v, Value::Function(_)) { Some(v) } else { None }
                                                } else { None }
                                            })
                                    });
                                if by_name.is_some() {
                                    by_name
                                } else {
                                    // No name in chunk (name=?): find constructor by arity (e.g. Call(0) -> *::new_0)
                                    let suffix = format!("::new_{}", arity);
                                    let candidates: Vec<Value> = chunk.global_names
                                        .iter()
                                        .filter(|(_, n)| n.ends_with(&suffix))
                                        .filter_map(|(_, n)| {
                                            global_names.iter()
                                                .find(|(_, nn)| nn.as_str() == n.as_str())
                                                .and_then(|(i, _)| {
                                                    if *i < globals.len() {
                                                        let id = globals[*i].resolve_to_value_id(value_store);
                                                        let v = load_value(id, value_store, heavy_store);
                                                        if matches!(&v, Value::Function(_)) { Some(v) } else { None }
                                                    } else { None }
                                                })
                                        })
                                        .collect();
                                    if candidates.len() == 1 {
                                        candidates.into_iter().next()
                                    } else {
                                        None
                                    }
                                }
                            } else {
                                None
                            };
                            if let Some(Value::Function(fn_idx)) = resolved {
                                if fn_idx < functions.len() {
                                    actual_callee = Value::Function(fn_idx);
                                    function_index_opt = Some(fn_idx);
                                    let fr = frames.last_mut().unwrap();
                                    fr.call_cache_ip = Some(current_ip);
                                    fr.call_cache_is_user_function = true;
                                }
                            }
                        }
                    }
                    // Resolve actual function index from callee when callee is a heap Function or ModuleFunction.
                    let function_index_resolved = if callee_tv.is_heap() {
                        let id = callee_tv.get_heap_id();
                        match value_store.get(id) {
                            Some(ValueCell::Function(i)) => {
                                if *i < functions.len() { Some(*i) } else { function_index_opt }
                            }
                            Some(ValueCell::ModuleFunction { module_id, local_index }) => {
                                unsafe { (*vm_ptr).get_module_function_index(*module_id, *local_index) }.or(function_index_opt)
                            }
                            _ => function_index_opt,
                        }
                    } else {
                        function_index_opt
                    };
                    let function_index_final = if let Some(function_index) = function_index_resolved {
                        if function_index < functions.len() {
                            function_index
                        } else {
                            // Index out of bounds: namespace may have raw module indices (remap missed or wrong VM).
                            // Resolve by name from previous LoadGlobal so merged module code can still call (e.g. DevSettings::new_0).
                            let fallback_name = frames.last().and_then(|f| {
                                let prev_ip = current_ip.saturating_sub(1);
                                if let Some(crate::bytecode::OpCode::LoadGlobal(idx)) = f.function.chunk.code.get(prev_ip) {
                                    f.function.chunk.global_names.get(idx).cloned()
                                } else {
                                    None
                                }
                            });
                            let by_name = fallback_name.clone().and_then(|name| {
                                let constructor_name = format!("{}::new_{}", name, arity);
                                functions.iter().position(|f| f.name == constructor_name)
                            });
                            if let Some(correct_idx) = by_name {
                                debug_println!(
                                    "[CALL] out of bounds: raw_index={}, functions.len()={}, resolved by name -> correct_idx={} ('{}')",
                                    function_index,
                                    functions.len(),
                                    correct_idx,
                                    functions.get(correct_idx).map(|f| f.name.as_str()).unwrap_or("?")
                                );
                                correct_idx
                            } else {
                                let expected_by_name = fallback_name.map(|n| format!("{}::new_{}", n, arity));
                                debug_println!(
                                    "[CALL] Function index {} out of bounds; functions.len()={}; expected by name: {:?}; DevSettings::new_0 would be at index {:?}",
                                    function_index,
                                    functions.len(),
                                    expected_by_name,
                                    functions.iter().position(|f| f.name == "DevSettings::new_0")
                                );
                                let error = ExceptionHandler::runtime_error(
                                    &frames,
                                    format!("Function index {} out of bounds", function_index),
                                    line,
                                );
                                match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error, value_store, heavy_store) {
                                    Ok(()) => return Ok(VMStatus::Continue),
                                    Err(e) => return Err(e),
                                }
                            }
                        }
                    } else if matches!(actual_callee, Value::NativeFunction(_)) {
                        // Callee is a builtin; function_index_resolved is None (heap cell is NativeFunction, not Function).
                        // Do not error: dispatch happens in match actual_callee below.
                        0
                    } else if let Value::Object(class_rc) = &actual_callee {
                        // Callee is a class Object; constructor may be ModuleFunction but registry lookup failed. Try by name.
                        let class_name = class_rc.borrow().get("__class_name").and_then(|v| if let Value::String(s) = v { Some(s.clone()) } else { None });
                        let constructor_name = class_name.as_ref().map(|n| format!("{}::new_{}", n, arity));
                        let by_name = constructor_name.as_ref().and_then(|name| functions.iter().position(|f| f.name.as_str() == name.as_str()));
                        let from_module = if by_name.is_none() {
                            // Class may come from current module or a submodule; try constructor by name in all loaded modules.
                            let cname = constructor_name.clone();
                            let modules = unsafe { (*vm_ptr).get_modules() };
                            let found: Option<Value> = cname.as_ref().and_then(|name| {
                                for (_mod_key, rc) in modules.iter() {
                                    if let Some(exp) = rc.borrow().get_export(name) {
                                        return Some(exp);
                                    }
                                }
                                None
                            });
                            drop(modules);
                            found.and_then(|exp| match &exp {
                                Value::Function(i) if *i < functions.len() => Some(*i),
                                Value::ModuleFunction { module_id, local_index } => {
                                    unsafe { (*vm_ptr).get_module_function_index(*module_id, *local_index) }
                                }
                                _ => None,
                            })
                        } else {
                            None
                        };
                        match by_name.or_else(|| from_module) {
                            Some(idx) => idx,
                            None => {
                                let error = ExceptionHandler::runtime_error(
                                    &frames,
                                    "Can only call functions".to_string(),
                                    line,
                                );
                                match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error, value_store, heavy_store) {
                                    Ok(()) => return Ok(VMStatus::Continue),
                                    Err(e) => return Err(e),
                                }
                            }
                        }
                    } else {
                        let error = ExceptionHandler::runtime_error(
                            &frames,
                            "Can only call functions".to_string(),
                            line,
                        );
                        match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error, value_store, heavy_store) {
                            Ok(()) => return Ok(VMStatus::Continue),
                            Err(e) => return Err(e),
                        }
                    };
                    if function_index_resolved.is_some() {
                    if let Some(_) = function_index_resolved {
                        let frame = frames.last_mut().unwrap();
                        frame.call_cache_ip = Some(current_ip);
                        frame.call_cache_is_user_function = true;
                    }
                    let function_index = function_index_final;
                            debug_println!(
                                "[CALL] function_index={}, functions.len()={}, function.name={}",
                                function_index,
                                functions.len(),
                                functions.get(function_index).map(|f| f.name.as_str()).unwrap_or("?")
                            );
                            let function = functions[function_index].clone();
                            
                            // Set __constructing_class__ to the class object before running a constructor, so Settings subclasses load model_config from the leaf class (e.g. DevSettings).
                            // Prefer constructing_class_opt (callee was class Object); no name lookup. If missing, fallback to globals then vm.modules.
                            // When entering a super() chain (caller is subclass, callee is superclass), do not overwrite so the leaf class stays in the slot.
                            if function.name.contains("::new_") {
                                const CONSTRUCTING_CLASS_NAME: &str = "__constructing_class__";
                                let skip_set_super_chain: bool = frames.last().and_then(|f| {
                                    let caller_name = f.function.name.as_str();
                                    let callee_name = function.name.split("::").next().unwrap_or("");
                                    if !caller_name.contains("::new_") || callee_name.is_empty() {
                                        return Some(false);
                                    }
                                    let caller_class = caller_name.split("::").next().unwrap_or("");
                                    if caller_class == callee_name {
                                        return Some(false);
                                    }
                                    let caller_class_idx = global_index_by_name(global_names, caller_class);
                                    let superclass_name: Option<String> = caller_class_idx.and_then(|idx| {
                                        if idx >= globals.len() {
                                            return None;
                                        }
                                        let id = globals[idx].resolve_to_value_id(value_store);
                                        let v = load_value(id, value_store, heavy_store);
                                        if let Value::Object(rc) = &v {
                                            let o = rc.borrow();
                                            if let Some(Value::String(s)) = o.get("__superclass") {
                                                return Some(s.clone());
                                            }
                                        }
                                        None
                                    });
                                    Some(superclass_name.as_deref() == Some(callee_name))
                                }).unwrap_or(false);
                                if !skip_set_super_chain {
                                    let class_to_set: Option<Value> = if let Some(ref class_val) = constructing_class_opt {
                                        Some(class_val.clone())
                                    } else if let Some(class_name) = function.name.split("::").next() {
                                        let indices_by_name: Vec<usize> = global_names
                                            .iter()
                                            .filter(|(_, n)| n.as_str() == class_name)
                                            .map(|(idx, _)| *idx)
                                            .collect();
                                        let by_name = indices_by_name
                                            .into_iter()
                                            .find_map(|i| {
                                                if i >= globals.len() {
                                                    return None;
                                                }
                                                let id = globals[i].resolve_to_value_id(value_store);
                                                let v = load_value(id, value_store, heavy_store);
                                                if let Value::Object(rc) = &v {
                                                    let o = rc.borrow();
                                                    if matches!(o.get("__class_name"), Some(Value::String(s)) if s.as_str() == class_name) {
                                                        return Some(v.clone());
                                                    }
                                                }
                                                None
                                            });
                                        by_name.or_else(|| {
                                            (0..globals.len()).find_map(|i| {
                                                let id = globals[i].resolve_to_value_id(value_store);
                                                let v = load_value(id, value_store, heavy_store);
                                                if let Value::Object(rc) = &v {
                                                    let o = rc.borrow();
                                                    if matches!(o.get("__class_name"), Some(Value::String(s)) if s.as_str() == class_name) {
                                                        return Some(v.clone());
                                                    }
                                                }
                                                None
                                            })
                                        }).or_else(|| {
                                            let modules = unsafe { (*vm_ptr).get_modules() };
                                            const MODULE_NAMES: &[&str] = &["config", "dev_config", "prod_config", "core.config"];
                                            for &mod_name in MODULE_NAMES {
                                                if let Some(rc) = modules.get(mod_name) {
                                                    if let Some(v) = rc.borrow().get_export(class_name) {
                                                        if let Value::Object(obj_rc) = &v {
                                                            let o = obj_rc.borrow();
                                                            let name_ok = matches!(o.get("__class_name"), Some(Value::String(s)) if s.as_str() == class_name)
                                                                || o.contains_key("new_0") || o.contains_key("new_1");
                                                            if name_ok {
                                                                return Some(v.clone());
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                            for (_mod_key, rc) in modules.iter() {
                                                if let Some(v) = rc.borrow().get_export(class_name) {
                                                    if let Value::Object(obj_rc) = &v {
                                                        let o = obj_rc.borrow();
                                                        let name_ok = matches!(o.get("__class_name"), Some(Value::String(s)) if s.as_str() == class_name)
                                                            || o.contains_key("new_0") || o.contains_key("new_1");
                                                        if name_ok {
                                                            return Some(v.clone());
                                                        }
                                                    }
                                                }
                                            }
                                            None
                                        })
                                    } else {
                                        None
                                    };
                                    if let Some(class_val) = class_to_set {
                                        // Use the index the constructor's bytecode actually reads (e.g. LoadGlobal(86)), not main's slot.
                                        // 1) From bytecode: LoadGlobal(idx) where chunk says idx is __constructing_class__ (correct after remap).
                                        // 2) From chunk.global_names (name == __constructing_class__).
                                        // 3) Main's global_names, or create new slot.
                                        let mut indices: Vec<usize> = Vec::new();
                                        for op in &function.chunk.code {
                                            if let crate::bytecode::OpCode::LoadGlobal(idx) = op {
                                                if function.chunk.global_names.get(idx).map(|n| n.as_str()) == Some(CONSTRUCTING_CLASS_NAME) {
                                                    indices.push(*idx);
                                                    break;
                                                }
                                            }
                                        }
                                        if indices.is_empty() {
                                            indices = function.chunk.global_names
                                                .iter()
                                                .filter(|(_, n)| n.as_str() == CONSTRUCTING_CLASS_NAME)
                                                .map(|(idx, _)| *idx)
                                                .collect();
                                            indices.sort_unstable();
                                        }
                                        if indices.is_empty() {
                                            let main_indices = global_indices_by_name(global_names, CONSTRUCTING_CLASS_NAME);
                                            if !main_indices.is_empty() {
                                                indices = main_indices;
                                            } else {
                                                // Last resort: first LoadGlobal in constructor (typically __constructing_class__ for model_config).
                                                for op in &function.chunk.code {
                                                    if let crate::bytecode::OpCode::LoadGlobal(idx) = op {
                                                        indices.push(*idx);
                                                        break;
                                                    }
                                                }
                                                if indices.is_empty() {
                                                    let new_idx = globals.len();
                                                    global_names.insert(new_idx, CONSTRUCTING_CLASS_NAME.to_string());
                                                    globals.resize(new_idx + 1, default_global_slot());
                                                    indices.push(new_idx);
                                                }
                                            }
                                        }
                                        if crate::common::debug::verbose_constructor_debug() {
                                            eprintln!("[executor] constructor '{}': __constructing_class__ set", function.name);
                                        }
                                        let class_id = store_value(class_val, value_store, heavy_store);
                                        for idx in indices {
                                            if idx >= globals.len() {
                                                globals.resize(idx + 1, default_global_slot());
                                            }
                                            globals[idx] = GlobalSlot::Heap(class_id);
                                        }
                                    } else if crate::common::debug::verbose_constructor_debug() {
                                        let class_name = function.name.split("::").next().unwrap_or("");
                                        eprintln!(
                                            "[executor] constructor '{}': class '{}' not found (globals + vm.modules)",
                                            function.name,
                                            class_name,
                                        );
                                    }
                                }
                            }
                            
                            debug_println!("[DEBUG executor Call] Вызываем функцию с индексом {}, имя: '{}', arity: {}, получено аргументов: {} (всего функций в VM: {})", 
                                function_index, function.name, function.arity, arity, functions.len());
                            
                            // Дополнительная отладка для конструкторов
                            if function.name.contains("::new_") {
                                debug_println!("[DEBUG executor Call] ВНИМАНИЕ: Вызывается конструктор '{}' с индексом функции {}", function.name, function_index);
                                debug_println!("[DEBUG executor Call] Ожидаем, что конструктор сохранит методы в объект через SetArrayElement");
                            }
                            
                            // Дополнительная отладка для методов класса
                            if function.name.contains("::method_") {
                                let current_ip_debug = frames.last().map(|f| f.ip - 1).unwrap_or(0);
                                debug_println!("[DEBUG executor Call] ВНИМАНИЕ: Вызывается метод класса '{}' с индексом функции {} на IP {}", function.name, function_index, current_ip_debug);
                                debug_println!("[DEBUG executor Call] Размер стека перед извлечением аргументов: {}, ожидается аргументов: {}", stack.len(), arity);
                            }
                            
                            // Собираем аргументы со стека (в обратном порядке, так как они были положены последними)
                            let mut args = Vec::new();
                            let mut arg_tvs: Vec<TaggedValue> = Vec::new();
                            
                            if arity > 0 {
                                let frame = frames.last().unwrap();
                                // stack_start указывает на начало стека для текущего frame
                                // Аргументы и функция были помещены на стек после stack_start
                                // После извлечения функции стек должен содержать минимум arity аргументов
                                let stack_size_before = stack.len();
                                debug_println!("[DEBUG executor Call] Размер стека перед извлечением аргументов: {}, stack_start: {}, ожидается аргументов: {}",
                                    stack_size_before, frame.stack_start, arity);

                                if crate::common::debug::is_debug_enabled() {
                                    debug_println!("[DEBUG executor Call] Полное содержимое стека ({} элементов):", stack.len());
                                    for (i, val_tv) in stack.iter().enumerate() {
                                        let val_id = tagged_to_value_id(*val_tv, value_store);
                                        let val = load_value(val_id, value_store, heavy_store);
                                        let val_type = match &val {
                                            Value::Number(_) => "Number",
                                            Value::String(_) => "String",
                                            Value::Bool(_) => "Bool",
                                            Value::Array(_) => "Array",
                                            Value::Object(_) => "Object",
                                            Value::Function(_) => "Function",
                                            Value::NativeFunction(_) => "NativeFunction",
                                            Value::Null => "Null",
                                            _ => "Other",
                                        };
                                        debug_println!("[DEBUG executor Call]   Стек[{}]: {} ({:?})", i, val_type, val);
                                    }
                                }

                                if stack.len() <= frame.stack_start {
                                    let error = ExceptionHandler::runtime_error(
                                        &frames,
                                        format!(
                                            "Not enough arguments on stack: expected {} but stack is empty",
                                            arity
                                        ),
                                        line,
                                    );
                                    match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error, value_store, heavy_store) {
                                        Ok(()) => return Ok(VMStatus::Continue),
                                        Err(e) => return Err(e),
                                    }
                                }
                                
                                // Проверяем, что на стеке достаточно аргументов (после извлечения функции)
                                let available_args = stack.len() - frame.stack_start;
                                debug_println!("[DEBUG executor Call] Доступно аргументов на стеке: {} (размер стека: {}, stack_start: {})",
                                    available_args, stack.len(), frame.stack_start);

                                if crate::common::debug::is_debug_enabled() && available_args > 0 {
                                    debug_println!("[DEBUG executor Call] Содержимое стека (последние {} элементов):", available_args.min(10));
                                    let start_idx = stack.len().saturating_sub(available_args.min(10));
                                    for (i, val_tv) in stack.iter().skip(start_idx).enumerate() {
                                        let val_id = tagged_to_value_id(*val_tv, value_store);
                                        let val = load_value(val_id, value_store, heavy_store);
                                        debug_println!("[DEBUG executor Call]   Стек[{}]: {:?}", start_idx + i, val);
                                    }
                                }

                                if available_args < arity {
                                    let error = ExceptionHandler::runtime_error(
                                        &frames,
                                        format!(
                                            "Not enough arguments on stack: expected {} but got {}",
                                            arity,
                                            available_args
                                        ),
                                        line,
                                    );
                                    match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error, value_store, heavy_store) {
                                        Ok(()) => return Ok(VMStatus::Continue),
                                        Err(e) => return Err(e),
                                    }
                                }
                                
                                arg_tvs.reserve(arity);
                                for i in 0..arity {
                                    let arg_tv = stack.pop().unwrap_or(TaggedValue::null());
                                    arg_tvs.push(arg_tv);
                                    let arg = slot_to_value(arg_tv, value_store, heavy_store);
                                    debug_println!("[DEBUG executor Call] Извлечен аргумент {}: {:?}", i, arg);
                                    args.push(arg);
                                }
                                arg_tvs.reverse();
                                args.reverse();
                                debug_println!("[DEBUG executor Call] Всего извлечено аргументов: {}, после reverse: {:?}", args.len(), args);
                            }
                            // Если arity == 0, args и arg_tvs остаются пустыми
                            
                            // Inject @class for methods that declare it: get class from args[0].__class and insert at position 1
                            if function.param_names.get(1).map(|s| s.as_str()) == Some("@class")
                                && args.len() + 1 == function.arity
                                && !args.is_empty()
                            {
                                let this_val = &args[0];
                                let class_val = match this_val {
                                    Value::Object(obj_rc) => obj_rc
                                        .borrow()
                                        .get("__class")
                                        .cloned()
                                        .unwrap_or(Value::Null),
                                    _ => Value::Null,
                                };
                                let class_id = store_value(class_val.clone(), value_store, heavy_store);
                                let class_tv = TaggedValue::from_heap(class_id);
                                args.insert(1, class_val);
                                arg_tvs.insert(1, class_tv);
                            }

                            // Проверяем количество аргументов (после возможной инъекции @class)
                            if args.len() != function.arity {
                                debug_println!(
                                    "[DEBUG Call] arity mismatch: function '{}' (index {}) expects {} args, got {} (caller thought this slot was a different function?)",
                                    function.name, function_index, function.arity, args.len()
                                );
                                let error = ExceptionHandler::runtime_error(
                                    &frames,
                                    format!(
                                        "Expected {} arguments but got {}",
                                        function.arity, args.len()
                                    ),
                                    line,
                                );
                                match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error, value_store, heavy_store) {
                                    Ok(()) => return Ok(VMStatus::Continue),
                                    Err(e) => return Err(e),
                                }
                            }
                            
                            // Проверяем типы аргументов, если указаны аннотации типов
                            for (i, (arg, expected_types)) in args.iter().zip(&function.param_types).enumerate() {
                                if let Some(type_names) = expected_types {
                                    if !crate::vm::calls::check_type_value(arg, type_names) {
                                        let param_name = function.param_names.get(i)
                                            .map(|s| s.as_str())
                                            .unwrap_or("unknown");
                                        let error = LangError::runtime_error_with_type(
                                            format!(
                                                "Argument '{}' expected type '{}', got '{}'",
                                                param_name, 
                                                crate::vm::calls::format_type_parts(type_names),
                                                crate::vm::calls::get_type_name_value(arg)
                                            ),
                                            line,
                                            ErrorType::TypeError,
                                        );
                                        match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error, value_store, heavy_store) {
                                            Ok(()) => return Ok(VMStatus::Continue),
                                            Err(e) => return Err(e),
                                        }
                                    }
                                }
                            }
                            
                            // Проверяем кэш, если функция помечена как кэшируемая
                            if function.is_cached {
                                use crate::bytecode::function::CacheKey;
                                
                                // Пытаемся создать ключ кэша
                                if let Some(cache_key) = CacheKey::new(&args) {
                                    // Получаем доступ к кэшу функции
                                    if let Some(cache_rc) = &function.cache {
                                        let cache = cache_rc.borrow();
                                        
                                        // Проверяем, есть ли результат в кэше
                                        if let Some(cached_result) = cache.map.get(&cache_key) {
                                            let id = store_value(cached_result.clone(), value_store, heavy_store);
                                            stack::push_id(stack, id);
                                            return Ok(VMStatus::Continue);
                                        }
                                        
                                        // Результат не найден - освобождаем borrow и продолжим выполнение
                                        drop(cache);
                                        
                                        // Выполним функцию и сохраним результат в кэш
                                        // (продолжаем выполнение ниже)
                                    }
                                }
                                // Если ключ не удалось создать (не-hashable аргументы),
                                // просто выполняем функцию без кэширования
                            }
                            
                            // Use popped arg_tvs so callee slots hold TaggedValue (immediates without store).
                            let stack_start = stack.len();
                            let mut new_frame = if function.is_cached {
                                CallFrame::new_with_cache(function.clone(), stack_start, arg_tvs.clone(), value_store, heavy_store)
                            } else {
                                CallFrame::new(function.clone(), stack_start, value_store, heavy_store)
                            };
                            if !function.chunk.error_type_table.is_empty() {
                                *error_type_table = function.chunk.error_type_table.clone();
                            }
                            if !frames.is_empty() && !function.captured_vars.is_empty() {
                                for captured_var in &function.captured_vars {
                                    if captured_var.local_slot_index >= new_frame.slots.len() {
                                        new_frame.slots.resize(captured_var.local_slot_index + 1, TaggedValue::null());
                                    }
                                    let ancestor_index = frames.len().saturating_sub(1 + captured_var.ancestor_depth);
                                    if ancestor_index < frames.len() {
                                        let ancestor_frame = &frames[ancestor_index];
                                        if captured_var.parent_slot_index < ancestor_frame.slots.len() {
                                            new_frame.slots[captured_var.local_slot_index] = ancestor_frame.slots[captured_var.parent_slot_index];
                                        } else {
                                            new_frame.slots[captured_var.local_slot_index] = TaggedValue::null();
                                        }
                                    } else {
                                        new_frame.slots[captured_var.local_slot_index] = TaggedValue::null();
                                    }
                                }
                            }
                            let param_start_index = function.captured_vars.len();
                            for (i, &arg_tv) in arg_tvs.iter().enumerate() {
                                let slot_index = param_start_index + i;
                                if slot_index >= new_frame.slots.len() {
                                    new_frame.slots.resize(slot_index + 1, TaggedValue::null());
                                }
                                new_frame.slots[slot_index] = arg_tv;
                            }
                            
                            // Добавляем новый frame
                            frames.push(new_frame);
                            return Ok(VMStatus::Continue);
                    }
                    match actual_callee {
                        Value::NativeFunction(native_index) => {
                            let builtin_count = natives.len();
                            if native_index >= builtin_count + abi_natives.len() {
                                let error = ExceptionHandler::runtime_error(
                            &frames,
                                    format!("Native function index {} out of bounds", native_index),
                                    line,
                                );
                            match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error, value_store, heavy_store) {
                                Ok(()) => return Ok(VMStatus::Continue),
                                Err(e) => return Err(e),
                            }
                            }

                            // Fast path for range(n) / range(start, end) / range(start, end, step): build Array of Number cells in ValueStore
                            // without calling native_range (no Vec<Value>, no store_value over elements).
                            const RANGE_NATIVE_INDEX: usize = 2;
                            if native_index == RANGE_NATIVE_INDEX && (arity == 1 || arity == 2 || arity == 3) {
                                let frame = frames.last().unwrap();
                                let available = stack.len().saturating_sub(frame.stack_start);
                                let need = if arity == 1 { 1 } else if arity == 2 { 2 } else { 3 };
                                if available >= need {
                                    let read_number = |store: &ValueStore, id: ValueId| -> Option<i64> {
                                        store.get(id).and_then(|c| match c {
                                            ValueCell::Number(n) => {
                                                let x = *n;
                                                if x.fract() == 0.0 && x >= i64::MIN as f64 && x <= i64::MAX as f64 {
                                                    Some(x as i64)
                                                } else {
                                                    None
                                                }
                                            }
                                            _ => None,
                                        })
                                    };
                                    let params = if arity == 1 {
                                        let n_tv = stack.pop().unwrap_or(TaggedValue::null());
                                        let n_id = tagged_to_value_id(n_tv, value_store);
                                        read_number(value_store, n_id).map(|n| (0_i64, n.max(0), 1_i64)).map_or_else(
                                            || { stack::push_id(stack, n_id); None },
                                            |t| Some(t),
                                        )
                                    } else if arity == 2 {
                                        let end_tv = stack.pop().unwrap_or(TaggedValue::null());
                                        let start_tv = stack.pop().unwrap_or(TaggedValue::null());
                                        let end_id = tagged_to_value_id(end_tv, value_store);
                                        let start_id = tagged_to_value_id(start_tv, value_store);
                                        match (read_number(value_store, start_id), read_number(value_store, end_id)) {
                                            (Some(start), Some(end)) => Some((start, end, 1_i64)),
                                            _ => {
                                                stack::push_id(stack, start_id);
                                                stack::push_id(stack, end_id);
                                                None
                                            }
                                        }
                                    } else {
                                        let step_tv = stack.pop().unwrap_or(TaggedValue::null());
                                        let end_tv = stack.pop().unwrap_or(TaggedValue::null());
                                        let start_tv = stack.pop().unwrap_or(TaggedValue::null());
                                        let step_id = tagged_to_value_id(step_tv, value_store);
                                        let end_id = tagged_to_value_id(end_tv, value_store);
                                        let start_id = tagged_to_value_id(start_tv, value_store);
                                        match (read_number(value_store, start_id), read_number(value_store, end_id), read_number(value_store, step_id)) {
                                            (Some(start), Some(end), Some(step)) if step != 0 => Some((start, end, step)),
                                            _ => {
                                                stack::push_id(stack, start_id);
                                                stack::push_id(stack, end_id);
                                                stack::push_id(stack, step_id);
                                                None
                                            }
                                        }
                                    };
                                    if let Some((start, end, step)) = params {
                                        let len = if step > 0 {
                                            if start >= end { 0 } else { ((end - start) as u64 / step as u64).min(usize::MAX as u64) as usize }
                                        } else {
                                            if start <= end { 0 } else { ((start - end) as u64 / (-step) as u64).min(usize::MAX as u64) as usize }
                                        };
                                        value_store.reserve_min(value_store.len() + 1);
                                        let mut slots = Vec::with_capacity(len);
                                        if step > 0 {
                                            let mut cur = start;
                                            while cur < end {
                                                slots.push(TaggedValue::from_f64(cur as f64));
                                                cur += step;
                                            }
                                        } else {
                                            let mut cur = start;
                                            while cur > end {
                                                slots.push(TaggedValue::from_f64(cur as f64));
                                                cur += step;
                                            }
                                        }
                                        let result_id = value_store.allocate_arena(ValueCell::Array(slots));
                                        stack::push_id(stack, result_id);
                                        return Ok(VMStatus::Continue);
                                    }
                                }
                            }

                            // Fast path for push(arr, item): mutate array in-place in ValueStore to avoid
                            // materializing and re-storing the entire array on every push (fixes memory blow-up on large datasets).
                            const PUSH_NATIVE_INDEX: usize = 35;
                            if native_index == PUSH_NATIVE_INDEX && arity == 2 {
                                let frame = frames.last().unwrap();
                                let available = stack.len().saturating_sub(frame.stack_start);
                                if available >= 2 {
                                    let item_tv = stack.pop().unwrap_or(TaggedValue::null());
                                    let array_tv = stack.pop().unwrap_or(TaggedValue::null());
                                    let array_id = tagged_to_value_id(array_tv, value_store);
                                    if let Some(ValueCell::Array(ref mut slots)) = value_store.get_mut(array_id) {
                                        slots.push(item_tv);
                                        stack::push_id(stack, array_id);
                                        return Ok(VMStatus::Continue);
                                    }
                                    stack::push_id(stack, array_id);
                                    stack::push(stack, item_tv);
                                }
                            }

                            // Fast path for len(x): avoid load_value/store_value when x is Array or String in ValueStore.
                            const LEN_NATIVE_INDEX: usize = 1;
                            if native_index == LEN_NATIVE_INDEX && arity == 1 {
                                let frame = frames.last().unwrap();
                                let available = stack.len().saturating_sub(frame.stack_start);
                                if available >= 1 {
                                    let arg_tv = stack.pop().unwrap_or(TaggedValue::null());
                                    let arg_id = tagged_to_value_id(arg_tv, value_store);
                                    if let Some(cell) = value_store.get(arg_id) {
                                        if let Some(len) = match cell {
                                            ValueCell::Array(ids) => Some(ids.len() as f64),
                                            ValueCell::String(sid) => value_store.get_string(*sid).map(|s| s.len() as f64),
                                            _ => None,
                                        } {
                                            let result_id = value_store.allocate(ValueCell::Number(len));
                                            stack::push_id(stack, result_id);
                                            return Ok(VMStatus::Continue);
                                        }
                                    }
                                    stack::push_id(stack, arg_id);
                                }
                            }

                            // Fast path for int(x), float(x), str(x), typeof(x): avoid load_value when arg is in ValueStore.
                            if arity == 1 {
                                let frame = frames.last().unwrap();
                                let available = stack.len().saturating_sub(frame.stack_start);
                                if available >= 1 {
                                    let arg_tv = stack.pop().unwrap_or(TaggedValue::null());
                                    let arg_id = tagged_to_value_id(arg_tv, value_store);
                                    let result_value = value_store.get(arg_id).and_then(|cell| {
                                        match (native_index, cell) {
                                            (3, ValueCell::Number(n)) => Some(Value::Number(n.trunc())), // int
                                            (3, ValueCell::Bool(b)) => Some(Value::Number(if *b { 1.0 } else { 0.0 })),
                                            (3, ValueCell::Null) => Some(Value::Number(0.0)),
                                            (4, ValueCell::Number(n)) => Some(Value::Number(*n)), // float
                                            (4, ValueCell::Bool(b)) => Some(Value::Number(if *b { 1.0 } else { 0.0 })),
                                            (4, ValueCell::Null) => Some(Value::Number(0.0)),
                                            (6, ValueCell::Number(n)) => Some(Value::String(n.to_string())), // str
                                            (6, ValueCell::Bool(b)) => Some(Value::String(if *b { "true".to_string() } else { "false".to_string() })),
                                            (6, ValueCell::String(sid)) => value_store.get_string(*sid).map(|s| Value::String(s.to_string())),
                                            (6, ValueCell::Null) => Some(Value::String("null".to_string())),
                                            (8, ValueCell::Number(n)) => Some(Value::String(if n.fract() == 0.0 { "int".to_string() } else { "float".to_string() })), // typeof
                                            (8, ValueCell::Bool(_)) => Some(Value::String("bool".to_string())),
                                            (8, ValueCell::String(_)) => Some(Value::String("string".to_string())),
                                            (8, ValueCell::Null) => Some(Value::String("null".to_string())),
                                            (8, ValueCell::Array(_)) => Some(Value::String("array".to_string())),
                                            (8, ValueCell::Tuple(_)) => Some(Value::String("tuple".to_string())),
                                            (8, ValueCell::Object(_)) => Some(Value::String("object".to_string())),
                                            (8, ValueCell::Path(_)) => Some(Value::String("path".to_string())),
                                            (8, ValueCell::Function(_)) | (8, ValueCell::NativeFunction(_)) => Some(Value::String("function".to_string())),
                                            _ => None,
                                        }
                                    });
                                    if let Some(v) = result_value {
                                        let result_id = store_value(v, value_store, heavy_store);
                                        stack::push_id(stack, result_id);
                                        return Ok(VMStatus::Continue);
                                    }
                                    stack::push_id(stack, arg_id);
                                }
                            }

                            // Fast path for table(data, headers): build table from ValueStore row-by-row to avoid
                            // materializing the entire data array at once (reduces peak memory for large datasets).
                            const TABLE_NATIVE_INDEX: usize = 43;
                            if native_index == TABLE_NATIVE_INDEX && arity == 2 {
                                let frame = frames.last().unwrap();
                                let available = stack.len().saturating_sub(frame.stack_start);
                                if available >= 2 {
                                    let headers_tv = stack.pop().unwrap_or(TaggedValue::null());
                                    let data_tv = stack.pop().unwrap_or(TaggedValue::null());
                                    let headers_id = tagged_to_value_id(headers_tv, value_store);
                                    let data_id = tagged_to_value_id(data_tv, value_store);
                                    let row_slots_opt = value_store.get(data_id).and_then(|c| if let ValueCell::Array(s) = c { Some(s.clone()) } else { None });
                                    if let Some(row_slots) = row_slots_opt {
                                        let num_cols = row_slots.first().and_then(|row_tv| {
                                            if row_tv.is_heap() {
                                                value_store.get(row_tv.get_heap_id()).and_then(|c| match c {
                                                    ValueCell::Array(s) => Some(s.len()),
                                                    _ => None,
                                                })
                                            } else {
                                                None
                                            }
                                        }).unwrap_or(0);
                                        let mut flat_cell_ids = Vec::with_capacity(row_slots.len() * num_cols.max(1));
                                        for row_tv in row_slots.iter() {
                                            if row_tv.is_heap() {
                                                let row_id = row_tv.get_heap_id();
                                                let cell_slots: Vec<TaggedValue> = value_store.get(row_id)
                                                    .and_then(|c| if let ValueCell::Array(s) = c { Some(s.clone()) } else { None })
                                                    .unwrap_or_default();
                                                for slot in cell_slots.iter() {
                                                    flat_cell_ids.push(tagged_to_value_id(*slot, value_store));
                                                }
                                            }
                                        }
                                        let headers: Vec<String> = {
                                            let header_slots: Vec<TaggedValue> = value_store.get(headers_id)
                                                .and_then(|c| if let ValueCell::Array(s) = c { Some(s.clone()) } else { None })
                                                .unwrap_or_default();
                                            let mut v = Vec::with_capacity(header_slots.len());
                                            for slot in header_slots.iter() {
                                                let val = load_value(tagged_to_value_id(*slot, value_store), value_store, heavy_store);
                                                v.push(match &val {
                                                    Value::String(s) => s.clone(),
                                                    _ => val.to_string(),
                                                });
                                            }
                                            if v.is_empty() {
                                                (0..num_cols).map(|i| format!("Column_{}", i)).collect()
                                            } else {
                                                v
                                            }
                                        };
                                        let table = Table::from_flat_view(flat_cell_ids, headers.len().max(1), headers);
                                        let table_val = Value::Table(Rc::new(RefCell::new(table)));
                                        let heavy_idx = heavy_store.push(table_val);
                                        let result_id = value_store.allocate(ValueCell::Heavy(heavy_idx));
                                        stack::push_id(stack, result_id);
                                        return Ok(VMStatus::Continue);
                                    }
                                    stack::push_id(stack, data_id);
                                    stack::push_id(stack, headers_id);
                                }
                            }

                            // Специальная обработка для методов тензора max_idx и min_idx (только встроенные нативы)
                            // Эти методы могут быть вызваны как tensor.max_idx() с arity=0,
                            // но тензор уже находится на стеке перед функцией
                            use crate::ml::natives as ml_natives;
                            let is_max_idx = native_index < builtin_count && std::ptr::eq(
                                natives[native_index] as *const (),
                                ml_natives::native_max_idx as *const ()
                            );
                            let is_min_idx = native_index < builtin_count && std::ptr::eq(
                                natives[native_index] as *const (),
                                ml_natives::native_min_idx as *const ()
                            );

                            // Специальная обработка для методов database engine: connect, execute, query
                            // Engine находится на стеке перед функцией
                            use crate::database::natives as db_natives;
                            let is_db_connect = native_index < builtin_count && std::ptr::eq(
                                natives[native_index] as *const (),
                                db_natives::native_engine_connect as *const ()
                            );
                            let is_db_execute = native_index < builtin_count && std::ptr::eq(
                                natives[native_index] as *const (),
                                db_natives::native_engine_execute as *const ()
                            );
                            let is_db_query = native_index < builtin_count && std::ptr::eq(
                                natives[native_index] as *const (),
                                db_natives::native_engine_query as *const ()
                            );
                            let is_db_run = native_index < builtin_count && std::ptr::eq(
                                natives[native_index] as *const (),
                                db_natives::native_engine_run as *const ()
                            );
                            let is_db_cluster_add = native_index < builtin_count && std::ptr::eq(
                                natives[native_index] as *const (),
                                db_natives::native_cluster_add as *const ()
                            );
                            let is_db_cluster_get = native_index < builtin_count && std::ptr::eq(
                                natives[native_index] as *const (),
                                db_natives::native_cluster_get as *const ()
                            );
                            let is_db_cluster_names = native_index < builtin_count && std::ptr::eq(
                                natives[native_index] as *const (),
                                db_natives::native_cluster_names as *const ()
                            );
                            let is_db_engine_method = is_db_connect || is_db_execute || is_db_query || is_db_run
                                || is_db_cluster_add || is_db_cluster_get || is_db_cluster_names;
                            
                            native_args_buffer.clear();
                            let mut native_arg_ids: Option<&mut Vec<ValueId>> = None;
                            if is_db_engine_method {
                                // Call(arity) already popped the method; stack has [receiver, arg1, ..., argN], arity = num args. Pop (arity+1) to include receiver; find receiver in all_popped; native gets (receiver, arg1, ...).
                                let frame = frames.last().unwrap();
                                let available = stack.len().saturating_sub(frame.stack_start);
                                let to_pop_total = (arity + 1).min(available);
                                reusable_all_popped.clear();
                                reusable_all_popped.reserve(to_pop_total);
                                for _ in 0..to_pop_total {
                                    let tv = stack.pop().unwrap_or(TaggedValue::null());
                                    let id = tagged_to_value_id(tv, value_store);
                                    reusable_all_popped.push(load_value(id, value_store, heavy_store));
                                }
                                reusable_all_popped.reverse();
                                if let Some(engine_idx) = reusable_all_popped.iter().position(|v| matches!(v, Value::DatabaseEngine(_) | Value::DatabaseCluster(_))) {
                                    let receiver = reusable_all_popped.remove(engine_idx);
                                    native_args_buffer.push(receiver);
                                }
                                native_args_buffer.extend(reusable_all_popped.drain(..));
                            }
                            if (is_max_idx || is_min_idx) && arity == 0 {
                                // Для методов тензора с arity=0, используем тензор со стека как первый аргумент
                                // Тензор был помещен на стек перед функцией при доступе к свойству
                                // Важно: нужно удалить тензор со стека после использования
                                if let Some(&top_tv) = stack.last() {
                                    let top_id = tagged_to_value_id(top_tv, value_store);
                                    let top_val = load_value(top_id, value_store, heavy_store);
                                    if let Value::Tensor(tensor_rc) = &top_val {
                                        native_args_buffer.push(Value::Tensor(Rc::clone(tensor_rc)));
                                        stack.pop();
                                    } else {
                                        let error = ExceptionHandler::runtime_error(&frames,
                                            "Tensor method called without tensor on stack".to_string(),
                                            line,
                                        );
                                        match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error, value_store, heavy_store) {
                                            Ok(()) => return Ok(VMStatus::Continue),
                                            Err(e) => return Err(e),
                                        }
                                    }
                                } else {
                                    let error = ExceptionHandler::runtime_error(&frames,
                                        "Tensor method called without tensor on stack".to_string(),
                                        line,
                                    );
                                    match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error, value_store, heavy_store) {
                                        Ok(()) => return Ok(VMStatus::Continue),
                                        Err(e) => return Err(e),
                                    }
                                }
                            } else if !is_db_engine_method {
                                // Обычная обработка аргументов
                                // ВАЖНО: Безопасно извлекаем аргументы, проверяя стек перед извлечением
                                let frame = frames.last().unwrap();
                                let available_args = stack.len() - frame.stack_start;
                                if available_args < arity {
                                    let error = ExceptionHandler::runtime_error(
                                        &frames,
                                        format!(
                                            "Not enough arguments on stack for native function: expected {} but got {}",
                                            arity,
                                            available_args
                                        ),
                                        line,
                                    );
                                    match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error, value_store, heavy_store) {
                                        Ok(()) => return Ok(VMStatus::Continue),
                                        Err(e) => return Err(e),
                                    }
                                }
                                // Fast path for table(data, headers): build Table as View over ValueStore without
                                // materializing the full array (avoids ~90k store get() for 10k rows × 8 cols).
                                const TABLE_NATIVE_INDEX: usize = 45;
                                if native_index == TABLE_NATIVE_INDEX && arity == 2 {
                                    let headers_tv = stack.pop().unwrap_or(TaggedValue::null());
                                    let data_tv = stack.pop().unwrap_or(TaggedValue::null());
                                    let headers_id = tagged_to_value_id(headers_tv, value_store);
                                    let data_id = tagged_to_value_id(data_tv, value_store);
                                    let row_slots_opt2 = value_store.get(data_id).and_then(|c| if let ValueCell::Array(s) = c { Some(s.clone()) } else { None });
                                    if let Some(row_slots) = row_slots_opt2 {
                                        let num_cols = row_slots.first().and_then(|row_tv| {
                                            if row_tv.is_heap() {
                                                value_store.get(row_tv.get_heap_id()).and_then(|c| match c {
                                                    ValueCell::Array(slots) => Some(slots.len()),
                                                    _ => None,
                                                })
                                            } else {
                                                None
                                            }
                                        }).unwrap_or(0);
                                        if num_cols > 0 && !row_slots.is_empty() {
                                            let mut flat_cell_ids =
                                                Vec::with_capacity(row_slots.len() * num_cols);
                                            let mut ok = true;
                                            for row_tv in row_slots.iter() {
                                                if !row_tv.is_heap() {
                                                    ok = false;
                                                    break;
                                                }
                                                let row_id = row_tv.get_heap_id();
                                                let cell_slots: Vec<TaggedValue> = value_store.get(row_id)
                                                    .and_then(|c| if let ValueCell::Array(s) = c { Some(s.clone()) } else { None })
                                                    .unwrap_or_default();
                                                if cell_slots.len() >= num_cols {
                                                    for slot in cell_slots.iter().take(num_cols) {
                                                        flat_cell_ids.push(tagged_to_value_id(*slot, value_store));
                                                    }
                                                } else {
                                                    ok = false;
                                                    break;
                                                }
                                            }
                                            if ok && flat_cell_ids.len() == row_slots.len() * num_cols
                                            {
                                                let headers_val =
                                                    load_value(headers_id, value_store, heavy_store);
                                                let header_strings: Vec<String> =
                                                    match &headers_val {
                                                        Value::Array(rc) => rc
                                                            .borrow()
                                                            .iter()
                                                            .map(|v| v.to_string())
                                                            .collect(),
                                                        _ => Vec::new(),
                                                    };
                                                if header_strings.len() >= num_cols {
                                                    let table = Table::from_flat_view(
                                                        flat_cell_ids,
                                                        num_cols,
                                                        header_strings,
                                                    );
                                                    let idx = heavy_store.push(Value::Table(
                                                        Rc::new(RefCell::new(table)),
                                                    ));
                                                    let result_id =
                                                        value_store.allocate(ValueCell::Heavy(idx));
                                                    stack::push_id(stack, result_id);
                                                    return Ok(VMStatus::Continue);
                                                }
                                            }
                                        }
                                    }
                                    stack::push_id(stack, data_id);
                                    stack::push_id(stack, headers_id);
                                }
                                reusable_native_arg_ids.clear();
                                reusable_native_arg_ids.reserve(arity);
                                native_args_buffer.reserve(arity);
                                for _ in 0..arity {
                                    let arg_tv = stack.pop().unwrap_or(TaggedValue::null());
                                    let arg_id = tagged_to_value_id(arg_tv, value_store);
                                    reusable_native_arg_ids.push(arg_id);
                                    native_args_buffer.push(load_value(arg_id, value_store, heavy_store));
                                }
                                reusable_native_arg_ids.reverse();
                                native_args_buffer.reverse();
                                native_arg_ids = Some(reusable_native_arg_ids);
                            }
                            
                            // Debug: log axis method calls
                            if native_index >= 2709 && native_index <= 2711 {
                                // axis methods: imshow (2709), set_title (2710), axis (2711)
                                let _method_name = match native_index {
                                    2709 => "imshow",
                                    2710 => "set_title",
                                    2711 => "axis",
                                    _ => "unknown",
                                };
                            }
                            
                            // Устанавливаем контекст VM для нативной функции
                            VM_CALL_CONTEXT.with(|ctx| {
                                *ctx.borrow_mut() = Some(vm_ptr);
                            });
                            
                            // Специальная проверка для range (принимает 1, 2 или 3 аргумента)
                            if native_index == 2 {
                                // range - индекс 2
                                if arity < 1 || arity > 3 {
                                    let error = ExceptionHandler::runtime_error(&frames,
                                        format!("range() expects 1, 2, or 3 arguments, got {}", arity),
                                        line,
                                    );
                            match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error, value_store, heavy_store) {
                                Ok(()) => return Ok(VMStatus::Continue),
                                Err(e) => return Err(e),
                            }
                                }
                                // Проверяем типы аргументов - все должны быть числами
                                for arg in native_args_buffer.iter() {
                                    if !matches!(arg, Value::Number(_)) {
                                        let error = ExceptionHandler::runtime_error(&frames,
                                            "range() arguments must be numbers".to_string(),
                                            line,
                                        );
                            match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error, value_store, heavy_store) {
                                Ok(()) => return Ok(VMStatus::Continue),
                                Err(e) => return Err(e),
                            }
                                    }
                                }
                                // Проверяем, что step не равен 0 (если передан)
                                if arity == 3 {
                                    if let Value::Number(step) = &native_args_buffer[2] {
                                        if *step == 0.0 {
                                            let error = ExceptionHandler::runtime_error(&frames,
                                                "range() step cannot be zero".to_string(),
                                                line,
                                            );
                            match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error, value_store, heavy_store) {
                                Ok(()) => return Ok(VMStatus::Continue),
                                Err(e) => return Err(e),
                            }
                                        }
                                    }
                                }
                            }
                            // enum(iterable): ровно 1 аргумент (индекс 72)
                            if native_index == 72 {
                                if arity != 1 {
                                    let error = ExceptionHandler::runtime_error(&frames,
                                        format!("enum() expects 1 argument, got {}", arity),
                                        line,
                                    );
                                    match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error, value_store, heavy_store) {
                                        Ok(()) => return Ok(VMStatus::Continue),
                                        Err(e) => return Err(e),
                                    }
                                }
                            }
                            
                            // ml.dataset(table, ...) requires get_column; View tables have get_column = None. Materialize View to Owned when first arg is a View table and arity is 3 (dataset(table, feature_cols, target_cols)).
                            if native_args_buffer.len() == 3 {
                                if let Some(Value::Table(rc)) = native_args_buffer.get(0) {
                                    let t = rc.borrow();
                                    if t.is_view() {
                                        let owned = t.materialize_with(|id| load_value(id, value_store, heavy_store));
                                        drop(t);
                                        native_args_buffer[0] = Value::Table(Rc::new(RefCell::new(owned)));
                                    }
                                }
                            }
                            
                            // Вызываем нативную функцию (встроенную или ABI)
                            let result = if native_index < builtin_count {
                                let native_fn = natives[native_index];
                                native_fn(native_args_buffer)
                            } else {
                                crate::vm::native_loader::call_abi_native(
                                    abi_natives[native_index - builtin_count],
                                    native_args_buffer,
                                )
                            };
                            
                            // Очищаем контекст VM после вызова нативной функции
                            VM_CALL_CONTEXT.with(|ctx| {
                                *ctx.borrow_mut() = None;
                            });
                            
                            // Проверяем ABI-ошибку (throw_error из нативного модуля)
                            if let Some(abi_err) = crate::vm::native_loader::take_last_abi_error() {
                                match ExceptionHandler::handle_exception(stack, frames, exception_handlers, abi_err, value_store, heavy_store) {
                                    Ok(()) => return Ok(VMStatus::Continue),
                                    Err(e) => return Err(e),
                                }
                            }
                            
                            // Если это relate(), получаем связи из VM-owned pending (natives push there when VM_CALL_CONTEXT is set)
                            if native_index == 65 {
                                // relate() - индекс 65
                                let relations = unsafe { (*vm_ptr).take_pending_relations() };
                                
                                // Находим имена таблиц по идентичности Rc (без сырых указателей)
                                for (table1_rc, col1_name, table2_rc, col2_name) in relations {
                                    let mut found_table1_name = None;
                                    let mut found_table2_name = None;
                                    
                                    for (index, slot) in globals.iter_mut().enumerate() {
                                        let value_id = slot.resolve_to_value_id(value_store);
                                        let value = load_value(value_id, value_store, heavy_store);
                                        if let Value::Table(table) = &value {
                                            if Rc::ptr_eq(table, &table1_rc) {
                                                if let Some(var_name) = explicit_global_names.get(&index) {
                                                    found_table1_name = Some(var_name.clone());
                                                }
                                            }
                                            if Rc::ptr_eq(table, &table2_rc) {
                                                if let Some(var_name) = explicit_global_names.get(&index) {
                                                    found_table2_name = Some(var_name.clone());
                                                }
                                            }
                                        }
                                    }
                                    
                                    // Если нашли обе таблицы, сохраняем связь
                                    // relate(pk_table["pk_column"], fk_table["fk_column"])
                                    // Первый аргумент - первичный ключ (целевая таблица)
                                    // Второй аргумент - внешний ключ (таблица, которая ссылается)
                                    if let (Some(table1_name), Some(table2_name)) = (found_table1_name, found_table2_name) {
                                        explicit_relations.push(ExplicitRelation {
                                            source_table_name: table2_name, // Таблица с внешним ключом
                                            source_column_name: col2_name,  // Внешний ключ
                                            target_table_name: table1_name, // Таблица с первичным ключом
                                            target_column_name: col1_name,   // Первичный ключ
                                        });
                                    }
                                }
                            }
                            
                            // Если это primary_key(), получаем первичные ключи из VM-owned pending
                            if native_index == 66 {
                                // primary_key() - индекс 66
                                let primary_keys = unsafe { (*vm_ptr).take_pending_primary_keys() };
                                
                                // Находим имена таблиц по идентичности Rc (без сырых указателей)
                                for (table_rc, col_name) in primary_keys {
                                    let mut found_table_name = None;
                                    
                                    for (index, slot) in globals.iter_mut().enumerate() {
                                        let value_id = slot.resolve_to_value_id(value_store);
                                        let value = load_value(value_id, value_store, heavy_store);
                                        if let Value::Table(table) = &value {
                                            if Rc::ptr_eq(table, &table_rc) {
                                                if let Some(var_name) = explicit_global_names.get(&index) {
                                                    found_table_name = Some(var_name.clone());
                                                }
                                            }
                                        }
                                    }
                                    
                                    // Если нашли таблицу, сохраняем первичный ключ
                                    if let Some(table_name) = found_table_name {
                                        explicit_primary_keys.push(ExplicitPrimaryKey {
                                            table_name,
                                            column_name: col_name,
                                        });
                                    }
                                }
                            }
                            
                            // Проверяем, не было ли ошибки в нативной функции
                            use crate::websocket::take_native_error;
                            if let Some(error_msg) = take_native_error() {
                                // Check if this is a GPU fallback warning (not a real error)
                                if error_msg.contains("Falling back to CPU") || 
                                   error_msg.contains("not available") && error_msg.contains("GPU") {
                                    // Print as warning and continue execution
                                    debug_println!("⚠️  Предупреждение: {}", error_msg);
                                    // Don't create an error, just continue
                                } else {
                                    // Determine error type based on error message
                                    // ML functions (tensor, etc.) use ValueError
                                    let error_type = if error_msg.contains("ShapeError") || 
                                                        error_msg.contains("Shape mismatch") ||
                                                        error_msg.starts_with("ShapeError:") {
                                        crate::common::error::ErrorType::ValueError
                                    } else {
                                        // Default to IOError for file/path related errors
                                        crate::common::error::ErrorType::IOError
                                    };
                                    
                                    let error = ExceptionHandler::runtime_error_with_type(
                                &frames,
                                        error_msg,
                                        line,
                                        error_type,
                                    );
                                    match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error, value_store, heavy_store) {
                                        Ok(()) => return Ok(VMStatus::Continue), // Исключение обработано
                                        Err(e) => return Err(e), // Исключение не обработано
                                    }
                                }
                            }
                            
                            // Write back any Array/Object args that the native may have mutated (e.g. push)
                            if let Some(ref ids) = native_arg_ids {
                                for (i, &id) in ids.iter().enumerate() {
                                    if i < native_args_buffer.len() {
                                        update_cell_if_mutable(id, &native_args_buffer[i], value_store, heavy_store);
                                    }
                                }
                            }
                            
                            stack::push_id(stack, store_value(result, value_store, heavy_store));
                            return Ok(VMStatus::Continue);
                        }
                        Value::Layer(layer_id) => {
                            // Layers can be called as functions: layer(input_tensor) -> output_tensor
                            if arity != 1 {
                                let error = ExceptionHandler::runtime_error(
                            &frames,
                                    format!("Layer call expects 1 argument (input tensor), got {}", arity),
                                    line,
                                );
                                match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error, value_store, heavy_store) {
                                    Ok(()) => {
                                        stack::push_id(stack, NULL_VALUE_ID);
                                        return Ok(VMStatus::Continue);
                                    }
                                    Err(e) => return Err(e),
                                }
                            }
                            
                            let input_value_id = pop_to_value_id(stack, frames, exception_handlers, value_store, heavy_store)?;
                            let input_value = load_value(input_value_id, value_store, heavy_store);
                            use crate::ml::natives;
                            let args = vec![Value::Layer(layer_id), input_value];
                            let result = natives::native_layer_call(&args);
                            stack::push_id(stack, store_value(result, value_store, heavy_store));
                            return Ok(VMStatus::Continue);
                        }
                        Value::NeuralNetwork(_) | Value::LinearRegression(_) => {
                            // Models can be called as functions: model(input_tensor) -> output_tensor
                            if arity != 1 {
                                let error = ExceptionHandler::runtime_error(
                            &frames,
                                    format!("Model call expects 1 argument (input tensor), got {}", arity),
                                    line,
                                );
                                match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error, value_store, heavy_store) {
                                    Ok(()) => {
                                        stack::push_id(stack, NULL_VALUE_ID);
                                        return Ok(VMStatus::Continue);
                                    }
                                    Err(e) => return Err(e),
                                }
                            }
                            
                            let input_value_id = pop_to_value_id(stack, frames, exception_handlers, value_store, heavy_store)?;
                            let input_value = load_value(input_value_id, value_store, heavy_store);
                            use crate::ml::natives;
                            let args = vec![actual_callee.clone(), input_value];
                            let result = natives::native_nn_forward(&args);
                            stack::push_id(stack, store_value(result, value_store, heavy_store));
                            return Ok(VMStatus::Continue);
                        }
                        _ => {
                            // Debug: callee type, line, function name (plan 1.1)
                            let callee_type = match &actual_callee {
                                Value::Null => "Null",
                                Value::Array(_) => "Array",
                                Value::Object(obj_rc) => {
                                    let obj = obj_rc.borrow();
                                    if obj.get("__class_name").is_some() {
                                        "Object(class)"
                                    } else {
                                        "Object"
                                    }
                                }
                                Value::Function(_) => "Function",
                                Value::NativeFunction(_) => "NativeFunction",
                                _ => "Other",
                            };
                            let frame_name = frames.last().map(|f| f.function.name.as_str()).unwrap_or("?");
                            if crate::common::debug::verbose_constructor_debug() {
                                eprintln!("[Call error] callee type={}, line={}, function={}", callee_type, line, frame_name);
                                if matches!(&actual_callee, Value::Array(_)) {
                                    if let Some(frame) = frames.last() {
                                        let prev_ip = frame.ip.saturating_sub(2);
                                        if let Some(crate::bytecode::OpCode::LoadGlobal(idx)) = frame.function.chunk.code.get(prev_ip) {
                                            let name = frame.function.chunk.global_names.get(idx).map(|s| s.as_str()).unwrap_or("?");
                                            eprintln!("[Call error] previous instruction at IP {}: LoadGlobal({}) name={}", prev_ip, idx, name);
                                        }
                                    }
                                }
                            }
                            // Try to provide more helpful error message
                            let error_msg = match &actual_callee {
                                Value::Null => "Cannot call null - function may not be imported or defined".to_string(),
                                Value::Object(obj_rc) => {
                                    let obj = obj_rc.borrow();
                                    if let Some(Value::String(class_name)) = obj.get("__class_name") {
                                        format!("Class '{}' cannot accept {} argument(s)", class_name, arity)
                                    } else {
                                        format!("Can only call functions, got: {:?}", std::mem::discriminant(&actual_callee))
                                    }
                                }
                                _ => format!("Can only call functions, got: {:?}", std::mem::discriminant(&actual_callee)),
                            };
                            let error = ExceptionHandler::runtime_error(
                            &frames,error_msg, line);
                            match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error, value_store, heavy_store) {
                                Ok(()) => {
                                    // Exception handled, but we need to push null to maintain stack consistency
                                    // since the caller expects a return value
                                    stack::push_id(stack, NULL_VALUE_ID);
                                    return Ok(VMStatus::Continue);
                                }
                                Err(e) => return Err(e),
                            }
                        }
                    }
                }
                OpCode::Return => {
                    // Получаем возвращаемое значение (если есть)
                    // Проверяем стек относительно stack_start текущего фрейма
                    let frame = frames.last().unwrap();
                    
                    // Логирование для конструкторов
                    let is_constructor = frame.function.name.contains("::new_");
                    if is_constructor {
                        debug_println!("[DEBUG executor Return] constructor '{}' line {} Return (stack len {}, stack_start {})", frame.function.name, line, stack.len(), frame.stack_start);
                    }
                    
                    let return_value_id = if stack.len() > frame.stack_start {
                        let tv = stack.pop().unwrap_or(TaggedValue::null());
                        tagged_to_value_id(tv, value_store)
                    } else {
                        NULL_VALUE_ID
                    };
                    if cfg!(debug_assertions) {
                        let ret_val = load_value(return_value_id, value_store, heavy_store);
                        if let Value::Object(obj_rc) = &ret_val {
                            if frame.function.name.contains("::new_") {
                                let key_count = obj_rc.borrow().len();
                                debug_println!("[DEBUG Return] constructor '{}' line {} returns Object ({} keys)", frame.function.name, line, key_count);
                            }
                        }
                    }
                    let frames_count = frames.len();
                    if frames_count > 1 {
                        if let Some(frame) = frames.last() {
                            if frame.function.is_cached {
                                if let Some(ref cached_args) = frame.cached_args {
                                    use crate::bytecode::function::CacheKey;
                                    let cached_vals: Vec<Value> = cached_args.iter().map(|&tv| slot_to_value(tv, value_store, heavy_store)).collect();
                                    if let Some(cache_key) = CacheKey::new(&cached_vals) {
                                        if let Some(cache_rc) = &frame.function.cache {
                                            let mut cache = cache_rc.borrow_mut();
                                            let result_val = load_value(return_value_id, value_store, heavy_store);
                                            cache.map.insert(cache_key, result_val);
                                        }
                                    }
                                }
                            }
                        }
                        frames.pop();
                        stack::push_id(stack, return_value_id);
                        return Ok(VMStatus::Continue);
                    } else {
                        return Ok(VMStatus::Return(return_value_id));
                    }
                }
                OpCode::Pop => {
                    // Безопасно извлекаем значение из стека
                    // Проверяем стек относительно stack_start текущего фрейма
                    if let Some(frame) = frames.last() {
                        if stack.len() > frame.stack_start {
                            // Безопасно извлекаем значение напрямую, так как мы уже проверили
                            // что стек не пуст относительно stack_start
                            stack.pop();
                        }
                        // Если стек пуст или ниже stack_start, просто игнорируем Pop
                        // Это может произойти, если функция не вернула значение
                    } else {
                        // Нет frame - безопасно извлекаем значение если стек не пуст
                        if !stack.is_empty() {
                            stack.pop();
                        }
                    }
                }
                OpCode::Dup => {
                    let top = pop_to_value_id(stack, frames, exception_handlers, value_store, heavy_store)?;
                    stack::push_id(stack, top);
                    stack::push_id(stack, top);
                }
                OpCode::MakeArray(count) => {
                    let cap = if count == 0 { 16384 } else { count };
                    let mut slots = Vec::with_capacity(cap);
                    for _ in 0..count {
                        slots.push(stack::pop(stack, frames, exception_handlers, value_store, heavy_store)?);
                    }
                    slots.reverse();
                    let result_id = value_store.allocate_arena(ValueCell::Array(slots));
                    stack::push_id(stack, result_id);
                }
                OpCode::MakeTuple(count) => {
                    let mut element_ids = Vec::with_capacity(count);
                    for _ in 0..count {
                        element_ids.push(pop_to_value_id(stack, frames, exception_handlers, value_store, heavy_store)?);
                    }
                    element_ids.reverse();
                    let result_id = value_store.allocate(ValueCell::Tuple(element_ids));
                    stack::push_id(stack, result_id);
                }
                OpCode::MakeObject(pair_count) => {
                    use std::collections::HashMap;
                    let mut map: HashMap<String, ValueId> = HashMap::with_capacity(pair_count);
                    for _ in 0..pair_count {
                        let value_id = pop_to_value_id(stack, frames, exception_handlers, value_store, heavy_store)?;
                        let key_id = pop_to_value_id(stack, frames, exception_handlers, value_store, heavy_store)?;
                        let key = match value_store.get(key_id) {
                            Some(ValueCell::String(sid)) => value_store.get_string(*sid).map(|s| s.to_string()),
                            _ => None,
                        };
                        let key = match key {
                            Some(k) => k,
                            None => {
                                let key_value = load_value(key_id, value_store, heavy_store);
                                match key_value {
                                    Value::String(s) => s,
                                    _ => {
                                        return Err(ExceptionHandler::runtime_error(
                                            &frames,
                                            "Object key must be a string".to_string(),
                                            line,
                                        ));
                                    }
                                }
                            }
                        };
                        map.insert(key, value_id);
                    }
                    let result_id = value_store.allocate(ValueCell::Object(map));
                    stack::push_id(stack, result_id);
                }
                OpCode::UnpackObject(count_slot) => {
                    let obj_id = pop_to_value_id(stack, frames, exception_handlers, value_store, heavy_store)?;
                    let pairs: Vec<(String, ValueId)> = match value_store.get(obj_id) {
                        Some(ValueCell::Object(m)) => m.iter().map(|(k, v)| (k.clone(), *v)).collect(),
                        _ => {
                            let val = load_value(obj_id, value_store, heavy_store);
                            if let Value::Object(rc) = &val {
                                rc.borrow().iter().map(|(k, v)| (k.clone(), store_value(v.clone(), value_store, heavy_store))).collect()
                            } else {
                                return Err(ExceptionHandler::runtime_error(
                                    &frames,
                                    "** unpacking requires an object".to_string(),
                                    line,
                                ));
                            }
                        }
                    };
                    let n = pairs.len();
                    for (k, v_id) in pairs {
                        let sid = value_store.intern_string(k);
                        let key_id = value_store.allocate(ValueCell::String(sid));
                        stack::push_id(stack, key_id);
                        stack::push_id(stack, v_id);
                    }
                    let frame = frames.last_mut().unwrap();
                    if count_slot >= frame.slots.len() {
                        frame.ensure_slot(count_slot);
                    }
                    let current = frame.slots[count_slot];
                    let cur_f64 = if current.is_number() {
                        current.get_f64()
                    } else if current.is_int() {
                        current.get_i32() as f64
                    } else {
                        0.0
                    };
                    frame.slots[count_slot] = TaggedValue::from_f64(cur_f64 + n as f64);
                }
                OpCode::MakeObjectDynamic => {
                    let count_id = pop_to_value_id(stack, frames, exception_handlers, value_store, heavy_store)?;
                    let pair_count = match value_store.get(count_id) {
                        Some(ValueCell::Number(n)) => {
                            let idx = *n as i64;
                            if idx < 0 {
                                return Err(ExceptionHandler::runtime_error(
                                    &frames,
                                    "Object pair count must be non-negative".to_string(),
                                    line,
                                ));
                            }
                            idx as usize
                        }
                        _ => {
                            let v = load_value(count_id, value_store, heavy_store);
                            match v {
                                Value::Number(n) => {
                                    let idx = n as i64;
                                    if idx < 0 {
                                        return Err(ExceptionHandler::runtime_error(
                                            &frames,
                                            "Object pair count must be non-negative".to_string(),
                                            line,
                                        ));
                                    }
                                    idx as usize
                                }
                                _ => {
                                    return Err(ExceptionHandler::runtime_error(
                                        &frames,
                                        "MakeObjectDynamic requires a number (pair count) on stack".to_string(),
                                        line,
                                    ));
                                }
                            }
                        }
                    };
                    use std::collections::HashMap;
                    let mut map: HashMap<String, ValueId> = HashMap::with_capacity(pair_count);
                    for _ in 0..pair_count {
                        let value_id = pop_to_value_id(stack, frames, exception_handlers, value_store, heavy_store)?;
                        let key_id = pop_to_value_id(stack, frames, exception_handlers, value_store, heavy_store)?;
                        let key = match value_store.get(key_id) {
                            Some(ValueCell::String(sid)) => value_store.get_string(*sid).map(|s| s.to_string()),
                            _ => None,
                        };
                        let key = match key {
                            Some(k) => k,
                            None => {
                                let key_value = load_value(key_id, value_store, heavy_store);
                                match key_value {
                                    Value::String(s) => s,
                                    _ => {
                                        return Err(ExceptionHandler::runtime_error(
                                            &frames,
                                            "Object key must be a string".to_string(),
                                            line,
                                        ));
                                    }
                                }
                            }
                        };
                        map.insert(key, value_id);
                    }
                    let result_id = value_store.allocate(ValueCell::Object(map));
                    stack::push_id(stack, result_id);
                }
                OpCode::MakeArrayDynamic => {
                    let count_id = pop_to_value_id(stack, frames, exception_handlers, value_store, heavy_store)?;
                    let count = match value_store.get(count_id) {
                        Some(ValueCell::Number(n)) => {
                            let idx = *n as i64;
                            if idx < 0 {
                                let error = ExceptionHandler::runtime_error(
                                    &frames,
                                    "Array size must be non-negative".to_string(),
                                    line,
                                );
                                match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error, value_store, heavy_store) {
                                    Ok(()) => return Ok(VMStatus::Continue),
                                    Err(e) => return Err(e),
                                }
                            }
                            idx as usize
                        }
                        _ => {
                            let count_value = load_value(count_id, value_store, heavy_store);
                            match count_value {
                                Value::Number(n) => {
                                    let idx = n as i64;
                                    if idx < 0 {
                                        let error = ExceptionHandler::runtime_error(
                                            &frames,
                                            "Array size must be non-negative".to_string(),
                                            line,
                                        );
                                        match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error, value_store, heavy_store) {
                                            Ok(()) => return Ok(VMStatus::Continue),
                                            Err(e) => return Err(e),
                                        }
                                    }
                                    idx as usize
                                }
                                _ => {
                                    let error = ExceptionHandler::runtime_error(
                                        &frames,
                                        "Array size must be a number".to_string(),
                                        line,
                                    );
                                    match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error, value_store, heavy_store) {
                                        Ok(()) => return Ok(VMStatus::Continue),
                                        Err(e) => return Err(e),
                                    }
                                }
                            }
                        }
                    };
                    let mut slots = Vec::with_capacity(count);
                    for _ in 0..count {
                        slots.push(stack::pop(stack, frames, exception_handlers, value_store, heavy_store)?);
                    }
                    slots.reverse();
                    let result_id = value_store.allocate_arena(ValueCell::Array(slots));
                    stack::push_id(stack, result_id);
                }
                OpCode::GetArrayLength => {
                    let array_id = pop_to_value_id(stack, frames, exception_handlers, value_store, heavy_store)?;
                    if let Some(ValueCell::Array(ref arr)) = value_store.get(array_id) {
                        let result_id = value_store.allocate(ValueCell::Number(arr.len() as f64));
                        stack::push_id(stack, result_id);
                        return Ok(VMStatus::Continue);
                    }
                    let array = load_value(array_id, value_store, heavy_store);
                    match array {
                        Value::Array(arr) => {
                            stack::push_id(stack, store_value(Value::Number(arr.borrow().len() as f64), value_store, heavy_store));
                        }
                        Value::ColumnReference { table, column_name } => {
                            let t = table.borrow();
                            if let Some(len) = crate::vm::table_ops::column_len(&*t, &column_name) {
                                stack::push_id(stack, store_value(Value::Number(len as f64), value_store, heavy_store));
                            } else {
                                let error = ExceptionHandler::runtime_error(
                                    &frames,
                                    format!("Column '{}' not found", column_name),
                                    line,
                                );
                                match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error, value_store, heavy_store) {
                                    Ok(()) => return Ok(VMStatus::Continue),
                                    Err(e) => return Err(e),
                                }
                            }
                        }
                        Value::Dataset(dataset) => {
                            let batch_size = dataset.borrow().batch_size();
                            stack::push_id(stack, store_value(Value::Number(batch_size as f64), value_store, heavy_store));
                        }
                        Value::Enumerate { data, .. } => {
                            stack::push_id(stack, store_value(Value::Number(data.borrow().len() as f64), value_store, heavy_store));
                        }
                        Value::Tuple(tuple) => {
                            stack::push_id(stack, store_value(Value::Number(tuple.borrow().len() as f64), value_store, heavy_store));
                        }
                        _ => {
                            let got_type = crate::vm::calls::get_type_name_value(&array);
                            let error = ExceptionHandler::runtime_error(
                                &frames,
                                format!("Expected array, column reference, dataset, enumerate, or tuple for GetArrayLength, got {}", got_type),
                                line,
                            );
                            match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error, value_store, heavy_store) {
                                Ok(()) => return Ok(VMStatus::Continue),
                                Err(e) => return Err(e),
                            }
                        }
                    }
                }
                OpCode::TableFilter => {
                    let value_tv = stack::pop(stack, frames, exception_handlers, value_store, heavy_store)?;
                    let op_tv = stack::pop(stack, frames, exception_handlers, value_store, heavy_store)?;
                    let column_tv = stack::pop(stack, frames, exception_handlers, value_store, heavy_store)?;
                    let table_tv = stack::pop(stack, frames, exception_handlers, value_store, heavy_store)?;
                    let value_id = tagged_to_value_id(value_tv, value_store);
                    let op_id = tagged_to_value_id(op_tv, value_store);
                    let column_id = tagged_to_value_id(column_tv, value_store);
                    let table_id = tagged_to_value_id(table_tv, value_store);
                    let table_val = load_value(table_id, value_store, heavy_store);
                    let column_val = load_value(column_id, value_store, heavy_store);
                    let op_val = load_value(op_id, value_store, heavy_store);
                    let filter_value = load_value(value_id, value_store, heavy_store);
                    let column_str = match &column_val {
                        Value::String(s) => s.as_str(),
                        _ => {
                            let error = ExceptionHandler::runtime_error(
                                &frames,
                                "Table filter column must be a string".to_string(),
                                line,
                            );
                            match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error, value_store, heavy_store) {
                                Ok(()) => return Ok(VMStatus::Continue),
                                Err(e) => return Err(e),
                            }
                        }
                    };
                    let op_str = match &op_val {
                        Value::String(s) => s.as_str(),
                        _ => {
                            let error = ExceptionHandler::runtime_error(
                                &frames,
                                "Table filter operator must be a string".to_string(),
                                line,
                            );
                            match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error, value_store, heavy_store) {
                                Ok(()) => return Ok(VMStatus::Continue),
                                Err(e) => return Err(e),
                            }
                        }
                    };
                    if let Value::Table(table_rc) = &table_val {
                        VM_CALL_CONTEXT.with(|ctx| {
                            *ctx.borrow_mut() = Some(vm_ptr);
                        });
                        let result = crate::vm::natives::table::table_where_impl(
                            table_rc,
                            column_str,
                            op_str,
                            &filter_value,
                        );
                        VM_CALL_CONTEXT.with(|ctx| {
                            *ctx.borrow_mut() = None;
                        });
                        stack::push_id(stack, store_value(result, value_store, heavy_store));
                    } else {
                        let got = match &table_val {
                            Value::Array(_) => "Array",
                            Value::Object(_) => "Object",
                            Value::Null => "Null",
                            Value::Number(_) => "Number",
                            Value::String(_) => "String",
                            Value::Bool(_) => "Bool",
                            _ => "other type",
                        };
                        let error = ExceptionHandler::runtime_error(
                            &frames,
                            format!("Table filter requires a table, got {}", got),
                            line,
                        );
                        match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error, value_store, heavy_store) {
                            Ok(()) => {}
                            Err(e) => return Err(e),
                        }
                    }
                }
                OpCode::GetArrayElement => {
                    let index_tv = stack::pop(stack, frames, exception_handlers, value_store, heavy_store)?;
                    let container_tv = stack::pop(stack, frames, exception_handlers, value_store, heavy_store)?;
                    let current_ip = {
                        let frame = frames.last().unwrap();
                        frame.ip - 1
                    };
                    {
                        let frame = frames.last_mut().unwrap();
                        let cache_array = frame.get_array_element_cache_ip == Some(current_ip) && frame.get_array_element_cache_array_number;
                        let cache_obj = frame.get_array_element_cache_ip == Some(current_ip) && frame.get_array_element_cache_object_string;
                        if cache_array && container_tv.is_heap() && index_tv.is_number() {
                            let container_id = container_tv.get_heap_id();
                            let idx = index_tv.get_f64() as i64;
                            if idx >= 0 {
                                if let Some(ValueCell::Array(ref vec)) = value_store.get(container_id) {
                                    let u = idx as usize;
                                    if u < vec.len() {
                                        stack::push(stack, vec[u]);
                                        frame.get_array_element_cache_ip = Some(current_ip);
                                        frame.get_array_element_cache_array_number = true;
                                        frame.get_array_element_cache_object_string = false;
                                        return Ok(VMStatus::Continue);
                                    }
                                }
                            }
                            frame.get_array_element_cache_array_number = false;
                        }
                        if cache_obj && container_tv.is_heap() && index_tv.is_heap() {
                            let container_id = container_tv.get_heap_id();
                            let index_value_id = index_tv.get_heap_id();
                            if let (Some(ValueCell::Object(ref map)), Some(ValueCell::String(key_id))) =
                                (value_store.get(container_id), value_store.get(index_value_id))
                            {
                                if !map.contains_key("__class_name") {
                                    if let Some(key_str) = value_store.get_string(*key_id) {
                                        let element_id = map.get(key_str).copied().unwrap_or(NULL_VALUE_ID);
                                        stack::push_id(stack, element_id);
                                        frame.get_array_element_cache_ip = Some(current_ip);
                                        frame.get_array_element_cache_array_number = false;
                                        frame.get_array_element_cache_object_string = true;
                                        return Ok(VMStatus::Continue);
                                    }
                                }
                            }
                            frame.get_array_element_cache_object_string = false;
                        }
                    }
                    let index_value_id = tagged_to_value_id(index_tv, value_store);
                    let container_id = tagged_to_value_id(container_tv, value_store);
                    if let (Some(ValueCell::Array(ref vec)), Some(ValueCell::Number(n))) =
                        (value_store.get(container_id), value_store.get(index_value_id))
                    {
                        let idx = *n as i64;
                        if idx >= 0 {
                            let u = idx as usize;
                            if u < vec.len() {
                                stack::push(stack, vec[u]);
                                let frame = frames.last_mut().unwrap();
                                frame.get_array_element_cache_ip = Some(current_ip);
                                frame.get_array_element_cache_array_number = true;
                                frame.get_array_element_cache_object_string = false;
                                return Ok(VMStatus::Continue);
                            }
                        }
                    }
                    // Fast path: ValueCell::Object + ValueCell::String key — no load_value for container/index.
                    // Skip fast path for class instances (they need private/protected checks).
                    if let (Some(ValueCell::Object(ref map)), Some(ValueCell::String(key_id))) =
                        (value_store.get(container_id), value_store.get(index_value_id))
                    {
                        if !map.contains_key("__class_name") {
                            if let Some(key_str) = value_store.get_string(*key_id) {
                                let element_id = map.get(key_str).copied().unwrap_or(NULL_VALUE_ID);
                                stack::push_id(stack, element_id);
                                let frame = frames.last_mut().unwrap();
                                frame.get_array_element_cache_ip = Some(current_ip);
                                frame.get_array_element_cache_array_number = false;
                                frame.get_array_element_cache_object_string = true;
                                return Ok(VMStatus::Continue);
                            }
                        }
                    }
                    let frame = frames.last_mut().unwrap();
                    frame.get_array_element_cache_array_number = false;
                    frame.get_array_element_cache_object_string = false;
                    let index_value = load_value(index_value_id, value_store, heavy_store);
                    let container = load_value(container_id, value_store, heavy_store);
                    let container_type = match &container {
                        Value::Array(_) => "Array",
                        Value::Enumerate { .. } => "Enumerate",
                        Value::Object(_) => "Object",
                        Value::Table(_) => "Table",
                        Value::Path(_) => "Path",
                        Value::Uuid(_, _) => "UUID",
                        _ => "Other",
                    };
                    let key_str = match &index_value {
                        Value::String(k) => k.clone(),
                        Value::Number(n) => format!("{}", n),
                        _ => format!("{:?}", index_value),
                    };
                    debug_println!("[DEBUG GetArrayElement] line {} IP {}: {} key '{}'", line, current_ip, container_type, key_str);
                    
                    match container {
                        Value::Array(arr) => {
                            if let Value::String(key) = &index_value {
                                // Array methods: push=35, pop=36, unique=37, reverse=38, sort=39, sum=40, average=41, count=42, any=43, all=44
                                let method_index = match key.as_str() {
                                    "push" => Some(35),
                                    "pop" => Some(36),
                                    "unique" => Some(37),
                                    "reverse" => Some(38),
                                    "sort" => Some(39),
                                    "sum" => Some(40),
                                    "average" => Some(41),
                                    "count" => Some(42),
                                    "any" => Some(43),
                                    "all" => Some(44),
                                    _ => None,
                                };
                                if let Some(idx) = method_index {
                                    stack::push_id(stack, store_value(Value::NativeFunction(idx), value_store, heavy_store));
                                    return Ok(VMStatus::Continue);
                                }
                                let error = ExceptionHandler::runtime_error(
                                    &frames,
                                    format!("Array has no property '{}'. Available: push, pop, unique, reverse, sort, sum, average, count, any, all, or use numeric index", key),
                                    line,
                                );
                                match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error, value_store, heavy_store) {
                                    Ok(()) => return Ok(VMStatus::Continue),
                                    Err(e) => return Err(e),
                                }
                            }
                            let index = match index_value {
                                Value::Number(n) => {
                                    let idx = n as i64;
                                    if idx < 0 {
                                        let error = ExceptionHandler::runtime_error(
                            &frames,
                                            "Array index must be non-negative".to_string(),
                                            line,
                                        );
                            match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error, value_store, heavy_store) {
                                Ok(()) => return Ok(VMStatus::Continue),
                                Err(e) => return Err(e),
                            }
                                    }
                                    idx as usize
                                }
                                _ => {
                                    let error = ExceptionHandler::runtime_error(
                            &frames,
                                        "Array index must be a number".to_string(),
                                        line,
                                    );
                            match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error, value_store, heavy_store) {
                                Ok(()) => return Ok(VMStatus::Continue),
                                Err(e) => return Err(e),
                            }
                                }
                            };
                            
                            // Push the element (TaggedValue) from the store so reference semantics are preserved.
                            if let Some(ValueCell::Array(slots)) = value_store.get(container_id) {
                                if index < slots.len() {
                                    stack::push(stack, slots[index]);
                                    return Ok(VMStatus::Continue);
                                }
                            }
                            let arr_ref = arr.borrow();
                            if index >= arr_ref.len() {
                                let error = ExceptionHandler::runtime_error_with_type(
                                &frames,
                                    format!("Array index {} out of bounds (length: {})", index, arr_ref.len()),
                                    line,
                                    ErrorType::IndexError,
                                );
                            match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error, value_store, heavy_store) {
                                Ok(()) => return Ok(VMStatus::Continue),
                                Err(e) => return Err(e),
                            }
                            }
                            let element = &arr_ref[index];
                            let value = match element {
                                Value::Array(arr_rc) => Value::Array(Rc::clone(arr_rc)),
                                Value::Table(table_rc) => Value::Table(Rc::clone(table_rc)),
                                Value::Axis(axis_rc) => Value::Axis(Rc::clone(axis_rc)),
                                Value::Figure(fig_rc) => Value::Figure(Rc::clone(fig_rc)),
                                Value::Image(img_rc) => Value::Image(Rc::clone(img_rc)),
                                Value::Window(handle) => Value::Window(*handle),
                                Value::Tensor(tensor_rc) => Value::Tensor(Rc::clone(tensor_rc)),
                                Value::Object(obj_rc) => Value::Object(obj_rc.clone()),
                                Value::DatabaseEngine(engine_rc) => Value::DatabaseEngine(Rc::clone(engine_rc)),
                                Value::DatabaseCluster(cluster_rc) => Value::DatabaseCluster(Rc::clone(cluster_rc)),
                                _ => element.clone(),
                            };
                            stack::push_id(stack, store_value(value, value_store, heavy_store));
                            return Ok(VMStatus::Continue);
                        }
                        Value::Tuple(tuple) => {
                            let index = match index_value {
                                Value::Number(n) => {
                                    let idx = n as i64;
                                    if idx < 0 {
                                        let error = ExceptionHandler::runtime_error(
                            &frames,
                                            "Tuple index must be non-negative".to_string(),
                                            line,
                                        );
                            match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error, value_store, heavy_store) {
                                Ok(()) => return Ok(VMStatus::Continue),
                                Err(e) => return Err(e),
                            }
                                    }
                                    idx as usize
                                }
                                _ => {
                                    let error = ExceptionHandler::runtime_error(
                            &frames,
                                        "Tuple index must be a number".to_string(),
                                        line,
                                    );
                            match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error, value_store, heavy_store) {
                                Ok(()) => return Ok(VMStatus::Continue),
                                Err(e) => return Err(e),
                            }
                                }
                            };
                            
                            let tuple_ref = tuple.borrow();
                            if index >= tuple_ref.len() {
                                let error = ExceptionHandler::runtime_error_with_type(
                                &frames,
                                    format!("Tuple index {} out of bounds (length: {})", index, tuple_ref.len()),
                                    line,
                                    ErrorType::IndexError,
                                );
                            match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error, value_store, heavy_store) {
                                Ok(()) => return Ok(VMStatus::Continue),
                                Err(e) => return Err(e),
                            }
                            }
                            // Для сложных типов возвращаем ссылку, для простых клонируем
                            let element = &tuple_ref[index];
                            let value = match element {
                                Value::Array(arr_rc) => Value::Array(Rc::clone(arr_rc)),
                                Value::Tuple(tuple_rc) => Value::Tuple(Rc::clone(tuple_rc)),
                                Value::Table(table_rc) => Value::Table(Rc::clone(table_rc)),
                                Value::Object(_) => element.clone(),
                                _ => element.clone(),
                            };
                            stack::push_id(stack, store_value(value, value_store, heavy_store));
                            return Ok(VMStatus::Continue);
                        }
                        Value::Enumerate { data, start } => {
                            let index = match index_value {
                                Value::Number(n) => {
                                    let idx = n as i64;
                                    if idx < 0 {
                                        let error = ExceptionHandler::runtime_error(
                                            &frames,
                                            "Enumerate index must be non-negative".to_string(),
                                            line,
                                        );
                                        match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error, value_store, heavy_store) {
                                            Ok(()) => return Ok(VMStatus::Continue),
                                            Err(e) => return Err(e),
                                        }
                                    }
                                    idx as usize
                                }
                                _ => {
                                    let error = ExceptionHandler::runtime_error(
                                        &frames,
                                        "Enumerate index must be a number".to_string(),
                                        line,
                                    );
                                    match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error, value_store, heavy_store) {
                                        Ok(()) => return Ok(VMStatus::Continue),
                                        Err(e) => return Err(e),
                                    }
                                }
                            };
                            let data_ref = data.borrow();
                            if index >= data_ref.len() {
                                let error = ExceptionHandler::runtime_error_with_type(
                                    &frames,
                                    format!("Enumerate index {} out of bounds (length: {})", index, data_ref.len()),
                                    line,
                                    ErrorType::IndexError,
                                );
                                match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error, value_store, heavy_store) {
                                    Ok(()) => return Ok(VMStatus::Continue),
                                    Err(e) => return Err(e),
                                }
                            }
                            let element = &data_ref[index];
                            let value = match element {
                                Value::Array(arr_rc) => Value::Array(Rc::clone(arr_rc)),
                                Value::Table(table_rc) => Value::Table(Rc::clone(table_rc)),
                                Value::Axis(axis_rc) => Value::Axis(Rc::clone(axis_rc)),
                                Value::Figure(fig_rc) => Value::Figure(Rc::clone(fig_rc)),
                                Value::Image(img_rc) => Value::Image(Rc::clone(img_rc)),
                                Value::Window(handle) => Value::Window(*handle),
                                Value::Tensor(tensor_rc) => Value::Tensor(Rc::clone(tensor_rc)),
                                Value::Object(obj_rc) => Value::Object(obj_rc.clone()),
                                Value::DatabaseEngine(engine_rc) => Value::DatabaseEngine(Rc::clone(engine_rc)),
                                Value::DatabaseCluster(cluster_rc) => Value::DatabaseCluster(Rc::clone(cluster_rc)),
                                _ => element.clone(),
                            };
                            let pair = Value::Tuple(Rc::new(RefCell::new(vec![
                                Value::Number((start + index as i64) as f64),
                                value,
                            ])));
                            stack::push_id(stack, store_value(pair, value_store, heavy_store));
                            return Ok(VMStatus::Continue);
                        }
                        Value::Table(table) => {
                            // Доступ к колонке таблицы по имени или строке по индексу
                            match index_value {
                                Value::String(property) => {
                                    // Специальные свойства таблицы
                                    if property == "rows" {
                                        let t = table.borrow();
                                        let rows: Vec<Value> = if t.is_view() {
                                            (0..t.len())
                                                .map(|i| {
                                                    let row = crate::vm::table_ops::get_row(&*t, i, value_store, heavy_store).unwrap_or_default();
                                                    Value::Array(Rc::new(RefCell::new(row)))
                                                })
                                                .collect()
                                        } else {
                                            t.rows_ref().unwrap().iter()
                                                .map(|row| Value::Array(Rc::new(RefCell::new(row.to_vec()))))
                                                .collect()
                                        };
                                        drop(t);
                                        stack::push_id(stack, store_value(Value::Array(Rc::new(RefCell::new(rows))), value_store, heavy_store));
                                    } else if property == "columns" {
                                        let table_ref = table.borrow();
                                        let columns: Vec<Value> = table_ref.headers().iter()
                                            .map(|header| Value::String(header.clone()))
                                            .collect();
                                        stack::push_id(stack, store_value(Value::Array(Rc::new(RefCell::new(columns))), value_store, heavy_store));
                                    } else {
                                        let mut table_ref = table.borrow_mut();
                                        let has_col = if table_ref.is_view() {
                                            table_ref.has_column(&property)
                                        } else {
                                            table_ref.get_column(&property).is_some()
                                        };
                                        if has_col {
                                            stack::push_id(stack, store_value(Value::ColumnReference {
                                                table: table.clone(),
                                                column_name: property,
                                            }, value_store, heavy_store));
                                        } else {
                                            let error = ExceptionHandler::runtime_error_with_type(
                                                &frames,
                                                format!("Column '{}' not found in table", property),
                                                line,
                                                ErrorType::KeyError,
                                            );
                                            match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error, value_store, heavy_store) {
                                                Ok(()) => return Ok(VMStatus::Continue),
                                                Err(e) => return Err(e),
                                            }
                                        }
                                    }
                                }
                                Value::Number(n) => {
                                    // Доступ к строке по индексу
                                    let idx = n as i64;
                                    if idx < 0 {
                                        let error = ExceptionHandler::runtime_error(
                            &frames,
                                            "Table row index must be non-negative".to_string(),
                                            line,
                                        );
                            match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error, value_store, heavy_store) {
                                Ok(()) => return Ok(VMStatus::Continue),
                                Err(e) => return Err(e),
                            }
                                    }
                                    let table_ref = table.borrow();
                                    let len = table_ref.len();
                                    if idx as usize >= len {
                                        let error = ExceptionHandler::runtime_error_with_type(
                                            &frames,
                                            format!("Row index {} out of bounds (length: {})", idx, len),
                                            line,
                                            ErrorType::IndexError,
                                        );
                                        match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error, value_store, heavy_store) {
                                            Ok(()) => return Ok(VMStatus::Continue),
                                            Err(e) => return Err(e),
                                        }
                                    }
                                    let row = if table_ref.is_view() {
                                        crate::vm::table_ops::get_row(&*table_ref, idx as usize, value_store, heavy_store)
                                    } else {
                                        table_ref.get_row(idx as usize).map(|r| r.to_vec())
                                    };
                                    if let Some(row) = row {
                                        use std::collections::HashMap;
                                        let mut row_dict = HashMap::new();
                                        for (i, header) in table_ref.headers().iter().enumerate() {
                                            if i < row.len() {
                                                row_dict.insert(header.clone(), row[i].clone());
                                            }
                                        }
                                        stack::push_id(stack, store_value(Value::Object(Rc::new(RefCell::new(row_dict))), value_store, heavy_store));
                                    } else {
                                        let error = ExceptionHandler::runtime_error_with_type(
                                            &frames,
                                            format!("Row index {} out of bounds", idx),
                                            line,
                                            ErrorType::IndexError,
                                        );
                                        match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error, value_store, heavy_store) {
                                            Ok(()) => return Ok(VMStatus::Continue),
                                            Err(e) => return Err(e),
                                        }
                                    }
                                }
                                _ => {
                                    let error = ExceptionHandler::runtime_error(
                            &frames,
                                        "Table index must be a string (column name) or number (row index)".to_string(),
                                        line,
                                    );
                            match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error, value_store, heavy_store) {
                                Ok(()) => return Ok(VMStatus::Continue),
                                Err(e) => return Err(e),
                            }
                                }
                            }
                        }
                        Value::Object(map_rc) => {
                            let map = map_rc.borrow();
                            
                            // Check if this is a layer accessor object (has __neural_network key)
                            if map.contains_key("__neural_network") {
                                // This is a layer accessor - handle indexing to get layers
                                match index_value {
                                    Value::Number(n) => {
                                        let idx = n as i64;
                                        if idx < 0 {
                                            let error = ExceptionHandler::runtime_error(&frames,
                                                "Layer index must be non-negative".to_string(),
                                                line,
                                            );
                                            match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error, value_store, heavy_store) {
                                                Ok(()) => return Ok(VMStatus::Continue),
                                                Err(e) => return Err(e),
                                            }
                                        }
                                        
                                        // Get the NeuralNetwork from the accessor
                                        if let Some(Value::NeuralNetwork(nn_rc)) = map.get("__neural_network") {
                                            // Call native_model_get_layer
                                            use crate::ml::natives;
                                            let args = vec![Value::NeuralNetwork(Rc::clone(nn_rc)), Value::Number(n)];
                                            let result = natives::native_model_get_layer(&args);
                                            stack::push_id(stack, store_value(result, value_store, heavy_store));
                                            return Ok(VMStatus::Continue);
                                        }
                                    }
                                    _ => {
                                        let error = ExceptionHandler::runtime_error(&frames,
                                            "Layer accessor index must be a number".to_string(),
                                            line,
                                        );
                                        match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error, value_store, heavy_store) {
                                            Ok(()) => return Ok(VMStatus::Continue),
                                            Err(e) => return Err(e),
                                        }
                                    }
                                }
                            }
                            
                            // Regular object access
                            match index_value {
                                Value::String(key) => {
                                    // Доступ к объекту класса: запрет чтения приватных переменных класса снаружи (ProtectError)
                                    if map.contains_key("__class_name") {
                                        if key != "model_config" {
                                            if let Some(Value::Array(private_vars_rc)) = map.get("__class_private_vars") {
                                                let private_vars = private_vars_rc.borrow();
                                                let is_private = private_vars.iter().any(|v| {
                                                    if let Value::String(s) = v { s.as_str() == key } else { false }
                                                });
                                                if is_private {
                                                    let class_name = map.get("__class_name")
                                                        .and_then(|v| if let Value::String(s) = v { Some(s.clone()) } else { None })
                                                        .unwrap_or_else(|| "?".to_string());
                                                    let msg = format!("Class variable '{}' is private in '{}' and cannot be accessed from outside the class", key, class_name);
                                                    let error = ExceptionHandler::runtime_error_with_type(
                                                        &frames,
                                                        msg,
                                                        line,
                                                        ErrorType::ProtectError,
                                                    );
                                                    match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error, value_store, heavy_store) {
                                                        Ok(()) => return Ok(VMStatus::Continue),
                                                        Err(e) => return Err(e),
                                                    }
                                                }
                                            }
                                        }
                                        // Class protected variables: allow only from this class or subclasses
                                        if let Some(Value::Array(protected_vars_rc)) = map.get("__class_protected_vars") {
                                            let protected_vars = protected_vars_rc.borrow();
                                            let is_protected = protected_vars.iter().any(|v| {
                                                if let Value::String(s) = v { s.as_str() == key } else { false }
                                            });
                                            if is_protected {
                                                let class_name_obj = map.get("__class_name");
                                                let in_hierarchy = if let Some(Value::String(ref obj_class_name)) = class_name_obj {
                                                    frames.iter().any(|f| {
                                                        let frame_class = f.function.name.split("::").next().unwrap_or("");
                                                        let frame_chain = get_superclass_chain(globals, global_names, frame_class, value_store, heavy_store);
                                                        frame_chain.iter().any(|c| c == obj_class_name)
                                                    })
                                                } else {
                                                    false
                                                };
                                                if !in_hierarchy {
                                                    let obj_class_name = map.get("__class_name")
                                                        .and_then(|v| if let Value::String(s) = v { Some(s.clone()) } else { None })
                                                        .unwrap_or_else(|| "?".to_string());
                                                    let frame_class_opt = frames.iter().rev()
                                                        .find_map(|f| {
                                                            if f.function.name.contains("::new_") || f.function.name.contains("::method_") {
                                                                f.function.name.split("::").next().map(String::from)
                                                            } else {
                                                                None
                                                            }
                                                        });
                                                    let msg = match &frame_class_opt {
                                                        Some(class) => format!("Class variable '{}' is protected in '{}' and cannot be accessed from subclass '{}'", key, obj_class_name, class),
                                                        None => format!("Class variable '{}' is protected in '{}' and cannot be accessed from outside the class", key, obj_class_name),
                                                    };
                                                    let error = ExceptionHandler::runtime_error_with_type(
                                                        &frames,
                                                        msg,
                                                        line,
                                                        ErrorType::ProtectError,
                                                    );
                                                    match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error, value_store, heavy_store) {
                                                        Ok(()) => return Ok(VMStatus::Continue),
                                                        Err(e) => return Err(e),
                                                    }
                                                }
                                            }
                                        }
                                    }
                                    // Instance private fields: allow only from the defining class (not subclasses)
                                    if let (Some(Value::String(ref class_name)), Some(Value::Array(private_fields_rc))) = (
                                        map.get("__class_name"),
                                        map.get("__private_fields"),
                                    ) {
                                        let private_fields_slice = private_fields_rc.borrow();
                                        let is_private_field = private_fields_slice.iter().any(|v| {
                                            if let Value::String(s) = v { s.as_str() == key } else { false }
                                        });
                                        if is_private_field {
                                            let defining_class = map.get("__private_field_defining_class")
                                                .and_then(|v| {
                                                    if let Value::Object(rc) = v {
                                                        rc.borrow().get(&key).and_then(|v| {
                                                            if let Value::String(s) = v { Some(s.clone()) } else { None }
                                                        })
                                                    } else { None }
                                                })
                                                .unwrap_or_else(|| class_name.clone());
                                            let in_defining_class = frames.iter().any(|f| {
                                                f.function.name.starts_with(&format!("{}::", defining_class))
                                            });
                                            if !in_defining_class {
                                                // Innermost frame that is a class method/constructor (::new_ or ::method_); skip <main>
                                                let frame_class_opt = frames.iter().rev()
                                                    .find_map(|f| {
                                                        if f.function.name.contains("::new_") || f.function.name.contains("::method_") {
                                                            f.function.name.split("::").next().map(String::from)
                                                        } else {
                                                            None
                                                        }
                                                    });
                                                let msg = match &frame_class_opt {
                                                    Some(class) => format!("Field '{}' is private in '{}' and cannot be accessed from subclass '{}'", key, defining_class, class),
                                                    None => format!("Field '{}' is private in '{}' and cannot be accessed from outside the class", key, defining_class),
                                                };
                                                let error = ExceptionHandler::runtime_error_with_type(
                                                    &frames,
                                                    msg,
                                                    line,
                                                    ErrorType::ProtectError,
                                                );
                                                match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error, value_store, heavy_store) {
                                                    Ok(()) => return Ok(VMStatus::Continue),
                                                    Err(e) => return Err(e),
                                                }
                                            }
                                        }
                                    }
                                    // Instance protected fields: allow from this class or any subclass (frame class in instance's chain)
                                    if let (Some(Value::String(ref instance_class)), Some(Value::Array(protected_fields_rc))) = (
                                        map.get("__class_name"),
                                        map.get("__protected_fields"),
                                    ) {
                                        let protected_fields_slice = protected_fields_rc.borrow();
                                        let is_protected_field = protected_fields_slice.iter().any(|v| {
                                            if let Value::String(s) = v { s.as_str() == key } else { false }
                                        });
                                        if is_protected_field {
                                            let instance_chain = get_superclass_chain(globals, global_names, instance_class, value_store, heavy_store);
                                            let in_hierarchy = frames.iter().any(|f| {
                                                let frame_class = f.function.name.split("::").next().unwrap_or("");
                                                instance_chain.iter().any(|c| c == frame_class)
                                            });
                                            if !in_hierarchy {
                                                let frame_class_opt = frames.iter().rev()
                                                    .find_map(|f| {
                                                        if f.function.name.contains("::new_") || f.function.name.contains("::method_") {
                                                            f.function.name.split("::").next().map(String::from)
                                                        } else {
                                                            None
                                                        }
                                                    });
                                                let msg = match &frame_class_opt {
                                                    Some(class) => format!("Field '{}' is protected in '{}' and cannot be accessed from subclass '{}'", key, instance_class, class),
                                                    None => format!("Field '{}' is protected in '{}' and cannot be accessed from outside the class", key, instance_class),
                                                };
                                                let error = ExceptionHandler::runtime_error_with_type(
                                                    &frames,
                                                    msg,
                                                    line,
                                                    ErrorType::ProtectError,
                                                );
                                                match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error, value_store, heavy_store) {
                                                    Ok(()) => return Ok(VMStatus::Continue),
                                                    Err(e) => return Err(e),
                                                }
                                            }
                                        }
                                    }
                                    if let Some(value) = map.get(&key) {
                                        stack::push_id(stack, store_value(value.clone(), value_store, heavy_store));
                                    } else {
                                        debug_println!("[DEBUG GetArrayElement] key '{}' not found, pushing Null", key);
                                        // Для отсутствующих ключей возвращаем null
                                        // Это позволяет классам иметь поля с значениями по умолчанию
                                        // В арифметических операциях null будет преобразован в 0
                                        stack::push_id(stack, NULL_VALUE_ID);
                                    }
                                }
                                _ => {
                                    let error = ExceptionHandler::runtime_error(&frames,
                                        "Object index must be a string".to_string(),
                                        line,
                                    );
                            match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error, value_store, heavy_store) {
                                Ok(()) => return Ok(VMStatus::Continue),
                                Err(e) => return Err(e),
                            }
                                }
                            }
                        }
                        Value::Figure(figure_rc) => {
                            // Доступ к свойствам фигуры по строковому ключу
                            match index_value {
                                Value::String(key) => {
                                    match key.as_str() {
                                        "axes" => {
                                            // Возвращаем 2D массив осей
                                            let figure_ref = figure_rc.borrow();
                                            let mut axes_array = Vec::new();
                                            for row in &figure_ref.axes {
                                                let mut row_array = Vec::new();
                                                for axis in row {
                                                    row_array.push(Value::Axis(axis.clone()));
                                                }
                                                axes_array.push(Value::Array(Rc::new(RefCell::new(row_array))));
                                            }
                                            stack::push_id(stack, store_value(Value::Array(Rc::new(RefCell::new(axes_array))), value_store, heavy_store));
                                        }
                                        _ => {
                                            let error = ExceptionHandler::runtime_error_with_type(&frames,
                                                format!("Figure has no property '{}'", key),
                                                line,
                                                ErrorType::KeyError,
                                            );
                                            match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error, value_store, heavy_store) {
                                                Ok(()) => return Ok(VMStatus::Continue),
                                                Err(e) => return Err(e),
                                            }
                                        }
                                    }
                                }
                                _ => {
                                    let error = ExceptionHandler::runtime_error(&frames,
                                        "Figure property access must use string key".to_string(),
                                        line,
                                    );
                                    match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error, value_store, heavy_store) {
                                        Ok(()) => return Ok(VMStatus::Continue),
                                        Err(e) => return Err(e),
                                    }
                                }
                            }
                            return Ok(VMStatus::Continue);
                        }
                        Value::Axis(_axis_rc) => {
                            // Доступ к методам оси по строковому ключу
                            match index_value {
                                Value::String(key) => {
                                    // Find the native function index for axis methods
                                    // These are registered after plot functions (starting at plot_native_start + 9)
                                    // We need to find them dynamically
                                    let method_name = match key.as_str() {
                                        "imshow" => "imshow",
                                        "set_title" => "set_title",
                                        "axis" => "axis",
                                        _ => {
                                            let error = ExceptionHandler::runtime_error_with_type(&frames,
                                                format!("Axis has no method '{}'", key),
                                                line,
                                                ErrorType::KeyError,
                                            );
                                            match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error, value_store, heavy_store) {
                                                Ok(()) => return Ok(VMStatus::Continue),
                                                Err(e) => return Err(e),
                                            }
                                        }
                                    };
                                    
                                    let method_index = if let Some((_plot_id, plot_val)) = globals.iter_mut().find_map(|slot| {
                                        let plot_id = slot.resolve_to_value_id(value_store);
                                        let plot_val = load_value(plot_id, value_store, heavy_store);
                                        if let Value::Object(map_rc) = &plot_val {
                                            if map_rc.borrow().contains_key("image") {
                                                return Some((plot_id, plot_val));
                                            }
                                        }
                                        None
                                    }) {
                                        if let Value::Object(map_rc) = &plot_val {
                                            let map = map_rc.borrow();
                                            let idx_key = match method_name {
                                                "imshow" => "__axis_imshow_idx",
                                                "set_title" => "__axis_set_title_idx",
                                                "axis" => "__axis_axis_idx",
                                                _ => {
                                                    let error = ExceptionHandler::runtime_error(&frames,
                                                        format!("Axis method '{}' not found", key),
                                                        line,
                                                    );
                                                    match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error, value_store, heavy_store) {
                                                        Ok(()) => return Ok(VMStatus::Continue),
                                                        Err(e) => return Err(e),
                                                    }
                                                }
                                            };
                                            if let Some(Value::Number(idx)) = map.get(idx_key) {
                                                *idx as usize
                                            } else {
                                                let error = ExceptionHandler::runtime_error(&frames,
                                                    format!("Axis method '{}' not registered", key),
                                                    line,
                                                );
                                                match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error, value_store, heavy_store) {
                                                    Ok(()) => return Ok(VMStatus::Continue),
                                                    Err(e) => return Err(e),
                                                }
                                            }
                                        } else {
                                            let error = ExceptionHandler::runtime_error(&frames,
                                                "Plot object not found".to_string(),
                                                line,
                                            );
                                            match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error, value_store, heavy_store) {
                                                Ok(()) => return Ok(VMStatus::Continue),
                                                Err(e) => return Err(e),
                                            }
                                        }
                                    } else {
                                        let error = ExceptionHandler::runtime_error(&frames,
                                            "Plot module not found".to_string(),
                                            line,
                                        );
                                        match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error, value_store, heavy_store) {
                                            Ok(()) => return Ok(VMStatus::Continue),
                                            Err(e) => return Err(e),
                                        }
                                    };
                                    // Return the native function
                                    // The compiler should arrange for axis to be passed as first argument
                                    stack::push_id(stack, store_value(Value::NativeFunction(method_index), value_store, heavy_store));
                                }
                                _ => {
                                    let error = ExceptionHandler::runtime_error(&frames,
                                        "Axis property access must use string key".to_string(),
                                        line,
                                    );
                                    match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error, value_store, heavy_store) {
                                        Ok(()) => return Ok(VMStatus::Continue),
                                        Err(e) => return Err(e),
                                    }
                                }
                            }
                            return Ok(VMStatus::Continue);
                        }
                        Value::Layer(_layer_id) => {
                            // Доступ к методам слоя по строковому ключу
                            match index_value {
                                Value::String(key) => {
                                    // Map method names to native function names in ml module
                                    let function_name = match key.as_str() {
                                        "freeze" => "layer_freeze",
                                        "unfreeze" => "layer_unfreeze",
                                        _ => {
                                            let error = ExceptionHandler::runtime_error_with_type(&frames,
                                                format!("Layer has no method '{}'. Available methods: freeze, unfreeze", key),
                                                line,
                                                ErrorType::KeyError,
                                            );
                                            match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error, value_store, heavy_store) {
                                                Ok(()) => return Ok(VMStatus::Continue),
                                                Err(e) => return Err(e),
                                            }
                                        }
                                    };
                                    
                                    // Get method index from ml object (stored during registration)
                                    let method_index = if let Some(ml_idx) = global_index_by_name(global_names, "ml") {
                                        if ml_idx >= globals.len() {
                                            let error = ExceptionHandler::runtime_error(&frames,
                                                "ML module not found in globals".to_string(),
                                                line,
                                            );
                                            match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error, value_store, heavy_store) {
                                                Ok(()) => return Ok(VMStatus::Continue),
                                                Err(e) => return Err(e),
                                            }
                                        }
                                        let ml_id = globals[ml_idx].resolve_to_value_id(value_store);
                                        let ml_val = load_value(ml_id, value_store, heavy_store);
                                        match &ml_val {
                                            Value::Object(map_rc) => {
                                                let map = map_rc.borrow();
                                                match map.get(function_name) {
                                                    Some(Value::NativeFunction(idx)) => *idx,
                                                    _ => {
                                                        let error = ExceptionHandler::runtime_error(&frames,
                                                            format!("Layer method '{}' not registered in ml module", key),
                                                            line,
                                                        );
                                                        match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error, value_store, heavy_store) {
                                                            Ok(()) => return Ok(VMStatus::Continue),
                                                            Err(e) => return Err(e),
                                                        }
                                                    }
                                                }
                                            }
                                            _ => {
                                                let error = ExceptionHandler::runtime_error(&frames,
                                                    "ML module is not an object".to_string(),
                                                    line,
                                                );
                                                match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error, value_store, heavy_store) {
                                                    Ok(()) => return Ok(VMStatus::Continue),
                                                    Err(e) => return Err(e),
                                                }
                                            }
                                        }
                                    } else {
                                        let error = ExceptionHandler::runtime_error(&frames,
                                            "ML module not found".to_string(),
                                            line,
                                        );
                                        match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error, value_store, heavy_store) {
                                            Ok(()) => return Ok(VMStatus::Continue),
                                            Err(e) => return Err(e),
                                        }
                                    };
                                    
                                    // Return the native function
                                    // The compiler should arrange for layer to be passed as first argument
                                    stack::push_id(stack, store_value(Value::NativeFunction(method_index), value_store, heavy_store));
                                }
                                _ => {
                                    let error = ExceptionHandler::runtime_error(&frames,
                                        "Layer property access must use string key".to_string(),
                                        line,
                                    );
                                    match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error, value_store, heavy_store) {
                                        Ok(()) => return Ok(VMStatus::Continue),
                                        Err(e) => return Err(e),
                                    }
                                }
                            }
                            return Ok(VMStatus::Continue);
                        }
                        Value::ColumnReference { table, column_name } => {
                            let index = match index_value {
                                Value::Number(n) => {
                                    let idx = n as i64;
                                    if idx < 0 {
                                        let error = ExceptionHandler::runtime_error(&frames,
                                            "Column index must be non-negative".to_string(),
                                            line,
                                        );
                            match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error, value_store, heavy_store) {
                                Ok(()) => return Ok(VMStatus::Continue),
                                Err(e) => return Err(e),
                            }
                                    }
                                    idx as usize
                                }
                                _ => {
                                    let error = ExceptionHandler::runtime_error(&frames,
                                        "Column index must be a number".to_string(),
                                        line,
                                    );
                            match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error, value_store, heavy_store) {
                                Ok(()) => return Ok(VMStatus::Continue),
                                Err(e) => return Err(e),
                            }
                                }
                            };
                            // Lazy: one cell via get_cell_value (no full column materialization). Cached column used when already built.
                            let pushed = if let Some(column) = table.borrow().get_column_cached(&column_name) {
                                if index < column.len() {
                                    stack::push_id(stack, store_value(column[index].clone(), value_store, heavy_store));
                                    true
                                } else { false }
                            } else {
                                let t = table.borrow();
                                if index < t.len() {
                                    if let Some(cell) = crate::vm::table_ops::get_cell_value(&*t, index, &column_name, value_store, heavy_store) {
                                        stack::push_id(stack, store_value(cell, value_store, heavy_store));
                                        true
                                    } else { false }
                                } else { false }
                            };
                            if !pushed {
                                let t = table.borrow();
                                let has = t.has_column(&column_name);
                                let len_opt = crate::vm::table_ops::column_len(&*t, &column_name);
                                let error = if has {
                                    ExceptionHandler::runtime_error_with_type(&frames,
                                        format!("Column index {} out of bounds{}", index,
                                            len_opt.map(|l| format!(" (length: {})", l)).unwrap_or_default()),
                                        line,
                                        ErrorType::IndexError,
                                    )
                                } else {
                                    ExceptionHandler::runtime_error_with_type(&frames,
                                        format!("Column '{}' not found", column_name),
                                        line,
                                        ErrorType::KeyError,
                                    )
                                };
                                match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error, value_store, heavy_store) {
                                    Ok(()) => return Ok(VMStatus::Continue),
                                    Err(e) => return Err(e),
                                }
                            }
                        }
                        Value::Path(path) => {
                            // Доступ к свойствам Path по строковому ключу
                            match index_value {
                                Value::String(property_name) => {
                                    match property_name.as_str() {
                                        "is_file" => {
                                            stack::push_id(stack, store_value(Value::Bool(path.is_file()), value_store, heavy_store));
                                        }
                                        "is_dir" => {
                                            stack::push_id(stack, store_value(Value::Bool(path.is_dir()), value_store, heavy_store));
                                        }
                                        "extension" => {
                                            if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
                                                stack::push_id(stack, store_value(Value::String(ext.to_string()), value_store, heavy_store));
                                            } else {
                                                stack::push_id(stack, NULL_VALUE_ID);
                                            }
                                        }
                                        "name" => {
                                            if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                                                stack::push_id(stack, store_value(Value::String(name.to_string()), value_store, heavy_store));
                                            } else {
                                                stack::push_id(stack, NULL_VALUE_ID);
                                            }
                                        }
                                        "parent" => {
                                            // Используем безопасную функцию для получения parent
                                            use crate::vm::natives::path::safe_path_parent;
                                            match safe_path_parent(&path) {
                                                Some(parent) => stack::push_id(stack, store_value(Value::Path(parent), value_store, heavy_store)),
                                                None => stack::push_id(stack, NULL_VALUE_ID),
                                            }
                                        }
                                        "exists" => {
                                            stack::push_id(stack, store_value(Value::Bool(path.exists()), value_store, heavy_store));
                                        }
                                        _ => {
                                            let error = ExceptionHandler::runtime_error(&frames,
                                                format!("Property '{}' not found on Path", property_name),
                                                line,
                                            );
                            match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error, value_store, heavy_store) {
                                Ok(()) => return Ok(VMStatus::Continue),
                                Err(e) => return Err(e),
                            }
                                        }
                                    }
                                }
                                _ => {
                                    let error = ExceptionHandler::runtime_error(&frames,
                                        "Path property access requires string index".to_string(),
                                        line,
                                    );
                            match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error, value_store, heavy_store) {
                                Ok(()) => return Ok(VMStatus::Continue),
                                Err(e) => return Err(e),
                            }
                                }
                            }
                        }
                        Value::Dataset(dataset) => {
                            let index = match index_value {
                                Value::Number(n) => {
                                    let idx = n as i64;
                                    if idx < 0 {
                                        let error = ExceptionHandler::runtime_error(&frames,
                                            "Dataset index must be non-negative".to_string(),
                                            line,
                                        );
                            match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error, value_store, heavy_store) {
                                Ok(()) => return Ok(VMStatus::Continue),
                                Err(e) => return Err(e),
                            }
                                    }
                                    idx as usize
                                }
                                _ => {
                                    let error = ExceptionHandler::runtime_error(&frames,
                                        "Dataset index must be a number".to_string(),
                                        line,
                                    );
                            match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error, value_store, heavy_store) {
                                Ok(()) => return Ok(VMStatus::Continue),
                                Err(e) => return Err(e),
                            }
                                }
                            };

                            let dataset_ref = dataset.borrow();
                            let batch_size = dataset_ref.batch_size();
                            
                            if index >= batch_size {
                                let error = ExceptionHandler::runtime_error_with_type(&frames,
                                    format!("Dataset index {} out of bounds (length: {})", index, batch_size),
                                    line,
                                    ErrorType::IndexError,
                                );
                            match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error, value_store, heavy_store) {
                                Ok(()) => return Ok(VMStatus::Continue),
                                Err(e) => return Err(e),
                            }
                            }

                            // Extract features for this sample
                            let num_features = dataset_ref.num_features();
                            let features_start = index * num_features;
                            let features_end = features_start + num_features;
                            // OPTIMIZATION: Use Vec::from for slice copy (slightly more efficient than to_vec)
                            let features_data: Vec<f32> = Vec::from(&dataset_ref.features().data[features_start..features_end]);
                            let features_tensor = Tensor::new(features_data, vec![num_features])
                                .map_err(|e| ExceptionHandler::runtime_error(&frames, format!("Failed to create features tensor: {}", e), line))?;

                            // Extract target for this sample
                            let num_targets = dataset_ref.num_targets();
                            let targets_start = index * num_targets;
                            let targets_end = targets_start + num_targets;
                            
                            // If target is a single value, return as Number; otherwise return as Tensor
                            let target_value = if num_targets == 1 {
                                Value::Number(dataset_ref.targets().data[targets_start] as f64)
                            } else {
                                // OPTIMIZATION: Use Vec::from for slice copy (slightly more efficient than to_vec)
                                let target_data: Vec<f32> = Vec::from(&dataset_ref.targets().data[targets_start..targets_end]);
                                let target_tensor = Tensor::new(target_data, vec![num_targets])
                                    .map_err(|e| ExceptionHandler::runtime_error(&frames, format!("Failed to create target tensor: {}", e), line))?;
                                Value::Tensor(Rc::new(RefCell::new(target_tensor)))
                            };

                            // Return [features, target] as array
                            let features_value = Value::Tensor(Rc::new(RefCell::new(features_tensor)));
                            let pair = vec![features_value, target_value];
                            stack::push_id(stack, store_value(Value::Array(Rc::new(RefCell::new(pair))), value_store, heavy_store));
                            return Ok(VMStatus::Continue);
                        }
                        Value::Tensor(tensor) => {
                            // Доступ к свойствам тензора по строковому ключу
                            match index_value {
                                Value::String(property_name) => {
                                    match property_name.as_str() {
                                        "shape" => {
                                            let tensor_ref = tensor.borrow();
                                            let shape_values: Vec<Value> = tensor_ref.shape.iter()
                                                .map(|&s| Value::Number(s as f64))
                                                .collect();
                                            stack::push_id(stack, store_value(Value::Array(Rc::new(RefCell::new(shape_values))), value_store, heavy_store));
                                        }
                                        "data" => {
                                            let tensor_ref = tensor.borrow();
                                            let data_values: Vec<Value> = tensor_ref.data.iter()
                                                .map(|&d| Value::Number(d as f64))
                                                .collect();
                                            stack::push_id(stack, store_value(Value::Array(Rc::new(RefCell::new(data_values))), value_store, heavy_store));
                                        }
                                        "max_idx" => {
                                            // Return a bound method: push tensor first, then function
                                            // When called, the function will receive tensor as first argument
                                            use crate::ml::natives;
                                            let max_idx_fn_ptr = natives::native_max_idx as *const ();
                                            let method_index = natives.iter().position(|&f| {
                                                let fn_ptr = f as *const ();
                                                std::ptr::eq(fn_ptr, max_idx_fn_ptr)
                                            });
                                            
                                            if let Some(idx) = method_index {
                                                // Push tensor onto stack first (will be used as first argument)
                                                stack::push_id(stack, store_value(Value::Tensor(Rc::clone(&tensor)), value_store, heavy_store));
                                                // Push native function
                                                stack::push_id(stack, store_value(Value::NativeFunction(idx), value_store, heavy_store));
                                            } else {
                                                let error = ExceptionHandler::runtime_error(&frames,
                                                    "max_idx method not found".to_string(),
                                                    line,
                                                );
                                                match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error, value_store, heavy_store) {
                                                    Ok(()) => return Ok(VMStatus::Continue),
                                                    Err(e) => return Err(e),
                                                }
                                            }
                                        }
                                        "min_idx" => {
                                            // Return a bound method: push tensor first, then function
                                            use crate::ml::natives;
                                            let min_idx_fn_ptr = natives::native_min_idx as *const ();
                                            let method_index = natives.iter().position(|&f| {
                                                let fn_ptr = f as *const ();
                                                std::ptr::eq(fn_ptr, min_idx_fn_ptr)
                                            });
                                            
                                            if let Some(idx) = method_index {
                                                // Push tensor onto stack first (will be used as first argument)
                                                stack::push_id(stack, store_value(Value::Tensor(Rc::clone(&tensor)), value_store, heavy_store));
                                                // Push native function
                                                stack::push_id(stack, store_value(Value::NativeFunction(idx), value_store, heavy_store));
                                            } else {
                                                let error = ExceptionHandler::runtime_error(&frames,
                                                    "min_idx method not found".to_string(),
                                                    line,
                                                );
                                                match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error, value_store, heavy_store) {
                                                    Ok(()) => return Ok(VMStatus::Continue),
                                                    Err(e) => return Err(e),
                                                }
                                            }
                                        }
                                        _ => {
                                            let error = ExceptionHandler::runtime_error(&frames,
                                                format!("Property '{}' not found on Tensor. Available properties: 'shape', 'data', 'max_idx', 'min_idx'", property_name),
                                                line,
                                            );
                                            match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error, value_store, heavy_store) {
                                                Ok(()) => return Ok(VMStatus::Continue),
                                                Err(e) => return Err(e),
                                            }
                                        }
                                    }
                                }
                                Value::Number(n) => {
                                    // Доступ к элементу тензора по индексу
                                    let idx = n as i64;
                                    if idx < 0 {
                                        let error = ExceptionHandler::runtime_error(&frames,
                                            "Tensor index must be non-negative".to_string(),
                                            line,
                                        );
                                        match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error, value_store, heavy_store) {
                                            Ok(()) => return Ok(VMStatus::Continue),
                                            Err(e) => return Err(e),
                                        }
                                    }
                                    let tensor_ref = tensor.borrow();
                                    let index = idx as usize;
                                    
                                    // For 1D tensors, return scalar (backward compatibility)
                                    if tensor_ref.ndim() == 1 {
                                        if index >= tensor_ref.shape[0] {
                                            let error = ExceptionHandler::runtime_error_with_type(&frames,
                                                format!("Tensor index {} out of bounds (size: {})", index, tensor_ref.shape[0]),
                                                line,
                                                ErrorType::IndexError,
                                            );
                                            match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error, value_store, heavy_store) {
                                                Ok(()) => return Ok(VMStatus::Continue),
                                                Err(e) => return Err(e),
                                            }
                                        }
                                        stack::push_id(stack, store_value(Value::Number(tensor_ref.data[index] as f64), value_store, heavy_store));
                                    } else {
                                        // For 2D+ tensors, return a slice (row) along first dimension
                                        match tensor_ref.get_row(index) {
                                            Ok(slice_tensor) => {
                                                stack::push_id(stack, store_value(Value::Tensor(Rc::new(RefCell::new(slice_tensor))), value_store, heavy_store));
                                            }
                                            Err(e) => {
                                                let error = ExceptionHandler::runtime_error_with_type(&frames,
                                                    e,
                                                    line,
                                                    ErrorType::IndexError,
                                                );
                                                match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error, value_store, heavy_store) {
                                                    Ok(()) => return Ok(VMStatus::Continue),
                                                    Err(e) => return Err(e),
                                                }
                                            }
                                        }
                                    }
                                }
                                _ => {
                                    let error = ExceptionHandler::runtime_error(&frames,
                                        "Tensor property access requires string key (e.g., 'shape', 'data') or numeric index".to_string(),
                                        line,
                                    );
                                    match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error, value_store, heavy_store) {
                                        Ok(()) => return Ok(VMStatus::Continue),
                                        Err(e) => return Err(e),
                                    }
                                }
                            }
                            return Ok(VMStatus::Continue);
                        }
                        Value::NeuralNetwork(nn_rc) => {
                            // Доступ к методам и свойствам нейронной сети по строковому ключу
                            match index_value {
                                Value::String(key) => {
                                    // Check if this is the "layers" property
                                    if key == "layers" {
                                        // Return a special object that can be indexed to get layers
                                        // This object stores a reference to the NeuralNetwork
                                        use std::collections::HashMap;
                                        let mut layer_accessor = HashMap::new();
                                        layer_accessor.insert("__neural_network".to_string(), Value::NeuralNetwork(Rc::clone(&nn_rc)));
                                        stack::push_id(stack, store_value(Value::Object(Rc::new(RefCell::new(layer_accessor))), value_store, heavy_store));
                                        return Ok(VMStatus::Continue);
                                    }
                                    
                                    // Map property names to native function names in ml module
                                    let function_name = match key.as_str() {
                                        "train" => "nn_train",
                                        "train_sh" => "nn_train_sh",
                                        "save" => "nn_save",
                                        "device" => "nn_set_device",
                                        "get_device" => "nn_get_device",
                                        _ => {
                                            let error = ExceptionHandler::runtime_error_with_type(&frames,
                                                format!("NeuralNetwork has no method '{}'. Available methods: train, train_sh, save, device, get_device, layers", key),
                                                line,
                                                ErrorType::KeyError,
                                            );
                                            match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error, value_store, heavy_store) {
                                                Ok(()) => return Ok(VMStatus::Continue),
                                                Err(e) => return Err(e),
                                            }
                                        }
                                    };
                                    
                                    // Get method index from ml object (stored during registration)
                                    let method_index = if let Some(ml_idx) = global_index_by_name(global_names, "ml") {
                                        if ml_idx >= globals.len() {
                                            let error = ExceptionHandler::runtime_error(&frames,
                                                "ML module not found in globals".to_string(),
                                                line,
                                            );
                                            match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error, value_store, heavy_store) {
                                                Ok(()) => return Ok(VMStatus::Continue),
                                                Err(e) => return Err(e),
                                            }
                                        }
                                        let ml_id = globals[ml_idx].resolve_to_value_id(value_store);
                                        let ml_val = load_value(ml_id, value_store, heavy_store);
                                        match &ml_val {
                                            Value::Object(map_rc) => {
                                                let map = map_rc.borrow();
                                                match map.get(function_name) {
                                                    Some(Value::NativeFunction(idx)) => *idx,
                                                    _ => {
                                                        let error = ExceptionHandler::runtime_error(&frames,
                                                            format!("NeuralNetwork method '{}' not registered in ml module", key),
                                                            line,
                                                        );
                                                        match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error, value_store, heavy_store) {
                                                            Ok(()) => return Ok(VMStatus::Continue),
                                                            Err(e) => return Err(e),
                                                        }
                                                    }
                                                }
                                            }
                                            _ => {
                                                let error = ExceptionHandler::runtime_error(&frames,
                                                    "ML module is not an object".to_string(),
                                                    line,
                                                );
                                                match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error, value_store, heavy_store) {
                                                    Ok(()) => return Ok(VMStatus::Continue),
                                                    Err(e) => return Err(e),
                                                }
                                            }
                                        }
                                    } else {
                                        let error = ExceptionHandler::runtime_error(&frames,
                                            "ML module not found".to_string(),
                                            line,
                                        );
                                        match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error, value_store, heavy_store) {
                                            Ok(()) => return Ok(VMStatus::Continue),
                                            Err(e) => return Err(e),
                                        }
                                    };
                                    // Return the native function
                                    // The compiler should arrange for neural network to be passed as first argument
                                    stack::push_id(stack, store_value(Value::NativeFunction(method_index), value_store, heavy_store));
                                }
                                _ => {
                                    let error = ExceptionHandler::runtime_error(&frames,
                                        "NeuralNetwork property access must use string key".to_string(),
                                        line,
                                    );
                                    match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error, value_store, heavy_store) {
                                        Ok(()) => return Ok(VMStatus::Continue),
                                        Err(e) => return Err(e),
                                    }
                                }
                            }
                            return Ok(VMStatus::Continue);
                        }
                        Value::DatabaseEngine(engine_rc) => {
                            // Access to engine methods: connect, execute, query
                            match &index_value {
                                Value::String(property_name) => {
                                    use crate::database::natives;
                                    let (connect_fn, execute_fn, query_fn, run_fn) = (
                                        natives::native_engine_connect as *const (),
                                        natives::native_engine_execute as *const (),
                                        natives::native_engine_query as *const (),
                                        natives::native_engine_run as *const (),
                                    );
                                    let method_index = match property_name.as_str() {
                                        "connect" => natives.iter().position(|&f| std::ptr::eq(f as *const (), connect_fn)),
                                        "execute" => natives.iter().position(|&f| std::ptr::eq(f as *const (), execute_fn)),
                                        "query" => natives.iter().position(|&f| std::ptr::eq(f as *const (), query_fn)),
                                        "run" => natives.iter().position(|&f| std::ptr::eq(f as *const (), run_fn)),
                                        _ => {
                                            let error = ExceptionHandler::runtime_error(&frames,
                                                format!("DatabaseEngine has no property '{}'. Available: connect, execute, query, run", property_name),
                                                line,
                                            );
                                            match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error, value_store, heavy_store) {
                                                Ok(()) => return Ok(VMStatus::Continue),
                                                Err(e) => return Err(e),
                                            }
                                        }
                                    };
                                    if let Some(idx) = method_index {
                                        stack::push_id(stack, store_value(Value::DatabaseEngine(Rc::clone(&engine_rc)), value_store, heavy_store));
                                        stack::push_id(stack, store_value(Value::NativeFunction(idx), value_store, heavy_store));
                                    } else {
                                        let error = ExceptionHandler::runtime_error(&frames,
                                            format!("Database engine method '{}' not found", property_name),
                                            line,
                                        );
                                        match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error, value_store, heavy_store) {
                                            Ok(()) => return Ok(VMStatus::Continue),
                                            Err(e) => return Err(e),
                                        }
                                    }
                                }
                                _ => {
                                    let error = ExceptionHandler::runtime_error(&frames,
                                        "DatabaseEngine property access requires string key (connect, execute, query, run)".to_string(),
                                        line,
                                    );
                                    match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error, value_store, heavy_store) {
                                        Ok(()) => return Ok(VMStatus::Continue),
                                        Err(e) => return Err(e),
                                    }
                                }
                            }
                            return Ok(VMStatus::Continue);
                        }
                        Value::DatabaseCluster(cluster_rc) => {
                            match &index_value {
                                Value::String(property_name) => {
                                    use crate::database::natives;
                                    let (add_fn, get_fn, names_fn) = (
                                        natives::native_cluster_add as *const (),
                                        natives::native_cluster_get as *const (),
                                        natives::native_cluster_names as *const (),
                                    );
                                    let method_index = match property_name.as_str() {
                                        "add" => natives.iter().position(|&f| std::ptr::eq(f as *const (), add_fn)),
                                        "get" => natives.iter().position(|&f| std::ptr::eq(f as *const (), get_fn)),
                                        "names" => natives.iter().position(|&f| std::ptr::eq(f as *const (), names_fn)),
                                        _ => {
                                            let error = ExceptionHandler::runtime_error(&frames,
                                                format!("DatabaseCluster has no property '{}'. Available: add, get, names", property_name),
                                                line,
                                            );
                                            match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error, value_store, heavy_store) {
                                                Ok(()) => return Ok(VMStatus::Continue),
                                                Err(e) => return Err(e),
                                            }
                                        }
                                    };
                                    if let Some(idx) = method_index {
                                        stack::push_id(stack, store_value(Value::DatabaseCluster(Rc::clone(&cluster_rc)), value_store, heavy_store));
                                        stack::push_id(stack, store_value(Value::NativeFunction(idx), value_store, heavy_store));
                                    } else {
                                        let error = ExceptionHandler::runtime_error(&frames,
                                            format!("Database cluster method '{}' not found", property_name),
                                            line,
                                        );
                                        match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error, value_store, heavy_store) {
                                            Ok(()) => return Ok(VMStatus::Continue),
                                            Err(e) => return Err(e),
                                        }
                                    }
                                }
                                _ => {
                                    let error = ExceptionHandler::runtime_error(&frames,
                                        "DatabaseCluster property access requires string key (add, get, names)".to_string(),
                                        line,
                                    );
                                    match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error, value_store, heavy_store) {
                                        Ok(()) => return Ok(VMStatus::Continue),
                                        Err(e) => return Err(e),
                                    }
                                }
                            }
                            return Ok(VMStatus::Continue);
                        }
                        Value::String(s) => {
                            match index_value {
                                Value::Number(n) => {
                                    let idx = n as i64;
                                    if idx < 0 {
                                        let error = ExceptionHandler::runtime_error(
                                            &frames,
                                            "String index must be non-negative".to_string(),
                                            line,
                                        );
                                        match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error, value_store, heavy_store) {
                                            Ok(()) => return Ok(VMStatus::Continue),
                                            Err(e) => return Err(e),
                                        }
                                    }
                                    let idx_usize = idx as usize;
                                    if let Some(ch) = s.chars().nth(idx_usize) {
                                        stack::push_id(stack, store_value(Value::String(ch.to_string()), value_store, heavy_store));
                                    } else {
                                        let error = ExceptionHandler::runtime_error_with_type(
                                            &frames,
                                            format!("String index {} out of bounds (length: {})", idx_usize, s.chars().count()),
                                            line,
                                            ErrorType::IndexError,
                                        );
                                        match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error, value_store, heavy_store) {
                                            Ok(()) => return Ok(VMStatus::Continue),
                                            Err(e) => return Err(e),
                                        }
                                    }
                                }
                                Value::String(key) => {
                                    // String methods: upper=27, lower=28, trim=29, split=30, join=31, contains=32, isupper=33, islower=34
                                    let method_index = match key.as_str() {
                                        "upper" => Some(27),
                                        "lower" => Some(28),
                                        "trim" => Some(29),
                                        "split" => Some(30),
                                        "join" => Some(31),
                                        "contains" => Some(32),
                                        "isupper" => Some(33),
                                        "islower" => Some(34),
                                        _ => None,
                                    };
                                    if let Some(idx) = method_index {
                                        stack::push_id(stack, store_value(Value::NativeFunction(idx), value_store, heavy_store));
                                    } else {
                                        let error = ExceptionHandler::runtime_error(
                                            &frames,
                                            format!("String has no property '{}'. Available: upper, lower, trim, split, join, contains, isupper, islower, or use numeric index", key),
                                            line,
                                        );
                                        match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error, value_store, heavy_store) {
                                            Ok(()) => return Ok(VMStatus::Continue),
                                            Err(e) => return Err(e),
                                        }
                                    }
                                }
                                _ => {
                                    let error = ExceptionHandler::runtime_error(
                                        &frames,
                                        "String index must be a number or a method name (upper, lower, trim, split, join, contains, isupper, islower)".to_string(),
                                        line,
                                    );
                                    match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error, value_store, heavy_store) {
                                        Ok(()) => return Ok(VMStatus::Continue),
                                        Err(e) => return Err(e),
                                    }
                                }
                            }
                            return Ok(VMStatus::Continue);
                        }
                        Value::NativeFunction(native_index) => {
                            // str[N] -> type descriptor for Column (string with max length N)
                            use crate::vm::natives::basic::native_str;
                            if native_index < natives.len()
                                && std::ptr::eq(natives[native_index] as *const (), native_str as *const ())
                            {
                                if let Value::Number(n) = index_value {
                                    let idx = n as i64;
                                    if idx >= 0 && n.fract() == 0.0 {
                                        let mut desc = std::collections::HashMap::new();
                                        desc.insert("__type".to_string(), Value::String("str".to_string()));
                                        desc.insert("__length".to_string(), Value::Number(idx as f64));
                                        stack::push_id(stack, store_value(Value::Object(Rc::new(RefCell::new(desc))), value_store, heavy_store));
                                        return Ok(VMStatus::Continue);
                                    }
                                }
                            }
                            let error = ExceptionHandler::runtime_error(
                                &frames,
                                "Expected array, tuple, column reference, table, object, path, dataset, tensor, neural network, database engine, or database cluster for GetArrayElement".to_string(),
                                line,
                            );
                            match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error, value_store, heavy_store) {
                                Ok(()) => return Ok(VMStatus::Continue),
                                Err(e) => return Err(e),
                            }
                        }
                        Value::Null => {
                            let error = ExceptionHandler::runtime_error(
                            &frames,
                                "Cannot access element of null value".to_string(),
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
                                "Expected array, tuple, column reference, table, object, path, dataset, tensor, neural network, database engine, or database cluster for GetArrayElement".to_string(),
                                line,
                            );
                            match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error, value_store, heavy_store) {
                                Ok(()) => return Ok(VMStatus::Continue),
                                Err(e) => return Err(e),
                            }
                        }
                    }
                    return Ok(VMStatus::Continue);
                }
                OpCode::SetArrayElement => {
                    // Stack order from compiler: [value, index, container] with container on top.
                    let container_id = pop_to_value_id(stack, frames, exception_handlers, value_store, heavy_store)?;
                    let index_tv = stack::pop(stack, frames, exception_handlers, value_store, heavy_store)?;
                    let value_tv = stack::pop(stack, frames, exception_handlers, value_store, heavy_store)?;
                    let value_id = tagged_to_value_id(value_tv, value_store);
                    // Fast path: ValueCell::Array + number index — store TaggedValue slot
                    if index_tv.is_number() {
                        let idx = index_tv.get_f64() as i64;
                        if idx >= 0 && index_tv.get_f64().fract() == 0.0 {
                            let u = idx as usize;
                            if let Some(ValueCell::Array(slots)) = value_store.get_mut(container_id) {
                                if u >= slots.len() {
                                    slots.resize(u + 1, TaggedValue::null());
                                }
                                slots[u] = value_tv;
                                stack::push_id(stack, container_id);
                                return Ok(VMStatus::Continue);
                            }
                        }
                    }
                    let container = load_value(container_id, value_store, heavy_store);
                    let index_value = load_value(tagged_to_value_id(index_tv, value_store), value_store, heavy_store);
                    let value = load_value(value_id, value_store, heavy_store);
                    let container_type = match &container {
                        Value::Array(_) => "Array",
                        Value::Object(_) => "Object",
                        Value::Table(_) => "Table",
                        _ => "Other",
                    };
                    let key_str = match &index_value {
                        Value::String(k) => k.clone(),
                        Value::Number(n) => format!("{}", n),
                        _ => format!("{:?}", index_value),
                    };
                    let value_type_str = match &value {
                        Value::Function(fn_idx) => {
                            if *fn_idx < functions.len() {
                                format!("Function({}, имя: '{}')", fn_idx, functions[*fn_idx].name)
                            } else {
                                format!("Function({}, OUT OF BOUNDS!)", fn_idx)
                            }
                        },
                        Value::NativeFunction(_) => "NativeFunction".to_string(),
                        _ => format!("{:?}", value),
                    };
                    debug_println!("[DEBUG SetArrayElement] line {}, {} key='{}' value={}", line, container_type, key_str, value_type_str);
                    
                    match container {
                        Value::Array(_) => {
                            let index = match index_value {
                                Value::Number(n) => {
                                    let idx = n as i64;
                                    if idx < 0 {
                                        let error = ExceptionHandler::runtime_error(
                                            &frames,
                                            "Array index must be non-negative".to_string(),
                                            line,
                                        );
                                        match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error, value_store, heavy_store) {
                                            Ok(()) => return Ok(VMStatus::Continue),
                                            Err(e) => return Err(e),
                                        }
                                    }
                                    idx as usize
                                }
                                _ => {
                                    let error = ExceptionHandler::runtime_error(
                                        &frames,
                                        "Array index must be a number".to_string(),
                                        line,
                                    );
                                    match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error, value_store, heavy_store) {
                                        Ok(()) => return Ok(VMStatus::Continue),
                                        Err(e) => return Err(e),
                                    }
                                }
                            };
                            // Update the array cell in place so all references see the change.
                            let slot_tv = match &value {
                                Value::Number(n) => TaggedValue::from_f64(*n),
                                Value::Bool(b) => TaggedValue::from_bool(*b),
                                Value::Null => TaggedValue::null(),
                                _ => TaggedValue::from_heap(store_value(value.clone(), value_store, heavy_store)),
                            };
                            if let Some(ValueCell::Array(slots)) = value_store.get_mut(container_id) {
                                if index >= slots.len() {
                                    slots.resize(index + 1, TaggedValue::null());
                                }
                                slots[index] = slot_tv;
                            }
                            stack::push_id(stack, container_id);
                            return Ok(VMStatus::Continue);
                        }
                        Value::Object(obj_rc) => {
                            // Для объектов индекс должен быть строкой
                            let key = match index_value {
                                Value::String(key) => key,
                                _ => {
                                    let error = ExceptionHandler::runtime_error(
                                        &frames,
                                        "Object key must be a string".to_string(),
                                        line,
                                    );
                                    match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error, value_store, heavy_store) {
                                        Ok(()) => return Ok(VMStatus::Continue),
                                        Err(e) => return Err(e),
                                    }
                                }
                            };
                            // Class private variables: forbid write from outside (ProtectError)
                            let is_class_private_var = {
                                let map = obj_rc.borrow();
                                map.contains_key("__class_name") && key != "model_config"
                                    && map.get("__class_private_vars").and_then(|v| {
                                        if let Value::Array(rc) = v {
                                            Some(rc.borrow().iter().any(|v| {
                                                if let Value::String(s) = v { s.as_str() == key } else { false }
                                            }))
                                        } else {
                                            None
                                        }
                                    }).unwrap_or(false)
                            };
                            // Allow write from single <main> frame (class variable initialization at module load)
                            let allow_main_init = frames.len() == 1
                                && frames.first().map(|f| f.function.name.as_str()) == Some("<main>");
                            if is_class_private_var && !allow_main_init {
                                let class_name = obj_rc.borrow().get("__class_name")
                                    .and_then(|v| if let Value::String(s) = v { Some(s.clone()) } else { None })
                                    .unwrap_or_else(|| "?".to_string());
                                let msg = format!("Class variable '{}' is private in '{}' and cannot be accessed from outside the class", key, class_name);
                                let error = ExceptionHandler::runtime_error_with_type(
                                    &frames,
                                    msg,
                                    line,
                                    ErrorType::ProtectError,
                                );
                                match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error, value_store, heavy_store) {
                                    Ok(()) => return Ok(VMStatus::Continue),
                                    Err(e) => return Err(e),
                                }
                            }
                            // Instance private fields: allow write only from the defining class (not subclasses)
                            let (is_instance_private_denied, private_error_msg) = {
                                let map = obj_rc.borrow();
                                if let (Some(Value::String(ref class_name)), Some(Value::Array(private_fields_rc))) = (
                                    map.get("__class_name"),
                                    map.get("__private_fields"),
                                ) {
                                    let is_private = private_fields_rc.borrow().iter().any(|v| {
                                        if let Value::String(s) = v { s.as_str() == key } else { false }
                                    });
                                    if !is_private {
                                        (false, String::new())
                                    } else {
                                        let defining_class = map.get("__private_field_defining_class")
                                            .and_then(|v| {
                                                if let Value::Object(rc) = v {
                                                    rc.borrow().get(&key).and_then(|v| {
                                                        if let Value::String(s) = v { Some(s.clone()) } else { None }
                                                    })
                                                } else { None }
                                            })
                                            .unwrap_or_else(|| class_name.clone());
                                        let in_defining_class = frames.iter().any(|f| {
                                            f.function.name.starts_with(&format!("{}::", defining_class))
                                        });
                                        if in_defining_class {
                                            (false, String::new())
                                        } else {
                                            let frame_class_opt = frames.iter().rev()
                                                .find_map(|f| {
                                                    if f.function.name.contains("::new_") || f.function.name.contains("::method_") {
                                                        f.function.name.split("::").next().map(String::from)
                                                    } else {
                                                        None
                                                    }
                                                });
                                            let msg = match &frame_class_opt {
                                                Some(class) => format!("Field '{}' is private in '{}' and cannot be accessed from subclass '{}'", key, defining_class, class),
                                                None => format!("Field '{}' is private in '{}' and cannot be accessed from outside the class", key, defining_class),
                                            };
                                            (true, msg)
                                        }
                                    }
                                } else {
                                    (false, String::new())
                                }
                            };
                            if is_instance_private_denied {
                                let error = ExceptionHandler::runtime_error_with_type(
                                    &frames,
                                    private_error_msg,
                                    line,
                                    ErrorType::ProtectError,
                                );
                                match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error, value_store, heavy_store) {
                                    Ok(()) => return Ok(VMStatus::Continue),
                                    Err(e) => return Err(e),
                                }
                            }
                            // Class protected variables: allow write only from this class or subclasses
                            let (is_class_protected_denied, class_protected_msg) = {
                                let map = obj_rc.borrow();
                                let is_protected_var = map.contains_key("__class_name") && key != "model_config"
                                    && map.get("__class_protected_vars").and_then(|v| {
                                        if let Value::Array(rc) = v {
                                            Some(rc.borrow().iter().any(|v| {
                                                if let Value::String(s) = v { s.as_str() == key } else { false }
                                            }))
                                        } else {
                                            None
                                        }
                                    }).unwrap_or(false);
                                if !is_protected_var {
                                    (false, String::new())
                                } else {
                                    let obj_class = map.get("__class_name").and_then(|v| if let Value::String(s) = v { Some(s.clone()) } else { None });
                                    let in_hierarchy = obj_class.as_ref().map(|obj_class| {
                                        frames.iter().any(|f| {
                                            let frame_class = f.function.name.split("::").next().unwrap_or("");
                                            let frame_chain = get_superclass_chain(globals, global_names, frame_class, value_store, heavy_store);
                                            frame_chain.iter().any(|c| c == obj_class)
                                        })
                                    }).unwrap_or(false);
                                    if in_hierarchy {
                                        (false, String::new())
                                    } else {
                                        let frame_class_opt = frames.iter().rev()
                                            .find_map(|f| {
                                                if f.function.name.contains("::new_") || f.function.name.contains("::method_") {
                                                    f.function.name.split("::").next().map(String::from)
                                                } else {
                                                    None
                                                }
                                            });
                                        let class_name = obj_class.as_deref().unwrap_or("?");
                                        let msg = match &frame_class_opt {
                                            Some(class) => format!("Class variable '{}' is protected in '{}' and cannot be accessed from subclass '{}'", key, class_name, class),
                                            None => format!("Class variable '{}' is protected in '{}' and cannot be accessed from outside the class", key, class_name),
                                        };
                                        (true, msg)
                                    }
                                }
                            };
                            if is_class_protected_denied {
                                let error = ExceptionHandler::runtime_error_with_type(
                                    &frames,
                                    class_protected_msg,
                                    line,
                                    ErrorType::ProtectError,
                                );
                                match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error, value_store, heavy_store) {
                                    Ok(()) => return Ok(VMStatus::Continue),
                                    Err(e) => return Err(e),
                                }
                            }
                            // Instance protected fields: allow write only from this class or subclasses
                            let (is_instance_protected_denied, instance_protected_msg) = {
                                let map = obj_rc.borrow();
                                if let (Some(Value::String(ref instance_class)), Some(Value::Array(protected_fields_rc))) = (
                                    map.get("__class_name"),
                                    map.get("__protected_fields"),
                                ) {
                                    let is_protected = protected_fields_rc.borrow().iter().any(|v| {
                                        if let Value::String(s) = v { s.as_str() == key } else { false }
                                    });
                                    let in_hierarchy = {
                                        let instance_chain = get_superclass_chain(globals, global_names, instance_class, value_store, heavy_store);
                                        frames.iter().any(|f| {
                                            let frame_class = f.function.name.split("::").next().unwrap_or("");
                                            instance_chain.iter().any(|c| c == frame_class)
                                        })
                                    };
                                    if is_protected && !in_hierarchy {
                                        let frame_class_opt = frames.iter().rev()
                                            .find_map(|f| {
                                                if f.function.name.contains("::new_") || f.function.name.contains("::method_") {
                                                    f.function.name.split("::").next().map(String::from)
                                                } else {
                                                    None
                                                }
                                            });
                                        let msg = match &frame_class_opt {
                                            Some(class) => format!("Field '{}' is protected in '{}' and cannot be accessed from subclass '{}'", key, instance_class, class),
                                            None => format!("Field '{}' is protected in '{}' and cannot be accessed from outside the class", key, instance_class),
                                        };
                                        (true, msg)
                                    } else {
                                        (false, String::new())
                                    }
                                } else {
                                    (false, String::new())
                                }
                            };
                            if is_instance_protected_denied {
                                let error = ExceptionHandler::runtime_error_with_type(
                                    &frames,
                                    instance_protected_msg,
                                    line,
                                    ErrorType::ProtectError,
                                );
                                match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error, value_store, heavy_store) {
                                    Ok(()) => return Ok(VMStatus::Continue),
                                    Err(e) => return Err(e),
                                }
                            }
                            // Update the object cell in place so all references see the change.
                            if let Some(ValueCell::Object(map)) = value_store.get_mut(container_id) {
                                map.insert(key.clone(), value_id);
                                debug_println!("[DEBUG SetArrayElement] object key '{}' set, keys now: {}", key, map.len());
                            }
                            stack::push_id(stack, container_id);
                            return Ok(VMStatus::Continue);
                        }
                        _ => {
                            let error = ExceptionHandler::runtime_error(
                                &frames,
                                format!("SetArrayElement only supports arrays and objects, got: {:?}", container),
                                line,
                            );
                            match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error, value_store, heavy_store) {
                                Ok(()) => return Ok(VMStatus::Continue),
                                Err(e) => return Err(e),
                            }
                        }
                    }
                }
                OpCode::Clone => {
                    let value_id = pop_to_value_id(stack, frames, exception_handlers, value_store, heavy_store)?;
                    let value = load_value(value_id, value_store, heavy_store);
                    let cloned = value.clone();
                    stack::push_id(stack, store_value(cloned, value_store, heavy_store));
                    return Ok(VMStatus::Continue);
                }
                OpCode::BeginTry(handler_index) => {
                    // Начало try блока - загружаем обработчик из chunk
                    let frame = frames.last().unwrap();
                    let chunk = &frame.function.chunk;
                    
                    // Загружаем информацию об обработчике из chunk
                    if handler_index < chunk.exception_handlers.len() {
                        let handler_info = &chunk.exception_handlers[handler_index];
                        
                        // Копируем таблицу типов ошибок в VM (если еще не скопирована)
                        if error_type_table.is_empty() {
                            *error_type_table = chunk.error_type_table.clone();
                        }
                        
                        // Сохраняем текущую высоту стека
                        let stack_height = stack.len();
                        
                        // Создаем обработчик с информацией из chunk
                        let frame_index = frames.len() - 1;
                        let handler = ExceptionHandler {
                            catch_ips: handler_info.catch_ips.clone(),
                            error_types: handler_info.error_types.clone(),
                            error_var_slots: handler_info.error_var_slots.clone(),
                            else_ip: handler_info.else_ip,
                            finally_ip: handler_info.finally_ip,
                            stack_height,
                            had_error: false,
                            frame_index,
                        };
                        exception_handlers.push(handler);
                    } else {
                        // Если обработчик не найден, создаем пустой (fallback)
                        let stack_height = stack.len();
                        let frame_index = frames.len() - 1;
                        let handler = ExceptionHandler {
                            catch_ips: Vec::new(),
                            error_types: Vec::new(),
                            error_var_slots: Vec::new(),
                            else_ip: None,
                            finally_ip: None,
                            stack_height,
                            had_error: false,
                            frame_index,
                        };
                        exception_handlers.push(handler);
                    }
                    return Ok(VMStatus::Continue);
                }
                OpCode::EndTry => {
                    // Конец try блока - если выполнение дошло сюда без ошибок
                    // Проверяем, была ли ошибка
                    if let Some(handler) = exception_handlers.last_mut() {
                        // Если не было ошибки и есть else блок, переходим к нему
                        if !handler.had_error {
                            if let Some(else_ip) = handler.else_ip {
                                let frame = frames.last_mut().unwrap();
                                frame.ip = else_ip;
                            }
                        }
                        // Удаляем обработчик из стека
                        exception_handlers.pop();
                    }
                    return Ok(VMStatus::Continue);
                }
                OpCode::Catch(_) => {
                    // Начало catch блока - этот опкод используется только для маркировки
                    // Реальная логика обработки выполняется в handle_exception()
                    // Здесь просто продолжаем выполнение
                    return Ok(VMStatus::Continue);
                }
                OpCode::EndCatch => {
                    // Конец catch блока - продолжаем выполнение после catch
                    // Обработчик будет удален при PopExceptionHandler
                    return Ok(VMStatus::Continue);
                }
                OpCode::Throw(_) => {
                    // Выбрасывание исключения
                    // Получаем значение со стека (сообщение об ошибке)
                    let error_value_id = pop_to_value_id(stack, frames, exception_handlers, value_store, heavy_store)?;
                    let error_value = load_value(error_value_id, value_store, heavy_store);
                    // Преобразуем значение в строку
                    let error_message = error_value.to_string();
                    
                    // Создаем LangError
                    let error = LangError::runtime_error(error_message, line);
                    
                    // Пытаемся найти обработчик исключения
                    match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error, value_store, heavy_store) {
                        Ok(()) => {
                            // Обработчик найден, выполнение продолжается в catch блоке
                            // handle_exception уже настроил стек и фреймы
                        }
                        Err(e) => {
                            // Обработчик не найден - возвращаем ошибку (программа завершается)
                            return Err(e);
                        }
                    }
                }
                OpCode::PopExceptionHandler => {
                    // Удаление обработчика исключений со стека
                    exception_handlers.pop();
                    return Ok(VMStatus::Continue);
                }
    }

    Ok(VMStatus::Continue)
}

