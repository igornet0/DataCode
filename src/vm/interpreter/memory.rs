//! LoadGlobal and StoreGlobal opcodes.

use crate::debug_println;
use crate::common::{error::LangError, value::Value, value_store::{ValueCell, ValueStore, NULL_VALUE_ID}, TaggedValue};
use crate::vm::types::VMStatus;
use crate::vm::frame::CallFrame;
use crate::vm::exceptions::ExceptionHandler;
use crate::vm::stack;
use crate::vm::modules;
use crate::vm::global_slot::{GlobalSlot, default_global_slot};
use crate::vm::module_object::BUILTIN_END;
use crate::vm::store_convert::{store_value, store_value_arena, load_value, tagged_to_value_id_arena};
use crate::vm::heavy_store::HeavyStore;
use crate::vm::executor::global_index_by_name;
use std::rc::Rc;
use std::cell::RefCell;

/// Execute LoadGlobal(index).
#[allow(clippy::too_many_arguments)]
pub fn op_load_global(
    index: usize,
    line: usize,
    stack: &mut Vec<TaggedValue>,
    frames: &mut Vec<CallFrame>,
    globals: &mut Vec<GlobalSlot>,
    global_names: &mut std::collections::BTreeMap<usize, String>,
    functions: &mut Vec<crate::bytecode::Function>,
    exception_handlers: &mut Vec<ExceptionHandler>,
    loaded_modules: &mut std::collections::HashSet<String>,
    value_store: &mut ValueStore,
    heavy_store: &mut HeavyStore,
    vm_ptr: *mut crate::vm::vm::Vm,
) -> Result<VMStatus, LangError> {
    let frame = frames.last().unwrap();
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
                if crate::common::debug::is_debug_enabled() {
                    let var_name = frame.function.chunk.global_names.get(&index).map(String::as_str);
                    let mod_keys: Vec<_> = unsafe { (*vm_ptr).get_modules() }.keys().cloned().collect();
                    debug_println!(
                        "[DEBUG LoadGlobal module] frame.module_name={:?} var_name={:?} index={} mod_found={} mod_keys_sample={:?}",
                        mod_name,
                        var_name,
                        index,
                        module_rc.is_some(),
                        mod_keys.get(..5.min(mod_keys.len())),
                    );
                }
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
                                if let Some(rc) = modules.get(class_name) {
                                    if let Some(v) = rc.borrow().get_export(class_name) {
                                        return Some(v);
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
                        if crate::common::debug::is_debug_enabled() {
                            let vt = match &value {
                                Value::NativeFunction(_) => "NativeFunction",
                                Value::Function(_) => "Function",
                                Value::ModuleFunction { .. } => "ModuleFunction",
                                Value::Object(_) => "Object",
                                _ => "Other",
                            };
                            debug_println!("[DEBUG LoadGlobal module] get_export({:?}) -> {} (using module path)", var_name, vt);
                        }
                        let id = store_value(value, value_store, heavy_store);
                        stack::push_id(stack, id);
                        return Ok(VMStatus::Continue);
                    }
                    if crate::common::debug::is_debug_enabled() {
                        debug_println!("[DEBUG LoadGlobal module] get_export({:?}) returned None, trying fallback modules", var_name);
                    }
                    // Fallback: primary module (e.g. "config") may not have the export; try all modules
                    // (e.g. "settings" lives in "core.config" top-level, not "config" submodule).
                    if let Some(name) = var_name {
                        if name != "__constructing_class__" {
                            let modules = unsafe { (*vm_ptr).get_modules() };
                            for (_mod_key, rc) in modules.iter() {
                                if let Some(v) = rc.borrow().get_export(name) {
                                    let id = store_value(v, value_store, heavy_store);
                                    stack::push_id(stack, id);
                                    return Ok(VMStatus::Continue);
                                }
                            }
                        }
                    }
                    if !(var_name == Some("__constructing_class__") && frame.function.name.contains("::new_")) {
                        if crate::common::debug::is_debug_enabled() {
                            debug_println!("[DEBUG LoadGlobal module] module {:?} not found or get_export failed for {:?}, falling through to globals", mod_name, var_name);
                        }
                        let err_name = var_name.unwrap_or("?").to_string();
                        return Err(LangError::runtime_error(
                            format!("Undefined variable: {}", err_name),
                            line,
                        ));
                    }
                } else if crate::common::debug::is_debug_enabled() {
                    let var_name = frame.function.chunk.global_names.get(&index).map(String::as_str);
                    debug_println!("[DEBUG LoadGlobal module] modules.get({:?}) = None, falling through to globals for {:?}", mod_name, var_name);
                }
            }
        }

        let mut effective_index = index;
        let argv_slot_for_resolve = unsafe { (*vm_ptr).get_argv_slot_index() };
        if index >= globals.len() {
            // Resolve by name from current chunk (e.g. merged module function with unpatched sentinel)
            if let Some(var_name) = frame.function.chunk.global_names.get(&index) {
                let real_idx_opt = if *var_name == "argv" && argv_slot_for_resolve.is_some() {
                    argv_slot_for_resolve
                } else {
                    global_names.iter()
                        .filter(|(_, n)| n.as_str() == var_name.as_str())
                        .map(|(i, _)| *i)
                        .min()
                };
                if let Some(ri) = real_idx_opt {
                    effective_index = ri;
                    if effective_index >= globals.len() {
                        globals.resize(effective_index + 1, default_global_slot());
                    }
                }
            }
        }
        // In main chunk, when loading argv by original index and chunk says this index is "argv", force effective_index to argv_slot.
        let name_at_index = frame.function.chunk.global_names.get(&index).map(|n| n.as_str());
        if frame.function.name == "<main>"
            && name_at_index == Some("argv")
            && unsafe { (*vm_ptr).get_argv_old_indices() }.map(|v| v.contains(&index)).unwrap_or(false)
        {
            if let Some(slot) = argv_slot_for_resolve {
                effective_index = slot;
                if effective_index >= globals.len() {
                    globals.resize(effective_index + 1, default_global_slot());
                }
            }
        }
        if effective_index >= globals.len() {
            let var_name = global_names.get(&effective_index)
                .or_else(|| frame.function.chunk.global_names.get(&index))
                .map(String::as_str);
            let error_message = match var_name {
                Some(name) if modules::is_known_module(name) && !loaded_modules.contains(name) => {
                    format!("Module {} not imported", name)
                }
                Some(name) => format!("Undefined variable: {}", name),
                None => "Undefined variable".to_string(),
            };
            // Use handle_exception so try/catch can catch undefined variable errors.
            // If no handler: returns Err (propagates). Never push Null.
            let error = ExceptionHandler::runtime_error(&frames, error_message, line);
            match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error, value_store, heavy_store) {
                Ok(()) => stack::push_id(stack, NULL_VALUE_ID),
                Err(e) => return Err(e),
            }
        } else {
            // When we're loading argv (by slot or by name in chunk or in VM global_names), use the canonical argv value id
            // or the slot at argv_slot_index if it holds an Array (script args), so we always get the script args.
            let argv_slot = unsafe { (*vm_ptr).get_argv_slot_index() };
            let name_in_chunk = frame.function.chunk.global_names.get(&index).map(|n| n.as_str());
            let name_in_vm = global_names.get(&effective_index).map(|n| n.as_str());
            let loading_argv = Some(effective_index) == argv_slot
                || name_in_chunk == Some("argv")
                || name_in_vm == Some("argv")
                || (frame.function.name == "<main>"
                    && name_in_chunk == Some("argv")
                    && unsafe { (*vm_ptr).get_argv_old_indices() }.map(|v| v.contains(&index)).unwrap_or(false));
            let mut id = if loading_argv {
                // In top-level main chunk, prefer canonical argv value id so script args are correct
                // even if the slot was overwritten after ImportFrom's update_chunk_indices_from_names.
                let in_main_chunk = frame.function.name == "<main>";
                let id_from_slot = argv_slot.and_then(|slot_idx| {
                    (slot_idx < globals.len()).then(|| globals[slot_idx].resolve_to_value_id(value_store))
                });
                if in_main_chunk {
                    unsafe { (*vm_ptr).get_current_argv_value_id() }
                        .or(id_from_slot)
                        .unwrap_or_else(|| globals[effective_index].resolve_to_value_id(value_store))
                } else {
                    unsafe { (*vm_ptr).get_current_argv_value_id() }
                        .or(id_from_slot)
                        .unwrap_or_else(|| globals[effective_index].resolve_to_value_id(value_store))
                }
            } else {
                let (inline_tv, heap_id_opt) = match &globals[effective_index] {
                    GlobalSlot::Inline(tv) => (Some(*tv), None),
                    GlobalSlot::Heap(slot_id) => (None, Some(*slot_id)),
                };
                if let Some(tv) = inline_tv {
                    stack::push(stack, tv);
                    return Ok(VMStatus::Continue);
                }
                heap_id_opt.unwrap()
            };
            // When loading argv in main chunk: use id restored after ImportFrom first, then VM/thread-local/slot.
            if loading_argv && frame.function.name == "<main>" {
                // Prefer SCRIPT_ARGV_VALUE_ID: set at run() start, survives nested module runs.
                let canonical_id = crate::vm::run_context::RunContext::get_script_argv_value_id()
                    .or_else(|| crate::vm::run_context::RunContext::get_restored_script_argv_after_import())
                    .or_else(|| unsafe { (*vm_ptr).get_current_argv_value_id() })
                    .or_else(|| crate::vm::run_context::RunContext::get_argv_value_id());
                if let Some(cid) = canonical_id {
                    id = cid;
                } else {
                    if let Some(slot_idx) = argv_slot {
                        if slot_idx < globals.len() {
                            let slot_id = globals[slot_idx].resolve_to_value_id(value_store);
                            let slot_val = load_value(slot_id, value_store, heavy_store);
                            if let Value::Array(a) = &slot_val {
                                if !a.borrow().is_empty() {
                                    id = slot_id;
                                }
                            }
                        }
                    }
                    let val = load_value(id, value_store, heavy_store);
                    let not_proper_argv = match &val {
                        Value::Array(a) => a.borrow().is_empty(),
                        _ => true,
                    };
                    if not_proper_argv {
                        if let Some(cid) = unsafe { (*vm_ptr).get_current_argv_value_id() }
                            .or_else(crate::vm::run_context::RunContext::get_argv_value_id)
                        {
                            id = cid;
                        }
                    }
                }
            } else if loading_argv {
                let cid = unsafe { (*vm_ptr).get_current_argv_value_id() }
                    .or_else(crate::vm::run_context::RunContext::get_argv_value_id);
                if let Some(cid) = cid {
                    id = cid;
                }
            }
            {
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

/// Execute StoreGlobal(index).
#[allow(clippy::too_many_arguments)]
pub fn op_store_global(
    index: usize,
    _line: usize,
    stack: &mut Vec<TaggedValue>,
    frames: &mut Vec<CallFrame>,
    globals: &mut Vec<GlobalSlot>,
    global_names: &mut std::collections::BTreeMap<usize, String>,
    exception_handlers: &mut Vec<ExceptionHandler>,
    value_store: &mut ValueStore,
    heavy_store: &mut HeavyStore,
    vm_ptr: *mut crate::vm::vm::Vm,
) -> Result<VMStatus, LangError> {
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
        // Store first so the same object (and Rc) is later modified by __col_* code; then register that stored value in metadata so create_all sees __col_*
        let final_id = store_value_arena(value, value_store, heavy_store);
        let stored_value = load_value(final_id, value_store, heavy_store);
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
                                        tables_rc.borrow_mut().push(stored_value.clone());
                                    }
                                    let mut meta = meta_rc.borrow_mut();
                                    if !meta.contains_key("classes") {
                                        meta.insert("classes".to_string(), Value::Object(Rc::new(RefCell::new(std::collections::HashMap::new()))));
                                    }
                                    let classes_rc_opt = meta.get("classes").cloned();
                                    drop(meta);
                                    if let Some(Value::Object(classes_rc)) = classes_rc_opt {
                                        let mut classes = classes_rc.borrow_mut();
                                        if let Some(ref name) = class_name {
                                            classes.insert(name.clone(), stored_value.clone());
                                        }
                                        let mut current_name = parent_name.clone();
                                        let mut seen_ancestors = std::collections::HashSet::new();
                                        loop {
                                            if !seen_ancestors.insert(current_name.clone()) {
                                                break;
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
        if let Some(name) = class_name {
            if let Some(Value::Object(meta_rc)) = class_meta_opt {
                let mut meta = meta_rc.borrow_mut();
                if !meta.contains_key("classes") {
                    meta.insert("classes".to_string(), Value::Object(Rc::new(RefCell::new(std::collections::HashMap::new()))));
                }
                let classes_rc_opt = meta.get("classes").cloned();
                drop(meta);
                if let Some(Value::Object(classes_rc)) = classes_rc_opt {
                    classes_rc.borrow_mut().insert(name, stored_value.clone());
                }
            }
        }
        if index >= globals.len() {
            globals.resize(index + 1, default_global_slot());
        }
        globals[index] = GlobalSlot::Heap(final_id);
        return Ok(VMStatus::Continue);
}
