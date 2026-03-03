//! Constructor call setup: set __constructing_class__ before running a constructor.

use crate::common::value::Value;
use crate::common::value_store::ValueStore;
use crate::vm::frame::CallFrame;
use crate::vm::global_slot::{GlobalSlot, default_global_slot};
use crate::vm::store_convert::{store_value, load_value};
use crate::vm::heavy_store::HeavyStore;
use crate::vm::executor::{global_index_by_name, global_indices_by_name};

/// Set __constructing_class__ in globals for constructor calls (function name contains "::new_").
/// So Settings subclasses load model_config from the leaf class (e.g. DevSettings).
pub fn set_constructing_class_for_call(
    function: &crate::bytecode::Function,
    constructing_class_opt: Option<&Value>,
    frames: &[CallFrame],
    globals: &mut Vec<GlobalSlot>,
    global_names: &mut std::collections::BTreeMap<usize, String>,
    value_store: &mut ValueStore,
    heavy_store: &mut HeavyStore,
    vm_ptr: *mut crate::vm::vm::Vm,
) {
    if !function.name.contains("::new_") {
        return;
    }
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
    if skip_set_super_chain {
        return;
    }
    let class_to_set: Option<Value> = if let Some(class_val) = constructing_class_opt {
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
    let class_to_set = class_to_set.or_else(|| {
        let caller_indices: Vec<usize> = frames.last()
            .and_then(|f| {
                let mut idx: Vec<usize> = f.function.chunk.global_names
                    .iter()
                    .filter(|(_, n)| n.as_str() == CONSTRUCTING_CLASS_NAME)
                    .map(|(idx, _)| *idx)
                    .collect();
                idx.sort_unstable();
                if idx.is_empty() { None } else { Some(idx) }
            })
            .unwrap_or_else(|| global_indices_by_name(global_names, CONSTRUCTING_CLASS_NAME));
        caller_indices.into_iter().find_map(|idx| {
            if idx >= globals.len() {
                return None;
            }
            let id = globals[idx].resolve_to_value_id(value_store);
            let v = load_value(id, value_store, heavy_store);
            if let Value::Object(rc) = &v {
                let o = rc.borrow();
                if o.contains_key("new_0") || o.contains_key("new_1") || o.get("__class_name").is_some() {
                    return Some(v.clone());
                }
            }
            None
        })
    });
    if let Some(class_val) = class_to_set {
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
