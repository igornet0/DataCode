//! GetArrayElement and SetArrayElement for Value::Object (class/instance fields, private/protected).

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use crate::common::{
    error::ErrorType,
    error::LangError,
    value::{ObjectKind, Value},
    value_store::{ValueCell, ValueStore, NULL_VALUE_ID},
};
use crate::debug_println;
use crate::vm::exceptions::ExceptionHandler;
use crate::vm::frame::CallFrame;
use crate::vm::global_utils::{get_superclass_chain, global_index_by_name};
use crate::vm::heavy_store::HeavyStore;
use crate::vm::stack;
use crate::vm::store_convert::{
    load_value, object_cell_try_lookup_by_key_id, object_map_upsert, promote_value_id_if_escapes,
    store_value, tagged_to_value_id,
};
use crate::vm::special_methods::{
    class_has_special, dispatch_special, dispatch_special_by_id, is_class_instance,
};
use crate::vm::types::VMStatus;

/// When a field is missing on the instance, class fallback may find a `Field(...)` descriptor object.
fn field_descriptor_default(method_val: &Value) -> Option<Value> {
    let Value::Object(rc) = method_val else {
        return None;
    };
    let map = rc.borrow();
    let legacy = map.legacy_ref()?;
    if legacy.contains_key("default_factory") {
        return None;
    }
    legacy.get("default").cloned()
}

fn lookup_field_on_class_hierarchy(
    class_name: &str,
    key: &str,
    globals: &mut [crate::vm::global_slot::GlobalSlot],
    global_names: &std::collections::BTreeMap<usize, String>,
    value_store: &mut ValueStore,
    heavy_store: &mut HeavyStore,
    modules: &std::collections::HashMap<
        String,
        Rc<RefCell<crate::vm::module_object::ModuleObject>>,
    >,
) -> Option<Value> {
    let chain = get_superclass_chain(
        globals,
        global_names,
        class_name,
        value_store,
        heavy_store,
    );
    for cn in chain {
        if let Some(idx) = global_index_by_name(global_names, &cn) {
            if idx < globals.len() {
                let id = globals[idx].resolve_to_value_id(value_store);
                let val = load_value(id, value_store, heavy_store);
                if let Value::Object(class_rc) = val {
                    if let Some(v) = class_rc.borrow().str_key_get(key) {
                        return Some(v.clone());
                    }
                }
            }
        }
        for (_mod_name, rc) in modules.iter() {
            if let Some(class_val) = rc.borrow().get_export(&cn) {
                if let Value::Object(class_rc) = &class_val {
                    if let Some(v) = class_rc.borrow().str_key_get(key) {
                        return Some(v.clone());
                    }
                }
            }
        }
    }
    None
}

/// Get field from Object. Handles metadata, private/protected, class method fallback.
#[allow(clippy::too_many_arguments)]
pub(crate) fn get_object(
    line: usize,
    stack: &mut Vec<crate::common::TaggedValue>,
    frames: &mut Vec<CallFrame>,
    globals: &mut Vec<crate::vm::global_slot::GlobalSlot>,
    global_names: &std::collections::BTreeMap<usize, String>,
    exception_handlers: &mut Vec<ExceptionHandler>,
    value_store: &mut ValueStore,
    heavy_store: &mut HeavyStore,
    vm_ptr: *mut crate::vm::vm::Vm,
    container_tv: crate::common::TaggedValue,
    map_rc: Rc<RefCell<ObjectKind>>,
    index_value: Value,
) -> Result<VMStatus, LangError> {
    let map = map_rc.borrow();

    // Regular object access
    match &index_value {
        Value::String(key) => {
            // Class.metadata must return the metadata value
            if key == "metadata" {
                if let Some(meta_val) = map.str_key_get("metadata") {
                    stack::push_id(
                        stack,
                        store_value(meta_val.clone(), value_store, heavy_store),
                    );
                    return Ok(VMStatus::Continue);
                }
                if map.str_key_contains("__class_name") {
                    let class_name_opt = map.str_key_get("__class_name").and_then(|v| {
                        if let Value::String(s) = v {
                            Some(s.clone())
                        } else {
                            None
                        }
                    });
                    if let Some(cn) = class_name_opt {
                        let chain = get_superclass_chain(
                            globals,
                            global_names,
                            &cn,
                            value_store,
                            heavy_store,
                        );
                        for c in &chain {
                            let class_val_opt = global_index_by_name(global_names, c)
                                .filter(|&idx| idx < globals.len())
                                .map(|idx| {
                                    load_value(
                                        globals[idx].resolve_to_value_id(value_store),
                                        value_store,
                                        heavy_store,
                                    )
                                });
                            let class_val_opt = class_val_opt.or_else(|| {
                                let modules = unsafe { (*vm_ptr).get_modules() };
                                for (_mod_name, rc) in modules.iter() {
                                    if let Some(class_val) = rc.borrow().get_export(c) {
                                        return Some(class_val);
                                    }
                                }
                                None
                            });
                            if let Some(Value::Object(class_rc)) = class_val_opt {
                                if let Some(meta) = class_rc.borrow().str_key_get("metadata") {
                                    stack::push_id(
                                        stack,
                                        store_value(meta.clone(), value_store, heavy_store),
                                    );
                                    return Ok(VMStatus::Continue);
                                }
                            }
                        }
                    }
                }
            }
            // Class private variables: forbid read from outside
            if map.str_key_contains("__class_name") && key != "model_config" {
                if let Some(Value::Array(private_vars_rc)) = map.str_key_get("__class_private_vars") {
                    let private_vars = private_vars_rc.borrow();
                    let is_private = private_vars.iter().any(|v| {
                        if let Value::String(s) = v {
                            s.as_str() == key
                        } else {
                            false
                        }
                    });
                    if is_private {
                        let class_name = map
                            .str_key_get("__class_name")
                            .and_then(|v| {
                                if let Value::String(s) = v {
                                    Some(s.clone())
                                } else {
                                    None
                                }
                            })
                            .unwrap_or_else(|| "?".to_string());
                        let msg = format!("Class variable '{}' is private in '{}' and cannot be accessed from outside the class", key, class_name);
                        let error = ExceptionHandler::runtime_error_with_type(
                            &frames,
                            msg,
                            line,
                            ErrorType::ProtectError,
                        );
                        return match ExceptionHandler::handle_exception(
                            stack,
                            frames,
                            exception_handlers,
                            error,
                            value_store,
                            heavy_store,
                        ) {
                            Ok(()) => Ok(VMStatus::Continue),
                            Err(e) => Err(e),
                        };
                    }
                }
            }
            // Class protected variables: allow only from this class or subclasses
            if let Some(Value::Array(protected_vars_rc)) = map.str_key_get("__class_protected_vars") {
                let protected_vars = protected_vars_rc.borrow();
                let is_protected = protected_vars.iter().any(|v| {
                    if let Value::String(s) = v {
                        s.as_str() == key
                    } else {
                        false
                    }
                });
                if is_protected {
                    let class_name_obj = map.str_key_get("__class_name");
                    let in_hierarchy =
                        if let Some(Value::String(ref obj_class_name)) = class_name_obj {
                            frames.iter().any(|f| {
                                let frame_class = f.function.name.split("::").next().unwrap_or("");
                                let frame_chain = get_superclass_chain(
                                    globals,
                                    global_names,
                                    frame_class,
                                    value_store,
                                    heavy_store,
                                );
                                frame_chain.iter().any(|c| c == obj_class_name)
                            })
                        } else {
                            false
                        };
                    if !in_hierarchy {
                        let obj_class_name = map
                            .str_key_get("__class_name")
                            .and_then(|v| {
                                if let Value::String(s) = v {
                                    Some(s.clone())
                                } else {
                                    None
                                }
                            })
                            .unwrap_or_else(|| "?".to_string());
                        let frame_class_opt = frames.iter().rev().find_map(|f| {
                            if f.function.name.contains("::new_")
                                || f.function.name.contains("::method_")
                            {
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
                        return match ExceptionHandler::handle_exception(
                            stack,
                            frames,
                            exception_handlers,
                            error,
                            value_store,
                            heavy_store,
                        ) {
                            Ok(()) => Ok(VMStatus::Continue),
                            Err(e) => Err(e),
                        };
                    }
                }
            }
            // Instance private fields: allow only from the defining class
            if let (Some(Value::String(ref class_name)), Some(Value::Array(private_fields_rc))) =
                (map.str_key_get("__class_name"), map.str_key_get("__private_fields"))
            {
                let private_fields_slice = private_fields_rc.borrow();
                let is_private_field = private_fields_slice.iter().any(|v| {
                    if let Value::String(s) = v {
                        s.as_str() == key
                    } else {
                        false
                    }
                });
                if is_private_field {
                    let defining_class = map
                        .str_key_get("__private_field_defining_class")
                        .and_then(|v| {
                            if let Value::Object(rc) = v {
                                rc.borrow().str_key_get(key.as_str()).and_then(|v| {
                                    if let Value::String(s) = v {
                                        Some(s.clone())
                                    } else {
                                        None
                                    }
                                })
                            } else {
                                None
                            }
                        })
                        .unwrap_or_else(|| class_name.clone());
                    let in_defining_class = frames.iter().any(|f| {
                        f.function
                            .name
                            .starts_with(&format!("{}::", defining_class))
                    });
                    if !in_defining_class {
                        let frame_class_opt = frames.iter().rev().find_map(|f| {
                            if f.function.name.contains("::new_")
                                || f.function.name.contains("::method_")
                            {
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
                        return match ExceptionHandler::handle_exception(
                            stack,
                            frames,
                            exception_handlers,
                            error,
                            value_store,
                            heavy_store,
                        ) {
                            Ok(()) => Ok(VMStatus::Continue),
                            Err(e) => Err(e),
                        };
                    }
                }
            }
            // Instance protected fields: allow from this class or any subclass
            if let (
                Some(Value::String(ref instance_class)),
                Some(Value::Array(protected_fields_rc)),
            ) = (map.str_key_get("__class_name"), map.str_key_get("__protected_fields"))
            {
                let protected_fields_slice = protected_fields_rc.borrow();
                let is_protected_field = protected_fields_slice.iter().any(|v| {
                    if let Value::String(s) = v {
                        s.as_str() == key
                    } else {
                        false
                    }
                });
                if is_protected_field {
                    let instance_chain = get_superclass_chain(
                        globals,
                        global_names,
                        instance_class,
                        value_store,
                        heavy_store,
                    );
                    let in_hierarchy = frames.iter().any(|f| {
                        let frame_class = f.function.name.split("::").next().unwrap_or("");
                        instance_chain.iter().any(|c| c == frame_class)
                    });
                    if !in_hierarchy {
                        let frame_class_opt = frames.iter().rev().find_map(|f| {
                            if f.function.name.contains("::new_")
                                || f.function.name.contains("::method_")
                            {
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
                        return match ExceptionHandler::handle_exception(
                            stack,
                            frames,
                            exception_handlers,
                            error,
                            value_store,
                            heavy_store,
                        ) {
                            Ok(()) => Ok(VMStatus::Continue),
                            Err(e) => Err(e),
                        };
                    }
                }
            }
            // Instance private methods: allow only from the defining class
            if let (Some(Value::String(ref class_name)), Some(Value::Array(private_methods_rc))) =
                (map.str_key_get("__class_name"), map.str_key_get("__private_methods"))
            {
                let private_methods_slice = private_methods_rc.borrow();
                let is_private_method = private_methods_slice.iter().any(|v| {
                    if let Value::String(s) = v {
                        s.as_str() == key
                    } else {
                        false
                    }
                });
                if is_private_method {
                    let defining_class = map
                        .str_key_get("__private_method_defining_class")
                        .and_then(|v| {
                            if let Value::Object(rc) = v {
                                rc.borrow().str_key_get(key.as_str()).and_then(|v| {
                                    if let Value::String(s) = v {
                                        Some(s.clone())
                                    } else {
                                        None
                                    }
                                })
                            } else {
                                None
                            }
                        })
                        .unwrap_or_else(|| class_name.clone());
                    let in_defining_class = frames.iter().any(|f| {
                        f.function
                            .name
                            .starts_with(&format!("{}::", defining_class))
                    });
                    if !in_defining_class {
                        let frame_class_opt = frames.iter().rev().find_map(|f| {
                            if f.function.name.contains("::new_")
                                || f.function.name.contains("::method_")
                            {
                                f.function.name.split("::").next().map(String::from)
                            } else {
                                None
                            }
                        });
                        let msg = match &frame_class_opt {
                            Some(class) => format!("Method '{}' is private in '{}' and cannot be accessed from subclass '{}'", key, defining_class, class),
                            None => format!("Method '{}' is private in '{}' and cannot be accessed from outside the class", key, defining_class),
                        };
                        let error = ExceptionHandler::runtime_error_with_type(
                            &frames,
                            msg,
                            line,
                            ErrorType::ProtectError,
                        );
                        return match ExceptionHandler::handle_exception(
                            stack,
                            frames,
                            exception_handlers,
                            error,
                            value_store,
                            heavy_store,
                        ) {
                            Ok(()) => Ok(VMStatus::Continue),
                            Err(e) => Err(e),
                        };
                    }
                }
            }
            // Instance protected methods: allow from this class or any subclass
            if let (
                Some(Value::String(ref instance_class)),
                Some(Value::Array(protected_methods_rc)),
            ) = (map.str_key_get("__class_name"), map.str_key_get("__protected_methods"))
            {
                let protected_methods_slice = protected_methods_rc.borrow();
                let is_protected_method = protected_methods_slice.iter().any(|v| {
                    if let Value::String(s) = v {
                        s.as_str() == key
                    } else {
                        false
                    }
                });
                if is_protected_method {
                    let instance_chain = get_superclass_chain(
                        globals,
                        global_names,
                        instance_class,
                        value_store,
                        heavy_store,
                    );
                    let in_hierarchy = frames.iter().any(|f| {
                        let frame_class = f.function.name.split("::").next().unwrap_or("");
                        instance_chain.iter().any(|c| c == frame_class)
                    });
                    if !in_hierarchy {
                        let msg = format!("Method '{}' is protected in '{}' and cannot be accessed outside class or subclass", key, instance_class);
                        let error = ExceptionHandler::runtime_error_with_type(
                            &frames,
                            msg,
                            line,
                            ErrorType::ProtectError,
                        );
                        return match ExceptionHandler::handle_exception(
                            stack,
                            frames,
                            exception_handlers,
                            error,
                            value_store,
                            heavy_store,
                        ) {
                            Ok(()) => Ok(VMStatus::Continue),
                            Err(e) => Err(e),
                        };
                    }
                }
            }
            let direct = map.str_key_get(key.as_str());
            let try_class_fallback = direct.map_or(true, |v| matches!(v, Value::Null));
            if try_class_fallback && map.str_key_contains("__class") && !in_user_constructor(frames)
                && !in_special_method(frames, "get")
            {
                let instance = Value::Object(map_rc.clone());
                if class_has_special(&instance, "@get") {
                    if let Ok(Some(result)) =
                        dispatch_special(&instance, "@get", &[index_value.clone()])
                    {
                        stack::push_id(
                            stack,
                            store_value(result, value_store, heavy_store),
                        );
                        return Ok(VMStatus::Continue);
                    }
                }
            }
            if let Some(value) = direct {
                if !try_class_fallback {
                    // Field reads must return the canonical ValueId stored in this object's bucket
                    // map. Re-storing via `store_value(...)` allocates a NEW id; mutations through
                    // that ephemeral object (e.g. `this.data[key] = v`) never update the instance
                    // field that still points at the original id (`private data: object`).
                    let container_id = tagged_to_value_id(container_tv, value_store);
                    let key_mat = Value::String(key.clone());
                    let kid = store_value(key_mat, value_store, heavy_store);
                    let heap_field_id =
                        object_cell_try_lookup_by_key_id(container_id, kid, value_store, heavy_store);
                    if let Some(vid) = heap_field_id {
                        stack::push_id(stack, vid);
                    } else {
                        stack::push_id(stack, store_value(value.clone(), value_store, heavy_store));
                    }
                    return Ok(VMStatus::Continue);
                }
            }
            if try_class_fallback {
                let class_name_opt = map.str_key_get("__class_name").and_then(|cn| {
                    if let Value::String(s) = cn {
                        Some(s.clone())
                    } else {
                        None
                    }
                });
                let from_class = class_name_opt.as_ref().and_then(|class_name| {
                    let modules = unsafe { (*vm_ptr).get_modules() };
                    lookup_field_on_class_hierarchy(
                        class_name,
                        key.as_str(),
                        globals,
                        global_names,
                        value_store,
                        heavy_store,
                        &modules,
                    )
                });
                if from_class.is_none() && class_name_opt.is_some() {
                    let class_name = class_name_opt.as_deref().unwrap();
                    let in_globals = global_index_by_name(global_names, class_name).is_some();
                    let modules = unsafe { (*vm_ptr).get_modules() };
                    let mod_keys: Vec<_> = modules.keys().cloned().collect();
                    let mut export_status = Vec::new();
                    for (mod_name, rc) in modules.iter() {
                        let class_val = rc.borrow().get_export(class_name);
                        let has_class = class_val.is_some();
                        let has_key = class_val
                            .as_ref()
                            .and_then(|v| {
                                if let Value::Object(or) = v {
                                    or.borrow().str_key_get(key.as_str()).map(|_| ())
                                } else {
                                    None
                                }
                            })
                            .is_some();
                        export_status.push((mod_name.clone(), has_class, has_key));
                    }
                    debug_println!(
                        "[DEBUG GetArrayElement fallback] class_name={} in_globals={} modules={:?} get_export_and_key={:?}",
                        class_name, in_globals, mod_keys, export_status
                    );
                }
                if let Some(method_val) = from_class {
                    if matches!(
                        &method_val,
                        Value::Function(_)
                            | Value::ModuleFunction { .. }
                            | Value::NativeFunction(_)
                    ) {
                        let container_id = tagged_to_value_id(container_tv, value_store);
                        stack::push_id(stack, container_id);
                        stack::push_id(stack, store_value(method_val, value_store, heavy_store));
                    } else if key == "metadata" {
                        stack::push_id(stack, store_value(method_val, value_store, heavy_store));
                    } else if let Some(default_val) = field_descriptor_default(&method_val) {
                        stack::push_id(
                            stack,
                            store_value(default_val, value_store, heavy_store),
                        );
                    } else {
                        debug_println!(
                            "[DEBUG GetArrayElement] key '{}' not found, pushing Null",
                            key
                        );
                        stack::push_id(stack, NULL_VALUE_ID);
                    }
                } else {
                    debug_println!(
                        "[DEBUG GetArrayElement] key '{}' not found, pushing Null",
                        key
                    );
                    stack::push_id(stack, NULL_VALUE_ID);
                }
            }
        }
        _ => {
            if map.str_key_contains("__class") {
                let instance = load_value(
                    tagged_to_value_id(container_tv, value_store),
                    value_store,
                    heavy_store,
                );
                if is_class_instance(&instance)
                    && class_has_special(&instance, "@get")
                    && !in_special_method(frames, "get")
                {
                    if let Ok(Some(result)) =
                        dispatch_special(&instance, "@get", &[index_value.clone()])
                    {
                        stack::push_id(
                            stack,
                            store_value(result, value_store, heavy_store),
                        );
                        return Ok(VMStatus::Continue);
                    }
                }
            }
            let error = ExceptionHandler::runtime_error(
                &frames,
                "Object index must be a string".to_string(),
                line,
            );
            return match ExceptionHandler::handle_exception(
                stack,
                frames,
                exception_handlers,
                error,
                value_store,
                heavy_store,
            ) {
                Ok(()) => Ok(VMStatus::Continue),
                Err(e) => Err(e),
            };
        }
    }
    Ok(VMStatus::Continue)
}

pub(crate) fn in_user_constructor(frames: &[CallFrame]) -> bool {
    frames
        .last()
        .map(|f| f.function.name.contains("::new_"))
        .unwrap_or(false)
}

pub(crate) fn in_special_method(frames: &[CallFrame], suffix: &str) -> bool {
    let marker = format!("::special_{suffix}");
    frames.iter().any(|f| f.function.name.ends_with(&marker))
}

/// Set field on Object. Handles private/protected, metadata registration.
#[allow(clippy::too_many_arguments)]
pub(crate) fn set_object(
    line: usize,
    stack: &mut Vec<crate::common::TaggedValue>,
    frames: &mut Vec<CallFrame>,
    globals: &mut Vec<crate::vm::global_slot::GlobalSlot>,
    global_names: &std::collections::BTreeMap<usize, String>,
    exception_handlers: &mut Vec<ExceptionHandler>,
    value_store: &mut ValueStore,
    heavy_store: &mut HeavyStore,
    container_id: crate::common::value_store::ValueId,
    obj_rc: Rc<RefCell<ObjectKind>>,
    key: String,
    value_id: crate::common::value_store::ValueId,
    value: Value,
) -> Result<VMStatus, LangError> {
    let instance = load_value(container_id, value_store, heavy_store);
    if is_class_instance(&instance)
        && class_has_special(&instance, "@set")
        && !in_user_constructor(frames)
        && !in_special_method(frames, "set")
    {
        let key_id = store_value(Value::String(key.clone()), value_store, heavy_store);
        if let Ok(Some(_)) =
            dispatch_special_by_id(container_id, "@set", &[key_id, value_id])
        {
            stack::push_id(stack, container_id);
            return Ok(VMStatus::Continue);
        }
    }
    // Class private variables: forbid write from outside
    let is_class_private_var = {
        let map = obj_rc.borrow();
        map.str_key_contains("__class_name")
            && key != "model_config"
            && map
                .str_key_get("__class_private_vars")
                .and_then(|v| {
                    if let Value::Array(rc) = v {
                        Some(rc.borrow().iter().any(|v| {
                            if let Value::String(s) = v {
                                s.as_str() == key
                            } else {
                                false
                            }
                        }))
                    } else {
                        None
                    }
                })
                .unwrap_or(false)
    };
    let allow_main_init =
        frames.len() == 1 && frames.first().map(|f| f.function.name.as_str()) == Some("<main>");
    if is_class_private_var && !allow_main_init {
        let class_name = obj_rc.borrow().str_key_get("__class_name")
            .and_then(|v| {
                if let Value::String(s) = v {
                    Some(s.clone())
                } else {
                    None
                }
            })
            .unwrap_or_else(|| "?".to_string());
        let msg = format!(
            "Class variable '{}' is private in '{}' and cannot be accessed from outside the class",
            key, class_name
        );
        let error =
            ExceptionHandler::runtime_error_with_type(&frames, msg, line, ErrorType::ProtectError);
        return match ExceptionHandler::handle_exception(
            stack,
            frames,
            exception_handlers,
            error,
            value_store,
            heavy_store,
        ) {
            Ok(()) => Ok(VMStatus::Continue),
            Err(e) => Err(e),
        };
    }
    // Instance private fields: allow write only from the defining class
    let (is_instance_private_denied, private_error_msg) = {
        let map = obj_rc.borrow();
        if let (Some(Value::String(ref class_name)), Some(Value::Array(private_fields_rc))) =
            (map.str_key_get("__class_name"), map.str_key_get("__private_fields"))
        {
            let is_private = private_fields_rc.borrow().iter().any(|v| {
                if let Value::String(s) = v {
                    s.as_str() == key
                } else {
                    false
                }
            });
            if !is_private {
                (false, String::new())
            } else {
                let defining_class = map
                    .str_key_get("__private_field_defining_class")
                    .and_then(|v| {
                        if let Value::Object(rc) = v {
                            rc.borrow().str_key_get(key.as_str()).and_then(|v| {
                                if let Value::String(s) = v {
                                    Some(s.clone())
                                } else {
                                    None
                                }
                            })
                        } else {
                            None
                        }
                    })
                    .unwrap_or_else(|| class_name.clone());
                let in_defining_class = frames.iter().any(|f| {
                    f.function
                        .name
                        .starts_with(&format!("{}::", defining_class))
                });
                if in_defining_class {
                    (false, String::new())
                } else {
                    let frame_class_opt = frames.iter().rev().find_map(|f| {
                        if f.function.name.contains("::new_")
                            || f.function.name.contains("::method_")
                        {
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
        return match ExceptionHandler::handle_exception(
            stack,
            frames,
            exception_handlers,
            error,
            value_store,
            heavy_store,
        ) {
            Ok(()) => Ok(VMStatus::Continue),
            Err(e) => Err(e),
        };
    }
    // Class protected variables: allow write only from this class or subclasses
    let (is_class_protected_denied, class_protected_msg) = {
        let map = obj_rc.borrow();
        let is_protected_var = map.str_key_contains("__class_name")
            && key != "model_config"
            && map
                .str_key_get("__class_protected_vars")
                .and_then(|v| {
                    if let Value::Array(rc) = v {
                        Some(rc.borrow().iter().any(|v| {
                            if let Value::String(s) = v {
                                s.as_str() == key
                            } else {
                                false
                            }
                        }))
                    } else {
                        None
                    }
                })
                .unwrap_or(false);
        if !is_protected_var {
            (false, String::new())
        } else {
            let obj_class = map.str_key_get("__class_name").and_then(|v| {
                if let Value::String(s) = v {
                    Some(s.clone())
                } else {
                    None
                }
            });
            let in_hierarchy = obj_class
                .as_ref()
                .map(|obj_class| {
                    frames.iter().any(|f| {
                        let frame_class = f.function.name.split("::").next().unwrap_or("");
                        let frame_chain = get_superclass_chain(
                            globals,
                            global_names,
                            frame_class,
                            value_store,
                            heavy_store,
                        );
                        frame_chain.iter().any(|c| c == obj_class)
                    })
                })
                .unwrap_or(false);
            if in_hierarchy {
                (false, String::new())
            } else {
                let frame_class_opt = frames.iter().rev().find_map(|f| {
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
        return match ExceptionHandler::handle_exception(
            stack,
            frames,
            exception_handlers,
            error,
            value_store,
            heavy_store,
        ) {
            Ok(()) => Ok(VMStatus::Continue),
            Err(e) => Err(e),
        };
    }
    // Instance protected fields: allow write only from this class or subclasses
    let (is_instance_protected_denied, instance_protected_msg) = {
        let map = obj_rc.borrow();
        if let (Some(Value::String(ref instance_class)), Some(Value::Array(protected_fields_rc))) =
            (map.str_key_get("__class_name"), map.str_key_get("__protected_fields"))
        {
            let is_protected = protected_fields_rc.borrow().iter().any(|v| {
                if let Value::String(s) = v {
                    s.as_str() == key
                } else {
                    false
                }
            });
            let in_hierarchy = {
                let instance_chain = get_superclass_chain(
                    globals,
                    global_names,
                    instance_class,
                    value_store,
                    heavy_store,
                );
                frames.iter().any(|f| {
                    let frame_class = f.function.name.split("::").next().unwrap_or("");
                    instance_chain.iter().any(|c| c == frame_class)
                })
            };
            if is_protected && !in_hierarchy {
                let frame_class_opt = frames.iter().rev().find_map(|f| {
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
        return match ExceptionHandler::handle_exception(
            stack,
            frames,
            exception_handlers,
            error,
            value_store,
            heavy_store,
        ) {
            Ok(()) => Ok(VMStatus::Continue),
            Err(e) => Err(e),
        };
    }
    // Update the object (clone ObjectMap briefly so [`object_map_upsert`] can read [`ValueStore`].)
    let key_material = Value::String(key.clone());
    let kid = store_value(key_material.clone(), value_store, heavy_store);
    let value_id = promote_value_id_if_escapes(container_id, value_id, value_store);
    if let Some(ValueCell::Object(mut omap)) = value_store.get(container_id).cloned() {
        object_map_upsert(&mut omap, value_store, heavy_store, &key_material, kid, value_id);
        debug_println!(
            "[DEBUG SetArrayElement] object key '{}' set, keys now: {}",
            key,
            omap.len()
        );
        if let Some(cell) = value_store.get_mut(container_id) {
            *cell = ValueCell::Object(omap);
        }
    } else {
        let v = load_value(value_id, value_store, heavy_store);
        let _ = obj_rc.borrow_mut().str_key_insert(key.clone(), v);
    }
    // When a class sets metadata = MetaData(), register the class in metadata.tables and metadata.classes
    if key == "metadata" {
        let is_meta = if let Value::Object(meta_rc) = &value {
            meta_rc
                .borrow()
                .str_key_get("__meta")
                .and_then(|v| {
                    if let Value::Bool(b) = v {
                        Some(*b)
                    } else {
                        None
                    }
                })
                .unwrap_or(false)
        } else {
            false
        };
        let class_name_opt = obj_rc.borrow().str_key_get("__class_name").and_then(|v| {
            if let Value::String(s) = v {
                Some(s.clone())
            } else {
                None
            }
        });
        if is_meta && class_name_opt.is_some() {
            if let Value::Object(meta_rc) = &value {
                if let Some(Value::Array(tables_rc)) = meta_rc.borrow().str_key_get("tables") {
                    tables_rc
                        .borrow_mut()
                        .push(Value::Object(Rc::clone(&obj_rc)));
                }
                let mut meta = meta_rc.borrow_mut();
                if !meta.str_key_contains("classes") {
                    meta.str_key_insert(
                        "classes".to_string(),
                        Value::legacy_object(HashMap::new()),
                    );
                }
                if let (Some(name), Some(Value::Object(classes_rc))) =
                    (class_name_opt, meta.str_key_get("classes"))
                {
                    if let Some(m) = classes_rc.borrow_mut().legacy_mut() {
                        m.insert(name, Value::Object(Rc::clone(&obj_rc)));
                    }
                }
            }
        }
    }
    stack::push_id(stack, container_id);
    Ok(VMStatus::Continue)
}
