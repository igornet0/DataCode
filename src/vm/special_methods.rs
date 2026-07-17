//! Runtime dispatch for class special methods (`@add`, `@string`, …).

use crate::common::error::LangError;
use crate::common::numeric::integer_value_as_i64_if_whole;
use crate::common::value::{ObjectKind, Value};
use crate::common::value_store::ValueId;
use crate::vm::memory::object_map_lookup_value;
use crate::vm::natives::utils::{call_user_function, call_user_function_with_arg_ids};
use crate::vm::store_convert::{load_value, store_value};
use crate::vm::vm::current_vm_ptr;
use std::cell::RefCell;

thread_local! {
    static STRING_DISPATCH_GUARD: RefCell<bool> = RefCell::new(false);
}

fn lookup_str_field(obj: &Value, key: &str) -> Option<Value> {
    let Value::Object(rc) = obj else {
        return None;
    };
    {
        let kind = rc.borrow();
        if let Some(v) = kind.str_key_get(key) {
            return Some(v.clone());
        }
        if let ObjectKind::Bucket(map) = &*kind {
            let vm_ptr = current_vm_ptr()?;
            return unsafe {
                (*vm_ptr).with_stores_mut(|store, heap| {
                    let key_id = store_value(Value::String(key.to_string()), store, heap);
                    object_map_lookup_value(map, key_id, store, heap)
                })
            };
        }
    }
    None
}

pub fn is_class_instance(v: &Value) -> bool {
    lookup_str_field(v, "__class_name").is_some() && lookup_str_field(v, "__class").is_some()
}

fn special_fn_global_name(instance: &Value, method_name: &str) -> Option<String> {
    let Value::String(class_name) = lookup_str_field(instance, "__class_name")? else {
        return None;
    };
    let suffix = method_name.strip_prefix('@')?;
    Some(format!("{class_name}::special_{suffix}"))
}

fn resolve_special_fn_index(instance: &Value, method_name: &str) -> Option<usize> {
    let fn_name = special_fn_global_name(instance, method_name)?;
    let vm_ptr = current_vm_ptr()?;
    unsafe {
        let vm = &*vm_ptr;
        vm.get_functions()
            .iter()
            .position(|f| f.name == fn_name)
    }
}

fn resolve_special_fn_index_by_id(receiver_id: ValueId, method_name: &str) -> Option<usize> {
    let vm_ptr = current_vm_ptr()?;
    unsafe {
        let vm = &*vm_ptr;
        let instance = load_value(receiver_id, vm.value_store(), vm.heavy_store());
        resolve_special_fn_index(&instance, method_name)
    }
}

pub fn resolve_special_fn(instance: &Value, method_name: &str) -> Option<Value> {
    let idx = resolve_special_fn_index(instance, method_name)?;
    Some(Value::Function(idx))
}

pub fn class_has_special(instance: &Value, method_name: &str) -> bool {
    let Some(class) = lookup_str_field(instance, "__class") else {
        return false;
    };
    let Some(Value::Array(names)) = lookup_str_field(&class, "__special_method_names") else {
        return false;
    };
    let names_ref = names.borrow();
    names_ref.iter().any(|v| {
        matches!(v, Value::String(s) if s.as_str() == method_name)
    })
}

pub fn class_instance_supports_hash(instance: &Value) -> bool {
    class_has_special(instance, "@hash")
}

pub fn dispatch_special(
    receiver: &Value,
    method_name: &str,
    args: &[Value],
) -> Result<Option<Value>, LangError> {
    let Some(fn_idx) = resolve_special_fn_index(receiver, method_name) else {
        return Ok(None);
    };
    let mut call_args = Vec::with_capacity(1 + args.len());
    call_args.push(receiver.clone());
    call_args.extend_from_slice(args);
    Ok(Some(call_user_function(fn_idx, &call_args)?))
}

/// Dispatch a special method using canonical [`ValueId`] for `this` (mutations persist on the instance).
pub fn dispatch_special_by_id(
    receiver_id: ValueId,
    method_name: &str,
    extra_arg_ids: &[ValueId],
) -> Result<Option<Value>, LangError> {
    let Some(fn_idx) = resolve_special_fn_index_by_id(receiver_id, method_name) else {
        return Ok(None);
    };
    let mut arg_ids = Vec::with_capacity(1 + extra_arg_ids.len());
    arg_ids.push(receiver_id);
    arg_ids.extend_from_slice(extra_arg_ids);
    Ok(Some(call_user_function_with_arg_ids(fn_idx, &arg_ids)?))
}

pub fn try_instance_hash(instance: &Value) -> Option<u64> {
    dispatch_special(instance, "@hash", &[])
        .ok()
        .flatten()
        .and_then(|v| integer_value_as_i64_if_whole(&v).map(|n| n as u64))
}

pub fn try_instance_string(instance: &Value) -> Option<String> {
    STRING_DISPATCH_GUARD.with(|guard| {
        if *guard.borrow() {
            return None;
        }
        *guard.borrow_mut() = true;
        let out = dispatch_special(instance, "@string", &[])
            .ok()
            .flatten()
            .map(|v| v.to_string());
        *guard.borrow_mut() = false;
        out
    })
}

pub fn try_resolve_callable_instance(instance: &Value) -> Option<Value> {
    resolve_special_fn(instance, "@call")
}

pub fn try_dispatch_binary_arithmetic(
    receiver: &Value,
    method_name: &str,
    other: &Value,
) -> Result<Option<Value>, LangError> {
    dispatch_special(receiver, method_name, &[other.clone()])
}

pub fn try_dispatch_binary_compare(
    receiver: &Value,
    method_name: &str,
    other: &Value,
) -> Result<Option<Value>, LangError> {
    dispatch_special(receiver, method_name, &[other.clone()])
}

pub fn try_dispatch_unary_special(
    receiver: &Value,
    method_name: &str,
) -> Result<Option<Value>, LangError> {
    dispatch_special(receiver, method_name, &[])
}

