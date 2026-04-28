// Utility functions for native functions

use crate::common::error::LangError;
use crate::common::value::Value;
use crate::vm::global_utils::global_index_by_name;
use crate::vm::store_convert::load_value;
use crate::vm::vm::{current_vm_ptr, Vm};

/// Resolve a global by name from VM; returns Some(Value) if found and it's an Object with __class_name.
/// Used by database_engine to walk class hierarchy (e.g. resolve "Base" for User.__superclass).
pub fn resolve_global_by_name(name: &str) -> Option<Value> {
    let vm_ptr = current_vm_ptr();
    let vm_ptr = vm_ptr?;
    unsafe {
        let vm = &mut *vm_ptr;
        let idx = global_index_by_name(vm.get_global_names(), name)?;
        if idx >= vm.get_globals().len() {
            return None;
        }
        let value_id = vm.resolve_global_to_value_id(idx);
        let value = load_value(value_id, vm.value_store(), vm.heavy_store());
        if let Value::Object(rc) = &value {
            if rc.borrow().get("__class_name").is_some() {
                return Some(value);
            }
        }
        None
    }
}

/// Вызвать пользовательскую функцию из нативной функции
/// Использует thread-local storage для доступа к VM
/// Invoke a [`Value::NativeFunction`] index: host natives (builtins + built-in modules) or ABI plugin.
pub fn call_native_by_index(native_index: usize, args: &[Value]) -> Result<Value, LangError> {
    let vm_ptr = current_vm_ptr().ok_or_else(|| {
        LangError::runtime_error(
            "Native call requires VM context".to_string(),
            0,
        )
    })?;
    unsafe {
        let vm: &mut Vm = &mut *vm_ptr;
        let host_len = vm.builtin_natives_count();
        if native_index < host_len {
            vm.get_natives()[native_index].invoke(args)
        } else {
            let abi_i = native_index - host_len;
            let abi = *vm.get_abi_natives().get(abi_i).ok_or_else(|| {
                LangError::runtime_error(
                    format!("Native index {} out of range", native_index),
                    0,
                )
            })?;
            Ok(crate::vm::native_loader::call_abi_native(
                abi,
                args,
                Some((vm.value_store(), vm.heavy_store())),
            ))
        }
    }
}

/// Dispatch a callable [`Value`] (native, user function, or module function).
pub fn invoke_value_callable(callee: &Value, args: &[Value]) -> Result<Value, LangError> {
    match callee {
        Value::NativeFunction(idx) => call_native_by_index(*idx, args),
        Value::Function(fn_idx) => call_user_function(*fn_idx, args),
        Value::ModuleFunction {
            module_uid,
            local_index,
        } => {
            let vm_ptr = current_vm_ptr().ok_or_else(|| {
                LangError::runtime_error(
                    "Callable requires VM context".to_string(),
                    0,
                )
            })?;
            unsafe {
                let vm = &mut *vm_ptr;
                let g_idx = vm
                    .get_module_function_index(*module_uid, *local_index)
                    .ok_or_else(|| {
                        LangError::runtime_error(
                            "ModuleFunction resolution failed".to_string(),
                            0,
                        )
                    })?;
                call_user_function(g_idx, args)
            }
        }
        _ => Err(LangError::runtime_error(
            "Not a callable (native, function, or module function)".to_string(),
            0,
        )),
    }
}

pub fn call_user_function(function_index: usize, args: &[Value]) -> Result<Value, LangError> {
    // Извлекаем указатель и сразу освобождаем заимствование контекста
    let vm_ptr = current_vm_ptr();

    if let Some(vm_ptr) = vm_ptr {
        unsafe {
            let vm = &mut *vm_ptr;
            vm.call_function_by_index(function_index, args)
        }
    } else {
        Err(LangError::runtime_error(
            "Cannot call user function: VM context not available".to_string(),
            0,
        ))
    }
}
