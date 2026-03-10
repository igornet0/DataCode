// Utility functions for native functions

use crate::common::value::Value;
use crate::common::error::LangError;
use crate::vm::global_utils::global_index_by_name;
use crate::vm::store_convert::load_value;
use crate::vm::vm::VM_CALL_CONTEXT;

/// Resolve a global by name from VM; returns Some(Value) if found and it's an Object with __class_name.
/// Used by database_engine to walk class hierarchy (e.g. resolve "Base" for User.__superclass).
pub fn resolve_global_by_name(name: &str) -> Option<Value> {
    let vm_ptr = VM_CALL_CONTEXT.with(|ctx| *ctx.borrow());
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
pub fn call_user_function(function_index: usize, args: &[Value]) -> Result<Value, LangError> {
    // Извлекаем указатель и сразу освобождаем заимствование контекста
    let vm_ptr = VM_CALL_CONTEXT.with(|ctx| {
        let ctx_ref = ctx.borrow();
        *ctx_ref
    });
    
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

