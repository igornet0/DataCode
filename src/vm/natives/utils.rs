// Utility functions for native functions

use crate::common::value::Value;
use crate::common::error::LangError;
use crate::vm::vm::VM_CALL_CONTEXT;

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

