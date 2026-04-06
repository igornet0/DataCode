//! Debug helpers (`debug.operators()`).

use crate::common::value::Value;
use crate::vm::vm::current_vm_ptr;

/// Returns a tab-separated table of registered infix operators (requires [`crate::vm::Vm::set_operator_registry_snapshot`]).
pub fn native_debug_operators(_args: &[Value]) -> Value {
    let Some(ptr) = current_vm_ptr() else {
        return Value::String("(no vm context)".to_string());
    };
    unsafe {
        let vm = &*ptr;
        let Some(ref reg) = vm.operator_registry_snapshot else {
            return Value::String(
                "(no operator registry snapshot — host did not set parse-time registry)".to_string(),
            );
        };
        let s = reg.format_debug_text();
        if s.is_empty() {
            Value::String("(empty operator registry)".to_string())
        } else {
            Value::String(s)
        }
    }
}
