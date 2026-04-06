//! Debug helpers (`debug.operators()`).

use crate::common::value::Value;
use crate::vm::vm::VM_CALL_CONTEXT;

/// Returns a tab-separated table of registered infix operators (requires [`crate::vm::Vm::set_operator_registry_snapshot`]).
pub fn native_debug_operators(_args: &[Value]) -> Value {
    VM_CALL_CONTEXT.with(|ctx| {
        let ptr = match *ctx.borrow() {
            Some(p) => p,
            None => return Value::String("(no vm context)".to_string()),
        };
        unsafe {
            let vm = &*ptr;
            let Some(ref reg) = vm.operator_registry_snapshot else {
                return Value::String(
                    "(no operator registry snapshot — host did not set parse-time registry)"
                        .to_string(),
                );
            };
            let s = reg.format_debug_text();
            if s.is_empty() {
                Value::String("(empty operator registry)".to_string())
            } else {
                Value::String(s)
            }
        }
    })
}
