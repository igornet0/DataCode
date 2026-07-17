//! Global builtin `copy(value)` — deep copy of containers.

use crate::common::value::Value;
use crate::vm::deep_copy::deep_copy;
use crate::vm::vm::current_vm_ptr;

/// `copy(value)` — recursive deep copy; scalars returned unchanged.
pub fn native_copy(args: &[Value]) -> Value {
    if args.is_empty() {
        return Value::Null;
    }
    let Some(vm_ptr) = current_vm_ptr() else {
        return Value::Null;
    };
    unsafe {
        (*vm_ptr).with_stores_mut(|store, heap| match deep_copy(&args[0], store, heap) {
            Ok(v) => v,
            Err(msg) => {
                crate::websocket::set_native_error(msg);
                Value::Null
            }
        })
    }
}
