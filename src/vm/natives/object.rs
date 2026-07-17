//! Plain bucket dict `.get(key, default=null)` / `.clear()` — binding; hot paths in VM.

use crate::common::value::{ObjectKind, Value};

/// Fallback when call does not hit the VM fast path (wrong receiver / arity).
pub fn native_object_get(_args: &[Value]) -> Value {
    Value::Null
}

/// `dict.clear()` — mutates materialized plain dict (`update_cell_if_mutable` writes back to store).
pub fn native_object_clear(args: &[Value]) -> Value {
    if args.is_empty() {
        return Value::Null;
    }
    let Value::Object(rc) = &args[0] else {
        return Value::Null;
    };
    if let ObjectKind::Bucket(ref mut omap) = *rc.borrow_mut() {
        omap.clear();
    }
    Value::Object(std::rc::Rc::clone(rc))
}
