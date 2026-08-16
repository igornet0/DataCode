//! Builtin `set()` and `set` methods (`add`, `remove`, …).

use crate::common::set_map::SetMap;
use crate::common::type_model::is_hashable_value;
use crate::common::value::Value;
use crate::vm::calls::get_type_name_value;
use crate::vm::set_ops::{
    set_contains_value, set_discard_material, set_insert_material, set_remove_material,
};
use crate::vm::store_convert::load_value;
use crate::vm::vm::current_vm_ptr;
use std::cell::RefCell;
use std::rc::Rc;

/// Insert all elements of `iterable` into `set` (same coerce rules as `for-in` / `set.update`).
fn extend_set_from_iterable(set: &Rc<RefCell<SetMap>>, iterable: Value) -> Result<(), String> {
    let Some(vm_ptr) = current_vm_ptr() else {
        return Err("internal: VM unavailable".to_string());
    };
    unsafe {
        let vm = &mut *vm_ptr;
        let coerced = crate::vm::iterable::coerce_to_iterable_value(iterable)
            .map_err(|e| e.to_string())?;
        let Value::Iterable(rc) = coerced else {
            return Err("TypeError: expected an iterable".to_string());
        };
        loop {
            let next = {
                let mut inner = rc.borrow_mut();
                crate::vm::iterable::iterable_next(&mut *inner, vm).map_err(|e| e.to_string())?
            };
            let Some(item) = next else {
                break;
            };
            let mut err_msg: Option<String> = None;
            vm.with_stores_mut(|store, heap| {
                if let Err(msg) = set_insert_material(&mut set.borrow_mut(), &item, store, heap) {
                    err_msg = Some(msg);
                }
            });
            if let Some(msg) = err_msg {
                return Err(msg);
            }
        }
    }
    Ok(())
}

fn is_iterable_coerce_error(msg: &str) -> bool {
    msg.contains("for-in / iterable:") || msg.contains("expected an iterable")
}

/// `set()` → empty set; `set(iterable)` → set from iterable elements (must be hashable).
pub fn native_set(args: &[Value]) -> Value {
    match args.len() {
        0 => Value::Set(Rc::new(RefCell::new(SetMap::new()))),
        1 => match &args[0] {
            // Fast path: avoid coerce/iterable_next for plain arrays.
            Value::Array(a) => {
                let Some(vm_ptr) = current_vm_ptr() else {
                    return Value::Null;
                };
                let mut m = SetMap::new();
                let mut failed = false;
                unsafe {
                    (*vm_ptr).with_stores_mut(|store, heap| {
                        for item in a.borrow().iter() {
                            if let Err(msg) = set_insert_material(&mut m, item, store, heap) {
                                crate::websocket::set_native_error(msg);
                                failed = true;
                                break;
                            }
                        }
                    });
                }
                if failed {
                    return Value::Null;
                }
                Value::Set(Rc::new(RefCell::new(m)))
            }
            other => {
                let s = Rc::new(RefCell::new(SetMap::new()));
                match extend_set_from_iterable(&s, other.clone()) {
                    Ok(()) => Value::Set(s),
                    Err(msg) => {
                        let msg = if is_iterable_coerce_error(&msg) {
                            format!(
                                "TypeError: set() expected an iterable, got {}",
                                get_type_name_value(other)
                            )
                        } else {
                            msg
                        };
                        crate::websocket::set_native_error(msg);
                        Value::Null
                    }
                }
            }
        },
        n => {
            crate::websocket::set_native_error(format!(
                "TypeError: set() expected at most 1 argument, got {}",
                n
            ));
            Value::Null
        }
    }
}

/// `set.add(value)` — two-arg call: (set, value). Mutates set in place.
pub fn native_set_add(args: &[Value]) -> Value {
    if args.len() < 2 {
        return Value::Null;
    }
    let Value::Set(s) = &args[0] else {
        return Value::Null;
    };
    let item = args[1].clone();
    let Some(vm_ptr) = current_vm_ptr() else {
        return Value::Null;
    };
    unsafe {
        (*vm_ptr).with_stores_mut(|store, heap| {
            if let Err(msg) = set_insert_material(&mut s.borrow_mut(), &item, store, heap) {
                crate::websocket::set_native_error(msg);
            }
        });
    }
    Value::Set(Rc::clone(s))
}

/// `set.remove(value)` — `KeyError` if missing or value unhashable (same message style as `add`).
pub fn native_set_remove(args: &[Value]) -> Value {
    if args.len() < 2 {
        return Value::Null;
    }
    let Value::Set(s) = &args[0] else {
        return Value::Null;
    };
    let item = &args[1];
    let Some(vm_ptr) = current_vm_ptr() else {
        return Value::Null;
    };
    if !is_hashable_value(item) {
        crate::websocket::set_native_error(format!(
            "unhashable type: {}",
            get_type_name_value(item)
        ));
        return Value::Set(Rc::clone(s));
    }
    let removed = unsafe {
        (*vm_ptr).with_stores_mut(|store, heap| {
            set_remove_material(&mut s.borrow_mut(), item, store, heap)
        })
    };
    if !removed {
        crate::websocket::set_native_error(format!("KeyError: {}", item.to_string()));
    }
    Value::Set(Rc::clone(s))
}

/// `set.discard(value)` — no error if missing.
pub fn native_set_discard(args: &[Value]) -> Value {
    if args.len() < 2 {
        return Value::Null;
    }
    let Value::Set(s) = &args[0] else {
        return Value::Null;
    };
    let item = &args[1];
    let Some(vm_ptr) = current_vm_ptr() else {
        return Value::Null;
    };
    if !is_hashable_value(item) {
        return Value::Set(Rc::clone(s));
    }
    unsafe {
        (*vm_ptr).with_stores_mut(|store, heap| {
            set_discard_material(&mut s.borrow_mut(), item, store, heap);
        });
    }
    Value::Set(Rc::clone(s))
}

/// `set.pop()` — arbitrary element; `KeyError` if empty.
pub fn native_set_pop(args: &[Value]) -> Value {
    if args.is_empty() {
        return Value::Null;
    }
    let Value::Set(s) = &args[0] else {
        return Value::Null;
    };
    let Some(vm_ptr) = current_vm_ptr() else {
        return Value::Null;
    };
    let popped = unsafe {
        (*vm_ptr).with_stores_mut(|store, heap| {
            s.borrow_mut()
                .pop_arbitrary(|canonical| store.intern_whole_i64(canonical))
                .map(|kid| load_value(kid, store, heap))
        })
    };
    match popped {
        Some(v) => v,
        None => {
            crate::websocket::set_native_error("KeyError: pop from empty set".to_string());
            Value::Null
        }
    }
}

/// `set.clear()`
pub fn native_set_clear(args: &[Value]) -> Value {
    if args.is_empty() {
        return Value::Null;
    }
    let Value::Set(s) = &args[0] else {
        return Value::Null;
    };
    s.borrow_mut().clear();
    Value::Set(Rc::clone(s))
}

/// `set.copy()` — deep copy of the set and its elements.
pub fn native_set_copy(args: &[Value]) -> Value {
    if args.is_empty() {
        return Value::Null;
    }
    let Some(vm_ptr) = current_vm_ptr() else {
        return Value::Null;
    };
    unsafe {
        (*vm_ptr).with_stores_mut(|store, heap| {
            match crate::vm::deep_copy::deep_copy(&args[0], store, heap) {
                Ok(v) => v,
                Err(msg) => {
                    crate::websocket::set_native_error(msg);
                    if let Value::Set(s) = &args[0] {
                        Value::Set(Rc::clone(s))
                    } else {
                        Value::Null
                    }
                }
            }
        })
    }
}

/// `set.update(iterable)`
pub fn native_set_update(args: &[Value]) -> Value {
    if args.len() < 2 {
        return Value::Null;
    }
    let Value::Set(s) = &args[0] else {
        return Value::Null;
    };
    if let Err(msg) = extend_set_from_iterable(s, args[1].clone()) {
        crate::websocket::set_native_error(msg);
    }
    Value::Set(Rc::clone(s))
}

/// `set.contains(value)` — same semantics as `value in set`.
pub fn native_set_contains(args: &[Value]) -> Value {
    if args.len() < 2 {
        return Value::Null;
    }
    let Value::Set(s) = &args[0] else {
        return Value::Null;
    };
    let item = &args[1];
    let Some(vm_ptr) = current_vm_ptr() else {
        return Value::Null;
    };
    let found = unsafe {
        (*vm_ptr).with_stores_mut(|store, heap| {
            set_contains_value(&s.borrow(), item, store, heap)
        })
    };
    Value::Bool(found)
}
