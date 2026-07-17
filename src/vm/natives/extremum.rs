//! Streaming ``min`` / ``max`` over iterable values (lazy, no intermediate array).

use std::cmp::Ordering;

use crate::common::value::{CallableSlot, Value};
use crate::common::value_ord::value_partial_cmp;
use crate::vm::iterable::{
    dispatch_callable, iterable_from_value, iterable_next, value_to_callable_slot,
};
use crate::vm::vm::current_vm_ptr;
use crate::websocket::set_native_error;

pub fn extremum_over_iterable(coll: &Value, key_fn: Option<&Value>, is_max: bool) -> Value {
    let vm_ptr =
        match current_vm_ptr() {
            Some(p) => p,
            None => {
                set_native_error("RuntimeError: min/max: VM context not available".to_string());
                return Value::Null;
            }
        };

    let rc = match iterable_from_value(coll) {
        Ok(rc) => rc,
        Err(e) => {
            set_native_error(e.to_string());
            return Value::Null;
        }
    };

    let (slot_opt, arity_opt) = if let Some(k) = key_fn {
        unsafe {
            let vm = &*vm_ptr;
            match value_to_callable_slot(k, vm) {
                Ok(pair) => {
                    if pair.1 != 1 && pair.1 != 2 {
                        set_native_error(
                            "min/max key function must have arity 1 or 2 (item[, index])".to_string(),
                        );
                        return Value::Null;
                    }
                    (Some(pair.0), Some(pair.1))
                }
                Err(e) => {
                    set_native_error(e.to_string());
                    return Value::Null;
                }
            }
        }
    } else {
        (None, None)
    };

    let op = if is_max { "max" } else { "min" };
    unsafe {
        let vm = &mut *vm_ptr;
        let mut inner = rc.borrow_mut();
        let first = match iterable_next(&mut *inner, vm) {
            Ok(None) => {
                let msg = if is_max {
                    "ValueError: max() arg is empty"
                } else {
                    "ValueError: min() arg is empty"
                };
                set_native_error(msg.to_string());
                return Value::Null;
            }
            Ok(Some(v)) => v,
            Err(e) => {
                set_native_error(e.to_string());
                return Value::Null;
            }
        };

        let mut best_elem = first;
        let mut best_key_val = match (&slot_opt, &arity_opt) {
            (&Some(ref slot), &Some(arity)) => {
                let k = compute_key(slot, arity, &best_elem, 0_u64, vm);
                match k {
                    Ok(v) => {
                        if matches!(v, Value::Null) {
                            let msg =
                                format!("TypeError: {op}() key function returned null");
                            set_native_error(msg);
                            return Value::Null;
                        }
                        Some(v)
                    }
                    Err(e) => {
                        set_native_error(e.to_string());
                        return Value::Null;
                    }
                }
            }
            _ => None,
        };

        let mut idx: u64 = 1;

        loop {
            let item = match iterable_next(&mut *inner, vm) {
                Ok(None) => break,
                Ok(Some(v)) => v,
                Err(e) => {
                    set_native_error(e.to_string());
                    return Value::Null;
                }
            };

            match (&slot_opt, arity_opt, &best_key_val) {
                (Some(slot), Some(arity), Some(cur_best_key)) => {
                    let k = match compute_key(slot, arity, &item, idx, vm) {
                        Ok(v) => v,
                        Err(e) => {
                            set_native_error(e.to_string());
                            return Value::Null;
                        }
                    };
                    if matches!(k, Value::Null) {
                        let msg = format!("TypeError: {op}() key function returned null");
                        set_native_error(msg);
                        return Value::Null;
                    }
                    let cmp_best = match value_partial_cmp(&k, cur_best_key) {
                        Ok(c) => c,
                        Err(msg) => {
                            set_native_error(msg);
                            return Value::Null;
                        }
                    };
                    if better_extremum(cmp_best, is_max) {
                        best_elem = item;
                        best_key_val = Some(k);
                    }
                }
                _ => match value_partial_cmp(&item, &best_elem) {
                    Ok(c) => {
                        if better_extremum(c, is_max) {
                            best_elem = item;
                        }
                    }
                    Err(msg) => {
                        set_native_error(msg);
                        return Value::Null;
                    }
                },
            }

            idx = idx.wrapping_add(1);
        }

        best_elem
    }
}

fn better_extremum(cmp: Ordering, is_max: bool) -> bool {
    if is_max {
        cmp == Ordering::Greater
    } else {
        cmp == Ordering::Less
    }
}

fn compute_key(
    slot: &CallableSlot,
    arity: u8,
    item: &Value,
    linear_index: u64,
    vm: &mut crate::vm::vm::Vm,
) -> Result<Value, crate::common::error::LangError> {
    if arity == 2 {
        dispatch_callable(
            slot,
            &[
                item.clone(),
                Value::Number(linear_index as f64),
            ],
            vm,
        )
    } else {
        dispatch_callable(slot, std::slice::from_ref(item), vm)
    }
}
