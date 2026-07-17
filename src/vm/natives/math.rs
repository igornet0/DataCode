// Mathematical native functions

use crate::common::numeric::{
    divmod_f64, divmod_i64, FloatValue, IntValue, integer_value_as_i64_if_whole,
};
use crate::common::value::Value;
use crate::vm::iterable::coerce_to_iterable_value;
use crate::vm::native_loader::call_abi_native;
use crate::vm::natives::extremum::extremum_over_iterable;
use crate::websocket::set_native_error;
use crate::vm::vm::current_vm_ptr;
use std::cell::RefCell;
use std::rc::Rc;

fn plugin_opaque_min_max_via_abi(arg: &Value, op: &str) -> Option<Value> {
    let Value::PluginOpaque { .. } = arg else {
        return None;
    };
    let vm_ptr = current_vm_ptr()?;
    unsafe {
        let vm = &*vm_ptr;
        let native_idx = vm.plugin_call_native?;
        let builtin_count = vm.builtin_natives_count();
        let abi = vm.get_abi_natives();
        if native_idx < builtin_count || native_idx >= builtin_count + abi.len() {
            return None;
        }
        let args = [arg.clone(), Value::String(op.to_string())];
        Some(call_abi_native(
            abi[native_idx - builtin_count],
            &args,
            Some((vm.value_store(), vm.heavy_store())),
        ))
    }
}

fn column_value_for_extremum(arg: &Value) -> Option<Value> {
    let Value::ColumnReference { table, column_name } = arg else {
        return None;
    };
    let vm_ptr = current_vm_ptr()?;
    unsafe {
        let vm = &mut *vm_ptr;
        let mut t = table.borrow_mut();
        crate::vm::table_ops::get_column(&mut *t, column_name, vm.value_store(), vm.heavy_store())
            .map(|col| Value::Array(Rc::new(RefCell::new(col))))
    }
}

fn value_is_extremum_iterable(v: &Value) -> bool {
    coerce_to_iterable_value(v.clone()).is_ok()
}

fn is_key_fn(v: &Value) -> bool {
    matches!(v, Value::Function(_) | Value::NativeFunction(_))
}

fn varargs_numbers_min(args: &[Value]) -> Value {
    let mut min_val: Option<f64> = None;

    for arg in args {
        match arg.as_ieee_f64() {
            Some(n) => {
                if let Some(current_min) = min_val {
                    if n < current_min {
                        min_val = Some(n);
                    }
                } else {
                    min_val = Some(n);
                }
            }
            None => return Value::Null,
        }
    }

    match min_val {
        Some(n) => Value::Number(n),
        None => Value::Null,
    }
}

fn varargs_numbers_max(args: &[Value]) -> Value {
    let mut max_val: Option<f64> = None;

    for arg in args {
        match arg.as_ieee_f64() {
            Some(n) => {
                if let Some(current_max) = max_val {
                    if n > current_max {
                        max_val = Some(n);
                    }
                } else {
                    max_val = Some(n);
                }
            }
            None => return Value::Null,
        }
    }

    match max_val {
        Some(n) => Value::Number(n),
        None => Value::Null,
    }
}

pub fn native_abs(args: &[Value]) -> Value {
    if args.is_empty() {
        return Value::Number(0.0);
    }

    match args[0].as_ieee_f64() {
        Some(n) => Value::Number(n.abs()),
        None => Value::Null,
    }
}

pub fn native_sqrt(args: &[Value]) -> Value {
    if args.is_empty() {
        return Value::Number(0.0);
    }

    match args[0].as_ieee_f64() {
        Some(n) if n >= 0.0 => Value::Number(n.sqrt()),
        Some(_) => Value::Null,
        None => Value::Null,
    }
}

pub fn native_pow(args: &[Value]) -> Value {
    if args.len() < 2 {
        return Value::Null;
    }

    let Some(base) = args[0].as_ieee_f64() else {
        return Value::Null;
    };
    let Some(exp) = args[1].as_ieee_f64() else {
        return Value::Null;
    };

    Value::Number(base.powf(exp))
}

pub fn native_min(args: &[Value]) -> Value {
    if args.is_empty() {
        set_native_error("ValueError: min() expects at least 1 argument".to_string());
        return Value::Null;
    }

    if args.len() == 1 {
        if let Some(v) = plugin_opaque_min_max_via_abi(&args[0], "min") {
            return v;
        }
        if matches!(&args[0], Value::PluginOpaque { .. }) {
            return Value::Null;
        }
        if let Some(arr) = column_value_for_extremum(&args[0]) {
            return extremum_over_iterable(&arr, None, false);
        }
        if value_is_extremum_iterable(&args[0]) {
            return extremum_over_iterable(&args[0], None, false);
        }
    }

    if args.len() >= 2 {
        let mut all_plain_numbers = true;
        for a in args {
            if a.as_ieee_f64().is_none() {
                all_plain_numbers = false;
                break;
            }
        }
        if all_plain_numbers {
            return varargs_numbers_min(args);
        }

        if args.len() == 2 {
            let coll = if let Some(arr) = column_value_for_extremum(&args[0]) {
                Some(arr)
            } else if value_is_extremum_iterable(&args[0]) {
                Some(args[0].clone())
            } else {
                None
            };

            if let Some(ref coll_v) = coll {
                if is_key_fn(&args[1]) {
                    return extremum_over_iterable(coll_v, Some(&args[1]), false);
                }
            }
        }

        return Value::Null;
    }

    varargs_numbers_min(args)
}

pub fn native_max(args: &[Value]) -> Value {
    if args.is_empty() {
        set_native_error("ValueError: max() expects at least 1 argument".to_string());
        return Value::Null;
    }

    if args.len() == 1 {
        if let Some(v) = plugin_opaque_min_max_via_abi(&args[0], "max") {
            return v;
        }
        if matches!(&args[0], Value::PluginOpaque { .. }) {
            return Value::Null;
        }
        if let Some(arr) = column_value_for_extremum(&args[0]) {
            return extremum_over_iterable(&arr, None, true);
        }
        if value_is_extremum_iterable(&args[0]) {
            return extremum_over_iterable(&args[0], None, true);
        }
    }

    if args.len() >= 2 {
        let mut all_plain_numbers = true;
        for a in args {
            if a.as_ieee_f64().is_none() {
                all_plain_numbers = false;
                break;
            }
        }
        if all_plain_numbers {
            return varargs_numbers_max(args);
        }

        if args.len() == 2 {
            let coll = if let Some(arr) = column_value_for_extremum(&args[0]) {
                Some(arr)
            } else if value_is_extremum_iterable(&args[0]) {
                Some(args[0].clone())
            } else {
                None
            };

            if let Some(ref coll_v) = coll {
                if is_key_fn(&args[1]) {
                    return extremum_over_iterable(coll_v, Some(&args[1]), true);
                }
            }
        }

        return Value::Null;
    }

    varargs_numbers_max(args)
}

pub fn native_round(args: &[Value]) -> Value {
    if args.is_empty() {
        return Value::Number(0.0);
    }

    match args[0].as_ieee_f64() {
        Some(n) if n >= 0.0 => {
            Value::Number(n.floor() + if n.fract() >= 0.5 { 1.0 } else { 0.0 })
        }
        Some(n) => {
            let abs_fract = n.abs().fract();
            if abs_fract > 0.5 {
                Value::Number(n.floor())
            } else {
                Value::Number(n.ceil())
            }
        }
        None => Value::Null,
    }
}

pub fn native_ceil(args: &[Value]) -> Value {
    if args.is_empty() {
        return Value::Number(0.0);
    }

    match args[0].as_ieee_f64() {
        Some(n) => Value::Number(n.ceil()),
        None => Value::Null,
    }
}

pub fn native_floor(args: &[Value]) -> Value {
    if args.is_empty() {
        return Value::Number(0.0);
    }

    match args[0].as_ieee_f64() {
        Some(n) => Value::Number(n.floor()),
        None => Value::Null,
    }
}

/// `divmod(a, b)` → `(q, r)` with Python floor-division semantics; returns a 2-tuple.
pub fn native_divmod(args: &[Value]) -> Value {
    if args.len() != 2 {
        set_native_error("TypeError: divmod() expects exactly 2 arguments".to_string());
        return Value::Null;
    }

    let a = &args[0];
    let b = &args[1];

    if let (Some(ai), Some(bi)) = (
        integer_value_as_i64_if_whole(a),
        integer_value_as_i64_if_whole(b),
    ) {
        if bi == 0 {
            set_native_error("ZeroDivisionError: integer division or modulo by zero".to_string());
            return Value::Null;
        }
        let (q, r) = divmod_i64(ai, bi);
        return Value::Tuple(Rc::new(RefCell::new(vec![
            Value::Int(IntValue::Finite(q)),
            Value::Int(IntValue::Finite(r)),
        ])));
    }

    let Some(x) = a.as_ieee_f64() else {
        set_native_error("TypeError: divmod() expected numeric operands".to_string());
        return Value::Null;
    };
    let Some(y) = b.as_ieee_f64() else {
        set_native_error("TypeError: divmod() expected numeric operands".to_string());
        return Value::Null;
    };

    if y == 0.0 {
        set_native_error("ZeroDivisionError: float division or modulo by zero".to_string());
        return Value::Null;
    }

    let (q, r) = divmod_f64(x, y);
    Value::Tuple(Rc::new(RefCell::new(vec![
        Value::Float(FloatValue::Finite(q)),
        Value::Float(FloatValue::Finite(r)),
    ])))
}
