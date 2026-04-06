// Mathematical native functions

use crate::common::value::Value;
use crate::vm::native_loader::call_abi_native;
use crate::vm::vm::VM_CALL_CONTEXT;
use std::cell::RefCell;
use std::rc::Rc;

fn plugin_opaque_min_max_via_abi(arg: &Value, op: &str) -> Option<Value> {
    let Value::PluginOpaque { .. } = arg else {
        return None;
    };
    let vm_ptr = VM_CALL_CONTEXT.with(|ctx| *ctx.borrow())?;
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

/// Min/max over numeric elements of an array; non-numbers are skipped. `None` if no numbers.
fn extremum_from_array(arr: &Rc<RefCell<Vec<Value>>>, is_max: bool) -> Value {
    let mut best: Option<f64> = None;
    for item in arr.borrow().iter() {
        if let Value::Number(n) = item {
            best = match best {
                None => Some(*n),
                Some(b) if is_max => Some(if *n > b { *n } else { b }),
                Some(b) => Some(if *n < b { *n } else { b }),
            };
        }
    }
    match best {
        Some(n) => Value::Number(n),
        None => Value::Null,
    }
}

pub fn native_abs(args: &[Value]) -> Value {
    if args.is_empty() {
        return Value::Number(0.0);
    }

    match &args[0] {
        Value::Number(n) => Value::Number(n.abs()),
        _ => Value::Null,
    }
}

pub fn native_sqrt(args: &[Value]) -> Value {
    if args.is_empty() {
        return Value::Number(0.0);
    }

    match &args[0] {
        Value::Number(n) => {
            if *n < 0.0 {
                Value::Null
            } else {
                Value::Number(n.sqrt())
            }
        }
        _ => Value::Null,
    }
}

pub fn native_pow(args: &[Value]) -> Value {
    if args.len() < 2 {
        return Value::Null;
    }

    let base = match &args[0] {
        Value::Number(n) => *n,
        _ => return Value::Null,
    };

    let exp = match &args[1] {
        Value::Number(n) => *n,
        _ => return Value::Null,
    };

    Value::Number(base.powf(exp))
}

pub fn native_min(args: &[Value]) -> Value {
    if args.is_empty() {
        return Value::Null;
    }

    if args.len() == 1 {
        if let Some(v) = plugin_opaque_min_max_via_abi(&args[0], "min") {
            return v;
        }
        if matches!(&args[0], Value::PluginOpaque { .. }) {
            return Value::Null;
        }
        if let Value::Array(a) = &args[0] {
            return extremum_from_array(a, false);
        }
    }

    let mut min_val: Option<f64> = None;

    for arg in args {
        match arg {
            Value::Number(n) => {
                if let Some(current_min) = min_val {
                    if *n < current_min {
                        min_val = Some(*n);
                    }
                } else {
                    min_val = Some(*n);
                }
            }
            _ => return Value::Null,
        }
    }

    match min_val {
        Some(n) => Value::Number(n),
        None => Value::Null,
    }
}

pub fn native_max(args: &[Value]) -> Value {
    if args.is_empty() {
        return Value::Null;
    }

    if args.len() == 1 {
        if let Some(v) = plugin_opaque_min_max_via_abi(&args[0], "max") {
            return v;
        }
        if matches!(&args[0], Value::PluginOpaque { .. }) {
            return Value::Null;
        }
        if let Value::Array(a) = &args[0] {
            return extremum_from_array(a, true);
        }
    }

    let mut max_val: Option<f64> = None;

    for arg in args {
        match arg {
            Value::Number(n) => {
                if let Some(current_max) = max_val {
                    if *n > current_max {
                        max_val = Some(*n);
                    }
                } else {
                    max_val = Some(*n);
                }
            }
            _ => return Value::Null,
        }
    }

    match max_val {
        Some(n) => Value::Number(n),
        None => Value::Null,
    }
}

pub fn native_round(args: &[Value]) -> Value {
    if args.is_empty() {
        return Value::Number(0.0);
    }

    match &args[0] {
        Value::Number(n) => {
            if *n >= 0.0 {
                Value::Number(n.floor() + if n.fract() >= 0.5 { 1.0 } else { 0.0 })
            } else {
                let abs_fract = n.abs().fract();
                if abs_fract > 0.5 {
                    Value::Number(n.floor())
                } else {
                    Value::Number(n.ceil())
                }
            }
        }
        _ => Value::Null,
    }
}
