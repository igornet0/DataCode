// Mathematical native functions

use crate::common::value::Value;

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
                } else if abs_fract < 0.5 {
                    Value::Number(n.ceil())
                } else {
                    Value::Number(n.ceil())
                }
            }
        }
        _ => Value::Null,
    }
}
