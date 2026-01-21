// Mathematical native functions

use crate::common::value::Value;
use std::rc::Rc;
use std::cell::RefCell;

pub fn native_abs(args: &[Value]) -> Value {
    if args.is_empty() {
        return Value::Number(0.0);
    }
    
    // Check if first argument is a tensor
    if let Value::Tensor(tensor) = &args[0] {
        let tensor_ref = tensor.borrow();
        let result = tensor_ref.abs();
        return Value::Tensor(Rc::new(RefCell::new(result)));
    }
    
    // Handle number (original behavior)
    match &args[0] {
        Value::Number(n) => Value::Number(n.abs()),
        _ => Value::Null,
    }
}

pub fn native_sqrt(args: &[Value]) -> Value {
    if args.is_empty() {
        return Value::Number(0.0);
    }
    
    // Check if first argument is a tensor
    if let Value::Tensor(tensor) = &args[0] {
        let tensor_ref = tensor.borrow();
        match tensor_ref.sqrt() {
            Ok(result) => return Value::Tensor(Rc::new(RefCell::new(result))),
            Err(_) => return Value::Null,
        }
    }
    
    // Handle number (original behavior)
    match &args[0] {
        Value::Number(n) => {
            if *n < 0.0 {
                Value::Null // Отрицательное число - возвращаем Null
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
    
    // Check if first argument is a tensor
    if let Value::Tensor(tensor) = &args[0] {
        let tensor_ref = tensor.borrow();
        match tensor_ref.to_cpu() {
            Ok(cpu_tensor) => {
                if cpu_tensor.data.is_empty() {
                    return Value::Null;
                }
                let min_val = cpu_tensor.data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
                return Value::Number(min_val as f64);
            }
            Err(_) => return Value::Null,
        }
    }
    
    // Handle multiple number arguments (original behavior)
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
            _ => return Value::Null, // Если есть нечисловой аргумент, возвращаем Null
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
    
    // Check if first argument is a tensor
    if let Value::Tensor(tensor) = &args[0] {
        let tensor_ref = tensor.borrow();
        match tensor_ref.to_cpu() {
            Ok(cpu_tensor) => {
                if cpu_tensor.data.is_empty() {
                    return Value::Null;
                }
                let max_val = cpu_tensor.data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                return Value::Number(max_val as f64);
            }
            Err(_) => return Value::Null,
        }
    }
    
    // Handle multiple number arguments (original behavior)
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
            _ => return Value::Null, // Если есть нечисловой аргумент, возвращаем Null
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
    
    // Check if first argument is a tensor
    if let Value::Tensor(tensor) = &args[0] {
        let tensor_ref = tensor.borrow();
        let result = tensor_ref.round();
        return Value::Tensor(Rc::new(RefCell::new(result)));
    }
    
    // Handle number (original behavior)
    match &args[0] {
        Value::Number(n) => {
            // Стандартное округление: к ближайшему целому
            // Для положительных: 3.5 -> 4, для отрицательных: -3.5 -> -3 (к нулю)
            if *n >= 0.0 {
                Value::Number(n.floor() + if n.fract() >= 0.5 { 1.0 } else { 0.0 })
            } else {
                // Для отрицательных: округляем к нулю
                // -3.5 -> -3, -3.6 -> -4
                // Для отрицательных чисел fract() возвращает положительное значение дробной части
                let abs_fract = n.abs().fract();
                if abs_fract > 0.5 {
                    // Округляем вниз (от нуля)
                    Value::Number(n.floor())
                } else if abs_fract < 0.5 {
                    // Округляем вверх (к нулю)
                    Value::Number(n.ceil())
                } else {
                    // Ровно 0.5 - округляем к нулю
                    Value::Number(n.ceil())
                }
            }
        }
        _ => Value::Null,
    }
}

