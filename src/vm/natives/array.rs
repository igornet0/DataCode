// Array manipulation native functions

use crate::common::value::Value;
use std::rc::Rc;
use std::cell::RefCell;
use std::collections::HashSet;

pub fn native_push(args: &[Value]) -> Value {
    if args.len() < 2 {
        return Value::Null;
    }
    
    let arr = match &args[0] {
        Value::Array(a) => a,
        _ => return Value::Null,
    };
    
    // Мутируем массив in-place, сохраняя ссылочную семантику
    // Если массив используется в нескольких местах, изменения будут видны через все ссылки
    let item = args[1].clone();
    arr.borrow_mut().push(item);
    
    // Возвращаем тот же массив (клонируем только Rc, не содержимое)
    Value::Array(Rc::clone(arr))
}

pub fn native_pop(args: &[Value]) -> Value {
    if args.is_empty() {
        return Value::Null;
    }
    
    let arr = match &args[0] {
        Value::Array(a) => a,
        _ => return Value::Null,
    };
    
    // Copy-on-Write: если массив используется в нескольких местах, клонируем
    let arr = if Rc::strong_count(arr) > 1 {
        let cloned_vec: Vec<Value> = arr.borrow().iter().map(|v| v.clone()).collect();
        Rc::new(RefCell::new(cloned_vec))
    } else {
        arr.clone()
    };
    
    let mut arr_ref = arr.borrow_mut();
    if arr_ref.is_empty() {
        return Value::Null;
    }
    
    // Возвращаем последний элемент (удаленный элемент)
    arr_ref.pop().unwrap_or(Value::Null)
}

pub fn native_unique(args: &[Value]) -> Value {
    if args.is_empty() {
        return Value::Array(Rc::new(RefCell::new(Vec::new())));
    }
    
    let arr = match &args[0] {
        Value::Array(a) => a,
        _ => return Value::Null,
    };
    
    // Создаем новый массив с уникальными элементами, сохраняя порядок первого вхождения
    // Используем HashSet для O(1) проверки вместо O(n) Vec::contains
    let arr_ref = arr.borrow();
    let mut seen = HashSet::new();
    let mut result = Vec::new();
    
    for item in arr_ref.iter() {
        // Используем строковое представление для хэширования, так как Value не Hash
        let item_str = item.to_string();
        if !seen.contains(&item_str) {
            seen.insert(item_str);
            result.push(item.clone());
        }
    }
    
    Value::Array(Rc::new(RefCell::new(result)))
}

pub fn native_reverse(args: &[Value]) -> Value {
    if args.is_empty() {
        return Value::Array(Rc::new(RefCell::new(Vec::new())));
    }
    
    let arr = match &args[0] {
        Value::Array(a) => a,
        _ => return Value::Null,
    };
    
    // Copy-on-Write: если массив используется в нескольких местах, клонируем
    let arr = if Rc::strong_count(arr) > 1 {
        let cloned_vec: Vec<Value> = arr.borrow().iter().map(|v| v.clone()).collect();
        Rc::new(RefCell::new(cloned_vec))
    } else {
        arr.clone()
    };
    
    // Мутируем in-place
    arr.borrow_mut().reverse();
    
    Value::Array(arr)
}

pub fn native_sort(args: &[Value]) -> Value {
    if args.is_empty() {
        return Value::Array(Rc::new(RefCell::new(Vec::new())));
    }
    
    let arr = match &args[0] {
        Value::Array(a) => a,
        _ => return Value::Null,
    };
    
    // Copy-on-Write: если массив используется в нескольких местах, клонируем
    let arr = if Rc::strong_count(arr) > 1 {
        let cloned_vec: Vec<Value> = arr.borrow().iter().map(|v| v.clone()).collect();
        Rc::new(RefCell::new(cloned_vec))
    } else {
        arr.clone()
    };
    
    // Сортируем in-place по строковому представлению для сравнения разных типов
    arr.borrow_mut().sort_by(|a, b| {
        let a_str = a.to_string();
        let b_str = b.to_string();
        a_str.cmp(&b_str)
    });
    
    Value::Array(arr)
}

pub fn native_sum(args: &[Value]) -> Value {
    if args.is_empty() {
        return Value::Number(0.0);
    }
    
    // Check if first argument is a tensor
    if let Value::Tensor(tensor) = &args[0] {
        let tensor_ref = tensor.borrow();
        let sum = tensor_ref.sum();
        return Value::Number(sum as f64);
    }
    
    // Handle array (original behavior)
    let arr = match &args[0] {
        Value::Array(a) => a,
        _ => return Value::Number(0.0),
    };
    
    let arr_ref = arr.borrow();
    let mut sum = 0.0;
    let mut has_numbers = false;
    
    for item in arr_ref.iter() {
        if let Value::Number(n) = item {
            sum += n;
            has_numbers = true;
        }
    }
    
    if has_numbers {
        Value::Number(sum)
    } else {
        Value::Number(0.0)
    }
}

pub fn native_average(args: &[Value]) -> Value {
    if args.is_empty() {
        return Value::Number(0.0);
    }
    
    // Check if first argument is a tensor
    if let Value::Tensor(tensor) = &args[0] {
        let tensor_ref = tensor.borrow();
        let mean = tensor_ref.mean();
        return Value::Number(mean as f64);
    }
    
    // Handle array (original behavior)
    let arr = match &args[0] {
        Value::Array(a) => a,
        _ => return Value::Number(0.0),
    };
    
    let arr_ref = arr.borrow();
    let mut sum = 0.0;
    let mut count = 0;
    
    for item in arr_ref.iter() {
        if let Value::Number(n) = item {
            sum += n;
            count += 1;
        }
    }
    
    if count > 0 {
        Value::Number(sum / count as f64)
    } else {
        Value::Number(0.0)
    }
}

pub fn native_count(args: &[Value]) -> Value {
    if args.is_empty() {
        return Value::Number(0.0);
    }
    
    // Check if first argument is a tensor
    if let Value::Tensor(tensor) = &args[0] {
        let tensor_ref = tensor.borrow();
        let count = tensor_ref.total_size();
        return Value::Number(count as f64);
    }
    
    // Handle array (original behavior)
    match &args[0] {
        Value::Array(arr) => Value::Number(arr.borrow().len() as f64),
        _ => Value::Number(0.0),
    }
}

pub fn native_any(args: &[Value]) -> Value {
    if args.is_empty() {
        return Value::Bool(false);
    }
    
    // Check if first argument is a tensor
    if let Value::Tensor(tensor) = &args[0] {
        let tensor_ref = tensor.borrow();
        match tensor_ref.to_cpu() {
            Ok(cpu_tensor) => {
                if cpu_tensor.data.is_empty() {
                    return Value::Bool(false);
                }
                // Check if there's at least one non-zero element
                for &val in cpu_tensor.data.iter() {
                    if val != 0.0 {
                        return Value::Bool(true);
                    }
                }
                return Value::Bool(false);
            }
            Err(_) => return Value::Bool(false),
        }
    }
    
    // Handle array (original behavior)
    let arr = match &args[0] {
        Value::Array(a) => a,
        _ => return Value::Bool(false),
    };
    
    let arr_ref = arr.borrow();
    
    // Для пустого массива возвращаем false
    if arr_ref.is_empty() {
        return Value::Bool(false);
    }
    
    // Проверяем, есть ли хотя бы одно истинное значение
    for item in arr_ref.iter() {
        if item.is_truthy() {
            return Value::Bool(true);
        }
    }
    
    Value::Bool(false)
}

pub fn native_all(args: &[Value]) -> Value {
    if args.is_empty() {
        return Value::Bool(false);
    }
    
    // Check if first argument is a tensor
    if let Value::Tensor(tensor) = &args[0] {
        let tensor_ref = tensor.borrow();
        match tensor_ref.to_cpu() {
            Ok(cpu_tensor) => {
                if cpu_tensor.data.is_empty() {
                    return Value::Bool(false);
                }
                // Check if all elements are non-zero
                for &val in cpu_tensor.data.iter() {
                    if val == 0.0 {
                        return Value::Bool(false);
                    }
                }
                return Value::Bool(true);
            }
            Err(_) => return Value::Bool(false),
        }
    }
    
    // Handle array (original behavior)
    let arr = match &args[0] {
        Value::Array(a) => a,
        _ => return Value::Bool(false),
    };
    
    let arr_ref = arr.borrow();
    
    // Для пустого массива возвращаем false (согласно плану)
    if arr_ref.is_empty() {
        return Value::Bool(false);
    }
    
    // Проверяем, все ли значения истинны
    for item in arr_ref.iter() {
        if !item.is_truthy() {
            return Value::Bool(false);
        }
    }
    
    Value::Bool(true)
}

