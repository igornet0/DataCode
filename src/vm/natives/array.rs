// Array manipulation native functions

use crate::common::value::Value;
use std::rc::Rc;
use std::cell::RefCell;
use std::collections::HashSet;

/// Returns an empty array with pre-allocated capacity to avoid reallocations when using push() in a loop.
pub fn native_array_with_capacity(args: &[Value]) -> Value {
    if args.is_empty() {
        return Value::Array(Rc::new(RefCell::new(Vec::new())));
    }
    let n = match &args[0] {
        Value::Number(x) => {
            let idx = *x as i64;
            if idx < 0 || idx > 1_000_000_000 {
                return Value::Null;
            }
            idx as usize
        }
        _ => return Value::Null,
    };
    Value::Array(Rc::new(RefCell::new(Vec::with_capacity(n))))
}

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
    let mut seen = HashSet::new();
    let mut result = Vec::new();
    match &args[0] {
        Value::Array(a) => {
            for item in a.borrow().iter() {
                let item_str = item.to_string();
                if !seen.contains(&item_str) {
                    seen.insert(item_str);
                    result.push(item.clone());
                }
            }
        }
        Value::ColumnReference { table, column_name } => {
            crate::vm::vm::with_current_stores(|store, heap| {
                let t = table.borrow();
                for i in 0..t.len() {
                    if let Some(item) = crate::vm::table_ops::get_cell_value(&*t, i, column_name, store, heap) {
                        let item_str = item.to_string();
                        if !seen.contains(&item_str) {
                            seen.insert(item_str);
                            result.push(item);
                        }
                    }
                }
            });
        }
        _ => return Value::Null,
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
    if let Value::Tensor(tensor) = &args[0] {
        let tensor_ref = tensor.borrow();
        return Value::Number(tensor_ref.sum() as f64);
    }
    let (mut sum, mut has_numbers) = (0.0, false);
    match &args[0] {
        Value::Array(a) => {
            for item in a.borrow().iter() {
                if let Value::Number(n) = item {
                    sum += n;
                    has_numbers = true;
                }
            }
        }
        Value::ColumnReference { table, column_name } => {
            crate::vm::vm::with_current_stores(|store, heap| {
                let t = table.borrow();
                for i in 0..t.len() {
                    if let Some(item) = crate::vm::table_ops::get_cell_value(&*t, i, column_name, store, heap) {
                        if let Value::Number(n) = item {
                            sum += n;
                            has_numbers = true;
                        }
                    }
                }
            });
        }
        _ => {}
    }
    if has_numbers { Value::Number(sum) } else { Value::Number(0.0) }
}

pub fn native_average(args: &[Value]) -> Value {
    if args.is_empty() {
        return Value::Number(0.0);
    }
    if let Value::Tensor(tensor) = &args[0] {
        let tensor_ref = tensor.borrow();
        return Value::Number(tensor_ref.mean() as f64);
    }
    let (mut sum, mut count) = (0.0, 0);
    match &args[0] {
        Value::Array(a) => {
            for item in a.borrow().iter() {
                if let Value::Number(n) = item {
                    sum += n;
                    count += 1;
                }
            }
        }
        Value::ColumnReference { table, column_name } => {
            crate::vm::vm::with_current_stores(|store, heap| {
                let t = table.borrow();
                for i in 0..t.len() {
                    if let Some(item) = crate::vm::table_ops::get_cell_value(&*t, i, column_name, store, heap) {
                        if let Value::Number(n) = item {
                            sum += n;
                            count += 1;
                        }
                    }
                }
            });
        }
        _ => {}
    }
    if count > 0 { Value::Number(sum / count as f64) } else { Value::Number(0.0) }
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
    
    // Handle array and column reference (lazy: no materialization)
    match &args[0] {
        Value::Array(arr) => Value::Number(arr.borrow().len() as f64),
        Value::ColumnReference { table, column_name } => {
            crate::vm::vm::with_current_stores(|_store, _heap| {
                let t = table.borrow();
                crate::vm::table_ops::column_len(&*t, column_name)
                    .map(|len| Value::Number(len as f64))
                    .unwrap_or(Value::Number(0.0))
            })
        }
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

