// String manipulation native functions

use crate::common::value::Value;
use std::rc::Rc;
use std::cell::RefCell;

pub fn native_upper(args: &[Value]) -> Value {
    if args.is_empty() {
        return Value::String(String::new());
    }
    
    match &args[0] {
        Value::String(s) => Value::String(s.to_uppercase()),
        _ => Value::Null,
    }
}

pub fn native_lower(args: &[Value]) -> Value {
    if args.is_empty() {
        return Value::String(String::new());
    }
    
    match &args[0] {
        Value::String(s) => Value::String(s.to_lowercase()),
        _ => Value::Null,
    }
}

pub fn native_trim(args: &[Value]) -> Value {
    if args.is_empty() {
        return Value::String(String::new());
    }
    
    match &args[0] {
        Value::String(s) => Value::String(s.trim().to_string()),
        _ => Value::Null,
    }
}

pub fn native_split(args: &[Value]) -> Value {
    if args.len() < 2 {
        return Value::Null;
    }
    
    let s = match &args[0] {
        Value::String(str) => str,
        _ => return Value::Null,
    };
    
    let delim = match &args[1] {
        Value::String(d) => d,
        _ => return Value::Null,
    };
    
    let parts: Vec<Value> = s.split(delim)
        .map(|part| Value::String(part.to_string()))
        .collect();
    
    Value::Array(Rc::new(RefCell::new(parts)))
}

pub fn native_join(args: &[Value]) -> Value {
    // Универсальная функция join: проверяем тип первого аргумента
    if args.is_empty() {
        return Value::Null;
    }
    
    // Если первый аргумент - таблица, это table join
    if matches!(&args[0], Value::Table(_)) && args.len() >= 3 {
        use super::join::native_table_join;
        return native_table_join(args);
    }
    
    // Иначе это array join (для обратной совместимости)
    if args.len() < 2 {
        return Value::Null;
    }
    
    let arr = match &args[0] {
        Value::Array(a) => a,
        _ => return Value::Null,
    };
    
    let delim = match &args[1] {
        Value::String(d) => d,
        _ => return Value::Null,
    };
    
    let arr_ref = arr.borrow();
    let parts: Vec<String> = arr_ref.iter()
        .map(|v| v.to_string())
        .collect();
    
    Value::String(parts.join(delim))
}

pub fn native_contains(args: &[Value]) -> Value {
    if args.len() < 2 {
        return Value::Bool(false);
    }
    
    let s = match &args[0] {
        Value::String(str) => str,
        _ => return Value::Bool(false),
    };
    
    let substr = match &args[1] {
        Value::String(sub) => sub,
        _ => return Value::Bool(false),
    };
    
    Value::Bool(s.contains(substr))
}

