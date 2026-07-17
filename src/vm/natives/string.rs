// String manipulation native functions

use crate::common::numeric::IntValue;
use crate::common::value::Value;
use crate::websocket::set_native_error;
use std::cell::RefCell;
use std::rc::Rc;

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

    let parts: Vec<Value> = s
        .split(delim)
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
    let parts: Vec<String> = arr_ref.iter().map(|v| v.to_string()).collect();

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

pub fn native_starts_with(args: &[Value]) -> Value {
    if args.len() < 2 {
        return Value::Bool(false);
    }

    let s = match &args[0] {
        Value::String(str) => str,
        _ => return Value::Bool(false),
    };

    let prefix = match &args[1] {
        Value::String(p) => p,
        _ => return Value::Bool(false),
    };

    Value::Bool(s.starts_with(prefix.as_str()))
}

pub fn native_ends_with(args: &[Value]) -> Value {
    if args.len() < 2 {
        return Value::Bool(false);
    }

    let s = match &args[0] {
        Value::String(str) => str,
        _ => return Value::Bool(false),
    };

    let suffix = match &args[1] {
        Value::String(suf) => suf,
        _ => return Value::Bool(false),
    };

    Value::Bool(s.ends_with(suffix.as_str()))
}

pub fn native_isupper(args: &[Value]) -> Value {
    if args.is_empty() {
        return Value::Bool(false);
    }
    match &args[0] {
        Value::String(s) => {
            Value::Bool(s.chars().next().map(|c| c.is_uppercase()).unwrap_or(false))
        }
        _ => Value::Bool(false),
    }
}
pub fn native_islower(args: &[Value]) -> Value {
    if args.is_empty() {
        return Value::Bool(false);
    }
    match &args[0] {
        Value::String(s) => {
            Value::Bool(s.chars().next().map(|c| c.is_lowercase()).unwrap_or(false))
        }
        _ => Value::Bool(false),
    }
}

/// `replace(str, find, replacement)` — replace all occurrences of `find` (Python `str.replace`).
pub fn native_replace(args: &[Value]) -> Value {
    if args.len() < 3 {
        return Value::Null;
    }
    match (&args[0], &args[1], &args[2]) {
        (Value::String(s), Value::String(find), Value::String(repl)) => {
            Value::String(s.replace(find.as_str(), repl.as_str()))
        }
        _ => Value::Null,
    }
}

/// `capitalize(str)` — first character uppercased, rest lowercased (Python `str.capitalize`).
pub fn native_capitalize(args: &[Value]) -> Value {
    if args.is_empty() {
        return Value::String(String::new());
    }
    match &args[0] {
        Value::String(s) => {
            let mut chars = s.chars();
            match chars.next() {
                None => Value::String(String::new()),
                Some(first) => {
                    let rest = chars.as_str().to_lowercase();
                    Value::String(format!("{}{}", first.to_uppercase(), rest))
                }
            }
        }
        _ => Value::Null,
    }
}

/// `ord(ch)` — Unicode code point of a single-character string (Python-compatible).
pub fn native_ord(args: &[Value]) -> Value {
    if args.len() != 1 {
        set_native_error(format!(
            "TypeError: ord() takes exactly one argument ({} given)",
            args.len()
        ));
        return Value::Null;
    }

    let Value::String(s) = &args[0] else {
        set_native_error("RuntimeError: ord() argument must be a string".to_string());
        return Value::Null;
    };

    let mut chars = s.chars();
    match (chars.next(), chars.next()) {
        (None, _) => {
            set_native_error(
                "TypeError: ord() expected a string of length 1, but string of length 0 found"
                    .to_string(),
            );
            Value::Null
        }
        (Some(c), None) => Value::Int(IntValue::Finite(c as u32 as i64)),
        (Some(_), Some(_)) => {
            let n = s.chars().count();
            set_native_error(format!(
                "TypeError: ord() expected a string of length 1, but string of length {} found",
                n
            ));
            Value::Null
        }
    }
}
