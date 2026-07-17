//! Validator factories for `Column(validators=[...])` and runtime checks on INSERT.
//!
//! On insert, per column: **validators** run on the raw cell value first, then **transform**
//! (e.g. password hash), then the row is written — so rules like `password_policy` apply to plaintext.

use crate::common::value::Value;
use crate::vm::natives::utils::invoke_value_callable;
use crate::websocket::set_native_error;
use regex::Regex;
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::OnceLock;

const TAG: &str = "__validator";

fn get_bool(v: &Value, default: bool) -> bool {
    match v {
        Value::Bool(b) => *b,
        _ => default,
    }
}

fn make_desc(kind: &str, mut fields: HashMap<String, Value>) -> Value {
    fields.insert(TAG.to_string(), Value::String(kind.to_string()));
    Value::legacy_object(fields)
}

fn usize_arg(v: &Value, _name: &str) -> Option<usize> {
    match v {
        Value::Number(x) if *x >= 0.0 && x.is_finite() && x.fract() == 0.0 => Some(*x as usize),
        _ => None,
    }
}

// --- Factories (Extended natives) ---

pub fn native_validator_min_length(args: &[Value]) -> Value {
    let Some(n) = args.first().and_then(|v| usize_arg(v, "n")) else {
        set_native_error("validators.min_length: expected non-negative integer".to_string());
        return Value::Null;
    };
    let mut m = HashMap::new();
    m.insert("n".to_string(), Value::Number(n as f64));
    make_desc("min_length", m)
}

pub fn native_validator_max_length(args: &[Value]) -> Value {
    let Some(n) = args.first().and_then(|v| usize_arg(v, "n")) else {
        set_native_error("validators.max_length: expected non-negative integer".to_string());
        return Value::Null;
    };
    let mut m = HashMap::new();
    m.insert("n".to_string(), Value::Number(n as f64));
    make_desc("max_length", m)
}

pub fn native_validator_length_between(args: &[Value]) -> Value {
    if args.len() < 2 {
        set_native_error("validators.length_between: expected (min, max)".to_string());
        return Value::Null;
    }
    let Some(lo) = usize_arg(&args[0], "min") else {
        set_native_error("validators.length_between: min must be non-negative integer".to_string());
        return Value::Null;
    };
    let Some(hi) = usize_arg(&args[1], "max") else {
        set_native_error("validators.length_between: max must be non-negative integer".to_string());
        return Value::Null;
    };
    if lo > hi {
        set_native_error("validators.length_between: min must be <= max".to_string());
        return Value::Null;
    }
    let mut m = HashMap::new();
    m.insert("min".to_string(), Value::Number(lo as f64));
    m.insert("max".to_string(), Value::Number(hi as f64));
    make_desc("length_between", m)
}

pub fn native_validator_regex(args: &[Value]) -> Value {
    let Some(Value::String(pattern)) = args.first() else {
        set_native_error("validators.regex: expected pattern string".to_string());
        return Value::Null;
    };
    if Regex::new(pattern).is_err() {
        set_native_error(format!("validators.regex: invalid pattern: {}", pattern));
        return Value::Null;
    }
    let mut m = HashMap::new();
    m.insert("pattern".to_string(), Value::String(pattern.clone()));
    make_desc("regex", m)
}

fn email_re() -> &'static Regex {
    static EMAIL: OnceLock<Regex> = OnceLock::new();
    EMAIL.get_or_init(|| {
        Regex::new(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$").unwrap()
    })
}

fn url_re() -> &'static Regex {
    static URL: OnceLock<Regex> = OnceLock::new();
    URL.get_or_init(|| Regex::new(r"^https?://[^\s]+$").unwrap())
}

fn username_re() -> &'static Regex {
    static USERNAME: OnceLock<Regex> = OnceLock::new();
    USERNAME.get_or_init(|| Regex::new(r"^[a-zA-Z0-9_]{3,32}$").unwrap())
}

pub fn native_validator_email(_args: &[Value]) -> Value {
    let _ = email_re();
    make_desc("email", HashMap::new())
}

pub fn native_validator_url(_args: &[Value]) -> Value {
    let _ = url_re();
    make_desc("url", HashMap::new())
}

pub fn native_validator_username(_args: &[Value]) -> Value {
    let _ = username_re();
    make_desc("username", HashMap::new())
}

pub fn native_validator_password_policy(args: &[Value]) -> Value {
    let (min_len, uppercase, digits, special) = match args.first() {
        Some(Value::Object(rc)) => {
            let o = rc.borrow();
            let ml = o
                .str_key_get("min_length")
                .and_then(|v| usize_arg(v, "min_length"))
                .unwrap_or(8);
            let uppercase = o
                .str_key_get("uppercase")
                .map(|v| get_bool(v, true))
                .unwrap_or(true);
            let digits = o.str_key_get("digits").map(|v| get_bool(v, true)).unwrap_or(true);
            let special = o.str_key_get("special").map(|v| get_bool(v, true)).unwrap_or(true);
            (ml, uppercase, digits, special)
        }
        Some(_) | None => (8, true, true, true),
    };
    let mut m = HashMap::new();
    m.insert("min_length".to_string(), Value::Number(min_len as f64));
    m.insert("uppercase".to_string(), Value::Bool(uppercase));
    m.insert("digits".to_string(), Value::Bool(digits));
    m.insert("special".to_string(), Value::Bool(special));
    make_desc("password_policy", m)
}

pub fn native_validator_one_of(args: &[Value]) -> Value {
    let Some(Value::Array(arr)) = args.first() else {
        set_native_error("validators.one_of: expected array of allowed values".to_string());
        return Value::Null;
    };
    let choices: Vec<Value> = arr.borrow().clone();
    let mut m = HashMap::new();
    m.insert(
        "choices".to_string(),
        Value::Array(Rc::new(RefCell::new(choices))),
    );
    make_desc("one_of", m)
}

pub fn native_validator_min_value(args: &[Value]) -> Value {
    let Some(Value::Number(n)) = args.first() else {
        set_native_error("validators.min_value: expected number".to_string());
        return Value::Null;
    };
    if !n.is_finite() {
        set_native_error("validators.min_value: number must be finite".to_string());
        return Value::Null;
    }
    let mut m = HashMap::new();
    m.insert("n".to_string(), Value::Number(*n));
    make_desc("min_value", m)
}

pub fn native_validator_max_value(args: &[Value]) -> Value {
    let Some(Value::Number(n)) = args.first() else {
        set_native_error("validators.max_value: expected number".to_string());
        return Value::Null;
    };
    if !n.is_finite() {
        set_native_error("validators.max_value: number must be finite".to_string());
        return Value::Null;
    }
    let mut m = HashMap::new();
    m.insert("n".to_string(), Value::Number(*n));
    make_desc("max_value", m)
}

pub fn native_validator_range_value(args: &[Value]) -> Value {
    if args.len() < 2 {
        set_native_error("validators.range_value: expected (min, max)".to_string());
        return Value::Null;
    }
    let Some(Value::Number(lo)) = args.first() else {
        set_native_error("validators.range_value: min must be number".to_string());
        return Value::Null;
    };
    let Some(Value::Number(hi)) = args.get(1) else {
        set_native_error("validators.range_value: max must be number".to_string());
        return Value::Null;
    };
    if !lo.is_finite() || !hi.is_finite() {
        set_native_error("validators.range_value: bounds must be finite".to_string());
        return Value::Null;
    }
    if lo > hi {
        set_native_error("validators.range_value: min must be <= max".to_string());
        return Value::Null;
    }
    let mut m = HashMap::new();
    m.insert("min".to_string(), Value::Number(*lo));
    m.insert("max".to_string(), Value::Number(*hi));
    make_desc("range_value", m)
}

pub fn native_validator_custom(args: &[Value]) -> Value {
    let Some(callee) = args.first() else {
        set_native_error("validators.custom: expected callable".to_string());
        return Value::Null;
    };
    if !matches!(
        callee,
        Value::NativeFunction(_)
            | Value::Function(_)
            | Value::ModuleFunction { .. }
    ) {
        set_native_error("validators.custom: expected callable".to_string());
        return Value::Null;
    }
    let mut m = HashMap::new();
    m.insert("callable".to_string(), callee.clone());
    make_desc("custom", m)
}

// --- Runtime ---

pub fn run_column_validators(
    validators_field: &Value,
    value: &Value,
    column_name: &str,
) -> Result<(), String> {
    match validators_field {
        Value::Null => Ok(()),
        Value::Array(arr) => {
            for d in arr.borrow().iter() {
                validate_one_descriptor(d, value, column_name)?;
            }
            Ok(())
        }
        _ => Err(format!(
            "column '{}': validators must be an array or null",
            column_name
        )),
    }
}

fn validate_one_descriptor(desc: &Value, value: &Value, column_name: &str) -> Result<(), String> {
    let Value::Object(rc) = desc else {
        return Err(format!(
            "column '{}': validator descriptor must be object",
            column_name
        ));
    };
    let obj = rc.borrow();
    let Some(Value::String(kind)) = obj.str_key_get(TAG) else {
        return Err(format!(
            "column '{}': validator missing __validator tag",
            column_name
        ));
    };
    let kind = kind.as_str();
    let res = match kind {
        "min_length" => {
            let s = require_str(value, column_name, "min_length")?;
            let n = obj
                .str_key_get("n")
                .and_then(|v| usize_arg(v, "n"))
                .ok_or_else(|| format!("column '{}': invalid min_length descriptor", column_name))?;
            if s.chars().count() >= n {
                Ok(())
            } else {
                Err(format!(
                    "column '{}': string shorter than min_length {}",
                    column_name, n
                ))
            }
        }
        "max_length" => {
            let s = require_str(value, column_name, "max_length")?;
            let n = obj
                .str_key_get("n")
                .and_then(|v| usize_arg(v, "n"))
                .ok_or_else(|| format!("column '{}': invalid max_length descriptor", column_name))?;
            if s.chars().count() <= n {
                Ok(())
            } else {
                Err(format!(
                    "column '{}': string longer than max_length {}",
                    column_name, n
                ))
            }
        }
        "length_between" => {
            let s = require_str(value, column_name, "length_between")?;
            let lo = obj
                .str_key_get("min")
                .and_then(|v| usize_arg(v, "min"))
                .ok_or_else(|| format!("column '{}': invalid length_between", column_name))?;
            let hi = obj
                .str_key_get("max")
                .and_then(|v| usize_arg(v, "max"))
                .ok_or_else(|| format!("column '{}': invalid length_between", column_name))?;
            let len = s.chars().count();
            if (lo..=hi).contains(&len) {
                Ok(())
            } else {
                Err(format!(
                    "column '{}': length {} not in [{}, {}]",
                    column_name, len, lo, hi
                ))
            }
        }
        "regex" => {
            let s = require_str(value, column_name, "regex")?;
            let pattern = obj
                .str_key_get("pattern")
                .and_then(|v| {
                    if let Value::String(p) = v {
                        Some(p.as_str())
                    } else {
                        None
                    }
                })
                .ok_or_else(|| format!("column '{}': regex descriptor missing pattern", column_name))?;
            let re = Regex::new(pattern).map_err(|e| {
                format!(
                    "column '{}': internal regex error: {}",
                    column_name, e
                )
            })?;
            if re.is_match(&s) {
                Ok(())
            } else {
                Err(format!(
                    "column '{}': value does not match regex",
                    column_name
                ))
            }
        }
        "email" => {
            let s = require_str(value, column_name, "email")?;
            if email_re().is_match(&s) {
                Ok(())
            } else {
                Err(format!("column '{}': invalid email format", column_name))
            }
        }
        "url" => {
            let s = require_str(value, column_name, "url")?;
            if url_re().is_match(&s) {
                Ok(())
            } else {
                Err(format!("column '{}': invalid url format", column_name))
            }
        }
        "username" => {
            let s = require_str(value, column_name, "username")?;
            if username_re().is_match(&s) {
                Ok(())
            } else {
                Err(format!(
                    "column '{}': invalid username format",
                    column_name
                ))
            }
        }
        "password_policy" => {
            let s = require_str(value, column_name, "password_policy")?;
            let min_len = obj
                .str_key_get("min_length")
                .and_then(|v| usize_arg(v, "min_length"))
                .unwrap_or(8);
            let need_upper = obj
                .str_key_get("uppercase")
                .map(|v| get_bool(v, true))
                .unwrap_or(true);
            let need_digits = obj.str_key_get("digits").map(|v| get_bool(v, true)).unwrap_or(true);
            let need_special = obj.str_key_get("special").map(|v| get_bool(v, true)).unwrap_or(true);
            if s.chars().count() < min_len {
                return Err(format!(
                    "column '{}': password shorter than {}",
                    column_name, min_len
                ));
            }
            if need_upper && !s.chars().any(|c| c.is_ascii_uppercase()) {
                return Err(format!(
                    "column '{}': password must contain uppercase letter",
                    column_name
                ));
            }
            if need_digits && !s.chars().any(|c| c.is_ascii_digit()) {
                return Err(format!(
                    "column '{}': password must contain a digit",
                    column_name
                ));
            }
            if need_special && !s.chars().any(|c| !c.is_ascii_alphanumeric()) {
                return Err(format!(
                    "column '{}': password must contain a non-alphanumeric character",
                    column_name
                ));
            }
            Ok(())
        }
        "one_of" => {
            let Some(Value::Array(choices_rc)) = obj.str_key_get("choices") else {
                return Err(format!(
                    "column '{}': one_of descriptor missing choices",
                    column_name
                ));
            };
            let choices = choices_rc.borrow();
            if choices.iter().any(|c| c == value) {
                Ok(())
            } else {
                Err(format!(
                    "column '{}': value not in allowed set",
                    column_name
                ))
            }
        }
        "min_value" => {
            let n = require_num(value, column_name, "min_value")?;
            let bound = obj
                .str_key_get("n")
                .and_then(|v| {
                    if let Value::Number(x) = v {
                        Some(*x)
                    } else {
                        None
                    }
                })
                .ok_or_else(|| format!("column '{}': invalid min_value", column_name))?;
            if n >= bound {
                Ok(())
            } else {
                Err(format!(
                    "column '{}': value {} below min {}",
                    column_name, n, bound
                ))
            }
        }
        "max_value" => {
            let n = require_num(value, column_name, "max_value")?;
            let bound = obj
                .str_key_get("n")
                .and_then(|v| {
                    if let Value::Number(x) = v {
                        Some(*x)
                    } else {
                        None
                    }
                })
                .ok_or_else(|| format!("column '{}': invalid max_value", column_name))?;
            if n <= bound {
                Ok(())
            } else {
                Err(format!(
                    "column '{}': value {} above max {}",
                    column_name, n, bound
                ))
            }
        }
        "range_value" => {
            let n = require_num(value, column_name, "range_value")?;
            let lo = obj
                .str_key_get("min")
                .and_then(|v| {
                    if let Value::Number(x) = v {
                        Some(*x)
                    } else {
                        None
                    }
                })
                .ok_or_else(|| format!("column '{}': invalid range_value", column_name))?;
            let hi = obj
                .str_key_get("max")
                .and_then(|v| {
                    if let Value::Number(x) = v {
                        Some(*x)
                    } else {
                        None
                    }
                })
                .ok_or_else(|| format!("column '{}': invalid range_value", column_name))?;
            if n >= lo && n <= hi {
                Ok(())
            } else {
                Err(format!(
                    "column '{}': value {} not in [{}, {}]",
                    column_name, n, lo, hi
                ))
            }
        }
        "custom" => {
            let Some(callable) = obj.str_key_get("callable") else {
                return Err(format!(
                    "column '{}': custom validator missing callable",
                    column_name
                ));
            };
            let args = [value.clone()];
            match invoke_value_callable(callable, &args) {
                Ok(Value::Bool(true)) => Ok(()),
                Ok(Value::Bool(false)) => Err(format!(
                    "column '{}': custom validator returned false",
                    column_name
                )),
                Ok(_) => Err(format!(
                    "column '{}': custom validator must return bool",
                    column_name
                )),
                Err(e) => Err(format!("column '{}': custom validator error: {}", column_name, e)),
            }
        }
        _ => Err(format!(
            "column '{}': unknown validator kind '{}'",
            column_name, kind
        )),
    };
    drop(obj);
    res
}

fn require_str<'a>(value: &'a Value, column: &str, vkind: &str) -> Result<&'a str, String> {
    match value {
        Value::String(s) => Ok(s.as_str()),
        _ => Err(format!(
            "column '{}': {} validator requires string value",
            column, vkind
        )),
    }
}

fn require_num(value: &Value, column: &str, vkind: &str) -> Result<f64, String> {
    match value {
        Value::Number(n) if n.is_finite() => Ok(*n),
        _ => Err(format!(
            "column '{}': {} validator requires number value",
            column, vkind
        )),
    }
}
