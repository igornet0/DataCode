// Native functions for settings_env module

use crate::common::value::Value;
use std::collections::HashMap;
use std::path::Path;
use std::rc::Rc;
use std::cell::RefCell;
use regex::Regex;

/// Parse a single .env line; returns (key_lowercase, value) or None if line should be skipped.
fn parse_env_line(line: &str) -> Option<(String, String)> {
    let line = line.trim();
    if line.is_empty() || line.starts_with('#') {
        return None;
    }
    let Some(eq_pos) = line.find('=') else {
        return None;
    };
    let key = line[..eq_pos].trim().to_string();
    let value = line[eq_pos + 1..].trim_start();
    // Remove optional surrounding quotes from value
    let value = value.trim_matches(|c| c == '"' || c == '\'');
    if key.is_empty() {
        return None;
    }
    Some((key.to_lowercase(), value.to_string()))
}

/// Coerce string value to Value: "true"/"false" -> Bool, numeric -> Number, else String.
fn coerce_value(s: &str) -> Value {
    let lower = s.to_lowercase();
    if lower == "true" {
        return Value::Bool(true);
    }
    if lower == "false" {
        return Value::Bool(false);
    }
    if let Ok(n) = s.parse::<f64>() {
        return Value::Number(n);
    }
    Value::String(s.to_string())
}

/// Resolve path: if relative, resolve against base_path from file_import; else use as-is.
fn resolve_env_path(path_str: &str) -> std::path::PathBuf {
    let path = Path::new(path_str);
    if path.is_absolute() {
        return path.to_path_buf();
    }
    if let Some(base) = crate::vm::file_import::get_base_path() {
        return base.join(path);
    }
    path.to_path_buf()
}

/// Helper: get string from model_config Object, or default.
fn get_config_str(map: &HashMap<String, Value>, key: &str, default: &str) -> String {
    map.get(key)
        .and_then(|v| {
            if let Value::String(s) = v {
                Some(s.clone())
            } else {
                None
            }
        })
        .unwrap_or_else(|| default.to_string())
}

/// Helper: get bool from model_config Object, or default.
fn get_config_bool(map: &HashMap<String, Value>, key: &str, default: bool) -> bool {
    map.get(key)
        .and_then(|v| {
            if let Value::Bool(b) = v {
                Some(*b)
            } else {
                None
            }
        })
        .unwrap_or(default)
}

/// load_env(path: str [, required_keys: array [, model_config: Object]]) -> Object
/// Reads .env file, returns object with keys (lowercase unless case_sensitive) and coerced values.
/// If required_keys is passed, each key must be present; otherwise error.
/// If model_config (SettingsConfigDict) is passed: env_prefix filters keys, env_file overrides path,
/// extra, case_sensitive, env_nested_delimiter are applied.
pub fn native_settings_env_load_env(args: &[Value]) -> Value {
    if args.is_empty() {
        return Value::Null;
    }
    let path_str = match &args[0] {
        Value::String(s) => s.clone(),
        Value::Path(p) => p.to_string_lossy().to_string(),
        _ => return Value::Null,
    };

    let (effective_path, env_prefix, case_sensitive, _extra) = if args.len() >= 3 {
        if let Value::Object(config_rc) = &args[2] {
            let config = config_rc.borrow();
            let env_file = config.get("env_file").and_then(|v| {
                if let Value::String(s) = v {
                    Some(s.clone())
                } else if matches!(v, Value::Null) {
                    None
                } else {
                    None
                }
            });
            let path = env_file.unwrap_or(path_str);
            let prefix = get_config_str(&config, "env_prefix", "");
            let case_sens = get_config_bool(&config, "case_sensitive", false);
            let ext = get_config_str(&config, "extra", "ignore");
            (path, prefix, case_sens, ext)
        } else {
            (path_str, String::new(), false, "ignore".to_string())
        }
    } else {
        (path_str.clone(), String::new(), false, "ignore".to_string())
    };

    // Relative path with no base path set (e.g. thread-local not set in this thread) would
    // resolve to cwd and cause non-deterministic behaviour. Fail explicitly.
    let path_is_relative = !Path::new(&effective_path).is_absolute();
    if path_is_relative && crate::vm::file_import::get_base_path().is_none() {
        crate::websocket::set_native_error(format!(
            "Relative path '{}' used but base path not set (e.g. run from script file so path is resolved relative to script directory)",
            effective_path
        ));
        return Value::Null;
    }

    let resolved = resolve_env_path(&effective_path);
    crate::debug_println!("[datacode load_env] path = {:?}, resolved = {:?}", effective_path, resolved);
    let content = match std::fs::read_to_string(&resolved) {
        Ok(c) => c,
        Err(_) => {
            // Return Null without setting native error so callers get Null for missing file (no throw)
            return Value::Null;
        }
    };

    let prefix_lower = env_prefix.to_lowercase();
    let mut map = HashMap::new();
    for line in content.lines() {
        if let Some((key, raw_value)) = parse_env_line(line) {
            let key_for_map = if case_sensitive {
                let line_trimmed = line.trim();
                let eq_pos = line_trimmed.find('=').unwrap_or(0);
                line_trimmed[..eq_pos].trim().to_string()
            } else {
                key.clone()
            };
            if !prefix_lower.is_empty() && !key_for_map.starts_with(&prefix_lower) {
                continue;
            }
            let key_final = if !prefix_lower.is_empty() && key_for_map.starts_with(&prefix_lower) {
                key_for_map[prefix_lower.len()..].to_lowercase()
            } else {
                key_for_map.to_lowercase()
            };
            map.insert(key_final, coerce_value(&raw_value));
        }
    }

    if args.len() >= 2 {
        let required_keys = match &args[1] {
            Value::Array(arr_rc) => {
                let arr = arr_rc.borrow();
                let mut keys = Vec::new();
                for v in arr.iter() {
                    if let Value::String(s) = v {
                        keys.push(s.clone());
                    }
                }
                keys
            }
            _ => vec![],
        };
        for key in &required_keys {
            if !map.contains_key(key) {
                crate::websocket::set_native_error(format!("Missing required env variable: {}", key));
                return Value::Null;
            }
        }
        // Опциональная отладка: если model_config с префиксом DB__ вернул только "url", а required_keys больше —
        // возможно в конструктор Config подставился класс DatabaseConfig (ошибка разрешения sentinel).
        if env_prefix == "DB__" && map.len() == 1 && map.contains_key("url") && required_keys.len() > 1 {
            crate::debug_println!(
                "[settings_env load_env] DB__ prefix returned only 'url' but required_keys has {} keys; \
                 if caller was Config constructor, check model_config class resolution (sentinel should map to Config)",
                required_keys.len()
            );
        }
    }

    Value::Object(Rc::new(RefCell::new(map)))
}

/// config(env_prefix?, extra?, env_file?, env_file_encoding?, case_sensitive?, env_nested_delimiter?) -> Object (SettingsConfigDict)
/// Returns an object with keys: env_prefix, extra, env_file, env_file_encoding, case_sensitive, env_nested_delimiter.
/// Args in order (named args compiled as positional): env_prefix (str), extra (str), env_file (str/null), env_file_encoding (str), case_sensitive (bool), env_nested_delimiter (str).
pub fn native_settings_env_config(args: &[Value]) -> Value {
    let env_prefix = args
        .get(0)
        .and_then(|v| if let Value::String(s) = v { Some(s.clone()) } else { None })
        .unwrap_or_default();
    let extra = args
        .get(1)
        .and_then(|v| if let Value::String(s) = v { Some(s.clone()) } else { None })
        .unwrap_or_else(|| "ignore".to_string());
    let env_file = args.get(2).cloned().filter(|v| !matches!(v, Value::Null));
    let env_file_val = env_file.unwrap_or(Value::Null);
    let env_file_encoding = args
        .get(3)
        .and_then(|v| if let Value::String(s) = v { Some(s.clone()) } else { None })
        .unwrap_or_else(|| "utf-8".to_string());
    let case_sensitive = args
        .get(4)
        .and_then(|v| if let Value::Bool(b) = v { Some(*b) } else { None })
        .unwrap_or(false);
    let env_nested_delimiter = args
        .get(5)
        .and_then(|v| if let Value::String(s) = v { Some(s.clone()) } else { None })
        .unwrap_or_else(|| "__".to_string());

    let mut map = HashMap::new();
    map.insert("env_prefix".to_string(), Value::String(env_prefix));
    map.insert("extra".to_string(), Value::String(extra));
    map.insert("env_file".to_string(), env_file_val);
    map.insert("env_file_encoding".to_string(), Value::String(env_file_encoding));
    map.insert("case_sensitive".to_string(), Value::Bool(case_sensitive));
    map.insert("env_nested_delimiter".to_string(), Value::String(env_nested_delimiter));
    Value::Object(Rc::new(RefCell::new(map)))
}

/// Settings(path: str) -> Object
/// Same as load_env(path); allows cfg = Settings("settings/dev.env").
pub fn native_settings_env_settings(args: &[Value]) -> Value {
    native_settings_env_load_env(args)
}

/// Order of parameters for Field; must match get_native_function_params("Field") in compiler/natives.rs.
const FIELD_PARAM_NAMES: &[&str] = &[
    "default",
    "default_factory",
    "alias",
    "title",
    "description",
    "examples",
    "exclude",
    "include",
    "const",
    "gt",
    "ge",
    "lt",
    "le",
    "multiple_of",
    "min_length",
    "max_length",
    "regex",
    "deprecated",
    "repr",
    "json_schema_extra",
    "validate_default",
    "frozen",
];

/// Field(default?, default_factory?, alias?, ...) -> Object (field descriptor)
/// Returns a field descriptor object for use in class field syntax, e.g. env: str = Field(default="dev").
/// Supports full Pydantic-style arguments: default, default_factory, alias, title, description,
/// examples, exclude, include, const, gt, ge, lt, le, multiple_of, min_length, max_length, regex,
/// deprecated, repr, json_schema_extra, validate_default, frozen.
/// Backward compat: single positional arg (non-Object) is treated as default.
pub fn native_settings_env_field(args: &[Value]) -> Value {
    let mut map = HashMap::new();

    // Backward compat: Field(x) or Field(default=x) — single arg as default. Field(...) = required.
    if args.len() == 1 {
        match &args[0] {
            Value::Ellipsis => {
                map.insert("required".to_string(), Value::Bool(true));
                return Value::Object(Rc::new(RefCell::new(map)));
            }
            Value::Object(map_rc) => {
                let obj = map_rc.borrow();
                // If it looks like a descriptor (has known keys), use as full descriptor
                let has_descriptor_keys = obj.keys().any(|k| {
                    FIELD_PARAM_NAMES.contains(&k.as_str())
                });
                if has_descriptor_keys {
                    for (k, v) in obj.iter() {
                        if !matches!(v, Value::Null) {
                            map.insert(k.clone(), v.clone());
                        }
                    }
                    return Value::Object(Rc::new(RefCell::new(map)));
                }
                // Else treat as legacy: object with "default" key -> return just default value for old compat?
                // Plan says: "Field(default=...)" — keep compat. So object with only "default" -> descriptor with default.
                if let Some(v) = obj.get("default") {
                    map.insert("default".to_string(), v.clone());
                }
                return Value::Object(Rc::new(RefCell::new(map)));
            }
            other => {
                // Single non-Object: treat as default (backward compat)
                map.insert("default".to_string(), other.clone());
                return Value::Object(Rc::new(RefCell::new(map)));
            }
        }
    }

    // Full positional args: args[i] corresponds to FIELD_PARAM_NAMES[i]; Null = not provided
    for (i, &name) in FIELD_PARAM_NAMES.iter().enumerate() {
        let val = args.get(i).cloned().unwrap_or(Value::Null);
        if !matches!(val, Value::Null) {
            map.insert(name.to_string(), val);
        }
    }

    // If no default and no default_factory, mark as required (Ellipsis semantics)
    if !map.contains_key("default") && !map.contains_key("default_factory") && !map.is_empty() {
        map.insert("required".to_string(), Value::Bool(true));
    }

    Value::Object(Rc::new(RefCell::new(map)))
}

/// Helper: get optional Number from descriptor map.
fn descriptor_number(map: &HashMap<String, Value>, key: &str) -> Option<f64> {
    map.get(key).and_then(|v| {
        if let Value::Number(n) = v {
            Some(*n)
        } else {
            None
        }
    })
}

/// Helper: get optional String from descriptor map.
fn descriptor_string(map: &HashMap<String, Value>, key: &str) -> Option<String> {
    map.get(key).and_then(|v| {
        if let Value::String(s) = v {
            Some(s.clone())
        } else {
            None
        }
    })
}

/// Validates a value against a Field descriptor (gt, ge, lt, le, multiple_of, min_length, max_length, regex).
/// On failure sets native error via set_native_error and returns None; on success returns Some(value).
/// Call this when applying a field value (e.g. from env or default) before storing in the Settings object.
pub fn apply_field_descriptor(
    field_name: &str,
    descriptor: &HashMap<String, Value>,
    value: Value,
) -> Option<Value> {
    let n = match &value {
        Value::Number(x) => Some(*x),
        _ => None,
    };
    if let Some(n) = n {
        if let Some(gt) = descriptor_number(descriptor, "gt") {
            if n <= gt {
                crate::websocket::set_native_error(format!(
                    "Field '{}': value {} must be > {}",
                    field_name, n, gt
                ));
                return None;
            }
        }
        if let Some(ge) = descriptor_number(descriptor, "ge") {
            if n < ge {
                crate::websocket::set_native_error(format!(
                    "Field '{}': value {} must be >= {}",
                    field_name, n, ge
                ));
                return None;
            }
        }
        if let Some(lt) = descriptor_number(descriptor, "lt") {
            if n >= lt {
                crate::websocket::set_native_error(format!(
                    "Field '{}': value {} must be < {}",
                    field_name, n, lt
                ));
                return None;
            }
        }
        if let Some(le) = descriptor_number(descriptor, "le") {
            if n > le {
                crate::websocket::set_native_error(format!(
                    "Field '{}': value {} must be <= {}",
                    field_name, n, le
                ));
                return None;
            }
        }
        if let Some(mo) = descriptor_number(descriptor, "multiple_of") {
            if mo == 0.0 {
                crate::websocket::set_native_error(format!(
                    "Field '{}': multiple_of must be non-zero",
                    field_name
                ));
                return None;
            }
            let remainder = (n / mo).round().mul_add(mo, -n).abs();
            if remainder > 1e-10 {
                crate::websocket::set_native_error(format!(
                    "Field '{}': value {} must be a multiple of {}",
                    field_name, n, mo
                ));
                return None;
            }
        }
    }

    let len = match &value {
        Value::String(s) => Some(s.len()),
        Value::Array(arr) => Some(arr.borrow().len()),
        Value::Tuple(t) => Some(t.borrow().len()),
        _ => None,
    };
    if let Some(len) = len {
        if let Some(min_len) = descriptor_number(descriptor, "min_length") {
            let min_len = min_len as usize;
            if len < min_len {
                crate::websocket::set_native_error(format!(
                    "Field '{}': length {} must be >= {}",
                    field_name, len, min_len
                ));
                return None;
            }
        }
        if let Some(max_len) = descriptor_number(descriptor, "max_length") {
            let max_len = max_len as usize;
            if len > max_len {
                crate::websocket::set_native_error(format!(
                    "Field '{}': length {} must be <= {}",
                    field_name, len, max_len
                ));
                return None;
            }
        }
    }

    if let Some(pattern) = descriptor_string(descriptor, "regex") {
        if let Value::String(s) = &value {
            if let Ok(re) = Regex::new(&pattern) {
                if !re.is_match(s) {
                    crate::websocket::set_native_error(format!(
                        "Field '{}': value does not match regex '{}'",
                        field_name, pattern
                    ));
                    return None;
                }
            }
        }
    }

    Some(value)
}
