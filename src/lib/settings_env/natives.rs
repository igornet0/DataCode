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

    // Prefer model_config.env_file when present (String or Path); otherwise use path argument.
    let env_file_from_config = if args.len() >= 3 {
        if let Value::Object(config_rc) = &args[2] {
            config_rc.borrow().get("env_file").and_then(|v| match v {
                Value::String(s) => Some(s.clone()),
                Value::Path(p) => Some(p.to_string_lossy().to_string()),
                _ => None,
            })
        } else {
            if crate::common::debug::verbose_constructor_debug() {
                let ty = match &args[2] {
                    Value::Null => "Null",
                    Value::Bool(_) => "Bool",
                    Value::Number(_) => "Number",
                    Value::String(_) => "String",
                    Value::Object(_) => "Object",
                    Value::Array(_) => "Array",
                    Value::Function(_) => "Function",
                    _ => "Other",
                };
                eprintln!("[settings_env load_env] args[2] (model_config) is {} (path_str={:?})", ty, path_str);
            }
            None
        }
    } else {
        None
    };
    let effective_path = env_file_from_config.unwrap_or_else(|| path_str.clone());

    let (env_prefix, case_sensitive, _extra) = if args.len() >= 3 {
        if let Value::Object(config_rc) = &args[2] {
            let config = config_rc.borrow();
            let prefix = get_config_str(&config, "env_prefix", "");
            let case_sens = get_config_bool(&config, "case_sensitive", false);
            let ext = get_config_str(&config, "extra", "ignore");
            (prefix, case_sens, ext)
        } else {
            (String::new(), false, "ignore".to_string())
        }
    } else {
        (String::new(), false, "ignore".to_string())
    };

    // When path and model_config.env_file are both empty we do not search for files; return empty env.
    // For env selection, Settings subclasses must set env_file in model_config.
    if effective_path.is_empty() {
        if crate::common::debug::verbose_constructor_debug() && args.len() >= 3 {
            if let Value::Object(config_rc) = &args[2] {
                let has_env_file = config_rc.borrow().contains_key("env_file");
                if !has_env_file {
                    eprintln!("[settings_env load_env] model_config has no env_file; path_str is empty; returning empty env");
                }
            }
        }
        return Value::Object(Rc::new(RefCell::new(HashMap::new())));
    }

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
        // When env_prefix was empty (e.g. fallback path), map may have "db__url" but required_keys ["url"]; derive missing keys from prefixed keys.
        if env_prefix.is_empty() {
            if !required_keys.is_empty() {
                let to_add: Vec<(String, Value)> = required_keys
                    .iter()
                    .filter(|key| !map.contains_key(*key))
                    .filter_map(|key| {
                        let suffix = format!("__{}", key);
                        map.iter()
                            .find(|(k, _)| k.ends_with(&suffix) || k.as_str() == key)
                            .map(|(_, v)| (key.clone(), v.clone()))
                    })
                    .collect();
                for (k, v) in to_add {
                    map.insert(k, v);
                }
            }
        }
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

    // Optional 4th argument: nested_specs — build nested Settings objects from prefixed keys and remove them from top-level.
    // Each spec: Object with "field" (str), "env_prefix" (str), "class_name" (str). Uses env_nested_delimiter from model_config.
    // Treat Null (e.g. parent class has no __nested_specs) as empty array.
    if args.len() >= 4 {
        let specs_rc = match &args[3] {
            Value::Array(a) => a.clone(),
            _ => Rc::new(RefCell::new(Vec::new())),
        };
        {
            let specs = specs_rc.borrow();
            let _delimiter = if args.len() >= 3 {
                if let Value::Object(cfg_rc) = &args[2] {
                    get_config_str(&cfg_rc.borrow(), "env_nested_delimiter", "__")
                } else {
                    "__".to_string()
                }
            } else {
                "__".to_string()
            };
            for spec_val in specs.iter() {
                if let Value::Object(spec_rc) = spec_val {
                    let spec = spec_rc.borrow();
                    let field_name = get_config_str(&spec, "field", "");
                    let nested_prefix = get_config_str(&spec, "env_prefix", "");
                    let class_name = get_config_str(&spec, "class_name", "");
                    if field_name.is_empty() || class_name.is_empty() {
                        continue;
                    }
                    let prefix_lower = nested_prefix.to_lowercase();
                    let mut nested_map = HashMap::new();
                    let keys_to_remove: Vec<String> = map
                        .keys()
                        .filter(|k| !prefix_lower.is_empty() && k.starts_with(&prefix_lower))
                        .cloned()
                        .collect();
                    for k in &keys_to_remove {
                        if let Some(v) = map.get(k) {
                            let inner_key = if prefix_lower.is_empty() {
                                k.clone()
                            } else {
                                k[prefix_lower.len()..].to_lowercase()
                            };
                            nested_map.insert(inner_key, v.clone());
                        }
                    }
                    for k in keys_to_remove {
                        map.remove(&k);
                    }
                    nested_map.insert("__class_name".to_string(), Value::String(class_name.clone()));
                    // If no env keys had the nested prefix, nested_map only has __class_name. Insert Null so the parent constructor's default_factory runs and builds the nested Settings (e.g. Config.db = DatabaseConfig(path, ...)) with __constructing_class__ set.
                    let nested_value = if nested_map.len() == 1 {
                        Value::Null
                    } else {
                        Value::Object(Rc::new(RefCell::new(nested_map)))
                    };
                    map.insert(field_name.to_string(), nested_value);
                }
            }
        }
    }

    Value::Object(Rc::new(RefCell::new(map)))
}

/// config(env_prefix?, extra?, env_file?, env_file_encoding?, case_sensitive?, env_nested_delimiter?) -> Object (SettingsConfigDict)
/// Returns an object with keys: env_prefix, extra, env_file, env_file_encoding, case_sensitive, env_nested_delimiter.
/// Args in order (named args compiled as positional): env_prefix (str), extra (str), env_file (str/null), env_file_encoding (str), case_sensitive (bool), env_nested_delimiter (str).
/// When called with first argument an Object (e.g. Config(**obj), possibly followed by Nulls), merges that object's keys with defaults.
pub fn native_settings_env_config(args: &[Value]) -> Value {
    if let Some(Value::Object(config_rc)) = args.get(0) {
        let config = config_rc.borrow();
        let mut map = HashMap::new();
        map.insert(
            "env_prefix".to_string(),
            Value::String(get_config_str(&config, "env_prefix", "")),
        );
        map.insert(
            "extra".to_string(),
            Value::String(get_config_str(&config, "extra", "ignore")),
        );
        map.insert(
            "env_file".to_string(),
            config.get("env_file").cloned().unwrap_or(Value::Null),
        );
        map.insert(
            "env_file_encoding".to_string(),
            Value::String(get_config_str(&config, "env_file_encoding", "utf-8")),
        );
        map.insert(
            "case_sensitive".to_string(),
            Value::Bool(get_config_bool(&config, "case_sensitive", false)),
        );
        map.insert(
            "env_nested_delimiter".to_string(),
            Value::String(get_config_str(&config, "env_nested_delimiter", "__")),
        );
        return Value::Object(Rc::new(RefCell::new(map)));
    }
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

/// Settings() -> Object (file from model_config) or Settings(path: str) -> Object (use path as env file).
/// Overload: no args — env file from __constructing_class__["model_config"] (compiler expands to pass it); if Null, empty env.
/// One arg str — path to .env; one arg Object (model_config) — use env_file from it.
pub fn native_settings_env_settings(args: &[Value]) -> Value {
    if args.is_empty() {
        return Value::Object(Rc::new(RefCell::new(HashMap::new())));
    }
    if args.len() == 1 {
        if matches!(&args[0], Value::Null) {
            return Value::Object(Rc::new(RefCell::new(HashMap::new())));
        }
        if let Value::Object(config_rc) = &args[0] {
            let config = config_rc.borrow();
            if config.contains_key("env_file") || config.contains_key("env_prefix") || config.contains_key("extra") {
                let path = get_config_str(&config, "env_file", "");
                let required = Value::Array(Rc::new(RefCell::new(vec![])));
                return native_settings_env_load_env(&[Value::String(path), required, args[0].clone()]);
            }
        }
    }
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
