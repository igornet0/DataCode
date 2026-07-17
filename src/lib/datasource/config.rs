//! Parse DataSource config from `Value::Object`.

use crate::common::value::{ObjectKind, Value};
use crate::datasource::error::DataSourceError;
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

#[derive(Debug, Clone)]
pub struct DataSourceConfig {
    pub connector_type: String,
    pub name: Option<String>,
    pub description: Option<String>,
    pub enabled: bool,
    pub url: Option<String>,
    pub host: Option<String>,
    pub port: Option<u16>,
    pub path: Option<String>,
    pub database: Option<String>,
    pub schema: Option<String>,
    pub username: Option<String>,
    pub password: Option<String>,
    pub token: Option<String>,
    pub api_key: Option<String>,
    pub verify_ssl: bool,
    pub timeout: Option<f64>,
    pub connect_timeout: Option<f64>,
    pub read_timeout: Option<f64>,
    pub retry_count: u32,
    pub headers: HashMap<String, String>,
    pub options: HashMap<String, Value>,
    pub raw: HashMap<String, Value>,
}

impl Default for DataSourceConfig {
    fn default() -> Self {
        Self {
            connector_type: String::new(),
            name: None,
            description: None,
            enabled: true,
            url: None,
            host: None,
            port: None,
            path: None,
            database: None,
            schema: None,
            username: None,
            password: None,
            token: None,
            api_key: None,
            verify_ssl: true,
            timeout: None,
            connect_timeout: None,
            read_timeout: None,
            retry_count: 0,
            headers: HashMap::new(),
            options: HashMap::new(),
            raw: HashMap::new(),
        }
    }
}

pub fn object_to_map(obj: &Value) -> Result<HashMap<String, Value>, DataSourceError> {
    let Value::Object(rc) = obj else {
        return Err(DataSourceError::Validation {
            message: "datasource() expects a config object".to_string(),
        });
    };
    let kind = rc.borrow();
    match &*kind {
        ObjectKind::Legacy(m) => Ok(m.clone()),
        ObjectKind::Inline(entries) => {
            let mut out = HashMap::new();
            for (k, v) in entries {
                if let Value::String(sk) = k {
                    out.insert(sk.clone(), v.clone());
                }
            }
            Ok(out)
        }
        ObjectKind::Bucket(_) => Err(DataSourceError::Validation {
            message: "datasource config must be a plain object".to_string(),
        }),
    }
}

fn get_string(map: &HashMap<String, Value>, key: &str) -> Option<String> {
    match map.get(key) {
        Some(Value::String(s)) => Some(s.clone()),
        Some(Value::Path(p)) => Some(p.to_string_lossy().into_owned()),
        _ => None,
    }
}

fn get_bool(map: &HashMap<String, Value>, key: &str, default: bool) -> bool {
    match map.get(key) {
        Some(Value::Bool(b)) => *b,
        _ => default,
    }
}

fn int_to_f64(i: crate::common::numeric::IntValue) -> Option<f64> {
    use crate::common::numeric::IntValue;
    match i {
        IntValue::Finite(n) => Some(n as f64),
        _ => None,
    }
}

fn get_f64(map: &HashMap<String, Value>, key: &str) -> Option<f64> {
    match map.get(key) {
        Some(Value::Number(n)) => Some(*n),
        Some(Value::Int(i)) => int_to_f64(*i),
        _ => None,
    }
}

fn get_u32(map: &HashMap<String, Value>, key: &str, default: u32) -> u32 {
    match map.get(key) {
        Some(Value::Number(n)) if *n >= 0.0 && n.fract() == 0.0 => *n as u32,
        Some(Value::Int(i)) => int_to_f64(*i).map(|n| n as u32).unwrap_or(default),
        _ => default,
    }
}

fn get_u16(map: &HashMap<String, Value>, key: &str) -> Option<u16> {
    let n = get_u32(map, key, 0);
    if map.contains_key(key) {
        Some(n.min(u16::MAX as u32) as u16)
    } else {
        None
    }
}

pub fn headers_from_value(v: &Value) -> HashMap<String, String> {
    let mut out = HashMap::new();
    if let Value::Object(rc) = v {
        let kind = rc.borrow();
        match &*kind {
            ObjectKind::Legacy(m) => {
                for (k, val) in m {
                    if let Value::String(s) = val {
                        out.insert(k.clone(), s.clone());
                    }
                }
            }
            ObjectKind::Inline(entries) => {
                for (k, val) in entries {
                    if let (Value::String(sk), Value::String(sv)) = (k, val) {
                        out.insert(sk.clone(), sv.clone());
                    }
                }
            }
            ObjectKind::Bucket(_) => {}
        }
    }
    out
}

pub fn parse_config(value: &Value) -> Result<DataSourceConfig, DataSourceError> {
    let map = object_to_map(value)?;
    let connector_type = get_string(&map, "type").ok_or_else(|| DataSourceError::Validation {
        message: "datasource config requires 'type' field".to_string(),
    })?;
    let mut cfg = DataSourceConfig {
        connector_type: connector_type.to_lowercase(),
        name: get_string(&map, "name"),
        description: get_string(&map, "description"),
        enabled: get_bool(&map, "enabled", true),
        url: get_string(&map, "url"),
        host: get_string(&map, "host"),
        port: get_u16(&map, "port"),
        path: get_string(&map, "path"),
        database: get_string(&map, "database"),
        schema: get_string(&map, "schema"),
        username: get_string(&map, "username"),
        password: get_string(&map, "password"),
        token: get_string(&map, "token"),
        api_key: get_string(&map, "api_key"),
        verify_ssl: get_bool(&map, "verify_ssl", true),
        timeout: get_f64(&map, "timeout"),
        connect_timeout: get_f64(&map, "connect_timeout"),
        read_timeout: get_f64(&map, "read_timeout"),
        retry_count: get_u32(&map, "retry_count", 0),
        raw: map.clone(),
        ..Default::default()
    };
    if let Some(h) = map.get("headers") {
        cfg.headers = headers_from_value(h);
    }
    if let Some(opts) = map.get("options") {
        cfg.options = object_to_map(opts).unwrap_or_default();
    }
    Ok(cfg)
}

pub fn sqlite_url_from_config(cfg: &DataSourceConfig) -> Result<String, DataSourceError> {
    if let Some(url) = &cfg.url {
        if url.to_lowercase().starts_with("sqlite:") {
            return Ok(url.clone());
        }
    }
    if let Some(db) = &cfg.database {
        if db.starts_with("sqlite:") {
            return Ok(db.clone());
        }
        return Ok(format!("sqlite:///{}", db));
    }
    if let Some(path) = &cfg.path {
        return Ok(format!("sqlite:///{}", path));
    }
    Err(DataSourceError::Validation {
        message: "sqlite datasource requires 'url', 'database', or 'path'".to_string(),
    })
}
