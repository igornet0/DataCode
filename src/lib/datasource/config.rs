//! Parse DataSource config from `Value::Object`.

use crate::common::value::{ObjectKind, Value};
use crate::datasource::error::DataSourceError;
use std::collections::HashMap;

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
    pub collection: Option<String>,
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
            collection: None,
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
    // Nested `connection: { uri, database, collection, ... }` merges into top-level fields.
    let mut merged = map.clone();
    if let Some(conn) = map.get("connection") {
        if let Ok(cm) = object_to_map(conn) {
            for (k, v) in cm {
                merged.entry(k).or_insert(v);
            }
        }
    }
    let mut cfg = DataSourceConfig {
        connector_type: connector_type.to_lowercase(),
        name: get_string(&merged, "name"),
        description: get_string(&merged, "description"),
        enabled: get_bool(&merged, "enabled", true),
        url: get_string(&merged, "url").or_else(|| get_string(&merged, "uri")),
        host: get_string(&merged, "host"),
        port: get_u16(&merged, "port"),
        path: get_string(&merged, "path"),
        database: get_string(&merged, "database"),
        schema: get_string(&merged, "schema"),
        collection: get_string(&merged, "collection"),
        username: get_string(&merged, "username").or_else(|| get_string(&merged, "user")),
        password: get_string(&merged, "password"),
        token: get_string(&merged, "token"),
        api_key: get_string(&merged, "api_key"),
        verify_ssl: get_bool(&merged, "verify_ssl", true),
        timeout: get_f64(&merged, "timeout"),
        connect_timeout: get_f64(&merged, "connect_timeout"),
        read_timeout: get_f64(&merged, "read_timeout"),
        retry_count: get_u32(&merged, "retry_count", 0),
        raw: map.clone(),
        ..Default::default()
    };
    if let Some(h) = merged.get("headers") {
        cfg.headers = headers_from_value(h);
    }
    if let Some(opts) = merged.get("options") {
        cfg.options = object_to_map(opts).unwrap_or_default();
    }
    Ok(cfg)
}

/// Build a SQL/Mongo connection URL from config fields when `url` is absent.
pub fn sql_url_from_config(cfg: &DataSourceConfig) -> Result<String, DataSourceError> {
    if let Some(url) = &cfg.url {
        return Ok(url.clone());
    }
    let ty = cfg.connector_type.as_str();
    if ty == "sqlite" || ty == "sql" {
        return sqlite_url_from_config(cfg);
    }
    let host = cfg.host.as_deref().unwrap_or("localhost");
    let user = cfg.username.as_deref().unwrap_or("");
    let pass = cfg.password.as_deref().unwrap_or("");
    let db = cfg.database.as_deref().unwrap_or("");
    let auth = if user.is_empty() {
        String::new()
    } else if pass.is_empty() {
        format!("{}@", user)
    } else {
        format!("{}:{}@", user, pass)
    };
    let (scheme, default_port) = match ty {
        "postgresql" | "postgres" => ("postgres", 5432u16),
        "mysql" | "mariadb" => ("mysql", 3306),
        "mssql" | "sqlserver" => ("mssql", 1433),
        "mongodb" => ("mongodb", 27017),
        other => {
            return Err(DataSourceError::Validation {
                message: format!("cannot build URL for datasource type '{}'", other),
            })
        }
    };
    let port = cfg.port.unwrap_or(default_port);
    Ok(format!("{}://{}{}:{}/{}", scheme, auth, host, port, db))
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
