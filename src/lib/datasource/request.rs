//! Request / get_table / send_table spec parsing.

use crate::common::value::Value;
use crate::datasource::config::object_to_map;
use crate::datasource::error::DataSourceError;
use std::collections::HashMap;

#[derive(Debug, Clone, Default)]
pub struct RequestSpec {
    pub method: String,
    pub path: Option<String>,
    pub url: Option<String>,
    pub sql: Option<String>,
    pub headers: HashMap<String, String>,
    pub query: HashMap<String, String>,
    pub body: Option<String>,
    pub json: Option<Value>,
    pub form: HashMap<String, String>,
    pub parameters: Vec<Value>,
    pub timeout: Option<f64>,
    /// Operation hint for non-HTTP backends: "find", "count", "list_databases", …
    pub op: Option<String>,
    pub collection: Option<String>,
    pub filter: Option<Value>,
    pub native: Option<Value>,
    pub aggregation: Option<Value>,
    pub limit: Option<usize>,
    pub offset: Option<usize>,
}

#[derive(Debug, Clone, Default)]
pub struct GetTableSpec {
    pub method: String,
    pub path: Option<String>,
    pub url: Option<String>,
    pub sql: Option<String>,
    pub parameters: Vec<Value>,
    pub format: Option<String>,
    pub root: Option<String>,
    pub headers: HashMap<String, String>,
    pub query: HashMap<String, String>,
    pub timeout: Option<f64>,
    pub limit: Option<usize>,
    pub offset: Option<usize>,
    pub collection: Option<String>,
    pub filter: Option<Value>,
    pub select: Option<Value>,
    pub sort: Option<Value>,
    pub aggregation: Option<Value>,
    pub native: Option<Value>,
    pub array_mode: Option<String>,
    pub flatten: bool,
    pub schema_sample_size: Option<usize>,
    pub batch_size: Option<usize>,
}

#[derive(Debug, Clone, Default)]
pub struct SendTableSpec {
    pub table: Option<Value>,
    pub mode: String,
    pub path: Option<String>,
    pub url: Option<String>,
    pub method: String,
    pub table_name: Option<String>,
    pub collection: Option<String>,
    pub sql: Option<String>,
    pub format: Option<String>,
    pub batch_size: usize,
}

fn string_from_value(v: &Value) -> Option<String> {
    match v {
        Value::String(s) => Some(s.clone()),
        Value::Path(p) => Some(p.to_string_lossy().into_owned()),
        _ => None,
    }
}

fn usize_from_value(v: &Value) -> Option<usize> {
    match v {
        Value::Number(n) if *n >= 0.0 => Some(*n as usize),
        Value::Int(i) => match i {
            crate::common::numeric::IntValue::Finite(n) if *n >= 0 => Some(*n as usize),
            _ => None,
        },
        _ => None,
    }
}

fn query_from_map(map: &HashMap<String, Value>) -> HashMap<String, String> {
    let mut out = HashMap::new();
    for (k, v) in map {
        let s = match v {
            Value::String(s) => s.clone(),
            Value::Number(n) => n.to_string(),
            Value::Bool(b) => b.to_string(),
            Value::Int(i) => i.to_display_string(),
            _ => continue,
        };
        out.insert(k.clone(), s);
    }
    out
}

pub fn parse_request_spec(value: &Value) -> Result<RequestSpec, DataSourceError> {
    let map = object_to_map(value)?;
    let mut spec = RequestSpec {
        method: map
            .get("method")
            .and_then(string_from_value)
            .unwrap_or_else(|| "GET".to_string())
            .to_uppercase(),
        path: map.get("path").and_then(string_from_value),
        url: map.get("url").and_then(string_from_value),
        sql: map.get("sql").and_then(string_from_value),
        body: map.get("body").and_then(string_from_value),
        json: map.get("json").cloned(),
        timeout: map.get("timeout").and_then(|v| match v {
            Value::Number(n) => Some(*n),
            _ => None,
        }),
        op: map.get("op").and_then(string_from_value),
        collection: map
            .get("collection")
            .and_then(string_from_value)
            .or_else(|| map.get("table_name").and_then(string_from_value)),
        filter: map.get("filter").cloned(),
        native: map.get("native").cloned(),
        aggregation: map.get("aggregation").cloned(),
        limit: map.get("limit").and_then(usize_from_value),
        offset: map.get("offset").and_then(usize_from_value),
        ..Default::default()
    };
    if let Some(h) = map.get("headers") {
        spec.headers = crate::datasource::config::headers_from_value(h);
    }
    if let Some(q) = map.get("query") {
        if let Ok(qm) = object_to_map(q) {
            spec.query = query_from_map(&qm);
        }
    }
    if let Some(f) = map.get("form") {
        if let Ok(fm) = object_to_map(f) {
            spec.form = query_from_map(&fm);
        }
    }
    if let Some(p) = map.get("parameters") {
        spec.parameters = match p {
            Value::Array(rc) => rc.borrow().clone(),
            other => vec![other.clone()],
        };
    }
    Ok(spec)
}

pub fn parse_get_table_spec(value: &Value) -> Result<GetTableSpec, DataSourceError> {
    let map = object_to_map(value)?;
    let flatten = match map.get("flatten") {
        Some(Value::Bool(b)) => *b,
        _ => false,
    };
    let mut spec = GetTableSpec {
        method: map
            .get("method")
            .and_then(string_from_value)
            .unwrap_or_else(|| "GET".to_string())
            .to_uppercase(),
        path: map.get("path").and_then(string_from_value),
        url: map.get("url").and_then(string_from_value),
        sql: map.get("sql").and_then(string_from_value),
        format: map.get("format").and_then(string_from_value),
        root: map.get("root").and_then(string_from_value),
        timeout: map.get("timeout").and_then(|v| match v {
            Value::Number(n) => Some(*n),
            _ => None,
        }),
        limit: map.get("limit").and_then(usize_from_value),
        offset: map.get("offset").and_then(usize_from_value),
        collection: map
            .get("collection")
            .and_then(string_from_value)
            .or_else(|| map.get("table_name").and_then(string_from_value)),
        filter: map.get("filter").cloned(),
        select: map
            .get("select")
            .cloned()
            .or_else(|| map.get("projection").cloned()),
        sort: map.get("sort").cloned(),
        aggregation: map.get("aggregation").cloned(),
        native: map.get("native").cloned(),
        array_mode: map.get("array_mode").and_then(string_from_value),
        flatten,
        schema_sample_size: map.get("schema_sample_size").and_then(usize_from_value),
        batch_size: map.get("batch_size").and_then(usize_from_value),
        ..Default::default()
    };
    if let Some(h) = map.get("headers") {
        spec.headers = crate::datasource::config::headers_from_value(h);
    }
    if let Some(q) = map.get("query") {
        if let Ok(qm) = object_to_map(q) {
            spec.query = query_from_map(&qm);
        }
    }
    if let Some(p) = map.get("parameters") {
        spec.parameters = match p {
            Value::Array(rc) => rc.borrow().clone(),
            other => vec![other.clone()],
        };
    }
    Ok(spec)
}

pub fn parse_send_table_spec(
    table_arg: Option<&Value>,
    spec_arg: Option<&Value>,
) -> Result<SendTableSpec, DataSourceError> {
    let mut spec = SendTableSpec {
        mode: "append".to_string(),
        method: "POST".to_string(),
        batch_size: 100,
        ..Default::default()
    };
    if let Some(t) = table_arg {
        if matches!(t, Value::Table(_)) {
            spec.table = Some(t.clone());
        } else if let Value::Object(_) = t {
            let map = object_to_map(t)?;
            if let Some(tbl) = map.get("table") {
                spec.table = Some(tbl.clone());
            }
            spec.mode = map
                .get("mode")
                .and_then(string_from_value)
                .unwrap_or_else(|| "append".to_string());
            spec.path = map.get("path").and_then(string_from_value);
            spec.url = map.get("url").and_then(string_from_value);
            spec.table_name = map.get("table_name").and_then(string_from_value);
            spec.collection = map
                .get("collection")
                .and_then(string_from_value)
                .or_else(|| map.get("table_name").and_then(string_from_value));
            spec.sql = map.get("sql").and_then(string_from_value);
            spec.format = map.get("format").and_then(string_from_value);
            if let Some(m) = map.get("method").and_then(string_from_value) {
                spec.method = m.to_uppercase();
            }
            if let Some(n) = map.get("batch_size").and_then(usize_from_value) {
                if n > 0 {
                    spec.batch_size = n;
                }
            }
            return Ok(spec);
        }
    }
    if let Some(s) = spec_arg {
        let map = object_to_map(s)?;
        if let Some(tbl) = map.get("table") {
            spec.table = Some(tbl.clone());
        }
        spec.mode = map
            .get("mode")
            .and_then(string_from_value)
            .unwrap_or_else(|| "append".to_string());
        spec.path = map.get("path").and_then(string_from_value);
        spec.url = map.get("url").and_then(string_from_value);
        spec.table_name = map.get("table_name").and_then(string_from_value);
        spec.collection = map
            .get("collection")
            .and_then(string_from_value)
            .or_else(|| map.get("table_name").and_then(string_from_value));
        spec.format = map.get("format").and_then(string_from_value);
    }
    if spec.table.is_none() {
        if let Some(t) = table_arg {
            if matches!(t, Value::Table(_)) {
                spec.table = Some(t.clone());
            }
        }
    }
    if spec.table.is_none() {
        return Err(DataSourceError::Validation {
            message: "send_table() requires a table argument".to_string(),
        });
    }
    Ok(spec)
}
