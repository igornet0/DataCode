//! Convert response bytes / JSON to Table.

use crate::common::table::Table;
use crate::common::value::Value;
use crate::datasource::error::DataSourceError;
use crate::datasource::request::GetTableSpec;
use crate::datasource::response::{format_from_spec_or_content_type, DataSourceResponse};
use crate::file_io::{read_bytes_from_memory, ReadOptions};
use std::path::PathBuf;

pub fn bytes_to_table(
    bytes: &[u8],
    virtual_path: &PathBuf,
    spec: &GetTableSpec,
    content_type: Option<&str>,
) -> Result<Table, DataSourceError> {
    let format = format_from_spec_or_content_type(
        spec.format.as_deref(),
        content_type,
        spec.path.as_deref().or(spec.url.as_deref()),
    );
    let mut path = virtual_path.clone();
    if path.extension().is_none() && format != "txt" {
        path = PathBuf::from(format!("response.{}", format));
    }
    let bytes = if format == "json" || path.extension().map(|e| e == "json").unwrap_or(false) {
        if let Some(root) = &spec.root {
            extract_json_root(bytes, root)?
        } else {
            bytes.to_vec()
        }
    } else {
        bytes.to_vec()
    };
    let value = read_bytes_from_memory(&path, &bytes, &ReadOptions::default()).map_err(|e| {
        DataSourceError::Parse {
            message: e,
        }
    })?;
    table_from_value(&value)
}

pub fn response_to_table(
    response: &DataSourceResponse,
    spec: &GetTableSpec,
) -> Result<Table, DataSourceError> {
    let path = response.virtual_path_for_format();
    bytes_to_table(
        &response.body,
        &path,
        spec,
        response.content_type.as_deref(),
    )
}

fn extract_json_root(bytes: &[u8], root: &str) -> Result<Vec<u8>, DataSourceError> {
    let text = std::str::from_utf8(bytes).map_err(|e| DataSourceError::Parse {
        message: format!("invalid UTF-8 in JSON response: {}", e),
    })?;
    let mut value = crate::file_io::value_serde::parse_json_str(text).map_err(|e| {
        DataSourceError::Parse {
            message: e.message(),
        }
    })?;
    for part in root.split('.') {
        value = navigate_json(value, part)?;
    }
  Ok(serde_json::to_vec(&value_to_json(&value)).map_err(|e| {
        DataSourceError::Parse {
            message: format!("JSON serialize: {}", e),
        }
    })?)
}

fn navigate_json(value: Value, key: &str) -> Result<Value, DataSourceError> {
    match value {
        Value::Object(rc) => {
            let kind = rc.borrow();
            match &*kind {
                crate::common::value::ObjectKind::Legacy(m) => m
                    .get(key)
                    .cloned()
                    .ok_or_else(|| DataSourceError::Parse {
                        message: format!("JSON root path: key '{}' not found", key),
                    }),
                crate::common::value::ObjectKind::Inline(entries) => entries
                    .iter()
                    .find(|(k, _)| match k {
                        Value::String(s) => s == key,
                        _ => false,
                    })
                    .map(|(_, v)| v.clone())
                    .ok_or_else(|| DataSourceError::Parse {
                        message: format!("JSON root path: key '{}' not found", key),
                    }),
                crate::common::value::ObjectKind::Bucket(_) => Err(DataSourceError::Parse {
                    message: "JSON root navigation requires plain object".to_string(),
                }),
            }
        }
        _ => Err(DataSourceError::Parse {
            message: format!("JSON root path: cannot navigate into '{}'", key),
        }),
    }
}

fn value_to_json(v: &Value) -> serde_json::Value {
    match v {
        Value::Null => serde_json::Value::Null,
        Value::Bool(b) => serde_json::Value::Bool(*b),
        Value::Number(n) => serde_json::json!(*n),
        Value::Int(i) => match i {
            crate::common::numeric::IntValue::Finite(n) => serde_json::json!(n),
            crate::common::numeric::IntValue::PosInfinity => serde_json::json!("inf"),
            crate::common::numeric::IntValue::NegInfinity => serde_json::json!("-inf"),
        },
        Value::String(s) => serde_json::Value::String(s.clone()),
        Value::Array(rc) => {
            let arr = rc.borrow();
            serde_json::Value::Array(arr.iter().map(value_to_json).collect())
        }
        Value::Object(rc) => {
            let kind = rc.borrow();
            let mut map = serde_json::Map::new();
            match &*kind {
                crate::common::value::ObjectKind::Legacy(m) => {
                    for (k, val) in m {
                        map.insert(k.clone(), value_to_json(val));
                    }
                }
                crate::common::value::ObjectKind::Inline(entries) => {
                    for (k, val) in entries {
                        if let Value::String(sk) = k {
                            map.insert(sk.clone(), value_to_json(val));
                        }
                    }
                }
                crate::common::value::ObjectKind::Bucket(_) => {}
            }
            serde_json::Value::Object(map)
        }
        other => serde_json::Value::String(other.to_string()),
    }
}

pub fn table_from_value(value: &Value) -> Result<Table, DataSourceError> {
    match value {
        Value::Table(rc) => Ok(rc.borrow().clone()),
        Value::Array(rc) => {
            let arr = rc.borrow();
            if arr.is_empty() {
                return Ok(Table::from_data(vec![], Some(vec![])));
            }
            if arr.iter().all(|v| matches!(v, Value::Array(_))) {
                let mut rows = Vec::new();
                for row in arr.iter() {
                    if let Value::Array(r) = row {
                        rows.push(r.borrow().clone());
                    }
                }
                let col_count = rows.first().map(|r| r.len()).unwrap_or(0);
                let headers: Vec<String> = (0..col_count).map(|i| format!("col{}", i)).collect();
                return Ok(Table::from_data(rows, Some(headers)));
            }
            if arr.iter().all(|v| matches!(v, Value::Object(_))) {
                return json_array_objects_to_table(&arr);
            }
            Err(DataSourceError::Parse {
                message: "cannot convert array to table".to_string(),
            })
        }
        Value::Object(_) => json_array_objects_to_table(&[value.clone()]),
        _ => Err(DataSourceError::Parse {
            message: format!(
                "expected table data, got {}",
                value.to_string()
            ),
        }),
    }
}

fn json_array_objects_to_table(items: &[Value]) -> Result<Table, DataSourceError> {
    let mut headers_set = std::collections::BTreeSet::new();
    let mut rows_data: Vec<std::collections::HashMap<String, Value>> = Vec::new();
    for item in items {
        let Value::Object(rc) = item else {
            continue;
        };
        let kind = rc.borrow();
        let mut row = std::collections::HashMap::new();
        match &*kind {
            crate::common::value::ObjectKind::Legacy(m) => {
                for (k, v) in m {
                    headers_set.insert(k.clone());
                    row.insert(k.clone(), v.clone());
                }
            }
            crate::common::value::ObjectKind::Inline(entries) => {
                for (k, v) in entries {
                    if let Value::String(sk) = k {
                        headers_set.insert(sk.clone());
                        row.insert(sk.clone(), v.clone());
                    }
                }
            }
            crate::common::value::ObjectKind::Bucket(_) => {}
        }
        rows_data.push(row);
    }
    let headers: Vec<String> = headers_set.into_iter().collect();
    let rows: Vec<Vec<Value>> = rows_data
        .into_iter()
        .map(|row| headers.iter().map(|h| row.get(h).cloned().unwrap_or(Value::Null)).collect())
        .collect();
    Ok(Table::from_data(rows, Some(headers)))
}
