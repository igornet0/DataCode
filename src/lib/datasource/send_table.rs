//! Send table to external source.

use crate::common::table::Table;
use crate::common::value::{ObjectKind, Value};
use crate::datasource::error::DataSourceError;
use crate::file_io::save_value;
use std::cell::RefCell;
use std::path::PathBuf;
use std::rc::Rc;

pub fn table_to_json_value(table: &Table) -> Value {
    let headers = table.headers().clone();
    let mut rows = Vec::new();
    if let Some(rr) = table.rows_ref() {
        for row in rr.iter() {
            let mut obj = std::collections::HashMap::new();
            for (i, h) in headers.iter().enumerate() {
                let val = row.get(i).cloned().unwrap_or(Value::Null);
                obj.insert(h.clone(), val);
            }
            rows.push(Value::Object(Rc::new(RefCell::new(ObjectKind::legacy(obj)))));
        }
    }
    Value::Array(Rc::new(RefCell::new(rows)))
}

pub fn save_table_to_path(table: &Value, path: &PathBuf) -> Result<(), DataSourceError> {
    save_value(table, &Value::Path(path.clone()))
        .map(|_| ())
        .map_err(|e| DataSourceError::Other { message: e })
}

pub fn table_rows_batch(table: &Table, batch_size: usize) -> Vec<Vec<Vec<Value>>> {
    let mut batches = Vec::new();
    let mut current = Vec::new();
    if let Some(rr) = table.rows_ref() {
        for row in rr.iter() {
            current.push(row.to_vec());
            if current.len() >= batch_size {
                batches.push(current);
                current = Vec::new();
            }
        }
    }
    if !current.is_empty() {
        batches.push(current);
    }
    batches
}
