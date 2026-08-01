//! Arrow IPC → DataCode `Table` conversion and strict column selection for `ws.source_table`.

use std::io::Cursor;

use arrow::array::{Array, AsArray, StringArray};
use arrow::datatypes::{DataType, TimeUnit};
use arrow::ipc::reader::{FileReader, StreamReader};
use arrow::record_batch::RecordBatch;

use crate::common::table::Table;
use crate::common::value::Value;

pub fn arrow_ipc_to_table(bytes: &[u8]) -> Result<Table, String> {
    let batches = read_ipc_batches(bytes)?;
    if batches.is_empty() {
        return Ok(Table::from_data(Vec::new(), None));
    }

    let schema = batches[0].schema();
    let headers: Vec<String> = schema
        .fields()
        .iter()
        .map(|f| f.name().clone())
        .collect();

    let mut rows: Vec<Vec<Value>> = Vec::new();
    for batch in &batches {
        let batch_rows = record_batch_to_rows(batch, schema.fields().len())?;
        rows.extend(batch_rows);
    }

    Ok(Table::from_data(rows, Some(headers)))
}

fn read_ipc_batches(bytes: &[u8]) -> Result<Vec<RecordBatch>, String> {
    let cursor = Cursor::new(bytes);
    if let Ok(mut reader) = FileReader::try_new(cursor, None) {
        let schema = reader.schema();
        let mut batches = Vec::new();
        while let Some(batch) = reader
            .next()
            .transpose()
            .map_err(|e| format!("Arrow IPC file read error: {e}"))?
        {
            batches.push(batch);
        }
        if !batches.is_empty() {
            return Ok(batches);
        }
        let _ = schema;
    }

    let cursor = Cursor::new(bytes);
    let mut reader = StreamReader::try_new(cursor, None)
        .map_err(|e| format!("Arrow IPC stream read error: {e}"))?;
    let mut batches = Vec::new();
    while let Some(batch) = reader
        .next()
        .transpose()
        .map_err(|e| format!("Arrow IPC stream read error: {e}"))?
    {
        batches.push(batch);
    }
    Ok(batches)
}

fn record_batch_to_rows(batch: &RecordBatch, num_cols: usize) -> Result<Vec<Vec<Value>>, String> {
    let num_rows = batch.num_rows();
    if num_rows == 0 {
        return Ok(Vec::new());
    }

    let mut column_values: Vec<Vec<Value>> = Vec::with_capacity(num_cols);
    for col_idx in 0..num_cols {
        let array = batch.column(col_idx);
        column_values.push(array_to_values(array)?);
    }

    let mut rows = Vec::with_capacity(num_rows);
    for row_idx in 0..num_rows {
        let mut row = Vec::with_capacity(num_cols);
        for col in &column_values {
            row.push(col.get(row_idx).cloned().unwrap_or(Value::Null));
        }
        rows.push(row);
    }
    Ok(rows)
}

fn array_to_values(array: &dyn Array) -> Result<Vec<Value>, String> {
    if array.is_null(0) && array.len() == 1 && matches!(array.data_type(), DataType::Null) {
        return Ok(vec![Value::Null; array.len()]);
    }

    match array.data_type() {
        DataType::Null => Ok(vec![Value::Null; array.len()]),
        DataType::Boolean => {
            let arr = array.as_boolean();
            Ok((0..arr.len())
                .map(|i| {
                    if arr.is_null(i) {
                        Value::Null
                    } else {
                        Value::Bool(arr.value(i))
                    }
                })
                .collect())
        }
        DataType::Int32 => {
            let arr = array.as_primitive::<arrow::datatypes::Int32Type>();
            Ok((0..arr.len())
                .map(|i| {
                    if arr.is_null(i) {
                        Value::Null
                    } else {
                        Value::Number(arr.value(i) as f64)
                    }
                })
                .collect())
        }
        DataType::Int64 => {
            let arr = array.as_primitive::<arrow::datatypes::Int64Type>();
            Ok((0..arr.len())
                .map(|i| {
                    if arr.is_null(i) {
                        Value::Null
                    } else {
                        Value::Number(arr.value(i) as f64)
                    }
                })
                .collect())
        }
        DataType::Float32 => {
            let arr = array.as_primitive::<arrow::datatypes::Float32Type>();
            Ok((0..arr.len())
                .map(|i| {
                    if arr.is_null(i) {
                        Value::Null
                    } else {
                        Value::Number(arr.value(i) as f64)
                    }
                })
                .collect())
        }
        DataType::Float64 => {
            let arr = array.as_primitive::<arrow::datatypes::Float64Type>();
            Ok((0..arr.len())
                .map(|i| {
                    if arr.is_null(i) {
                        Value::Null
                    } else {
                        Value::Number(arr.value(i))
                    }
                })
                .collect())
        }
        DataType::Utf8 => string_array_to_values(array.as_string::<i32>()),
        DataType::LargeUtf8 => {
            let arr = array.as_string::<i64>();
            Ok((0..arr.len())
                .map(|i| {
                    if arr.is_null(i) {
                        Value::Null
                    } else {
                        Value::String(arr.value(i).to_string())
                    }
                })
                .collect())
        }
        DataType::Date32 => {
            let arr = array.as_primitive::<arrow::datatypes::Date32Type>();
            Ok((0..arr.len())
                .map(|i| {
                    if arr.is_null(i) {
                        Value::Null
                    } else {
                        Value::Number(arr.value(i) as f64)
                    }
                })
                .collect())
        }
        DataType::Date64 => {
            let arr = array.as_primitive::<arrow::datatypes::Date64Type>();
            Ok((0..arr.len())
                .map(|i| {
                    if arr.is_null(i) {
                        Value::Null
                    } else {
                        Value::Number(arr.value(i) as f64)
                    }
                })
                .collect())
        }
        DataType::Timestamp(unit, _) => timestamp_array_to_values(array, *unit),
        other => {
            if array.is_null(0) && array.len() > 0 {
                // Fallback: stringify unsupported types row-by-row via display
            }
            Err(format!(
                "Unsupported Arrow column type for ws.source_table: {other:?}"
            ))
        }
    }
}

fn string_array_to_values(arr: &StringArray) -> Result<Vec<Value>, String> {
    Ok((0..arr.len())
        .map(|i| {
            if arr.is_null(i) {
                Value::Null
            } else {
                Value::String(arr.value(i).to_string())
            }
        })
        .collect())
}

fn timestamp_array_to_values(array: &dyn Array, unit: TimeUnit) -> Result<Vec<Value>, String> {
    match unit {
        TimeUnit::Second => {
            let arr = array.as_primitive::<arrow::datatypes::TimestampSecondType>();
            Ok((0..arr.len())
                .map(|i| {
                    if arr.is_null(i) {
                        Value::Null
                    } else {
                        Value::Number(arr.value(i) as f64)
                    }
                })
                .collect())
        }
        TimeUnit::Millisecond => {
            let arr = array.as_primitive::<arrow::datatypes::TimestampMillisecondType>();
            Ok((0..arr.len())
                .map(|i| {
                    if arr.is_null(i) {
                        Value::Null
                    } else {
                        Value::Number(arr.value(i) as f64)
                    }
                })
                .collect())
        }
        TimeUnit::Microsecond => {
            let arr = array.as_primitive::<arrow::datatypes::TimestampMicrosecondType>();
            Ok((0..arr.len())
                .map(|i| {
                    if arr.is_null(i) {
                        Value::Null
                    } else {
                        Value::Number(arr.value(i) as f64)
                    }
                })
                .collect())
        }
        TimeUnit::Nanosecond => {
            let arr = array.as_primitive::<arrow::datatypes::TimestampNanosecondType>();
            Ok((0..arr.len())
                .map(|i| {
                    if arr.is_null(i) {
                        Value::Null
                    } else {
                        Value::Number(arr.value(i) as f64)
                    }
                })
                .collect())
        }
    }
}

/// Strict column filter for `ws.source_table` (errors on missing columns; object = select + rename).
pub fn apply_source_table_columns(table: Table, columns: Option<&Value>) -> Result<Table, String> {
    let Some(columns) = columns else {
        return Ok(table);
    };

    match columns {
        Value::Null => Ok(table),
        Value::Array(cols_arr) => {
            let cols_arr_ref = cols_arr.borrow();
            let selected: Vec<String> = cols_arr_ref
                .iter()
                .map(|v| match v {
                    Value::String(s) => Ok(s.clone()),
                    _ => Err("TypeError: columns array must contain only strings".to_string()),
                })
                .collect::<Result<_, _>>()?;

            if selected.is_empty() {
                return Ok(table);
            }

            let headers = table.headers();
            let mut col_indices = Vec::new();
            let mut new_headers = Vec::new();
            for col_name in &selected {
                let idx = headers
                    .iter()
                    .position(|h| h == col_name)
                    .ok_or_else(|| format!("Column '{col_name}' not found in table"))?;
                col_indices.push(idx);
                new_headers.push(col_name.clone());
            }

            let rr = table
                .rows_ref()
                .ok_or_else(|| "Internal error: table has no row view".to_string())?;
            let mut new_rows = Vec::new();
            for row in rr.iter() {
                let mut new_row = Vec::new();
                for &idx in &col_indices {
                    new_row.push(row.get(idx).cloned().unwrap_or(Value::Null));
                }
                new_rows.push(new_row);
            }
            Ok(Table::from_data(new_rows, Some(new_headers)))
        }
        Value::Object(rename_map_rc) => {
            let rename_map = rename_map_rc.borrow();
            let headers = table.headers();
            let mut selected_pairs: Vec<(usize, String)> = Vec::new();

            for (old_name, rename_val) in rename_map.str_key_pairs() {
                let idx = headers
                    .iter()
                    .position(|h| h == &old_name)
                    .ok_or_else(|| format!("Column '{old_name}' not found in table"))?;
                let new_name = match rename_val {
                    Value::String(s) => s.clone(),
                    Value::Null => old_name.clone(),
                    _ => {
                        return Err(format!(
                            "TypeError: rename for column '{old_name}' must be string or null"
                        ));
                    }
                };
                selected_pairs.push((idx, new_name));
            }

            if selected_pairs.is_empty() {
                return Ok(table);
            }

            let new_headers: Vec<String> = selected_pairs.iter().map(|(_, n)| n.clone()).collect();
            let rr = table
                .rows_ref()
                .ok_or_else(|| "Internal error: table has no row view".to_string())?;
            let mut new_rows = Vec::new();
            for row in rr.iter() {
                let mut new_row = Vec::new();
                for (idx, _) in &selected_pairs {
                    new_row.push(row.get(*idx).cloned().unwrap_or(Value::Null));
                }
                new_rows.push(new_row);
            }
            Ok(Table::from_data(new_rows, Some(new_headers)))
        }
        _ => Err("TypeError: columns must be null, array, or object".to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::rc::Rc;
    use std::cell::RefCell;

    fn sample_table() -> Table {
        Table::from_data(
            vec![
                vec![
                    Value::Number(1.0),
                    Value::String("a".to_string()),
                    Value::Number(10.0),
                ],
                vec![
                    Value::Number(2.0),
                    Value::String("b".to_string()),
                    Value::Number(20.0),
                ],
            ],
            Some(vec!["id".to_string(), "date".to_string(), "value".to_string()]),
        )
    }

    #[test]
    fn apply_columns_null_keeps_all() {
        let table = sample_table();
        let out = apply_source_table_columns(table, Some(&Value::Null)).unwrap();
        assert_eq!(out.headers(), &["id", "date", "value"]);
    }

    #[test]
    fn apply_columns_array_selects_in_order() {
        let table = sample_table();
        let cols = Value::Array(Rc::new(RefCell::new(vec![
            Value::String("date".to_string()),
            Value::String("id".to_string()),
        ])));
        let out = apply_source_table_columns(table, Some(&cols)).unwrap();
        assert_eq!(out.headers(), &["date", "id"]);
        let rr = out.rows_ref().unwrap();
        assert_eq!(rr.row(0).unwrap()[0], Value::String("a".to_string()));
        assert_eq!(rr.row(0).unwrap()[1], Value::Number(1.0));
    }

    #[test]
    fn apply_columns_array_missing_errors() {
        let table = sample_table();
        let cols = Value::Array(Rc::new(RefCell::new(vec![Value::String(
            "missing".to_string(),
        )])));
        let err = apply_source_table_columns(table, Some(&cols)).unwrap_err();
        assert!(err.contains("missing"));
    }

    #[test]
    fn apply_columns_object_select_rename() {
        let table = sample_table();
        let mut map = HashMap::new();
        map.insert("id".to_string(), Value::Null);
        map.insert("value".to_string(), Value::String("amount".to_string()));
        let obj = Value::legacy_object(map);
        let out = apply_source_table_columns(table, Some(&obj)).unwrap();
        let headers = out.headers();
        assert!(headers.contains(&"id".to_string()));
        assert!(headers.contains(&"amount".to_string()));
        assert_eq!(headers.len(), 2);
    }

    #[test]
    fn apply_columns_object_missing_errors() {
        let table = sample_table();
        let mut map = HashMap::new();
        map.insert("nope".to_string(), Value::Null);
        let obj = Value::legacy_object(map);
        let err = apply_source_table_columns(table, Some(&obj)).unwrap_err();
        assert!(err.contains("nope"));
    }
}
