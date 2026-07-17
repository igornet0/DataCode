//! `table.push(data [, ignore=false])` — append rows from object, array, or table.

use crate::common::numeric::{FloatValue, IntValue};
use crate::common::table::Table;
use crate::common::value::{ObjectKind, Value};
use crate::vm::natives::table::try_parse_date;
use crate::vm::store_convert::load_value;
use crate::vm::vm::with_current_stores;
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ColumnType {
    Any,
    Int,
    Float,
    Number,
    String,
    Bool,
    Date,
    Duration,
    Array,
    Mixed,
}

fn push_error(msg: impl Into<String>) -> Value {
    crate::websocket::set_native_error(msg.into());
    Value::Null
}

fn parse_ignore(args: &[Value]) -> bool {
    args.get(2).is_some_and(|v| matches!(v, Value::Bool(true)))
}

fn is_object(v: &Value) -> bool {
    matches!(v, Value::Object(_))
}

fn value_column_type(v: &Value) -> ColumnType {
    match v {
        Value::Null => ColumnType::Any,
        Value::Int(_) => ColumnType::Int,
        Value::Float(_) => ColumnType::Float,
        Value::Number(_) => ColumnType::Number,
        Value::String(_) => ColumnType::String,
        Value::Bool(_) => ColumnType::Bool,
        Value::Date(_) => ColumnType::Date,
        Value::Duration(_) => ColumnType::Duration,
        Value::Array(_) => ColumnType::Array,
        _ => ColumnType::Mixed,
    }
}

fn infer_column_type(table: &Table, col: &str) -> ColumnType {
    if let Some(vals) = table.column_values_owned(col) {
        for v in &vals {
            if !matches!(v, Value::Null) {
                return value_column_type(v);
            }
        }
    }
    ColumnType::Any
}

fn parse_strict_int_string(s: &str) -> Result<i64, ()> {
    let s = s.trim();
    if s.is_empty() {
        return Err(());
    }
    if let Ok(n) = s.parse::<i64>() {
        return Ok(n);
    }
    if let Ok(n) = s.parse::<f64>() {
        if n.fract() == 0.0 && n >= i64::MIN as f64 && n <= i64::MAX as f64 {
            return Ok(n as i64);
        }
    }
    Err(())
}

fn parse_strict_float_string(s: &str) -> Result<f64, ()> {
    let s = s.trim();
    s.parse::<f64>().map_err(|_| ())
}

fn coerce_for_column(value: &Value, expected: ColumnType) -> Result<Value, String> {
    if matches!(value, Value::Null) {
        return Ok(Value::Null);
    }
    match expected {
        ColumnType::Any => Ok(value.clone()),
        ColumnType::Int => match value {
            Value::Int(i) => Ok(Value::Int(*i)),
            Value::Number(n) => {
                if n.fract() != 0.0 {
                    return Err("Cannot convert float to int".to_string());
                }
                Ok(Value::Int(IntValue::Finite(*n as i64)))
            }
            Value::String(s) => parse_strict_int_string(s)
                .map(|n| Value::Int(IntValue::Finite(n)))
                .map_err(|_| "Cannot convert string to int".to_string()),
            Value::Bool(b) => Ok(Value::Int(IntValue::Finite(if *b { 1 } else { 0 }))),
            _ => Err("Cannot convert value to int".to_string()),
        },
        ColumnType::Float | ColumnType::Number => match value {
            Value::Float(f) => Ok(Value::Float(*f)),
            Value::Int(i) => Ok(Value::Float(i.widen_to_float())),
            Value::Number(n) => Ok(Value::Number(*n)),
            Value::String(s) => parse_strict_float_string(s)
                .map(|n| Value::Number(n))
                .map_err(|_| "Cannot convert string to number".to_string()),
            Value::Bool(b) => Ok(Value::Number(if *b { 1.0 } else { 0.0 })),
            _ => Err("Cannot convert value to number".to_string()),
        },
        ColumnType::String => Ok(Value::String(value_to_display_string(value))),
        ColumnType::Bool => match value {
            Value::Bool(b) => Ok(Value::Bool(*b)),
            Value::String(s) => match s.to_ascii_lowercase().as_str() {
                "true" => Ok(Value::Bool(true)),
                "false" => Ok(Value::Bool(false)),
                _ => Err("Cannot convert string to bool".to_string()),
            },
            _ => Err("Cannot convert value to bool".to_string()),
        },
        ColumnType::Date => match value {
            Value::Date(d) => Ok(Value::Date(*d)),
            Value::String(s) => try_parse_date(s)
                .map(Value::Date)
                .ok_or_else(|| "Cannot convert string to date".to_string()),
            _ => Err("Cannot convert value to date".to_string()),
        },
        ColumnType::Duration => match value {
            Value::Duration(d) => Ok(Value::Duration(*d)),
            _ => Err("Cannot convert value to duration".to_string()),
        },
        ColumnType::Array => match value {
            Value::Array(a) => Ok(Value::Array(Rc::clone(a))),
            _ => Err("Cannot convert value to array".to_string()),
        },
        ColumnType::Mixed => Ok(value.clone()),
    }
}

fn value_to_display_string(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Null => "null".to_string(),
        Value::Bool(b) => b.to_string(),
        Value::Int(i) => i.to_display_string(),
        Value::Float(f) => match f {
            FloatValue::Finite(n) => n.to_string(),
            FloatValue::NaN => "nan".to_string(),
            FloatValue::PosInfinity => "inf".to_string(),
            FloatValue::NegInfinity => "-inf".to_string(),
        },
        Value::Number(n) => n.to_string(),
        _ => v.to_string(),
    }
}

fn object_entries(obj: &Value) -> Result<Vec<(String, Value)>, String> {
    let Value::Object(rc) = obj else {
        return Err("expected object".to_string());
    };
    let kind = rc.borrow();
    match &*kind {
        ObjectKind::Legacy(map) => Ok(map
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect()),
        ObjectKind::Inline(pairs) => {
            let mut out = Vec::with_capacity(pairs.len());
            for (k, v) in pairs {
                let key = match k {
                    Value::String(s) => s.clone(),
                    _ => return Err("object keys must be strings".to_string()),
                };
                out.push((key, v.clone()));
            }
            Ok(out)
        }
        ObjectKind::Bucket(omap) => with_current_stores(|store, heap| {
            let mut out = Vec::new();
            for (_, key_id, val_id) in omap.iter_entries() {
                let key_val = load_value(key_id, store, heap);
                let val = load_value(val_id, store, heap);
                let key = match key_val {
                    Value::String(s) => s,
                    _ => return Err("object keys must be strings".to_string()),
                };
                out.push((key, val));
            }
            Ok(out)
        }),
    }
}

fn ensure_columns(table: &mut Table, keys: &[String], ignore: bool) -> Result<(), String> {
    if ignore {
        let missing: Vec<String> = keys
            .iter()
            .filter(|k| !table.has_column(k))
            .cloned()
            .collect();
        if !missing.is_empty() {
            table.extend_columns(&missing)?;
        }
        Ok(())
    } else {
        for key in keys {
            if !table.has_column(key) {
                return Err(format!("Column \"{}\" does not exist", key));
            }
        }
        Ok(())
    }
}

fn build_row_from_object(
    table: &Table,
    entries: &[(String, Value)],
    ignore: bool,
) -> Result<Vec<Value>, String> {
    if !ignore {
        for (k, _) in entries {
            if !table.has_column(k) {
                return Err(format!("Column \"{}\" does not exist", k));
            }
        }
    }
    let map: HashMap<&str, &Value> = entries.iter().map(|(k, v)| (k.as_str(), v)).collect();
    let mut row = Vec::with_capacity(table.column_count());
    for header in table.headers() {
        let raw = map.get(header.as_str()).copied().unwrap_or(&Value::Null);
        let expected = infer_column_type(table, header);
        let coerced = coerce_for_column(raw, expected)?;
        row.push(coerced);
    }
    Ok(row)
}

fn push_object(table: &mut Table, obj: &Value, ignore: bool) -> Result<usize, String> {
    let entries = object_entries(obj)?;
    if ignore {
        let keys: Vec<String> = entries.iter().map(|(k, _)| k.clone()).collect();
        ensure_columns(table, &keys, true)?;
    }
    let row = build_row_from_object(table, &entries, ignore)?;
    table.append_rows(&[row])
}

fn push_array_objects(table: &mut Table, arr: &Rc<RefCell<Vec<Value>>>, ignore: bool) -> Result<usize, String> {
    let items = arr.borrow();
    if items.is_empty() {
        return Ok(0);
    }
    if !is_object(&items[0]) {
        return Err("expected array of objects".to_string());
    }
    for item in items.iter() {
        if !is_object(item) {
            return Err("expected array of objects".to_string());
        }
    }
    if ignore {
        let mut all_keys = Vec::new();
        for item in items.iter() {
            let entries = object_entries(item)?;
            for (k, _) in entries {
                if !all_keys.contains(&k) {
                    all_keys.push(k);
                }
            }
        }
        ensure_columns(table, &all_keys, true)?;
    }
    let mut rows = Vec::with_capacity(items.len());
    for item in items.iter() {
        let entries = object_entries(item)?;
        rows.push(build_row_from_object(table, &entries, ignore)?);
    }
    table.append_rows(&rows)
}

fn push_array_row(table: &mut Table, arr: &Rc<RefCell<Vec<Value>>>, _ignore: bool) -> Result<usize, String> {
    let items = arr.borrow();
    if table.column_count() == 0 {
        return Err("table has no columns".to_string());
    }
    if items.len() != table.column_count() {
        return Err(format!(
            "Invalid row length\nExpected {} columns\nGot {}",
            table.column_count(),
            items.len()
        ));
    }
    let headers = table.headers().clone();
    let mut row = Vec::with_capacity(items.len());
    for (i, v) in items.iter().enumerate() {
        let col = &headers[i];
        let expected = infer_column_type(table, col);
        row.push(coerce_for_column(v, expected)?);
    }
    table.append_rows(&[row])
}

fn push_table(dst: &mut Table, src: &Table, ignore: bool) -> Result<usize, String> {
    let src_len = src.len();
    if src_len == 0 {
        return Ok(0);
    }

    let src_owned = if src.is_view() {
        with_current_stores(|store, heap| {
            src.materialize_with(|id| load_value(id, store, heap))
        })
    } else {
        src.clone()
    };

    let extra: Vec<String> = src_owned
        .headers()
        .iter()
        .filter(|h| !dst.has_column(h))
        .cloned()
        .collect();

    if !extra.is_empty() {
        if ignore {
            dst.extend_columns(&extra)?;
        } else {
            return Err(format!("Unknown column \"{}\"", extra[0]));
        }
    }

    let dst_headers = dst.headers().clone();
    let src_headers = src_owned.headers();

    if dst_headers.len() == src_headers.len()
        && dst_headers.iter().zip(src_headers.iter()).all(|(a, b)| a == b)
    {
        if let Some(chunk) = src_owned.owned_flat() {
            return dst.append_flat_chunk(chunk);
        }
    }

    let mut rows = Vec::with_capacity(src_len);
    for row_idx in 0..src_len {
        let src_row = if let Some(r) = src_owned.get_row(row_idx) {
            r.to_vec()
        } else {
            with_current_stores(|store, heap| {
                crate::vm::table_ops::get_row(&src_owned, row_idx, store, heap)
                    .unwrap_or_default()
            })
        };
        let src_map: HashMap<&str, Value> = src_headers
            .iter()
            .zip(src_row.iter())
            .map(|(h, v)| (h.as_str(), v.clone()))
            .collect();
        let mut aligned = Vec::with_capacity(dst_headers.len());
        for h in &dst_headers {
            let raw = src_map.get(h.as_str()).cloned().unwrap_or(Value::Null);
            let expected = infer_column_type(dst, h);
            aligned.push(coerce_for_column(&raw, expected)?);
        }
        rows.push(aligned);
    }
    dst.append_rows(&rows)
}

fn push_dispatch(table: &mut Table, data: &Value, ignore: bool) -> Result<usize, String> {
    match data {
        Value::Object(_) => push_object(table, data, ignore),
        Value::Array(arr) => {
            let borrowed = arr.borrow();
            if borrowed.is_empty() {
                Ok(0)
            } else if is_object(&borrowed[0]) {
                drop(borrowed);
                push_array_objects(table, arr, ignore)
            } else {
                drop(borrowed);
                push_array_row(table, arr, ignore)
            }
        }
        Value::Table(src) => {
            let src_ref = src.borrow();
            push_table(table, &src_ref, ignore)
        }
        _ => Err("push() expects object, array, or table".to_string()),
    }
}

pub fn native_table_push(args: &[Value]) -> Value {
    if args.len() < 2 {
        return push_error("push() expects at least one argument");
    }
    let Value::Table(table_rc) = &args[0] else {
        return Value::Null;
    };
    let ignore = parse_ignore(args);
    let data = &args[1];

    let count = with_current_stores(|store, heap| {
        let mut table_ref = table_rc.borrow_mut();
        table_ref.ensure_owned(|id| load_value(id, store, heap));
        push_dispatch(&mut table_ref, data, ignore)
    });

    match count {
        Ok(n) => Value::Number(n as f64),
        Err(e) => push_error(e),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::table::Table;

    #[test]
    fn coerce_int_rejects_hello() {
        let err = coerce_for_column(&Value::String("hello".to_string()), ColumnType::Int);
        assert!(err.is_err());
        assert!(err.unwrap_err().contains("Cannot convert string to int"));
    }

    #[test]
    fn coerce_int_accepts_25_string() {
        let v = coerce_for_column(&Value::String("25".to_string()), ColumnType::Int).unwrap();
        assert!(matches!(v, Value::Int(IntValue::Finite(25))));
    }

    #[test]
    fn extend_columns_nulls_old_rows() {
        let mut t = Table::from_data(
            vec![vec![Value::Number(1.0), Value::String("a".to_string())]],
            Some(vec!["id".to_string(), "name".to_string()]),
        );
        t.extend_columns(&["age".to_string()]).unwrap();
        assert_eq!(t.column_count(), 3);
        let row = t.get_row(0).unwrap();
        assert_eq!(row.len(), 3);
        assert!(matches!(row[2], Value::Null));
    }
}
