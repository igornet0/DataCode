//! Universal `save()` dispatch by value type and file extension.

use crate::common::table::Table;
use crate::common::table_csv_export::write_table_csv;
use crate::common::value::{ObjectKind, Value};
use crate::file_io::path_input::{extension_lower, format_path_for_error, path_from_value, write_bytes_to_path};
use crate::file_io::value_serde::{
    value_to_json_string_pretty, value_to_toml_string_pretty, value_to_xml_string,
    value_to_yaml_string,
};
use crate::sqlite_export::export_single_table;
use std::cell::RefCell;
use std::path::PathBuf;
use std::rc::Rc;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ValueKind {
    Table,
    Object,
    String,
    Bytes,
    Other,
}

pub fn value_kind(v: &Value) -> ValueKind {
    match v {
        Value::Table(_) => ValueKind::Table,
        Value::Object(_) => ValueKind::Object,
        Value::String(_) => ValueKind::String,
        Value::ByteBuffer(_) => ValueKind::Bytes,
        _ => ValueKind::Other,
    }
}

pub fn type_name_for_error(v: &Value) -> &'static str {
    match value_kind(v) {
        ValueKind::Table => "table",
        ValueKind::Object => "object",
        ValueKind::String => "string",
        ValueKind::Bytes => "bytes",
        ValueKind::Other => "value",
    }
}

fn err(path: &PathBuf, ext: &str, ty: &str, msg: impl Into<String>) -> String {
    format!(
        "Cannot save {} as {} (file: {}): {}",
        ty,
        ext.to_uppercase(),
        format_path_for_error(path),
        msg.into()
    )
}

fn incompatible(path: &PathBuf, ext: &str, ty: &str) -> String {
    format!(
        "Unsupported output format '{}' for type {} (file: {})",
        ext,
        ty,
        format_path_for_error(path)
    )
}

pub fn save_value(data: &Value, path_arg: &Value) -> Result<String, String> {
    let path = path_from_value(path_arg)?;
    let ext = extension_lower(&path);
    let ty = type_name_for_error(data);
    let kind = value_kind(data);

    let out_path = match (kind, ext.as_str()) {
        (ValueKind::Table, "csv") => {
            let Value::Table(table_rc) = data else {
                unreachable!()
            };
            let table = materialize_table(table_rc);
            let mut out = path.clone();
            if out.extension().is_none() {
                out.set_extension("csv");
            }
            let resolved = crate::file_io::path_input::resolve_local_path(&out)?;
            ensure_parent(&resolved)?;
            write_table_csv(&table, &resolved).map_err(|e| err(&path, &ext, ty, e))?;
            resolved
        }
        (ValueKind::Table, "json") => {
            let json = table_to_json_string(data)?;
            write_bytes_to_path(&path, json.as_bytes())?
        }
        (ValueKind::Table, "sqlite" | "db") => {
            let Value::Table(table_rc) = data else {
                unreachable!()
            };
            let table = materialize_table(table_rc);
            let name = table.name.clone().ok_or_else(|| {
                err(
                    &path,
                    &ext,
                    ty,
                    "table has no name; assign table to a variable before save",
                )
            })?;
            let mut out = path.clone();
            if out.extension().is_none() {
                out.set_extension("sqlite");
            }
            let resolved = crate::file_io::path_input::resolve_local_path(&out)?;
            ensure_parent(&resolved)?;
            export_single_table(&table, &resolved, &name)
                .map_err(|e| err(&path, &ext, ty, e))?;
            resolved
        }
        (ValueKind::Object, "json") => {
            let json =
                value_to_json_string_pretty(data).map_err(|e| err(&path, &ext, ty, e.message()))?;
            write_bytes_to_path(&path, json.as_bytes())?
        }
        (ValueKind::Object, "toml") => {
            let toml =
                value_to_toml_string_pretty(data).map_err(|e| err(&path, &ext, ty, e.message()))?;
            write_bytes_to_path(&path, toml.as_bytes())?
        }
        (ValueKind::Object, "yaml" | "yml") => {
            let yaml = value_to_yaml_string(data).map_err(|e| err(&path, &ext, ty, e.message()))?;
            write_bytes_to_path(&path, yaml.as_bytes())?
        }
        (ValueKind::Object, "xml") => {
            let xml = value_to_xml_string(data).map_err(|e| err(&path, &ext, ty, e.message()))?;
            write_bytes_to_path(&path, xml.as_bytes())?
        }
        (ValueKind::String, "txt" | "text" | "md" | "html") => {
            let Value::String(s) = data else {
                unreachable!()
            };
            write_bytes_to_path(&path, s.as_bytes())?
        }
        (ValueKind::Bytes, "bin") => {
            let Value::ByteBuffer(b) = data else {
                unreachable!()
            };
            let slice = &b.bytes[b.offset..b.offset + b.len];
            write_bytes_to_path(&path, slice)?
        }
        (ValueKind::Table, _) => return Err(incompatible(&path, &ext, ty)),
        (ValueKind::Object, _) => return Err(incompatible(&path, &ext, ty)),
        (ValueKind::String, _) => return Err(incompatible(&path, &ext, ty)),
        (ValueKind::Bytes, _) => return Err(incompatible(&path, &ext, ty)),
        (ValueKind::Other, _) => return Err(incompatible(&path, &ext, ty)),
    };

    Ok(out_path.to_string_lossy().to_string())
}

fn ensure_parent(resolved: &PathBuf) -> Result<(), String> {
    if let Some(parent) = resolved.parent() {
        if !parent.as_os_str().is_empty() && parent != std::path::Path::new(".") && !parent.exists()
        {
            return Err(format!(
                "Directory does not exist: {}",
                parent.to_string_lossy()
            ));
        }
    }
    Ok(())
}

fn materialize_table(table_rc: &Rc<RefCell<Table>>) -> Table {
    crate::vm::vm::with_current_stores(|store, heap| {
        let table_ref = table_rc.borrow();
        if table_ref.is_view() {
            table_ref.materialize_with(|id| crate::vm::store_convert::load_value(id, store, heap))
        } else {
            table_ref.clone()
        }
    })
}

fn table_to_json_string(data: &Value) -> Result<String, String> {
    let Value::Table(table_rc) = data else {
        return Err("expected table".to_string());
    };
    let table = materialize_table(table_rc);
    let headers = table.headers().clone();
    let mut rows_json = Vec::new();
    if let Some(rr) = table.rows_ref() {
        for row in rr.iter() {
            let pairs: Vec<(Value, Value)> = headers
                .iter()
                .zip(row.iter())
                .map(|(h, v)| (Value::String(h.clone()), v.clone()))
                .collect();
            rows_json.push(Value::Object(Rc::new(RefCell::new(ObjectKind::Inline(pairs)))));
        }
    }
    let arr = Value::Array(Rc::new(RefCell::new(rows_json)));
    value_to_json_string_pretty(&arr).map_err(|e| e.message())
}
