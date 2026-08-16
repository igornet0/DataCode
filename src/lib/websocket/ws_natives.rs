//! Built-in `ws` module for safe access to the current DCP WebSocket session.

use std::collections::HashMap;
use std::rc::Rc;
use std::cell::RefCell;

use crate::common::value::{ByteBuffer, Value};
use crate::dcp::{
    apply_source_table_columns, arrow_ipc_to_table, dcp_session_active, get_dcp_content_assets,
    get_dcp_metadata, get_dcp_tables, get_dcp_vfs, validate_table_asset_refs,
};
use crate::plot::Image;
use crate::websocket::set_native_error;

fn arg_string(args: &[Value], fn_name: &str, arg_name: &str) -> Option<String> {
    match args.first() {
        Some(Value::String(s)) => Some(s.clone()),
        _ => {
            set_native_error(format!(
                "TypeError: {}() requires a string {arg_name}",
                fn_name
            ));
            None
        }
    }
}

fn require_session() -> bool {
    if dcp_session_active() {
        true
    } else {
        set_native_error("ws: no active DCP session".to_string());
        false
    }
}

/// `ws.tables()` — names of ARROW_TABLE sections in the current DCP package.
pub fn native_ws_tables(_args: &[Value]) -> Value {
    if !require_session() {
        return Value::Null;
    }
    let Some(tables) = get_dcp_tables() else {
        return Value::Array(Rc::new(RefCell::new(Vec::new())));
    };
    let names: Vec<Value> = tables.names().into_iter().map(Value::String).collect();
    Value::Array(Rc::new(RefCell::new(names)))
}

/// `ws.has_table(name)`
pub fn native_ws_has_table(args: &[Value]) -> Value {
    if !require_session() {
        return Value::Null;
    }
    let Some(name) = arg_string(args, "has_table", "name") else {
        return Value::Null;
    };
    let has = get_dcp_tables()
        .map(|t| t.has(&name))
        .unwrap_or(false);
    Value::Bool(has)
}

/// `ws.source_table(name, columns?)`
pub fn native_ws_source_table(args: &[Value]) -> Value {
    if !require_session() {
        return Value::Null;
    }
    let Some(name) = arg_string(args, "source_table", "name") else {
        return Value::Null;
    };
    let columns = args.get(1);

    let Some(tables) = get_dcp_tables() else {
        set_native_error(format!("Table '{name}' not found in DCP"));
        return Value::Null;
    };

    let Some(bytes) = tables.get(&name) else {
        set_native_error(format!("Table '{name}' not found in DCP"));
        return Value::Null;
    };

    let table = match arrow_ipc_to_table(bytes) {
        Ok(t) => t,
        Err(e) => {
            set_native_error(e);
            return Value::Null;
        }
    };

    let table = match apply_source_table_columns(table, columns) {
        Ok(t) => t,
        Err(e) => {
            set_native_error(e);
            return Value::Null;
        }
    };

    if let Some(store) = get_dcp_content_assets() {
        if let Err(e) = validate_table_asset_refs(&name, &table, &store) {
            set_native_error(e);
            return Value::Null;
        }
    }

    Value::Table(Rc::new(RefCell::new(table)))
}

/// `ws.assets()` — logical asset paths (no server filesystem paths).
pub fn native_ws_assets(_args: &[Value]) -> Value {
    if !require_session() {
        return Value::Null;
    }
    let paths = get_dcp_vfs()
        .map(|vfs| vfs.asset_paths())
        .unwrap_or_default();
    let items: Vec<Value> = paths.into_iter().map(Value::String).collect();
    Value::Array(Rc::new(RefCell::new(items)))
}

/// `ws.has_asset(name)`
pub fn native_ws_has_asset(args: &[Value]) -> Value {
    if !require_session() {
        return Value::Null;
    }
    let Some(name) = arg_string(args, "has_asset", "name") else {
        return Value::Null;
    };
    let has = get_dcp_vfs()
        .map(|vfs| vfs.contains_file(&name))
        .unwrap_or(false);
    Value::Bool(has)
}

/// `ws.metadata()` — custom DCP metadata key/value strings.
pub fn native_ws_metadata(_args: &[Value]) -> Value {
    if !require_session() {
        return Value::Null;
    }
    let meta = get_dcp_metadata().unwrap_or_default();
    let map: HashMap<String, Value> = meta
        .into_iter()
        .map(|(k, v)| (k, Value::String(v)))
        .collect();
    Value::legacy_object(map)
}

/// `ws.metadata_get(key)`
pub fn native_ws_metadata_get(args: &[Value]) -> Value {
    if !require_session() {
        return Value::Null;
    }
    let Some(key) = arg_string(args, "metadata_get", "key") else {
        return Value::Null;
    };
    match get_dcp_metadata().and_then(|m| m.get(&key).cloned()) {
        Some(v) => Value::String(v),
        None => Value::Null,
    }
}

/// `ws.content_assets()` — content-addressed asset ids (SHA-256 hex).
pub fn native_ws_content_assets(_args: &[Value]) -> Value {
    if !require_session() {
        return Value::Null;
    }
    let ids = get_dcp_content_assets()
        .map(|s| s.ids())
        .unwrap_or_default();
    let items: Vec<Value> = ids.into_iter().map(Value::String).collect();
    Value::Array(Rc::new(RefCell::new(items)))
}

/// `ws.has_content_asset(id)`
pub fn native_ws_has_content_asset(args: &[Value]) -> Value {
    if !require_session() {
        return Value::Null;
    }
    let Some(id) = arg_string(args, "has_content_asset", "id") else {
        return Value::Null;
    };
    let has = get_dcp_content_assets()
        .map(|s| s.contains(&id))
        .unwrap_or(false);
    Value::Bool(has)
}

/// `ws.content_asset(id)` — bytes, or `image` when kind/mime is image.
pub fn native_ws_content_asset(args: &[Value]) -> Value {
    if !require_session() {
        return Value::Null;
    }
    let Some(id) = arg_string(args, "content_asset", "id") else {
        return Value::Null;
    };
    let Some(store) = get_dcp_content_assets() else {
        set_native_error(format!("content asset '{id}' not found"));
        return Value::Null;
    };
    let Some(asset) = store.get(&id) else {
        set_native_error(format!("content asset '{id}' not found"));
        return Value::Null;
    };

    let is_image = asset.meta.kind == "image" || asset.meta.mime_type.starts_with("image/");
    if is_image {
        match Image::from_bytes(&asset.data) {
            Ok(img) => Value::Image(Rc::new(RefCell::new(img))),
            Err(e) => {
                set_native_error(format!("Failed to decode image asset '{id}': {e}"));
                Value::Null
            }
        }
    } else {
        Value::ByteBuffer(ByteBuffer::from_vec(asset.data.clone()))
    }
}

/// `ws.package_info()` — safe summary without host/port/paths/secrets.
pub fn native_ws_package_info(_args: &[Value]) -> Value {
    if !require_session() {
        return Value::Null;
    }

    let table_count = get_dcp_tables().map(|t| t.count()).unwrap_or(0);
    let asset_count = get_dcp_vfs().map(|v| v.asset_count()).unwrap_or(0);
    let content_asset_count = get_dcp_content_assets().map(|s| s.len()).unwrap_or(0);
    let has_metadata = get_dcp_metadata().map(|m| !m.is_empty()).unwrap_or(false);

    let mut map = HashMap::new();
    map.insert("table_count".to_string(), Value::Number(table_count as f64));
    map.insert("asset_count".to_string(), Value::Number(asset_count as f64));
    map.insert(
        "content_asset_count".to_string(),
        Value::Number(content_asset_count as f64),
    );
    map.insert("has_code".to_string(), Value::Bool(true));
    map.insert("has_metadata".to_string(), Value::Bool(has_metadata));
    Value::legacy_object(map)
}
