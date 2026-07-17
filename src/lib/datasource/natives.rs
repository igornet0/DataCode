//! VM native functions for DataSource API.

use crate::common::table::Table;
use crate::common::value::Value;
use crate::datasource::config::parse_config;
use crate::datasource::datasource::DataSource;
use crate::datasource::get_table::response_to_table;
use crate::datasource::request::{
    parse_get_table_spec, parse_request_spec, parse_send_table_spec,
};
use crate::datasource::response::DataSourceResponse;
use crate::file_io::{path_from_value, write_bytes_to_path};
use crate::file_io::value_serde::{parse_json_str, value_to_json_string_pretty};
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

fn datasource_from_args(args: &[Value]) -> Result<Rc<RefCell<DataSource>>, String> {
    if args.is_empty() {
        return Err("datasource method expects datasource as first argument".to_string());
    }
    match &args[0] {
        Value::DataSource(rc) => Ok(Rc::clone(rc)),
        _ => Err("Expected datasource object".to_string()),
    }
}

fn response_from_args(args: &[Value]) -> Result<Rc<RefCell<DataSourceResponse>>, String> {
    if args.is_empty() {
        return Err("response method expects response as first argument".to_string());
    }
    match &args[0] {
        Value::DataSourceResponse(rc) => Ok(Rc::clone(rc)),
        _ => Err("Expected response object".to_string()),
    }
}

fn set_error(msg: impl Into<String>) -> Value {
    crate::websocket::set_native_error(msg.into());
    Value::Null
}

fn map_err(e: crate::datasource::error::DataSourceError) -> Value {
    set_error(e.display())
}

/// `datasource(config)` — create DataSource from config object.
pub fn native_datasource(args: &[Value]) -> Value {
    if args.len() != 1 {
        return set_error("datasource() expects one config object argument");
    }
    let cfg = match parse_config(&args[0]) {
        Ok(c) => c,
        Err(e) => return map_err(e),
    };
    match DataSource::new(cfg) {
        Ok(ds) => Value::DataSource(Rc::new(RefCell::new(ds))),
        Err(e) => map_err(e),
    }
}

pub fn native_datasource_request(args: &[Value]) -> Value {
    if args.len() < 2 {
        return set_error("request() expects a spec object argument");
    }
    let rc = match datasource_from_args(args) {
        Ok(r) => r,
        Err(e) => return set_error(e),
    };
    let spec = match parse_request_spec(&args[1]) {
        Ok(s) => s,
        Err(e) => return map_err(e),
    };
    let ds = rc.borrow();
    match ds.request(&spec) {
        Ok(resp) => Value::DataSourceResponse(Rc::new(RefCell::new(resp))),
        Err(e) => map_err(e),
    }
}

pub fn native_datasource_get_table(args: &[Value]) -> Value {
    if args.len() < 2 {
        return set_error("get_table() expects a spec object argument");
    }
    let rc = match datasource_from_args(args) {
        Ok(r) => r,
        Err(e) => return set_error(e),
    };
    let spec = match parse_get_table_spec(&args[1]) {
        Ok(s) => s,
        Err(e) => return map_err(e),
    };
    let ds = rc.borrow();
    match ds.get_table(&spec) {
        Ok(table) => Value::Table(Rc::new(RefCell::new(table))),
        Err(e) => map_err(e),
    }
}

pub fn native_datasource_send_table(args: &[Value]) -> Value {
    let rc = match datasource_from_args(args) {
        Ok(r) => r,
        Err(e) => return set_error(e),
    };
    let mut spec = match parse_send_table_spec(
        args.get(1),
        if args.len() > 2 { args.get(2) } else { None },
    ) {
        Ok(s) => s,
        Err(e) => return map_err(e),
    };
    if let Some(Value::Table(table_rc)) = spec.table.as_ref() {
        let needs_materialize = table_rc.borrow().is_view();
        if needs_materialize {
            let table_ref = table_rc.borrow();
            use crate::vm::table_ops;
            use crate::vm::vm::with_current_stores;
            let owned = with_current_stores(|store, heap| {
                let rows = table_ops::materialize_rows(&table_ref, store, heap);
                Table::from_data(rows, Some(table_ref.headers().to_vec()))
            });
            drop(table_ref);
            spec.table = Some(Value::Table(Rc::new(RefCell::new(owned))));
        }
    }
    let ds = rc.borrow();
    match ds.send_table(&spec) {
        Ok(()) => Value::Null,
        Err(e) => map_err(e),
    }
}

pub fn native_datasource_connect(args: &[Value]) -> Value {
    let rc = match datasource_from_args(args) {
        Ok(r) => r,
        Err(e) => return set_error(e),
    };
    let ds = rc.borrow();
    match ds.connect() {
        Ok(()) => Value::Null,
        Err(e) => map_err(e),
    }
}

pub fn native_datasource_disconnect(args: &[Value]) -> Value {
    let rc = match datasource_from_args(args) {
        Ok(r) => r,
        Err(e) => return set_error(e),
    };
    rc.borrow().disconnect();
    Value::Null
}

pub fn native_datasource_ping(args: &[Value]) -> Value {
    let rc = match datasource_from_args(args) {
        Ok(r) => r,
        Err(e) => return set_error(e),
    };
    let ds = rc.borrow();
    match ds.ping() {
        Ok(ok) => Value::Bool(ok),
        Err(e) => map_err(e),
    }
}

pub fn native_datasource_test(args: &[Value]) -> Value {
    let rc = match datasource_from_args(args) {
        Ok(r) => r,
        Err(e) => return set_error(e),
    };
    let ds = rc.borrow();
    match ds.test() {
        Ok(v) => v,
        Err(e) => map_err(e),
    }
}

pub fn native_datasource_clone(args: &[Value]) -> Value {
    let rc = match datasource_from_args(args) {
        Ok(r) => r,
        Err(e) => return set_error(e),
    };
    let ds = rc.borrow();
    match ds.clone_handle() {
        Ok(new_ds) => Value::DataSource(Rc::new(RefCell::new(new_ds))),
        Err(e) => map_err(e),
    }
}

pub fn native_response_json(args: &[Value]) -> Value {
    let rc = match response_from_args(args) {
        Ok(r) => r,
        Err(e) => return set_error(e),
    };
    let resp = rc.borrow();
    let text = match resp.text() {
        Ok(t) => t,
        Err(e) => return set_error(e),
    };
    match parse_json_str(&text) {
        Ok(v) => v,
        Err(e) => set_error(e.message()),
    }
}

pub fn native_response_table(args: &[Value]) -> Value {
    let rc = match response_from_args(args) {
        Ok(r) => r,
        Err(e) => return set_error(e),
    };
    let resp = rc.borrow();
    let spec = crate::datasource::request::GetTableSpec::default();
    match response_to_table(&resp, &spec) {
        Ok(table) => Value::Table(Rc::new(RefCell::new(table))),
        Err(e) => map_err(e),
    }
}

pub fn native_response_csv(args: &[Value]) -> Value {
    native_response_table(args)
}

pub fn native_response_save(args: &[Value]) -> Value {
    if args.len() < 2 {
        return set_error("save() expects a path argument");
    }
    let rc = match response_from_args(args) {
        Ok(r) => r,
        Err(e) => return set_error(e),
    };
    let path = match path_from_value(&args[1]) {
        Ok(p) => p,
        Err(e) => return set_error(e),
    };
    let resp = rc.borrow();
    if let Err(e) = write_bytes_to_path(&path, &resp.body) {
        return set_error(e);
    }
    Value::String(path.to_string_lossy().into_owned())
}

pub fn native_response_save_text(args: &[Value]) -> Value {
    if args.len() < 2 {
        return set_error("save_text() expects a path argument");
    }
    let rc = match response_from_args(args) {
        Ok(r) => r,
        Err(e) => return set_error(e),
    };
    let path = match path_from_value(&args[1]) {
        Ok(p) => p,
        Err(e) => return set_error(e),
    };
    let resp = rc.borrow();
    let text = match resp.text() {
        Ok(t) => t,
        Err(e) => return set_error(e),
    };
    if let Err(e) = std::fs::write(&path, text) {
        return set_error(e.to_string());
    }
    Value::String(path.to_string_lossy().into_owned())
}

pub fn native_response_save_json(args: &[Value]) -> Value {
    if args.len() < 2 {
        return set_error("save_json() expects a path argument");
    }
    let rc = match response_from_args(args) {
        Ok(r) => r,
        Err(e) => return set_error(e),
    };
    let path = match path_from_value(&args[1]) {
        Ok(p) => p,
        Err(e) => return set_error(e),
    };
    let resp = rc.borrow();
    let text = match resp.text() {
        Ok(t) => t,
        Err(e) => return set_error(e),
    };
    let value = match parse_json_str(&text) {
        Ok(v) => v,
        Err(e) => return set_error(e.message()),
    };
    let pretty = match value_to_json_string_pretty(&value) {
        Ok(s) => s,
        Err(e) => return set_error(e.message()),
    };
    if let Err(e) = std::fs::write(&path, pretty) {
        return set_error(e.to_string());
    }
    Value::String(path.to_string_lossy().into_owned())
}

/// Build headers object for response property access.
pub fn response_headers_value(headers: &HashMap<String, String>) -> Value {
    use crate::common::value::ObjectKind;
    let mut m = HashMap::new();
    for (k, v) in headers {
        m.insert(k.clone(), Value::String(v.clone()));
    }
    Value::Object(Rc::new(RefCell::new(ObjectKind::legacy(m))))
}

/// Build body value for response property access.
pub fn response_body_value(resp: &DataSourceResponse) -> Value {
    Value::ByteBuffer(resp.byte_buffer())
}

/// Convert response bytes to table with optional spec (for csv/table methods with args).
pub fn response_table_with_spec(
    resp: &DataSourceResponse,
    spec: &crate::datasource::request::GetTableSpec,
) -> Result<Table, crate::datasource::error::DataSourceError> {
    response_to_table(resp, spec)
}
