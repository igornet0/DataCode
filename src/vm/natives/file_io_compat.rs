//! Additive-compat aliases: `read()` / `save()` alongside `read_file` / `read_file_bin`.

use crate::common::value::Value;
use crate::file_io::{read_value, save_value};

pub fn native_read(args: &[Value]) -> Value {
    match read_value(args) {
        Ok(v) => v,
        Err(msg) => {
            crate::websocket::set_native_error(msg);
            Value::Null
        }
    }
}

pub fn native_save(args: &[Value]) -> Value {
    if args.len() < 2 {
        crate::websocket::set_native_error(
            "save() expects data and path arguments".to_string(),
        );
        return Value::Null;
    }
    match save_value(&args[0], &args[1]) {
        Ok(path) => Value::String(path),
        Err(msg) => {
            crate::websocket::set_native_error(msg);
            Value::Null
        }
    }
}
