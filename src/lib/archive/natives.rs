//! VM native functions for Archive API.

use crate::archive::archive::Archive;
use crate::common::value::Value;
use crate::file_io::path_from_value;
use std::cell::RefCell;
use std::path::PathBuf;
use std::rc::Rc;

fn archive_from_args(args: &[Value]) -> Result<Rc<RefCell<Archive>>, String> {
    if args.is_empty() {
        return Err("archive method expects archive as first argument".to_string());
    }
    match &args[0] {
        Value::Archive(rc) => Ok(Rc::clone(rc)),
        _ => Err("Expected archive object".to_string()),
    }
}

/// `archive(path)` — open archive and return Archive object.
pub fn native_archive(args: &[Value]) -> Value {
    if args.len() != 1 {
        crate::websocket::set_native_error("archive() expects one path argument".to_string());
        return Value::Null;
    }
    let path: PathBuf = match path_from_value(&args[0]) {
        Ok(p) => p,
        Err(e) => {
            crate::websocket::set_native_error(e);
            return Value::Null;
        }
    };
    match Archive::open(path) {
        Ok(arch) => Value::Archive(Rc::new(RefCell::new(arch))),
        Err(e) => {
            crate::websocket::set_native_error(e);
            Value::Null
        }
    }
}

/// `zip.read(path)` — read entry with auto type detection.
pub fn native_archive_read(args: &[Value]) -> Value {
    if args.len() < 2 {
        crate::websocket::set_native_error("read() expects path argument".to_string());
        return Value::Null;
    }
    let path = match &args[1] {
        Value::String(s) => s.clone(),
        Value::Path(p) => p.to_string_lossy().into_owned(),
        _ => {
            crate::websocket::set_native_error("read() path must be string or path".to_string());
            return Value::Null;
        }
    };
    let rc = match archive_from_args(args) {
        Ok(r) => r,
        Err(e) => {
            crate::websocket::set_native_error(e);
            return Value::Null;
        }
    };
    let result = {
        let mut arch = rc.borrow_mut();
        arch.read(&path)
    };
    match result {
        Ok(v) => v,
        Err(e) => {
            crate::websocket::set_native_error(e);
            Value::Null
        }
    }
}

/// `zip.read_text(path)` — read entry as UTF-8 string.
pub fn native_archive_read_text(args: &[Value]) -> Value {
    if args.len() < 2 {
        crate::websocket::set_native_error("read_text() expects path argument".to_string());
        return Value::Null;
    }
    let path = match &args[1] {
        Value::String(s) => s.clone(),
        Value::Path(p) => p.to_string_lossy().into_owned(),
        _ => {
            crate::websocket::set_native_error(
                "read_text() path must be string or path".to_string(),
            );
            return Value::Null;
        }
    };
    let rc = match archive_from_args(args) {
        Ok(r) => r,
        Err(e) => {
            crate::websocket::set_native_error(e);
            return Value::Null;
        }
    };
    let result = {
        let mut arch = rc.borrow_mut();
        arch.read_text(&path)
    };
    match result {
        Ok(v) => v,
        Err(e) => {
            crate::websocket::set_native_error(e);
            Value::Null
        }
    }
}

/// `zip.extract(dest)` — extract all entries.
pub fn native_archive_extract(args: &[Value]) -> Value {
    if args.len() < 2 {
        crate::websocket::set_native_error("extract() expects destination path".to_string());
        return Value::Null;
    }
    let dest = match path_from_value(&args[1]) {
        Ok(p) => p,
        Err(e) => {
            crate::websocket::set_native_error(e);
            return Value::Null;
        }
    };
    let rc = match archive_from_args(args) {
        Ok(r) => r,
        Err(e) => {
            crate::websocket::set_native_error(e);
            return Value::Null;
        }
    };
    let result = {
        let mut arch = rc.borrow_mut();
        arch.extract(&dest)
    };
    match result {
        Ok(()) => Value::Null,
        Err(e) => {
            crate::websocket::set_native_error(e);
            Value::Null
        }
    }
}

/// `zip.close()` — release resources.
pub fn native_archive_close(args: &[Value]) -> Value {
    let rc = match archive_from_args(args) {
        Ok(r) => r,
        Err(e) => {
            crate::websocket::set_native_error(e);
            return Value::Null;
        }
    };
    rc.borrow_mut().close();
    Value::Null
}
