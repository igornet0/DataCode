// Path manipulation native functions

use crate::common::value::Value;
use std::path::PathBuf;

fn vfs_lookup_key(path: &PathBuf) -> Option<String> {
    crate::dcp::normalize_vfs_path(path).ok()
}

fn vfs_exists(path: &PathBuf) -> bool {
    let Some(vfs) = crate::dcp::get_dcp_vfs() else {
        return path.exists();
    };
    let Some(key) = vfs_lookup_key(path) else {
        return false;
    };
    vfs.exists(&key)
}

fn vfs_is_file(path: &PathBuf) -> bool {
    let Some(vfs) = crate::dcp::get_dcp_vfs() else {
        return path.is_file();
    };
    let Some(key) = vfs_lookup_key(path) else {
        return false;
    };
    vfs.contains_file(&key)
}

fn vfs_is_dir(path: &PathBuf) -> bool {
    let Some(vfs) = crate::dcp::get_dcp_vfs() else {
        return path.is_dir();
    };
    let Some(key) = vfs_lookup_key(path) else {
        return false;
    };
    vfs.is_dir(&key)
}

// Helper function to safely get parent path
pub fn safe_path_parent(path: &PathBuf) -> Option<PathBuf> {
    if crate::dcp::dcp_vfs_active() {
        let key = vfs_lookup_key(path)?;
        if key.is_empty() {
            return None;
        }
        let parent_key = key.rfind('/').map(|idx| key[..idx].to_string())?;
        return Some(crate::dcp::logical_path_to_pathbuf(
            if parent_key.is_empty() { "." } else { &parent_key },
        ));
    }

    use crate::websocket::{get_use_ve, get_user_session_path};

    if !get_use_ve() {
        return path.parent().map(|p| p.to_path_buf());
    }

    let session_path = get_user_session_path()?;

    let session_path_normalized = match session_path.canonicalize() {
        Ok(p) => p,
        Err(_) => session_path.clone(),
    };

    let parent = path.parent()?;

    let parent_normalized = match parent.canonicalize() {
        Ok(p) => p,
        Err(_) => parent.to_path_buf(),
    };

    if parent_normalized.starts_with(&session_path_normalized) {
        Some(parent.to_path_buf())
    } else {
        None
    }
}

pub fn native_path(args: &[Value]) -> Value {
    if args.is_empty() {
        return Value::Path(PathBuf::new());
    }

    match &args[0] {
        Value::String(s) => Value::Path(PathBuf::from(s)),
        Value::Path(p) => Value::Path(p.clone()),
        _ => Value::Path(PathBuf::from(args[0].to_string())),
    }
}

pub fn native_path_name(args: &[Value]) -> Value {
    if args.is_empty() {
        return Value::String(String::new());
    }

    match &args[0] {
        Value::Path(p) => {
            if let Some(name) = p.file_name() {
                Value::String(name.to_string_lossy().to_string())
            } else {
                Value::String(String::new())
            }
        }
        _ => Value::String(String::new()),
    }
}

pub fn native_path_parent(args: &[Value]) -> Value {
    if args.is_empty() {
        return Value::Null;
    }

    match &args[0] {
        Value::Path(p) => match safe_path_parent(p) {
            Some(parent) => Value::Path(parent),
            None => Value::Null,
        },
        _ => Value::Null,
    }
}

pub fn native_path_exists(args: &[Value]) -> Value {
    if args.is_empty() {
        return Value::Bool(false);
    }

    match &args[0] {
        Value::Path(p) => Value::Bool(vfs_exists(p)),
        _ => Value::Bool(false),
    }
}

pub fn native_path_is_file(args: &[Value]) -> Value {
    if args.is_empty() {
        return Value::Bool(false);
    }

    match &args[0] {
        Value::Path(p) => Value::Bool(vfs_is_file(p)),
        _ => Value::Bool(false),
    }
}

pub fn native_path_is_dir(args: &[Value]) -> Value {
    if args.is_empty() {
        return Value::Bool(false);
    }

    match &args[0] {
        Value::Path(p) => Value::Bool(vfs_is_dir(p)),
        _ => Value::Bool(false),
    }
}

pub fn native_path_extension(args: &[Value]) -> Value {
    if args.is_empty() {
        return Value::String(String::new());
    }

    match &args[0] {
        Value::Path(p) => {
            if let Some(ext) = p.extension() {
                Value::String(ext.to_string_lossy().to_string())
            } else {
                Value::String(String::new())
            }
        }
        _ => Value::String(String::new()),
    }
}

pub fn native_path_stem(args: &[Value]) -> Value {
    if args.is_empty() {
        return Value::String(String::new());
    }

    match &args[0] {
        Value::Path(p) => {
            if let Some(stem) = p.file_stem() {
                Value::String(stem.to_string_lossy().to_string())
            } else {
                Value::String(String::new())
            }
        }
        _ => Value::String(String::new()),
    }
}

pub fn native_path_len(args: &[Value]) -> Value {
    if args.is_empty() {
        return Value::Number(0.0);
    }

    match &args[0] {
        Value::Path(p) => {
            let len = p.to_string_lossy().len();
            Value::Number(len as f64)
        }
        _ => Value::Number(0.0),
    }
}
