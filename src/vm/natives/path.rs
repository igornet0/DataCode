// Path manipulation native functions

use crate::common::value::Value;
use std::path::PathBuf;

// Helper function to safely get parent path
pub fn safe_path_parent(path: &PathBuf) -> Option<PathBuf> {
    use crate::websocket::{get_user_session_path, get_use_ve};
    
    if !get_use_ve() {
        // В обычном режиме просто возвращаем parent как есть
        return path.parent().map(|p| p.to_path_buf());
    }
    
    let session_path = match get_user_session_path() {
        Some(p) => p,
        None => return None, // Нет пути сессии - не возвращаем parent
    };
    
    // Нормализуем session_path для корректного сравнения
    let session_path_normalized = match session_path.canonicalize() {
        Ok(p) => p,
        Err(_) => session_path.clone(),
    };
    
    // Получаем parent путь
    let parent = match path.parent() {
        Some(p) => p,
        None => return None,
    };
    
    // Нормализуем parent для корректного сравнения
    let parent_normalized = match parent.canonicalize() {
        Ok(p) => p,
        Err(_) => parent.to_path_buf(),
    };
    
    // Проверяем, что parent находится внутри session_path
    if parent_normalized.starts_with(&session_path_normalized) {
        Some(parent.to_path_buf())
    } else {
        // Если parent находится вне session_path, возвращаем None
        // Это предотвращает выход за пределы виртуальной среды
        None
    }
}

pub fn native_path(args: &[Value]) -> Value {
    if args.is_empty() {
        return Value::Path(PathBuf::new());
    }
    
    match &args[0] {
        Value::String(s) => {
            // Создаем путь из строки
            Value::Path(PathBuf::from(s))
        }
        Value::Path(p) => {
            // Если уже путь, возвращаем копию
            Value::Path(p.clone())
        }
        _ => {
            // Для других типов преобразуем в строку и создаем путь
            Value::Path(PathBuf::from(args[0].to_string()))
        }
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
        Value::Path(p) => {
            // Используем безопасную функцию для получения parent
            match safe_path_parent(p) {
                Some(parent) => Value::Path(parent),
                None => Value::Null,
            }
        }
        _ => Value::Null,
    }
}

pub fn native_path_exists(args: &[Value]) -> Value {
    if args.is_empty() {
        return Value::Bool(false);
    }
    
    match &args[0] {
        Value::Path(p) => Value::Bool(p.exists()),
        _ => Value::Bool(false),
    }
}

pub fn native_path_is_file(args: &[Value]) -> Value {
    if args.is_empty() {
        return Value::Bool(false);
    }
    
    match &args[0] {
        Value::Path(p) => Value::Bool(p.is_file()),
        _ => Value::Bool(false),
    }
}

pub fn native_path_is_dir(args: &[Value]) -> Value {
    if args.is_empty() {
        return Value::Bool(false);
    }
    
    match &args[0] {
        Value::Path(p) => Value::Bool(p.is_dir()),
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

