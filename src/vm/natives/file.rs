// File operations native functions

use crate::common::value::Value;
use std::path::PathBuf;
use std::rc::Rc;
use std::cell::RefCell;
use std::fs;
use std::env;
use chrono::Utc;

pub fn native_now(_args: &[Value]) -> Value {
    // Возвращаем текущее время в формате RFC3339 (ISO 8601)
    // Формат: YYYY-MM-DDTHH:MM:SSZ
    let now = Utc::now();
    Value::String(now.format("%Y-%m-%dT%H:%M:%SZ").to_string())
}

pub fn native_getcwd(_args: &[Value]) -> Value {
    use crate::websocket::get_use_ve;
    
    // Если включен режим use_ve, возвращаем пустой путь для безопасности
    if get_use_ve() {
        Value::Path(PathBuf::new()) // Пустой путь в режиме use_ve для безопасности
    } else {
        // Возвращаем текущую рабочую директорию
        match env::current_dir() {
            Ok(path) => Value::Path(path),
            Err(_) => Value::Path(PathBuf::new()), // При ошибке возвращаем пустой путь
        }
    }
}

/// Безопасное разрешение пути относительно папки сессии в режиме --use-ve
pub fn resolve_path_in_session(path: &PathBuf) -> Result<PathBuf, String> {
    use crate::websocket::{get_user_session_path, get_use_ve};
    
    if !get_use_ve() {
        // В обычном режиме просто возвращаем путь как есть
        return Ok(path.clone());
    }
    
    let session_path = match get_user_session_path() {
        Some(p) => p,
        None => return Err("Session path not available".to_string()),
    };
    
    // Нормализуем session_path для корректного сравнения
    let session_path_normalized = match session_path.canonicalize() {
        Ok(p) => p,
        Err(_) => {
            // Если канонизация не удалась, используем исходный путь
            // но нормализуем его (убираем лишние компоненты)
            session_path.clone()
        },
    };
    
    // Проверяем на directory traversal атаки
    let path_str = path.to_string_lossy().to_string();
    if path_str.contains("..") {
        return Err("Path traversal not allowed in --use-ve mode".to_string());
    }
    
    // Обрабатываем пустой путь и "." как текущую директорию сессии
    if path_str.is_empty() || path_str == "." {
        return Ok(session_path_normalized.clone());
    }
    
    // Проверяем абсолютные пути
    if path.is_absolute() {
        // Для абсолютных путей проверяем, что они внутри сессии
        // Пытаемся канонизировать оба пути для корректного сравнения
        let path_normalized = if path.exists() {
            match path.canonicalize() {
                Ok(p) => p,
                Err(_) => {
                    // Если канонизация не удалась, но путь существует,
                    // проверяем через starts_with с исходными путями
                    if path.starts_with(&session_path) || path.starts_with(&session_path_normalized) {
                        path.clone()
                    } else {
                        return Err("Access outside session directory not allowed".to_string());
                    }
                },
            }
        } else {
            // Если файл не существует, пытаемся канонизировать родительскую директорию
            // для проверки безопасности
            if let Some(parent) = path.parent() {
                match parent.canonicalize() {
                    Ok(p) => {
                        if p.starts_with(&session_path_normalized) {
                            path.clone()
                        } else {
                            // Дополнительная проверка через сравнение компонентов
                            let parent_components: Vec<_> = p.components().collect();
                            let session_components: Vec<_> = session_path_normalized.components().collect();
                            if parent_components.len() >= session_components.len() {
                                let mut matches = true;
                                for (i, session_comp) in session_components.iter().enumerate() {
                                    if i >= parent_components.len() || parent_components[i] != *session_comp {
                                        matches = false;
                                        break;
                                    }
                                }
                                if matches {
                                    path.clone()
                                } else {
                                    return Err("Path resolved outside session directory".to_string());
                                }
                            } else {
                                return Err("Path resolved outside session directory".to_string());
                            }
                        }
                    },
                    Err(_) => {
                        // Если родительская директория не может быть канонизирована,
                        // проверяем через starts_with
                        if parent.starts_with(&session_path) || parent.starts_with(&session_path_normalized) {
                            path.clone()
                        } else {
                            return Err("Access outside session directory not allowed".to_string());
                        }
                    },
                }
            } else {
                // Нет родительской директории - проверяем напрямую
                if path.starts_with(&session_path) || path.starts_with(&session_path_normalized) {
                    path.clone()
                } else {
                    return Err("Access outside session directory not allowed".to_string());
                }
            }
        };
        
        // Финальная проверка безопасности через канонизированные пути
        if path_normalized.starts_with(&session_path_normalized) {
            Ok(path_normalized)
        } else if path_normalized.starts_with(&session_path) {
            // Если session_path не был канонизирован, но путь начинается с него
            Ok(path_normalized)
        } else {
            // Дополнительная проверка через сравнение компонентов
            let path_components: Vec<_> = path_normalized.components().collect();
            let session_components: Vec<_> = session_path_normalized.components().collect();
            if path_components.len() >= session_components.len() {
                let mut matches = true;
                for (i, session_comp) in session_components.iter().enumerate() {
                    if i >= path_components.len() || path_components[i] != *session_comp {
                        matches = false;
                        break;
                    }
                }
                if matches {
                    Ok(path_normalized)
                } else {
                    Err("Path resolved outside session directory".to_string())
                }
            } else {
                Err("Access outside session directory not allowed".to_string())
            }
        }
    } else {
        // Относительный путь разрешаем относительно папки сессии
        let resolved = session_path.join(path);
        
        // Нормализуем путь и проверяем, что он все еще внутри сессии
        let normalized = if resolved.exists() {
            match resolved.canonicalize() {
                Ok(p) => p,
                Err(_) => resolved,
            }
        } else {
            // Если файл не существует, проверяем родительскую директорию
            if let Some(parent) = resolved.parent() {
                match parent.canonicalize() {
                    Ok(p) => {
                        if p.starts_with(&session_path_normalized) {
                            resolved
                        } else {
                            return Err("Path resolved outside session directory".to_string());
                        }
                    },
                    Err(_) => {
                        if parent.starts_with(&session_path) {
                            resolved
                        } else {
                            return Err("Path resolved outside session directory".to_string());
                        }
                    },
                }
            } else {
                resolved
            }
        };
        
        if normalized.starts_with(&session_path_normalized) || normalized.starts_with(&session_path) {
            Ok(normalized)
        } else {
            Err("Path resolved outside session directory".to_string())
        }
    }
}

pub fn native_list_files(args: &[Value]) -> Value {
    if args.is_empty() {
        return Value::Array(Rc::new(RefCell::new(Vec::new())));
    }

    // Первый аргумент - путь к директории
    let dir_path = match &args[0] {
        Value::Path(p) => p.clone(),
        Value::String(s) => PathBuf::from(s),
        _ => return Value::Array(Rc::new(RefCell::new(Vec::new()))),
    };

    let dir_path_str = dir_path.to_string_lossy().to_string();

    // Проверяем, является ли это SMB путем (lib://)
    if dir_path_str.starts_with("lib://") {
        // Извлекаем имя шары и путь к директории
        let path_without_prefix = &dir_path_str[6..]; // Убираем "lib://"
        let parts: Vec<&str> = path_without_prefix.splitn(2, '/').collect();
        
        if parts.is_empty() {
            return Value::Array(Rc::new(RefCell::new(Vec::new())));
        }
        
        let share_name = parts[0];
        let dir_path_on_share = if parts.len() > 1 { parts[1] } else { "" };
        
        // Получаем SmbManager из thread-local storage
        if let Some(smb_manager) = crate::vm::file_ops::get_smb_manager() {
            match smb_manager.lock().unwrap().list_files(share_name, dir_path_on_share) {
                Ok(files) => {
                    let file_values: Vec<Value> = files.iter()
                        .map(|f| {
                            // Конструируем полный путь для SMB
                            let full_path = if dir_path_on_share.is_empty() {
                                format!("lib://{}/{}", share_name, f)
                            } else {
                                format!("lib://{}/{}/{}", share_name, dir_path_on_share, f)
                            };
                            Value::Path(PathBuf::from(full_path))
                        })
                        .collect();
                    Value::Array(Rc::new(RefCell::new(file_values)))
                }
                Err(_) => Value::Array(Rc::new(RefCell::new(Vec::new()))),
            }
        } else {
            Value::Array(Rc::new(RefCell::new(Vec::new())))
        }
    } else {
        // Обычная локальная директория
        // Разрешаем путь относительно папки сессии в режиме --use-ve
        let resolved_path = match resolve_path_in_session(&dir_path) {
            Ok(p) => p,
            Err(err_msg) => {
                // При ошибке безопасности сохраняем сообщение об ошибке
                use crate::websocket::set_native_error;
                set_native_error(err_msg);
                return Value::Array(Rc::new(RefCell::new(Vec::new())));
            }
        };
        
        if !resolved_path.exists() || !resolved_path.is_dir() {
            return Value::Array(Rc::new(RefCell::new(Vec::new())));
        }
        
        match fs::read_dir(&resolved_path) {
            Ok(entries) => {
                let mut files = Vec::new();
                for entry in entries {
                    if let Ok(entry) = entry {
                        let entry_path = entry.path();
                        if let Some(file_name) = entry_path.file_name() {
                            if let Some(name_str) = file_name.to_str() {
                                // Пропускаем служебные файлы
                                if !name_str.starts_with(".") && name_str != ".DS_Store" {
                                    files.push(Value::Path(entry_path));
                                }
                            }
                        }
                    }
                }
                Value::Array(Rc::new(RefCell::new(files)))
            }
            Err(_) => Value::Array(Rc::new(RefCell::new(Vec::new()))),
        }
    }
}

