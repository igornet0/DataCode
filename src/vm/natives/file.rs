// File operations native functions

use crate::common::value::Value;
use std::path::PathBuf;
use std::rc::Rc;
use std::cell::RefCell;
use std::fs;
use std::env;
use chrono::Utc;
use regex::Regex;

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

/// Безопасное форматирование пути для сообщений об ошибках
/// В режиме --use-ve преобразует полный путь в относительный
pub fn format_path_for_error(path: &PathBuf) -> String {
    use crate::websocket::{get_use_ve, get_user_session_path};
    
    if get_use_ve() {
        if let Some(session_path) = get_user_session_path() {
            // Канонизируем оба пути для корректного сравнения
            let canonical_session = session_path.canonicalize().ok().unwrap_or(session_path.clone());
            let canonical_path = path.canonicalize().ok().unwrap_or(path.clone());
            
            // Проверяем, начинается ли путь с пути сессии
            if let Ok(stripped) = canonical_path.strip_prefix(&canonical_session) {
                // Формируем относительный путь с префиксом ./
                let relative = stripped.to_string_lossy().to_string();
                if relative.is_empty() || relative == "." {
                    "./".to_string()
                } else {
                    // Убираем начальные слеши и добавляем ./
                    let trimmed = relative.trim_start_matches(|c| c == '/' || c == '\\');
                    if trimmed.is_empty() {
                        "./".to_string()
                    } else {
                        format!("./{}", trimmed)
                    }
                }
            } else {
                // Путь вне сессии - возвращаем как есть (не канонизированный для сохранения оригинального формата)
                path.to_string_lossy().to_string()
            }
        } else {
            // Нет пути сессии - возвращаем как есть
            path.to_string_lossy().to_string()
        }
    } else {
        // Не режим --use-ve - возвращаем полный путь
        path.to_string_lossy().to_string()
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

/// Конвертирует glob паттерн в regex
/// Поддерживает: *, ?, [abc], | (альтернативы), экранирование специальных символов
fn glob_to_regex(glob: &str) -> String {
    // Если паттерн содержит |, разделяем на альтернативы
    if glob.contains('|') {
        let alternatives: Vec<&str> = glob.split('|').collect();
        let regex_alternatives: Vec<String> = alternatives
            .iter()
            .map(|alt| glob_to_regex_single(alt.trim()))
            .collect();
        return format!("^({})$", regex_alternatives.join("|"));
    }
    
    // Для одиночного паттерна добавляем якоря
    format!("^{}$", glob_to_regex_single(glob))
}

/// Конвертирует один glob паттерн (без |) в regex
/// Не добавляет якоря ^ и $ - они добавляются в вызывающей функции
fn glob_to_regex_single(glob: &str) -> String {
    let mut regex = String::new();
    let mut chars = glob.chars().peekable();
    
    while let Some(ch) = chars.next() {
        match ch {
            '*' => {
                // Проверяем на ** (рекурсивный glob)
                if chars.peek() == Some(&'*') {
                    chars.next(); // Пропускаем второй *
                    regex.push_str(".*");
                } else {
                    regex.push_str(".*");
                }
            }
            '?' => {
                regex.push('.');
            }
            '.' | '(' | ')' | '[' | ']' | '{' | '}' | '+' | '^' | '$' | '\\' => {
                // Экранируем специальные символы regex (кроме |, который обрабатывается отдельно)
                regex.push('\\');
                regex.push(ch);
            }
            _ => {
                regex.push(ch);
            }
        }
    }
    
    regex
}

/// Рекурсивно обходит директорию и собирает все файлы
/// Применяет regex фильтрацию к именам файлов, если regex задан
fn list_files_recursive(
    dir: &PathBuf,
    regex: Option<&Regex>,
    session_path: &PathBuf,
) -> Vec<Value> {
    let mut files = Vec::new();
    
    // Проверяем безопасность пути в режиме --use-ve
    let resolved_dir = match resolve_path_in_session(dir) {
        Ok(p) => p,
        Err(_) => return files, // Пропускаем недоступные пути
    };
    
    if !resolved_dir.exists() || !resolved_dir.is_dir() {
        return files;
    }
    
    match fs::read_dir(&resolved_dir) {
        Ok(entries) => {
            for entry in entries {
                if let Ok(entry) = entry {
                    let entry_path = entry.path();
                    
                    // Пропускаем служебные файлы
                    if let Some(file_name) = entry_path.file_name() {
                        if let Some(name_str) = file_name.to_str() {
                            if name_str.starts_with(".") || name_str == ".DS_Store" {
                                continue;
                            }
                        }
                    }
                    
                    let metadata = match entry.metadata() {
                        Ok(m) => m,
                        Err(_) => continue,
                    };
                    
                    if metadata.is_file() {
                        // Для файлов проверяем regex фильтрацию
                        if let Some(re) = regex {
                            if let Some(file_name) = entry_path.file_name() {
                                if let Some(name_str) = file_name.to_str() {
                                    if !re.is_match(name_str) {
                                        continue; // Файл не соответствует regex
                                    }
                                } else {
                                    continue; // Невалидное имя файла
                                }
                            } else {
                                continue; // Нет имени файла
                            }
                        }
                        files.push(Value::Path(entry_path));
                    } else if metadata.is_dir() {
                        // Для директорий рекурсивно обходим содержимое
                        let sub_files = list_files_recursive(&entry_path, regex, session_path);
                        files.extend(sub_files);
                    }
                }
            }
        }
        Err(_) => {
            // Игнорируем ошибки чтения директории
        }
    }
    
    files
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

    // Второй аргумент (опциональный) - regex паттерн для фильтрации
    let regex_pattern = if args.len() > 1 {
        match &args[1] {
            Value::String(s) if !s.is_empty() => Some(s.clone()),
            _ => None,
        }
    } else {
        None
    };
    
    // Компилируем regex, если он задан
    let regex = if let Some(pattern) = &regex_pattern {
        // Проверяем, является ли паттерн glob-подобным (содержит * или ?)
        let is_glob = pattern.contains('*') || pattern.contains('?');
        
        let regex_pattern_str = if is_glob {
            // Конвертируем glob в regex
            glob_to_regex(pattern)
        } else {
            // Используем как есть (предполагаем, что это уже regex)
            pattern.clone()
        };
        
        match Regex::new(&regex_pattern_str) {
            Ok(re) => Some(re),
            Err(e) => {
                // При ошибке компиляции regex устанавливаем ошибку и возвращаем пустой массив
                use crate::websocket::set_native_error;
                set_native_error(format!("Invalid regex pattern '{}': {}", pattern, e));
                return Value::Array(Rc::new(RefCell::new(Vec::new())));
            }
        }
    } else {
        None
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
            let regex_str = regex_pattern.as_deref();
            match smb_manager.lock().unwrap().list_files(share_name, dir_path_on_share, regex_str, true) {
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
        
        // Получаем путь сессии для проверки безопасности в рекурсивной функции
        use crate::websocket::{get_user_session_path, get_use_ve};
        let session_path = if get_use_ve() {
            get_user_session_path().unwrap_or_else(|| PathBuf::from("."))
        } else {
            PathBuf::from(".")
        };
        
        // Используем рекурсивную функцию для обхода директории
        let files = list_files_recursive(&resolved_path, regex.as_ref(), &session_path);
        Value::Array(Rc::new(RefCell::new(files)))
    }
}

