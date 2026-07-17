//! WebSocket message routing: custom @ws_route handlers and built-in types.

use crate::common::value::Value;
use crate::run_with_vm_with_policy;
use crate::sqlite_export;
use crate::vm::PermissionPolicy;
use crate::websocket::app::{app_config, call_route_handler, route_handler_index};
use crate::websocket::config::is_builtin_disabled;
use crate::websocket::output_capture::OutputCapture;
use crate::websocket::smb::{SmbConnection, SmbManager};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::cell::RefCell;
use std::collections::HashMap;
use std::env;
use std::fs;
use std::rc::Rc;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use super::get_user_session_path;

#[derive(Debug, Serialize, Deserialize)]
pub struct ExecuteResponse {
    pub success: bool,
    pub output: String,
    pub error: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sqlite_db: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct SmbConnectResponse {
    success: bool,
    message: String,
    error: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct SmbListFilesResponse {
    success: bool,
    files: Vec<String>,
    error: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct SmbReadFileResponse {
    success: bool,
    content: Option<String>,
    error: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct UploadFileResponse {
    success: bool,
    message: String,
    error: Option<String>,
}

/// Per-connection context passed to the router.
pub struct ClientContext {
    pub smb_manager: Arc<Mutex<SmbManager>>,
    pub use_ve: bool,
    pub build_model: bool,
}

fn json_to_value(v: &serde_json::Value) -> Value {
    match v {
        serde_json::Value::Null => Value::Null,
        serde_json::Value::Bool(b) => Value::Bool(*b),
        serde_json::Value::Number(n) => Value::Number(n.as_f64().unwrap_or(0.0)),
        serde_json::Value::String(s) => Value::String(s.clone()),
        serde_json::Value::Array(arr) => {
            let items: Vec<Value> = arr.iter().map(json_to_value).collect();
            Value::Array(Rc::new(RefCell::new(items)))
        }
        serde_json::Value::Object(map) => {
            let hm: HashMap<String, Value> = map
                .iter()
                .map(|(k, v)| (k.clone(), json_to_value(v)))
                .collect();
            Value::legacy_object(hm)
        }
    }
}

fn value_to_serde_json(v: &Value) -> Result<serde_json::Value, ()> {
    Ok(match v {
        Value::Null => json!(null),
        Value::Bool(b) => json!(b),
        Value::Number(n) => json!(*n),
        Value::String(s) => json!(s),
        Value::Array(rc) => {
            let arr = rc.borrow();
            let out: Result<Vec<_>, _> = arr.iter().map(value_to_serde_json).collect();
            json!(out?)
        }
        Value::Object(rc) => {
            let map = rc.borrow();
            let out: Result<HashMap<String, _>, _> = map
                .str_key_pairs()
                .into_iter()
                .map(|(k, v)| value_to_serde_json(v).map(|j| (k, j)))
                .collect();
            json!(out?)
        }
        _ => json!(v.to_string()),
    })
}

fn value_to_json_string(v: &Value) -> String {
    value_to_serde_json(v)
        .ok()
        .and_then(|j| serde_json::to_string(&j).ok())
        .unwrap_or_else(|| "{\"success\":false,\"error\":\"Failed to serialize handler response\"}".to_string())
}

fn error_json(success: bool, error: &str) -> String {
    serde_json::to_string(&ExecuteResponse {
        success,
        output: String::new(),
        error: Some(error.to_string()),
        sqlite_db: None,
    })
    .unwrap_or_else(|_| format!("{{\"success\":false,\"error\":{}}}", serde_json::to_string(error).unwrap_or_default()))
}

fn extract_message_type(payload: &serde_json::Value) -> Option<String> {
    if let Some(t) = payload.get("type").and_then(|v| v.as_str()) {
        return Some(t.to_string());
    }
    // Legacy: {"code": "..."} without type
    if payload.get("code").is_some() && payload.get("type").is_none() {
        return Some("execute".to_string());
    }
    None
}

/// Route an incoming JSON text message to custom or built-in handlers.
pub fn dispatch_message(raw_json: &str, ctx: &ClientContext) -> String {
    let payload: serde_json::Value = match serde_json::from_str(raw_json) {
        Ok(v) => v,
        Err(e) => {
            return error_json(
                false,
                &format!(
                    "Ошибка парсинга JSON: {}. Ожидается JSON с полем type или code",
                    e
                ),
            );
        }
    };

    let msg_type = match extract_message_type(&payload) {
        Some(t) => t,
        None => {
            return error_json(
                false,
                "Ошибка парсинга запроса. Ожидается JSON с полем type (или legacy code)",
            );
        }
    };

    // Custom @ws_route handler (including overrides of built-in types)
    if let Some(handler_idx) = route_handler_index(&msg_type) {
        let request_value = json_to_value(&payload);
        return match call_route_handler(handler_idx, request_value) {
            Ok(val) => value_to_json_string(&val),
            Err(e) => error_json(false, &e),
        };
    }

    let config = app_config();

    if is_builtin_disabled(&config, &msg_type) {
        return error_json(
            false,
            &format!("Message type '{}' is disabled by ws_app configuration", msg_type),
        );
    }

    match handle_builtin(&msg_type, &payload, ctx, &config) {
        Ok(json) => json,
        Err(e) => error_json(false, &e),
    }
}

fn handle_builtin(
    msg_type: &str,
    payload: &serde_json::Value,
    ctx: &ClientContext,
    config: &crate::websocket::config::WsAppConfig,
) -> Result<String, String> {
    match msg_type {
        "execute" => {
            let code = payload
                .get("code")
                .and_then(|v| v.as_str())
                .ok_or("execute request requires 'code' field")?;
            let response = execute_code(code, &ctx.smb_manager, ctx.build_model, config.execute_permission_policy);
            serde_json::to_string(&response).map_err(|e| e.to_string())
        }
        "smb_connect" => {
            let ip = str_field(payload, "ip")?;
            let login = str_field(payload, "login")?;
            let password = str_field(payload, "password")?;
            let domain = str_field(payload, "domain")?;
            let share_name = str_field(payload, "share_name")?;
            let connection = SmbConnection::new(ip, login, password, domain, share_name);
            let result = ctx.smb_manager.lock().unwrap().connect(connection);
            let response = match result {
                Ok(msg) => SmbConnectResponse {
                    success: true,
                    message: msg,
                    error: None,
                },
                Err(e) => SmbConnectResponse {
                    success: false,
                    message: String::new(),
                    error: Some(e),
                },
            };
            serde_json::to_string(&response).map_err(|e| e.to_string())
        }
        "smb_list_files" => {
            let share_name = str_field(payload, "share_name")?;
            let path = str_field(payload, "path")?;
            let result = ctx
                .smb_manager
                .lock()
                .unwrap()
                .list_files(&share_name, &path, None, true);
            let response = match result {
                Ok(files) => SmbListFilesResponse {
                    success: true,
                    files,
                    error: None,
                },
                Err(e) => SmbListFilesResponse {
                    success: false,
                    files: Vec::new(),
                    error: Some(e),
                },
            };
            serde_json::to_string(&response).map_err(|e| e.to_string())
        }
        "smb_read_file" => {
            let share_name = str_field(payload, "share_name")?;
            let file_path = str_field(payload, "file_path")?;
            let result = ctx
                .smb_manager
                .lock()
                .unwrap()
                .read_file(&share_name, &file_path);
            let response = match result {
                Ok(content) => match String::from_utf8(content.clone()) {
                    Ok(text) => SmbReadFileResponse {
                        success: true,
                        content: Some(text),
                        error: None,
                    },
                    Err(_) => {
                        use base64::Engine;
                        let base64_content =
                            base64::engine::general_purpose::STANDARD.encode(&content);
                        SmbReadFileResponse {
                            success: true,
                            content: Some(format!("base64:{}", base64_content)),
                            error: None,
                        }
                    }
                },
                Err(e) => SmbReadFileResponse {
                    success: false,
                    content: None,
                    error: Some(e),
                },
            };
            serde_json::to_string(&response).map_err(|e| e.to_string())
        }
        "upload_file" => {
            let filename = str_field(payload, "filename")?;
            let content = str_field(payload, "content")?;
            let response = handle_upload_file(&filename, &content, ctx.use_ve);
            serde_json::to_string(&response).map_err(|e| e.to_string())
        }
        other => Err(format!("Unknown message type: {}", other)),
    }
}

fn str_field(payload: &serde_json::Value, key: &str) -> Result<String, String> {
    payload
        .get(key)
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
        .ok_or_else(|| format!("Missing or invalid '{}' field", key))
}

fn handle_upload_file(filename: &str, content: &str, use_ve: bool) -> UploadFileResponse {
    if !use_ve {
        return UploadFileResponse {
            success: false,
            message: String::new(),
            error: Some("Режим --use-ve не включен".to_string()),
        };
    }
    let Some(session_path) = get_user_session_path() else {
        return UploadFileResponse {
            success: false,
            message: String::new(),
            error: Some("Сессия пользователя не найдена".to_string()),
        };
    };
    let file_path = session_path.join(filename);
    let Some(parent) = file_path.parent() else {
        return UploadFileResponse {
            success: false,
            message: String::new(),
            error: Some("Некорректный путь к файлу".to_string()),
        };
    };
    if let Err(e) = fs::create_dir_all(parent) {
        return UploadFileResponse {
            success: false,
            message: String::new(),
            error: Some(format!("Ошибка создания директории: {}", e)),
        };
    }
    let file_content_result = if content.starts_with("base64:") {
        use base64::Engine;
        base64::engine::general_purpose::STANDARD
            .decode(&content[7..])
            .map_err(|e| format!("Ошибка декодирования base64: {}", e))
    } else {
        Ok(content.as_bytes().to_vec())
    };
    match file_content_result {
        Ok(file_content) => match fs::write(&file_path, file_content) {
            Ok(_) => UploadFileResponse {
                success: true,
                message: format!("Файл {} успешно загружен", filename),
                error: None,
            },
            Err(e) => UploadFileResponse {
                success: false,
                message: String::new(),
                error: Some(format!("Ошибка записи файла: {}", e)),
            },
        },
        Err(e) => UploadFileResponse {
            success: false,
            message: String::new(),
            error: Some(e),
        },
    }
}

fn execute_code(
    code: &str,
    smb_manager: &Arc<Mutex<SmbManager>>,
    build_model: bool,
    policy: PermissionPolicy,
) -> ExecuteResponse {
    crate::vm::file_ops::set_smb_manager(smb_manager.clone());

    let output_capture = OutputCapture::new();
    output_capture.set_capture(true);

    let result = if policy == PermissionPolicy::AllowAll {
        crate::run_with_vm(code)
    } else {
        run_with_vm_with_policy(code, policy)
    };

    let output = output_capture.get_output();
    output_capture.set_capture(false);

    match result {
        Ok((_, mut vm)) => {
            let mut sqlite_db = None;

            if build_model {
                match sqlite_export::get_global_tables(&mut vm) {
                    Ok(tables) if !tables.is_empty() => {
                        let timestamp = SystemTime::now()
                            .duration_since(UNIX_EPOCH)
                            .unwrap()
                            .as_nanos();
                        let temp_db_path =
                            env::temp_dir().join(format!("datacode_export_{}.db", timestamp));

                        if sqlite_export::export_to_sqlite(
                            &mut vm,
                            temp_db_path.to_str().unwrap(),
                            false,
                        )
                        .is_ok()
                        {
                            if let Ok(db_bytes) = fs::read(&temp_db_path) {
                                use base64::Engine;
                                sqlite_db = Some(
                                    base64::engine::general_purpose::STANDARD.encode(&db_bytes),
                                );
                            }
                            let _ = fs::remove_file(&temp_db_path);
                        }
                    }
                    Ok(_) => {}
                    Err(e) => {
                        eprintln!("⚠️  Ошибка проверки таблиц: {}", e);
                    }
                }
            }

            ExecuteResponse {
                success: true,
                output,
                error: None,
                sqlite_db,
            }
        }
        Err(e) => ExecuteResponse {
            success: false,
            output,
            error: Some(e.to_string()),
            sqlite_db: None,
        },
    }
}
