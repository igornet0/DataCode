//! WebSocket message routing: DCP binary execution, custom @ws_route handlers, and SMB.

use crate::common::value::Value;
use crate::dcp::{clear_dcp_vfs, DcpDecoder, DcpVfs};
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

/// Per-connection context passed to the router.
pub struct ClientContext {
    pub smb_manager: Arc<Mutex<SmbManager>>,
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

pub fn error_json(success: bool, error: &str) -> String {
    serde_json::to_string(&ExecuteResponse {
        success,
        output: String::new(),
        error: Some(error.to_string()),
        sqlite_db: None,
    })
    .unwrap_or_else(|_| {
        format!(
            "{{\"success\":false,\"error\":{}}}",
            serde_json::to_string(error).unwrap_or_default()
        )
    })
}

fn extract_message_type(payload: &serde_json::Value) -> Option<String> {
    payload
        .get("type")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
}

/// Decode a binary DCP package, mount assets in VFS, execute code, return JSON response.
pub fn dispatch_dcp(data: &[u8], ctx: &ClientContext) -> String {
    let decoded = match DcpDecoder::decode(data) {
        Ok(pkg) => pkg,
        Err(e) => return error_json(false, &e.to_string()),
    };

    let vfs = match DcpVfs::from_assets(decoded.assets) {
        Ok(v) => std::sync::Arc::new(v),
        Err(e) => return error_json(false, &e.to_string()),
    };
    crate::dcp::set_dcp_vfs(Some(vfs));

    let config = app_config();
    let response = execute_code(
        &decoded.code,
        &ctx.smb_manager,
        ctx.build_model,
        config.execute_permission_policy,
    );
    clear_dcp_vfs();
    serde_json::to_string(&response)
        .unwrap_or_else(|e| error_json(false, &format!("Failed to serialize response: {e}")))
}

/// Route an incoming JSON text message to custom or built-in SMB handlers.
pub fn dispatch_message(raw_json: &str, ctx: &ClientContext) -> String {
    let payload: serde_json::Value = match serde_json::from_str(raw_json) {
        Ok(v) => v,
        Err(e) => {
            return error_json(
                false,
                &format!(
                    "JSON parse error: {e}. Expected JSON with field type for SMB/custom routes"
                ),
            );
        }
    };

    let msg_type = match extract_message_type(&payload) {
        Some(t) => t,
        None => {
            return error_json(
                false,
                "JSON request requires field type. DCP packages must be sent as binary frames",
            );
        }
    };

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
            &format!("Message type '{msg_type}' is disabled by ws_app configuration"),
        );
    }

    match handle_builtin(&msg_type, &payload, ctx) {
        Ok(json) => json,
        Err(e) => error_json(false, &e),
    }
}

fn handle_builtin(
    msg_type: &str,
    payload: &serde_json::Value,
    ctx: &ClientContext,
) -> Result<String, String> {
    match msg_type {
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
                            content: Some(format!("base64:{base64_content}")),
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
        other => Err(format!(
            "Unknown message type: {other}. DCP execution requires a binary frame"
        )),
    }
}

fn str_field(payload: &serde_json::Value, key: &str) -> Result<String, String> {
    payload
        .get(key)
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
        .ok_or_else(|| format!("Missing or invalid '{key}' field"))
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
                            env::temp_dir().join(format!("datacode_export_{timestamp}.db"));

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
                        eprintln!("⚠️  Table export error: {e}");
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
