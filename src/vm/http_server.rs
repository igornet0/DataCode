// HTTP server (datacode-server)
// Stage 1: GET / returns fixed string. Stage 2: load app.dc with @route handlers, dispatch to VM.
// VM is not Send, so we run the server on the current thread and store VM in thread-local.

use crate::common::value::Value;
use crate::vm::cli::HttpServerConfig;
use axum::{
    body::Body,
    extract::connect_info::ConnectInfo,
    extract::Request,
    http::StatusCode,
    response::IntoResponse,
    Router,
};
use tower::util::MapRequestLayer;
use hyper::server::conn::http1::Builder as Http1Builder;
use hyper_util::rt::TokioIo;
use hyper_util::service::TowerToHyperService;
use tokio::net::TcpListener;
use tokio::task::LocalSet;
use std::collections::HashMap;
use std::net::SocketAddr;
use std::path::Path;
use std::rc::Rc;
use std::cell::RefCell;
use tokio::runtime::Builder;

/// Route entry: (method, path_template, handler_index). Path template may contain {name} segments.
type RouteEntry = (String, String, usize);

thread_local! {
    static HTTP_VM: RefCell<Option<crate::vm::Vm>> = RefCell::new(None);
    static ROUTE_TABLE: RefCell<Option<Vec<RouteEntry>>> = RefCell::new(None);
}

type ResponsePayload = (u16, Vec<(String, String)>, Vec<u8>);

/// Match request path against template like /users/{id}, return param map or None.
fn match_path_template(template: &str, path: &str) -> Option<HashMap<String, String>> {
    let t_segments: Vec<&str> = template.split('/').filter(|s| !s.is_empty()).collect();
    let p_segments: Vec<&str> = path.split('/').filter(|s| !s.is_empty()).collect();
    if t_segments.len() != p_segments.len() {
        return None;
    }
    let mut params = HashMap::new();
    for (t, p) in t_segments.iter().zip(p_segments.iter()) {
        if t.starts_with('{') && t.ends_with('}') {
            let name = &t[1..t.len() - 1];
            params.insert(name.to_string(), (*p).to_string());
        } else if t != p {
            return None;
        }
    }
    Some(params)
}

/// Build Request value for DataCode handler: { method, path, headers, query, body, params }.
fn build_request_value(
    method: &str,
    path: &str,
    headers: &[(String, String)],
    query: &str,
    body: &[u8],
    params: &HashMap<String, String>,
) -> Value {
    let body_str = String::from_utf8_lossy(body).to_string();
    let mut req = HashMap::new();
    req.insert("method".to_string(), Value::String(method.to_string()));
    req.insert("path".to_string(), Value::String(path.to_string()));
    req.insert("query".to_string(), Value::String(query.to_string()));
    let headers_map: HashMap<String, Value> = headers
        .iter()
        .map(|(k, v)| (k.clone(), Value::String(v.clone())))
        .collect();
    req.insert(
        "headers".to_string(),
        Value::Object(Rc::new(RefCell::new(headers_map))),
    );
    req.insert("body".to_string(), Value::String(body_str));
    let params_map: HashMap<String, Value> = params
        .iter()
        .map(|(k, v)| (k.clone(), Value::String(v.clone())))
        .collect();
    req.insert(
        "params".to_string(),
        Value::Object(Rc::new(RefCell::new(params_map))),
    );
    Value::Object(Rc::new(RefCell::new(req)))
}

/// Convert handler return Value to HTTP (status, headers, body).
fn value_to_http_response(v: Value) -> ResponsePayload {
    match &v {
        Value::String(s) => (200, vec![], s.as_bytes().to_vec()),
        Value::Object(rc) => {
            let status = {
                let map = rc.borrow();
                map.get("status")
                    .and_then(|n| {
                        if let Value::Number(x) = n {
                            Some(*x as u16)
                        } else {
                            None
                        }
                    })
                    .unwrap_or(200)
            };
            let headers: Vec<(String, String)> = {
                let map = rc.borrow();
                map.get("headers")
                    .and_then(|h| {
                        if let Value::Object(hrc) = h {
                            let hm = hrc.borrow();
                            Some(
                                hm.iter()
                                    .filter_map(|(k, v)| {
                                        if let Value::String(s) = v {
                                            Some((k.clone(), s.clone()))
                                        } else {
                                            None
                                        }
                                    })
                                    .collect(),
                            )
                        } else {
                            None
                        }
                    })
                    .unwrap_or_default()
            };
            let body = {
                let body_opt = {
                    let map = rc.borrow();
                    map.get("body").and_then(|b| {
                        if let Value::String(s) = b {
                            Some(s.as_bytes().to_vec())
                        } else {
                            None
                        }
                    })
                };
                body_opt.unwrap_or_else(|| value_to_json_bytes(&v).unwrap_or_default())
            };
            (status, headers, body)
        }
        _ => (
            200,
            vec![("content-type".to_string(), "application/json".to_string())],
            value_to_json_bytes(&v).unwrap_or_else(|_| b"{}".to_vec()),
        ),
    }
}

fn value_to_json_bytes(v: &Value) -> Result<Vec<u8>, ()> {
    let j = value_to_serde_json(v)?;
    serde_json::to_vec(&j).map_err(|_| ())
}

fn value_to_serde_json(v: &Value) -> Result<serde_json::Value, ()> {
    use serde_json::json;
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
                .iter()
                .map(|(k, v)| value_to_serde_json(v).map(|j| (k.clone(), j)))
                .collect();
            json!(out?)
        }
        _ => json!(v.to_string()),
    })
}

fn format_size(bytes: usize) -> String {
    if bytes < 1024 {
        format!("{}B", bytes)
    } else if bytes < 1024 * 1024 {
        format!("{:.1}KB", bytes as f64 / 1024.0)
    } else {
        format!("{:.1}MB", bytes as f64 / (1024.0 * 1024.0))
    }
}

fn body_preview(body: &[u8], max_len: usize) -> String {
    let s = String::from_utf8_lossy(body);
    let one_line: String = s.replace('\n', " ").replace('\r', " ").chars().take(max_len).collect();
    if body.len() > max_len {
        format!("{}...", one_line.trim())
    } else {
        one_line.trim().to_string()
    }
}

/// ANSI color codes for terminal output (reset after each colored segment).
const ANSI_RESET: &str = "\x1b[0m";
const ANSI_GREEN: &str = "\x1b[32m";
const ANSI_YELLOW: &str = "\x1b[33m";
const ANSI_RED: &str = "\x1b[31m";

fn status_color(status: u16) -> &'static str {
    if (200..300).contains(&status) {
        ANSI_GREEN
    } else if (500..600).contains(&status) {
        ANSI_RED
    } else {
        ANSI_YELLOW
    }
}

/// Column widths for aligned [RESPONSE] log output (visible chars, not counting ANSI codes).
const COL_IP: usize = 13;
const COL_METHOD: usize = 6;
const COL_PATH: usize = 40;
const COL_STATUS: usize = 12;
const COL_SIZE: usize = 8;
const COL_DURATION: usize = 6; // width for the number, then "ms"
const COL_PREVIEW: usize = 40;

fn truncate_pad_left(s: &str, width: usize) -> String {
    let truncate_at = width.saturating_sub(3);
    let s = if s.chars().count() > width {
        let truncated: String = s.chars().take(truncate_at).collect();
        format!("{}...", truncated)
    } else {
        s.to_string()
    };
    format!("{:<width$}", s, width = width)
}

fn log_response(
    client_ip: &str,
    method: &str,
    path: &str,
    status: u16,
    body_len: usize,
    duration_ms: u64,
    body: &[u8],
) {
    let reason = status_reason(status);
    let size = format_size(body_len);
    let preview = body_preview(body, COL_PREVIEW);
    let color = status_color(status);
    let status_reason = format!("{} {}", status, reason);
    let ip = truncate_pad_left(client_ip, COL_IP);
    let method_pad = format!("{:<width$}", method, width = COL_METHOD);
    let path_pad = truncate_pad_left(path, COL_PATH);
    let status_pad = truncate_pad_left(&status_reason, COL_STATUS);
    let size_pad = format!("{:>width$}", size, width = COL_SIZE);
    let duration_pad = format!("{:>width$}ms", duration_ms, width = COL_DURATION);
    let preview_pad = truncate_pad_left(&preview, COL_PREVIEW);
    println!(
        "[RESPONSE] {} → {}{}{} {} | {}{}{} | {} | {} | {}",
        ip,
        color,
        method_pad,
        ANSI_RESET,
        path_pad,
        color,
        status_pad,
        ANSI_RESET,
        size_pad,
        duration_pad,
        preview_pad
    );
}

fn status_reason(status: u16) -> &'static str {
    match status {
        200 => "OK",
        201 => "Created",
        204 => "No Content",
        400 => "Bad Request",
        401 => "Unauthorized",
        403 => "Forbidden",
        404 => "Not Found",
        405 => "Method Not Allowed",
        500 => "Internal Server Error",
        _ => "Unknown",
    }
}

fn client_ip_from_headers(headers: &axum::http::HeaderMap) -> String {
    if let Some(v) = headers.get("x-real-ip") {
        if let Ok(s) = v.to_str() {
            return s.trim().to_string();
        }
    }
    if let Some(v) = headers.get("x-forwarded-for") {
        if let Ok(s) = v.to_str() {
            return s.split(',').next().map(|x| x.trim()).unwrap_or("-").to_string();
        }
    }
    "-".to_string()
}

async fn root_handler_with_log(
    ConnectInfo(peer_addr): ConnectInfo<SocketAddr>,
    req: Request,
) -> impl IntoResponse {
    let start = std::time::Instant::now();
    let from_headers = client_ip_from_headers(req.headers());
    let client_ip = if from_headers == "-" {
        peer_addr.ip().to_string()
    } else {
        from_headers
    };
    let body = "Hello from DataCode";
    let duration_ms = start.elapsed().as_millis() as u64;
    log_response(
        &client_ip,
        "GET",
        "/",
        200,
        body.len(),
        duration_ms,
        body.as_bytes(),
    );
    body.into_response()
}

async fn vm_handler(ConnectInfo(peer_addr): ConnectInfo<SocketAddr>, req: Request) -> impl IntoResponse {
    let start = std::time::Instant::now();
    let method = req.method().to_string();
    let path = req.uri().path().to_string();
    let query = req.uri().query().unwrap_or("").to_string();
    let from_headers = client_ip_from_headers(req.headers());
    let client_ip = if from_headers == "-" {
        peer_addr.ip().to_string()
    } else {
        from_headers
    };
    let headers_vec: Vec<(String, String)> = req
        .headers()
        .iter()
        .map(|(k, v)| {
            (
                k.as_str().to_string(),
                v.to_str().unwrap_or("").to_string(),
            )
        })
        .collect();
    let body = axum::body::to_bytes(req.into_body(), 1024 * 1024)
        .await
        .unwrap_or_default();
    let body = body.to_vec();

    let result = HTTP_VM.with(|vm_cell| {
        ROUTE_TABLE.with(|table_cell| {
            let mut vm_opt = vm_cell.borrow_mut();
            let vm = vm_opt.as_mut().ok_or("VM not set")?;
            let table = table_cell.borrow();
            let route_table = table.as_ref().ok_or("Route table not set")?;
            let (handler_idx, params) = route_table
                .iter()
                .find(|(m, template, _)| *m == method && *template == path)
                .map(|(_, _, idx)| (*idx, HashMap::new()))
                .or_else(|| {
                    route_table.iter().find_map(|(m, template, idx)| {
                        if *m == method {
                            match_path_template(template, &path).map(|p| (*idx, p))
                        } else {
                            None
                        }
                    })
                })
                .ok_or_else(|| format!("No route for {} {}", method, path))?;
            let request_value = build_request_value(
                &method,
                &path,
                &headers_vec,
                &query,
                &body,
                &params,
            );
            vm.call_function_by_index(handler_idx, &[request_value])
                .map(value_to_http_response)
                .map_err(|e| e.to_string())
        })
    });

    let duration_ms = start.elapsed().as_millis() as u64;

    match result {
        Ok((status, headers, body)) => {
            HTTP_VM.with(|vm_cell| {
                if let Some(vm) = vm_cell.borrow_mut().as_mut() {
                    vm.reset_stores_and_globals_for_stateless();
                }
            });
            log_response(
                &client_ip,
                &method,
                &path,
                status,
                body.len(),
                duration_ms,
                &body,
            );
            let mut r = axum::response::Response::builder().status(status);
            for (k, v) in &headers {
                if let (Ok(name), Ok(value)) = (
                    axum::http::header::HeaderName::try_from(k.as_str()),
                    axum::http::header::HeaderValue::try_from(v.as_str()),
                ) {
                    r = r.header(name, value);
                }
            }
            r.body(Body::from(body)).unwrap().into_response()
        }
        Err(e) => {
            let body_bytes = e.as_bytes();
            log_response(
                &client_ip,
                &method,
                &path,
                500,
                body_bytes.len(),
                duration_ms,
                body_bytes,
            );
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                [("content-type", "text/plain")],
                e,
            )
                .into_response()
        }
    }
}

/// If host is localhost or 0.0.0.0, print a hint with the machine's LAN IP for connecting from other devices.
fn print_local_network_hint(host: &str, port: u16) {
    if !host.eq_ignore_ascii_case("localhost") && host != "0.0.0.0" {
        return;
    }
    if let Ok(ip) = local_ip_address::local_ip() {
        println!("Local network: http://{}:{}", ip, port);
    }
}

/// Start HTTP server. If config.app_file is set, load app and dispatch to @route handlers.
pub fn start_http_server(config: HttpServerConfig) -> Result<(), String> {
    // SocketAddr::parse() accepts only "ip:port", not hostnames like "localhost"
    let bind_host = if config.host.eq_ignore_ascii_case("localhost") {
        "127.0.0.1"
    } else {
        config.host.as_str()
    };
    let addr: SocketAddr = format!("{}:{}", bind_host, config.port)
        .parse()
        .map_err(|e| format!("Invalid address: {}", e))?;

    if let Some(ref app_path) = config.app_file {
        // Stage 2: load app, build route table, run VM thread + axum
        let source = std::fs::read_to_string(app_path)
            .map_err(|e| format!("Failed to read app file {}: {}", app_path, e))?;
        let base_path = Path::new(app_path)
            .parent()
            .map(|p| p.to_path_buf())
            .ok_or_else(|| "Invalid app path".to_string())?;
        let lib_path = base_path.join("__lib__.dc");
        let lib_path_opt = if lib_path.exists() {
            Some(lib_path.as_path())
        } else {
            None
        };
        crate::vm::file_import::set_base_path(Some(base_path.clone()));
        let (_, vm) = crate::run_with_vm_with_args_and_lib(
            &source,
            None,
            lib_path_opt,
            Some(base_path.as_path()),
            Some(app_path.as_ref()),
        )
        .map_err(|e| format!("Failed to run app: {}", e))?;

        let mut route_table: Vec<RouteEntry> = Vec::new();
        for (i, f) in vm.get_functions().iter().enumerate() {
            if let (Some(ref m), Some(ref p)) = (&f.route_method, &f.route_path) {
                route_table.push((m.clone(), p.clone(), i));
            }
        }
        if route_table.is_empty() {
            return Err("No @route handlers found in app".to_string());
        }

        HTTP_VM.with(|cell| *cell.borrow_mut() = Some(vm));
        ROUTE_TABLE.with(|cell| *cell.borrow_mut() = Some(route_table.clone()));

        println!("HTTP server (DataCode) starting with app: {}", app_path);
        println!("Address: http://{}", addr);
        print_local_network_hint(&config.host, config.port);
        for (m, p, _) in &route_table {
            println!("  {} {}", m, p);
        }
        println!();

        let router = Router::new().fallback(vm_handler).with_state(());

        // Single-threaded runtime so VM (not Send) stays on this thread
        let rt = Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| format!("Failed to create tokio runtime: {}", e))?;
        rt.block_on(async {
            let listener = TcpListener::bind(addr)
                .await
                .map_err(|e| format!("Failed to bind {}: {}", addr, e))?;
            let local_set = Rc::new(LocalSet::new());
            let local_set_clone = local_set.clone();
            #[allow(unreachable_code)]
            local_set
                .run_until(async move {
                    loop {
                        let (stream, peer_addr) = listener
                            .accept()
                            .await
                            .map_err(|e| format!("Accept error: {}", e))?;
                        let layer = MapRequestLayer::new(
                            move |mut req: hyper::Request<Body>| {
                                req.extensions_mut()
                                    .insert(ConnectInfo(peer_addr));
                                req
                            },
                        );
                        let app = TowerToHyperService::new(
                            router.clone().layer(layer).with_state(()),
                        );
                        local_set_clone.spawn_local(async move {
                            let io = TokioIo::new(stream);
                            if let Err(e) = Http1Builder::new().serve_connection(io, app).await {
                                eprintln!("Connection error: {}", e);
                            }
                        });
                    }
                    Ok::<(), String>(())
                })
                .await
                .map_err(|e| format!("HTTP server error: {}", e))
        })
    } else {
        // Stage 1: fixed GET /
        println!("HTTP server (DataCode) starting...");
        println!("Address: http://{}", addr);
        print_local_network_hint(&config.host, config.port);
        println!("GET / -> \"Hello from DataCode\"");
        println!();

        let router = Router::new()
            .route("/", axum::routing::get(root_handler_with_log))
            .with_state(());
        let rt = tokio::runtime::Runtime::new()
            .map_err(|e| format!("Failed to create tokio runtime: {}", e))?;
        rt.block_on(async {
            let listener = TcpListener::bind(addr)
                .await
                .map_err(|e| format!("Failed to bind {}: {}", addr, e))?;
            loop {
                let (stream, peer_addr) = listener
                    .accept()
                    .await
                    .map_err(|e| format!("Accept error: {}", e))?;
                let layer = MapRequestLayer::new(
                    move |mut req: hyper::Request<Body>| {
                        req.extensions_mut()
                            .insert(ConnectInfo(peer_addr));
                        req
                    },
                );
                let app = TowerToHyperService::new(
                    router.clone().layer(layer).with_state(()),
                );
                tokio::spawn(async move {
                    let io = TokioIo::new(stream);
                    if let Err(e) = Http1Builder::new().serve_connection(io, app).await {
                        eprintln!("Connection error: {}", e);
                    }
                });
            }
        })
    }
}
