use futures_util::{SinkExt, StreamExt};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use tokio::net::{TcpListener, TcpStream};
use tokio_tungstenite::{accept_async, tungstenite::Message};

pub mod app;
pub mod config;
pub mod natives;
pub mod output_capture;
pub mod router;
pub mod smb;
pub mod ws_natives;

use app::bootstrap_app;
use crate::dcp::clear_dcp_session;
use router::{ClientContext, dispatch_dcp, dispatch_message};
use smb::SmbManager;

thread_local! {
    static USER_SESSION_PATH: std::cell::RefCell<Option<PathBuf>> = std::cell::RefCell::new(None);
    static USE_VE_FLAG: std::cell::RefCell<bool> = std::cell::RefCell::new(false);
    static NATIVE_ERROR: std::cell::RefCell<Option<String>> = std::cell::RefCell::new(None);
}

pub fn set_user_session_path(path: Option<PathBuf>) {
    USER_SESSION_PATH.with(|p| *p.borrow_mut() = path);
}

pub fn get_user_session_path() -> Option<PathBuf> {
    USER_SESSION_PATH.with(|p| p.borrow().clone())
}

pub fn set_use_ve(use_ve: bool) {
    USE_VE_FLAG.with(|f| *f.borrow_mut() = use_ve);
}

pub fn get_use_ve() -> bool {
    USE_VE_FLAG.with(|f| *f.borrow())
}

pub fn set_native_error(msg: String) {
    NATIVE_ERROR.with(|e| *e.borrow_mut() = Some(msg));
}

pub fn take_native_error() -> Option<String> {
    NATIVE_ERROR.with(|e| e.borrow_mut().take())
}

/// Start the WebSocket server on the given address.
pub async fn start_server(
    address: &str,
    _use_ve: bool,
    build_model: bool,
    app_file: Option<&str>,
    base_dir: Option<&str>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    if let Some(app) = app_file {
        let resolved = app::resolve_app_path(app, base_dir);
        bootstrap_app(resolved.to_str().unwrap_or(app), base_dir)
            .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> { e.into() })?;
    }

    let listener = TcpListener::bind(address).await?;
    println!("🚀 DataCode WebSocket Server running on {address}");
    println!("📡 Waiting for connections...");
    if app_file.is_some() {
        println!("📜 ws_app mode: @ws_route handlers + websocket.configure()");
    }
    println!("💡 Send a binary WebSocket frame with a DCP package (.dcp bytes, magic DCPK)");
    println!("💡 DCP assets are served from in-memory VFS (no temp_sessions on disk)");
    println!("💡 Response: {{\"success\": true/false, \"output\": \"...\", \"error\": null/\"...\"}}");
    if build_model {
        println!("💡 --build_model: successful runs include sqlite_db (base64) in the response");
    }
    println!("💡 SMB control messages still use JSON text: smb_connect, smb_list_files, smb_read_file");
    println!();

    // Each client runs on its own multi-thread runtime (dedicated OS thread).
    // A LocalSet was previously used for thread-locals, but LocalSet forbids
    // `tokio::task::block_in_place`, which MongoDB/MSSQL/browser bridges need.
    loop {
        let (stream, addr) = match listener.accept().await {
            Ok((s, a)) => (s, a),
            Err(e) => {
                eprintln!("❌ Accept error: {e}");
                continue;
            }
        };

        println!("✅ New connection from {addr}");
        std::thread::Builder::new()
            .name(format!("datacode-ws-{addr}"))
            .spawn(move || {
                let rt = tokio::runtime::Builder::new_multi_thread()
                    .worker_threads(2)
                    .enable_all()
                    .thread_name(format!("ws-client-{addr}"))
                    .build()
                    .expect("failed to create per-client tokio runtime");
                rt.block_on(handle_client(stream, build_model));
            })
            .expect("failed to spawn WebSocket client thread");
    }
}

async fn handle_client(stream: TcpStream, build_model: bool) {
    let ws_stream = match accept_async(stream).await {
        Ok(ws) => ws,
        Err(e) => {
            eprintln!("❌ WebSocket handshake error: {e}");
            return;
        }
    };

    let (mut write, mut read) = ws_stream.split();
    let smb_manager = Arc::new(Mutex::new(SmbManager::new()));

    crate::vm::file_ops::set_smb_manager(smb_manager.clone());
    set_use_ve(true);
    clear_dcp_session();

    let ctx = ClientContext {
        smb_manager: smb_manager.clone(),
        build_model,
    };

    while let Some(msg) = read.next().await {
        match msg {
            Ok(Message::Binary(data)) => {
                let response_json = dispatch_dcp(&data, &ctx);
                if let Err(e) = write.send(Message::Text(response_json)).await {
                    eprintln!("❌ Failed to send response: {e}");
                    break;
                }
            }
            Ok(Message::Text(text)) => {
                let response_json = dispatch_message(&text, &ctx);
                if let Err(e) = write.send(Message::Text(response_json)).await {
                    eprintln!("❌ Failed to send response: {e}");
                    break;
                }
            }
            Ok(Message::Close(_)) => {
                println!("🔌 Client disconnected");
                break;
            }
            Ok(Message::Ping(data)) => {
                if let Err(e) = write.send(Message::Pong(data)).await {
                    eprintln!("❌ Failed to send Pong: {e}");
                    break;
                }
            }
            Err(e) => {
                eprintln!("❌ Read error: {e}");
                break;
            }
            _ => {}
        }
    }

    cleanup_client(&smb_manager);
}

fn cleanup_client(smb_manager: &Arc<Mutex<SmbManager>>) {
    let mut manager = smb_manager.lock().unwrap();
    let shares: Vec<String> = manager.list_connections();
    for share in shares {
        let _ = manager.disconnect(&share);
    }
    drop(manager);

    crate::vm::file_ops::clear_smb_manager();
    clear_dcp_session();
    crate::web::cleanup_all();
    set_use_ve(false);
}
