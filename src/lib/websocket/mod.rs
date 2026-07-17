use futures_util::{SinkExt, StreamExt};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use tokio::net::{TcpListener, TcpStream};
use tokio_tungstenite::{accept_async, tungstenite::Message};

pub mod app;
pub mod config;
pub mod natives;
pub mod output_capture;
pub mod router;
pub mod smb;

use app::bootstrap_app;
use router::{ClientContext, dispatch_message};
use smb::SmbManager;

// Thread-local storage для хранения пути к папке пользователя
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

/// Запустить WebSocket сервер на указанном адресе
pub async fn start_server(
    address: &str,
    use_ve: bool,
    build_model: bool,
    app_file: Option<&str>,
    base_dir: Option<&str>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    if let Some(app) = app_file {
        let resolved = app::resolve_app_path(app, base_dir);
        bootstrap_app(
            resolved.to_str().unwrap_or(app),
            base_dir,
        )
        .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> { e.into() })?;
    }

    let listener = TcpListener::bind(address).await?;
    println!("🚀 DataCode WebSocket Server запущен на {}", address);
    println!("📡 Ожидание подключений...");
    if app_file.is_some() {
        println!("📜 Режим ws_app: маршруты @ws_route + websocket.configure()");
    }
    println!("💡 Отправьте JSON запрос: {{\"type\": \"execute\", \"code\": \"ваш код\"}}");
    println!("💡 Ответ: {{\"success\": true/false, \"output\": \"...\", \"error\": null/\"...\"}}");
    if use_ve {
        println!("💡 Режим --use-ve: upload_file загружает файлы в изолированную сессию клиента");
    }
    if build_model {
        println!("💡 Режим --build_model: при успешном execute в ответе будет поле sqlite_db (base64)");
    }
    println!("💡 Для SMB: сначала smb_connect, затем execute с lib://share_name/path");
    println!();

    if use_ve {
        let temp_sessions_dir = Path::new("src/temp_sessions");
        if !temp_sessions_dir.exists() {
            if let Err(e) = fs::create_dir_all(temp_sessions_dir) {
                eprintln!(
                    "⚠️  Предупреждение: не удалось создать папку temp_sessions: {}",
                    e
                );
            } else {
                println!(
                    "📁 Создана папка для сессий: {}",
                    temp_sessions_dir.display()
                );
            }
        }
    }

    let local_set = tokio::task::LocalSet::new();

    local_set
        .run_until(async {
            loop {
                let (stream, addr) = match listener.accept().await {
                    Ok((s, a)) => (s, a),
                    Err(e) => {
                        eprintln!("❌ Ошибка принятия подключения: {}", e);
                        continue;
                    }
                };

                println!("✅ Новое подключение от {}", addr);
                local_set.spawn_local(handle_client(stream, use_ve, build_model));
            }
        })
        .await;

    Ok(())
}

/// Обработать клиентское подключение
async fn handle_client(stream: TcpStream, use_ve: bool, build_model: bool) {
    let ws_stream = match accept_async(stream).await {
        Ok(ws) => ws,
        Err(e) => {
            eprintln!("❌ Ошибка при принятии WebSocket соединения: {}", e);
            return;
        }
    };

    let (mut write, mut read) = ws_stream.split();
    let smb_manager = Arc::new(Mutex::new(SmbManager::new()));

    crate::vm::file_ops::set_smb_manager(smb_manager.clone());
    set_use_ve(use_ve);

    let user_session_path = if use_ve {
        use std::time::{SystemTime, UNIX_EPOCH};
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let user_id = format!("user_{}", timestamp);
        let user_dir = Path::new("src/temp_sessions").join(&user_id);

        let user_dir_absolute = match user_dir.canonicalize() {
            Ok(p) => p,
            Err(_) => match env::current_dir() {
                Ok(cwd) => cwd.join(&user_dir),
                Err(_) => user_dir,
            },
        };

        if let Err(e) = fs::create_dir_all(&user_dir_absolute) {
            eprintln!("❌ Ошибка создания папки пользователя: {}", e);
            None
        } else {
            println!(
                "📁 Создана папка пользователя: {}",
                user_dir_absolute.display()
            );
            Some(user_dir_absolute)
        }
    } else {
        None
    };

    set_user_session_path(user_session_path.clone());

    let ctx = ClientContext {
        smb_manager: smb_manager.clone(),
        use_ve,
        build_model,
    };

    while let Some(msg) = read.next().await {
        match msg {
            Ok(Message::Text(text)) => {
                let response_json = dispatch_message(&text, &ctx);
                if let Err(e) = write.send(Message::Text(response_json)).await {
                    eprintln!("❌ Ошибка отправки ответа: {}", e);
                    break;
                }
            }
            Ok(Message::Close(_)) => {
                println!("🔌 Клиент отключился");
                break;
            }
            Ok(Message::Ping(data)) => {
                if let Err(e) = write.send(Message::Pong(data)).await {
                    eprintln!("❌ Ошибка отправки Pong: {}", e);
                    break;
                }
            }
            Err(e) => {
                eprintln!("❌ Ошибка чтения сообщения: {}", e);
                if use_ve {
                    if let Some(session_path) = get_user_session_path() {
                        if session_path.exists() {
                            let _ = fs::remove_dir_all(&session_path);
                        }
                    }
                }
                break;
            }
            _ => {}
        }
    }

    cleanup_client(&smb_manager, use_ve);
}

fn cleanup_client(smb_manager: &Arc<Mutex<SmbManager>>, use_ve: bool) {
    let mut manager = smb_manager.lock().unwrap();
    let shares: Vec<String> = manager.list_connections();
    for share in shares {
        let _ = manager.disconnect(&share);
    }
    drop(manager);

    if use_ve {
        if let Some(session_path) = get_user_session_path() {
            if session_path.exists() {
                if let Err(e) = fs::remove_dir_all(&session_path) {
                    eprintln!(
                        "⚠️  Ошибка удаления папки пользователя {}: {}",
                        session_path.display(),
                        e
                    );
                } else {
                    println!("🗑️  Удалена папка пользователя: {}", session_path.display());
                }
            }
        }
    }

    crate::vm::file_ops::clear_smb_manager();
    set_user_session_path(None);
    set_use_ve(false);
}
