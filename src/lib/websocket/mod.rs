use crate::run_with_vm;
use crate::sqlite_export;
use serde::{Deserialize, Serialize};
use tokio_tungstenite::{accept_async, tungstenite::Message};
use futures_util::{SinkExt, StreamExt};
use tokio::net::{TcpListener, TcpStream};
use std::sync::{Arc, Mutex};
use std::path::{Path, PathBuf};
use std::fs;
use std::env;
use std::time::{SystemTime, UNIX_EPOCH};

pub mod output_capture;
pub mod smb;

use output_capture::OutputCapture;
use smb::{SmbManager, SmbConnection};

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type")]
enum WebSocketRequest {
    #[serde(rename = "execute")]
    Execute { code: String },
    #[serde(rename = "smb_connect")]
    SmbConnect {
        ip: String,
        login: String,
        password: String,
        domain: String,
        share_name: String,
    },
    #[serde(rename = "smb_list_files")]
    SmbListFiles {
        share_name: String,
        path: String,
    },
    #[serde(rename = "smb_read_file")]
    SmbReadFile {
        share_name: String,
        file_path: String,
    },
    #[serde(rename = "upload_file")]
    UploadFile {
        filename: String,
        content: String,
    },
}

#[derive(Debug, Serialize, Deserialize)]
struct ExecuteRequest {
    code: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct ExecuteResponse {
    success: bool,
    output: String,
    error: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    sqlite_db: Option<String>,
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
pub async fn start_server(address: &str, use_ve: bool, build_model: bool) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let listener = TcpListener::bind(address).await?;
    println!("🚀 DataCode WebSocket Server запущен на {}", address);
    println!("📡 Ожидание подключений...");
    println!("💡 Отправьте JSON запрос: {{\"code\": \"ваш код\"}}");
    println!("💡 Ответ будет в формате: {{\"success\": true/false, \"output\": \"...\", \"error\": null/\"...\"}}");
    println!();

    // Если включен режим use_ve, создаем папку temp_sessions
    if use_ve {
        let temp_sessions_dir = Path::new("src/temp_sessions");
        if !temp_sessions_dir.exists() {
            if let Err(e) = fs::create_dir_all(temp_sessions_dir) {
                eprintln!("⚠️  Предупреждение: не удалось создать папку temp_sessions: {}", e);
            } else {
                println!("📁 Создана папка для сессий: {}", temp_sessions_dir.display());
            }
        }
    }

    // Используем LocalSet для локальных задач, так как Interpreter не является Send
    let local_set = tokio::task::LocalSet::new();
    
    // Создаем listener внутри LocalSet и обрабатываем подключения
    local_set.run_until(async {
        loop {
            let (stream, addr) = match listener.accept().await {
                Ok((s, a)) => (s, a),
                Err(e) => {
                    eprintln!("❌ Ошибка принятия подключения: {}", e);
                    continue;
                }
            };
            
            println!("✅ Новое подключение от {}", addr);
            // DIAG: Log task spawn
            // eprintln!("[DIAG] WebSocket: Spawning client handler task for {}", addr);
            local_set.spawn_local(handle_client(stream, use_ve, build_model));
        }
    }).await;

    Ok(())
}

/// Обработать клиентское подключение
async fn handle_client(stream: TcpStream, use_ve: bool, build_model: bool) {
    let _client_start_time = std::time::Instant::now();
    let _client_addr = stream.peer_addr().ok();
    
    let ws_stream = match accept_async(stream).await {
        Ok(ws) => ws,
        Err(e) => {
            eprintln!("❌ Ошибка при принятии WebSocket соединения: {}", e);
            return;
        }
    };

    let (mut write, mut read) = ws_stream.split();
    // Создаем отдельный SmbManager для каждого клиента
    let smb_manager = Arc::new(Mutex::new(SmbManager::new()));
    
    // Устанавливаем SmbManager в thread-local storage для доступа из функций файловых операций
    crate::vm::file_ops::set_smb_manager(smb_manager.clone());
    
    // Устанавливаем флаг use_ve
    set_use_ve(use_ve);
    
    // Если включен режим use_ve, создаем папку для пользователя
    let user_session_path = if use_ve {
        // Генерируем уникальный ID для пользователя на основе времени и случайного числа
        use std::time::{SystemTime, UNIX_EPOCH};
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let user_id = format!("user_{}", timestamp);
        let user_dir = Path::new("src/temp_sessions").join(&user_id);
        
        // Преобразуем в абсолютный путь для корректной работы с путями от list_files
        let user_dir_absolute = match user_dir.canonicalize() {
            Ok(p) => p,
            Err(_) => {
                // Если канонизация не удалась (папка еще не существует), 
                // создаем абсолютный путь через current_dir
                match env::current_dir() {
                    Ok(cwd) => cwd.join(&user_dir),
                    Err(_) => user_dir, // Fallback к относительному пути
                }
            },
        };
        
        if let Err(e) = fs::create_dir_all(&user_dir_absolute) {
            eprintln!("❌ Ошибка создания папки пользователя: {}", e);
            None
        } else {
            println!("📁 Создана папка пользователя: {}", user_dir_absolute.display());
            Some(user_dir_absolute)
        }
    } else {
        None
    };
    
    // Устанавливаем путь к папке пользователя в thread-local storage
    set_user_session_path(user_session_path.clone());

    while let Some(msg) = read.next().await {
        match msg {
            Ok(Message::Text(text)) => {
                // Пытаемся распарсить как новый формат с типом команды
                if let Ok(request) = serde_json::from_str::<WebSocketRequest>(&text) {
                    match request {
                        WebSocketRequest::Execute { code } => {
                            // Выполняем код
                            let response = execute_code(&code, &smb_manager, build_model);
                            
                            // Отправляем ответ
                            if let Ok(json) = serde_json::to_string(&response) {
                                if let Err(e) = write.send(Message::Text(json)).await {
                                    eprintln!("❌ Ошибка отправки ответа: {}", e);
                                    break;
                                }
                            }
                        }
                        WebSocketRequest::SmbConnect { ip, login, password, domain, share_name } => {
                            let connection = SmbConnection::new(ip, login, password, domain, share_name);
                            let result = smb_manager.lock().unwrap().connect(connection);
                            
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
                            
                            if let Ok(json) = serde_json::to_string(&response) {
                                if let Err(e) = write.send(Message::Text(json)).await {
                                    eprintln!("❌ Ошибка отправки ответа: {}", e);
                                    break;
                                }
                            }
                        }
                        WebSocketRequest::SmbListFiles { share_name, path } => {
                            let result = smb_manager.lock().unwrap().list_files(&share_name, &path, None, true);
                            
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
                            
                            if let Ok(json) = serde_json::to_string(&response) {
                                if let Err(e) = write.send(Message::Text(json)).await {
                                    eprintln!("❌ Ошибка отправки ответа: {}", e);
                                    break;
                                }
                            }
                        }
                        WebSocketRequest::SmbReadFile { share_name, file_path } => {
                            let result = smb_manager.lock().unwrap().read_file(&share_name, &file_path);
                            
                            let response = match result {
                                Ok(content) => {
                                    // Пытаемся декодировать как UTF-8, если не получается - возвращаем base64
                                    match String::from_utf8(content.clone()) {
                                        Ok(text) => SmbReadFileResponse {
                                            success: true,
                                            content: Some(text),
                                            error: None,
                                        },
                                        Err(_) => {
                                            // Если не UTF-8, возвращаем base64
                                            use base64::Engine;
                                            let base64_content = base64::engine::general_purpose::STANDARD.encode(&content);
                                            SmbReadFileResponse {
                                                success: true,
                                                content: Some(format!("base64:{}", base64_content)),
                                                error: None,
                                            }
                                        }
                                    }
                                }
                                Err(e) => SmbReadFileResponse {
                                    success: false,
                                    content: None,
                                    error: Some(e),
                                },
                            };
                            
                            if let Ok(json) = serde_json::to_string(&response) {
                                if let Err(e) = write.send(Message::Text(json)).await {
                                    eprintln!("❌ Ошибка отправки ответа: {}", e);
                                    break;
                                }
                            }
                        }
                        WebSocketRequest::UploadFile { filename, content } => {
                            let response = if use_ve {
                                if let Some(session_path) = get_user_session_path() {
                                    let file_path = session_path.join(&filename);
                                    
                                    // Создаем родительские директории если нужно
                                    if let Some(parent) = file_path.parent() {
                                        match fs::create_dir_all(parent) {
                                            Ok(_) => {
                                                // Декодируем base64 контент если нужно
                                                let file_content_result = if content.starts_with("base64:") {
                                                    use base64::Engine;
                                                    base64::engine::general_purpose::STANDARD.decode(&content[7..])
                                                        .map_err(|e| format!("Ошибка декодирования base64: {}", e))
                                                } else {
                                                    Ok(content.as_bytes().to_vec())
                                                };
                                                
                                                match file_content_result {
                                                    Ok(file_content) => {
                                                        match fs::write(&file_path, file_content) {
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
                                                        }
                                                    }
                                                    Err(e) => UploadFileResponse {
                                                        success: false,
                                                        message: String::new(),
                                                        error: Some(e),
                                                    },
                                                }
                                            }
                                            Err(e) => UploadFileResponse {
                                                success: false,
                                                message: String::new(),
                                                error: Some(format!("Ошибка создания директории: {}", e)),
                                            },
                                        }
                                    } else {
                                        UploadFileResponse {
                                            success: false,
                                            message: String::new(),
                                            error: Some("Некорректный путь к файлу".to_string()),
                                        }
                                    }
                                } else {
                                    UploadFileResponse {
                                        success: false,
                                        message: String::new(),
                                        error: Some("Сессия пользователя не найдена".to_string()),
                                    }
                                }
                            } else {
                                UploadFileResponse {
                                    success: false,
                                    message: String::new(),
                                    error: Some("Режим --use-ve не включен".to_string()),
                                }
                            };
                            
                            if let Ok(json) = serde_json::to_string(&response) {
                                if let Err(e) = write.send(Message::Text(json)).await {
                                    eprintln!("❌ Ошибка отправки ответа: {}", e);
                                    break;
                                }
                            }
                        }
                    }
                } else {
                    // Пытаемся распарсить как старый формат для обратной совместимости
                    if let Ok(request) = serde_json::from_str::<ExecuteRequest>(&text) {
                        let response = execute_code(&request.code, &smb_manager, build_model);
                        
                        if let Ok(json) = serde_json::to_string(&response) {
                            if let Err(e) = write.send(Message::Text(json)).await {
                                eprintln!("❌ Ошибка отправки ответа: {}", e);
                                break;
                            }
                        }
                    } else {
                        let error_response = ExecuteResponse {
                            success: false,
                            output: String::new(),
                            error: Some(format!("Ошибка парсинга запроса. Ожидается JSON с полями: type, code (или smb_connect, smb_list_files, smb_read_file)")),
                            sqlite_db: None,
                        };
                        if let Ok(json) = serde_json::to_string(&error_response) {
                            let _ = write.send(Message::Text(json)).await;
                        }
                    }
                }
            }
            Ok(Message::Close(_)) => {
                println!("🔌 Клиент отключился");
                // Отключаем все SMB подключения при отключении клиента
                let mut manager = smb_manager.lock().unwrap();
                let shares: Vec<String> = manager.list_connections();
                for share in shares {
                    let _ = manager.disconnect(&share);
                }
                
                // Если включен режим use_ve, удаляем папку пользователя
                if use_ve {
                    if let Some(session_path) = get_user_session_path() {
                        if session_path.exists() {
                            if let Err(e) = fs::remove_dir_all(&session_path) {
                                eprintln!("⚠️  Ошибка удаления папки пользователя {}: {}", session_path.display(), e);
                            } else {
                                println!("🗑️  Удалена папка пользователя: {}", session_path.display());
                            }
                        }
                    }
                }
                
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
                
                // Если включен режим use_ve, удаляем папку пользователя при ошибке
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
    
    // Если включен режим use_ve, удаляем папку пользователя при выходе из цикла
    if use_ve {
        if let Some(session_path) = get_user_session_path() {
            if session_path.exists() {
                if let Err(e) = fs::remove_dir_all(&session_path) {
                    eprintln!("⚠️  Ошибка удаления папки пользователя {}: {}", session_path.display(), e);
                } else {
                    println!("🗑️  Удалена папка пользователя: {}", session_path.display());
                }
            }
        }
    }
    
    // Очищаем thread-local storage
    crate::vm::file_ops::clear_smb_manager();
    set_user_session_path(None);
    set_use_ve(false);
    
    // DIAG: Log client handler completion
    // let client_duration = client_start_time.elapsed();
    // eprintln!(
    //     "[DIAG] WebSocket: Client handler finished - addr={:?}, duration={}ms",
    //     client_addr, client_duration.as_millis()
    // );
}

/// Выполнить код и вернуть результат
fn execute_code(
    code: &str,
    smb_manager: &Arc<Mutex<SmbManager>>,
    build_model: bool,
) -> ExecuteResponse {
    // Устанавливаем SmbManager в thread-local storage для доступа из функций файловых операций
    crate::vm::file_ops::set_smb_manager(smb_manager.clone());
    
    // Создаем буфер для перехвата вывода
    let output_capture = OutputCapture::new();
    
    // Устанавливаем буфер для текущего потока
    output_capture.set_capture(true);

    // Выполняем код используя новую архитектуру VM и получаем VM для доступа к глобальным переменным
    let result = run_with_vm(code);

    // Получаем вывод
    let output = output_capture.get_output();
    output_capture.set_capture(false);

    // Формируем ответ
    let response = match result {
        Ok((_, mut vm)) => {
            let mut sqlite_db = None;
            
            // Если включен build_model, проверяем наличие таблиц и экспортируем их
            if build_model {
                // Проверяем наличие таблиц
                match sqlite_export::get_global_tables(&mut vm) {
                    Ok(tables) if !tables.is_empty() => {
                        // Создаем временный файл для SQLite БД
                        let timestamp = SystemTime::now()
                            .duration_since(UNIX_EPOCH)
                            .unwrap()
                            .as_nanos();
                        let temp_db_path = env::temp_dir().join(format!("datacode_export_{}.db", timestamp));
                        
                        // Экспортируем таблицы в SQLite
                        match sqlite_export::export_to_sqlite(&mut vm, temp_db_path.to_str().unwrap()) {
                            Ok(_) => {
                                // Читаем SQLite БД как байты
                                match fs::read(&temp_db_path) {
                                    Ok(db_bytes) => {
                                        // Кодируем в base64
                                        use base64::Engine;
                                        let base64_db = base64::engine::general_purpose::STANDARD.encode(&db_bytes);
                                        sqlite_db = Some(base64_db);
                                        
                                        // Удаляем временный файл
                                        let _ = fs::remove_file(&temp_db_path);
                                    }
                                    Err(e) => {
                                        eprintln!("⚠️  Ошибка чтения SQLite БД: {}", e);
                                        let _ = fs::remove_file(&temp_db_path);
                                    }
                                }
                            }
                            Err(e) => {
                                eprintln!("⚠️  Ошибка экспорта в SQLite: {}", e);
                            }
                        }
                    }
                    Ok(_) => {
                        // Таблиц нет, но это не ошибка
                    }
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
    };
    
    response
}

