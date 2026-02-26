// CLI argument parsing and structures

use std::env;
use std::path::Path;

const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Configuration for WebSocket server
#[derive(Debug, Clone)]
pub struct WebSocketConfig {
    pub host: String,
    pub port: u16,
    pub use_ve: bool,
    pub build_model: bool,
}

/// Configuration for HTTP server (datacode-server)
#[derive(Debug, Clone)]
pub struct HttpServerConfig {
    pub host: String,
    pub port: u16,
    /// Optional app entry file (e.g. app.dc) with @route handlers; if None, GET / returns fixed string.
    pub app_file: Option<String>,
}

/// Configuration for file execution
#[derive(Debug, Clone)]
pub struct FileExecutionConfig {
    pub filename: String,
    /// When set, resolve relative script path and load_env base from this directory (e.g. when running via cargo from subdirectory).
    pub base_dir: Option<String>,
    pub build_model: bool,
    pub output_db: Option<String>,
    pub debug: bool,
    /// When true, run in main thread without GUI event loop (script output visible; no plot windows)
    pub no_gui: bool,
    pub script_args: Vec<String>, // Аргументы для передачи в скрипт (позиционные + из --имя=значение по первому параметру __main__)
    /// Полная командная строка для извлечения опций вида --X=value по имени первого параметра __main__.
    pub raw_args: Vec<String>,
}

/// Parsed CLI arguments
#[derive(Debug)]
pub enum CliArgs {
    Help,
    Version,
    WebSocket(WebSocketConfig),
    HttpServer(HttpServerConfig),
    FileExecution(FileExecutionConfig),
    Repl,
}

/// Print help message
pub fn print_help() {
    println!("🧠 DataCode - Interactive Programming Language");
    println!();
    println!("Usage:");
    println!("  datacode                   # Start interactive REPL (default)");
    println!("  datacode main.dc           # Execute DataCode file");
    println!("  datacode main.dc --<param>=value  # fn __main__(param, ...) → --param=value adds to argv");
    println!("  datacode main.dc --build_model  # Execute and export tables to SQLite");
    println!("  datacode main.dc --build_model output.db  # Export to specific file");
    println!("  datacode main.dc --no-gui       # Run in main thread (script output visible, no plot windows)");
    println!("  datacode --base-dir <dir> main.dc  # Resolve script and .env paths from <dir> (see Path resolution below)");
    println!("  datacode --websocket       # Start WebSocket server for remote code execution");
    println!("  datacode --http            # Start HTTP server (or use datacode-server binary)");
    println!("  datacode --help            # Show this help");
    println!();
    println!("File Execution:");
    println!("  • Create files with .dc extension");
    println!("  • Execute: datacode filename.dc  [arg1 arg2 ...]  [--<name>=value] [--debug]");
    println!("  • argv = values from --<first __main__ param>=value (e.g. fn __main__(env, ...) → --env=prod) + positional");
    println!();
    println!("Path resolution (--base-dir):");
    println!("  • Relative script path (e.g. src/main.dc) is resolved from current working directory (CWD).");
    println!("  • When running from repo root or IDE, CWD may be the project root — then load_env reads");
    println!("    settings from that tree (e.g. project_root/src/settings/dev.env), not from the script dir.");
    println!("  • Use --base-dir <dir> to fix: datacode --base-dir sandbox/config_test src/main.dc");
    println!("  • Or run from the script directory: cd sandbox/config_test && cargo run -- src/main.dc");
    println!("  • Use --debug to print CWD, script path and resolved .env path to stderr.");
    println!();
    println!("SQLite Export (--build_model):");
    println!("  • Exports all tables from global variables to SQLite database");
    println!("  • Automatically detects foreign key relationships");
    println!("  • Creates metadata table _datacode_variables with all variable info");
    println!("  • Default output: <script_name>.db");
    println!("  • Custom output: --build_model output.db");
    println!("  • Environment variable: DATACODE_SQLITE_OUTPUT=path.db");
    println!();
    println!("WebSocket Server:");
    println!("  • Start server: datacode --websocket");
    println!("  • Default address: ws://127.0.0.1:8080");
    println!("  • Custom host/port: datacode --websocket --host 0.0.0.0 --port 8899");
    println!("  • Or use env var: DATACODE_WS_ADDRESS=0.0.0.0:3000 datacode --websocket");
    println!("  • Virtual environment mode: datacode --websocket --use-ve");
    println!("    - Creates isolated session folders in src/temp_sessions");
    println!("    - getcwd() returns empty string");
    println!("    - Supports file uploads via upload_file request");
    println!("    - Session folder is deleted on disconnect");
    println!("  • Send JSON: {{\"code\": \"print('Hello World')\"}}");
    println!("  • Receive JSON: {{\"success\": true, \"output\": \"Hello World\\n\", \"error\": null}}");
    println!("      • Upload file: {{\"type\": \"upload_file\", \"filename\": \"test.txt\", \"content\": \"...\"}}");
    println!();
    println!("HTTP Server (datacode-server):");
    println!("  • Start: datacode --http  or  datacode-server");
    println!("  • Default: http://127.0.0.1:8080");
    println!("  • Custom: datacode-server --host 0.0.0.0 --port 3000");
    println!("  • GET / returns \"Hello from DataCode\" (Stage 1)");
    println!();
    println!("Features:");
    println!("  • Interactive REPL with multiline support");
    println!("  • User-defined functions with local scope");
    println!("  • Arithmetic and logical operations");
    println!("  • File system operations");
    println!("  • For loops and control structures");
    println!("  • Improved error messages with line numbers");
    println!("  • Path manipulation");
    println!("  • Functional programming methods (map, filter, reduce)");
    println!("  • WebSocket server for remote code execution");
    println!();
    println!("Example DataCode file (example.dc):");
    println!("  # Simple DataCode program");
    println!("  fn greet(name) {{");
    println!("      return 'Hello, ' + name + '!'");
    println!("  }}");
    println!("  ");
    println!("  global message = greet('DataCode')");
    println!("  print(message)");
    println!();
    println!("Run with: datacode example.dc");
    println!("Debug run: datacode example.dc --debug");
}

/// Print version
pub fn print_version() {
    println!("DataCode v{}", VERSION);
}

/// Собирает значения опций вида --param_name и --param_name=value из полного argv.
/// Используется для передачи в скрипт по имени первого параметра fn __main__(param_name, ...).
pub fn extract_param_args(args: &[String], param_name: &str) -> Vec<String> {
    let mut out = Vec::new();
    let prefix_eq = format!("--{}=", param_name);
    let flag = format!("--{}", param_name);
    let mut i = 0;
    while i < args.len() {
        let arg = &args[i];
        if arg == &flag {
            if i + 1 < args.len() && !args[i + 1].starts_with('-') {
                out.push(args[i + 1].clone());
                i += 2;
            } else {
                i += 1;
            }
        } else if arg.starts_with(&prefix_eq) {
            let value = arg.strip_prefix(&prefix_eq).unwrap_or("").to_string();
            out.push(value);
            i += 1;
        } else {
            i += 1;
        }
    }
    out
}

/// Parse --build_model, --debug, --no-gui, --base-dir and script args from a slice.
/// skip_script_arg_index: if Some(i), args[i] is the .dc filename and is not added to script_args.
fn parse_file_execution_flags(
    args: &[String],
    skip_script_arg_index: Option<usize>,
) -> Result<
    (
        bool,
        Option<String>,
        bool,
        bool,
        Option<String>,
        Vec<String>,
    ),
    String,
> {
    let mut build_model = false;
    let mut output_db: Option<String> = None;
    let mut debug = false;
    let mut no_gui = false;
    let mut base_dir: Option<String> = None;
    let mut script_args = Vec::new();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--build_model" | "--build-model" => {
                build_model = true;
                if i + 1 < args.len() && !args[i + 1].starts_with('-') {
                    output_db = Some(args[i + 1].clone());
                    i += 2;
                } else {
                    i += 1;
                }
            }
            "--debug" => {
                debug = true;
                i += 1;
            }
            "--no-gui" => {
                no_gui = true;
                i += 1;
            }
            "--base-dir" | "--base_dir" => {
                if i + 1 < args.len() {
                    base_dir = Some(args[i + 1].clone());
                    i += 2;
                } else {
                    return Err("Ошибка: --base-dir требует значение (путь к директории)".to_string());
                }
            }
            arg => {
                if !arg.starts_with('-') {
                    if Some(i) != skip_script_arg_index {
                        script_args.push(arg.to_string());
                    }
                }
                i += 1;
            }
        }
    }
    Ok((build_model, output_db, debug, no_gui, base_dir, script_args))
}

/// Parse CLI arguments
pub fn parse_args(args: Vec<String>) -> Result<CliArgs, String> {
    if args.len() > 1 {
        let arg = &args[1];
        
        match arg.as_str() {
            "-h" | "--help" => {
                return Ok(CliArgs::Help);
            }
            "-v" | "--version" => {
                return Ok(CliArgs::Version);
            }
            "--websocket" => {
                let mut host = "127.0.0.1".to_string();
                let mut port = 8080u16;
                let mut use_ve = false;
                let mut build_model = false;
                
                // Check environment variable
                if let Ok(ws_address) = env::var("DATACODE_WS_ADDRESS") {
                    if let Some(colon_pos) = ws_address.find(':') {
                        host = ws_address[..colon_pos].to_string();
                        if let Ok(p) = ws_address[colon_pos + 1..].parse::<u16>() {
                            port = p;
                        }
                    } else {
                        host = ws_address;
                    }
                }
                
                // Parse command line arguments
                let mut i = 2;
                while i < args.len() {
                    match args[i].as_str() {
                        "--host" => {
                            if i + 1 < args.len() {
                                host = args[i + 1].clone();
                                i += 2;
                            } else {
                                return Err("Ошибка: --host требует значение".to_string());
                            }
                        }
                        "--port" => {
                            if i + 1 < args.len() {
                                if let Ok(p) = args[i + 1].parse::<u16>() {
                                    port = p;
                                    i += 2;
                                } else {
                                    return Err("Ошибка: неверный номер порта".to_string());
                                }
                            } else {
                                return Err("Ошибка: --port требует значение".to_string());
                            }
                        }
                        "--use-ve" => {
                            use_ve = true;
                            i += 1;
                        }
                        "--build_model" | "--build-model" => {
                            build_model = true;
                            i += 1;
                        }
                        _ => {
                            return Err(format!("Неизвестный аргумент: {}", args[i]));
                        }
                    }
                }
                
                return Ok(CliArgs::WebSocket(WebSocketConfig {
                    host,
                    port,
                    use_ve,
                    build_model,
                }));
            }
            "--http" => {
                let mut host = "127.0.0.1".to_string();
                let mut port = 8080u16;
                let mut app_file: Option<String> = None;
                if let Ok(addr) = env::var("DATACODE_HTTP_ADDRESS") {
                    if let Some(colon_pos) = addr.find(':') {
                        host = addr[..colon_pos].to_string();
                        if let Ok(p) = addr[colon_pos + 1..].parse::<u16>() {
                            port = p;
                        }
                    } else {
                        host = addr;
                    }
                }
                let mut i = 2;
                while i < args.len() {
                    match args[i].as_str() {
                        "--host" => {
                            if i + 1 < args.len() {
                                host = args[i + 1].clone();
                                i += 2;
                            } else {
                                return Err("Ошибка: --host требует значение".to_string());
                            }
                        }
                        "--port" => {
                            if i + 1 < args.len() {
                                if let Ok(p) = args[i + 1].parse::<u16>() {
                                    port = p;
                                    i += 2;
                                } else {
                                    return Err("Ошибка: неверный номер порта".to_string());
                                }
                            } else {
                                return Err("Ошибка: --port требует значение".to_string());
                            }
                        }
                        arg if !arg.starts_with('-') && arg.ends_with(".dc") => {
                            app_file = Some(arg.to_string());
                            i += 1;
                        }
                        _ => {
                            i += 1;
                        }
                    }
                }
                return Ok(CliArgs::HttpServer(HttpServerConfig { host, port, app_file }));
            }
            _ => {
                // Allow flags before filename (e.g. datacode --no-gui script.dc); if first arg is a flag, find .dc file in args
                if arg.starts_with('-') {
                    let dc_idx = args[1..].iter().position(|a| a.ends_with(".dc"));
                    let dc_idx = dc_idx.map(|i| i + 1);
                    if let Some(idx) = dc_idx {
                        let filename = args[idx].clone();
                        let (build_model, output_db, debug, no_gui, base_dir, script_args) =
                            parse_file_execution_flags(&args[1..], Some(idx - 1))?;
                        let script_path_for_check: std::path::PathBuf = if let Some(ref b) = base_dir {
                            Path::new(b).join(&filename)
                        } else {
                            Path::new(&filename).to_path_buf()
                        };
                        if !script_path_for_check.exists() {
                            return Err(format!("Ошибка: файл '{}' не найден", script_path_for_check.display()));
                        }
                        if !filename.ends_with(".dc") {
                            eprintln!("Предупреждение: файл '{}' не имеет расширения .dc", filename);
                        }
                        return Ok(CliArgs::FileExecution(FileExecutionConfig {
                            filename,
                            base_dir,
                            build_model,
                            output_db,
                            debug,
                            no_gui,
                            script_args,
                            raw_args: args.clone(),
                        }));
                    }
                    return Err(format!("Неизвестная опция: {}\nИспользуйте --help для справки", arg));
                }
            }
        }
        
        // File execution: first arg is the .dc filename
        let filename = arg.clone();
        let (build_model, output_db, debug, no_gui, base_dir, script_args) =
            parse_file_execution_flags(&args[2..], None)?;
        
        // Check file existence: if base_dir set and path relative, resolve relative to base_dir
        let script_path_for_check: std::path::PathBuf = if let Some(ref b) = base_dir {
            Path::new(b).join(&filename)
        } else {
            Path::new(&filename).to_path_buf()
        };
        if !script_path_for_check.exists() {
            return Err(format!("Ошибка: файл '{}' не найден", script_path_for_check.display()));
        }
        
        if !filename.ends_with(".dc") {
            eprintln!("Предупреждение: файл '{}' не имеет расширения .dc", filename);
        }
        
        Ok(CliArgs::FileExecution(FileExecutionConfig {
            filename,
            base_dir,
            build_model,
            output_db,
            debug,
            no_gui,
            script_args,
            raw_args: args.clone(),
        }))
    } else {
        // REPL mode (interactive)
        Ok(CliArgs::Repl)
    }
}

/// Get version string
pub fn version() -> &'static str {
    VERSION
}

