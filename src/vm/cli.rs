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

/// Configuration for file execution
#[derive(Debug, Clone)]
pub struct FileExecutionConfig {
    pub filename: String,
    pub build_model: bool,
    pub output_db: Option<String>,
    pub debug: bool,
}

/// Parsed CLI arguments
#[derive(Debug)]
pub enum CliArgs {
    Help,
    Version,
    WebSocket(WebSocketConfig),
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
    println!("  datacode main.dc --build_model  # Execute and export tables to SQLite");
    println!("  datacode main.dc --build_model output.db  # Export to specific file");
    println!("  datacode --websocket       # Start WebSocket server for remote code execution");
    println!("  datacode --help            # Show this help");
    println!();
    println!("File Execution:");
    println!("  • Create files with .dc extension");
    println!("  • Write DataCode programs in files");
    println!("  • Execute with: datacode filename.dc");

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
    println!("  • Upload file: {{\"type\": \"upload_file\", \"filename\": \"test.txt\", \"content\": \"...\"}}");
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
            _ => {
                // Check if it's an option (starts with -)
                if arg.starts_with('-') {
                    return Err(format!("Неизвестная опция: {}\nИспользуйте --help для справки", arg));
                }
            }
        }
        
        // File execution
        let filename = arg.clone();
        
        // Check file existence
        if !Path::new(&filename).exists() {
            return Err(format!("Ошибка: файл '{}' не найден", filename));
        }
        
        // Check file extension (optional, but useful)
        if !filename.ends_with(".dc") {
            eprintln!("Предупреждение: файл '{}' не имеет расширения .dc", filename);
        }
        
        // Check for --build_model and --debug flags
        let mut build_model = false;
        let mut output_db: Option<String> = None;
        let mut debug = false;
        let mut i = 2;
        while i < args.len() {
            match args[i].as_str() {
                "--build_model" | "--build-model" => {
                    build_model = true;
                    // Check next argument - might be filename
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
                _ => {
                    i += 1;
                }
            }
        }
        
        Ok(CliArgs::FileExecution(FileExecutionConfig {
            filename,
            build_model,
            output_db,
            debug,
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

