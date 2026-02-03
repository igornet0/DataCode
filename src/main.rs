use data_code::compile;
use data_code::sqlite_export;
use data_code::common::debug;
use data_code::vm::{cli, repl, gui, websocket, http_server};
use std::env;
use std::fs;
use std::path::PathBuf;

fn main() {
    let args: Vec<String> = env::args().collect();

    match cli::parse_args(args) {
        Ok(cli::CliArgs::Help) => {
            cli::print_help();
        }
        Ok(cli::CliArgs::Version) => {
            cli::print_version();
        }
        Ok(cli::CliArgs::WebSocket(config)) => {
            if let Err(e) = websocket::start_websocket_server(config) {
                eprintln!("{}", e);
                std::process::exit(1);
            }
        }
        Ok(cli::CliArgs::HttpServer(config)) => {
            if let Err(e) = http_server::start_http_server(config) {
                eprintln!("{}", e);
                std::process::exit(1);
            }
        }
        Ok(cli::CliArgs::FileExecution(config)) => {
            // Set debug mode if flag is present
            debug::set_debug(config.debug);
            execute_file(config);
        }
        Ok(cli::CliArgs::Repl) => {
            if let Err(e) = gui::run_with_event_loop(|| repl::run_repl()) {
                eprintln!("Ошибка: {}", e);
            }
        }
        Err(e) => {
            eprintln!("{}", e);
            std::process::exit(1);
        }
    }
}

fn execute_file(config: cli::FileExecutionConfig) {
    use std::path::Path;
    use data_code::dpm::{resolve_env_and_packages, find_project_root, load_manifest, lock_file_name, datacode_version_satisfies};

    // Определяем путь к файлу
    let file_path = Path::new(&config.filename);

    // Подготавливаем аргументы командной строки для передачи в скрипт
    let script_args = config.script_args.clone();

    // Базовый путь скрипта (директория файла) — передаём в VM для разрешения импортов и load_env.
    // Если задан --base-dir и путь относительный, разрешаем относительно base_dir (стабильно при cargo run из подкаталога).
    // Иначе: CWD + относительный путь, затем канонизация.
    let script_abs: PathBuf = if file_path.is_absolute() {
        file_path.to_path_buf()
    } else if let Some(ref base_dir) = config.base_dir {
        Path::new(base_dir).join(file_path)
    } else {
        std::env::current_dir()
            .ok()
            .unwrap_or_else(|| PathBuf::from("."))
            .join(file_path)
    };
    let script_canonical = fs::canonicalize(&script_abs).unwrap_or(script_abs.clone());
    let script_base_path = script_canonical.parent().map(PathBuf::from);

    // Путь к файлу для чтения и для __lib__.dc — всегда канонический
    let script_path_for_read = &script_canonical;

    // Определяем путь к __lib__.dc для передачи в run_with_vm_with_args_and_lib (в той же директории, что и скрипт)
    let lib_path = script_base_path.as_ref().and_then(|dir| {
        let lib_path = dir.join("__lib__.dc");
        if lib_path.exists() {
            Some(lib_path)
        } else {
            None
        }
    });

    // Устанавливаем базовый путь для импортов и load_env — всегда абсолютный,
    // чтобы разрешение путей не зависело от current_dir() (иначе при cargo run из подкаталога — нестабильно).
    use data_code::vm::file_import;
    if let Some(ref base) = script_base_path {
        file_import::set_base_path(Some(base.clone()));
    }

    // Отладочный вывод путей (при --debug): при null/null в load_env сразу видно CWD и какой файл читается
    if config.debug {
        eprintln!("[datacode path] CWD = {:?}", std::env::current_dir().ok());
        eprintln!("[datacode path] script_canonical = {:?}", script_canonical);
        eprintln!("[datacode path] script_base_path = {:?}", script_base_path);
    }

    // DPM: найти dpm.toml, проверить dpm.lock, установить пути пакетов для импорта (от разрешённого пути скрипта)
    let dpm_package_paths = match resolve_env_and_packages(script_path_for_read.as_path()) {
        Ok(Some((_env_root, paths))) => {
            if let Some(project_root) = find_project_root(script_path_for_read.as_path()) {
                if let Ok(manifest) = load_manifest(&project_root) {
                    let lock_file = lock_file_name(&manifest);
                    let lock_path = project_root.join(lock_file);
                    if !manifest.dependencies.is_empty() && !lock_path.exists() {
                        eprintln!(
                            "Предупреждение: dpm.toml найден, но {} отсутствует. Выполните: dpm init",
                            lock_file
                        );
                    }
                    if let Some(ref proj) = manifest.project {
                        if let Some(ref req) = proj.datacode {
                            let current = env!("CARGO_PKG_VERSION");
                            if !datacode_version_satisfies(req, current) {
                                eprintln!(
                                    "Предупреждение: dpm.toml требует datacode {}, установлена {}",
                                    req, current
                                );
                            }
                        }
                    }
                }
            }
            paths
        }
        Ok(None) => Vec::new(),
        Err(e) => {
            eprintln!("Предупреждение DPM: {}", e);
            Vec::new()
        }
    };
    file_import::set_dpm_package_paths(dpm_package_paths.clone());
    
    // Унифицированный подход: используем run_with_vm_with_args_and_lib для обоих случаев
    // Читаем файл по разрешённому пути (каноническому), чтобы при --base-dir использовался правильный файл
    match fs::read_to_string(script_path_for_read) {
        Ok(source) => {
            // If debug mode, print bytecode first
            if config.debug {
                match compile(&source) {
                    Ok((chunk, functions)) => {
                        println!("{}", chunk.disassemble("<main>"));
                        for function in &functions {
                            println!("{}", function.chunk.disassemble(&function.name));
                        }
                    }
                    Err(_) => {
                        // Continue execution even if compilation fails
                    }
                }
            }
            
            if config.build_model {
                // Execute with SQLite export
                match data_code::run_with_vm_with_args_and_lib(&source, Some(script_args), lib_path.as_deref(), script_base_path.as_deref()) {
                    Ok((_, vm)) => {
                        // Determine output database filename
                        let db_filename = if let Some(db) = config.output_db {
                            db
                        } else if let Ok(env_db) = env::var("DATACODE_SQLITE_OUTPUT") {
                            env_db
                        } else {
                            // Default: script name with .db extension
                            let path = PathBuf::from(&config.filename);
                            let stem = path.file_stem()
                                .and_then(|s| s.to_str())
                                .unwrap_or("output");
                            format!("{}.db", stem)
                        };
                        
                        // Export tables to SQLite
                        match sqlite_export::export_to_sqlite(&vm, &db_filename) {
                            Ok(_) => {
                                println!("✅ База данных создана: {}", db_filename);
                            }
                            Err(e) => {
                                eprintln!("❌ Ошибка экспорта в SQLite: {}", e);
                                std::process::exit(1);
                            }
                        }
                    }
                    Err(e) => {
                        eprintln!("Ошибка выполнения: {}", e);
                        std::process::exit(1);
                    }
                }
            } else {
                // Normal execution without export
                if config.no_gui {
                    // Run in main thread: script output (print) is visible; no plot windows
                    match data_code::run_with_vm_with_args_and_lib(&source, Some(script_args), lib_path.as_deref(), script_base_path.as_deref()) {
                        Ok(_) => {}
                        Err(e) => {
                            eprintln!("Ошибка выполнения: {}", e);
                            std::process::exit(1);
                        }
                    }
                } else {
                    // Use run_with_event_loop to support plot functionality
                    let source_clone = source.clone();
                    let script_args_clone = script_args.clone();
                    let lib_path_clone = lib_path.clone();
                    let script_base_path_clone = script_base_path.clone();
                    let dpm_paths_clone = dpm_package_paths.clone();

                    match gui::run_with_event_loop(move || {
                        // Устанавливаем базовый путь (абсолютный) и DPM пути в GUI потоке (thread-local)
                        use data_code::vm::file_import;
                        if let Some(ref base) = script_base_path_clone {
                            file_import::set_base_path(Some(base.clone()));
                        }
                        file_import::set_dpm_package_paths(dpm_paths_clone.clone());

                        // Используем run_with_vm_with_args_and_lib для передачи пути к __lib__.dc и base_path
                        // __lib__.dc будет выполнен внутри GUI потока перед основным скриптом
                        data_code::run_with_vm_with_args_and_lib(&source_clone, Some(script_args_clone), lib_path_clone.as_deref(), script_base_path_clone.as_deref())
                            .map(|_| ()) // Ignore return value
                            .map_err(|e| e.to_string())
                    }) {
                        Ok(_) => {}
                        Err(e) => {
                            eprintln!("Ошибка выполнения: {}", e);
                            std::process::exit(1);
                        }
                    }
                }
            }
        }
        Err(e) => {
            eprintln!("Ошибка чтения файла '{}': {}", script_path_for_read.display(), e);
            std::process::exit(1);
        }
    }
}
