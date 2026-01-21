// Main entry point для DataCode интерпретатора

use data_code::{run, run_with_vm, compile};
use data_code::sqlite_export;
use data_code::common::debug;
use data_code::vm::{cli, repl, gui, websocket};
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
    if config.build_model {
        // Execute with SQLite export
        match fs::read_to_string(&config.filename) {
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
                match run_with_vm(&source) {
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
                }
                Err(e) => {
                eprintln!("Ошибка чтения файла '{}': {}", config.filename, e);
                    std::process::exit(1);
                }
            }
        } else {
        // Normal execution without export
            // Use run_with_event_loop to support plot functionality
        match fs::read_to_string(&config.filename) {
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
                    let source_clone = source.clone();
                match gui::run_with_event_loop(move || {
                        run(&source_clone)
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
                Err(e) => {
                eprintln!("Ошибка чтения файла '{}': {}", config.filename, e);
                    std::process::exit(1);
            }
        }
    }
}
