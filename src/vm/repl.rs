// REPL mode for interactive code execution

use crate::run;
use crate::common::value::Value;
use crate::vm::cli::version;

/// Run REPL mode
pub fn run_repl() -> Result<(), String> {
    println!("ДатаКод v{} - Bytecode VM", version());
    println!("Введите код (Ctrl+D или 'exit' для выхода):");
    println!();
    
    let mut input = String::new();
    loop {
        use std::io::{self, Write};
        
        // Show prompt
        print!("datacode> ");
        io::stdout().flush().unwrap();
        
        match io::stdin().read_line(&mut input) {
            Ok(0) => {
                // EOF (Ctrl+D)
                println!("\nДо свидания!");
                break;
            }
            Ok(_) => {
                let trimmed = input.trim();
                
                // Check for exit command
                if trimmed == "exit" || trimmed == "quit" {
                    println!("До свидания!");
                    break;
                }
                
                if trimmed.is_empty() {
                    input.clear();
                    continue;
                }
                
                // Execute code
                // Note: run() is called inside run_with_event_loop, so PlotSystem is already initialized
                match run(trimmed) {
                    Ok(value) => {
                        // If there's a result, show it
                        if !matches!(value, Value::Null) {
                            println!("=> {:?}", value);
                        }
                    }
                    Err(e) => {
                        eprintln!("Ошибка: {}", e);
                    }
                }
                input.clear();
            }
            Err(e) => {
                eprintln!("Ошибка чтения: {}", e);
                break;
            }
        }
    }
    Ok(())
}

