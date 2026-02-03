// datacode-server binary: HTTP server for DataCode.
// Usage: datacode-server [app.dc] [--host HOST] [--port PORT]

use data_code::vm::{http_server, cli};
use std::env;

fn main() {
    let args: Vec<String> = env::args().collect();
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

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--host" => {
                if i + 1 < args.len() {
                    host = args[i + 1].clone();
                    i += 2;
                } else {
                    eprintln!("Error: --host requires a value");
                    std::process::exit(1);
                }
            }
            "--port" => {
                if i + 1 < args.len() {
                    if let Ok(p) = args[i + 1].parse::<u16>() {
                        port = p;
                        i += 2;
                    } else {
                        eprintln!("Error: invalid port number");
                        std::process::exit(1);
                    }
                } else {
                    eprintln!("Error: --port requires a value");
                    std::process::exit(1);
                }
            }
            "-h" | "--help" => {
                println!("datacode-server - HTTP server for DataCode");
                println!();
                println!("Usage: datacode-server [app.dc] [--host HOST] [--port PORT]");
                println!("  app.dc        Optional app file with @route handlers");
                println!("  --host HOST   Bind address (default: 127.0.0.1)");
                println!("  --port PORT   Port (default: 8080)");
                println!("  --help        Show this help");
                println!();
                println!("Environment: DATACODE_HTTP_ADDRESS=host:port");
                std::process::exit(0);
            }
            arg if arg.ends_with(".dc") => {
                app_file = Some(arg.to_string());
                i += 1;
            }
            _ => {
                i += 1;
            }
        }
    }

    let config = cli::HttpServerConfig {
        host,
        port,
        app_file,
    };
    if let Err(e) = http_server::start_http_server(config) {
        eprintln!("{}", e);
        std::process::exit(1);
    }
}
