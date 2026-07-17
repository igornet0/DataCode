//! WebSocket ws_app.dc and CLI tests

use data_code::infra::cli;

#[test]
fn websocket_cli_websocket_flag_anywhere() {
    let args = vec![
        "datacode".to_string(),
        "examples/en/07-websocket/dc/ws_app.dc".to_string(),
        "--websocket".to_string(),
        "--port".to_string(),
        "8899".to_string(),
    ];
    let parsed = cli::parse_args(args).expect("parse");
    match parsed {
        cli::CliArgs::WebSocket(cfg) => {
            assert_eq!(cfg.port, 8899);
            assert_eq!(
                cfg.app_file.as_deref(),
                Some("examples/en/07-websocket/dc/ws_app.dc")
            );
        }
        other => panic!("expected WebSocket, got {:?}", other),
    }
}

#[test]
fn websocket_cli_websocket_first() {
    let args = vec![
        "datacode".to_string(),
        "--websocket".to_string(),
        "examples/en/07-websocket/dc/ws_app.dc".to_string(),
    ];
    let parsed = cli::parse_args(args).expect("parse");
    match parsed {
        cli::CliArgs::WebSocket(cfg) => {
            assert_eq!(
                cfg.app_file.as_deref(),
                Some("examples/en/07-websocket/dc/ws_app.dc")
            );
        }
        other => panic!("expected WebSocket, got {:?}", other),
    }
}

#[test]
fn websocket_cli_no_app_default() {
    let args = vec!["datacode".to_string(), "--websocket".to_string()];
    let parsed = cli::parse_args(args).expect("parse");
    match parsed {
        cli::CliArgs::WebSocket(cfg) => {
            assert!(cfg.app_file.is_none());
            assert!(!cfg.use_ve);
        }
        other => panic!("expected WebSocket, got {:?}", other),
    }
}

#[test]
fn ws_route_metadata_on_function() {
    use data_code::compile;
    let source = r#"
from websocket import configure
configure({"execute_policy": "restricted"})

@ws_route("ping")
fn ping(req) {
    return {"success": true, "message": "pong"}
}
"#;
    let (_chunk, functions) = compile(source).expect("compile ws_app snippet");
    let has_ws_route = functions.iter().any(|f| f.ws_route_type.as_deref() == Some("ping"));
    assert!(has_ws_route, "expected @ws_route on ping function");
}

#[test]
fn websocket_router_dispatch_custom_route() {
    use data_code::websocket::app::{bootstrap_app, route_handler_index};
    use data_code::websocket::router::{dispatch_message, ClientContext};
    use data_code::websocket::smb::SmbManager;
    use std::sync::{Arc, Mutex};

    let app_path = "examples/en/07-websocket/dc/ws_app.dc";
    if !std::path::Path::new(app_path).exists() {
        return;
    }
    bootstrap_app(app_path, None).expect("bootstrap");

    assert!(route_handler_index("ping").is_some());

    let ctx = ClientContext {
        smb_manager: Arc::new(Mutex::new(SmbManager::new())),
        use_ve: false,
        build_model: false,
    };
    let resp = dispatch_message(r#"{"type":"ping"}"#, &ctx);
    assert!(resp.contains("pong"), "response: {}", resp);
}
