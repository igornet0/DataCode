//! A DCP session on a client thread where ws_app.dc is loaded must not import the
//! developer's `.dc` modules from the app directory: loading ws_app.dc points the
//! thread's base path there, and a session must not inherit it.
//!
//! `bootstrap_app` sets process-wide state, so this binary holds a single test.

mod common;

use common::dcp::run_session;
use data_code::websocket::app::{bootstrap_app, route_handler_index};

#[test]
fn session_does_not_import_modules_from_the_app_directory() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(
        dir.path().join("ws_app.dc"),
        r#"from app_helper import greeting

@ws_route("hello")
fn hello(req) {
    return {"success": true, "message": greeting()}
}
"#,
    )
    .unwrap();
    std::fs::write(
        dir.path().join("app_helper.dc"),
        "SECRET = \"app-module-secret\"\n\nfn greeting() {\n    return \"hi\"\n}\n",
    )
    .unwrap();
    bootstrap_app(dir.path().join("ws_app.dc").to_str().unwrap(), None).expect("bootstrap");

    let responses = std::thread::spawn(|| {
        // Load ws_app.dc on this client thread, as the first routed message does.
        assert!(route_handler_index("hello").is_some(), "@ws_route not loaded on the client thread");
        [
            run_session("import app_helper\nprint(app_helper.SECRET)", &[]),
            run_session("from app_helper import SECRET\nprint(SECRET)", &[]),
        ]
    })
    .join()
    .expect("client thread");

    for resp in responses {
        assert!(
            !resp.raw.contains("app-module-secret"),
            "a session imported a module from the ws_app directory: {}",
            resp.raw
        );
    }
}
