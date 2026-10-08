//! ws_app.dc configuration reaches every client thread (issue #15), sessions are
//! `restricted` unless the developer opts in (issue #10), and writes stay denied
//! in a session even under `allow_all` (issue #9).
//!
//! `bootstrap_app` sets process-wide state, so this binary holds a single
//! sequential test.

mod common;

use common::dcp::run_session;
use data_code::websocket::app::{app_config, bootstrap_app, route_handler_index};
use data_code::websocket::router::{dispatch_message, ClientContext};
use data_code::websocket::smb::SmbManager;
use data_code::PermissionPolicy;
use std::sync::{Arc, Mutex};

const PROCESS_EXEC: &str = "import system\nprint(system.process.exec(\"echo developer-opt-in\"))";

fn on_client_thread<T: Send + 'static>(f: impl FnOnce() -> T + Send + 'static) -> T {
    std::thread::spawn(f).join().expect("client thread")
}

#[test]
fn ws_app_policy_and_routes_apply_to_client_threads() {
    // 1. No ws_app.dc: sessions are sandboxed by default.
    assert_eq!(app_config().execute_permission_policy, PermissionPolicy::Restricted);
    let resp = on_client_thread(|| run_session(PROCESS_EXEC, &[]));
    assert!(
        resp.output.contains("permission denied: process.exec"),
        "default session must deny process.exec: {}",
        resp.raw
    );

    // 2. Developer opts in to allow_all and registers a route.
    let dir = tempfile::tempdir().unwrap();
    let app = dir.path().join("ws_app.dc");
    std::fs::write(
        &app,
        r#"from websocket import configure
configure({"execute_policy": "allow_all"})

@ws_route("ping")
fn ping(req) {
    return {"success": true, "message": "pong"}
}
"#,
    )
    .unwrap();
    bootstrap_app(app.to_str().unwrap(), None).expect("bootstrap");

    // 3. Config and routes are visible on a fresh client thread.
    let (policy, route, reply) = on_client_thread(|| {
        let ctx = ClientContext {
            smb_manager: Arc::new(Mutex::new(SmbManager::new())),
            build_model: false,
        };
        (
            app_config().execute_permission_policy,
            route_handler_index("ping"),
            dispatch_message(r#"{"type":"ping"}"#, &ctx),
        )
    });
    assert_eq!(policy, PermissionPolicy::AllowAll);
    assert!(route.is_some(), "@ws_route not found on a client thread");
    assert!(reply.contains("pong"), "route handler did not run on a client thread: {reply}");

    // 4. The opt-in applies to DCP sessions on client threads…
    let resp = on_client_thread(|| run_session(PROCESS_EXEC, &[]));
    assert!(resp.output.contains("developer-opt-in"), "allow_all opt-in not applied: {}", resp.raw);

    // 5. …but writes to the server disk stay denied in a session.
    let name = "dc_policy_probe_out.csv";
    let target = std::env::current_dir().unwrap().join(name);
    let _ = std::fs::remove_file(&target);
    let code = format!("t = table([[1]], [\"a\"])\nprint(save(t, \"{name}\"))");
    let resp = on_client_thread(move || run_session(&code, &[]));
    let created = target.exists();
    let _ = std::fs::remove_file(&target);
    assert!(!created, "save() wrote to the server disk under allow_all: {}", resp.raw);
    assert!(!resp.success, "save() must be denied in a session: {}", resp.raw);
}
