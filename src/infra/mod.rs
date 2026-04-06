// Infrastructure layer: CLI, REPL, GUI, HTTP/WebSocket servers.
// VM does not depend on this module; binaries (main, datacode-server) use it.

pub mod cli;
pub mod gui;
pub mod http_server;
pub mod repl;
pub mod websocket;
pub mod window_events;

// Re-export public API for binaries
pub use cli::{
    extract_param_args, parse_args, print_help, print_version, version, CliArgs,
    FileExecutionConfig, HttpServerConfig, WebSocketConfig,
};
pub use gui::run_with_event_loop;
pub use http_server::start_http_server;
pub use repl::run_repl;
pub use websocket::start_websocket_server;
