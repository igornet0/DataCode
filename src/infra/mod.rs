// Infrastructure layer: CLI, REPL, GUI, HTTP/WebSocket servers.
// VM does not depend on this module; binaries (main, datacode-server) use it.

pub mod cli;
pub mod window_events;
pub mod gui;
pub mod repl;
pub mod websocket;
pub mod http_server;

// Re-export public API for binaries
pub use cli::{CliArgs, WebSocketConfig, HttpServerConfig, FileExecutionConfig, parse_args, print_help, print_version, version, extract_param_args};
pub use repl::run_repl;
pub use gui::run_with_event_loop;
pub use websocket::start_websocket_server;
pub use http_server::start_http_server;
