pub mod vm;
pub mod frame;
pub mod file_ops;

pub mod natives;

pub mod types;
pub mod exceptions;
pub mod operations;
pub mod stack;
pub mod globals;
pub mod modules;
pub mod calls;
pub mod executor;

// CLI and runtime modules
pub mod cli;
pub mod repl;
pub mod gui;
pub mod window_events;
pub mod websocket;

pub use vm::Vm;
pub use types::{ExplicitRelation, ExplicitPrimaryKey};

// Re-export public APIs
pub use cli::{CliArgs, WebSocketConfig, FileExecutionConfig, parse_args, print_help, print_version, version};
pub use repl::run_repl;
pub use gui::run_with_event_loop;
pub use websocket::start_websocket_server;

