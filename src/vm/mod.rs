pub mod vm;
pub mod run_context;
pub mod module_cache;
pub mod module_object;
pub mod dcb;
pub mod frame;
pub mod heavy_store;
pub mod store_convert;
pub mod table_ops;
pub mod file_ops;
pub mod file_import;
pub mod abi_bridge;
pub mod native_loader;

pub mod natives;

pub mod types;
pub mod exceptions;
pub mod operations;
pub mod stack;
pub mod global_slot;
pub mod globals;
pub mod modules;
pub mod calls;
pub mod executor;
pub mod profile;

// CLI and runtime modules
pub mod cli;
pub mod repl;
pub mod gui;
pub mod window_events;
pub mod websocket;
pub mod http_server;

pub use vm::Vm;
pub use types::{ExplicitRelation, ExplicitPrimaryKey};

// Re-export public APIs
pub use cli::{CliArgs, WebSocketConfig, HttpServerConfig, FileExecutionConfig, parse_args, print_help, print_version, version};
pub use repl::run_repl;
pub use gui::run_with_event_loop;
pub use websocket::start_websocket_server;
pub use http_server::start_http_server;

