pub mod vm;
pub mod context_guards;
pub mod run;
pub mod core;
pub mod global_utils;
pub mod run_context;
pub mod module_cache;
pub mod module_object;
pub mod dcb;
pub mod heavy_store;
pub mod store_convert;
pub mod table_ops;
pub mod file_ops;
pub mod file_import;
pub mod abi_bridge;
pub mod host;
pub mod native_loader;
pub mod native_registry;

pub mod natives;

pub mod types;
pub mod exception;
pub mod exceptions;
pub mod operations;
pub mod global_slot;
pub mod globals;
pub mod modules;
pub mod calls;
pub mod executor;
pub mod interpreter;
pub mod module_system;
pub mod memory;
pub mod runtime;
pub mod profile;

// Re-export core types for backward compatibility (crate::vm::frame, crate::vm::stack)
pub use core::{frame, stack};

pub use vm::Vm;
pub use types::{ExplicitRelation, ExplicitPrimaryKey, ModuleInfo};

