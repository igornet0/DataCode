pub mod abi_bridge;
pub mod array_view;
pub mod context_guards;
pub mod core;
pub mod dcb;
pub mod file_import;
pub mod file_ops;
pub mod global_utils;
pub mod heavy_store;
pub mod host;
pub mod import_scan;
pub mod iterable;
pub mod module_cache;
pub mod module_object;
pub mod native_call_registry;
pub mod native_loader;
pub mod native_registry;
pub mod operator_registry;
pub mod run;
pub mod run_context;
pub mod store_convert;
pub mod table_ops;
pub mod vm;

pub mod natives;

pub mod calls;
pub mod exception;
pub mod exceptions;
pub mod executor;
pub mod generator;
pub mod global_slot;
pub mod globals;
pub mod interpreter;
pub mod memory;
pub mod module_system;
pub mod modules;
pub mod operations;
pub mod permission_policy;
pub mod profile;
pub mod runtime;
pub mod types;

// Re-export core types for backward compatibility (crate::vm::frame, crate::vm::stack)
pub use core::{frame, stack};

pub use native_call_registry::{NativeCallParamRegistry, SharedNativeCallParamRegistry};
pub use operator_registry::{
    Associativity, OperatorInfo, OperatorRegistry, SharedOperatorRegistry,
};
pub use permission_policy::PermissionPolicy;
pub use types::{ExplicitPrimaryKey, ExplicitRelation, ModuleInfo};
pub use vm::Vm;
