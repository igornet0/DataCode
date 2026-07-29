// Публичный API языка DataCode (новая архитектура Bytecode + VM)

#[cfg(feature = "allocator_jemalloc")]
#[global_allocator]
static GLOBAL: jemallocator::Jemalloc = jemallocator::Jemalloc;

pub mod abi;
pub mod abi_policy;
pub mod bytecode;
pub mod common;
pub mod compiler;
pub mod compute;
pub mod dcmodule;
pub mod dpm;
pub mod infra;
pub mod lexer;
pub mod parser;
pub mod semantic;
pub mod vm;

#[path = "lib/crypto/mod.rs"]
pub mod crypto;
#[path = "lib/database_engine/mod.rs"]
pub mod database_engine;
#[path = "lib/plot/mod.rs"]
pub mod plot;
#[path = "lib/settings_env/mod.rs"]
pub mod settings_env;
#[path = "lib/sqlite_export/mod.rs"]
pub mod sqlite_export;
#[path = "lib/system/mod.rs"]
pub mod system;
#[path = "lib/uuid/mod.rs"]
pub mod uuid;
#[path = "lib/heapq/mod.rs"]
pub mod heapq;
#[path = "lib/pathfind/mod.rs"]
pub mod pathfind;
#[path = "lib/grid/mod.rs"]
pub mod grid;
#[path = "lib/file_io/mod.rs"]
pub mod file_io;
#[path = "lib/archive/mod.rs"]
pub mod archive;
#[path = "lib/datasource/mod.rs"]
pub mod datasource;
#[path = "lib/dcp/mod.rs"]
pub mod dcp;
#[path = "lib/websocket/mod.rs"]
pub mod websocket;

mod run_api;

// Публичный API для запуска интерпретатора и компиляции
pub use bytecode::Chunk;
pub use common::{error::LangError, value::Value};
pub use run_api::{
    compile, extract_globals_from_vm, get_main_entry_params, run, run_debug, run_lib_file,
    run_with_base_path, run_with_existing_vm, run_with_options, run_with_vm, run_with_vm_and_path,
    run_with_vm_with_args, run_with_vm_with_args_and_lib, run_with_vm_with_policy, PreloadContext,
    RunOptions,
};
pub use vm::PermissionPolicy;
pub use vm::Vm;

/// Crate-internal helpers used by VM / file import (export remapping, parse-time preload).
pub(crate) use run_api::{
    module_uid, preload_native_call_registry_for_parse, preload_operator_registry_for_parse,
    remap_function_constants_in_chunks, remap_native_indices_in_exports,
    replace_function_with_module_function_in_exports,
};
