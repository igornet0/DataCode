//! Execution of native (builtin and ABI) function calls.

mod execute;
mod fast_paths;

pub(crate) use execute::execute_native_call;
