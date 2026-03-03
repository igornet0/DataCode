//! Call engine: execution of Call and CallWithUnpack.
//! Dispatches to user functions, natives, constructors, and special callables.

mod call_dispatch;
mod closure_call;
mod constructor_call;
mod method_call;
mod native_call;

pub use call_dispatch::{execute_call, execute_call_with_unpack};
