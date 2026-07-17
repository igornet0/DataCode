//! Execution of native (builtin and ABI) function calls.

mod execute;
mod fast_paths;
mod heapq_fast;
mod object_clear_fast;
mod object_get_fast;
mod set_fast;

pub(crate) use execute::execute_native_call;
pub(crate) use heapq_fast::{heappop_store, heappop_unpack2_locals, heappush_flat_pair, heappush_two_slot_pair};
pub(crate) use object_clear_fast::object_clear_from_stack;
pub(crate) use object_get_fast::object_get_from_stack;
pub(crate) use set_fast::{set_mut_integral_from_stack, SetIntegralMut};
