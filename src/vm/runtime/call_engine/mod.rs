//! Call engine: execution of Call and CallWithUnpack.
//! Dispatches to user functions, natives, constructors, and special callables.

mod call_dispatch;
mod closure_call;
mod constructor_call;
mod method_call;
mod native_call;

pub(crate) use call_dispatch::{execute_call, execute_call_variadic, execute_call_with_unpack};
pub(crate) use native_call::{
    heappop_store, heappop_unpack2_locals, heappush_flat_pair, heappush_two_slot_pair,
    object_clear_from_stack, object_get_from_stack, set_mut_integral_from_stack, SetIntegralMut,
};
