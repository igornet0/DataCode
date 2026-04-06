//! Call Engine: execution of `Call` and `CallWithUnpack` opcodes.

mod dispatch_arms;
mod execute_call;
mod resolve_callee;
mod unpack;

pub(crate) use execute_call::execute_call;
pub(crate) use unpack::execute_call_with_unpack;
