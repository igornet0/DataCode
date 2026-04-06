// Bytecode interpreter: dispatch and opcode handlers.
// One opcode = one function.

pub mod arithmetic;
pub mod comparison;
pub mod control_flow;
pub(crate) mod element;
pub(crate) use element as element_ops;
pub mod for_iterable;
pub(crate) mod helpers;
pub mod memory;
pub mod object;
pub mod stack_ops;
