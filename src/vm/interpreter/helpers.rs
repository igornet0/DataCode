// Shared helpers for interpreter opcode handlers.

use crate::common::{error::LangError, value_store::{ValueId, ValueStore}};
use crate::vm::exceptions::ExceptionHandler;
use crate::vm::frame::CallFrame;
use crate::vm::heavy_store::HeavyStore;
use crate::vm::stack;
use crate::vm::store_convert::tagged_to_value_id;
use crate::common::TaggedValue;

/// Pop one TaggedValue and convert to ValueId (for opcodes that need store id).
pub fn pop_to_value_id(
    stack: &mut Vec<TaggedValue>,
    frames: &mut Vec<CallFrame>,
    exception_handlers: &mut Vec<ExceptionHandler>,
    value_store: &mut ValueStore,
    heavy_store: &mut HeavyStore,
) -> Result<ValueId, LangError> {
    let tv = stack::pop(stack, frames, exception_handlers, value_store, heavy_store)?;
    Ok(tagged_to_value_id(tv, value_store))
}
