//! `@init` invocation from class constructors (canonical `this` [`ValueId`]).

use crate::common::error::LangError;
use crate::common::value_store::{ValueId, ValueStore};
use crate::common::TaggedValue;
use crate::vm::exceptions::ExceptionHandler;
use crate::vm::frame::CallFrame;
use crate::vm::heavy_store::HeavyStore;
use crate::vm::store_convert::tagged_to_value_id;
use crate::vm::types::VMStatus;

pub fn op_invoke_special_init(
    this_slot: usize,
    param_count: usize,
    line: usize,
    stack: &mut Vec<TaggedValue>,
    frames: &mut Vec<CallFrame>,
    exception_handlers: &mut Vec<ExceptionHandler>,
    value_store: &mut ValueStore,
    heavy_store: &mut HeavyStore,
) -> Result<VMStatus, LangError> {
    let frame = frames.last_mut().unwrap();
    if this_slot >= frame.slots.len() {
        frame.ensure_slot(this_slot);
    }
    let this_id = tagged_to_value_id(frame.slots[this_slot], value_store);
    let mut extra: Vec<ValueId> = Vec::with_capacity(param_count);
    for i in 0..param_count {
        if i >= frame.slots.len() {
            frame.ensure_slot(i);
        }
        extra.push(tagged_to_value_id(frame.slots[i], value_store));
    }
    match crate::vm::special_methods::dispatch_special_by_id(this_id, "@init", &extra) {
        Ok(Some(_)) | Ok(None) => Ok(VMStatus::Continue),
        Err(e) => {
            let error = ExceptionHandler::runtime_error(frames, e.to_string(), line);
            match ExceptionHandler::handle_exception(
                stack,
                frames,
                exception_handlers,
                error,
                value_store,
                heavy_store,
            ) {
                Ok(()) => Ok(VMStatus::Continue),
                Err(err) => Err(err),
            }
        }
    }
}
