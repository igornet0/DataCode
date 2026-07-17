//! Early `dict.clear()` on plain [`ValueCell::Object`] — no args materialization.

use crate::common::error::{ErrorType, LangError};
use crate::common::value_store::ValueStore;
use crate::common::TaggedValue;
use crate::vm::exceptions::ExceptionHandler;
use crate::vm::frame::CallFrame;
use crate::vm::heavy_store::HeavyStore;
use crate::vm::native_indices::builtin;
use crate::vm::stack;
use crate::vm::types::VMStatus;

/// Stack `[obj]` → clear plain dict in place, push obj id.
pub(crate) fn object_clear_from_stack(
    line: usize,
    stack: &mut Vec<TaggedValue>,
    frames: &mut Vec<CallFrame>,
    exception_handlers: &mut Vec<ExceptionHandler>,
    value_store: &mut ValueStore,
    heavy_store: &mut HeavyStore,
) -> Result<VMStatus, LangError> {
    let obj_tv = stack::pop(stack, frames, exception_handlers, value_store, heavy_store)?;
    if !obj_tv.is_heap() {
        let error = ExceptionHandler::runtime_error_with_type(
            frames,
            "TypeError: dict.clear() expects a plain dict object".to_string(),
            line,
            ErrorType::TypeError,
        );
        return ExceptionHandler::handle_exception_vm(
            stack,
            frames,
            exception_handlers,
            error,
            value_store,
            heavy_store,
        );
    }
    let obj_id = obj_tv.get_heap_id();
    if value_store.plain_object_clear(obj_id) || value_store.plain_set_clear(obj_id) {
        stack::push_id(stack, obj_id);
        return Ok(VMStatus::Continue);
    }
    let error = ExceptionHandler::runtime_error_with_type(
        frames,
        "TypeError: .clear() expects a plain dict or set object".to_string(),
        line,
        ErrorType::TypeError,
    );
    return ExceptionHandler::handle_exception_vm(
        stack,
        frames,
        exception_handlers,
        error,
        value_store,
        heavy_store,
    );
}

pub(super) fn try_object_clear_early(
    native_index: usize,
    arity: usize,
    line: usize,
    stack: &mut Vec<TaggedValue>,
    frames: &mut Vec<CallFrame>,
    exception_handlers: &mut Vec<ExceptionHandler>,
    value_store: &mut ValueStore,
    heavy_store: &mut HeavyStore,
) -> Option<Result<VMStatus, LangError>> {
    if native_index != builtin::OBJECT_CLEAR || arity != 1 {
        return None;
    }
    let frame = frames.last()?;
    if crate::vm::stack::available_in_frame(stack, frame.stack_start) < 1 {
        return None;
    }
    let obj_tv = crate::vm::stack::pop_direct(stack).unwrap_or(TaggedValue::null());
    stack::push(stack, obj_tv);
    Some(object_clear_from_stack(
        line,
        stack,
        frames,
        exception_handlers,
        value_store,
        heavy_store,
    ))
}
