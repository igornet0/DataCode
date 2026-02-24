// Stack operations for VM: stack holds TaggedValue (immediates + heap refs).

use crate::common::{error::LangError, value_store::{ValueId, ValueStore}};
use crate::common::TaggedValue;

/// Push a heap value by id (converts to TaggedValue on stack).
#[inline]
pub fn push_id(stack: &mut Vec<TaggedValue>, id: ValueId) {
    stack.push(TaggedValue::from_heap(id));
}
use crate::vm::frame::CallFrame;
use crate::vm::exceptions::ExceptionHandler;
use crate::vm::heavy_store::HeavyStore;

#[inline]
pub fn push(stack: &mut Vec<TaggedValue>, tv: TaggedValue) {
    stack.push(tv);
}

pub fn pop(
    stack: &mut Vec<TaggedValue>,
    frames: &mut Vec<CallFrame>,
    exception_handlers: &mut Vec<ExceptionHandler>,
    value_store: &mut ValueStore,
    heavy_store: &mut HeavyStore,
) -> Result<TaggedValue, LangError> {
    let line = if let Some(frame) = frames.last() {
        if frame.ip > 0 {
            frame.function.chunk.get_line(frame.ip - 1)
        } else {
            0
        }
    } else {
        0
    };

    if let Some(frame) = frames.last() {
        if stack.len() <= frame.stack_start {
            let error = ExceptionHandler::runtime_error(
                frames,
                "Stack underflow".to_string(),
                line,
            );
            match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error.clone(), value_store, heavy_store) {
                Ok(_) => return Err(error),
                Err(e) => return Err(e),
            }
        }
    }

    stack.pop().ok_or_else(|| {
        let error = ExceptionHandler::runtime_error(
            frames,
            "Stack underflow".to_string(),
            line,
        );
        match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error.clone(), value_store, heavy_store) {
            Ok(_) => error,
            Err(e) => e,
        }
    })
}

pub fn peek<'a>(stack: &'a [TaggedValue], distance: usize, frames: &[CallFrame]) -> Result<TaggedValue, LangError> {
    if distance >= stack.len() {
        let line = if let Some(frame) = frames.last() {
            if frame.ip > 0 {
                frame.function.chunk.get_line(frame.ip - 1)
            } else {
                0
            }
        } else {
            0
        };
        return Err(ExceptionHandler::runtime_error(
            frames,
            "Stack underflow".to_string(),
            line,
        ));
    }
    Ok(stack[stack.len() - 1 - distance])
}

