// Stack operations for VM

use crate::common::{error::LangError, value::Value};
use crate::vm::frame::CallFrame;
use crate::vm::exceptions::ExceptionHandler;

pub fn push(stack: &mut Vec<Value>, value: Value) {
    stack.push(value);
}

pub fn pop(
    stack: &mut Vec<Value>,
    frames: &mut Vec<CallFrame>,
    exception_handlers: &mut Vec<ExceptionHandler>,
) -> Result<Value, LangError> {
    let line = if let Some(frame) = frames.last() {
        if frame.ip > 0 {
            frame.function.chunk.get_line(frame.ip - 1)
        } else {
            0
        }
    } else {
        0
    };
    
    // Проверяем, что стек не пуст относительно stack_start текущего фрейма
    if let Some(frame) = frames.last() {
        if stack.len() <= frame.stack_start {
            let error = ExceptionHandler::runtime_error(
                frames,
                "Stack underflow".to_string(),
                line,
            );
            // Attempt to handle the exception, but if it fails, return the original error
            match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error.clone()) {
                Ok(_) => return Err(error), // If handled, return the error to signal control flow change
                Err(e) => return Err(e), // If not handled, propagate the error
            }
        }
    }
    
    stack.pop().ok_or_else(|| {
        let error = ExceptionHandler::runtime_error(
            frames,
            "Stack underflow".to_string(),
            line,
        );
        // Attempt to handle the exception, but if it fails, return the original error
        match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error.clone()) {
            Ok(_) => error, // If handled, return the error to signal control flow change
            Err(e) => e, // If not handled, propagate the error
        }
    })
}

pub fn peek<'a>(stack: &'a [Value], distance: usize, frames: &[CallFrame]) -> Result<&'a Value, LangError> {
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
    Ok(&stack[stack.len() - 1 - distance])
}

