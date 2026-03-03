// Exception opcode handlers: BeginTry, EndTry, Catch, EndCatch, Throw, PopExceptionHandler.
// Logic preserved 1:1 from executor.rs — no semantic changes.

use crate::common::error::LangError;
use crate::vm::exceptions::ExceptionHandler;
use crate::vm::frame::CallFrame;
use crate::vm::heavy_store::HeavyStore;
use crate::vm::store_convert::load_value;
use crate::vm::types::VMStatus;
use crate::common::value_store::ValueStore;

use crate::vm::interpreter::helpers::pop_to_value_id;

pub fn op_begin_try(
    handler_index: usize,
    stack: &mut Vec<crate::common::TaggedValue>,
    frames: &mut Vec<CallFrame>,
    exception_handlers: &mut Vec<ExceptionHandler>,
    error_type_table: &mut Vec<String>,
) -> Result<VMStatus, LangError> {
    let frame = frames.last().unwrap();
    let chunk = &frame.function.chunk;

    if handler_index < chunk.exception_handlers.len() {
        let handler_info = &chunk.exception_handlers[handler_index];

        if error_type_table.is_empty() {
            *error_type_table = chunk.error_type_table.clone();
        }

        let stack_height = stack.len();
        let frame_index = frames.len() - 1;
        let handler = ExceptionHandler {
            catch_ips: handler_info.catch_ips.clone(),
            error_types: handler_info.error_types.clone(),
            error_var_slots: handler_info.error_var_slots.clone(),
            else_ip: handler_info.else_ip,
            finally_ip: handler_info.finally_ip,
            stack_height,
            had_error: false,
            frame_index,
        };
        exception_handlers.push(handler);
    } else {
        let stack_height = stack.len();
        let frame_index = frames.len() - 1;
        let handler = ExceptionHandler {
            catch_ips: Vec::new(),
            error_types: Vec::new(),
            error_var_slots: Vec::new(),
            else_ip: None,
            finally_ip: None,
            stack_height,
            had_error: false,
            frame_index,
        };
        exception_handlers.push(handler);
    }
    Ok(VMStatus::Continue)
}

pub fn op_end_try(
    frames: &mut Vec<CallFrame>,
    exception_handlers: &mut Vec<ExceptionHandler>,
) -> Result<VMStatus, LangError> {
    if let Some(handler) = exception_handlers.last_mut() {
        if !handler.had_error {
            if let Some(else_ip) = handler.else_ip {
                let frame = frames.last_mut().unwrap();
                frame.ip = else_ip;
            }
        }
        exception_handlers.pop();
    }
    Ok(VMStatus::Continue)
}

pub fn op_catch() -> Result<VMStatus, LangError> {
    Ok(VMStatus::Continue)
}

pub fn op_end_catch() -> Result<VMStatus, LangError> {
    Ok(VMStatus::Continue)
}

pub fn op_throw(
    line: usize,
    stack: &mut Vec<crate::common::TaggedValue>,
    frames: &mut Vec<CallFrame>,
    exception_handlers: &mut Vec<ExceptionHandler>,
    value_store: &mut ValueStore,
    heavy_store: &mut HeavyStore,
) -> Result<VMStatus, LangError> {
    let error_value_id = pop_to_value_id(stack, frames, exception_handlers, value_store, heavy_store)?;
    let error_value = load_value(error_value_id, value_store, heavy_store);
    let error_message = error_value.to_string();
    let error = LangError::runtime_error(error_message, line);

    match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error, value_store, heavy_store) {
        Ok(()) => {}
        Err(e) => return Err(e),
    }
    Ok(VMStatus::Continue)
}

pub fn op_pop_exception_handler(exception_handlers: &mut Vec<ExceptionHandler>) -> Result<VMStatus, LangError> {
    exception_handlers.pop();
    Ok(VMStatus::Continue)
}
