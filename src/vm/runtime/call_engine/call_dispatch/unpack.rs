//! `CallWithUnpack` opcode: kwargs object unpacking into a function call.

use crate::common::{
    error::LangError,
    value::Value,
    value_store::ValueStore,
    TaggedValue,
};
use crate::vm::exceptions::ExceptionHandler;
use crate::vm::frame::CallFrame;
use crate::vm::heavy_store::HeavyStore;
use crate::vm::stack;
use crate::vm::store_convert::{load_value, store_value, tagged_to_value_id};
use crate::vm::types::VMStatus;
/// Execute CallWithUnpack(unpack_arity): kwargs object unpacking into function call.
#[allow(clippy::too_many_arguments)]
pub(crate) fn execute_call_with_unpack(
    unpack_arity: usize,
    line: usize,
    stack: &mut Vec<TaggedValue>,
    frames: &mut Vec<CallFrame>,
    functions: &mut Vec<crate::bytecode::Function>,
    exception_handlers: &mut Vec<ExceptionHandler>,
    error_type_table: &mut Vec<String>,
    value_store: &mut ValueStore,
    heavy_store: &mut HeavyStore,
    vm_ptr: *mut crate::vm::vm::Vm,
) -> Result<VMStatus, LangError> {
    // Body extracted from executor OpCode::CallWithUnpack
    let callee_tv = stack::pop(stack, frames, exception_handlers, value_store, heavy_store)?;
    if unpack_arity != 1 {
        let error = ExceptionHandler::runtime_error(
            &frames,
            format!(
                "CallWithUnpack expects 1 argument (kwargs object), got {}",
                unpack_arity
            ),
            line,
        );
        match ExceptionHandler::handle_exception(
            stack,
            frames,
            exception_handlers,
            error,
            value_store,
            heavy_store,
        ) {
            Ok(()) => return Ok(VMStatus::Continue),
            Err(e) => return Err(e),
        }
    }
    let kwargs_tv = stack::pop(stack, frames, exception_handlers, value_store, heavy_store)?;
    let kwargs_id = tagged_to_value_id(kwargs_tv, value_store);
    let kwargs_val = load_value(kwargs_id, value_store, heavy_store);
    let obj_map = match &kwargs_val {
        Value::Object(rc) => rc.borrow().clone(),
        _ => {
            let error = ExceptionHandler::runtime_error(
                &frames,
                "** unpacking in call requires an object (dict), not a value of another type"
                    .to_string(),
                line,
            );
            match ExceptionHandler::handle_exception(
                stack,
                frames,
                exception_handlers,
                error,
                value_store,
                heavy_store,
            ) {
                Ok(()) => return Ok(VMStatus::Continue),
                Err(e) => return Err(e),
            }
        }
    };
    let callee_id = tagged_to_value_id(callee_tv, value_store);
    let callee_val = load_value(callee_id, value_store, heavy_store);
    let (_function_index, function) = match &callee_val {
        Value::Function(i) if *i < functions.len() => (*i, functions[*i].clone()),
        Value::ModuleFunction {
            module_uid,
            local_index,
        } => match unsafe { (*vm_ptr).get_module_function_index(*module_uid, *local_index) } {
            Some(real_idx) if real_idx < functions.len() => (real_idx, functions[real_idx].clone()),
            _ => {
                let error = ExceptionHandler::runtime_error(
                    &frames,
                    "** unpacking is only supported for user-defined functions".to_string(),
                    line,
                );
                match ExceptionHandler::handle_exception(
                    stack,
                    frames,
                    exception_handlers,
                    error,
                    value_store,
                    heavy_store,
                ) {
                    Ok(()) => return Ok(VMStatus::Continue),
                    Err(e) => return Err(e),
                }
            }
        },
        _ => {
            let error = ExceptionHandler::runtime_error(
                &frames,
                "** unpacking is only supported for user-defined functions".to_string(),
                line,
            );
            match ExceptionHandler::handle_exception(
                stack,
                frames,
                exception_handlers,
                error,
                value_store,
                heavy_store,
            ) {
                Ok(()) => return Ok(VMStatus::Continue),
                Err(e) => return Err(e),
            }
        }
    };
    let param_names = &function.param_names;
    let obj_keys: std::collections::HashSet<&String> = obj_map.keys().collect();
    let param_set: std::collections::HashSet<&String> = param_names.iter().collect();
    if obj_keys != param_set {
        let expected: Vec<&str> = param_names.iter().map(|s| s.as_str()).collect();
        let got: Vec<&str> = obj_map.keys().map(|s| s.as_str()).collect();
        let error = ExceptionHandler::runtime_error(
            &frames,
            format!(
                "Object keys must match function parameters. Expected keys: [{}], got keys: [{}]",
                expected.join(", "),
                got.join(", ")
            ),
            line,
        );
        match ExceptionHandler::handle_exception(
            stack,
            frames,
            exception_handlers,
            error,
            value_store,
            heavy_store,
        ) {
            Ok(()) => return Ok(VMStatus::Continue),
            Err(e) => return Err(e),
        }
    }
    let mut arg_tvs = Vec::with_capacity(param_names.len());
    for p in param_names {
        let v = obj_map.get(p).cloned().unwrap_or(Value::Null);
        let id = store_value(v, value_store, heavy_store);
        arg_tvs.push(TaggedValue::from_heap(id));
    }
    let stack_start = stack.len();
    let mut new_frame = if function.is_cached {
        CallFrame::new_with_cache(
            function.clone(),
            stack_start,
            arg_tvs.clone(),
            value_store,
            heavy_store,
        )
    } else {
        CallFrame::new(function.clone(), stack_start, value_store, heavy_store)
    };
    if !function.chunk.error_type_table.is_empty() {
        *error_type_table = function.chunk.error_type_table.clone();
    }
    if !frames.is_empty() && !function.captured_vars.is_empty() {
        for captured_var in &function.captured_vars {
            if captured_var.local_slot_index >= new_frame.slots.len() {
                new_frame
                    .slots
                    .resize(captured_var.local_slot_index + 1, TaggedValue::null());
            }
            let ancestor_index = frames.len().saturating_sub(1 + captured_var.ancestor_depth);
            if ancestor_index < frames.len() {
                let ancestor_frame = &frames[ancestor_index];
                if captured_var.parent_slot_index < ancestor_frame.slots.len() {
                    new_frame.slots[captured_var.local_slot_index] =
                        ancestor_frame.slots[captured_var.parent_slot_index];
                } else {
                    new_frame.slots[captured_var.local_slot_index] = TaggedValue::null();
                }
            } else {
                new_frame.slots[captured_var.local_slot_index] = TaggedValue::null();
            }
        }
    }
    let param_start_index = function.captured_vars.len();
    for (i, &arg_tv) in arg_tvs.iter().enumerate() {
        let slot_index = param_start_index + i;
        if slot_index >= new_frame.slots.len() {
            new_frame.slots.resize(slot_index + 1, TaggedValue::null());
        }
        new_frame.slots[slot_index] = arg_tv;
    }
    frames.push(new_frame);
    Ok(VMStatus::Continue)
}
