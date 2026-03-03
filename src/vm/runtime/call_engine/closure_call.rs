//! Execution of user function and closure calls (including constructors and methods).

use super::{constructor_call, method_call};
use crate::debug_println;
use crate::common::{error::LangError, value::Value, value_store::ValueStore, TaggedValue};
use crate::vm::types::VMStatus;
use crate::vm::frame::CallFrame;
use crate::vm::exceptions::ExceptionHandler;
use crate::vm::stack;
use crate::vm::global_slot::GlobalSlot;
use crate::vm::store_convert::{store_value, slot_to_value};
use crate::vm::heavy_store::HeavyStore;
use crate::common::error::ErrorType;

/// Execute a user function or closure call (including constructor and method dispatch).
/// Called from call_dispatch when function_index_resolved.is_some().
#[allow(clippy::too_many_arguments)]
pub fn execute_closure_call(
    current_ip: usize,
    function_index: usize,
    constructing_class_opt: Option<Value>,
    arity: usize,
    line: usize,
    stack: &mut Vec<TaggedValue>,
    frames: &mut Vec<CallFrame>,
    globals: &mut Vec<GlobalSlot>,
    global_names: &mut std::collections::BTreeMap<usize, String>,
    functions: &mut Vec<crate::bytecode::Function>,
    exception_handlers: &mut Vec<ExceptionHandler>,
    error_type_table: &mut Vec<String>,
    value_store: &mut ValueStore,
    heavy_store: &mut HeavyStore,
    vm_ptr: *mut crate::vm::vm::Vm,
) -> Result<VMStatus, LangError> {
    if let Some(frame) = frames.last_mut() {
        frame.call_cache_ip = Some(current_ip);
        frame.call_cache_is_user_function = true;
    }
    let function = functions[function_index].clone();

    constructor_call::set_constructing_class_for_call(
        &function,
        constructing_class_opt.as_ref(),
        &*frames,
        globals,
        global_names,
        value_store,
        heavy_store,
        vm_ptr,
    );

    debug_println!(
        "[CALL] function_index={}, functions.len()={}, function.name={}",
        function_index,
        functions.len(),
        functions.get(function_index).map(|f| f.name.as_str()).unwrap_or("?")
    );

    let mut args = Vec::new();
    let mut arg_tvs: Vec<TaggedValue> = Vec::new();

    if arity > 0 {
        let frame = frames.last().unwrap();
        if stack.len() <= frame.stack_start {
            let error = ExceptionHandler::runtime_error(
                &frames,
                format!(
                    "Not enough arguments on stack: expected {} but stack is empty",
                    arity
                ),
                line,
            );
            match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error, value_store, heavy_store) {
                Ok(()) => return Ok(VMStatus::Continue),
                Err(e) => return Err(e),
            }
        }
        let available_args = stack.len() - frame.stack_start;
        if available_args < arity {
            let error = ExceptionHandler::runtime_error(
                &frames,
                format!(
                    "Not enough arguments on stack: expected {} but got {}",
                    arity,
                    available_args
                ),
                line,
            );
            match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error, value_store, heavy_store) {
                Ok(()) => return Ok(VMStatus::Continue),
                Err(e) => return Err(e),
            }
        }
        arg_tvs.reserve(arity);
        for _ in 0..arity {
            let arg_tv = stack.pop().unwrap_or(TaggedValue::null());
            arg_tvs.push(arg_tv);
            args.push(slot_to_value(arg_tv, value_store, heavy_store));
        }
        arg_tvs.reverse();
        args.reverse();
    }

    method_call::prepare_method_args(&function, &mut args, &mut arg_tvs, value_store, heavy_store);

    if args.len() != function.arity {
        let error = ExceptionHandler::runtime_error(
            &frames,
            format!(
                "Expected {} arguments but got {}",
                function.arity, args.len()
            ),
            line,
        );
        match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error, value_store, heavy_store) {
            Ok(()) => return Ok(VMStatus::Continue),
            Err(e) => return Err(e),
        }
    }

    for (i, (arg, expected_types)) in args.iter().zip(&function.param_types).enumerate() {
        if let Some(type_names) = expected_types {
            if !crate::vm::calls::check_type_value(arg, type_names) {
                let param_name = function.param_names.get(i)
                    .map(|s| s.as_str())
                    .unwrap_or("unknown");
                let error = LangError::runtime_error_with_type(
                    format!(
                        "Argument '{}' expected type '{}', got '{}'",
                        param_name,
                        crate::vm::calls::format_type_parts(type_names),
                        crate::vm::calls::get_type_name_value(arg)
                    ),
                    line,
                    ErrorType::TypeError,
                );
                match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error, value_store, heavy_store) {
                    Ok(()) => return Ok(VMStatus::Continue),
                    Err(e) => return Err(e),
                }
            }
        }
    }

    if function.is_cached {
        use crate::bytecode::function::CacheKey;
        if let Some(cache_key) = CacheKey::new(&args) {
            if let Some(cache_rc) = &function.cache {
                let cache = cache_rc.borrow();
                if let Some(cached_result) = cache.map.get(&cache_key) {
                    let id = store_value(cached_result.clone(), value_store, heavy_store);
                    stack::push_id(stack, id);
                    return Ok(VMStatus::Continue);
                }
            }
        }
    }

    let stack_start = stack.len();
    let mut new_frame = if function.is_cached {
        CallFrame::new_with_cache(function.clone(), stack_start, arg_tvs.clone(), value_store, heavy_store)
    } else {
        CallFrame::new(function.clone(), stack_start, value_store, heavy_store)
    };
    if !function.chunk.error_type_table.is_empty() {
        *error_type_table = function.chunk.error_type_table.clone();
    }
    if !frames.is_empty() && !function.captured_vars.is_empty() {
        for captured_var in &function.captured_vars {
            if captured_var.local_slot_index >= new_frame.slots.len() {
                new_frame.slots.resize(captured_var.local_slot_index + 1, TaggedValue::null());
            }
            let ancestor_index = frames.len().saturating_sub(1 + captured_var.ancestor_depth);
            if ancestor_index < frames.len() {
                let ancestor_frame = &frames[ancestor_index];
                if captured_var.parent_slot_index < ancestor_frame.slots.len() {
                    new_frame.slots[captured_var.local_slot_index] = ancestor_frame.slots[captured_var.parent_slot_index];
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
