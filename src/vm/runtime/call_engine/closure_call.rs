//! Execution of user function and closure calls (including constructors and methods).

use super::{constructor_call, method_call};
use crate::common::error::ErrorType;
use crate::common::{
    error::LangError, value::GeneratorState, value::Value, value_store::ValueStore, TaggedValue,
};
use crate::debug_println;
use crate::vm::calls::ancestor_frame_index_for_capture;
use crate::vm::exceptions::ExceptionHandler;
use crate::vm::frame::CallFrame;
use crate::vm::global_slot::GlobalSlot;
use crate::vm::heavy_store::HeavyStore;
use crate::vm::stack;
use crate::vm::store_convert::{slot_to_value, store_value};
use crate::vm::types::VMStatus;
use std::cell::RefCell;
use std::rc::Rc;

/// Execute a user function or closure call (including constructor and method dispatch).
/// Called from call_dispatch when function_index_resolved.is_some().
#[allow(clippy::too_many_arguments)]
pub(crate) fn execute_closure_call(
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
        functions
            .get(function_index)
            .map(|f| f.name.as_str())
            .unwrap_or("?")
    );

    let mut args = Vec::new();
    let mut arg_tvs: Vec<TaggedValue> = Vec::new();

    if arity > 0 {
        let frame = frames.last().unwrap();
        if crate::vm::stack::available_in_frame(stack, frame.stack_start) == 0 {
            let error = ExceptionHandler::runtime_error(
                &frames,
                format!(
                    "Not enough arguments on stack: expected {} but stack is empty",
                    arity
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
        let available_args = crate::vm::stack::available_in_frame(stack, frame.stack_start);
        if available_args < arity {
            let error = ExceptionHandler::runtime_error(
                &frames,
                format!(
                    "Not enough arguments on stack: expected {} but got {}",
                    arity, available_args
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
        arg_tvs.reserve(arity);
        for _ in 0..arity {
            let arg_tv = stack::pop_direct(stack).unwrap_or(TaggedValue::null());
            arg_tvs.push(arg_tv);
            args.push(slot_to_value(arg_tv, value_store, heavy_store));
        }
        // Stack convention from `compile_module_method`: callee is popped first elsewhere,then this
        // loop consumes `arity` values *below* it. Typical layout bottom→top is `[receiver,
        // arg₁, ..., argₙ]` before the callee pushed on top → pops yield `[argₙ, ..., receiver]`
        // and reversing restores `[receiver, arg₁, ...]`.
        //
        // Some call sites duplicate the receiver for `GetArrayElement(method)` so the deepest
        // receiver disappears when the callee is popped but the upper receiver ends up nearer
        // the callee than the last (`this`, `array`-like) arguments. First popped value becomes
        // `this` already in left‑to‑right order → reversing wrongly swaps bindings (constructor
        // helpers like `_init(items)`). Detect the common arity‑2 `{this, array}` edge and skip.
        let skip_reverse_two_arg_this_receiver_on_top =
            arity == 2
                && function.param_names.first().map(|s| s.as_str()) == Some("this")
                && function.param_names.len() >= 2
                && crate::vm::calls::get_type_name_value(&args[0]) == "object"
                && crate::vm::calls::get_type_name_value(&args[1]) == "array";
        if !(arity == 2 && skip_reverse_two_arg_this_receiver_on_top) {
            arg_tvs.reverse();
            args.reverse();
        }
    }

    method_call::prepare_method_args(&function, &mut args, &mut arg_tvs, value_store, heavy_store);

    if args.len() != function.arity {
        let error = ExceptionHandler::runtime_error(
            &frames,
            format!(
                "Expected {} arguments but got {}",
                function.arity,
                args.len()
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

    for (i, (arg, expected_types)) in args.iter().zip(&function.param_types).enumerate() {
        if let Some(type_names) = expected_types {
            if !crate::vm::calls::check_type_value_with_globals(
                arg,
                type_names,
                globals,
                global_names,
                value_store,
                heavy_store,
            ) {
                let param_name = function
                    .param_names
                    .get(i)
                    .map(|s| s.as_str())
                    .unwrap_or("unknown");
                let error = LangError::runtime_error_with_type(
                    format!(
                        "Argument '{}' expected type '{}', got '{}'",
                        param_name,
                        crate::vm::calls::format_type_parts(type_names),
                        crate::vm::type_compat::display_value_type(arg)
                    ),
                    line,
                    ErrorType::TypeError,
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
        }
    }

    if function.is_stream {
        if !function.captured_vars.is_empty() {
            return Err(LangError::runtime_error(
                "stream fn with captured variables is not supported yet".to_string(),
                line,
            ));
        }
        let gen = GeneratorState {
            fn_index: function_index,
            ip: 0,
            slots: Vec::new(),
            finished: false,
            final_value: None,
            pending_args: Some(args),
            waiting_for_input: false,
            pending_first_send: None,
            cold_send_first_yield: None,
            pending_deferred_yield: None,
        };
        let id = store_value(
            Value::Generator(Rc::new(RefCell::new(gen))),
            value_store,
            heavy_store,
        );
        stack::push_id(stack, id);
        return Ok(VMStatus::Continue);
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

    let stack_start = crate::vm::stack::truncate_to_current_sp(stack);
    let mut new_frame = if function.is_cached {
        CallFrame::new_with_cache(
            function.clone(),
            function_index,
            stack_start,
            arg_tvs.clone(),
            value_store,
            heavy_store,
        )
    } else {
        CallFrame::new(
            function.clone(),
            function_index,
            stack_start,
            value_store,
            heavy_store,
        )
    };
    if new_frame.module_name.is_none() {
        let registry = unsafe { (*vm_ptr).get_module_registry() };
        if let Some(name) =
            crate::vm::module_object::module_name_for_function_index(&registry, function_index)
        {
            new_frame.module_name = Some(name);
        }
    }
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
            if let Some(ancestor_index) = ancestor_frame_index_for_capture(frames, captured_var) {
                if ancestor_index < frames.len() {
                    let ancestor_frame = &frames[ancestor_index];
                    if captured_var.parent_slot_index < ancestor_frame.slots.len() {
                        new_frame.slots[captured_var.local_slot_index] =
                            ancestor_frame.slots[captured_var.parent_slot_index];
                        continue;
                    }
                }
            }
            new_frame.slots[captured_var.local_slot_index] = TaggedValue::null();
        }
    }
    let param_start_index = function
        .captured_vars
        .iter()
        .map(|c| c.local_slot_index)
        .max()
        .map(|m| m.saturating_add(1))
        .unwrap_or(0);
    for (i, &arg_tv) in arg_tvs.iter().enumerate() {
        let slot_index = param_start_index + i;
        if slot_index >= new_frame.slots.len() {
            new_frame.slots.resize(slot_index + 1, TaggedValue::null());
        }
        new_frame.slots[slot_index] = arg_tv;
    }

    frames.push(new_frame);
    value_store.enter_ephemeral();
    Ok(VMStatus::Continue)
}
