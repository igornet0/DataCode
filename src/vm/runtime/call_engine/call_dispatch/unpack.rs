//! `CallWithUnpack` / `CallVariadic` opcodes: kwargs and variadic function calls.

use crate::common::{
    error::LangError,
    value::Value,
    value_store::ValueStore,
    TaggedValue,
};
use crate::vm::calls::ancestor_frame_index_for_capture;
use crate::vm::exceptions::ExceptionHandler;
use crate::vm::frame::CallFrame;
use crate::vm::heavy_store::HeavyStore;
use crate::vm::stack;
use crate::vm::store_convert::{load_value, store_value, tagged_to_value_id};
use crate::vm::types::VMStatus;
use crate::vm::variadic_bind::{bind_function_args, bind_native_varkw_args};
use std::collections::HashMap;

fn pop_value(
    stack: &mut Vec<TaggedValue>,
    frames: &mut Vec<CallFrame>,
    exception_handlers: &mut Vec<ExceptionHandler>,
    value_store: &mut ValueStore,
    heavy_store: &mut HeavyStore,
) -> Result<Value, LangError> {
    let tv = stack::pop(stack, frames, exception_handlers, value_store, heavy_store)?;
    let id = tagged_to_value_id(tv, value_store);
    Ok(load_value(id, value_store, heavy_store))
}

fn push_user_frame(
    function_index: usize,
    function: crate::bytecode::Function,
    arg_tvs: Vec<TaggedValue>,
    stack: &mut Vec<TaggedValue>,
    frames: &mut Vec<CallFrame>,
    error_type_table: &mut Vec<String>,
    value_store: &mut ValueStore,
    heavy_store: &mut HeavyStore,
) -> Result<(), LangError> {
    let stack_start = crate::vm::stack::logical_len(stack);
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
    let param_start_index = function.captured_vars.len();
    for (i, &arg_tv) in arg_tvs.iter().enumerate() {
        let slot_index = param_start_index + i;
        if slot_index >= new_frame.slots.len() {
            new_frame.slots.resize(slot_index + 1, TaggedValue::null());
        }
        new_frame.slots[slot_index] = arg_tv;
    }
    frames.push(new_frame);
    value_store.enter_ephemeral();
    Ok(())
}

/// Execute CallVariadic(packed): `(n_pos) | (n_star << 8) | (n_named << 16) | (n_starstar << 24)`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn execute_call_variadic(
    packed: u32,
    line: usize,
    stack: &mut Vec<TaggedValue>,
    frames: &mut Vec<CallFrame>,
    globals: &mut Vec<crate::vm::global_slot::GlobalSlot>,
    _global_names: &mut std::collections::BTreeMap<usize, String>,
    explicit_global_names: &std::collections::BTreeMap<usize, String>,
    functions: &mut Vec<crate::bytecode::Function>,
    natives: &[crate::vm::host::HostEntry],
    exception_handlers: &mut Vec<ExceptionHandler>,
    error_type_table: &mut Vec<String>,
    explicit_relations: &mut Vec<crate::vm::types::ExplicitRelation>,
    explicit_primary_keys: &mut Vec<crate::vm::types::ExplicitPrimaryKey>,
    abi_natives: &mut Vec<crate::abi::NativeAbiFn>,
    value_store: &mut ValueStore,
    heavy_store: &mut HeavyStore,
    native_args_buffer: &mut Vec<Value>,
    reusable_native_arg_ids: &mut Vec<crate::common::value_store::ValueId>,
    reusable_all_popped: &mut Vec<Value>,
    vm_ptr: *mut crate::vm::vm::Vm,
) -> Result<VMStatus, LangError> {
    let n_pos = (packed & 0xFF) as usize;
    let n_star = ((packed >> 8) & 0xFF) as usize;
    let n_named = ((packed >> 16) & 0xFF) as usize;
    let n_starstar = ((packed >> 24) & 0xFF) as usize;

    let callee_tv = stack::pop(stack, frames, exception_handlers, value_store, heavy_store)?;
    let callee_id = tagged_to_value_id(callee_tv, value_store);
    let callee_val = load_value(callee_id, value_store, heavy_store);

    let mut starstar_vals = Vec::with_capacity(n_starstar);
    for _ in 0..n_starstar {
        starstar_vals.push(pop_value(
            stack,
            frames,
            exception_handlers,
            value_store,
            heavy_store,
        )?);
    }
    let mut named = HashMap::new();
    for _ in 0..n_named {
        let val = pop_value(
            stack,
            frames,
            exception_handlers,
            value_store,
            heavy_store,
        )?;
        let key_val = pop_value(
            stack,
            frames,
            exception_handlers,
            value_store,
            heavy_store,
        )?;
        let Value::String(key) = key_val else {
            return Err(LangError::runtime_error(
                "named argument key must be a string".to_string(),
                line,
            ));
        };
        if named.contains_key(&key) {
            return Err(LangError::runtime_error(
                format!("got multiple values for argument '{}'", key),
                line,
            ));
        }
        named.insert(key, val);
    }
    let mut star_vals = Vec::with_capacity(n_star);
    for _ in 0..n_star {
        star_vals.push(pop_value(
            stack,
            frames,
            exception_handlers,
            value_store,
            heavy_store,
        )?);
    }
    let mut pos_rev = Vec::with_capacity(n_pos);
    for _ in 0..n_pos {
        pos_rev.push(pop_value(
            stack,
            frames,
            exception_handlers,
            value_store,
            heavy_store,
        )?);
    }
    let positional: Vec<Value> = pos_rev.into_iter().rev().collect();

    match &callee_val {
        Value::Function(i) if *i < functions.len() => {
            let function_index = *i;
            let function = functions[function_index].clone();
            let bound = bind_function_args(
                &function,
                positional,
                named,
                &star_vals,
                &starstar_vals,
            )
            .map_err(|e| LangError::runtime_error(e, line))?;
            let arg_tvs: Vec<TaggedValue> = bound
                .iter()
                .map(|v| TaggedValue::from_heap(store_value(v.clone(), value_store, heavy_store)))
                .collect();
            push_user_frame(
                function_index,
                function,
                arg_tvs,
                stack,
                frames,
                error_type_table,
                value_store,
                heavy_store,
            )?;
        }
        Value::NativeFunction(native_index) => {
            let native_index = *native_index;
            let bound = {
                let name = crate::vm::native_indices::builtin_native_name(
                    match &callee_val {
                        Value::NativeFunction(i) => *i,
                        _ => unreachable!(),
                    },
                );
                let param_names = crate::compiler::natives::get_native_function_params(name)
                    .unwrap_or_default();
                let varkw = crate::compiler::natives::get_native_varkw_param(name);
                bind_native_varkw_args(
                    &param_names,
                    varkw,
                    positional,
                    named,
                    &star_vals,
                    &starstar_vals,
                )
                .map_err(|e| LangError::runtime_error(e, line))?
            };
            let arity = bound.len();
            for v in bound {
                let id = store_value(v, value_store, heavy_store);
                stack::push(stack, TaggedValue::from_heap(id));
            }
            return super::super::native_call::execute_native_call(
                native_index,
                arity,
                line,
                stack,
                frames,
                natives,
                exception_handlers,
                value_store,
                heavy_store,
                native_args_buffer,
                reusable_native_arg_ids,
                reusable_all_popped,
                abi_natives,
                explicit_relations,
                explicit_primary_keys,
                globals,
                explicit_global_names,
                vm_ptr,
            );
        }
        _ => {
            return Err(LangError::runtime_error(
                "CallVariadic callee must be a function".to_string(),
                line,
            ));
        }
    }
    Ok(VMStatus::Continue)
}

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
    let (function_index, function) = match &callee_val {
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
    let mut obj_keys: Vec<String> = obj_map
        .str_key_pairs()
        .into_iter()
        .map(|(k, _)| k)
        .collect();
    obj_keys.sort();
    let param_set: std::collections::HashSet<&str> =
        param_names.iter().map(|s| s.as_str()).collect();
    let mut unknown: Vec<&str> = Vec::new();
    for k in &obj_keys {
        if !param_set.contains(k.as_str()) {
            unknown.push(k.as_str());
        }
    }
    if !unknown.is_empty() {
        unknown.sort();
        let expected: Vec<&str> = param_names.iter().map(|s| s.as_str()).collect();
        let error = ExceptionHandler::runtime_error(
            &frames,
            format!(
                "** call got unexpected keys [{}]; allowed parameter names are [{}]",
                unknown.join(", "),
                expected.join(", ")
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
        let v = obj_map.str_key_get(p.as_str()).cloned().unwrap_or(Value::Null);
        let id = store_value(v, value_store, heavy_store);
        arg_tvs.push(TaggedValue::from_heap(id));
    }
    let stack_start = crate::vm::stack::logical_len(stack);
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
    let param_start_index = function.captured_vars.len();
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
