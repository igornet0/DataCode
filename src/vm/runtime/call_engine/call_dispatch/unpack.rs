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

fn is_class_object(rc: &std::rc::Rc<std::cell::RefCell<crate::common::value::ObjectKind>>) -> bool {
    let obj = rc.borrow();
    obj.str_key_get("__class_name").is_some() && !obj.str_key_contains("__class")
}

fn is_module_namespace(v: &Value) -> bool {
    matches!(v, Value::Object(rc)
        if rc.borrow().str_key_contains(crate::vm::module_object::MODULE_MARKER_KEY))
}

/// Constructors of a class in binding order: fixed-arity overloads (fewest parameters first),
/// then `*args` / `**kwargs` overloads (most regular parameters first).
fn constructor_candidates(
    class_obj: &crate::common::value::ObjectKind,
    class_name: &str,
    functions: &[crate::bytecode::Function],
    vm_ptr: *mut crate::vm::vm::Vm,
) -> Vec<usize> {
    let mut found: Vec<usize> = class_obj
        .str_key_pairs()
        .iter()
        .filter(|(k, _)| k.starts_with("new_"))
        .filter_map(|(_, v)| match v {
            Value::Function(i) if *i < functions.len() => Some(*i),
            Value::ModuleFunction {
                module_uid,
                local_index,
            } => unsafe { (*vm_ptr).get_module_function_index(*module_uid, *local_index) },
            _ => None,
        })
        .collect();
    if found.is_empty() && !class_name.is_empty() {
        let prefix = format!("{}::new_", class_name);
        found = functions
            .iter()
            .enumerate()
            .filter(|(_, f)| f.name.starts_with(&prefix))
            .map(|(i, _)| i)
            .collect();
    }
    found.sort_by_key(|&i| {
        let f = &functions[i];
        let variadic = crate::vm::variadic_bind::function_accepts_variadic(f);
        let fixed = crate::vm::variadic_bind::fixed_param_count(f);
        (variadic, if variadic { usize::MAX - fixed } else { fixed })
    });
    found.dedup();
    found
}

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
    // Argument-binding errors go through the exception handler (catchable by try/catch),
    // like the arity errors of a plain Call.
    macro_rules! raise {
        ($msg:expr) => {{
            let error = ExceptionHandler::runtime_error(&frames, $msg, line);
            return match ExceptionHandler::handle_exception(
                stack,
                frames,
                exception_handlers,
                error,
                value_store,
                heavy_store,
            ) {
                Ok(()) => Ok(VMStatus::Continue),
                Err(e) => Err(e),
            };
        }};
    }
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
    // `**a, **b` are popped last-first; restore call-site order.
    starstar_vals.reverse();
    // Named arguments in call-site order (needed when a native without known parameter names
    // takes them positionally, see below).
    let mut named_order: Vec<(String, Value)> = Vec::with_capacity(n_named);
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
            raise!("named argument key must be a string".to_string());
        };
        if named.contains_key(&key) {
            raise!(format!("got multiple values for argument '{}'", key));
        }
        named.insert(key.clone(), val.clone());
        named_order.push((key, val));
    }
    named_order.reverse();
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
    // `f(*a, *b)` must bind `a` before `b`: they were popped last-first.
    star_vals.reverse();
    let mut positional: Vec<Value> = pos_rev.into_iter().rev().collect();

    let user_function_index = match &callee_val {
        Value::Function(i) if *i < functions.len() => Some(*i),
        Value::ModuleFunction {
            module_uid,
            local_index,
        } => unsafe { (*vm_ptr).get_module_function_index(*module_uid, *local_index) }
            .filter(|i| *i < functions.len()),
        _ => None,
    };

    match &callee_val {
        _ if user_function_index.is_some() => {
            let function_index = user_function_index.unwrap();
            let function = functions[function_index].clone();
            // Same receiver rules as a plain Call (`m.f(...)` drops the module namespace,
            // `@class` methods get their class injected).
            let mut receiver_tvs = vec![TaggedValue::null(); positional.len()];
            super::super::method_call::prepare_method_args(
                function_index,
                &function,
                &mut positional,
                &mut receiver_tvs,
                value_store,
                heavy_store,
                vm_ptr,
            );
            let bound = match bind_function_args(
                &function,
                positional,
                named,
                &star_vals,
                &starstar_vals,
            ) {
                Ok(bound) => bound,
                Err(msg) => raise!(msg),
            };
            if let Some(msg) = crate::vm::calls::param_type_error(
                &function,
                &bound,
                globals,
                _global_names,
                value_store,
                heavy_store,
            ) {
                let error = crate::common::error::LangError::runtime_error_with_type(
                    msg,
                    line,
                    crate::common::error::ErrorType::TypeError,
                );
                return match ExceptionHandler::handle_exception(
                    stack,
                    frames,
                    exception_handlers,
                    error,
                    value_store,
                    heavy_store,
                ) {
                    Ok(()) => Ok(VMStatus::Continue),
                    Err(e) => Err(e),
                };
            }
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
                if param_names.is_empty()
                    && varkw.is_none()
                    && !named_order.is_empty()
                    && star_vals.is_empty()
                    && starstar_vals.is_empty()
                {
                    // Parameter names unknown (plugin natives): pass named values positionally
                    // in call-site order, as a plain Call did before runtime binding.
                    positional
                        .into_iter()
                        .chain(named_order.into_iter().map(|(_, v)| v))
                        .collect()
                } else {
                    match bind_native_varkw_args(
                        &param_names,
                        varkw,
                        positional,
                        named,
                        &star_vals,
                        &starstar_vals,
                    ) {
                        Ok(bound) => bound,
                        Err(msg) => raise!(msg),
                    }
                }
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
        Value::Object(class_rc) if is_class_object(class_rc) => {
            // `Class(a, *xs, k = v, **opts)` / `m.Class(...)`: pick the constructor overload that
            // binds these arguments, bind them here, then run it like a plain constructor Call.
            if positional.first().is_some_and(is_module_namespace) {
                positional.remove(0);
            }
            for arr in &star_vals {
                let Value::Array(rc) = arr else {
                    raise!("* unpacking requires an array".to_string());
                };
                positional.extend(rc.borrow().iter().cloned());
            }
            for obj in &starstar_vals {
                let Value::Object(rc) = obj else {
                    raise!("** unpacking requires an object with string keys".to_string());
                };
                for (k, v) in rc.borrow().str_key_entries_cloned() {
                    if named.contains_key(&k) {
                        raise!(format!("got multiple values for argument '{}'", k));
                    }
                    named.insert(k, v);
                }
            }
            let class_name = class_rc
                .borrow()
                .str_key_get("__class_name")
                .map(|v| v.to_string())
                .unwrap_or_default();
            let mut last_error = None;
            let mut chosen = None;
            for idx in constructor_candidates(&class_rc.borrow(), &class_name, functions, vm_ptr) {
                match bind_function_args(&functions[idx], positional.clone(), named.clone(), &[], &[]) {
                    Ok(bound) => {
                        chosen = Some((idx, bound));
                        break;
                    }
                    Err(e) => last_error = Some(e),
                }
            }
            let Some((function_index, bound)) = chosen else {
                raise!(last_error.unwrap_or_else(|| format!(
                    "Class '{}' has no constructor for these arguments",
                    class_name
                )));
            };
            let arity = bound.len();
            for v in bound {
                let id = store_value(v, value_store, heavy_store);
                stack::push(stack, TaggedValue::from_heap(id));
            }
            unsafe { (*vm_ptr).prebound_call_args = true };
            let current_ip = frames.last().map(|f| f.ip.saturating_sub(1)).unwrap_or(0);
            return super::super::closure_call::execute_closure_call(
                current_ip,
                function_index,
                Some(callee_val.clone()),
                arity,
                line,
                stack,
                frames,
                globals,
                _global_names,
                functions,
                exception_handlers,
                error_type_table,
                value_store,
                heavy_store,
                vm_ptr,
            );
        }
        _ => {
            raise!("CallVariadic callee must be a function".to_string());
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
