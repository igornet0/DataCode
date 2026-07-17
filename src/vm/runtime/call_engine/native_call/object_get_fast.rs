//! Early `dict.get(key [, default])` — stack operands, no `native_args_buffer` / per-arg `load_value`.

use crate::common::error::ErrorType;
use crate::common::numeric::tagged_integral_canonical_if_whole;
use crate::common::value::Value;
use crate::common::value_store::{ValueCell, ValueId, ValueStore};
use crate::common::TaggedValue;
use crate::vm::exceptions::ExceptionHandler;
use crate::vm::frame::CallFrame;
use crate::vm::heavy_store::HeavyStore;
use crate::vm::interpreter::element::object_map_needs_visibility_checks;
use crate::vm::memory::{
    canonical_integral_from_key_id, load_value, object_cell_try_lookup_by_key_id,
    push_integral_slot, push_stack_value_id, store_value,
};
use crate::vm::native_indices::builtin;
use crate::vm::natives::utils::call_user_function;
use crate::vm::stack;
use crate::vm::store_convert::tagged_to_value_id;
use crate::vm::types::VMStatus;

use crate::common::error::LangError;

fn push_default(
    stack: &mut Vec<TaggedValue>,
    value_store: &mut ValueStore,
    default_tv: TaggedValue,
) {
    if default_tv.is_number()
        || default_tv.is_int()
        || default_tv.is_bool()
        || default_tv.is_null()
    {
        stack::push(stack, default_tv);
    } else if default_tv.is_heap() {
        push_stack_value_id(stack, value_store, default_tv.get_heap_id());
    } else {
        stack::push(stack, default_tv);
    }
}

fn class_instance_get_type_error(
    line: usize,
    stack: &mut Vec<TaggedValue>,
    frames: &mut Vec<CallFrame>,
    exception_handlers: &mut Vec<ExceptionHandler>,
    value_store: &mut ValueStore,
    heavy_store: &mut HeavyStore,
    message: &str,
) -> Result<VMStatus, LangError> {
    let error = ExceptionHandler::runtime_error_with_type(
        frames,
        message.to_string(),
        line,
        ErrorType::TypeError,
    );
    ExceptionHandler::handle_exception_vm(
        stack,
        frames,
        exception_handlers,
        error,
        value_store,
        heavy_store,
    )
}

/// Class instance with `fn get`: invoke bound method instead of plain-dict key lookup.
/// Returns `Ok(true)` when handled (result pushed), `Ok(false)` when not a class instance.
fn try_invoke_class_instance_get(
    line: usize,
    stack: &mut Vec<TaggedValue>,
    frames: &mut Vec<CallFrame>,
    exception_handlers: &mut Vec<ExceptionHandler>,
    value_store: &mut ValueStore,
    heavy_store: &mut HeavyStore,
    obj_id: ValueId,
    key_id: ValueId,
) -> Result<bool, LangError> {
    let Some(ValueCell::Object(omap)) = value_store.get(obj_id) else {
        return Ok(false);
    };
    if !object_map_needs_visibility_checks(omap, value_store, heavy_store) {
        return Ok(false);
    }

    let get_name_id = store_value(Value::String("get".to_string()), value_store, heavy_store);
    let Some(method_id) =
        object_cell_try_lookup_by_key_id(obj_id, get_name_id, value_store, heavy_store)
    else {
        return class_instance_get_type_error(
            line,
            stack,
            frames,
            exception_handlers,
            value_store,
            heavy_store,
            "TypeError: class instance has no .get() method",
        )
        .map(|_| true);
    };

    let method = load_value(method_id, value_store, heavy_store);
    let instance = load_value(obj_id, value_store, heavy_store);
    let key = load_value(key_id, value_store, heavy_store);

    let result = match method {
        Value::Function(fn_idx) => call_user_function(fn_idx, &[instance, key]),
        _ => {
            return class_instance_get_type_error(
                line,
                stack,
                frames,
                exception_handlers,
                value_store,
                heavy_store,
                "TypeError: class .get() is not a user function",
            )
            .map(|_| true);
        }
    };

    match result {
        Ok(value) => {
            let out_id = store_value(value, value_store, heavy_store);
            push_stack_value_id(stack, value_store, out_id);
            Ok(true)
        }
        Err(e) => {
            ExceptionHandler::handle_exception_vm(
                stack,
                frames,
                exception_handlers,
                e,
                value_store,
                heavy_store,
            )?;
            Ok(true)
        }
    }
}

/// Stack `[obj, key, default]` → lookup result (integral fast path + generic fallback).
pub(crate) fn object_get_from_stack(
    line: usize,
    stack: &mut Vec<TaggedValue>,
    frames: &mut Vec<CallFrame>,
    exception_handlers: &mut Vec<ExceptionHandler>,
    value_store: &mut ValueStore,
    heavy_store: &mut HeavyStore,
) -> Result<VMStatus, LangError> {
    let default_tv = stack::pop(stack, frames, exception_handlers, value_store, heavy_store)?;
    let key_tv = stack::pop(stack, frames, exception_handlers, value_store, heavy_store)?;
    let obj_tv = stack::pop(stack, frames, exception_handlers, value_store, heavy_store)?;

    if !obj_tv.is_heap() {
        let error = ExceptionHandler::runtime_error_with_type(
            frames,
            "TypeError: .get() expects a plain dict object".to_string(),
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
    let key_id = tagged_to_value_id(key_tv, value_store);

    if try_invoke_class_instance_get(
        line,
        stack,
        frames,
        exception_handlers,
        value_store,
        heavy_store,
        obj_id,
        key_id,
    )? {
        return Ok(VMStatus::Continue);
    }

    let key_canonical = tagged_integral_canonical_if_whole(key_tv).or_else(|| {
        if key_tv.is_heap() {
            canonical_integral_from_key_id(key_tv.get_heap_id(), value_store)
        } else {
            None
        }
    });

    if let Some(canonical) = key_canonical {
        match value_store.get_mut(obj_id) {
            Some(ValueCell::Object(omap)) => {
                if let Some(slot) = omap.find_integral_slot(canonical) {
                    push_integral_slot(stack, value_store, slot);
                } else {
                    push_default(stack, value_store, default_tv);
                }
                return Ok(VMStatus::Continue);
            }
            _ => {
                let error = ExceptionHandler::runtime_error_with_type(
                    frames,
                    "TypeError: .get() expects a plain dict object".to_string(),
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
        }
    }

    if canonical_integral_from_key_id(key_id, value_store).is_none() {
        let key_ok = value_store
            .get(key_id)
            .map(crate::common::type_model::value_cell_is_hashable_key)
            .unwrap_or(false);
        if !key_ok {
            let key_material = load_value(key_id, value_store, heavy_store);
            if !crate::common::type_model::is_hashable_value(&key_material) {
                let tn = crate::vm::calls::get_type_name_value(&key_material);
                let error = ExceptionHandler::runtime_error_with_type(
                    frames,
                    format!("unhashable type: {}", tn),
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
        }
    }

    let out_id = if matches!(value_store.get(obj_id), Some(ValueCell::Object(_))) {
        match object_cell_try_lookup_by_key_id(obj_id, key_id, value_store, heavy_store) {
            Some(vid) => vid,
            None => {
                push_default(stack, value_store, default_tv);
                return Ok(VMStatus::Continue);
            }
        }
    } else {
        let error = ExceptionHandler::runtime_error_with_type(
            frames,
            "TypeError: .get() expects a plain dict object".to_string(),
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
    };

    push_stack_value_id(stack, value_store, out_id);
    Ok(VMStatus::Continue)
}

fn restore_stack_args(
    stack: &mut Vec<TaggedValue>,
    obj_tv: TaggedValue,
    key_tv: TaggedValue,
    default_tv: Option<TaggedValue>,
) {
    stack::push(stack, obj_tv);
    stack::push(stack, key_tv);
    if let Some(d) = default_tv {
        stack::push(stack, d);
    }
}

/// Run before native arg materialization in [`super::execute_native_call`].
pub(super) fn try_object_get_early(
    native_index: usize,
    arity: usize,
    line: usize,
    stack: &mut Vec<TaggedValue>,
    frames: &mut Vec<CallFrame>,
    exception_handlers: &mut Vec<ExceptionHandler>,
    value_store: &mut ValueStore,
    heavy_store: &mut HeavyStore,
) -> Option<Result<VMStatus, LangError>> {
    if native_index != builtin::OBJECT_GET || !(arity == 2 || arity == 3) {
        return None;
    }
    let frame = frames.last()?;
    let available = crate::vm::stack::available_in_frame(stack, frame.stack_start);
    if available < arity {
        return None;
    }

    let default_tv = if arity == 3 {
        Some(crate::vm::stack::pop_direct(stack).unwrap_or(TaggedValue::null()))
    } else {
        None
    };
    let key_tv = crate::vm::stack::pop_direct(stack).unwrap_or(TaggedValue::null());
    let obj_tv = crate::vm::stack::pop_direct(stack).unwrap_or(TaggedValue::null());

    if !obj_tv.is_heap() {
        restore_stack_args(stack, obj_tv, key_tv, default_tv);
        return None;
    }

    let default_for_stack = default_tv.unwrap_or(TaggedValue::null());
    stack::push(stack, obj_tv);
    stack::push(stack, key_tv);
    stack::push(stack, default_for_stack);
    Some(object_get_from_stack(
        line,
        stack,
        frames,
        exception_handlers,
        value_store,
        heavy_store,
    ))
}
