//! Early `set.add` / `set.discard` on [`ValueCell::Set`] — mutate in place; never `load_value` the whole set.

use crate::common::error::{ErrorType, LangError};
use crate::common::numeric::tagged_integral_canonical_if_whole;
use crate::common::value_store::{ValueCell, ValueStore, NULL_VALUE_ID};
use crate::common::TaggedValue;
use crate::vm::exceptions::ExceptionHandler;
use crate::vm::frame::CallFrame;
use crate::vm::heavy_store::HeavyStore;
use crate::vm::native_indices::builtin;
use crate::vm::set_ops::{set_discard_material, set_insert_material};
use crate::vm::stack;
use crate::vm::store_convert::{load_value, tagged_to_value_id};
use crate::vm::types::VMStatus;

enum SetOp {
    Add,
    Discard,
}

pub(crate) enum SetIntegralMut {
    Add,
    Discard,
}

/// Hash-based set mutation when the key is not a whole number (tuple/string members).
fn set_mut_material_from_stack(
    op: SetIntegralMut,
    line: usize,
    stack: &mut Vec<TaggedValue>,
    frames: &mut Vec<CallFrame>,
    exception_handlers: &mut Vec<ExceptionHandler>,
    value_store: &mut ValueStore,
    heavy_store: &mut HeavyStore,
) -> Result<VMStatus, LangError> {
    let item_tv = stack::pop(stack, frames, exception_handlers, value_store, heavy_store)?;
    let set_tv = stack::pop(stack, frames, exception_handlers, value_store, heavy_store)?;
    if !set_tv.is_heap() {
        let error = ExceptionHandler::runtime_error_with_type(
            frames,
            "TypeError: set mutation expects a set object".to_string(),
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
    let set_id = set_tv.get_heap_id();
    let item_id = tagged_to_value_id(item_tv, value_store);
    let item = load_value(item_id, value_store, heavy_store);
    let mut smap = match value_store.get_mut(set_id) {
        Some(ValueCell::Set(m)) => std::mem::take(m),
        _ => {
            let error = ExceptionHandler::runtime_error_with_type(
                frames,
                "TypeError: set mutation expects a set object".to_string(),
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
    };
    match op {
        SetIntegralMut::Add => {
            if let Err(msg) = set_insert_material(&mut smap, &item, value_store, heavy_store) {
                if let Some(ValueCell::Set(slot)) = value_store.get_mut(set_id) {
                    *slot = smap;
                }
                let error = ExceptionHandler::runtime_error_with_type(
                    frames,
                    msg,
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
        SetIntegralMut::Discard => {
            set_discard_material(&mut smap, &item, value_store, heavy_store);
        }
    }
    if let Some(ValueCell::Set(slot)) = value_store.get_mut(set_id) {
        *slot = smap;
    }
    stack::push_id(stack, set_id);
    Ok(VMStatus::Continue)
}

/// Stack `[set, item]` → mutate plain set in place, push set id (integral key fast path).
pub(crate) fn set_mut_integral_from_stack(
    op: SetIntegralMut,
    line: usize,
    stack: &mut Vec<TaggedValue>,
    frames: &mut Vec<CallFrame>,
    exception_handlers: &mut Vec<crate::vm::exceptions::ExceptionHandler>,
    value_store: &mut ValueStore,
    heavy_store: &mut crate::vm::heavy_store::HeavyStore,
) -> Result<VMStatus, LangError> {
    let item_tv = crate::vm::stack::pop(stack, frames, exception_handlers, value_store, heavy_store)?;
    let set_tv = crate::vm::stack::pop(stack, frames, exception_handlers, value_store, heavy_store)?;

    if !set_tv.is_heap() {
        let error = ExceptionHandler::runtime_error_with_type(
            frames,
            "TypeError: set mutation expects a set object".to_string(),
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
    let set_id = set_tv.get_heap_id();
    let Some((canonical, key_id)) = integral_key_for_set_op(item_tv, value_store) else {
        // Non-integral key (tuple, string, …): fall back to material hash path.
        stack::push(stack, set_tv);
        stack::push(stack, item_tv);
        return set_mut_material_from_stack(
            op,
            line,
            stack,
            frames,
            exception_handlers,
            value_store,
            heavy_store,
        );
    };

    match value_store.get_mut(set_id) {
        Some(ValueCell::Set(smap)) => match op {
            SetIntegralMut::Add => {
                smap.insert_integral_only(canonical, key_id);
            }
            SetIntegralMut::Discard => {
                smap.discard_integral(canonical);
            }
        },
        _ => {
            let error = ExceptionHandler::runtime_error_with_type(
                frames,
                "TypeError: set mutation expects a set object".to_string(),
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
    stack::push_id(stack, set_id);
    Ok(VMStatus::Continue)
}

fn restore_stack_args(stack: &mut Vec<TaggedValue>, set_tv: TaggedValue, item_tv: TaggedValue) {
    stack::push(stack, set_tv);
    stack::push(stack, item_tv);
}

fn integral_key_for_set_op(
    item_tv: TaggedValue,
    store: &mut ValueStore,
) -> Option<(i64, crate::common::value_store::ValueId)> {
    let canonical = tagged_integral_canonical_if_whole(item_tv).or_else(|| {
        if item_tv.is_heap() {
            crate::vm::memory::canonical_integral_from_key_id_mut(item_tv.get_heap_id(), store)
        } else {
            None
        }
    })?;
    let _ = store;
    Some((canonical, NULL_VALUE_ID))
}

/// Run before native arg materialization in [`super::execute_native_call`].
pub(super) fn try_set_integral_mut_early(
    native_index: usize,
    arity: usize,
    _line: usize,
    stack: &mut Vec<TaggedValue>,
    frames: &mut Vec<CallFrame>,
    _exception_handlers: &mut Vec<crate::vm::exceptions::ExceptionHandler>,
    value_store: &mut ValueStore,
    _heavy_store: &mut crate::vm::heavy_store::HeavyStore,
) -> Option<Result<VMStatus, LangError>> {
    let op = match native_index {
        i if i == builtin::SET_ADD && arity == 2 => SetOp::Add,
        i if i == builtin::SET_DISCARD && arity == 2 => SetOp::Discard,
        _ => return None,
    };
    let frame = frames.last()?;
    if crate::vm::stack::available_in_frame(stack, frame.stack_start) < 2 {
        return None;
    }

    let item_tv = crate::vm::stack::pop_direct(stack).unwrap_or(TaggedValue::null());
    let set_tv = crate::vm::stack::pop_direct(stack).unwrap_or(TaggedValue::null());
    if !set_tv.is_heap() {
        restore_stack_args(stack, set_tv, item_tv);
        return None;
    }
    let set_id = set_tv.get_heap_id();
    if !matches!(value_store.get_mut(set_id), Some(ValueCell::Set(_))) {
        restore_stack_args(stack, set_tv, item_tv);
        return None;
    }

    let Some((canonical, key_id)) = integral_key_for_set_op(item_tv, value_store) else {
        restore_stack_args(stack, set_tv, item_tv);
        return None;
    };

    match op {
        SetOp::Add => {
            value_store.plain_set_insert_integral(set_id, canonical, key_id);
        }
        SetOp::Discard => {
            value_store.plain_set_discard_integral(set_id, canonical);
        }
    }
    stack::push_id(stack, set_id);
    Some(Ok(VMStatus::Continue))
}
