//! Opcodes for `for x in` over lazy [`Value::Iterable`].

use crate::common::error::LangError;
use crate::common::value::Value;
use crate::common::value_store::ValueStore;
use crate::common::TaggedValue;
use crate::vm::exceptions::ExceptionHandler;
use crate::vm::frame::CallFrame;
use crate::vm::heavy_store::HeavyStore;
use crate::vm::iterable::{iterable_next, prepare_for_in_iterable};
use crate::vm::stack;
use crate::vm::store_convert::{load_value, store_value, tagged_to_value_id};
use crate::vm::types::VMStatus;
use crate::vm::vm::VM_CALL_CONTEXT;

/// Replace local `iter_local` with `prepare_for_in_iterable` result.
pub fn op_coerce_for_in_iterable(
    iter_local: usize,
    line: usize,
    stack: &mut Vec<TaggedValue>,
    frames: &mut Vec<CallFrame>,
    exception_handlers: &mut Vec<ExceptionHandler>,
    value_store: &mut ValueStore,
    heavy_store: &mut HeavyStore,
) -> Result<VMStatus, LangError> {
    let frame = frames.last_mut().unwrap();
    if iter_local >= frame.slots.len() {
        frame.ensure_slot(iter_local);
    }
    let tv = frame.slots[iter_local];
    let id = tagged_to_value_id(tv, value_store);
    let v = load_value(id, value_store, heavy_store);
    let coerced = match prepare_for_in_iterable(v) {
        Ok(x) => x,
        Err(e) => {
            let msg = format!("{}", e);
            let error = ExceptionHandler::runtime_error(frames, msg, line);
            return match ExceptionHandler::handle_exception(
                stack,
                frames,
                exception_handlers,
                error,
                value_store,
                heavy_store,
            ) {
                Ok(()) => Ok(VMStatus::Continue),
                Err(err) => Err(err),
            };
        }
    };
    let new_id = store_value(coerced, value_store, heavy_store);
    frame.slots[iter_local] = TaggedValue::from_heap(new_id);
    Ok(VMStatus::Continue)
}

/// Next element from iterable in `iter_local`: push `value, true` or `false` if exhausted.
pub fn op_for_iterable_next(
    iter_local: usize,
    line: usize,
    stack: &mut Vec<TaggedValue>,
    frames: &mut Vec<CallFrame>,
    exception_handlers: &mut Vec<ExceptionHandler>,
    value_store: &mut ValueStore,
    heavy_store: &mut HeavyStore,
) -> Result<VMStatus, LangError> {
    let vm_ptr = VM_CALL_CONTEXT.with(|ctx| *ctx.borrow()).ok_or_else(|| {
        LangError::runtime_error(
            "ForIterableNext: VM context not available".to_string(),
            line,
        )
    })?;
    let frame = frames.last_mut().unwrap();
    if iter_local >= frame.slots.len() {
        frame.ensure_slot(iter_local);
    }
    let tv = frame.slots[iter_local];
    let id = tagged_to_value_id(tv, value_store);
    let v = load_value(id, value_store, heavy_store);
    let Value::Iterable(rc) = v else {
        let error = ExceptionHandler::runtime_error(
            frames,
            format!(
                "internal: ForIterableNext expected iterable after coerce, got {}",
                crate::vm::calls::get_type_name_value(&v)
            ),
            line,
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
    };
    unsafe {
        let vm = &mut *vm_ptr;
        let mut inner = rc.borrow_mut();
        match iterable_next(&mut *inner, vm) {
            Ok(None) => {
                stack::push(stack, TaggedValue::from_bool(false));
            }
            Ok(Some(el)) => {
                let vid = store_value(el, value_store, heavy_store);
                stack::push(stack, TaggedValue::from_heap(vid));
                stack::push(stack, TaggedValue::from_bool(true));
            }
            Err(e) => {
                let error = ExceptionHandler::runtime_error(frames, e.to_string(), line);
                return match ExceptionHandler::handle_exception(
                    stack,
                    frames,
                    exception_handlers,
                    error,
                    value_store,
                    heavy_store,
                ) {
                    Ok(()) => Ok(VMStatus::Continue),
                    Err(err) => Err(err),
                };
            }
        }
    }
    Ok(VMStatus::Continue)
}
