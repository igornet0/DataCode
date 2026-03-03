// Comparison opcodes: Equal, NotEqual, Greater, Less, GreaterEqual, LessEqual, In.
// Logic preserved 1:1 from executor.rs — no semantic changes.

use crate::common::{error::LangError, value::Value, value_store::ValueStore, TaggedValue};
use crate::vm::exceptions::ExceptionHandler;
use crate::vm::frame::CallFrame;
use crate::vm::heavy_store::HeavyStore;
use crate::vm::operations;
use crate::vm::stack;
use crate::vm::store_convert::{load_value, store_value, tagged_to_value_id};
use crate::vm::types::VMStatus;

use super::helpers::pop_to_value_id;

pub fn op_equal(
    stack: &mut Vec<TaggedValue>,
    frames: &mut Vec<CallFrame>,
    exception_handlers: &mut Vec<ExceptionHandler>,
    value_store: &mut ValueStore,
    heavy_store: &mut HeavyStore,
) -> Result<VMStatus, LangError> {
    let b_tv = stack::pop(stack, frames, exception_handlers, value_store, heavy_store)?;
    let a_tv = stack::pop(stack, frames, exception_handlers, value_store, heavy_store)?;
    if a_tv.is_number() && b_tv.is_number() {
        stack::push(stack, TaggedValue::from_bool(a_tv.get_f64() == b_tv.get_f64()));
        return Ok(VMStatus::Continue);
    }
    if a_tv.is_bool() && b_tv.is_bool() {
        stack::push(stack, TaggedValue::from_bool(a_tv.get_bool() == b_tv.get_bool()));
        return Ok(VMStatus::Continue);
    }
    if a_tv.is_null() && b_tv.is_null() {
        stack::push(stack, TaggedValue::from_bool(true));
        return Ok(VMStatus::Continue);
    }
    let a_id = tagged_to_value_id(a_tv, value_store);
    let b_id = tagged_to_value_id(b_tv, value_store);
    let a = load_value(a_id, value_store, heavy_store);
    let b = load_value(b_id, value_store, heavy_store);
    let result = match (&a, &b) {
        (Value::Number(n1), Value::Number(n2)) => Value::Bool(n1 == n2),
        _ => operations::binary_equal(&a, &b),
    };
    stack::push_id(stack, store_value(result, value_store, heavy_store));
    Ok(VMStatus::Continue)
}

pub fn op_not_equal(
    stack: &mut Vec<TaggedValue>,
    frames: &mut Vec<CallFrame>,
    exception_handlers: &mut Vec<ExceptionHandler>,
    value_store: &mut ValueStore,
    heavy_store: &mut HeavyStore,
) -> Result<VMStatus, LangError> {
    let b_tv = stack::pop(stack, frames, exception_handlers, value_store, heavy_store)?;
    let a_tv = stack::pop(stack, frames, exception_handlers, value_store, heavy_store)?;
    if a_tv.is_number() && b_tv.is_number() {
        stack::push(stack, TaggedValue::from_bool(a_tv.get_f64() != b_tv.get_f64()));
        return Ok(VMStatus::Continue);
    }
    if a_tv.is_bool() && b_tv.is_bool() {
        stack::push(stack, TaggedValue::from_bool(a_tv.get_bool() != b_tv.get_bool()));
        return Ok(VMStatus::Continue);
    }
    if a_tv.is_null() && b_tv.is_null() {
        stack::push(stack, TaggedValue::from_bool(false));
        return Ok(VMStatus::Continue);
    }
    let a_id = tagged_to_value_id(a_tv, value_store);
    let b_id = tagged_to_value_id(b_tv, value_store);
    let a = load_value(a_id, value_store, heavy_store);
    let b = load_value(b_id, value_store, heavy_store);
    let result = match (&a, &b) {
        (Value::Number(n1), Value::Number(n2)) => Value::Bool(n1 != n2),
        _ => operations::binary_not_equal(&a, &b),
    };
    stack::push_id(stack, store_value(result, value_store, heavy_store));
    Ok(VMStatus::Continue)
}

pub fn op_greater(
    stack: &mut Vec<TaggedValue>,
    frames: &mut Vec<CallFrame>,
    exception_handlers: &mut Vec<ExceptionHandler>,
    value_store: &mut ValueStore,
    heavy_store: &mut HeavyStore,
) -> Result<VMStatus, LangError> {
    let b_tv = stack::pop(stack, frames, exception_handlers, value_store, heavy_store)?;
    let a_tv = stack::pop(stack, frames, exception_handlers, value_store, heavy_store)?;
    if a_tv.is_number() && b_tv.is_number() {
        stack::push(stack, TaggedValue::from_bool(a_tv.get_f64() > b_tv.get_f64()));
        return Ok(VMStatus::Continue);
    }
    let a_id = tagged_to_value_id(a_tv, value_store);
    let b_id = tagged_to_value_id(b_tv, value_store);
    let a = load_value(a_id, value_store, heavy_store);
    let b = load_value(b_id, value_store, heavy_store);
    let result = match (&a, &b) {
        (Value::Number(n1), Value::Number(n2)) => Value::Bool(n1 > n2),
        _ => operations::binary_greater(&a, &b, frames, stack, exception_handlers, value_store, heavy_store)?,
    };
    stack::push_id(stack, store_value(result, value_store, heavy_store));
    Ok(VMStatus::Continue)
}

pub fn op_less(
    stack: &mut Vec<TaggedValue>,
    frames: &mut Vec<CallFrame>,
    exception_handlers: &mut Vec<ExceptionHandler>,
    value_store: &mut ValueStore,
    heavy_store: &mut HeavyStore,
) -> Result<VMStatus, LangError> {
    let b_tv = stack::pop(stack, frames, exception_handlers, value_store, heavy_store)?;
    let a_tv = stack::pop(stack, frames, exception_handlers, value_store, heavy_store)?;
    if a_tv.is_number() && b_tv.is_number() {
        stack::push(stack, TaggedValue::from_bool(a_tv.get_f64() < b_tv.get_f64()));
        return Ok(VMStatus::Continue);
    }
    let a_id = tagged_to_value_id(a_tv, value_store);
    let b_id = tagged_to_value_id(b_tv, value_store);
    let a = load_value(a_id, value_store, heavy_store);
    let b = load_value(b_id, value_store, heavy_store);
    let result = match (&a, &b) {
        (Value::Number(n1), Value::Number(n2)) => Value::Bool(n1 < n2),
        _ => operations::binary_less(&a, &b, frames, stack, exception_handlers, value_store, heavy_store)?,
    };
    stack::push_id(stack, store_value(result, value_store, heavy_store));
    Ok(VMStatus::Continue)
}

pub fn op_greater_equal(
    stack: &mut Vec<TaggedValue>,
    frames: &mut Vec<CallFrame>,
    exception_handlers: &mut Vec<ExceptionHandler>,
    value_store: &mut ValueStore,
    heavy_store: &mut HeavyStore,
) -> Result<VMStatus, LangError> {
    let b_tv = stack::pop(stack, frames, exception_handlers, value_store, heavy_store)?;
    let a_tv = stack::pop(stack, frames, exception_handlers, value_store, heavy_store)?;
    if a_tv.is_number() && b_tv.is_number() {
        stack::push(stack, TaggedValue::from_bool(a_tv.get_f64() >= b_tv.get_f64()));
        return Ok(VMStatus::Continue);
    }
    let a_id = tagged_to_value_id(a_tv, value_store);
    let b_id = tagged_to_value_id(b_tv, value_store);
    let a = load_value(a_id, value_store, heavy_store);
    let b = load_value(b_id, value_store, heavy_store);
    let result = match (&a, &b) {
        (Value::Number(n1), Value::Number(n2)) => Value::Bool(n1 >= n2),
        _ => operations::binary_greater_equal(&a, &b, frames, stack, exception_handlers, value_store, heavy_store)?,
    };
    stack::push_id(stack, store_value(result, value_store, heavy_store));
    Ok(VMStatus::Continue)
}

pub fn op_less_equal(
    stack: &mut Vec<TaggedValue>,
    frames: &mut Vec<CallFrame>,
    exception_handlers: &mut Vec<ExceptionHandler>,
    value_store: &mut ValueStore,
    heavy_store: &mut HeavyStore,
) -> Result<VMStatus, LangError> {
    let b_tv = stack::pop(stack, frames, exception_handlers, value_store, heavy_store)?;
    let a_tv = stack::pop(stack, frames, exception_handlers, value_store, heavy_store)?;
    if a_tv.is_number() && b_tv.is_number() {
        stack::push(stack, TaggedValue::from_bool(a_tv.get_f64() <= b_tv.get_f64()));
        return Ok(VMStatus::Continue);
    }
    let a_id = tagged_to_value_id(a_tv, value_store);
    let b_id = tagged_to_value_id(b_tv, value_store);
    let a = load_value(a_id, value_store, heavy_store);
    let b = load_value(b_id, value_store, heavy_store);
    let result = match (&a, &b) {
        (Value::Number(n1), Value::Number(n2)) => Value::Bool(n1 <= n2),
        _ => operations::binary_less_equal(&a, &b, frames, stack, exception_handlers, value_store, heavy_store)?,
    };
    stack::push_id(stack, store_value(result, value_store, heavy_store));
    Ok(VMStatus::Continue)
}

pub fn op_in(
    line: usize,
    stack: &mut Vec<TaggedValue>,
    frames: &mut Vec<CallFrame>,
    exception_handlers: &mut Vec<ExceptionHandler>,
    value_store: &mut ValueStore,
    heavy_store: &mut HeavyStore,
) -> Result<VMStatus, LangError> {
    let array_id = pop_to_value_id(stack, frames, exception_handlers, value_store, heavy_store)?;
    let value_id = pop_to_value_id(stack, frames, exception_handlers, value_store, heavy_store)?;
    let array = load_value(array_id, value_store, heavy_store);
    let value = load_value(value_id, value_store, heavy_store);
    match array {
        Value::Array(arr) => {
            let arr_ref = arr.borrow();
            let found = arr_ref.iter().any(|item| item == &value);
            stack::push_id(stack, store_value(Value::Bool(found), value_store, heavy_store));
        }
        _ => {
            let error = ExceptionHandler::runtime_error(
                &frames,
                "Right operand of 'in' operator must be an array".to_string(),
                line,
            );
            match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error, value_store, heavy_store) {
                Ok(()) => return Ok(VMStatus::Continue),
                Err(e) => return Err(e),
            }
        }
    }
    Ok(VMStatus::Continue)
}
