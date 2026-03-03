//! Get/Set for Array, Tuple, Enumerate.

use std::cell::RefCell;
use std::rc::Rc;

use crate::common::{error::LangError, value::Value, value_store::{ValueCell, ValueStore}, TaggedValue};
use crate::common::error::ErrorType;
use crate::vm::exceptions::ExceptionHandler;
use crate::vm::frame::CallFrame;
use crate::vm::heavy_store::HeavyStore;
use crate::vm::stack;
use crate::vm::store_convert::store_value;
use crate::vm::types::VMStatus;

/// Get element from Array. Called when container is Value::Array.
#[allow(clippy::too_many_arguments)]
pub fn get_array(
    line: usize,
    stack: &mut Vec<TaggedValue>,
    frames: &mut Vec<CallFrame>,
    exception_handlers: &mut Vec<ExceptionHandler>,
    value_store: &mut ValueStore,
    heavy_store: &mut HeavyStore,
    container_id: crate::common::value_store::ValueId,
    arr: Rc<RefCell<Vec<Value>>>,
    index_value: Value,
) -> Result<VMStatus, LangError> {
    if let Value::String(key) = &index_value {
        let method_index = match key.as_str() {
            "push" => Some(35),
            "pop" => Some(36),
            "unique" => Some(37),
            "reverse" => Some(38),
            "sort" => Some(39),
            "sum" => Some(40),
            "average" => Some(41),
            "count" => Some(42),
            "any" => Some(43),
            "all" => Some(44),
            _ => None,
        };
        if let Some(idx) = method_index {
            stack::push_id(stack, store_value(Value::NativeFunction(idx), value_store, heavy_store));
            return Ok(VMStatus::Continue);
        }
        let error = ExceptionHandler::runtime_error(
            &frames,
            format!("Array has no property '{}'. Available: push, pop, unique, reverse, sort, sum, average, count, any, all, or use numeric index", key),
            line,
        );
        return match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error, value_store, heavy_store) {
            Ok(()) => Ok(VMStatus::Continue),
            Err(e) => Err(e),
        };
    }
    let index = match index_value {
        Value::Number(n) => {
            let idx = n as i64;
            if idx < 0 {
                let error = ExceptionHandler::runtime_error(&frames, "Array index must be non-negative".to_string(), line);
                match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error, value_store, heavy_store) {
                    Ok(()) => return Ok(VMStatus::Continue),
                    Err(e) => return Err(e),
                }
            }
            idx as usize
        }
        _ => {
            let error = ExceptionHandler::runtime_error(&frames, "Array index must be a number".to_string(), line);
            match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error, value_store, heavy_store) {
                Ok(()) => return Ok(VMStatus::Continue),
                Err(e) => return Err(e),
            }
        }
    };
    if let Some(ValueCell::Array(slots)) = value_store.get(container_id) {
        if index < slots.len() {
            stack::push(stack, slots[index]);
            return Ok(VMStatus::Continue);
        }
    }
    let arr_ref = arr.borrow();
    if index >= arr_ref.len() {
        let error = ExceptionHandler::runtime_error_with_type(
            &frames,
            format!("Array index {} out of bounds (length: {})", index, arr_ref.len()),
            line,
            ErrorType::IndexError,
        );
        return match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error, value_store, heavy_store) {
            Ok(()) => Ok(VMStatus::Continue),
            Err(e) => Err(e),
        };
    }
    let element = &arr_ref[index];
    let value = match element {
        Value::Array(arr_rc) => Value::Array(Rc::clone(arr_rc)),
        Value::Table(table_rc) => Value::Table(Rc::clone(table_rc)),
        Value::Axis(axis_rc) => Value::Axis(Rc::clone(axis_rc)),
        Value::Figure(fig_rc) => Value::Figure(Rc::clone(fig_rc)),
        Value::Image(img_rc) => Value::Image(Rc::clone(img_rc)),
        Value::Window(handle) => Value::Window(*handle),
        Value::Tensor(tensor_rc) => Value::Tensor(Rc::clone(tensor_rc)),
        Value::Object(obj_rc) => Value::Object(obj_rc.clone()),
        Value::DatabaseEngine(engine_rc) => Value::DatabaseEngine(Rc::clone(engine_rc)),
        Value::DatabaseCluster(cluster_rc) => Value::DatabaseCluster(Rc::clone(cluster_rc)),
        _ => element.clone(),
    };
    stack::push_id(stack, store_value(value, value_store, heavy_store));
    Ok(VMStatus::Continue)
}

/// Get element from Tuple.
#[allow(clippy::too_many_arguments)]
pub fn get_tuple(
    line: usize,
    stack: &mut Vec<TaggedValue>,
    frames: &mut Vec<CallFrame>,
    exception_handlers: &mut Vec<ExceptionHandler>,
    value_store: &mut ValueStore,
    heavy_store: &mut HeavyStore,
    tuple: Rc<RefCell<Vec<Value>>>,
    index_value: Value,
) -> Result<VMStatus, LangError> {
    let index = match index_value {
        Value::Number(n) => {
            let idx = n as i64;
            if idx < 0 {
                let error = ExceptionHandler::runtime_error(&frames, "Tuple index must be non-negative".to_string(), line);
                match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error, value_store, heavy_store) {
                    Ok(()) => return Ok(VMStatus::Continue),
                    Err(e) => return Err(e),
                }
            }
            idx as usize
        }
        _ => {
            let error = ExceptionHandler::runtime_error(&frames, "Tuple index must be a number".to_string(), line);
            match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error, value_store, heavy_store) {
                Ok(()) => return Ok(VMStatus::Continue),
                Err(e) => return Err(e),
            }
        }
    };
    let tuple_ref = tuple.borrow();
    if index >= tuple_ref.len() {
        let error = ExceptionHandler::runtime_error_with_type(
            &frames,
            format!("Tuple index {} out of bounds (length: {})", index, tuple_ref.len()),
            line,
            ErrorType::IndexError,
        );
        return match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error, value_store, heavy_store) {
            Ok(()) => Ok(VMStatus::Continue),
            Err(e) => Err(e),
        };
    }
    let element = &tuple_ref[index];
    let value = match element {
        Value::Array(arr_rc) => Value::Array(Rc::clone(arr_rc)),
        Value::Tuple(tuple_rc) => Value::Tuple(Rc::clone(tuple_rc)),
        Value::Table(table_rc) => Value::Table(Rc::clone(table_rc)),
        Value::Object(_) => element.clone(),
        _ => element.clone(),
    };
    stack::push_id(stack, store_value(value, value_store, heavy_store));
    Ok(VMStatus::Continue)
}

/// Get element from Enumerate.
#[allow(clippy::too_many_arguments)]
pub fn get_enumerate(
    line: usize,
    stack: &mut Vec<TaggedValue>,
    frames: &mut Vec<CallFrame>,
    exception_handlers: &mut Vec<ExceptionHandler>,
    value_store: &mut ValueStore,
    heavy_store: &mut HeavyStore,
    data: Rc<RefCell<Vec<Value>>>,
    start: i64,
    index_value: Value,
) -> Result<VMStatus, LangError> {
    let index = match index_value {
        Value::Number(n) => {
            let idx = n as i64;
            if idx < 0 {
                let error = ExceptionHandler::runtime_error(&frames, "Enumerate index must be non-negative".to_string(), line);
                match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error, value_store, heavy_store) {
                    Ok(()) => return Ok(VMStatus::Continue),
                    Err(e) => return Err(e),
                }
            }
            idx as usize
        }
        _ => {
            let error = ExceptionHandler::runtime_error(&frames, "Enumerate index must be a number".to_string(), line);
            match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error, value_store, heavy_store) {
                Ok(()) => return Ok(VMStatus::Continue),
                Err(e) => return Err(e),
            }
        }
    };
    let data_ref = data.borrow();
    if index >= data_ref.len() {
        let error = ExceptionHandler::runtime_error_with_type(
            &frames,
            format!("Enumerate index {} out of bounds (length: {})", index, data_ref.len()),
            line,
            ErrorType::IndexError,
        );
        return match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error, value_store, heavy_store) {
            Ok(()) => Ok(VMStatus::Continue),
            Err(e) => Err(e),
        };
    }
    let element = &data_ref[index];
    let value = match element {
        Value::Array(arr_rc) => Value::Array(Rc::clone(arr_rc)),
        Value::Table(table_rc) => Value::Table(Rc::clone(table_rc)),
        Value::Axis(axis_rc) => Value::Axis(Rc::clone(axis_rc)),
        Value::Figure(fig_rc) => Value::Figure(Rc::clone(fig_rc)),
        Value::Image(img_rc) => Value::Image(Rc::clone(img_rc)),
        Value::Window(handle) => Value::Window(*handle),
        Value::Tensor(tensor_rc) => Value::Tensor(Rc::clone(tensor_rc)),
        Value::Object(obj_rc) => Value::Object(obj_rc.clone()),
        Value::DatabaseEngine(engine_rc) => Value::DatabaseEngine(Rc::clone(engine_rc)),
        Value::DatabaseCluster(cluster_rc) => Value::DatabaseCluster(Rc::clone(cluster_rc)),
        _ => element.clone(),
    };
    let pair = Value::Tuple(Rc::new(RefCell::new(vec![
        Value::Number((start + index as i64) as f64),
        value,
    ])));
    stack::push_id(stack, store_value(pair, value_store, heavy_store));
    Ok(VMStatus::Continue)
}

/// Set element in Array. Called when container is Value::Array.
#[allow(clippy::too_many_arguments)]
pub fn set_array(
    line: usize,
    stack: &mut Vec<TaggedValue>,
    frames: &mut Vec<CallFrame>,
    exception_handlers: &mut Vec<ExceptionHandler>,
    value_store: &mut ValueStore,
    heavy_store: &mut HeavyStore,
    container_id: crate::common::value_store::ValueId,
    index_value: Value,
    value: Value,
) -> Result<VMStatus, LangError> {
    let index = match index_value {
        Value::Number(n) => {
            let idx = n as i64;
            if idx < 0 {
                let error = ExceptionHandler::runtime_error(&frames, "Array index must be non-negative".to_string(), line);
                match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error, value_store, heavy_store) {
                    Ok(()) => return Ok(VMStatus::Continue),
                    Err(e) => return Err(e),
                }
            }
            idx as usize
        }
        _ => {
            let error = ExceptionHandler::runtime_error(&frames, "Array index must be a number".to_string(), line);
            match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error, value_store, heavy_store) {
                Ok(()) => return Ok(VMStatus::Continue),
                Err(e) => return Err(e),
            }
        }
    };
    let slot_tv = match &value {
        Value::Number(n) => TaggedValue::from_f64(*n),
        Value::Bool(b) => TaggedValue::from_bool(*b),
        Value::Null => TaggedValue::null(),
        _ => TaggedValue::from_heap(store_value(value.clone(), value_store, heavy_store)),
    };
    if let Some(ValueCell::Array(slots)) = value_store.get_mut(container_id) {
        if index >= slots.len() {
            slots.resize(index + 1, TaggedValue::null());
        }
        slots[index] = slot_tv;
    }
    stack::push_id(stack, container_id);
    Ok(VMStatus::Continue)
}
