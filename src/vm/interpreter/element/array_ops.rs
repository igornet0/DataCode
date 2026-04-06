//! Get/Set for Array, Tuple, Enumerate.

use std::cell::RefCell;
use std::rc::Rc;

use crate::common::array_slice::{contiguous_positive_slice_bounds, slice_indices};
use crate::common::error::ErrorType;
use crate::common::value::{ArrayViewData, ArrayViewSource, ByteBuffer, Value};
use crate::common::{
    error::LangError,
    value_store::{ValueCell, ValueStore},
    TaggedValue,
};
use crate::vm::array_view::{
    origin_to_view_data, push_array_view_value, resolve_slice_origin, validate_view_physical,
    view_get_element, SliceOrigin,
};
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
            "chunk" => Some(80),
            _ => None,
        };
        if let Some(idx) = method_index {
            stack::push_id(
                stack,
                store_value(Value::NativeFunction(idx), value_store, heavy_store),
            );
            return Ok(VMStatus::Continue);
        }
        let error = ExceptionHandler::runtime_error(
            &frames,
            format!("Array has no property '{}'. Available: push, pop, unique, reverse, sort, sum, average, count, any, all, chunk, or use numeric index", key),
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
    }
    let len = if let Some(ValueCell::Array(slots)) = value_store.get(container_id) {
        slots.len()
    } else {
        arr.borrow().len()
    };
    let index = match index_value {
        Value::Number(n) => {
            if n.fract() != 0.0 && (n - n.round()).abs() > 1e-9 {
                let error = ExceptionHandler::runtime_error(
                    &frames,
                    "Array index must be an integer".to_string(),
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
            let mut idx = n as i64;
            if idx < 0 {
                idx += len as i64;
            }
            if idx < 0 || (idx as usize) >= len {
                let error = ExceptionHandler::runtime_error_with_type(
                    &frames,
                    format!("Array index {} out of bounds (length: {})", n as i64, len),
                    line,
                    ErrorType::IndexError,
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
            idx as usize
        }
        _ => {
            let error = ExceptionHandler::runtime_error(
                &frames,
                "Array index must be a number".to_string(),
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
            format!(
                "Array index {} out of bounds (length: {})",
                index,
                arr_ref.len()
            ),
            line,
            ErrorType::IndexError,
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
    let element = &arr_ref[index];
    let value = match element {
        Value::Array(arr_rc) => Value::Array(Rc::clone(arr_rc)),
        Value::Table(table_rc) => Value::Table(Rc::clone(table_rc)),
        Value::Axis(axis_rc) => Value::Axis(Rc::clone(axis_rc)),
        Value::Figure(fig_rc) => Value::Figure(Rc::clone(fig_rc)),
        Value::Image(img_rc) => Value::Image(Rc::clone(img_rc)),
        Value::Window(handle) => Value::Window(*handle),
        Value::PluginOpaque { tag, id } => Value::PluginOpaque { tag: *tag, id: *id },
        Value::Object(obj_rc) => Value::Object(obj_rc.clone()),
        Value::DatabaseEngine(engine_rc) => Value::DatabaseEngine(Rc::clone(engine_rc)),
        Value::DatabaseCluster(cluster_rc) => Value::DatabaseCluster(Rc::clone(cluster_rc)),
        _ => element.clone(),
    };
    stack::push_id(stack, store_value(value, value_store, heavy_store));
    Ok(VMStatus::Continue)
}

/// Get element or method from [`Value::ArrayView`].
#[allow(clippy::too_many_arguments)]
pub fn get_array_view(
    line: usize,
    stack: &mut Vec<TaggedValue>,
    frames: &mut Vec<CallFrame>,
    exception_handlers: &mut Vec<ExceptionHandler>,
    value_store: &mut ValueStore,
    heavy_store: &mut HeavyStore,
    _container_id: crate::common::value_store::ValueId,
    av: &ArrayViewData,
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
            "chunk" => Some(80),
            _ => None,
        };
        if let Some(idx) = method_index {
            stack::push_id(
                stack,
                store_value(Value::NativeFunction(idx), value_store, heavy_store),
            );
            return Ok(VMStatus::Continue);
        }
        let error = ExceptionHandler::runtime_error(
            &frames,
            format!("Array has no property '{}'. Available: push, pop, unique, reverse, sort, sum, average, count, any, all, chunk, or use numeric index", key),
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
    }
    let len = av.length;
    let index = match index_value {
        Value::Number(n) => {
            if n.fract() != 0.0 && (n - n.round()).abs() > 1e-9 {
                let error = ExceptionHandler::runtime_error(
                    &frames,
                    "Array index must be an integer".to_string(),
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
            let mut idx = n as i64;
            if idx < 0 {
                idx += len as i64;
            }
            if idx < 0 || (idx as usize) >= len {
                let error = ExceptionHandler::runtime_error_with_type(
                    &frames,
                    format!("Array index {} out of bounds (length: {})", n as i64, len),
                    line,
                    ErrorType::IndexError,
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
            idx as usize
        }
        _ => {
            let error = ExceptionHandler::runtime_error(
                &frames,
                "Array index must be a number".to_string(),
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
    let element = match view_get_element(av, index, value_store, heavy_store) {
        Some(e) => e,
        None => {
            let error = ExceptionHandler::runtime_error_with_type(
                &frames,
                format!("Array index {} out of bounds (length: {})", index, len),
                line,
                ErrorType::IndexError,
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
    };
    let value = match &element {
        Value::Array(arr_rc) => Value::Array(Rc::clone(arr_rc)),
        Value::Table(table_rc) => Value::Table(Rc::clone(table_rc)),
        Value::Axis(axis_rc) => Value::Axis(Rc::clone(axis_rc)),
        Value::Figure(fig_rc) => Value::Figure(Rc::clone(fig_rc)),
        Value::Image(img_rc) => Value::Image(Rc::clone(img_rc)),
        Value::Window(handle) => Value::Window(*handle),
        Value::PluginOpaque { tag, id } => Value::PluginOpaque { tag: *tag, id: *id },
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
                let error = ExceptionHandler::runtime_error(
                    &frames,
                    "Tuple index must be non-negative".to_string(),
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
            idx as usize
        }
        _ => {
            let error = ExceptionHandler::runtime_error(
                &frames,
                "Tuple index must be a number".to_string(),
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
    let tuple_ref = tuple.borrow();
    if index >= tuple_ref.len() {
        let error = ExceptionHandler::runtime_error_with_type(
            &frames,
            format!(
                "Tuple index {} out of bounds (length: {})",
                index,
                tuple_ref.len()
            ),
            line,
            ErrorType::IndexError,
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
                let error = ExceptionHandler::runtime_error(
                    &frames,
                    "Enumerate index must be non-negative".to_string(),
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
            idx as usize
        }
        _ => {
            let error = ExceptionHandler::runtime_error(
                &frames,
                "Enumerate index must be a number".to_string(),
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
    let data_ref = data.borrow();
    if index >= data_ref.len() {
        let error = ExceptionHandler::runtime_error_with_type(
            &frames,
            format!(
                "Enumerate index {} out of bounds (length: {})",
                index,
                data_ref.len()
            ),
            line,
            ErrorType::IndexError,
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
    let element = &data_ref[index];
    let value = match element {
        Value::Array(arr_rc) => Value::Array(Rc::clone(arr_rc)),
        Value::Table(table_rc) => Value::Table(Rc::clone(table_rc)),
        Value::Axis(axis_rc) => Value::Axis(Rc::clone(axis_rc)),
        Value::Figure(fig_rc) => Value::Figure(Rc::clone(fig_rc)),
        Value::Image(img_rc) => Value::Image(Rc::clone(img_rc)),
        Value::Window(handle) => Value::Window(*handle),
        Value::PluginOpaque { tag, id } => Value::PluginOpaque { tag: *tag, id: *id },
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
    let len = if let Some(ValueCell::Array(slots)) = value_store.get(container_id) {
        slots.len()
    } else {
        0
    };
    let index = match index_value {
        Value::Number(n) => {
            if n.fract() != 0.0 && (n - n.round()).abs() > 1e-9 {
                let error = ExceptionHandler::runtime_error(
                    &frames,
                    "Array index must be an integer".to_string(),
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
            let mut idx = n as i64;
            if idx < 0 {
                idx += len as i64;
            }
            if idx < 0 {
                let error = ExceptionHandler::runtime_error_with_type(
                    &frames,
                    format!("Array index {} out of bounds (length: {})", n as i64, len),
                    line,
                    ErrorType::IndexError,
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
            idx as usize
        }
        _ => {
            let error = ExceptionHandler::runtime_error(
                &frames,
                "Array index must be a number".to_string(),
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

fn value_to_slot_tv(
    value: &Value,
    value_store: &mut ValueStore,
    heavy_store: &mut HeavyStore,
) -> TaggedValue {
    match value {
        Value::Number(n) => TaggedValue::from_f64(*n),
        Value::Bool(b) => TaggedValue::from_bool(*b),
        Value::Null => TaggedValue::null(),
        _ => TaggedValue::from_heap(store_value(value.clone(), value_store, heavy_store)),
    }
}

/// `step == 1` contiguous slice of [`ByteBuffer`] (zero-copy on shared `Rc<Vec<u8>>`).
pub fn byte_buffer_slice_value(
    bb: &ByteBuffer,
    start: Option<i64>,
    stop: Option<i64>,
    step: Option<i64>,
) -> Result<Value, String> {
    let st = step.unwrap_or(1);
    if st != 1 {
        return Err("ByteBuffer slice requires step 1".to_string());
    }
    let (a, b) = contiguous_positive_slice_bounds(bb.len, start, stop);
    bb.slice_range(a, b)
        .map(Value::ByteBuffer)
        .ok_or_else(|| "invalid byte slice".to_string())
}

/// Срез массива или представления: при `step == 1` — [`Value::ArrayView`] (zero-copy); иначе — копия в новый [`Value::Array`].
#[allow(clippy::too_many_arguments)]
pub fn get_array_slice_from_container(
    line: usize,
    stack: &mut Vec<TaggedValue>,
    frames: &mut Vec<CallFrame>,
    exception_handlers: &mut Vec<ExceptionHandler>,
    value_store: &mut ValueStore,
    heavy_store: &mut HeavyStore,
    container_id: crate::common::value_store::ValueId,
    start: Option<i64>,
    stop: Option<i64>,
    step: Option<i64>,
) -> Result<VMStatus, LangError> {
    let Some(origin) = resolve_slice_origin(container_id, value_store, heavy_store) else {
        let error = ExceptionHandler::runtime_error(
            frames,
            "GetArraySlice requires an array or array view".to_string(),
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
    let st = step.unwrap_or(1);
    let logical_len = origin.len();

    if st == 1 {
        let (a, b) = contiguous_positive_slice_bounds(logical_len, start, stop);
        let new_len = b.saturating_sub(a);
        let av = match &origin {
            SliceOrigin::Store {
                base_id,
                offset,
                length: _,
            } => ArrayViewData {
                source: ArrayViewSource::Store { base_id: *base_id },
                offset: offset + a,
                length: new_len,
            },
            SliceOrigin::Heap {
                vec,
                offset,
                length: _,
            } => ArrayViewData {
                source: ArrayViewSource::Heap(vec.clone()),
                offset: offset + a,
                length: new_len,
            },
        };
        if let Err(msg) = validate_view_physical(&av, value_store) {
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
                Err(e) => Err(e),
            };
        }
        push_array_view_value(av, stack, value_store, heavy_store);
        return Ok(VMStatus::Continue);
    }

    let indices = match slice_indices(logical_len, start, stop, st) {
        Ok(ix) => ix,
        Err(msg) => {
            let error = ExceptionHandler::runtime_error(&frames, msg.to_string(), line);
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
    };
    let base = origin_to_view_data(&origin);
    let mut out: Vec<Value> = Vec::with_capacity(indices.len());
    for &i in &indices {
        let v = view_get_element(&base, i, value_store, heavy_store).unwrap_or(Value::Null);
        out.push(v);
    }
    let arr = Value::Array(Rc::new(RefCell::new(out)));
    stack::push_id(stack, store_value(arr, value_store, heavy_store));
    Ok(VMStatus::Continue)
}

/// Присваивание срезу: только step 1 (или по умолчанию). `rhs` — массив вставляемых элементов.
#[allow(clippy::too_many_arguments)]
pub fn set_array_slice_splice(
    line: usize,
    stack: &mut Vec<TaggedValue>,
    frames: &mut Vec<CallFrame>,
    exception_handlers: &mut Vec<ExceptionHandler>,
    value_store: &mut ValueStore,
    heavy_store: &mut HeavyStore,
    container_id: crate::common::value_store::ValueId,
    start: Option<i64>,
    stop: Option<i64>,
    step: Option<i64>,
    rhs: Value,
) -> Result<VMStatus, LangError> {
    let st = step.unwrap_or(1);
    if st != 1 {
        let error = ExceptionHandler::runtime_error(
            &frames,
            "Slice assignment with step other than 1 is not supported".to_string(),
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
    }
    let Value::Array(rhs_rc) = rhs else {
        let error = ExceptionHandler::runtime_error(
            &frames,
            "Slice assignment requires an array on the right-hand side".to_string(),
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
    let insert: Vec<TaggedValue> = rhs_rc
        .borrow()
        .iter()
        .map(|v| value_to_slot_tv(v, value_store, heavy_store))
        .collect();

    let Some(ValueCell::Array(slots)) = value_store.get_mut(container_id) else {
        let error = ExceptionHandler::runtime_error(
            &frames,
            "SetArraySlice expects array storage".to_string(),
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
    let (a, b) = contiguous_positive_slice_bounds(slots.len(), start, stop);
    slots.splice(a..b, insert);
    stack::push_id(stack, container_id);
    Ok(VMStatus::Continue)
}
