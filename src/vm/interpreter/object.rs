// Object/array opcodes: MakeArray, MakeTuple, MakeObject, UnpackObject, MakeObjectDynamic, MakeArrayDynamic,
// GetArrayLength, TableFilter, GetArrayElement, SetArrayElement, Clone.
// Logic preserved 1:1 from executor.rs — no semantic changes.

use std::collections::HashMap;

use crate::common::{error::LangError, value::Value, value_store::{ValueCell, ValueId, ValueStore}};
use crate::vm::store_convert::tagged_to_value_id;
use crate::vm::vm::VM_CALL_CONTEXT;
use crate::vm::exceptions::ExceptionHandler;
use crate::vm::frame::CallFrame;
use crate::vm::heavy_store::HeavyStore;
use crate::vm::stack;
use crate::vm::store_convert::{load_value, store_value};
use crate::vm::types::VMStatus;

use super::helpers::pop_to_value_id;

pub fn op_make_array(
    count: usize,
    stack: &mut Vec<crate::common::TaggedValue>,
    frames: &mut Vec<CallFrame>,
    exception_handlers: &mut Vec<ExceptionHandler>,
    value_store: &mut ValueStore,
    heavy_store: &mut HeavyStore,
) -> Result<VMStatus, LangError> {
    let cap = if count == 0 { 16384 } else { count };
    let mut slots = Vec::with_capacity(cap);
    for _ in 0..count {
        slots.push(stack::pop(stack, frames, exception_handlers, value_store, heavy_store)?);
    }
    slots.reverse();
    let result_id = value_store.allocate_arena(ValueCell::Array(slots));
    stack::push_id(stack, result_id);
    Ok(VMStatus::Continue)
}

pub fn op_make_tuple(
    count: usize,
    stack: &mut Vec<crate::common::TaggedValue>,
    frames: &mut Vec<CallFrame>,
    exception_handlers: &mut Vec<ExceptionHandler>,
    value_store: &mut ValueStore,
    heavy_store: &mut HeavyStore,
) -> Result<VMStatus, LangError> {
    let mut element_ids = Vec::with_capacity(count);
    for _ in 0..count {
        element_ids.push(pop_to_value_id(stack, frames, exception_handlers, value_store, heavy_store)?);
    }
    element_ids.reverse();
    let result_id = value_store.allocate(ValueCell::Tuple(element_ids));
    stack::push_id(stack, result_id);
    Ok(VMStatus::Continue)
}

pub fn op_make_object(
    pair_count: usize,
    line: usize,
    stack: &mut Vec<crate::common::TaggedValue>,
    frames: &mut Vec<CallFrame>,
    exception_handlers: &mut Vec<ExceptionHandler>,
    value_store: &mut ValueStore,
    heavy_store: &mut HeavyStore,
) -> Result<VMStatus, LangError> {
    let mut map: HashMap<String, ValueId> = HashMap::with_capacity(pair_count);
    for _ in 0..pair_count {
        let value_id = pop_to_value_id(stack, frames, exception_handlers, value_store, heavy_store)?;
        let key_id = pop_to_value_id(stack, frames, exception_handlers, value_store, heavy_store)?;
        let key = match value_store.get(key_id) {
            Some(ValueCell::String(sid)) => value_store.get_string(*sid).map(|s| s.to_string()),
            _ => None,
        };
        let key = match key {
            Some(k) => k,
            None => {
                let key_value = load_value(key_id, value_store, heavy_store);
                match key_value {
                    Value::String(s) => s,
                    _ => {
                        return Err(ExceptionHandler::runtime_error(
                            &frames,
                            "Object key must be a string".to_string(),
                            line,
                        ));
                    }
                }
            }
        };
        map.insert(key, value_id);
    }
    let result_id = value_store.allocate(ValueCell::Object(map));
    stack::push_id(stack, result_id);
    Ok(VMStatus::Continue)
}

pub fn op_unpack_object(
    count_slot: usize,
    line: usize,
    stack: &mut Vec<crate::common::TaggedValue>,
    frames: &mut Vec<CallFrame>,
    exception_handlers: &mut Vec<ExceptionHandler>,
    value_store: &mut ValueStore,
    heavy_store: &mut HeavyStore,
) -> Result<VMStatus, LangError> {
    let obj_id = pop_to_value_id(stack, frames, exception_handlers, value_store, heavy_store)?;
    let pairs: Vec<(String, ValueId)> = match value_store.get(obj_id) {
        Some(ValueCell::Object(m)) => m.iter().map(|(k, v)| (k.clone(), *v)).collect(),
        _ => {
            let val = load_value(obj_id, value_store, heavy_store);
            if let Value::Object(rc) = &val {
                rc.borrow().iter().map(|(k, v)| (k.clone(), store_value(v.clone(), value_store, heavy_store))).collect()
            } else {
                return Err(ExceptionHandler::runtime_error(
                    &frames,
                    "** unpacking requires an object".to_string(),
                    line,
                ));
            }
        }
    };
    let n = pairs.len();
    for (k, v_id) in pairs {
        let sid = value_store.intern_string(k);
        let key_id = value_store.allocate(ValueCell::String(sid));
        stack::push_id(stack, key_id);
        stack::push_id(stack, v_id);
    }
    let frame = frames.last_mut().unwrap();
    if count_slot >= frame.slots.len() {
        frame.ensure_slot(count_slot);
    }
    let current = frame.slots[count_slot];
    let cur_f64 = if current.is_number() {
        current.get_f64()
    } else if current.is_int() {
        current.get_i32() as f64
    } else {
        0.0
    };
    frame.slots[count_slot] = crate::common::TaggedValue::from_f64(cur_f64 + n as f64);
    Ok(VMStatus::Continue)
}

pub fn op_make_object_dynamic(
    line: usize,
    stack: &mut Vec<crate::common::TaggedValue>,
    frames: &mut Vec<CallFrame>,
    exception_handlers: &mut Vec<ExceptionHandler>,
    value_store: &mut ValueStore,
    heavy_store: &mut HeavyStore,
) -> Result<VMStatus, LangError> {
    let count_id = pop_to_value_id(stack, frames, exception_handlers, value_store, heavy_store)?;
    let pair_count = match value_store.get(count_id) {
        Some(ValueCell::Number(n)) => {
            let idx = *n as i64;
            if idx < 0 {
                return Err(ExceptionHandler::runtime_error(
                    &frames,
                    "Object pair count must be non-negative".to_string(),
                    line,
                ));
            }
            idx as usize
        }
        _ => {
            let v = load_value(count_id, value_store, heavy_store);
            match v {
                Value::Number(n) => {
                    let idx = n as i64;
                    if idx < 0 {
                        return Err(ExceptionHandler::runtime_error(
                            &frames,
                            "Object pair count must be non-negative".to_string(),
                            line,
                        ));
                    }
                    idx as usize
                }
                _ => {
                    return Err(ExceptionHandler::runtime_error(
                        &frames,
                        "MakeObjectDynamic requires a number (pair count) on stack".to_string(),
                        line,
                    ));
                }
            }
        }
    };
    let mut map: HashMap<String, ValueId> = HashMap::with_capacity(pair_count);
    for _ in 0..pair_count {
        let value_id = pop_to_value_id(stack, frames, exception_handlers, value_store, heavy_store)?;
        let key_id = pop_to_value_id(stack, frames, exception_handlers, value_store, heavy_store)?;
        let key = match value_store.get(key_id) {
            Some(ValueCell::String(sid)) => value_store.get_string(*sid).map(|s| s.to_string()),
            _ => None,
        };
        let key = match key {
            Some(k) => k,
            None => {
                let key_value = load_value(key_id, value_store, heavy_store);
                match key_value {
                    Value::String(s) => s,
                    _ => {
                        return Err(ExceptionHandler::runtime_error(
                            &frames,
                            "Object key must be a string".to_string(),
                            line,
                        ));
                    }
                }
            }
        };
        map.insert(key, value_id);
    }
    let result_id = value_store.allocate(ValueCell::Object(map));
    stack::push_id(stack, result_id);
    Ok(VMStatus::Continue)
}

pub fn op_get_array_length(
    line: usize,
    stack: &mut Vec<crate::common::TaggedValue>,
    frames: &mut Vec<CallFrame>,
    exception_handlers: &mut Vec<ExceptionHandler>,
    value_store: &mut ValueStore,
    heavy_store: &mut HeavyStore,
) -> Result<VMStatus, LangError> {
    let array_id = pop_to_value_id(stack, frames, exception_handlers, value_store, heavy_store)?;
    if let Some(ValueCell::Array(ref arr)) = value_store.get(array_id) {
        let result_id = value_store.allocate(ValueCell::Number(arr.len() as f64));
        stack::push_id(stack, result_id);
        return Ok(VMStatus::Continue);
    }
    let array = load_value(array_id, value_store, heavy_store);
    match array {
        Value::Array(arr) => {
            stack::push_id(stack, store_value(Value::Number(arr.borrow().len() as f64), value_store, heavy_store));
        }
        Value::ColumnReference { table, column_name } => {
            let t = table.borrow();
            if let Some(len) = crate::vm::table_ops::column_len(&*t, &column_name) {
                stack::push_id(stack, store_value(Value::Number(len as f64), value_store, heavy_store));
            } else {
                let error = ExceptionHandler::runtime_error(
                    &frames,
                    format!("Column '{}' not found", column_name),
                    line,
                );
                match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error, value_store, heavy_store) {
                    Ok(()) => return Ok(VMStatus::Continue),
                    Err(e) => return Err(e),
                }
            }
        }
        Value::Dataset(dataset) => {
            let batch_size = dataset.borrow().batch_size();
            stack::push_id(stack, store_value(Value::Number(batch_size as f64), value_store, heavy_store));
        }
        Value::Enumerate { data, .. } => {
            stack::push_id(stack, store_value(Value::Number(data.borrow().len() as f64), value_store, heavy_store));
        }
        Value::Tuple(tuple) => {
            stack::push_id(stack, store_value(Value::Number(tuple.borrow().len() as f64), value_store, heavy_store));
        }
        _ => {
            let got_type = crate::vm::calls::get_type_name_value(&array);
            let error = ExceptionHandler::runtime_error(
                &frames,
                format!("Expected array, column reference, dataset, enumerate, or tuple for GetArrayLength, got {}", got_type),
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

pub fn op_table_filter(
    line: usize,
    stack: &mut Vec<crate::common::TaggedValue>,
    frames: &mut Vec<CallFrame>,
    exception_handlers: &mut Vec<ExceptionHandler>,
    value_store: &mut ValueStore,
    heavy_store: &mut HeavyStore,
    vm_ptr: *mut crate::vm::vm::Vm,
) -> Result<VMStatus, LangError> {
    let value_tv = stack::pop(stack, frames, exception_handlers, value_store, heavy_store)?;
    let op_tv = stack::pop(stack, frames, exception_handlers, value_store, heavy_store)?;
    let column_tv = stack::pop(stack, frames, exception_handlers, value_store, heavy_store)?;
    let table_tv = stack::pop(stack, frames, exception_handlers, value_store, heavy_store)?;
    let value_id = tagged_to_value_id(value_tv, value_store);
    let op_id = tagged_to_value_id(op_tv, value_store);
    let column_id = tagged_to_value_id(column_tv, value_store);
    let table_id = tagged_to_value_id(table_tv, value_store);
    let table_val = load_value(table_id, value_store, heavy_store);
    let column_val = load_value(column_id, value_store, heavy_store);
    let op_val = load_value(op_id, value_store, heavy_store);
    let filter_value = load_value(value_id, value_store, heavy_store);
    let column_str = match &column_val {
        Value::String(s) => s.as_str(),
        _ => {
            let error = ExceptionHandler::runtime_error(
                &frames,
                "Table filter column must be a string".to_string(),
                line,
            );
            match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error, value_store, heavy_store) {
                Ok(()) => return Ok(VMStatus::Continue),
                Err(e) => return Err(e),
            }
        }
    };
    let op_str = match &op_val {
        Value::String(s) => s.as_str(),
        _ => {
            let error = ExceptionHandler::runtime_error(
                &frames,
                "Table filter operator must be a string".to_string(),
                line,
            );
            match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error, value_store, heavy_store) {
                Ok(()) => return Ok(VMStatus::Continue),
                Err(e) => return Err(e),
            }
        }
    };
    if let Value::Table(table_rc) = &table_val {
        VM_CALL_CONTEXT.with(|ctx| {
            *ctx.borrow_mut() = Some(vm_ptr);
        });
        let result = crate::vm::natives::table::table_where_impl(
            table_rc,
            column_str,
            op_str,
            &filter_value,
        );
        VM_CALL_CONTEXT.with(|ctx| {
            *ctx.borrow_mut() = None;
        });
        stack::push_id(stack, store_value(result, value_store, heavy_store));
    } else {
        let got = match &table_val {
            Value::Array(_) => "Array",
            Value::Object(_) => "Object",
            Value::Null => "Null",
            Value::Number(_) => "Number",
            Value::String(_) => "String",
            Value::Bool(_) => "Bool",
            _ => "other type",
        };
        let error = ExceptionHandler::runtime_error(
            &frames,
            format!("Table filter requires a table, got {}", got),
            line,
        );
        match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error, value_store, heavy_store) {
            Ok(()) => {}
            Err(e) => return Err(e),
        }
    }
    Ok(VMStatus::Continue)
}

pub fn op_clone(
    stack: &mut Vec<crate::common::TaggedValue>,
    frames: &mut Vec<CallFrame>,
    exception_handlers: &mut Vec<ExceptionHandler>,
    value_store: &mut ValueStore,
    heavy_store: &mut HeavyStore,
) -> Result<VMStatus, LangError> {
    let value_id = pop_to_value_id(stack, frames, exception_handlers, value_store, heavy_store)?;
    let value = load_value(value_id, value_store, heavy_store);
    let cloned = value.clone();
    stack::push_id(stack, store_value(cloned, value_store, heavy_store));
    Ok(VMStatus::Continue)
}

pub fn op_make_array_dynamic(
    line: usize,
    stack: &mut Vec<crate::common::TaggedValue>,
    frames: &mut Vec<CallFrame>,
    exception_handlers: &mut Vec<ExceptionHandler>,
    value_store: &mut ValueStore,
    heavy_store: &mut HeavyStore,
) -> Result<VMStatus, LangError> {
    let count_id = pop_to_value_id(stack, frames, exception_handlers, value_store, heavy_store)?;
    let count: usize = match value_store.get(count_id) {
        Some(ValueCell::Number(n)) => {
            let idx = *n as i64;
            if idx < 0 {
                let error = ExceptionHandler::runtime_error(
                    &frames,
                    "Array size must be non-negative".to_string(),
                    line,
                );
                return match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error, value_store, heavy_store) {
                    Ok(()) => Ok(VMStatus::Continue),
                    Err(e) => Err(e),
                };
            }
            idx as usize
        }
        _ => {
            let count_value = load_value(count_id, value_store, heavy_store);
            match count_value {
                Value::Number(n) => {
                    let idx = n as i64;
                    if idx < 0 {
                        let error = ExceptionHandler::runtime_error(
                            &frames,
                            "Array size must be non-negative".to_string(),
                            line,
                        );
                        return match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error, value_store, heavy_store) {
                            Ok(()) => Ok(VMStatus::Continue),
                            Err(e) => Err(e),
                        };
                    }
                    idx as usize
                }
                _ => {
                    let error = ExceptionHandler::runtime_error(
                        &frames,
                        "Array size must be a number".to_string(),
                        line,
                    );
                    return match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error, value_store, heavy_store) {
                        Ok(()) => Ok(VMStatus::Continue),
                        Err(e) => Err(e),
                    };
                }
            }
        }
    };
    let mut slots = Vec::with_capacity(count);
    for _ in 0..count {
        slots.push(stack::pop(stack, frames, exception_handlers, value_store, heavy_store)?);
    }
    slots.reverse();
    let result_id = value_store.allocate_arena(ValueCell::Array(slots));
    stack::push_id(stack, result_id);
    Ok(VMStatus::Continue)
}
