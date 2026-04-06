//! Builtin native fast paths extracted from [`super::execute_native_call`] for readability and profiling.

use crate::common::{
    value::Value,
    value_store::{ValueCell, ValueId, ValueStore},
    TaggedValue,
};
use crate::vm::frame::CallFrame;
use crate::vm::native_indices::builtin;
use crate::vm::native_indices::TABLE_DATA_HEADERS_FAST_PATH_LEGACY;
use crate::vm::heavy_store::HeavyStore;
use crate::vm::stack;
use crate::vm::store_convert::{store_value, tagged_to_value_id};
use crate::vm::types::VMStatus;
use std::cell::RefCell;
use std::rc::Rc;

use crate::common::table::Table;

/// `range(...)` fast path. Returns `Some(Continue)` when handled; `None` to fall through to generic native dispatch.
pub(super) fn try_range_fast_path(
    native_index: usize,
    arity: usize,
    stack: &mut Vec<TaggedValue>,
    frames: &mut Vec<CallFrame>,
    value_store: &mut ValueStore,
) -> Option<VMStatus> {
    if native_index != builtin::RANGE || !(arity == 1 || arity == 2 || arity == 3) {
        return None;
    }
    let frame = frames.last().unwrap();
    let available = stack.len().saturating_sub(frame.stack_start);
    let need = if arity == 1 {
        1
    } else if arity == 2 {
        2
    } else {
        3
    };
    if available < need {
        return None;
    }
    let read_number = |store: &ValueStore, id: ValueId| -> Option<i64> {
        store.get(id).and_then(|c| match c {
            ValueCell::Number(n) => {
                let x = *n;
                if x.fract() == 0.0 && x >= i64::MIN as f64 && x <= i64::MAX as f64 {
                    Some(x as i64)
                } else {
                    None
                }
            }
            _ => None,
        })
    };
    let params = if arity == 1 {
        let n_tv = stack.pop().unwrap_or(TaggedValue::null());
        let n_id = tagged_to_value_id(n_tv, value_store);
        read_number(value_store, n_id)
            .map(|n| (0_i64, n.max(0), 1_i64))
            .map_or_else(
                || {
                    stack::push_id(stack, n_id);
                    None
                },
                |t| Some(t),
            )
    } else if arity == 2 {
        let end_tv = stack.pop().unwrap_or(TaggedValue::null());
        let start_tv = stack.pop().unwrap_or(TaggedValue::null());
        let end_id = tagged_to_value_id(end_tv, value_store);
        let start_id = tagged_to_value_id(start_tv, value_store);
        match (
            read_number(value_store, start_id),
            read_number(value_store, end_id),
        ) {
            (Some(start), Some(end)) => Some((start, end, 1_i64)),
            _ => {
                stack::push_id(stack, start_id);
                stack::push_id(stack, end_id);
                None
            }
        }
    } else {
        let step_tv = stack.pop().unwrap_or(TaggedValue::null());
        let end_tv = stack.pop().unwrap_or(TaggedValue::null());
        let start_tv = stack.pop().unwrap_or(TaggedValue::null());
        let step_id = tagged_to_value_id(step_tv, value_store);
        let end_id = tagged_to_value_id(end_tv, value_store);
        let start_id = tagged_to_value_id(start_tv, value_store);
        match (
            read_number(value_store, start_id),
            read_number(value_store, end_id),
            read_number(value_store, step_id),
        ) {
            (Some(start), Some(end), Some(step)) if step != 0 => Some((start, end, step)),
            _ => {
                stack::push_id(stack, start_id);
                stack::push_id(stack, end_id);
                stack::push_id(stack, step_id);
                None
            }
        }
    };
    let (start, end, step) = params?;
    let len = if step > 0 {
        if start >= end {
            0
        } else {
            ((end - start) as u64 / step as u64).min(usize::MAX as u64) as usize
        }
    } else if start <= end {
        0
    } else {
        ((start - end) as u64 / (-step) as u64).min(usize::MAX as u64) as usize
    };
    value_store.reserve_min(value_store.len() + 1);
    let mut slots = Vec::with_capacity(len);
    if step > 0 {
        let mut cur = start;
        while cur < end {
            slots.push(TaggedValue::from_f64(cur as f64));
            cur += step;
        }
    } else {
        let mut cur = start;
        while cur > end {
            slots.push(TaggedValue::from_f64(cur as f64));
            cur += step;
        }
    }
    let result_id = value_store.allocate_arena(ValueCell::Array(slots));
    stack::push_id(stack, result_id);
    Some(VMStatus::Continue)
}

/// `push(arr, item)` in-place array append.
pub(super) fn try_push_fast_path(
    native_index: usize,
    arity: usize,
    stack: &mut Vec<TaggedValue>,
    frames: &mut Vec<CallFrame>,
    value_store: &mut ValueStore,
) -> Option<VMStatus> {
    if native_index != builtin::PUSH || arity != 2 {
        return None;
    }
    let frame = frames.last().unwrap();
    let available = stack.len().saturating_sub(frame.stack_start);
    if available < 2 {
        return None;
    }
    let item_tv = stack.pop().unwrap_or(TaggedValue::null());
    let arr_tv = stack.pop().unwrap_or(TaggedValue::null());
    let arr_id = tagged_to_value_id(arr_tv, value_store);
    if let Some(ValueCell::Array(slots)) = value_store.get_mut(arr_id) {
        slots.push(item_tv);
        stack::push_id(stack, arr_id);
        return Some(VMStatus::Continue);
    }
    stack::push(stack, arr_tv);
    stack::push(stack, item_tv);
    None
}

/// `len(x)` for array/string cells.
pub(super) fn try_len_fast_path(
    native_index: usize,
    arity: usize,
    stack: &mut Vec<TaggedValue>,
    frames: &mut Vec<CallFrame>,
    value_store: &mut ValueStore,
) -> Option<VMStatus> {
    if native_index != builtin::LEN || arity != 1 {
        return None;
    }
    let frame = frames.last().unwrap();
    let available = stack.len().saturating_sub(frame.stack_start);
    if available < 1 {
        return None;
    }
    let arg_tv = stack.pop().unwrap_or(TaggedValue::null());
    let arg_id = tagged_to_value_id(arg_tv, value_store);
    if let Some(cell) = value_store.get(arg_id) {
        if let Some(len) = match cell {
            ValueCell::Array(ids) => Some(ids.len() as f64),
            ValueCell::String(sid) => value_store.get_string(*sid).map(|s| s.len() as f64),
            _ => None,
        } {
            let result_id = value_store.allocate(ValueCell::Number(len));
            stack::push_id(stack, result_id);
            return Some(VMStatus::Continue);
        }
    }
    stack::push_id(stack, arg_id);
    None
}

/// `int` / `float` / `str` / `typeof` unary fast paths (no generic invoke).
pub(super) fn try_cast_typeof_fast_path(
    native_index: usize,
    arity: usize,
    stack: &mut Vec<TaggedValue>,
    frames: &mut Vec<CallFrame>,
    value_store: &mut ValueStore,
    heavy_store: &mut HeavyStore,
) -> Option<VMStatus> {
    if arity != 1
        || !matches!(
            native_index,
            builtin::INT | builtin::FLOAT | builtin::STR | builtin::TYPEOF
        )
    {
        return None;
    }
    let frame = frames.last().unwrap();
    let available = stack.len().saturating_sub(frame.stack_start);
    if available < 1 {
        return None;
    }
    let arg_tv = stack.pop().unwrap_or(TaggedValue::null());
    let arg_id = tagged_to_value_id(arg_tv, value_store);
    let result_value = value_store.get(arg_id).and_then(|cell| {
        match (native_index, cell) {
            (idx, ValueCell::Number(n)) if idx == builtin::INT => Some(Value::Number(n.trunc())),
            (idx, ValueCell::Bool(b)) if idx == builtin::INT => {
                Some(Value::Number(if *b { 1.0 } else { 0.0 }))
            }
            (idx, ValueCell::Null) if idx == builtin::INT => Some(Value::Number(0.0)),
            (idx, ValueCell::Number(n)) if idx == builtin::FLOAT => Some(Value::Number(*n)),
            (idx, ValueCell::Bool(b)) if idx == builtin::FLOAT => {
                Some(Value::Number(if *b { 1.0 } else { 0.0 }))
            }
            (idx, ValueCell::Null) if idx == builtin::FLOAT => Some(Value::Number(0.0)),
            (idx, ValueCell::Number(n)) if idx == builtin::STR => Some(Value::String(n.to_string())),
            (idx, ValueCell::Bool(b)) if idx == builtin::STR => Some(Value::String(if *b {
                "true".to_string()
            } else {
                "false".to_string()
            })),
            (idx, ValueCell::String(sid)) if idx == builtin::STR => value_store
                .get_string(*sid)
                .map(|s| Value::String(s.to_string())),
            (idx, ValueCell::Null) if idx == builtin::STR => Some(Value::String("null".to_string())),
            (idx, ValueCell::Number(n)) if idx == builtin::TYPEOF => {
                Some(Value::String(if n.fract() == 0.0 {
                    "int".to_string()
                } else {
                    "float".to_string()
                }))
            }
            (idx, ValueCell::Bool(_)) if idx == builtin::TYPEOF => {
                Some(Value::String("bool".to_string()))
            }
            (idx, ValueCell::String(_)) if idx == builtin::TYPEOF => {
                Some(Value::String("string".to_string()))
            }
            (idx, ValueCell::Null) if idx == builtin::TYPEOF => {
                Some(Value::String("null".to_string()))
            }
            (idx, ValueCell::Array(_)) if idx == builtin::TYPEOF => {
                Some(Value::String("array".to_string()))
            }
            (idx, ValueCell::Tuple(_)) if idx == builtin::TYPEOF => {
                Some(Value::String("tuple".to_string()))
            }
            (idx, ValueCell::Path(_)) if idx == builtin::TYPEOF => {
                Some(Value::String("path".to_string()))
            }
            (idx, ValueCell::Function(_)) | (idx, ValueCell::NativeFunction(_))
                if idx == builtin::TYPEOF =>
            {
                Some(Value::String("function".to_string()))
            }
            _ => None,
        }
    });
    if let Some(v) = result_value {
        let result_id = store_value(v, value_store, heavy_store);
        stack::push_id(stack, result_id);
        Some(VMStatus::Continue)
    } else {
        stack::push_id(stack, arg_id);
        None
    }
}

/// Legacy `table(data, headers)` fast path (see `TABLE_DATA_HEADERS_FAST_PATH_LEGACY`).
pub(super) fn try_table_legacy_fast_path(
    native_index: usize,
    arity: usize,
    stack: &mut Vec<TaggedValue>,
    frames: &mut Vec<CallFrame>,
    value_store: &mut ValueStore,
    heavy_store: &mut HeavyStore,
) -> Option<VMStatus> {
    if native_index != TABLE_DATA_HEADERS_FAST_PATH_LEGACY || arity != 2 {
        return None;
    }
    let frame = frames.last().unwrap();
    let available = stack.len().saturating_sub(frame.stack_start);
    if available < 2 {
        return None;
    }
    let headers_tv = stack.pop().unwrap_or(TaggedValue::null());
    let data_tv = stack.pop().unwrap_or(TaggedValue::null());
    let headers_id = tagged_to_value_id(headers_tv, value_store);
    let data_id = tagged_to_value_id(data_tv, value_store);
    let row_slots_opt = value_store.get(data_id).and_then(|c| {
        if let ValueCell::Array(s) = c {
            Some(s.clone())
        } else {
            None
        }
    });
    let Some(row_slots) = row_slots_opt else {
        stack::push_id(stack, data_id);
        stack::push_id(stack, headers_id);
        return None;
    };
    let num_cols = row_slots
        .first()
        .and_then(|row_tv| {
            if row_tv.is_heap() {
                value_store.get(row_tv.get_heap_id()).and_then(|c| match c {
                    ValueCell::Array(s) => Some(s.len()),
                    _ => None,
                })
            } else {
                None
            }
        })
        .unwrap_or(0);
    let mut flat_cell_ids = Vec::with_capacity(row_slots.len() * num_cols.max(1));
    for row_tv in row_slots.iter() {
        if row_tv.is_heap() {
            let row_id = row_tv.get_heap_id();
            let cell_slots: Vec<TaggedValue> = value_store
                .get(row_id)
                .and_then(|c| {
                    if let ValueCell::Array(s) = c {
                        Some(s.clone())
                    } else {
                        None
                    }
                })
                .unwrap_or_default();
            for slot in cell_slots.iter() {
                flat_cell_ids.push(tagged_to_value_id(*slot, value_store));
            }
        }
    }
    let headers: Vec<String> = {
        let header_slots: Vec<TaggedValue> = value_store
            .get(headers_id)
            .and_then(|c| {
                if let ValueCell::Array(s) = c {
                    Some(s.clone())
                } else {
                    None
                }
            })
            .unwrap_or_default();
        let mut v = Vec::with_capacity(header_slots.len());
        for slot in header_slots.iter() {
            let val = crate::vm::store_convert::load_value(
                tagged_to_value_id(*slot, value_store),
                value_store,
                heavy_store,
            );
            v.push(match &val {
                Value::String(s) => s.clone(),
                _ => val.to_string(),
            });
        }
        if v.is_empty() {
            (0..num_cols).map(|i| format!("Column_{}", i)).collect()
        } else {
            v
        }
    };
    let table = Table::from_flat_view(flat_cell_ids, headers.len().max(1), headers);
    let table_val = Value::Table(Rc::new(RefCell::new(table)));
    let heavy_idx = heavy_store.push(table_val);
    let result_id = value_store.allocate(ValueCell::Heavy(heavy_idx));
    stack::push_id(stack, result_id);
    Some(VMStatus::Continue)
}
