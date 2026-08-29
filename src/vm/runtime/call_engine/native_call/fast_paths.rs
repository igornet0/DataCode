//! Builtin native fast paths extracted from [`super::execute_native_call`] for readability and profiling.

use crate::common::{
    numeric::{coerce_to_int_value, float_is_int_surface, int_value_from_f64_lossy, number_is_int_surface, FloatValue, IntValue},
    value::Value,
    value_store::{ValueCell, ValueId, ValueStore},
    TaggedValue,
};
use crate::common::error::LangError;
use crate::vm::exceptions::ExceptionHandler;
use crate::vm::frame::CallFrame;
use crate::vm::native_indices::builtin;
use crate::vm::native_indices::TABLE_DATA_HEADERS_FAST_PATH_LEGACY;
use crate::vm::heavy_store::HeavyStore;
use crate::vm::stack;
use crate::vm::special_methods::{class_has_special, dispatch_special};
use crate::vm::store_convert::{load_value, store_value, tagged_to_value_id};
use crate::vm::types::VMStatus;
use std::cell::RefCell;
use std::rc::Rc;

use crate::common::table::Table;

/// `range(...)` fast path — lazy [`IterableInner::Range`], no materialization.
pub(super) fn try_range_fast_path(
    native_index: usize,
    arity: usize,
    line: usize,
    stack: &mut Vec<TaggedValue>,
    frames: &mut Vec<CallFrame>,
    exception_handlers: &mut Vec<ExceptionHandler>,
    value_store: &mut ValueStore,
    heavy_store: &mut HeavyStore,
) -> Result<Option<VMStatus>, LangError> {
    if native_index != builtin::RANGE || !(arity == 1 || arity == 2 || arity == 3) {
        return Ok(None);
    }
    let frame = frames.last().unwrap();
    let available = crate::vm::stack::available_in_frame(stack, frame.stack_start);
    let need = if arity == 1 { 1 } else if arity == 2 { 2 } else { 3 };
    if available < need {
        return Ok(None);
    }
    let read_integral = |store: &ValueStore, id: ValueId| -> Option<i64> {
        store.get(id).and_then(crate::common::numeric::integer_cell_as_i64_if_whole)
    };
    let pop_restore = |stack: &mut Vec<TaggedValue>, ids: &[ValueId]| {
        for id in ids.iter().rev() {
            stack::push_id(stack, *id);
        }
    };
    let args: Vec<Value> = if arity == 1 {
        let n_tv = crate::vm::stack::pop_direct(stack).unwrap_or(TaggedValue::null());
        let n_id = tagged_to_value_id(n_tv, value_store);
        let Some(end) = read_integral(value_store, n_id) else {
            pop_restore(stack, &[n_id]);
            return Ok(None);
        };
        vec![Value::Number(end as f64)]
    } else if arity == 2 {
        let end_tv = crate::vm::stack::pop_direct(stack).unwrap_or(TaggedValue::null());
        let start_tv = crate::vm::stack::pop_direct(stack).unwrap_or(TaggedValue::null());
        let end_id = tagged_to_value_id(end_tv, value_store);
        let start_id = tagged_to_value_id(start_tv, value_store);
        let (Some(start), Some(end)) = (
            read_integral(value_store, start_id),
            read_integral(value_store, end_id),
        ) else {
            pop_restore(stack, &[start_id, end_id]);
            return Ok(None);
        };
        vec![Value::Number(start as f64), Value::Number(end as f64)]
    } else {
        let step_tv = crate::vm::stack::pop_direct(stack).unwrap_or(TaggedValue::null());
        let end_tv = crate::vm::stack::pop_direct(stack).unwrap_or(TaggedValue::null());
        let start_tv = crate::vm::stack::pop_direct(stack).unwrap_or(TaggedValue::null());
        let step_id = tagged_to_value_id(step_tv, value_store);
        let end_id = tagged_to_value_id(end_tv, value_store);
        let start_id = tagged_to_value_id(start_tv, value_store);
        let (Some(start), Some(end), Some(step)) = (
            read_integral(value_store, start_id),
            read_integral(value_store, end_id),
            read_integral(value_store, step_id),
        ) else {
            pop_restore(stack, &[start_id, end_id, step_id]);
            return Ok(None);
        };
        vec![
            Value::Number(start as f64),
            Value::Number(end as f64),
            Value::Number(step as f64),
        ]
    };
    match crate::common::range_args::range_spec_from_values(&args) {
        Ok(spec) => {
            let v = crate::common::range_args::value_from_range_spec(spec);
            let result_id = store_value(v, value_store, heavy_store);
            stack::push_id(stack, result_id);
            Ok(Some(VMStatus::Continue))
        }
        Err("zero_step") => {
            let error = ExceptionHandler::runtime_error(
                &frames,
                "range() step cannot be zero".to_string(),
                line,
            );
            let status = ExceptionHandler::handle_exception_vm(
                stack,
                frames,
                exception_handlers,
                error,
                value_store,
                heavy_store,
            )?;
            Ok(Some(status))
        }
        Err(_) => Ok(None),
    }
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
    let available = crate::vm::stack::available_in_frame(stack, frame.stack_start);
    if available < 2 {
        return None;
    }
    let item_tv = crate::vm::stack::pop_direct(stack).unwrap_or(TaggedValue::null());
    let arr_tv = crate::vm::stack::pop_direct(stack).unwrap_or(TaggedValue::null());
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

/// `pop(array [, idx])` — remove at index and return the slot (preserves heap object identity).
pub(super) fn try_pop_fast_path(
    native_index: usize,
    arity: usize,
    stack: &mut Vec<TaggedValue>,
    frames: &mut Vec<CallFrame>,
    value_store: &mut ValueStore,
) -> Option<VMStatus> {
    if native_index != builtin::POP || !(arity == 1 || arity == 2) {
        return None;
    }
    let frame = frames.last().unwrap();
    let available = crate::vm::stack::available_in_frame(stack, frame.stack_start);
    if available < arity {
        return None;
    }
    let idx_tv = if arity == 2 {
        crate::vm::stack::pop_direct(stack).unwrap_or(TaggedValue::null())
    } else {
        TaggedValue::null()
    };
    let arr_tv = crate::vm::stack::pop_direct(stack).unwrap_or(TaggedValue::null());
    let arr_id = tagged_to_value_id(arr_tv, value_store);
    if value_store.is_flat_heap(arr_id) {
        stack::push(stack, arr_tv);
        if arity == 2 {
            stack::push(stack, idx_tv);
        }
        return None;
    }
    let mut idx: i64 = -1;
    if arity == 2 && !idx_tv.is_null() {
        if idx_tv.is_int() {
            idx = idx_tv.get_i32() as i64;
        } else if idx_tv.is_number() {
            let n = idx_tv.get_f64();
            if n.fract() != 0.0 {
                stack::push(stack, arr_tv);
                stack::push(stack, idx_tv);
                return None;
            }
            idx = n as i64;
        } else {
            stack::push(stack, arr_tv);
            stack::push(stack, idx_tv);
            return None;
        }
    }
    if let Some(ValueCell::Array(slots)) = value_store.get_mut(arr_id) {
        let n = slots.len();
        if n == 0 {
            stack::push(stack, TaggedValue::null());
            return Some(VMStatus::Continue);
        }
        if idx < 0 {
            idx += n as i64;
        }
        if idx < 0 || idx >= n as i64 {
            stack::push(stack, arr_tv);
            if arity == 2 {
                stack::push(stack, idx_tv);
            }
            return None;
        }
        let removed = slots.remove(idx as usize);
        stack::push(stack, removed);
        return Some(VMStatus::Continue);
    }
    stack::push(stack, arr_tv);
    if arity == 2 {
        stack::push(stack, idx_tv);
    }
    None
}

/// `len(x)` for array/string cells.
pub(super) fn try_len_fast_path(
    native_index: usize,
    arity: usize,
    stack: &mut Vec<TaggedValue>,
    frames: &mut Vec<CallFrame>,
    value_store: &mut ValueStore,
    heavy_store: &mut HeavyStore,
) -> Option<VMStatus> {
    if native_index != builtin::LEN || arity != 1 {
        return None;
    }
    let frame = frames.last().unwrap();
    let available = crate::vm::stack::available_in_frame(stack, frame.stack_start);
    if available < 1 {
        return None;
    }
    let arg_tv = crate::vm::stack::pop_direct(stack).unwrap_or(TaggedValue::null());
    let arg_id = tagged_to_value_id(arg_tv, value_store);
    if let Some(cell) = value_store.get(arg_id) {
        if let ValueCell::Object(_) = cell {
            let arg = load_value(arg_id, value_store, heavy_store);
            if class_has_special(&arg, "@len") {
                if let Ok(Some(v)) = dispatch_special(&arg, "@len", &[]) {
                    let result_id = store_value(v, value_store, heavy_store);
                    stack::push_id(stack, result_id);
                    return Some(VMStatus::Continue);
                }
            }
        }
        if let Some(len) = match cell {
            ValueCell::Array(ids) => {
                let n = if value_store.is_flat_heap(arg_id) {
                    ids.len() / 2
                } else {
                    ids.len()
                };
                Some(n as f64)
            }
            ValueCell::String(sid) => value_store.get_string(*sid).map(|s| s.len() as f64),
            ValueCell::Object(omap) => Some(omap.len() as f64),
            ValueCell::Set(smap) => Some(smap.len() as f64),
            ValueCell::ObjectFieldList { element_ids, .. } => {
                Some(element_ids.len() as f64)
            }
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
    let available = crate::vm::stack::available_in_frame(stack, frame.stack_start);
    if available < 1 {
        return None;
    }
    let arg_tv = crate::vm::stack::pop_direct(stack).unwrap_or(TaggedValue::null());
    let arg_id = tagged_to_value_id(arg_tv, value_store);
    let result_value = value_store.get(arg_id).and_then(|cell| {
        match (native_index, cell) {
            (idx, ValueCell::Number(n)) if idx == builtin::INT => {
                Some(Value::Int(int_value_from_f64_lossy(*n)))
            }
            (idx, ValueCell::Int(i)) if idx == builtin::INT => Some(Value::Int(*i)),
            (idx, ValueCell::Float(f)) if idx == builtin::INT => {
                Some(Value::Int(coerce_to_int_value(&Value::Float(*f))))
            }
            (idx, ValueCell::Bool(b)) if idx == builtin::INT => {
                Some(Value::Int(IntValue::Finite(if *b { 1 } else { 0 })))
            }
            (idx, ValueCell::Null) if idx == builtin::INT => Some(Value::Int(IntValue::Finite(0))),
            (idx, ValueCell::Number(n)) if idx == builtin::FLOAT => {
                Some(Value::Float(FloatValue::classify_f64(*n)))
            }
            (idx, ValueCell::Int(i)) if idx == builtin::FLOAT => {
                Some(Value::Float(i.widen_to_float()))
            }
            (idx, ValueCell::Float(f)) if idx == builtin::FLOAT => Some(Value::Float(*f)),
            (idx, ValueCell::Bool(b)) if idx == builtin::FLOAT => {
                Some(Value::Float(FloatValue::Finite(if *b { 1.0 } else { 0.0 })))
            }
            (idx, ValueCell::Null) if idx == builtin::FLOAT => {
                Some(Value::Float(FloatValue::Finite(0.0)))
            }
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
            (idx, ValueCell::Int(_)) if idx == builtin::TYPEOF => {
                Some(Value::String("int".to_string()))
            }
            (idx, ValueCell::Float(f)) if idx == builtin::TYPEOF => {
                Some(Value::String(if float_is_int_surface(*f) {
                    "int".to_string()
                } else {
                    "float".to_string()
                }))
            }
            (idx, ValueCell::Number(n)) if idx == builtin::TYPEOF => {
                Some(Value::String(if number_is_int_surface(*n) {
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
            (idx, ValueCell::ObjectFieldList { .. }) if idx == builtin::TYPEOF => {
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
    let available = crate::vm::stack::available_in_frame(stack, frame.stack_start);
    if available < 2 {
        return None;
    }
    let headers_tv = crate::vm::stack::pop_direct(stack).unwrap_or(TaggedValue::null());
    let data_tv = crate::vm::stack::pop_direct(stack).unwrap_or(TaggedValue::null());
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
            (0..1).map(|i| format!("Column_{}", i)).collect()
        } else {
            v
        }
    };
    let header_count = headers.len().max(1);
    let Some((flat_cell_ids, num_cols)) =
        crate::vm::table_ops::build_view_flat_from_row_slots(
            &row_slots,
            header_count,
            value_store,
            heavy_store,
        )
    else {
        stack::push_id(stack, data_id);
        stack::push_id(stack, headers_id);
        return None;
    };
    let table = Table::from_flat_view(flat_cell_ids, num_cols, headers);
    let table_val = Value::Table(Rc::new(RefCell::new(table)));
    let heavy_idx = heavy_store.push(table_val);
    let result_id = value_store.allocate(ValueCell::Heavy(heavy_idx));
    stack::push_id(stack, result_id);
    Some(VMStatus::Continue)
}

/// `abs(x)` for int / whole number without generic native invoke.
pub(super) fn try_abs_fast_path(
    native_index: usize,
    arity: usize,
    stack: &mut Vec<TaggedValue>,
    frames: &mut Vec<CallFrame>,
    _value_store: &mut ValueStore,
) -> Option<VMStatus> {
    if native_index != builtin::ABS || arity != 1 {
        return None;
    }
    let frame = frames.last().unwrap();
    let available = crate::vm::stack::available_in_frame(stack, frame.stack_start);
    if available < 1 {
        return None;
    }
    let arg_tv = crate::vm::stack::pop_direct(stack).unwrap_or(TaggedValue::null());
    if arg_tv.is_int() {
        let n = arg_tv.get_i32();
        let abs_n = if n == i32::MIN {
            i64::from(n).unsigned_abs() as f64
        } else {
            n.unsigned_abs() as f64
        };
        stack::push(stack, TaggedValue::from_f64(abs_n));
        return Some(VMStatus::Continue);
    }
    if arg_tv.is_number() {
        let n = arg_tv.get_f64();
        if n.is_finite() && n.fract() == 0.0 {
            let abs_n = if n >= 0.0 { n } else { -n };
            stack::push(stack, TaggedValue::from_f64(abs_n));
            return Some(VMStatus::Continue);
        }
        if n.is_finite() {
            stack::push(stack, TaggedValue::from_f64(n.abs()));
            return Some(VMStatus::Continue);
        }
    }
    stack::push(stack, arg_tv);
    None
}
