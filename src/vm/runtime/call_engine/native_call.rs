//! Execution of native (builtin and ABI) function calls.

use crate::debug_println;
use crate::common::{error::LangError, value::Value, value_store::{ValueCell, ValueId, ValueStore}, TaggedValue};
use crate::common::table::Table;
use crate::vm::types::VMStatus;
use crate::vm::frame::CallFrame;
use crate::vm::exceptions::ExceptionHandler;
use crate::vm::stack;
use crate::vm::vm::VM_CALL_CONTEXT;
use crate::vm::global_slot::GlobalSlot;
use crate::vm::store_convert::{store_value, load_value, update_cell_if_mutable, tagged_to_value_id};
use crate::vm::heavy_store::HeavyStore;
use crate::common::error::ErrorType;
use crate::vm::host::HostEntry;
use crate::vm::types::{ExplicitRelation, ExplicitPrimaryKey};
use std::rc::Rc;
use std::cell::RefCell;

/// Execute a native (builtin or ABI) call. Called from call_dispatch when callee is Value::NativeFunction(native_index).
#[allow(clippy::too_many_arguments)]
pub fn execute_native_call(
    native_index: usize,
    arity: usize,
    line: usize,
    stack: &mut Vec<TaggedValue>,
    frames: &mut Vec<CallFrame>,
    natives: &[HostEntry],
    exception_handlers: &mut Vec<ExceptionHandler>,
    value_store: &mut ValueStore,
    heavy_store: &mut HeavyStore,
    native_args_buffer: &mut Vec<Value>,
    reusable_native_arg_ids: &mut Vec<ValueId>,
    reusable_all_popped: &mut Vec<Value>,
    abi_natives: &mut Vec<crate::abi::NativeAbiFn>,
    explicit_relations: &mut Vec<ExplicitRelation>,
    explicit_primary_keys: &mut Vec<ExplicitPrimaryKey>,
    globals: &mut Vec<GlobalSlot>,
    explicit_global_names: &std::collections::BTreeMap<usize, String>,
    vm_ptr: *mut crate::vm::vm::Vm,
) -> Result<VMStatus, LangError> {
    let builtin_count = natives.len();
    if native_index >= builtin_count + abi_natives.len() {
        let error = ExceptionHandler::runtime_error(
            &frames,
            format!("Native function index {} out of bounds", native_index),
            line,
        );
        match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error, value_store, heavy_store) {
            Ok(()) => return Ok(VMStatus::Continue),
            Err(e) => return Err(e),
        }
    }

    // Fast path for range(n) / range(start, end) / range(start, end, step)
    const RANGE_NATIVE_INDEX: usize = 2;
    if native_index == RANGE_NATIVE_INDEX && (arity == 1 || arity == 2 || arity == 3) {
        let frame = frames.last().unwrap();
        let available = stack.len().saturating_sub(frame.stack_start);
        let need = if arity == 1 { 1 } else if arity == 2 { 2 } else { 3 };
        if available >= need {
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
                read_number(value_store, n_id).map(|n| (0_i64, n.max(0), 1_i64)).map_or_else(
                    || { stack::push_id(stack, n_id); None },
                    |t| Some(t),
                )
            } else if arity == 2 {
                let end_tv = stack.pop().unwrap_or(TaggedValue::null());
                let start_tv = stack.pop().unwrap_or(TaggedValue::null());
                let end_id = tagged_to_value_id(end_tv, value_store);
                let start_id = tagged_to_value_id(start_tv, value_store);
                match (read_number(value_store, start_id), read_number(value_store, end_id)) {
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
                match (read_number(value_store, start_id), read_number(value_store, end_id), read_number(value_store, step_id)) {
                    (Some(start), Some(end), Some(step)) if step != 0 => Some((start, end, step)),
                    _ => {
                        stack::push_id(stack, start_id);
                        stack::push_id(stack, end_id);
                        stack::push_id(stack, step_id);
                        None
                    }
                }
            };
            if let Some((start, end, step)) = params {
                let len = if step > 0 {
                    if start >= end { 0 } else { ((end - start) as u64 / step as u64).min(usize::MAX as u64) as usize }
                } else {
                    if start <= end { 0 } else { ((start - end) as u64 / (-step) as u64).min(usize::MAX as u64) as usize }
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
                return Ok(VMStatus::Continue);
            }
        }
    }

    // Fast path for push(arr, item)
    const PUSH_NATIVE_INDEX: usize = 35;
    if native_index == PUSH_NATIVE_INDEX && arity == 2 {
        let frame = frames.last().unwrap();
        let available = stack.len().saturating_sub(frame.stack_start);
        if available >= 2 {
            let item_tv = stack.pop().unwrap_or(TaggedValue::null());
            let array_tv = stack.pop().unwrap_or(TaggedValue::null());
            let array_id = tagged_to_value_id(array_tv, value_store);
            if let Some(ValueCell::Array(ref mut slots)) = value_store.get_mut(array_id) {
                slots.push(item_tv);
                stack::push_id(stack, array_id);
                return Ok(VMStatus::Continue);
            }
            stack::push_id(stack, array_id);
            stack::push(stack, item_tv);
        }
    }

    // Fast path for len(x)
    const LEN_NATIVE_INDEX: usize = 1;
    if native_index == LEN_NATIVE_INDEX && arity == 1 {
        let frame = frames.last().unwrap();
        let available = stack.len().saturating_sub(frame.stack_start);
        if available >= 1 {
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
                    return Ok(VMStatus::Continue);
                }
            }
            stack::push_id(stack, arg_id);
        }
    }

    // Fast path for int(x), float(x), str(x), typeof(x)
    if arity == 1 {
        let frame = frames.last().unwrap();
        let available = stack.len().saturating_sub(frame.stack_start);
        if available >= 1 {
            let arg_tv = stack.pop().unwrap_or(TaggedValue::null());
            let arg_id = tagged_to_value_id(arg_tv, value_store);
            let result_value = value_store.get(arg_id).and_then(|cell| {
                match (native_index, cell) {
                    (3, ValueCell::Number(n)) => Some(Value::Number(n.trunc())),
                    (3, ValueCell::Bool(b)) => Some(Value::Number(if *b { 1.0 } else { 0.0 })),
                    (3, ValueCell::Null) => Some(Value::Number(0.0)),
                    (4, ValueCell::Number(n)) => Some(Value::Number(*n)),
                    (4, ValueCell::Bool(b)) => Some(Value::Number(if *b { 1.0 } else { 0.0 })),
                    (4, ValueCell::Null) => Some(Value::Number(0.0)),
                    (6, ValueCell::Number(n)) => Some(Value::String(n.to_string())),
                    (6, ValueCell::Bool(b)) => Some(Value::String(if *b { "true".to_string() } else { "false".to_string() })),
                    (6, ValueCell::String(sid)) => value_store.get_string(*sid).map(|s| Value::String(s.to_string())),
                    (6, ValueCell::Null) => Some(Value::String("null".to_string())),
                    (8, ValueCell::Number(n)) => Some(Value::String(if n.fract() == 0.0 { "int".to_string() } else { "float".to_string() })),
                    (8, ValueCell::Bool(_)) => Some(Value::String("bool".to_string())),
                    (8, ValueCell::String(_)) => Some(Value::String("string".to_string())),
                    (8, ValueCell::Null) => Some(Value::String("null".to_string())),
                    (8, ValueCell::Array(_)) => Some(Value::String("array".to_string())),
                    (8, ValueCell::Tuple(_)) => Some(Value::String("tuple".to_string())),
                    (8, ValueCell::Object(_)) => Some(Value::String("object".to_string())),
                    (8, ValueCell::Path(_)) => Some(Value::String("path".to_string())),
                    (8, ValueCell::Function(_)) | (8, ValueCell::NativeFunction(_)) => Some(Value::String("function".to_string())),
                    _ => None,
                }
            });
            if let Some(v) = result_value {
                let result_id = store_value(v, value_store, heavy_store);
                stack::push_id(stack, result_id);
                return Ok(VMStatus::Continue);
            }
            stack::push_id(stack, arg_id);
        }
    }

    // Fast path for table(data, headers) - index 43
    const TABLE_NATIVE_INDEX_43: usize = 43;
    if native_index == TABLE_NATIVE_INDEX_43 && arity == 2 {
        let frame = frames.last().unwrap();
        let available = stack.len().saturating_sub(frame.stack_start);
        if available >= 2 {
            let headers_tv = stack.pop().unwrap_or(TaggedValue::null());
            let data_tv = stack.pop().unwrap_or(TaggedValue::null());
            let headers_id = tagged_to_value_id(headers_tv, value_store);
            let data_id = tagged_to_value_id(data_tv, value_store);
            let row_slots_opt = value_store.get(data_id).and_then(|c| if let ValueCell::Array(s) = c { Some(s.clone()) } else { None });
            if let Some(row_slots) = row_slots_opt {
                let num_cols = row_slots.first().and_then(|row_tv| {
                    if row_tv.is_heap() {
                        value_store.get(row_tv.get_heap_id()).and_then(|c| match c {
                            ValueCell::Array(s) => Some(s.len()),
                            _ => None,
                        })
                    } else {
                        None
                    }
                }).unwrap_or(0);
                let mut flat_cell_ids = Vec::with_capacity(row_slots.len() * num_cols.max(1));
                for row_tv in row_slots.iter() {
                    if row_tv.is_heap() {
                        let row_id = row_tv.get_heap_id();
                        let cell_slots: Vec<TaggedValue> = value_store.get(row_id)
                            .and_then(|c| if let ValueCell::Array(s) = c { Some(s.clone()) } else { None })
                            .unwrap_or_default();
                        for slot in cell_slots.iter() {
                            flat_cell_ids.push(tagged_to_value_id(*slot, value_store));
                        }
                    }
                }
                let headers: Vec<String> = {
                    let header_slots: Vec<TaggedValue> = value_store.get(headers_id)
                        .and_then(|c| if let ValueCell::Array(s) = c { Some(s.clone()) } else { None })
                        .unwrap_or_default();
                    let mut v = Vec::with_capacity(header_slots.len());
                    for slot in header_slots.iter() {
                        let val = load_value(tagged_to_value_id(*slot, value_store), value_store, heavy_store);
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
                return Ok(VMStatus::Continue);
            }
            stack::push_id(stack, data_id);
            stack::push_id(stack, headers_id);
        }
    }

    // Tensor methods max_idx / min_idx and database engine methods
    use crate::ml::natives as ml_natives;
    let max_idx_ptr = ml_natives::native_max_idx as *const ();
    let is_max_idx = native_index < builtin_count && natives[native_index].as_fn_ptr() == Some(max_idx_ptr);
    let min_idx_ptr = ml_natives::native_min_idx as *const ();
    let is_min_idx = native_index < builtin_count && natives[native_index].as_fn_ptr() == Some(min_idx_ptr);

    use crate::database_engine::natives as db_natives;
    let is_db_connect = native_index < builtin_count && natives[native_index].as_fn_ptr() == Some(db_natives::native_engine_connect as *const ());
    let is_db_execute = native_index < builtin_count && natives[native_index].as_fn_ptr() == Some(db_natives::native_engine_execute as *const ());
    let is_db_query = native_index < builtin_count && natives[native_index].as_fn_ptr() == Some(db_natives::native_engine_query as *const ());
    let is_db_run = native_index < builtin_count && natives[native_index].as_fn_ptr() == Some(db_natives::native_engine_run as *const ());
    let is_db_cluster_add = native_index < builtin_count && natives[native_index].as_fn_ptr() == Some(db_natives::native_cluster_add as *const ());
    let is_db_cluster_get = native_index < builtin_count && natives[native_index].as_fn_ptr() == Some(db_natives::native_cluster_get as *const ());
    let is_db_cluster_names = native_index < builtin_count && natives[native_index].as_fn_ptr() == Some(db_natives::native_cluster_names as *const ());
    let is_db_column = native_index < builtin_count && natives[native_index].as_fn_ptr() == Some(db_natives::native_column as *const ());
    let is_db_engine_method = is_db_connect || is_db_execute || is_db_query || is_db_run
        || is_db_cluster_add || is_db_cluster_get || is_db_cluster_names;

    native_args_buffer.clear();
    let mut native_arg_ids: Option<&mut Vec<ValueId>> = None;
    if is_db_engine_method {
        let frame = frames.last().unwrap();
        let available = stack.len().saturating_sub(frame.stack_start);
        let to_pop_total = arity.min(available);
        reusable_all_popped.clear();
        reusable_all_popped.reserve(to_pop_total);
        for _ in 0..to_pop_total {
            let tv = stack.pop().unwrap_or(TaggedValue::null());
            let id = tagged_to_value_id(tv, value_store);
            reusable_all_popped.push(load_value(id, value_store, heavy_store));
        }
        reusable_all_popped.reverse();
        let receiver_predicate: fn(&Value) -> bool = if is_db_cluster_add || is_db_cluster_get || is_db_cluster_names {
            |v| matches!(v, Value::DatabaseCluster(_))
        } else {
            |v| matches!(v, Value::DatabaseEngine(_) | Value::DatabaseCluster(_))
        };
        if let Some(engine_idx) = reusable_all_popped.iter().position(receiver_predicate) {
            let receiver = reusable_all_popped.remove(engine_idx);
            native_args_buffer.push(receiver);
        }
        native_args_buffer.extend(reusable_all_popped.drain(..));
    }
    if (is_max_idx || is_min_idx) && arity == 0 {
        if let Some(&top_tv) = stack.last() {
            let top_id = tagged_to_value_id(top_tv, value_store);
            let top_val = load_value(top_id, value_store, heavy_store);
            if let Value::Tensor(tensor_rc) = &top_val {
                native_args_buffer.push(Value::Tensor(Rc::clone(tensor_rc)));
                stack.pop();
            } else {
                let error = ExceptionHandler::runtime_error(&frames,
                    "Tensor method called without tensor on stack".to_string(),
                    line,
                );
                match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error, value_store, heavy_store) {
                    Ok(()) => return Ok(VMStatus::Continue),
                    Err(e) => return Err(e),
                }
            }
        } else {
            let error = ExceptionHandler::runtime_error(&frames,
                "Tensor method called without tensor on stack".to_string(),
                line,
            );
            match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error, value_store, heavy_store) {
                Ok(()) => return Ok(VMStatus::Continue),
                Err(e) => return Err(e),
            }
        }
    } else if !is_db_engine_method {
        let frame = frames.last().unwrap();
        let available_args = stack.len() - frame.stack_start;
        if available_args < arity {
            let error = ExceptionHandler::runtime_error(
                &frames,
                format!(
                    "Not enough arguments on stack for native function: expected {} but got {}",
                    arity,
                    available_args
                ),
                line,
            );
            match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error, value_store, heavy_store) {
                Ok(()) => return Ok(VMStatus::Continue),
                Err(e) => return Err(e),
            }
        }
        // Fast path table(data, headers) - index 45
        const TABLE_NATIVE_INDEX_45: usize = 45;
        if native_index == TABLE_NATIVE_INDEX_45 && arity == 2 {
            let headers_tv = stack.pop().unwrap_or(TaggedValue::null());
            let data_tv = stack.pop().unwrap_or(TaggedValue::null());
            let headers_id = tagged_to_value_id(headers_tv, value_store);
            let data_id = tagged_to_value_id(data_tv, value_store);
            let row_slots_opt2 = value_store.get(data_id).and_then(|c| if let ValueCell::Array(s) = c { Some(s.clone()) } else { None });
            if let Some(row_slots) = row_slots_opt2 {
                let num_cols = row_slots.first().and_then(|row_tv| {
                    if row_tv.is_heap() {
                        value_store.get(row_tv.get_heap_id()).and_then(|c| match c {
                            ValueCell::Array(slots) => Some(slots.len()),
                            _ => None,
                        })
                    } else {
                        None
                    }
                }).unwrap_or(0);
                if num_cols > 0 && !row_slots.is_empty() {
                    let mut flat_cell_ids = Vec::with_capacity(row_slots.len() * num_cols);
                    let mut ok = true;
                    for row_tv in row_slots.iter() {
                        if !row_tv.is_heap() {
                            ok = false;
                            break;
                        }
                        let row_id = row_tv.get_heap_id();
                        let cell_slots: Vec<TaggedValue> = value_store.get(row_id)
                            .and_then(|c| if let ValueCell::Array(s) = c { Some(s.clone()) } else { None })
                            .unwrap_or_default();
                        if cell_slots.len() >= num_cols {
                            for slot in cell_slots.iter().take(num_cols) {
                                flat_cell_ids.push(tagged_to_value_id(*slot, value_store));
                            }
                        } else {
                            ok = false;
                            break;
                        }
                    }
                    if ok && flat_cell_ids.len() == row_slots.len() * num_cols {
                        let headers_val = load_value(headers_id, value_store, heavy_store);
                        let header_strings: Vec<String> = match &headers_val {
                            Value::Array(rc) => rc.borrow().iter().map(|v| v.to_string()).collect(),
                            _ => Vec::new(),
                        };
                        if header_strings.len() >= num_cols {
                            let table = Table::from_flat_view(flat_cell_ids, num_cols, header_strings);
                            let idx = heavy_store.push(Value::Table(Rc::new(RefCell::new(table))));
                            let result_id = value_store.allocate(ValueCell::Heavy(idx));
                            stack::push_id(stack, result_id);
                            return Ok(VMStatus::Continue);
                        }
                    }
                }
            }
            stack::push_id(stack, data_id);
            stack::push_id(stack, headers_id);
        }
        reusable_native_arg_ids.clear();
        reusable_native_arg_ids.reserve(arity);
        native_args_buffer.reserve(arity);
        for _ in 0..arity {
            let arg_tv = stack.pop().unwrap_or(TaggedValue::null());
            let arg_id = tagged_to_value_id(arg_tv, value_store);
            reusable_native_arg_ids.push(arg_id);
            native_args_buffer.push(load_value(arg_id, value_store, heavy_store));
        }
        reusable_native_arg_ids.reverse();
        native_args_buffer.reverse();
        native_arg_ids = Some(reusable_native_arg_ids);
    }

    VM_CALL_CONTEXT.with(|ctx| {
        *ctx.borrow_mut() = Some(vm_ptr);
    });

    if native_index == 2 {
        if arity < 1 || arity > 3 {
            let error = ExceptionHandler::runtime_error(&frames,
                format!("range() expects 1, 2, or 3 arguments, got {}", arity),
                line,
            );
            match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error, value_store, heavy_store) {
                Ok(()) => return Ok(VMStatus::Continue),
                Err(e) => return Err(e),
            }
        }
        for arg in native_args_buffer.iter() {
            if !matches!(arg, Value::Number(_)) {
                let error = ExceptionHandler::runtime_error(&frames,
                    "range() arguments must be numbers".to_string(),
                    line,
                );
                match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error, value_store, heavy_store) {
                    Ok(()) => return Ok(VMStatus::Continue),
                    Err(e) => return Err(e),
                }
            }
        }
        if arity == 3 {
            if let Value::Number(step) = &native_args_buffer[2] {
                if *step == 0.0 {
                    let error = ExceptionHandler::runtime_error(&frames,
                        "range() step cannot be zero".to_string(),
                        line,
                    );
                    match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error, value_store, heavy_store) {
                        Ok(()) => return Ok(VMStatus::Continue),
                        Err(e) => return Err(e),
                    }
                }
            }
        }
    }
    if native_index == 72 {
        if arity != 1 {
            let error = ExceptionHandler::runtime_error(&frames,
                format!("enum() expects 1 argument, got {}", arity),
                line,
            );
            match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error, value_store, heavy_store) {
                Ok(()) => return Ok(VMStatus::Continue),
                Err(e) => return Err(e),
            }
        }
    }

    if native_args_buffer.len() == 3 {
        if let Some(Value::Table(rc)) = native_args_buffer.get(0) {
            let t = rc.borrow();
            if t.is_view() {
                let owned = t.materialize_with(|id| load_value(id, value_store, heavy_store));
                drop(t);
                native_args_buffer[0] = Value::Table(Rc::new(RefCell::new(owned)));
            }
        }
    } else if native_args_buffer.len() == 4 {
        if let Some(Value::Table(rc)) = native_args_buffer.get(1) {
            let t = rc.borrow();
            if t.is_view() {
                let owned = t.materialize_with(|id| load_value(id, value_store, heavy_store));
                drop(t);
                native_args_buffer[1] = Value::Table(Rc::new(RefCell::new(owned)));
            }
        }
    }

    const INSTANCEOF_NATIVE_INDEX: usize = 9;
    let second_is_class = native_args_buffer.len() >= 2
        && matches!(&native_args_buffer[1], Value::Object(rc) if rc.borrow().get("__class_name").is_some());
    let skip_drop = arity == 1
        || native_index == INSTANCEOF_NATIVE_INDEX
        || second_is_class
        || (native_index == 1 && native_args_buffer.len() == 1)
        || is_db_column;
    if !skip_drop && !native_args_buffer.is_empty() {
        if let Value::Object(_) = &native_args_buffer[0] {
            native_args_buffer.remove(0);
        }
    }

    let result = if native_index < builtin_count {
        match natives[native_index].invoke(&native_args_buffer) {
            Ok(v) => v,
            Err(e) => {
                match ExceptionHandler::handle_exception(stack, frames, exception_handlers, e, value_store, heavy_store) {
                    Ok(()) => return Ok(VMStatus::Continue),
                    Err(ee) => return Err(ee),
                }
            }
        }
    } else {
        crate::vm::native_loader::call_abi_native(
            abi_natives[native_index - builtin_count],
            &native_args_buffer,
        )
    };

    VM_CALL_CONTEXT.with(|ctx| {
        *ctx.borrow_mut() = None;
    });

    if let Some(abi_err) = crate::vm::native_loader::take_last_abi_error() {
        match ExceptionHandler::handle_exception(stack, frames, exception_handlers, abi_err, value_store, heavy_store) {
            Ok(()) => return Ok(VMStatus::Continue),
            Err(e) => return Err(e),
        }
    }

    if native_index == 65 {
        let relations = unsafe { (*vm_ptr).take_pending_relations() };
        for (table1_rc, col1_name, table2_rc, col2_name) in relations {
            let mut found_table1_name = None;
            let mut found_table2_name = None;
            for (index, slot) in globals.iter_mut().enumerate() {
                let value_id = slot.resolve_to_value_id(value_store);
                let value = load_value(value_id, value_store, heavy_store);
                if let Value::Table(table) = &value {
                    if Rc::ptr_eq(table, &table1_rc) {
                        if let Some(var_name) = explicit_global_names.get(&index) {
                            found_table1_name = Some(var_name.clone());
                        }
                    }
                    if Rc::ptr_eq(table, &table2_rc) {
                        if let Some(var_name) = explicit_global_names.get(&index) {
                            found_table2_name = Some(var_name.clone());
                        }
                    }
                }
            }
            if let (Some(table1_name), Some(table2_name)) = (found_table1_name, found_table2_name) {
                explicit_relations.push(ExplicitRelation {
                    source_table_name: table2_name,
                    source_column_name: col2_name,
                    target_table_name: table1_name,
                    target_column_name: col1_name,
                });
            }
        }
    }

    if native_index == 66 {
        let primary_keys = unsafe { (*vm_ptr).take_pending_primary_keys() };
        for (table_rc, col_name) in primary_keys {
            let mut found_table_name = None;
            for (index, slot) in globals.iter_mut().enumerate() {
                let value_id = slot.resolve_to_value_id(value_store);
                let value = load_value(value_id, value_store, heavy_store);
                if let Value::Table(table) = &value {
                    if Rc::ptr_eq(table, &table_rc) {
                        if let Some(var_name) = explicit_global_names.get(&index) {
                            found_table_name = Some(var_name.clone());
                        }
                    }
                }
            }
            if let Some(table_name) = found_table_name {
                explicit_primary_keys.push(ExplicitPrimaryKey {
                    table_name,
                    column_name: col_name,
                });
            }
        }
    }

    use crate::websocket::take_native_error;
    if let Some(error_msg) = take_native_error() {
        if error_msg.contains("Falling back to CPU") ||
            error_msg.contains("not available") && error_msg.contains("GPU") {
            debug_println!("⚠️  Предупреждение: {}", error_msg);
        } else {
            let error_type = if error_msg.contains("ShapeError") ||
                error_msg.contains("Shape mismatch") ||
                error_msg.starts_with("ShapeError:") {
                ErrorType::ValueError
            } else {
                ErrorType::IOError
            };
            let error = ExceptionHandler::runtime_error_with_type(&frames, error_msg, line, error_type);
            match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error, value_store, heavy_store) {
                Ok(()) => return Ok(VMStatus::Continue),
                Err(e) => return Err(e),
            }
        }
    }

    if let Some(ref ids) = native_arg_ids {
        for (i, &id) in ids.iter().enumerate() {
            if i < native_args_buffer.len() {
                update_cell_if_mutable(id, &native_args_buffer[i], value_store, heavy_store);
            }
        }
    }

    stack::push_id(stack, store_value(result, value_store, heavy_store));
    Ok(VMStatus::Continue)
}
