// Object/array opcodes: MakeArray, MakeTuple, MakeObject, UnpackObject, MakeObjectDynamic, MakeArrayDynamic,
// GetArrayLength, TableFilter, GetArrayElement, SetArrayElement, Clone.
// Logic preserved 1:1 from executor.rs — no semantic changes.

use crate::common::{
    error::LangError,
    object_map::ObjectMap,
    set_map::SetMap,
    value::Value,
    value_store::{ValueCell, ValueId, ValueStore},
};
use crate::vm::exceptions::ExceptionHandler;
use crate::vm::frame::CallFrame;
use crate::vm::heavy_store::HeavyStore;
use crate::vm::native_loader::call_abi_native;
use crate::vm::stack;
use crate::vm::store_convert::tagged_to_value_id;
use crate::vm::store_convert::{load_value, object_map_upsert, store_value};
use crate::vm::types::VMStatus;
use crate::vm::execution_context::RestoreVmCallContextGuard;
use crate::vm::vm::{current_vm_ptr, Vm};

use super::helpers::pop_to_value_id;

/// Opaque length: delegates to the loaded plugin's `native_plugin_call(opaque, "len")` if exported.
/// No host knowledge of plugin tags or `dataset_len` — plugins implement `"len"` for types they own.
pub(crate) fn plugin_opaque_len_via_plugin_call(opaque: &Value) -> Option<Value> {
    if !matches!(opaque, Value::PluginOpaque { .. }) {
        return None;
    }
    let vm_ptr = current_vm_ptr()?;
    unsafe {
        let vm = &*vm_ptr;
        let native_idx = vm.plugin_call_native?;
        let builtin_count = vm.builtin_natives_count();
        let abi = vm.get_abi_natives();
        if native_idx < builtin_count || native_idx >= builtin_count + abi.len() {
            return None;
        }
        let v = call_abi_native(
            abi[native_idx - builtin_count],
            &[opaque.clone(), Value::String("len".to_string())],
            Some((vm.value_store(), vm.heavy_store())),
        );
        if crate::vm::native_loader::take_last_abi_error().is_some() {
            return None;
        }
        Some(v)
    }
}

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
        slots.push(stack::pop(
            stack,
            frames,
            exception_handlers,
            value_store,
            heavy_store,
        )?);
    }
    slots.reverse();
    let result_id = value_store.allocate_ephemeral_arena(ValueCell::Array(slots));
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
        element_ids.push(pop_to_value_id(
            stack,
            frames,
            exception_handlers,
            value_store,
            heavy_store,
        )?);
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
    let mut omap = ObjectMap::with_capacity_plain(pair_count);
    for _ in 0..pair_count {
        let value_id =
            pop_to_value_id(stack, frames, exception_handlers, value_store, heavy_store)?;
        let key_id = pop_to_value_id(stack, frames, exception_handlers, value_store, heavy_store)?;
        let key_value = load_value(key_id, value_store, heavy_store);
        if !crate::common::type_model::is_hashable_value(&key_value) {
            return Err(ExceptionHandler::runtime_error(
                &frames,
                format!(
                    "unhashable type: {}",
                    crate::vm::calls::get_type_name_value(&key_value)
                ),
                line,
            ));
        }
        object_map_upsert(
            &mut omap,
            value_store,
            heavy_store,
            &key_value,
            key_id,
            value_id,
        );
    }
    let result_id = value_store.allocate_ephemeral(ValueCell::Object(omap));
    value_store.mark_plain_object(result_id);
    stack::push_id(stack, result_id);
    Ok(VMStatus::Continue)
}

pub fn op_make_set(
    count: usize,
    line: usize,
    stack: &mut Vec<crate::common::TaggedValue>,
    frames: &mut Vec<CallFrame>,
    exception_handlers: &mut Vec<ExceptionHandler>,
    value_store: &mut ValueStore,
    heavy_store: &mut HeavyStore,
) -> Result<VMStatus, LangError> {
    let mut smap = SetMap::with_capacity(count);
    for _ in 0..count {
        let elem_id =
            pop_to_value_id(stack, frames, exception_handlers, value_store, heavy_store)?;
        let elem = load_value(elem_id, value_store, heavy_store);
        if !crate::common::type_model::is_hashable_value(&elem) {
            return Err(ExceptionHandler::runtime_error(
                &frames,
                format!(
                    "unhashable type: {}",
                    crate::vm::calls::get_type_name_value(&elem)
                ),
                line,
            ));
        }
        let h = crate::common::type_model::object_key_hash_value(&elem).expect("hashable");
        let canonical = crate::common::numeric::integer_value_as_i64_if_whole(&elem);
        smap.insert(
            h,
            elem_id,
            |id| load_value(id, value_store, heavy_store) == elem,
            canonical,
        );
    }
    let result_id = value_store.allocate_ephemeral(ValueCell::Set(smap));
    value_store.mark_plain_set(result_id);
    stack::push_id(stack, result_id);
    Ok(VMStatus::Continue)
}

pub fn op_make_set_dynamic(
    line: usize,
    stack: &mut Vec<crate::common::TaggedValue>,
    frames: &mut Vec<CallFrame>,
    exception_handlers: &mut Vec<ExceptionHandler>,
    value_store: &mut ValueStore,
    heavy_store: &mut HeavyStore,
) -> Result<VMStatus, LangError> {
    let count_tv = stack::pop(
        stack,
        frames,
        exception_handlers,
        value_store,
        heavy_store,
    )?;
    let count = if count_tv.is_number() {
        count_tv.get_f64() as usize
    } else if count_tv.is_int() {
        count_tv.get_i32() as usize
    } else {
        return Err(ExceptionHandler::runtime_error(
            &frames,
            "MakeSetDynamic: count must be a number".to_string(),
            line,
        ));
    };
    op_make_set(count, line, stack, frames, exception_handlers, value_store, heavy_store)
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
        Some(ValueCell::Object(m)) => m
            .iter_entries()
            .map(|(_, kid, vid)| {
                let k = match load_value(kid, value_store, heavy_store) {
                    Value::String(s) => s,
                    Value::Number(n) => format!("{}", n),
                    other => format!("{:?}", other),
                };
                (k, vid)
            })
            .collect(),
        _ => {
            let val = load_value(obj_id, value_store, heavy_store);
            if let Value::Object(rc) = &val {
                rc.borrow()
                    .str_key_entries_cloned()
                    .into_iter()
                    .map(|(k, v)| (k, store_value(v, value_store, heavy_store)))
                    .collect()
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
    let count_tv = stack::pop(stack, frames, exception_handlers, value_store, heavy_store)?;
    let pair_count = if count_tv.is_number() {
        let n = count_tv.get_f64();
        if n < 0.0 || n.fract() != 0.0 {
            return Err(ExceptionHandler::runtime_error(
                &frames,
                "Object pair count must be a non-negative whole number".to_string(),
                line,
            ));
        }
        n as usize
    } else if count_tv.is_int() {
        let n = count_tv.get_i32();
        if n < 0 {
            return Err(ExceptionHandler::runtime_error(
                &frames,
                "Object pair count must be non-negative".to_string(),
                line,
            ));
        }
        n as usize
    } else {
        let count_id = tagged_to_value_id(count_tv, value_store);
        match value_store.get(count_id) {
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
                    Value::Int(crate::common::numeric::IntValue::Finite(n)) if n >= 0 => n as usize,
                    _ => {
                        return Err(ExceptionHandler::runtime_error(
                            &frames,
                            "MakeObjectDynamic requires a number (pair count) on stack".to_string(),
                            line,
                        ));
                    }
                }
            }
        }
    };
    let mut omap = ObjectMap::with_capacity_plain(pair_count);
    for _ in 0..pair_count {
        let value_id =
            pop_to_value_id(stack, frames, exception_handlers, value_store, heavy_store)?;
        let key_id = pop_to_value_id(stack, frames, exception_handlers, value_store, heavy_store)?;
        let key_value = load_value(key_id, value_store, heavy_store);
        if !crate::common::type_model::is_hashable_value(&key_value) {
            return Err(ExceptionHandler::runtime_error(
                &frames,
                format!(
                    "unhashable type: {}",
                    crate::vm::calls::get_type_name_value(&key_value)
                ),
                line,
            ));
        }
        object_map_upsert(
            &mut omap,
            value_store,
            heavy_store,
            &key_value,
            key_id,
            value_id,
        );
    }
    let result_id = value_store.allocate(ValueCell::Object(omap));
    value_store.mark_plain_object(result_id);
    stack::push_id(stack, result_id);
    Ok(VMStatus::Continue)
}

pub(crate) fn op_get_array_length(
    line: usize,
    stack: &mut Vec<crate::common::TaggedValue>,
    frames: &mut Vec<CallFrame>,
    exception_handlers: &mut Vec<ExceptionHandler>,
    value_store: &mut ValueStore,
    heavy_store: &mut HeavyStore,
    _vm_ptr: *mut Vm,
) -> Result<VMStatus, LangError> {
    let array_id = pop_to_value_id(stack, frames, exception_handlers, value_store, heavy_store)?;
    if let Some(ValueCell::Array(ref arr)) = value_store.get(array_id) {
        let len = if value_store.is_flat_heap(array_id) {
            arr.len() / 2
        } else {
            arr.len()
        };
        let result_id = value_store.allocate(ValueCell::Number(len as f64));
        stack::push_id(stack, result_id);
        return Ok(VMStatus::Continue);
    }
    if let Some(ValueCell::ArrayView { length, .. }) = value_store.get(array_id) {
        let result_id = value_store.allocate(ValueCell::Number(*length as f64));
        stack::push_id(stack, result_id);
        return Ok(VMStatus::Continue);
    }
    let array = load_value(array_id, value_store, heavy_store);
    match array {
        Value::Array(arr) => {
            stack::push_id(
                stack,
                store_value(
                    Value::Number(arr.borrow().len() as f64),
                    value_store,
                    heavy_store,
                ),
            );
        }
        Value::ArrayView(av) => {
            stack::push_id(
                stack,
                store_value(Value::Number(av.length as f64), value_store, heavy_store),
            );
        }
        Value::ColumnReference { table, column_name } => {
            let t = table.borrow();
            if let Some(len) = crate::vm::table_ops::column_len(&*t, &column_name) {
                stack::push_id(
                    stack,
                    store_value(Value::Number(len as f64), value_store, heavy_store),
                );
            } else {
                let error = ExceptionHandler::runtime_error(
                    &frames,
                    format!("Column '{}' not found", column_name),
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
        }
        Value::ColumnsReference { table, .. } => {
            stack::push_id(
                stack,
                store_value(
                    Value::Number(table.borrow().len() as f64),
                    value_store,
                    heavy_store,
                ),
            );
        }
        Value::PluginOpaque { .. } => {
            if let Some(v) = plugin_opaque_len_via_plugin_call(&array) {
                stack::push_id(stack, store_value(v, value_store, heavy_store));
            } else {
                stack::push_id(stack, store_value(Value::Null, value_store, heavy_store));
            }
        }
        Value::Enumerate { data, .. } => {
            stack::push_id(
                stack,
                store_value(
                    Value::Number(data.borrow().len() as f64),
                    value_store,
                    heavy_store,
                ),
            );
        }
        Value::ByteBuffer(b) => {
            stack::push_id(
                stack,
                store_value(Value::Number(b.len as f64), value_store, heavy_store),
            );
        }
        Value::Tuple(tuple) => {
            stack::push_id(
                stack,
                store_value(
                    Value::Number(tuple.borrow().len() as f64),
                    value_store,
                    heavy_store,
                ),
            );
        }
        _ => {
            let got_type = crate::vm::calls::get_type_name_value(&array);
            let error = ExceptionHandler::runtime_error(
                &frames,
                format!("Expected array, column reference, dataset, enumerate, or tuple for GetArrayLength, got {}", got_type),
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
    }
    Ok(VMStatus::Continue)
}

pub(crate) fn op_table_filter(
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
    let op_str = match &op_val {
        Value::String(s) => s.as_str(),
        _ => {
            let error = ExceptionHandler::runtime_error(
                &frames,
                "Table filter operator must be a string".to_string(),
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
    if let Value::Table(table_rc) = &table_val {
        let _ctx = RestoreVmCallContextGuard::push(vm_ptr);
        let result = crate::vm::natives::table::table_where_impl(
            table_rc,
            column_str,
            op_str,
            &filter_value,
        );
        let result_id = store_value(result, value_store, heavy_store);
        stack::push_id(stack, result_id);
        if let Some(sp) = stack::active_sp_for(stack) {
            let stack_start = frames.last().map(|f| f.stack_start).unwrap_or(0);
            stack::compact_after_native_result(stack, sp, stack_start);
        }
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
        match ExceptionHandler::handle_exception(
            stack,
            frames,
            exception_handlers,
            error,
            value_store,
            heavy_store,
        ) {
            Ok(()) => {}
            Err(e) => return Err(e),
        }
    }
    Ok(VMStatus::Continue)
}

pub(crate) fn op_table_filter_pred(
    pred_index: usize,
    line: usize,
    stack: &mut Vec<crate::common::TaggedValue>,
    frames: &mut Vec<CallFrame>,
    exception_handlers: &mut Vec<ExceptionHandler>,
    value_store: &mut ValueStore,
    heavy_store: &mut HeavyStore,
    vm_ptr: *mut crate::vm::vm::Vm,
) -> Result<VMStatus, LangError> {
    let values_tv = stack::pop(stack, frames, exception_handlers, value_store, heavy_store)?;
    let table_tv = stack::pop(stack, frames, exception_handlers, value_store, heavy_store)?;
    let values_id = tagged_to_value_id(values_tv, value_store);
    let table_id = tagged_to_value_id(table_tv, value_store);
    let table_val = load_value(table_id, value_store, heavy_store);
    let values_val = load_value(values_id, value_store, heavy_store);

    let pred = frames
        .last()
        .and_then(|f| f.function.chunk.constants.get(pred_index))
        .cloned()
        .unwrap_or(Value::Null);

    let values_vec: Vec<Value> = match &values_val {
        Value::Array(arr) => arr.borrow().clone(),
        _ => Vec::new(),
    };

    if let Value::Table(table_rc) = &table_val {
        let _ctx = RestoreVmCallContextGuard::push(vm_ptr);
        let result =
            crate::vm::table_filter_pred::table_filter_pred_impl(table_rc, &pred, &values_vec);
        let result_id = store_value(result, value_store, heavy_store);
        stack::push_id(stack, result_id);
        if let Some(sp) = stack::active_sp_for(stack) {
            let stack_start = frames.last().map(|f| f.stack_start).unwrap_or(0);
            stack::compact_after_native_result(stack, sp, stack_start);
        }
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
    if crate::vm::special_methods::is_class_instance(&value) {
        if let Ok(Some(cloned)) =
            crate::vm::special_methods::try_dispatch_unary_special(&value, "@clone")
        {
            stack::push_id(stack, store_value(cloned, value_store, heavy_store));
            return Ok(VMStatus::Continue);
        }
    }
    let cloned = match crate::vm::deep_copy::deep_copy(&value, value_store, heavy_store) {
        Ok(v) => v,
        Err(msg) => {
            let line = frames.last().map(|f| f.function.chunk.get_line(f.ip)).unwrap_or(0);
            let error = ExceptionHandler::runtime_error(&frames, msg, line);
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
                        "Array size must be a number".to_string(),
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
            }
        }
    };
    let mut slots = Vec::with_capacity(count);
    for _ in 0..count {
        slots.push(stack::pop(
            stack,
            frames,
            exception_handlers,
            value_store,
            heavy_store,
        )?);
    }
    slots.reverse();
    let result_id = value_store.allocate_ephemeral_arena(ValueCell::Array(slots));
    stack::push_id(stack, result_id);
    Ok(VMStatus::Continue)
}
