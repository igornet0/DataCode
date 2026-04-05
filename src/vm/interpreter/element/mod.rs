//! GetArrayElement and SetArrayElement opcodes (large bodies, kept separate).

mod array_ops;
mod indexing;
mod indexing_lib;
mod object_fields;
mod table_ops;

use crate::common::{error::LangError, value::{IterableInner, Value}, value_store::{ValueCell, ValueStore, NULL_VALUE_ID}, TaggedValue};
use crate::debug_println;
use crate::vm::exceptions::ExceptionHandler;
use crate::vm::frame::CallFrame;
use crate::vm::heavy_store::HeavyStore;
use crate::vm::stack;
use crate::vm::store_convert::{load_value, store_value, tagged_to_value_id};
use crate::vm::types::VMStatus;

use super::helpers::pop_to_value_id;
use crate::vm::array_view::resolve_slice_origin;
use crate::vm::iterable::{chunk_source_count, materialize_chunk_at};

#[allow(clippy::too_many_arguments)]
pub fn op_get_array_element(
    line: usize,
    stack: &mut Vec<TaggedValue>,
    frames: &mut Vec<CallFrame>,
    globals: &mut Vec<crate::vm::global_slot::GlobalSlot>,
    global_names: &std::collections::BTreeMap<usize, String>,
    _functions: &[crate::bytecode::Function],
    natives: &[crate::vm::host::HostEntry],
    exception_handlers: &mut Vec<ExceptionHandler>,
    value_store: &mut ValueStore,
    heavy_store: &mut HeavyStore,
    vm_ptr: *mut crate::vm::vm::Vm,
) -> Result<VMStatus, LangError> {
    let index_tv = stack::pop(stack, frames, exception_handlers, value_store, heavy_store)?;
    let container_tv = stack::pop(stack, frames, exception_handlers, value_store, heavy_store)?;
    let current_ip = {
        let frame = frames.last().unwrap();
        frame.ip - 1
    };
    {
        let frame = frames.last_mut().unwrap();
        let cache_array = frame.get_array_element_cache_ip == Some(current_ip) && frame.get_array_element_cache_array_number;
        let cache_obj = frame.get_array_element_cache_ip == Some(current_ip) && frame.get_array_element_cache_object_string;
        if cache_array && container_tv.is_heap() && index_tv.is_number() {
            let container_id = container_tv.get_heap_id();
            let idx = index_tv.get_f64() as i64;
            if idx >= 0 {
                if let Some(ValueCell::Array(ref vec)) = value_store.get(container_id) {
                    let u = idx as usize;
                    if u < vec.len() {
                        stack::push(stack, vec[u]);
                        frame.get_array_element_cache_ip = Some(current_ip);
                        frame.get_array_element_cache_array_number = true;
                        frame.get_array_element_cache_object_string = false;
                        return Ok(VMStatus::Continue);
                    }
                }
            }
            frame.get_array_element_cache_array_number = false;
        }
        if cache_obj && container_tv.is_heap() && index_tv.is_heap() {
            let container_id = container_tv.get_heap_id();
            let index_value_id = index_tv.get_heap_id();
            if let (Some(ValueCell::Object(ref map)), Some(ValueCell::String(key_id))) =
                (value_store.get(container_id), value_store.get(index_value_id))
            {
                if !map.contains_key("__class_name") {
                    if let Some(key_str) = value_store.get_string(*key_id) {
                        let element_id = map.get(key_str).copied().unwrap_or(NULL_VALUE_ID);
                        stack::push_id(stack, element_id);
                        frame.get_array_element_cache_ip = Some(current_ip);
                        frame.get_array_element_cache_array_number = false;
                        frame.get_array_element_cache_object_string = true;
                        return Ok(VMStatus::Continue);
                    }
                }
            }
            frame.get_array_element_cache_object_string = false;
        }
    }
    let index_value_id = tagged_to_value_id(index_tv, value_store);
    let container_id = tagged_to_value_id(container_tv, value_store);
    if let (Some(ValueCell::Array(ref vec)), Some(ValueCell::Number(n))) =
        (value_store.get(container_id), value_store.get(index_value_id))
    {
        let idx = *n as i64;
        if idx >= 0 {
            let u = idx as usize;
            if u < vec.len() {
                stack::push(stack, vec[u]);
                let frame = frames.last_mut().unwrap();
                frame.get_array_element_cache_ip = Some(current_ip);
                frame.get_array_element_cache_array_number = true;
                frame.get_array_element_cache_object_string = false;
                return Ok(VMStatus::Continue);
            }
        }
    }
    // Fast path: ValueCell::Object + ValueCell::String key — no load_value for container/index.
    // Skip fast path for class instances (they need private/protected checks).
    if let (Some(ValueCell::Object(ref map)), Some(ValueCell::String(key_id))) =
        (value_store.get(container_id), value_store.get(index_value_id))
    {
        if !map.contains_key("__class_name") {
            if let Some(key_str) = value_store.get_string(*key_id) {
                let element_id = map.get(key_str).copied().unwrap_or(NULL_VALUE_ID);
                stack::push_id(stack, element_id);
                let frame = frames.last_mut().unwrap();
                frame.get_array_element_cache_ip = Some(current_ip);
                frame.get_array_element_cache_array_number = false;
                frame.get_array_element_cache_object_string = true;
                return Ok(VMStatus::Continue);
            }
        }
    }
    let frame = frames.last_mut().unwrap();
    frame.get_array_element_cache_array_number = false;
    frame.get_array_element_cache_object_string = false;
    let index_value = load_value(index_value_id, value_store, heavy_store);
    let container = load_value(container_id, value_store, heavy_store);
    let container_type = match &container {
        Value::Array(_) => "Array",
        Value::ArrayView(_) => "ArrayView",
        Value::Enumerate { .. } => "Enumerate",
        Value::Iterable(_) => "Iterable",
        Value::Generator(_) => "Generator",
        Value::Object(_) => "Object",
        Value::Table(_) => "Table",
        Value::Path(_) => "Path",
        Value::Uuid(_, _) => "UUID",
        _ => "Other",
    };
    let key_str = match &index_value {
        Value::String(k) => k.clone(),
        Value::Number(n) => format!("{}", n),
        _ => format!("{:?}", index_value),
    };
    debug_println!("[DEBUG GetArrayElement] line {} IP {}: {} key '{}'", line, current_ip, container_type, key_str);
    
    match container {
        Value::Array(arr) => {
            return array_ops::get_array(
                line, stack, frames, exception_handlers, value_store, heavy_store,
                container_id, arr, index_value,
            );
        }
        Value::ArrayView(av) => {
            return array_ops::get_array_view(
                line, stack, frames, exception_handlers, value_store, heavy_store,
                container_id, &av, index_value,
            );
        }
        Value::Tuple(tuple) => {
            return array_ops::get_tuple(
                line, stack, frames, exception_handlers, value_store, heavy_store,
                tuple, index_value,
            );
        }
        Value::Enumerate { data, start } => {
            return array_ops::get_enumerate(
                line, stack, frames, exception_handlers, value_store, heavy_store,
                data, start, index_value,
            );
        }
        Value::ByteBuffer(bb) => {
            if let Value::String(key) = &index_value {
                if key.as_str() == "chunk" {
                    const NATIVE_CHUNK: usize = 80;
                    stack::push_id(
                        stack,
                        store_value(Value::NativeFunction(NATIVE_CHUNK), value_store, heavy_store),
                    );
                    return Ok(VMStatus::Continue);
                }
                let error = ExceptionHandler::runtime_error(
                    &frames,
                    format!(
                        "ByteBuffer has no property '{}'. Available: chunk, or numeric index",
                        key
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
            }
            if let Value::Number(n) = &index_value {
                if n.fract() == 0.0 && *n >= 0.0 {
                    let idx = *n as usize;
                    if idx < bb.len {
                        let byte = bb.bytes[bb.offset + idx];
                        stack::push_id(
                            stack,
                            store_value(Value::Number(byte as f64), value_store, heavy_store),
                        );
                        return Ok(VMStatus::Continue);
                    }
                }
            }
            let error = ExceptionHandler::runtime_error(
                &frames,
                "ByteBuffer index out of range or invalid".to_string(),
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
        Value::Generator(rc) => {
            if let Value::String(key) = &index_value {
                if key.as_str() == "live" {
                    let live = !rc.borrow().finished;
                    stack::push_id(
                        stack,
                        store_value(Value::Bool(live), value_store, heavy_store),
                    );
                    return Ok(VMStatus::Continue);
                }
                if key.as_str() == "final" {
                    const GENERATOR_FINAL_NATIVE_INDEX: usize = 81;
                    stack::push_id(
                        stack,
                        store_value(Value::NativeFunction(GENERATOR_FINAL_NATIVE_INDEX), value_store, heavy_store),
                    );
                    return Ok(VMStatus::Continue);
                }
                if key.as_str() == "next" {
                    const GENERATOR_NEXT_NATIVE_INDEX: usize = 82;
                    stack::push_id(
                        stack,
                        store_value(Value::NativeFunction(GENERATOR_NEXT_NATIVE_INDEX), value_store, heavy_store),
                    );
                    return Ok(VMStatus::Continue);
                }
                if key.as_str() == "send" {
                    const GENERATOR_SEND_NATIVE_INDEX: usize = 83;
                    stack::push_id(
                        stack,
                        store_value(Value::NativeFunction(GENERATOR_SEND_NATIVE_INDEX), value_store, heavy_store),
                    );
                    return Ok(VMStatus::Continue);
                }
            }
            let error = ExceptionHandler::runtime_error(
                &frames,
                "Generator supports .live, .final(), .next(), .send() (string keys)".to_string(),
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
        Value::Iterable(rc) => {
            let chunk_info = {
                let inner = rc.borrow();
                if let IterableInner::Chunks {
                    source,
                    chunk_size,
                    ..
                } = &*inner
                {
                    if let Value::Number(n) = &index_value {
                        if n.fract() == 0.0 && *n >= 0.0 {
                            let k = *n as usize;
                            if k < chunk_source_count(source, *chunk_size) {
                                Some((source.clone(), *chunk_size, k))
                            } else {
                                None
                            }
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                } else {
                    None
                }
            };
            if let Some((source, chunk_size, k)) = chunk_info {
                let vm = unsafe { &mut *vm_ptr };
                match materialize_chunk_at(vm, &source, chunk_size, k) {
                    Ok(v) => {
                        stack::push_id(stack, store_value(v, value_store, heavy_store));
                        return Ok(VMStatus::Continue);
                    }
                    Err(e) => {
                        let error = ExceptionHandler::runtime_error(&frames, e.to_string(), line);
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
            let error = ExceptionHandler::runtime_error(
                &frames,
                "GetArrayElement: only chunk(...) iterables support numeric indexing (chunk index)".to_string(),
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
        Value::Table(table) => {
            return table_ops::get_table(
                line, stack, frames, exception_handlers, value_store, heavy_store,
                table, index_value,
            );
        }
        Value::Object(map_rc) => {
            return object_fields::get_object(
                line, stack, frames, globals, global_names, exception_handlers,
                value_store, heavy_store, vm_ptr, container_tv, map_rc, index_value,
            );
        }
        Value::Figure(figure_rc) => {
            return indexing_lib::get_figure(
                line, stack, frames, exception_handlers, value_store, heavy_store,
                figure_rc, index_value,
            );
        }
        Value::Axis(_axis_rc) => {
            return indexing_lib::get_axis(
                line, stack, frames, globals, exception_handlers, value_store, heavy_store,
                index_value,
            );
        }
        Value::ColumnReference { table, column_name } => {
            return indexing::get_column_reference(
                line, stack, frames, exception_handlers, value_store, heavy_store,
                table, column_name, index_value,
            );
        }
        Value::Path(path) => {
            return indexing::get_path(
                line, stack, frames, exception_handlers, value_store, heavy_store,
                path, index_value,
            );
        }
        Value::PluginOpaque { .. } => {
            let Some(native_idx) = (unsafe { (*vm_ptr).plugin_call_native }) else {
                let error = ExceptionHandler::runtime_error(
                    &frames,
                    "GetArrayElement on plugin opaque values requires native_plugin_call (import a native module that exports it)".to_string(),
                    line,
                );
                return match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error, value_store, heavy_store) {
                    Ok(()) => Ok(VMStatus::Continue),
                    Err(e) => Err(e),
                };
            };
            let builtin_count = natives.len();
            let abi_slice = unsafe { (*vm_ptr).get_abi_natives() };
            if native_idx < builtin_count || native_idx >= builtin_count + abi_slice.len() {
                let error = ExceptionHandler::runtime_error(
                    &frames,
                    "native_plugin_call index is invalid (reload native module)".to_string(),
                    line,
                );
                return match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error, value_store, heavy_store) {
                    Ok(()) => Ok(VMStatus::Continue),
                    Err(e) => Err(e),
                };
            }
            let args = [container.clone(), index_value.clone()];
            let result = crate::vm::native_loader::call_abi_native(
                abi_slice[native_idx - builtin_count],
                &args,
                Some((value_store, heavy_store)),
            );
            if let Some(abi_err) = crate::vm::native_loader::take_last_abi_error() {
                return match ExceptionHandler::handle_exception(stack, frames, exception_handlers, abi_err, value_store, heavy_store) {
                    Ok(()) => Ok(VMStatus::Continue),
                    Err(e) => Err(e),
                };
            }
            stack::push_id(stack, store_value(result, value_store, heavy_store));
            return Ok(VMStatus::Continue);
        }
        Value::DatabaseEngine(_engine_rc) => {
            return indexing_lib::get_database_engine(
                line, stack, frames, exception_handlers, value_store, heavy_store,
                natives, index_value,
            );
        }
        Value::DatabaseCluster(cluster_rc) => {
            return indexing_lib::get_database_cluster(
                line, stack, frames, exception_handlers, value_store, heavy_store,
                natives, cluster_rc, index_value,
            );
        }
        Value::String(s) => {
            return indexing::get_string(
                line, stack, frames, exception_handlers, value_store, heavy_store,
                s, index_value,
            );
        }
        Value::NativeFunction(native_index) => {
            use crate::vm::natives::basic::native_str;
            const STR_NATIVE_INDEX: usize = 6;
            if native_index < natives.len()
                && (natives[native_index].as_fn_ptr() == Some(native_str as *const ()) || native_index == STR_NATIVE_INDEX)
            {
                return indexing::get_native_str(
                    line, stack, frames, exception_handlers, value_store, heavy_store,
                    index_value,
                );
            }
            let error = ExceptionHandler::runtime_error(
                &frames,
                "Expected array, tuple, column reference, table, object, path, database engine, or database cluster for GetArrayElement".to_string(),
                line,
            );
            match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error, value_store, heavy_store) {
                Ok(()) => return Ok(VMStatus::Continue),
                Err(e) => return Err(e),
            }
        }
        Value::Null => {
            let error = ExceptionHandler::runtime_error(
            &frames,
                "Cannot access element of null value".to_string(),
                line,
            );
            match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error, value_store, heavy_store) {
                Ok(()) => return Ok(VMStatus::Continue),
                Err(e) => return Err(e),
            }
        }
        _ => {
            let error = ExceptionHandler::runtime_error(
            &frames,
                "Expected array, tuple, column reference, table, object, path, database engine, or database cluster for GetArrayElement".to_string(),
                line,
            );
            match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error, value_store, heavy_store) {
                Ok(()) => return Ok(VMStatus::Continue),
                Err(e) => return Err(e),
            }
        }
    }
}

fn slice_bound_from_tagged(
    tv: TaggedValue,
    line: usize,
    value_store: &mut ValueStore,
    heavy_store: &mut HeavyStore,
) -> Result<Option<i64>, LangError> {
    let v = load_value(tagged_to_value_id(tv, value_store), value_store, heavy_store);
    match v {
        Value::Null => Ok(None),
        Value::Number(n) => {
            if n.fract() != 0.0 && (n - n.round()).abs() > 1e-9 {
                return Err(LangError::runtime_error(
                    "Slice bound must be an integer or null".to_string(),
                    line,
                ));
            }
            Ok(Some(n as i64))
        }
        _ => Err(LangError::runtime_error(
            "Slice bound must be a number or null".to_string(),
            line,
        )),
    }
}

#[allow(clippy::too_many_arguments)]
pub fn op_get_array_slice(
    line: usize,
    stack: &mut Vec<TaggedValue>,
    frames: &mut Vec<CallFrame>,
    exception_handlers: &mut Vec<ExceptionHandler>,
    value_store: &mut ValueStore,
    heavy_store: &mut HeavyStore,
) -> Result<VMStatus, LangError> {
    let step_tv = stack::pop(stack, frames, exception_handlers, value_store, heavy_store)?;
    let stop_tv = stack::pop(stack, frames, exception_handlers, value_store, heavy_store)?;
    let start_tv = stack::pop(stack, frames, exception_handlers, value_store, heavy_store)?;
    let container_id = pop_to_value_id(stack, frames, exception_handlers, value_store, heavy_store)?;
    let start = slice_bound_from_tagged(start_tv, line, value_store, heavy_store)?;
    let stop = slice_bound_from_tagged(stop_tv, line, value_store, heavy_store)?;
    let step = slice_bound_from_tagged(step_tv, line, value_store, heavy_store)?;
    let container_v = load_value(container_id, value_store, heavy_store);
    if let Value::ByteBuffer(ref bb) = container_v {
        match array_ops::byte_buffer_slice_value(bb, start, stop, step) {
            Ok(v) => {
                stack::push_id(stack, store_value(v, value_store, heavy_store));
                return Ok(VMStatus::Continue);
            }
            Err(msg) => {
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
        }
    }
    if resolve_slice_origin(container_id, value_store, heavy_store).is_some() {
        return array_ops::get_array_slice_from_container(
            line,
            stack,
            frames,
            exception_handlers,
            value_store,
            heavy_store,
            container_id,
            start,
            stop,
            step,
        );
    }
    let error = ExceptionHandler::runtime_error(frames, "GetArraySlice requires an array".to_string(), line);
    match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error, value_store, heavy_store) {
        Ok(()) => Ok(VMStatus::Continue),
        Err(e) => Err(e),
    }
}

#[allow(clippy::too_many_arguments)]
pub fn op_set_array_slice(
    line: usize,
    stack: &mut Vec<TaggedValue>,
    frames: &mut Vec<CallFrame>,
    exception_handlers: &mut Vec<ExceptionHandler>,
    value_store: &mut ValueStore,
    heavy_store: &mut HeavyStore,
) -> Result<VMStatus, LangError> {
    let container_id = pop_to_value_id(stack, frames, exception_handlers, value_store, heavy_store)?;
    let step_tv = stack::pop(stack, frames, exception_handlers, value_store, heavy_store)?;
    let stop_tv = stack::pop(stack, frames, exception_handlers, value_store, heavy_store)?;
    let start_tv = stack::pop(stack, frames, exception_handlers, value_store, heavy_store)?;
    let value_tv = stack::pop(stack, frames, exception_handlers, value_store, heavy_store)?;
    let value_id = tagged_to_value_id(value_tv, value_store);
    let value = load_value(value_id, value_store, heavy_store);
    let start = slice_bound_from_tagged(start_tv, line, value_store, heavy_store)?;
    let stop = slice_bound_from_tagged(stop_tv, line, value_store, heavy_store)?;
    let step = slice_bound_from_tagged(step_tv, line, value_store, heavy_store)?;
    array_ops::set_array_slice_splice(
        line,
        stack,
        frames,
        exception_handlers,
        value_store,
        heavy_store,
        container_id,
        start,
        stop,
        step,
        value,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn op_set_array_element(
    line: usize,
    stack: &mut Vec<TaggedValue>,
    frames: &mut Vec<CallFrame>,
    globals: &mut Vec<crate::vm::global_slot::GlobalSlot>,
    global_names: &std::collections::BTreeMap<usize, String>,
    functions: &[crate::bytecode::Function],
    _natives: &[crate::vm::host::HostEntry],
    exception_handlers: &mut Vec<ExceptionHandler>,
    value_store: &mut ValueStore,
    heavy_store: &mut HeavyStore,
) -> Result<VMStatus, LangError> {
    // Stack order from compiler: [value, index, container] with container on top.
    let container_id = pop_to_value_id(stack, frames, exception_handlers, value_store, heavy_store)?;
    let index_tv = stack::pop(stack, frames, exception_handlers, value_store, heavy_store)?;
    let value_tv = stack::pop(stack, frames, exception_handlers, value_store, heavy_store)?;
    let value_id = tagged_to_value_id(value_tv, value_store);
    // Fast path: ValueCell::Array + number index — store TaggedValue slot
    if index_tv.is_number() {
        let idx = index_tv.get_f64() as i64;
        if idx >= 0 && index_tv.get_f64().fract() == 0.0 {
            let u = idx as usize;
            if let Some(ValueCell::Array(slots)) = value_store.get_mut(container_id) {
                if u >= slots.len() {
                    slots.resize(u + 1, TaggedValue::null());
                }
                slots[u] = value_tv;
                stack::push_id(stack, container_id);
                return Ok(VMStatus::Continue);
            }
        }
    }
    let container = load_value(container_id, value_store, heavy_store);
    let index_value = load_value(tagged_to_value_id(index_tv, value_store), value_store, heavy_store);
    let value = load_value(value_id, value_store, heavy_store);
    let container_type = match &container {
        Value::Array(_) => "Array",
        Value::Object(_) => "Object",
        Value::Table(_) => "Table",
        _ => "Other",
    };
    let key_str = match &index_value {
        Value::String(k) => k.clone(),
        Value::Number(n) => format!("{}", n),
        _ => format!("{:?}", index_value),
    };
    let value_type_str = match &value {
        Value::Function(fn_idx) => {
            if *fn_idx < functions.len() {
                format!("Function({}, имя: '{}')", fn_idx, functions[*fn_idx].name)
            } else {
                format!("Function({}, OUT OF BOUNDS!)", fn_idx)
            }
        },
        Value::NativeFunction(_) => "NativeFunction".to_string(),
        _ => format!("{:?}", value),
    };
    debug_println!("[DEBUG SetArrayElement] line {}, {} key='{}' value={}", line, container_type, key_str, value_type_str);
    match container {
        Value::Array(_) => {
            return array_ops::set_array(
                line, stack, frames, exception_handlers, value_store, heavy_store,
                container_id, index_value, value,
            );
        }
        Value::Object(obj_rc) => {
            let key = match index_value {
                Value::String(key) => key,
                _ => {
                    let error = ExceptionHandler::runtime_error(
                        &frames,
                        "Object key must be a string".to_string(),
                        line,
                    );
                    match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error, value_store, heavy_store) {
                        Ok(()) => return Ok(VMStatus::Continue),
                        Err(e) => return Err(e),
                    }
                }
            };
            return object_fields::set_object(
                line, stack, frames, globals, global_names, exception_handlers,
                value_store, heavy_store, container_id, obj_rc, key, value_id, value,
            );
        }
        _ => {
            let error = ExceptionHandler::runtime_error(
                &frames,
                format!("SetArrayElement only supports arrays and objects, got: {:?}", container),
                line,
            );
            match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error, value_store, heavy_store) {
                Ok(()) => return Ok(VMStatus::Continue),
                Err(e) => return Err(e),
            }
        }
    }
}
