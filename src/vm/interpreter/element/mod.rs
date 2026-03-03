//! GetArrayElement and SetArrayElement opcodes (large bodies, kept separate).

mod array_ops;
mod indexing;
mod indexing_lib;
mod object_fields;
mod table_ops;

use crate::common::{error::LangError, value::Value, value_store::{ValueCell, ValueStore, NULL_VALUE_ID}, TaggedValue};
use crate::debug_println;
use crate::vm::exceptions::ExceptionHandler;
use crate::vm::frame::CallFrame;
use crate::vm::heavy_store::HeavyStore;
use crate::vm::stack;
use crate::vm::store_convert::{load_value, tagged_to_value_id};
use crate::vm::types::VMStatus;

use super::helpers::pop_to_value_id;

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
        Value::Enumerate { .. } => "Enumerate",
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
        Value::Layer(_layer_id) => {
            return indexing_lib::get_layer(
                line, stack, frames, globals, global_names, exception_handlers,
                value_store, heavy_store, index_value,
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
        Value::Dataset(dataset) => {
            return indexing_lib::get_dataset(
                line, stack, frames, exception_handlers, value_store, heavy_store,
                dataset, index_value,
            );
        }
        Value::Tensor(tensor) => {
            return indexing_lib::get_tensor(
                line, stack, frames, exception_handlers, value_store, heavy_store,
                natives, tensor, index_value,
            );
        }
        Value::NeuralNetwork(nn_rc) => {
            return indexing_lib::get_neural_network(
                line, stack, frames, globals, global_names, exception_handlers,
                value_store, heavy_store, nn_rc, index_value,
            );
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
                "Expected array, tuple, column reference, table, object, path, dataset, tensor, neural network, database engine, or database cluster for GetArrayElement".to_string(),
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
                "Expected array, tuple, column reference, table, object, path, dataset, tensor, neural network, database engine, or database cluster for GetArrayElement".to_string(),
                line,
            );
            match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error, value_store, heavy_store) {
                Ok(()) => return Ok(VMStatus::Continue),
                Err(e) => return Err(e),
            }
        }
    }
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
