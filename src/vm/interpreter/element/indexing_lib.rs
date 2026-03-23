//! GetArrayElement for Figure, Axis, DatabaseEngine, DatabaseCluster.
//! (Tensor/Dataset/NN indexing lives in the ML dylib; use `import ml`.)

use std::cell::RefCell;
use std::rc::Rc;

use crate::common::{error::LangError, error::ErrorType, value::Value, value_store::ValueStore};
use crate::vm::exceptions::ExceptionHandler;
use crate::vm::frame::CallFrame;
use crate::vm::heavy_store::HeavyStore;
use crate::vm::stack;
use crate::vm::store_convert::{load_value, store_value};
use crate::vm::types::VMStatus;

/// Get Figure property (axes).
#[allow(clippy::too_many_arguments)]
pub fn get_figure(
    line: usize,
    stack: &mut Vec<crate::common::TaggedValue>,
    frames: &mut Vec<CallFrame>,
    exception_handlers: &mut Vec<ExceptionHandler>,
    value_store: &mut ValueStore,
    heavy_store: &mut HeavyStore,
    figure_rc: Rc<RefCell<crate::plot::Figure>>,
    index_value: Value,
) -> Result<VMStatus, LangError> {
    match index_value {
        Value::String(key) => {
            match key.as_str() {
                "axes" => {
                    let figure_ref = figure_rc.borrow();
                    let mut axes_array = Vec::new();
                    for row in &figure_ref.axes {
                        let mut row_array = Vec::new();
                        for axis in row {
                            row_array.push(Value::Axis(axis.clone()));
                        }
                        axes_array.push(Value::Array(Rc::new(RefCell::new(row_array))));
                    }
                    stack::push_id(
                        stack,
                        store_value(
                            Value::Array(Rc::new(RefCell::new(axes_array))),
                            value_store,
                            heavy_store,
                        ),
                    );
                }
                _ => {
                    let error = ExceptionHandler::runtime_error_with_type(
                        &frames,
                        format!("Figure has no property '{}'", key),
                        line,
                        ErrorType::KeyError,
                    );
                    return match ExceptionHandler::handle_exception(
                        stack, frames, exception_handlers, error, value_store, heavy_store,
                    ) {
                        Ok(()) => Ok(VMStatus::Continue),
                        Err(e) => Err(e),
                    };
                }
            }
        }
        _ => {
            let error = ExceptionHandler::runtime_error(
                &frames,
                "Figure property access must use string key".to_string(),
                line,
            );
            return match ExceptionHandler::handle_exception(
                stack, frames, exception_handlers, error, value_store, heavy_store,
            ) {
                Ok(()) => Ok(VMStatus::Continue),
                Err(e) => Err(e),
            };
        }
    }
    Ok(VMStatus::Continue)
}

/// Get Axis method (imshow, set_title, axis).
#[allow(clippy::too_many_arguments)]
pub fn get_axis(
    line: usize,
    stack: &mut Vec<crate::common::TaggedValue>,
    frames: &mut Vec<CallFrame>,
    globals: &mut Vec<crate::vm::global_slot::GlobalSlot>,
    exception_handlers: &mut Vec<ExceptionHandler>,
    value_store: &mut ValueStore,
    heavy_store: &mut HeavyStore,
    index_value: Value,
) -> Result<VMStatus, LangError> {
    match index_value {
        Value::String(key) => {
            let method_name = match key.as_str() {
                "imshow" => "imshow",
                "set_title" => "set_title",
                "axis" => "axis",
                _ => {
                    let error = ExceptionHandler::runtime_error_with_type(
                        &frames,
                        format!("Axis has no method '{}'", key),
                        line,
                        ErrorType::KeyError,
                    );
                    return match ExceptionHandler::handle_exception(
                        stack, frames, exception_handlers, error, value_store, heavy_store,
                    ) {
                        Ok(()) => Ok(VMStatus::Continue),
                        Err(e) => Err(e),
                    };
                }
            };
            let method_index = if let Some((_plot_id, plot_val)) = globals.iter_mut().find_map(|slot| {
                let plot_id = slot.resolve_to_value_id(value_store);
                let plot_val = load_value(plot_id, value_store, heavy_store);
                if let Value::Object(map_rc) = &plot_val {
                    if map_rc.borrow().contains_key("image") {
                        return Some((plot_id, plot_val));
                    }
                }
                None
            }) {
                if let Value::Object(map_rc) = &plot_val {
                    let map = map_rc.borrow();
                    let idx_key = match method_name {
                        "imshow" => "__axis_imshow_idx",
                        "set_title" => "__axis_set_title_idx",
                        "axis" => "__axis_axis_idx",
                        _ => unreachable!(),
                    };
                    if let Some(Value::Number(idx)) = map.get(idx_key) {
                        *idx as usize
                    } else {
                        let error = ExceptionHandler::runtime_error(
                            &frames,
                            format!("Axis method '{}' not registered", key),
                            line,
                        );
                        return match ExceptionHandler::handle_exception(
                            stack, frames, exception_handlers, error, value_store, heavy_store,
                        ) {
                            Ok(()) => Ok(VMStatus::Continue),
                            Err(e) => Err(e),
                        };
                    }
                } else {
                    let error = ExceptionHandler::runtime_error(
                        &frames,
                        "Plot object not found".to_string(),
                        line,
                    );
                    return match ExceptionHandler::handle_exception(
                        stack, frames, exception_handlers, error, value_store, heavy_store,
                    ) {
                        Ok(()) => Ok(VMStatus::Continue),
                        Err(e) => Err(e),
                    };
                }
            } else {
                let error = ExceptionHandler::runtime_error(
                    &frames,
                    "Plot module not found".to_string(),
                    line,
                );
                return match ExceptionHandler::handle_exception(
                    stack, frames, exception_handlers, error, value_store, heavy_store,
                ) {
                    Ok(()) => Ok(VMStatus::Continue),
                    Err(e) => Err(e),
                };
            };
            stack::push_id(stack, store_value(Value::NativeFunction(method_index), value_store, heavy_store));
        }
        _ => {
            let error = ExceptionHandler::runtime_error(
                &frames,
                "Axis property access must use string key".to_string(),
                line,
            );
            return match ExceptionHandler::handle_exception(
                stack, frames, exception_handlers, error, value_store, heavy_store,
            ) {
                Ok(()) => Ok(VMStatus::Continue),
                Err(e) => Err(e),
            };
        }
    }
    Ok(VMStatus::Continue)
}

pub fn get_database_engine(
    line: usize,
    stack: &mut Vec<crate::common::TaggedValue>,
    frames: &mut Vec<CallFrame>,
    exception_handlers: &mut Vec<ExceptionHandler>,
    value_store: &mut ValueStore,
    heavy_store: &mut HeavyStore,
    natives: &[crate::vm::host::HostEntry],
    index_value: Value,
) -> Result<VMStatus, LangError> {
    match &index_value {
        Value::String(property_name) => {
            use crate::database_engine::natives as db_natives;
            let (connect_fn, execute_fn, query_fn, run_fn) = (
                db_natives::native_engine_connect as *const (),
                db_natives::native_engine_execute as *const (),
                db_natives::native_engine_query as *const (),
                db_natives::native_engine_run as *const (),
            );
            let method_index = match property_name.as_str() {
                "connect" => natives.iter().position(|e| e.as_fn_ptr() == Some(connect_fn)),
                "execute" => natives.iter().position(|e| e.as_fn_ptr() == Some(execute_fn)),
                "query" => natives.iter().position(|e| e.as_fn_ptr() == Some(query_fn)),
                "run" => natives.iter().position(|e| e.as_fn_ptr() == Some(run_fn)),
                _ => {
                    let error = ExceptionHandler::runtime_error(
                        &frames,
                        format!(
                            "DatabaseEngine has no property '{}'. Available: connect, execute, query, run",
                            property_name
                        ),
                        line,
                    );
                    return match ExceptionHandler::handle_exception(
                        stack, frames, exception_handlers, error, value_store, heavy_store,
                    ) {
                        Ok(()) => Ok(VMStatus::Continue),
                        Err(e) => Err(e),
                    };
                }
            };
            if let Some(idx) = method_index {
                stack::push_id(
                    stack,
                    store_value(Value::NativeFunction(idx), value_store, heavy_store),
                );
            } else {
                let error = ExceptionHandler::runtime_error(
                    &frames,
                    format!("Database engine method '{}' not found", property_name),
                    line,
                );
                return match ExceptionHandler::handle_exception(
                    stack, frames, exception_handlers, error, value_store, heavy_store,
                ) {
                    Ok(()) => Ok(VMStatus::Continue),
                    Err(e) => Err(e),
                };
            }
        }
        _ => {
            let error = ExceptionHandler::runtime_error(
                &frames,
                "DatabaseEngine property access requires string key (connect, execute, query, run)".to_string(),
                line,
            );
            return match ExceptionHandler::handle_exception(
                stack, frames, exception_handlers, error, value_store, heavy_store,
            ) {
                Ok(()) => Ok(VMStatus::Continue),
                Err(e) => Err(e),
            };
        }
    }
    Ok(VMStatus::Continue)
}

/// Get DatabaseCluster method (add, get, names).
#[allow(clippy::too_many_arguments)]
pub fn get_database_cluster(
    line: usize,
    stack: &mut Vec<crate::common::TaggedValue>,
    frames: &mut Vec<CallFrame>,
    exception_handlers: &mut Vec<ExceptionHandler>,
    value_store: &mut ValueStore,
    heavy_store: &mut HeavyStore,
    natives: &[crate::vm::host::HostEntry],
    cluster_rc: Rc<RefCell<crate::database_engine::cluster::DatabaseCluster>>,
    index_value: Value,
) -> Result<VMStatus, LangError> {
    match &index_value {
        Value::String(property_name) => {
            use crate::database_engine::natives as db_natives;
            let (add_fn, get_fn, names_fn) = (
                db_natives::native_cluster_add as *const (),
                db_natives::native_cluster_get as *const (),
                db_natives::native_cluster_names as *const (),
            );
            let method_index = match property_name.as_str() {
                "add" => natives.iter().position(|e| e.as_fn_ptr() == Some(add_fn)),
                "get" => natives.iter().position(|e| e.as_fn_ptr() == Some(get_fn)),
                "names" => natives.iter().position(|e| e.as_fn_ptr() == Some(names_fn)),
                _ => {
                    let error = ExceptionHandler::runtime_error(
                        &frames,
                        format!(
                            "DatabaseCluster has no property '{}'. Available: add, get, names",
                            property_name
                        ),
                        line,
                    );
                    return match ExceptionHandler::handle_exception(
                        stack, frames, exception_handlers, error, value_store, heavy_store,
                    ) {
                        Ok(()) => Ok(VMStatus::Continue),
                        Err(e) => Err(e),
                    };
                }
            };
            if let Some(idx) = method_index {
                stack::push_id(
                    stack,
                    store_value(Value::DatabaseCluster(Rc::clone(&cluster_rc)), value_store, heavy_store),
                );
                stack::push_id(
                    stack,
                    store_value(Value::NativeFunction(idx), value_store, heavy_store),
                );
            } else {
                let error = ExceptionHandler::runtime_error(
                    &frames,
                    format!("Database cluster method '{}' not found", property_name),
                    line,
                );
                return match ExceptionHandler::handle_exception(
                    stack, frames, exception_handlers, error, value_store, heavy_store,
                ) {
                    Ok(()) => Ok(VMStatus::Continue),
                    Err(e) => Err(e),
                };
            }
        }
        _ => {
            let error = ExceptionHandler::runtime_error(
                &frames,
                "DatabaseCluster property access requires string key (add, get, names)".to_string(),
                line,
            );
            return match ExceptionHandler::handle_exception(
                stack, frames, exception_handlers, error, value_store, heavy_store,
            ) {
                Ok(()) => Ok(VMStatus::Continue),
                Err(e) => Err(e),
            };
        }
    }
    Ok(VMStatus::Continue)
}
