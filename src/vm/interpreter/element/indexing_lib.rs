//! GetArrayElement for Figure, Axis, DatabaseEngine, DatabaseCluster.

use std::cell::RefCell;
use std::rc::Rc;

use crate::common::{error::ErrorType, error::LangError, value::Value, value_store::ValueStore};
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
        Value::String(key) => match key.as_str() {
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
        },
        _ => {
            let error = ExceptionHandler::runtime_error(
                &frames,
                "Figure property access must use string key".to_string(),
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
            let method_index = if let Some((_plot_id, plot_val)) =
                globals.iter_mut().find_map(|slot| {
                    let plot_id = slot.resolve_to_value_id(value_store);
                    let plot_val = load_value(plot_id, value_store, heavy_store);
                    if let Value::Object(map_rc) = &plot_val {
                        if map_rc.borrow().str_key_contains("image") {
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
                    if let Some(Value::Number(idx)) = map.str_key_get(idx_key) {
                        *idx as usize
                    } else {
                        let error = ExceptionHandler::runtime_error(
                            &frames,
                            format!("Axis method '{}' not registered", key),
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
                } else {
                    let error = ExceptionHandler::runtime_error(
                        &frames,
                        "Plot object not found".to_string(),
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
            } else {
                let error = ExceptionHandler::runtime_error(
                    &frames,
                    "Plot module not found".to_string(),
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
            stack::push_id(
                stack,
                store_value(
                    Value::NativeFunction(method_index),
                    value_store,
                    heavy_store,
                ),
            );
        }
        _ => {
            let error = ExceptionHandler::runtime_error(
                &frames,
                "Axis property access must use string key".to_string(),
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
                "connect" => natives
                    .iter()
                    .position(|e| e.as_fn_ptr() == Some(connect_fn)),
                "execute" => natives
                    .iter()
                    .position(|e| e.as_fn_ptr() == Some(execute_fn)),
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
        _ => {
            let error = ExceptionHandler::runtime_error(
                &frames,
                "DatabaseEngine property access requires string key (connect, execute, query, run)"
                    .to_string(),
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
            if let Some(idx) = method_index {
                stack::push_id(
                    stack,
                    store_value(
                        Value::DatabaseCluster(Rc::clone(&cluster_rc)),
                        value_store,
                        heavy_store,
                    ),
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
        _ => {
            let error = ExceptionHandler::runtime_error(
                &frames,
                "DatabaseCluster property access requires string key (add, get, names)".to_string(),
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
    Ok(VMStatus::Continue)
}

/// Get Archive properties and methods.
#[allow(clippy::too_many_arguments)]
pub fn get_archive(
    line: usize,
    stack: &mut Vec<crate::common::TaggedValue>,
    frames: &mut Vec<CallFrame>,
    exception_handlers: &mut Vec<ExceptionHandler>,
    value_store: &mut ValueStore,
    heavy_store: &mut HeavyStore,
    archive_rc: Rc<RefCell<crate::archive::Archive>>,
    index_value: Value,
) -> Result<VMStatus, LangError> {
    use crate::vm::native_indices::builtin;
    match index_value {
        Value::String(key) => {
            let arch = archive_rc.borrow();
            match key.as_str() {
                "path" => {
                    stack::push_id(
                        stack,
                        store_value(
                            Value::Path(arch.path.clone()),
                            value_store,
                            heavy_store,
                        ),
                    );
                }
                "format" => {
                    stack::push_id(
                        stack,
                        store_value(
                            Value::String(arch.format.as_str().to_string()),
                            value_store,
                            heavy_store,
                        ),
                    );
                }
                "files" => {
                    let files = arch.files_as_values();
                    stack::push_id(
                        stack,
                        store_value(
                            Value::Array(Rc::new(RefCell::new(files))),
                            value_store,
                            heavy_store,
                        ),
                    );
                }
                "count" => {
                    stack::push_id(
                        stack,
                        store_value(
                            Value::Number(arch.count as f64),
                            value_store,
                            heavy_store,
                        ),
                    );
                }
                "size" => {
                    stack::push_id(
                        stack,
                        store_value(
                            Value::Number(arch.size as f64),
                            value_store,
                            heavy_store,
                        ),
                    );
                }
                "compressed_size" => {
                    stack::push_id(
                        stack,
                        store_value(
                            Value::Number(arch.compressed_size as f64),
                            value_store,
                            heavy_store,
                        ),
                    );
                }
                "read" => {
                    stack::push_id(
                        stack,
                        store_value(
                            Value::NativeFunction(builtin::ARCHIVE_READ),
                            value_store,
                            heavy_store,
                        ),
                    );
                }
                "read_text" => {
                    stack::push_id(
                        stack,
                        store_value(
                            Value::NativeFunction(builtin::ARCHIVE_READ_TEXT),
                            value_store,
                            heavy_store,
                        ),
                    );
                }
                "extract" => {
                    stack::push_id(
                        stack,
                        store_value(
                            Value::NativeFunction(builtin::ARCHIVE_EXTRACT),
                            value_store,
                            heavy_store,
                        ),
                    );
                }
                "close" => {
                    stack::push_id(
                        stack,
                        store_value(
                            Value::NativeFunction(builtin::ARCHIVE_CLOSE),
                            value_store,
                            heavy_store,
                        ),
                    );
                }
                _ => {
                    let error = ExceptionHandler::runtime_error_with_type(
                        &frames,
                        format!(
                            "Archive has no property '{}'. Available: path, format, files, count, size, compressed_size, read, read_text, extract, close",
                            key
                        ),
                        line,
                        ErrorType::KeyError,
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
        _ => {
            let error = ExceptionHandler::runtime_error(
                &frames,
                "Archive property access must use string key".to_string(),
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
    Ok(VMStatus::Continue)
}

/// Get DataSource properties and methods.
#[allow(clippy::too_many_arguments)]
pub fn get_datasource(
    line: usize,
    stack: &mut Vec<crate::common::TaggedValue>,
    frames: &mut Vec<CallFrame>,
    exception_handlers: &mut Vec<ExceptionHandler>,
    value_store: &mut ValueStore,
    heavy_store: &mut HeavyStore,
    ds_rc: Rc<RefCell<crate::datasource::DataSource>>,
    index_value: Value,
) -> Result<VMStatus, LangError> {
    use crate::vm::native_indices::builtin;
    match index_value {
        Value::String(key) => {
            let ds = ds_rc.borrow();
            match key.as_str() {
                "type" => {
                    stack::push_id(
                        stack,
                        store_value(
                            Value::String(ds.connector_type().to_string()),
                            value_store,
                            heavy_store,
                        ),
                    );
                }
                "name" => {
                    stack::push_id(
                        stack,
                        store_value(
                            Value::String(ds.config.name.clone().unwrap_or_default()),
                            value_store,
                            heavy_store,
                        ),
                    );
                }
                "url" => {
                    stack::push_id(
                        stack,
                        store_value(
                            Value::String(ds.config.url.clone().unwrap_or_default()),
                            value_store,
                            heavy_store,
                        ),
                    );
                }
                "enabled" => {
                    stack::push_id(
                        stack,
                        store_value(Value::Bool(ds.config.enabled), value_store, heavy_store),
                    );
                }
                "request" => {
                    stack::push_id(
                        stack,
                        store_value(
                            Value::NativeFunction(builtin::DATASOURCE_REQUEST),
                            value_store,
                            heavy_store,
                        ),
                    );
                }
                "get_table" => {
                    stack::push_id(
                        stack,
                        store_value(
                            Value::NativeFunction(builtin::DATASOURCE_GET_TABLE),
                            value_store,
                            heavy_store,
                        ),
                    );
                }
                "send_table" => {
                    stack::push_id(
                        stack,
                        store_value(
                            Value::NativeFunction(builtin::DATASOURCE_SEND_TABLE),
                            value_store,
                            heavy_store,
                        ),
                    );
                }
                "connect" => {
                    stack::push_id(
                        stack,
                        store_value(
                            Value::NativeFunction(builtin::DATASOURCE_CONNECT),
                            value_store,
                            heavy_store,
                        ),
                    );
                }
                "disconnect" => {
                    stack::push_id(
                        stack,
                        store_value(
                            Value::NativeFunction(builtin::DATASOURCE_DISCONNECT),
                            value_store,
                            heavy_store,
                        ),
                    );
                }
                "ping" => {
                    stack::push_id(
                        stack,
                        store_value(
                            Value::NativeFunction(builtin::DATASOURCE_PING),
                            value_store,
                            heavy_store,
                        ),
                    );
                }
                "test" => {
                    stack::push_id(
                        stack,
                        store_value(
                            Value::NativeFunction(builtin::DATASOURCE_TEST),
                            value_store,
                            heavy_store,
                        ),
                    );
                }
                "clone" => {
                    stack::push_id(
                        stack,
                        store_value(
                            Value::NativeFunction(builtin::DATASOURCE_CLONE),
                            value_store,
                            heavy_store,
                        ),
                    );
                }
                _ => {
                    let error = ExceptionHandler::runtime_error_with_type(
                        &frames,
                        format!(
                            "DataSource has no property '{}'. Available: type, name, url, enabled, request, get_table, send_table, connect, disconnect, ping, test, clone",
                            key
                        ),
                        line,
                        ErrorType::KeyError,
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
        _ => {
            let error = ExceptionHandler::runtime_error(
                &frames,
                "DataSource property access must use string key".to_string(),
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
    Ok(VMStatus::Continue)
}

/// Get DataSourceResponse properties and methods.
#[allow(clippy::too_many_arguments)]
pub fn get_datasource_response(
    line: usize,
    stack: &mut Vec<crate::common::TaggedValue>,
    frames: &mut Vec<CallFrame>,
    exception_handlers: &mut Vec<ExceptionHandler>,
    value_store: &mut ValueStore,
    heavy_store: &mut HeavyStore,
    resp_rc: Rc<RefCell<crate::datasource::DataSourceResponse>>,
    index_value: Value,
) -> Result<VMStatus, LangError> {
    use crate::datasource::natives::{response_body_value, response_headers_value};
    use crate::vm::native_indices::builtin;
    match index_value {
        Value::String(key) => {
            let resp = resp_rc.borrow();
            match key.as_str() {
                "status" => {
                    stack::push_id(
                        stack,
                        store_value(
                            Value::Number(resp.status as f64),
                            value_store,
                            heavy_store,
                        ),
                    );
                }
                "success" => {
                    stack::push_id(
                        stack,
                        store_value(Value::Bool(resp.success), value_store, heavy_store),
                    );
                }
                "url" => {
                    stack::push_id(
                        stack,
                        store_value(
                            Value::String(resp.url.clone()),
                            value_store,
                            heavy_store,
                        ),
                    );
                }
                "headers" => {
                    stack::push_id(
                        stack,
                        store_value(
                            response_headers_value(&resp.headers),
                            value_store,
                            heavy_store,
                        ),
                    );
                }
                "content_type" => {
                    stack::push_id(
                        stack,
                        store_value(
                            Value::String(resp.content_type.clone().unwrap_or_default()),
                            value_store,
                            heavy_store,
                        ),
                    );
                }
                "content_length" => {
                    let len = resp
                        .content_length
                        .map(|n| Value::Number(n as f64))
                        .unwrap_or(Value::Null);
                    stack::push_id(stack, store_value(len, value_store, heavy_store));
                }
                "encoding" => {
                    stack::push_id(
                        stack,
                        store_value(
                            Value::String(resp.encoding.clone().unwrap_or_default()),
                            value_store,
                            heavy_store,
                        ),
                    );
                }
                "elapsed" => {
                    stack::push_id(
                        stack,
                        store_value(Value::Number(resp.elapsed_ms), value_store, heavy_store),
                    );
                }
                "body" | "bytes" => {
                    stack::push_id(
                        stack,
                        store_value(response_body_value(&resp), value_store, heavy_store),
                    );
                }
                "text" => {
                    let text = resp.text().unwrap_or_default();
                    stack::push_id(
                        stack,
                        store_value(Value::String(text), value_store, heavy_store),
                    );
                }
                "json" => {
                    stack::push_id(
                        stack,
                        store_value(
                            Value::NativeFunction(builtin::RESPONSE_JSON),
                            value_store,
                            heavy_store,
                        ),
                    );
                }
                "table" => {
                    stack::push_id(
                        stack,
                        store_value(
                            Value::NativeFunction(builtin::RESPONSE_TABLE),
                            value_store,
                            heavy_store,
                        ),
                    );
                }
                "csv" => {
                    stack::push_id(
                        stack,
                        store_value(
                            Value::NativeFunction(builtin::RESPONSE_CSV),
                            value_store,
                            heavy_store,
                        ),
                    );
                }
                "save" => {
                    stack::push_id(
                        stack,
                        store_value(
                            Value::NativeFunction(builtin::RESPONSE_SAVE),
                            value_store,
                            heavy_store,
                        ),
                    );
                }
                "save_text" => {
                    stack::push_id(
                        stack,
                        store_value(
                            Value::NativeFunction(builtin::RESPONSE_SAVE_TEXT),
                            value_store,
                            heavy_store,
                        ),
                    );
                }
                "save_json" => {
                    stack::push_id(
                        stack,
                        store_value(
                            Value::NativeFunction(builtin::RESPONSE_SAVE_JSON),
                            value_store,
                            heavy_store,
                        ),
                    );
                }
                _ => {
                    let error = ExceptionHandler::runtime_error_with_type(
                        &frames,
                        format!(
                            "Response has no property '{}'. Available: status, success, url, headers, content_type, content_length, encoding, elapsed, body, text, bytes, json, table, csv, save, save_text, save_json",
                            key
                        ),
                        line,
                        ErrorType::KeyError,
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
        _ => {
            let error = ExceptionHandler::runtime_error(
                &frames,
                "Response property access must use string key".to_string(),
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
    Ok(VMStatus::Continue)
}
