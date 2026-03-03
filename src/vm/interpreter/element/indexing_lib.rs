//! GetArrayElement for Figure, Axis, Layer, Dataset, Tensor, NeuralNetwork,
//! DatabaseEngine, DatabaseCluster.

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use crate::common::{error::LangError, error::ErrorType, value::Value, value_store::ValueStore};
use crate::ml::tensor::Tensor;
use crate::vm::exceptions::ExceptionHandler;
use crate::vm::frame::CallFrame;
use crate::vm::global_utils::global_index_by_name;
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

/// Get Layer method (freeze, unfreeze).
#[allow(clippy::too_many_arguments)]
pub fn get_layer(
    line: usize,
    stack: &mut Vec<crate::common::TaggedValue>,
    frames: &mut Vec<CallFrame>,
    globals: &mut Vec<crate::vm::global_slot::GlobalSlot>,
    global_names: &std::collections::BTreeMap<usize, String>,
    exception_handlers: &mut Vec<ExceptionHandler>,
    value_store: &mut ValueStore,
    heavy_store: &mut HeavyStore,
    index_value: Value,
) -> Result<VMStatus, LangError> {
    match index_value {
        Value::String(key) => {
            let function_name = match key.as_str() {
                "freeze" => "layer_freeze",
                "unfreeze" => "layer_unfreeze",
                _ => {
                    let error = ExceptionHandler::runtime_error_with_type(
                        &frames,
                        format!("Layer has no method '{}'. Available methods: freeze, unfreeze", key),
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
            let method_index = if let Some(ml_idx) = global_index_by_name(global_names, "ml") {
                if ml_idx >= globals.len() {
                    let error = ExceptionHandler::runtime_error(
                        &frames,
                        "ML module not found in globals".to_string(),
                        line,
                    );
                    return match ExceptionHandler::handle_exception(
                        stack, frames, exception_handlers, error, value_store, heavy_store,
                    ) {
                        Ok(()) => Ok(VMStatus::Continue),
                        Err(e) => Err(e),
                    };
                }
                let ml_id = globals[ml_idx].resolve_to_value_id(value_store);
                let ml_val = load_value(ml_id, value_store, heavy_store);
                match &ml_val {
                    Value::Object(map_rc) => {
                        let map = map_rc.borrow();
                        match map.get(function_name) {
                            Some(Value::NativeFunction(idx)) => *idx,
                            _ => {
                                let error = ExceptionHandler::runtime_error(
                                    &frames,
                                    format!("Layer method '{}' not registered in ml module", key),
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
                    }
                    _ => {
                        let error = ExceptionHandler::runtime_error(
                            &frames,
                            "ML module is not an object".to_string(),
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
            } else {
                let error = ExceptionHandler::runtime_error(
                    &frames,
                    "ML module not found".to_string(),
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
                "Layer property access must use string key".to_string(),
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

/// Get Dataset sample by index [features, target].
#[allow(clippy::too_many_arguments)]
pub fn get_dataset(
    line: usize,
    stack: &mut Vec<crate::common::TaggedValue>,
    frames: &mut Vec<CallFrame>,
    exception_handlers: &mut Vec<ExceptionHandler>,
    value_store: &mut ValueStore,
    heavy_store: &mut HeavyStore,
    dataset: Rc<RefCell<crate::ml::dataset::Dataset>>,
    index_value: Value,
) -> Result<VMStatus, LangError> {
    let index = match index_value {
        Value::Number(n) => {
            let idx = n as i64;
            if idx < 0 {
                let error = ExceptionHandler::runtime_error(
                    &frames,
                    "Dataset index must be non-negative".to_string(),
                    line,
                );
                return match ExceptionHandler::handle_exception(
                    stack, frames, exception_handlers, error, value_store, heavy_store,
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
                "Dataset index must be a number".to_string(),
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
    let dataset_ref = dataset.borrow();
    let batch_size = dataset_ref.batch_size();
    if index >= batch_size {
        let error = ExceptionHandler::runtime_error_with_type(
            &frames,
            format!("Dataset index {} out of bounds (length: {})", index, batch_size),
            line,
            ErrorType::IndexError,
        );
        return match ExceptionHandler::handle_exception(
            stack, frames, exception_handlers, error, value_store, heavy_store,
        ) {
            Ok(()) => Ok(VMStatus::Continue),
            Err(e) => Err(e),
        };
    }
    let num_features = dataset_ref.num_features();
    let features_start = index * num_features;
    let features_end = features_start + num_features;
    let features_data: Vec<f32> = Vec::from(&dataset_ref.features().data[features_start..features_end]);
    let features_tensor = Tensor::new(features_data, vec![num_features]).map_err(|e| {
        ExceptionHandler::runtime_error(&frames, format!("Failed to create features tensor: {}", e), line)
    })?;
    let num_targets = dataset_ref.num_targets();
    let targets_start = index * num_targets;
    let targets_end = targets_start + num_targets;
    let target_value = if num_targets == 1 {
        Value::Number(dataset_ref.targets().data[targets_start] as f64)
    } else {
        let target_data: Vec<f32> = Vec::from(&dataset_ref.targets().data[targets_start..targets_end]);
        let target_tensor =
            Tensor::new(target_data, vec![num_targets]).map_err(|e| {
                ExceptionHandler::runtime_error(&frames, format!("Failed to create target tensor: {}", e), line)
            })?;
        Value::Tensor(Rc::new(RefCell::new(target_tensor)))
    };
    let features_value = Value::Tensor(Rc::new(RefCell::new(features_tensor)));
    let pair = vec![features_value, target_value];
    stack::push_id(
        stack,
        store_value(Value::Array(Rc::new(RefCell::new(pair))), value_store, heavy_store),
    );
    Ok(VMStatus::Continue)
}

/// Get Tensor property (shape, data, max_idx, min_idx) or element by index.
#[allow(clippy::too_many_arguments)]
pub fn get_tensor(
    line: usize,
    stack: &mut Vec<crate::common::TaggedValue>,
    frames: &mut Vec<CallFrame>,
    exception_handlers: &mut Vec<ExceptionHandler>,
    value_store: &mut ValueStore,
    heavy_store: &mut HeavyStore,
    natives: &[crate::vm::host::HostEntry],
    tensor: Rc<RefCell<Tensor>>,
    index_value: Value,
) -> Result<VMStatus, LangError> {
    match index_value {
        Value::String(property_name) => {
            match property_name.as_str() {
                "shape" => {
                    let tensor_ref = tensor.borrow();
                    let shape_values: Vec<Value> =
                        tensor_ref.shape.iter().map(|&s| Value::Number(s as f64)).collect();
                    stack::push_id(
                        stack,
                        store_value(
                            Value::Array(Rc::new(RefCell::new(shape_values))),
                            value_store,
                            heavy_store,
                        ),
                    );
                }
                "data" => {
                    let tensor_ref = tensor.borrow();
                    let data_values: Vec<Value> =
                        tensor_ref.data.iter().map(|&d| Value::Number(d as f64)).collect();
                    stack::push_id(
                        stack,
                        store_value(
                            Value::Array(Rc::new(RefCell::new(data_values))),
                            value_store,
                            heavy_store,
                        ),
                    );
                }
                "max_idx" => {
                    use crate::ml::natives as ml_natives;
                    let max_idx_fn_ptr = ml_natives::native_max_idx as *const ();
                    let method_index = natives.iter().position(|e| e.as_fn_ptr() == Some(max_idx_fn_ptr));
                    if let Some(idx) = method_index {
                        // Push only the method; receiver is already on stack from compile_module_method step 1.
                        // Pushing tensor here would leave a stray value (e.g. "y: X pred: Y" → "22").
                        stack::push_id(
                            stack,
                            store_value(Value::NativeFunction(idx), value_store, heavy_store),
                        );
                    } else {
                        let error = ExceptionHandler::runtime_error(
                            &frames,
                            "max_idx method not found".to_string(),
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
                "min_idx" => {
                    use crate::ml::natives as ml_natives;
                    let min_idx_fn_ptr = ml_natives::native_min_idx as *const ();
                    let method_index = natives.iter().position(|e| e.as_fn_ptr() == Some(min_idx_fn_ptr));
                    if let Some(idx) = method_index {
                        // Push only the method; receiver is already on stack (same as max_idx).
                        stack::push_id(
                            stack,
                            store_value(Value::NativeFunction(idx), value_store, heavy_store),
                        );
                    } else {
                        let error = ExceptionHandler::runtime_error(
                            &frames,
                            "min_idx method not found".to_string(),
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
                        format!(
                            "Property '{}' not found on Tensor. Available properties: 'shape', 'data', 'max_idx', 'min_idx'",
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
            }
        }
        Value::Number(n) => {
            let idx = n as i64;
            if idx < 0 {
                let error = ExceptionHandler::runtime_error(
                    &frames,
                    "Tensor index must be non-negative".to_string(),
                    line,
                );
                return match ExceptionHandler::handle_exception(
                    stack, frames, exception_handlers, error, value_store, heavy_store,
                ) {
                    Ok(()) => Ok(VMStatus::Continue),
                    Err(e) => Err(e),
                };
            }
            let tensor_ref = tensor.borrow();
            let index = idx as usize;
            if tensor_ref.ndim() == 1 {
                if index >= tensor_ref.shape[0] {
                    let error = ExceptionHandler::runtime_error_with_type(
                        &frames,
                        format!("Tensor index {} out of bounds (size: {})", index, tensor_ref.shape[0]),
                        line,
                        ErrorType::IndexError,
                    );
                    return match ExceptionHandler::handle_exception(
                        stack, frames, exception_handlers, error, value_store, heavy_store,
                    ) {
                        Ok(()) => Ok(VMStatus::Continue),
                        Err(e) => Err(e),
                    };
                }
                stack::push_id(
                    stack,
                    store_value(Value::Number(tensor_ref.data[index] as f64), value_store, heavy_store),
                );
            } else {
                match tensor_ref.get_row(index) {
                    Ok(slice_tensor) => {
                        stack::push_id(
                            stack,
                            store_value(
                                Value::Tensor(Rc::new(RefCell::new(slice_tensor))),
                                value_store,
                                heavy_store,
                            ),
                        );
                    }
                    Err(e) => {
                        let error = ExceptionHandler::runtime_error_with_type(
                            &frames,
                            e,
                            line,
                            ErrorType::IndexError,
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
        }
        _ => {
            let error = ExceptionHandler::runtime_error(
                &frames,
                "Tensor property access requires string key (e.g., 'shape', 'data') or numeric index".to_string(),
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

/// Get NeuralNetwork property (layers) or method (train, train_sh, save, etc.).
#[allow(clippy::too_many_arguments)]
pub fn get_neural_network(
    line: usize,
    stack: &mut Vec<crate::common::TaggedValue>,
    frames: &mut Vec<CallFrame>,
    globals: &mut Vec<crate::vm::global_slot::GlobalSlot>,
    global_names: &std::collections::BTreeMap<usize, String>,
    exception_handlers: &mut Vec<ExceptionHandler>,
    value_store: &mut ValueStore,
    heavy_store: &mut HeavyStore,
    nn_rc: Rc<RefCell<crate::ml::model::NeuralNetwork>>,
    index_value: Value,
) -> Result<VMStatus, LangError> {
    match index_value {
        Value::String(key) => {
            if key == "layers" {
                let mut layer_accessor = HashMap::new();
                layer_accessor.insert(
                    "__neural_network".to_string(),
                    Value::NeuralNetwork(Rc::clone(&nn_rc)),
                );
                stack::push_id(
                    stack,
                    store_value(
                        Value::Object(Rc::new(RefCell::new(layer_accessor))),
                        value_store,
                        heavy_store,
                    ),
                );
                return Ok(VMStatus::Continue);
            }
            let function_name = match key.as_str() {
                "train" => "nn_train",
                "train_sh" => "nn_train_sh",
                "save" => "nn_save",
                "device" => "nn_set_device",
                "get_device" => "nn_get_device",
                _ => {
                    let error = ExceptionHandler::runtime_error_with_type(
                        &frames,
                        format!(
                            "NeuralNetwork has no method '{}'. Available methods: train, train_sh, save, device, get_device, layers",
                            key
                        ),
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
            let method_index = if let Some(ml_idx) = global_index_by_name(global_names, "ml") {
                if ml_idx >= globals.len() {
                    let error = ExceptionHandler::runtime_error(
                        &frames,
                        "ML module not found in globals".to_string(),
                        line,
                    );
                    return match ExceptionHandler::handle_exception(
                        stack, frames, exception_handlers, error, value_store, heavy_store,
                    ) {
                        Ok(()) => Ok(VMStatus::Continue),
                        Err(e) => Err(e),
                    };
                }
                let ml_id = globals[ml_idx].resolve_to_value_id(value_store);
                let ml_val = load_value(ml_id, value_store, heavy_store);
                match &ml_val {
                    Value::Object(map_rc) => {
                        let map = map_rc.borrow();
                        match map.get(function_name) {
                            Some(Value::NativeFunction(idx)) => *idx,
                            _ => {
                                let error = ExceptionHandler::runtime_error(
                                    &frames,
                                    format!("NeuralNetwork method '{}' not registered in ml module", key),
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
                    }
                    _ => {
                        let error = ExceptionHandler::runtime_error(
                            &frames,
                            "ML module is not an object".to_string(),
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
            } else {
                let error = ExceptionHandler::runtime_error(
                    &frames,
                    "ML module not found".to_string(),
                    line,
                );
                return match ExceptionHandler::handle_exception(
                    stack, frames, exception_handlers, error, value_store, heavy_store,
                ) {
                    Ok(()) => Ok(VMStatus::Continue),
                    Err(e) => Err(e),
                };
            };
            stack::push_id(
                stack,
                store_value(Value::NativeFunction(method_index), value_store, heavy_store),
            );
        }
        _ => {
            let error = ExceptionHandler::runtime_error(
                &frames,
                "NeuralNetwork property access must use string key".to_string(),
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

/// Get DatabaseEngine method (connect, execute, query, run).
#[allow(clippy::too_many_arguments)]
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
