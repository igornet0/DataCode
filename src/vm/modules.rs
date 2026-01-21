// Module operations for VM

use crate::common::{error::LangError, value::Value};
use std::collections::HashMap;

/// Check if a name is a known module name
pub fn is_known_module(name: &str) -> bool {
    matches!(name, "ml" | "plot")
}

/// Register a module by name
/// This function registers native functions for the module and creates the module object in globals
pub fn register_module(
    module_name: &str,
    natives: &mut Vec<fn(&[Value]) -> Value>,
    globals: &mut Vec<Value>,
    global_names: &mut std::collections::HashMap<usize, String>,
) -> Result<(), LangError> {
    match module_name {
        "ml" => register_ml_module(natives, globals, global_names),
        "plot" => register_plot_module(natives, globals, global_names),
        _ => Err(LangError::runtime_error(
            format!("Unknown module: {}", module_name),
            0,
        )),
    }
}

fn register_ml_module(
    natives: &mut Vec<fn(&[Value]) -> Value>,
    globals: &mut Vec<Value>,
    global_names: &mut std::collections::HashMap<usize, String>,
) -> Result<(), LangError> {
    use crate::ml::natives;
    
    // Register ML native functions
    let ml_native_start = natives.len();
    natives.push(natives::native_tensor);
    natives.push(natives::native_shape);
    natives.push(natives::native_data);
    natives.push(natives::native_add);
    natives.push(natives::native_sub);
    natives.push(natives::native_mul);
    natives.push(natives::native_matmul);
    natives.push(natives::native_transpose);
    natives.push(natives::native_sum);
    natives.push(natives::native_mean);
    natives.push(natives::native_max_idx);
    natives.push(natives::native_min_idx);
    // Graph functions
    natives.push(natives::native_graph);
    natives.push(natives::native_graph_add_input);
    natives.push(natives::native_graph_add_op);
    natives.push(natives::native_graph_forward);
    natives.push(natives::native_graph_get_output);
    // Autograd functions
    natives.push(natives::native_graph_backward);
    natives.push(natives::native_graph_get_gradient);
    natives.push(natives::native_graph_zero_grad);
    natives.push(natives::native_graph_set_requires_grad);
    // Linear Regression functions
    natives.push(natives::native_linear_regression);
    natives.push(natives::native_lr_predict);
    natives.push(natives::native_lr_train);
    natives.push(natives::native_lr_evaluate);
    // Optimizer functions
    natives.push(natives::native_sgd);
    natives.push(natives::native_sgd_step);
    natives.push(natives::native_sgd_zero_grad);
    natives.push(natives::native_adam);
    natives.push(natives::native_adam_step);
    // Loss functions
    natives.push(natives::native_mse_loss);
    natives.push(natives::native_cross_entropy_loss);
    natives.push(natives::native_binary_cross_entropy_loss);
    natives.push(natives::native_mae_loss);
    natives.push(natives::native_huber_loss);
    natives.push(natives::native_hinge_loss);
    natives.push(natives::native_kl_divergence);
    natives.push(natives::native_smooth_l1_loss);
    // Dataset functions
    natives.push(natives::native_dataset);
    natives.push(natives::native_dataset_features);
    natives.push(natives::native_dataset_targets);
    natives.push(natives::native_onehot);
    natives.push(natives::native_load_mnist);
    // Layer functions
    natives.push(natives::native_linear_layer);
    natives.push(natives::native_relu_layer);
    natives.push(natives::native_softmax_layer);
    natives.push(natives::native_flatten_layer);
    natives.push(natives::native_layer_call);
    // Neural network functions
    natives.push(natives::native_sequential);
    natives.push(natives::native_sequential_add);
    natives.push(natives::native_neural_network);
    natives.push(natives::native_nn_forward);
    natives.push(natives::native_nn_train);
    natives.push(natives::native_nn_train_sh);
    natives.push(natives::native_nn_save);
    natives.push(natives::native_nn_load);
    natives.push(natives::native_categorical_cross_entropy_loss);
    natives.push(natives::native_ml_save_model);
    natives.push(natives::native_ml_load_model);
    // Device management functions
    natives.push(natives::native_ml_set_device);
    natives.push(natives::native_ml_get_device);
    natives.push(natives::native_nn_set_device);
    natives.push(natives::native_nn_get_device);
    natives.push(natives::native_devices);
    natives.push(natives::native_ml_validate_model);
    natives.push(natives::native_ml_model_info);
    // Layer freeze/unfreeze functions
    natives.push(natives::native_model_get_layer);
    natives.push(natives::native_layer_freeze);
    natives.push(natives::native_layer_unfreeze);
    
    // Create ML module object with native function references
    let mut ml_object = HashMap::new();
    ml_object.insert("tensor".to_string(), Value::NativeFunction(ml_native_start + 0));
    ml_object.insert("shape".to_string(), Value::NativeFunction(ml_native_start + 1));
    ml_object.insert("data".to_string(), Value::NativeFunction(ml_native_start + 2));
    ml_object.insert("add".to_string(), Value::NativeFunction(ml_native_start + 3));
    ml_object.insert("sub".to_string(), Value::NativeFunction(ml_native_start + 4));
    ml_object.insert("mul".to_string(), Value::NativeFunction(ml_native_start + 5));
    ml_object.insert("matmul".to_string(), Value::NativeFunction(ml_native_start + 6));
    ml_object.insert("transpose".to_string(), Value::NativeFunction(ml_native_start + 7));
    ml_object.insert("sum".to_string(), Value::NativeFunction(ml_native_start + 8));
    ml_object.insert("mean".to_string(), Value::NativeFunction(ml_native_start + 9));
    ml_object.insert("max_idx".to_string(), Value::NativeFunction(ml_native_start + 10));
    ml_object.insert("min_idx".to_string(), Value::NativeFunction(ml_native_start + 11));
    // Graph functions
    ml_object.insert("graph".to_string(), Value::NativeFunction(ml_native_start + 12));
    ml_object.insert("graph_add_input".to_string(), Value::NativeFunction(ml_native_start + 13));
    ml_object.insert("graph_add_op".to_string(), Value::NativeFunction(ml_native_start + 14));
    ml_object.insert("graph_forward".to_string(), Value::NativeFunction(ml_native_start + 15));
    ml_object.insert("graph_get_output".to_string(), Value::NativeFunction(ml_native_start + 16));
    // Autograd functions
    ml_object.insert("graph_backward".to_string(), Value::NativeFunction(ml_native_start + 17));
    ml_object.insert("graph_get_gradient".to_string(), Value::NativeFunction(ml_native_start + 18));
    ml_object.insert("graph_zero_grad".to_string(), Value::NativeFunction(ml_native_start + 19));
    ml_object.insert("graph_set_requires_grad".to_string(), Value::NativeFunction(ml_native_start + 20));
    // Linear Regression functions
    ml_object.insert("linear_regression".to_string(), Value::NativeFunction(ml_native_start + 21));
    ml_object.insert("lr_predict".to_string(), Value::NativeFunction(ml_native_start + 22));
    ml_object.insert("lr_train".to_string(), Value::NativeFunction(ml_native_start + 23));
    ml_object.insert("lr_evaluate".to_string(), Value::NativeFunction(ml_native_start + 24));
    // Optimizer functions
    ml_object.insert("sgd".to_string(), Value::NativeFunction(ml_native_start + 25));
    ml_object.insert("sgd_step".to_string(), Value::NativeFunction(ml_native_start + 26));
    ml_object.insert("sgd_zero_grad".to_string(), Value::NativeFunction(ml_native_start + 27));
    // Loss functions
    ml_object.insert("mse_loss".to_string(), Value::NativeFunction(ml_native_start + 30));
    ml_object.insert("cross_entropy_loss".to_string(), Value::NativeFunction(ml_native_start + 31));
    ml_object.insert("binary_cross_entropy_loss".to_string(), Value::NativeFunction(ml_native_start + 32));
    ml_object.insert("mae_loss".to_string(), Value::NativeFunction(ml_native_start + 33));
    ml_object.insert("huber_loss".to_string(), Value::NativeFunction(ml_native_start + 34));
    ml_object.insert("hinge_loss".to_string(), Value::NativeFunction(ml_native_start + 35));
    ml_object.insert("kl_divergence".to_string(), Value::NativeFunction(ml_native_start + 36));
    ml_object.insert("smooth_l1_loss".to_string(), Value::NativeFunction(ml_native_start + 37));
    // Dataset functions
    ml_object.insert("dataset".to_string(), Value::NativeFunction(ml_native_start + 38));
    ml_object.insert("dataset_features".to_string(), Value::NativeFunction(ml_native_start + 39));
    ml_object.insert("dataset_targets".to_string(), Value::NativeFunction(ml_native_start + 40));
    ml_object.insert("onehot".to_string(), Value::NativeFunction(ml_native_start + 41));
    ml_object.insert("load_mnist".to_string(), Value::NativeFunction(ml_native_start + 42));
    // Layer object with all layer functions
    let mut layer_object = HashMap::new();
    layer_object.insert("linear".to_string(), Value::NativeFunction(ml_native_start + 43));
    layer_object.insert("relu".to_string(), Value::NativeFunction(ml_native_start + 44));
    layer_object.insert("softmax".to_string(), Value::NativeFunction(ml_native_start + 45));
    layer_object.insert("flatten".to_string(), Value::NativeFunction(ml_native_start + 46));
    ml_object.insert("layer".to_string(), Value::Object(layer_object));
    // Neural network functions
    ml_object.insert("sequential".to_string(), Value::NativeFunction(ml_native_start + 48));
    ml_object.insert("sequential_add".to_string(), Value::NativeFunction(ml_native_start + 49));
    ml_object.insert("neural_network".to_string(), Value::NativeFunction(ml_native_start + 50));
    ml_object.insert("nn_forward".to_string(), Value::NativeFunction(ml_native_start + 51));
    ml_object.insert("nn_train".to_string(), Value::NativeFunction(ml_native_start + 52));
    ml_object.insert("nn_train_sh".to_string(), Value::NativeFunction(ml_native_start + 53));
    ml_object.insert("nn_save".to_string(), Value::NativeFunction(ml_native_start + 54));
    ml_object.insert("nn_load".to_string(), Value::NativeFunction(ml_native_start + 55));
    ml_object.insert("categorical_cross_entropy_loss".to_string(), Value::NativeFunction(ml_native_start + 56));
    ml_object.insert("save_model".to_string(), Value::NativeFunction(ml_native_start + 57));
    ml_object.insert("load".to_string(), Value::NativeFunction(ml_native_start + 58));
    // Device management
    ml_object.insert("set_device".to_string(), Value::NativeFunction(ml_native_start + 59));
    ml_object.insert("get_device".to_string(), Value::NativeFunction(ml_native_start + 60));
    ml_object.insert("nn_set_device".to_string(), Value::NativeFunction(ml_native_start + 61));
    ml_object.insert("nn_get_device".to_string(), Value::NativeFunction(ml_native_start + 62));
    ml_object.insert("devices".to_string(), Value::NativeFunction(ml_native_start + 63));
    ml_object.insert("validate_model".to_string(), Value::NativeFunction(ml_native_start + 64));
    ml_object.insert("model_info".to_string(), Value::NativeFunction(ml_native_start + 65));
    // Layer freeze/unfreeze functions
    ml_object.insert("model_get_layer".to_string(), Value::NativeFunction(ml_native_start + 66));
    ml_object.insert("layer_freeze".to_string(), Value::NativeFunction(ml_native_start + 67));
    ml_object.insert("layer_unfreeze".to_string(), Value::NativeFunction(ml_native_start + 68));
    
    // Register ml as a global variable
    // First, check if "ml" is already in global_names (from compiler)
    let ml_index = if let Some((&idx, _)) = global_names.iter().find(|(_, name)| name.as_str() == "ml") {
        // ml is already registered by compiler, use that index
        // Make sure globals vector is large enough
        if idx >= globals.len() {
            globals.resize(idx + 1, Value::Null);
        }
        idx
    } else if let Some(idx) = globals.iter().position(|v| {
        if let Value::Object(map) = v {
            map.contains_key("tensor")
        } else {
            false
        }
    }) {
        // ml object already exists at this index
        idx
    } else {
        // Create new global index
        let idx = globals.len();
        // Push a placeholder, will be set below
        globals.push(Value::Null);
        global_names.insert(idx, "ml".to_string());
        idx
    };
    
    // Store ml object in globals (always store the original, not a clone)
    // Ensure the vector is large enough
    if ml_index >= globals.len() {
        globals.resize(ml_index + 1, Value::Null);
    }
    // Verify ml_object is not empty before storing
    if ml_object.is_empty() {
        return Err(LangError::runtime_error(
            "ML module object is empty - native functions not registered".to_string(),
            0,
        ));
    }
    // Store the module object
    globals[ml_index] = Value::Object(ml_object);
    
    // Verify it was stored correctly
    match &globals[ml_index] {
        Value::Object(map) => {
            if map.is_empty() {
                return Err(LangError::runtime_error(
                    "ML module object stored but is empty".to_string(),
                    0,
                ));
            }
        }
        _ => {
            return Err(LangError::runtime_error(
                format!("ML module not stored as Object, found: {:?}", 
                    std::mem::discriminant(&globals[ml_index])),
                0,
            ));
        }
    }
    
    Ok(())
}

fn register_plot_module(
    natives: &mut Vec<fn(&[Value]) -> Value>,
    globals: &mut Vec<Value>,
    global_names: &mut std::collections::HashMap<usize, String>,
) -> Result<(), LangError> {
    use crate::plot::natives;
    
    // Register plot native functions
    let plot_native_start = natives.len();
    natives.push(natives::native_plot_image);
    natives.push(natives::native_plot_window);
    natives.push(natives::native_window_draw);
    natives.push(natives::native_plot_wait);
    natives.push(natives::native_plot_show);
    natives.push(natives::native_plot_show_grid);
    natives.push(natives::native_plot_subplots);
    natives.push(natives::native_plot_tight_layout);
    natives.push(natives::native_plot_show_figure);
    natives.push(natives::native_axis_imshow);
    natives.push(natives::native_axis_set_title);
    natives.push(natives::native_axis_axis);
    natives.push(natives::native_plot_xlabel);
    natives.push(natives::native_plot_ylabel);
    natives.push(natives::native_plot_line);
    natives.push(natives::native_plot_bar);
    natives.push(natives::native_plot_pie);
    natives.push(natives::native_plot_heatmap);
    
    // Create plot module object with native function references
    let mut plot_object = HashMap::new();
    plot_object.insert("image".to_string(), Value::NativeFunction(plot_native_start + 0));
    plot_object.insert("window".to_string(), Value::NativeFunction(plot_native_start + 1));
    plot_object.insert("draw".to_string(), Value::NativeFunction(plot_native_start + 2));
    plot_object.insert("wait".to_string(), Value::NativeFunction(plot_native_start + 3));
    plot_object.insert("show".to_string(), Value::NativeFunction(plot_native_start + 4));
    plot_object.insert("show_grid".to_string(), Value::NativeFunction(plot_native_start + 5));
    plot_object.insert("subplots".to_string(), Value::NativeFunction(plot_native_start + 6));
    plot_object.insert("tight_layout".to_string(), Value::NativeFunction(plot_native_start + 7));
    plot_object.insert("xlabel".to_string(), Value::NativeFunction(plot_native_start + 12));
    plot_object.insert("ylabel".to_string(), Value::NativeFunction(plot_native_start + 13));
    plot_object.insert("line".to_string(), Value::NativeFunction(plot_native_start + 14));
    plot_object.insert("bar".to_string(), Value::NativeFunction(plot_native_start + 15));
    plot_object.insert("pie".to_string(), Value::NativeFunction(plot_native_start + 16));
    plot_object.insert("heatmap".to_string(), Value::NativeFunction(plot_native_start + 17));
    // show_figure is handled by checking if argument is Figure in native_plot_show
    
    // Store axis method indices for later lookup
    // imshow = plot_native_start + 9, set_title = +10, axis = +11
    let axis_imshow_idx = plot_native_start + 9;
    let axis_set_title_idx = plot_native_start + 10;
    let axis_axis_idx = plot_native_start + 11;
    
    // Store in plot object for access (we'll use a special key)
    plot_object.insert("__axis_imshow_idx".to_string(), Value::Number(axis_imshow_idx as f64));
    plot_object.insert("__axis_set_title_idx".to_string(), Value::Number(axis_set_title_idx as f64));
    plot_object.insert("__axis_axis_idx".to_string(), Value::Number(axis_axis_idx as f64));
    
    // Register plot as a global variable
    let plot_index = if let Some((&idx, _)) = global_names.iter().find(|(_, name)| name.as_str() == "plot") {
        if idx >= globals.len() {
            globals.resize(idx + 1, Value::Null);
        }
        idx
    } else if let Some(idx) = globals.iter().position(|v| {
        if let Value::Object(map) = v {
            map.contains_key("image")
        } else {
            false
        }
    }) {
        idx
    } else {
        let idx = globals.len();
        globals.push(Value::Object(plot_object.clone()));
        global_names.insert(idx, "plot".to_string());
        idx
    };
    
    // Store plot object in globals
    globals[plot_index] = Value::Object(plot_object);
    
    Ok(())
}
