// Module operations for VM (globals as Vec<GlobalSlot>)

use crate::common::{error::LangError, value::Value, value_store::ValueStore};
use crate::vm::global_slot::{GlobalSlot, default_global_slot};
use crate::vm::heavy_store::HeavyStore;
use crate::vm::host::HostEntry;
use crate::vm::store_convert::{store_value_arena, load_value};
use std::collections::HashMap;
use std::rc::Rc;
use std::cell::RefCell;

/// Built-in module names (for error messages and is_known_module)
const BUILTIN_MODULE_NAMES: &[&str] = &["ml", "plot", "settings_env", "uuid", "database_engine"];

/// Check if a name is a known module name
pub fn is_known_module(name: &str) -> bool {
    BUILTIN_MODULE_NAMES.contains(&name)
}

/// Comma-separated list of built-in module names for error messages
pub fn builtin_modules_list() -> String {
    BUILTIN_MODULE_NAMES.join(", ")
}

/// Deterministic global slot by name (min index when multiple; stable across HashMap iteration).
fn global_index_by_name(global_names: &std::collections::BTreeMap<usize, String>, name: &str) -> Option<usize> {
    global_names
        .iter()
        .filter(|(_, n)| n.as_str() == name)
        .map(|(idx, _)| *idx)
        .min()
}

/// Register a module by name (globals as Vec<GlobalSlot>, store_value for module object)
pub fn register_module(
    module_name: &str,
    natives: &mut Vec<HostEntry>,
    globals: &mut Vec<GlobalSlot>,
    global_names: &mut std::collections::BTreeMap<usize, String>,
    store: &mut ValueStore,
    heap: &mut HeavyStore,
) -> Result<(), LangError> {
    match module_name {
        "ml" => register_ml_module(natives, globals, global_names, store, heap),
        "plot" => register_plot_module(natives, globals, global_names, store, heap),
        "settings_env" => register_settings_env_module(natives, globals, global_names, store, heap),
        "uuid" => register_uuid_module(natives, globals, global_names, store, heap),
        "database_engine" => register_database_module(natives, globals, global_names, store, heap),
        _ => Err(LangError::runtime_error(
            format!("Unknown module: {}", module_name),
            0,
        )),
    }
}

fn register_ml_module(
    natives: &mut Vec<HostEntry>,
    globals: &mut Vec<GlobalSlot>,
    global_names: &mut std::collections::BTreeMap<usize, String>,
    store: &mut ValueStore,
    heap: &mut HeavyStore,
) -> Result<(), LangError> {
    use crate::ml::natives;
    
    // Register ML native functions (Extended: fn pointer for ptr::eq in call_engine)
    let ml_native_start = natives.len();
    natives.push(HostEntry::Extended(natives::native_tensor));
    natives.push(HostEntry::Extended(natives::native_shape));
    natives.push(HostEntry::Extended(natives::native_data));
    natives.push(HostEntry::Extended(natives::native_add));
    natives.push(HostEntry::Extended(natives::native_sub));
    natives.push(HostEntry::Extended(natives::native_mul));
    natives.push(HostEntry::Extended(natives::native_matmul));
    natives.push(HostEntry::Extended(natives::native_transpose));
    natives.push(HostEntry::Extended(natives::native_sum));
    natives.push(HostEntry::Extended(natives::native_mean));
    natives.push(HostEntry::Extended(natives::native_max_idx));
    natives.push(HostEntry::Extended(natives::native_min_idx));
    // Graph functions
    natives.push(HostEntry::Extended(natives::native_graph));
    natives.push(HostEntry::Extended(natives::native_graph_add_input));
    natives.push(HostEntry::Extended(natives::native_graph_add_op));
    natives.push(HostEntry::Extended(natives::native_graph_forward));
    natives.push(HostEntry::Extended(natives::native_graph_get_output));
    // Autograd functions
    natives.push(HostEntry::Extended(natives::native_graph_backward));
    natives.push(HostEntry::Extended(natives::native_graph_get_gradient));
    natives.push(HostEntry::Extended(natives::native_graph_zero_grad));
    natives.push(HostEntry::Extended(natives::native_graph_set_requires_grad));
    // Linear Regression functions
    natives.push(HostEntry::Extended(natives::native_linear_regression));
    natives.push(HostEntry::Extended(natives::native_lr_predict));
    natives.push(HostEntry::Extended(natives::native_lr_train));
    natives.push(HostEntry::Extended(natives::native_lr_evaluate));
    // Optimizer functions
    natives.push(HostEntry::Extended(natives::native_sgd));
    natives.push(HostEntry::Extended(natives::native_sgd_step));
    natives.push(HostEntry::Extended(natives::native_sgd_zero_grad));
    natives.push(HostEntry::Extended(natives::native_adam));
    natives.push(HostEntry::Extended(natives::native_adam_step));
    // Loss functions
    natives.push(HostEntry::Extended(natives::native_mse_loss));
    natives.push(HostEntry::Extended(natives::native_cross_entropy_loss));
    natives.push(HostEntry::Extended(natives::native_binary_cross_entropy_loss));
    natives.push(HostEntry::Extended(natives::native_mae_loss));
    natives.push(HostEntry::Extended(natives::native_huber_loss));
    natives.push(HostEntry::Extended(natives::native_hinge_loss));
    natives.push(HostEntry::Extended(natives::native_kl_divergence));
    natives.push(HostEntry::Extended(natives::native_smooth_l1_loss));
    // Dataset functions
    natives.push(HostEntry::Extended(natives::native_dataset));
    natives.push(HostEntry::Extended(natives::native_dataset_features));
    natives.push(HostEntry::Extended(natives::native_dataset_targets));
    natives.push(HostEntry::Extended(natives::native_onehot));
    natives.push(HostEntry::Extended(natives::native_load_mnist));
    // Layer functions
    natives.push(HostEntry::Extended(natives::native_linear_layer));
    natives.push(HostEntry::Extended(natives::native_relu_layer));
    natives.push(HostEntry::Extended(natives::native_softmax_layer));
    natives.push(HostEntry::Extended(natives::native_flatten_layer));
    natives.push(HostEntry::Extended(natives::native_layer_call));
    // Neural network functions
    natives.push(HostEntry::Extended(natives::native_sequential));
    natives.push(HostEntry::Extended(natives::native_sequential_add));
    natives.push(HostEntry::Extended(natives::native_neural_network));
    natives.push(HostEntry::Extended(natives::native_nn_forward));
    natives.push(HostEntry::Extended(natives::native_nn_train));
    natives.push(HostEntry::Extended(natives::native_nn_train_sh));
    natives.push(HostEntry::Extended(natives::native_nn_save));
    natives.push(HostEntry::Extended(natives::native_nn_load));
    natives.push(HostEntry::Extended(natives::native_categorical_cross_entropy_loss));
    natives.push(HostEntry::Extended(natives::native_ml_save_model));
    natives.push(HostEntry::Extended(natives::native_ml_load_model));
    // Device management functions
    natives.push(HostEntry::Extended(natives::native_ml_set_device));
    natives.push(HostEntry::Extended(natives::native_ml_get_device));
    natives.push(HostEntry::Extended(natives::native_nn_set_device));
    natives.push(HostEntry::Extended(natives::native_nn_get_device));
    natives.push(HostEntry::Extended(natives::native_devices));
    natives.push(HostEntry::Extended(natives::native_ml_validate_model));
    natives.push(HostEntry::Extended(natives::native_ml_model_info));
    // Layer freeze/unfreeze functions
    natives.push(HostEntry::Extended(natives::native_model_get_layer));
    natives.push(HostEntry::Extended(natives::native_layer_freeze));
    natives.push(HostEntry::Extended(natives::native_layer_unfreeze));
    
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
    ml_object.insert("layer".to_string(), Value::Object(Rc::new(RefCell::new(layer_object))));
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
    
    let ml_index = if let Some(idx) = global_index_by_name(global_names, "ml") {
        if idx >= globals.len() {
            globals.resize(idx + 1, default_global_slot());
        }
        idx
    } else {
        let ml_idx_opt = globals.iter_mut().enumerate().find_map(|(i, slot)| {
            let id = slot.resolve_to_value_id(store);
            let v = load_value(id, store, heap);
            if let Value::Object(map_rc) = &v {
                if map_rc.borrow().contains_key("tensor") {
                    return Some(i);
                }
            }
            None
        });
        if let Some(idx) = ml_idx_opt {
            idx
        } else {
            let idx = globals.len();
            globals.push(default_global_slot());
            global_names.insert(idx, "ml".to_string());
            idx
        }
    };

    if ml_index >= globals.len() {
        globals.resize(ml_index + 1, default_global_slot());
    }
    if ml_object.is_empty() {
        return Err(LangError::runtime_error(
            "ML module object is empty - native functions not registered".to_string(),
            0,
        ));
    }
    globals[ml_index] = GlobalSlot::Heap(store_value_arena(Value::Object(Rc::new(RefCell::new(ml_object))), store, heap));

    let id = globals[ml_index].resolve_to_value_id(store);
    let v = load_value(id, store, heap);
    match &v {
        Value::Object(map_rc) => {
            let map = map_rc.borrow();
            if map.is_empty() {
                return Err(LangError::runtime_error(
                    "ML module object stored but is empty".to_string(),
                    0,
                ));
            }
        }
        _ => {
            return Err(LangError::runtime_error(
                format!("ML module not stored as Object, found: {:?}", std::mem::discriminant(&v)),
                0,
            ));
        }
    }

    Ok(())
}

fn register_plot_module(
    natives: &mut Vec<HostEntry>,
    globals: &mut Vec<GlobalSlot>,
    global_names: &mut std::collections::BTreeMap<usize, String>,
    store: &mut ValueStore,
    heap: &mut HeavyStore,
) -> Result<(), LangError> {
    use crate::plot::natives;
    
    // Register plot native functions
    let plot_native_start = natives.len();
    natives.push(HostEntry::Extended(natives::native_plot_image));
    natives.push(HostEntry::Extended(natives::native_plot_window));
    natives.push(HostEntry::Extended(natives::native_window_draw));
    natives.push(HostEntry::Extended(natives::native_plot_wait));
    natives.push(HostEntry::Extended(natives::native_plot_show));
    natives.push(HostEntry::Extended(natives::native_plot_show_grid));
    natives.push(HostEntry::Extended(natives::native_plot_subplots));
    natives.push(HostEntry::Extended(natives::native_plot_tight_layout));
    natives.push(HostEntry::Extended(natives::native_plot_show_figure));
    natives.push(HostEntry::Extended(natives::native_axis_imshow));
    natives.push(HostEntry::Extended(natives::native_axis_set_title));
    natives.push(HostEntry::Extended(natives::native_axis_axis));
    natives.push(HostEntry::Extended(natives::native_plot_xlabel));
    natives.push(HostEntry::Extended(natives::native_plot_ylabel));
    natives.push(HostEntry::Extended(natives::native_plot_line));
    natives.push(HostEntry::Extended(natives::native_plot_bar));
    natives.push(HostEntry::Extended(natives::native_plot_pie));
    natives.push(HostEntry::Extended(natives::native_plot_heatmap));
    
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
    
    let plot_index = if let Some(idx) = global_index_by_name(global_names, "plot") {
        if idx >= globals.len() {
            globals.resize(idx + 1, default_global_slot());
        }
        idx
    } else {
        let plot_idx_opt = globals.iter_mut().enumerate().find_map(|(i, slot)| {
            let id = slot.resolve_to_value_id(store);
            let v = load_value(id, store, heap);
            if let Value::Object(map_rc) = &v {
                if map_rc.borrow().contains_key("image") {
                    return Some(i);
                }
            }
            None
        });
        if let Some(idx) = plot_idx_opt {
            idx
        } else {
            let idx = globals.len();
            globals.push(GlobalSlot::Heap(store_value_arena(Value::Object(Rc::new(RefCell::new(plot_object.clone()))), store, heap)));
            global_names.insert(idx, "plot".to_string());
            idx
        }
    };

    globals[plot_index] = GlobalSlot::Heap(store_value_arena(Value::Object(Rc::new(RefCell::new(plot_object))), store, heap));

    Ok(())
}

fn register_settings_env_module(
    natives: &mut Vec<HostEntry>,
    globals: &mut Vec<GlobalSlot>,
    global_names: &mut std::collections::BTreeMap<usize, String>,
    store: &mut ValueStore,
    heap: &mut HeavyStore,
) -> Result<(), LangError> {
    use crate::settings_env::natives;
    
    let settings_env_native_start = natives.len();
    natives.push(HostEntry::Extended(natives::native_settings_env_load_env));
    natives.push(HostEntry::Extended(natives::native_settings_env_settings));
    natives.push(HostEntry::Extended(natives::native_settings_env_field));
    natives.push(HostEntry::Extended(natives::native_settings_env_config));
    
    let mut settings_env_object = HashMap::new();
    let load_env_fn = Value::NativeFunction(settings_env_native_start + 0);
    let settings_call = Value::NativeFunction(settings_env_native_start + 1);
    let config_fn = Value::NativeFunction(settings_env_native_start + 3);
    let mut settings_object = HashMap::new();
    settings_object.insert("__call__".to_string(), settings_call);
    settings_object.insert("config".to_string(), config_fn.clone());
    let settings_value = Value::Object(Rc::new(RefCell::new(settings_object)));
    settings_env_object.insert("load_env".to_string(), load_env_fn);
    settings_env_object.insert("Settings".to_string(), settings_value.clone());
    settings_env_object.insert("settings".to_string(), settings_value);
    settings_env_object.insert("Field".to_string(), Value::NativeFunction(settings_env_native_start + 2));
    settings_env_object.insert("Config".to_string(), config_fn);

    let settings_env_index = if let Some(idx) = global_index_by_name(global_names, "settings_env") {
        if idx >= globals.len() {
            globals.resize(idx + 1, default_global_slot());
        }
        idx
    } else {
        let idx = globals.len();
        globals.push(default_global_slot());
        global_names.insert(idx, "settings_env".to_string());
        idx
    };

    globals[settings_env_index] = GlobalSlot::Heap(store_value_arena(Value::Object(Rc::new(RefCell::new(settings_env_object))), store, heap));

    Ok(())
}

fn register_uuid_module(
    natives: &mut Vec<HostEntry>,
    globals: &mut Vec<GlobalSlot>,
    global_names: &mut std::collections::BTreeMap<usize, String>,
    store: &mut ValueStore,
    heap: &mut HeavyStore,
) -> Result<(), LangError> {
    use crate::uuid::natives;
    
    let uuid_native_start = natives.len();
    natives.push(HostEntry::Extended(natives::native_uuid_v4));
    natives.push(HostEntry::Extended(natives::native_uuid_v7));
    natives.push(HostEntry::Extended(natives::native_uuid_new));
    natives.push(HostEntry::Extended(natives::native_uuid_random));
    natives.push(HostEntry::Extended(natives::native_uuid_parse));
    natives.push(HostEntry::Extended(natives::native_uuid_to_string));
    natives.push(HostEntry::Extended(natives::native_uuid_to_bytes));
    natives.push(HostEntry::Extended(natives::native_uuid_from_bytes));
    natives.push(HostEntry::Extended(natives::native_uuid_version));
    natives.push(HostEntry::Extended(natives::native_uuid_variant));
    natives.push(HostEntry::Extended(natives::native_uuid_timestamp));
    natives.push(HostEntry::Extended(natives::native_uuid_v3));
    natives.push(HostEntry::Extended(natives::native_uuid_v5));
    
    let start = uuid_native_start;
    let mut uuid_object = HashMap::new();
    uuid_object.insert("v4".to_string(), Value::NativeFunction(start + 0));
    uuid_object.insert("v7".to_string(), Value::NativeFunction(start + 1));
    uuid_object.insert("new".to_string(), Value::NativeFunction(start + 2));
    uuid_object.insert("random".to_string(), Value::NativeFunction(start + 3));
    uuid_object.insert("parse".to_string(), Value::NativeFunction(start + 4));
    uuid_object.insert("to_string".to_string(), Value::NativeFunction(start + 5));
    uuid_object.insert("to_bytes".to_string(), Value::NativeFunction(start + 6));
    uuid_object.insert("from_bytes".to_string(), Value::NativeFunction(start + 7));
    uuid_object.insert("version".to_string(), Value::NativeFunction(start + 8));
    uuid_object.insert("variant".to_string(), Value::NativeFunction(start + 9));
    uuid_object.insert("timestamp".to_string(), Value::NativeFunction(start + 10));
    uuid_object.insert("v3".to_string(), Value::NativeFunction(start + 11));
    uuid_object.insert("v5".to_string(), Value::NativeFunction(start + 12));
    uuid_object.insert("DNS".to_string(), natives::uuid_namespace_dns());
    uuid_object.insert("URL".to_string(), natives::uuid_namespace_url());
    uuid_object.insert("OID".to_string(), natives::uuid_namespace_oid());
    
    let uuid_index = if let Some(idx) = global_index_by_name(global_names, "uuid") {
        if idx >= globals.len() {
            globals.resize(idx + 1, default_global_slot());
        }
        idx
    } else {
        let idx = globals.len();
        globals.push(default_global_slot());
        global_names.insert(idx, "uuid".to_string());
        idx
    };

    globals[uuid_index] = GlobalSlot::Heap(store_value_arena(Value::Object(Rc::new(RefCell::new(uuid_object))), store, heap));

    Ok(())
}

fn register_database_module(
    natives: &mut Vec<HostEntry>,
    globals: &mut Vec<GlobalSlot>,
    global_names: &mut std::collections::BTreeMap<usize, String>,
    store: &mut ValueStore,
    heap: &mut HeavyStore,
) -> Result<(), LangError> {
    use crate::database_engine::natives;

    let db_native_start = natives.len();
    natives.push(HostEntry::Extended(natives::native_engine));
    natives.push(HostEntry::Extended(natives::native_engine_connect));
    natives.push(HostEntry::Extended(natives::native_engine_execute));
    natives.push(HostEntry::Extended(natives::native_engine_query));
    natives.push(HostEntry::Extended(natives::native_metadata));
    natives.push(HostEntry::Extended(natives::native_column));
    natives.push(HostEntry::Extended(natives::native_now_call));
    natives.push(HostEntry::Extended(natives::native_select));
    natives.push(HostEntry::Extended(natives::native_engine_run));
    natives.push(HostEntry::Extended(natives::native_cluster));
    natives.push(HostEntry::Extended(natives::native_cluster_add));
    natives.push(HostEntry::Extended(natives::native_cluster_get));
    natives.push(HostEntry::Extended(natives::native_cluster_names));

    let start = db_native_start;
    let mut database_object = HashMap::new();
    database_object.insert("engine".to_string(), Value::NativeFunction(start + 0));
    database_object.insert("MetaData".to_string(), Value::NativeFunction(start + 4));
    database_object.insert("Column".to_string(), Value::NativeFunction(start + 5));
    database_object.insert("Сolumn".to_string(), Value::NativeFunction(start + 5)); // Cyrillic С
    database_object.insert("now_call".to_string(), Value::NativeFunction(start + 6));
    database_object.insert("select".to_string(), Value::NativeFunction(start + 7));
    database_object.insert("DatabaseCluster".to_string(), Value::NativeFunction(start + 9));
    // connect, execute, query, run are methods on engine - accessed via GetArrayElement on DatabaseEngine
    // add, get, names are methods on cluster - accessed via GetArrayElement on DatabaseCluster

    let database_index = if let Some(idx) = global_index_by_name(global_names, "database_engine") {
        if idx >= globals.len() {
            globals.resize(idx + 1, default_global_slot());
        }
        idx
    } else {
        let idx = globals.len();
        globals.push(default_global_slot());
        global_names.insert(idx, "database_engine".to_string());
        idx
    };

    globals[database_index] = GlobalSlot::Heap(store_value_arena(Value::Object(Rc::new(RefCell::new(database_object))), store, heap));

    Ok(())
}
