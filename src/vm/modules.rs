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
const BUILTIN_MODULE_NAMES: &[&str] = &["plot", "settings_env", "uuid", "database_engine"];

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
