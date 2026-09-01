// Module operations for VM (globals as Vec<GlobalSlot>)

use crate::common::{error::LangError, value::Value, value_store::ValueStore};
use crate::vm::global_slot::{default_global_slot, GlobalSlot};
use crate::vm::heavy_store::HeavyStore;
use crate::vm::host::HostEntry;
use crate::vm::store_convert::{load_value, store_value_arena};
use std::collections::HashMap;

/// Built-in module names (for error messages, is_known_module, and register_all_builtin_modules / loaded_modules).
pub const BUILTIN_MODULE_NAMES: &[&str] = &[
    "plot",
    "settings_env",
    "uuid",
    "crypto",
    "database_engine",
    "system",
    "debug",
    "heapq",
    "pathfind",
    "grid",
    "websocket",
    "ws",
    "web",
];

/// Check if a name is a known module name
pub fn is_known_module(name: &str) -> bool {
    BUILTIN_MODULE_NAMES.contains(&name)
}

/// Comma-separated list of built-in module names for error messages
pub fn builtin_modules_list() -> String {
    BUILTIN_MODULE_NAMES.join(", ")
}

/// Top-level public exports of a built-in module (no `__*` keys).
/// Used by the compiler so `from M import *` can register call targets at compile time.
pub fn builtin_module_exports(name: &str) -> Option<&'static [&'static str]> {
    Some(match name {
        "ws" => &[
            "assets",
            "has_asset",
            "has_table",
            "metadata",
            "metadata_get",
            "package_info",
            "source_table",
            "tables",
        ],
        "web" => &["http", "browser", "data"],
        "websocket" => &["configure", "disable_builtin", "enable_builtin"],
        "heapq" => &[
            "heap_clear",
            "heappeek",
            "heapify",
            "heappop",
            "heappush",
            "heapreplace",
        ],
        "debug" => &["operators"],
        "uuid" => &[
            "DNS", "OID", "URL", "from_bytes", "new", "parse", "random", "timestamp",
            "to_bytes", "to_string", "v3", "v4", "v5", "v7", "variant", "version",
        ],
        "settings_env" => &["Config", "Field", "Settings", "load_env", "settings"],
        "crypto" => &["Argon2", "bcrypt", "secure_compare"],
        "pathfind" => &["astar_grid"],
        "system" => &[
            "env", "fs", "hardware", "log", "net", "permissions", "process", "runtime", "time",
        ],
        "database_engine" => &[
            "Column", "DatabaseCluster", "MetaData", "SQLEnum", "bool", "date", "engine",
            "float", "int", "now_call", "select", "str", "validators",
        ],
        "plot" => &[
            "bar", "draw", "heatmap", "image", "line", "pie", "show", "show_grid", "subplots",
            "tight_layout", "wait", "window", "xlabel", "ylabel",
        ],
        "grid" => &[
            "alloc_i32", "alloc_u8", "astar", "astar_from_set", "astar_step", "bitmap_bytes",
            "bitmap_from_ids", "fill_i32", "fill_u8", "get_i32", "get_u8", "heap_alloc",
            "heap_clear", "heap_len", "heap_pop", "heap_push", "set_blocked", "set_i32", "set_u8",
            "shrink", "shrink_all", "store_len", "test_blocked",
        ],
        _ => return None,
    })
}

/// Deterministic global slot by name (min index when multiple; stable across HashMap iteration).
fn global_index_by_name(
    global_names: &std::collections::BTreeMap<usize, String>,
    name: &str,
) -> Option<usize> {
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
        "crypto" => register_crypto_module(natives, globals, global_names, store, heap),
        "database_engine" => register_database_module(natives, globals, global_names, store, heap),
        "system" => register_system_module(natives, globals, global_names, store, heap),
        "debug" => register_debug_module(natives, globals, global_names, store, heap),
        "heapq" => register_heapq_module(natives, globals, global_names, store, heap),
        "pathfind" => register_pathfind_module(natives, globals, global_names, store, heap),
        "grid" => register_grid_module(natives, globals, global_names, store, heap),
        "websocket" => register_websocket_module(natives, globals, global_names, store, heap),
        "ws" => register_ws_module(natives, globals, global_names, store, heap),
        "web" => register_web_module(natives, globals, global_names, store, heap),
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
    plot_object.insert(
        "image".to_string(),
        Value::NativeFunction(plot_native_start + 0),
    );
    plot_object.insert(
        "window".to_string(),
        Value::NativeFunction(plot_native_start + 1),
    );
    plot_object.insert(
        "draw".to_string(),
        Value::NativeFunction(plot_native_start + 2),
    );
    plot_object.insert(
        "wait".to_string(),
        Value::NativeFunction(plot_native_start + 3),
    );
    plot_object.insert(
        "show".to_string(),
        Value::NativeFunction(plot_native_start + 4),
    );
    plot_object.insert(
        "show_grid".to_string(),
        Value::NativeFunction(plot_native_start + 5),
    );
    plot_object.insert(
        "subplots".to_string(),
        Value::NativeFunction(plot_native_start + 6),
    );
    plot_object.insert(
        "tight_layout".to_string(),
        Value::NativeFunction(plot_native_start + 7),
    );
    plot_object.insert(
        "xlabel".to_string(),
        Value::NativeFunction(plot_native_start + 12),
    );
    plot_object.insert(
        "ylabel".to_string(),
        Value::NativeFunction(plot_native_start + 13),
    );
    plot_object.insert(
        "line".to_string(),
        Value::NativeFunction(plot_native_start + 14),
    );
    plot_object.insert(
        "bar".to_string(),
        Value::NativeFunction(plot_native_start + 15),
    );
    plot_object.insert(
        "pie".to_string(),
        Value::NativeFunction(plot_native_start + 16),
    );
    plot_object.insert(
        "heatmap".to_string(),
        Value::NativeFunction(plot_native_start + 17),
    );
    // show_figure is handled by checking if argument is Figure in native_plot_show

    // Store axis method indices for later lookup
    // imshow = plot_native_start + 9, set_title = +10, axis = +11
    let axis_imshow_idx = plot_native_start + 9;
    let axis_set_title_idx = plot_native_start + 10;
    let axis_axis_idx = plot_native_start + 11;

    // Store in plot object for access (we'll use a special key)
    plot_object.insert(
        "__axis_imshow_idx".to_string(),
        Value::Number(axis_imshow_idx as f64),
    );
    plot_object.insert(
        "__axis_set_title_idx".to_string(),
        Value::Number(axis_set_title_idx as f64),
    );
    plot_object.insert(
        "__axis_axis_idx".to_string(),
        Value::Number(axis_axis_idx as f64),
    );

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
                if map_rc.borrow().str_key_contains("image") {
                    return Some(i);
                }
            }
            None
        });
        if let Some(idx) = plot_idx_opt {
            idx
        } else {
            let idx = globals.len();
            globals.push(GlobalSlot::Heap(store_value_arena(
                Value::legacy_object(plot_object.clone()),
                store,
                heap,
            )));
            global_names.insert(idx, "plot".to_string());
            idx
        }
    };

    globals[plot_index] = GlobalSlot::Heap(store_value_arena(
        Value::legacy_object(plot_object),
        store,
        heap,
    ));

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
    let settings_value = Value::legacy_object(settings_object);
    settings_env_object.insert("load_env".to_string(), load_env_fn);
    settings_env_object.insert("Settings".to_string(), settings_value.clone());
    settings_env_object.insert("settings".to_string(), settings_value);
    settings_env_object.insert(
        "Field".to_string(),
        Value::NativeFunction(settings_env_native_start + 2),
    );
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

    globals[settings_env_index] = GlobalSlot::Heap(store_value_arena(
        Value::legacy_object(settings_env_object),
        store,
        heap,
    ));

    Ok(())
}

fn register_debug_module(
    natives: &mut Vec<HostEntry>,
    globals: &mut Vec<GlobalSlot>,
    global_names: &mut std::collections::BTreeMap<usize, String>,
    store: &mut ValueStore,
    heap: &mut HeavyStore,
) -> Result<(), LangError> {
    use crate::vm::natives::native_debug_operators;

    let start = natives.len();
    natives.push(HostEntry::Extended(native_debug_operators));

    let mut debug_object = HashMap::new();
    debug_object.insert("operators".to_string(), Value::NativeFunction(start));

    let debug_index = if let Some(idx) = global_index_by_name(global_names, "debug") {
        if idx >= globals.len() {
            globals.resize(idx + 1, default_global_slot());
        }
        idx
    } else {
        let idx = globals.len();
        globals.push(default_global_slot());
        global_names.insert(idx, "debug".to_string());
        idx
    };

    globals[debug_index] = GlobalSlot::Heap(store_value_arena(
        Value::legacy_object(debug_object),
        store,
        heap,
    ));

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

    globals[uuid_index] = GlobalSlot::Heap(store_value_arena(
        Value::legacy_object(uuid_object),
        store,
        heap,
    ));

    Ok(())
}

fn register_crypto_module(
    natives: &mut Vec<HostEntry>,
    globals: &mut Vec<GlobalSlot>,
    global_names: &mut std::collections::BTreeMap<usize, String>,
    store: &mut ValueStore,
    heap: &mut HeavyStore,
) -> Result<(), LangError> {
    use crate::crypto::natives as crypto_natives;

    let crypto_start = natives.len();
    natives.push(HostEntry::Extended(crypto_natives::native_crypto_argon2_hash));
    natives.push(HostEntry::Extended(crypto_natives::native_crypto_argon2_verify));
    natives.push(HostEntry::Extended(crypto_natives::native_crypto_bcrypt_hash));
    natives.push(HostEntry::Extended(crypto_natives::native_crypto_bcrypt_verify));
    natives.push(HostEntry::Extended(crypto_natives::native_crypto_secure_compare));

    let s = crypto_start;
    let mut argon2_obj = HashMap::new();
    argon2_obj.insert("hash".to_string(), Value::NativeFunction(s + 0));
    argon2_obj.insert("verify".to_string(), Value::NativeFunction(s + 1));
    let mut bcrypt_obj = HashMap::new();
    bcrypt_obj.insert("hash".to_string(), Value::NativeFunction(s + 2));
    bcrypt_obj.insert("verify".to_string(), Value::NativeFunction(s + 3));

    let mut crypto_object = HashMap::new();
    crypto_object.insert(
        "Argon2".to_string(),
        Value::legacy_object(argon2_obj),
    );
    crypto_object.insert(
        "bcrypt".to_string(),
        Value::legacy_object(bcrypt_obj),
    );
    crypto_object.insert(
        "secure_compare".to_string(),
        Value::NativeFunction(s + 4),
    );

    let crypto_index = if let Some(idx) = global_index_by_name(global_names, "crypto") {
        if idx >= globals.len() {
            globals.resize(idx + 1, default_global_slot());
        }
        idx
    } else {
        let idx = globals.len();
        globals.push(default_global_slot());
        global_names.insert(idx, "crypto".to_string());
        idx
    };

    globals[crypto_index] = GlobalSlot::Heap(store_value_arena(
        Value::legacy_object(crypto_object),
        store,
        heap,
    ));

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
    use crate::database_engine::sqenum;

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
    natives.push(HostEntry::Extended(natives::native_engine_schemas));
    natives.push(HostEntry::Extended(natives::native_engine_tables));
    natives.push(HostEntry::Extended(natives::native_engine_views));
    natives.push(HostEntry::Extended(natives::native_engine_columns));
    natives.push(HostEntry::Extended(natives::native_engine_indexes));
    natives.push(HostEntry::Extended(natives::native_engine_primary_key));
    natives.push(HostEntry::Extended(natives::native_engine_foreign_keys));
    natives.push(HostEntry::Extended(natives::native_engine_inspect));
    natives.push(HostEntry::Extended(natives::native_engine_table));

    let sqenum_add_idx = natives.len();
    natives.push(HostEntry::Extended(sqenum::native_sqenum_add_member));
    natives.push(HostEntry::Extended(sqenum::native_sqenum_finalize));

    use crate::database_engine::validators as db_validators;
    let v0 = natives.len();
    natives.push(HostEntry::Extended(db_validators::native_validator_min_length));
    natives.push(HostEntry::Extended(db_validators::native_validator_max_length));
    natives.push(HostEntry::Extended(db_validators::native_validator_length_between));
    natives.push(HostEntry::Extended(db_validators::native_validator_regex));
    natives.push(HostEntry::Extended(db_validators::native_validator_email));
    natives.push(HostEntry::Extended(db_validators::native_validator_url));
    natives.push(HostEntry::Extended(db_validators::native_validator_username));
    natives.push(HostEntry::Extended(db_validators::native_validator_password_policy));
    natives.push(HostEntry::Extended(db_validators::native_validator_one_of));
    natives.push(HostEntry::Extended(db_validators::native_validator_min_value));
    natives.push(HostEntry::Extended(db_validators::native_validator_max_value));
    natives.push(HostEntry::Extended(db_validators::native_validator_range_value));
    natives.push(HostEntry::Extended(db_validators::native_validator_custom));

    let mut validators_obj = HashMap::new();
    validators_obj.insert("min_length".to_string(), Value::NativeFunction(v0));
    validators_obj.insert("max_length".to_string(), Value::NativeFunction(v0 + 1));
    validators_obj.insert("length_between".to_string(), Value::NativeFunction(v0 + 2));
    validators_obj.insert("regex".to_string(), Value::NativeFunction(v0 + 3));
    validators_obj.insert("email".to_string(), Value::NativeFunction(v0 + 4));
    validators_obj.insert("url".to_string(), Value::NativeFunction(v0 + 5));
    validators_obj.insert("username".to_string(), Value::NativeFunction(v0 + 6));
    validators_obj.insert("password_policy".to_string(), Value::NativeFunction(v0 + 7));
    validators_obj.insert("one_of".to_string(), Value::NativeFunction(v0 + 8));
    validators_obj.insert("min_value".to_string(), Value::NativeFunction(v0 + 9));
    validators_obj.insert("max_value".to_string(), Value::NativeFunction(v0 + 10));
    validators_obj.insert("range_value".to_string(), Value::NativeFunction(v0 + 11));
    validators_obj.insert("custom".to_string(), Value::NativeFunction(v0 + 12));

    let start = db_native_start;
    let mut database_object = HashMap::new();

    let mut sqenum_marker = HashMap::new();
    sqenum_marker.insert(
        "__class_name".to_string(),
        Value::String("SQLEnum".to_string()),
    );
    sqenum_marker.insert(
        crate::database_engine::sqenum::KEY_BUILTIN_SQENUM.to_string(),
        Value::Bool(true),
    );
    sqenum_marker.insert(
        crate::database_engine::sqenum::KEY_EXTENDS_SQENUM.to_string(),
        Value::Bool(true),
    );
    sqenum_marker.insert(
        "add_member".to_string(),
        Value::NativeFunction(sqenum_add_idx),
    );
    sqenum_marker.insert(
        "finalize".to_string(),
        Value::NativeFunction(sqenum_add_idx + 1),
    );
    database_object.insert(
        "SQLEnum".to_string(),
        Value::legacy_object(sqenum_marker),
    );

    database_object.insert("engine".to_string(), Value::NativeFunction(start + 0));
    database_object.insert("MetaData".to_string(), Value::NativeFunction(start + 4));
    database_object.insert("Column".to_string(), Value::NativeFunction(start + 5));
    database_object.insert("now_call".to_string(), Value::NativeFunction(start + 6));
    database_object.insert("select".to_string(), Value::NativeFunction(start + 7));
    database_object.insert(
        "DatabaseCluster".to_string(),
        Value::NativeFunction(start + 9),
    );
    // connect, execute, query, run are methods on engine - accessed via GetArrayElement on DatabaseEngine
    // add, get, names are methods on cluster - accessed via GetArrayElement on DatabaseCluster
    //
    // Re-export builtin type constructors for `from database_engine import int, str, ...` (tests / ORM).
    // Indices must match `native_registry::register_builtin_natives`.
    database_object.insert("int".to_string(), Value::NativeFunction(3));
    database_object.insert("float".to_string(), Value::NativeFunction(4));
    database_object.insert("bool".to_string(), Value::NativeFunction(5));
    database_object.insert("str".to_string(), Value::NativeFunction(6));
    database_object.insert("date".to_string(), Value::NativeFunction(10));
    database_object.insert(
        "validators".to_string(),
        Value::legacy_object(validators_obj),
    );

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

    globals[database_index] = GlobalSlot::Heap(store_value_arena(
        Value::legacy_object(database_object),
        store,
        heap,
    ));

    Ok(())
}

fn register_system_module(
    natives: &mut Vec<HostEntry>,
    globals: &mut Vec<GlobalSlot>,
    global_names: &mut std::collections::BTreeMap<usize, String>,
    store: &mut ValueStore,
    heap: &mut HeavyStore,
) -> Result<(), LangError> {
    use crate::system::natives;

    let start = natives.len();
    natives.push(HostEntry::Extended(natives::native_system_get_os));
    natives.push(HostEntry::Extended(natives::native_system_get_arch));
    natives.push(HostEntry::Extended(natives::native_system_get_os_version));
    natives.push(HostEntry::Extended(natives::native_system_get_hostname));
    natives.push(HostEntry::Extended(natives::native_system_get_username));
    natives.push(HostEntry::Extended(natives::native_system_get_home_dir));
    natives.push(HostEntry::Extended(natives::native_system_get_temp_dir));
    natives.push(HostEntry::Extended(natives::native_system_env_get));
    natives.push(HostEntry::Extended(natives::native_system_env_set));
    natives.push(HostEntry::Extended(
        natives::native_system_get_datacode_version,
    ));
    natives.push(HostEntry::Extended(natives::native_system_get_vm_version));
    natives.push(HostEntry::Extended(natives::native_system_get_module_path));
    natives.push(HostEntry::Extended(natives::native_system_get_venv_path));
    natives.push(HostEntry::Extended(
        natives::native_system_get_loaded_modules,
    ));
    natives.push(HostEntry::Extended(natives::native_system_get_registry_url));
    natives.push(HostEntry::Extended(natives::native_system_cpu_count));
    natives.push(HostEntry::Extended(natives::native_system_memory_total));
    natives.push(HostEntry::Extended(natives::native_system_memory_free));
    natives.push(HostEntry::Extended(natives::native_system_gpu_count));
    natives.push(HostEntry::Extended(natives::native_system_gpu_info));
    natives.push(HostEntry::Extended(natives::native_system_time_now));
    natives.push(HostEntry::Extended(natives::native_system_sleep_ms));
    natives.push(HostEntry::Extended(natives::native_system_uptime));
    natives.push(HostEntry::Extended(natives::native_system_has_permission));
    natives.push(HostEntry::Extended(
        natives::native_system_request_permission,
    ));
    natives.push(HostEntry::Extended(natives::native_system_log_info));
    natives.push(HostEntry::Extended(natives::native_system_log_warn));
    natives.push(HostEntry::Extended(natives::native_system_log_error));
    natives.push(HostEntry::Extended(natives::native_system_log_debug));
    natives.push(HostEntry::Extended(natives::native_system_net_get_ip));
    natives.push(HostEntry::Extended(
        natives::native_system_net_get_interfaces,
    ));
    natives.push(HostEntry::Extended(natives::native_system_process_exec));
    natives.push(HostEntry::Extended(natives::native_system_fs_read));
    natives.push(HostEntry::Extended(natives::native_system_fs_write));
    natives.push(HostEntry::Extended(natives::native_system_get_dpm_env_base));
    natives.push(HostEntry::Extended(natives::native_system_get_dpm_env_root));
    natives.push(HostEntry::Extended(
        natives::native_system_time_monotonic_ms,
    ));
    natives.push(HostEntry::Extended(natives::native_system_time_perf_counter));

    // process compute API (append only — keep native indices stable)
    use crate::system::process_compute as proc;
    natives.push(HostEntry::Extended(proc::native_process_get_device));
    natives.push(HostEntry::Extended(proc::native_process_set_device));
    natives.push(HostEntry::Extended(proc::native_process_get_gpu_min_size));
    natives.push(HostEntry::Extended(proc::native_process_set_gpu_min_size));
    natives.push(HostEntry::Extended(proc::native_process_auto_device));
    natives.push(HostEntry::Extended(proc::native_process_has_gpu));
    natives.push(HostEntry::Extended(proc::native_process_has_cuda));
    natives.push(HostEntry::Extended(proc::native_process_has_metal));
    natives.push(HostEntry::Extended(proc::native_process_info));
    natives.push(HostEntry::Extended(proc::native_process_vector_add));
    natives.push(HostEntry::Extended(proc::native_process_vector_mul));
    natives.push(HostEntry::Extended(proc::native_process_run));
    natives.push(HostEntry::Extended(natives::native_system_trim_allocator));

    let mut env = HashMap::new();
    env.insert("get_os".to_string(), Value::NativeFunction(start + 0));
    env.insert("get_arch".to_string(), Value::NativeFunction(start + 1));
    env.insert("get_version".to_string(), Value::NativeFunction(start + 2));
    env.insert("get_hostname".to_string(), Value::NativeFunction(start + 3));
    env.insert("get_username".to_string(), Value::NativeFunction(start + 4));
    env.insert("get_home_dir".to_string(), Value::NativeFunction(start + 5));
    env.insert("get_temp_dir".to_string(), Value::NativeFunction(start + 6));
    env.insert("get".to_string(), Value::NativeFunction(start + 7));
    env.insert("set_env".to_string(), Value::NativeFunction(start + 8));

    let mut runtime = HashMap::new();
    runtime.insert(
        "get_datacode_version".to_string(),
        Value::NativeFunction(start + 9),
    );
    runtime.insert(
        "get_vm_version".to_string(),
        Value::NativeFunction(start + 10),
    );
    runtime.insert(
        "get_module_path".to_string(),
        Value::NativeFunction(start + 11),
    );
    runtime.insert(
        "get_venv_path".to_string(),
        Value::NativeFunction(start + 12),
    );
    runtime.insert(
        "get_loaded_modules".to_string(),
        Value::NativeFunction(start + 13),
    );
    runtime.insert(
        "get_registry_url".to_string(),
        Value::NativeFunction(start + 14),
    );
    runtime.insert(
        "get_dpm_env_base".to_string(),
        Value::NativeFunction(start + 34),
    );
    runtime.insert(
        "get_dpm_env_root".to_string(),
        Value::NativeFunction(start + 35),
    );
    runtime.insert(
        "trim_allocator".to_string(),
        Value::NativeFunction(start + 50),
    );

    let mut hardware = HashMap::new();
    hardware.insert("cpu_count".to_string(), Value::NativeFunction(start + 15));
    hardware.insert(
        "memory_total".to_string(),
        Value::NativeFunction(start + 16),
    );
    hardware.insert("memory_free".to_string(), Value::NativeFunction(start + 17));
    hardware.insert("gpu_count".to_string(), Value::NativeFunction(start + 18));
    hardware.insert("gpu_info".to_string(), Value::NativeFunction(start + 19));

    let mut time = HashMap::new();
    time.insert("now".to_string(), Value::NativeFunction(start + 20));
    time.insert("sleep".to_string(), Value::NativeFunction(start + 21));
    time.insert("uptime".to_string(), Value::NativeFunction(start + 22));
    time.insert(
        "monotonic_ms".to_string(),
        Value::NativeFunction(start + 36),
    );
    time.insert(
        "perf_counter".to_string(),
        Value::NativeFunction(start + 37),
    );

    let mut permissions = HashMap::new();
    permissions.insert(
        "has_permission".to_string(),
        Value::NativeFunction(start + 23),
    );
    permissions.insert(
        "request_permission".to_string(),
        Value::NativeFunction(start + 24),
    );

    let mut log = HashMap::new();
    log.insert("log_info".to_string(), Value::NativeFunction(start + 25));
    log.insert("log_warn".to_string(), Value::NativeFunction(start + 26));
    log.insert("log_error".to_string(), Value::NativeFunction(start + 27));
    log.insert("debug".to_string(), Value::NativeFunction(start + 28));

    let mut net = HashMap::new();
    net.insert("get_ip".to_string(), Value::NativeFunction(start + 29));
    net.insert(
        "get_interfaces".to_string(),
        Value::NativeFunction(start + 30),
    );

    let mut process = HashMap::new();
    process.insert("exec".to_string(), Value::NativeFunction(start + 31));
    process.insert("cpu".to_string(), proc::device_constant(crate::compute::device::DeviceKind::Cpu));
    process.insert("gpu".to_string(), proc::device_constant(crate::compute::device::DeviceKind::Gpu));
    process.insert("cuda".to_string(), proc::device_constant(crate::compute::device::DeviceKind::Cuda));
    process.insert("metal".to_string(), proc::device_constant(crate::compute::device::DeviceKind::Metal));
    process.insert("auto".to_string(), proc::device_constant(crate::compute::device::DeviceKind::Auto));
    process.insert("get_device".to_string(), Value::NativeFunction(start + 38));
    process.insert("set_device".to_string(), Value::NativeFunction(start + 39));
    process.insert("get_gpu_min_size".to_string(), Value::NativeFunction(start + 40));
    process.insert("set_gpu_min_size".to_string(), Value::NativeFunction(start + 41));
    process.insert("auto_device".to_string(), Value::NativeFunction(start + 42));
    process.insert("has_gpu".to_string(), Value::NativeFunction(start + 43));
    process.insert("has_cuda".to_string(), Value::NativeFunction(start + 44));
    process.insert("has_metal".to_string(), Value::NativeFunction(start + 45));
    process.insert("info".to_string(), Value::NativeFunction(start + 46));
    process.insert("vector_add".to_string(), Value::NativeFunction(start + 47));
    process.insert("vector_mul".to_string(), Value::NativeFunction(start + 48));
    process.insert("run".to_string(), Value::NativeFunction(start + 49));

    let mut fs = HashMap::new();
    fs.insert("read".to_string(), Value::NativeFunction(start + 32));
    fs.insert("write".to_string(), Value::NativeFunction(start + 33));

    let mut system_object = HashMap::new();
    system_object.insert("env".to_string(), Value::legacy_object(env));
    system_object.insert(
        "runtime".to_string(),
        Value::legacy_object(runtime),
    );
    system_object.insert(
        "hardware".to_string(),
        Value::legacy_object(hardware),
    );
    system_object.insert(
        "time".to_string(),
        Value::legacy_object(time),
    );
    system_object.insert(
        "permissions".to_string(),
        Value::legacy_object(permissions),
    );
    system_object.insert("log".to_string(), Value::legacy_object(log));
    system_object.insert("net".to_string(), Value::legacy_object(net));
    system_object.insert(
        "process".to_string(),
        Value::legacy_object(process),
    );
    system_object.insert("fs".to_string(), Value::legacy_object(fs));

    let system_index = if let Some(idx) = global_index_by_name(global_names, "system") {
        if idx >= globals.len() {
            globals.resize(idx + 1, default_global_slot());
        }
        idx
    } else {
        let idx = globals.len();
        globals.push(default_global_slot());
        global_names.insert(idx, "system".to_string());
        idx
    };

    globals[system_index] = GlobalSlot::Heap(store_value_arena(
        Value::legacy_object(system_object),
        store,
        heap,
    ));

    Ok(())
}

fn register_heapq_module(
    natives: &mut Vec<HostEntry>,
    globals: &mut Vec<GlobalSlot>,
    global_names: &mut std::collections::BTreeMap<usize, String>,
    store: &mut ValueStore,
    heap: &mut HeavyStore,
) -> Result<(), LangError> {
    use crate::heapq::natives as heapq_natives;

    let start = natives.len();
    natives.push(HostEntry::Extended(heapq_natives::native_heapq_heappush));
    natives.push(HostEntry::Extended(heapq_natives::native_heapq_heappop));
    natives.push(HostEntry::Extended(heapq_natives::native_heapq_heapify));
    natives.push(HostEntry::Extended(heapq_natives::native_heapq_heappeek));
    natives.push(HostEntry::Extended(heapq_natives::native_heapq_heapreplace));
    natives.push(HostEntry::Extended(heapq_natives::native_heapq_heap_clear));

    let mut heapq_object = HashMap::new();
    heapq_object.insert("heappush".to_string(), Value::NativeFunction(start + 0));
    heapq_object.insert("heappop".to_string(), Value::NativeFunction(start + 1));
    heapq_object.insert("heapify".to_string(), Value::NativeFunction(start + 2));
    heapq_object.insert("heappeek".to_string(), Value::NativeFunction(start + 3));
    heapq_object.insert("heapreplace".to_string(), Value::NativeFunction(start + 4));
    heapq_object.insert("heap_clear".to_string(), Value::NativeFunction(start + 5));

    let heapq_index = if let Some(idx) = global_index_by_name(global_names, "heapq") {
        if idx >= globals.len() {
            globals.resize(idx + 1, default_global_slot());
        }
        idx
    } else {
        let idx = globals.len();
        globals.push(default_global_slot());
        global_names.insert(idx, "heapq".to_string());
        idx
    };

    globals[heapq_index] = GlobalSlot::Heap(store_value_arena(
        Value::legacy_object(heapq_object),
        store,
        heap,
    ));

    Ok(())
}

fn register_pathfind_module(
    natives: &mut Vec<HostEntry>,
    globals: &mut Vec<GlobalSlot>,
    global_names: &mut std::collections::BTreeMap<usize, String>,
    store: &mut ValueStore,
    heap: &mut HeavyStore,
) -> Result<(), LangError> {
    use crate::pathfind::natives as pathfind_natives;

    let start = natives.len();
    natives.push(HostEntry::Extended(
        pathfind_natives::native_pathfind_astar_grid,
    ));

    let mut pathfind_object = HashMap::new();
    pathfind_object.insert(
        "astar_grid".to_string(),
        Value::NativeFunction(start + 0),
    );

    let pathfind_index = if let Some(idx) = global_index_by_name(global_names, "pathfind") {
        if idx >= globals.len() {
            globals.resize(idx + 1, default_global_slot());
        }
        idx
    } else {
        let idx = globals.len();
        globals.push(default_global_slot());
        global_names.insert(idx, "pathfind".to_string());
        idx
    };

    globals[pathfind_index] = GlobalSlot::Heap(store_value_arena(
        Value::legacy_object(pathfind_object),
        store,
        heap,
    ));

    Ok(())
}

fn register_grid_module(
    natives: &mut Vec<HostEntry>,
    globals: &mut Vec<GlobalSlot>,
    global_names: &mut std::collections::BTreeMap<usize, String>,
    store: &mut ValueStore,
    heap: &mut HeavyStore,
) -> Result<(), LangError> {
    use crate::grid::natives as grid_natives;

    let start = natives.len();
    natives.push(HostEntry::Extended(grid_natives::native_grid_alloc_i32));
    natives.push(HostEntry::Extended(grid_natives::native_grid_alloc_u8));
    natives.push(HostEntry::Extended(grid_natives::native_grid_fill_i32));
    natives.push(HostEntry::Extended(grid_natives::native_grid_fill_u8));
    natives.push(HostEntry::Extended(grid_natives::native_grid_get_i32));
    natives.push(HostEntry::Extended(grid_natives::native_grid_set_i32));
    natives.push(HostEntry::Extended(grid_natives::native_grid_get_u8));
    natives.push(HostEntry::Extended(grid_natives::native_grid_set_u8));
    natives.push(HostEntry::Extended(grid_natives::native_grid_test_blocked));
    natives.push(HostEntry::Extended(grid_natives::native_grid_set_blocked));
    natives.push(HostEntry::Extended(grid_natives::native_grid_shrink));
    natives.push(HostEntry::Extended(grid_natives::native_grid_bitmap_bytes));
    natives.push(HostEntry::Extended(grid_natives::native_grid_astar));
    natives.push(HostEntry::Extended(grid_natives::native_grid_bitmap_from_ids));
    natives.push(HostEntry::Extended(grid_natives::native_grid_store_len));
    natives.push(HostEntry::Extended(grid_natives::native_grid_astar_from_set));
    natives.push(HostEntry::Extended(grid_natives::native_grid_heap_alloc));
    natives.push(HostEntry::Extended(grid_natives::native_grid_heap_clear));
    natives.push(HostEntry::Extended(grid_natives::native_grid_heap_push));
    natives.push(HostEntry::Extended(grid_natives::native_grid_heap_pop));
    natives.push(HostEntry::Extended(grid_natives::native_grid_heap_len));
    natives.push(HostEntry::Extended(grid_natives::native_grid_shrink_all));
    natives.push(HostEntry::Extended(grid_natives::native_grid_astar_step));

    let mut grid_object = HashMap::new();
    grid_object.insert("alloc_i32".to_string(), Value::NativeFunction(start + 0));
    grid_object.insert("alloc_u8".to_string(), Value::NativeFunction(start + 1));
    grid_object.insert("fill_i32".to_string(), Value::NativeFunction(start + 2));
    grid_object.insert("fill_u8".to_string(), Value::NativeFunction(start + 3));
    grid_object.insert("get_i32".to_string(), Value::NativeFunction(start + 4));
    grid_object.insert("set_i32".to_string(), Value::NativeFunction(start + 5));
    grid_object.insert("get_u8".to_string(), Value::NativeFunction(start + 6));
    grid_object.insert("set_u8".to_string(), Value::NativeFunction(start + 7));
    grid_object.insert("test_blocked".to_string(), Value::NativeFunction(start + 8));
    grid_object.insert("set_blocked".to_string(), Value::NativeFunction(start + 9));
    grid_object.insert("shrink".to_string(), Value::NativeFunction(start + 10));
    grid_object.insert("bitmap_bytes".to_string(), Value::NativeFunction(start + 11));
    grid_object.insert("astar".to_string(), Value::NativeFunction(start + 12));
    grid_object.insert(
        "bitmap_from_ids".to_string(),
        Value::NativeFunction(start + 13),
    );
    grid_object.insert("store_len".to_string(), Value::NativeFunction(start + 14));
    grid_object.insert(
        "astar_from_set".to_string(),
        Value::NativeFunction(start + 15),
    );
    grid_object.insert("heap_alloc".to_string(), Value::NativeFunction(start + 16));
    grid_object.insert("heap_clear".to_string(), Value::NativeFunction(start + 17));
    grid_object.insert("heap_push".to_string(), Value::NativeFunction(start + 18));
    grid_object.insert("heap_pop".to_string(), Value::NativeFunction(start + 19));
    grid_object.insert("heap_len".to_string(), Value::NativeFunction(start + 20));
    grid_object.insert("shrink_all".to_string(), Value::NativeFunction(start + 21));
    grid_object.insert("astar_step".to_string(), Value::NativeFunction(start + 22));

    let grid_index = if let Some(idx) = global_index_by_name(global_names, "grid") {
        if idx >= globals.len() {
            globals.resize(idx + 1, default_global_slot());
        }
        idx
    } else {
        let idx = globals.len();
        globals.push(default_global_slot());
        global_names.insert(idx, "grid".to_string());
        idx
    };

    globals[grid_index] = GlobalSlot::Heap(store_value_arena(
        Value::legacy_object(grid_object),
        store,
        heap,
    ));

    Ok(())
}

fn register_websocket_module(
    natives: &mut Vec<HostEntry>,
    globals: &mut Vec<GlobalSlot>,
    global_names: &mut std::collections::BTreeMap<usize, String>,
    store: &mut ValueStore,
    heap: &mut HeavyStore,
) -> Result<(), LangError> {
    use crate::websocket::natives as ws_natives;

    let start = natives.len();
    natives.push(HostEntry::Extended(ws_natives::native_websocket_configure));
    natives.push(HostEntry::Extended(ws_natives::native_websocket_disable_builtin));
    natives.push(HostEntry::Extended(ws_natives::native_websocket_enable_builtin));

    let mut ws_object = HashMap::new();
    ws_object.insert("configure".to_string(), Value::NativeFunction(start));
    ws_object.insert(
        "disable_builtin".to_string(),
        Value::NativeFunction(start + 1),
    );
    ws_object.insert(
        "enable_builtin".to_string(),
        Value::NativeFunction(start + 2),
    );

    let ws_index = if let Some(idx) = global_index_by_name(global_names, "websocket") {
        if idx >= globals.len() {
            globals.resize(idx + 1, default_global_slot());
        }
        idx
    } else {
        let idx = globals.len();
        globals.push(default_global_slot());
        global_names.insert(idx, "websocket".to_string());
        idx
    };

    globals[ws_index] = GlobalSlot::Heap(store_value_arena(
        Value::legacy_object(ws_object),
        store,
        heap,
    ));

    Ok(())
}

fn register_ws_module(
    natives: &mut Vec<HostEntry>,
    globals: &mut Vec<GlobalSlot>,
    global_names: &mut std::collections::BTreeMap<usize, String>,
    store: &mut ValueStore,
    heap: &mut HeavyStore,
) -> Result<(), LangError> {
    use crate::websocket::ws_natives;

    let start = natives.len();
    natives.push(HostEntry::Extended(ws_natives::native_ws_tables));
    natives.push(HostEntry::Extended(ws_natives::native_ws_has_table));
    natives.push(HostEntry::Extended(ws_natives::native_ws_source_table));
    natives.push(HostEntry::Extended(ws_natives::native_ws_assets));
    natives.push(HostEntry::Extended(ws_natives::native_ws_has_asset));
    natives.push(HostEntry::Extended(ws_natives::native_ws_metadata));
    natives.push(HostEntry::Extended(ws_natives::native_ws_metadata_get));
    natives.push(HostEntry::Extended(ws_natives::native_ws_package_info));
    natives.push(HostEntry::Extended(ws_natives::native_ws_content_assets));
    natives.push(HostEntry::Extended(ws_natives::native_ws_has_content_asset));
    natives.push(HostEntry::Extended(ws_natives::native_ws_content_asset));

    let mut ws_object = HashMap::new();
    ws_object.insert("tables".to_string(), Value::NativeFunction(start));
    ws_object.insert("has_table".to_string(), Value::NativeFunction(start + 1));
    ws_object.insert("source_table".to_string(), Value::NativeFunction(start + 2));
    ws_object.insert("assets".to_string(), Value::NativeFunction(start + 3));
    ws_object.insert("has_asset".to_string(), Value::NativeFunction(start + 4));
    ws_object.insert("metadata".to_string(), Value::NativeFunction(start + 5));
    ws_object.insert("metadata_get".to_string(), Value::NativeFunction(start + 6));
    ws_object.insert("package_info".to_string(), Value::NativeFunction(start + 7));
    ws_object.insert("content_assets".to_string(), Value::NativeFunction(start + 8));
    ws_object.insert(
        "has_content_asset".to_string(),
        Value::NativeFunction(start + 9),
    );
    ws_object.insert("content_asset".to_string(), Value::NativeFunction(start + 10));

    let ws_index = if let Some(idx) = global_index_by_name(global_names, "ws") {
        if idx >= globals.len() {
            globals.resize(idx + 1, default_global_slot());
        }
        idx
    } else {
        let idx = globals.len();
        globals.push(default_global_slot());
        global_names.insert(idx, "ws".to_string());
        idx
    };

    globals[ws_index] = GlobalSlot::Heap(store_value_arena(
        Value::legacy_object(ws_object),
        store,
        heap,
    ));

    Ok(())
}

fn register_web_module(
    natives: &mut Vec<HostEntry>,
    globals: &mut Vec<GlobalSlot>,
    global_names: &mut std::collections::BTreeMap<usize, String>,
    store: &mut ValueStore,
    heap: &mut HeavyStore,
) -> Result<(), LangError> {
    use crate::web::natives as web_natives;

    let start = natives.len();
    // http: 0..7
    natives.push(HostEntry::Extended(web_natives::native_http_get));
    natives.push(HostEntry::Extended(web_natives::native_http_post));
    natives.push(HostEntry::Extended(web_natives::native_http_put));
    natives.push(HostEntry::Extended(web_natives::native_http_patch));
    natives.push(HostEntry::Extended(web_natives::native_http_delete));
    natives.push(HostEntry::Extended(web_natives::native_http_head));
    natives.push(HostEntry::Extended(web_natives::native_http_options));
    natives.push(HostEntry::Extended(web_natives::native_http_get_table));
    // browser.open: 8
    natives.push(HostEntry::Extended(web_natives::native_browser_open));
    // page methods: 9..25
    natives.push(HostEntry::Extended(web_natives::native_page_goto));
    natives.push(HostEntry::Extended(web_natives::native_page_close));
    natives.push(HostEntry::Extended(web_natives::native_page_click));
    natives.push(HostEntry::Extended(web_natives::native_page_type));
    natives.push(HostEntry::Extended(web_natives::native_page_fill));
    natives.push(HostEntry::Extended(web_natives::native_page_clear));
    natives.push(HostEntry::Extended(web_natives::native_page_select));
    natives.push(HostEntry::Extended(web_natives::native_page_text));
    natives.push(HostEntry::Extended(web_natives::native_page_html));
    natives.push(HostEntry::Extended(web_natives::native_page_screenshot));
    natives.push(HostEntry::Extended(web_natives::native_page_wait));
    natives.push(HostEntry::Extended(web_natives::native_page_wait_for));
    natives.push(HostEntry::Extended(web_natives::native_page_wait_for_navigation));
    natives.push(HostEntry::Extended(web_natives::native_page_find));
    natives.push(HostEntry::Extended(web_natives::native_page_find_all));
    natives.push(HostEntry::Extended(web_natives::native_page_set_cookie));
    natives.push(HostEntry::Extended(web_natives::native_page_delete_cookie));
    // element methods: 26..32
    natives.push(HostEntry::Extended(web_natives::native_element_click));
    natives.push(HostEntry::Extended(web_natives::native_element_text));
    natives.push(HostEntry::Extended(web_natives::native_element_html));
    natives.push(HostEntry::Extended(web_natives::native_element_type));
    natives.push(HostEntry::Extended(web_natives::native_element_fill));
    natives.push(HostEntry::Extended(web_natives::native_element_clear));
    natives.push(HostEntry::Extended(web_natives::native_element_attr));
    // data: 33..34
    natives.push(HostEntry::Extended(web_natives::native_data_table));
    natives.push(HostEntry::Extended(web_natives::native_data_extract));

    let mut http = HashMap::new();
    http.insert("get".to_string(), Value::NativeFunction(start));
    http.insert("post".to_string(), Value::NativeFunction(start + 1));
    http.insert("put".to_string(), Value::NativeFunction(start + 2));
    http.insert("patch".to_string(), Value::NativeFunction(start + 3));
    http.insert("delete".to_string(), Value::NativeFunction(start + 4));
    http.insert("head".to_string(), Value::NativeFunction(start + 5));
    http.insert("options".to_string(), Value::NativeFunction(start + 6));
    http.insert("get_table".to_string(), Value::NativeFunction(start + 7));

    let mut browser = HashMap::new();
    browser.insert("open".to_string(), Value::NativeFunction(start + 8));

    let mut data = HashMap::new();
    data.insert("table".to_string(), Value::NativeFunction(start + 33));
    data.insert("extract".to_string(), Value::NativeFunction(start + 34));

    let mut web_object = HashMap::new();
    web_object.insert("http".to_string(), Value::legacy_object(http));
    web_object.insert("browser".to_string(), Value::legacy_object(browser));
    web_object.insert("data".to_string(), Value::legacy_object(data));

    let web_index = if let Some(idx) = global_index_by_name(global_names, "web") {
        if idx >= globals.len() {
            globals.resize(idx + 1, default_global_slot());
        }
        idx
    } else {
        let idx = globals.len();
        globals.push(default_global_slot());
        global_names.insert(idx, "web".to_string());
        idx
    };

    globals[web_index] = GlobalSlot::Heap(store_value_arena(
        Value::legacy_object(web_object),
        store,
        heap,
    ));

    Ok(())
}
