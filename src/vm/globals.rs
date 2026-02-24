// Global variables operations for VM (globals as Vec<GlobalSlot>)

use crate::common::value_store::{ValueStore, ValueCell};
use crate::vm::global_slot::{GlobalSlot, default_global_slot};

/// Канонические имена встроенных глобалов по индексу 0..75 (для merge_globals_from: не перезаписывать правильное значение ошибочным).
pub const BUILTIN_GLOBAL_NAMES: [&str; 75] = [
    "print", "len", "range", "int", "float", "bool", "str", "array", "typeof", "isinstance",
    "date", "money", "path", "path_name", "path_parent", "path_exists", "path_is_file", "path_is_dir",
    "path_extension", "path_stem", "path_len", "abs", "sqrt", "pow", "min", "max", "round",
    "upper", "lower", "trim", "split", "join", "contains", "isupper", "islower", "push", "pop", "unique", "reverse",
    "sort", "sum", "average", "count", "any", "all", "table", "read_file", "table_info", "table_head",
    "table_tail", "table_select", "table_sort", "table_where", "show_table", "merge_tables", "now",
    "getcwd", "list_files", "inner_join", "left_join", "right_join", "full_join", "cross_join",
    "semi_join", "anti_join", "zip_join", "asof_join", "apply_join", "join_on", "table_suffixes",
    "relate", "primary_key", "enum", "Table", "array_with_capacity",
];

/// Возвращает каноническое имя встроенной глобальной переменной по индексу (0..75).
pub fn builtin_global_name(index: usize) -> Option<&'static str> {
    (index < 75).then(|| BUILTIN_GLOBAL_NAMES[index])
}

/// Возвращает канонический индекс встроенной глобальной переменной по имени (для set_functions).
pub fn builtin_global_index(name: &str) -> Option<usize> {
    BUILTIN_GLOBAL_NAMES
        .iter()
        .position(|&n| n == name)
}

/// Регистрирует нативные функции в глобальных переменных (GlobalSlot::Heap(ValueId))
pub fn register_native_globals(
    globals: &mut Vec<GlobalSlot>,
    global_names: &mut std::collections::HashMap<usize, String>,
    store: &mut ValueStore,
) {
    let max_global_index = global_names.keys().max().copied().unwrap_or(74);
    let min_size = (75).max(max_global_index + 1);
    globals.resize(min_size, default_global_slot());

    for (index, name) in BUILTIN_GLOBAL_NAMES.iter().enumerate() {
        globals[index] = GlobalSlot::Heap(store.allocate(ValueCell::NativeFunction(index)));
        if !global_names.contains_key(&index) {
            global_names.insert(index, (*name).to_string());
        }
    }
}

/// Заполняет имена глобальных переменных из chunk.
/// Does not overwrite: (1) builtin slots 0..74 with a different name; (2) any existing name; (3) never insert at idx >= 75 (VM already has 75+ from ensure_globals_from_chunk).
pub fn merge_global_names(
    global_names: &mut std::collections::HashMap<usize, String>,
    explicit_global_names: &mut std::collections::HashMap<usize, String>,
    chunk_global_names: &std::collections::HashMap<usize, String>,
    chunk_explicit_global_names: &std::collections::HashMap<usize, String>,
) {
    const BUILTIN_END: usize = 75;
    for (idx, name) in chunk_global_names {
        // Never merge high indices (75+): chunk may already be patched; VM slots 75+ are from ensure_globals_from_chunk and must not be overwritten.
        if *idx >= BUILTIN_END {
            continue;
        }
        // Do not overwrite builtin indices 0..74 with a different name (would break "print", "len", "ml", etc.).
        if let Some(existing) = global_names.get(idx) {
            if existing != name {
                continue;
            }
        }
        // If this name already exists at any index, do not insert (would create duplicate or wrong mapping).
        if global_names.values().any(|n| n == name) {
            continue;
        }
        global_names.insert(*idx, name.clone());
    }
    for (idx, name) in chunk_explicit_global_names {
        if *idx >= BUILTIN_END {
            continue;
        }
        if let Some(existing) = explicit_global_names.get(idx) {
            if existing != name {
                continue;
            }
        }
        if explicit_global_names.values().any(|n| n == name) {
            continue;
        }
        explicit_global_names.insert(*idx, name.clone());
    }
}
