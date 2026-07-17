// Global variables operations for VM (globals as Vec<GlobalSlot>)

use crate::common::value_store::{ValueCell, ValueStore};
use crate::vm::global_slot::{default_global_slot, GlobalSlot};

/// Количество встроенных глобалов (индексы `0..BUILTIN_GLOBAL_COUNT`).
pub const BUILTIN_GLOBAL_COUNT: usize = 117;

/// Канонические имена встроенных глобалов по индексу `0..BUILTIN_GLOBAL_COUNT` (для legacy `legacy_merge::merge_globals_from`: не перезаписывать правильное значение ошибочным).
pub const BUILTIN_GLOBAL_NAMES: [&str; BUILTIN_GLOBAL_COUNT] = [
    "print",
    "len",
    "range",
    "int",
    "float",
    "bool",
    "str",
    "array",
    "typeof",
    "isinstance",
    "date",
    "money",
    "path",
    "path_name",
    "path_parent",
    "path_exists",
    "path_is_file",
    "path_is_dir",
    "path_extension",
    "path_stem",
    "path_len",
    "abs",
    "sqrt",
    "pow",
    "min",
    "max",
    "round",
    "ceil",
    "floor",
    "upper",
    "lower",
    "trim",
    "split",
    "join",
    "contains",
    "starts_with",
    "ends_with",
    "isupper",
    "islower",
    "replace",
    "capitalize",
    "push",
    "pop",
    "unique",
    "reverse",
    "sort",
    "sum",
    "average",
    "count",
    "any",
    "all",
    "table",
    "read_file",
    "read_file_bin",
    "table_info",
    "table_head",
    "table_tail",
    "table_select",
    "table_sort",
    "table_where",
    "table_drop_nulls",
    "table_replace_nulls",
    "table_rename",
    "table_drop_column",
    "table_add_column",
    "table_map",
    "table_split_column",
    "table_join_columns",
    "show_table",
    "merge_tables",
    "now",
    "getcwd",
    "list_files",
    "inner_join",
    "left_join",
    "right_join",
    "full_join",
    "cross_join",
    "semi_join",
    "anti_join",
    "zip_join",
    "asof_join",
    "apply_join",
    "join_on",
    "table_suffixes",
    "relate",
    "primary_key",
    "enum",
    "Table",
    "array_with_capacity",
    "map",
    "filter",
    "reduce",
    "sha256",
    "sha512",
    "hmac_sha256",
    "hmac_sha512",
    "random_bytes",
    "random_int",
    "random_seed",
    "random",
    "date_to_unix",
    "parse_date",
    "format_date",
    "duration",
    "set",
    "divmod",
    "isinf",
    "copy",
    "ord",
    "table_row_number",
    "table_distinct",
    "table_value_map",
    "table_aggregate",
    "table_aggregate_group",
    "archive",
    "datasource",
];

/// Возвращает каноническое имя встроенной глобальной переменной по индексу (`0..BUILTIN_GLOBAL_COUNT`).
pub fn builtin_global_name(index: usize) -> Option<&'static str> {
    (index < BUILTIN_GLOBAL_COUNT).then(|| BUILTIN_GLOBAL_NAMES[index])
}

/// Additive builtins registered above `BUILTIN_GLOBAL_COUNT` (chunk global index != native index).
pub fn extended_builtin_native_index(name: &str) -> Option<usize> {
    use crate::vm::native_indices::builtin;
    match name {
        "read" => Some(builtin::READ_FILE),
        "save" => Some(builtin::SAVE),
        "save_tables_sqlite" => Some(builtin::SAVE_TABLES_SQLITE),
        _ => None,
    }
}

/// Возвращает канонический индекс встроенной глобальной переменной по имени (для set_functions).
pub fn builtin_global_index(name: &str) -> Option<usize> {
    BUILTIN_GLOBAL_NAMES
        .iter()
        .position(|&n| n == name)
        .or_else(|| extended_builtin_native_index(name))
}

/// Регистрирует нативные функции в глобальных переменных (GlobalSlot::Heap(ValueId))
pub fn register_native_globals(
    globals: &mut Vec<GlobalSlot>,
    global_names: &mut std::collections::BTreeMap<usize, String>,
    store: &mut ValueStore,
) {
    let max_global_index = global_names
        .keys()
        .max()
        .copied()
        .unwrap_or(BUILTIN_GLOBAL_COUNT - 1);
    let min_size = (BUILTIN_GLOBAL_COUNT).max(max_global_index + 1);
    globals.resize(min_size, default_global_slot());

    for (index, name) in BUILTIN_GLOBAL_NAMES.iter().enumerate() {
        globals[index] = GlobalSlot::Heap(store.allocate(ValueCell::NativeFunction(index)));
        global_names
            .entry(index)
            .or_insert_with(|| (*name).to_string());
    }
}

/// Заполняет имена глобальных переменных из chunk.
/// Does not overwrite: (1) builtin slots below `BUILTIN_GLOBAL_COUNT` with a different name; (2) any existing name; (3) never insert at idx >= `BUILTIN_GLOBAL_COUNT` (VM already has high slots from ensure_globals_from_chunk).
pub fn merge_global_names(
    global_names: &mut std::collections::BTreeMap<usize, String>,
    explicit_global_names: &mut std::collections::BTreeMap<usize, String>,
    chunk_global_names: &std::collections::BTreeMap<usize, String>,
    chunk_explicit_global_names: &std::collections::BTreeMap<usize, String>,
) {
    const BUILTIN_END: usize = BUILTIN_GLOBAL_COUNT;
    for (idx, name) in chunk_global_names {
        // Never merge high indices: chunk may already be patched; VM high slots are from ensure_globals_from_chunk and must not be overwritten.
        if *idx >= BUILTIN_END {
            continue;
        }
        // Do not overwrite builtin indices below BUILTIN_END with a different name (would break "print", "len", etc.).
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
