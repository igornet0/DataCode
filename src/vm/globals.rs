// Global variables operations for VM

use crate::common::value::Value;

/// Канонические имена встроенных глобалов по индексу 0..74 (для merge_globals_from: не перезаписывать правильное значение ошибочным).
pub const BUILTIN_GLOBAL_NAMES: [&str; 74] = [
    "print", "len", "range", "int", "float", "bool", "str", "array", "typeof", "isinstance",
    "date", "money", "path", "path_name", "path_parent", "path_exists", "path_is_file", "path_is_dir",
    "path_extension", "path_stem", "path_len", "abs", "sqrt", "pow", "min", "max", "round",
    "upper", "lower", "trim", "split", "join", "contains", "isupper", "islower", "push", "pop", "unique", "reverse",
    "sort", "sum", "average", "count", "any", "all", "table", "read_file", "table_info", "table_head",
    "table_tail", "table_select", "table_sort", "table_where", "show_table", "merge_tables", "now",
    "getcwd", "list_files", "inner_join", "left_join", "right_join", "full_join", "cross_join",
    "semi_join", "anti_join", "zip_join", "asof_join", "apply_join", "join_on", "table_suffixes",
    "relate", "primary_key", "enum", "Table",
];

/// Возвращает каноническое имя встроенной глобальной переменной по индексу (0..74).
pub fn builtin_global_name(index: usize) -> Option<&'static str> {
    (index < 74).then(|| BUILTIN_GLOBAL_NAMES[index])
}

/// Возвращает канонический индекс встроенной глобальной переменной по имени (для set_functions).
pub fn builtin_global_index(name: &str) -> Option<usize> {
    BUILTIN_GLOBAL_NAMES
        .iter()
        .position(|&n| n == name)
}

/// Регистрирует нативные функции в глобальных переменных
/// Порядок должен соответствовать register_natives() в vm.rs
pub fn register_native_globals(
    globals: &mut Vec<Value>,
    global_names: &mut std::collections::HashMap<usize, String>,
) {
    // Ensure globals vector is large enough for native functions (74) and any globals from compiler
    let max_global_index = global_names.keys().max().copied().unwrap_or(73);
    let min_size = (74).max(max_global_index + 1);
    globals.resize(min_size, Value::Null);

    for (index, name) in BUILTIN_GLOBAL_NAMES.iter().enumerate() {
        globals[index] = Value::NativeFunction(index);
        if !global_names.contains_key(&index) {
            global_names.insert(index, (*name).to_string());
        }
    }
}/// Заполняет имена глобальных переменных из chunk
pub fn merge_global_names(
    global_names: &mut std::collections::HashMap<usize, String>,
    explicit_global_names: &mut std::collections::HashMap<usize, String>,
    chunk_global_names: &std::collections::HashMap<usize, String>,
    chunk_explicit_global_names: &std::collections::HashMap<usize, String>,
) {
    // Merge with existing global_names instead of overwriting to preserve modules registered during execution
    for (idx, name) in chunk_global_names {
        global_names.insert(*idx, name.clone());
    }
    for (idx, name) in chunk_explicit_global_names {
        explicit_global_names.insert(*idx, name.clone());
    }
}
