/// Регистрация нативных функций компилятора

/// Регистрирует все нативные функции в таблице глобальных переменных
/// Порядок должен соответствовать порядку в VM::register_native_globals()
pub fn register_natives(globals: &mut std::collections::HashMap<String, usize>) {
    // Базовые функции
    register(globals, "print");
    register(globals, "len");
    register(globals, "range");
    register(globals, "int");
    register(globals, "float");
    register(globals, "bool");
    register(globals, "str");
    register(globals, "array");
    register(globals, "typeof");
    register(globals, "isinstance");
    register(globals, "date");
    register(globals, "money");

    // Path функции
    register(globals, "path");
    register(globals, "path_name");
    register(globals, "path_parent");
    register(globals, "path_exists");
    register(globals, "path_is_file");
    register(globals, "path_is_dir");
    register(globals, "path_extension");
    register(globals, "path_stem");
    register(globals, "path_len");

    // Математические функции
    register(globals, "abs");
    register(globals, "sqrt");
    register(globals, "pow");
    register(globals, "min");
    register(globals, "max");
    register(globals, "round");

    // Строковые функции
    register(globals, "upper");
    register(globals, "lower");
    register(globals, "trim");
    register(globals, "split");
    register(globals, "join");
    register(globals, "contains");
    register(globals, "isupper");
    register(globals, "islower");

    // Функции массивов
    register(globals, "push");
    register(globals, "pop");
    register(globals, "unique");
    register(globals, "reverse");
    register(globals, "sort");
    register(globals, "sum");
    register(globals, "average");
    register(globals, "count");
    register(globals, "any");
    register(globals, "all");

    // Функции для работы с таблицами
    register(globals, "table");
    register(globals, "read_file");
    register(globals, "read_file_bin");
    register(globals, "table_info");
    register(globals, "table_head");
    register(globals, "table_tail");
    register(globals, "table_select");
    register(globals, "table_sort");
    register(globals, "table_where");
    register(globals, "show_table");
    register(globals, "merge_tables");
    register(globals, "now");
    register(globals, "getcwd");
    register(globals, "list_files");

    // JOIN операции
    register(globals, "inner_join");
    register(globals, "left_join");
    register(globals, "right_join");
    register(globals, "full_join");
    register(globals, "cross_join");
    register(globals, "semi_join");
    register(globals, "anti_join");
    register(globals, "zip_join");
    register(globals, "asof_join");
    register(globals, "apply_join");
    register(globals, "join_on");
    register(globals, "table_suffixes");
    register(globals, "relate");
    register(globals, "primary_key");
    register(globals, "enum");
    register(globals, "Table");
    register(globals, "array_with_capacity");
    register(globals, "map");
    register(globals, "filter");
    register(globals, "reduce");
    register(globals, "sha256");
    register(globals, "sha512");
    register(globals, "hmac_sha256");
    register(globals, "hmac_sha512");
    register(globals, "random_bytes");
    register(globals, "random_int");
    register(globals, "random_seed");
    register(globals, "date_to_unix");
    register(globals, "duration");
    register(globals, "set");
    register(globals, "divmod");
    register(globals, "isinf");

    // Built-in module globals (plot, uuid, debug, …) so `uuid.foo()` / `debug.operators()` resolve without `import`.
    // Order matches `crate::vm::modules::BUILTIN_MODULE_NAMES` (deterministic indices for chunk/linker).
    for name in crate::vm::modules::BUILTIN_MODULE_NAMES {
        if !globals.contains_key(*name) {
            register(globals, name);
        }
    }
}

fn register(globals: &mut std::collections::HashMap<String, usize>, name: &str) {
    let index = globals.len();
    globals.insert(name.to_string(), index);
}

/// Возвращает имена параметров для нативной функции, если она поддерживает именованные аргументы
/// None — когда kwargs не задаются здесь (`print`, `array`, …). У `min`/`max` два параметра `(iterable, key)` для `key=…`; при этом многозначные только-числовые вызовы по-прежнему поддерживаются (см. `resolve_function_args`).
pub fn get_native_function_params(function_name: &str) -> Option<Vec<String>> {
    match function_name {
        // Функции с переменным числом аргументов — kwargs не задаём здесь
        "print" | "array" | "set" => None,

        // iterable + опциональный key (именованный key= поддерживается; varargs через позиционные вызовы)
        "min" => Some(vec!["iterable".to_string(), "key".to_string()]),
        "max" => Some(vec!["iterable".to_string(), "key".to_string()]),

        // Функции с одним параметром
        "len" => Some(vec!["value".to_string()]),
        "enum" => Some(vec!["iterable".to_string()]),
        "int" => Some(vec!["value".to_string()]),
        "float" => Some(vec!["value".to_string()]),
        "isinf" => Some(vec!["value".to_string()]),
        "bool" => Some(vec!["value".to_string()]),
        "str" => Some(vec!["value".to_string()]),
        "typeof" => Some(vec!["value".to_string()]),
        "date" => Some(vec!["value".to_string()]),
        "path" => Some(vec!["value".to_string()]),
        "path_name" => Some(vec!["path".to_string()]),
        "path_parent" => Some(vec!["path".to_string()]),
        "path_exists" => Some(vec!["path".to_string()]),
        "path_is_file" => Some(vec!["path".to_string()]),
        "path_is_dir" => Some(vec!["path".to_string()]),
        "path_extension" => Some(vec!["path".to_string()]),
        "path_stem" => Some(vec!["path".to_string()]),
        "path_len" => Some(vec!["path".to_string()]),
        "abs" => Some(vec!["n".to_string()]),
        "sqrt" => Some(vec!["n".to_string()]),
        "round" => Some(vec!["n".to_string()]),
        "upper" => Some(vec!["str".to_string()]),
        "lower" => Some(vec!["str".to_string()]),
        "trim" => Some(vec!["str".to_string()]),
        "isupper" => Some(vec!["str".to_string()]),
        "islower" => Some(vec!["str".to_string()]),
        "pop" => Some(vec!["array".to_string(), "idx".to_string()]),
        "unique" => Some(vec!["array".to_string()]),
        "reverse" => Some(vec!["array".to_string()]),
        "sort" => Some(vec!["iterable".to_string()]),
        "sum" => Some(vec!["array".to_string()]),
        "average" => Some(vec!["array".to_string()]),
        "count" => Some(vec!["array".to_string()]),
        "table_info" => Some(vec!["table".to_string()]),
        "show_table" => Some(vec!["table".to_string()]),
        "now" => Some(vec![]),
        "getcwd" => Some(vec![]),
        "array_with_capacity" => Some(vec!["n".to_string()]),
        "map" => Some(vec!["collection".to_string(), "fn".to_string()]),
        "filter" => Some(vec!["collection".to_string(), "predicate".to_string()]),
        "reduce" => Some(vec![
            "collection".to_string(),
            "fn".to_string(),
            "initial".to_string(),
        ]),
        "sha256" => Some(vec!["data".to_string()]),
        "sha512" => Some(vec!["data".to_string()]),
        "hmac_sha256" => Some(vec!["key".to_string(), "data".to_string()]),
        "hmac_sha512" => Some(vec!["key".to_string(), "data".to_string()]),
        "random_bytes" => Some(vec!["size".to_string()]),
        "random_int" => Some(vec!["min".to_string(), "max".to_string()]),
        "random_seed" => Some(vec!["seed".to_string()]),
        "date_to_unix" => Some(vec!["value".to_string()]),
        "duration" => Some(vec![
            "seconds".to_string(),
            "minutes".to_string(),
            "hours".to_string(),
            "days".to_string(),
            "milliseconds".to_string(),
        ]),

        // Функции с двумя параметрами
        "range" => Some(vec![
            "start".to_string(),
            "end".to_string(),
            "step".to_string(),
        ]),
        "pow" => Some(vec!["base".to_string(), "exp".to_string()]),
        "divmod" => Some(vec!["a".to_string(), "b".to_string()]),
        "split" => Some(vec!["str".to_string(), "delim".to_string()]),
        "join" => Some(vec!["array".to_string(), "delim".to_string()]),
        "contains" => Some(vec!["str".to_string(), "substr".to_string()]),
        "push" => Some(vec!["array".to_string(), "item".to_string()]),
        "isinstance" => Some(vec!["value".to_string(), "type".to_string()]),
        "money" => Some(vec!["amount".to_string(), "format".to_string()]),
        "list_files" => Some(vec!["path".to_string(), "regex".to_string()]),

        // Функции с опциональными параметрами
        "table" => Some(vec!["data".to_string(), "headers".to_string()]),
        "read_file" => Some(vec![
            "path".to_string(),
            "header_row".to_string(),
            "sheet_name".to_string(),
            "header".to_string(),
        ]),
        "read_file_bin" => Some(vec!["path".to_string()]),
        "table_head" => Some(vec!["table".to_string(), "n".to_string()]),
        "table_tail" => Some(vec!["table".to_string(), "n".to_string()]),
        "table_select" => Some(vec!["table".to_string(), "cols".to_string()]),
        "table_sort" => Some(vec![
            "table".to_string(),
            "col".to_string(),
            "asc".to_string(),
        ]),
        "table_where" => Some(vec![
            "table".to_string(),
            "col".to_string(),
            "op".to_string(),
            "value".to_string(),
        ]),
        "merge_tables" => Some(vec!["tables".to_string(), "mode".to_string()]),
        "cross_join" => Some(vec!["left".to_string(), "right".to_string()]),
        "table_suffixes" => Some(vec![
            "left".to_string(),
            "right".to_string(),
            "left_suffix".to_string(),
            "right_suffix".to_string(),
        ]),
        "relate" => Some(vec!["col1".to_string(), "col2".to_string()]),
        "primary_key" => Some(vec!["col".to_string()]),
        "Table" => Some(vec!["path".to_string()]),

        // database module (from database_engine import engine, ...)
        "engine" => Some(vec![
            "url".to_string(),
            "echo".to_string(),
            "echo_pool".to_string(),
            "pool_size".to_string(),
            "max_overflow".to_string(),
            "timeout".to_string(),
            "connect_args".to_string(),
        ]),
        "MetaData" => Some(vec![
            "schema".to_string(),
            "quote_schema".to_string(),
            "naming_convention".to_string(),
            "info".to_string(),
        ]),
        "Column" => Some(vec![
            "type".to_string(),
            "primary_key".to_string(),
            "autoincrement".to_string(),
            "unique".to_string(),
            "default".to_string(),
            "nullable".to_string(),
            "onupdate".to_string(),
            "transform".to_string(),
            "validators".to_string(),
        ]),

        // JOIN функции - они все имеют одинаковую структуру (left, right, on, type?, suffixes?)
        "inner_join" | "left_join" | "right_join" | "full_join" | "semi_join" | "anti_join"
        | "zip_join" | "asof_join" | "join_on" | "apply_join" => Some(vec![
            "left".to_string(),
            "right".to_string(),
            "on".to_string(),
            "type".to_string(),
            "suffixes".to_string(),
        ]),

        // Module methods
        "show" => Some(vec!["image".to_string(), "title".to_string()]),
        "line" => Some(vec![
            "x".to_string(),
            "y".to_string(),
            "show_points".to_string(),
            "point_size".to_string(),
            "line_width".to_string(),
            "color".to_string(),
        ]),

        // settings_env.Config(...) / Settings.config(...) — config dict for load_env
        "Config" | "config" => Some(vec![
            "env_prefix".to_string(),
            "extra".to_string(),
            "env_file".to_string(),
            "env_file_encoding".to_string(),
            "case_sensitive".to_string(),
            "env_nested_delimiter".to_string(),
        ]),

        // settings_env.Field(...) — full Pydantic-style field descriptor
        "Field" => Some(vec![
            "default".to_string(),
            "default_factory".to_string(),
            "alias".to_string(),
            "title".to_string(),
            "description".to_string(),
            "examples".to_string(),
            "exclude".to_string(),
            "include".to_string(),
            "const".to_string(),
            "gt".to_string(),
            "ge".to_string(),
            "lt".to_string(),
            "le".to_string(),
            "multiple_of".to_string(),
            "min_length".to_string(),
            "max_length".to_string(),
            "regex".to_string(),
            "deprecated".to_string(),
            "repr".to_string(),
            "json_schema_extra".to_string(),
            "validate_default".to_string(),
            "frozen".to_string(),
        ]),

        // Функция не найдена или не поддерживает именованные аргументы
        _ => None,
    }
}
