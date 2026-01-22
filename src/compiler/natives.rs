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
    register(globals, "concat");
}

fn register(globals: &mut std::collections::HashMap<String, usize>, name: &str) {
    let index = globals.len();
    globals.insert(name.to_string(), index);
}

/// Возвращает имена параметров для нативной функции, если она поддерживает именованные аргументы
/// None возвращается для функций с переменным числом аргументов (print, min, max, array)
pub fn get_native_function_params(function_name: &str) -> Option<Vec<String>> {
    match function_name {
        // Функции с переменным числом аргументов - именованные аргументы не поддерживаются
        "print" | "min" | "max" | "array" => None,
        
        // Функции с одним параметром
        "len" => Some(vec!["value".to_string()]),
        "int" => Some(vec!["value".to_string()]),
        "float" => Some(vec!["value".to_string()]),
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
        "pop" => Some(vec!["array".to_string()]),
        "unique" => Some(vec!["array".to_string()]),
        "reverse" => Some(vec!["array".to_string()]),
        "sort" => Some(vec!["array".to_string()]),
        "sum" => Some(vec!["array".to_string()]),
        "average" => Some(vec!["array".to_string()]),
        "count" => Some(vec!["array".to_string()]),
        "table_info" => Some(vec!["table".to_string()]),
        "show_table" => Some(vec!["table".to_string()]),
        "now" => Some(vec![]),
        "getcwd" => Some(vec![]),
        
        // Функции с двумя параметрами
        "range" => Some(vec!["start".to_string(), "end".to_string(), "step".to_string()]),
        "pow" => Some(vec!["base".to_string(), "exp".to_string()]),
        "split" => Some(vec!["str".to_string(), "delim".to_string()]),
        "join" => Some(vec!["array".to_string(), "delim".to_string()]),
        "contains" => Some(vec!["str".to_string(), "substr".to_string()]),
        "push" => Some(vec!["array".to_string(), "item".to_string()]),
        "isinstance" => Some(vec!["value".to_string(), "type".to_string()]),
        "money" => Some(vec!["amount".to_string(), "format".to_string()]),
        "list_files" => Some(vec!["path".to_string(), "regex".to_string()]),
        
        // Функции с опциональными параметрами
        "table" => Some(vec!["data".to_string(), "headers".to_string()]),
        "read_file" => Some(vec!["path".to_string(), "header_row".to_string(), "sheet_name".to_string(), "header".to_string()]),
        "table_head" => Some(vec!["table".to_string(), "n".to_string()]),
        "table_tail" => Some(vec!["table".to_string(), "n".to_string()]),
        "table_select" => Some(vec!["table".to_string(), "cols".to_string()]),
        "table_sort" => Some(vec!["table".to_string(), "col".to_string(), "asc".to_string()]),
        "table_where" => Some(vec!["table".to_string(), "col".to_string(), "op".to_string(), "value".to_string()]),
        "merge_tables" => Some(vec!["tables".to_string(), "mode".to_string()]),
        "cross_join" => Some(vec!["left".to_string(), "right".to_string()]),
        "table_suffixes" => Some(vec!["left".to_string(), "right".to_string(), "left_suffix".to_string(), "right_suffix".to_string()]),
        "relate" => Some(vec!["col1".to_string(), "col2".to_string()]),
        "primary_key" => Some(vec!["col".to_string()]),
        
        // JOIN функции - они все имеют одинаковую структуру (left, right, on, type?, suffixes?)
        "inner_join" | "left_join" | "right_join" | "full_join" | "semi_join" | "anti_join" | "zip_join" | "asof_join" | "join_on" | "apply_join" => {
            Some(vec!["left".to_string(), "right".to_string(), "on".to_string(), "type".to_string(), "suffixes".to_string()])
        },
        
        // Module methods
        "show" => Some(vec!["image".to_string(), "title".to_string()]),
        
        // ML functions
        "nn_train" => Some(vec![
            "nn".to_string(),  // Model object (first parameter, added separately for method calls)
            "x".to_string(),
            "y".to_string(),
            "epochs".to_string(),
            "batch_size".to_string(),
            "learning_rate".to_string(),
            "loss".to_string(),
            "optimizer".to_string(),
            "x_val".to_string(),
            "y_val".to_string(),
        ]),
        "nn_train_sh" => Some(vec![
            "nn".to_string(),  // Model object (first parameter, added separately for method calls)
            "x".to_string(),
            "y".to_string(),
            "epochs".to_string(),
            "batch_size".to_string(),
            "learning_rate".to_string(),
            "loss".to_string(),
            "optimizer".to_string(),
            "monitor".to_string(),
            "patience".to_string(),
            "min_delta".to_string(),
            "restore_best".to_string(),
            "x_val".to_string(),
            "y_val".to_string(),
        ]),
        
        // Функция не найдена или не поддерживает именованные аргументы
        _ => None,
    }
}






