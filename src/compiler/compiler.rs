// Компилятор AST → Bytecode

use crate::parser::ast::{Expr, Stmt, Arg, UnpackPattern, ImportStmt, ImportItem};
use crate::bytecode::{Chunk, OpCode, Function, CapturedVar};
use crate::common::error::LangError;
use crate::common::value::Value;
use crate::lexer::TokenKind;

// Структура для отслеживания обработчиков исключений
#[derive(Clone)]
struct ExceptionHandler {
    catch_ips: Vec<usize>,           // IP начала каждого catch блока
    error_types: Vec<Option<usize>>, // Типы ошибок для каждого catch (None для catch всех)
    error_var_slots: Vec<Option<usize>>, // Слоты для переменных ошибок
    else_ip: Option<usize>,          // IP начала else блока
    stack_height: usize,            // Высота стека при входе в try
}

// Структура для отслеживания контекста циклов
struct LoopContext {
    continue_label: usize,   // Метка для continue (начало следующей итерации или инкремент)
    break_label: usize,      // Метка для break (конец цикла)
}

pub struct Compiler {
    chunk: Chunk,
    functions: Vec<Function>,
    function_names: Vec<String>, // Имена функций для поиска
    current_function: Option<usize>, // Индекс текущей компилируемой функции
    globals: std::collections::HashMap<String, usize>, // Глобальные переменные и функции
    locals: Vec<std::collections::HashMap<String, usize>>, // Локальные переменные для каждой функции (стек областей видимости)
    local_count: usize, // Счетчик локальных переменных в текущей функции
    current_line: usize, // Текущий номер строки (для отладки и ошибок)
    exception_handlers: Vec<ExceptionHandler>, // Стек обработчиков исключений
    error_type_table: Vec<String>, // Таблица типов ошибок для текущей функции
    loop_contexts: Vec<LoopContext>, // Стек контекстов циклов для break/continue
    // Система меток для эталонного алгоритма апгрейда jump-инструкций
    label_counter: usize, // Счетчик для генерации уникальных ID меток
    labels: std::collections::HashMap<usize, usize>, // Маппинг label_id -> индекс инструкции
    pending_jumps: Vec<(usize, usize, bool)>, // (индекс_инструкции, label_id, is_conditional)
}

impl Compiler {
    pub fn new() -> Self {
        let mut compiler = Self {
            chunk: Chunk::new(),
            functions: Vec::new(),
            function_names: Vec::new(),
            current_function: None,
            globals: std::collections::HashMap::new(),
            locals: Vec::new(),
            local_count: 0,
            current_line: 0,
            exception_handlers: Vec::new(),
            error_type_table: Vec::new(),
            loop_contexts: Vec::new(),
            label_counter: 0,
            labels: std::collections::HashMap::new(),
            pending_jumps: Vec::new(),
        };
        compiler.register_natives();
        compiler
    }

    // Получить индекс типа ошибки в таблице типов
    fn get_error_type_index(&mut self, error_type_name: &str) -> usize {
        // Ищем в существующей таблице
        if let Some(index) = self.error_type_table.iter().position(|s| s == error_type_name) {
            return index;
        }
        // Добавляем новый тип
        let index = self.error_type_table.len();
        self.error_type_table.push(error_type_name.to_string());
        index
    }

    fn register_natives(&mut self) {
        // Регистрируем нативные функции как глобальные переменные
        // Порядок должен соответствовать порядку в VM::register_native_globals()
        let print_index = self.globals.len();
        self.globals.insert("print".to_string(), print_index);
        
        let len_index = self.globals.len();
        self.globals.insert("len".to_string(), len_index);
        
        let range_index = self.globals.len();
        self.globals.insert("range".to_string(), range_index);
        
        let int_index = self.globals.len();
        self.globals.insert("int".to_string(), int_index);
        
        let float_index = self.globals.len();
        self.globals.insert("float".to_string(), float_index);
        
        let bool_index = self.globals.len();
        self.globals.insert("bool".to_string(), bool_index);
        
        let str_index = self.globals.len();
        self.globals.insert("str".to_string(), str_index);
        
        let array_index = self.globals.len();
        self.globals.insert("array".to_string(), array_index);
        
        let typeof_index = self.globals.len();
        self.globals.insert("typeof".to_string(), typeof_index);
        
        let isinstance_index = self.globals.len();
        self.globals.insert("isinstance".to_string(), isinstance_index);
        
        let date_index = self.globals.len();
        self.globals.insert("date".to_string(), date_index);
        
        let money_index = self.globals.len();
        self.globals.insert("money".to_string(), money_index);
        
        let path_index = self.globals.len();
        self.globals.insert("path".to_string(), path_index);
        
        let path_name_index = self.globals.len();
        self.globals.insert("path_name".to_string(), path_name_index);
        
        let path_parent_index = self.globals.len();
        self.globals.insert("path_parent".to_string(), path_parent_index);
        
        let path_exists_index = self.globals.len();
        self.globals.insert("path_exists".to_string(), path_exists_index);
        
        let path_is_file_index = self.globals.len();
        self.globals.insert("path_is_file".to_string(), path_is_file_index);
        
        let path_is_dir_index = self.globals.len();
        self.globals.insert("path_is_dir".to_string(), path_is_dir_index);
        
        let path_extension_index = self.globals.len();
        self.globals.insert("path_extension".to_string(), path_extension_index);
        
        let path_stem_index = self.globals.len();
        self.globals.insert("path_stem".to_string(), path_stem_index);
        
        let path_len_index = self.globals.len();
        self.globals.insert("path_len".to_string(), path_len_index);
        
        // Математические функции
        let abs_index = self.globals.len();
        self.globals.insert("abs".to_string(), abs_index);
        
        let sqrt_index = self.globals.len();
        self.globals.insert("sqrt".to_string(), sqrt_index);
        
        let pow_index = self.globals.len();
        self.globals.insert("pow".to_string(), pow_index);
        
        let min_index = self.globals.len();
        self.globals.insert("min".to_string(), min_index);
        
        let max_index = self.globals.len();
        self.globals.insert("max".to_string(), max_index);
        
        let round_index = self.globals.len();
        self.globals.insert("round".to_string(), round_index);
        
        // Строковые функции
        let upper_index = self.globals.len();
        self.globals.insert("upper".to_string(), upper_index);
        
        let lower_index = self.globals.len();
        self.globals.insert("lower".to_string(), lower_index);
        
        let trim_index = self.globals.len();
        self.globals.insert("trim".to_string(), trim_index);
        
        let split_index = self.globals.len();
        self.globals.insert("split".to_string(), split_index);
        
        let join_index = self.globals.len();
        self.globals.insert("join".to_string(), join_index);
        
        let contains_index = self.globals.len();
        self.globals.insert("contains".to_string(), contains_index);
        
        // Функции массивов
        let push_index = self.globals.len();
        self.globals.insert("push".to_string(), push_index);
        
        let pop_index = self.globals.len();
        self.globals.insert("pop".to_string(), pop_index);
        
        let unique_index = self.globals.len();
        self.globals.insert("unique".to_string(), unique_index);
        
        let reverse_index = self.globals.len();
        self.globals.insert("reverse".to_string(), reverse_index);
        
        let sort_index = self.globals.len();
        self.globals.insert("sort".to_string(), sort_index);
        
        let sum_index = self.globals.len();
        self.globals.insert("sum".to_string(), sum_index);
        
        let average_index = self.globals.len();
        self.globals.insert("average".to_string(), average_index);
        
        let count_index = self.globals.len();
        self.globals.insert("count".to_string(), count_index);
        
        let any_index = self.globals.len();
        self.globals.insert("any".to_string(), any_index);
        
        let all_index = self.globals.len();
        self.globals.insert("all".to_string(), all_index);
        
        // Функции для работы с таблицами
        let table_index = self.globals.len();
        self.globals.insert("table".to_string(), table_index);
        
        let read_file_index = self.globals.len();
        self.globals.insert("read_file".to_string(), read_file_index);
        
        let table_info_index = self.globals.len();
        self.globals.insert("table_info".to_string(), table_info_index);
        
        let table_head_index = self.globals.len();
        self.globals.insert("table_head".to_string(), table_head_index);
        
        let table_tail_index = self.globals.len();
        self.globals.insert("table_tail".to_string(), table_tail_index);
        
        let table_select_index = self.globals.len();
        self.globals.insert("table_select".to_string(), table_select_index);
        
        let table_sort_index = self.globals.len();
        self.globals.insert("table_sort".to_string(), table_sort_index);
        
        let table_where_index = self.globals.len();
        self.globals.insert("table_where".to_string(), table_where_index);
        
        let show_table_index = self.globals.len();
        self.globals.insert("show_table".to_string(), show_table_index);

        let merge_tables_index = self.globals.len();
        self.globals.insert("merge_tables".to_string(), merge_tables_index);

        let now_index = self.globals.len();
        self.globals.insert("now".to_string(), now_index);

        let getcwd_index = self.globals.len();
        self.globals.insert("getcwd".to_string(), getcwd_index);

        let list_files_index = self.globals.len();
        self.globals.insert("list_files".to_string(), list_files_index);

        // JOIN операции
        let inner_join_index = self.globals.len();
        self.globals.insert("inner_join".to_string(), inner_join_index);

        let left_join_index = self.globals.len();
        self.globals.insert("left_join".to_string(), left_join_index);

        let right_join_index = self.globals.len();
        self.globals.insert("right_join".to_string(), right_join_index);

        let full_join_index = self.globals.len();
        self.globals.insert("full_join".to_string(), full_join_index);

        let cross_join_index = self.globals.len();
        self.globals.insert("cross_join".to_string(), cross_join_index);

        let semi_join_index = self.globals.len();
        self.globals.insert("semi_join".to_string(), semi_join_index);

        let anti_join_index = self.globals.len();
        self.globals.insert("anti_join".to_string(), anti_join_index);

        let zip_join_index = self.globals.len();
        self.globals.insert("zip_join".to_string(), zip_join_index);

        let asof_join_index = self.globals.len();
        self.globals.insert("asof_join".to_string(), asof_join_index);

        let apply_join_index = self.globals.len();
        self.globals.insert("apply_join".to_string(), apply_join_index);

        let join_on_index = self.globals.len();
        self.globals.insert("join_on".to_string(), join_on_index);

        let table_suffixes_index = self.globals.len();
        self.globals.insert("table_suffixes".to_string(), table_suffixes_index);

        let relate_index = self.globals.len();
        self.globals.insert("relate".to_string(), relate_index);

        let primary_key_index = self.globals.len();
        self.globals.insert("primary_key".to_string(), primary_key_index);

        let concat_table_index = self.globals.len();
        self.globals.insert("concat".to_string(), concat_table_index);
    }

    pub fn compile(&mut self, statements: &[Stmt]) -> Result<Chunk, LangError> {
        // Начинаем область видимости для главной функции
        self.begin_scope();
        
        // Первый проход: объявляем все функции (forward declaration), включая вложенные
        self.collect_all_functions(statements)?;
        
        // Второй проход: компилируем все statements
        for (i, stmt) in statements.iter().enumerate() {
            let is_last = i == statements.len() - 1;
            self.compile_stmt_with_pop(stmt, !is_last)?;
        }
        self.chunk.write_with_line(OpCode::Return, self.current_line);
        
        // Заканчиваем область видимости главной функции
        self.end_scope();
        
        // Эталонный алгоритм апгрейда jump-инструкций: стабилизация layout и финализация
        self.stabilize_layout()?;
        self.finalize_jumps()?;
        
        // Очищаем метки после финализации главного скрипта
        self.clear_labels();
        
        Ok(self.chunk.clone())
    }

    /// Рекурсивно собирает все объявления функций на всех уровнях вложенности
    fn collect_all_functions(&mut self, statements: &[Stmt]) -> Result<(), LangError> {
        for stmt in statements {
            match stmt {
                Stmt::Function { name, params, body, is_cached, .. } => {
                    // Объявляем функцию с правильной сигнатурой сразу
                    let arity = params.len();
                    let param_names: Vec<String> = params.iter().map(|p| p.name.clone()).collect();
                    
                    let mut function = if *is_cached {
                        Function::with_cache(name.clone(), arity)
                    } else {
                        Function::new(name.clone(), arity)
                    };
                    
                    // Устанавливаем имена параметров сразу (значения по умолчанию обработаем позже)
                    function.param_names = param_names;
                    // Инициализируем default_values как None для всех параметров (обработаем позже)
                    function.default_values = vec![None; params.len()];
                    
                    self.functions.push(function);
                    self.function_names.push(name.clone());
                    // Регистрируем функцию в глобальной таблице
                    let global_index = self.globals.len();
                    self.globals.insert(name.clone(), global_index);
                    
                    // Рекурсивно собираем функции из тела этой функции
                    self.collect_all_functions(body)?;
                }
                Stmt::If { then_branch, else_branch, .. } => {
                    // Рекурсивно собираем функции из веток if
                    self.collect_all_functions(then_branch)?;
                    if let Some(else_branch) = else_branch {
                        self.collect_all_functions(else_branch)?;
                    }
                }
                Stmt::While { body, .. } => {
                    // Рекурсивно собираем функции из тела while
                    self.collect_all_functions(body)?;
                }
                Stmt::For { body, .. } => {
                    // Рекурсивно собираем функции из тела for
                    self.collect_all_functions(body)?;
                }
                Stmt::Try { try_block, catch_blocks, else_block, .. } => {
                    // Рекурсивно собираем функции из try блока
                    self.collect_all_functions(try_block)?;
                    // Рекурсивно собираем функции из catch блоков
                    for catch_block in catch_blocks {
                        self.collect_all_functions(&catch_block.body)?;
                    }
                    // Рекурсивно собираем функции из else блока (если есть)
                    if let Some(else_block) = else_block {
                        self.collect_all_functions(else_block)?;
                    }
                }
                _ => {
                    // Другие statements не содержат вложенных statements
                }
            }
        }
        Ok(())
    }

    pub fn get_functions(self) -> Vec<Function> {
        self.functions
    }

    /// Возвращает имена параметров для нативной функции, если она поддерживает именованные аргументы
    /// None возвращается для функций с переменным числом аргументов (print, min, max, array)
    fn get_native_function_params(&self, function_name: &str) -> Option<Vec<String>> {
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
            "list_files" => Some(vec!["path".to_string()]),
            
            // Функции с опциональными параметрами
            "table" => Some(vec!["data".to_string(), "headers".to_string()]),
            "read_file" => Some(vec!["path".to_string(), "header_row".to_string(), "sheet_name".to_string()]),
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

    fn compile_stmt(&mut self, stmt: &Stmt) -> Result<(), LangError> {
        self.compile_stmt_with_pop(stmt, true)
    }

    fn compile_stmt_with_pop(&mut self, stmt: &Stmt, pop_value: bool) -> Result<(), LangError> {
        let stmt_line = stmt.line();
        self.current_line = stmt_line;
        match stmt {
            Stmt::Import { import_stmt, line } => {
                match import_stmt {
                    ImportStmt::Modules(modules) => {
                        // import ml, plot
                        for module in modules {
                            // Import statements are handled at runtime by the VM
                            // We compile them as a special opcode that the VM will handle
                            // Store the module name as a constant and emit Import opcode
                            let module_index = self.chunk.add_constant(Value::String(module.clone()));
                            self.chunk.write_with_line(OpCode::Import(module_index), *line);
                            
                            // Register the module name in globals so subsequent uses are recognized
                            // This allows the compiler to know about the module even though it's loaded at runtime
                            if !self.globals.contains_key(module) {
                                let global_index = self.globals.len();
                                self.globals.insert(module.clone(), global_index);
                                self.chunk.global_names.insert(global_index, module.clone());
                            }
                        }
                    }
                    ImportStmt::From { module, items } => {
                        // from ml import load_mnist, *
                        // Создаем массив элементов импорта в константах
                        use std::rc::Rc;
                        use std::cell::RefCell;
                        let mut item_strings = Vec::new();
                        for item in items {
                            match item {
                                ImportItem::Named(name) => {
                                    item_strings.push(Value::String(name.clone()));
                                }
                                ImportItem::Aliased { name, alias } => {
                                    // Формат: "name:alias"
                                    item_strings.push(Value::String(format!("{}:{}", name, alias)));
                                }
                                ImportItem::All => {
                                    item_strings.push(Value::String("*".to_string()));
                                }
                            }
                        }
                        let items_array = Value::Array(Rc::new(RefCell::new(item_strings)));
                        let items_index = self.chunk.add_constant(items_array);
                        
                        // Store the module name as a constant
                        let module_index = self.chunk.add_constant(Value::String(module.clone()));
                        self.chunk.write_with_line(OpCode::ImportFrom(module_index, items_index), *line);
                        
                        // Register the module name in globals
                        if !self.globals.contains_key(module) {
                            let global_index = self.globals.len();
                            self.globals.insert(module.clone(), global_index);
                            self.chunk.global_names.insert(global_index, module.clone());
                        }
                        
                        // Register imported item names in globals for from-import
                        for item in items {
                            match item {
                                ImportItem::Named(name) => {
                                    if !self.globals.contains_key(name) {
                                        let global_index = self.globals.len();
                                        self.globals.insert(name.clone(), global_index);
                                        self.chunk.global_names.insert(global_index, name.clone());
                                    }
                                }
                                ImportItem::Aliased { alias, .. } => {
                                    if !self.globals.contains_key(alias) {
                                        let global_index = self.globals.len();
                                        self.globals.insert(alias.clone(), global_index);
                                        self.chunk.global_names.insert(global_index, alias.clone());
                                    }
                                }
                                ImportItem::All => {
                                    // All items will be imported at runtime, we can't register them here
                                    // But we need to handle this case in VM
                                }
                            }
                        }
                    }
                }
            }
            Stmt::Let { name, value, is_global, line } => {
                self.current_line = *line;
                // Проверяем, является ли value UnpackAssign (распаковка кортежа)
                if let Expr::UnpackAssign { names, value: tuple_value, .. } = value {
                    // Распаковка кортежа в let statement
                    self.compile_expr(tuple_value)?;
                    
                    // Сохраняем кортеж во временную переменную
                    let tuple_temp = self.declare_local(&format!("__tuple_temp_{}", line));
                    self.chunk.write_with_line(OpCode::StoreLocal(tuple_temp), *line);
                    
                    // Для каждой переменной извлекаем элемент кортежа и сохраняем
                    for (index, var_name) in names.iter().enumerate() {
                        // Загружаем кортеж
                        self.chunk.write_with_line(OpCode::LoadLocal(tuple_temp), *line);
                        // Загружаем индекс
                        let index_const = self.chunk.add_constant(Value::Number(index as f64));
                        self.chunk.write_with_line(OpCode::Constant(index_const), *line);
                        // Получаем элемент по индексу
                        self.chunk.write_with_line(OpCode::GetArrayElement, *line);
                        
                        // Сохраняем в переменную
                        if *is_global {
                            // Глобальная переменная
                            let global_index = if let Some(&idx) = self.globals.get(var_name) {
                                idx
                            } else {
                                let idx = self.globals.len();
                                self.globals.insert(var_name.clone(), idx);
                                idx
                            };
                            self.chunk.global_names.insert(global_index, var_name.clone());
                            self.chunk.explicit_global_names.insert(global_index, var_name.clone());
                            self.chunk.write_with_line(OpCode::StoreGlobal(global_index), *line);
                        } else {
                            // Локальная переменная
                            if let Some(local_index) = self.resolve_local(var_name) {
                                self.chunk.write_with_line(OpCode::StoreLocal(local_index), *line);
                            } else if self.current_function.is_some() {
                                let var_index = self.declare_local(var_name);
                                self.chunk.write_with_line(OpCode::StoreLocal(var_index), *line);
                            } else {
                                // На верхнем уровне - проверяем, есть ли глобальная переменная
                                if let Some(&global_index) = self.globals.get(var_name) {
                                    // Глобальная переменная найдена - обновляем
                                    self.chunk.global_names.insert(global_index, var_name.clone());
                                    self.chunk.write_with_line(OpCode::StoreGlobal(global_index), *line);
                                } else {
                                    // Новая глобальная переменная на верхнем уровне
                                    let global_index = self.globals.len();
                                    self.globals.insert(var_name.clone(), global_index);
                                    self.chunk.global_names.insert(global_index, var_name.clone());
                                    self.chunk.write_with_line(OpCode::StoreGlobal(global_index), *line);
                                }
                            }
                        }
                    }
                    
                    // Загружаем последнее значение на стек
                    if let Some(last_name) = names.last() {
                        if let Some(local_index) = self.resolve_local(last_name) {
                            self.chunk.write_with_line(OpCode::LoadLocal(local_index), *line);
                        } else if let Some(&global_index) = self.globals.get(last_name) {
                            self.chunk.write_with_line(OpCode::LoadGlobal(global_index), *line);
                        }
                    }
                    return Ok(());
                }
                
                // Обычное присваивание
                self.compile_expr(value)?;
                // Не клонируем автоматически - переменные должны разделять ссылки на массивы/таблицы/объекты
                // Клонирование происходит только при явном вызове .clone()
                if *is_global {
                    // Явное объявление глобальной переменной
                    // Всегда изменяет глобальную переменную, даже внутри функций
                    let global_index = if let Some(&idx) = self.globals.get(name) {
                        idx
                    } else {
                        let idx = self.globals.len();
                        self.globals.insert(name.clone(), idx);
                        idx
                    };
                    // Сохраняем имя глобальной переменной для использования в JOIN
                    self.chunk.global_names.insert(global_index, name.clone());
                    // Сохраняем имя явно объявленной глобальной переменной для экспорта в SQLite
                    self.chunk.explicit_global_names.insert(global_index, name.clone());
                    self.chunk.write_with_line(OpCode::StoreGlobal(global_index), *line);
                } else {
                    // Локальная переменная или глобальная на верхнем уровне
                    // Проверяем, есть ли переменная в локальной области видимости
                    if let Some(local_index) = self.resolve_local(name) {
                        // Локальная переменная уже объявлена - обновляем
                        self.chunk.write_with_line(OpCode::StoreLocal(local_index), *line);
                    } else if self.current_function.is_some() {
                        // Мы находимся внутри функции - объявляем новую локальную переменную
                        let index = self.declare_local(name);
                        self.chunk.write_with_line(OpCode::StoreLocal(index), *line);
                    } else {
                        // Переменная не найдена локально - проверяем, является ли она глобальной
                        // На верхнем уровне главной функции переменные без 'global' все равно глобальные
                        if let Some(&global_index) = self.globals.get(name) {
                            // Глобальная переменная уже существует - обновляем
                            // Сохраняем имя глобальной переменной для использования в JOIN
                            self.chunk.global_names.insert(global_index, name.clone());
                            self.chunk.write_with_line(OpCode::StoreGlobal(global_index), *line);
                        } else {
                            // Новая глобальная переменная на верхнем уровне
                            let global_index = self.globals.len();
                            self.globals.insert(name.clone(), global_index);
                            // Сохраняем имя глобальной переменной для использования в JOIN
                            self.chunk.global_names.insert(global_index, name.clone());
                            self.chunk.write_with_line(OpCode::StoreGlobal(global_index), *line);
                        }
                    }
                }
                // StoreGlobal/StoreLocal уже удаляют значение со стека, поэтому дополнительный Pop не нужен
            }
            Stmt::Expr { expr, line } => {
                self.current_line = *line;
                self.compile_expr(expr)?;
                if pop_value {
                    self.chunk.write_with_line(OpCode::Pop, *line);
                }
            }
            Stmt::If { condition, then_branch, else_branch, line } => {
                self.current_line = *line;
                self.compile_expr(condition)?;
                
                // Создаем метки для else и end
                let else_label = self.create_label();
                let end_label = self.create_label();
                
                // Jump if false к else (или end, если else нет)
                let target_label = if else_branch.is_some() { else_label } else { end_label };
                self.emit_jump(true, target_label)?;
                
                // Компилируем then ветку (с новой областью видимости)
                self.begin_scope();
                for (i, stmt) in then_branch.iter().enumerate() {
                    let is_last = i == then_branch.len() - 1;
                    self.compile_stmt_with_pop(stmt, !is_last || pop_value)?;
                }
                self.end_scope();
                
                // Jump к end после then
                self.emit_jump(false, end_label)?;
                
                // Помечаем метку else (если есть)
                if else_branch.is_some() {
                    self.mark_label(else_label);
                
                // Компилируем else ветку (с новой областью видимости)
                    self.begin_scope();
                    for (i, stmt) in else_branch.as_ref().unwrap().iter().enumerate() {
                        let is_last = i == else_branch.as_ref().unwrap().len() - 1;
                        self.compile_stmt_with_pop(stmt, !is_last || pop_value)?;
                    }
                    self.end_scope();
                }
                
                // Помечаем метку end
                self.mark_label(end_label);
            }
            Stmt::While { condition, body, line } => {
                self.current_line = *line;
                
                // Создаем метки для начала и конца цикла
                let loop_start_label = self.create_label();
                let loop_end_label = self.create_label();
                
                // Помечаем начало цикла
                self.mark_label(loop_start_label);
                
                self.compile_expr(condition)?;
                // Jump if false к концу цикла
                self.emit_jump(true, loop_end_label)?;
                
                // Создаем контекст цикла
                let loop_context = LoopContext {
                    continue_label: loop_start_label, // continue возвращается к началу цикла
                    break_label: loop_end_label,      // break переходит к концу цикла
                };
                self.loop_contexts.push(loop_context);
                
                // Компилируем тело цикла (с новой областью видимости)
                self.begin_scope();
                for stmt in body {
                    self.compile_stmt(stmt)?;
                }
                self.end_scope();
                
                // Jump к началу цикла
                self.emit_loop(loop_start_label)?;
                
                // Помечаем конец цикла
                self.mark_label(loop_end_label);
                
                // Контекст цикла уже содержит метки для break/continue
                
                // Условие уже удалено JumpIfFalse при выходе из цикла
                self.loop_contexts.pop();
            }
            Stmt::For { pattern, iterable, body, line } => {
                self.current_line = *line;
                
                // Начинаем новую область видимости для переменных цикла
                self.begin_scope();
                
                // Компилируем итерируемое выражение (оно должно быть массивом)
                self.compile_expr(iterable)?;
                
                // Сохраняем массив во временную переменную (локальную)
                // Создаем скрытую переменную для массива
                let array_local = self.declare_local("__array_iter");
                self.chunk.write_with_line(OpCode::StoreLocal(array_local), *line);
                
                // Создаем переменную для индекса
                let index_local = self.declare_local("__index_iter");
                // Инициализируем индекс как 0
                let zero_index = self.chunk.add_constant(Value::Number(0.0));
                self.chunk.write_with_line(OpCode::Constant(zero_index), *line);
                self.chunk.write_with_line(OpCode::StoreLocal(index_local), *line);
                
                // Проверяем, является ли это простым случаем (одна переменная без распаковки)
                let is_simple_case = pattern.len() == 1 && matches!(pattern[0], UnpackPattern::Variable(_));
                
                // Подсчитываем количество переменных (не wildcard) для проверки
                let expected_count = if is_simple_case {
                    0 // Не распаковываем, просто присваиваем
                } else {
                    self.count_unpack_variables(pattern)
                };
                
                // Объявляем переменные из паттерна распаковки
                let var_locals = if is_simple_case {
                    // Простой случай: одна переменная
                    if let UnpackPattern::Variable(name) = &pattern[0] {
                        let index = self.declare_local(name);
                        vec![Some(index)]
                    } else {
                        vec![None]
                    }
                } else {
                    self.declare_unpack_pattern_variables(pattern, *line)?
                };
                
                // Создаем метки для цикла
                let loop_start_label = self.create_label();
                let continue_label = self.create_label();
                let loop_end_label = self.create_label();
                
                // Помечаем начало цикла
                self.mark_label(loop_start_label);
                
                // Проверяем условие: индекс < длина массива
                // Загружаем индекс
                self.chunk.write_with_line(OpCode::LoadLocal(index_local), *line);
                // Загружаем массив
                self.chunk.write_with_line(OpCode::LoadLocal(array_local), *line);
                // Получаем длину массива
                self.chunk.write_with_line(OpCode::GetArrayLength, *line);
                // Сравниваем индекс < длина
                self.chunk.write_with_line(OpCode::Less, *line);
                
                // Если условие false, выходим из цикла
                self.emit_jump(true, loop_end_label)?;
                
                // Загружаем элемент массива по индексу
                self.chunk.write_with_line(OpCode::LoadLocal(array_local), *line);
                self.chunk.write_with_line(OpCode::LoadLocal(index_local), *line);
                self.chunk.write_with_line(OpCode::GetArrayElement, *line);
                
                // Распаковываем элемент в переменные (или просто присваиваем для простого случая)
                // Элемент находится на стеке
                if is_simple_case {
                    // Простой случай: просто сохраняем элемент в переменную
                    if let Some(local_index) = var_locals[0] {
                        self.chunk.write_with_line(OpCode::StoreLocal(local_index), *line);
                    }
                } else {
                    self.compile_unpack_pattern(pattern, &var_locals, expected_count, *line)?;
                }
                
                // Создаем контекст цикла
                let loop_context = LoopContext {
                    continue_label: continue_label, // continue переходит к инкременту
                    break_label: loop_end_label,    // break переходит к концу цикла
                };
                self.loop_contexts.push(loop_context);
                
                // Компилируем тело цикла
                for stmt in body {
                    self.compile_stmt(stmt)?;
                }
                
                // Помечаем метку continue (начало инкремента индекса)
                self.mark_label(continue_label);
                
                // Инкрементируем индекс
                self.chunk.write_with_line(OpCode::LoadLocal(index_local), *line);
                let one_index = self.chunk.add_constant(Value::Number(1.0));
                self.chunk.write_with_line(OpCode::Constant(one_index), *line);
                self.chunk.write_with_line(OpCode::Add, *line);
                self.chunk.write_with_line(OpCode::StoreLocal(index_local), *line);
                
                // Переход к началу цикла
                self.emit_loop(loop_start_label)?;
                
                // Помечаем конец цикла
                self.mark_label(loop_end_label);
                
                // Контекст цикла уже содержит метки для break/continue
                
                // Очищаем стек от условия (оно уже удалено JumpIfFalse, но на всякий случай)
                
                // Заканчиваем область видимости
                self.end_scope();
                self.loop_contexts.pop();
            }
            Stmt::Function { name, params, body, is_cached, line } => {
                self.current_line = *line;
                // Находим индекс функции (она уже объявлена в первом проходе)
                let function_index = self.function_names.iter()
                    .position(|n| n == name)
                    .ok_or_else(|| LangError::ParseError {
                        message: format!("Function '{}' not found in forward declarations", name),
                        line: *line,
                    })?;
                
                // Получаем функцию и обновляем количество параметров и флаг кэширования
                let mut function = self.functions[function_index].clone();
                function.arity = params.len();
                function.is_cached = *is_cached;
                
                // Сохраняем имена параметров и вычисляем значения по умолчанию
                let mut param_names = Vec::new();
                let mut default_values = Vec::new();
                
                for param in params.iter() {
                    param_names.push(param.name.clone());
                    
                    // Вычисляем значение по умолчанию во время компиляции
                    if let Some(ref default_expr) = param.default_value {
                        // Пытаемся вычислить константное выражение
                        match self.evaluate_constant_expr(default_expr) {
                            Ok(Some(constant_value)) => {
                                default_values.push(Some(constant_value));
                            }
                            Ok(None) => {
                                // Выражение не константное - пока требуем константные значения по умолчанию
                                return Err(LangError::ParseError {
                                    message: format!(
                                        "Default value for parameter '{}' must be a constant expression",
                                        param.name
                                    ),
                                    line: default_expr.line(),
                                });
                            }
                            Err(e) => {
                                return Err(e);
                            }
                        }
                    } else {
                        default_values.push(None);
                    }
                }
                
                function.param_names = param_names;
                function.default_values = default_values;
                // Если кэш включен, но еще не инициализирован, инициализируем его
                if *is_cached && function.cache.is_none() {
                    use std::rc::Rc;
                    use std::cell::RefCell;
                    use crate::bytecode::function::FnCache;
                    function.cache = Some(Rc::new(RefCell::new(FnCache::new())));
                }
                
                // ВАЖНО: Сохраняем сигнатуру функции ДО компиляции тела, чтобы рекурсивные вызовы
                // могли видеть правильное количество параметров и их имена
                self.functions[function_index] = function.clone();
                
                // Сохраняем текущие локальные области видимости для доступа к переменным родительских функций
                // Это нужно для поддержки замыканий - переменные из родительских функций должны быть доступны
                let parent_locals_snapshot: Vec<std::collections::HashMap<String, usize>> = self.locals.iter()
                    .map(|scope| scope.clone())
                    .collect();
                
                // Компилируем тело функции в chunk функции
                let saved_chunk = std::mem::replace(&mut self.chunk, function.chunk.clone());
                let saved_exception_handlers = self.exception_handlers.clone();
                let saved_error_type_table = self.error_type_table.clone();
                let saved_function = self.current_function;
                let saved_local_count = self.local_count;
                self.current_function = Some(function_index);
                self.local_count = 0;
                // Очищаем обработчики и таблицу типов ошибок для новой функции
                self.exception_handlers.clear();
                self.error_type_table.clear();
                
                // Начинаем новую область видимости для функции
                self.begin_scope();
                
                // Находим переменные, которые используются в теле функции, но не объявлены в ней
                // (захваченные из родительских функций)
                let param_names: Vec<String> = params.iter().map(|p| p.name.clone()).collect();
                let captured_vars = self.find_captured_variables(body, &parent_locals_snapshot, &param_names);
        
                
                // Создаем локальные слоты для захваченных переменных (перед параметрами)
                // Эти слоты будут заполнены значениями из родительских функций при вызове
                let mut captured_vars_info = Vec::new();
                
                
                for var_name in &captured_vars {
                    let local_slot_index = self.declare_local(var_name);
                    
                    // Находим slot index в родительской функции и глубину предка
                    // Ищем переменную в parent_locals_snapshot (от последней области к первой)
                    // Последняя область - это самая внутренняя, которая соответствует текущему контексту
                    // Это должна быть область видимости непосредственного родителя (родительской функции)
                    let mut parent_slot_index = None;
                    let mut ancestor_depth = 0;
                    // Ищем в обратном порядке, чтобы найти в самой внутренней (последней) области сначала
                    // parent_locals_snapshot[0] - самый внешний предок (например, outer)
                    // parent_locals_snapshot[n-1] - ближайший родитель (например, middle)
                    
                    
                    for (depth, scope) in parent_locals_snapshot.iter().rev().enumerate() {
                        if let Some(&slot_idx) = scope.get(var_name) {
                            parent_slot_index = Some(slot_idx);
                            // depth = 0 означает ближайший родитель, depth = 1 означает дедушку и т.д.
                            ancestor_depth = depth;
                            
                            break;
                        }
                    }
                    
                    // Если переменная не найдена в родительских локальных, она может быть глобальной
                    // В этом случае мы не можем захватить её как локальную переменную
                    // Но для правильного кода она должна быть найдена
                    if parent_slot_index.is_none() {
                        return Err(LangError::ParseError {
                            message: format!("Captured variable '{}' not found in parent scopes", var_name),
                            line: *line,
                        });
                    }
                    
                    let parent_slot = parent_slot_index.unwrap();
                    
                    captured_vars_info.push(CapturedVar {
                        name: var_name.clone(),
                        parent_slot_index: parent_slot,
                        local_slot_index,
                        ancestor_depth,
                    });
                }
                
                // Объявляем параметры как локальные переменные (после захваченных переменных)
                for param in params {
                    self.declare_local(&param.name);
                }
                
                // Компилируем тело функции
                // При компиляции переменных, resolve_local будет искать в текущих областях видимости,
                // которые включают параметры функции. Если переменная не найдена, она будет искаться
                // в родительских областях через parent_locals_snapshot в resolve_local
                for stmt in body {
                    self.compile_stmt(stmt)?;
                }
                
                // Если функция не вернула значение явно, добавляем неявный return
                // который вернет последнее значение на стеке (если есть) или null
                if self.chunk.code.is_empty() || 
                   !matches!(self.chunk.code.last(), Some(OpCode::Return)) {
                    // Не добавляем Constant(Null) - просто возвращаем то, что на стеке
                    self.chunk.write_with_line(OpCode::Return, *line);
                }
                
                // Заканчиваем область видимости функции
                self.end_scope();
                
                // Эталонный алгоритм апгрейда jump-инструкций: стабилизация layout и финализация
                self.stabilize_layout()?;
                self.finalize_jumps()?;
                
                // Сохраняем скомпилированную функцию (обработчики уже сохранены в chunk при компиляции try/catch)
                let function_chunk = std::mem::replace(&mut self.chunk, saved_chunk);
                function.chunk = function_chunk;
                function.captured_vars = captured_vars_info;
                self.functions[function_index] = function;
                
                // Восстанавливаем состояние компилятора
                self.exception_handlers = saved_exception_handlers;
                self.error_type_table = saved_error_type_table;
                self.current_function = saved_function;
                self.local_count = saved_local_count;
                
                // Сохраняем функцию в глобальную таблицу (уже сделано в первом проходе)
                let global_index = *self.globals.get(name).unwrap();
                
                // Сохраняем имя глобальной переменной для использования в JOIN
                self.chunk.global_names.insert(global_index, name.clone());
                
                // Сохраняем функцию как константу
                let constant_index = self.chunk.add_constant(Value::Function(function_index));
                
                // Сохраняем функцию в глобальную переменную
                self.chunk.write_with_line(OpCode::Constant(constant_index), *line);
                self.chunk.write_with_line(OpCode::StoreGlobal(global_index), *line);
            }
            Stmt::Return { value, line } => {
                self.current_line = *line;
                if let Some(expr) = value {
                    self.compile_expr(expr)?;
                } else {
                    let const_index = self.chunk.add_constant(Value::Null);
                    self.chunk.write_with_line(OpCode::Constant(const_index), *line);
                }
                self.chunk.write_with_line(OpCode::Return, *line);
            }
            Stmt::Break { line } => {
                self.current_line = *line;
                if self.loop_contexts.is_empty() {
                    return Err(LangError::ParseError {
                        message: "break statement outside of loop".to_string(),
                        line: *line,
                    });
                }
                // Jump к метке конца цикла
                let break_label = self.loop_contexts.last().unwrap().break_label;
                self.emit_jump(false, break_label)?;
            }
            Stmt::Continue { line } => {
                self.current_line = *line;
                if self.loop_contexts.is_empty() {
                    return Err(LangError::ParseError {
                        message: "continue statement outside of loop".to_string(),
                        line: *line,
                    });
                }
                // Jump к метке continue
                let continue_label = self.loop_contexts.last().unwrap().continue_label;
                self.emit_jump(false, continue_label)?;
            }
            Stmt::Throw { value, line } => {
                self.current_line = *line;
                // Компилируем выражение (оно оставит значение на стеке)
                self.compile_expr(value)?;
                // Генерируем Throw опкод (None означает RuntimeError без конкретного типа)
                self.chunk.write_with_line(OpCode::Throw(None), *line);
            }
            Stmt::Try { try_block, catch_blocks, else_block, line } => {
                self.current_line = *line;
                self.compile_try(try_block, catch_blocks, else_block.as_deref(), *line)?;
            }
        }
        Ok(())
    }


    /// Разрешает аргументы функции: именованные -> позиционные, применяет значения по умолчанию
    fn resolve_function_args(
        &self,
        function_name: &str,
        args: &[Arg],
        function_info: Option<(usize, &Function)>,
        line: usize,
    ) -> Result<Vec<Arg>, LangError> {
        // Если это встроенная функция, проверяем, поддерживает ли она именованные аргументы
        if function_info.is_none() {
            // Проверяем, есть ли именованные аргументы
            let has_named = args.iter().any(|a| matches!(a, Arg::Named { .. }));
            
            if has_named {
                // Проверяем, поддерживает ли эта нативная функция именованные аргументы
                if let Some(param_names) = self.get_native_function_params(function_name) {
                    // Нативная функция поддерживает именованные аргументы
                    // Разрешаем их аналогично пользовательским функциям
                    let mut resolved = vec![None; param_names.len()];
                    let mut positional_count = 0;
                    
                    // Для методов объектов (например, nn_train), первый параметр - это объект,
                    // который не передается в args метода, поэтому пропускаем позицию 0
                    let start_position = if (function_name == "nn_train" || function_name == "nn_train_sh") && !param_names.is_empty() && param_names[0] == "nn" {
                        1  // Пропускаем первый параметр "nn" (объект метода)
                    } else {
                        0
                    };
                    
                    // Обрабатываем аргументы
                    for arg in args {
                        match arg {
                            Arg::Positional(expr) => {
                                let target_position = start_position + positional_count;
                                if target_position >= param_names.len() {
                                    return Err(LangError::ParseError {
                                        message: format!(
                                            "Function '{}' takes at most {} arguments but {} positional arguments were provided",
                                            function_name, param_names.len() - start_position, positional_count + 1
                                        ),
                                        line,
                                    });
                                }
                                resolved[target_position] = Some(Arg::Positional(expr.clone()));
                                positional_count += 1;
                            }
                            Arg::Named { name, value } => {
                                // Находим индекс параметра по имени
                                if let Some(param_index) = param_names.iter().position(|n| n == name) {
                                    if resolved[param_index].is_some() {
                                        return Err(LangError::ParseError {
                                            message: format!(
                                                "Function '{}' got multiple values for argument '{}'",
                                                function_name, name
                                            ),
                                            line,
                                        });
                                    }
                                    resolved[param_index] = Some(Arg::Named {
                                        name: name.clone(),
                                        value: value.clone(),
                                    });
                                } else {
                                    return Err(LangError::ParseError {
                                        message: format!(
                                            "Function '{}' got an unexpected keyword argument '{}'",
                                            function_name, name
                                        ),
                                        line,
                                    });
                                }
                            }
                        }
                    }
                    
                    // Собираем итоговый список аргументов в правильном порядке
                    // Включаем только предоставленные параметры (нативные функции сами обрабатывают опциональные)
                    let mut final_args = Vec::new();
                    for i in 0..param_names.len() {
                        if let Some(arg) = resolved[i].take() {
                            match arg {
                                Arg::Positional(expr) => final_args.push(Arg::Positional(expr)),
                                Arg::Named { value, .. } => final_args.push(Arg::Positional(value)),
                            }
                        }
                    }
                    
                    return Ok(final_args);
                } else {
                    // Нативная функция не поддерживает именованные аргументы (например, print, min, max)
                    return Err(LangError::ParseError {
                        message: format!(
                            "Named arguments are not supported for built-in function '{}'",
                            function_name
                        ),
                        line,
                    });
                }
            } else {
                // Нет именованных аргументов, просто возвращаем позиционные
                return Ok(args.iter().map(|a| match a {
                    Arg::Positional(e) => Arg::Positional(e.clone()),
                    Arg::Named { .. } => unreachable!(),
                }).collect());
            }
        }
        
        let (_, function) = function_info.unwrap();
        let param_names = &function.param_names;
        let default_values = &function.default_values;
        
        // Создаем массив для разрешенных аргументов
        let mut resolved = vec![None; param_names.len()];
        let mut positional_count = 0;
        
        // Обрабатываем аргументы
        for arg in args {
            match arg {
                Arg::Positional(expr) => {
                    if positional_count >= param_names.len() {
                        return Err(LangError::ParseError {
                            message: format!(
                                "Function '{}' takes {} arguments but {} positional arguments were provided",
                                function_name, param_names.len(), positional_count + 1
                            ),
                            line,
                        });
                    }
                    resolved[positional_count] = Some(Arg::Positional(expr.clone()));
                    positional_count += 1;
                }
                Arg::Named { name, value } => {
                    // Находим индекс параметра по имени
                    if let Some(param_index) = param_names.iter().position(|n| n == name) {
                        if resolved[param_index].is_some() {
                            return Err(LangError::ParseError {
                                message: format!(
                                    "Function '{}' got multiple values for argument '{}'",
                                    function_name, name
                                ),
                                line,
                            });
                        }
                        resolved[param_index] = Some(Arg::Named {
                            name: name.clone(),
                            value: value.clone(),
                        });
                    } else {
                        return Err(LangError::ParseError {
                            message: format!(
                                "Function '{}' got an unexpected keyword argument '{}'",
                                function_name, name
                            ),
                            line,
                        });
                    }
                }
            }
        }
        
        // Применяем значения по умолчанию для незаполненных параметров
        let mut final_args = Vec::new();
        for (i, param_name) in param_names.iter().enumerate() {
            if let Some(arg) = resolved[i].take() {
                // Аргумент был предоставлен
                match arg {
                    Arg::Positional(expr) => {
                        final_args.push(Arg::Positional(expr));
                    }
                    Arg::Named { value, .. } => {
                        final_args.push(Arg::Positional(value));
                    }
                }
            } else {
                // Аргумент не был предоставлен - используем значение по умолчанию
                if let Some(default_value) = default_values.get(i).and_then(|v| v.as_ref()) {
                    // Значение по умолчанию было вычислено во время компиляции
                    final_args.push(Arg::Positional(Expr::Literal {
                        value: default_value.clone(),
                        line,
                    }));
                } else {
                    // Обязательный параметр не был предоставлен
                    return Err(LangError::ParseError {
                        message: format!(
                            "Function '{}' missing required argument '{}'",
                            function_name, param_name
                        ),
                        line,
                    });
                }
            }
        }
        
        Ok(final_args)
    }

    fn compile_expr(&mut self, expr: &Expr) -> Result<(), LangError> {
        let expr_line = expr.line();
        self.current_line = expr_line;
        
        // Оптимизация: вычисляем константные выражения во время компиляции
        if let Some(constant_value) = self.evaluate_constant_expr(expr)? {
            let constant_index = self.chunk.add_constant(constant_value);
            self.chunk.write_with_line(OpCode::Constant(constant_index), expr_line);
            return Ok(());
        }
        
        match expr {
            Expr::Literal { value, line } => {
                let constant_index = self.chunk.add_constant(value.clone());
                self.chunk.write_with_line(OpCode::Constant(constant_index), *line);
            }
            Expr::Assign { name, value, line } => {
                // Компилируем значение
                self.compile_expr(value)?;
                // Не клонируем автоматически - переменные должны разделять ссылки на массивы/таблицы/объекты
                // Клонирование происходит только при явном вызове .clone()
                // Проверяем, является ли переменная локальной или глобальной
                if let Some(local_index) = self.resolve_local(name) {
                    // Локальная переменная найдена - обновляем
                    self.chunk.write_with_line(OpCode::StoreLocal(local_index), *line);
                    self.chunk.write_with_line(OpCode::LoadLocal(local_index), *line);
                } else if self.current_function.is_some() {
                    // Мы находимся внутри функции - создаем локальную переменную, которая скрывает глобальную
                    let index = self.declare_local(name);
                    self.chunk.write_with_line(OpCode::StoreLocal(index), *line);
                    self.chunk.write_with_line(OpCode::LoadLocal(index), *line);
                } else if let Some(&global_index) = self.globals.get(name) {
                    // Мы в главной функции, глобальная переменная найдена - обновляем
                    // Сохраняем имя глобальной переменной для использования в JOIN
                    self.chunk.global_names.insert(global_index, name.clone());
                    self.chunk.write_with_line(OpCode::StoreGlobal(global_index), *line);
                    self.chunk.write_with_line(OpCode::LoadGlobal(global_index), *line);
                } else {
                    // Переменная не найдена ни локально, ни глобально
                    if self.current_function.is_some() {
                        // Мы находимся внутри функции - создаем локальную переменную
                        let index = self.declare_local(name);
                        self.chunk.write_with_line(OpCode::StoreLocal(index), *line);
                        self.chunk.write_with_line(OpCode::LoadLocal(index), *line);
                    } else {
                        // На верхнем уровне - создаем новую глобальную переменную
                        let global_index = self.globals.len();
                        self.globals.insert(name.clone(), global_index);
                        self.chunk.global_names.insert(global_index, name.clone());
                        self.chunk.write_with_line(OpCode::StoreGlobal(global_index), *line);
                        self.chunk.write_with_line(OpCode::LoadGlobal(global_index), *line);
                    }
                }
            }
            Expr::UnpackAssign { names, value, line } => {
                // Распаковка кортежа: a, b, c = tuple_expr
                // Компилируем правую часть (должна вернуть кортеж)
                self.compile_expr(value)?;
                
                // Сохраняем кортеж во временную переменную, чтобы можно было извлекать элементы
                let tuple_temp = self.declare_local(&format!("__tuple_temp_{}", line));
                self.chunk.write_with_line(OpCode::StoreLocal(tuple_temp), *line);
                
                // Для каждой переменной извлекаем элемент кортежа и сохраняем
                for (index, name) in names.iter().enumerate() {
                    // Загружаем кортеж
                    self.chunk.write_with_line(OpCode::LoadLocal(tuple_temp), *line);
                    // Загружаем индекс
                    let index_const = self.chunk.add_constant(Value::Number(index as f64));
                    self.chunk.write_with_line(OpCode::Constant(index_const), *line);
                    // Получаем элемент по индексу
                    self.chunk.write_with_line(OpCode::GetArrayElement, *line);
                    
                    // Сохраняем в переменную
                    // Проверяем, находимся ли мы в контексте let statement (по имени первой переменной)
                    // Если это let statement, нужно объявить все переменные
                    if let Some(local_index) = self.resolve_local(name) {
                        // Локальная переменная найдена - обновляем
                        self.chunk.write_with_line(OpCode::StoreLocal(local_index), *line);
                    } else if self.current_function.is_some() {
                        // Мы находимся внутри функции - создаем локальную переменную
                        let var_index = self.declare_local(name);
                        self.chunk.write_with_line(OpCode::StoreLocal(var_index), *line);
                    } else {
                        // На верхнем уровне - проверяем, есть ли глобальная переменная
                        if let Some(&global_index) = self.globals.get(name) {
                            // Глобальная переменная найдена - обновляем
                            self.chunk.global_names.insert(global_index, name.clone());
                            self.chunk.write_with_line(OpCode::StoreGlobal(global_index), *line);
                        } else {
                            // Новая глобальная переменная на верхнем уровне
                            let global_index = self.globals.len();
                            self.globals.insert(name.clone(), global_index);
                            self.chunk.global_names.insert(global_index, name.clone());
                            self.chunk.write_with_line(OpCode::StoreGlobal(global_index), *line);
                        }
                    }
                }
                
                // Удаляем временную переменную кортежа
                // (она будет автоматически удалена при выходе из области видимости)
                
                // Загружаем последнее значение на стек (для возврата результата)
                if let Some(last_name) = names.last() {
                    if let Some(local_index) = self.resolve_local(last_name) {
                        self.chunk.write_with_line(OpCode::LoadLocal(local_index), *line);
                    } else if let Some(&global_index) = self.globals.get(last_name) {
                        self.chunk.write_with_line(OpCode::LoadGlobal(global_index), *line);
                    }
                }
            }
            Expr::AssignOp { name, op, value, line } => {
                // Оператор присваивания: a += b эквивалентно a = a + b
                // Проверяем, является ли переменная локальной или глобальной
                if let Some(local_index) = self.resolve_local(name) {
                    // Локальная переменная найдена - загружаем её значение
                    self.chunk.write_with_line(OpCode::LoadLocal(local_index), *line);
                    
                    // Компилируем правую часть
                    self.compile_expr(value)?;
                    
                    // Выполняем операцию
                    match op {
                        TokenKind::PlusEqual => {
                            self.chunk.write_with_line(OpCode::Add, *line);
                        }
                        TokenKind::MinusEqual => {
                            self.chunk.write_with_line(OpCode::Sub, *line);
                        }
                        TokenKind::StarEqual => {
                            self.chunk.write_with_line(OpCode::Mul, *line);
                        }
                        TokenKind::StarStarEqual => {
                            self.chunk.write_with_line(OpCode::Pow, *line);
                        }
                        TokenKind::SlashEqual => {
                            self.chunk.write_with_line(OpCode::Div, *line);
                        }
                        TokenKind::SlashSlashEqual => {
                            self.chunk.write_with_line(OpCode::IntDiv, *line);
                        }
                        TokenKind::PercentEqual => {
                            self.chunk.write_with_line(OpCode::Mod, *line);
                        }
                        _ => {
                            return Err(LangError::ParseError {
                                message: format!("Unknown assignment operator: {:?}", op),
                                line: *line,
                            });
                        }
                    }
                    
                    // Сохраняем результат обратно в локальную переменную
                    self.chunk.write_with_line(OpCode::StoreLocal(local_index), *line);
                    // Загружаем значение обратно, чтобы присваивание возвращало значение
                    self.chunk.write_with_line(OpCode::LoadLocal(local_index), *line);
                } else if self.current_function.is_some() {
                    // Мы находимся внутри функции - создаем локальную переменную, которая скрывает глобальную
                    let index = self.declare_local(name);
                    // Загружаем 0 как начальное значение
                    let zero_index = self.chunk.add_constant(Value::Number(0.0));
                    self.chunk.write_with_line(OpCode::Constant(zero_index), *line);
                    
                    // Компилируем правую часть
                    self.compile_expr(value)?;
                    
                    // Выполняем операцию
                    match op {
                        TokenKind::PlusEqual => {
                            self.chunk.write_with_line(OpCode::Add, *line);
                        }
                        TokenKind::MinusEqual => {
                            self.chunk.write_with_line(OpCode::Sub, *line);
                        }
                        TokenKind::StarEqual => {
                            self.chunk.write_with_line(OpCode::Mul, *line);
                        }
                        TokenKind::StarStarEqual => {
                            self.chunk.write_with_line(OpCode::Pow, *line);
                        }
                        TokenKind::SlashEqual => {
                            self.chunk.write_with_line(OpCode::Div, *line);
                        }
                        TokenKind::SlashSlashEqual => {
                            self.chunk.write_with_line(OpCode::IntDiv, *line);
                        }
                        TokenKind::PercentEqual => {
                            self.chunk.write_with_line(OpCode::Mod, *line);
                        }
                        _ => {
                            return Err(LangError::ParseError {
                                message: format!("Unknown assignment operator: {:?}", op),
                                line: *line,
                            });
                        }
                    }
                    
                    // Сохраняем результат обратно в локальную переменную
                    self.chunk.write_with_line(OpCode::StoreLocal(index), *line);
                    // Загружаем значение обратно, чтобы присваивание возвращало значение
                    self.chunk.write_with_line(OpCode::LoadLocal(index), *line);
                } else if let Some(&global_index) = self.globals.get(name) {
                    // Мы в главной функции, глобальная переменная найдена - загружаем её значение
                    self.chunk.write_with_line(OpCode::LoadGlobal(global_index), *line);
                    
                    // Компилируем правую часть
                    self.compile_expr(value)?;
                    
                    // Выполняем операцию
                    match op {
                        TokenKind::PlusEqual => {
                            self.chunk.write_with_line(OpCode::Add, *line);
                        }
                        TokenKind::MinusEqual => {
                            self.chunk.write_with_line(OpCode::Sub, *line);
                        }
                        TokenKind::StarEqual => {
                            self.chunk.write_with_line(OpCode::Mul, *line);
                        }
                        TokenKind::StarStarEqual => {
                            self.chunk.write_with_line(OpCode::Pow, *line);
                        }
                        TokenKind::SlashEqual => {
                            self.chunk.write_with_line(OpCode::Div, *line);
                        }
                        TokenKind::SlashSlashEqual => {
                            self.chunk.write_with_line(OpCode::IntDiv, *line);
                        }
                        TokenKind::PercentEqual => {
                            self.chunk.write_with_line(OpCode::Mod, *line);
                        }
                        _ => {
                            return Err(LangError::ParseError {
                                message: format!("Unknown assignment operator: {:?}", op),
                                line: *line,
                            });
                        }
                    }
                    
                    // Сохраняем результат обратно в глобальную переменную
                    // Сохраняем имя глобальной переменной для использования в JOIN
                    self.chunk.global_names.insert(global_index, name.clone());
                    self.chunk.write_with_line(OpCode::StoreGlobal(global_index), *line);
                    // Загружаем значение обратно, чтобы присваивание возвращало значение
                    self.chunk.write_with_line(OpCode::LoadGlobal(global_index), *line);
                } else {
                    // Переменная не найдена ни локально, ни глобально - создаем новую локальную переменную
                    let index = self.declare_local(name);
                    // Загружаем 0 как начальное значение
                    let zero_index = self.chunk.add_constant(Value::Number(0.0));
                    self.chunk.write_with_line(OpCode::Constant(zero_index), *line);
                    
                    // Компилируем правую часть
                    self.compile_expr(value)?;
                    
                    // Выполняем операцию
                    match op {
                        TokenKind::PlusEqual => {
                            self.chunk.write_with_line(OpCode::Add, *line);
                        }
                        TokenKind::MinusEqual => {
                            self.chunk.write_with_line(OpCode::Sub, *line);
                        }
                        TokenKind::StarEqual => {
                            self.chunk.write_with_line(OpCode::Mul, *line);
                        }
                        TokenKind::StarStarEqual => {
                            self.chunk.write_with_line(OpCode::Pow, *line);
                        }
                        TokenKind::SlashEqual => {
                            self.chunk.write_with_line(OpCode::Div, *line);
                        }
                        TokenKind::SlashSlashEqual => {
                            self.chunk.write_with_line(OpCode::IntDiv, *line);
                        }
                        TokenKind::PercentEqual => {
                            self.chunk.write_with_line(OpCode::Mod, *line);
                        }
                        _ => {
                            return Err(LangError::ParseError {
                                message: format!("Unknown assignment operator: {:?}", op),
                                line: *line,
                            });
                        }
                    }
                    
                    // Сохраняем результат обратно в локальную переменную
                    self.chunk.write_with_line(OpCode::StoreLocal(index), *line);
                    // Загружаем значение обратно, чтобы присваивание возвращало значение
                    self.chunk.write_with_line(OpCode::LoadLocal(index), *line);
                }
            }
            Expr::Variable { name, line } => {
                // Определяем, глобальная или локальная переменная
                if let Some(local_index) = self.resolve_local(name) {
                    // Локальная переменная
                    self.chunk.write_with_line(OpCode::LoadLocal(local_index), *line);
                } else if let Some(&global_index) = self.globals.get(name) {
                    // Глобальная переменная или функция
                    self.chunk.write_with_line(OpCode::LoadGlobal(global_index), *line);
                } else {
                    // Переменная не найдена - создаем новый глобальный индекс
                    // Это позволит проверить переменную во время выполнения
                    let global_index = self.globals.len();
                    self.globals.insert(name.clone(), global_index);
                    self.chunk.global_names.insert(global_index, name.clone());
                    self.chunk.write_with_line(OpCode::LoadGlobal(global_index), *line);
                }
            }
            Expr::Unary { op, right, line } => {
                self.current_line = *line;
                match op {
                    TokenKind::Minus => {
                        // Унарный минус: -value
                        self.compile_expr(right)?;
                        self.chunk.write_with_line(OpCode::Negate, *line);
                    }
                    TokenKind::Bang => {
                        // Унарный Bang: !value (логическое отрицание)
                        self.compile_expr(right)?;
                        self.chunk.write_with_line(OpCode::Not, *line);
                    }
                    _ => {
                        return Err(LangError::ParseError {
                            message: format!("Unknown unary operator: {:?}", op),
                            line: *line,
                        });
                    }
                }
            }
            Expr::Binary { left, op, right, line } => {
                self.current_line = *line;
                // Специальная обработка логических операторов
                if *op == TokenKind::EqualEqual {
                    self.compile_expr(left)?;
                    self.compile_expr(right)?;
                    self.chunk.write_with_line(OpCode::Equal, *line);
                } else if *op == TokenKind::BangEqual {
                    self.compile_expr(left)?;
                    self.compile_expr(right)?;
                    self.chunk.write_with_line(OpCode::NotEqual, *line);
                } else {
                    self.compile_expr(left)?;
                    self.compile_expr(right)?;
                    match op {
                        TokenKind::Plus => self.chunk.write_with_line(OpCode::Add, *line),
                        TokenKind::Minus => self.chunk.write_with_line(OpCode::Sub, *line),
                        TokenKind::Star => self.chunk.write_with_line(OpCode::Mul, *line),
                        TokenKind::StarStar => self.chunk.write_with_line(OpCode::Pow, *line),
                        TokenKind::Slash => self.chunk.write_with_line(OpCode::Div, *line),
                        TokenKind::SlashSlash => self.chunk.write_with_line(OpCode::IntDiv, *line),
                        TokenKind::Percent => self.chunk.write_with_line(OpCode::Mod, *line),
                        TokenKind::Greater => self.chunk.write_with_line(OpCode::Greater, *line),
                        TokenKind::Less => self.chunk.write_with_line(OpCode::Less, *line),
                        TokenKind::GreaterEqual => self.chunk.write_with_line(OpCode::GreaterEqual, *line),
                        TokenKind::LessEqual => self.chunk.write_with_line(OpCode::LessEqual, *line),
                        TokenKind::In => self.chunk.write_with_line(OpCode::In, *line),
                        TokenKind::Or => self.chunk.write_with_line(OpCode::Or, *line),
                        TokenKind::And => self.chunk.write_with_line(OpCode::And, *line),
                        _ => {
                            return Err(LangError::ParseError {
                                message: format!("Unknown binary operator: {:?}", op),
                                line: *line,
                            });
                        }
                    }
                }
            }
            Expr::Call { name, args, line } => {
                self.current_line = *line;
                
                // Находим функцию для получения информации о параметрах
                let function_info = if let Some(function_index) = self.function_names.iter().position(|n| n == name) {
                    Some((function_index, &self.functions[function_index]))
                } else {
                    None
                };
                
                // Разрешаем аргументы: именованные -> позиционные, применяем значения по умолчанию
                let resolved_args = self.resolve_function_args(name, args, function_info, *line)?;
                
                // Специальная обработка для isinstance: преобразуем идентификаторы типов в строки
                let processed_args = if name == "isinstance" && resolved_args.len() >= 2 {
                    let mut new_args = resolved_args.clone();
                    if let Arg::Positional(Expr::Variable { name: type_name, .. }) = &resolved_args[1] {
                        let type_names = vec!["int", "str", "bool", "array", "null", "num", "float"];
                        if type_names.contains(&type_name.as_str()) {
                            new_args[1] = Arg::Positional(Expr::Literal {
                                value: Value::String(type_name.clone()),
                                line: match &resolved_args[1] {
                                    Arg::Positional(e) => e.line(),
                                    Arg::Named { value, .. } => value.line(),
                                },
                            });
                        }
                    }
                    new_args
                } else {
                    resolved_args
                };
                
                // Специальная обработка для функций, которые модифицируют первый аргумент in-place
                let in_place_functions = vec!["push", "reverse", "sort"];
                let should_assign_back = in_place_functions.contains(&name.as_str()) 
                    && !processed_args.is_empty()
                    && matches!(&processed_args[0], Arg::Positional(Expr::Variable { .. }));
                
                // Если нужно присвоить обратно, сохраняем имя переменной
                let var_name_to_assign = if should_assign_back {
                    if let Arg::Positional(Expr::Variable { name: var_name, .. }) = &processed_args[0] {
                        Some(var_name.clone())
                    } else {
                        None
                    }
                } else {
                    None
                };
                
                // Компилируем аргументы на стек
                for arg in &processed_args {
                    match arg {
                        Arg::Positional(expr) => {
                            self.compile_expr(expr)?;
                        }
                        Arg::Named { value, .. } => {
                            // Именованные аргументы уже разрешены в позиционные
                            self.compile_expr(value)?;
                        }
                    }
                }
                
                // Загружаем функцию: сначала проверяем переменные (локальные или глобальные),
                // затем ищем функцию по имени
                if let Some(local_index) = self.resolve_local(name) {
                    // Локальная переменная содержит функцию
                    self.chunk.write_with_line(OpCode::LoadLocal(local_index), self.current_line);
                } else if let Some(&global_index) = self.globals.get(name) {
                    // Глобальная переменная содержит функцию
                    self.chunk.write_with_line(OpCode::LoadGlobal(global_index), self.current_line);
                } else {
                    // Ищем функцию по имени в списке функций
                    if let Some(function_index) = self.function_names.iter().position(|n| n == name) {
                        let constant_index = self.chunk.add_constant(Value::Function(function_index));
                        self.chunk.write_with_line(OpCode::Constant(constant_index), self.current_line);
                    } else {
                        // Функция не найдена - это может быть функция, импортированная через import *
                        // Регистрируем её как глобальную переменную для разрешения во время выполнения
                        let global_index = if let Some(&idx) = self.globals.get(name) {
                            idx
                        } else {
                            let idx = self.globals.len();
                            self.globals.insert(name.clone(), idx);
                            self.chunk.global_names.insert(idx, name.clone());
                            idx
                        };
                        self.chunk.write_with_line(OpCode::LoadGlobal(global_index), self.current_line);
                    }
                }
                
                // Вызываем функцию с количеством аргументов
                self.chunk.write_with_line(OpCode::Call(processed_args.len()), *line);
                
                // Если нужно присвоить результат обратно в переменную
                if let Some(var_name) = var_name_to_assign {
                    // Определяем, глобальная или локальная переменная
                    let is_local = !self.locals.is_empty() && self.resolve_local(&var_name).is_some();
                    
                    if is_local {
                        // Локальная переменная
                        let local_index = self.resolve_local(&var_name).unwrap();
                        self.chunk.write_with_line(OpCode::StoreLocal(local_index), *line);
                        // Загружаем значение обратно на стек, чтобы выражение возвращало результат
                        self.chunk.write_with_line(OpCode::LoadLocal(local_index), *line);
                    } else {
                        // Глобальная переменная
                        let global_index = if let Some(&idx) = self.globals.get(&var_name) {
                            idx
                        } else {
                            let idx = self.globals.len();
                            self.globals.insert(var_name.clone(), idx);
                            idx
                        };
                        // Сохраняем имя глобальной переменной для использования в JOIN
                        self.chunk.global_names.insert(global_index, var_name.clone());
                        self.chunk.write_with_line(OpCode::StoreGlobal(global_index), *line);
                        // Загружаем значение обратно на стек, чтобы выражение возвращало результат
                        self.chunk.write_with_line(OpCode::LoadGlobal(global_index), *line);
                    }
                }
            }
            Expr::ArrayLiteral { elements, line } => {
                // Компилируем каждый элемент массива
                for element in elements {
                    self.compile_expr(element)?;
                }
                // Создаем массив из элементов на стеке
                let arity = elements.len();
                self.chunk.write_with_line(OpCode::MakeArray(arity), *line);
            }
            Expr::TupleLiteral { elements, line } => {
                // Компилируем каждый элемент кортежа
                for element in elements {
                    self.compile_expr(element)?;
                }
                // Создаем кортеж из элементов на стеке
                // Используем MakeArray, но в VM будем создавать Tuple
                let arity = elements.len();
                self.chunk.write_with_line(OpCode::MakeTuple(arity), *line);
            }
            Expr::ArrayIndex { array, index, line } => {
                // Компилируем выражение массива (оно должно быть на стеке первым)
                self.compile_expr(array)?;
                // Компилируем индексное выражение
                self.compile_expr(index)?;
                // Получаем элемент массива по индексу
                self.chunk.write_with_line(OpCode::GetArrayElement, *line);
            }
            Expr::Property { object, name, line } => {
                // Компилируем объект
                self.compile_expr(object)?;
                // Для table.idx мы просто оставляем таблицу на стеке
                // Затем при индексации [i] это будет обработано как table[i]
                if name != "idx" {
                    // Для других свойств создаем строку и используем индексацию
                    let name_index = self.chunk.add_constant(Value::String(name.clone()));
                    self.chunk.write_with_line(OpCode::Constant(name_index), *line);
                    self.chunk.write_with_line(OpCode::GetArrayElement, *line);
                }
                // Для "idx" просто оставляем объект на стеке
            }
            Expr::MethodCall { object, method, args, line } => {
                // Компилируем объект
                self.compile_expr(object)?;
                
                // Специальная обработка для метода clone()
                if method == "clone" {
                    // Для clone() не нужны аргументы
                    if !args.is_empty() {
                        return Err(LangError::ParseError {
                            message: "clone() method takes no arguments".to_string(),
                            line: *line,
                        });
                    }
                    // Используем специальный opcode для клонирования
                    // Пока используем существующий механизм - просто клонируем значение на стеке
                    // Это будет обработано в VM
                    self.chunk.write_with_line(OpCode::Clone, *line);
                } else if method == "suffixes" {
                    // Метод suffixes для применения суффиксов к колонкам таблицы
                    // Проверяем количество аргументов (должно быть 2)
                    if args.len() != 2 {
                        return Err(LangError::ParseError {
                            message: format!("suffixes() method expects 2 arguments (left_suffix, right_suffix), got {}", args.len()),
                            line: *line,
                        });
                    }
                    
                    // Сохраняем объект (таблицу) во временную локальную переменную
                    let temp_object_slot = self.declare_local("__method_object");
                    self.chunk.write_with_line(OpCode::StoreLocal(temp_object_slot), *line);
                    
                    // Компилируем аргументы в нормальном порядке (left_suffix, right_suffix)
                    for arg in args {
                        match arg {
                            Arg::Positional(expr) => self.compile_expr(expr)?,
                            Arg::Named { value, .. } => self.compile_expr(value)?,
                        }
                    }
                    
                    // Загружаем объект обратно (он должен быть первым аргументом)
                    // Порядок на стеке должен быть: [left_suffix, right_suffix, table]
                    // Но при вызове Call аргументы извлекаются в обратном порядке и реверсируются
                    // Поэтому нужно: [table, right_suffix, left_suffix] на стеке
                    // Чтобы получить это, загружаем объект последним
                    self.chunk.write_with_line(OpCode::LoadLocal(temp_object_slot), *line);
                    
                    // Находим индекс функции table_suffixes и загружаем её на стек
                    if let Some(&function_index) = self.globals.get("table_suffixes") {
                        // Загружаем функцию на стек
                        // Порядок на стеке: [left_suffix, right_suffix, table, function]
                        // При Call(3): извлекается function, table, right_suffix, left_suffix
                        // Реверсируется: [left_suffix, right_suffix, table] - неправильно!
                        // Нужно: [table, left_suffix, right_suffix]
                        // Поэтому нужно переставить аргументы
                        
                        // Удаляем аргументы со стека
                        for _ in 0..3 {
                            self.chunk.write_with_line(OpCode::Pop, *line);
                        }
                        
                        // Загружаем в правильном порядке: table, left_suffix, right_suffix
                        self.chunk.write_with_line(OpCode::LoadLocal(temp_object_slot), *line);
                        // Компилируем аргументы заново
                        for arg in args {
                            match arg {
                                Arg::Positional(expr) => self.compile_expr(expr)?,
                                Arg::Named { value, .. } => self.compile_expr(value)?,
                            }
                        }
                        
                        // Загружаем функцию на стек
                        self.chunk.write_with_line(OpCode::LoadGlobal(function_index), *line);
                        // Вызываем функцию с 3 аргументами: table, left_suffix, right_suffix
                        self.chunk.write_with_line(OpCode::Call(3), *line);
                    } else {
                        return Err(LangError::ParseError {
                            message: "Function 'table_suffixes' not found".to_string(),
                            line: *line,
                        });
                    }
                } else if matches!(method.as_str(), "inner_join" | "left_join" | "right_join" | "full_join" | 
                                   "cross_join" | "semi_join" | "anti_join" | "zip_join" | "asof_join" | 
                                   "apply_join" | "join_on") {
                    // JOIN методы для таблиц
                    // Объект уже на стеке. Нужно вызвать функцию с объектом как первым аргументом.
                    // Сохраняем объект во временную локальную переменную
                    let temp_object_slot = self.declare_local("__method_object");
                    self.chunk.write_with_line(OpCode::StoreLocal(temp_object_slot), *line);
                    
                    // Компилируем аргументы в нормальном порядке
                    for arg in args {
                        match arg {
                            Arg::Positional(expr) => self.compile_expr(expr)?,
                            Arg::Named { value, .. } => self.compile_expr(value)?,
                        }
                    }
                    
                    // Загружаем объект обратно (он должен быть первым аргументом)
                    self.chunk.write_with_line(OpCode::LoadLocal(temp_object_slot), *line);
                    
                    // Определяем имя функции для вызова
                    let function_name = match method.as_str() {
                        "inner_join" => "inner_join",
                        "left_join" => "left_join",
                        "right_join" => "right_join",
                        "full_join" => "full_join",
                        "cross_join" => "cross_join",
                        "semi_join" => "semi_join",
                        "anti_join" => "anti_join",
                        "zip_join" => "zip_join",
                        "asof_join" => "asof_join",
                        "apply_join" => "apply_join",
                        "join_on" => "join_on",
                        _ => unreachable!(),
                    };
                    
                    // Находим индекс функции
                    if let Some(&function_index) = self.globals.get(function_name) {
                        // На стеке сейчас: [object, arg_n, ..., arg_2, arg_1]
                        // При вызове функции аргументы извлекаются в обратном порядке: [arg_1, arg_2, ..., arg_n, object]
                        // Но нам нужно [object, arg_1, arg_2, ..., arg_n]
                        // Поэтому нужно переставить: компилируем аргументы в обратном порядке
                        
                        // Перекомпилируем аргументы в обратном порядке
                        // Удаляем текущие аргументы со стека (кроме объекта)
                        for _ in 0..args.len() {
                            self.chunk.write_with_line(OpCode::Pop, *line);
                        }
                        
                        // Компилируем аргументы в обратном порядке
                        for arg in args.iter().rev() {
                            match arg {
                                Arg::Positional(expr) => self.compile_expr(expr)?,
                                Arg::Named { value, .. } => self.compile_expr(value)?,
                            }
                        }
                        
                        // Загружаем объект обратно
                        self.chunk.write_with_line(OpCode::LoadLocal(temp_object_slot), *line);
                        
                        // Загружаем функцию на стек
                        self.chunk.write_with_line(OpCode::LoadGlobal(function_index), *line);
                        
                        // Вызываем функцию с количеством аргументов (object + args)
                        self.chunk.write_with_line(OpCode::Call(args.len() + 1), *line);
                    } else {
                        return Err(LangError::ParseError {
                            message: format!("Function '{}' not found", function_name),
                            line: *line,
                        });
                    }
                } else {
                    // Общий случай: метод может быть нативной функцией или обычной функцией в объекте
                    // Объект уже на стеке. Нужно получить свойство (метод) и вызвать его.
                    // Для обычных объектов: получаем свойство объекта
                    // Сначала сохраняем объект во временную переменную
                    let temp_object_slot = self.declare_local("__method_object");
                    self.chunk.write_with_line(OpCode::StoreLocal(temp_object_slot), *line);
                    
                    // Проверяем, является ли это методом объекта (например, axis.imshow)
                    // или функцией модуля (например, ml.load_mnist)
                    // Методы объектов (Axis) нуждаются в объекте как первом аргументе,
                    // а функции модулей (ml, plot) - нет
                    // Определяем это по имени метода
                    let is_axis_method = matches!(method.as_str(), "imshow" | "set_title" | "axis");
                    let is_nn_method = matches!(method.as_str(), "device" | "get_device" | "save" | "train" | "train_sh");
                    let is_layer_method = matches!(method.as_str(), "freeze" | "unfreeze");
                    
                    if is_nn_method {
                        // Для методов device, get_device и save на NeuralNetwork, вызываем соответствующие нативные функции
                        // Загружаем объект первым
                        self.chunk.write_with_line(OpCode::LoadLocal(temp_object_slot), *line);
                        
                        // Определяем имя функции в ml модуле
                        let function_name = if method == "device" {
                            "nn_set_device"
                        } else if method == "get_device" {
                            "nn_get_device"
                        } else if method == "save" {
                            "nn_save"
                        } else if method == "train" {
                            "nn_train"
                        } else if method == "train_sh" {
                            "nn_train_sh"
                        } else {
                            return Err(LangError::ParseError {
                                message: format!("Unknown NeuralNetwork method: {}", method),
                                line: *line,
                            });
                        };
                        
                        // Определяем фактическое количество аргументов для Call инструкции
                        let actual_arg_count = if method == "get_device" {
                            if !args.is_empty() {
                                return Err(LangError::ParseError {
                                    message: "get_device() takes no arguments".to_string(),
                                    line: *line,
                                });
                            }
                            0
                        } else if method == "train" {
                            // Разрешаем именованные аргументы для train метода
                            // Используем resolve_function_args для правильного маппинга именованных аргументов
                            let resolved_args = match self.resolve_function_args("nn_train", args, None, *line) {
                                Ok(resolved) => resolved,
                                Err(e) => return Err(e),
                            };
                            
                            // Компилируем разрешенные аргументы в правильном порядке
                            for arg in &resolved_args {
                                match arg {
                                    Arg::Positional(expr) => self.compile_expr(expr)?,
                                    Arg::Named { value, .. } => self.compile_expr(value)?,
                                }
                            }
                            
                            // Используем количество разрешенных аргументов
                            resolved_args.len()
                        } else if method == "train_sh" {
                            // Разрешаем именованные аргументы для train_sh метода
                            let resolved_args = match self.resolve_function_args("nn_train_sh", args, None, *line) {
                                Ok(resolved) => resolved,
                                Err(e) => return Err(e),
                            };
                            
                            // Компилируем разрешенные аргументы в правильном порядке
                            for arg in &resolved_args {
                                match arg {
                                    Arg::Positional(expr) => self.compile_expr(expr)?,
                                    Arg::Named { value, .. } => self.compile_expr(value)?,
                                }
                            }
                            
                            // Используем количество разрешенных аргументов
                            resolved_args.len()
                        } else {
                            // Для device и save компилируем аргументы
                            // device(device_string) или save(path_string)
                            if args.len() != 1 {
                                return Err(LangError::ParseError {
                                    message: format!("{}() takes exactly 1 argument", method),
                                    line: *line,
                                });
                            }
                            for arg in args {
                                match arg {
                                    Arg::Positional(expr) => self.compile_expr(expr)?,
                                    Arg::Named { value, .. } => self.compile_expr(value)?,
                                }
                            }
                            args.len()
                        };
                        
                        // Теперь на стеке: [object, arg_1] (для device/save), [object, arg_1, arg_2, ...] (для train) или [object] (для get_device)
                        
                        // Загружаем функцию из ml модуля
                        if let Some(&ml_index) = self.globals.get("ml") {
                            self.chunk.write_with_line(OpCode::LoadGlobal(ml_index), *line);
                            let method_name_index = self.chunk.add_constant(Value::String(function_name.to_string()));
                            self.chunk.write_with_line(OpCode::Constant(method_name_index), *line);
                            self.chunk.write_with_line(OpCode::GetArrayElement, *line);
                            // Теперь на стеке: [object, arg_1, ..., NativeFunction]
                            // При вызове Call: pop N раз, reverse -> функция получает [object, arg_1, ...] в правильном порядке
                            self.chunk.write_with_line(OpCode::Call(actual_arg_count + 1), *line);
                        } else {
                            return Err(LangError::ParseError {
                                message: "ml module not found".to_string(),
                                line: *line,
                            });
                        }
                    } else if is_axis_method {
                        // Для методов Axis компилируем аргументы сразу (они не поддерживают именованные аргументы через разрешение)
                        for arg in args {
                            match arg {
                                Arg::Positional(expr) => self.compile_expr(expr)?,
                                Arg::Named { value, .. } => self.compile_expr(value)?,
                            }
                        }
                        // Теперь на стеке: [arg_n, ..., arg_1]
                        // Для методов Axis: нужно изменить порядок аргументов так, чтобы объект был первым
                        // Удаляем текущие аргументы со стека
                        for _ in 0..args.len() {
                             self.chunk.write_with_line(OpCode::Pop, *line);
                        }
                        
                        // Загружаем объект первым
                        self.chunk.write_with_line(OpCode::LoadLocal(temp_object_slot), *line);
                        // Теперь на стеке: [object]
                        
                        // Компилируем аргументы в нормальном порядке (они будут после объекта)
                        for arg in args {
                            match arg {
                                Arg::Positional(expr) => self.compile_expr(expr)?,
                                Arg::Named { value, .. } => self.compile_expr(value)?,
                            }
                        }
                        // Теперь на стеке: [object, arg_1, ..., arg_n]
                        
                        // Получаем свойство объекта по имени метода
                        // Загружаем объект еще раз для получения метода
                        self.chunk.write_with_line(OpCode::LoadLocal(temp_object_slot), *line);
                        // Теперь на стеке: [object, arg_1, ..., arg_n, object]
                        
                        let method_name_index = self.chunk.add_constant(Value::String(method.clone()));
                        self.chunk.write_with_line(OpCode::Constant(method_name_index), *line);
                        self.chunk.write_with_line(OpCode::GetArrayElement, *line);
                        // Теперь на стеке: [object, arg_1, ..., arg_n, NativeFunction]
                        
                        // Вызываем метод
                        self.chunk.write_with_line(OpCode::Call(args.len() + 1), *line);
                    } else if is_layer_method {
                        // Для методов Layer (freeze, unfreeze) компилируем аргументы сразу
                        // Эти методы не принимают аргументов, кроме самого layer
                        if !args.is_empty() {
                            return Err(LangError::ParseError {
                                message: format!("layer.{}() takes no arguments", method),
                                line: *line,
                            });
                        }
                        
                        // Загружаем объект первым
                        self.chunk.write_with_line(OpCode::LoadLocal(temp_object_slot), *line);
                        // Теперь на стеке: [object]
                        
                        // Получаем свойство объекта по имени метода
                        self.chunk.write_with_line(OpCode::LoadLocal(temp_object_slot), *line);
                        // Теперь на стеке: [object, object]
                        
                        let method_name_index = self.chunk.add_constant(Value::String(method.clone()));
                        self.chunk.write_with_line(OpCode::Constant(method_name_index), *line);
                        self.chunk.write_with_line(OpCode::GetArrayElement, *line);
                        // Теперь на стеке: [object, NativeFunction]
                        
                        // При вызове Call(1): pop 1 раз, reverse -> функция получает [object] в правильном порядке
                        self.chunk.write_with_line(OpCode::Call(1), *line);
                    } else {
                            // Для функций модулей: получаем метод, не добавляем объект
                            // Пытаемся разрешить именованные аргументы
                            let resolved_args = match self.resolve_function_args(method, args, None, *line) {
                                Ok(resolved) => {
                                    resolved
                                }
                                Err(e) => {
                                    // Проверяем, является ли это ошибкой "not supported"
                                    let error_msg = match &e {
                                        LangError::ParseError { message, .. } => message,
                                        _ => "",
                                    };
                                    
                                    if error_msg.contains("not supported") || error_msg.contains("Named arguments are not supported") {
                                        // Fallback: компилируем аргументы как есть (именованные аргументы будут преобразованы в объекты)
                                        args.iter().map(|a| match a {
                                            Arg::Positional(e) => Arg::Positional(e.clone()),
                                            Arg::Named { value, .. } => Arg::Positional(value.clone()),
                                        }).collect()
                                    } else {
                                        return Err(e);
                                    }
                                }
                            };
                            
                            // Компилируем разрешенные аргументы
                            for arg in &resolved_args {
                                match arg {
                                    Arg::Positional(expr) => self.compile_expr(expr)?,
                                    Arg::Named { .. } => {
                                        return Err(LangError::ParseError {
                                            message: format!("Unexpected named argument in resolved args for method '{}'", method),
                                            line: *line,
                                        });
                                    }
                                }
                            }
                            // Теперь на стеке: [arg_n, ..., arg_1]
                            
                            // Загружаем объект обратно для получения метода
                            self.chunk.write_with_line(OpCode::LoadLocal(temp_object_slot), *line);
                            // Теперь на стеке: [arg_n, ..., arg_1, object]
                            
                            // Получаем свойство объекта по имени метода
                            let method_name_index = self.chunk.add_constant(Value::String(method.clone()));
                            self.chunk.write_with_line(OpCode::Constant(method_name_index), *line);
                            self.chunk.write_with_line(OpCode::GetArrayElement, *line);
                            // Теперь на стеке: [arg_n, ..., arg_1, NativeFunction]
                            
                            // Вызываем функцию без объекта как первого аргумента
                            self.chunk.write_with_line(OpCode::Call(resolved_args.len()), *line);
                        }
                    }
                }
            }
        Ok(())
    }

    /// Вычисляет размер инструкции в байтах для эталонного алгоритма апгрейда jump-инструкций
    fn instruction_size(opcode: &OpCode) -> usize {
        match opcode {
            // Jump инструкции с относительными смещениями
            OpCode::Jump8(_) | OpCode::JumpIfFalse8(_) => 2,  // 1 байт opcode + 1 байт смещение
            OpCode::Jump16(_) | OpCode::JumpIfFalse16(_) => 3, // 1 байт opcode + 2 байта смещение
            OpCode::Jump32(_) | OpCode::JumpIfFalse32(_) => 5, // 1 байт opcode + 4 байта смещение
            OpCode::JumpLabel(_) | OpCode::JumpIfFalseLabel(_) => 2, // Временно считаем как Jump8 до финализации
            
            // Инструкции с параметрами
            OpCode::Constant(_) => 2,  // 1 байт opcode + 1 байт индекс константы (usize может быть больше, но упрощаем)
            OpCode::LoadLocal(_) | OpCode::StoreLocal(_) => 2, // 1 байт opcode + 1 байт индекс
            OpCode::LoadGlobal(_) | OpCode::StoreGlobal(_) => 2, // 1 байт opcode + 1 байт индекс
            OpCode::Call(_) => 2, // 1 байт opcode + 1 байт количество аргументов
            OpCode::MakeArray(_) => 2, // 1 байт opcode + 1 байт размер
            OpCode::MakeTuple(_) => 2, // 1 байт opcode + 1 байт размер
            OpCode::MakeArrayDynamic => 1, // 1 байт opcode (размер на стеке) 1 байт opcode + 1 байт количество элементов
            OpCode::BeginTry(_) => 2, // 1 байт opcode + 1 байт индекс обработчика
            OpCode::Catch(Some(_)) => 2, // 1 байт opcode + 1 байт тип ошибки
            OpCode::Catch(None) => 1, // 1 байт opcode
            OpCode::Throw(Some(_)) => 2, // 1 байт opcode + 1 байт тип ошибки
            OpCode::Throw(None) => 1, // 1 байт opcode
            
            // Все остальные инструкции занимают 1 байт
            _ => 1,
        }
    }

    /// Вычисляет абсолютные адреса всех инструкций с учетом их размеров
    fn compute_instruction_addresses(&self) -> Vec<usize> {
        let mut addresses = Vec::with_capacity(self.chunk.code.len());
        let mut current_addr = 0;
        
        for opcode in &self.chunk.code {
            addresses.push(current_addr);
            current_addr += Self::instruction_size(opcode);
        }
        
        addresses
    }

    /// Апгрейдит jump-инструкции до минимально достаточного формата
    /// Возвращает true если были изменения
    fn upgrade_jump_instructions(&mut self) -> bool {
        let addresses = self.compute_instruction_addresses();
        let mut changed = false;
        
        // Адреса меток будут использованы при вычислении смещений
        
        // Обрабатываем все pending jumps (JumpLabel и JumpIfFalseLabel)
        for (jump_index, label_id, is_conditional) in self.pending_jumps.iter() {
            if *jump_index >= self.chunk.code.len() {
                continue;
            }
            
            // Получаем адрес целевой метки
            let dst_instruction_index = *self.labels.get(label_id).unwrap_or(jump_index);
            if dst_instruction_index >= addresses.len() {
                continue;
            }
            
            // VM использует индексы инструкций, а не байтовые адреса
            // IP инкрементируется на 1 после каждой инструкции
            // Поэтому смещение вычисляется как: offset = dst_index - (src_index + 1)
            // где +1 - это автоматический инкремент IP после выполнения jump-инструкции
            let src_index = *jump_index;
            let dst_index = dst_instruction_index;
            
            // Вычисляем относительное смещение в индексах инструкций
            // offset = dst_index - (src_index + 1), так как IP инкрементируется на 1 после jump
            let offset = (dst_index as i64 - (src_index as i64 + 1)) as i32;
            
            // Определяем текущий размер jump-инструкции (для апгрейда)
            let current_opcode = &self.chunk.code[*jump_index];
            
            // Определяем минимально достаточный формат
            let new_opcode = if offset >= -128 && offset <= 127 {
                // Jump8 достаточно
                if *is_conditional {
                    OpCode::JumpIfFalse8(offset as i8)
                } else {
                    OpCode::Jump8(offset as i8)
                }
            } else if offset >= -32768 && offset <= 32767 {
                // Нужен Jump16
                if *is_conditional {
                    OpCode::JumpIfFalse16(offset as i16)
                } else {
                    OpCode::Jump16(offset as i16)
                }
            } else {
                // Нужен Jump32
                if *is_conditional {
                    OpCode::JumpIfFalse32(offset)
                } else {
                    OpCode::Jump32(offset)
                }
            };
            
            // Проверяем, нужно ли апгрейдить
            // НЕ заменяем JumpLabel здесь - это делает finalize_jumps
            if matches!(current_opcode, OpCode::JumpLabel(_) | OpCode::JumpIfFalseLabel(_)) {
                // Пропускаем JumpLabel - они будут обработаны в finalize_jumps
                continue;
            }
            
            let new_size = Self::instruction_size(&new_opcode);
            let current_size = Self::instruction_size(current_opcode);
            if new_size != current_size {
                self.chunk.code[*jump_index] = new_opcode;
                changed = true;
            } else {
                // Если размер не изменился, но смещение могло измениться, обновляем
                match current_opcode {
                    OpCode::Jump8(_) | OpCode::Jump16(_) | OpCode::Jump32(_) |
                    OpCode::JumpIfFalse8(_) | OpCode::JumpIfFalse16(_) | OpCode::JumpIfFalse32(_) => {
                        self.chunk.code[*jump_index] = new_opcode;
                        // Не помечаем как changed, если размер не изменился
                    }
                    _ => {}
                }
            }
        }
        
        changed
    }

    /// Итеративно стабилизирует layout до полной фиксации размеров
    fn stabilize_layout(&mut self) -> Result<(), LangError> {
        let mut iterations = 0;
        const MAX_ITERATIONS: usize = 100; // Защита от бесконечного цикла
        
        loop {
            let changed = self.upgrade_jump_instructions();
            iterations += 1;
            
            if !changed {
                break; // Layout стабилизирован
            }
            
            if iterations >= MAX_ITERATIONS {
                return Err(LangError::ParseError {
                    message: "Layout stabilization failed: too many iterations".to_string(),
                    line: self.current_line,
                });
            }
        }
        
        Ok(())
    }

    /// Финализирует jump-инструкции: заменяет все JumpLabel на финальные инструкции
    /// Вызывается после стабилизации layout
    fn finalize_jumps(&mut self) -> Result<(), LangError> {
        let addresses = self.compute_instruction_addresses();
        
        // Заменяем все оставшиеся JumpLabel на финальные инструкции
        // Обрабатываем все pending_jumps, а также ищем все JumpLabel в коде
        let mut jumps_to_finalize = self.pending_jumps.clone();
        
        // Также ищем все JumpLabel в коде, которые могут не быть в pending_jumps
        for (jump_index, opcode) in self.chunk.code.iter().enumerate() {
            match opcode {
                OpCode::JumpLabel(label_id) => {
                    if !jumps_to_finalize.iter().any(|(idx, _, _)| *idx == jump_index) {
                        jumps_to_finalize.push((jump_index, *label_id, false));
                    }
                }
                OpCode::JumpIfFalseLabel(label_id) => {
                    if !jumps_to_finalize.iter().any(|(idx, _, _)| *idx == jump_index) {
                        jumps_to_finalize.push((jump_index, *label_id, true));
                    }
                }
                _ => {}
            }
        }
        
        for (jump_index, label_id, is_conditional) in jumps_to_finalize.iter() {
            if *jump_index >= self.chunk.code.len() {
                continue;
            }
            
            let current_opcode = &self.chunk.code[*jump_index];
            
            // Пропускаем, если уже финализировано
            if !matches!(current_opcode, OpCode::JumpLabel(_) | OpCode::JumpIfFalseLabel(_)) {
                continue;
            }
            
            // Получаем адрес целевой метки
            let dst_instruction_index = *self.labels.get(label_id)
                .ok_or_else(|| LangError::ParseError {
                    message: format!("Label {} not found", label_id),
                    line: self.current_line,
                })?;
            
            // Если метка указывает за пределы массива, это означает, что метка помечена после последней инструкции
            // В этом случае используем индекс последней инструкции
            let dst_instruction_index = if dst_instruction_index >= self.chunk.code.len() {
                if self.chunk.code.is_empty() {
            return Err(LangError::ParseError {
                        message: format!("Label {} points to empty code", label_id),
                line: self.current_line,
            });
        }
                self.chunk.code.len() - 1
            } else {
                dst_instruction_index
            };
            
            if dst_instruction_index >= addresses.len() {
                return Err(LangError::ParseError {
                    message: format!("Label {} instruction index {} >= addresses len {} (code len: {})", 
                        label_id, dst_instruction_index, addresses.len(), self.chunk.code.len()),
                    line: self.current_line,
                });
            }
            
            // VM использует индексы инструкций, а не байтовые адреса
            // IP инкрементируется на 1 после каждой инструкции
            // Поэтому смещение вычисляется как: offset = dst_index - (src_index + 1)
            let src_index = *jump_index;
            let dst_index = dst_instruction_index;
            
            // Вычисляем относительное смещение в индексах инструкций
            // offset = dst_index - (src_index + 1), так как IP инкрементируется на 1 после jump
            let offset = (dst_index as i64 - (src_index as i64 + 1)) as i32;
            
            // Определяем финальную инструкцию
            let final_opcode = if offset >= -128 && offset <= 127 {
                if *is_conditional {
                    OpCode::JumpIfFalse8(offset as i8)
                } else {
                    OpCode::Jump8(offset as i8)
                }
            } else if offset >= -32768 && offset <= 32767 {
                if *is_conditional {
                    OpCode::JumpIfFalse16(offset as i16)
                } else {
                    OpCode::Jump16(offset as i16)
                }
            } else {
                if *is_conditional {
                    OpCode::JumpIfFalse32(offset)
                } else {
                    OpCode::Jump32(offset)
                }
            };
            
            self.chunk.code[*jump_index] = final_opcode;
        }
        
        // Очищаем только pending_jumps, но НЕ очищаем labels и label_counter,
        // так как они могут использоваться в других функциях или в главном скрипте
        // Метки будут очищены при компиляции следующей функции или в конце компиляции главного скрипта
        self.pending_jumps.clear();
        
        Ok(())
    }
    
    /// Очищает все временные структуры меток (вызывается в конце компиляции)
    fn clear_labels(&mut self) {
        self.labels.clear();
        self.label_counter = 0;
        self.pending_jumps.clear();
    }

    /// Создает новую метку и возвращает её ID
    fn create_label(&mut self) -> usize {
        let label_id = self.label_counter;
        self.label_counter += 1;
        label_id
    }

    /// Помечает текущую позицию инструкции меткой
    /// Метка указывает на следующую инструкцию, которая будет добавлена
    fn mark_label(&mut self, label_id: usize) {
        let instruction_index = self.chunk.code.len();
        // Проверяем, что метка не указывает за пределы массива
        if instruction_index > self.chunk.code.len() {
            // Это не должно происходить, но на всякий случай
            return;
        }
        self.labels.insert(label_id, instruction_index);
    }

    /// Создает jump-инструкцию с меткой (для эталонного алгоритма)
    fn emit_jump(&mut self, is_conditional: bool, label_id: usize) -> Result<usize, LangError> {
        let jump_index = self.chunk.code.len();
        let opcode = if is_conditional {
            OpCode::JumpIfFalseLabel(label_id)
        } else {
            OpCode::JumpLabel(label_id)
        };
        self.chunk.write_with_line(opcode, self.current_line);
        self.pending_jumps.push((jump_index, label_id, is_conditional));
        Ok(jump_index)
    }


    /// Создает jump-инструкцию для цикла (переход к началу цикла)
    /// Использует метку для эталонного алгоритма
    fn emit_loop(&mut self, loop_label_id: usize) -> Result<(), LangError> {
        self.emit_jump(false, loop_label_id)?;
        Ok(())
    }

    fn begin_scope(&mut self) {
        self.locals.push(std::collections::HashMap::new());
    }

    fn end_scope(&mut self) {
        if let Some(scope) = self.locals.pop() {
            // Уменьшаем счетчик локальных переменных на количество переменных в этой области
            self.local_count -= scope.len();
        }
    }

    fn declare_local(&mut self, name: &str) -> usize {
        let index = self.local_count;
        if let Some(scope) = self.locals.last_mut() {
            scope.insert(name.to_string(), index);
        }
        self.local_count += 1;
        index
    }

    fn resolve_local(&mut self, name: &str) -> Option<usize> {
        // Ищем переменную в текущих областях видимости (от последней к первой)
        for scope in self.locals.iter().rev() {
            if let Some(&index) = scope.get(name) {
                return Some(index);
            }
        }
        None
    }

    /// Подсчитывает количество фиксированных переменных (не wildcard, не variadic) в паттерне распаковки
    /// Для вложенных паттернов считает только переменные текущего уровня (вложенный паттерн = 1 элемент)
    fn count_unpack_variables(&self, pattern: &[UnpackPattern]) -> usize {
        let mut count = 0;
        for pat in pattern {
            match pat {
                UnpackPattern::Variable(_) => count += 1,
                UnpackPattern::Wildcard => {}, // Wildcard не считается
                UnpackPattern::Variadic(_) | UnpackPattern::VariadicWildcard => {
                    // Variadic не считается в фиксированных переменных
                }
                UnpackPattern::Nested(_) => count += 1, // Вложенный паттерн считается как один элемент
            }
        }
        count
    }
    

    /// Объявляет переменные из паттерна распаковки и возвращает их локальные индексы
    fn declare_unpack_pattern_variables(&mut self, pattern: &[UnpackPattern], line: usize) -> Result<Vec<Option<usize>>, LangError> {
        let mut var_locals = Vec::new();
        for pat in pattern {
            match pat {
                UnpackPattern::Variable(name) => {
                    let index = self.declare_local(name);
                    var_locals.push(Some(index));
                }
                UnpackPattern::Wildcard => {
                    // Wildcard не создает переменную
                    var_locals.push(None);
                }
                UnpackPattern::Variadic(name) => {
                    // Variadic переменная создает переменную
                    let index = self.declare_local(name);
                    var_locals.push(Some(index));
                }
                UnpackPattern::VariadicWildcard => {
                    // Variadic wildcard не создает переменную
                    var_locals.push(None);
                }
                UnpackPattern::Nested(nested) => {
                    // Рекурсивно обрабатываем вложенные паттерны
                    let nested_locals = self.declare_unpack_pattern_variables(nested, line)?;
                    var_locals.extend(nested_locals);
                }
            }
        }
        Ok(var_locals)
    }

    /// Компилирует код для распаковки значения в переменные
    /// Предполагается, что значение находится на вершине стека
    fn compile_unpack_pattern(&mut self, pattern: &[UnpackPattern], var_locals: &[Option<usize>], expected_count: usize, line: usize) -> Result<(), LangError> {
        // Сохраняем элемент во временную переменную
        let temp_local = self.declare_local("__unpack_temp");
        self.chunk.write_with_line(OpCode::StoreLocal(temp_local), line);
        
        // Проверяем минимальную длину элемента (M >= N_fixed)
        // Загружаем элемент
        self.chunk.write_with_line(OpCode::LoadLocal(temp_local), line);
        // Получаем длину
        self.chunk.write_with_line(OpCode::GetArrayLength, line);
        // Загружаем минимальную требуемую длину (N_fixed)
        let n_fixed_const = self.chunk.add_constant(Value::Number(expected_count as f64));
        self.chunk.write_with_line(OpCode::Constant(n_fixed_const), line);
        // Сравниваем: length < N_fixed?
        self.chunk.write_with_line(OpCode::Less, line);
        
        // Если length >= N_fixed, пропускаем ошибку
        let skip_error_label = self.create_label();
        self.emit_jump(true, skip_error_label)?;
        
        // Выбрасываем ошибку: длина меньше минимально требуемой
        let error_msg = format!("Unpack pattern requires at least {} elements, but got array with length less than {}", expected_count, expected_count);
        let error_msg_index = self.chunk.add_constant(Value::String(error_msg));
        self.chunk.write_with_line(OpCode::Constant(error_msg_index), line);
        self.chunk.write_with_line(OpCode::Throw(None), line);
        
        // Метка для пропуска ошибки
        self.mark_label(skip_error_label);
        
        // Распаковываем значения
        // var_locals соответствует структуре pattern, итерируем их вместе
        let mut var_index = 0;
        self.compile_unpack_pattern_recursive(pattern, var_locals, &mut var_index, temp_local, line)?;
        
        Ok(())
    }

    /// Собирает имена переменных из паттерна распаковки
    fn collect_unpack_pattern_variables(&self, pattern: &[UnpackPattern], vars: &mut std::collections::HashSet<String>) {
        for pat in pattern {
            match pat {
                UnpackPattern::Variable(name) => {
                    vars.insert(name.clone());
                }
                UnpackPattern::Wildcard => {}
                UnpackPattern::Variadic(name) => {
                    vars.insert(name.clone());
                }
                UnpackPattern::VariadicWildcard => {}
                UnpackPattern::Nested(nested) => {
                    self.collect_unpack_pattern_variables(nested, vars);
                }
            }
        }
    }

    /// Рекурсивно компилирует распаковку вложенных паттернов
    fn compile_unpack_pattern_recursive(&mut self, pattern: &[UnpackPattern], var_locals: &[Option<usize>], var_index: &mut usize, source_local: usize, line: usize) -> Result<(), LangError> {
        // Находим позицию variadic в паттерне (если есть)
        let variadic_pos = pattern.iter().position(|p| matches!(p, UnpackPattern::Variadic(_) | UnpackPattern::VariadicWildcard));
        
        // Обрабатываем фиксированные переменные до variadic
        let end_pos = variadic_pos.unwrap_or(pattern.len());
        for (i, pat) in pattern[..end_pos].iter().enumerate() {
            match pat {
                UnpackPattern::Variable(_) => {
                    // Получаем значение по индексу
                    self.chunk.write_with_line(OpCode::LoadLocal(source_local), line);
                    let index_const = self.chunk.add_constant(Value::Number(*var_index as f64));
                    self.chunk.write_with_line(OpCode::Constant(index_const), line);
                    self.chunk.write_with_line(OpCode::GetArrayElement, line);
                    
                    // Сохраняем в переменную
                    if let Some(local_index) = var_locals.get(i).and_then(|&x| x) {
                        self.chunk.write_with_line(OpCode::StoreLocal(local_index), line);
                    }
                    *var_index += 1;
                }
                UnpackPattern::Wildcard => {
                    // Получаем значение, но не сохраняем (просто удаляем со стека)
                    self.chunk.write_with_line(OpCode::LoadLocal(source_local), line);
                    let index_const = self.chunk.add_constant(Value::Number(*var_index as f64));
                    self.chunk.write_with_line(OpCode::Constant(index_const), line);
                    self.chunk.write_with_line(OpCode::GetArrayElement, line);
                    self.chunk.write_with_line(OpCode::Pop, line); // Удаляем значение
                    *var_index += 1;
                }
                UnpackPattern::Variadic(_) | UnpackPattern::VariadicWildcard => {
                    // Не должно быть здесь, так как мы обрабатываем только до variadic
                    unreachable!("Variadic should be handled separately");
                }
                UnpackPattern::Nested(nested) => {
                    // Для вложенной распаковки нужно получить элемент и рекурсивно распаковать
                    self.chunk.write_with_line(OpCode::LoadLocal(source_local), line);
                    let index_const = self.chunk.add_constant(Value::Number(*var_index as f64));
                    self.chunk.write_with_line(OpCode::Constant(index_const), line);
                    self.chunk.write_with_line(OpCode::GetArrayElement, line);
                    
                    // Сохраняем вложенный элемент во временную переменную
                    let nested_temp = self.declare_local("__unpack_nested_temp");
                    self.chunk.write_with_line(OpCode::StoreLocal(nested_temp), line);
                    
                    // Упрощенный подход: для вложенных паттернов создаем новые локальные переменные
                    let nested_var_locals = self.declare_unpack_pattern_variables(nested, line)?;
                    let mut nested_var_index = 0;
                    self.compile_unpack_pattern_recursive(nested, &nested_var_locals, &mut nested_var_index, nested_temp, line)?;
                    
                    *var_index += 1;
                }
            }
        }
        
        // Обрабатываем variadic (если есть)
        if let Some(pos) = variadic_pos {
            let variadic_pattern = &pattern[pos];
            match variadic_pattern {
                UnpackPattern::Variadic(_) => {
                    // Создаем массив из оставшихся элементов
                    // 1. Вычисляем count = length - var_index
                    self.chunk.write_with_line(OpCode::LoadLocal(source_local), line);
                    self.chunk.write_with_line(OpCode::GetArrayLength, line);
                    let var_index_const = self.chunk.add_constant(Value::Number(*var_index as f64));
                    self.chunk.write_with_line(OpCode::Constant(var_index_const), line);
                    self.chunk.write_with_line(OpCode::Sub, line);
                    // Теперь на стеке: count
                    
                    // Сохраняем count во временную переменную
                    let count_temp = self.declare_local("__variadic_count");
                    self.chunk.write_with_line(OpCode::StoreLocal(count_temp), line);
                    
                    // 2. Загружаем элементы от length-1 до var_index (в обратном порядке индексов)
                    // чтобы они были на стеке в правильном порядке для MakeArrayDynamic
                    let loop_start_label = self.create_label();
                    let loop_end_label = self.create_label();
                    let loop_index_temp = self.declare_local("__variadic_loop_idx");
                    
                    // Начинаем с length-1
                    self.chunk.write_with_line(OpCode::LoadLocal(source_local), line);
                    self.chunk.write_with_line(OpCode::GetArrayLength, line);
                    let one_const = self.chunk.add_constant(Value::Number(1.0));
                    self.chunk.write_with_line(OpCode::Constant(one_const), line);
                    self.chunk.write_with_line(OpCode::Sub, line);
                    self.chunk.write_with_line(OpCode::StoreLocal(loop_index_temp), line);
                    
                    // Начало цикла
                    self.mark_label(loop_start_label);
                    
                    // Проверяем условие: loop_index >= var_index
                    self.chunk.write_with_line(OpCode::LoadLocal(loop_index_temp), line);
                    self.chunk.write_with_line(OpCode::Constant(var_index_const), line);
                    self.chunk.write_with_line(OpCode::GreaterEqual, line);
                    
                    // Если условие false, выходим из цикла
                    self.emit_jump(true, loop_end_label)?;
                    
                    // Загружаем элемент по индексу loop_index
                    self.chunk.write_with_line(OpCode::LoadLocal(source_local), line);
                    self.chunk.write_with_line(OpCode::LoadLocal(loop_index_temp), line);
                    self.chunk.write_with_line(OpCode::GetArrayElement, line);
                    
                    // Декрементируем loop_index
                    self.chunk.write_with_line(OpCode::LoadLocal(loop_index_temp), line);
                    self.chunk.write_with_line(OpCode::Constant(one_const), line);
                    self.chunk.write_with_line(OpCode::Sub, line);
                    self.chunk.write_with_line(OpCode::StoreLocal(loop_index_temp), line);
                    
                    // Переход к началу цикла
                    self.emit_loop(loop_start_label)?;
                    
                    // Конец цикла
                    self.mark_label(loop_end_label);
                    
                    // 3. Создаем массив динамически: загружаем count и используем MakeArrayDynamic
                    self.chunk.write_with_line(OpCode::LoadLocal(count_temp), line);
                    self.chunk.write_with_line(OpCode::MakeArrayDynamic, line);
                    
                    // 4. Сохраняем в variadic переменную
                    if let Some(local_index) = var_locals.get(pos).and_then(|&x| x) {
                        self.chunk.write_with_line(OpCode::StoreLocal(local_index), line);
                    }
                }
                UnpackPattern::VariadicWildcard => {
                    // Variadic wildcard: пропускаем оставшиеся элементы
                    // Ничего не делаем, элементы уже пропущены
                }
                _ => {}
            }
        }
        
        Ok(())
    }
    
    /// Находит все переменные, используемые в выражениях и statements
    fn find_used_variables_in_expr(&self, expr: &Expr) -> std::collections::HashSet<String> {
        let mut vars = std::collections::HashSet::new();
        match expr {
            Expr::Variable { name, .. } => {
                vars.insert(name.clone());
            }
            Expr::Assign { name, value, .. } => {
                vars.insert(name.clone());
                vars.extend(self.find_used_variables_in_expr(value));
            }
            Expr::UnpackAssign { names, value, .. } => {
                for name in names {
                    vars.insert(name.clone());
                }
                vars.extend(self.find_used_variables_in_expr(value));
            }
            Expr::Binary { left, right, .. } => {
                vars.extend(self.find_used_variables_in_expr(left));
                vars.extend(self.find_used_variables_in_expr(right));
            }
            Expr::Unary { right, .. } => {
                vars.extend(self.find_used_variables_in_expr(right));
            }
            Expr::Call { args, .. } => {
                for arg in args {
                    match arg {
                        Arg::Positional(expr) => {
                            vars.extend(self.find_used_variables_in_expr(expr));
                        }
                        Arg::Named { value, .. } => {
                            vars.extend(self.find_used_variables_in_expr(value));
                        }
                    }
                }
            }
            Expr::ArrayLiteral { elements, .. } => {
                for elem in elements {
                    vars.extend(self.find_used_variables_in_expr(elem));
                }
            }
            Expr::TupleLiteral { elements, .. } => {
                for elem in elements {
                    vars.extend(self.find_used_variables_in_expr(elem));
                }
            }
            Expr::ArrayIndex { array, index, .. } => {
                vars.extend(self.find_used_variables_in_expr(array));
                vars.extend(self.find_used_variables_in_expr(index));
            }
            Expr::Property { object, .. } => {
                vars.extend(self.find_used_variables_in_expr(object));
            }
            Expr::MethodCall { object, args, .. } => {
                vars.extend(self.find_used_variables_in_expr(object));
                for arg in args {
                    match arg {
                        Arg::Positional(expr) => {
                            vars.extend(self.find_used_variables_in_expr(expr));
                        }
                        Arg::Named { value, .. } => {
                            vars.extend(self.find_used_variables_in_expr(value));
                        }
                    }
                }
            }
            _ => {}
        }
        vars
    }
    
    /// Находит все переменные, используемые в statements
    fn find_used_variables_in_stmt(&self, stmt: &Stmt) -> std::collections::HashSet<String> {
        let mut vars = std::collections::HashSet::new();
        match stmt {
            Stmt::Import { .. } => {
                // Import statements don't use variables
            }
            Stmt::Let { value, .. } => {
                vars.extend(self.find_used_variables_in_expr(value));
            }
            Stmt::Expr { expr, .. } => {
                vars.extend(self.find_used_variables_in_expr(expr));
            }
            Stmt::If { condition, then_branch, else_branch, .. } => {
                vars.extend(self.find_used_variables_in_expr(condition));
                for stmt in then_branch {
                    vars.extend(self.find_used_variables_in_stmt(stmt));
                }
                if let Some(else_branch) = else_branch {
                    for stmt in else_branch {
                        vars.extend(self.find_used_variables_in_stmt(stmt));
                    }
                }
            }
            Stmt::While { condition, body, .. } => {
                vars.extend(self.find_used_variables_in_expr(condition));
                for stmt in body {
                    vars.extend(self.find_used_variables_in_stmt(stmt));
                }
            }
            Stmt::For { iterable, body, .. } => {
                vars.extend(self.find_used_variables_in_expr(iterable));
                for stmt in body {
                    vars.extend(self.find_used_variables_in_stmt(stmt));
                }
            }
            Stmt::Function { body, .. } => {
                for stmt in body {
                    vars.extend(self.find_used_variables_in_stmt(stmt));
                }
            }
            Stmt::Return { value, .. } => {
                if let Some(expr) = value {
                    vars.extend(self.find_used_variables_in_expr(expr));
                }
            }
            Stmt::Break { .. } => {
                // break не использует переменные
            }
            Stmt::Continue { .. } => {
                // continue не использует переменные
            }
            Stmt::Try { try_block, catch_blocks, else_block, .. } => {
                // Находим переменные в try блоке
                for stmt in try_block {
                    vars.extend(self.find_used_variables_in_stmt(stmt));
                }
                // Находим переменные в catch блоках
                for catch_block in catch_blocks {
                    for stmt in &catch_block.body {
                        vars.extend(self.find_used_variables_in_stmt(stmt));
                    }
                }
                // Находим переменные в else блоке (если есть)
                if let Some(else_block) = else_block {
                    for stmt in else_block {
                        vars.extend(self.find_used_variables_in_stmt(stmt));
                    }
                }
            }
            Stmt::Throw { value, .. } => {
                // Находим переменные в выражении throw
                vars.extend(self.find_used_variables_in_expr(value));
            }
        }
        vars
    }
    
    /// Находит все переменные, объявленные локально в теле функции
    /// (через let и for, рекурсивно проверяя вложенные блоки)
    fn find_locally_declared_variables(&self, body: &[Stmt]) -> std::collections::HashSet<String> {
        let mut declared_vars = std::collections::HashSet::new();
        
        for stmt in body {
            match stmt {
                Stmt::Let { name, is_global, .. } => {
                    // Добавляем только локальные переменные (не глобальные)
                    if !is_global {
                        declared_vars.insert(name.clone());
                    }
                }
                Stmt::For { pattern, body, .. } => {
                    // Переменные цикла for объявляются локально
                    self.collect_unpack_pattern_variables(pattern, &mut declared_vars);
                    // Рекурсивно проверяем тело цикла
                    declared_vars.extend(self.find_locally_declared_variables(body));
                }
                Stmt::If { then_branch, else_branch, .. } => {
                    // Рекурсивно проверяем ветки if
                    declared_vars.extend(self.find_locally_declared_variables(then_branch));
                    if let Some(else_branch) = else_branch {
                        declared_vars.extend(self.find_locally_declared_variables(else_branch));
                    }
                }
                Stmt::While { body, .. } => {
                    // Рекурсивно проверяем тело while
                    declared_vars.extend(self.find_locally_declared_variables(body));
                }
                Stmt::Function { body, .. } => {
                    // Рекурсивно проверяем тело вложенной функции
                    declared_vars.extend(self.find_locally_declared_variables(body));
                }
                Stmt::Try { try_block, catch_blocks, else_block, .. } => {
                    // Рекурсивно проверяем try блок
                    declared_vars.extend(self.find_locally_declared_variables(try_block));
                    // Рекурсивно проверяем catch блоки
                    for catch_block in catch_blocks {
                        declared_vars.extend(self.find_locally_declared_variables(&catch_block.body));
                    }
                    // Рекурсивно проверяем else блок (если есть)
                    if let Some(else_block) = else_block {
                        declared_vars.extend(self.find_locally_declared_variables(else_block));
                    }
                }
                _ => {
                    // Expr, Return, Break, Continue не объявляют переменные
                }
            }
        }
        
        declared_vars
    }
    
    /// Находит переменные, которые используются в теле функции, но не объявлены в ней
    /// (т.е. захваченные из родительских функций)
    fn find_captured_variables(
        &self,
        body: &[Stmt],
        parent_locals: &[std::collections::HashMap<String, usize>],
        params: &[String],
    ) -> Vec<String> {
        let mut used_vars = std::collections::HashSet::new();
        for stmt in body {
            used_vars.extend(self.find_used_variables_in_stmt(stmt));
        }
        
        // Исключаем параметры функции
        let param_set: std::collections::HashSet<String> = params.iter().cloned().collect();
        used_vars.retain(|v| !param_set.contains(v));
        
        // Находим все переменные, объявленные локально в теле функции
        let locally_declared = self.find_locally_declared_variables(body);
        
        // Исключаем локально объявленные переменные из проверки захвата
        // Они локальные, не требуют захвата из родительских областей
        used_vars.retain(|v| !locally_declared.contains(v));
        
        // Ищем переменные, которые используются, но не найдены в текущих областях видимости
        // но найдены в родительских областях видимости
        let mut captured = Vec::new();
        
        for var_name in &used_vars {
            // Проверяем, найдена ли переменная в текущей области видимости функции
            // (только в последней области, которая была создана для этой функции)
            // НЕ проверяем в родительских областях, которые все еще в self.locals
            let found_in_current_scope = self.locals.last()
                .map(|scope| scope.contains_key(var_name))
                .unwrap_or(false);
            
            if !found_in_current_scope {
                // Проверяем, найдена ли переменная в родительских областях видимости
                let found_in_parent = parent_locals.iter().any(|scope| scope.contains_key(var_name));
            
                if found_in_parent {
                    captured.push(var_name.clone());
                }
            }
        }
        
        captured
    }

    /// Оптимизация: вычисляет константные выражения во время компиляции
    fn evaluate_constant_expr(&self, expr: &Expr) -> Result<Option<Value>, LangError> {
        match expr {
            Expr::Literal { value, .. } => Ok(Some(value.clone())),
            Expr::ArrayLiteral { .. } => Ok(None), // Не можем вычислить во время компиляции
            Expr::TupleLiteral { .. } => Ok(None), // Не можем вычислить во время компиляции
            Expr::Property { .. } => Ok(None), // Не можем вычислить во время компиляции
            Expr::MethodCall { .. } => Ok(None), // Не можем вычислить во время компиляции
            Expr::Binary { left, op, right, .. } => {
                // Пытаемся вычислить бинарное выражение, если оба операнда константы
                let left_val = self.evaluate_constant_expr(left)?;
                let right_val = self.evaluate_constant_expr(right)?;
                
                if let (Some(l), Some(r)) = (left_val, right_val) {
                    match op {
                        TokenKind::Plus => {
                            match (l, r) {
                                (Value::Number(n1), Value::Number(n2)) => Ok(Some(Value::Number(n1 + n2))),
                                (Value::String(s1), Value::String(s2)) => Ok(Some(Value::String(format!("{}{}", s1, s2)))),
                                (Value::String(s), Value::Number(n)) => Ok(Some(Value::String(format!("{}{}", s, n)))),
                                (Value::Number(n), Value::String(s)) => Ok(Some(Value::String(format!("{}{}", n, s)))),
                                _ => Ok(None),
                            }
                        }
                        TokenKind::Minus => {
                            if let (Value::Number(n1), Value::Number(n2)) = (l, r) {
                                Ok(Some(Value::Number(n1 - n2)))
                            } else {
                                Ok(None)
                            }
                        }
                        TokenKind::Star => {
                            match (&l, &r) {
                                (Value::Number(n1), Value::Number(n2)) => Ok(Some(Value::Number(n1 * n2))),
                                (Value::String(s), Value::Number(n)) => {
                                    let count = *n as i64;
                                    if count <= 0 {
                                        Ok(Some(Value::String(String::new())))
                                    } else {
                                        Ok(Some(Value::String(s.repeat(count as usize))))
                                    }
                                }
                                (Value::Number(n), Value::String(s)) => {
                                    let count = *n as i64;
                                    if count <= 0 {
                                        Ok(Some(Value::String(String::new())))
                                    } else {
                                        Ok(Some(Value::String(s.repeat(count as usize))))
                                    }
                                }
                                _ => Ok(None),
                            }
                        }
                        TokenKind::Slash => {
                            if let (Value::Number(n1), Value::Number(n2)) = (l, r) {
                                if n2 == 0.0 {
                                    // Don't constant-fold division by zero - let it be a runtime error
                                    // so we can provide proper stack traces
                                    return Ok(None);
                                }
                                Ok(Some(Value::Number(n1 / n2)))
                            } else {
                                Ok(None)
                            }
                        }
                        TokenKind::SlashSlash => {
                            if let (Value::Number(n1), Value::Number(n2)) = (l, r) {
                                if n2 == 0.0 {
                                    return Ok(None);
                                }
                                Ok(Some(Value::Number((n1 / n2).floor())))
                            } else {
                                Ok(None)
                            }
                        }
                        TokenKind::EqualEqual => Ok(Some(Value::Bool(l == r))),
                        TokenKind::BangEqual => Ok(Some(Value::Bool(l != r))),
                        TokenKind::Greater => {
                            if let (Value::Number(n1), Value::Number(n2)) = (l, r) {
                                Ok(Some(Value::Bool(n1 > n2)))
                            } else {
                                Ok(None)
                            }
                        }
                        TokenKind::Less => {
                            if let (Value::Number(n1), Value::Number(n2)) = (l, r) {
                                Ok(Some(Value::Bool(n1 < n2)))
                            } else {
                                Ok(None)
                            }
                        }
                        TokenKind::GreaterEqual => {
                            if let (Value::Number(n1), Value::Number(n2)) = (l, r) {
                                Ok(Some(Value::Bool(n1 >= n2)))
                            } else {
                                Ok(None)
                            }
                        }
                        TokenKind::LessEqual => {
                            if let (Value::Number(n1), Value::Number(n2)) = (l, r) {
                                Ok(Some(Value::Bool(n1 <= n2)))
                            } else {
                                Ok(None)
                            }
                        }
                        _ => Ok(None),
                    }
                } else {
                    Ok(None)
                }
            }
            Expr::Unary { op, right, .. } => {
                let right_val = self.evaluate_constant_expr(right)?;
                if let Some(r) = right_val {
                    match op {
                        TokenKind::Minus => {
                            if let Value::Number(n) = r {
                                Ok(Some(Value::Number(-n)))
                            } else {
                                Ok(None)
                            }
                        }
                        TokenKind::Bang => {
                            Ok(Some(Value::Bool(!r.is_truthy())))
                        }
                        _ => Ok(None),
                    }
                } else {
                    Ok(None)
                }
            }
            _ => Ok(None), // Переменные, вызовы функций и присваивания не могут быть вычислены во время компиляции
        }
    }

    fn compile_try(
        &mut self,
        try_block: &[Stmt],
        catch_blocks: &[crate::parser::ast::CatchBlock],
        else_block: Option<&[Stmt]>,
        line: usize,
    ) -> Result<(), LangError> {
        // Сохраняем текущую высоту стека
        let stack_height = self.chunk.code.len();
        
        // Создаем обработчик исключений
        let mut handler = ExceptionHandler {
            catch_ips: Vec::new(),
            error_types: Vec::new(),
            error_var_slots: Vec::new(),
            else_ip: None,
            stack_height,
        };
        
        // Начинаем новую область видимости для try блока
        self.begin_scope();
        
        // Генерируем BeginTry (пока без индекса обработчика, патчим позже)
        let begin_try_ip = self.chunk.code.len();
        self.chunk.write_with_line(OpCode::BeginTry(0), line); // Временное значение
        
        // Компилируем try блок
        for stmt in try_block {
            self.compile_stmt(stmt)?;
        }
        
        // Генерируем EndTry
        self.chunk.write_with_line(OpCode::EndTry, line);
        
        // Завершаем область видимости try блока
        self.end_scope();
        
        // Создаем метки для try/catch блоков
        let after_try_label = self.create_label();
        let after_catch_label = if else_block.is_some() {
            Some(self.create_label())
        } else {
            None
        };
        
        // Добавляем Jump после EndTry, чтобы пропустить catch блоки, если ошибки не было
        self.emit_jump(false, after_try_label)?;
        
        // Компилируем catch блоки
        for catch_block in catch_blocks {
            // Сохраняем IP начала catch блока
            let catch_ip = self.chunk.code.len();
            handler.catch_ips.push(catch_ip);
            
            // Определяем тип ошибки
            let error_type_index = if let Some(ref error_type_name) = catch_block.error_type {
                Some(self.get_error_type_index(error_type_name))
            } else {
                None // catch всех
            };
            handler.error_types.push(error_type_index);
            
            // Если есть переменная ошибки, создаем для неё слот
            let error_var_slot = if let Some(ref error_var) = catch_block.error_var {
                self.begin_scope();
                let slot = self.declare_local(error_var);
                Some(slot)
            } else {
                None
            };
            handler.error_var_slots.push(error_var_slot);
            
            // Генерируем Catch опкод
            self.chunk.write_with_line(
                OpCode::Catch(error_type_index),
                catch_block.line,
            );
            
            // Компилируем тело catch блока
            for stmt in &catch_block.body {
                self.compile_stmt(stmt)?;
            }
            
            // Генерируем EndCatch
            self.chunk.write_with_line(OpCode::EndCatch, catch_block.line);
            
            // Завершаем область видимости catch блока
            if error_var_slot.is_some() {
                self.end_scope();
            }
            
            // Добавляем Jump после EndCatch, чтобы пропустить else блок
            // (catch блок уже обработал ошибку, else блок не нужен)
            if let Some(ref after_catch_label) = after_catch_label {
                self.emit_jump(false, *after_catch_label)?;
            }
        }
        
        // Помечаем метку после try (начало catch блоков)
        self.mark_label(after_try_label);
        
        // Компилируем else блок (если есть)
        if let Some(else_block) = else_block {
            let else_ip = self.chunk.code.len();
            handler.else_ip = Some(else_ip);
            
            self.begin_scope();
            for stmt in else_block {
                self.compile_stmt(stmt)?;
            }
            self.end_scope();
            
            // Помечаем метку после catch блоков (если есть else)
            if let Some(ref after_catch_label) = after_catch_label {
                self.mark_label(*after_catch_label);
            }
        }
        
        // Добавляем обработчик в стек компилятора
        let handler_index = self.exception_handlers.len();
        self.exception_handlers.push(handler.clone());
        
        // Сохраняем обработчик в chunk
        let handler_info = crate::bytecode::ExceptionHandlerInfo {
            catch_ips: handler.catch_ips.clone(),
            error_types: handler.error_types.clone(),
            error_var_slots: handler.error_var_slots.clone(),
            else_ip: handler.else_ip,
            stack_height: handler.stack_height,
        };
        self.chunk.exception_handlers.push(handler_info);
        
        // Копируем таблицу типов ошибок в chunk (если еще не скопирована)
        if self.chunk.error_type_table.is_empty() {
            self.chunk.error_type_table = self.error_type_table.clone();
        }
        
        // Патчим BeginTry с правильным индексом обработчика
        if let Some(OpCode::BeginTry(_)) = self.chunk.code.get_mut(begin_try_ip) {
            *self.chunk.code.get_mut(begin_try_ip).unwrap() = OpCode::BeginTry(handler_index);
        }
        
        // Генерируем PopExceptionHandler в конце
        self.chunk.write_with_line(OpCode::PopExceptionHandler, line);
        
        Ok(())
    }
}

