/// Контекст компиляции для передачи между модулями

use crate::bytecode::{Chunk, Function};
use crate::compiler::scope::ScopeManager;
use crate::compiler::labels::LabelManager;

// Структура для отслеживания обработчиков исключений
#[derive(Clone)]
pub struct ExceptionHandler {
    pub catch_ips: Vec<usize>,           // IP начала каждого catch блока
    pub error_types: Vec<Option<usize>>, // Типы ошибок для каждого catch (None для catch всех)
    pub error_var_slots: Vec<Option<usize>>, // Слоты для переменных ошибок
    pub else_ip: Option<usize>,          // IP начала else блока
    pub finally_ip: Option<usize>,       // IP начала finally блока
    pub stack_height: usize,            // Высота стека при входе в try
}

// Структура для отслеживания контекста циклов
pub struct LoopContext {
    pub continue_label: usize,   // Метка для continue (начало следующей итерации или инкремент)
    pub break_label: usize,      // Метка для break (конец цикла)
    /// true для for i in range(...); при break нужно снять состояние с for_range_stack
    pub is_for_range: bool,
}

/// Контекст компиляции для передачи между модулями
pub struct CompilationContext<'a> {
    pub chunk: &'a mut Chunk,
    pub scope: &'a mut ScopeManager,
    pub labels: &'a mut LabelManager,
    pub functions: &'a mut Vec<Function>,
    pub function_names: &'a mut Vec<String>,
    pub current_function: Option<usize>,
    pub current_line: &'a mut usize,
    pub exception_handlers: &'a mut Vec<ExceptionHandler>,
    pub error_type_table: &'a mut Vec<String>,
    pub loop_contexts: &'a mut Vec<LoopContext>,
    /// Symbols imported via `from module import X`: name -> module. Used to treat uppercase names as class constructor calls only for file modules (not builtins).
    pub imported_symbols: &'a mut std::collections::HashMap<String, String>,
    /// Class name -> list of private field names (filled at end of each class; used to merge in subclass constructors).
    pub class_private_fields: &'a mut std::collections::HashMap<String, Vec<String>>,
    /// Class name -> list of protected field names (for inheritance: merge in subclass constructors).
    pub class_protected_fields: &'a mut std::collections::HashMap<String, Vec<String>>,
    /// Class name -> list of private method names (for inheritance: merge in subclass constructors).
    pub class_private_methods: &'a mut std::collections::HashMap<String, Vec<String>>,
    /// Class name -> list of protected method names (for inheritance: merge in subclass constructors).
    pub class_protected_methods: &'a mut std::collections::HashMap<String, Vec<String>>,
    /// Subclass name -> superclass name, set when implicit constructor was skipped (superclass has no matching constructor).
    pub class_superclass: &'a mut std::collections::HashMap<String, String>,
    /// Class name -> true if the class extends Table (directly or indirectly). Used for isinstance(x, Table).
    pub class_extends_table: &'a mut std::collections::HashMap<String, bool>,
    /// Class name -> (constructor_name, function_index). Used to resolve named-arg constructor calls (e.g. User(name="Alice")) for extends_table classes.
    pub class_constructor: &'a mut std::collections::HashMap<String, (String, usize)>,
    /// Class names marked with @Abstract; calls to these must load the class object (not constructor) so VM can check __abstract.
    pub abstract_classes: &'a mut std::collections::HashSet<String>,
    /// Class name -> env_prefix from model_config (for Settings subclasses). Used to build nested_specs when calling load_env.
    pub class_settings_env_prefix: &'a mut std::collections::HashMap<String, String>,
    /// Class name -> nested_specs Value (array) for Settings subclasses. Used so subclasses (e.g. DevSettings) pass parent's nested_specs when calling super.
    pub class_nested_specs_value: &'a mut std::collections::HashMap<String, crate::common::value::Value>,
    /// Class name -> default required_keys array for Settings subclasses. Used at call site for Config(path) to pass 3 args.
    pub class_required_keys_value: &'a mut std::collections::HashMap<String, crate::common::value::Value>,
    /// Current class being compiled (for super.method() resolution).
    pub current_class: Option<String>,
    /// Superclass of current class (for super() and super.method() resolution).
    pub current_superclass: Option<String>,
    /// Whether we are currently compiling a constructor body.
    pub in_constructor: bool,
    /// When in constructor body, the slot index for "this" (arity). Used so body always uses correct slot regardless of scope.
    pub constructor_this_slot: Option<usize>,
    /// Source file path for error messages (propagated to chunk.source_name).
    pub source_name: Option<&'a str>,
}

impl<'a> CompilationContext<'a> {
    pub fn get_error_type_index(&mut self, error_type_name: &str) -> usize {
        // Ищем в существующей таблице
        if let Some(index) = self.error_type_table.iter().position(|s| s == error_type_name) {
            return index;
        }
        // Добавляем новый тип
        let index = self.error_type_table.len();
        self.error_type_table.push(error_type_name.to_string());
        index
    }
}

