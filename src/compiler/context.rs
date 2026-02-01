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
    /// Subclass name -> superclass name, set when implicit constructor was skipped (superclass has no matching constructor).
    pub class_superclass: &'a mut std::collections::HashMap<String, String>,
    /// Current class being compiled (for super.method() resolution).
    pub current_class: Option<String>,
    /// Superclass of current class (for super() and super.method() resolution).
    pub current_superclass: Option<String>,
    /// Whether we are currently compiling a constructor body.
    pub in_constructor: bool,
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

