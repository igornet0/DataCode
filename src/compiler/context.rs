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

