// Exception handling for VM

use crate::common::{error::{LangError, StackTraceEntry, ErrorType}, value::Value};
use crate::vm::frame::CallFrame;

// Структура для обработчика исключений в VM
pub struct ExceptionHandler {
    pub catch_ips: Vec<usize>,           // IP начала каждого catch блока
    pub error_types: Vec<Option<usize>>, // Типы ошибок для каждого catch (None для catch всех)
    pub error_var_slots: Vec<Option<usize>>, // Слоты для переменных ошибок
    pub else_ip: Option<usize>,         // IP начала else блока
    pub finally_ip: Option<usize>,      // IP начала finally блока
    pub stack_height: usize,             // Высота стека при входе в try
    pub had_error: bool,                 // Флаг, указывающий, была ли ошибка в try блоке
    pub frame_index: usize,              // Индекс фрейма, к которому относится этот обработчик
}

impl ExceptionHandler {
    pub fn new(frame_index: usize, stack_height: usize) -> Self {
        Self {
            catch_ips: Vec::new(),
            error_types: Vec::new(),
            error_var_slots: Vec::new(),
            else_ip: None,
            finally_ip: None,
            stack_height,
            had_error: false,
            frame_index,
        }
    }

    pub fn build_stack_trace(frames: &[CallFrame]) -> Vec<StackTraceEntry> {
        let mut trace = Vec::new();
        for frame in frames {
            let line = if frame.ip > 0 {
                frame.function.chunk.get_line(frame.ip - 1)
            } else {
                0
            };
            trace.push(StackTraceEntry {
                function_name: frame.function.name.clone(),
                line,
            });
        }
        trace.reverse(); // Начинаем с самой глубокой функции
        trace
    }

    pub fn runtime_error(frames: &[CallFrame], message: String, line: usize) -> LangError {
        LangError::runtime_error_with_trace(message, line, Self::build_stack_trace(frames))
    }

    pub fn runtime_error_with_type(frames: &[CallFrame], message: String, line: usize, error_type: ErrorType) -> LangError {
        LangError::runtime_error_with_type_and_trace(message, line, error_type, Self::build_stack_trace(frames))
    }

    /// Обрабатывает исключение - проверяет стек обработчиков и переходит к соответствующему catch блоку
    pub fn handle_exception(
        vm_stack: &mut Vec<Value>,
        vm_frames: &mut Vec<CallFrame>,
        exception_handlers: &mut Vec<ExceptionHandler>,
        error: LangError,
    ) -> Result<(), LangError> {
        // Получаем текущий IP для проверки, не находимся ли мы уже внутри catch блока
        let current_ip = if let Some(frame) = vm_frames.last() {
            frame.ip
        } else {
            0
        };
        
        // Проверяем стек обработчиков (сверху вниз)
        // Обработчики привязаны к конкретным фреймам через frame_index
        for handler in exception_handlers.iter_mut().rev() {
            let handler_frame_index = handler.frame_index;
            
            if handler_frame_index >= vm_frames.len() {
                continue;
            }
            
            // Получаем chunk функции для этого фрейма
            let frame = &vm_frames[handler_frame_index];
            let chunk = &frame.function.chunk;
            
            // Проверяем, не находимся ли мы уже внутри catch блока этого обработчика
            if handler_frame_index == vm_frames.len() - 1 {
                let is_inside_catch = handler.catch_ips.iter().enumerate().any(|(i, &catch_ip)| {
                    if current_ip < catch_ip {
                        return false;
                    }
                    // Проверяем, не прошли ли мы этот catch блок
                    if let Some(&next_catch_ip) = handler.catch_ips.get(i + 1) {
                        current_ip < next_catch_ip
                    } else {
                        // Это последний catch блок, проверяем, что current_ip < else_ip (если есть)
                        if let Some(else_ip) = handler.else_ip {
                            current_ip < else_ip
                        } else {
                            true // Нет else блока, значит catch блок последний, и мы внутри него
                        }
                    }
                });
                
                // Если мы находимся внутри catch блока этого обработчика, пропускаем его
                if is_inside_catch {
                    continue;
                }
            }
            
            // Проверяем каждый catch блок
            for (i, catch_ip) in handler.catch_ips.iter().enumerate() {
                let error_type = handler.error_types.get(i);
                let error_var_slot = handler.error_var_slots.get(i);
                
                // Используем таблицу типов ошибок из chunk функции
                let error_type_table = &chunk.error_type_table;
                
                // Проверяем, подходит ли этот catch блок для данной ошибки
                let matches = match error_type {
                    Some(Some(expected_type_index)) => {
                        // Типизированный catch - проверяем тип ошибки
                        if let Some(error_type_name) = error_type_table.get(*expected_type_index) {
                            if let Some(et) = ErrorType::from_name(error_type_name) {
                                error.is_instance_of(&et)
                            } else {
                                false
                            }
                        } else {
                            false
                        }
                    }
                    Some(None) => {
                        // catch всех
                        true
                    }
                    None => false,
                };
                
                if matches {
                    // Нашли подходящий catch блок
                    handler.had_error = true;
                    
                    // Очищаем стек до нужной высоты
                    while vm_stack.len() > handler.stack_height {
                        vm_stack.pop();
                    }
                    
                    // Удаляем все фреймы до фрейма с обработчиком
                    while vm_frames.len() > handler_frame_index + 1 {
                        vm_frames.pop();
                    }
                    
                    // Сохраняем ошибку в переменную (если указана)
                    if let Some(Some(slot)) = error_var_slot {
                        // Преобразуем ошибку в строку для сохранения в переменную
                        let error_string = format!("{}", error);
                        let frame = vm_frames.last_mut().unwrap();
                        if *slot >= frame.slots.len() {
                            frame.slots.resize(*slot + 1, Value::Null);
                        }
                        frame.slots[*slot] = Value::String(error_string);
                    }
                    
                    // Переходим к catch блоку в правильном фрейме
                    let frame = vm_frames.last_mut().unwrap();
                    frame.ip = *catch_ip;
                    
                    return Ok(());
                }
            }
        }
        
        // Обработчик не найден - возвращаем ошибку
        Err(error)
    }
}

