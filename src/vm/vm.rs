// Виртуальная машина

use crate::bytecode::{Chunk, OpCode};
use crate::common::{error::{LangError, StackTraceEntry, ErrorType}, value::Value};
use crate::vm::frame::CallFrame;
use crate::vm::natives;
use crate::ml::tensor::Tensor;
use std::rc::Rc;
use std::cell::RefCell;

pub type NativeFn = fn(&[Value]) -> Value;

// Thread-local storage для хранения контекста VM во время вызова нативных функций
// Это позволяет нативным функциям вызывать пользовательские функции
thread_local! {
    pub(crate) static VM_CALL_CONTEXT: RefCell<Option<*mut Vm>> = RefCell::new(None);
}

/// Структура для хранения явной связи между колонками таблиц
#[derive(Debug, Clone)]
pub struct ExplicitRelation {
    pub source_table_name: String,
    pub source_column_name: String,
    pub target_table_name: String,
    pub target_column_name: String,
}

/// Структура для хранения явного первичного ключа таблицы
#[derive(Debug, Clone)]
pub struct ExplicitPrimaryKey {
    pub table_name: String,
    pub column_name: String,
}

// Структура для обработчика исключений в VM
struct ExceptionHandler {
    catch_ips: Vec<usize>,           // IP начала каждого catch блока
    error_types: Vec<Option<usize>>, // Типы ошибок для каждого catch (None для catch всех)
    error_var_slots: Vec<Option<usize>>, // Слоты для переменных ошибок
    else_ip: Option<usize>,         // IP начала else блока
    stack_height: usize,             // Высота стека при входе в try
    had_error: bool,                 // Флаг, указывающий, была ли ошибка в try блоке
    frame_index: usize,              // Индекс фрейма, к которому относится этот обработчик
}

/// Статус выполнения одного шага VM
#[derive(Debug)]
enum VMStatus {
    Continue,        // Продолжить выполнение
    Return(Value),   // Возврат из функции с значением
    FrameEnded,      // Фрейм завершился без return
}

pub struct Vm {
    stack: Vec<Value>,
    frames: Vec<CallFrame>,
    globals: Vec<Value>,
    functions: Vec<crate::bytecode::Function>,
    natives: Vec<NativeFn>,
    exception_handlers: Vec<ExceptionHandler>, // Стек обработчиков исключений
    error_type_table: Vec<String>, // Таблица типов ошибок для текущей функции
    global_names: std::collections::HashMap<usize, String>, // Маппинг индексов глобальных переменных на их имена
    explicit_global_names: std::collections::HashMap<usize, String>, // Маппинг индексов переменных, явно объявленных с ключевым словом 'global'
    explicit_relations: Vec<ExplicitRelation>, // Явные связи, созданные через relate()
    explicit_primary_keys: Vec<ExplicitPrimaryKey>, // Явные первичные ключи, созданные через primary_key()
    loaded_modules: std::collections::HashSet<String>, // Множество загруженных модулей
}

impl Vm {
    pub fn new() -> Self {
        let mut vm = Self {
            stack: Vec::new(),
            frames: Vec::new(),
            globals: Vec::new(),
            functions: Vec::new(),
            natives: Vec::new(),
            exception_handlers: Vec::new(),
            error_type_table: Vec::new(),
            global_names: std::collections::HashMap::new(),
            explicit_global_names: std::collections::HashMap::new(),
            explicit_relations: Vec::new(),
            explicit_primary_keys: Vec::new(),
            loaded_modules: std::collections::HashSet::new(),
        };
        vm.register_natives();
        vm
    }

    fn register_natives(&mut self) {
        // Регистрируем нативные функции
        // Порядок важен - индексы должны соответствовать register_native_globals
        self.natives.push(natives::native_print);      // 0
        self.natives.push(natives::native_len);        // 1
        self.natives.push(natives::native_range);      // 2
        self.natives.push(natives::native_int);        // 3
        self.natives.push(natives::native_float);        // 4
        self.natives.push(natives::native_bool);        // 5
        self.natives.push(natives::native_str);        // 6
        self.natives.push(natives::native_array);      // 7
        self.natives.push(natives::native_typeof);     // 8
        self.natives.push(natives::native_isinstance); // 9
        self.natives.push(natives::native_date);      // 10
        self.natives.push(natives::native_money);     // 11
        self.natives.push(natives::native_path);     // 12
        self.natives.push(natives::native_path_name);     // 13
        self.natives.push(natives::native_path_parent);   // 14
        self.natives.push(natives::native_path_exists);   // 15
        self.natives.push(natives::native_path_is_file); // 16
        self.natives.push(natives::native_path_is_dir);  // 17
        self.natives.push(natives::native_path_extension); // 18
        self.natives.push(natives::native_path_stem);    // 19
        self.natives.push(natives::native_path_len);     // 20
        // Математические функции
        self.natives.push(natives::native_abs);          // 21
        self.natives.push(natives::native_sqrt);         // 22
        self.natives.push(natives::native_pow);          // 23
        self.natives.push(natives::native_min);          // 24
        self.natives.push(natives::native_max);          // 25
        self.natives.push(natives::native_round);        // 26
        // Строковые функции
        self.natives.push(natives::native_upper);        // 27
        self.natives.push(natives::native_lower);        // 28
        self.natives.push(natives::native_trim);         // 29
        self.natives.push(natives::native_split);        // 30
        self.natives.push(natives::native_join);         // 31
        self.natives.push(natives::native_contains);     // 32
        // Функции массивов
        self.natives.push(natives::native_push);         // 33
        self.natives.push(natives::native_pop);          // 34
        self.natives.push(natives::native_unique);       // 35
        self.natives.push(natives::native_reverse);      // 36
        self.natives.push(natives::native_sort);        // 37
        self.natives.push(natives::native_sum);          // 38
        self.natives.push(natives::native_average);     // 39
        self.natives.push(natives::native_count);        // 40
        self.natives.push(natives::native_any);          // 41
        self.natives.push(natives::native_all);          // 42
        // Функции для работы с таблицами
        self.natives.push(natives::native_table);        // 43
        self.natives.push(natives::native_read_file);    // 44
        self.natives.push(natives::native_table_info);   // 45
        self.natives.push(natives::native_table_head);   // 46
        self.natives.push(natives::native_table_tail);   // 47
        self.natives.push(natives::native_table_select); // 48
        self.natives.push(natives::native_table_sort);   // 49
        self.natives.push(natives::native_table_where);  // 50
        self.natives.push(natives::native_show_table);   // 51
        self.natives.push(natives::native_merge_tables); // 52
        self.natives.push(natives::native_now);          // 53
        self.natives.push(natives::native_getcwd);       // 54
        self.natives.push(natives::native_list_files);   // 55
        // JOIN операции
        self.natives.push(natives::native_inner_join);   // 56
        self.natives.push(natives::native_left_join);    // 57
        self.natives.push(natives::native_right_join);  // 58
        self.natives.push(natives::native_full_join);    // 59
        self.natives.push(natives::native_cross_join);   // 60
        self.natives.push(natives::native_semi_join);   // 61
        self.natives.push(natives::native_anti_join);   // 62
        self.natives.push(natives::native_zip_join);    // 63
        self.natives.push(natives::native_asof_join);   // 64
        self.natives.push(natives::native_apply_join);   // 65
        self.natives.push(natives::native_join_on);     // 66
        self.natives.push(natives::native_table_suffixes); // 67
        self.natives.push(natives::native_relate);      // 68
        self.natives.push(natives::native_primary_key); // 69
    }

    pub fn set_functions(&mut self, functions: Vec<crate::bytecode::Function>) {
        self.functions = functions;
        // Заполняем имена глобальных переменных из chunk главной функции (первая функция)
        // Merge with existing global_names instead of overwriting to preserve any pre-registered globals
        if let Some(main_function) = self.functions.first() {
            for (idx, name) in &main_function.chunk.global_names {
                self.global_names.insert(*idx, name.clone());
            }
            for (idx, name) in &main_function.chunk.explicit_global_names {
                self.explicit_global_names.insert(*idx, name.clone());
            }
        }
    }

    pub fn register_native_globals(&mut self) {
        // Регистрируем нативные функции в глобальных переменных
        // Порядок должен соответствовать register_natives()
        // Ensure globals vector is large enough for native functions (70) and any globals from compiler
        let max_global_index = self.global_names.keys().max().copied().unwrap_or(69);
        let min_size = (70).max(max_global_index + 1);
        self.globals.resize(min_size, Value::Null);
        
        self.globals[0] = Value::NativeFunction(0);  // print
        self.globals[1] = Value::NativeFunction(1);  // len
        self.globals[2] = Value::NativeFunction(2);  // range
        self.globals[3] = Value::NativeFunction(3);  // int
        self.globals[4] = Value::NativeFunction(4);  // float
        self.globals[5] = Value::NativeFunction(5);  // bool
        self.globals[6] = Value::NativeFunction(6);  // str
        self.globals[7] = Value::NativeFunction(7);  // array
        self.globals[8] = Value::NativeFunction(8);  // typeof
        self.globals[9] = Value::NativeFunction(9);  // isinstance
        self.globals[10] = Value::NativeFunction(10);  // date
        self.globals[11] = Value::NativeFunction(11);  // money
        self.globals[12] = Value::NativeFunction(12);  // path
        self.globals[13] = Value::NativeFunction(13);  // path_name
        self.globals[14] = Value::NativeFunction(14);  // path_parent
        self.globals[15] = Value::NativeFunction(15);  // path_exists
        self.globals[16] = Value::NativeFunction(16);  // path_is_file
        self.globals[17] = Value::NativeFunction(17);  // path_is_dir
        self.globals[18] = Value::NativeFunction(18);  // path_extension
        self.globals[19] = Value::NativeFunction(19);  // path_stem
        self.globals[20] = Value::NativeFunction(20);  // path_len
        // Математические функции
        self.globals[21] = Value::NativeFunction(21);  // abs
        self.globals[22] = Value::NativeFunction(22);  // sqrt
        self.globals[23] = Value::NativeFunction(23);  // pow
        self.globals[24] = Value::NativeFunction(24);  // min
        self.globals[25] = Value::NativeFunction(25);  // max
        self.globals[26] = Value::NativeFunction(26);  // round
        // Строковые функции
        self.globals[27] = Value::NativeFunction(27);  // upper
        self.globals[28] = Value::NativeFunction(28);  // lower
        self.globals[29] = Value::NativeFunction(29);  // trim
        self.globals[30] = Value::NativeFunction(30);  // split
        self.globals[31] = Value::NativeFunction(31);  // join
        self.globals[32] = Value::NativeFunction(32);  // contains
        // Функции массивов
        self.globals[33] = Value::NativeFunction(33);  // push
        self.globals[34] = Value::NativeFunction(34);  // pop
        self.globals[35] = Value::NativeFunction(35);  // unique
        self.globals[36] = Value::NativeFunction(36);  // reverse
        self.globals[37] = Value::NativeFunction(37);  // sort
        self.globals[38] = Value::NativeFunction(38);  // sum
        self.globals[39] = Value::NativeFunction(39);  // average
        self.globals[40] = Value::NativeFunction(40);  // count
        self.globals[41] = Value::NativeFunction(41);  // any
        self.globals[42] = Value::NativeFunction(42);  // all
        // Функции для работы с таблицами
        self.globals[43] = Value::NativeFunction(43);  // table
        self.globals[44] = Value::NativeFunction(44);  // read_file
        self.globals[45] = Value::NativeFunction(45);  // table_info
        self.globals[46] = Value::NativeFunction(46);  // table_head
        self.globals[47] = Value::NativeFunction(47);  // table_tail
        self.globals[48] = Value::NativeFunction(48);  // table_select
        self.globals[49] = Value::NativeFunction(49);  // table_sort
        self.globals[50] = Value::NativeFunction(50);  // table_where
        self.globals[51] = Value::NativeFunction(51);  // show_table
        self.globals[52] = Value::NativeFunction(52);  // merge_tables
        self.globals[53] = Value::NativeFunction(53);  // now
        self.globals[54] = Value::NativeFunction(54);  // getcwd
        self.globals[55] = Value::NativeFunction(55);  // list_files
        // JOIN операции
        self.globals[56] = Value::NativeFunction(56);  // inner_join
        self.globals[57] = Value::NativeFunction(57);  // left_join
        self.globals[58] = Value::NativeFunction(58);  // right_join
        self.globals[59] = Value::NativeFunction(59);  // full_join
        self.globals[60] = Value::NativeFunction(60);  // cross_join
        self.globals[61] = Value::NativeFunction(61);  // semi_join
        self.globals[62] = Value::NativeFunction(62);  // anti_join
        self.globals[63] = Value::NativeFunction(63);  // zip_join
        self.globals[64] = Value::NativeFunction(64);  // asof_join
        self.globals[65] = Value::NativeFunction(65);  // apply_join
        self.globals[66] = Value::NativeFunction(66);  // join_on
        self.globals[67] = Value::NativeFunction(67);  // table_suffixes
        self.globals[68] = Value::NativeFunction(68);  // relate
        self.globals[69] = Value::NativeFunction(69);  // primary_key
    }

    fn build_stack_trace(&self) -> Vec<StackTraceEntry> {
        let mut trace = Vec::new();
        for frame in &self.frames {
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

    fn runtime_error(&self, message: String, line: usize) -> LangError {
        LangError::runtime_error_with_trace(message, line, self.build_stack_trace())
    }

    fn runtime_error_with_type(&self, message: String, line: usize, error_type: ErrorType) -> LangError {
        LangError::runtime_error_with_type_and_trace(message, line, error_type, self.build_stack_trace())
    }

    /// Обрабатывает исключение - проверяет стек обработчиков и переходит к соответствующему catch блоку
    fn handle_exception(&mut self, error: LangError) -> Result<(), LangError> {
        // Получаем текущий IP для проверки, не находимся ли мы уже внутри catch блока
        let current_ip = if let Some(frame) = self.frames.last() {
            frame.ip
        } else {
            0
        };
        
        // Проверяем стек обработчиков (сверху вниз)
        // Обработчики привязаны к конкретным фреймам через frame_index
        for handler in self.exception_handlers.iter_mut().rev() {
            let handler_frame_index = handler.frame_index;
            
            if handler_frame_index >= self.frames.len() {
                continue;
            }
            
            // Получаем chunk функции для этого фрейма
            let frame = &self.frames[handler_frame_index];
            let chunk = &frame.function.chunk;
            
            // Проверяем, не находимся ли мы уже внутри catch блока этого обработчика
            // Если мы в том же фрейме и current_ip >= catch_ip для какого-то catch блока,
            // и current_ip < следующего catch_ip (или else_ip, если это последний catch),
            // значит мы внутри catch блока
            if handler_frame_index == self.frames.len() - 1 {
                let is_inside_catch = handler.catch_ips.iter().enumerate().any(|(i, &catch_ip)| {
                    if current_ip < catch_ip {
                        return false;
                    }
                    // Проверяем, не прошли ли мы этот catch блок
                    // Если есть следующий catch блок, проверяем, что current_ip < следующего catch_ip
                    if let Some(&next_catch_ip) = handler.catch_ips.get(i + 1) {
                        current_ip < next_catch_ip
                    } else {
                        // Это последний catch блок, проверяем, что current_ip < else_ip (если есть)
                        // или просто что мы >= catch_ip (если else_ip нет, значит catch блок последний)
                        if let Some(else_ip) = handler.else_ip {
                            current_ip < else_ip
                        } else {
                            true // Нет else блока, значит catch блок последний, и мы внутри него
                        }
                    }
                });
                
                // Если мы находимся внутри catch блока этого обработчика, пропускаем его
                // и ищем следующий обработчик выше по стеку
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
                    // Устанавливаем флаг ошибки
                    handler.had_error = true;
                    
                    // Очищаем стек до нужной высоты
                    while self.stack.len() > handler.stack_height {
                        self.stack.pop();
                    }
                    
                    // Удаляем все фреймы до фрейма с обработчиком
                    while self.frames.len() > handler_frame_index + 1 {
                        self.frames.pop();
                    }
                    
                    // Сохраняем ошибку в переменную (если указана)
                    if let Some(Some(slot)) = error_var_slot {
                        // Преобразуем ошибку в строку для сохранения в переменную
                        let error_string = format!("{}", error);
                        let frame = self.frames.last_mut().unwrap();
                        if *slot >= frame.slots.len() {
                            frame.slots.resize(*slot + 1, Value::Null);
                        }
                        frame.slots[*slot] = Value::String(error_string);
                    }
                    
                    // Переходим к catch блоку в правильном фрейме
                    let frame = self.frames.last_mut().unwrap();
                    frame.ip = *catch_ip;
                    
                    return Ok(());
                }
            }
        }
        
        // Обработчик не найден - возвращаем ошибку
        Err(error)
    }

    pub fn run(&mut self, chunk: &Chunk) -> Result<Value, LangError> {
        // Заполняем имена глобальных переменных из chunk
        // Merge with existing global_names instead of overwriting to preserve modules registered during execution
        for (idx, name) in &chunk.global_names {
            self.global_names.insert(*idx, name.clone());
        }
        for (idx, name) in &chunk.explicit_global_names {
            self.explicit_global_names.insert(*idx, name.clone());
        }
        
        // Создаем начальный frame
        let function = crate::bytecode::Function::new("<main>".to_string(), 0);
        let mut function = function;
        function.chunk = chunk.clone();
        let frame = CallFrame::new(function, 0);
        self.frames.push(frame);

        loop {
            match self.step()? {
                VMStatus::Continue => {}
                VMStatus::Return(v) => return Ok(v),
                VMStatus::FrameEnded => break,
            }
        }

        // После завершения выполнения возвращаем последнее значение на стеке
        if !self.stack.is_empty() {
            Ok(self.stack.pop().unwrap())
        } else {
            Ok(Value::Null)
        }
    }

    /// Выполнить один шаг VM - получить следующую инструкцию и выполнить её
    fn step(&mut self) -> Result<VMStatus, LangError> {
        let frame = match self.frames.last_mut() {
            Some(f) => f,
            None => return Ok(VMStatus::FrameEnded),
        };

        if frame.ip >= frame.function.chunk.code.len() {
            self.frames.pop();
            return Ok(VMStatus::FrameEnded);
        }

        let ip = frame.ip;
        let instruction = frame.function.chunk.code[ip].clone();
        let line = frame.function.chunk.get_line(ip);
        frame.ip += 1;

        self.execute_instruction(instruction, line)
    }

    /// Выполнить одну инструкцию
    /// Возвращает VMStatus, указывающий, что делать дальше
    fn execute_instruction(&mut self, instruction: OpCode, line: usize) -> Result<VMStatus, LangError> {
        let frame = self.frames.last_mut().unwrap();
        
        match instruction {
                OpCode::Import(module_index) => {
                    let module_name = match &frame.function.chunk.constants[module_index] {
                        Value::String(name) => name.clone(),
                        _ => {
                            let error = self.runtime_error(
                                "Import expects module name as string".to_string(),
                                line,
                            );
                            match self.handle_exception(error) {
                                Ok(()) => return Ok(VMStatus::Continue),
                                Err(e) => return Err(e),
                            }
                        }
                    };
                    
                    // Check if module is already loaded
                    if self.loaded_modules.contains(&module_name) {
                        return Ok(VMStatus::Continue);
                    }
                    
                    // Register the module
                    self.register_module(&module_name)?;
                    self.loaded_modules.insert(module_name);
                    return Ok(VMStatus::Continue);
                }
                OpCode::ImportFrom(module_index, items_index) => {
                    // Get module name
                    let module_name = match &frame.function.chunk.constants[module_index] {
                        Value::String(name) => name.clone(),
                        _ => {
                            let error = self.runtime_error(
                                "ImportFrom expects module name as string".to_string(),
                                line,
                            );
                            match self.handle_exception(error) {
                                Ok(()) => return Ok(VMStatus::Continue),
                                Err(e) => return Err(e),
                            }
                        }
                    };
                    
                    // Get items array
                    let items_array = match &frame.function.chunk.constants[items_index] {
                        Value::Array(arr) => arr.borrow().clone(),
                        _ => {
                            let error = self.runtime_error(
                                "ImportFrom expects items array".to_string(),
                                line,
                            );
                            match self.handle_exception(error) {
                                Ok(()) => return Ok(VMStatus::Continue),
                                Err(e) => return Err(e),
                            }
                        }
                    };
                    
                    // Register the module if not already loaded
                    if !self.loaded_modules.contains(&module_name) {
                        self.register_module(&module_name)?;
                        self.loaded_modules.insert(module_name.clone());
                    }
                    
                    // Get the module object from globals
                    let module_global_index = if let Some((&idx, _)) = self.global_names.iter().find(|(_, name)| name.as_str() == module_name) {
                        idx
                    } else {
                        let error = self.runtime_error(
                            format!("Module {} not found in globals", module_name),
                            line,
                        );
                        match self.handle_exception(error) {
                            Ok(()) => return Ok(VMStatus::Continue),
                            Err(e) => return Err(e),
                        }
                    };
                    
                    // Ensure globals vector is large enough
                    if module_global_index >= self.globals.len() {
                        self.globals.resize(module_global_index + 1, Value::Null);
                    }
                    
                    // Get the module object from globals
                    // First verify it exists and is an Object
                    if module_global_index >= self.globals.len() {
                        let error = self.runtime_error(
                            format!("Module {} global index {} out of bounds (globals.len() = {})", 
                                module_name, module_global_index, self.globals.len()),
                            line,
                        );
                        match self.handle_exception(error) {
                            Ok(()) => return Ok(VMStatus::Continue),
                            Err(e) => return Err(e),
                        }
                    }
                    
                    let module_object_ref = match &self.globals[module_global_index] {
                        Value::Object(map) => map,
                        Value::Null => {
                            let error = self.runtime_error(
                                format!("Module {} is Null - module registration may have failed", module_name),
                                line,
                            );
                            match self.handle_exception(error) {
                                Ok(()) => return Ok(VMStatus::Continue),
                                Err(e) => return Err(e),
                            }
                        }
                        _ => {
                            let error = self.runtime_error(
                                format!("Module {} is not an object (found: {:?})", module_name, 
                                    std::mem::discriminant(&self.globals[module_global_index])),
                                line,
                            );
                            match self.handle_exception(error) {
                                Ok(()) => return Ok(VMStatus::Continue),
                                Err(e) => return Err(e),
                            }
                        }
                    };
                    // Clone the HashMap to avoid borrowing issues
                    let module_object = module_object_ref.clone();
                    
                    // Import items
                    for item_value in items_array {
                        match item_value {
                            Value::String(item_str) => {
                                if item_str == "*" {
                                    // Import all items
                                    // First pass: collect all global indices we need without modifying globals
                                    let mut indices_to_set: Vec<(usize, String, Value)> = Vec::new();
                                    let mut max_index_needed = self.globals.len();
                                    let mut new_indices = Vec::new(); // Track new indices we need to create
                                    
                                    for (key, value) in &module_object {
                                        // Find or create global index for this name
                                        let global_index = self.global_names.iter()
                                            .find(|(_, name)| name.as_str() == key.as_str())
                                            .map(|(&idx, _)| idx);
                                        
                                        let global_index = match global_index {
                                            Some(idx) => idx,
                                            None => {
                                                // Name not found - we'll create new global index after calculating all
                                                // Use a temporary index based on current length + new indices count
                                                let idx = self.globals.len() + new_indices.len();
                                                new_indices.push((idx, key.clone()));
                                                idx
                                            }
                                        };
                                        
                                        max_index_needed = max_index_needed.max(global_index + 1);
                                        indices_to_set.push((global_index, key.clone(), value.clone()));
                                    }
                                    
                                    // Resize globals vector once to accommodate all indices
                                    if max_index_needed > self.globals.len() {
                                        self.globals.resize(max_index_needed, Value::Null);
                                    }
                                    
                                    // Register new global names
                                    for (idx, name) in new_indices {
                                        self.global_names.insert(idx, name);
                                    }
                                    
                                    // Second pass: set all values
                                    for (global_index, _key, value) in indices_to_set {
                                        // Store the value at the correct index
                                        self.globals[global_index] = value;
                                    }
                                } else if item_str.contains(':') {
                                    // Aliased import: "name:alias"
                                    let parts: Vec<&str> = item_str.split(':').collect();
                                    if parts.len() == 2 {
                                        let name = parts[0];
                                        let alias = parts[1];
                                        
                                        if let Some(value) = module_object.get(name) {
                                            // Register the alias in globals
                                            let global_index = if let Some(&idx) = self.global_names.iter().find(|(_, n)| n.as_str() == alias).map(|(idx, _)| idx) {
                                                idx
                                            } else {
                                                let idx = self.globals.len();
                                                self.globals.push(value.clone());
                                                self.global_names.insert(idx, alias.to_string());
                                                idx
                                            };
                                            // Update the global value
                                            if global_index >= self.globals.len() {
                                                self.globals.resize(global_index + 1, Value::Null);
                                            }
                                            self.globals[global_index] = value.clone();
                                        } else {
                                            let error = self.runtime_error_with_type(
                                                format!("Module '{}' has no attribute '{}'", module_name, name),
                                                line,
                                                crate::common::error::ErrorType::KeyError,
                                            );
                                            match self.handle_exception(error) {
                                                Ok(()) => continue,
                                                Err(e) => return Err(e),
                                            }
                                        }
                                    }
                                } else {
                                    // Named import: just the name
                                    if let Some(value) = module_object.get(&item_str) {
                                        // Register the name in globals
                                        let global_index = if let Some(&idx) = self.global_names.iter().find(|(_, n)| n.as_str() == item_str.as_str()).map(|(idx, _)| idx) {
                                            idx
                                        } else {
                                            let idx = self.globals.len();
                                            self.globals.push(value.clone());
                                            self.global_names.insert(idx, item_str.clone());
                                            idx
                                        };
                                        // Update the global value
                                        if global_index >= self.globals.len() {
                                            self.globals.resize(global_index + 1, Value::Null);
                                        }
                                        self.globals[global_index] = value.clone();
                                    } else {
                                        let error = self.runtime_error_with_type(
                                            format!("Module '{}' has no attribute '{}'", module_name, item_str),
                                            line,
                                            crate::common::error::ErrorType::KeyError,
                                        );
                                        match self.handle_exception(error) {
                                            Ok(()) => continue,
                                            Err(e) => return Err(e),
                                        }
                                    }
                                }
                            }
                            _ => {
                                let error = self.runtime_error(
                                    "ImportFrom item must be a string".to_string(),
                                    line,
                                );
                                match self.handle_exception(error) {
                                    Ok(()) => continue,
                                    Err(e) => return Err(e),
                                }
                            }
                        }
                    }
                    
                    return Ok(VMStatus::Continue);
                }
                OpCode::Constant(index) => {
                    let value = frame.function.chunk.constants[index].clone();
                    self.push(value);
                    return Ok(VMStatus::Continue);
                }
                OpCode::LoadLocal(index) => {
                    // Для сложных типов (Array, Table) возвращаем ссылку (shallow copy Rc)
                    // Для простых типов клонируем значение
                    let value = &frame.slots[index];
                    let loaded_value = match value {
                        Value::Array(arr_rc) => Value::Array(Rc::clone(arr_rc)),
                        Value::Table(table_rc) => Value::Table(Rc::clone(table_rc)),
                        _ => value.clone(), // Простые типы клонируем
                    };
                    self.push(loaded_value);
                    return Ok(VMStatus::Continue);
                }
                OpCode::StoreLocal(index) => {
                    let value = self.pop()?;
                    // Clone уже создает глубокую копию для массивов и таблиц
                    let frame = self.frames.last_mut().unwrap();
                    if index >= frame.slots.len() {
                        frame.slots.resize(index + 1, Value::Null);
                    }
                    frame.slots[index] = value;
                    return Ok(VMStatus::Continue);
                }
                OpCode::LoadGlobal(index) => {
                    if index >= self.globals.len() {
                        // Check if this is a known module that hasn't been imported
                        let error_message = if let Some(var_name) = self.global_names.get(&index) {
                            if Self::is_known_module(var_name) && !self.loaded_modules.contains(var_name) {
                                format!("Module {} not imported", var_name)
                            } else {
                                format!("Undefined variable: {}", var_name)
                            }
                        } else {
                            format!("Undefined variable")
                        };
                        
                        let error = self.runtime_error(
                            error_message,
                            line,
                        );
                        match self.handle_exception(error) {
                            Ok(()) => {
                                // Исключение обработано, кладем Null на стек
                                self.push(Value::Null);
                            }
                            Err(e) => return Err(e), // Исключение не обработано
                        }
                    } else {
                        // Для сложных типов (Array, Table) возвращаем ссылку (shallow copy Rc)
                        // Для простых типов клонируем значение
                        let value = &self.globals[index];
                        let loaded_value = match value {
                            Value::Array(arr_rc) => Value::Array(Rc::clone(arr_rc)),
                            Value::Table(table_rc) => Value::Table(Rc::clone(table_rc)),
                            _ => value.clone(), // Простые типы клонируем
                        };
                        self.push(loaded_value);
                    }
                    return Ok(VMStatus::Continue);
                }
                OpCode::StoreGlobal(index) => {
                    let mut value = self.pop()?;
                    // Если значение - таблица, устанавливаем её имя из global_names
                    if let Value::Table(table_rc) = &mut value {
                        if let Some(var_name) = self.global_names.get(&index) {
                            table_rc.borrow_mut().set_name(var_name.clone());
                        }
                    }
                    // Clone уже создает глубокую копию для массивов и таблиц
                    if index >= self.globals.len() {
                        self.globals.resize(index + 1, Value::Null);
                    }
                    // Важно: присваиваем value после установки имени, чтобы имя сохранилось
                    self.globals[index] = value;
                    return Ok(VMStatus::Continue);
                }
                OpCode::Add => {
                    let b = self.pop()?;
                    let a = self.pop()?;
                    let result = self.binary_add(&a, &b)?;
                    self.push(result);
                    return Ok(VMStatus::Continue);
                }
                OpCode::Sub => {
                    let b = self.pop()?;
                    let a = self.pop()?;
                    let result = self.binary_sub(&a, &b)?;
                    self.push(result);
                    return Ok(VMStatus::Continue);
                }
                OpCode::Mul => {
                    let b = self.pop()?;
                    let a = self.pop()?;
                    let result = self.binary_mul(&a, &b)?;
                    self.push(result);
                    return Ok(VMStatus::Continue);
                }
                OpCode::Div => {
                    let b = self.pop()?;
                    let a = self.pop()?;
                    let result = self.binary_div(&a, &b)?;
                    self.push(result);
                    return Ok(VMStatus::Continue);
                }
                OpCode::IntDiv => {
                    let b = self.pop()?;
                    let a = self.pop()?;
                    let result = self.binary_int_div(&a, &b)?;
                    self.push(result);
                    return Ok(VMStatus::Continue);
                }
                OpCode::Mod => {
                    let b = self.pop()?;
                    let a = self.pop()?;
                    let result = self.binary_mod(&a, &b)?;
                    self.push(result);
                    return Ok(VMStatus::Continue);
                }
                OpCode::Pow => {
                    let b = self.pop()?;
                    let a = self.pop()?;
                    let result = self.binary_pow(&a, &b)?;
                    self.push(result);
                }
                OpCode::Negate => {
                    let value = self.pop()?;
                    match value {
                        Value::Number(n) => self.push(Value::Number(-n)),
                        _ => {
                            let error = self.runtime_error(
                                "Operand must be a number".to_string(),
                                line,
                            );
                            match self.handle_exception(error) {
                                Ok(()) => return Ok(VMStatus::Continue), // Исключение обработано, продолжаем выполнение
                                Err(e) => return Err(e), // Исключение не обработано
                            }
                        }
                    }
                }
                OpCode::Not => {
                    let value = self.pop()?;
                    self.push(Value::Bool(!value.is_truthy()));
                }
                OpCode::Or => {
                    let b = self.pop()?;
                    let a = self.pop()?;
                    // Если a истинно, возвращаем a, иначе возвращаем b
                    if a.is_truthy() {
                        self.push(a);
                    } else {
                        self.push(b);
                    }
                }
                OpCode::And => {
                    let b = self.pop()?;
                    let a = self.pop()?;
                    // Если a ложно, возвращаем a, иначе возвращаем b
                    if !a.is_truthy() {
                        self.push(a);
                    } else {
                        self.push(b);
                    }
                }
                OpCode::Equal => {
                    let b = self.pop()?;
                    let a = self.pop()?;
                    self.push(Value::Bool(a == b));
                }
                OpCode::NotEqual => {
                    let b = self.pop()?;
                    let a = self.pop()?;
                    self.push(Value::Bool(a != b));
                }
                OpCode::Greater => {
                    let b = self.pop()?;
                    let a = self.pop()?;
                    let result = self.binary_greater(&a, &b)?;
                    self.push(result);
                    return Ok(VMStatus::Continue);
                }
                OpCode::Less => {
                    let b = self.pop()?;
                    let a = self.pop()?;
                    let result = self.binary_less(&a, &b)?;
                    self.push(result);
                    return Ok(VMStatus::Continue);
                }
                OpCode::GreaterEqual => {
                    let b = self.pop()?;
                    let a = self.pop()?;
                    let result = self.binary_greater_equal(&a, &b)?;
                    self.push(result);
                    return Ok(VMStatus::Continue);
                }
                OpCode::LessEqual => {
                    let b = self.pop()?;
                    let a = self.pop()?;
                    let result = self.binary_less_equal(&a, &b)?;
                    self.push(result);
                }
                OpCode::In => {
                    let array = self.pop()?; // Правый операнд - массив
                    let value = self.pop()?; // Левый операнд - значение для поиска
                    
                    match array {
                        Value::Array(arr) => {
                            let arr_ref = arr.borrow();
                            let found = arr_ref.iter().any(|item| item == &value);
                            self.push(Value::Bool(found));
                        }
                        _ => {
                            let error = self.runtime_error(
                                "Right operand of 'in' operator must be an array".to_string(),
                                line,
                            );
                            match self.handle_exception(error) {
                                Ok(()) => return Ok(VMStatus::Continue),
                                Err(e) => return Err(e),
                            }
                        }
                    }
                }
                OpCode::Jump8(offset) => {
                    frame.ip = (frame.ip as i32 + offset as i32) as usize;
                }
                OpCode::Jump16(offset) => {
                    frame.ip = (frame.ip as i32 + offset as i32) as usize;
                }
                OpCode::Jump32(offset) => {
                    frame.ip = (frame.ip as i64 + offset as i64) as usize;
                }
                OpCode::JumpIfFalse8(offset) => {
                    let condition = self.pop()?;
                    let frame = self.frames.last_mut().unwrap();
                    if !condition.is_truthy() {
                        frame.ip = (frame.ip as i32 + offset as i32) as usize;
                    }
                }
                OpCode::JumpIfFalse16(offset) => {
                    let condition = self.pop()?;
                    let frame = self.frames.last_mut().unwrap();
                    if !condition.is_truthy() {
                        frame.ip = (frame.ip as i32 + offset as i32) as usize;
                    }
                }
                OpCode::JumpIfFalse32(offset) => {
                    let condition = self.pop()?;
                    let frame = self.frames.last_mut().unwrap();
                    if !condition.is_truthy() {
                        frame.ip = (frame.ip as i64 + offset as i64) as usize;
                    }
                }
                OpCode::JumpLabel(_) | OpCode::JumpIfFalseLabel(_) => {
                    return Err(crate::common::error::LangError::runtime_error(
                        "JumpLabel found in VM - compilation not finalized".to_string(),
                        frame.function.chunk.get_line(frame.ip),
                    ));
                }
                OpCode::Call(arity) => {
                    // Получаем функцию со стека
                    let function_value = self.pop()?;
                    match function_value {
                        Value::Function(function_index) => {
                            if function_index >= self.functions.len() {
                                let error = self.runtime_error(
                                    format!("Function index {} out of bounds", function_index),
                                    line,
                                );
                            match self.handle_exception(error) {
                                Ok(()) => return Ok(VMStatus::Continue),
                                Err(e) => return Err(e),
                            }
                            }
                            
                            let function = self.functions[function_index].clone();
                            
                            // Проверяем количество аргументов
                            if arity != function.arity {
                                let error = self.runtime_error(
                                    format!(
                                        "Expected {} arguments but got {}",
                                        function.arity, arity
                                    ),
                                    line,
                                );
                            match self.handle_exception(error) {
                                Ok(()) => return Ok(VMStatus::Continue),
                                Err(e) => return Err(e),
                            }
                            }
                            
                            // Собираем аргументы со стека (в обратном порядке, так как они были положены последними)
                            let mut args = Vec::new();
                            for _ in 0..arity {
                                args.push(self.pop()?);
                            }
                            args.reverse(); // Теперь args[0] - первый аргумент
                            
                            // Проверяем кэш, если функция помечена как кэшируемая
                            if function.is_cached {
                                use crate::bytecode::function::CacheKey;
                                
                                // Пытаемся создать ключ кэша
                                if let Some(cache_key) = CacheKey::new(&args) {
                                    // Получаем доступ к кэшу функции
                                    if let Some(cache_rc) = &function.cache {
                                        let cache = cache_rc.borrow();
                                        
                                        // Проверяем, есть ли результат в кэше
                                        if let Some(cached_result) = cache.map.get(&cache_key) {
                                            // Результат найден в кэше - возвращаем его без выполнения функции
                                            self.push(cached_result.clone());
                                            return Ok(VMStatus::Continue); // Пропускаем выполнение функции
                                        }
                                        
                                        // Результат не найден - освобождаем borrow и продолжим выполнение
                                        drop(cache);
                                        
                                        // Выполним функцию и сохраним результат в кэш
                                        // (продолжаем выполнение ниже)
                                    }
                                }
                                // Если ключ не удалось создать (не-hashable аргументы),
                                // просто выполняем функцию без кэширования
                            }
                            
                            // Создаем новый CallFrame
                            let stack_start = self.stack.len();
                            let mut new_frame = if function.is_cached {
                                // Сохраняем аргументы для кэширования
                                CallFrame::new_with_cache(function.clone(), stack_start, args.clone())
                            } else {
                                CallFrame::new(function.clone(), stack_start)
                            };
                            
                            // Копируем таблицу типов ошибок из chunk функции в VM
                            if !function.chunk.error_type_table.is_empty() {
                                self.error_type_table = function.chunk.error_type_table.clone();
                            }
                            
                            // Копируем захваченные переменные из родительских frames (если есть)
                            // Используем ancestor_depth для поиска переменной в правильном предке
                            if !self.frames.is_empty() && !function.captured_vars.is_empty() {
                                for captured_var in &function.captured_vars {
                                    // Убеждаемся, что слот существует в новом frame
                                    if captured_var.local_slot_index >= new_frame.slots.len() {
                                        new_frame.slots.resize(captured_var.local_slot_index + 1, Value::Null);
                                    }
                                    
                                    // Находим предка на нужной глубине
                                    // ancestor_depth = 0 означает ближайший родитель (последний frame в стеке)
                                    // ancestor_depth = 1 означает дедушку (предпоследний frame) и т.д.
                                    let ancestor_index = self.frames.len().saturating_sub(1 + captured_var.ancestor_depth);
                                    
                                    if ancestor_index < self.frames.len() {
                                        let ancestor_frame = &self.frames[ancestor_index];
                                        
                                        // Копируем значение из предка
                                        if captured_var.parent_slot_index < ancestor_frame.slots.len() {
                                            let captured_value = ancestor_frame.slots[captured_var.parent_slot_index].clone();
                                            
                                            new_frame.slots[captured_var.local_slot_index] = captured_value;
                                        } else {
                                            // Если слот не существует в предке, используем Null
                                            new_frame.slots[captured_var.local_slot_index] = Value::Null;
                                        }
                                    } else {
                                        // Если предок не существует, используем Null
                                        new_frame.slots[captured_var.local_slot_index] = Value::Null;
                                    }
                                }
                            }
                            
                            // Инициализируем параметры функции в slots (после захваченных переменных)
                            let param_start_index = function.captured_vars.len();
                            for (i, arg) in args.iter().enumerate() {
                                let slot_index = param_start_index + i;
                                if slot_index >= new_frame.slots.len() {
                                    new_frame.slots.resize(slot_index + 1, Value::Null);
                                }
                                new_frame.slots[slot_index] = arg.clone();
                            }
                            
                            // Добавляем новый frame
                            self.frames.push(new_frame);
                            return Ok(VMStatus::Continue);
                        }
                        Value::NativeFunction(native_index) => {
                            if native_index >= self.natives.len() {
                                let error = self.runtime_error(
                                    format!("Native function index {} out of bounds", native_index),
                                    line,
                                );
                            match self.handle_exception(error) {
                                Ok(()) => return Ok(VMStatus::Continue),
                                Err(e) => return Err(e),
                            }
                            }
                            
                            // Специальная обработка для методов тензора max_idx и min_idx
                            // Эти методы могут быть вызваны как tensor.max_idx() с arity=0,
                            // но тензор уже находится на стеке перед функцией
                            use crate::ml::natives;
                            let is_max_idx = std::ptr::eq(
                                self.natives[native_index] as *const (),
                                natives::native_max_idx as *const ()
                            );
                            let is_min_idx = std::ptr::eq(
                                self.natives[native_index] as *const (),
                                natives::native_min_idx as *const ()
                            );
                            
                            let mut args = Vec::new();
                            if (is_max_idx || is_min_idx) && arity == 0 {
                                // Для методов тензора с arity=0, используем тензор со стека как первый аргумент
                                // Тензор был помещен на стек перед функцией при доступе к свойству
                                // Важно: нужно удалить тензор со стека после использования
                                if let Some(Value::Tensor(tensor_rc)) = self.stack.last() {
                                    args.push(Value::Tensor(Rc::clone(tensor_rc)));
                                    // Удаляем тензор со стека, так как он был использован как аргумент
                                    self.pop()?;
                                } else {
                                    let error = self.runtime_error(
                                        "Tensor method called without tensor on stack".to_string(),
                                        line,
                                    );
                                    match self.handle_exception(error) {
                                        Ok(()) => return Ok(VMStatus::Continue),
                                        Err(e) => return Err(e),
                                    }
                                }
                            } else {
                                // Обычная обработка аргументов
                                for _ in 0..arity {
                                    args.push(self.pop()?);
                                }
                                args.reverse(); // Теперь args[0] - первый аргумент
                            }
                            
                            // Debug: log axis method calls
                            if native_index >= 2709 && native_index <= 2711 {
                                // axis methods: imshow (2709), set_title (2710), axis (2711)
                                let _method_name = match native_index {
                                    2709 => "imshow",
                                    2710 => "set_title",
                                    2711 => "axis",
                                    _ => "unknown",
                                };
                            }
                            
                            // Устанавливаем контекст VM для нативной функции
                            VM_CALL_CONTEXT.with(|ctx| {
                                *ctx.borrow_mut() = Some(self as *mut Vm);
                            });
                            
                            // Специальная проверка для range (принимает 1, 2 или 3 аргумента)
                            if native_index == 2 {
                                // range - индекс 2
                                if arity < 1 || arity > 3 {
                                    let error = self.runtime_error(
                                        format!("range() expects 1, 2, or 3 arguments, got {}", arity),
                                        line,
                                    );
                            match self.handle_exception(error) {
                                Ok(()) => return Ok(VMStatus::Continue),
                                Err(e) => return Err(e),
                            }
                                }
                                // Проверяем типы аргументов - все должны быть числами
                                for arg in &args {
                                    if !matches!(arg, Value::Number(_)) {
                                        let error = self.runtime_error(
                                            "range() arguments must be numbers".to_string(),
                                            line,
                                        );
                            match self.handle_exception(error) {
                                Ok(()) => return Ok(VMStatus::Continue),
                                Err(e) => return Err(e),
                            }
                                    }
                                }
                                // Проверяем, что step не равен 0 (если передан)
                                if arity == 3 {
                                    if let Value::Number(step) = &args[2] {
                                        if *step == 0.0 {
                                            let error = self.runtime_error(
                                                "range() step cannot be zero".to_string(),
                                                line,
                                            );
                            match self.handle_exception(error) {
                                Ok(()) => return Ok(VMStatus::Continue),
                                Err(e) => return Err(e),
                            }
                                        }
                                    }
                                }
                            }
                            
                            // Вызываем нативную функцию
                            let native_fn = self.natives[native_index];
                            let result = native_fn(&args);
                            
                            // Очищаем контекст VM после вызова нативной функции
                            VM_CALL_CONTEXT.with(|ctx| {
                                *ctx.borrow_mut() = None;
                            });
                            
                            // Если это relate(), получаем связи из thread-local storage
                            if native_index == 65 {
                                // relate() - индекс 65
                                use crate::vm::natives::take_relations;
                                let relations = take_relations();
                                
                                // Находим имена таблиц по указателям
                                for (table1_ptr, col1_name, table2_ptr, col2_name) in relations {
                                    let mut found_table1_name = None;
                                    let mut found_table2_name = None;
                                    
                                    // Ищем таблицы в глобальных переменных
                                    for (index, value) in self.globals.iter().enumerate() {
                                        if let Value::Table(table) = value {
                                            if Rc::as_ptr(table) == table1_ptr {
                                                if let Some(var_name) = self.explicit_global_names.get(&index) {
                                                    found_table1_name = Some(var_name.clone());
                                                }
                                            }
                                            if Rc::as_ptr(table) == table2_ptr {
                                                if let Some(var_name) = self.explicit_global_names.get(&index) {
                                                    found_table2_name = Some(var_name.clone());
                                                }
                                            }
                                        }
                                    }
                                    
                                    // Если нашли обе таблицы, сохраняем связь
                                    // relate(pk_table["pk_column"], fk_table["fk_column"])
                                    // Первый аргумент - первичный ключ (целевая таблица)
                                    // Второй аргумент - внешний ключ (таблица, которая ссылается)
                                    if let (Some(table1_name), Some(table2_name)) = (found_table1_name, found_table2_name) {
                                        self.explicit_relations.push(ExplicitRelation {
                                            source_table_name: table2_name, // Таблица с внешним ключом
                                            source_column_name: col2_name,  // Внешний ключ
                                            target_table_name: table1_name, // Таблица с первичным ключом
                                            target_column_name: col1_name,   // Первичный ключ
                                        });
                                    }
                                }
                            }
                            
                            // Если это primary_key(), получаем первичные ключи из thread-local storage
                            if native_index == 66 {
                                // primary_key() - индекс 66
                                use crate::vm::natives::take_primary_keys;
                                let primary_keys = take_primary_keys();
                                
                                // Находим имена таблиц по указателям
                                for (table_ptr, col_name) in primary_keys {
                                    let mut found_table_name = None;
                                    
                                    // Ищем таблицу в глобальных переменных
                                    for (index, value) in self.globals.iter().enumerate() {
                                        if let Value::Table(table) = value {
                                            if Rc::as_ptr(table) == table_ptr {
                                                if let Some(var_name) = self.explicit_global_names.get(&index) {
                                                    found_table_name = Some(var_name.clone());
                                                }
                                            }
                                        }
                                    }
                                    
                                    // Если нашли таблицу, сохраняем первичный ключ
                                    if let Some(table_name) = found_table_name {
                                        self.explicit_primary_keys.push(ExplicitPrimaryKey {
                                            table_name,
                                            column_name: col_name,
                                        });
                                    }
                                }
                            }
                            
                            // Проверяем, не было ли ошибки в нативной функции
                            use crate::websocket::take_native_error;
                            if let Some(error_msg) = take_native_error() {
                                // Check if this is a GPU fallback warning (not a real error)
                                if error_msg.contains("Falling back to CPU") || 
                                   error_msg.contains("not available") && error_msg.contains("GPU") {
                                    // Print as warning and continue execution
                                    eprintln!("⚠️  Предупреждение: {}", error_msg);
                                    // Don't create an error, just continue
                                } else {
                                    // Determine error type based on error message
                                    // ML functions (tensor, etc.) use ValueError
                                    let error_type = if error_msg.contains("ShapeError") || 
                                                        error_msg.contains("Shape mismatch") ||
                                                        error_msg.starts_with("ShapeError:") {
                                        crate::common::error::ErrorType::ValueError
                                    } else {
                                        // Default to IOError for file/path related errors
                                        crate::common::error::ErrorType::IOError
                                    };
                                    
                                    let error = self.runtime_error_with_type(
                                        error_msg,
                                        line,
                                        error_type,
                                    );
                                    match self.handle_exception(error) {
                                        Ok(()) => return Ok(VMStatus::Continue), // Исключение обработано
                                        Err(e) => return Err(e), // Исключение не обработано
                                    }
                                }
                            }
                            
                            // Помещаем результат на стек
                            self.push(result);
                            return Ok(VMStatus::Continue);
                        }
                        Value::Layer(layer_id) => {
                            // Layers can be called as functions: layer(input_tensor) -> output_tensor
                            if arity != 1 {
                                let error = self.runtime_error(
                                    format!("Layer call expects 1 argument (input tensor), got {}", arity),
                                    line,
                                );
                                match self.handle_exception(error) {
                                    Ok(()) => {
                                        self.push(Value::Null);
                                        return Ok(VMStatus::Continue);
                                    }
                                    Err(e) => return Err(e),
                                }
                            }
                            
                            // Get input tensor from stack
                            let input_value = self.pop()?;
                            
                            // Call native layer_call function directly
                            use crate::ml::natives;
                            let args = vec![Value::Layer(layer_id), input_value];
                            let result = natives::native_layer_call(&args);
                            
                            self.push(result);
                            return Ok(VMStatus::Continue);
                        }
                        Value::NeuralNetwork(_) | Value::LinearRegression(_) => {
                            // Models can be called as functions: model(input_tensor) -> output_tensor
                            if arity != 1 {
                                let error = self.runtime_error(
                                    format!("Model call expects 1 argument (input tensor), got {}", arity),
                                    line,
                                );
                                match self.handle_exception(error) {
                                    Ok(()) => {
                                        self.push(Value::Null);
                                        return Ok(VMStatus::Continue);
                                    }
                                    Err(e) => return Err(e),
                                }
                            }
                            
                            // Get input tensor from stack
                            let input_value = self.pop()?;
                            
                            // Call native_nn_forward function (it handles both NeuralNetwork and LinearRegression)
                            use crate::ml::natives;
                            let args = vec![function_value.clone(), input_value];
                            let result = natives::native_nn_forward(&args);
                            
                            self.push(result);
                            return Ok(VMStatus::Continue);
                        }
                        _ => {
                            // Try to provide more helpful error message
                            let error_msg = match &function_value {
                                Value::Null => "Cannot call null - function may not be imported or defined".to_string(),
                                _ => format!("Can only call functions, got: {:?}", std::mem::discriminant(&function_value)),
                            };
                            let error = self.runtime_error(error_msg, line);
                            match self.handle_exception(error) {
                                Ok(()) => {
                                    // Exception handled, but we need to push null to maintain stack consistency
                                    // since the caller expects a return value
                                    self.push(Value::Null);
                                    return Ok(VMStatus::Continue);
                                }
                                Err(e) => return Err(e),
                            }
                        }
                    }
                }
                OpCode::Return => {
                    // Получаем возвращаемое значение (если есть)
                    let return_value = if !self.stack.is_empty() {
                        self.pop().ok()
                    } else {
                        Some(Value::Null)
                    };
                    
                    let frames_count = self.frames.len();
                    if frames_count > 1 {
                        // Сохраняем результат в кэш, если функция кэшируемая
                        if let Some(frame) = self.frames.last() {
                            if frame.function.is_cached {
                                if let Some(ref cached_args) = frame.cached_args {
                                    use crate::bytecode::function::CacheKey;
                                    
                                    // Пытаемся создать ключ кэша
                                    if let Some(cache_key) = CacheKey::new(cached_args) {
                                        // Получаем доступ к кэшу функции
                                        if let Some(cache_rc) = &frame.function.cache {
                                            let mut cache = cache_rc.borrow_mut();
                                            
                                            // Сохраняем результат в кэш
                                            if let Some(ref result) = return_value {
                                                cache.map.insert(cache_key, result.clone());
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        
                        // Возврат из функции - удаляем текущий frame
                        self.frames.pop();
                        
                        // Помещаем возвращаемое значение на стек для вызывающей функции
                        if let Some(value) = return_value {
                            self.push(value);
                        }
                        // Продолжаем выполнение вызывающей функции
                        return Ok(VMStatus::Continue);
                    } else {
                        // Возврат из главной функции - завершаем выполнение
                        // Возвращаем значение со стека, если есть
                        if let Some(value) = return_value {
                            return Ok(VMStatus::Return(value));
                        } else {
                            return Ok(VMStatus::Return(Value::Null));
                        }
                    }
                }
                OpCode::Pop => {
                    self.pop()?;
                }
                OpCode::MakeArray(count) => {
                    let mut elements = Vec::new();
                    for _ in 0..count {
                        elements.push(self.pop()?);
                    }
                    elements.reverse(); // Восстанавливаем правильный порядок
                    self.push(Value::Array(Rc::new(RefCell::new(elements))));
                }
                OpCode::MakeTuple(count) => {
                    let mut elements = Vec::new();
                    for _ in 0..count {
                        elements.push(self.pop()?);
                    }
                    elements.reverse(); // Восстанавливаем правильный порядок
                    self.push(Value::Tuple(Rc::new(RefCell::new(elements))));
                }
                OpCode::MakeArrayDynamic => {
                    // Размер массива находится на стеке
                    let count_value = self.pop()?;
                    let count = match count_value {
                        Value::Number(n) => {
                            let idx = n as i64;
                            if idx < 0 {
                                let error = self.runtime_error(
                                    "Array size must be non-negative".to_string(),
                                    line,
                                );
                            match self.handle_exception(error) {
                                Ok(()) => return Ok(VMStatus::Continue),
                                Err(e) => return Err(e),
                            }
                            }
                            idx as usize
                        }
                        _ => {
                            let error = self.runtime_error(
                                "Array size must be a number".to_string(),
                                line,
                            );
                            match self.handle_exception(error) {
                                Ok(()) => return Ok(VMStatus::Continue),
                                Err(e) => return Err(e),
                            }
                        }
                    };
                    
                    let mut elements = Vec::new();
                    for _ in 0..count {
                        elements.push(self.pop()?);
                    }
                    elements.reverse(); // Восстанавливаем правильный порядок
                    self.push(Value::Array(Rc::new(RefCell::new(elements))));
                }
                OpCode::GetArrayLength => {
                    let array = self.pop()?;
                    match array {
                        Value::Array(arr) => {
                            self.push(Value::Number(arr.borrow().len() as f64));
                        }
                        Value::ColumnReference { table, column_name } => {
                            let table_ref = table.borrow();
                            if let Some(column) = table_ref.get_column(&column_name) {
                                self.push(Value::Number(column.len() as f64));
                            } else {
                                let error = self.runtime_error(
                                    format!("Column '{}' not found", column_name),
                                    line,
                                );
                            match self.handle_exception(error) {
                                Ok(()) => return Ok(VMStatus::Continue),
                                Err(e) => return Err(e),
                            }
                            }
                        }
                        Value::Dataset(dataset) => {
                            let batch_size = dataset.borrow().batch_size();
                            self.push(Value::Number(batch_size as f64));
                        }
                        _ => {
                            let error = self.runtime_error(
                                "Expected array, column reference, or dataset for GetArrayLength".to_string(),
                                line,
                            );
                            match self.handle_exception(error) {
                                Ok(()) => return Ok(VMStatus::Continue), // Исключение обработано, продолжаем выполнение
                                Err(e) => return Err(e), // Исключение не обработано
                            }
                        }
                    }
                }
                OpCode::GetArrayElement => {
                    let index_value = self.pop()?;
                    let container = self.pop()?;
                    
                    match container {
                        Value::Array(arr) => {
                            let index = match index_value {
                                Value::Number(n) => {
                                    let idx = n as i64;
                                    if idx < 0 {
                                        let error = self.runtime_error(
                                            "Array index must be non-negative".to_string(),
                                            line,
                                        );
                            match self.handle_exception(error) {
                                Ok(()) => return Ok(VMStatus::Continue),
                                Err(e) => return Err(e),
                            }
                                    }
                                    idx as usize
                                }
                                _ => {
                                    let error = self.runtime_error(
                                        "Array index must be a number".to_string(),
                                        line,
                                    );
                            match self.handle_exception(error) {
                                Ok(()) => return Ok(VMStatus::Continue),
                                Err(e) => return Err(e),
                            }
                                }
                            };
                            
                            let arr_ref = arr.borrow();
                            if index >= arr_ref.len() {
                                let error = self.runtime_error_with_type(
                                    format!("Array index {} out of bounds (length: {})", index, arr_ref.len()),
                                    line,
                                    ErrorType::IndexError,
                                );
                            match self.handle_exception(error) {
                                Ok(()) => return Ok(VMStatus::Continue),
                                Err(e) => return Err(e),
                            }
                            }
                            // Для сложных типов (Array, Table, Object, Axis, etc.) возвращаем ссылку (shallow copy Rc)
                            // Для простых типов клонируем значение
                            let element = &arr_ref[index];
                            let value = match element {
                                Value::Array(arr_rc) => Value::Array(Rc::clone(arr_rc)),
                                Value::Table(table_rc) => Value::Table(Rc::clone(table_rc)),
                                Value::Axis(axis_rc) => Value::Axis(Rc::clone(axis_rc)), // Clone Rc, not the Axis itself
                                Value::Figure(fig_rc) => Value::Figure(Rc::clone(fig_rc)),
                                Value::Image(img_rc) => Value::Image(Rc::clone(img_rc)),
                                Value::Window(handle) => Value::Window(*handle), // PlotWindowHandle is Copy
                                Value::Tensor(tensor_rc) => Value::Tensor(Rc::clone(tensor_rc)),
                                Value::Object(_) => element.clone(), // Object uses HashMap, clone is needed
                                _ => element.clone(), // Простые типы клонируем
                            };
                            self.push(value);
                            return Ok(VMStatus::Continue);
                        }
                        Value::Tuple(tuple) => {
                            let index = match index_value {
                                Value::Number(n) => {
                                    let idx = n as i64;
                                    if idx < 0 {
                                        let error = self.runtime_error(
                                            "Tuple index must be non-negative".to_string(),
                                            line,
                                        );
                            match self.handle_exception(error) {
                                Ok(()) => return Ok(VMStatus::Continue),
                                Err(e) => return Err(e),
                            }
                                    }
                                    idx as usize
                                }
                                _ => {
                                    let error = self.runtime_error(
                                        "Tuple index must be a number".to_string(),
                                        line,
                                    );
                            match self.handle_exception(error) {
                                Ok(()) => return Ok(VMStatus::Continue),
                                Err(e) => return Err(e),
                            }
                                }
                            };
                            
                            let tuple_ref = tuple.borrow();
                            if index >= tuple_ref.len() {
                                let error = self.runtime_error_with_type(
                                    format!("Tuple index {} out of bounds (length: {})", index, tuple_ref.len()),
                                    line,
                                    ErrorType::IndexError,
                                );
                            match self.handle_exception(error) {
                                Ok(()) => return Ok(VMStatus::Continue),
                                Err(e) => return Err(e),
                            }
                            }
                            // Для сложных типов возвращаем ссылку, для простых клонируем
                            let element = &tuple_ref[index];
                            let value = match element {
                                Value::Array(arr_rc) => Value::Array(Rc::clone(arr_rc)),
                                Value::Tuple(tuple_rc) => Value::Tuple(Rc::clone(tuple_rc)),
                                Value::Table(table_rc) => Value::Table(Rc::clone(table_rc)),
                                Value::Object(_) => element.clone(),
                                _ => element.clone(),
                            };
                            self.push(value);
                            return Ok(VMStatus::Continue);
                        }
                        Value::Table(table) => {
                            // Доступ к колонке таблицы по имени или строке по индексу
                            match index_value {
                                Value::String(property) => {
                                    let table_ref = table.borrow();
                                    
                                    // Специальные свойства таблицы
                                    if property == "rows" {
                                        // Возвращаем массив строк (каждая строка - массив значений)
                                        let rows: Vec<Value> = table_ref.rows.iter()
                                            .map(|row| {
                                                Value::Array(Rc::new(RefCell::new(row.clone())))
                                            })
                                            .collect();
                                        self.push(Value::Array(Rc::new(RefCell::new(rows))));
                                    } else if property == "columns" {
                                        // Возвращаем массив имен колонок (заголовки)
                                        let columns: Vec<Value> = table_ref.headers.iter()
                                            .map(|header| Value::String(header.clone()))
                                            .collect();
                                        self.push(Value::Array(Rc::new(RefCell::new(columns))));
                                    } else {
                                        // Доступ к колонке по имени
                                        if table_ref.get_column(&property).is_some() {
                                            // Возвращаем ColumnReference для использования в relate()
                                            self.push(Value::ColumnReference {
                                                table: table.clone(),
                                                column_name: property,
                                            });
                                        } else {
                                            let error = self.runtime_error_with_type(
                                                format!("Column '{}' not found in table", property),
                                                line,
                                                ErrorType::KeyError,
                                            );
                            match self.handle_exception(error) {
                                Ok(()) => return Ok(VMStatus::Continue),
                                Err(e) => return Err(e),
                            }
                                        }
                                    }
                                }
                                Value::Number(n) => {
                                    // Доступ к строке по индексу
                                    let idx = n as i64;
                                    if idx < 0 {
                                        let error = self.runtime_error(
                                            "Table row index must be non-negative".to_string(),
                                            line,
                                        );
                            match self.handle_exception(error) {
                                Ok(()) => return Ok(VMStatus::Continue),
                                Err(e) => return Err(e),
                            }
                                    }
                                    let table_ref = table.borrow();
                                    if idx as usize >= table_ref.rows.len() {
                                        let error = self.runtime_error_with_type(
                                            format!("Row index {} out of bounds (length: {})", idx, table_ref.rows.len()),
                                            line,
                                            ErrorType::IndexError,
                                        );
                            match self.handle_exception(error) {
                                Ok(()) => return Ok(VMStatus::Continue),
                                Err(e) => return Err(e),
                            }
                                    }
                                    if let Some(row) = table_ref.get_row(idx as usize) {
                                        // Создаем словарь из строки таблицы
                                        use std::collections::HashMap;
                                        let mut row_dict = HashMap::new();
                                        for (i, header) in table_ref.headers.iter().enumerate() {
                                            if i < row.len() {
                                                row_dict.insert(header.clone(), row[i].clone());
                                            }
                                        }
                                        self.push(Value::Object(row_dict));
                                    } else {
                                        let error = self.runtime_error_with_type(
                                            format!("Row index {} out of bounds", idx),
                                            line,
                                            ErrorType::IndexError,
                                        );
                            match self.handle_exception(error) {
                                Ok(()) => return Ok(VMStatus::Continue),
                                Err(e) => return Err(e),
                            }
                                    }
                                }
                                _ => {
                                    let error = self.runtime_error(
                                        "Table index must be a string (column name) or number (row index)".to_string(),
                                        line,
                                    );
                            match self.handle_exception(error) {
                                Ok(()) => return Ok(VMStatus::Continue),
                                Err(e) => return Err(e),
                            }
                                }
                            }
                        }
                        Value::Object(map) => {
                            // Check if this is a layer accessor object (has __neural_network key)
                            if map.contains_key("__neural_network") {
                                // This is a layer accessor - handle indexing to get layers
                                match index_value {
                                    Value::Number(n) => {
                                        let idx = n as i64;
                                        if idx < 0 {
                                            let error = self.runtime_error(
                                                "Layer index must be non-negative".to_string(),
                                                line,
                                            );
                                            match self.handle_exception(error) {
                                                Ok(()) => return Ok(VMStatus::Continue),
                                                Err(e) => return Err(e),
                                            }
                                        }
                                        
                                        // Get the NeuralNetwork from the accessor
                                        if let Some(Value::NeuralNetwork(nn_rc)) = map.get("__neural_network") {
                                            // Call native_model_get_layer
                                            use crate::ml::natives;
                                            let args = vec![Value::NeuralNetwork(Rc::clone(nn_rc)), Value::Number(n)];
                                            let result = natives::native_model_get_layer(&args);
                                            self.push(result);
                                            return Ok(VMStatus::Continue);
                                        }
                                    }
                                    _ => {
                                        let error = self.runtime_error(
                                            "Layer accessor index must be a number".to_string(),
                                            line,
                                        );
                                        match self.handle_exception(error) {
                                            Ok(()) => return Ok(VMStatus::Continue),
                                            Err(e) => return Err(e),
                                        }
                                    }
                                }
                            }
                            
                            // Regular object access
                            match index_value {
                                Value::String(key) => {
                                    if let Some(value) = map.get(&key) {
                                        self.push(value.clone());
                                    } else {
                                        let error = self.runtime_error_with_type(
                                            format!("Key '{}' not found in object", key),
                                            line,
                                            ErrorType::KeyError,
                                        );
                            match self.handle_exception(error) {
                                Ok(()) => return Ok(VMStatus::Continue),
                                Err(e) => return Err(e),
                            }
                                    }
                                }
                                _ => {
                                    let error = self.runtime_error(
                                        "Object index must be a string".to_string(),
                                        line,
                                    );
                            match self.handle_exception(error) {
                                Ok(()) => return Ok(VMStatus::Continue),
                                Err(e) => return Err(e),
                            }
                                }
                            }
                        }
                        Value::Figure(figure_rc) => {
                            // Доступ к свойствам фигуры по строковому ключу
                            match index_value {
                                Value::String(key) => {
                                    match key.as_str() {
                                        "axes" => {
                                            // Возвращаем 2D массив осей
                                            let figure_ref = figure_rc.borrow();
                                            let mut axes_array = Vec::new();
                                            for row in &figure_ref.axes {
                                                let mut row_array = Vec::new();
                                                for axis in row {
                                                    row_array.push(Value::Axis(axis.clone()));
                                                }
                                                axes_array.push(Value::Array(Rc::new(RefCell::new(row_array))));
                                            }
                                            self.push(Value::Array(Rc::new(RefCell::new(axes_array))));
                                        }
                                        _ => {
                                            let error = self.runtime_error_with_type(
                                                format!("Figure has no property '{}'", key),
                                                line,
                                                ErrorType::KeyError,
                                            );
                                            match self.handle_exception(error) {
                                                Ok(()) => return Ok(VMStatus::Continue),
                                                Err(e) => return Err(e),
                                            }
                                        }
                                    }
                                }
                                _ => {
                                    let error = self.runtime_error(
                                        "Figure property access must use string key".to_string(),
                                        line,
                                    );
                                    match self.handle_exception(error) {
                                        Ok(()) => return Ok(VMStatus::Continue),
                                        Err(e) => return Err(e),
                                    }
                                }
                            }
                            return Ok(VMStatus::Continue);
                        }
                        Value::Axis(_axis_rc) => {
                            // Доступ к методам оси по строковому ключу
                            match index_value {
                                Value::String(key) => {
                                    // Find the native function index for axis methods
                                    // These are registered after plot functions (starting at plot_native_start + 9)
                                    // We need to find them dynamically
                                    let method_name = match key.as_str() {
                                        "imshow" => "imshow",
                                        "set_title" => "set_title",
                                        "axis" => "axis",
                                        _ => {
                                            let error = self.runtime_error_with_type(
                                                format!("Axis has no method '{}'", key),
                                                line,
                                                ErrorType::KeyError,
                                            );
                                            match self.handle_exception(error) {
                                                Ok(()) => return Ok(VMStatus::Continue),
                                                Err(e) => return Err(e),
                                            }
                                        }
                                    };
                                    
                                    // Get method index from plot object (stored during registration)
                                    let method_index = if let Some(plot_obj) = self.globals.iter().find(|v| {
                                        if let Value::Object(map) = v {
                                            map.contains_key("image")
                                        } else {
                                            false
                                        }
                                    }) {
                                        if let Value::Object(map) = plot_obj {
                                            let idx_key = match method_name {
                                                "imshow" => "__axis_imshow_idx",
                                                "set_title" => "__axis_set_title_idx",
                                                "axis" => "__axis_axis_idx",
                                                _ => {
                                                    let error = self.runtime_error(
                                                        format!("Axis method '{}' not found", key),
                                                        line,
                                                    );
                                                    match self.handle_exception(error) {
                                                        Ok(()) => return Ok(VMStatus::Continue),
                                                        Err(e) => return Err(e),
                                                    }
                                                }
                                            };
                                            if let Some(Value::Number(idx)) = map.get(idx_key) {
                                                *idx as usize
                                            } else {
                                                let error = self.runtime_error(
                                                    format!("Axis method '{}' not registered", key),
                                                    line,
                                                );
                                                match self.handle_exception(error) {
                                                    Ok(()) => return Ok(VMStatus::Continue),
                                                    Err(e) => return Err(e),
                                                }
                                            }
                                        } else {
                                            let error = self.runtime_error(
                                                "Plot object not found".to_string(),
                                                line,
                                            );
                                            match self.handle_exception(error) {
                                                Ok(()) => return Ok(VMStatus::Continue),
                                                Err(e) => return Err(e),
                                            }
                                        }
                                    } else {
                                        let error = self.runtime_error(
                                            "Plot module not found".to_string(),
                                            line,
                                        );
                                        match self.handle_exception(error) {
                                            Ok(()) => return Ok(VMStatus::Continue),
                                            Err(e) => return Err(e),
                                        }
                                    };
                                    // Return the native function
                                    // The compiler should arrange for axis to be passed as first argument
                                    self.push(Value::NativeFunction(method_index));
                                }
                                _ => {
                                    let error = self.runtime_error(
                                        "Axis property access must use string key".to_string(),
                                        line,
                                    );
                                    match self.handle_exception(error) {
                                        Ok(()) => return Ok(VMStatus::Continue),
                                        Err(e) => return Err(e),
                                    }
                                }
                            }
                            return Ok(VMStatus::Continue);
                        }
                        Value::Layer(_layer_id) => {
                            // Доступ к методам слоя по строковому ключу
                            match index_value {
                                Value::String(key) => {
                                    // Map method names to native function names in ml module
                                    let function_name = match key.as_str() {
                                        "freeze" => "layer_freeze",
                                        "unfreeze" => "layer_unfreeze",
                                        _ => {
                                            let error = self.runtime_error_with_type(
                                                format!("Layer has no method '{}'. Available methods: freeze, unfreeze", key),
                                                line,
                                                ErrorType::KeyError,
                                            );
                                            match self.handle_exception(error) {
                                                Ok(()) => return Ok(VMStatus::Continue),
                                                Err(e) => return Err(e),
                                            }
                                        }
                                    };
                                    
                                    // Get method index from ml object (stored during registration)
                                    let method_index = if let Some((&ml_idx, _)) = self.global_names.iter().find(|(_, name)| name.as_str() == "ml") {
                                        if ml_idx >= self.globals.len() {
                                            let error = self.runtime_error(
                                                "ML module not found in globals".to_string(),
                                                line,
                                            );
                                            match self.handle_exception(error) {
                                                Ok(()) => return Ok(VMStatus::Continue),
                                                Err(e) => return Err(e),
                                            }
                                        }
                                        
                                        match &self.globals[ml_idx] {
                                            Value::Object(map) => {
                                                match map.get(function_name) {
                                                    Some(Value::NativeFunction(idx)) => *idx,
                                                    _ => {
                                                        let error = self.runtime_error(
                                                            format!("Layer method '{}' not registered in ml module", key),
                                                            line,
                                                        );
                                                        match self.handle_exception(error) {
                                                            Ok(()) => return Ok(VMStatus::Continue),
                                                            Err(e) => return Err(e),
                                                        }
                                                    }
                                                }
                                            }
                                            _ => {
                                                let error = self.runtime_error(
                                                    "ML module is not an object".to_string(),
                                                    line,
                                                );
                                                match self.handle_exception(error) {
                                                    Ok(()) => return Ok(VMStatus::Continue),
                                                    Err(e) => return Err(e),
                                                }
                                            }
                                        }
                                    } else {
                                        let error = self.runtime_error(
                                            "ML module not found".to_string(),
                                            line,
                                        );
                                        match self.handle_exception(error) {
                                            Ok(()) => return Ok(VMStatus::Continue),
                                            Err(e) => return Err(e),
                                        }
                                    };
                                    
                                    // Return the native function
                                    // The compiler should arrange for layer to be passed as first argument
                                    self.push(Value::NativeFunction(method_index));
                                }
                                _ => {
                                    let error = self.runtime_error(
                                        "Layer property access must use string key".to_string(),
                                        line,
                                    );
                                    match self.handle_exception(error) {
                                        Ok(()) => return Ok(VMStatus::Continue),
                                        Err(e) => return Err(e),
                                    }
                                }
                            }
                            return Ok(VMStatus::Continue);
                        }
                        Value::ColumnReference { table, column_name } => {
                            // Доступ к элементу колонки по индексу (как массив)
                            let index = match index_value {
                                Value::Number(n) => {
                                    let idx = n as i64;
                                    if idx < 0 {
                                        let error = self.runtime_error(
                                            "Column index must be non-negative".to_string(),
                                            line,
                                        );
                            match self.handle_exception(error) {
                                Ok(()) => return Ok(VMStatus::Continue),
                                Err(e) => return Err(e),
                            }
                                    }
                                    idx as usize
                                }
                                _ => {
                                    let error = self.runtime_error(
                                        "Column index must be a number".to_string(),
                                        line,
                                    );
                            match self.handle_exception(error) {
                                Ok(()) => return Ok(VMStatus::Continue),
                                Err(e) => return Err(e),
                            }
                                }
                            };
                            
                            let table_ref = table.borrow();
                            if let Some(column) = table_ref.get_column(&column_name) {
                                if index >= column.len() {
                                    let error = self.runtime_error_with_type(
                                        format!("Column index {} out of bounds (length: {})", index, column.len()),
                                        line,
                                        ErrorType::IndexError,
                                    );
                            match self.handle_exception(error) {
                                Ok(()) => return Ok(VMStatus::Continue),
                                Err(e) => return Err(e),
                            }
                                }
                                self.push(column[index].clone());
                            } else {
                                let error = self.runtime_error_with_type(
                                    format!("Column '{}' not found", column_name),
                                    line,
                                    ErrorType::KeyError,
                                );
                            match self.handle_exception(error) {
                                Ok(()) => return Ok(VMStatus::Continue),
                                Err(e) => return Err(e),
                            }
                            }
                        }
                        Value::Path(path) => {
                            // Доступ к свойствам Path по строковому ключу
                            match index_value {
                                Value::String(property_name) => {
                                    match property_name.as_str() {
                                        "is_file" => {
                                            self.push(Value::Bool(path.is_file()));
                                        }
                                        "is_dir" => {
                                            self.push(Value::Bool(path.is_dir()));
                                        }
                                        "extension" => {
                                            if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
                                                self.push(Value::String(ext.to_string()));
                                            } else {
                                                self.push(Value::Null);
                                            }
                                        }
                                        "name" => {
                                            if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                                                self.push(Value::String(name.to_string()));
                                            } else {
                                                self.push(Value::Null);
                                            }
                                        }
                                        "parent" => {
                                            // Используем безопасную функцию для получения parent
                                            use crate::vm::natives::safe_path_parent;
                                            match safe_path_parent(&path) {
                                                Some(parent) => self.push(Value::Path(parent)),
                                                None => self.push(Value::Null),
                                            }
                                        }
                                        "exists" => {
                                            self.push(Value::Bool(path.exists()));
                                        }
                                        _ => {
                                            let error = self.runtime_error(
                                                format!("Property '{}' not found on Path", property_name),
                                                line,
                                            );
                            match self.handle_exception(error) {
                                Ok(()) => return Ok(VMStatus::Continue),
                                Err(e) => return Err(e),
                            }
                                        }
                                    }
                                }
                                _ => {
                                    let error = self.runtime_error(
                                        "Path property access requires string index".to_string(),
                                        line,
                                    );
                            match self.handle_exception(error) {
                                Ok(()) => return Ok(VMStatus::Continue),
                                Err(e) => return Err(e),
                            }
                                }
                            }
                        }
                        Value::Dataset(dataset) => {
                            let index = match index_value {
                                Value::Number(n) => {
                                    let idx = n as i64;
                                    if idx < 0 {
                                        let error = self.runtime_error(
                                            "Dataset index must be non-negative".to_string(),
                                            line,
                                        );
                            match self.handle_exception(error) {
                                Ok(()) => return Ok(VMStatus::Continue),
                                Err(e) => return Err(e),
                            }
                                    }
                                    idx as usize
                                }
                                _ => {
                                    let error = self.runtime_error(
                                        "Dataset index must be a number".to_string(),
                                        line,
                                    );
                            match self.handle_exception(error) {
                                Ok(()) => return Ok(VMStatus::Continue),
                                Err(e) => return Err(e),
                            }
                                }
                            };

                            let dataset_ref = dataset.borrow();
                            let batch_size = dataset_ref.batch_size();
                            
                            if index >= batch_size {
                                let error = self.runtime_error_with_type(
                                    format!("Dataset index {} out of bounds (length: {})", index, batch_size),
                                    line,
                                    ErrorType::IndexError,
                                );
                            match self.handle_exception(error) {
                                Ok(()) => return Ok(VMStatus::Continue),
                                Err(e) => return Err(e),
                            }
                            }

                            // Extract features for this sample
                            let num_features = dataset_ref.num_features();
                            let features_start = index * num_features;
                            let features_end = features_start + num_features;
                            let features_data: Vec<f32> = dataset_ref.features().data[features_start..features_end].to_vec();
                            let features_tensor = Tensor::new(features_data, vec![num_features])
                                .map_err(|e| self.runtime_error(format!("Failed to create features tensor: {}", e), line))?;

                            // Extract target for this sample
                            let num_targets = dataset_ref.num_targets();
                            let targets_start = index * num_targets;
                            let targets_end = targets_start + num_targets;
                            
                            // If target is a single value, return as Number; otherwise return as Tensor
                            let target_value = if num_targets == 1 {
                                Value::Number(dataset_ref.targets().data[targets_start] as f64)
                            } else {
                                let target_data: Vec<f32> = dataset_ref.targets().data[targets_start..targets_end].to_vec();
                                let target_tensor = Tensor::new(target_data, vec![num_targets])
                                    .map_err(|e| self.runtime_error(format!("Failed to create target tensor: {}", e), line))?;
                                Value::Tensor(Rc::new(RefCell::new(target_tensor)))
                            };

                            // Return [features, target] as array
                            let features_value = Value::Tensor(Rc::new(RefCell::new(features_tensor)));
                            let pair = vec![features_value, target_value];
                            self.push(Value::Array(Rc::new(RefCell::new(pair))));
                            return Ok(VMStatus::Continue);
                        }
                        Value::Tensor(tensor) => {
                            // Доступ к свойствам тензора по строковому ключу
                            match index_value {
                                Value::String(property_name) => {
                                    match property_name.as_str() {
                                        "shape" => {
                                            let tensor_ref = tensor.borrow();
                                            let shape_values: Vec<Value> = tensor_ref.shape.iter()
                                                .map(|&s| Value::Number(s as f64))
                                                .collect();
                                            self.push(Value::Array(Rc::new(RefCell::new(shape_values))));
                                        }
                                        "data" => {
                                            let tensor_ref = tensor.borrow();
                                            let data_values: Vec<Value> = tensor_ref.data.iter()
                                                .map(|&d| Value::Number(d as f64))
                                                .collect();
                                            self.push(Value::Array(Rc::new(RefCell::new(data_values))));
                                        }
                                        "max_idx" => {
                                            // Return a bound method: push tensor first, then function
                                            // When called, the function will receive tensor as first argument
                                            use crate::ml::natives;
                                            let max_idx_fn_ptr = natives::native_max_idx as *const ();
                                            let method_index = self.natives.iter().position(|&f| {
                                                let fn_ptr = f as *const ();
                                                std::ptr::eq(fn_ptr, max_idx_fn_ptr)
                                            });
                                            
                                            if let Some(idx) = method_index {
                                                // Push tensor onto stack first (will be used as first argument)
                                                self.push(Value::Tensor(Rc::clone(&tensor)));
                                                // Push native function
                                                self.push(Value::NativeFunction(idx));
                                            } else {
                                                let error = self.runtime_error(
                                                    "max_idx method not found".to_string(),
                                                    line,
                                                );
                                                match self.handle_exception(error) {
                                                    Ok(()) => return Ok(VMStatus::Continue),
                                                    Err(e) => return Err(e),
                                                }
                                            }
                                        }
                                        "min_idx" => {
                                            // Return a bound method: push tensor first, then function
                                            use crate::ml::natives;
                                            let min_idx_fn_ptr = natives::native_min_idx as *const ();
                                            let method_index = self.natives.iter().position(|&f| {
                                                let fn_ptr = f as *const ();
                                                std::ptr::eq(fn_ptr, min_idx_fn_ptr)
                                            });
                                            
                                            if let Some(idx) = method_index {
                                                // Push tensor onto stack first (will be used as first argument)
                                                self.push(Value::Tensor(Rc::clone(&tensor)));
                                                // Push native function
                                                self.push(Value::NativeFunction(idx));
                                            } else {
                                                let error = self.runtime_error(
                                                    "min_idx method not found".to_string(),
                                                    line,
                                                );
                                                match self.handle_exception(error) {
                                                    Ok(()) => return Ok(VMStatus::Continue),
                                                    Err(e) => return Err(e),
                                                }
                                            }
                                        }
                                        _ => {
                                            let error = self.runtime_error(
                                                format!("Property '{}' not found on Tensor. Available properties: 'shape', 'data', 'max_idx', 'min_idx'", property_name),
                                                line,
                                            );
                                            match self.handle_exception(error) {
                                                Ok(()) => return Ok(VMStatus::Continue),
                                                Err(e) => return Err(e),
                                            }
                                        }
                                    }
                                }
                                Value::Number(n) => {
                                    // Доступ к элементу тензора по индексу
                                    let idx = n as i64;
                                    if idx < 0 {
                                        let error = self.runtime_error(
                                            "Tensor index must be non-negative".to_string(),
                                            line,
                                        );
                                        match self.handle_exception(error) {
                                            Ok(()) => return Ok(VMStatus::Continue),
                                            Err(e) => return Err(e),
                                        }
                                    }
                                    let tensor_ref = tensor.borrow();
                                    let index = idx as usize;
                                    
                                    // For 1D tensors, return scalar (backward compatibility)
                                    if tensor_ref.ndim() == 1 {
                                        if index >= tensor_ref.shape[0] {
                                            let error = self.runtime_error_with_type(
                                                format!("Tensor index {} out of bounds (size: {})", index, tensor_ref.shape[0]),
                                                line,
                                                ErrorType::IndexError,
                                            );
                                            match self.handle_exception(error) {
                                                Ok(()) => return Ok(VMStatus::Continue),
                                                Err(e) => return Err(e),
                                            }
                                        }
                                        self.push(Value::Number(tensor_ref.data[index] as f64));
                                    } else {
                                        // For 2D+ tensors, return a slice (row) along first dimension
                                        match tensor_ref.get_row(index) {
                                            Ok(slice_tensor) => {
                                                self.push(Value::Tensor(Rc::new(RefCell::new(slice_tensor))));
                                            }
                                            Err(e) => {
                                                let error = self.runtime_error_with_type(
                                                    e,
                                                    line,
                                                    ErrorType::IndexError,
                                                );
                                                match self.handle_exception(error) {
                                                    Ok(()) => return Ok(VMStatus::Continue),
                                                    Err(e) => return Err(e),
                                                }
                                            }
                                        }
                                    }
                                }
                                _ => {
                                    let error = self.runtime_error(
                                        "Tensor property access requires string key (e.g., 'shape', 'data') or numeric index".to_string(),
                                        line,
                                    );
                                    match self.handle_exception(error) {
                                        Ok(()) => return Ok(VMStatus::Continue),
                                        Err(e) => return Err(e),
                                    }
                                }
                            }
                            return Ok(VMStatus::Continue);
                        }
                        Value::NeuralNetwork(nn_rc) => {
                            // Доступ к методам и свойствам нейронной сети по строковому ключу
                            match index_value {
                                Value::String(key) => {
                                    // Check if this is the "layers" property
                                    if key == "layers" {
                                        // Return a special object that can be indexed to get layers
                                        // This object stores a reference to the NeuralNetwork
                                        use std::collections::HashMap;
                                        let mut layer_accessor = HashMap::new();
                                        layer_accessor.insert("__neural_network".to_string(), Value::NeuralNetwork(Rc::clone(&nn_rc)));
                                        self.push(Value::Object(layer_accessor));
                                        return Ok(VMStatus::Continue);
                                    }
                                    
                                    // Map property names to native function names in ml module
                                    let function_name = match key.as_str() {
                                        "train" => "nn_train",
                                        "train_sh" => "nn_train_sh",
                                        "save" => "nn_save",
                                        "device" => "nn_set_device",
                                        "get_device" => "nn_get_device",
                                        _ => {
                                            let error = self.runtime_error_with_type(
                                                format!("NeuralNetwork has no method '{}'. Available methods: train, train_sh, save, device, get_device, layers", key),
                                                line,
                                                ErrorType::KeyError,
                                            );
                                            match self.handle_exception(error) {
                                                Ok(()) => return Ok(VMStatus::Continue),
                                                Err(e) => return Err(e),
                                            }
                                        }
                                    };
                                    
                                    // Get method index from ml object (stored during registration)
                                    // Find ml module using global_names
                                    let method_index = if let Some((&ml_idx, _)) = self.global_names.iter().find(|(_, name)| name.as_str() == "ml") {
                                        // Ensure globals vector is large enough
                                        if ml_idx >= self.globals.len() {
                                            let error = self.runtime_error(
                                                "ML module not found in globals".to_string(),
                                                line,
                                            );
                                            match self.handle_exception(error) {
                                                Ok(()) => return Ok(VMStatus::Continue),
                                                Err(e) => return Err(e),
                                            }
                                        }
                                        
                                        match &self.globals[ml_idx] {
                                            Value::Object(map) => {
                                                match map.get(function_name) {
                                                    Some(Value::NativeFunction(idx)) => *idx,
                                                    _ => {
                                                        let error = self.runtime_error(
                                                            format!("NeuralNetwork method '{}' not registered in ml module", key),
                                                            line,
                                                        );
                                                        match self.handle_exception(error) {
                                                            Ok(()) => return Ok(VMStatus::Continue),
                                                            Err(e) => return Err(e),
                                                        }
                                                    }
                                                }
                                            }
                                            _ => {
                                                let error = self.runtime_error(
                                                    "ML module is not an object".to_string(),
                                                    line,
                                                );
                                                match self.handle_exception(error) {
                                                    Ok(()) => return Ok(VMStatus::Continue),
                                                    Err(e) => return Err(e),
                                                }
                                            }
                                        }
                                    } else {
                                        let error = self.runtime_error(
                                            "ML module not found".to_string(),
                                            line,
                                        );
                                        match self.handle_exception(error) {
                                            Ok(()) => return Ok(VMStatus::Continue),
                                            Err(e) => return Err(e),
                                        }
                                    };
                                    // Return the native function
                                    // The compiler should arrange for neural network to be passed as first argument
                                    self.push(Value::NativeFunction(method_index));
                                }
                                _ => {
                                    let error = self.runtime_error(
                                        "NeuralNetwork property access must use string key".to_string(),
                                        line,
                                    );
                                    match self.handle_exception(error) {
                                        Ok(()) => return Ok(VMStatus::Continue),
                                        Err(e) => return Err(e),
                                    }
                                }
                            }
                            return Ok(VMStatus::Continue);
                        }
                        Value::Null => {
                            let error = self.runtime_error(
                                "Cannot access element of null value".to_string(),
                                line,
                            );
                            match self.handle_exception(error) {
                                Ok(()) => return Ok(VMStatus::Continue),
                                Err(e) => return Err(e),
                            }
                        }
                        _ => {
                            let error = self.runtime_error(
                                "Expected array, tuple, column reference, table, object, path, dataset, tensor, or neural network for GetArrayElement".to_string(),
                                line,
                            );
                            match self.handle_exception(error) {
                                Ok(()) => return Ok(VMStatus::Continue),
                                Err(e) => return Err(e),
                            }
                        }
                    }
                    return Ok(VMStatus::Continue);
                }
                OpCode::Clone => {
                    // Глубокое клонирование значения на стеке
                    let value = self.pop()?;
                    let cloned = value.clone(); // Используем реализованный Clone для Value
                    self.push(cloned);
                    return Ok(VMStatus::Continue);
                }
                OpCode::BeginTry(handler_index) => {
                    // Начало try блока - загружаем обработчик из chunk
                    let frame = self.frames.last().unwrap();
                    let chunk = &frame.function.chunk;
                    
                    // Загружаем информацию об обработчике из chunk
                    if handler_index < chunk.exception_handlers.len() {
                        let handler_info = &chunk.exception_handlers[handler_index];
                        
                        // Копируем таблицу типов ошибок в VM (если еще не скопирована)
                        if self.error_type_table.is_empty() {
                            self.error_type_table = chunk.error_type_table.clone();
                        }
                        
                        // Сохраняем текущую высоту стека
                        let stack_height = self.stack.len();
                        
                        // Создаем обработчик с информацией из chunk
                        let frame_index = self.frames.len() - 1;
                        let handler = ExceptionHandler {
                            catch_ips: handler_info.catch_ips.clone(),
                            error_types: handler_info.error_types.clone(),
                            error_var_slots: handler_info.error_var_slots.clone(),
                            else_ip: handler_info.else_ip,
                            stack_height,
                            had_error: false,
                            frame_index,
                        };
                        self.exception_handlers.push(handler);
                    } else {
                        // Если обработчик не найден, создаем пустой (fallback)
                        let stack_height = self.stack.len();
                        let frame_index = self.frames.len() - 1;
                        let handler = ExceptionHandler {
                            catch_ips: Vec::new(),
                            error_types: Vec::new(),
                            error_var_slots: Vec::new(),
                            else_ip: None,
                            stack_height,
                            had_error: false,
                            frame_index,
                        };
                        self.exception_handlers.push(handler);
                    }
                    return Ok(VMStatus::Continue);
                }
                OpCode::EndTry => {
                    // Конец try блока - если выполнение дошло сюда без ошибок
                    // Проверяем, была ли ошибка
                    if let Some(handler) = self.exception_handlers.last_mut() {
                        // Если не было ошибки и есть else блок, переходим к нему
                        if !handler.had_error {
                            if let Some(else_ip) = handler.else_ip {
                                let frame = self.frames.last_mut().unwrap();
                                frame.ip = else_ip;
                            }
                        }
                        // Удаляем обработчик из стека
                        self.exception_handlers.pop();
                    }
                    return Ok(VMStatus::Continue);
                }
                OpCode::Catch(_) => {
                    // Начало catch блока - этот опкод используется только для маркировки
                    // Реальная логика обработки выполняется в handle_exception()
                    // Здесь просто продолжаем выполнение
                    return Ok(VMStatus::Continue);
                }
                OpCode::EndCatch => {
                    // Конец catch блока - продолжаем выполнение после catch
                    // Обработчик будет удален при PopExceptionHandler
                    return Ok(VMStatus::Continue);
                }
                OpCode::Throw(_) => {
                    // Выбрасывание исключения
                    // Получаем значение со стека (сообщение об ошибке)
                    let error_value = self.pop()?;
                    
                    // Преобразуем значение в строку
                    let error_message = error_value.to_string();
                    
                    // Создаем LangError
                    let error = LangError::runtime_error(error_message, line);
                    
                    // Пытаемся найти обработчик исключения
                    match self.handle_exception(error) {
                        Ok(()) => {
                            // Обработчик найден, выполнение продолжается в catch блоке
                            // handle_exception уже настроил стек и фреймы
                        }
                        Err(e) => {
                            // Обработчик не найден - возвращаем ошибку (программа завершается)
                            return Err(e);
                        }
                    }
                }
                OpCode::PopExceptionHandler => {
                    // Удаление обработчика исключений со стека
                    self.exception_handlers.pop();
                    return Ok(VMStatus::Continue);
                }
            }
        
        Ok(VMStatus::Continue)
    }

    fn push(&mut self, value: Value) {
        self.stack.push(value);
    }

    fn pop(&mut self) -> Result<Value, LangError> {
        let line = if let Some(frame) = self.frames.last() {
            if frame.ip > 0 {
                frame.function.chunk.get_line(frame.ip - 1)
            } else {
                0
            }
        } else {
            0
        };
        self.stack.pop().ok_or_else(|| self.runtime_error(
            "Stack underflow".to_string(),
            line,
        ))
    }

    fn binary_add(&mut self, a: &Value, b: &Value) -> Result<Value, LangError> {
        let line = if let Some(frame) = self.frames.last() {
            if frame.ip > 0 {
                frame.function.chunk.get_line(frame.ip - 1)
            } else {
                0
            }
        } else {
            0
        };
        match (a, b) {
            (Value::Number(n1), Value::Number(n2)) => Ok(Value::Number(n1 + n2)),
            (Value::String(s1), Value::String(s2)) => Ok(Value::String(format!("{}{}", s1, s2))),
            (Value::String(s), Value::Number(n)) => Ok(Value::String(format!("{}{}", s, n))),
            (Value::Number(n), Value::String(s)) => Ok(Value::String(format!("{}{}", n, s))),
            (Value::Array(arr1), Value::Array(arr2)) => {
                // Array concatenation
                let mut result = arr1.borrow().clone();
                result.extend_from_slice(&arr2.borrow());
                Ok(Value::Array(Rc::new(RefCell::new(result))))
            },
            _ => {
                let error = self.runtime_error(
                    "Operands must be numbers or strings".to_string(),
                    line,
                );
                match self.handle_exception(error) {
                    Ok(()) => Ok(Value::Null),
                    Err(e) => Err(e),
                }
            }
        }
    }

    fn binary_sub(&mut self, a: &Value, b: &Value) -> Result<Value, LangError> {
        let line = if let Some(frame) = self.frames.last() {
            if frame.ip > 0 {
                frame.function.chunk.get_line(frame.ip - 1)
            } else {
                0
            }
        } else {
            0
        };
        match (a, b) {
            (Value::Number(n1), Value::Number(n2)) => Ok(Value::Number(n1 - n2)),
            _ => {
                let error = self.runtime_error(
                    "Operands must be numbers".to_string(),
                    line,
                );
                match self.handle_exception(error) {
                    Ok(()) => Ok(Value::Null),
                    Err(e) => Err(e),
                }
            }
        }
    }

    fn binary_mul(&mut self, a: &Value, b: &Value) -> Result<Value, LangError> {
        let line = if let Some(frame) = self.frames.last() {
            if frame.ip > 0 {
                frame.function.chunk.get_line(frame.ip - 1)
            } else {
                0
            }
        } else {
            0
        };
        match (a, b) {
            (Value::Number(n1), Value::Number(n2)) => Ok(Value::Number(n1 * n2)),
            (Value::String(s), Value::Number(n)) => {
                let count = *n as i64;
                if count <= 0 {
                    Ok(Value::String(String::new()))
                } else {
                    Ok(Value::String(s.repeat(count as usize)))
                }
            }
            (Value::Number(n), Value::String(s)) => {
                let count = *n as i64;
                if count <= 0 {
                    Ok(Value::String(String::new()))
                } else {
                    Ok(Value::String(s.repeat(count as usize)))
                }
            }
            _ => {
                let error = self.runtime_error(
                    "Operands must be numbers, or string and number for repetition".to_string(),
                    line,
                );
                match self.handle_exception(error) {
                    Ok(()) => Ok(Value::Null),
                    Err(e) => Err(e),
                }
            }
        }
    }

    fn binary_div(&mut self, a: &Value, b: &Value) -> Result<Value, LangError> {
        let line = if let Some(frame) = self.frames.last() {
            if frame.ip > 0 {
                frame.function.chunk.get_line(frame.ip - 1)
            } else {
                0
            }
        } else {
            0
        };
        match (a, b) {
            (Value::Number(n1), Value::Number(n2)) => {
                if *n2 == 0.0 {
                    let error = self.runtime_error(
                        "Division by zero".to_string(),
                        line,
                    );
                    match self.handle_exception(error) {
                        Ok(()) => {
                            // Исключение обработано, возвращаем Null как значение после обработки
                            Ok(Value::Null)
                        }
                        Err(e) => Err(e),
                    }
                } else {
                    Ok(Value::Number(n1 / n2))
                }
            }
            // Tensor / Number
            (Value::Tensor(t1), Value::Number(n2)) => {
                if *n2 == 0.0 {
                    let error = self.runtime_error(
                        "Division by zero".to_string(),
                        line,
                    );
                    match self.handle_exception(error) {
                        Ok(()) => Ok(Value::Null),
                        Err(e) => Err(e),
                    }
                } else {
                    match t1.borrow().div_scalar(*n2 as f32) {
                        Ok(result) => Ok(Value::Tensor(std::rc::Rc::new(std::cell::RefCell::new(result)))),
                        Err(e) => {
                            let error = self.runtime_error(e, line);
                            match self.handle_exception(error) {
                                Ok(()) => Ok(Value::Null),
                                Err(e) => Err(e),
                            }
                        }
                    }
                }
            }
            // Number / Tensor
            (Value::Number(n1), Value::Tensor(t2)) => {
                let tensor_ref = t2.borrow();
                match tensor_ref.to_cpu() {
                    Ok(cpu_tensor) => {
                        // Check for division by zero
                        if cpu_tensor.data.iter().any(|&x| x == 0.0) {
                            let error = self.runtime_error(
                                "Division by zero".to_string(),
                                line,
                            );
                            match self.handle_exception(error) {
                                Ok(()) => Ok(Value::Null),
                                Err(e) => Err(e),
                            }
                        } else {
                            // Create a tensor filled with n1 and divide element-wise
                            let scalar = *n1 as f32;
                            let data: Vec<f32> = cpu_tensor.data.iter().map(|&x| scalar / x).collect();
                            Ok(Value::Tensor(std::rc::Rc::new(std::cell::RefCell::new(crate::ml::tensor::Tensor {
                                data,
                                shape: cpu_tensor.shape.clone(),
                                device: crate::ml::device::Device::Cpu,
                                #[cfg(feature = "gpu")]
                                gpu_tensor: None,
                            }))))
                        }
                    }
                    Err(e) => {
                        let error = self.runtime_error(e, line);
                        match self.handle_exception(error) {
                            Ok(()) => Ok(Value::Null),
                            Err(e) => Err(e),
                        }
                    }
                }
            }
            // Tensor / Tensor
            (Value::Tensor(t1), Value::Tensor(t2)) => {
                match t1.borrow().div(&t2.borrow()) {
                    Ok(result) => Ok(Value::Tensor(std::rc::Rc::new(std::cell::RefCell::new(result)))),
                    Err(e) => {
                        let error = self.runtime_error(e, line);
                        match self.handle_exception(error) {
                            Ok(()) => Ok(Value::Null),
                            Err(e) => Err(e),
                        }
                    }
                }
            }
            // Конкатенация путей: Path / String -> Path
            (Value::Path(p), Value::String(s)) => {
                let mut new_path = p.clone();
                new_path.push(s);
                Ok(Value::Path(new_path))
            }
            // Конкатенация путей: String / String -> Path (если контекст предполагает путь)
            (Value::String(s1), Value::String(s2)) => {
                use std::path::PathBuf;
                let mut path = PathBuf::from(s1);
                path.push(s2);
                Ok(Value::Path(path))
            }
            _ => {
                let error = self.runtime_error(
                    "Operands must be numbers, tensors, or paths".to_string(),
                    line,
                );
                match self.handle_exception(error) {
                    Ok(()) => Ok(Value::Null),
                    Err(e) => Err(e),
                }
            }
        }
    }

    fn binary_int_div(&mut self, a: &Value, b: &Value) -> Result<Value, LangError> {
        let line = if let Some(frame) = self.frames.last() {
            if frame.ip > 0 {
                frame.function.chunk.get_line(frame.ip - 1)
            } else {
                0
            }
        } else {
            0
        };
        match (a, b) {
            (Value::Number(n1), Value::Number(n2)) => {
                if *n2 == 0.0 {
                    let error = self.runtime_error(
                        "Division by zero".to_string(),
                        line,
                    );
                    match self.handle_exception(error) {
                        Ok(()) => Ok(Value::Null),
                        Err(e) => Err(e),
                    }
                } else {
                    // Целочисленное деление: отбрасываем дробную часть
                    Ok(Value::Number((n1 / n2).floor()))
                }
            }
            _ => {
                let error = self.runtime_error(
                    "Operands must be numbers".to_string(),
                    line,
                );
                match self.handle_exception(error) {
                    Ok(()) => Ok(Value::Null),
                    Err(e) => Err(e),
                }
            }
        }
    }

    fn binary_mod(&mut self, a: &Value, b: &Value) -> Result<Value, LangError> {
        let line = if let Some(frame) = self.frames.last() {
            if frame.ip > 0 {
                frame.function.chunk.get_line(frame.ip - 1)
            } else {
                0
            }
        } else {
            0
        };
        match (a, b) {
            (Value::Number(n1), Value::Number(n2)) => {
                if *n2 == 0.0 {
                    let error = self.runtime_error(
                        "Modulo by zero".to_string(),
                        line,
                    );
                    match self.handle_exception(error) {
                        Ok(()) => Ok(Value::Null),
                        Err(e) => Err(e),
                    }
                } else {
                    Ok(Value::Number(n1 % n2))
                }
            }
            _ => {
                let error = self.runtime_error(
                    "Operands must be numbers".to_string(),
                    line,
                );
                match self.handle_exception(error) {
                    Ok(()) => Ok(Value::Null),
                    Err(e) => Err(e),
                }
            }
        }
    }

    fn binary_pow(&mut self, a: &Value, b: &Value) -> Result<Value, LangError> {
        let line = if let Some(frame) = self.frames.last() {
            if frame.ip > 0 {
                frame.function.chunk.get_line(frame.ip - 1)
            } else {
                0
            }
        } else {
            0
        };
        match (a, b) {
            (Value::Number(n1), Value::Number(n2)) => Ok(Value::Number(n1.powf(*n2))),
            _ => {
                let error = self.runtime_error(
                    "Operands must be numbers".to_string(),
                    line,
                );
                match self.handle_exception(error) {
                    Ok(()) => Ok(Value::Null),
                    Err(e) => Err(e),
                }
            }
        }
    }

    fn binary_greater(&mut self, a: &Value, b: &Value) -> Result<Value, LangError> {
        let line = if let Some(frame) = self.frames.last() {
            if frame.ip > 0 {
                frame.function.chunk.get_line(frame.ip - 1)
            } else {
                0
            }
        } else {
            0
        };
        match (a, b) {
            (Value::Number(n1), Value::Number(n2)) => Ok(Value::Bool(n1 > n2)),
            (Value::String(s1), Value::String(s2)) => Ok(Value::Bool(s1 > s2)),
            _ => {
                let error = self.runtime_error(
                    "Operands must be numbers or strings".to_string(),
                    line,
                );
                match self.handle_exception(error) {
                    Ok(()) => Ok(Value::Null),
                    Err(e) => Err(e),
                }
            }
        }
    }

    fn binary_less(&mut self, a: &Value, b: &Value) -> Result<Value, LangError> {
        let line = if let Some(frame) = self.frames.last() {
            if frame.ip > 0 {
                frame.function.chunk.get_line(frame.ip - 1)
            } else {
                0
            }
        } else {
            0
        };
        match (a, b) {
            (Value::Number(n1), Value::Number(n2)) => Ok(Value::Bool(n1 < n2)),
            (Value::String(s1), Value::String(s2)) => Ok(Value::Bool(s1 < s2)),
            _ => {
                let error = self.runtime_error(
                    "Operands must be numbers or strings".to_string(),
                    line,
                );
                match self.handle_exception(error) {
                    Ok(()) => Ok(Value::Null),
                    Err(e) => Err(e),
                }
            }
        }
    }

    fn binary_greater_equal(&mut self, a: &Value, b: &Value) -> Result<Value, LangError> {
        let line = if let Some(frame) = self.frames.last() {
            if frame.ip > 0 {
                frame.function.chunk.get_line(frame.ip - 1)
            } else {
                0
            }
        } else {
            0
        };
        match (a, b) {
            (Value::Number(n1), Value::Number(n2)) => Ok(Value::Bool(n1 >= n2)),
            (Value::String(s1), Value::String(s2)) => Ok(Value::Bool(s1 >= s2)),
            _ => {
                let error = self.runtime_error(
                    "Operands must be numbers or strings".to_string(),
                    line,
                );
                match self.handle_exception(error) {
                    Ok(()) => Ok(Value::Null),
                    Err(e) => Err(e),
                }
            }
        }
    }

    fn binary_less_equal(&mut self, a: &Value, b: &Value) -> Result<Value, LangError> {
        let line = if let Some(frame) = self.frames.last() {
            if frame.ip > 0 {
                frame.function.chunk.get_line(frame.ip - 1)
            } else {
                0
            }
        } else {
            0
        };
        match (a, b) {
            (Value::Number(n1), Value::Number(n2)) => Ok(Value::Bool(n1 <= n2)),
            (Value::String(s1), Value::String(s2)) => Ok(Value::Bool(s1 <= s2)),
            _ => {
                let error = self.runtime_error(
                    "Operands must be numbers or strings".to_string(),
                    line,
                );
                match self.handle_exception(error) {
                    Ok(()) => Ok(Value::Null),
                    Err(e) => Err(e),
                }
            }
        }
    }

    /// Check if a name is a known module name
    fn is_known_module(name: &str) -> bool {
        matches!(name, "ml" | "plot")
    }

    fn register_module(&mut self, module_name: &str) -> Result<(), LangError> {
        match module_name {
            "ml" => {
                use crate::ml::natives;
                use std::collections::HashMap;
                
                // Register ML native functions
                let ml_native_start = self.natives.len();
                self.natives.push(natives::native_tensor);
                self.natives.push(natives::native_shape);
                self.natives.push(natives::native_data);
                self.natives.push(natives::native_add);
                self.natives.push(natives::native_sub);
                self.natives.push(natives::native_mul);
                self.natives.push(natives::native_matmul);
                self.natives.push(natives::native_transpose);
                self.natives.push(natives::native_sum);
                self.natives.push(natives::native_mean);
                self.natives.push(natives::native_max_idx);
                self.natives.push(natives::native_min_idx);
                // Graph functions
                self.natives.push(natives::native_graph);
                self.natives.push(natives::native_graph_add_input);
                self.natives.push(natives::native_graph_add_op);
                self.natives.push(natives::native_graph_forward);
                self.natives.push(natives::native_graph_get_output);
                // Autograd functions
                self.natives.push(natives::native_graph_backward);
                self.natives.push(natives::native_graph_get_gradient);
                self.natives.push(natives::native_graph_zero_grad);
                self.natives.push(natives::native_graph_set_requires_grad);
                // Linear Regression functions
                self.natives.push(natives::native_linear_regression);
                self.natives.push(natives::native_lr_predict);
                self.natives.push(natives::native_lr_train);
                self.natives.push(natives::native_lr_evaluate);
                // Optimizer functions
                self.natives.push(natives::native_sgd);
                self.natives.push(natives::native_sgd_step);
                self.natives.push(natives::native_sgd_zero_grad);
                self.natives.push(natives::native_adam);
                self.natives.push(natives::native_adam_step);
                // Loss functions
                self.natives.push(natives::native_mse_loss);
                self.natives.push(natives::native_cross_entropy_loss);
                self.natives.push(natives::native_binary_cross_entropy_loss);
                self.natives.push(natives::native_mae_loss);
                self.natives.push(natives::native_huber_loss);
                self.natives.push(natives::native_hinge_loss);
                self.natives.push(natives::native_kl_divergence);
                self.natives.push(natives::native_smooth_l1_loss);
                // Dataset functions
                self.natives.push(natives::native_dataset);
                self.natives.push(natives::native_dataset_features);
                self.natives.push(natives::native_dataset_targets);
                self.natives.push(natives::native_onehot);
                self.natives.push(natives::native_load_mnist);
                // Layer functions
                self.natives.push(natives::native_linear_layer);
                self.natives.push(natives::native_relu_layer);
                self.natives.push(natives::native_softmax_layer);
                self.natives.push(natives::native_flatten_layer);
                self.natives.push(natives::native_layer_call);
                // Neural network functions
                self.natives.push(natives::native_sequential);
                self.natives.push(natives::native_sequential_add);
                self.natives.push(natives::native_neural_network);
                self.natives.push(natives::native_nn_forward);
                self.natives.push(natives::native_nn_train);
                self.natives.push(natives::native_nn_train_sh);
                self.natives.push(natives::native_nn_save);
                self.natives.push(natives::native_nn_load);
                self.natives.push(natives::native_categorical_cross_entropy_loss);
                self.natives.push(natives::native_ml_save_model);
                self.natives.push(natives::native_ml_load_model);
                // Device management functions
                self.natives.push(natives::native_ml_set_device);
                self.natives.push(natives::native_ml_get_device);
                self.natives.push(natives::native_nn_set_device);
                self.natives.push(natives::native_nn_get_device);
                self.natives.push(natives::native_ml_validate_model);
                self.natives.push(natives::native_ml_model_info);
                // Layer freeze/unfreeze functions
                self.natives.push(natives::native_model_get_layer);
                self.natives.push(natives::native_layer_freeze);
                self.natives.push(natives::native_layer_unfreeze);
                
                // Create ML module object with native function references
                let mut ml_object = HashMap::new();
                ml_object.insert("tensor".to_string(), Value::NativeFunction(ml_native_start + 0));
                ml_object.insert("shape".to_string(), Value::NativeFunction(ml_native_start + 1));
                ml_object.insert("data".to_string(), Value::NativeFunction(ml_native_start + 2));
                ml_object.insert("add".to_string(), Value::NativeFunction(ml_native_start + 3));
                ml_object.insert("sub".to_string(), Value::NativeFunction(ml_native_start + 4));
                ml_object.insert("mul".to_string(), Value::NativeFunction(ml_native_start + 5));
                ml_object.insert("matmul".to_string(), Value::NativeFunction(ml_native_start + 6));
                ml_object.insert("transpose".to_string(), Value::NativeFunction(ml_native_start + 7));
                ml_object.insert("sum".to_string(), Value::NativeFunction(ml_native_start + 8));
                ml_object.insert("mean".to_string(), Value::NativeFunction(ml_native_start + 9));
                ml_object.insert("max_idx".to_string(), Value::NativeFunction(ml_native_start + 10));
                ml_object.insert("min_idx".to_string(), Value::NativeFunction(ml_native_start + 11));
                // Graph functions
                ml_object.insert("graph".to_string(), Value::NativeFunction(ml_native_start + 12));
                ml_object.insert("graph_add_input".to_string(), Value::NativeFunction(ml_native_start + 13));
                ml_object.insert("graph_add_op".to_string(), Value::NativeFunction(ml_native_start + 14));
                ml_object.insert("graph_forward".to_string(), Value::NativeFunction(ml_native_start + 15));
                ml_object.insert("graph_get_output".to_string(), Value::NativeFunction(ml_native_start + 16));
                // Autograd functions
                ml_object.insert("graph_backward".to_string(), Value::NativeFunction(ml_native_start + 17));
                ml_object.insert("graph_get_gradient".to_string(), Value::NativeFunction(ml_native_start + 18));
                ml_object.insert("graph_zero_grad".to_string(), Value::NativeFunction(ml_native_start + 19));
                ml_object.insert("graph_set_requires_grad".to_string(), Value::NativeFunction(ml_native_start + 20));
                // Linear Regression functions
                ml_object.insert("linear_regression".to_string(), Value::NativeFunction(ml_native_start + 21));
                ml_object.insert("lr_predict".to_string(), Value::NativeFunction(ml_native_start + 22));
                ml_object.insert("lr_train".to_string(), Value::NativeFunction(ml_native_start + 23));
                ml_object.insert("lr_evaluate".to_string(), Value::NativeFunction(ml_native_start + 24));
                // Optimizer functions
                ml_object.insert("sgd".to_string(), Value::NativeFunction(ml_native_start + 25));
                ml_object.insert("sgd_step".to_string(), Value::NativeFunction(ml_native_start + 26));
                ml_object.insert("sgd_zero_grad".to_string(), Value::NativeFunction(ml_native_start + 27));
                // Loss functions
                ml_object.insert("mse_loss".to_string(), Value::NativeFunction(ml_native_start + 30));
                ml_object.insert("cross_entropy_loss".to_string(), Value::NativeFunction(ml_native_start + 31));
                ml_object.insert("binary_cross_entropy_loss".to_string(), Value::NativeFunction(ml_native_start + 32));
                ml_object.insert("mae_loss".to_string(), Value::NativeFunction(ml_native_start + 33));
                ml_object.insert("huber_loss".to_string(), Value::NativeFunction(ml_native_start + 34));
                ml_object.insert("hinge_loss".to_string(), Value::NativeFunction(ml_native_start + 35));
                ml_object.insert("kl_divergence".to_string(), Value::NativeFunction(ml_native_start + 36));
                ml_object.insert("smooth_l1_loss".to_string(), Value::NativeFunction(ml_native_start + 37));
                // Dataset functions
                ml_object.insert("dataset".to_string(), Value::NativeFunction(ml_native_start + 38));
                ml_object.insert("dataset_features".to_string(), Value::NativeFunction(ml_native_start + 39));
                ml_object.insert("dataset_targets".to_string(), Value::NativeFunction(ml_native_start + 40));
                ml_object.insert("onehot".to_string(), Value::NativeFunction(ml_native_start + 41));
                ml_object.insert("load_mnist".to_string(), Value::NativeFunction(ml_native_start + 42));
                // Layer object with all layer functions
                let mut layer_object = HashMap::new();
                layer_object.insert("linear".to_string(), Value::NativeFunction(ml_native_start + 43));
                layer_object.insert("relu".to_string(), Value::NativeFunction(ml_native_start + 44));
                layer_object.insert("softmax".to_string(), Value::NativeFunction(ml_native_start + 45));
                layer_object.insert("flatten".to_string(), Value::NativeFunction(ml_native_start + 46));
                ml_object.insert("layer".to_string(), Value::Object(layer_object));
                // Neural network functions
                ml_object.insert("sequential".to_string(), Value::NativeFunction(ml_native_start + 48));
                ml_object.insert("sequential_add".to_string(), Value::NativeFunction(ml_native_start + 49));
                ml_object.insert("neural_network".to_string(), Value::NativeFunction(ml_native_start + 50));
                ml_object.insert("nn_forward".to_string(), Value::NativeFunction(ml_native_start + 51));
                ml_object.insert("nn_train".to_string(), Value::NativeFunction(ml_native_start + 52));
                ml_object.insert("nn_train_sh".to_string(), Value::NativeFunction(ml_native_start + 53));
                ml_object.insert("nn_save".to_string(), Value::NativeFunction(ml_native_start + 54));
                ml_object.insert("nn_load".to_string(), Value::NativeFunction(ml_native_start + 55));
                ml_object.insert("categorical_cross_entropy_loss".to_string(), Value::NativeFunction(ml_native_start + 56));
                ml_object.insert("save_model".to_string(), Value::NativeFunction(ml_native_start + 57));
                ml_object.insert("load".to_string(), Value::NativeFunction(ml_native_start + 58));
                // Device management
                ml_object.insert("set_device".to_string(), Value::NativeFunction(ml_native_start + 59));
                ml_object.insert("get_device".to_string(), Value::NativeFunction(ml_native_start + 60));
                ml_object.insert("nn_set_device".to_string(), Value::NativeFunction(ml_native_start + 61));
                ml_object.insert("nn_get_device".to_string(), Value::NativeFunction(ml_native_start + 62));
                ml_object.insert("validate_model".to_string(), Value::NativeFunction(ml_native_start + 63));
                ml_object.insert("model_info".to_string(), Value::NativeFunction(ml_native_start + 64));
                // Layer freeze/unfreeze functions
                ml_object.insert("model_get_layer".to_string(), Value::NativeFunction(ml_native_start + 65));
                ml_object.insert("layer_freeze".to_string(), Value::NativeFunction(ml_native_start + 66));
                ml_object.insert("layer_unfreeze".to_string(), Value::NativeFunction(ml_native_start + 67));
                
                // Register ml as a global variable
                // First, check if "ml" is already in global_names (from compiler)
                let ml_index = if let Some((&idx, _)) = self.global_names.iter().find(|(_, name)| name.as_str() == "ml") {
                    // ml is already registered by compiler, use that index
                    // Make sure globals vector is large enough
                    if idx >= self.globals.len() {
                        self.globals.resize(idx + 1, Value::Null);
                    }
                    idx
                } else if let Some(idx) = self.globals.iter().position(|v| {
                    if let Value::Object(map) = v {
                        map.contains_key("tensor")
                    } else {
                        false
                    }
                }) {
                    // ml object already exists at this index
                    idx
                } else {
                    // Create new global index
                    let idx = self.globals.len();
                    // Push a placeholder, will be set below
                    self.globals.push(Value::Null);
                    self.global_names.insert(idx, "ml".to_string());
                    idx
                };
                
                // Store ml object in globals (always store the original, not a clone)
                // Ensure the vector is large enough
                if ml_index >= self.globals.len() {
                    self.globals.resize(ml_index + 1, Value::Null);
                }
                // Verify ml_object is not empty before storing
                if ml_object.is_empty() {
                    return Err(LangError::runtime_error(
                        "ML module object is empty - native functions not registered".to_string(),
                        0,
                    ));
                }
                // Store the module object
                self.globals[ml_index] = Value::Object(ml_object);
                
                // Verify it was stored correctly
                match &self.globals[ml_index] {
                    Value::Object(map) => {
                        if map.is_empty() {
                            return Err(LangError::runtime_error(
                                "ML module object stored but is empty".to_string(),
                                0,
                            ));
                        }
                    }
                    _ => {
                        return Err(LangError::runtime_error(
                            format!("ML module not stored as Object, found: {:?}", 
                                std::mem::discriminant(&self.globals[ml_index])),
                            0,
                        ));
                    }
                }
            }
            "plot" => {
                use crate::plot::natives;
                use std::collections::HashMap;
                
                // Register plot native functions
                let plot_native_start = self.natives.len();
                self.natives.push(natives::native_plot_image);
                self.natives.push(natives::native_plot_window);
                self.natives.push(natives::native_window_draw);
                self.natives.push(natives::native_plot_wait);
                self.natives.push(natives::native_plot_show);
                self.natives.push(natives::native_plot_show_grid);
                self.natives.push(natives::native_plot_subplots);
                self.natives.push(natives::native_plot_tight_layout);
                self.natives.push(natives::native_plot_show_figure);
                self.natives.push(natives::native_axis_imshow);
                self.natives.push(natives::native_axis_set_title);
                self.natives.push(natives::native_axis_axis);
                self.natives.push(natives::native_plot_xlabel);
                self.natives.push(natives::native_plot_ylabel);
                self.natives.push(natives::native_plot_line);
                self.natives.push(natives::native_plot_bar);
                self.natives.push(natives::native_plot_pie);
                self.natives.push(natives::native_plot_heatmap);
                
                // Create plot module object with native function references
                let mut plot_object = HashMap::new();
                plot_object.insert("image".to_string(), Value::NativeFunction(plot_native_start + 0));
                plot_object.insert("window".to_string(), Value::NativeFunction(plot_native_start + 1));
                plot_object.insert("draw".to_string(), Value::NativeFunction(plot_native_start + 2));
                plot_object.insert("wait".to_string(), Value::NativeFunction(plot_native_start + 3));
                plot_object.insert("show".to_string(), Value::NativeFunction(plot_native_start + 4));
                plot_object.insert("show_grid".to_string(), Value::NativeFunction(plot_native_start + 5));
                plot_object.insert("subplots".to_string(), Value::NativeFunction(plot_native_start + 6));
                plot_object.insert("tight_layout".to_string(), Value::NativeFunction(plot_native_start + 7));
                plot_object.insert("xlabel".to_string(), Value::NativeFunction(plot_native_start + 12));
                plot_object.insert("ylabel".to_string(), Value::NativeFunction(plot_native_start + 13));
                plot_object.insert("line".to_string(), Value::NativeFunction(plot_native_start + 14));
                plot_object.insert("bar".to_string(), Value::NativeFunction(plot_native_start + 15));
                plot_object.insert("pie".to_string(), Value::NativeFunction(plot_native_start + 16));
                plot_object.insert("heatmap".to_string(), Value::NativeFunction(plot_native_start + 17));
                // show_figure is handled by checking if argument is Figure in native_plot_show
                
                // Store axis method indices for later lookup
                // imshow = plot_native_start + 9, set_title = +10, axis = +11
                let axis_imshow_idx = plot_native_start + 9;
                let axis_set_title_idx = plot_native_start + 10;
                let axis_axis_idx = plot_native_start + 11;
                
                // Store in plot object for access (we'll use a special key)
                plot_object.insert("__axis_imshow_idx".to_string(), Value::Number(axis_imshow_idx as f64));
                plot_object.insert("__axis_set_title_idx".to_string(), Value::Number(axis_set_title_idx as f64));
                plot_object.insert("__axis_axis_idx".to_string(), Value::Number(axis_axis_idx as f64));
                
                // Register plot as a global variable
                let plot_index = if let Some((&idx, _)) = self.global_names.iter().find(|(_, name)| name.as_str() == "plot") {
                    if idx >= self.globals.len() {
                        self.globals.resize(idx + 1, Value::Null);
                    }
                    idx
                } else if let Some(idx) = self.globals.iter().position(|v| {
                    if let Value::Object(map) = v {
                        map.contains_key("image")
                    } else {
                        false
                    }
                }) {
                    idx
                } else {
                    let idx = self.globals.len();
                    self.globals.push(Value::Object(plot_object.clone()));
                    self.global_names.insert(idx, "plot".to_string());
                    idx
                };
                
                // Store plot object in globals
                self.globals[plot_index] = Value::Object(plot_object);
            }
            _ => {
                return Err(LangError::runtime_error(
                    format!("Unknown module: {}", module_name),
                    0,
                ));
            }
        }
        Ok(())
    }

    /// Получить доступ к глобальным переменным (для экспорта)
    pub fn get_globals(&self) -> &Vec<Value> {
        &self.globals
    }

    /// Получить доступ к именам глобальных переменных
    pub fn get_global_names(&self) -> &std::collections::HashMap<usize, String> {
        &self.global_names
    }

    /// Получить доступ к именам переменных, явно объявленных с ключевым словом 'global'
    pub fn get_explicit_global_names(&self) -> &std::collections::HashMap<usize, String> {
        &self.explicit_global_names
    }

    /// Добавить явную связь между колонками таблиц
    pub fn add_explicit_relation(&mut self, relation: ExplicitRelation) {
        self.explicit_relations.push(relation);
    }

    /// Получить все явные связи
    pub fn get_explicit_relations(&self) -> &Vec<ExplicitRelation> {
        &self.explicit_relations
    }

    /// Добавить явный первичный ключ таблицы
    pub fn add_explicit_primary_key(&mut self, primary_key: ExplicitPrimaryKey) {
        self.explicit_primary_keys.push(primary_key);
    }

    /// Получить явные первичные ключи таблиц
    pub fn get_explicit_primary_keys(&self) -> &Vec<ExplicitPrimaryKey> {
        &self.explicit_primary_keys
    }

    /// Вызвать пользовательскую функцию по индексу с заданными аргументами
    /// Используется нативными функциями для вызова пользовательских функций
    pub fn call_function_by_index(&mut self, function_index: usize, args: &[Value]) -> Result<Value, LangError> {
        if function_index >= self.functions.len() {
            return Err(LangError::runtime_error(
                format!("Function index {} out of bounds", function_index),
                0,
            ));
        }

        let function = self.functions[function_index].clone();

        // Проверяем количество аргументов
        if args.len() != function.arity {
            return Err(LangError::runtime_error(
                format!(
                    "Expected {} arguments but got {}",
                    function.arity, args.len()
                ),
                0,
            ));
        }

        // Проверяем кэш, если функция помечена как кэшируемая
        if function.is_cached {
            use crate::bytecode::function::CacheKey;
            
            if let Some(cache_key) = CacheKey::new(args) {
                if let Some(cache_rc) = &function.cache {
                    let cache = cache_rc.borrow();
                    if let Some(cached_result) = cache.map.get(&cache_key) {
                        return Ok(cached_result.clone());
                    }
                    drop(cache);
                }
            }
        }

        // Создаем новый CallFrame
        let stack_start = self.stack.len();
        let mut new_frame = if function.is_cached {
            CallFrame::new_with_cache(function.clone(), stack_start, args.to_vec())
        } else {
            CallFrame::new(function.clone(), stack_start)
        };

        // Копируем таблицу типов ошибок из chunk функции в VM
        if !function.chunk.error_type_table.is_empty() {
            self.error_type_table = function.chunk.error_type_table.clone();
        }

        // Копируем захваченные переменные из родительских frames (если есть)
        if !self.frames.is_empty() && !function.captured_vars.is_empty() {
            for captured_var in &function.captured_vars {
                if captured_var.local_slot_index >= new_frame.slots.len() {
                    new_frame.slots.resize(captured_var.local_slot_index + 1, Value::Null);
                }
                
                let ancestor_index = self.frames.len().saturating_sub(1 + captured_var.ancestor_depth);
                
                if ancestor_index < self.frames.len() {
                    let ancestor_frame = &self.frames[ancestor_index];
                    if captured_var.parent_slot_index < ancestor_frame.slots.len() {
                        let captured_value = ancestor_frame.slots[captured_var.parent_slot_index].clone();
                        new_frame.slots[captured_var.local_slot_index] = captured_value;
                    } else {
                        new_frame.slots[captured_var.local_slot_index] = Value::Null;
                    }
                } else {
                    new_frame.slots[captured_var.local_slot_index] = Value::Null;
                }
            }
        }

        // Инициализируем параметры функции в slots
        let param_start_index = function.captured_vars.len();
        for (i, arg) in args.iter().enumerate() {
            let slot_index = param_start_index + i;
            if slot_index >= new_frame.slots.len() {
                new_frame.slots.resize(slot_index + 1, Value::Null);
            }
            new_frame.slots[slot_index] = arg.clone();
        }

        // Добавляем новый frame
        self.frames.push(new_frame);

        // Выполняем функцию используя существующую логику выполнения VM
        // Сохраняем текущее состояние стека для восстановления после выполнения
        let initial_stack_size = self.stack.len();
        let initial_frames_count = self.frames.len();
        
        // Выполняем функцию до Return, используя step()
        loop {
            if self.frames.len() < initial_frames_count {
                // Функция завершилась (frame был удален через Return)
                break;
            }

            match self.step()? {
                VMStatus::Continue => {}
                VMStatus::Return(v) => {
                    // Функция вернула значение
                    return Ok(v);
                }
                VMStatus::FrameEnded => {
                    // Фрейм завершился без return
                    break;
                }
            }
        }

        // Если дошли сюда, значит функция завершилась без return
        // Возвращаем значение со стека, если оно было добавлено после initial_stack_size
        if self.stack.len() > initial_stack_size {
            Ok(self.pop().unwrap_or(Value::Null))
        } else {
            Ok(Value::Null)
        }
    }
}

