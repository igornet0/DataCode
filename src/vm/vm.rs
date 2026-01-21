// Виртуальная машина

use crate::bytecode::Chunk;
use crate::common::{error::LangError, value::Value};
use crate::vm::frame::CallFrame;
use crate::vm::natives;
use crate::vm::types::{ExplicitRelation, ExplicitPrimaryKey, VMStatus};
use crate::vm::exceptions::ExceptionHandler;
use crate::vm::globals;
use crate::vm::calls;
use crate::vm::executor;
use std::cell::RefCell;

pub type NativeFn = fn(&[Value]) -> Value;

// Thread-local storage для хранения контекста VM во время вызова нативных функций
// Это позволяет нативным функциям вызывать пользовательские функции
thread_local! {
    pub(crate) static VM_CALL_CONTEXT: RefCell<Option<*mut Vm>> = RefCell::new(None);
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
        globals::register_native_globals(&mut self.globals, &self.global_names);
    }

    // Exception handling methods moved to exceptions.rs

    pub fn run(&mut self, chunk: &Chunk) -> Result<Value, LangError> {
        // Заполняем имена глобальных переменных из chunk
        globals::merge_global_names(
            &mut self.global_names,
            &mut self.explicit_global_names,
            &chunk.global_names,
            &chunk.explicit_global_names,
        );
        
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
        // Get raw pointer to self before any mutable borrows
        let vm_ptr = self as *mut Vm;
        
        // Get instruction and line from executor::step (which already increments IP)
        let (instruction, line) = {
            let frames_ref = &mut self.frames;
            match executor::step(frames_ref)? {
                Some((inst, ln)) => (inst, ln),
            None => return Ok(VMStatus::FrameEnded),
            }
        };
        
        // Execute the instruction (frames_ref is dropped, so we can borrow again)
        executor::execute_instruction(
            instruction,
                                line,
            &mut self.stack,
            &mut self.frames,
            &mut self.globals,
            &mut self.global_names,
            &self.explicit_global_names,
            &self.functions,
            &mut self.natives,
            &mut self.exception_handlers,
            &mut self.error_type_table,
            &mut self.explicit_relations,
            &mut self.explicit_primary_keys,
            &mut self.loaded_modules,
            vm_ptr,
        )
    }

    // execute_instruction moved to executor.rs
    // Stack operations moved to stack.rs
    // Binary and unary operations moved to operations.rs

    // Module registration methods moved to modules.rs

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
        // Setup function call (creates frame, handles cache, sets up captured variables)
        if let Some(cached_result) = calls::setup_function_call(
            function_index,
            args,
            &self.functions,
            &mut self.stack,
            &mut self.frames,
            &mut self.error_type_table,
        )? {
            return Ok(cached_result);
        }

        // Execute the function using step()
        let initial_stack_size = self.stack.len();
        let initial_frames_count = self.frames.len();
        
        loop {
            if self.frames.len() < initial_frames_count {
                break;
            }

            match self.step()? {
                VMStatus::Continue => {}
                VMStatus::Return(v) => {
                    return Ok(v);
                }
                VMStatus::FrameEnded => {
                    break;
                }
            }
        }

        // Return value from stack if any
        // Проверяем относительно текущего frame's stack_start (caller's frame)
        // после того как функция вернулась и её frame был удалён
        // Используем безопасное извлечение без вызова stack::pop, чтобы избежать
        // ошибки stack underflow, которая может быть неправильно обработана
        if let Some(frame) = self.frames.last() {
            if self.stack.len() > frame.stack_start {
                // Безопасно извлекаем значение напрямую, так как мы уже проверили
                // что стек не пуст относительно stack_start
                Ok(self.stack.pop().unwrap_or(Value::Null))
            } else {
                Ok(Value::Null)
            }
        } else {
            // Нет frame - проверяем относительно initial_stack_size
            if self.stack.len() > initial_stack_size {
                // Безопасно извлекаем значение напрямую
                Ok(self.stack.pop().unwrap_or(Value::Null))
            } else {
                Ok(Value::Null)
            }
        }
    }
}

