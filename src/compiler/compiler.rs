// Компилятор AST → Bytecode

use crate::parser::ast::{Expr, Stmt, Arg, UnpackPattern, ImportStmt, ImportItem};
use crate::bytecode::{Chunk, OpCode, Function, CapturedVar};
use crate::common::error::LangError;
use crate::common::value::Value;
use crate::lexer::TokenKind;
use crate::compiler::natives;
use crate::compiler::scope::ScopeManager;
use crate::compiler::labels::LabelManager;
use crate::compiler::context::{ExceptionHandler, LoopContext};
use crate::compiler::constant_fold;
use crate::compiler::expr;
use crate::compiler::stmt;
use crate::compiler::unpack;
use crate::compiler::closure;
use crate::compiler::args;

pub struct Compiler {
    chunk: Chunk,
    functions: Vec<Function>,
    function_names: Vec<String>, // Имена функций для поиска
    current_function: Option<usize>, // Индекс текущей компилируемой функции
    scope: ScopeManager, // Управление областями видимости
    labels: LabelManager, // Управление метками и jump-инструкциями
    current_line: usize, // Текущий номер строки (для отладки и ошибок)
    exception_handlers: Vec<ExceptionHandler>, // Стек обработчиков исключений
    error_type_table: Vec<String>, // Таблица типов ошибок для текущей функции
    loop_contexts: Vec<LoopContext>, // Стек контекстов циклов для break/continue
}

impl Compiler {
    pub fn new() -> Self {
        let mut compiler = Self {
            chunk: Chunk::new(),
            functions: Vec::new(),
            function_names: Vec::new(),
            current_function: None,
            scope: ScopeManager::new(),
            labels: LabelManager::new(),
            current_line: 0,
            exception_handlers: Vec::new(),
            error_type_table: Vec::new(),
            loop_contexts: Vec::new(),
        };
        compiler.register_natives();
        compiler
    }


    fn register_natives(&mut self) {
        natives::register_natives(&mut self.scope.globals);
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
        self.labels.stabilize_layout(&mut self.chunk, self.current_line)?;
        self.labels.finalize_jumps(&mut self.chunk, self.current_line)?;
        
        // Очищаем метки после финализации главного скрипта
        self.labels.clear();
        
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
                    let global_index = self.scope.globals.len();
                    self.scope.globals.insert(name.clone(), global_index);
                    
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


    fn compile_stmt(&mut self, stmt: &Stmt) -> Result<(), LangError> {
        self.compile_stmt_with_pop(stmt, true)
    }

    fn compile_stmt_with_pop(&mut self, stmt: &Stmt, pop_value: bool) -> Result<(), LangError> {
        let mut ctx = self.create_context();
        stmt::compile_stmt(&mut ctx, stmt, pop_value)
    }

    fn create_context(&mut self) -> crate::compiler::context::CompilationContext {
        crate::compiler::context::CompilationContext {
            chunk: &mut self.chunk,
            scope: &mut self.scope,
            labels: &mut self.labels,
            functions: &mut self.functions,
            function_names: &mut self.function_names,
            current_function: self.current_function,
            current_line: &mut self.current_line,
            exception_handlers: &mut self.exception_handlers,
            error_type_table: &mut self.error_type_table,
            loop_contexts: &mut self.loop_contexts,
        }
    }

    #[allow(dead_code)]
    fn compile_stmt_with_pop_old(&mut self, stmt: &Stmt, pop_value: bool) -> Result<(), LangError> {
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
                            if !self.scope.globals.contains_key(module) {
                                let global_index = self.scope.globals.len();
                                self.scope.globals.insert(module.clone(), global_index);
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
                        if !self.scope.globals.contains_key(module) {
                            let global_index = self.scope.globals.len();
                            self.scope.globals.insert(module.clone(), global_index);
                            self.chunk.global_names.insert(global_index, module.clone());
                        }
                        
                        // Register imported item names in globals for from-import
                        for item in items {
                            match item {
                                ImportItem::Named(name) => {
                                    if !self.scope.globals.contains_key(name) {
                                        let global_index = self.scope.globals.len();
                                        self.scope.globals.insert(name.clone(), global_index);
                                        self.chunk.global_names.insert(global_index, name.clone());
                                    }
                                }
                                ImportItem::Aliased { alias, .. } => {
                                    if !self.scope.globals.contains_key(alias) {
                                        let global_index = self.scope.globals.len();
                                        self.scope.globals.insert(alias.clone(), global_index);
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
                            let global_index = if let Some(&idx) = self.scope.globals.get(var_name) {
                                idx
                            } else {
                                let idx = self.scope.globals.len();
                                self.scope.globals.insert(var_name.clone(), idx);
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
                                if let Some(&global_index) = self.scope.globals.get(var_name) {
                                    // Глобальная переменная найдена - обновляем
                                    self.chunk.global_names.insert(global_index, var_name.clone());
                                    self.chunk.write_with_line(OpCode::StoreGlobal(global_index), *line);
                                } else {
                                    // Новая глобальная переменная на верхнем уровне
                                    let global_index = self.scope.globals.len();
                                    self.scope.globals.insert(var_name.clone(), global_index);
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
                        } else if let Some(&global_index) = self.scope.globals.get(last_name) {
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
                    let global_index = if let Some(&idx) = self.scope.globals.get(name) {
                        idx
                    } else {
                        let idx = self.scope.globals.len();
                        self.scope.globals.insert(name.clone(), idx);
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
                        if let Some(&global_index) = self.scope.globals.get(name) {
                            // Глобальная переменная уже существует - обновляем
                            // Сохраняем имя глобальной переменной для использования в JOIN
                            self.chunk.global_names.insert(global_index, name.clone());
                            self.chunk.write_with_line(OpCode::StoreGlobal(global_index), *line);
                        } else {
                            // Новая глобальная переменная на верхнем уровне
                            let global_index = self.scope.globals.len();
                            self.scope.globals.insert(name.clone(), global_index);
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
                let else_label = self.labels.create_label();
                let end_label = self.labels.create_label();
                
                // Jump if false к else (или end, если else нет)
                let target_label = if else_branch.is_some() { else_label } else { end_label };
                self.labels.emit_jump(&mut self.chunk, self.current_line,true, target_label)?;
                
                // Компилируем then ветку (с новой областью видимости)
                self.begin_scope();
                for (i, stmt) in then_branch.iter().enumerate() {
                    let is_last = i == then_branch.len() - 1;
                    self.compile_stmt_with_pop(stmt, !is_last || pop_value)?;
                }
                self.end_scope();
                
                // Jump к end после then
                self.labels.emit_jump(&mut self.chunk, self.current_line,false, end_label)?;
                
                // Помечаем метку else (если есть)
                if else_branch.is_some() {
                    self.labels.mark_label(else_label, self.chunk.code.len());
                
                // Компилируем else ветку (с новой областью видимости)
                    self.begin_scope();
                    for (i, stmt) in else_branch.as_ref().unwrap().iter().enumerate() {
                        let is_last = i == else_branch.as_ref().unwrap().len() - 1;
                        self.compile_stmt_with_pop(stmt, !is_last || pop_value)?;
                    }
                    self.end_scope();
                }
                
                // Помечаем метку end
                self.labels.mark_label(end_label, self.chunk.code.len());
            }
            Stmt::While { condition, body, line } => {
                self.current_line = *line;
                
                // Создаем метки для начала и конца цикла
                let loop_start_label = self.labels.create_label();
                let loop_end_label = self.labels.create_label();
                
                // Помечаем начало цикла
                self.labels.mark_label(loop_start_label, self.chunk.code.len());
                
                self.compile_expr(condition)?;
                // Jump if false к концу цикла
                self.labels.emit_jump(&mut self.chunk, self.current_line,true, loop_end_label)?;
                
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
                self.labels.emit_loop(&mut self.chunk, self.current_line,loop_start_label)?;
                
                // Помечаем конец цикла
                self.labels.mark_label(loop_end_label, self.chunk.code.len());
                
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
                    unpack::count_unpack_variables(pattern)
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
                    unpack::declare_unpack_pattern_variables(pattern, &mut self.scope, *line)?
                };
                
                // Создаем метки для цикла
                let loop_start_label = self.labels.create_label();
                let continue_label = self.labels.create_label();
                let loop_end_label = self.labels.create_label();
                
                // Помечаем начало цикла
                self.labels.mark_label(loop_start_label, self.chunk.code.len());
                
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
                self.labels.emit_jump(&mut self.chunk, self.current_line,true, loop_end_label)?;
                
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
                    unpack::compile_unpack_pattern(pattern, &var_locals, expected_count, &mut self.chunk, &mut self.scope, &mut self.labels, self.current_line, *line)?;
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
                self.labels.mark_label(continue_label, self.chunk.code.len());
                
                // Инкрементируем индекс
                self.chunk.write_with_line(OpCode::LoadLocal(index_local), *line);
                let one_index = self.chunk.add_constant(Value::Number(1.0));
                self.chunk.write_with_line(OpCode::Constant(one_index), *line);
                self.chunk.write_with_line(OpCode::Add, *line);
                self.chunk.write_with_line(OpCode::StoreLocal(index_local), *line);
                
                // Переход к началу цикла
                self.labels.emit_loop(&mut self.chunk, self.current_line,loop_start_label)?;
                
                // Помечаем конец цикла
                self.labels.mark_label(loop_end_label, self.chunk.code.len());
                
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
                        match constant_fold::evaluate_constant_expr(default_expr) {
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
                let parent_locals_snapshot: Vec<std::collections::HashMap<String, usize>> = self.scope.locals.iter()
                    .map(|scope| scope.clone())
                    .collect();
                
                // Компилируем тело функции в chunk функции
                let saved_chunk = std::mem::replace(&mut self.chunk, function.chunk.clone());
                let saved_exception_handlers = self.exception_handlers.clone();
                let saved_error_type_table = self.error_type_table.clone();
                let saved_function = self.current_function;
                let saved_local_count = self.scope.local_count;
                self.current_function = Some(function_index);
                self.scope.local_count = 0;
                // Очищаем обработчики и таблицу типов ошибок для новой функции
                self.exception_handlers.clear();
                self.error_type_table.clear();
                
                // Начинаем новую область видимости для функции
                self.begin_scope();
                
                // Находим переменные, которые используются в теле функции, но не объявлены в ней
                // (захваченные из родительских функций)
                let param_names: Vec<String> = params.iter().map(|p| p.name.clone()).collect();
                let current_scope = self.scope.locals.last().cloned().unwrap_or_default();
                let captured_vars = closure::find_captured_variables(body, &parent_locals_snapshot, &param_names, &current_scope);
        
                
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
                self.labels.stabilize_layout(&mut self.chunk, *line)?;
                self.labels.finalize_jumps(&mut self.chunk, *line)?;
                
                // Сохраняем скомпилированную функцию (обработчики уже сохранены в chunk при компиляции try/catch)
                let function_chunk = std::mem::replace(&mut self.chunk, saved_chunk);
                function.chunk = function_chunk;
                function.captured_vars = captured_vars_info;
                self.functions[function_index] = function;
                
                // Восстанавливаем состояние компилятора
                self.exception_handlers = saved_exception_handlers;
                self.error_type_table = saved_error_type_table;
                self.current_function = saved_function;
                self.scope.local_count = saved_local_count;
                
                // Сохраняем функцию в глобальную таблицу (уже сделано в первом проходе)
                let global_index = *self.scope.globals.get(name).unwrap();
                
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
                self.labels.emit_jump(&mut self.chunk, self.current_line,false, break_label)?;
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
                self.labels.emit_jump(&mut self.chunk, self.current_line,false, continue_label)?;
            }
            Stmt::Throw { value, line } => {
                self.current_line = *line;
                // Компилируем выражение (оно оставит значение на стеке)
                self.compile_expr(value)?;
                // Генерируем Throw опкод (None означает RuntimeError без конкретного типа)
                self.chunk.write_with_line(OpCode::Throw(None), *line);
            }
            Stmt::Try { .. } => {
                // Обрабатывается в compile_stmt_with_pop через stmt::compile_stmt
                unreachable!("Try statement should be handled by stmt::compile_stmt")
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
        args::resolve_function_args(function_name, args, function_info, line)
    }

    fn compile_expr(&mut self, expr: &Expr) -> Result<(), LangError> {
        let expr_line = expr.line();
        self.current_line = expr_line;
        
        // Оптимизация: вычисляем константные выражения во время компиляции
        if let Some(constant_value) = constant_fold::evaluate_constant_expr(expr)? {
            let constant_index = self.chunk.add_constant(constant_value);
            self.chunk.write_with_line(OpCode::Constant(constant_index), expr_line);
            return Ok(());
        }
        
        let mut ctx = self.create_context();
        expr::compile_expr(&mut ctx, expr)
    }

    #[allow(dead_code)]
    fn compile_expr_old(&mut self, expr: &Expr) -> Result<(), LangError> {
        let expr_line = expr.line();
        self.current_line = expr_line;
        
        // Оптимизация: вычисляем константные выражения во время компиляции
        if let Some(constant_value) = constant_fold::evaluate_constant_expr(expr)? {
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
                } else if let Some(&global_index) = self.scope.globals.get(name) {
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
                        let global_index = self.scope.globals.len();
                        self.scope.globals.insert(name.clone(), global_index);
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
                        if let Some(&global_index) = self.scope.globals.get(name) {
                            // Глобальная переменная найдена - обновляем
                            self.chunk.global_names.insert(global_index, name.clone());
                            self.chunk.write_with_line(OpCode::StoreGlobal(global_index), *line);
                        } else {
                            // Новая глобальная переменная на верхнем уровне
                            let global_index = self.scope.globals.len();
                            self.scope.globals.insert(name.clone(), global_index);
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
                    } else if let Some(&global_index) = self.scope.globals.get(last_name) {
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
                } else if let Some(&global_index) = self.scope.globals.get(name) {
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
                } else if let Some(&global_index) = self.scope.globals.get(name) {
                    // Глобальная переменная или функция
                    self.chunk.write_with_line(OpCode::LoadGlobal(global_index), *line);
                } else {
                    // Переменная не найдена - создаем новый глобальный индекс
                    // Это позволит проверить переменную во время выполнения
                    let global_index = self.scope.globals.len();
                    self.scope.globals.insert(name.clone(), global_index);
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
                
                // Загружаем функцию: сначала проверяем локальные переменные,
                // затем пользовательские функции (они имеют приоритет над встроенными),
                // затем глобальные переменные (встроенные функции)
                if let Some(local_index) = self.resolve_local(name) {
                    // Локальная переменная содержит функцию
                    self.chunk.write_with_line(OpCode::LoadLocal(local_index), self.current_line);
                } else if let Some(function_index) = self.function_names.iter().position(|n| n == name) {
                    // Пользовательская функция найдена - она имеет приоритет над встроенными
                    let constant_index = self.chunk.add_constant(Value::Function(function_index));
                    self.chunk.write_with_line(OpCode::Constant(constant_index), self.current_line);
                } else if let Some(&global_index) = self.scope.globals.get(name) {
                    // Глобальная переменная содержит функцию (встроенная функция)
                    self.chunk.write_with_line(OpCode::LoadGlobal(global_index), self.current_line);
                } else {
                    // Функция не найдена - это может быть функция, импортированная через import *
                    // Регистрируем её как глобальную переменную для разрешения во время выполнения
                    let global_index = if let Some(&idx) = self.scope.globals.get(name) {
                        idx
                    } else {
                        let idx = self.scope.globals.len();
                        self.scope.globals.insert(name.clone(), idx);
                        self.chunk.global_names.insert(idx, name.clone());
                        idx
                    };
                    self.chunk.write_with_line(OpCode::LoadGlobal(global_index), self.current_line);
                }
                
                // Вызываем функцию с количеством аргументов
                self.chunk.write_with_line(OpCode::Call(processed_args.len()), *line);
                
                // Если нужно присвоить результат обратно в переменную
                if let Some(var_name) = var_name_to_assign {
                    // Определяем, глобальная или локальная переменная
                    let is_local = !self.scope.locals.is_empty() && self.resolve_local(&var_name).is_some();
                    
                    if is_local {
                        // Локальная переменная
                        let local_index = self.resolve_local(&var_name).unwrap();
                        self.chunk.write_with_line(OpCode::StoreLocal(local_index), *line);
                        // Загружаем значение обратно на стек, чтобы выражение возвращало результат
                        self.chunk.write_with_line(OpCode::LoadLocal(local_index), *line);
                    } else {
                        // Глобальная переменная
                        let global_index = if let Some(&idx) = self.scope.globals.get(&var_name) {
                            idx
                        } else {
                            let idx = self.scope.globals.len();
                            self.scope.globals.insert(var_name.clone(), idx);
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
            Expr::ObjectLiteral { pairs, line } => {
                // Компилируем пары (ключ, значение) в обратном порядке
                // На стеке будут: [key1, value1, key2, value2, ...]
                for (key, value) in pairs.iter().rev() {
                    // Сначала добавляем ключ как строку
                    let key_index = self.chunk.add_constant(Value::String(key.clone()));
                    self.chunk.write_with_line(OpCode::Constant(key_index), *line);
                    // Затем компилируем значение
                    self.compile_expr(value)?;
                }
                // Создаем объект из пар на стеке
                let pair_count = pairs.len();
                self.chunk.write_with_line(OpCode::MakeObject(pair_count), *line);
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
                    if let Some(&function_index) = self.scope.globals.get("table_suffixes") {
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
                    if let Some(&function_index) = self.scope.globals.get(function_name) {
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
                        if let Some(&ml_index) = self.scope.globals.get("ml") {
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


    fn begin_scope(&mut self) {
        self.scope.begin_scope();
    }

    fn end_scope(&mut self) {
        self.scope.end_scope();
    }

    fn declare_local(&mut self, name: &str) -> usize {
        self.scope.declare_local(name)
    }

    fn resolve_local(&self, name: &str) -> Option<usize> {
        self.scope.resolve_local(name)
    }

    


}

