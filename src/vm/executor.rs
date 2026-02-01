// Opcode execution for VM

use crate::debug_println;
use crate::bytecode::OpCode;
use crate::common::{error::LangError, value::Value};
use crate::vm::types::VMStatus;
use crate::vm::frame::CallFrame;
use crate::vm::exceptions::ExceptionHandler;
use crate::vm::operations;
use crate::vm::stack;
use crate::vm::modules;
use crate::vm::vm::VM_CALL_CONTEXT;
use crate::ml::tensor::Tensor;
use crate::common::error::ErrorType;
use crate::vm::types::{ExplicitRelation, ExplicitPrimaryKey};
use std::rc::Rc;
use std::cell::RefCell;

/// Returns [class_name, superclass, ...] for VM protected access checks.
fn get_superclass_chain(
    globals: &[Value],
    global_names: &std::collections::HashMap<usize, String>,
    class_name: &str,
) -> Vec<String> {
    let mut chain = vec![class_name.to_string()];
    let mut current = class_name.to_string();
    loop {
        let super_name_opt = global_names
            .iter()
            .find(|(_, name)| name.as_str() == current)
            .and_then(|(idx, _)| globals.get(*idx))
            .and_then(|v| {
                if let Value::Object(rc) = v {
                    let map = rc.borrow();
                    map.get("__superclass").cloned()
                } else {
                    None
                }
            });
        let super_name = match super_name_opt {
            Some(Value::String(s)) => s,
            _ => break,
        };
        chain.push(super_name.clone());
        current = super_name;
    }
    chain
}

/// Execute one step of the VM - get next instruction and execute it
pub fn step(
    frames: &mut Vec<CallFrame>,
) -> Result<Option<(OpCode, usize)>, LangError> {
    loop {
        let frame = match frames.last_mut() {
            Some(f) => f,
            None => return Ok(None),
        };

        if frame.ip >= frame.function.chunk.code.len() {
            // Frame exhausted (e.g. empty method body); pop and continue with caller
            frames.pop();
            continue;
        }

        let ip = frame.ip;
        let instruction = frame.function.chunk.code[ip].clone();
        let line = frame.function.chunk.get_line(ip);
        frame.ip += 1;

        return Ok(Some((instruction, line)));
    }
}

/// Execute a single instruction
/// Returns VMStatus indicating what to do next
/// vm_ptr is used for VM_CALL_CONTEXT when calling native functions
pub fn execute_instruction(
    instruction: OpCode,
    line: usize,
    stack: &mut Vec<Value>,
    frames: &mut Vec<CallFrame>,
    globals: &mut Vec<Value>,
    global_names: &mut std::collections::HashMap<usize, String>,
    explicit_global_names: &std::collections::HashMap<usize, String>,
    functions: &[crate::bytecode::Function],
    natives: &mut Vec<fn(&[Value]) -> Value>,
    exception_handlers: &mut Vec<ExceptionHandler>,
    error_type_table: &mut Vec<String>,
    explicit_relations: &mut Vec<crate::vm::types::ExplicitRelation>,
    explicit_primary_keys: &mut Vec<crate::vm::types::ExplicitPrimaryKey>,
    loaded_modules: &mut std::collections::HashSet<String>,
    abi_natives: &mut Vec<crate::abi::NativeAbiFn>,
    loaded_native_libraries: &mut Vec<libloading::Library>,
    vm_ptr: *mut crate::vm::vm::Vm,
) -> Result<VMStatus, LangError> {
    let frame = frames.last_mut().unwrap();
    let current_ip = frame.ip - 1; // IP уже инкрементирован в step()

    // Логирование выполнения конструктора
    let is_constructor = frame.function.name.contains("::new_");
    
    if is_constructor {
        let is_return = matches!(instruction, OpCode::Return);
        let _is_load_local = matches!(instruction, OpCode::LoadLocal(_));
        let _is_store_local = matches!(instruction, OpCode::StoreLocal(_));
        debug_println!("[DEBUG executor constructor] '{}' IP {} line {}: {:?} (stack len {})", 
            frame.function.name, current_ip, line, instruction, stack.len());
        if is_return && !stack.is_empty() {
            let return_value = &stack[stack.len() - 1];
            let val_type = match return_value {
                Value::Object(_) => {
                    if let Value::Object(obj_rc) = return_value {
                        let map = obj_rc.borrow();
                        let keys: Vec<String> = map.keys().cloned().collect();
                        format!("Object с ключами: {:?}", keys)
                    } else {
                        "Object".to_string()
                    }
                },
                _ => format!("{:?}", return_value),
            };
            debug_println!("[DEBUG executor constructor] Возвращаемое значение: {}", val_type);
        }
    }
    
    match instruction {
                OpCode::Import(module_index) => {
                    let module_name = match &frame.function.chunk.constants[module_index] {
                        Value::String(name) => name.clone(),
                        _ => {
                            let error = ExceptionHandler::runtime_error(
                                &frames,
                                "Import expects module name as string".to_string(),
                                line,
                            );
                            match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error) {
                                Ok(()) => return Ok(VMStatus::Continue),
                                Err(e) => return Err(e),
                            }
                        }
                    };
                    
                    // Check if module is already loaded
                    if loaded_modules.contains(&module_name) {
                        return Ok(VMStatus::Continue);
                    }
                    
                    // Built-in (ml, plot, settings_env), then .dc file, then native ABI module
                    if modules::is_known_module(&module_name) {
                        modules::register_module(&module_name, natives, globals, global_names)?;
                        loaded_modules.insert(module_name);
                        return Ok(VMStatus::Continue);
                    }
                    let base_path = unsafe { (*vm_ptr).get_base_path() }.or_else(crate::vm::file_import::get_base_path);
                    let load_dc_err = if let Some(ref base_path) = base_path {
                        match crate::vm::file_import::load_local_module_with_vm(&module_name, base_path) {
                            Ok((module_object, module_vm)) => {
                                let start_function_index = unsafe {
                                    let vm_ref = &mut *vm_ptr;
                                    vm_ref.add_functions(module_vm.get_functions().clone())
                                };
                                if let Value::Object(module_obj_rc) = &module_object {
                                    let mut module_obj = module_obj_rc.borrow_mut();
                                    module_obj.insert("__start_function_index".to_string(), Value::Number(start_function_index as f64));
                                }
                                if let Some((&idx, _)) = global_names.iter().find(|(_, n)| n.as_str() == module_name.as_str()) {
                                    if idx < globals.len() { globals[idx] = module_object; } else { globals.resize(idx + 1, Value::Null); globals[idx] = module_object; }
                                } else {
                                    let idx = globals.len();
                                    globals.push(module_object);
                                    global_names.insert(idx, module_name.clone());
                                }
                                loaded_modules.insert(module_name);
                                return Ok(VMStatus::Continue);
                            }
                            Err(e) => Some(e),
                        }
                    } else {
                        None
                    };
                    if let Ok(module_object) = crate::vm::native_loader::try_load_native_module(
                        &module_name,
                        base_path.as_deref(),
                        natives.len(),
                        abi_natives,
                        loaded_native_libraries,
                    ) {
                        let module_value = Value::Object(Rc::new(RefCell::new(module_object)));
                        if let Some((&idx, _)) = global_names.iter().find(|(_, n)| n.as_str() == module_name.as_str()) {
                            if idx < globals.len() { globals[idx] = module_value; } else { globals.resize(idx + 1, Value::Null); globals[idx] = module_value; }
                        } else {
                            let idx = globals.len();
                            globals.push(module_value);
                            global_names.insert(idx, module_name.clone());
                        }
                        loaded_modules.insert(module_name);
                        return Ok(VMStatus::Continue);
                    }
                    return Err(load_dc_err.map_or_else(
                        || LangError::runtime_error(
                            format!("Module '{}' not found (built-in, .dc file, or native module)", module_name),
                            line,
                        ),
                        |e| LangError::runtime_error(
                            format!("Failed to load module '{}': {}", module_name, e),
                            line,
                        ),
                    ));
                }
                OpCode::ImportFrom(module_index, items_index) => {
                    // Get module name
                    let module_name = match &frame.function.chunk.constants[module_index] {
                        Value::String(name) => name.clone(),
                        _ => {
                            let error = ExceptionHandler::runtime_error(
                            &frames,
                                "ImportFrom expects module name as string".to_string(),
                                line,
                            );
                            match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error) {
                                Ok(()) => return Ok(VMStatus::Continue),
                                Err(e) => return Err(e),
                            }
                        }
                    };
                    
                    // Get items array
                    let items_array = match &frame.function.chunk.constants[items_index] {
                        Value::Array(arr) => arr.borrow().clone(),
                        _ => {
                            let error = ExceptionHandler::runtime_error(
                            &frames,
                                "ImportFrom expects items array".to_string(),
                                line,
                            );
                            match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error) {
                                Ok(()) => return Ok(VMStatus::Continue),
                                Err(e) => return Err(e),
                            }
                        }
                    };
                    
                    // Register the module if not already loaded
                    if !loaded_modules.contains(&module_name) {
                        // Сначала попробуем зарегистрировать как встроенный модуль
                        if modules::is_known_module(&module_name) {
                            modules::register_module(&module_name, natives, globals, global_names)?;
                            loaded_modules.insert(module_name.clone());
                        } else {
                            // Попробуем загрузить как локальный файл (VM base_path затем thread-local)
                            use crate::vm::file_import;
                            let base_path = unsafe { (*vm_ptr).get_base_path() }.or_else(file_import::get_base_path);
                            if let Some(ref base_path) = base_path {
                                match file_import::load_local_module_with_vm(&module_name, base_path) {
                                    Ok((module_object, module_vm)) => {
                                        debug_println!("[DEBUG ImportFrom] Загружен модуль '{}'", module_name);
                                        debug_println!("[DEBUG ImportFrom] Функций в модуле: {}", module_vm.get_functions().len());
                                        
                                        // Сливаем функции и глобалы модуля в main VM (merge_globals_from добавляет функции и обновляет индексы в глобалах)
                                        let start_function_index = unsafe {
                                            let vm_ref = &mut *vm_ptr;
                                            let start_idx = vm_ref.functions_count();
                                            debug_println!("[DEBUG ImportFrom] Начальный индекс функций в основном VM: {}", start_idx);
                                            vm_ref.merge_globals_from(&module_vm);
                                            let added_count = vm_ref.functions_count() - start_idx;
                                            debug_println!("[DEBUG ImportFrom] Добавлено функций: {}, начальный индекс: {}", added_count, start_idx);
                                            // Обновляем LoadGlobal/StoreGlobal в байткоде добавленных функций на индексы main VM
                                            let global_names = vm_ref.get_global_names().clone();
                                            let end = vm_ref.functions_count();
                                            for i in start_idx..end {
                                                crate::vm::vm::Vm::update_chunk_indices_from_names(
                                                    &mut vm_ref.get_functions_mut()[i].chunk,
                                                    &global_names,
                                                    None, // vm_ref already borrowed mutably for chunk
                                                );
                                            }
                                            start_idx
                                        };

                                        // Обновляем индексы глобалов в главном chunk, чтобы они соответствовали
                                        // текущему global_names после merge (иначе последующий LOAD_GLOBAL загрузит неверный слот)
                                        if let Some(main_frame) = frames.first_mut() {
                                            crate::vm::vm::Vm::update_chunk_indices_from_names(
                                                &mut main_frame.function.chunk,
                                                global_names,
                                                Some(globals.as_slice()),
                                            );
                                        }

                                        // Сохраняем start_function_index в объекте модуля как метаданные
                                        if let Value::Object(module_obj_rc) = &module_object {
                                            let mut module_obj = module_obj_rc.borrow_mut();
                                            module_obj.insert("__start_function_index".to_string(), Value::Number(start_function_index as f64));
                                            debug_println!("[DEBUG ImportFrom] Сохранен start_function_index={} в объекте модуля", start_function_index);
                                            debug_println!("[DEBUG ImportFrom] Ключи в объекте модуля: {:?}", module_obj.keys().collect::<Vec<_>>());
                                        }
                                        
                                        // Регистрируем модуль в глобальных переменных
                                        // Проверяем, есть ли уже модуль в globals
                                        if let Some((&idx, _)) = global_names.iter().find(|(_, n)| n.as_str() == module_name.as_str()) {
                                            // Модуль уже зарегистрирован, обновляем значение
                                            if idx < globals.len() {
                                                globals[idx] = module_object;
                                            } else {
                                                // Индекс выходит за границы, расширяем массив
                                                globals.resize(idx + 1, Value::Null);
                                                globals[idx] = module_object;
                                            }
                                        } else {
                                            // Новый модуль, создаем новый индекс
                                            let idx = globals.len();
                                            globals.push(module_object);
                                            global_names.insert(idx, module_name.clone());
                                        }
                                        loaded_modules.insert(module_name.clone());
                                    }
                                    Err(load_err) => {
                                        // Попробуем загрузить как нативный ABI-модуль (.so/.dylib)
                                        match crate::vm::native_loader::try_load_native_module(
                                            &module_name,
                                            Some(&base_path),
                                            natives.len(),
                                            abi_natives,
                                            loaded_native_libraries,
                                        ) {
                                            Ok(module_object) => {
                                                let module_value = Value::Object(Rc::new(RefCell::new(module_object)));
                                                if let Some((&idx, _)) = global_names.iter().find(|(_, n)| n.as_str() == module_name.as_str()) {
                                                    if idx < globals.len() {
                                                        globals[idx] = module_value;
                                                    } else {
                                                        globals.resize(idx + 1, Value::Null);
                                                        globals[idx] = module_value;
                                                    }
                                                } else {
                                                    let idx = globals.len();
                                                    globals.push(module_value);
                                                    global_names.insert(idx, module_name.clone());
                                                }
                                                loaded_modules.insert(module_name.clone());
                                            }
                                            Err(_) => {
                                                let error = ExceptionHandler::runtime_error(
                                                    &frames,
                                                    format!("Failed to load module '{}': {}", module_name, load_err),
                                                    line,
                                                );
                                                match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error) {
                                                    Ok(()) => return Ok(VMStatus::Continue),
                                                    Err(e) => return Err(e),
                                                }
                                            }
                                        }
                                    }
                                }
                            } else {
                                // Базовый путь не установлен — локальные .dc модули недоступны
                                let builtins = modules::builtin_modules_list();
                                let error = ExceptionHandler::runtime_error(
                                    &frames,
                                    format!(
                                        "Module '{}' not found. Built-in modules: {}. For local .dc modules (e.g. from config import Config), run a script file from CLI or use run_with_vm_with_args_and_lib(..., base_path).",
                                        module_name, builtins
                                    ),
                                    line,
                                );
                                match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error) {
                                    Ok(()) => return Ok(VMStatus::Continue),
                                    Err(e) => return Err(e),
                                }
                            }
                        }
                    }
                    
                    // Get the module object from globals
                    let module_global_index = if let Some((&idx, _)) = global_names.iter().find(|(_, name)| name.as_str() == module_name) {
                        idx
                    } else {
                        let error = ExceptionHandler::runtime_error(
                            &frames,
                            format!("Module {} not found in globals", module_name),
                            line,
                        );
                        match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error) {
                            Ok(()) => return Ok(VMStatus::Continue),
                            Err(e) => return Err(e),
                        }
                    };
                    
                    // Ensure globals vector is large enough
                    if module_global_index >= globals.len() {
                        globals.resize(module_global_index + 1, Value::Null);
                    }
                    
                    // Get the module object from globals
                    // First verify it exists and is an Object
                    if module_global_index >= globals.len() {
                        let error = ExceptionHandler::runtime_error(
                            &frames,
                            format!("Module {} global index {} out of bounds (globals.len() = {})", 
                                module_name, module_global_index, globals.len()),
                            line,
                        );
                        match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error) {
                            Ok(()) => return Ok(VMStatus::Continue),
                            Err(e) => return Err(e),
                        }
                    }
                    
                    // Get the module object - clone the Rc to avoid borrow conflicts
                    let module_object_rc = match &globals[module_global_index] {
                        Value::Object(map_rc) => map_rc.clone(),
                        Value::Null => {
                            let error = ExceptionHandler::runtime_error(
                            &frames,
                                format!("Module {} is Null - module registration may have failed", module_name),
                                line,
                            );
                            match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error) {
                                Ok(()) => return Ok(VMStatus::Continue),
                                Err(e) => return Err(e),
                            }
                        }
                        _ => {
                            let error = ExceptionHandler::runtime_error(
                            &frames,
                                format!("Module {} is not an object (found: {:?})", module_name, 
                                    std::mem::discriminant(&globals[module_global_index])),
                                line,
                            );
                            match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error) {
                                Ok(()) => return Ok(VMStatus::Continue),
                                Err(e) => return Err(e),
                            }
                        }
                    };
                    // Clone the HashMap to avoid borrowing issues - we can now mutate globals
                    let module_object = module_object_rc.borrow().clone();
                    
                    // Import items
                    for item_value in items_array {
                        match item_value {
                            Value::String(item_str) => {
                                if item_str == "*" {
                                    // Import all items
                                    // First pass: collect all global indices we need without modifying globals
                                    let mut indices_to_set: Vec<(usize, String, Value)> = Vec::new();
                                    let mut max_index_needed = globals.len();
                                    let mut new_indices = Vec::new(); // Track new indices we need to create
                                    
                                    for (key, value) in &module_object {
                                        // Find or create global index for this name
                                        let global_index = global_names.iter()
                                            .find(|(_, name)| name.as_str() == key.as_str())
                                            .map(|(&idx, _)| idx);
                                        
                                        let global_index = match global_index {
                                            Some(idx) => idx,
                                            None => {
                                                // Name not found - we'll create new global index after calculating all
                                                // Use a temporary index based on current length + new indices count
                                                let idx = globals.len() + new_indices.len();
                                                new_indices.push((idx, key.clone()));
                                                idx
                                            }
                                        };
                                        
                                        max_index_needed = max_index_needed.max(global_index + 1);
                                        indices_to_set.push((global_index, key.clone(), value.clone()));
                                    }
                                    
                                    // Resize globals vector once to accommodate all indices
                                    if max_index_needed > globals.len() {
                                        globals.resize(max_index_needed, Value::Null);
                                    }
                                    
                                    // Register new global names
                                    for (idx, name) in new_indices {
                                        global_names.insert(idx, name);
                                    }
                                    
                                    // Second pass: set all values
                                    for (global_index, _key, value) in indices_to_set {
                                        // Store the value at the correct index
                                        globals[global_index] = value;
                                    }
                                } else if item_str.contains(':') {
                                    // Aliased import: "name:alias"
                                    let parts: Vec<&str> = item_str.split(':').collect();
                                    if parts.len() == 2 {
                                        let name = parts[0];
                                        let alias = parts[1];
                                        
                                        if let Some(value) = module_object.get(name) {
                                            // Register the alias in globals
                                            let global_index = if let Some(&idx) = global_names.iter().find(|(_, n)| n.as_str() == alias).map(|(idx, _)| idx) {
                                                idx
                                            } else {
                                                let idx = globals.len();
                                                globals.push(value.clone());
                                                global_names.insert(idx, alias.to_string());
                                                idx
                                            };
                                            // Update the global value
                                            if global_index >= globals.len() {
                                                globals.resize(global_index + 1, Value::Null);
                                            }
                                            globals[global_index] = value.clone();
                                        } else {
                                            let error = ExceptionHandler::runtime_error_with_type(
                                &frames,
                                                format!("Module '{}' has no attribute '{}'", module_name, name),
                                                line,
                                                crate::common::error::ErrorType::KeyError,
                                            );
                                            match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error) {
                                                Ok(()) => continue,
                                                Err(e) => return Err(e),
                                            }
                                        }
                                    }
                                } else {
                                    // Named import: just the name
                                    debug_println!("[DEBUG ImportFrom] Импортируем '{}' из модуля '{}'", item_str, module_name);
                                    debug_println!("[DEBUG ImportFrom] Доступные ключи в модуле: {:?}", module_object.keys().collect::<Vec<_>>());
                                    
                                    if let Some(value) = module_object.get(&item_str) {
                                        debug_println!("[DEBUG ImportFrom] Найден '{}' в модуле, тип: {:?}", item_str, match value {
                                            Value::Object(_) => "Object",
                                            Value::Function(_) => "Function",
                                            Value::Null => "Null",
                                            _ => "Other",
                                        });
                                        
                                        // Register the name in globals
                                        let global_index = if let Some(&idx) = global_names.iter().find(|(_, n)| n.as_str() == item_str.as_str()).map(|(idx, _)| idx) {
                                            debug_println!("[DEBUG ImportFrom] '{}' уже существует в globals с индексом {}", item_str, idx);
                                            idx
                                        } else {
                                            let idx = globals.len();
                                            globals.push(value.clone());
                                            global_names.insert(idx, item_str.clone());
                                            debug_println!("[DEBUG ImportFrom] Создан новый глобальный индекс {} для '{}'", idx, item_str);
                                            idx
                                        };
                                        // Update the global value
                                        if global_index >= globals.len() {
                                            globals.resize(global_index + 1, Value::Null);
                                        }
                                        globals[global_index] = value.clone();
                                        debug_println!("[DEBUG ImportFrom] '{}' установлен в globals[{}]", item_str, global_index);
                                        
                                        // Если импортируется класс (объект с метаданными класса), также импортируем все конструкторы
                                        // Конструкторы имеют формат ClassName::new_<arity>
                                        if let Value::Object(class_obj_rc) = value {
                                            let class_obj = class_obj_rc.borrow();
                                            debug_println!("[DEBUG ImportFrom] Проверяем, является ли '{}' классом...", item_str);
                                            // Проверяем, что это класс (имеет метаданные __class_name)
                                            if class_obj.contains_key("__class_name") {
                                                debug_println!("[DEBUG ImportFrom] '{}' является классом! Импортируем конструкторы...", item_str);
                                                // Получаем start_function_index из объекта модуля
                                                // Сначала получаем объект модуля из глобальных переменных
                                                let start_function_index = if let Some(&module_global_idx) = global_names.iter().find(|(_, n)| n.as_str() == module_name).map(|(idx, _)| idx) {
                                                    if module_global_idx < globals.len() {
                                                        if let Value::Object(module_obj_rc) = &globals[module_global_idx] {
                                                            let module_obj = module_obj_rc.borrow();
                                                            if let Some(Value::Number(idx)) = module_obj.get("__start_function_index") {
                                                                debug_println!("[DEBUG ImportFrom] Найден start_function_index={} для модуля '{}'", *idx, module_name);
                                                                *idx as usize
                                                            } else {
                                                                debug_println!("[DEBUG ImportFrom] WARNING: start_function_index не найден в модуле '{}', используем 0", module_name);
                                                                0 // Если не найден, используем 0 (функции уже добавлены)
                                                            }
                                                        } else {
                                                            debug_println!("[DEBUG ImportFrom] WARNING: Модуль '{}' не является объектом", module_name);
                                                            0
                                                        }
                                                    } else {
                                                        debug_println!("[DEBUG ImportFrom] WARNING: Индекс модуля {} выходит за границы globals (len={})", module_global_idx, globals.len());
                                                        0
                                                    }
                                                } else {
                                                    debug_println!("[DEBUG ImportFrom] WARNING: Модуль '{}' не найден в global_names", module_name);
                                                    0
                                                };
                                                
                                                // Импортируем все конструкторы этого класса из модуля
                                                let constructor_prefix = format!("{}::new_", item_str);
                                                debug_println!("[DEBUG ImportFrom] Ищем конструкторы с префиксом '{}'", constructor_prefix);
                                                let mut found_constructors = 0;
                                                for (key, val) in module_object.iter() {
                                                    if key.starts_with(&constructor_prefix) {
                                                        found_constructors += 1;
                                                        debug_println!("[DEBUG ImportFrom] Найден конструктор: {}", key);
                                                        // Обновляем индекс функции в конструкторе
                                                        let (updated_val, new_function_index) = match val {
                                                            Value::Function(function_index) => {
                                                                let new_index = start_function_index + *function_index;
                                                                debug_println!("[DEBUG ImportFrom] Обновляем индекс функции: {} -> {}", function_index, new_index);
                                                                // Обновляем индекс функции с учетом уже добавленных функций
                                                                (Value::Function(new_index), new_index)
                                                            }
                                                            _ => {
                                                                debug_println!("[DEBUG ImportFrom] WARNING: Конструктор {} не является функцией", key);
                                                                (val.clone(), 0)
                                                            }
                                                        };
                                                        
                                                        // Импортируем конструктор
                                                        let constructor_global_index = if let Some(&idx) = global_names.iter().find(|(_, n)| n.as_str() == key.as_str()).map(|(idx, _)| idx) {
                                                            debug_println!("[DEBUG ImportFrom] Конструктор '{}' уже существует в globals с индексом {}, обновляем индекс функции", key, idx);
                                                            // Обновляем индекс функции в существующем конструкторе
                                                            // Это важно, потому что конструктор мог быть создан при merge из __lib__.dc
                                                            // с неправильным индексом функции (0), и теперь нужно обновить его на правильный
                                                            if idx < globals.len() {
                                                                if let Value::Function(old_fn_idx) = &globals[idx] {
                                                                    debug_println!("[DEBUG ImportFrom] Старый индекс функции: {}, новый индекс функции: {} (функции из модуля добавлены в VM)", old_fn_idx, new_function_index);
                                                                }
                                                                globals[idx] = updated_val.clone();
                                                            } else {
                                                                globals.resize(idx + 1, Value::Null);
                                                                globals[idx] = updated_val.clone();
                                                            }
                                                            idx
                                                        } else {
                                                            let idx = globals.len();
                                                            globals.push(updated_val.clone());
                                                            global_names.insert(idx, key.clone());
                                                            debug_println!("[DEBUG ImportFrom] Создан новый глобальный индекс {} для конструктора '{}'", idx, key);
                                                            idx
                                                        };
                                                        // Update the global value
                                                        if constructor_global_index >= globals.len() {
                                                            globals.resize(constructor_global_index + 1, Value::Null);
                                                        }
                                                        globals[constructor_global_index] = updated_val;
                                                        debug_println!("[DEBUG ImportFrom] Конструктор '{}' установлен в globals[{}] с индексом функции {}", key, constructor_global_index, new_function_index);
                                                    }
                                                }
                                                debug_println!("[DEBUG ImportFrom] Всего найдено конструкторов: {}", found_constructors);
                                                // Методы класса живут только внутри объекта класса (getBalance, deposit и т.д.), не экспортируем их в globals при ImportFrom.
                                            } else {
                                                debug_println!("[DEBUG ImportFrom] '{}' не является классом (нет ключа __class_name)", item_str);
                                            }
                                        }
                                    } else {
                                        debug_println!("[DEBUG ImportFrom] ERROR: '{}' не найден в модуле '{}'", item_str, module_name);
                                        let error = ExceptionHandler::runtime_error_with_type(
                                &frames,
                                            format!("Module '{}' has no attribute '{}'", module_name, item_str),
                                            line,
                                            crate::common::error::ErrorType::KeyError,
                                        );
                                        match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error) {
                                            Ok(()) => continue,
                                            Err(e) => return Err(e),
                                        }
                                    }
                                }
                            }
                            _ => {
                                let error = ExceptionHandler::runtime_error(
                            &frames,
                                    "ImportFrom item must be a string".to_string(),
                                    line,
                                );
                                match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error) {
                                    Ok(()) => continue,
                                    Err(e) => return Err(e),
                                }
                            }
                        }
                    }
                    
                    // After importing items (builtin or file), update main chunk's LoadGlobal/StoreGlobal
                    // to the current global_names so subsequent instructions see the correct slots.
                    if let Some(main_frame) = frames.first_mut() {
                        crate::vm::vm::Vm::update_chunk_indices_from_names(
                            &mut main_frame.function.chunk,
                            global_names,
                            Some(globals.as_slice()),
                        );
                    }
                    return Ok(VMStatus::Continue);
                }
                OpCode::Constant(index) => {
                    let value = frame.function.chunk.constants[index].clone();
                    stack::push(stack, value);
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
                    stack::push(stack, loaded_value);
                    return Ok(VMStatus::Continue);
                }
                OpCode::StoreLocal(index) => {
                    let value = stack::pop(stack, frames, exception_handlers)?;
                    
                    // Логирование для объектов (проверка клонирования)
                    if let Value::Object(obj_rc) = &value {
                        let _obj_ptr = Rc::as_ptr(obj_rc);
                        let frame = frames.last().unwrap();
                        let is_constructor = frame.function.name.contains("::new_");
                        let current_ip = frame.ip - 1;
                        
                        if is_constructor {
                            let map = obj_rc.borrow();
                            let key_count = map.len();
                            debug_println!("[DEBUG StoreLocal] constructor '{}' IP {} slot {}: Object ({} keys)", frame.function.name, current_ip, index, key_count);
                        }
                    }
                    
                    // Clone уже создает глубокую копию для массивов и таблиц
                    let frame = frames.last_mut().unwrap();
                    if index >= frame.slots.len() {
                        frame.slots.resize(index + 1, Value::Null);
                    }
                    frame.slots[index] = value;
                    return Ok(VMStatus::Continue);
                }
                OpCode::LoadGlobal(index) => {
                    if index >= globals.len() {
                        // Check if this is a known module that hasn't been imported
                        let error_message = if let Some(var_name) = global_names.get(&index) {
                            if modules::is_known_module(var_name) && !loaded_modules.contains(var_name) {
                                format!("Module {} not imported", var_name)
                            } else {
                                format!("Undefined variable: {}", var_name)
                            }
                        } else {
                            format!("Undefined variable")
                        };
                        
                        let error = ExceptionHandler::runtime_error(
                            &frames,
                            error_message,
                            line,
                        );
                        match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error) {
                            Ok(()) => {
                                // Исключение обработано, кладем Null на стек
                                stack::push(stack, Value::Null);
                            }
                            Err(e) => return Err(e), // Исключение не обработано
                        }
                    } else {
                        // Для сложных типов (Array, Table) возвращаем ссылку (shallow copy Rc)
                        // Для простых типов клонируем значение
                        let value = &globals[index];
                        // Отладочный вывод для проверки значений
                        if let Some(var_name) = global_names.get(&index) {
                            if var_name.contains("Data") || var_name.contains("range") || var_name.contains("print") || var_name.contains("::new_") {
                                let value_type_str = match value {
                                    Value::Null => "Null".to_string(),
                                    Value::Function(fn_idx) => {
                                        if *fn_idx < functions.len() {
                                            format!("Function({}, имя: '{}')", fn_idx, functions[*fn_idx].name)
                                        } else {
                                            format!("Function({}, OUT OF BOUNDS!)", fn_idx)
                                        }
                                    },
                                    Value::Object(_) => "Object".to_string(),
                                    Value::NativeFunction(_) => "NativeFunction".to_string(),
                                    _ => "Other".to_string(),
                                };
                                debug_println!("[DEBUG LoadGlobal] Загружаем '{}' из globals[{}], значение: {}", var_name, index, value_type_str);
                                
                                // Дополнительная проверка для конструкторов
                                if var_name.contains("::new_") {
                                    if let Value::Function(fn_idx) = value {
                                        if *fn_idx < functions.len() {
                                            let func = &functions[*fn_idx];
                                            debug_println!("[DEBUG LoadGlobal] Конструктор '{}' имеет индекс функции {}, имя функции: '{}', arity: {}", 
                                                var_name, fn_idx, func.name, func.arity);
                                        } else {
                                            debug_println!("[DEBUG LoadGlobal] ОШИБКА: Конструктор '{}' имеет индекс функции {} (выходит за границы, всего функций: {})", 
                                                var_name, fn_idx, functions.len());
                                        }
                                    }
                                }
                            }
                        }
                        if matches!(value, Value::Null) {
                            // Если значение Null, проверяем, не является ли это функцией, которая должна быть установлена
                            if let Some(var_name) = global_names.get(&index) {
                                if var_name.contains("Data") || var_name.contains("range") || var_name.contains("print") {
                                    debug_println!("[DEBUG LoadGlobal] WARNING: '{}' в globals[{}] равен Null", var_name, index);
                                }
                            }
                        }
                        let loaded_value = match value {
                            Value::Array(arr_rc) => Value::Array(Rc::clone(arr_rc)),
                            Value::Table(table_rc) => Value::Table(Rc::clone(table_rc)),
                            _ => value.clone(), // Простые типы клонируем
                        };
                        stack::push(stack, loaded_value);
                    }
                    return Ok(VMStatus::Continue);
                }
                OpCode::StoreGlobal(index) => {
                    let mut value = stack::pop(stack, frames, exception_handlers)?;
                    // Если значение - таблица, устанавливаем её имя из global_names
                    if let Value::Table(table_rc) = &mut value {
                        if let Some(var_name) = global_names.get(&index) {
                            table_rc.borrow_mut().set_name(var_name.clone());
                        }
                    }
                    // Clone уже создает глубокую копию для массивов и таблиц
                    if index >= globals.len() {
                        globals.resize(index + 1, Value::Null);
                    }
                    // Важно: присваиваем value после установки имени, чтобы имя сохранилось
                    globals[index] = value;
                    return Ok(VMStatus::Continue);
                }
                OpCode::Add => {
                    let b = stack::pop(stack, frames, exception_handlers)?;
                    let a = stack::pop(stack, frames, exception_handlers)?;
                    let result = operations::binary_add(&a, &b, frames, stack, exception_handlers)?;
                    stack::push(stack, result);
                    return Ok(VMStatus::Continue);
                }
                OpCode::Sub => {
                    let b = stack::pop(stack, frames, exception_handlers)?;
                    let a = stack::pop(stack, frames, exception_handlers)?;
                    let result = operations::binary_sub(&a, &b, frames, stack, exception_handlers)?;
                    stack::push(stack, result);
                    return Ok(VMStatus::Continue);
                }
                OpCode::Mul => {
                    let b = stack::pop(stack, frames, exception_handlers)?;
                    let a = stack::pop(stack, frames, exception_handlers)?;
                    let result = operations::binary_mul(&a, &b, frames, stack, exception_handlers)?;
                    stack::push(stack, result);
                    return Ok(VMStatus::Continue);
                }
                OpCode::Div => {
                    let b = stack::pop(stack, frames, exception_handlers)?;
                    let a = stack::pop(stack, frames, exception_handlers)?;
                    let result = operations::binary_div(&a, &b, frames, stack, exception_handlers)?;
                    stack::push(stack, result);
                    return Ok(VMStatus::Continue);
                }
                OpCode::IntDiv => {
                    let b = stack::pop(stack, frames, exception_handlers)?;
                    let a = stack::pop(stack, frames, exception_handlers)?;
                    let result = operations::binary_int_div(&a, &b, frames, stack, exception_handlers)?;
                    stack::push(stack, result);
                    return Ok(VMStatus::Continue);
                }
                OpCode::Mod => {
                    let b = stack::pop(stack, frames, exception_handlers)?;
                    let a = stack::pop(stack, frames, exception_handlers)?;
                    let result = operations::binary_mod(&a, &b, frames, stack, exception_handlers)?;
                    stack::push(stack, result);
                    return Ok(VMStatus::Continue);
                }
                OpCode::Pow => {
                    let b = stack::pop(stack, frames, exception_handlers)?;
                    let a = stack::pop(stack, frames, exception_handlers)?;
                    let result = operations::binary_pow(&a, &b, frames, stack, exception_handlers)?;
                    stack::push(stack, result);
                }
                OpCode::Negate => {
                    let value = stack::pop(stack, frames, exception_handlers)?;
                    let result = operations::unary_negate(&value, frames, stack, exception_handlers)?;
                    stack::push(stack, result);
                }
                OpCode::Not => {
                    let value = stack::pop(stack, frames, exception_handlers)?;
                    let result = operations::unary_not(&value);
                    stack::push(stack, result);
                }
                OpCode::Or => {
                    let b = stack::pop(stack, frames, exception_handlers)?;
                    let a = stack::pop(stack, frames, exception_handlers)?;
                    let result = operations::binary_or(&a, &b);
                    stack::push(stack, result);
                }
                OpCode::And => {
                    let b = stack::pop(stack, frames, exception_handlers)?;
                    let a = stack::pop(stack, frames, exception_handlers)?;
                    let result = operations::binary_and(&a, &b);
                    stack::push(stack, result);
                }
                OpCode::Equal => {
                    let b = stack::pop(stack, frames, exception_handlers)?;
                    let a = stack::pop(stack, frames, exception_handlers)?;
                    let result = operations::binary_equal(&a, &b);
                    stack::push(stack, result);
                }
                OpCode::NotEqual => {
                    let b = stack::pop(stack, frames, exception_handlers)?;
                    let a = stack::pop(stack, frames, exception_handlers)?;
                    let result = operations::binary_not_equal(&a, &b);
                    stack::push(stack, result);
                }
                OpCode::Greater => {
                    let b = stack::pop(stack, frames, exception_handlers)?;
                    let a = stack::pop(stack, frames, exception_handlers)?;
                    let result = operations::binary_greater(&a, &b, frames, stack, exception_handlers)?;
                    stack::push(stack, result);
                    return Ok(VMStatus::Continue);
                }
                OpCode::Less => {
                    let b = stack::pop(stack, frames, exception_handlers)?;
                    let a = stack::pop(stack, frames, exception_handlers)?;
                    let result = operations::binary_less(&a, &b, frames, stack, exception_handlers)?;
                    stack::push(stack, result);
                    return Ok(VMStatus::Continue);
                }
                OpCode::GreaterEqual => {
                    let b = stack::pop(stack, frames, exception_handlers)?;
                    let a = stack::pop(stack, frames, exception_handlers)?;
                    let result = operations::binary_greater_equal(&a, &b, frames, stack, exception_handlers)?;
                    stack::push(stack, result);
                    return Ok(VMStatus::Continue);
                }
                OpCode::LessEqual => {
                    let b = stack::pop(stack, frames, exception_handlers)?;
                    let a = stack::pop(stack, frames, exception_handlers)?;
                    let result = operations::binary_less_equal(&a, &b, frames, stack, exception_handlers)?;
                    stack::push(stack, result);
                }
                OpCode::In => {
                    let array = stack::pop(stack, frames, exception_handlers)?; // Правый операнд - массив
                    let value = stack::pop(stack, frames, exception_handlers)?; // Левый операнд - значение для поиска
                    
                    match array {
                        Value::Array(arr) => {
                            let arr_ref = arr.borrow();
                            let found = arr_ref.iter().any(|item| item == &value);
                            stack::push(stack, Value::Bool(found));
                        }
                        _ => {
                            let error = ExceptionHandler::runtime_error(
                                &frames,
                                "Right operand of 'in' operator must be an array".to_string(),
                                line,
                            );
                            match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error) {
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
                    let condition = stack::pop(stack, frames, exception_handlers)?;
                    let frame = frames.last_mut().unwrap();
                    if !condition.is_truthy() {
                        frame.ip = (frame.ip as i32 + offset as i32) as usize;
                    }
                }
                OpCode::JumpIfFalse16(offset) => {
                    let condition = stack::pop(stack, frames, exception_handlers)?;
                    let frame = frames.last_mut().unwrap();
                    if !condition.is_truthy() {
                        frame.ip = (frame.ip as i32 + offset as i32) as usize;
                    }
                }
                OpCode::JumpIfFalse32(offset) => {
                    let condition = stack::pop(stack, frames, exception_handlers)?;
                    let frame = frames.last_mut().unwrap();
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
                    let frame = frames.last().unwrap();
                    let current_ip = frame.ip - 1; // IP уже инкрементирован в step(), возвращаемся назад
                    debug_println!("[DEBUG executor OpCode::Call] Получен OpCode::Call({}) на строке {}, IP: {}", arity, line, current_ip);
                    
                    // Проверяем, что инструкция в байткоде действительно имеет правильный arity
                    if let Some(OpCode::Call(recorded_arity)) = frame.function.chunk.code.get(current_ip) {
                        if *recorded_arity != arity {
                            debug_println!("[ERROR executor OpCode::Call] КРИТИЧЕСКАЯ ОШИБКА: В байткоде на IP {} записано Call({}), но прочитано Call({})!", 
                                current_ip, recorded_arity, arity);
                            // Дополнительная отладка: показываем окружающие инструкции
                            let start = current_ip.saturating_sub(5);
                            let end = (current_ip + 5).min(frame.function.chunk.code.len());
                            debug_println!("[ERROR executor OpCode::Call] Окружающие инструкции (IP {} - {}):", start, end);
                            for i in start..end {
                                let marker = if i == current_ip { " <-- ТЕКУЩАЯ" } else { "" };
                                debug_println!("[ERROR executor OpCode::Call]   IP {}: {:?}{}", i, frame.function.chunk.code.get(i), marker);
                            }
                        } else {
                            debug_println!("[DEBUG executor OpCode::Call] Подтверждено: Call({}) правильно прочитан из байткода на IP {}", arity, current_ip);
                        }
                    } else {
                        debug_println!("[ERROR executor OpCode::Call] КРИТИЧЕСКАЯ ОШИБКА: На IP {} не найдена инструкция Call!", current_ip);
                        // Показываем, что там на самом деле
                        if let Some(opcode) = frame.function.chunk.code.get(current_ip) {
                            debug_println!("[ERROR executor OpCode::Call] На IP {} найдена инструкция: {:?}", current_ip, opcode);
                        }
                    }
                    
                    // Получаем функцию со стека
                    // Используем stack::pop для правильного номера строки при ошибке
                    let stack_size_before_pop = stack.len();
                    debug_println!("[DEBUG executor OpCode::Call] Размер стека перед извлечением функции: {}", stack_size_before_pop);
                    
                    // Детальное логирование стека для IP 15 и IP 40 (вызовы методов)
                    if current_ip == 15 || current_ip == 40 {
                        debug_println!("[DEBUG executor OpCode::Call] ДЕТАЛЬНОЕ ЛОГИРОВАНИЕ СТЕКА для IP 15:");
                        debug_println!("[DEBUG executor OpCode::Call] Размер стека: {}", stack.len());
                        debug_println!("[DEBUG executor OpCode::Call] stack_start текущего frame: {}", frame.stack_start);
                        debug_println!("[DEBUG executor OpCode::Call] Байткод вокруг IP 15:");
                        let start = current_ip.saturating_sub(5);
                        let end = (current_ip + 5).min(frame.function.chunk.code.len());
                        for i in start..end {
                            let marker = if i == current_ip { " <-- ТЕКУЩАЯ" } else { "" };
                            debug_println!("[DEBUG executor OpCode::Call]   IP {}: {:?}{}", i, frame.function.chunk.code.get(i), marker);
                        }
                        for (i, val) in stack.iter().enumerate() {
                            let val_type = match val {
                                Value::Number(_) => "Number".to_string(),
                                Value::String(_) => "String".to_string(),
                                Value::Bool(_) => "Bool".to_string(),
                                Value::Array(_) => "Array".to_string(),
                                Value::Object(_) => "Object".to_string(),
                                Value::Function(fn_idx) => {
                                    if *fn_idx < functions.len() {
                                        format!("Function({}, имя: '{}')", fn_idx, functions[*fn_idx].name)
                                    } else {
                                        format!("Function({}, OUT OF BOUNDS!)", fn_idx)
                                    }
                                },
                                Value::NativeFunction(_) => "NativeFunction".to_string(),
                                Value::Null => "Null".to_string(),
                                _ => "Other".to_string(),
                            };
                            debug_println!("[DEBUG executor OpCode::Call]   Стек[{}]: {} ({:?})", i, val_type, val);
                        }
                    }
                    
                    let function_value = stack::pop(stack, frames, exception_handlers)?;
                    let stack_size_after_pop = stack.len();
                    debug_println!("[DEBUG executor OpCode::Call] Размер стека после извлечения функции: {}", stack_size_after_pop);
                    
                    // Отладочный вывод для проверки вызываемой функции
                    let function_type = match &function_value {
                        Value::Null => "Null",
                        Value::Function(_) => "Function",
                        Value::NativeFunction(_) => "NativeFunction",
                        _ => "Other",
                    };
                    debug_println!("[DEBUG executor OpCode::Call] Значение на стеке перед вызовом: тип = {}, значение = {:?}", function_type, function_value);
                    
                    // Дополнительное логирование для IP 15 и IP 40
                    if current_ip == 15 || current_ip == 40 {
                        debug_println!("[DEBUG executor OpCode::Call] Извлечена функция для IP {}: {:?}", current_ip, function_value);
                        if let Value::Function(fn_idx) = &function_value {
                            if *fn_idx < functions.len() {
                                debug_println!("[DEBUG executor OpCode::Call] Функция на IP {}: индекс={}, имя='{}', arity={}, ожидается аргументов: {}", 
                                    current_ip, fn_idx, functions[*fn_idx].name, functions[*fn_idx].arity, arity);
                            }
                        }
                    }
                    if matches!(&function_value, Value::Null) {
                        debug_println!("[DEBUG Call] Пытаемся вызвать Null с {} аргументами на строке {}", arity, line);
                    }
                    // If callee is a class Object (from import), resolve to constructor from globals or from class object
                    // If callee is Object without __class_name but has __call__, use __call__ as the actual callee (e.g. Settings)
                    let actual_callee: Value = {
                        if let Value::Object(obj_rc) = &function_value {
                            let class_name_opt = obj_rc.borrow().get("__class_name").cloned();
                            if let Some(Value::String(class_name)) = class_name_opt {
                                let constructor_name = format!("{}::new_{}", class_name, arity);
                                let constructor_value = global_names
                                    .iter()
                                    .find(|(_, n)| *n == &constructor_name)
                                    .and_then(|(idx, _)| globals.get(*idx).cloned());
                                if let Some(Value::Function(constructor_fn_idx)) = constructor_value {
                                    debug_println!("[DEBUG executor OpCode::Call] Class object '{}' resolved to constructor '{}'", class_name, constructor_name);
                                    Value::Function(constructor_fn_idx)
                                } else {
                                    // Fallback: constructor may live only on the class object (e.g. after import)
                                    let method_key = format!("new_{}", arity);
                                    let from_class = obj_rc.borrow().get(&method_key).cloned();
                                    if let Some(Value::Function(constructor_fn_idx)) = from_class {
                                        debug_println!("[DEBUG executor OpCode::Call] Class object '{}' resolved to constructor from class key '{}'", class_name, method_key);
                                        Value::Function(constructor_fn_idx)
                                    } else {
                                        function_value
                                    }
                                }
                            } else {
                                let call_opt = obj_rc.borrow().get("__call__").cloned();
                                if let Some(Value::Function(_)) | Some(Value::NativeFunction(_)) = call_opt.as_ref() {
                                    call_opt.unwrap()
                                } else {
                                    function_value
                                }
                            }
                        } else {
                            function_value
                        }
                    };
                    match actual_callee {
                        Value::Function(function_index) => {
                            if function_index >= functions.len() {
                                let error = ExceptionHandler::runtime_error(
                            &frames,
                                    format!("Function index {} out of bounds", function_index),
                                    line,
                                );
                            match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error) {
                                Ok(()) => return Ok(VMStatus::Continue),
                                Err(e) => return Err(e),
                            }
                            }
                            
                            let function = functions[function_index].clone();
                            
                            debug_println!("[DEBUG executor Call] Вызываем функцию с индексом {}, имя: '{}', arity: {}, получено аргументов: {} (всего функций в VM: {})", 
                                function_index, function.name, function.arity, arity, functions.len());
                            
                            // Дополнительная отладка для конструкторов
                            if function.name.contains("::new_") {
                                debug_println!("[DEBUG executor Call] ВНИМАНИЕ: Вызывается конструктор '{}' с индексом функции {}", function.name, function_index);
                                debug_println!("[DEBUG executor Call] Ожидаем, что конструктор сохранит методы в объект через SetArrayElement");
                            }
                            
                            // Дополнительная отладка для методов класса
                            if function.name.contains("::method_") {
                                let current_ip_debug = frames.last().map(|f| f.ip - 1).unwrap_or(0);
                                debug_println!("[DEBUG executor Call] ВНИМАНИЕ: Вызывается метод класса '{}' с индексом функции {} на IP {}", function.name, function_index, current_ip_debug);
                                debug_println!("[DEBUG executor Call] Размер стека перед извлечением аргументов: {}, ожидается аргументов: {}", stack.len(), arity);
                            }
                            
                            // Проверяем количество аргументов
                            if arity != function.arity {
                                let error = ExceptionHandler::runtime_error(
                            &frames,
                                    format!(
                                        "Expected {} arguments but got {}",
                                        function.arity, arity
                                    ),
                                    line,
                                );
                            match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error) {
                                Ok(()) => return Ok(VMStatus::Continue),
                                Err(e) => return Err(e),
                            }
                            }
                            
                            // Собираем аргументы со стека (в обратном порядке, так как они были положены последними)
                            // ВАЖНО: Проверяем стек перед извлечением аргументов, чтобы избежать ошибки stack underflow
                            let mut args = Vec::new();
                            
                            if arity > 0 {
                                let frame = frames.last().unwrap();
                                // stack_start указывает на начало стека для текущего frame
                                // Аргументы и функция были помещены на стек после stack_start
                                // После извлечения функции стек должен содержать минимум arity аргументов
                                let stack_size_before = stack.len();
                                debug_println!("[DEBUG executor Call] Размер стека перед извлечением аргументов: {}, stack_start: {}, ожидается аргументов: {}", 
                                    stack_size_before, frame.stack_start, arity);
                                
                                // Дополнительная отладка: показываем все элементы стека
                                debug_println!("[DEBUG executor Call] Полное содержимое стека ({} элементов):", stack.len());
                                for (i, val) in stack.iter().enumerate() {
                                    let val_type = match val {
                                        Value::Number(_) => "Number",
                                        Value::String(_) => "String",
                                        Value::Bool(_) => "Bool",
                                        Value::Array(_) => "Array",
                                        Value::Object(_) => "Object",
                                        Value::Function(_) => "Function",
                                        Value::NativeFunction(_) => "NativeFunction",
                                        Value::Null => "Null",
                                        _ => "Other",
                                    };
                                    debug_println!("[DEBUG executor Call]   Стек[{}]: {} ({:?})", i, val_type, val);
                                }
                                
                                if stack.len() <= frame.stack_start {
                                    let error = ExceptionHandler::runtime_error(
                                        &frames,
                                        format!(
                                            "Not enough arguments on stack: expected {} but stack is empty",
                                            arity
                                        ),
                                        line,
                                    );
                                    match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error) {
                                        Ok(()) => return Ok(VMStatus::Continue),
                                        Err(e) => return Err(e),
                                    }
                                }
                                
                                // Проверяем, что на стеке достаточно аргументов (после извлечения функции)
                                let available_args = stack.len() - frame.stack_start;
                                debug_println!("[DEBUG executor Call] Доступно аргументов на стеке: {} (размер стека: {}, stack_start: {})", 
                                    available_args, stack.len(), frame.stack_start);
                                
                                // Логируем содержимое стека для отладки
                                if available_args > 0 {
                                    debug_println!("[DEBUG executor Call] Содержимое стека (последние {} элементов):", available_args.min(10));
                                    let start_idx = stack.len().saturating_sub(available_args.min(10));
                                    for (i, val) in stack.iter().skip(start_idx).enumerate() {
                                        debug_println!("[DEBUG executor Call]   Стек[{}]: {:?}", start_idx + i, val);
                                    }
                                }
                                
                                if available_args < arity {
                                    let error = ExceptionHandler::runtime_error(
                                        &frames,
                                        format!(
                                            "Not enough arguments on stack: expected {} but got {}",
                                            arity,
                                            available_args
                                        ),
                                        line,
                                    );
                                    match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error) {
                                        Ok(()) => return Ok(VMStatus::Continue),
                                        Err(e) => return Err(e),
                                    }
                                }
                                
                                for i in 0..arity {
                                    // Безопасно извлекаем аргументы напрямую, так как мы уже проверили стек
                                    let arg = stack.pop().unwrap_or(Value::Null);
                                    debug_println!("[DEBUG executor Call] Извлечен аргумент {}: {:?}", i, arg);
                                    args.push(arg);
                                }
                                args.reverse(); // Теперь args[0] - первый аргумент
                                debug_println!("[DEBUG executor Call] Всего извлечено аргументов: {}, после reverse: {:?}", args.len(), args);
                            }
                            // Если arity == 0, args остается пустым вектором
                            
                            // Проверяем типы аргументов, если указаны аннотации типов
                            for (i, (arg, expected_types)) in args.iter().zip(&function.param_types).enumerate() {
                                if let Some(type_names) = expected_types {
                                    if !crate::vm::calls::check_type_value(arg, type_names) {
                                        let param_name = function.param_names.get(i)
                                            .map(|s| s.as_str())
                                            .unwrap_or("unknown");
                                        let error = LangError::runtime_error_with_type(
                                            format!(
                                                "Argument '{}' expected type '{}', got '{}'",
                                                param_name, 
                                                crate::vm::calls::format_type_names(type_names),
                                                crate::vm::calls::get_type_name_value(arg)
                                            ),
                                            line,
                                            ErrorType::TypeError,
                                        );
                                        match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error) {
                                            Ok(()) => return Ok(VMStatus::Continue),
                                            Err(e) => return Err(e),
                                        }
                                    }
                                }
                            }
                            
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
                                            stack::push(stack, cached_result.clone());
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
                            let stack_start = stack.len();
                            let mut new_frame = if function.is_cached {
                                // Сохраняем аргументы для кэширования
                                CallFrame::new_with_cache(function.clone(), stack_start, args.clone())
                            } else {
                                CallFrame::new(function.clone(), stack_start)
                            };
                            
                            // Копируем таблицу типов ошибок из chunk функции в VM
                            if !function.chunk.error_type_table.is_empty() {
                                *error_type_table = function.chunk.error_type_table.clone();
                            }
                            
                            // Копируем захваченные переменные из родительских frames (если есть)
                            // Используем ancestor_depth для поиска переменной в правильном предке
                            if !frames.is_empty() && !function.captured_vars.is_empty() {
                                for captured_var in &function.captured_vars {
                                    // Убеждаемся, что слот существует в новом frame
                                    if captured_var.local_slot_index >= new_frame.slots.len() {
                                        new_frame.slots.resize(captured_var.local_slot_index + 1, Value::Null);
                                    }
                                    
                                    // Находим предка на нужной глубине
                                    // ancestor_depth = 0 означает ближайший родитель (последний frame в стеке)
                                    // ancestor_depth = 1 означает дедушку (предпоследний frame) и т.д.
                                    let ancestor_index = frames.len().saturating_sub(1 + captured_var.ancestor_depth);
                                    
                                    if ancestor_index < frames.len() {
                                        let ancestor_frame = &frames[ancestor_index];
                                        
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
                            frames.push(new_frame);
                            return Ok(VMStatus::Continue);
                        }
                        Value::NativeFunction(native_index) => {
                            let builtin_count = natives.len();
                            if native_index >= builtin_count + abi_natives.len() {
                                let error = ExceptionHandler::runtime_error(
                            &frames,
                                    format!("Native function index {} out of bounds", native_index),
                                    line,
                                );
                            match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error) {
                                Ok(()) => return Ok(VMStatus::Continue),
                                Err(e) => return Err(e),
                            }
                            }
                            
                            // Специальная обработка для методов тензора max_idx и min_idx (только встроенные нативы)
                            // Эти методы могут быть вызваны как tensor.max_idx() с arity=0,
                            // но тензор уже находится на стеке перед функцией
                            use crate::ml::natives as ml_natives;
                            let is_max_idx = native_index < builtin_count && std::ptr::eq(
                                natives[native_index] as *const (),
                                ml_natives::native_max_idx as *const ()
                            );
                            let is_min_idx = native_index < builtin_count && std::ptr::eq(
                                natives[native_index] as *const (),
                                ml_natives::native_min_idx as *const ()
                            );
                            
                            let mut args = Vec::new();
                            if (is_max_idx || is_min_idx) && arity == 0 {
                                // Для методов тензора с arity=0, используем тензор со стека как первый аргумент
                                // Тензор был помещен на стек перед функцией при доступе к свойству
                                // Важно: нужно удалить тензор со стека после использования
                                if let Some(Value::Tensor(tensor_rc)) = stack.last() {
                                    args.push(Value::Tensor(Rc::clone(tensor_rc)));
                                    // Удаляем тензор со стека, так как он был использован как аргумент
                                    // Безопасно извлекаем тензор напрямую
                                    stack.pop();
                                } else {
                                    let error = ExceptionHandler::runtime_error(&frames,
                                        "Tensor method called without tensor on stack".to_string(),
                                        line,
                                    );
                                    match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error) {
                                        Ok(()) => return Ok(VMStatus::Continue),
                                        Err(e) => return Err(e),
                                    }
                                }
                            } else {
                                // Обычная обработка аргументов
                                // ВАЖНО: Безопасно извлекаем аргументы, проверяя стек перед извлечением
                                let frame = frames.last().unwrap();
                                let available_args = stack.len() - frame.stack_start;
                                if available_args < arity {
                                    let error = ExceptionHandler::runtime_error(
                                        &frames,
                                        format!(
                                            "Not enough arguments on stack for native function: expected {} but got {}",
                                            arity,
                                            available_args
                                        ),
                                        line,
                                    );
                                    match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error) {
                                        Ok(()) => return Ok(VMStatus::Continue),
                                        Err(e) => return Err(e),
                                    }
                                }
                                
                                for _ in 0..arity {
                                    // Безопасно извлекаем аргументы напрямую, так как мы уже проверили стек
                                    args.push(stack.pop().unwrap_or(Value::Null));
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
                                *ctx.borrow_mut() = Some(vm_ptr);
                            });
                            
                            // Специальная проверка для range (принимает 1, 2 или 3 аргумента)
                            if native_index == 2 {
                                // range - индекс 2
                                if arity < 1 || arity > 3 {
                                    let error = ExceptionHandler::runtime_error(&frames,
                                        format!("range() expects 1, 2, or 3 arguments, got {}", arity),
                                        line,
                                    );
                            match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error) {
                                Ok(()) => return Ok(VMStatus::Continue),
                                Err(e) => return Err(e),
                            }
                                }
                                // Проверяем типы аргументов - все должны быть числами
                                for arg in &args {
                                    if !matches!(arg, Value::Number(_)) {
                                        let error = ExceptionHandler::runtime_error(&frames,
                                            "range() arguments must be numbers".to_string(),
                                            line,
                                        );
                            match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error) {
                                Ok(()) => return Ok(VMStatus::Continue),
                                Err(e) => return Err(e),
                            }
                                    }
                                }
                                // Проверяем, что step не равен 0 (если передан)
                                if arity == 3 {
                                    if let Value::Number(step) = &args[2] {
                                        if *step == 0.0 {
                                            let error = ExceptionHandler::runtime_error(&frames,
                                                "range() step cannot be zero".to_string(),
                                                line,
                                            );
                            match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error) {
                                Ok(()) => return Ok(VMStatus::Continue),
                                Err(e) => return Err(e),
                            }
                                        }
                                    }
                                }
                            }
                            
                            // Вызываем нативную функцию (встроенную или ABI)
                            let result = if native_index < builtin_count {
                                let native_fn = natives[native_index];
                                native_fn(&args)
                            } else {
                                crate::vm::native_loader::call_abi_native(
                                    abi_natives[native_index - builtin_count],
                                    &args,
                                )
                            };
                            
                            // Очищаем контекст VM после вызова нативной функции
                            VM_CALL_CONTEXT.with(|ctx| {
                                *ctx.borrow_mut() = None;
                            });
                            
                            // Проверяем ABI-ошибку (throw_error из нативного модуля)
                            if let Some(abi_err) = crate::vm::native_loader::take_last_abi_error() {
                                match ExceptionHandler::handle_exception(stack, frames, exception_handlers, abi_err) {
                                    Ok(()) => return Ok(VMStatus::Continue),
                                    Err(e) => return Err(e),
                                }
                            }
                            
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
                                    for (index, value) in globals.iter().enumerate() {
                                        if let Value::Table(table) = value {
                                            if Rc::as_ptr(table) == table1_ptr {
                                                if let Some(var_name) = explicit_global_names.get(&index) {
                                                    found_table1_name = Some(var_name.clone());
                                                }
                                            }
                                            if Rc::as_ptr(table) == table2_ptr {
                                                if let Some(var_name) = explicit_global_names.get(&index) {
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
                                        explicit_relations.push(ExplicitRelation {
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
                                    for (index, value) in globals.iter().enumerate() {
                                        if let Value::Table(table) = value {
                                            if Rc::as_ptr(table) == table_ptr {
                                                if let Some(var_name) = explicit_global_names.get(&index) {
                                                    found_table_name = Some(var_name.clone());
                                                }
                                            }
                                        }
                                    }
                                    
                                    // Если нашли таблицу, сохраняем первичный ключ
                                    if let Some(table_name) = found_table_name {
                                        explicit_primary_keys.push(ExplicitPrimaryKey {
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
                                    debug_println!("⚠️  Предупреждение: {}", error_msg);
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
                                    
                                    let error = ExceptionHandler::runtime_error_with_type(
                                &frames,
                                        error_msg,
                                        line,
                                        error_type,
                                    );
                                    match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error) {
                                        Ok(()) => return Ok(VMStatus::Continue), // Исключение обработано
                                        Err(e) => return Err(e), // Исключение не обработано
                                    }
                                }
                            }
                            
                            // Помещаем результат на стек
                            stack::push(stack, result);
                            return Ok(VMStatus::Continue);
                        }
                        Value::Layer(layer_id) => {
                            // Layers can be called as functions: layer(input_tensor) -> output_tensor
                            if arity != 1 {
                                let error = ExceptionHandler::runtime_error(
                            &frames,
                                    format!("Layer call expects 1 argument (input tensor), got {}", arity),
                                    line,
                                );
                                match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error) {
                                    Ok(()) => {
                                        stack::push(stack, Value::Null);
                                        return Ok(VMStatus::Continue);
                                    }
                                    Err(e) => return Err(e),
                                }
                            }
                            
                            // Get input tensor from stack
                            let input_value = stack::pop(stack, frames, exception_handlers)?;
                            
                            // Call native layer_call function directly
                            use crate::ml::natives;
                            let args = vec![Value::Layer(layer_id), input_value];
                            let result = natives::native_layer_call(&args);
                            
                            stack::push(stack, result);
                            return Ok(VMStatus::Continue);
                        }
                        Value::NeuralNetwork(_) | Value::LinearRegression(_) => {
                            // Models can be called as functions: model(input_tensor) -> output_tensor
                            if arity != 1 {
                                let error = ExceptionHandler::runtime_error(
                            &frames,
                                    format!("Model call expects 1 argument (input tensor), got {}", arity),
                                    line,
                                );
                                match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error) {
                                    Ok(()) => {
                                        stack::push(stack, Value::Null);
                                        return Ok(VMStatus::Continue);
                                    }
                                    Err(e) => return Err(e),
                                }
                            }
                            
                            // Get input tensor from stack
                            let input_value = stack::pop(stack, frames, exception_handlers)?;
                            
                            // Call native_nn_forward function (it handles both NeuralNetwork and LinearRegression)
                            use crate::ml::natives;
                            let args = vec![actual_callee.clone(), input_value];
                            let result = natives::native_nn_forward(&args);
                            
                            stack::push(stack, result);
                            return Ok(VMStatus::Continue);
                        }
                        _ => {
                            // Try to provide more helpful error message
                            let error_msg = match &actual_callee {
                                Value::Null => "Cannot call null - function may not be imported or defined".to_string(),
                                Value::Object(obj_rc) => {
                                    let obj = obj_rc.borrow();
                                    if let Some(Value::String(class_name)) = obj.get("__class_name") {
                                        format!("Class '{}' cannot accept {} argument(s)", class_name, arity)
                                    } else {
                                        format!("Can only call functions, got: {:?}", std::mem::discriminant(&actual_callee))
                                    }
                                }
                                _ => format!("Can only call functions, got: {:?}", std::mem::discriminant(&actual_callee)),
                            };
                            let error = ExceptionHandler::runtime_error(
                            &frames,error_msg, line);
                            match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error) {
                                Ok(()) => {
                                    // Exception handled, but we need to push null to maintain stack consistency
                                    // since the caller expects a return value
                                    stack::push(stack, Value::Null);
                                    return Ok(VMStatus::Continue);
                                }
                                Err(e) => return Err(e),
                            }
                        }
                    }
                }
                OpCode::Return => {
                    // Получаем возвращаемое значение (если есть)
                    // Проверяем стек относительно stack_start текущего фрейма
                    let frame = frames.last().unwrap();
                    
                    // Логирование для конструкторов
                    let is_constructor = frame.function.name.contains("::new_");
                    if is_constructor {
                        debug_println!("[DEBUG executor Return] constructor '{}' line {} Return (stack len {}, stack_start {})", frame.function.name, line, stack.len(), frame.stack_start);
                    }
                    
                    // ВАЖНО: Проверяем, что стек не ниже stack_start (это может произойти
                    // если цикл или другой statement оставил стек в некорректном состоянии)
                    // Используем безопасное извлечение без вызова stack::pop, чтобы избежать
                    // ошибки stack underflow, которая может быть неправильно обработана
                    let return_value = if stack.len() > frame.stack_start {
                        // Безопасно извлекаем значение напрямую, так как мы уже проверили
                        // что стек не пуст относительно stack_start
                        Some(stack.pop().unwrap_or(Value::Null))
                    } else {
                        // Стек пуст или ниже stack_start - возвращаем Null
                        // Это нормальная ситуация для функций без явного return
                        Some(Value::Null)
                    };
                    
                    // Отладка: логируем содержимое объекта после выполнения конструктора
                    if let Some(ref ret_val) = return_value {
                        if let Value::Object(obj_rc) = ret_val {
                            // Проверяем, является ли это конструктором (по имени функции)
                            if frame.function.name.contains("::new_") {
                                let key_count = obj_rc.borrow().len();
                                debug_println!("[DEBUG Return] constructor '{}' line {} returns Object ({} keys)", frame.function.name, line, key_count);
                            }
                        }
                    }
                    
                    let frames_count = frames.len();
                    if frames_count > 1 {
                        // Сохраняем результат в кэш, если функция кэшируемая
                        if let Some(frame) = frames.last() {
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
                        frames.pop();
                        
                        // Помещаем возвращаемое значение на стек для вызывающей функции
                        if let Some(value) = return_value {
                            stack::push(stack, value);
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
                    // Безопасно извлекаем значение из стека
                    // Проверяем стек относительно stack_start текущего фрейма
                    if let Some(frame) = frames.last() {
                        if stack.len() > frame.stack_start {
                            // Безопасно извлекаем значение напрямую, так как мы уже проверили
                            // что стек не пуст относительно stack_start
                            stack.pop();
                        }
                        // Если стек пуст или ниже stack_start, просто игнорируем Pop
                        // Это может произойти, если функция не вернула значение
                    } else {
                        // Нет frame - безопасно извлекаем значение если стек не пуст
                        if !stack.is_empty() {
                            stack.pop();
                        }
                    }
                }
                OpCode::MakeArray(count) => {
                    let mut elements = Vec::new();
                    for _ in 0..count {
                        elements.push(stack::pop(stack, frames, exception_handlers)?);
                    }
                    elements.reverse(); // Восстанавливаем правильный порядок
                    stack::push(stack, Value::Array(Rc::new(RefCell::new(elements))));
                }
                OpCode::MakeTuple(count) => {
                    let mut elements = Vec::new();
                    for _ in 0..count {
                        elements.push(stack::pop(stack, frames, exception_handlers)?);
                    }
                    elements.reverse(); // Восстанавливаем правильный порядок
                    stack::push(stack, Value::Tuple(Rc::new(RefCell::new(elements))));
                }
                OpCode::MakeObject(pair_count) => {
                    use std::collections::HashMap;
                    let mut object = HashMap::new();
                    // Извлекаем пары (ключ, значение) со стека
                    // На стеке: [key1, value1, key2, value2, ...]
                    // Извлекаем в обратном порядке: сначала последнюю пару
                    for _ in 0..pair_count {
                        let value = stack::pop(stack, frames, exception_handlers)?;
                        let key_value = stack::pop(stack, frames, exception_handlers)?;
                        let key = match key_value {
                            Value::String(s) => s,
                            _ => {
                                return Err(ExceptionHandler::runtime_error(
                                    &frames,
                                    "Object key must be a string".to_string(),
                                    line,
                                ));
                            }
                        };
                        object.insert(key, value);
                    }
                    stack::push(stack, Value::Object(Rc::new(RefCell::new(object))));
                }
                OpCode::MakeArrayDynamic => {
                    // Размер массива находится на стеке
                    let count_value = stack::pop(stack, frames, exception_handlers)?;
                    let count = match count_value {
                        Value::Number(n) => {
                            let idx = n as i64;
                            if idx < 0 {
                                let error = ExceptionHandler::runtime_error(
                            &frames,
                                    "Array size must be non-negative".to_string(),
                                    line,
                                );
                            match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error) {
                                Ok(()) => return Ok(VMStatus::Continue),
                                Err(e) => return Err(e),
                            }
                            }
                            idx as usize
                        }
                        _ => {
                            let error = ExceptionHandler::runtime_error(
                            &frames,
                                "Array size must be a number".to_string(),
                                line,
                            );
                            match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error) {
                                Ok(()) => return Ok(VMStatus::Continue),
                                Err(e) => return Err(e),
                            }
                        }
                    };
                    
                    let mut elements = Vec::new();
                    for _ in 0..count {
                        elements.push(stack::pop(stack, frames, exception_handlers)?);
                    }
                    elements.reverse(); // Восстанавливаем правильный порядок
                    stack::push(stack, Value::Array(Rc::new(RefCell::new(elements))));
                }
                OpCode::GetArrayLength => {
                    let array = stack::pop(stack, frames, exception_handlers)?;
                    match array {
                        Value::Array(arr) => {
                            stack::push(stack, Value::Number(arr.borrow().len() as f64));
                        }
                        Value::ColumnReference { table, column_name } => {
                            let table_ref = table.borrow();
                            if let Some(column) = table_ref.get_column(&column_name) {
                                stack::push(stack, Value::Number(column.len() as f64));
                            } else {
                                let error = ExceptionHandler::runtime_error(
                            &frames,
                                    format!("Column '{}' not found", column_name),
                                    line,
                                );
                            match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error) {
                                Ok(()) => return Ok(VMStatus::Continue),
                                Err(e) => return Err(e),
                            }
                            }
                        }
                        Value::Dataset(dataset) => {
                            let batch_size = dataset.borrow().batch_size();
                            stack::push(stack, Value::Number(batch_size as f64));
                        }
                        _ => {
                            let error = ExceptionHandler::runtime_error(
                            &frames,
                                "Expected array, column reference, or dataset for GetArrayLength".to_string(),
                                line,
                            );
                            match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error) {
                                Ok(()) => return Ok(VMStatus::Continue), // Исключение обработано, продолжаем выполнение
                                Err(e) => return Err(e), // Исключение не обработано
                            }
                        }
                    }
                }
                OpCode::GetArrayElement => {
                    let frame = frames.last().unwrap();
                    let current_ip = frame.ip - 1; // IP уже инкрементирован в step()
                    
                    let index_value = stack::pop(stack, frames, exception_handlers)?;
                    let container = stack::pop(stack, frames, exception_handlers)?;
                    
                    let container_type = match &container {
                        Value::Array(_) => "Array",
                        Value::Object(_) => "Object",
                        Value::Table(_) => "Table",
                        Value::Path(_) => "Path",
                        Value::Uuid(_, _) => "UUID",
                        _ => "Other",
                    };
                    let key_str = match &index_value {
                        Value::String(k) => k.clone(),
                        Value::Number(n) => format!("{}", n),
                        _ => format!("{:?}", index_value),
                    };
                    debug_println!("[DEBUG GetArrayElement] line {} IP {}: {} key '{}'", line, current_ip, container_type, key_str);
                    
                    match container {
                        Value::Array(arr) => {
                            let index = match index_value {
                                Value::Number(n) => {
                                    let idx = n as i64;
                                    if idx < 0 {
                                        let error = ExceptionHandler::runtime_error(
                            &frames,
                                            "Array index must be non-negative".to_string(),
                                            line,
                                        );
                            match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error) {
                                Ok(()) => return Ok(VMStatus::Continue),
                                Err(e) => return Err(e),
                            }
                                    }
                                    idx as usize
                                }
                                _ => {
                                    let error = ExceptionHandler::runtime_error(
                            &frames,
                                        "Array index must be a number".to_string(),
                                        line,
                                    );
                            match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error) {
                                Ok(()) => return Ok(VMStatus::Continue),
                                Err(e) => return Err(e),
                            }
                                }
                            };
                            
                            let arr_ref = arr.borrow();
                            if index >= arr_ref.len() {
                                let error = ExceptionHandler::runtime_error_with_type(
                                &frames,
                                    format!("Array index {} out of bounds (length: {})", index, arr_ref.len()),
                                    line,
                                    ErrorType::IndexError,
                                );
                            match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error) {
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
                                Value::Object(obj_rc) => Value::Object(obj_rc.clone()), // Object uses Rc<RefCell>, clone Rc
                                _ => element.clone(), // Простые типы клонируем
                            };
                            stack::push(stack, value);
                            return Ok(VMStatus::Continue);
                        }
                        Value::Tuple(tuple) => {
                            let index = match index_value {
                                Value::Number(n) => {
                                    let idx = n as i64;
                                    if idx < 0 {
                                        let error = ExceptionHandler::runtime_error(
                            &frames,
                                            "Tuple index must be non-negative".to_string(),
                                            line,
                                        );
                            match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error) {
                                Ok(()) => return Ok(VMStatus::Continue),
                                Err(e) => return Err(e),
                            }
                                    }
                                    idx as usize
                                }
                                _ => {
                                    let error = ExceptionHandler::runtime_error(
                            &frames,
                                        "Tuple index must be a number".to_string(),
                                        line,
                                    );
                            match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error) {
                                Ok(()) => return Ok(VMStatus::Continue),
                                Err(e) => return Err(e),
                            }
                                }
                            };
                            
                            let tuple_ref = tuple.borrow();
                            if index >= tuple_ref.len() {
                                let error = ExceptionHandler::runtime_error_with_type(
                                &frames,
                                    format!("Tuple index {} out of bounds (length: {})", index, tuple_ref.len()),
                                    line,
                                    ErrorType::IndexError,
                                );
                            match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error) {
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
                            stack::push(stack, value);
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
                                        stack::push(stack, Value::Array(Rc::new(RefCell::new(rows))));
                                    } else if property == "columns" {
                                        // Возвращаем массив имен колонок (заголовки)
                                        let columns: Vec<Value> = table_ref.headers.iter()
                                            .map(|header| Value::String(header.clone()))
                                            .collect();
                                        stack::push(stack, Value::Array(Rc::new(RefCell::new(columns))));
                                    } else {
                                        // Доступ к колонке по имени
                                        if table_ref.get_column(&property).is_some() {
                                            // Возвращаем ColumnReference для использования в relate()
                                            stack::push(stack, Value::ColumnReference {
                                                table: table.clone(),
                                                column_name: property,
                                            });
                                        } else {
                                            let error = ExceptionHandler::runtime_error_with_type(
                                &frames,
                                                format!("Column '{}' not found in table", property),
                                                line,
                                                ErrorType::KeyError,
                                            );
                            match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error) {
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
                                        let error = ExceptionHandler::runtime_error(
                            &frames,
                                            "Table row index must be non-negative".to_string(),
                                            line,
                                        );
                            match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error) {
                                Ok(()) => return Ok(VMStatus::Continue),
                                Err(e) => return Err(e),
                            }
                                    }
                                    let table_ref = table.borrow();
                                    if idx as usize >= table_ref.rows.len() {
                                        let error = ExceptionHandler::runtime_error_with_type(
                                &frames,
                                            format!("Row index {} out of bounds (length: {})", idx, table_ref.rows.len()),
                                            line,
                                            ErrorType::IndexError,
                                        );
                            match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error) {
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
                                        stack::push(stack, Value::Object(Rc::new(RefCell::new(row_dict))));
                                    } else {
                                        let error = ExceptionHandler::runtime_error_with_type(
                                &frames,
                                            format!("Row index {} out of bounds", idx),
                                            line,
                                            ErrorType::IndexError,
                                        );
                            match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error) {
                                Ok(()) => return Ok(VMStatus::Continue),
                                Err(e) => return Err(e),
                            }
                                    }
                                }
                                _ => {
                                    let error = ExceptionHandler::runtime_error(
                            &frames,
                                        "Table index must be a string (column name) or number (row index)".to_string(),
                                        line,
                                    );
                            match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error) {
                                Ok(()) => return Ok(VMStatus::Continue),
                                Err(e) => return Err(e),
                            }
                                }
                            }
                        }
                        Value::Object(map_rc) => {
                            let map = map_rc.borrow();
                            
                            // Check if this is a layer accessor object (has __neural_network key)
                            if map.contains_key("__neural_network") {
                                // This is a layer accessor - handle indexing to get layers
                                match index_value {
                                    Value::Number(n) => {
                                        let idx = n as i64;
                                        if idx < 0 {
                                            let error = ExceptionHandler::runtime_error(&frames,
                                                "Layer index must be non-negative".to_string(),
                                                line,
                                            );
                                            match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error) {
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
                                            stack::push(stack, result);
                                            return Ok(VMStatus::Continue);
                                        }
                                    }
                                    _ => {
                                        let error = ExceptionHandler::runtime_error(&frames,
                                            "Layer accessor index must be a number".to_string(),
                                            line,
                                        );
                                        match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error) {
                                            Ok(()) => return Ok(VMStatus::Continue),
                                            Err(e) => return Err(e),
                                        }
                                    }
                                }
                            }
                            
                            // Regular object access
                            match index_value {
                                Value::String(key) => {
                                    // Доступ к объекту класса: запрет чтения приватных переменных класса снаружи (ProtectError)
                                    if map.contains_key("__class_name") {
                                        if key != "model_config" {
                                            if let Some(Value::Array(private_vars_rc)) = map.get("__class_private_vars") {
                                                let private_vars = private_vars_rc.borrow();
                                                let is_private = private_vars.iter().any(|v| {
                                                    if let Value::String(s) = v { s.as_str() == key } else { false }
                                                });
                                                if is_private {
                                                    let class_name = map.get("__class_name")
                                                        .and_then(|v| if let Value::String(s) = v { Some(s.clone()) } else { None })
                                                        .unwrap_or_else(|| "?".to_string());
                                                    let msg = format!("Class variable '{}' is private in '{}' and cannot be accessed from outside the class", key, class_name);
                                                    let error = ExceptionHandler::runtime_error_with_type(
                                                        &frames,
                                                        msg,
                                                        line,
                                                        ErrorType::ProtectError,
                                                    );
                                                    match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error) {
                                                        Ok(()) => return Ok(VMStatus::Continue),
                                                        Err(e) => return Err(e),
                                                    }
                                                }
                                            }
                                        }
                                        // Class protected variables: allow only from this class or subclasses
                                        if let Some(Value::Array(protected_vars_rc)) = map.get("__class_protected_vars") {
                                            let protected_vars = protected_vars_rc.borrow();
                                            let is_protected = protected_vars.iter().any(|v| {
                                                if let Value::String(s) = v { s.as_str() == key } else { false }
                                            });
                                            if is_protected {
                                                let class_name_obj = map.get("__class_name");
                                                let in_hierarchy = if let Some(Value::String(ref obj_class_name)) = class_name_obj {
                                                    frames.iter().any(|f| {
                                                        let frame_class = f.function.name.split("::").next().unwrap_or("");
                                                        let frame_chain = get_superclass_chain(globals, global_names, frame_class);
                                                        frame_chain.iter().any(|c| c == obj_class_name)
                                                    })
                                                } else {
                                                    false
                                                };
                                                if !in_hierarchy {
                                                    let obj_class_name = map.get("__class_name")
                                                        .and_then(|v| if let Value::String(s) = v { Some(s.clone()) } else { None })
                                                        .unwrap_or_else(|| "?".to_string());
                                                    let frame_class_opt = frames.iter().rev()
                                                        .find_map(|f| {
                                                            if f.function.name.contains("::new_") || f.function.name.contains("::method_") {
                                                                f.function.name.split("::").next().map(String::from)
                                                            } else {
                                                                None
                                                            }
                                                        });
                                                    let msg = match &frame_class_opt {
                                                        Some(class) => format!("Class variable '{}' is protected in '{}' and cannot be accessed from subclass '{}'", key, obj_class_name, class),
                                                        None => format!("Class variable '{}' is protected in '{}' and cannot be accessed from outside the class", key, obj_class_name),
                                                    };
                                                    let error = ExceptionHandler::runtime_error_with_type(
                                                        &frames,
                                                        msg,
                                                        line,
                                                        ErrorType::ProtectError,
                                                    );
                                                    match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error) {
                                                        Ok(()) => return Ok(VMStatus::Continue),
                                                        Err(e) => return Err(e),
                                                    }
                                                }
                                            }
                                        }
                                    }
                                    // Instance private fields: allow only from the defining class (not subclasses)
                                    if let (Some(Value::String(ref class_name)), Some(Value::Array(private_fields_rc))) = (
                                        map.get("__class_name"),
                                        map.get("__private_fields"),
                                    ) {
                                        let private_fields_slice = private_fields_rc.borrow();
                                        let is_private_field = private_fields_slice.iter().any(|v| {
                                            if let Value::String(s) = v { s.as_str() == key } else { false }
                                        });
                                        if is_private_field {
                                            let defining_class = map.get("__private_field_defining_class")
                                                .and_then(|v| {
                                                    if let Value::Object(rc) = v {
                                                        rc.borrow().get(&key).and_then(|v| {
                                                            if let Value::String(s) = v { Some(s.clone()) } else { None }
                                                        })
                                                    } else { None }
                                                })
                                                .unwrap_or_else(|| class_name.clone());
                                            let in_defining_class = frames.iter().any(|f| {
                                                f.function.name.starts_with(&format!("{}::", defining_class))
                                            });
                                            if !in_defining_class {
                                                // Innermost frame that is a class method/constructor (::new_ or ::method_); skip <main>
                                                let frame_class_opt = frames.iter().rev()
                                                    .find_map(|f| {
                                                        if f.function.name.contains("::new_") || f.function.name.contains("::method_") {
                                                            f.function.name.split("::").next().map(String::from)
                                                        } else {
                                                            None
                                                        }
                                                    });
                                                let msg = match &frame_class_opt {
                                                    Some(class) => format!("Field '{}' is private in '{}' and cannot be accessed from subclass '{}'", key, defining_class, class),
                                                    None => format!("Field '{}' is private in '{}' and cannot be accessed from outside the class", key, defining_class),
                                                };
                                                let error = ExceptionHandler::runtime_error_with_type(
                                                    &frames,
                                                    msg,
                                                    line,
                                                    ErrorType::ProtectError,
                                                );
                                                match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error) {
                                                    Ok(()) => return Ok(VMStatus::Continue),
                                                    Err(e) => return Err(e),
                                                }
                                            }
                                        }
                                    }
                                    // Instance protected fields: allow from this class or any subclass (frame class in instance's chain)
                                    if let (Some(Value::String(ref instance_class)), Some(Value::Array(protected_fields_rc))) = (
                                        map.get("__class_name"),
                                        map.get("__protected_fields"),
                                    ) {
                                        let protected_fields_slice = protected_fields_rc.borrow();
                                        let is_protected_field = protected_fields_slice.iter().any(|v| {
                                            if let Value::String(s) = v { s.as_str() == key } else { false }
                                        });
                                        if is_protected_field {
                                            let instance_chain = get_superclass_chain(globals, global_names, instance_class);
                                            let in_hierarchy = frames.iter().any(|f| {
                                                let frame_class = f.function.name.split("::").next().unwrap_or("");
                                                instance_chain.iter().any(|c| c == frame_class)
                                            });
                                            if !in_hierarchy {
                                                let frame_class_opt = frames.iter().rev()
                                                    .find_map(|f| {
                                                        if f.function.name.contains("::new_") || f.function.name.contains("::method_") {
                                                            f.function.name.split("::").next().map(String::from)
                                                        } else {
                                                            None
                                                        }
                                                    });
                                                let msg = match &frame_class_opt {
                                                    Some(class) => format!("Field '{}' is protected in '{}' and cannot be accessed from subclass '{}'", key, instance_class, class),
                                                    None => format!("Field '{}' is protected in '{}' and cannot be accessed from outside the class", key, instance_class),
                                                };
                                                let error = ExceptionHandler::runtime_error_with_type(
                                                    &frames,
                                                    msg,
                                                    line,
                                                    ErrorType::ProtectError,
                                                );
                                                match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error) {
                                                    Ok(()) => return Ok(VMStatus::Continue),
                                                    Err(e) => return Err(e),
                                                }
                                            }
                                        }
                                    }
                                    if let Some(value) = map.get(&key) {
                                        stack::push(stack, value.clone());
                                    } else {
                                        debug_println!("[DEBUG GetArrayElement] key '{}' not found, pushing Null", key);
                                        // Для отсутствующих ключей возвращаем null
                                        // Это позволяет классам иметь поля с значениями по умолчанию
                                        // В арифметических операциях null будет преобразован в 0
                                        stack::push(stack, Value::Null);
                                    }
                                }
                                _ => {
                                    let error = ExceptionHandler::runtime_error(&frames,
                                        "Object index must be a string".to_string(),
                                        line,
                                    );
                            match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error) {
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
                                            stack::push(stack, Value::Array(Rc::new(RefCell::new(axes_array))));
                                        }
                                        _ => {
                                            let error = ExceptionHandler::runtime_error_with_type(&frames,
                                                format!("Figure has no property '{}'", key),
                                                line,
                                                ErrorType::KeyError,
                                            );
                                            match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error) {
                                                Ok(()) => return Ok(VMStatus::Continue),
                                                Err(e) => return Err(e),
                                            }
                                        }
                                    }
                                }
                                _ => {
                                    let error = ExceptionHandler::runtime_error(&frames,
                                        "Figure property access must use string key".to_string(),
                                        line,
                                    );
                                    match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error) {
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
                                            let error = ExceptionHandler::runtime_error_with_type(&frames,
                                                format!("Axis has no method '{}'", key),
                                                line,
                                                ErrorType::KeyError,
                                            );
                                            match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error) {
                                                Ok(()) => return Ok(VMStatus::Continue),
                                                Err(e) => return Err(e),
                                            }
                                        }
                                    };
                                    
                                    // Get method index from plot object (stored during registration)
                                    let method_index = if let Some(plot_obj) = globals.iter().find(|v| {
                                        if let Value::Object(map_rc) = v {
                                            map_rc.borrow().contains_key("image")
                                        } else {
                                            false
                                        }
                                    }) {
                                        if let Value::Object(map_rc) = plot_obj {
                                            let map = map_rc.borrow();
                                            let idx_key = match method_name {
                                                "imshow" => "__axis_imshow_idx",
                                                "set_title" => "__axis_set_title_idx",
                                                "axis" => "__axis_axis_idx",
                                                _ => {
                                                    let error = ExceptionHandler::runtime_error(&frames,
                                                        format!("Axis method '{}' not found", key),
                                                        line,
                                                    );
                                                    match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error) {
                                                        Ok(()) => return Ok(VMStatus::Continue),
                                                        Err(e) => return Err(e),
                                                    }
                                                }
                                            };
                                            if let Some(Value::Number(idx)) = map.get(idx_key) {
                                                *idx as usize
                                            } else {
                                                let error = ExceptionHandler::runtime_error(&frames,
                                                    format!("Axis method '{}' not registered", key),
                                                    line,
                                                );
                                                match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error) {
                                                    Ok(()) => return Ok(VMStatus::Continue),
                                                    Err(e) => return Err(e),
                                                }
                                            }
                                        } else {
                                            let error = ExceptionHandler::runtime_error(&frames,
                                                "Plot object not found".to_string(),
                                                line,
                                            );
                                            match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error) {
                                                Ok(()) => return Ok(VMStatus::Continue),
                                                Err(e) => return Err(e),
                                            }
                                        }
                                    } else {
                                        let error = ExceptionHandler::runtime_error(&frames,
                                            "Plot module not found".to_string(),
                                            line,
                                        );
                                        match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error) {
                                            Ok(()) => return Ok(VMStatus::Continue),
                                            Err(e) => return Err(e),
                                        }
                                    };
                                    // Return the native function
                                    // The compiler should arrange for axis to be passed as first argument
                                    stack::push(stack, Value::NativeFunction(method_index));
                                }
                                _ => {
                                    let error = ExceptionHandler::runtime_error(&frames,
                                        "Axis property access must use string key".to_string(),
                                        line,
                                    );
                                    match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error) {
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
                                            let error = ExceptionHandler::runtime_error_with_type(&frames,
                                                format!("Layer has no method '{}'. Available methods: freeze, unfreeze", key),
                                                line,
                                                ErrorType::KeyError,
                                            );
                                            match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error) {
                                                Ok(()) => return Ok(VMStatus::Continue),
                                                Err(e) => return Err(e),
                                            }
                                        }
                                    };
                                    
                                    // Get method index from ml object (stored during registration)
                                    let method_index = if let Some((&ml_idx, _)) = global_names.iter().find(|(_, name)| name.as_str() == "ml") {
                                        if ml_idx >= globals.len() {
                                            let error = ExceptionHandler::runtime_error(&frames,
                                                "ML module not found in globals".to_string(),
                                                line,
                                            );
                                            match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error) {
                                                Ok(()) => return Ok(VMStatus::Continue),
                                                Err(e) => return Err(e),
                                            }
                                        }
                                        
                                        match &globals[ml_idx] {
                                            Value::Object(map_rc) => {
                                                let map = map_rc.borrow();
                                                match map.get(function_name) {
                                                    Some(Value::NativeFunction(idx)) => *idx,
                                                    _ => {
                                                        let error = ExceptionHandler::runtime_error(&frames,
                                                            format!("Layer method '{}' not registered in ml module", key),
                                                            line,
                                                        );
                                                        match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error) {
                                                            Ok(()) => return Ok(VMStatus::Continue),
                                                            Err(e) => return Err(e),
                                                        }
                                                    }
                                                }
                                            }
                                            _ => {
                                                let error = ExceptionHandler::runtime_error(&frames,
                                                    "ML module is not an object".to_string(),
                                                    line,
                                                );
                                                match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error) {
                                                    Ok(()) => return Ok(VMStatus::Continue),
                                                    Err(e) => return Err(e),
                                                }
                                            }
                                        }
                                    } else {
                                        let error = ExceptionHandler::runtime_error(&frames,
                                            "ML module not found".to_string(),
                                            line,
                                        );
                                        match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error) {
                                            Ok(()) => return Ok(VMStatus::Continue),
                                            Err(e) => return Err(e),
                                        }
                                    };
                                    
                                    // Return the native function
                                    // The compiler should arrange for layer to be passed as first argument
                                    stack::push(stack, Value::NativeFunction(method_index));
                                }
                                _ => {
                                    let error = ExceptionHandler::runtime_error(&frames,
                                        "Layer property access must use string key".to_string(),
                                        line,
                                    );
                                    match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error) {
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
                                        let error = ExceptionHandler::runtime_error(&frames,
                                            "Column index must be non-negative".to_string(),
                                            line,
                                        );
                            match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error) {
                                Ok(()) => return Ok(VMStatus::Continue),
                                Err(e) => return Err(e),
                            }
                                    }
                                    idx as usize
                                }
                                _ => {
                                    let error = ExceptionHandler::runtime_error(&frames,
                                        "Column index must be a number".to_string(),
                                        line,
                                    );
                            match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error) {
                                Ok(()) => return Ok(VMStatus::Continue),
                                Err(e) => return Err(e),
                            }
                                }
                            };
                            
                            let table_ref = table.borrow();
                            if let Some(column) = table_ref.get_column(&column_name) {
                                if index >= column.len() {
                                    let error = ExceptionHandler::runtime_error_with_type(&frames,
                                        format!("Column index {} out of bounds (length: {})", index, column.len()),
                                        line,
                                        ErrorType::IndexError,
                                    );
                            match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error) {
                                Ok(()) => return Ok(VMStatus::Continue),
                                Err(e) => return Err(e),
                            }
                                }
                                stack::push(stack, column[index].clone());
                            } else {
                                let error = ExceptionHandler::runtime_error_with_type(&frames,
                                    format!("Column '{}' not found", column_name),
                                    line,
                                    ErrorType::KeyError,
                                );
                            match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error) {
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
                                            stack::push(stack, Value::Bool(path.is_file()));
                                        }
                                        "is_dir" => {
                                            stack::push(stack, Value::Bool(path.is_dir()));
                                        }
                                        "extension" => {
                                            if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
                                                stack::push(stack, Value::String(ext.to_string()));
                                            } else {
                                                stack::push(stack, Value::Null);
                                            }
                                        }
                                        "name" => {
                                            if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                                                stack::push(stack, Value::String(name.to_string()));
                                            } else {
                                                stack::push(stack, Value::Null);
                                            }
                                        }
                                        "parent" => {
                                            // Используем безопасную функцию для получения parent
                                            use crate::vm::natives::path::safe_path_parent;
                                            match safe_path_parent(&path) {
                                                Some(parent) => stack::push(stack, Value::Path(parent)),
                                                None => stack::push(stack, Value::Null),
                                            }
                                        }
                                        "exists" => {
                                            stack::push(stack, Value::Bool(path.exists()));
                                        }
                                        _ => {
                                            let error = ExceptionHandler::runtime_error(&frames,
                                                format!("Property '{}' not found on Path", property_name),
                                                line,
                                            );
                            match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error) {
                                Ok(()) => return Ok(VMStatus::Continue),
                                Err(e) => return Err(e),
                            }
                                        }
                                    }
                                }
                                _ => {
                                    let error = ExceptionHandler::runtime_error(&frames,
                                        "Path property access requires string index".to_string(),
                                        line,
                                    );
                            match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error) {
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
                                        let error = ExceptionHandler::runtime_error(&frames,
                                            "Dataset index must be non-negative".to_string(),
                                            line,
                                        );
                            match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error) {
                                Ok(()) => return Ok(VMStatus::Continue),
                                Err(e) => return Err(e),
                            }
                                    }
                                    idx as usize
                                }
                                _ => {
                                    let error = ExceptionHandler::runtime_error(&frames,
                                        "Dataset index must be a number".to_string(),
                                        line,
                                    );
                            match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error) {
                                Ok(()) => return Ok(VMStatus::Continue),
                                Err(e) => return Err(e),
                            }
                                }
                            };

                            let dataset_ref = dataset.borrow();
                            let batch_size = dataset_ref.batch_size();
                            
                            if index >= batch_size {
                                let error = ExceptionHandler::runtime_error_with_type(&frames,
                                    format!("Dataset index {} out of bounds (length: {})", index, batch_size),
                                    line,
                                    ErrorType::IndexError,
                                );
                            match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error) {
                                Ok(()) => return Ok(VMStatus::Continue),
                                Err(e) => return Err(e),
                            }
                            }

                            // Extract features for this sample
                            let num_features = dataset_ref.num_features();
                            let features_start = index * num_features;
                            let features_end = features_start + num_features;
                            // OPTIMIZATION: Use Vec::from for slice copy (slightly more efficient than to_vec)
                            let features_data: Vec<f32> = Vec::from(&dataset_ref.features().data[features_start..features_end]);
                            let features_tensor = Tensor::new(features_data, vec![num_features])
                                .map_err(|e| ExceptionHandler::runtime_error(&frames, format!("Failed to create features tensor: {}", e), line))?;

                            // Extract target for this sample
                            let num_targets = dataset_ref.num_targets();
                            let targets_start = index * num_targets;
                            let targets_end = targets_start + num_targets;
                            
                            // If target is a single value, return as Number; otherwise return as Tensor
                            let target_value = if num_targets == 1 {
                                Value::Number(dataset_ref.targets().data[targets_start] as f64)
                            } else {
                                // OPTIMIZATION: Use Vec::from for slice copy (slightly more efficient than to_vec)
                                let target_data: Vec<f32> = Vec::from(&dataset_ref.targets().data[targets_start..targets_end]);
                                let target_tensor = Tensor::new(target_data, vec![num_targets])
                                    .map_err(|e| ExceptionHandler::runtime_error(&frames, format!("Failed to create target tensor: {}", e), line))?;
                                Value::Tensor(Rc::new(RefCell::new(target_tensor)))
                            };

                            // Return [features, target] as array
                            let features_value = Value::Tensor(Rc::new(RefCell::new(features_tensor)));
                            let pair = vec![features_value, target_value];
                            stack::push(stack, Value::Array(Rc::new(RefCell::new(pair))));
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
                                            stack::push(stack, Value::Array(Rc::new(RefCell::new(shape_values))));
                                        }
                                        "data" => {
                                            let tensor_ref = tensor.borrow();
                                            let data_values: Vec<Value> = tensor_ref.data.iter()
                                                .map(|&d| Value::Number(d as f64))
                                                .collect();
                                            stack::push(stack, Value::Array(Rc::new(RefCell::new(data_values))));
                                        }
                                        "max_idx" => {
                                            // Return a bound method: push tensor first, then function
                                            // When called, the function will receive tensor as first argument
                                            use crate::ml::natives;
                                            let max_idx_fn_ptr = natives::native_max_idx as *const ();
                                            let method_index = natives.iter().position(|&f| {
                                                let fn_ptr = f as *const ();
                                                std::ptr::eq(fn_ptr, max_idx_fn_ptr)
                                            });
                                            
                                            if let Some(idx) = method_index {
                                                // Push tensor onto stack first (will be used as first argument)
                                                stack::push(stack, Value::Tensor(Rc::clone(&tensor)));
                                                // Push native function
                                                stack::push(stack, Value::NativeFunction(idx));
                                            } else {
                                                let error = ExceptionHandler::runtime_error(&frames,
                                                    "max_idx method not found".to_string(),
                                                    line,
                                                );
                                                match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error) {
                                                    Ok(()) => return Ok(VMStatus::Continue),
                                                    Err(e) => return Err(e),
                                                }
                                            }
                                        }
                                        "min_idx" => {
                                            // Return a bound method: push tensor first, then function
                                            use crate::ml::natives;
                                            let min_idx_fn_ptr = natives::native_min_idx as *const ();
                                            let method_index = natives.iter().position(|&f| {
                                                let fn_ptr = f as *const ();
                                                std::ptr::eq(fn_ptr, min_idx_fn_ptr)
                                            });
                                            
                                            if let Some(idx) = method_index {
                                                // Push tensor onto stack first (will be used as first argument)
                                                stack::push(stack, Value::Tensor(Rc::clone(&tensor)));
                                                // Push native function
                                                stack::push(stack, Value::NativeFunction(idx));
                                            } else {
                                                let error = ExceptionHandler::runtime_error(&frames,
                                                    "min_idx method not found".to_string(),
                                                    line,
                                                );
                                                match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error) {
                                                    Ok(()) => return Ok(VMStatus::Continue),
                                                    Err(e) => return Err(e),
                                                }
                                            }
                                        }
                                        _ => {
                                            let error = ExceptionHandler::runtime_error(&frames,
                                                format!("Property '{}' not found on Tensor. Available properties: 'shape', 'data', 'max_idx', 'min_idx'", property_name),
                                                line,
                                            );
                                            match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error) {
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
                                        let error = ExceptionHandler::runtime_error(&frames,
                                            "Tensor index must be non-negative".to_string(),
                                            line,
                                        );
                                        match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error) {
                                            Ok(()) => return Ok(VMStatus::Continue),
                                            Err(e) => return Err(e),
                                        }
                                    }
                                    let tensor_ref = tensor.borrow();
                                    let index = idx as usize;
                                    
                                    // For 1D tensors, return scalar (backward compatibility)
                                    if tensor_ref.ndim() == 1 {
                                        if index >= tensor_ref.shape[0] {
                                            let error = ExceptionHandler::runtime_error_with_type(&frames,
                                                format!("Tensor index {} out of bounds (size: {})", index, tensor_ref.shape[0]),
                                                line,
                                                ErrorType::IndexError,
                                            );
                                            match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error) {
                                                Ok(()) => return Ok(VMStatus::Continue),
                                                Err(e) => return Err(e),
                                            }
                                        }
                                        stack::push(stack, Value::Number(tensor_ref.data[index] as f64));
                                    } else {
                                        // For 2D+ tensors, return a slice (row) along first dimension
                                        match tensor_ref.get_row(index) {
                                            Ok(slice_tensor) => {
                                                stack::push(stack, Value::Tensor(Rc::new(RefCell::new(slice_tensor))));
                                            }
                                            Err(e) => {
                                                let error = ExceptionHandler::runtime_error_with_type(&frames,
                                                    e,
                                                    line,
                                                    ErrorType::IndexError,
                                                );
                                                match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error) {
                                                    Ok(()) => return Ok(VMStatus::Continue),
                                                    Err(e) => return Err(e),
                                                }
                                            }
                                        }
                                    }
                                }
                                _ => {
                                    let error = ExceptionHandler::runtime_error(&frames,
                                        "Tensor property access requires string key (e.g., 'shape', 'data') or numeric index".to_string(),
                                        line,
                                    );
                                    match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error) {
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
                                        stack::push(stack, Value::Object(Rc::new(RefCell::new(layer_accessor))));
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
                                            let error = ExceptionHandler::runtime_error_with_type(&frames,
                                                format!("NeuralNetwork has no method '{}'. Available methods: train, train_sh, save, device, get_device, layers", key),
                                                line,
                                                ErrorType::KeyError,
                                            );
                                            match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error) {
                                                Ok(()) => return Ok(VMStatus::Continue),
                                                Err(e) => return Err(e),
                                            }
                                        }
                                    };
                                    
                                    // Get method index from ml object (stored during registration)
                                    // Find ml module using global_names
                                    let method_index = if let Some((&ml_idx, _)) = global_names.iter().find(|(_, name)| name.as_str() == "ml") {
                                        // Ensure globals vector is large enough
                                        if ml_idx >= globals.len() {
                                            let error = ExceptionHandler::runtime_error(&frames,
                                                "ML module not found in globals".to_string(),
                                                line,
                                            );
                                            match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error) {
                                                Ok(()) => return Ok(VMStatus::Continue),
                                                Err(e) => return Err(e),
                                            }
                                        }
                                        
                                        match &globals[ml_idx] {
                                            Value::Object(map_rc) => {
                                                let map = map_rc.borrow();
                                                match map.get(function_name) {
                                                    Some(Value::NativeFunction(idx)) => *idx,
                                                    _ => {
                                                        let error = ExceptionHandler::runtime_error(&frames,
                                                            format!("NeuralNetwork method '{}' not registered in ml module", key),
                                                            line,
                                                        );
                                                        match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error) {
                                                            Ok(()) => return Ok(VMStatus::Continue),
                                                            Err(e) => return Err(e),
                                                        }
                                                    }
                                                }
                                            }
                                            _ => {
                                                let error = ExceptionHandler::runtime_error(&frames,
                                                    "ML module is not an object".to_string(),
                                                    line,
                                                );
                                                match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error) {
                                                    Ok(()) => return Ok(VMStatus::Continue),
                                                    Err(e) => return Err(e),
                                                }
                                            }
                                        }
                                    } else {
                                        let error = ExceptionHandler::runtime_error(&frames,
                                            "ML module not found".to_string(),
                                            line,
                                        );
                                        match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error) {
                                            Ok(()) => return Ok(VMStatus::Continue),
                                            Err(e) => return Err(e),
                                        }
                                    };
                                    // Return the native function
                                    // The compiler should arrange for neural network to be passed as first argument
                                    stack::push(stack, Value::NativeFunction(method_index));
                                }
                                _ => {
                                    let error = ExceptionHandler::runtime_error(&frames,
                                        "NeuralNetwork property access must use string key".to_string(),
                                        line,
                                    );
                                    match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error) {
                                        Ok(()) => return Ok(VMStatus::Continue),
                                        Err(e) => return Err(e),
                                    }
                                }
                            }
                            return Ok(VMStatus::Continue);
                        }
                        Value::Null => {
                            let error = ExceptionHandler::runtime_error(
                            &frames,
                                "Cannot access element of null value".to_string(),
                                line,
                            );
                            match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error) {
                                Ok(()) => return Ok(VMStatus::Continue),
                                Err(e) => return Err(e),
                            }
                        }
                        _ => {
                            let error = ExceptionHandler::runtime_error(
                            &frames,
                                "Expected array, tuple, column reference, table, object, path, dataset, tensor, or neural network for GetArrayElement".to_string(),
                                line,
                            );
                            match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error) {
                                Ok(()) => return Ok(VMStatus::Continue),
                                Err(e) => return Err(e),
                            }
                        }
                    }
                    return Ok(VMStatus::Continue);
                }
                OpCode::SetArrayElement => {
                    // Установка элемента массива/объекта: [value, index/key, container]
                    // Стек: последний элемент наверху, поэтому порядок обратный
                    let container = stack::pop(stack, frames, exception_handlers)?;
                    let index_value = stack::pop(stack, frames, exception_handlers)?;
                    let value = stack::pop(stack, frames, exception_handlers)?;
                    
                    // Универсальное логирование для всех случаев SetArrayElement
                    let container_type = match &container {
                        Value::Array(_) => "Array",
                        Value::Object(_) => "Object",
                        Value::Table(_) => "Table",
                        _ => "Other",
                    };
                    let key_str = match &index_value {
                        Value::String(k) => k.clone(),
                        Value::Number(n) => format!("{}", n),
                        _ => format!("{:?}", index_value),
                    };
                    let value_type_str = match &value {
                        Value::Function(fn_idx) => {
                            if *fn_idx < functions.len() {
                                format!("Function({}, имя: '{}')", fn_idx, functions[*fn_idx].name)
                            } else {
                                format!("Function({}, OUT OF BOUNDS!)", fn_idx)
                            }
                        },
                        Value::NativeFunction(_) => "NativeFunction".to_string(),
                        _ => format!("{:?}", value),
                    };
                    debug_println!("[DEBUG SetArrayElement] line {}, {} key='{}' value={}", line, container_type, key_str, value_type_str);
                    
                    match container {
                        Value::Array(arr) => {
                            let index = match index_value {
                                Value::Number(n) => {
                                    let idx = n as i64;
                                    if idx < 0 {
                                        let error = ExceptionHandler::runtime_error(
                                            &frames,
                                            "Array index must be non-negative".to_string(),
                                            line,
                                        );
                                        match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error) {
                                            Ok(()) => return Ok(VMStatus::Continue),
                                            Err(e) => return Err(e),
                                        }
                                    }
                                    idx as usize
                                }
                                _ => {
                                    let error = ExceptionHandler::runtime_error(
                                        &frames,
                                        "Array index must be a number".to_string(),
                                        line,
                                    );
                                    match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error) {
                                        Ok(()) => return Ok(VMStatus::Continue),
                                        Err(e) => return Err(e),
                                    }
                                }
                            };
                            
                            let mut arr_ref = arr.borrow_mut();
                            if index >= arr_ref.len() {
                                // Расширяем массив, если индекс выходит за границы
                                arr_ref.resize(index + 1, Value::Null);
                            }
                            arr_ref[index] = value;
                            // Возвращаем обновленный массив (через Rc)
                            stack::push(stack, Value::Array(arr.clone()));
                            return Ok(VMStatus::Continue);
                        }
                        Value::Object(obj_rc) => {
                            // Для объектов индекс должен быть строкой
                            let key = match index_value {
                                Value::String(key) => key,
                                _ => {
                                    let error = ExceptionHandler::runtime_error(
                                        &frames,
                                        "Object key must be a string".to_string(),
                                        line,
                                    );
                                    match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error) {
                                        Ok(()) => return Ok(VMStatus::Continue),
                                        Err(e) => return Err(e),
                                    }
                                }
                            };
                            // Class private variables: forbid write from outside (ProtectError)
                            let is_class_private_var = {
                                let map = obj_rc.borrow();
                                map.contains_key("__class_name") && key != "model_config"
                                    && map.get("__class_private_vars").and_then(|v| {
                                        if let Value::Array(rc) = v {
                                            Some(rc.borrow().iter().any(|v| {
                                                if let Value::String(s) = v { s.as_str() == key } else { false }
                                            }))
                                        } else {
                                            None
                                        }
                                    }).unwrap_or(false)
                            };
                            if is_class_private_var {
                                let class_name = obj_rc.borrow().get("__class_name")
                                    .and_then(|v| if let Value::String(s) = v { Some(s.clone()) } else { None })
                                    .unwrap_or_else(|| "?".to_string());
                                let msg = format!("Class variable '{}' is private in '{}' and cannot be accessed from outside the class", key, class_name);
                                let error = ExceptionHandler::runtime_error_with_type(
                                    &frames,
                                    msg,
                                    line,
                                    ErrorType::ProtectError,
                                );
                                match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error) {
                                    Ok(()) => return Ok(VMStatus::Continue),
                                    Err(e) => return Err(e),
                                }
                            }
                            // Instance private fields: allow write only from the defining class (not subclasses)
                            let (is_instance_private_denied, private_error_msg) = {
                                let map = obj_rc.borrow();
                                if let (Some(Value::String(ref class_name)), Some(Value::Array(private_fields_rc))) = (
                                    map.get("__class_name"),
                                    map.get("__private_fields"),
                                ) {
                                    let is_private = private_fields_rc.borrow().iter().any(|v| {
                                        if let Value::String(s) = v { s.as_str() == key } else { false }
                                    });
                                    if !is_private {
                                        (false, String::new())
                                    } else {
                                        let defining_class = map.get("__private_field_defining_class")
                                            .and_then(|v| {
                                                if let Value::Object(rc) = v {
                                                    rc.borrow().get(&key).and_then(|v| {
                                                        if let Value::String(s) = v { Some(s.clone()) } else { None }
                                                    })
                                                } else { None }
                                            })
                                            .unwrap_or_else(|| class_name.clone());
                                        let in_defining_class = frames.iter().any(|f| {
                                            f.function.name.starts_with(&format!("{}::", defining_class))
                                        });
                                        if in_defining_class {
                                            (false, String::new())
                                        } else {
                                            let frame_class_opt = frames.iter().rev()
                                                .find_map(|f| {
                                                    if f.function.name.contains("::new_") || f.function.name.contains("::method_") {
                                                        f.function.name.split("::").next().map(String::from)
                                                    } else {
                                                        None
                                                    }
                                                });
                                            let msg = match &frame_class_opt {
                                                Some(class) => format!("Field '{}' is private in '{}' and cannot be accessed from subclass '{}'", key, defining_class, class),
                                                None => format!("Field '{}' is private in '{}' and cannot be accessed from outside the class", key, defining_class),
                                            };
                                            (true, msg)
                                        }
                                    }
                                } else {
                                    (false, String::new())
                                }
                            };
                            if is_instance_private_denied {
                                let error = ExceptionHandler::runtime_error_with_type(
                                    &frames,
                                    private_error_msg,
                                    line,
                                    ErrorType::ProtectError,
                                );
                                match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error) {
                                    Ok(()) => return Ok(VMStatus::Continue),
                                    Err(e) => return Err(e),
                                }
                            }
                            // Class protected variables: allow write only from this class or subclasses
                            let (is_class_protected_denied, class_protected_msg) = {
                                let map = obj_rc.borrow();
                                let is_protected_var = map.contains_key("__class_name") && key != "model_config"
                                    && map.get("__class_protected_vars").and_then(|v| {
                                        if let Value::Array(rc) = v {
                                            Some(rc.borrow().iter().any(|v| {
                                                if let Value::String(s) = v { s.as_str() == key } else { false }
                                            }))
                                        } else {
                                            None
                                        }
                                    }).unwrap_or(false);
                                if !is_protected_var {
                                    (false, String::new())
                                } else {
                                    let obj_class = map.get("__class_name").and_then(|v| if let Value::String(s) = v { Some(s.clone()) } else { None });
                                    let in_hierarchy = obj_class.as_ref().map(|obj_class| {
                                        frames.iter().any(|f| {
                                            let frame_class = f.function.name.split("::").next().unwrap_or("");
                                            let frame_chain = get_superclass_chain(globals, global_names, frame_class);
                                            frame_chain.iter().any(|c| c == obj_class)
                                        })
                                    }).unwrap_or(false);
                                    if in_hierarchy {
                                        (false, String::new())
                                    } else {
                                        let frame_class_opt = frames.iter().rev()
                                            .find_map(|f| {
                                                if f.function.name.contains("::new_") || f.function.name.contains("::method_") {
                                                    f.function.name.split("::").next().map(String::from)
                                                } else {
                                                    None
                                                }
                                            });
                                        let class_name = obj_class.as_deref().unwrap_or("?");
                                        let msg = match &frame_class_opt {
                                            Some(class) => format!("Class variable '{}' is protected in '{}' and cannot be accessed from subclass '{}'", key, class_name, class),
                                            None => format!("Class variable '{}' is protected in '{}' and cannot be accessed from outside the class", key, class_name),
                                        };
                                        (true, msg)
                                    }
                                }
                            };
                            if is_class_protected_denied {
                                let error = ExceptionHandler::runtime_error_with_type(
                                    &frames,
                                    class_protected_msg,
                                    line,
                                    ErrorType::ProtectError,
                                );
                                match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error) {
                                    Ok(()) => return Ok(VMStatus::Continue),
                                    Err(e) => return Err(e),
                                }
                            }
                            // Instance protected fields: allow write only from this class or subclasses
                            let (is_instance_protected_denied, instance_protected_msg) = {
                                let map = obj_rc.borrow();
                                if let (Some(Value::String(ref instance_class)), Some(Value::Array(protected_fields_rc))) = (
                                    map.get("__class_name"),
                                    map.get("__protected_fields"),
                                ) {
                                    let is_protected = protected_fields_rc.borrow().iter().any(|v| {
                                        if let Value::String(s) = v { s.as_str() == key } else { false }
                                    });
                                    let in_hierarchy = {
                                        let instance_chain = get_superclass_chain(globals, global_names, instance_class);
                                        frames.iter().any(|f| {
                                            let frame_class = f.function.name.split("::").next().unwrap_or("");
                                            instance_chain.iter().any(|c| c == frame_class)
                                        })
                                    };
                                    if is_protected && !in_hierarchy {
                                        let frame_class_opt = frames.iter().rev()
                                            .find_map(|f| {
                                                if f.function.name.contains("::new_") || f.function.name.contains("::method_") {
                                                    f.function.name.split("::").next().map(String::from)
                                                } else {
                                                    None
                                                }
                                            });
                                        let msg = match &frame_class_opt {
                                            Some(class) => format!("Field '{}' is protected in '{}' and cannot be accessed from subclass '{}'", key, instance_class, class),
                                            None => format!("Field '{}' is protected in '{}' and cannot be accessed from outside the class", key, instance_class),
                                        };
                                        (true, msg)
                                    } else {
                                        (false, String::new())
                                    }
                                } else {
                                    (false, String::new())
                                }
                            };
                            if is_instance_protected_denied {
                                let error = ExceptionHandler::runtime_error_with_type(
                                    &frames,
                                    instance_protected_msg,
                                    line,
                                    ErrorType::ProtectError,
                                );
                                match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error) {
                                    Ok(()) => return Ok(VMStatus::Continue),
                                    Err(e) => return Err(e),
                                }
                            }
                            obj_rc.borrow_mut().insert(key.clone(), value);
                            let keys_after: Vec<String> = obj_rc.borrow().keys().cloned().collect();
                            debug_println!("[DEBUG SetArrayElement] object key '{}' set, keys now: {}", key, keys_after.len());
                            stack::push(stack, Value::Object(obj_rc.clone()));
                            return Ok(VMStatus::Continue);
                        }
                        _ => {
                            let error = ExceptionHandler::runtime_error(
                                &frames,
                                format!("SetArrayElement only supports arrays and objects, got: {:?}", container),
                                line,
                            );
                            match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error) {
                                Ok(()) => return Ok(VMStatus::Continue),
                                Err(e) => return Err(e),
                            }
                        }
                    }
                }
                OpCode::Clone => {
                    // Глубокое клонирование значения на стеке
                    let value = stack::pop(stack, frames, exception_handlers)?;
                    let cloned = value.clone(); // Используем реализованный Clone для Value
                    stack::push(stack, cloned);
                    return Ok(VMStatus::Continue);
                }
                OpCode::BeginTry(handler_index) => {
                    // Начало try блока - загружаем обработчик из chunk
                    let frame = frames.last().unwrap();
                    let chunk = &frame.function.chunk;
                    
                    // Загружаем информацию об обработчике из chunk
                    if handler_index < chunk.exception_handlers.len() {
                        let handler_info = &chunk.exception_handlers[handler_index];
                        
                        // Копируем таблицу типов ошибок в VM (если еще не скопирована)
                        if error_type_table.is_empty() {
                            *error_type_table = chunk.error_type_table.clone();
                        }
                        
                        // Сохраняем текущую высоту стека
                        let stack_height = stack.len();
                        
                        // Создаем обработчик с информацией из chunk
                        let frame_index = frames.len() - 1;
                        let handler = ExceptionHandler {
                            catch_ips: handler_info.catch_ips.clone(),
                            error_types: handler_info.error_types.clone(),
                            error_var_slots: handler_info.error_var_slots.clone(),
                            else_ip: handler_info.else_ip,
                            finally_ip: handler_info.finally_ip,
                            stack_height,
                            had_error: false,
                            frame_index,
                        };
                        exception_handlers.push(handler);
                    } else {
                        // Если обработчик не найден, создаем пустой (fallback)
                        let stack_height = stack.len();
                        let frame_index = frames.len() - 1;
                        let handler = ExceptionHandler {
                            catch_ips: Vec::new(),
                            error_types: Vec::new(),
                            error_var_slots: Vec::new(),
                            else_ip: None,
                            finally_ip: None,
                            stack_height,
                            had_error: false,
                            frame_index,
                        };
                        exception_handlers.push(handler);
                    }
                    return Ok(VMStatus::Continue);
                }
                OpCode::EndTry => {
                    // Конец try блока - если выполнение дошло сюда без ошибок
                    // Проверяем, была ли ошибка
                    if let Some(handler) = exception_handlers.last_mut() {
                        // Если не было ошибки и есть else блок, переходим к нему
                        if !handler.had_error {
                            if let Some(else_ip) = handler.else_ip {
                                let frame = frames.last_mut().unwrap();
                                frame.ip = else_ip;
                            }
                        }
                        // Удаляем обработчик из стека
                        exception_handlers.pop();
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
                    let error_value = stack::pop(stack, frames, exception_handlers)?;
                    
                    // Преобразуем значение в строку
                    let error_message = error_value.to_string();
                    
                    // Создаем LangError
                    let error = LangError::runtime_error(error_message, line);
                    
                    // Пытаемся найти обработчик исключения
                    match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error) {
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
                    exception_handlers.pop();
                    return Ok(VMStatus::Continue);
                }
    }

    Ok(VMStatus::Continue)
}

