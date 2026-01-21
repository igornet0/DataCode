// Opcode execution for VM

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

/// Execute one step of the VM - get next instruction and execute it
pub fn step(
    frames: &mut Vec<CallFrame>,
) -> Result<Option<(OpCode, usize)>, LangError> {
    let frame = match frames.last_mut() {
        Some(f) => f,
        None => return Ok(None),
    };

    if frame.ip >= frame.function.chunk.code.len() {
        frames.pop();
        return Ok(None);
    }

    let ip = frame.ip;
    let instruction = frame.function.chunk.code[ip].clone();
    let line = frame.function.chunk.get_line(ip);
    frame.ip += 1;

    Ok(Some((instruction, line)))
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
    vm_ptr: *mut crate::vm::vm::Vm,
) -> Result<VMStatus, LangError> {
    let frame = frames.last_mut().unwrap();
    
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
                    
                    // Register the module
                    modules::register_module(&module_name, natives, globals, global_names)?;
                    loaded_modules.insert(module_name);
                    return Ok(VMStatus::Continue);
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
                        modules::register_module(&module_name, natives, globals, global_names)?;
                        loaded_modules.insert(module_name.clone());
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
                    
                    let module_object_ref = match &globals[module_global_index] {
                        Value::Object(map) => map,
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
                                    if let Some(value) = module_object.get(&item_str) {
                                        // Register the name in globals
                                        let global_index = if let Some(&idx) = global_names.iter().find(|(_, n)| n.as_str() == item_str.as_str()).map(|(idx, _)| idx) {
                                            idx
                                        } else {
                                            let idx = globals.len();
                                            globals.push(value.clone());
                                            global_names.insert(idx, item_str.clone());
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
                    // Получаем функцию со стека
                    // Используем stack::pop для правильного номера строки при ошибке
                    let function_value = stack::pop(stack, frames, exception_handlers)?;
                    match function_value {
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
                                
                                for _ in 0..arity {
                                    // Безопасно извлекаем аргументы напрямую, так как мы уже проверили стек
                                    args.push(stack.pop().unwrap_or(Value::Null));
                                }
                                args.reverse(); // Теперь args[0] - первый аргумент
                            }
                            // Если arity == 0, args остается пустым вектором
                            
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
                            if native_index >= natives.len() {
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
                            
                            // Специальная обработка для методов тензора max_idx и min_idx
                            // Эти методы могут быть вызваны как tensor.max_idx() с arity=0,
                            // но тензор уже находится на стеке перед функцией
                            use crate::ml::natives;
                            let is_max_idx = std::ptr::eq(
                                natives[native_index] as *const (),
                                natives::native_max_idx as *const ()
                            );
                            let is_min_idx = std::ptr::eq(
                                natives[native_index] as *const (),
                                natives::native_min_idx as *const ()
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
                            
                            // Вызываем нативную функцию
                            let native_fn = natives[native_index];
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
                            let args = vec![function_value.clone(), input_value];
                            let result = natives::native_nn_forward(&args);
                            
                            stack::push(stack, result);
                            return Ok(VMStatus::Continue);
                        }
                        _ => {
                            // Try to provide more helpful error message
                            let error_msg = match &function_value {
                                Value::Null => "Cannot call null - function may not be imported or defined".to_string(),
                                _ => format!("Can only call functions, got: {:?}", std::mem::discriminant(&function_value)),
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
                    stack::push(stack, Value::Object(object));
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
                    let index_value = stack::pop(stack, frames, exception_handlers)?;
                    let container = stack::pop(stack, frames, exception_handlers)?;
                    
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
                                Value::Object(_) => element.clone(), // Object uses HashMap, clone is needed
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
                                        stack::push(stack, Value::Object(row_dict));
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
                        Value::Object(map) => {
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
                                    if let Some(value) = map.get(&key) {
                                        stack::push(stack, value.clone());
                                    } else {
                                        let error = ExceptionHandler::runtime_error_with_type(&frames,
                                            format!("Key '{}' not found in object", key),
                                            line,
                                            ErrorType::KeyError,
                                        );
                            match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error) {
                                Ok(()) => return Ok(VMStatus::Continue),
                                Err(e) => return Err(e),
                            }
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
                                            Value::Object(map) => {
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
                                        stack::push(stack, Value::Object(layer_accessor));
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
                                            Value::Object(map) => {
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
