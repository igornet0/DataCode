// Function call operations for VM

use crate::common::{error::LangError, value::Value};
use crate::vm::frame::CallFrame;

/// Setup a function call by creating a new call frame and setting up captured variables
/// Returns the cached result if available, or None if execution is needed
pub fn setup_function_call(
    function_index: usize,
    args: &[Value],
    functions: &[crate::bytecode::Function],
    stack: &mut Vec<Value>,
    frames: &mut Vec<CallFrame>,
    error_type_table: &mut Vec<String>,
) -> Result<Option<Value>, LangError> {
    if function_index >= functions.len() {
        return Err(LangError::runtime_error(
            format!("Function index {} out of bounds", function_index),
            0,
        ));
    }

    let function = functions[function_index].clone();

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
                    return Ok(Some(cached_result.clone()));
                }
                drop(cache);
            }
        }
    }

    // Создаем новый CallFrame
    let stack_start = stack.len();
    let mut new_frame = if function.is_cached {
        CallFrame::new_with_cache(function.clone(), stack_start, args.to_vec())
    } else {
        CallFrame::new(function.clone(), stack_start)
    };

    // Копируем таблицу типов ошибок из chunk функции в VM
    if !function.chunk.error_type_table.is_empty() {
        *error_type_table = function.chunk.error_type_table.clone();
    }

    // Копируем захваченные переменные из родительских frames (если есть)
    if !frames.is_empty() && !function.captured_vars.is_empty() {
        for captured_var in &function.captured_vars {
            if captured_var.local_slot_index >= new_frame.slots.len() {
                new_frame.slots.resize(captured_var.local_slot_index + 1, Value::Null);
            }
            
            let ancestor_index = frames.len().saturating_sub(1 + captured_var.ancestor_depth);
            
            if ancestor_index < frames.len() {
                let ancestor_frame = &frames[ancestor_index];
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
    frames.push(new_frame);    Ok(None) // No cached result, execution needed
}
