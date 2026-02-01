// Function call operations for VM

use crate::debug_println;
use crate::common::{error::{LangError, ErrorType}, value::Value};
use crate::vm::frame::CallFrame;

/// Проверяет, соответствует ли значение одному типу
fn check_single_type(value: &Value, type_name: &str) -> bool {
    let type_name_lower = type_name.to_lowercase();
    match (value, type_name_lower.as_str()) {
        // Числовые типы
        (Value::Number(n), "int" | "integer") => n.fract() == 0.0,
        (Value::Number(_), "float" | "num" | "number") => true,
        // Строковые типы
        (Value::String(_), "str" | "string") => true,
        // Булевы типы
        (Value::Bool(_), "bool" | "boolean") => true,
        // Коллекции
        (Value::Array(_), "array" | "list") => true,
        (Value::Tuple(_), "tuple") => true,
        (Value::Object(_), "object" | "dict" | "dictionary") => true,
        (Value::Table(_), "table") => true,
        // Специальные типы
        (Value::Null, "null" | "none") => true,
        (Value::Path(_), "path") => true,
        (Value::Function(_) | Value::NativeFunction(_), "function" | "fn") => true,
        // ML типы
        (Value::Tensor(_), "tensor") => true,
        (Value::Graph(_), "graph") => true,
        (Value::Dataset(_), "dataset") => true,
        (Value::NeuralNetwork(_), "neural_network" | "neuralnetwork") => true,
        (Value::Sequential(_), "sequential") => true,
        (Value::Layer(_), "layer") => true,
        // Оптимизаторы
        (Value::LinearRegression(_), "linear_regression" | "linearregression") => true,
        (Value::SGD(_), "sgd") => true,
        (Value::Momentum(_), "momentum") => true,
        (Value::NAG(_), "nag") => true,
        (Value::Adagrad(_), "adagrad") => true,
        (Value::RMSprop(_), "rmsprop") => true,
        (Value::Adam(_), "adam") => true,
        (Value::AdamW(_), "adamw") => true,
        // Графические типы
        (Value::Window(_), "window") => true,
        (Value::Image(_), "image") => true,
        (Value::Figure(_), "figure") => true,
        (Value::Axis(_), "axis") => true,
        (Value::ColumnReference { .. }, "column") => true,
        _ => false,
    }
}

/// Проверяет, соответствует ли значение хотя бы одному из типов (union типы)
pub fn check_type_value(value: &Value, type_names: &[String]) -> bool {
    type_names.iter().any(|type_name| check_single_type(value, type_name))
}

/// Форматирует список типов для сообщения об ошибке
pub fn format_type_names(type_names: &[String]) -> String {
    type_names.join(" | ")
}

/// Возвращает имя типа значения
pub fn get_type_name_value(value: &Value) -> &'static str {
    match value {
        Value::Number(n) => {
            if n.fract() == 0.0 { "int" } else { "float" }
        }
        Value::Bool(_) => "bool",
        Value::String(_) => "str",
        Value::Array(_) => "array",
        Value::Tuple(_) => "tuple",
        Value::Object(_) => "object",
        Value::Table(_) => "table",
        Value::Null => "null",
        Value::Path(_) => "path",
        Value::Uuid(_, _) => "uuid",
        Value::Function(_) | Value::NativeFunction(_) => "function",
        Value::Tensor(_) => "tensor",
        Value::Graph(_) => "graph",
        Value::Dataset(_) => "dataset",
        Value::NeuralNetwork(_) => "neural_network",
        Value::Sequential(_) => "sequential",
        Value::Layer(_) => "layer",
        Value::LinearRegression(_) => "linear_regression",
        Value::SGD(_) => "sgd",
        Value::Momentum(_) => "momentum",
        Value::NAG(_) => "nag",
        Value::Adagrad(_) => "adagrad",
        Value::RMSprop(_) => "rmsprop",
        Value::Adam(_) => "adam",
        Value::AdamW(_) => "adamw",
        Value::Window(_) => "window",
        Value::Image(_) => "image",
        Value::Figure(_) => "figure",
        Value::Axis(_) => "axis",
        Value::ColumnReference { .. } => "column",
        Value::Ellipsis => "ellipsis",
    }
}

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
            format!("Function index {} out of bounds (functions.len() = {})", function_index, functions.len()),
            0,
        ));
    }
    
    let function = functions[function_index].clone();
    
    debug_println!("[DEBUG setup_function_call] Вызываем функцию с индексом {}, имя: '{}', arity: {}, получено аргументов: {} (всего функций в VM: {})", 
        function_index, function.name, function.arity, args.len(), functions.len());

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

    // Проверяем типы аргументов, если указаны аннотации типов
    for (i, (arg, expected_types)) in args.iter().zip(&function.param_types).enumerate() {
        if let Some(type_names) = expected_types {
            if !check_type_value(arg, type_names) {
                let param_name = function.param_names.get(i)
                    .map(|s| s.as_str())
                    .unwrap_or("unknown");
                return Err(LangError::runtime_error_with_type(
                    format!(
                        "Argument '{}' expected type '{}', got '{}'",
                        param_name, format_type_names(type_names), get_type_name_value(arg)
                    ),
                    0,
                    ErrorType::TypeError,
                ));
            }
        }
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