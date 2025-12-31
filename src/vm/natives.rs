// Встроенные функции (native functions)

use crate::common::value::Value;
use crate::common::table::Table;
use crate::vm::vm::VM_CALL_CONTEXT;
use crate::common::error::LangError;
use std::path::PathBuf;
use std::rc::Rc;
use std::cell::RefCell;
use std::fs;
use std::io;
use std::env;
use chrono::Utc;

// Thread-local storage для хранения временных связей, созданных через relate()
// Храним указатели на таблицы (Rc::as_ptr) и имена колонок
thread_local! {
    static RELATIONS: RefCell<Vec<(*const RefCell<Table>, String, *const RefCell<Table>, String)>> = RefCell::new(Vec::new());
}

// Thread-local storage для хранения временных первичных ключей, созданных через primary_key()
thread_local! {
    static PRIMARY_KEYS: RefCell<Vec<(*const RefCell<Table>, String)>> = RefCell::new(Vec::new());
}

/// Получить все временные связи (для использования в VM)
pub fn take_relations() -> Vec<(*const RefCell<Table>, String, *const RefCell<Table>, String)> {
    RELATIONS.with(|r| {
        let mut relations = r.borrow_mut();
        let result = relations.clone();
        relations.clear();
        result
    })
}

/// Получить все временные первичные ключи (для использования в VM)
pub fn take_primary_keys() -> Vec<(*const RefCell<Table>, String)> {
    PRIMARY_KEYS.with(|pk| {
        let mut primary_keys = pk.borrow_mut();
        let result = primary_keys.clone();
        primary_keys.clear();
        result
    })
}

/// Вызвать пользовательскую функцию из нативной функции
/// Использует thread-local storage для доступа к VM
pub fn call_user_function(function_index: usize, args: &[Value]) -> Result<Value, LangError> {
    // Извлекаем указатель и сразу освобождаем заимствование контекста
    let vm_ptr = VM_CALL_CONTEXT.with(|ctx| {
        let ctx_ref = ctx.borrow();
        *ctx_ref
    });
    
    if let Some(vm_ptr) = vm_ptr {
        unsafe {
            let vm = &mut *vm_ptr;
            vm.call_function_by_index(function_index, args)
        }
    } else {
        Err(LangError::runtime_error(
            "Cannot call user function: VM context not available".to_string(),
            0,
        ))
    }
}

/// Нативная функция для создания связи между колонками таблиц
pub fn native_relate(args: &[Value]) -> Value {
    if args.len() != 2 {
        return Value::Null;
    }

    let col1 = match &args[0] {
        Value::ColumnReference { table, column_name } => (table, column_name),
        _ => return Value::Null,
    };

    let col2 = match &args[1] {
        Value::ColumnReference { table, column_name } => (table, column_name),
        _ => return Value::Null,
    };

    // Проверяем совместимость типов колонок (оба должны быть одинакового типа)
    let table1_ref = col1.0.borrow();
    let table2_ref = col2.0.borrow();
    
    let col1_data = match table1_ref.get_column(col1.1) {
        Some(col) => col,
        None => return Value::Null,
    };
    
    let col2_data = match table2_ref.get_column(col2.1) {
        Some(col) => col,
        None => return Value::Null,
    };

    // Проверяем совместимость типов (простая проверка - оба должны иметь значения одного типа)
    // Это базовая проверка, более точная проверка будет сделана при экспорте
    if !col1_data.is_empty() && !col2_data.is_empty() {
        // Проверяем первый не-null элемент каждой колонки
        let type1 = &col1_data[0];
        let type2 = &col2_data[0];
        
        // Проверяем совместимость типов (Number <-> Number, String <-> String)
        match (type1, type2) {
            (Value::Number(_), Value::Number(_)) => {},
            (Value::String(_), Value::String(_)) => {},
            _ => {
                // Типы не совместимы, но все равно сохраняем связь
                // Более строгая проверка будет при экспорте
            }
        }
    }

    // Сохраняем связь в thread-local storage
    RELATIONS.with(|r| {
        let mut relations = r.borrow_mut();
        relations.push((
            Rc::as_ptr(col1.0),
            col1.1.clone(),
            Rc::as_ptr(col2.0),
            col2.1.clone(),
        ));
    });

    Value::Null
}

/// Нативная функция для указания первичного ключа таблицы
pub fn native_primary_key(args: &[Value]) -> Value {
    if args.len() != 1 {
        return Value::Null;
    }

    let col = match &args[0] {
        Value::ColumnReference { table, column_name } => (table, column_name),
        _ => return Value::Null,
    };

    // Проверяем, что колонка существует
    let table_ref = col.0.borrow();
    if table_ref.get_column(col.1).is_none() {
        return Value::Null;
    }

    // Сохраняем первичный ключ в thread-local storage
    PRIMARY_KEYS.with(|pk| {
        let mut primary_keys = pk.borrow_mut();
        primary_keys.push((
            Rc::as_ptr(col.0),
            col.1.clone(),
        ));
    });

    Value::Null
}

pub fn native_print(args: &[Value]) -> Value {
    use crate::websocket::output_capture::OutputCapture;
    
    if args.is_empty() {
        if OutputCapture::is_capturing() {
            OutputCapture::write_output("");
        } else {
            println!();
        }
    } else {
        let mut output = String::new();
        for (i, arg) in args.iter().enumerate() {
            if i > 0 {
                output.push(' ');
            }
            output.push_str(&arg.to_string());
        }
        if OutputCapture::is_capturing() {
            OutputCapture::write_output(&output);
        } else {
            println!("{}", output);
        }
    }
    Value::Null
}

pub fn native_len(args: &[Value]) -> Value {
    if let Some(arg) = args.first() {
        match arg {
            Value::String(s) => Value::Number(s.len() as f64),
            Value::Array(arr) => Value::Number(arr.borrow().len() as f64),
            Value::Table(table) => Value::Number(table.borrow().len() as f64),
            Value::Object(map) => Value::Number(map.len() as f64),
            Value::ColumnReference { table, column_name } => {
                let table_ref = table.borrow();
                if let Some(column) = table_ref.get_column(column_name) {
                    Value::Number(column.len() as f64)
                } else {
                    Value::Null
                }
            },
            Value::Dataset(dataset) => {
                let batch_size = dataset.borrow().batch_size();
                Value::Number(batch_size as f64)
            },
            _ => Value::Null,
        }
    } else {
        Value::Null
    }
}

pub fn native_range(args: &[Value]) -> Value {
    // Определяем параметры в зависимости от количества аргументов
    let (start, end, step) = match args.len() {
        1 => {
            // range(10) → range(0, 10, 1)
            let end = match &args[0] {
                Value::Number(n) => *n as i64,
                _ => return Value::Null,
            };
            (0, end, 1)
        }
        2 => {
            // range(1, 10) → range(1, 10, 1)
            let start = match &args[0] {
                Value::Number(n) => *n as i64,
                _ => return Value::Null,
            };
            let end = match &args[1] {
                Value::Number(n) => *n as i64,
                _ => return Value::Null,
            };
            (start, end, 1)
        }
        3 => {
            // range(1, 10, 2) → range(1, 10, 2)
            let start = match &args[0] {
                Value::Number(n) => *n as i64,
                _ => return Value::Null,
            };
            let end = match &args[1] {
                Value::Number(n) => *n as i64,
                _ => return Value::Null,
            };
            let step = match &args[2] {
                Value::Number(n) => *n as i64,
                _ => return Value::Null,
            };
            if step == 0 {
                return Value::Null; // Ошибка: шаг не может быть 0
            }
            (start, end, step)
        }
        _ => {
            // Ошибка будет обработана в VM при вызове
            return Value::Null;
        }
    };
    
    // Генерация массива с учетом шага
    let mut result = Vec::new();
    if step > 0 {
        let mut current = start;
        while current < end {
            result.push(Value::Number(current as f64));
            current += step;
        }
    } else {
        // Отрицательный шаг: идем в обратном направлении
        let mut current = start;
        while current > end {
            result.push(Value::Number(current as f64));
            current += step; // step уже отрицательный
        }
    }
    
    Value::Array(Rc::new(RefCell::new(result)))
}

// Функции преобразования типов

pub fn native_int(args: &[Value]) -> Value {
    if args.is_empty() {
        return Value::Number(0.0);
    }
    match &args[0] {
        Value::Number(n) => Value::Number(n.trunc()), // Округление вниз до целого
        Value::String(s) => {
            // Парсинг строки в число
            match s.parse::<f64>() {
                Ok(n) => Value::Number(n.trunc()),
                Err(_) => Value::Number(0.0), // При ошибке парсинга возвращаем 0
            }
        }
        Value::Bool(b) => Value::Number(if *b { 1.0 } else { 0.0 }),
        Value::Null => Value::Number(0.0),
        _ => Value::Number(0.0),
    }
}

pub fn native_float(args: &[Value]) -> Value {
    if args.is_empty() {
        return Value::Number(0.0);
    }
    match &args[0] {
        Value::Number(n) => Value::Number(*n), // Уже число
        Value::String(s) => {
            // Парсинг строки в число
            match s.parse::<f64>() {
                Ok(n) => Value::Number(n),
                Err(_) => Value::Number(0.0), // При ошибке парсинга возвращаем 0.0
            }
        }
        Value::Bool(b) => Value::Number(if *b { 1.0 } else { 0.0 }),
        Value::Null => Value::Number(0.0),
        _ => Value::Number(0.0),
    }
}

pub fn native_bool(args: &[Value]) -> Value {
    if args.is_empty() {
        return Value::Bool(false);
    }
    Value::Bool(args[0].is_truthy())
}

pub fn native_str(args: &[Value]) -> Value {
    if args.is_empty() {
        return Value::String(String::new());
    }
    Value::String(args[0].to_string())
}

pub fn native_array(args: &[Value]) -> Value {
    // Если передан один аргумент и это тензор, преобразуем его в массив чисел
    if args.len() == 1 {
        if let Value::Tensor(tensor) = &args[0] {
            let tensor_ref = tensor.borrow();
            match tensor_ref.to_cpu() {
                Ok(cpu_tensor) => {
                    let data_values: Vec<Value> = cpu_tensor.data.iter()
                        .map(|&d| Value::Number(d as f64))
                        .collect();
                    return Value::Array(Rc::new(RefCell::new(data_values)));
                }
                Err(_) => {
                    // Если не удалось преобразовать в CPU, возвращаем пустой массив
                    return Value::Array(Rc::new(RefCell::new(Vec::new())));
                }
            }
        }
    }
    
    // Если передано несколько аргументов, преобразуем тензоры в массивы
    let mut result = Vec::new();
    for arg in args {
        if let Value::Tensor(tensor) = arg {
            let tensor_ref = tensor.borrow();
            match tensor_ref.to_cpu() {
                Ok(cpu_tensor) => {
                    let data_values: Vec<Value> = cpu_tensor.data.iter()
                        .map(|&d| Value::Number(d as f64))
                        .collect();
                    result.push(Value::Array(Rc::new(RefCell::new(data_values))));
                }
                Err(_) => {
                    // Если не удалось преобразовать в CPU, добавляем пустой массив
                    result.push(Value::Array(Rc::new(RefCell::new(Vec::new()))));
                }
            }
        } else {
            // Для не-тензоров добавляем как есть
            result.push(arg.clone());
        }
    }
    
    Value::Array(Rc::new(RefCell::new(result)))
}

pub fn native_date(args: &[Value]) -> Value {
    if args.is_empty() {
        return Value::String(String::new());
    }
    
    match &args[0] {
        Value::String(s) => {
            // Парсим строку даты и нормализуем в ISO формат
            // Поддерживаем форматы: YYYY-MM-DD, YYYY-MM-DDTHH:MM:SSZ, и другие ISO форматы
            let date_str = s.trim();
            
            // Если уже в формате ISO (YYYY-MM-DD или YYYY-MM-DDTHH:MM:SSZ), возвращаем как есть
            if date_str.len() >= 10 && date_str.chars().nth(4) == Some('-') && date_str.chars().nth(7) == Some('-') {
                Value::String(date_str.to_string())
            } else {
                // Для других форматов пока возвращаем как есть
                // В будущем можно добавить парсинг других форматов
                Value::String(date_str.to_string())
            }
        }
        Value::Number(n) => {
            // Если передано число (timestamp), конвертируем в ISO формат
            // Для простоты пока возвращаем как строку числа
            Value::String(format!("{}", n))
        }
        _ => Value::String(String::new()),
    }
}

pub fn native_money(args: &[Value]) -> Value {
    if args.is_empty() {
        return Value::String("0".to_string());
    }
    
    if args.len() < 2 {
        // Если формат не указан, просто возвращаем число как строку
        return Value::String(args[0].to_string());
    }
    
    let amount = match &args[0] {
        Value::Number(n) => *n,
        Value::String(s) => {
            // Пытаемся распарсить строку как число
            s.parse::<f64>().unwrap_or(0.0)
        }
        _ => 0.0,
    };
    
    let format_str = match &args[1] {
        Value::String(s) => s.clone(),
        _ => String::new(),
    };
    
    // Простое форматирование денег
    // Поддерживаем базовые паттерны: "$0.00", "0,0 $", "0 EUR"
    // Сначала проверяем паттерн "0,0" (запятая как десятичный разделитель)
    let formatted = if format_str.contains("0,0") {
        // Формат с запятой как десятичным разделителем
        let formatted_amount = format!("{:.2}", amount).replace('.', ",");
        if format_str.contains("$") {
            // "$0,0" или "0,0 $" - доллар в начале или конце
            if format_str.starts_with("$") {
                format!("${}", formatted_amount)
            } else {
                format!("{} $", formatted_amount)
            }
        } else if format_str.contains("EUR") || format_str.contains("€") {
            // "0,0 EUR" - евро в конце
            format!("{} EUR", formatted_amount)
        } else {
            // "0,0" без валюты - просто число с запятой
            let parts: Vec<&str> = format_str.split_whitespace().collect();
            if parts.len() > 1 {
                format!("{} {}", formatted_amount, parts[parts.len() - 1])
            } else {
                formatted_amount
            }
        }
    } else if format_str.contains("$") {
        // Формат с долларом (точка как десятичный разделитель)
        if format_str.contains("0.00") {
            format!("${:.2}", amount)
        } else {
            format!("${}", amount)
        }
    } else if format_str.contains("EUR") || format_str.contains("€") {
        // Формат с евро (точка как десятичный разделитель)
        if format_str.contains("0.0") {
            format!("{:.2} EUR", amount)
        } else {
            format!("{} EUR", amount)
        }
    } else {
        // Простое форматирование с двумя знаками после запятой (точка)
        format!("{:.2}", amount)
    };
    
    Value::String(formatted)
}

// Функции работы с типами

pub fn native_typeof(args: &[Value]) -> Value {
    if args.is_empty() {
        return Value::String("null".to_string());
    }
    let type_name = match &args[0] {
        Value::Number(n) => {
            // Различаем int и float по дробной части
            if n.fract() == 0.0 {
                "int"
            } else {
                "float"
            }
        }
        Value::Bool(_) => "bool",
        Value::String(s) => {
            // Проверяем, является ли строка датой или деньгами
            let s_trimmed = s.trim();
            // Проверка на дату: формат YYYY-MM-DD или ISO формат
            if s_trimmed.len() >= 10 && s_trimmed.chars().nth(4) == Some('-') && s_trimmed.chars().nth(7) == Some('-') {
                "date"
            } else if s_trimmed.starts_with('$') || s_trimmed.contains("EUR") || s_trimmed.contains("€") {
                // Проверка на деньги: содержит валютные символы
                "money"
            } else {
                "string"
            }
        }
        Value::Array(_) => "array",
        Value::Tuple(_) => "tuple",
        Value::Path(_) => "path",
        Value::Table(_) => "table",
        Value::Object(_) => "object",
        Value::ColumnReference { .. } => "column",
        Value::Null => "null",
        Value::Function(_) => "function",
        Value::NativeFunction(_) => "function",
        Value::Tensor(_) => "tensor",
        Value::Graph(_) => "graph",
        Value::LinearRegression(_) => "linear_regression",
        Value::SGD(_) => "sgd",
        Value::Momentum(_) => "momentum",
        Value::NAG(_) => "nag",
        Value::Adagrad(_) => "adagrad",
        Value::RMSprop(_) => "rmsprop",
        Value::Adam(_) => "adam",
        Value::AdamW(_) => "adamw",
        Value::Dataset(_) => "dataset",
        Value::NeuralNetwork(_) => "neural_network",
        Value::Sequential(_) => "sequential",
        Value::Layer(_) => "layer",
        Value::Window(_) => "window",
        Value::Image(_) => "image",
        Value::Figure(_) => "figure",
        Value::Axis(_) => "axis",
    };
    Value::String(type_name.to_string())
}

pub fn native_isinstance(args: &[Value]) -> Value {
    if args.len() < 2 {
        return Value::Bool(false);
    }
    
    let value = &args[0];
    // Извлекаем имя типа из второго аргумента
    // Поддерживаем как строки, так и другие типы (для констант типов, которые являются строками)
    let type_name_str = match &args[1] {
        Value::String(s) => s.clone(),
        Value::NativeFunction(index) => {
            // Если передан NativeFunction, извлекаем имя типа по индексу
            // Индексы: 0=print, 1=len, 2=range, 3=int, 4=float, 5=bool, 6=str, 7=array, 8=typeof, 9=isinstance, 10=date, 11=money, 12=path
            match *index {
                3 => "int".to_string(),
                4 => "float".to_string(),
                5 => "bool".to_string(),
                6 => "string".to_string(),
                7 => "array".to_string(),
                10 => "date".to_string(),
                11 => "money".to_string(),
                12 => "path".to_string(),
                _ => "unknown".to_string(),
            }
        }
        // Для обратной совместимости: если это не строка, пытаемся преобразовать в строку
        // Это позволит работать с константами типов, которые уже являются строками
        other => other.to_string(),
    };
    
    // Нормализуем имя типа (приводим к нижнему регистру)
    let type_name_lower = type_name_str.to_lowercase();
    
    let matches = match value {
        Value::Number(n) => {
            // Для чисел проверяем int, float и money
            if type_name_lower == "int" || type_name_lower == "integer" || type_name_lower == "num" || type_name_lower == "number" {
                true
            } else if type_name_lower == "float" {
                n.fract() != 0.0
            } else if type_name_lower == "money" {
                // Числа могут быть деньгами
                true
            } else {
                false
            }
        }
        Value::Bool(_) => type_name_lower == "bool" || type_name_lower == "boolean",
        Value::String(s) => {
            let s_trimmed = s.trim();
            if type_name_lower == "string" || type_name_lower == "str" {
                true
            } else if type_name_lower == "date" {
                // Проверка на формат даты: YYYY-MM-DD или ISO формат
                s_trimmed.len() >= 10 && s_trimmed.chars().nth(4) == Some('-') && s_trimmed.chars().nth(7) == Some('-')
            } else if type_name_lower == "money" {
                // Проверка на деньги: содержит валютные символы
                s_trimmed.starts_with('$') || s_trimmed.contains("EUR") || s_trimmed.contains("€")
            } else {
                false
            }
        }
        Value::Path(_) => type_name_lower == "path",
        Value::Array(_) => type_name_lower == "array" || type_name_lower == "list",
        Value::Tuple(_) => type_name_lower == "tuple",
        Value::Table(_) => type_name_lower == "table",
        Value::Object(_) => type_name_lower == "object" || type_name_lower == "dict" || type_name_lower == "dictionary",
        Value::ColumnReference { .. } => type_name_lower == "column",
        Value::Null => type_name_lower == "null" || type_name_lower == "none",
        Value::Function(_) | Value::NativeFunction(_) => type_name_lower == "function",
        Value::Tensor(_) => type_name_lower == "tensor",
        Value::Graph(_) => type_name_lower == "graph",
        Value::LinearRegression(_) => type_name_lower == "linear_regression",
        Value::SGD(_) => type_name_lower == "sgd",
        Value::Momentum(_) => type_name_lower == "momentum",
        Value::NAG(_) => type_name_lower == "nag",
        Value::Adagrad(_) => type_name_lower == "adagrad",
        Value::RMSprop(_) => type_name_lower == "rmsprop",
        Value::Adam(_) => type_name_lower == "adam",
        Value::AdamW(_) => type_name_lower == "adamw",
        Value::Dataset(_) => type_name_lower == "dataset",
        Value::NeuralNetwork(_) => type_name_lower == "neural_network",
        Value::Sequential(_) => type_name_lower == "sequential",
        Value::Layer(_) => type_name_lower == "layer",
        Value::Window(_) => type_name_lower == "window",
        Value::Image(_) => type_name_lower == "image",
        Value::Figure(_) => type_name_lower == "figure",
        Value::Axis(_) => type_name_lower == "axis",
    };
    
    Value::Bool(matches)
}

// Функции для работы с путями

pub fn native_path(args: &[Value]) -> Value {
    if args.is_empty() {
        return Value::Path(PathBuf::new());
    }
    
    match &args[0] {
        Value::String(s) => {
            // Создаем путь из строки
            Value::Path(PathBuf::from(s))
        }
        Value::Path(p) => {
            // Если уже путь, возвращаем копию
            Value::Path(p.clone())
        }
        _ => {
            // Для других типов преобразуем в строку и создаем путь
            Value::Path(PathBuf::from(args[0].to_string()))
        }
    }
}

pub fn native_path_name(args: &[Value]) -> Value {
    if args.is_empty() {
        return Value::String(String::new());
    }
    
    match &args[0] {
        Value::Path(p) => {
            if let Some(name) = p.file_name() {
                Value::String(name.to_string_lossy().to_string())
            } else {
                Value::String(String::new())
            }
        }
        _ => Value::String(String::new()),
    }
}

pub fn native_path_parent(args: &[Value]) -> Value {
    if args.is_empty() {
        return Value::Null;
    }
    
    match &args[0] {
        Value::Path(p) => {
            // Используем безопасную функцию для получения parent
            match safe_path_parent(p) {
                Some(parent) => Value::Path(parent),
                None => Value::Null,
            }
        }
        _ => Value::Null,
    }
}

pub fn native_path_exists(args: &[Value]) -> Value {
    if args.is_empty() {
        return Value::Bool(false);
    }
    
    match &args[0] {
        Value::Path(p) => Value::Bool(p.exists()),
        _ => Value::Bool(false),
    }
}

pub fn native_path_is_file(args: &[Value]) -> Value {
    if args.is_empty() {
        return Value::Bool(false);
    }
    
    match &args[0] {
        Value::Path(p) => Value::Bool(p.is_file()),
        _ => Value::Bool(false),
    }
}

pub fn native_path_is_dir(args: &[Value]) -> Value {
    if args.is_empty() {
        return Value::Bool(false);
    }
    
    match &args[0] {
        Value::Path(p) => Value::Bool(p.is_dir()),
        _ => Value::Bool(false),
    }
}

pub fn native_path_extension(args: &[Value]) -> Value {
    if args.is_empty() {
        return Value::String(String::new());
    }
    
    match &args[0] {
        Value::Path(p) => {
            if let Some(ext) = p.extension() {
                Value::String(ext.to_string_lossy().to_string())
            } else {
                Value::String(String::new())
            }
        }
        _ => Value::String(String::new()),
    }
}

pub fn native_path_stem(args: &[Value]) -> Value {
    if args.is_empty() {
        return Value::String(String::new());
    }
    
    match &args[0] {
        Value::Path(p) => {
            if let Some(stem) = p.file_stem() {
                Value::String(stem.to_string_lossy().to_string())
            } else {
                Value::String(String::new())
            }
        }
        _ => Value::String(String::new()),
    }
}

pub fn native_path_len(args: &[Value]) -> Value {
    if args.is_empty() {
        return Value::Number(0.0);
    }
    
    match &args[0] {
        Value::Path(p) => {
            let len = p.to_string_lossy().len();
            Value::Number(len as f64)
        }
        _ => Value::Number(0.0),
    }
}

// Математические функции

pub fn native_abs(args: &[Value]) -> Value {
    if args.is_empty() {
        return Value::Number(0.0);
    }
    
    // Check if first argument is a tensor
    if let Value::Tensor(tensor) = &args[0] {
        let tensor_ref = tensor.borrow();
        let result = tensor_ref.abs();
        return Value::Tensor(std::rc::Rc::new(std::cell::RefCell::new(result)));
    }
    
    // Handle number (original behavior)
    match &args[0] {
        Value::Number(n) => Value::Number(n.abs()),
        _ => Value::Null,
    }
}

pub fn native_sqrt(args: &[Value]) -> Value {
    if args.is_empty() {
        return Value::Number(0.0);
    }
    
    // Check if first argument is a tensor
    if let Value::Tensor(tensor) = &args[0] {
        let tensor_ref = tensor.borrow();
        match tensor_ref.sqrt() {
            Ok(result) => return Value::Tensor(std::rc::Rc::new(std::cell::RefCell::new(result))),
            Err(_) => return Value::Null,
        }
    }
    
    // Handle number (original behavior)
    match &args[0] {
        Value::Number(n) => {
            if *n < 0.0 {
                Value::Null // Отрицательное число - возвращаем Null
            } else {
                Value::Number(n.sqrt())
            }
        }
        _ => Value::Null,
    }
}

pub fn native_pow(args: &[Value]) -> Value {
    if args.len() < 2 {
        return Value::Null;
    }
    
    let base = match &args[0] {
        Value::Number(n) => *n,
        _ => return Value::Null,
    };
    
    let exp = match &args[1] {
        Value::Number(n) => *n,
        _ => return Value::Null,
    };
    
    Value::Number(base.powf(exp))
}

pub fn native_min(args: &[Value]) -> Value {
    if args.is_empty() {
        return Value::Null;
    }
    
    // Check if first argument is a tensor
    if let Value::Tensor(tensor) = &args[0] {
        let tensor_ref = tensor.borrow();
        match tensor_ref.to_cpu() {
            Ok(cpu_tensor) => {
                if cpu_tensor.data.is_empty() {
                    return Value::Null;
                }
                let min_val = cpu_tensor.data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
                return Value::Number(min_val as f64);
            }
            Err(_) => return Value::Null,
        }
    }
    
    // Handle multiple number arguments (original behavior)
    let mut min_val: Option<f64> = None;
    
    for arg in args {
        match arg {
            Value::Number(n) => {
                if let Some(current_min) = min_val {
                    if *n < current_min {
                        min_val = Some(*n);
                    }
                } else {
                    min_val = Some(*n);
                }
            }
            _ => return Value::Null, // Если есть нечисловой аргумент, возвращаем Null
        }
    }
    
    match min_val {
        Some(n) => Value::Number(n),
        None => Value::Null,
    }
}

pub fn native_max(args: &[Value]) -> Value {
    if args.is_empty() {
        return Value::Null;
    }
    
    // Check if first argument is a tensor
    if let Value::Tensor(tensor) = &args[0] {
        let tensor_ref = tensor.borrow();
        match tensor_ref.to_cpu() {
            Ok(cpu_tensor) => {
                if cpu_tensor.data.is_empty() {
                    return Value::Null;
                }
                let max_val = cpu_tensor.data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                return Value::Number(max_val as f64);
            }
            Err(_) => return Value::Null,
        }
    }
    
    // Handle multiple number arguments (original behavior)
    let mut max_val: Option<f64> = None;
    
    for arg in args {
        match arg {
            Value::Number(n) => {
                if let Some(current_max) = max_val {
                    if *n > current_max {
                        max_val = Some(*n);
                    }
                } else {
                    max_val = Some(*n);
                }
            }
            _ => return Value::Null, // Если есть нечисловой аргумент, возвращаем Null
        }
    }
    
    match max_val {
        Some(n) => Value::Number(n),
        None => Value::Null,
    }
}

pub fn native_round(args: &[Value]) -> Value {
    if args.is_empty() {
        return Value::Number(0.0);
    }
    
    // Check if first argument is a tensor
    if let Value::Tensor(tensor) = &args[0] {
        let tensor_ref = tensor.borrow();
        let result = tensor_ref.round();
        return Value::Tensor(std::rc::Rc::new(std::cell::RefCell::new(result)));
    }
    
    // Handle number (original behavior)
    match &args[0] {
        Value::Number(n) => {
            // Стандартное округление: к ближайшему целому
            // Для положительных: 3.5 -> 4, для отрицательных: -3.5 -> -3 (к нулю)
            if *n >= 0.0 {
                Value::Number(n.floor() + if n.fract() >= 0.5 { 1.0 } else { 0.0 })
            } else {
                // Для отрицательных: округляем к нулю
                // -3.5 -> -3, -3.6 -> -4
                // Для отрицательных чисел fract() возвращает положительное значение дробной части
                let abs_fract = n.abs().fract();
                if abs_fract > 0.5 {
                    // Округляем вниз (от нуля)
                    Value::Number(n.floor())
                } else if abs_fract < 0.5 {
                    // Округляем вверх (к нулю)
                    Value::Number(n.ceil())
                } else {
                    // Ровно 0.5 - округляем к нулю
                    Value::Number(n.ceil())
                }
            }
        }
        _ => Value::Null,
    }
}

// Строковые функции

pub fn native_upper(args: &[Value]) -> Value {
    if args.is_empty() {
        return Value::String(String::new());
    }
    
    match &args[0] {
        Value::String(s) => Value::String(s.to_uppercase()),
        _ => Value::Null,
    }
}

pub fn native_lower(args: &[Value]) -> Value {
    if args.is_empty() {
        return Value::String(String::new());
    }
    
    match &args[0] {
        Value::String(s) => Value::String(s.to_lowercase()),
        _ => Value::Null,
    }
}

pub fn native_trim(args: &[Value]) -> Value {
    if args.is_empty() {
        return Value::String(String::new());
    }
    
    match &args[0] {
        Value::String(s) => Value::String(s.trim().to_string()),
        _ => Value::Null,
    }
}

pub fn native_split(args: &[Value]) -> Value {
    if args.len() < 2 {
        return Value::Null;
    }
    
    let s = match &args[0] {
        Value::String(str) => str,
        _ => return Value::Null,
    };
    
    let delim = match &args[1] {
        Value::String(d) => d,
        _ => return Value::Null,
    };
    
    let parts: Vec<Value> = s.split(delim)
        .map(|part| Value::String(part.to_string()))
        .collect();
    
    Value::Array(Rc::new(RefCell::new(parts)))
}

pub fn native_join(args: &[Value]) -> Value {
    // Универсальная функция join: проверяем тип первого аргумента
    if args.is_empty() {
        return Value::Null;
    }
    
    // Если первый аргумент - таблица, это table join
    if matches!(&args[0], Value::Table(_)) && args.len() >= 3 {
        return native_table_join(args);
    }
    
    // Иначе это array join (для обратной совместимости)
    if args.len() < 2 {
        return Value::Null;
    }
    
    let arr = match &args[0] {
        Value::Array(a) => a,
        _ => return Value::Null,
    };
    
    let delim = match &args[1] {
        Value::String(d) => d,
        _ => return Value::Null,
    };
    
    let arr_ref = arr.borrow();
    let parts: Vec<String> = arr_ref.iter()
        .map(|v| v.to_string())
        .collect();
    
    Value::String(parts.join(delim))
}

pub fn native_contains(args: &[Value]) -> Value {
    if args.len() < 2 {
        return Value::Bool(false);
    }
    
    let s = match &args[0] {
        Value::String(str) => str,
        _ => return Value::Bool(false),
    };
    
    let substr = match &args[1] {
        Value::String(sub) => sub,
        _ => return Value::Bool(false),
    };
    
    Value::Bool(s.contains(substr))
}

// Функции для работы с массивами

pub fn native_push(args: &[Value]) -> Value {
    if args.len() < 2 {
        return Value::Null;
    }
    
    let arr = match &args[0] {
        Value::Array(a) => a,
        _ => return Value::Null,
    };
    
    // Мутируем массив in-place, сохраняя ссылочную семантику
    // Если массив используется в нескольких местах, изменения будут видны через все ссылки
    let item = args[1].clone();
    arr.borrow_mut().push(item);
    
    // Возвращаем тот же массив (клонируем только Rc, не содержимое)
    Value::Array(Rc::clone(arr))
}

pub fn native_pop(args: &[Value]) -> Value {
    if args.is_empty() {
        return Value::Null;
    }
    
    let arr = match &args[0] {
        Value::Array(a) => a,
        _ => return Value::Null,
    };
    
    // Copy-on-Write: если массив используется в нескольких местах, клонируем
    let arr = if Rc::strong_count(arr) > 1 {
        let cloned_vec: Vec<Value> = arr.borrow().iter().map(|v| v.clone()).collect();
        Rc::new(RefCell::new(cloned_vec))
    } else {
        arr.clone()
    };
    
    let mut arr_ref = arr.borrow_mut();
    if arr_ref.is_empty() {
        return Value::Null;
    }
    
    // Возвращаем последний элемент (удаленный элемент)
    arr_ref.pop().unwrap_or(Value::Null)
}

pub fn native_unique(args: &[Value]) -> Value {
    use std::collections::HashSet;
    
    if args.is_empty() {
        return Value::Array(Rc::new(RefCell::new(Vec::new())));
    }
    
    let arr = match &args[0] {
        Value::Array(a) => a,
        _ => return Value::Null,
    };
    
    // Создаем новый массив с уникальными элементами, сохраняя порядок первого вхождения
    // Используем HashSet для O(1) проверки вместо O(n) Vec::contains
    let arr_ref = arr.borrow();
    let mut seen = HashSet::new();
    let mut result = Vec::new();
    
    for item in arr_ref.iter() {
        // Используем строковое представление для хэширования, так как Value не Hash
        let item_str = item.to_string();
        if !seen.contains(&item_str) {
            seen.insert(item_str);
            result.push(item.clone());
        }
    }
    
    Value::Array(Rc::new(RefCell::new(result)))
}

pub fn native_reverse(args: &[Value]) -> Value {
    if args.is_empty() {
        return Value::Array(Rc::new(RefCell::new(Vec::new())));
    }
    
    let arr = match &args[0] {
        Value::Array(a) => a,
        _ => return Value::Null,
    };
    
    // Copy-on-Write: если массив используется в нескольких местах, клонируем
    let arr = if Rc::strong_count(arr) > 1 {
        let cloned_vec: Vec<Value> = arr.borrow().iter().map(|v| v.clone()).collect();
        Rc::new(RefCell::new(cloned_vec))
    } else {
        arr.clone()
    };
    
    // Мутируем in-place
    arr.borrow_mut().reverse();
    
    Value::Array(arr)
}

pub fn native_sort(args: &[Value]) -> Value {
    if args.is_empty() {
        return Value::Array(Rc::new(RefCell::new(Vec::new())));
    }
    
    let arr = match &args[0] {
        Value::Array(a) => a,
        _ => return Value::Null,
    };
    
    // Copy-on-Write: если массив используется в нескольких местах, клонируем
    let arr = if Rc::strong_count(arr) > 1 {
        let cloned_vec: Vec<Value> = arr.borrow().iter().map(|v| v.clone()).collect();
        Rc::new(RefCell::new(cloned_vec))
    } else {
        arr.clone()
    };
    
    // Сортируем in-place по строковому представлению для сравнения разных типов
    arr.borrow_mut().sort_by(|a, b| {
        let a_str = a.to_string();
        let b_str = b.to_string();
        a_str.cmp(&b_str)
    });
    
    Value::Array(arr)
}

pub fn native_sum(args: &[Value]) -> Value {
    if args.is_empty() {
        return Value::Number(0.0);
    }
    
    // Check if first argument is a tensor
    if let Value::Tensor(tensor) = &args[0] {
        let tensor_ref = tensor.borrow();
        let sum = tensor_ref.sum();
        return Value::Number(sum as f64);
    }
    
    // Handle array (original behavior)
    let arr = match &args[0] {
        Value::Array(a) => a,
        _ => return Value::Number(0.0),
    };
    
    let arr_ref = arr.borrow();
    let mut sum = 0.0;
    let mut has_numbers = false;
    
    for item in arr_ref.iter() {
        if let Value::Number(n) = item {
            sum += n;
            has_numbers = true;
        }
    }
    
    if has_numbers {
        Value::Number(sum)
    } else {
        Value::Number(0.0)
    }
}

pub fn native_average(args: &[Value]) -> Value {
    if args.is_empty() {
        return Value::Number(0.0);
    }
    
    // Check if first argument is a tensor
    if let Value::Tensor(tensor) = &args[0] {
        let tensor_ref = tensor.borrow();
        let mean = tensor_ref.mean();
        return Value::Number(mean as f64);
    }
    
    // Handle array (original behavior)
    let arr = match &args[0] {
        Value::Array(a) => a,
        _ => return Value::Number(0.0),
    };
    
    let arr_ref = arr.borrow();
    let mut sum = 0.0;
    let mut count = 0;
    
    for item in arr_ref.iter() {
        if let Value::Number(n) = item {
            sum += n;
            count += 1;
        }
    }
    
    if count > 0 {
        Value::Number(sum / count as f64)
    } else {
        Value::Number(0.0)
    }
}

pub fn native_count(args: &[Value]) -> Value {
    if args.is_empty() {
        return Value::Number(0.0);
    }
    
    // Check if first argument is a tensor
    if let Value::Tensor(tensor) = &args[0] {
        let tensor_ref = tensor.borrow();
        let count = tensor_ref.total_size();
        return Value::Number(count as f64);
    }
    
    // Handle array (original behavior)
    match &args[0] {
        Value::Array(arr) => Value::Number(arr.borrow().len() as f64),
        _ => Value::Number(0.0),
    }
}

pub fn native_any(args: &[Value]) -> Value {
    if args.is_empty() {
        return Value::Bool(false);
    }
    
    // Check if first argument is a tensor
    if let Value::Tensor(tensor) = &args[0] {
        let tensor_ref = tensor.borrow();
        match tensor_ref.to_cpu() {
            Ok(cpu_tensor) => {
                if cpu_tensor.data.is_empty() {
                    return Value::Bool(false);
                }
                // Check if there's at least one non-zero element
                for &val in cpu_tensor.data.iter() {
                    if val != 0.0 {
                        return Value::Bool(true);
                    }
                }
                return Value::Bool(false);
            }
            Err(_) => return Value::Bool(false),
        }
    }
    
    // Handle array (original behavior)
    let arr = match &args[0] {
        Value::Array(a) => a,
        _ => return Value::Bool(false),
    };
    
    let arr_ref = arr.borrow();
    
    // Для пустого массива возвращаем false
    if arr_ref.is_empty() {
        return Value::Bool(false);
    }
    
    // Проверяем, есть ли хотя бы одно истинное значение
    for item in arr_ref.iter() {
        if item.is_truthy() {
            return Value::Bool(true);
        }
    }
    
    Value::Bool(false)
}

pub fn native_all(args: &[Value]) -> Value {
    if args.is_empty() {
        return Value::Bool(false);
    }
    
    // Check if first argument is a tensor
    if let Value::Tensor(tensor) = &args[0] {
        let tensor_ref = tensor.borrow();
        match tensor_ref.to_cpu() {
            Ok(cpu_tensor) => {
                if cpu_tensor.data.is_empty() {
                    return Value::Bool(false);
                }
                // Check if all elements are non-zero
                for &val in cpu_tensor.data.iter() {
                    if val == 0.0 {
                        return Value::Bool(false);
                    }
                }
                return Value::Bool(true);
            }
            Err(_) => return Value::Bool(false),
        }
    }
    
    // Handle array (original behavior)
    let arr = match &args[0] {
        Value::Array(a) => a,
        _ => return Value::Bool(false),
    };
    
    let arr_ref = arr.borrow();
    
    // Для пустого массива возвращаем false (согласно плану)
    if arr_ref.is_empty() {
        return Value::Bool(false);
    }
    
    // Проверяем, все ли значения истинны
    for item in arr_ref.iter() {
        if !item.is_truthy() {
            return Value::Bool(false);
        }
    }
    
    Value::Bool(true)
}

// Функции для работы с таблицами

pub fn native_table(args: &[Value]) -> Value {
    if args.is_empty() {
        return Value::Null;
    }

    // Первый аргумент - данные (массив массивов)
    let data = match &args[0] {
        Value::Array(arr) => {
            // Преобразуем массив массивов в Vec<Vec<Value>>
            let arr_ref = arr.borrow();
            let mut rows = Vec::new();
            for row_val in arr_ref.iter() {
                match row_val {
                    Value::Array(row) => rows.push(row.borrow().clone()),
                    _ => {
                        // Если элемент не массив, создаем строку с одним элементом
                        rows.push(vec![row_val.clone()]);
                    }
                }
            }
            rows
        }
        _ => return Value::Null,
    };

    // Второй аргумент (опциональный) - заголовки
    let headers = if args.len() > 1 {
        match &args[1] {
            Value::Array(headers_arr) => {
                let headers_arr_ref = headers_arr.borrow();
                let mut header_strings = Vec::new();
                for header_val in headers_arr_ref.iter() {
                    match header_val {
                        Value::String(s) => header_strings.push(s.clone()),
                        _ => header_strings.push(header_val.to_string()),
                    }
                }
                Some(header_strings)
            }
            _ => None,
        }
    } else {
        None
    };

    let table = Table::from_data(data, headers);
    Value::Table(Rc::new(RefCell::new(table)))
}

pub fn native_read_file(args: &[Value]) -> Value {
    if args.is_empty() {
        return Value::Null;
    }

    // Первый аргумент - путь к файлу
    // Для Path значений используем PathBuf напрямую, чтобы не терять информацию при конвертации
    let file_path = match &args[0] {
        Value::Path(p) => p.clone(),
        Value::String(s) => PathBuf::from(s),
        _ => return Value::Null,
    };
    
    let file_path_str = file_path.to_string_lossy().to_string();

    // Проверяем, является ли это SMB путем (lib://)
    if file_path_str.starts_with("lib://") {
        // Извлекаем имя шары и путь к файлу
        let path_without_prefix = &file_path_str[6..]; // Убираем "lib://"
        let parts: Vec<&str> = path_without_prefix.splitn(2, '/').collect();
        
        if parts.is_empty() {
            return Value::Null;
        }
        
        let share_name = parts[0];
        let file_path_on_share = if parts.len() > 1 { parts[1] } else { "" };
        
        // Получаем SmbManager из thread-local storage
        if let Some(smb_manager) = crate::vm::file_ops::get_smb_manager() {
            match smb_manager.lock().unwrap().read_file(share_name, file_path_on_share) {
                Ok(content) => {
                    // Определяем тип файла по расширению
                    let extension = std::path::Path::new(file_path_on_share)
                        .extension()
                        .and_then(|ext| ext.to_str())
                        .unwrap_or("")
                        .to_lowercase();
                    
                    match extension.as_str() {
                        "csv" => {
                            // Парсим CSV из байтов
                            // Создаем временный файл для парсинга CSV
                            use std::io::Write;
                            let temp_file = std::env::temp_dir().join(format!("datacode_smb_{}.csv", std::process::id()));
                            if let Ok(mut file) = fs::File::create(&temp_file) {
                                if file.write_all(&content).is_ok() {
                                    match read_csv_file(&temp_file) {
                                        Ok(table) => {
                                            let _ = fs::remove_file(&temp_file);
                                            return Value::Table(Rc::new(RefCell::new(table)));
                                        }
                                        Err(_) => {
                                            let _ = fs::remove_file(&temp_file);
                                        }
                                    }
                                }
                            }
                            Value::Null
                        }
                        "xlsx" => {
                            // Создаем временный файл для парсинга XLSX
                            use std::io::Write;
                            let temp_file = std::env::temp_dir().join(format!("datacode_smb_{}.xlsx", std::process::id()));
                            if let Ok(mut file) = fs::File::create(&temp_file) {
                                if file.write_all(&content).is_ok() {
                                    let header_row = if args.len() > 1 {
                                        match &args[1] {
                                            Value::Number(n) => *n as usize,
                                            _ => 0,
                                        }
                                    } else {
                                        0
                                    };
                                    
                                    let sheet_name = if args.len() > 2 {
                                        match &args[2] {
                                            Value::String(s) => Some(s.clone()),
                                            _ => None,
                                        }
                                    } else if args.len() > 1 {
                                        match &args[1] {
                                            Value::String(s) => Some(s.clone()),
                                            _ => None,
                                        }
                                    } else {
                                        None
                                    };
                                    
                                    match read_xlsx_file(&temp_file, header_row, sheet_name.as_deref()) {
                                        Ok(table) => {
                                            let _ = fs::remove_file(&temp_file);
                                            return Value::Table(Rc::new(RefCell::new(table)));
                                        }
                                        Err(_) => {
                                            let _ = fs::remove_file(&temp_file);
                                        }
                                    }
                                }
                            }
                            Value::Null
                        }
                        "txt" | "text" => {
                            match String::from_utf8(content) {
                                Ok(text) => Value::String(text),
                                Err(_) => Value::Null,
                            }
                        }
                        _ => {
                            // По умолчанию пытаемся прочитать как текст
                            match String::from_utf8(content) {
                                Ok(text) => Value::String(text),
                                Err(_) => Value::Null,
                            }
                        }
                    }
                }
                Err(_) => Value::Null,
            }
        } else {
            Value::Null
        }
    } else {
        // Обычный локальный файл
        // Используем file_path напрямую (уже PathBuf) вместо создания нового из строки
        // Разрешаем путь относительно папки сессии в режиме --use-ve
        let resolved_path = match resolve_path_in_session(&file_path) {
            Ok(p) => p,
            Err(err_msg) => {
                // При ошибке безопасности сохраняем сообщение об ошибке
                use crate::websocket::set_native_error;
                set_native_error(format!("Path resolution error: {}", err_msg));
                return Value::Null;
            }
        };
        
        // Проверяем существование файла перед чтением
        if !resolved_path.exists() {
            use crate::websocket::set_native_error;
            set_native_error(format!("File does not exist: {}", resolved_path.display()));
            return Value::Null;
        }
        
        if !resolved_path.is_file() {
            use crate::websocket::set_native_error;
            set_native_error(format!("Path is not a file: {}", resolved_path.display()));
            return Value::Null;
        }
        
        // Проверяем расширение файла
        let extension = resolved_path.extension()
            .and_then(|ext| ext.to_str())
            .unwrap_or("")
            .to_lowercase();

    match extension.as_str() {
        "csv" => {
            // Читаем CSV файл
            match read_csv_file(&resolved_path) {
                Ok(table) => Value::Table(Rc::new(RefCell::new(table))),
                Err(e) => {
                    use crate::websocket::set_native_error;
                    set_native_error(format!("Error reading CSV file: {}", e));
                    Value::Null
                },
            }
        }
        "xlsx" => {
            // Читаем XLSX файл
            let header_row = if args.len() > 1 {
                match &args[1] {
                    Value::Number(n) => *n as usize,
                    _ => 0,
                }
            } else {
                0
            };
            
            let sheet_name = if args.len() > 2 {
                match &args[2] {
                    Value::String(s) => Some(s.clone()),
                    _ => None,
                }
            } else if args.len() > 1 {
                match &args[1] {
                    Value::String(s) => Some(s.clone()),
                    _ => None,
                }
            } else {
                None
            };

            match read_xlsx_file(&resolved_path, header_row, sheet_name.as_deref()) {
                Ok(table) => Value::Table(Rc::new(RefCell::new(table))),
                Err(e) => {
                    use crate::websocket::set_native_error;
                    set_native_error(format!("Error reading XLSX file: {}", e));
                    Value::Null
                },
            }
        }
        "txt" | "text" => {
            // Читаем текстовый файл как строку
            match fs::read_to_string(&resolved_path) {
                Ok(content) => Value::String(content),
                Err(e) => {
                    use crate::websocket::set_native_error;
                    set_native_error(format!("Error reading text file: {}", e));
                    Value::Null
                },
            }
        }
        _ => {
            // По умолчанию пытаемся прочитать как текст
            match fs::read_to_string(&resolved_path) {
                Ok(content) => Value::String(content),
                Err(e) => {
                    use crate::websocket::set_native_error;
                    set_native_error(format!("Error reading file: {}", e));
                    Value::Null
                },
            }
        }
    }
    }
}

fn read_csv_file(path: &PathBuf) -> Result<Table, io::Error> {
    use csv::ReaderBuilder;
    
    let mut reader = ReaderBuilder::new()
        .has_headers(true)
        .from_path(path)?;

    // Читаем заголовки
    let headers: Vec<String> = reader.headers()?
        .iter()
        .map(|s| s.to_string())
        .collect();

    // Читаем данные
    let mut rows = Vec::new();
    for result in reader.records() {
        let record = result?;
        let row: Vec<Value> = record.iter()
            .map(|field| {
                // Пытаемся определить тип данных
                if let Ok(num) = field.parse::<f64>() {
                    Value::Number(num)
                } else if field == "true" || field == "True" {
                    Value::Bool(true)
                } else if field == "false" || field == "False" {
                    Value::Bool(false)
                } else if field.is_empty() {
                    Value::Null
                } else {
                    Value::String(field.to_string())
                }
            })
            .collect();
        rows.push(row);
    }

    Ok(Table::from_data(rows, Some(headers)))
}

fn read_xlsx_file(path: &PathBuf, header_row: usize, sheet_name: Option<&str>) -> Result<Table, Box<dyn std::error::Error>> {
    use calamine::{open_workbook, Reader, Xlsx};
    
    let mut workbook: Xlsx<_> = open_workbook(path)?;
    
    // Выбираем лист
    let sheet = if let Some(name) = sheet_name {
        workbook.worksheet_range(name)?
    } else {
        // Берем первый лист
        let sheet_names = workbook.sheet_names();
        if sheet_names.is_empty() {
            return Err("No sheets found".into());
        }
        workbook.worksheet_range(&sheet_names[0])?
    };

    let mut rows = Vec::new();
    let mut headers = Vec::new();
    
    for (row_idx, row) in sheet.rows().enumerate() {
        let values: Vec<Value> = row.iter()
            .map(|cell| {
                match cell {
                    calamine::Data::Int(n) => Value::Number(*n as f64),
                    calamine::Data::Float(n) => Value::Number(*n),
                    calamine::Data::String(s) => Value::String(s.clone()),
                    calamine::Data::Bool(b) => Value::Bool(*b),
                    calamine::Data::DateTime(dt) => Value::String(dt.to_string()),
                    calamine::Data::DateTimeIso(s) => Value::String(s.clone()),
                    calamine::Data::DurationIso(s) => Value::String(s.clone()),
                    calamine::Data::Error(_) => Value::Null,
                    calamine::Data::Empty => Value::Null,
                }
            })
            .collect();

        if row_idx == header_row {
            // Это строка заголовков
            headers = values.iter()
                .map(|v| match v {
                    Value::String(s) => s.clone(),
                    _ => v.to_string(),
                })
                .collect();
        } else {
            rows.push(values);
        }
    }

    // Если заголовки не были найдены, генерируем их
    if headers.is_empty() && !rows.is_empty() {
        let num_cols = rows[0].len();
        headers = (0..num_cols)
            .map(|i| format!("Column_{}", i))
            .collect();
    }

    Ok(Table::from_data(rows, Some(headers)))
}

// Дополнительные функции для работы с таблицами

pub fn native_table_info(args: &[Value]) -> Value {
    if args.is_empty() {
        return Value::String("Table: empty".to_string());
    }

    match &args[0] {
        Value::Table(table) => {
            let table_ref = table.borrow();
            let row_count = table_ref.len();
            let col_count = table_ref.column_count();
            let headers = table_ref.headers.clone();
            
            let mut info = format!("Table: {} rows, {} columns\n", row_count, col_count);
            info.push_str("Columns:\n");
            for header in &headers {
                if let Some(column) = table_ref.get_column(header) {
                    let col_type = if column.is_empty() {
                        "empty".to_string()
                    } else {
                        // Определяем тип по первому не-null значению
                        let first_val = column.iter().find(|v| !matches!(v, Value::Null));
                        match first_val {
                            Some(Value::Number(_)) => "number".to_string(),
                            Some(Value::String(_)) => "string".to_string(),
                            Some(Value::Bool(_)) => "bool".to_string(),
                            Some(Value::Array(_)) => "array".to_string(),
                            _ => "mixed".to_string(),
                        }
                    };
                    info.push_str(&format!("  - {}: {} ({} values)\n", header, col_type, column.len()));
                }
            }
            Value::String(info)
        }
        _ => Value::String("Not a table".to_string()),
    }
}

pub fn native_table_head(args: &[Value]) -> Value {
    if args.is_empty() {
        return Value::Null;
    }

    let n = if args.len() > 1 {
        match &args[1] {
            Value::Number(num) => *num as usize,
            _ => 5,
        }
    } else {
        5
    };

    match &args[0] {
        Value::Table(table) => {
            let table_ref = table.borrow();
            let row_count = table_ref.len();
            let take_n = if n > row_count { row_count } else { n };
            
            // Оптимизация: предварительное выделение памяти
            let mut new_rows = Vec::with_capacity(take_n);
            for i in 0..take_n {
                if let Some(row) = table_ref.get_row(i) {
                    new_rows.push(row.clone());
                }
            }
            
            let new_table = Table::from_data(new_rows, Some(table_ref.headers.clone()));
            Value::Table(Rc::new(RefCell::new(new_table)))
        }
        _ => Value::Null,
    }
}

pub fn native_table_tail(args: &[Value]) -> Value {
    if args.is_empty() {
        return Value::Null;
    }

    let n = if args.len() > 1 {
        match &args[1] {
            Value::Number(num) => *num as usize,
            _ => 5,
        }
    } else {
        5
    };

    match &args[0] {
        Value::Table(table) => {
            let table_ref = table.borrow();
            let row_count = table_ref.len();
            let take_n = if n > row_count { row_count } else { n };
            let start_idx = if row_count > take_n { row_count - take_n } else { 0 };
            
            // Оптимизация: предварительное выделение памяти
            let mut new_rows = Vec::with_capacity(take_n);
            for i in start_idx..row_count {
                if let Some(row) = table_ref.get_row(i) {
                    new_rows.push(row.clone());
                }
            }
            
            let new_table = Table::from_data(new_rows, Some(table_ref.headers.clone()));
            Value::Table(Rc::new(RefCell::new(new_table)))
        }
        _ => Value::Null,
    }
}

pub fn native_table_select(args: &[Value]) -> Value {
    if args.len() < 2 {
        return Value::Null;
    }

    let columns_to_select = match &args[1] {
        Value::Array(arr) => {
            let arr_ref = arr.borrow();
            let mut cols = Vec::new();
            for val in arr_ref.iter() {
                match val {
                    Value::String(s) => cols.push(s.clone()),
                    _ => return Value::Null,
                }
            }
            cols
        }
        _ => return Value::Null,
    };

    match &args[0] {
        Value::Table(table) => {
            let table_ref = table.borrow();
            let mut new_rows = Vec::new();
            
            // Создаем индексы колонок для выборки
            let mut col_indices = Vec::new();
            for col_name in &columns_to_select {
                if let Some(idx) = table_ref.headers.iter().position(|h| h == col_name) {
                    col_indices.push(idx);
                } else {
                    return Value::Null; // Колонка не найдена
                }
            }
            
            // Создаем новые строки только с выбранными колонками
            for row in &table_ref.rows {
                let mut new_row = Vec::new();
                for &idx in &col_indices {
                    if idx < row.len() {
                        new_row.push(row[idx].clone());
                    } else {
                        new_row.push(Value::Null);
                    }
                }
                new_rows.push(new_row);
            }
            
            let new_table = Table::from_data(new_rows, Some(columns_to_select));
            Value::Table(Rc::new(RefCell::new(new_table)))
        }
        _ => Value::Null,
    }
}

pub fn native_table_sort(args: &[Value]) -> Value {
    if args.len() < 2 {
        return Value::Null;
    }

    let column_name = match &args[1] {
        Value::String(s) => s.clone(),
        _ => return Value::Null,
    };

    let ascending = if args.len() > 2 {
        match &args[2] {
            Value::Bool(b) => *b,
            Value::Number(n) => *n != 0.0,
            _ => true,
        }
    } else {
        true
    };

    match &args[0] {
        Value::Table(table) => {
            let table_ref = table.borrow();
            
            // Получаем колонку для сортировки
            let sort_column = match table_ref.get_column(&column_name) {
                Some(col) => col.clone(),
                None => return Value::Null,
            };
            
            // Создаем вектор индексов для сортировки
            let mut indices: Vec<usize> = (0..table_ref.rows.len()).collect();
            
            // Сортируем индексы по значениям в колонке
            indices.sort_by(|&a, &b| {
                let val_a = &sort_column[a];
                let val_b = &sort_column[b];
                
                let cmp = match (val_a, val_b) {
                    (Value::Number(n1), Value::Number(n2)) => n1.partial_cmp(n2).unwrap_or(std::cmp::Ordering::Equal),
                    (Value::String(s1), Value::String(s2)) => s1.cmp(s2),
                    (Value::Bool(b1), Value::Bool(b2)) => b1.cmp(b2),
                    (Value::Null, Value::Null) => std::cmp::Ordering::Equal,
                    (Value::Null, _) => std::cmp::Ordering::Less,
                    (_, Value::Null) => std::cmp::Ordering::Greater,
                    _ => val_a.to_string().cmp(&val_b.to_string()),
                };
                
                if ascending {
                    cmp
                } else {
                    cmp.reverse()
                }
            });
            
            // Создаем новые строки в отсортированном порядке
            let mut new_rows = Vec::new();
            for &idx in &indices {
                if let Some(row) = table_ref.get_row(idx) {
                    new_rows.push(row.clone());
                }
            }
            
            let new_table = Table::from_data(new_rows, Some(table_ref.headers.clone()));
            Value::Table(Rc::new(RefCell::new(new_table)))
        }
        _ => Value::Null,
    }
}

pub fn native_table_where(args: &[Value]) -> Value {
    if args.len() < 4 {
        return Value::Null;
    }

    let column_name = match &args[1] {
        Value::String(s) => s.clone(),
        _ => return Value::Null,
    };

    let operator = match &args[2] {
        Value::String(s) => s.as_str(),
        _ => return Value::Null,
    };

    let filter_value = args[3].clone();

    match &args[0] {
        Value::Table(table) => {
            let table_ref = table.borrow();
            
            // Получаем колонку для фильтрации
            let filter_column = match table_ref.get_column(&column_name) {
                Some(col) => col.clone(),
                None => return Value::Null,
            };
            
            // Определяем, какие строки проходят фильтр
            let mut matching_indices = Vec::new();
            for (i, val) in filter_column.iter().enumerate() {
                let matches = match operator {
                    ">" => compare_values(val, &filter_value) == std::cmp::Ordering::Greater,
                    "<" => compare_values(val, &filter_value) == std::cmp::Ordering::Less,
                    ">=" => {
                        let cmp = compare_values(val, &filter_value);
                        cmp == std::cmp::Ordering::Greater || cmp == std::cmp::Ordering::Equal
                    }
                    "<=" => {
                        let cmp = compare_values(val, &filter_value);
                        cmp == std::cmp::Ordering::Less || cmp == std::cmp::Ordering::Equal
                    }
                    "==" | "=" => compare_values(val, &filter_value) == std::cmp::Ordering::Equal,
                    "!=" | "<>" => compare_values(val, &filter_value) != std::cmp::Ordering::Equal,
                    _ => false,
                };
                
                if matches {
                    matching_indices.push(i);
                }
            }
            
            // Создаем новые строки только с подходящими индексами
            let mut new_rows = Vec::new();
            for &idx in &matching_indices {
                if let Some(row) = table_ref.get_row(idx) {
                    new_rows.push(row.clone());
                }
            }
            
            let new_table = Table::from_data(new_rows, Some(table_ref.headers.clone()));
            Value::Table(Rc::new(RefCell::new(new_table)))
        }
        _ => Value::Null,
    }
}

fn compare_values(a: &Value, b: &Value) -> std::cmp::Ordering {
    match (a, b) {
        (Value::Number(n1), Value::Number(n2)) => n1.partial_cmp(n2).unwrap_or(std::cmp::Ordering::Equal),
        (Value::String(s1), Value::String(s2)) => s1.cmp(s2),
        (Value::Bool(b1), Value::Bool(b2)) => b1.cmp(b2),
        (Value::Null, Value::Null) => std::cmp::Ordering::Equal,
        (Value::Null, _) => std::cmp::Ordering::Less,
        (_, Value::Null) => std::cmp::Ordering::Greater,
        _ => a.to_string().cmp(&b.to_string()),
    }
}

// Вспомогательная функция для выполнения ASOF join для одной группы
fn asof_join_single_group(
    left_rows: &[Vec<Value>],
    right_rows: &[Vec<Value>],
    left_time_idx: usize,
    right_time_idx: usize,
    direction: &str,
    null_right_row: &[Value],
) -> Vec<Vec<Value>> {
    let mut result = Vec::new();
    
    // Сортируем правую таблицу по времени (для эффективного поиска)
    let mut right_indices: Vec<usize> = (0..right_rows.len()).collect();
    right_indices.sort_by(|&a, &b| {
        let time_a = &right_rows[a][right_time_idx];
        let time_b = &right_rows[b][right_time_idx];
        compare_values(time_a, time_b)
    });
    
    for left_row in left_rows {
        let left_time = &left_row[left_time_idx];
        
        // Ищем ближайшую строку в правой таблице
        let mut best_match: Option<usize> = None;
        let mut best_diff: Option<f64> = None;
        
        for &right_idx in &right_indices {
            let right_row = &right_rows[right_idx];
            let right_time = &right_row[right_time_idx];
            
            // Вычисляем разницу времени (упрощенная версия - только для чисел)
            let diff = match (left_time, right_time) {
                (Value::Number(l), Value::Number(r)) => {
                    let diff_val = match direction {
                        "backward" => *l - *r,  // left_time >= right_time
                        "forward" => *r - *l,   // right_time >= left_time
                        "nearest" => (*l - *r).abs(),
                        _ => *l - *r,
                    };
                    Some(diff_val)
                }
                _ => None,
            };
            
            if let Some(d) = diff {
                let matches_direction = match direction {
                    "backward" => d >= 0.0,
                    "forward" => d >= 0.0,
                    "nearest" => true,
                    _ => d >= 0.0,
                };
                
                if matches_direction {
                    let should_update = match best_diff {
                        None => true,
                        Some(bd) => match direction {
                            "nearest" => d < bd,
                            _ => d < bd,
                        },
                    };
                    
                    if should_update {
                        best_match = Some(right_idx);
                        best_diff = Some(d);
                    }
                }
            }
        }
        
        if let Some(right_idx) = best_match {
            let mut new_row = left_row.clone();
            new_row.extend_from_slice(&right_rows[right_idx]);
            result.push(new_row);
        } else {
            // Нет совпадения - добавляем строку с NULL справа (для left join семантики)
            let mut new_row = left_row.clone();
            new_row.extend_from_slice(null_right_row);
            result.push(new_row);
        }
    }
    
    result
}

pub fn native_show_table(args: &[Value]) -> Value {
    if args.is_empty() {
        return Value::Null;
    }

    match &args[0] {
        Value::Table(table) => {
            let table_ref = table.borrow();
            
            if table_ref.rows.is_empty() {
                println!("Empty table");
                return Value::Null;
            }
            
            // Вычисляем ширину колонок
            let mut col_widths = Vec::new();
            for header in &table_ref.headers {
                let mut max_width = header.len();
                if let Some(column) = table_ref.get_column(header) {
                    for val in column {
                        let val_str = val.to_string();
                        if val_str.len() > max_width {
                            max_width = val_str.len();
                        }
                    }
                }
                col_widths.push(max_width.max(3)); // Минимум 3 символа
            }
            
            // Печатаем верхнюю границу
            print!("┌");
            for (i, &width) in col_widths.iter().enumerate() {
                if i > 0 {
                    print!("┬");
                }
                print!("{}", "─".repeat(width + 2));
            }
            println!("┐");
            
            // Печатаем заголовки
            print!("│");
            for (i, header) in table_ref.headers.iter().enumerate() {
                if i > 0 {
                    print!("│");
                }
                print!(" {:<width$} ", header, width = col_widths[i]);
            }
            println!("│");
            
            // Печатаем разделитель
            print!("├");
            for (i, &width) in col_widths.iter().enumerate() {
                if i > 0 {
                    print!("┼");
                }
                print!("{}", "─".repeat(width + 2));
            }
            println!("┤");
            
            // Печатаем строки (максимум 20 для больших таблиц)
            let max_rows = 20;
            let rows_to_show = table_ref.rows.len().min(max_rows);
            for row_idx in 0..rows_to_show {
                if let Some(row) = table_ref.get_row(row_idx) {
                    print!("│");
                    for (i, val) in row.iter().enumerate() {
                        if i > 0 {
                            print!("│");
                        }
                        let val_str = val.to_string();
                        print!(" {:<width$} ", val_str, width = col_widths[i]);
                    }
                    println!("│");
                }
            }
            
            // Печатаем нижнюю границу
            print!("└");
            for (i, &width) in col_widths.iter().enumerate() {
                if i > 0 {
                    print!("┴");
                }
                print!("{}", "─".repeat(width + 2));
            }
            println!("┘");
            
            if table_ref.rows.len() > max_rows {
                println!("... ({} more rows)", table_ref.rows.len() - max_rows);
            }
            
            Value::Null
        }
        _ => Value::Null,
    }
}

pub fn native_merge_tables(args: &[Value]) -> Value {
    use std::collections::HashSet;
    
    if args.is_empty() {
        return Value::Null;
    }

    // Извлекаем массив таблиц
    let tables_array = match &args[0] {
        Value::Array(arr) => arr.borrow().clone(),
        _ => return Value::Null,
    };

    // Обработка пустого массива
    if tables_array.is_empty() {
        return Value::Null;
    }

    // Определяем режим (по умолчанию "outer")
    let mode = if args.len() > 1 {
        match &args[1] {
            Value::String(s) => s.as_str(),
            _ => "outer",
        }
    } else {
        "outer"
    };

    // Извлекаем таблицы из массива
    let mut tables = Vec::new();
    for val in &tables_array {
        match val {
            Value::Table(table) => {
                tables.push(table.clone());
            }
            _ => return Value::Null, // Все элементы должны быть таблицами
        }
    }

    // Если только одна таблица, возвращаем её копию
    if tables.len() == 1 {
        let table_ref = tables[0].borrow();
        let new_table = Table::from_data(table_ref.rows.clone(), Some(table_ref.headers.clone()));
        return Value::Table(Rc::new(RefCell::new(new_table)));
    }

    // Собираем все уникальные колонки
    let mut all_columns_set = HashSet::new();
    let mut column_order = Vec::new();
    
    // Сначала добавляем колонки первой таблицы для сохранения порядка
    let first_table = tables[0].borrow();
    for header in &first_table.headers {
        if all_columns_set.insert(header.clone()) {
            column_order.push(header.clone());
        }
    }
    
    // Затем добавляем колонки из остальных таблиц
    for table_rc in &tables[1..] {
        let table_ref = table_rc.borrow();
        for header in &table_ref.headers {
            if all_columns_set.insert(header.clone()) {
                column_order.push(header.clone());
            }
        }
    }

    // Для inner mode - оставляем только колонки, присутствующие во всех таблицах
    let result_columns = if mode == "inner" {
        let mut common_columns = Vec::new();
        for col in &column_order {
            let mut in_all = true;
            for table_rc in &tables {
                let table_ref = table_rc.borrow();
                if !table_ref.headers.contains(col) {
                    in_all = false;
                    break;
                }
            }
            if in_all {
                common_columns.push(col.clone());
            }
        }
        common_columns
    } else {
        column_order
    };

    // Создаем объединенные строки
    let mut merged_rows = Vec::new();
    
    for table_rc in &tables {
        let table_ref = table_rc.borrow();
        
        // Для каждой строки в таблице
        for row in &table_ref.rows {
            let mut new_row = Vec::new();
            
            // Для каждой колонки в результате
            for col_name in &result_columns {
                // Находим индекс колонки в исходной таблице
                if let Some(col_idx) = table_ref.headers.iter().position(|h| h == col_name) {
                    // Берем значение из соответствующей позиции в строке
                    if col_idx < row.len() {
                        new_row.push(row[col_idx].clone());
                    } else {
                        new_row.push(Value::Null);
                    }
                } else {
                    // Колонки нет в этой таблице - добавляем null
                    new_row.push(Value::Null);
                }
            }
            
            merged_rows.push(new_row);
        }
    }

    // Создаем результирующую таблицу
    let merged_table = Table::from_data(merged_rows, Some(result_columns));
    Value::Table(Rc::new(RefCell::new(merged_table)))
}

pub fn native_now(_args: &[Value]) -> Value {
    // Возвращаем текущее время в формате RFC3339 (ISO 8601)
    // Формат: YYYY-MM-DDTHH:MM:SSZ
    let now = Utc::now();
    Value::String(now.format("%Y-%m-%dT%H:%M:%SZ").to_string())
}

/// Безопасное получение parent пути в режиме --use-ve
/// Возвращает parent только если он находится внутри папки сессии
/// Если parent выходит за пределы папки сессии, возвращает None (не позволяет получить доступ к путям вне сессии)
pub fn safe_path_parent(path: &PathBuf) -> Option<PathBuf> {
    use crate::websocket::{get_user_session_path, get_use_ve};
    
    if !get_use_ve() {
        // В обычном режиме просто возвращаем parent как есть
        return path.parent().map(|p| p.to_path_buf());
    }
    
    let session_path = match get_user_session_path() {
        Some(p) => p,
        None => return None, // Нет пути сессии - не возвращаем parent
    };
    
    // Нормализуем session_path для корректного сравнения
    let session_path_normalized = match session_path.canonicalize() {
        Ok(p) => p,
        Err(_) => session_path.clone(),
    };
    
    // Получаем parent путь
    let parent = match path.parent() {
        Some(p) => p.to_path_buf(),
        None => return None,
    };
    
    // Канонизируем parent для проверки
    let parent_normalized = match parent.canonicalize() {
        Ok(p) => p,
        Err(_) => parent.clone(),
    };
    
    // Проверяем, что parent находится внутри папки сессии
    if parent_normalized.starts_with(&session_path_normalized) {
        // Если parent уже является самой папкой сессии, не возвращаем parent (защита от выхода наружу)
        if parent_normalized == session_path_normalized {
            None
        } else {
            Some(parent_normalized)
        }
    } else {
        // Parent выходит за пределы сессии - возвращаем None (не позволяем доступ к путям вне сессии)
        None
    }
}

/// Безопасное разрешение пути относительно папки сессии в режиме --use-ve
fn resolve_path_in_session(path: &PathBuf) -> Result<PathBuf, String> {
    use crate::websocket::{get_user_session_path, get_use_ve};
    
    if !get_use_ve() {
        // В обычном режиме просто возвращаем путь как есть
        return Ok(path.clone());
    }
    
    let session_path = match get_user_session_path() {
        Some(p) => p,
        None => return Err("Session path not available".to_string()),
    };
    
    // Нормализуем session_path для корректного сравнения
    let session_path_normalized = match session_path.canonicalize() {
        Ok(p) => p,
        Err(_) => {
            // Если канонизация не удалась, используем исходный путь
            // но нормализуем его (убираем лишние компоненты)
            session_path.clone()
        },
    };
    
    // Проверяем на directory traversal атаки
    let path_str = path.to_string_lossy().to_string();
    if path_str.contains("..") {
        return Err("Path traversal not allowed in --use-ve mode".to_string());
    }
    
    // Обрабатываем пустой путь и "." как текущую директорию сессии
    if path_str.is_empty() || path_str == "." {
        return Ok(session_path_normalized.clone());
    }
    
    // Проверяем абсолютные пути
    if path.is_absolute() {
        // Для абсолютных путей проверяем, что они внутри сессии
        // Пытаемся канонизировать оба пути для корректного сравнения
        let path_normalized = if path.exists() {
            match path.canonicalize() {
                Ok(p) => p,
                Err(_) => {
                    // Если канонизация не удалась, но путь существует,
                    // проверяем через starts_with с исходными путями
                    if path.starts_with(&session_path) || path.starts_with(&session_path_normalized) {
                        path.clone()
                    } else {
                        return Err("Access outside session directory not allowed".to_string());
                    }
                },
            }
        } else {
            // Если файл не существует, пытаемся канонизировать родительскую директорию
            // для проверки безопасности
            if let Some(parent) = path.parent() {
                match parent.canonicalize() {
                    Ok(p) => {
                        // Проверяем, что родительская директория внутри сессии
                        if p.starts_with(&session_path_normalized) {
                            path.clone()
                        } else {
                            // Дополнительная проверка через сравнение компонентов
                            let parent_components: Vec<_> = p.components().collect();
                            let session_components: Vec<_> = session_path_normalized.components().collect();
                            if parent_components.len() >= session_components.len() {
                                let mut matches = true;
                                for (i, session_comp) in session_components.iter().enumerate() {
                                    if i >= parent_components.len() || parent_components[i] != *session_comp {
                                        matches = false;
                                        break;
                                    }
                                }
                                if matches {
                                    path.clone()
                                } else {
                                    return Err("Path resolved outside session directory".to_string());
                                }
                            } else {
                                return Err("Path resolved outside session directory".to_string());
                            }
                        }
                    },
                    Err(_) => {
                        // Если родительская директория не может быть канонизирована,
                        // проверяем через starts_with
                        if parent.starts_with(&session_path) || parent.starts_with(&session_path_normalized) {
                            path.clone()
                        } else {
                            return Err("Access outside session directory not allowed".to_string());
                        }
                    },
                }
            } else {
                // Нет родительской директории - проверяем напрямую
                if path.starts_with(&session_path) || path.starts_with(&session_path_normalized) {
                    path.clone()
                } else {
                    return Err("Access outside session directory not allowed".to_string());
                }
            }
        };
        
        // Финальная проверка безопасности через канонизированные пути
        if path_normalized.starts_with(&session_path_normalized) {
            Ok(path_normalized)
        } else if path_normalized.starts_with(&session_path) {
            // Если session_path не был канонизирован, но путь начинается с него
            Ok(path_normalized)
        } else {
            // Дополнительная проверка через сравнение компонентов
            let path_components: Vec<_> = path_normalized.components().collect();
            let session_components: Vec<_> = session_path_normalized.components().collect();
            if path_components.len() >= session_components.len() {
                let mut matches = true;
                for (i, session_comp) in session_components.iter().enumerate() {
                    if i >= path_components.len() || path_components[i] != *session_comp {
                        matches = false;
                        break;
                    }
                }
                if matches {
                    Ok(path_normalized)
                } else {
                    Err("Path resolved outside session directory".to_string())
                }
            } else {
                Err("Access outside session directory not allowed".to_string())
            }
        }
    } else {
        // Относительный путь разрешаем относительно папки сессии
        let resolved = session_path.join(path);
        
        // Нормализуем путь и проверяем, что он все еще внутри сессии
        let normalized = if resolved.exists() {
            match resolved.canonicalize() {
                Ok(p) => p,
                Err(_) => resolved,
            }
        } else {
            // Если файл не существует, проверяем родительскую директорию
            if let Some(parent) = resolved.parent() {
                match parent.canonicalize() {
                    Ok(p) => {
                        if p.starts_with(&session_path_normalized) {
                            resolved
                        } else {
                            return Err("Path resolved outside session directory".to_string());
                        }
                    },
                    Err(_) => {
                        if parent.starts_with(&session_path) {
                            resolved
                        } else {
                            return Err("Path resolved outside session directory".to_string());
                        }
                    },
                }
            } else {
                resolved
            }
        };
        
        if normalized.starts_with(&session_path_normalized) || normalized.starts_with(&session_path) {
            Ok(normalized)
        } else {
            Err("Path resolved outside session directory".to_string())
        }
    }
}

pub fn native_getcwd(_args: &[Value]) -> Value {
    use crate::websocket::{get_use_ve};
    
    // Если включен режим use_ve, возвращаем пустой путь для безопасности
    if get_use_ve() {
        Value::Path(PathBuf::new()) // Пустой путь в режиме use_ve для безопасности
    } else {
        // Возвращаем текущую рабочую директорию
        match env::current_dir() {
            Ok(path) => Value::Path(path),
            Err(_) => Value::Path(PathBuf::new()), // При ошибке возвращаем пустой путь
        }
    }
}

pub fn native_list_files(args: &[Value]) -> Value {
    if args.is_empty() {
        return Value::Array(Rc::new(RefCell::new(Vec::new())));
    }

    // Первый аргумент - путь к директории
    let dir_path = match &args[0] {
        Value::Path(p) => p.clone(),
        Value::String(s) => PathBuf::from(s),
        _ => return Value::Array(Rc::new(RefCell::new(Vec::new()))),
    };

    let dir_path_str = dir_path.to_string_lossy().to_string();

    // Проверяем, является ли это SMB путем (lib://)
    if dir_path_str.starts_with("lib://") {
        // Извлекаем имя шары и путь к директории
        let path_without_prefix = &dir_path_str[6..]; // Убираем "lib://"
        let parts: Vec<&str> = path_without_prefix.splitn(2, '/').collect();
        
        if parts.is_empty() {
            return Value::Array(Rc::new(RefCell::new(Vec::new())));
        }
        
        let share_name = parts[0];
        let dir_path_on_share = if parts.len() > 1 { parts[1] } else { "" };
        
        // Получаем SmbManager из thread-local storage
        if let Some(smb_manager) = crate::vm::file_ops::get_smb_manager() {
            match smb_manager.lock().unwrap().list_files(share_name, dir_path_on_share) {
                Ok(files) => {
                    let file_values: Vec<Value> = files.iter()
                        .map(|f| {
                            // Конструируем полный путь для SMB
                            let full_path = if dir_path_on_share.is_empty() {
                                format!("lib://{}/{}", share_name, f)
                            } else {
                                format!("lib://{}/{}/{}", share_name, dir_path_on_share, f)
                            };
                            Value::Path(PathBuf::from(full_path))
                        })
                        .collect();
                    Value::Array(Rc::new(RefCell::new(file_values)))
                }
                Err(_) => Value::Array(Rc::new(RefCell::new(Vec::new()))),
            }
        } else {
            Value::Array(Rc::new(RefCell::new(Vec::new())))
        }
    } else {
        // Обычная локальная директория
        // Разрешаем путь относительно папки сессии в режиме --use-ve
        let resolved_path = match resolve_path_in_session(&dir_path) {
            Ok(p) => p,
            Err(err_msg) => {
                // При ошибке безопасности сохраняем сообщение об ошибке
                use crate::websocket::set_native_error;
                set_native_error(err_msg);
                return Value::Array(Rc::new(RefCell::new(Vec::new())));
            }
        };
        
        if !resolved_path.exists() || !resolved_path.is_dir() {
            return Value::Array(Rc::new(RefCell::new(Vec::new())));
        }
        
        match fs::read_dir(&resolved_path) {
            Ok(entries) => {
                let mut files = Vec::new();
                for entry in entries {
                    if let Ok(entry) = entry {
                        let entry_path = entry.path();
                        if let Some(file_name) = entry_path.file_name() {
                            if let Some(name_str) = file_name.to_str() {
                                // Пропускаем служебные файлы
                                if !name_str.starts_with(".") && name_str != ".DS_Store" {
                                    files.push(Value::Path(entry_path));
                                }
                            }
                        }
                    }
                }
                Value::Array(Rc::new(RefCell::new(files)))
            }
            Err(_) => Value::Array(Rc::new(RefCell::new(Vec::new()))),
        }
    }
}

// ============================================================================
// JOIN Operations Infrastructure
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum JoinType {
    Inner,
    Left,
    Right,
    Full,
    Cross,
    Semi,
    Anti,
}

#[derive(Debug, Clone)]
struct JoinKey {
    left_col: String,
    right_col: String,
}

// Парсинг ключей JOIN из Value
// Поддерживает форматы:
// - String: имя колонки (автоматическое сопоставление)
// - Array с кортежами: [("left_col", "right_col"), ...]
// - Array с одним кортежем: [("left_col", "right_col")]
fn parse_join_keys(value: &Value, left_table: &Table, right_table: &Table) -> Result<Vec<JoinKey>, String> {
    match value {
        Value::String(col_name) => {
            // Автоматическое сопоставление: ищем колонку с таким именем в обеих таблицах
            if left_table.get_column(col_name).is_some() && right_table.get_column(col_name).is_some() {
                Ok(vec![JoinKey {
                    left_col: col_name.clone(),
                    right_col: col_name.clone(),
                }])
            } else {
                Err(format!("Column '{}' not found in both tables", col_name))
            }
        }
        Value::Array(arr) => {
            let arr_ref = arr.borrow();
            if arr_ref.is_empty() {
                return Err("Join keys array cannot be empty".to_string());
            }
            
            let mut keys = Vec::new();
            for item in arr_ref.iter() {
                match item {
                    Value::Array(tuple) => {
                        let tuple_ref = tuple.borrow();
                        if tuple_ref.len() != 2 {
                            return Err("Join key tuple must have exactly 2 elements".to_string());
                        }
                        let left_col = match &tuple_ref[0] {
                            Value::String(s) => s.clone(),
                            _ => return Err("Join key must be a string".to_string()),
                        };
                        let right_col = match &tuple_ref[1] {
                            Value::String(s) => s.clone(),
                            _ => return Err("Join key must be a string".to_string()),
                        };
                        
                        if left_table.get_column(&left_col).is_none() {
                            return Err(format!("Column '{}' not found in left table", left_col));
                        }
                        if right_table.get_column(&right_col).is_none() {
                            return Err(format!("Column '{}' not found in right table", right_col));
                        }
                        
                        keys.push(JoinKey { left_col, right_col });
                    }
                    Value::String(col_name) => {
                        // Одиночная строка в массиве - автоматическое сопоставление
                        if left_table.get_column(col_name).is_some() && right_table.get_column(col_name).is_some() {
                            keys.push(JoinKey {
                                left_col: col_name.clone(),
                                right_col: col_name.clone(),
                            });
                        } else {
                            return Err(format!("Column '{}' not found in both tables", col_name));
                        }
                    }
                    _ => return Err("Join key must be a string or tuple [string, string]".to_string()),
                }
            }
            Ok(keys)
        }
        _ => Err("Join keys must be a string or array of tuples".to_string()),
    }
}

// Применение алиасов таблиц к именам колонок для разрешения коллизий
// Формат: <table_alias>.<column> (например, "left.id", "right.id")
fn apply_column_aliases(
    headers: &[String],
    alias: &str,
    existing_names: &std::collections::HashSet<String>,
) -> Vec<String> {
    headers
        .iter()
        .map(|h| {
            if existing_names.contains(h) {
                format!("{}.{}", alias, h)  // Конфликт: left.id
            } else {
                h.clone()  // Нет конфликта: name
            }
        })
        .collect()
}

// Сравнение значений для JOIN (с учетом nulls_equal)
// Примечание: в текущей реализации JOIN используется хеш-таблица (build_right_hash_table),
// поэтому эти функции не используются. Они оставлены на случай будущей реализации
// non-equi joins или других типов JOIN, где может понадобиться прямое сравнение.
#[allow(dead_code)]
fn values_match(left_val: &Value, right_val: &Value, nulls_equal: bool) -> bool {
    match (left_val, right_val) {
        (Value::Null, Value::Null) => nulls_equal,
        (Value::Null, _) | (_, Value::Null) => false,
        _ => left_val == right_val,
    }
}

// Сравнение ключей для JOIN
// Примечание: в текущей реализации JOIN используется хеш-таблица (build_right_hash_table),
// поэтому эта функция не используется. Она оставлена на случай будущей реализации
// non-equi joins или других типов JOIN, где может понадобиться прямое сравнение.
#[allow(dead_code)]
fn keys_match(left_row: &[Value], right_row: &[Value], keys: &[JoinKey], 
              left_table: &Table, right_table: &Table, nulls_equal: bool) -> bool {
    for key in keys {
        let left_idx = left_table.headers.iter().position(|h| h == &key.left_col);
        let right_idx = right_table.headers.iter().position(|h| h == &key.right_col);
        
        match (left_idx, right_idx) {
            (Some(li), Some(ri)) => {
                let left_val = &left_row[li];
                let right_val = &right_row[ri];
                if !values_match(left_val, right_val, nulls_equal) {
                    return false;
                }
            }
            _ => return false,
        }
    }
    true
}

// Построение хеш-таблицы для правой таблицы по ключам
use std::collections::HashMap;
use std::hash::Hash;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct KeyHash {
    values: Vec<Value>,
}

fn build_right_hash_table(
    right_table: &Table,
    keys: &[JoinKey],
    nulls_equal: bool,
) -> HashMap<KeyHash, Vec<usize>> {
    let mut hash_map = HashMap::new();
    
    for (row_idx, row) in right_table.rows.iter().enumerate() {
        let mut key_values = Vec::new();
        let mut valid_key = true;
        
        for key in keys {
            if let Some(col_idx) = right_table.headers.iter().position(|h| h == &key.right_col) {
                if col_idx < row.len() {
                    let val = &row[col_idx];
                    // Если nulls_equal=false и значение NULL, пропускаем эту строку
                    if matches!(val, Value::Null) && !nulls_equal {
                        valid_key = false;
                        break;
                    }
                    key_values.push(val.clone());
                } else {
                    valid_key = false;
                    break;
                }
            } else {
                valid_key = false;
                break;
            }
        }
        
        if valid_key {
            let key_hash = KeyHash { values: key_values };
            hash_map.entry(key_hash).or_insert_with(Vec::new).push(row_idx);
        }
    }
    
    hash_map
}

// Извлечение ключа из строки левой таблицы
fn extract_left_key(left_row: &[Value], keys: &[JoinKey], left_table: &Table) -> Option<KeyHash> {
    let mut key_values = Vec::new();
    
    for key in keys {
        if let Some(col_idx) = left_table.headers.iter().position(|h| h == &key.left_col) {
            if col_idx < left_row.len() {
                key_values.push(left_row[col_idx].clone());
            } else {
                return None;
            }
        } else {
            return None;
        }
    }
    
    Some(KeyHash { values: key_values })
}

// Универсальная функция JOIN для таблиц
// join(left: Table, right: Table, on: JoinKeys, type: JoinType = "inner", 
//      suffixes: (string, string) = ("_left", "_right"), nulls_equal: boolean = false) -> Table
pub fn native_table_join(args: &[Value]) -> Value {
    if args.len() < 3 {
        return Value::Null;
    }

    // Извлекаем left и right таблицы
    let left_table = match &args[0] {
        Value::Table(t) => t.borrow().clone(),
        _ => return Value::Null,
    };
    
    let right_table = match &args[1] {
        Value::Table(t) => t.borrow().clone(),
        _ => return Value::Null,
    };

    // Парсим тип JOIN (по умолчанию inner) - нужно проверить ДО парсинга ключей
    let join_type = if args.len() > 3 {
        match &args[3] {
            Value::String(s) => match s.as_str() {
                "inner" => JoinType::Inner,
                "left" => JoinType::Left,
                "right" => JoinType::Right,
                "full" => JoinType::Full,
                "cross" => JoinType::Cross,
                "semi" => JoinType::Semi,
                "anti" => JoinType::Anti,
                _ => JoinType::Inner,
            },
            _ => JoinType::Inner,
        }
    } else {
        JoinType::Inner
    };

    // Для CROSS JOIN не нужны ключи - пропускаем парсинг
    let keys = if join_type == JoinType::Cross {
        Vec::new()
    } else {
        // Парсим параметр on (ключи)
        match parse_join_keys(&args[2], &left_table, &right_table) {
            Ok(k) => k,
            Err(err_msg) => {
                use crate::websocket::set_native_error;
                set_native_error(err_msg);
                return Value::Null;
            }
        }
    };

    // Используем имена таблиц из table.name, если они установлены, иначе используем переданные алиасы или значения по умолчанию
    let left_alias = left_table.name.as_ref()
        .map(|n| n.clone())
        .unwrap_or_else(|| {
            if args.len() > 4 {
                match &args[4] {
                    Value::Array(arr) => {
                        let arr_ref = arr.borrow();
                        if arr_ref.len() >= 1 {
                            match &arr_ref[0] {
                                Value::String(s) => s.clone(),
                                _ => "left".to_string(),
                            }
                        } else {
                            "left".to_string()
                        }
                    }
                    _ => "left".to_string(),
                }
            } else {
                "left".to_string()
            }
        });
    
    let right_alias = right_table.name.as_ref()
        .map(|n| n.clone())
        .unwrap_or_else(|| {
            if args.len() > 4 {
                match &args[4] {
                    Value::Array(arr) => {
                        let arr_ref = arr.borrow();
                        if arr_ref.len() >= 2 {
                            match &arr_ref[1] {
                                Value::String(s) => s.clone(),
                                _ => "right".to_string(),
                            }
                        } else {
                            "right".to_string()
                        }
                    }
                    _ => "right".to_string(),
                }
            } else {
                "right".to_string()
            }
        });

    // Парсим nulls_equal (по умолчанию false)
    let nulls_equal = if args.len() > 5 {
        match &args[5] {
            Value::Bool(b) => *b,
            Value::Number(n) => *n != 0.0,
            _ => false,
        }
    } else {
        false
    };

    // CROSS JOIN - особый случай (декартово произведение)
    if join_type == JoinType::Cross {
        return perform_cross_join(&left_table, &right_table, &left_alias, &right_alias);
    }

    // Для остальных типов JOIN нужны ключи
    if keys.is_empty() {
        return Value::Null;
    }

    // Выполняем JOIN в зависимости от типа
    match join_type {
        JoinType::Inner => perform_inner_join(&left_table, &right_table, &keys, &left_alias, &right_alias, nulls_equal),
        JoinType::Left => perform_left_join(&left_table, &right_table, &keys, &left_alias, &right_alias, nulls_equal),
        JoinType::Right => perform_right_join(&left_table, &right_table, &keys, &left_alias, &right_alias, nulls_equal),
        JoinType::Full => perform_full_join(&left_table, &right_table, &keys, &left_alias, &right_alias, nulls_equal),
        JoinType::Semi => perform_semi_join(&left_table, &right_table, &keys, nulls_equal),
        JoinType::Anti => perform_anti_join(&left_table, &right_table, &keys, nulls_equal),
        JoinType::Cross => unreachable!(), // Уже обработано выше
    }
}

// Выполнение INNER JOIN
fn perform_inner_join(
    left_table: &Table,
    right_table: &Table,
    keys: &[JoinKey],
    left_alias: &str,
    right_alias: &str,
    nulls_equal: bool,
) -> Value {
    let right_hash = build_right_hash_table(right_table, keys, nulls_equal);
    
    let mut result_rows = Vec::new();
    let mut result_headers = Vec::new();
    
    // Создаем заголовки с учетом алиасов таблиц
    let left_headers_set: std::collections::HashSet<String> = left_table.headers.iter().cloned().collect();
    let right_headers_set: std::collections::HashSet<String> = right_table.headers.iter().cloned().collect();
    
    let mut left_headers = apply_column_aliases(&left_table.headers, left_alias, &right_headers_set);
    let mut right_headers = apply_column_aliases(&right_table.headers, right_alias, &left_headers_set);
    
    result_headers.append(&mut left_headers);
    result_headers.append(&mut right_headers);
    
    // Проходим по левой таблице и ищем совпадения
    for left_row in left_table.rows.iter() {
        if let Some(left_key) = extract_left_key(left_row, keys, left_table) {
            if let Some(right_indices) = right_hash.get(&left_key) {
                for &right_idx in right_indices {
                    if let Some(right_row) = right_table.get_row(right_idx) {
                        let mut new_row = left_row.clone();
                        new_row.extend_from_slice(right_row);
                        result_rows.push(new_row);
                    }
                }
            }
        }
    }
    
    Value::Table(Rc::new(RefCell::new(Table::from_data(result_rows, Some(result_headers)))))
}

// Выполнение LEFT JOIN
fn perform_left_join(
    left_table: &Table,
    right_table: &Table,
    keys: &[JoinKey],
    left_alias: &str,
    right_alias: &str,
    nulls_equal: bool,
) -> Value {
    let right_hash = build_right_hash_table(right_table, keys, nulls_equal);
    
    let mut result_rows = Vec::new();
    let mut result_headers = Vec::new();
    
    // Создаем заголовки с учетом алиасов таблиц
    let left_headers_set: std::collections::HashSet<String> = left_table.headers.iter().cloned().collect();
    let right_headers_set: std::collections::HashSet<String> = right_table.headers.iter().cloned().collect();
    
    let mut left_headers = apply_column_aliases(&left_table.headers, left_alias, &right_headers_set);
    let mut right_headers = apply_column_aliases(&right_table.headers, right_alias, &left_headers_set);
    
    result_headers.append(&mut left_headers);
    result_headers.append(&mut right_headers);
    
    // Создаем NULL-строку для правой таблицы
    let null_right_row: Vec<Value> = (0..right_table.headers.len()).map(|_| Value::Null).collect();
    
    // Проходим по левой таблице
    for left_row in left_table.rows.iter() {
        if let Some(left_key) = extract_left_key(left_row, keys, left_table) {
            if let Some(right_indices) = right_hash.get(&left_key) {
                // Есть совпадения - добавляем все совпадения
                for &right_idx in right_indices {
                    if let Some(right_row) = right_table.get_row(right_idx) {
                        let mut new_row = left_row.clone();
                        new_row.extend_from_slice(right_row);
                        result_rows.push(new_row);
                    }
                }
            } else {
                // Нет совпадений - добавляем строку с NULL справа
                let mut new_row = left_row.clone();
                new_row.extend_from_slice(&null_right_row);
                result_rows.push(new_row);
            }
        } else {
            // Невалидный ключ - добавляем строку с NULL справа
            let mut new_row = left_row.clone();
            new_row.extend_from_slice(&null_right_row);
            result_rows.push(new_row);
        }
    }
    
    Value::Table(Rc::new(RefCell::new(Table::from_data(result_rows, Some(result_headers)))))
}

// Выполнение RIGHT JOIN
fn perform_right_join(
    left_table: &Table,
    right_table: &Table,
    keys: &[JoinKey],
    left_alias: &str,
    right_alias: &str,
    nulls_equal: bool,
) -> Value {
    // RIGHT JOIN - это LEFT JOIN с переставленными таблицами
    let result = perform_left_join(right_table, left_table, 
        &keys.iter().map(|k| JoinKey {
            left_col: k.right_col.clone(),
            right_col: k.left_col.clone(),
        }).collect::<Vec<_>>(),
        right_alias, left_alias, nulls_equal);
    
    // Переставляем заголовки обратно
    if let Value::Table(t) = result {
        let table_ref = t.borrow();
        // Меняем порядок заголовков: сначала right, потом left
        let mut new_headers = Vec::new();
        let right_count = right_table.headers.len();
        let left_count = left_table.headers.len();
        
        // Сначала заголовки правой таблицы
        for i in 0..right_count {
            if i < table_ref.headers.len() {
                new_headers.push(table_ref.headers[i].clone());
            }
        }
        // Потом заголовки левой таблицы
        for i in right_count..(right_count + left_count) {
            if i < table_ref.headers.len() {
                new_headers.push(table_ref.headers[i].clone());
            }
        }
        
        // Переставляем данные в строках тоже
        let mut new_rows = Vec::new();
        for row in &table_ref.rows {
            let mut new_row = Vec::new();
            // Сначала данные правой таблицы
            for i in 0..right_count {
                if i < row.len() {
                    new_row.push(row[i].clone());
                }
            }
            // Потом данные левой таблицы
            for i in right_count..(right_count + left_count) {
                if i < row.len() {
                    new_row.push(row[i].clone());
                }
            }
            new_rows.push(new_row);
        }
        
        // Освобождаем заимствование перед созданием новой таблицы
        drop(table_ref);
        Value::Table(Rc::new(RefCell::new(Table::from_data(new_rows, Some(new_headers)))))
    } else {
        result
    }
}

// Выполнение FULL JOIN
fn perform_full_join(
    left_table: &Table,
    right_table: &Table,
    keys: &[JoinKey],
    left_alias: &str,
    right_alias: &str,
    nulls_equal: bool,
) -> Value {
    let right_hash = build_right_hash_table(right_table, keys, nulls_equal);
    
    let mut result_rows = Vec::new();
    let mut result_headers = Vec::new();
    
    // Создаем заголовки с учетом алиасов таблиц
    let left_headers_set: std::collections::HashSet<String> = left_table.headers.iter().cloned().collect();
    let right_headers_set: std::collections::HashSet<String> = right_table.headers.iter().cloned().collect();
    
    let mut left_headers = apply_column_aliases(&left_table.headers, left_alias, &right_headers_set);
    let mut right_headers = apply_column_aliases(&right_table.headers, right_alias, &left_headers_set);
    
    result_headers.append(&mut left_headers);
    result_headers.append(&mut right_headers);
    
    // Создаем NULL-строки
    let null_left_row: Vec<Value> = (0..left_table.headers.len()).map(|_| Value::Null).collect();
    let null_right_row: Vec<Value> = (0..right_table.headers.len()).map(|_| Value::Null).collect();
    
    let mut matched_right_indices = std::collections::HashSet::new();
    
    // Проходим по левой таблице
    for left_row in &left_table.rows {
        if let Some(left_key) = extract_left_key(left_row, keys, left_table) {
            if let Some(right_indices) = right_hash.get(&left_key) {
                for &right_idx in right_indices {
                    matched_right_indices.insert(right_idx);
                    if let Some(right_row) = right_table.get_row(right_idx) {
                        let mut new_row = left_row.clone();
                        new_row.extend_from_slice(right_row);
                        result_rows.push(new_row);
                    }
                }
            } else {
                // Нет совпадений справа - добавляем строку с NULL справа
                let mut new_row = left_row.clone();
                new_row.extend_from_slice(&null_right_row);
                result_rows.push(new_row);
            }
        } else {
            // Невалидный ключ - добавляем строку с NULL справа
            let mut new_row = left_row.clone();
            new_row.extend_from_slice(&null_right_row);
            result_rows.push(new_row);
        }
    }
    
    // Проходим по правой таблице и добавляем несовпадающие строки
    for (right_idx, right_row) in right_table.rows.iter().enumerate() {
        if !matched_right_indices.contains(&right_idx) {
            let mut new_row = null_left_row.clone();
            new_row.extend_from_slice(right_row);
            result_rows.push(new_row);
        }
    }
    
    Value::Table(Rc::new(RefCell::new(Table::from_data(result_rows, Some(result_headers)))))
}

// Выполнение SEMI JOIN (только строки left, колонки right не включаются)
fn perform_semi_join(
    left_table: &Table,
    right_table: &Table,
    keys: &[JoinKey],
    nulls_equal: bool,
) -> Value {
    let right_hash = build_right_hash_table(right_table, keys, nulls_equal);
    
    let mut result_rows = Vec::new();
    
    // Проходим по левой таблице и проверяем наличие совпадений
    for left_row in &left_table.rows {
        if let Some(left_key) = extract_left_key(left_row, keys, left_table) {
            if right_hash.contains_key(&left_key) {
                // Есть совпадение - добавляем строку из left
                result_rows.push(left_row.clone());
            }
        }
    }
    
    Value::Table(Rc::new(RefCell::new(Table::from_data(result_rows, Some(left_table.headers.clone())))))
}

// Выполнение ANTI JOIN (строки left без совпадений в right)
fn perform_anti_join(
    left_table: &Table,
    right_table: &Table,
    keys: &[JoinKey],
    nulls_equal: bool,
) -> Value {
    let right_hash = build_right_hash_table(right_table, keys, nulls_equal);
    
    let mut result_rows = Vec::new();
    
    // Проходим по левой таблице и проверяем отсутствие совпадений
    for left_row in &left_table.rows {
        if let Some(left_key) = extract_left_key(left_row, keys, left_table) {
            if !right_hash.contains_key(&left_key) {
                // Нет совпадения - добавляем строку из left
                result_rows.push(left_row.clone());
            }
        } else {
            // Невалидный ключ - добавляем строку
            result_rows.push(left_row.clone());
        }
    }
    
    Value::Table(Rc::new(RefCell::new(Table::from_data(result_rows, Some(left_table.headers.clone())))))
}

// Выполнение CROSS JOIN (декартово произведение)
fn perform_cross_join(
    left_table: &Table,
    right_table: &Table,
    left_alias: &str,
    right_alias: &str,
) -> Value {
    let mut result_rows = Vec::new();
    let mut result_headers = Vec::new();
    
    // Создаем заголовки с учетом алиасов таблиц
    let left_headers_set: std::collections::HashSet<String> = left_table.headers.iter().cloned().collect();
    let right_headers_set: std::collections::HashSet<String> = right_table.headers.iter().cloned().collect();
    
    let mut left_headers = apply_column_aliases(&left_table.headers, left_alias, &right_headers_set);
    let mut right_headers = apply_column_aliases(&right_table.headers, right_alias, &left_headers_set);
    
    result_headers.append(&mut left_headers);
    result_headers.append(&mut right_headers);
    
    // Декартово произведение
    for left_row in &left_table.rows {
        for right_row in &right_table.rows {
            let mut new_row = left_row.clone();
            new_row.extend_from_slice(right_row);
            result_rows.push(new_row);
        }
    }
    
    Value::Table(Rc::new(RefCell::new(Table::from_data(result_rows, Some(result_headers)))))
}

// ============================================================================
// Specialized JOIN Functions (syntactic sugar over native_table_join)
// ============================================================================

pub fn native_inner_join(args: &[Value]) -> Value {
    if args.len() < 3 {
        return Value::Null;
    }
    // Вызываем native_table_join с type="inner"
    let mut new_args = args.to_vec();
    
    // Если переданы два отдельных строковых аргумента (left_col, right_col)
    if new_args.len() == 4 {
        if let (Value::String(_), Value::String(_)) = (&new_args[2], &new_args[3]) {
            // Создаем массив кортежей: [["left_col", "right_col"]]
            let tuple = Value::Array(Rc::new(RefCell::new(vec![
                new_args[2].clone(),
                new_args[3].clone(),
            ])));
            let keys_array = Value::Array(Rc::new(RefCell::new(vec![tuple])));
            new_args[2] = keys_array;
            new_args[3] = Value::String("inner".to_string());
        } else {
            // Иначе перезаписываем тип join
            new_args[3] = Value::String("inner".to_string());
        }
    } else if new_args.len() == 3 {
        new_args.push(Value::String("inner".to_string()));
    } else {
        new_args[3] = Value::String("inner".to_string());
    }
    native_table_join(&new_args)
}

pub fn native_left_join(args: &[Value]) -> Value {
    if args.len() < 3 {
        return Value::Null;
    }
    let mut new_args = args.to_vec();
    
    // Если переданы два отдельных строковых аргумента (left_col, right_col)
    if new_args.len() == 4 {
        if let (Value::String(_), Value::String(_)) = (&new_args[2], &new_args[3]) {
            // Создаем массив кортежей: [["left_col", "right_col"]]
            let tuple = Value::Array(Rc::new(RefCell::new(vec![
                new_args[2].clone(),
                new_args[3].clone(),
            ])));
            let keys_array = Value::Array(Rc::new(RefCell::new(vec![tuple])));
            new_args[2] = keys_array;
            new_args[3] = Value::String("left".to_string());
        } else {
            // Иначе перезаписываем тип join
            new_args[3] = Value::String("left".to_string());
        }
    } else if new_args.len() == 3 {
        new_args.push(Value::String("left".to_string()));
    } else {
        new_args[3] = Value::String("left".to_string());
    }
    native_table_join(&new_args)
}

pub fn native_right_join(args: &[Value]) -> Value {
    if args.len() < 3 {
        return Value::Null;
    }
    let mut new_args = args.to_vec();
    
    // Если переданы два отдельных строковых аргумента (left_col, right_col)
    if new_args.len() == 4 {
        if let (Value::String(_), Value::String(_)) = (&new_args[2], &new_args[3]) {
            // Создаем массив кортежей: [["left_col", "right_col"]]
            let tuple = Value::Array(Rc::new(RefCell::new(vec![
                new_args[2].clone(),
                new_args[3].clone(),
            ])));
            let keys_array = Value::Array(Rc::new(RefCell::new(vec![tuple])));
            new_args[2] = keys_array;
            new_args[3] = Value::String("right".to_string());
        } else {
            // Иначе перезаписываем тип join
            new_args[3] = Value::String("right".to_string());
        }
    } else if new_args.len() == 3 {
        new_args.push(Value::String("right".to_string()));
    } else {
        new_args[3] = Value::String("right".to_string());
    }
    native_table_join(&new_args)
}

pub fn native_full_join(args: &[Value]) -> Value {
    if args.len() < 3 {
        return Value::Null;
    }
    let mut new_args = args.to_vec();
    if new_args.len() == 3 {
        new_args.push(Value::String("full".to_string()));
    } else {
        new_args[3] = Value::String("full".to_string());
    }
    native_table_join(&new_args)
}

pub fn native_cross_join(args: &[Value]) -> Value {
    if args.len() < 2 {
        return Value::Null;
    }
    // CROSS JOIN не требует ключей, используем пустой массив
    let mut new_args = vec![args[0].clone(), args[1].clone()];
    new_args.push(Value::Array(Rc::new(RefCell::new(Vec::new()))));
    new_args.push(Value::String("cross".to_string()));
    native_table_join(&new_args)
}

pub fn native_semi_join(args: &[Value]) -> Value {
    if args.len() < 3 {
        return Value::Null;
    }
    let mut new_args = args.to_vec();
    
    // Если переданы два отдельных строковых аргумента (left_col, right_col)
    if new_args.len() == 4 {
        if let (Value::String(_), Value::String(_)) = (&new_args[2], &new_args[3]) {
            // Создаем массив кортежей: [["left_col", "right_col"]]
            let tuple = Value::Array(Rc::new(RefCell::new(vec![
                new_args[2].clone(),
                new_args[3].clone(),
            ])));
            let keys_array = Value::Array(Rc::new(RefCell::new(vec![tuple])));
            new_args[2] = keys_array;
            new_args[3] = Value::String("semi".to_string());
        } else {
            // Иначе перезаписываем тип join
            new_args[3] = Value::String("semi".to_string());
        }
    } else if new_args.len() == 3 {
        new_args.push(Value::String("semi".to_string()));
    } else {
        new_args[3] = Value::String("semi".to_string());
    }
    native_table_join(&new_args)
}

pub fn native_anti_join(args: &[Value]) -> Value {
    if args.len() < 3 {
        return Value::Null;
    }
    let mut new_args = args.to_vec();
    
    // Если переданы два отдельных строковых аргумента (left_col, right_col)
    if new_args.len() == 4 {
        if let (Value::String(_), Value::String(_)) = (&new_args[2], &new_args[3]) {
            // Создаем массив кортежей: [["left_col", "right_col"]]
            let tuple = Value::Array(Rc::new(RefCell::new(vec![
                new_args[2].clone(),
                new_args[3].clone(),
            ])));
            let keys_array = Value::Array(Rc::new(RefCell::new(vec![tuple])));
            new_args[2] = keys_array;
            new_args[3] = Value::String("anti".to_string());
        } else {
            // Иначе перезаписываем тип join
            new_args[3] = Value::String("anti".to_string());
        }
    } else if new_args.len() == 3 {
        new_args.push(Value::String("anti".to_string()));
    } else {
        new_args[3] = Value::String("anti".to_string());
    }
    native_table_join(&new_args)
}

// ZIP JOIN - позиционное соединение по индексу строки
pub fn native_zip_join(args: &[Value]) -> Value {
    if args.len() < 2 {
        return Value::Null;
    }

    let left_table = match &args[0] {
        Value::Table(t) => t.borrow().clone(),
        _ => return Value::Null,
    };
    
    let right_table = match &args[1] {
        Value::Table(t) => t.borrow().clone(),
        _ => return Value::Null,
    };

    // Используем имена таблиц из table.name, если они установлены
    let left_alias = left_table.name.as_ref()
        .map(|n| n.clone())
        .unwrap_or_else(|| {
            if args.len() > 2 {
                match &args[2] {
                    Value::Array(arr) => {
                        let arr_ref = arr.borrow();
                        if arr_ref.len() >= 1 {
                            match &arr_ref[0] {
                                Value::String(s) => s.clone(),
                                _ => "left".to_string(),
                            }
                        } else {
                            "left".to_string()
                        }
                    }
                    _ => "left".to_string(),
                }
            } else {
                "left".to_string()
            }
        });
    
    let right_alias = right_table.name.as_ref()
        .map(|n| n.clone())
        .unwrap_or_else(|| {
            if args.len() > 2 {
                match &args[2] {
                    Value::Array(arr) => {
                        let arr_ref = arr.borrow();
                        if arr_ref.len() >= 2 {
                            match &arr_ref[1] {
                                Value::String(s) => s.clone(),
                                _ => "right".to_string(),
                            }
                        } else {
                            "right".to_string()
                        }
                    }
                    _ => "right".to_string(),
                }
            } else {
                "right".to_string()
            }
        });

    let mut result_rows = Vec::new();
    let mut result_headers = Vec::new();
    
    // Создаем заголовки с учетом алиасов таблиц
    let left_headers_set: std::collections::HashSet<String> = left_table.headers.iter().cloned().collect();
    let right_headers_set: std::collections::HashSet<String> = right_table.headers.iter().cloned().collect();
    
    let mut left_headers = apply_column_aliases(&left_table.headers, &left_alias, &right_headers_set);
    let mut right_headers = apply_column_aliases(&right_table.headers, &right_alias, &left_headers_set);
    
    result_headers.append(&mut left_headers);
    result_headers.append(&mut right_headers);
    
    // Соединяем строки по позиции (индексу)
    let min_len = std::cmp::min(left_table.rows.len(), right_table.rows.len());
    for i in 0..min_len {
        let mut new_row = left_table.rows[i].clone();
        new_row.extend_from_slice(&right_table.rows[i]);
        result_rows.push(new_row);
    }
    
    Value::Table(Rc::new(RefCell::new(Table::from_data(result_rows, Some(result_headers)))))
}

// APPLY JOIN / LATERAL JOIN - для каждой строки left вызывает функцию
// apply_join(left: Table, fn: Function, type: "inner" | "left" = "inner") -> Table
pub fn native_apply_join(args: &[Value]) -> Value {
    if args.len() < 2 {
        return Value::Null;
    }

    let left_table = match &args[0] {
        Value::Table(t) => t.borrow().clone(),
        _ => return Value::Null,
    };

    // Извлекаем функцию из аргументов
    let function_index = match &args[1] {
        Value::Function(idx) => *idx,
        _ => {
            // Если функция не передана, возвращаем Null
            return Value::Null;
        }
    };

    // Извлекаем тип JOIN (по умолчанию "inner")
    let join_type = if args.len() > 2 {
        match &args[2] {
            Value::String(s) => s.as_str(),
            _ => "inner",
        }
    } else {
        "inner"
    };

    // Получаем доступ к VM через thread-local storage
    // Сохраняем указатель на VM, чтобы восстановить контекст после вызовов нативных функций
    let vm_ptr = VM_CALL_CONTEXT.with(|ctx| {
        let ctx_ref = ctx.borrow();
        *ctx_ref
    });
    
    let mut result_rows = Vec::new();
    let mut result_headers = Vec::new();

    // Создаем заголовки результата (начнем с заголовков левой таблицы)
    result_headers.extend_from_slice(&left_table.headers);

    // Отслеживаем максимальное количество колонок правой таблицы для корректной обработки NULLs
    let mut max_right_columns = 0;
    let mut has_seen_table = false;

    // Для каждой строки левой таблицы вызываем функцию
    for left_row in &left_table.rows {
        // Восстанавливаем контекст перед каждым вызовом, так как он мог быть очищен
        // при вызове нативных функций внутри пользовательской функции
        if let Some(vm_ptr) = vm_ptr {
            VM_CALL_CONTEXT.with(|ctx| {
                *ctx.borrow_mut() = Some(vm_ptr);
            });
        }
        
        // Вызываем функцию с аргументом - массив значений строки
        let row_array = Value::Array(Rc::new(RefCell::new(left_row.clone())));
        let function_result = match call_user_function(function_index, &[row_array]) {
            Ok(result) => result,
            Err(_e) => {
                // Если произошла ошибка при вызове функции, пропускаем строку для inner join
                // или добавляем с NULLs для left join
                if join_type == "left" {
                    let new_row = left_row.clone();
                    // Добавим NULL значения позже, когда узнаем количество колонок
                    result_rows.push(new_row);
                }
                continue;
            }
        };
        

        match function_result {
            Value::Table(right_table) => {
                let right_table_ref = right_table.borrow();
                has_seen_table = true;
                
                // Обновляем максимальное количество колонок
                if right_table_ref.headers.len() > max_right_columns {
                    max_right_columns = right_table_ref.headers.len();
                }
                
                // Если заголовки результата еще не установлены полностью, добавляем заголовки правой таблицы
                if result_headers.len() == left_table.headers.len() {
                    // Проверяем конфликты имен колонок
                    let left_headers_set: std::collections::HashSet<String> = 
                        left_table.headers.iter().cloned().collect();
                    let mut right_headers = Vec::new();
                    for header in &right_table_ref.headers {
                        if left_headers_set.contains(header) {
                            right_headers.push(format!("right_{}", header));
                        } else {
                            right_headers.push(header.clone());
                        }
                    }
                    result_headers.extend_from_slice(&right_headers);
                }

                // Для каждой строки в результате функции добавляем комбинацию left_row + right_row
                for right_row in &right_table_ref.rows {
                    let mut new_row = left_row.clone();
                    new_row.extend_from_slice(right_row);
                    result_rows.push(new_row);
                }
            }
            Value::Null => {
                // Если функция вернула Null и это left join, добавляем строку с NULLs
                if join_type == "left" {
                    let new_row = left_row.clone();
                    result_rows.push(new_row);
                }
            }
            _ => {
                // Если функция вернула что-то другое, игнорируем для inner join
                // или добавляем с NULLs для left join
                if join_type == "left" {
                    let new_row = left_row.clone();
                    result_rows.push(new_row);
                }
            }
        }
    }

    // Если это left join и были строки без правой части, добавляем NULL значения
    if join_type == "left" && !has_seen_table && max_right_columns == 0 {
        // Если мы не видели ни одной таблицы, но есть строки с NULLs,
        // нужно добавить заголовки для правой части (пустые)
        // Но мы не знаем, какие заголовки должны быть, поэтому оставляем как есть
    } else if join_type == "left" && max_right_columns > 0 {
        // Добавляем NULL значения для строк, которые не имеют правой части
        // Находим строки, которые короче ожидаемого (left + right колонки)
        let expected_length = left_table.headers.len() + max_right_columns;
        for row in &mut result_rows {
            while row.len() < expected_length {
                row.push(Value::Null);
            }
        }
    }

    Value::Table(Rc::new(RefCell::new(Table::from_data(result_rows, Some(result_headers)))))
}

// ASOF JOIN - временное соединение
// asof_join(left: Table, right: Table, on: Column, by: Column | List<Column>, 
//           direction: "backward" | "forward" | "nearest" = "backward") -> Table
pub fn native_asof_join(args: &[Value]) -> Value {
    if args.len() < 3 {
        return Value::Null;
    }

    let left_table = match &args[0] {
        Value::Table(t) => t.borrow().clone(),
        _ => return Value::Null,
    };
    
    let right_table = match &args[1] {
        Value::Table(t) => t.borrow().clone(),
        _ => return Value::Null,
    };

    // Парсим временную колонку
    let time_column = match &args[2] {
        Value::String(s) => s.clone(),
        _ => return Value::Null,
    };

    // Проверяем наличие временной колонки в обеих таблицах
    if left_table.get_column(&time_column).is_none() {
        return Value::Null;
    }
    if right_table.get_column(&time_column).is_none() {
        return Value::Null;
    }

    // Парсим by (группирующие колонки) - опционально
    let by_columns: Vec<String> = if args.len() > 3 {
        match &args[3] {
            Value::String(s) => vec![s.clone()],
            Value::Array(arr) => {
                let arr_ref = arr.borrow();
                let mut cols = Vec::new();
                for val in arr_ref.iter() {
                    if let Value::String(s) = val {
                        cols.push(s.clone());
                    }
                }
                cols
            }
            _ => Vec::new(),
        }
    } else {
        Vec::new()
    };

    // Парсим direction (по умолчанию "backward")
    let direction = if args.len() > 4 {
        match &args[4] {
            Value::String(s) => s.as_str(),
            _ => "backward",
        }
    } else {
        "backward"
    };

    // Используем имена таблиц из table.name, если они установлены
    let left_alias = left_table.name.as_ref()
        .map(|n| n.clone())
        .unwrap_or_else(|| {
            if args.len() > 5 {
                match &args[5] {
                    Value::Array(arr) => {
                        let arr_ref = arr.borrow();
                        if arr_ref.len() >= 1 {
                            match &arr_ref[0] {
                                Value::String(s) => s.clone(),
                                _ => "left".to_string(),
                            }
                        } else {
                            "left".to_string()
                        }
                    }
                    _ => "left".to_string(),
                }
            } else {
                "left".to_string()
            }
        });
    
    let right_alias = right_table.name.as_ref()
        .map(|n| n.clone())
        .unwrap_or_else(|| {
            if args.len() > 5 {
                match &args[5] {
                    Value::Array(arr) => {
                        let arr_ref = arr.borrow();
                        if arr_ref.len() >= 2 {
                            match &arr_ref[1] {
                                Value::String(s) => s.clone(),
                                _ => "right".to_string(),
                            }
                        } else {
                            "right".to_string()
                        }
                    }
                    _ => "right".to_string(),
                }
            } else {
                "right".to_string()
            }
        });

    // Создаем индексы для временной колонки
    let left_time_idx = left_table.headers.iter().position(|h| h == &time_column).unwrap();
    let right_time_idx = right_table.headers.iter().position(|h| h == &time_column).unwrap();

    // Если есть by колонки, группируем данные
    let mut result_rows = Vec::new();
    let mut result_headers = Vec::new();
    
    // Создаем заголовки с учетом алиасов таблиц
    let left_headers_set: std::collections::HashSet<String> = left_table.headers.iter().cloned().collect();
    let right_headers_set: std::collections::HashSet<String> = right_table.headers.iter().cloned().collect();
    
    let mut left_headers = apply_column_aliases(&left_table.headers, &left_alias, &right_headers_set);
    let mut right_headers = apply_column_aliases(&right_table.headers, &right_alias, &left_headers_set);
    
    result_headers.append(&mut left_headers);
    result_headers.append(&mut right_headers);

    // Создаем NULL-строку для правой таблицы
    let null_right_row: Vec<Value> = (0..right_table.headers.len()).map(|_| Value::Null).collect();

    if by_columns.is_empty() {
        // Нет группировки - простой ASOF join
        // Сортируем правую таблицу по времени (для бинарного поиска)
        let mut right_indices: Vec<usize> = (0..right_table.rows.len()).collect();
        right_indices.sort_by(|&a, &b| {
            let time_a = &right_table.rows[a][right_time_idx];
            let time_b = &right_table.rows[b][right_time_idx];
            compare_values(time_a, time_b)
        });

        for left_row in &left_table.rows {
            let left_time = &left_row[left_time_idx];
            
            // Ищем ближайшую строку в правой таблице
            let mut best_match: Option<usize> = None;
            let mut best_diff: Option<f64> = None;

            for &right_idx in &right_indices {
                let right_row = &right_table.rows[right_idx];
                let right_time = &right_row[right_time_idx];

                // Вычисляем разницу времени (упрощенная версия - только для чисел)
                let diff = match (left_time, right_time) {
                    (Value::Number(l), Value::Number(r)) => {
                        let diff_val = match direction {
                            "backward" => *l - *r,  // left_time >= right_time
                            "forward" => *r - *l,   // right_time >= left_time
                            "nearest" => (*l - *r).abs(),
                            _ => *l - *r,
                        };
                        Some(diff_val)
                    }
                    _ => None,
                };

                if let Some(d) = diff {
                    let matches_direction = match direction {
                        "backward" => d >= 0.0,
                        "forward" => d >= 0.0,
                        "nearest" => true,
                        _ => d >= 0.0,
                    };

                    if matches_direction {
                        let should_update = match best_diff {
                            None => true,
                            Some(bd) => match direction {
                                "nearest" => d < bd,
                                _ => d < bd,
                            },
                        };

                        if should_update {
                            best_match = Some(right_idx);
                            best_diff = Some(d);
                        }
                    }
                }
            }

            if let Some(right_idx) = best_match {
                let mut new_row = left_row.clone();
                new_row.extend_from_slice(&right_table.rows[right_idx]);
                result_rows.push(new_row);
            } else {
                // Нет совпадения - добавляем строку с NULL справа (для left join семантики)
                let mut new_row = left_row.clone();
                new_row.extend_from_slice(&null_right_row);
                result_rows.push(new_row);
            }
        }
    } else {
        // Есть группировка - группируем по by колонкам
        // Создаем HashMap для группировки данных
        
        // Получаем индексы by колонок
        let mut by_indices_left = Vec::new();
        let mut by_indices_right = Vec::new();
        
        for by_col in &by_columns {
            if let Some(idx) = left_table.headers.iter().position(|h| h == by_col) {
                by_indices_left.push(idx);
            } else {
                return Value::Null; // Колонка не найдена
            }
            if let Some(idx) = right_table.headers.iter().position(|h| h == by_col) {
                by_indices_right.push(idx);
            } else {
                return Value::Null; // Колонка не найдена
            }
        }
        
        // Группируем левую таблицу по by колонкам
        let mut left_groups: std::collections::HashMap<Vec<Value>, Vec<Vec<Value>>> = std::collections::HashMap::new();
        for row in &left_table.rows {
            let key: Vec<Value> = by_indices_left.iter().map(|&idx| row[idx].clone()).collect();
            left_groups.entry(key).or_insert_with(Vec::new).push(row.clone());
        }
        
        // Группируем правую таблицу по by колонкам
        let mut right_groups: std::collections::HashMap<Vec<Value>, Vec<Vec<Value>>> = std::collections::HashMap::new();
        for row in &right_table.rows {
            let key: Vec<Value> = by_indices_right.iter().map(|&idx| row[idx].clone()).collect();
            right_groups.entry(key).or_insert_with(Vec::new).push(row.clone());
        }
        
        // Для каждой группы в левой таблице выполняем ASOF join
        for (group_key, left_group_rows) in left_groups {
            if let Some(right_group_rows) = right_groups.get(&group_key) {
                // Выполняем ASOF join для этой группы
                let group_results = asof_join_single_group(
                    &left_group_rows,
                    right_group_rows,
                    left_time_idx,
                    right_time_idx,
                    direction,
                    &null_right_row,
                );
                result_rows.extend(group_results);
            } else {
                // Группы нет в правой таблице - добавляем строки с NULLs (для left join семантики)
                for left_row in left_group_rows {
                    let mut new_row = left_row.clone();
                    new_row.extend_from_slice(&null_right_row);
                    result_rows.push(new_row);
                }
            }
        }
    }

    Value::Table(Rc::new(RefCell::new(Table::from_data(result_rows, Some(result_headers)))))
}

// JOIN ON - non-equi join с произвольным условием
// join_on(left: Table, right: Table, condition: Expr, type: JoinType = "inner") -> Table
// Примечание: Полная поддержка выражений требует дополнительной инфраструктуры.
// Пока реализована упрощенная версия для простых условий сравнения колонок.
pub fn native_join_on(args: &[Value]) -> Value {
    if args.len() < 3 {
        return Value::Null;
    }

    let left_table = match &args[0] {
        Value::Table(t) => t.borrow().clone(),
        _ => return Value::Null,
    };
    
    let right_table = match &args[1] {
        Value::Table(t) => t.borrow().clone(),
        _ => return Value::Null,
    };

    // Парсим условие - пока упрощенная версия
    // Ожидаем строку вида "left_col >= right_col" или массив ["left_col", ">=", "right_col"]
    let condition = &args[2];
    
    // Парсим тип JOIN (по умолчанию inner)
    let join_type = if args.len() > 3 {
        match &args[3] {
            Value::String(s) => match s.as_str() {
                "inner" => JoinType::Inner,
                "left" => JoinType::Left,
                "right" => JoinType::Right,
                "full" => JoinType::Full,
                _ => JoinType::Inner,
            },
            _ => JoinType::Inner,
        }
    } else {
        JoinType::Inner
    };

    // Используем имена таблиц из table.name, если они установлены
    let left_alias = left_table.name.as_ref()
        .map(|n| n.clone())
        .unwrap_or_else(|| {
            if args.len() > 4 {
                match &args[4] {
                    Value::Array(arr) => {
                        let arr_ref = arr.borrow();
                        if arr_ref.len() >= 1 {
                            match &arr_ref[0] {
                                Value::String(s) => s.clone(),
                                _ => "left".to_string(),
                            }
                        } else {
                            "left".to_string()
                        }
                    }
                    _ => "left".to_string(),
                }
            } else {
                "left".to_string()
            }
        });
    
    let right_alias = right_table.name.as_ref()
        .map(|n| n.clone())
        .unwrap_or_else(|| {
            if args.len() > 4 {
                match &args[4] {
                    Value::Array(arr) => {
                        let arr_ref = arr.borrow();
                        if arr_ref.len() >= 2 {
                            match &arr_ref[1] {
                                Value::String(s) => s.clone(),
                                _ => "right".to_string(),
                            }
                        } else {
                            "right".to_string()
                        }
                    }
                    _ => "right".to_string(),
                }
            } else {
                "right".to_string()
            }
        });

    // Парсим условие - упрощенная версия
    // Формат: ["left_col", ">=", "right_col"] или строка "left_col >= right_col"
    let (left_col, op, right_col) = match condition {
        Value::String(s) => {
            // Парсим строку вида "left_col >= right_col"
            let parts: Vec<&str> = s.split_whitespace().collect();
            if parts.len() >= 3 {
                (parts[0].to_string(), parts[1].to_string(), parts[2].to_string())
            } else {
                return Value::Null;
            }
        }
        Value::Array(arr) => {
            let arr_ref = arr.borrow();
            if arr_ref.len() >= 3 {
                let left = match &arr_ref[0] {
                    Value::String(s) => s.clone(),
                    _ => return Value::Null,
                };
                let op = match &arr_ref[1] {
                    Value::String(s) => s.clone(),
                    _ => return Value::Null,
                };
                let right = match &arr_ref[2] {
                    Value::String(s) => s.clone(),
                    _ => return Value::Null,
                };
                (left, op, right)
            } else {
                return Value::Null;
            }
        }
        _ => return Value::Null,
    };

    // Проверяем наличие колонок
    if left_table.get_column(&left_col).is_none() {
        return Value::Null;
    }
    if right_table.get_column(&right_col).is_none() {
        return Value::Null;
    }

    let left_col_idx = left_table.headers.iter().position(|h| h == &left_col).unwrap();
    let right_col_idx = right_table.headers.iter().position(|h| h == &right_col).unwrap();

    let mut result_rows = Vec::new();
    let mut result_headers = Vec::new();
    
    // Создаем заголовки с учетом алиасов таблиц
    let left_headers_set: std::collections::HashSet<String> = left_table.headers.iter().cloned().collect();
    let right_headers_set: std::collections::HashSet<String> = right_table.headers.iter().cloned().collect();
    
    let mut left_headers = apply_column_aliases(&left_table.headers, &left_alias, &right_headers_set);
    let mut right_headers = apply_column_aliases(&right_table.headers, &right_alias, &left_headers_set);
    
    result_headers.append(&mut left_headers);
    result_headers.append(&mut right_headers);

    // Создаем NULL-строки
    let null_left_row: Vec<Value> = (0..left_table.headers.len()).map(|_| Value::Null).collect();
    let null_right_row: Vec<Value> = (0..right_table.headers.len()).map(|_| Value::Null).collect();

    // Nested loop join с проверкой условия
    let mut matched_right_indices = std::collections::HashSet::new();

    for left_row in &left_table.rows {
        let left_val = &left_row[left_col_idx];
        let mut found_match = false;

        for (right_idx, right_row) in right_table.rows.iter().enumerate() {
            let right_val = &right_row[right_col_idx];
            
            // Проверяем условие
            let condition_met = match op.as_str() {
                ">" => compare_values(left_val, right_val) == std::cmp::Ordering::Greater,
                "<" => compare_values(left_val, right_val) == std::cmp::Ordering::Less,
                ">=" => {
                    let cmp = compare_values(left_val, right_val);
                    cmp == std::cmp::Ordering::Greater || cmp == std::cmp::Ordering::Equal
                }
                "<=" => {
                    let cmp = compare_values(left_val, right_val);
                    cmp == std::cmp::Ordering::Less || cmp == std::cmp::Ordering::Equal
                }
                "==" | "=" => compare_values(left_val, right_val) == std::cmp::Ordering::Equal,
                "!=" | "<>" => compare_values(left_val, right_val) != std::cmp::Ordering::Equal,
                _ => false,
            };

            if condition_met {
                found_match = true;
                matched_right_indices.insert(right_idx);
                let mut new_row = left_row.clone();
                new_row.extend_from_slice(right_row);
                result_rows.push(new_row);
            }
        }

        // Для LEFT JOIN добавляем строки без совпадений
        if !found_match && (join_type == JoinType::Left || join_type == JoinType::Full) {
            let mut new_row = left_row.clone();
            new_row.extend_from_slice(&null_right_row);
            result_rows.push(new_row);
        }
    }

    // Для RIGHT и FULL JOIN добавляем несовпадающие строки справа
    if join_type == JoinType::Right || join_type == JoinType::Full {
        for (right_idx, right_row) in right_table.rows.iter().enumerate() {
            if !matched_right_indices.contains(&right_idx) {
                let mut new_row = null_left_row.clone();
                new_row.extend_from_slice(right_row);
                result_rows.push(new_row);
            }
        }
    }

    Value::Table(Rc::new(RefCell::new(Table::from_data(result_rows, Some(result_headers)))))
}

// Применение суффиксов к колонкам таблицы после join
// Заменяет префиксы "left." и "right." на пользовательские суффиксы
pub fn native_table_suffixes(args: &[Value]) -> Value {
    if args.len() < 3 {
        return Value::Null;
    }

    // Аргументы приходят в порядке: [table, left_suffix, right_suffix]
    // Извлекаем таблицу
    let table = match &args[0] {
        Value::Table(t) => t.borrow().clone(),
        _ => return Value::Null,
    };

    // Извлекаем суффиксы
    let left_suffix = match &args[1] {
        Value::String(s) => s.clone(),
        _ => return Value::Null,
    };

    let right_suffix = match &args[2] {
        Value::String(s) => s.clone(),
        _ => return Value::Null,
    };

    // Создаем новую таблицу с переименованными колонками
    let mut new_headers = Vec::new();
    let mut column_mapping = std::collections::HashMap::new(); // старое имя -> новое имя

    // Определяем, какие колонки относятся к левой таблице, а какие к правой
    // Префиксы могут быть вида "left.", "right." или "table_name."
    // Определяем первые уникальные префиксы как левую таблицу, остальные - как правую
    let mut seen_prefixes = Vec::new();
    let mut prefix_to_table = std::collections::HashMap::new(); // prefix -> "left" или "right"
    
    // Сначала проходим по всем заголовкам и собираем уникальные префиксы
    for header in &table.headers {
        if let Some(dot_pos) = header.find('.') {
            let prefix = &header[..dot_pos];
            if !seen_prefixes.contains(&prefix.to_string()) {
                seen_prefixes.push(prefix.to_string());
            }
        }
    }
    
    // Первый уникальный префикс относится к левой таблице, остальные - к правой
    for (i, prefix) in seen_prefixes.iter().enumerate() {
        if i == 0 {
            prefix_to_table.insert(prefix.clone(), "left");
        } else {
            prefix_to_table.insert(prefix.clone(), "right");
        }
    }
    
    // Также добавляем стандартные префиксы
    prefix_to_table.insert("left".to_string(), "left");
    prefix_to_table.insert("right".to_string(), "right");
    
    // Проходим по всем заголовкам и переименовываем колонки с префиксами
    for header in &table.headers {
        let new_header = if let Some(dot_pos) = header.find('.') {
            let prefix = &header[..dot_pos];
            let base_name = &header[dot_pos + 1..];
            
            // Определяем, к какой таблице относится колонка
            if let Some(table_side) = prefix_to_table.get(prefix) {
                if *table_side == "left" {
                    // Колонка левой таблицы - применяем left_suffix
                    let new_name = format!("{}{}", base_name, left_suffix);
                    column_mapping.insert(header.clone(), new_name.clone());
                    new_name
                } else {
                    // Колонка правой таблицы - применяем right_suffix
                    let new_name = format!("{}{}", base_name, right_suffix);
                    column_mapping.insert(header.clone(), new_name.clone());
                    new_name
                }
            } else {
                // Неизвестный префикс - оставляем без изменений
                column_mapping.insert(header.clone(), header.clone());
                header.clone()
            }
        } else {
            // Колонка без префикса - оставляем без изменений
            column_mapping.insert(header.clone(), header.clone());
            header.clone()
        };
        new_headers.push(new_header);
    }

    // Создаем новую структуру колонок с переименованными ключами
    let mut new_columns = std::collections::HashMap::new();
    for (old_header, new_header) in &column_mapping {
        if let Some(column_data) = table.columns.get(old_header) {
            new_columns.insert(new_header.clone(), column_data.clone());
        }
    }

    // Создаем новую таблицу
    let mut new_table = Table::new();
    new_table.headers = new_headers;
    new_table.columns = new_columns;
    new_table.rows = table.rows.clone();
    new_table.name = table.name.clone();

    Value::Table(Rc::new(RefCell::new(new_table)))
}

