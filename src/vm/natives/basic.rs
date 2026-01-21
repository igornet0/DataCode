// Basic native functions: print, len, range, type conversions, typeof, isinstance

use crate::common::value::Value;
use std::rc::Rc;
use std::cell::RefCell;

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

