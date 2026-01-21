// Table manipulation native functions

use crate::common::value::Value;
use crate::common::table::Table;
use std::path::PathBuf;
use std::rc::Rc;
use std::cell::RefCell;
use std::fs;
use std::io;

// Import resolve_path_in_session from file module
use super::file::resolve_path_in_session;

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

pub fn native_read_file(args: &[Value]) -> Value {
    if args.is_empty() {
        return Value::Null;
    }

    // Первый аргумент - путь к файлу
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

pub fn compare_values(a: &Value, b: &Value) -> std::cmp::Ordering {
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
                
                let cmp = compare_values(val_a, val_b);
                
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
