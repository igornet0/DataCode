// Table manipulation native functions

use crate::common::value::Value;
use crate::common::table::Table;
use std::path::PathBuf;
use std::rc::Rc;
use std::cell::RefCell;
use std::fs;
use std::io;

// Import resolve_path_in_session and format_path_for_error from file module
use super::file::{resolve_path_in_session, format_path_for_error};

pub fn native_table(args: &[Value]) -> Value {
    if args.is_empty() {
        return Value::Null;
    }

    // Первый аргумент - данные (массив массивов). Table::from_data stores rows only; columns built lazily.
    let data = match &args[0] {
        Value::Array(arr) => {
            let arr_ref = arr.borrow();
            let mut rows = Vec::with_capacity(arr_ref.len());
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

/// Применяет фильтр header к таблице
/// Если header - массив, фильтрует колонки (оставляет только указанные)
/// Если header - словарь, переименовывает колонки
fn apply_header_filter(table: Table, header_arg: Option<&Value>) -> Table {
    let header_arg = match header_arg {
        Some(v) => v,
        None => return table,
    };

    match header_arg {
        Value::Array(cols_arr) => {
            // Фильтруем колонки: оставляем только указанные в массиве
            let cols_arr_ref = cols_arr.borrow();
            let mut selected_cols = Vec::new();
            
            // Извлекаем имена колонок из массива
            for col_val in cols_arr_ref.iter() {
                match col_val {
                    Value::String(s) => selected_cols.push(s.clone()),
                    _ => {
                        // Игнорируем не-строковые значения
                        continue;
                    }
                }
            }
            
            if selected_cols.is_empty() {
                return table;
            }
            
            // Создаем индексы колонок для выборки
            let mut col_indices = Vec::new();
            let mut new_headers = Vec::new();
            
            for col_name in &selected_cols {
                if let Some(idx) = table.headers().iter().position(|h| h == col_name) {
                    col_indices.push(idx);
                    new_headers.push(col_name.clone());
                }
                // Игнорируем несуществующие колонки
            }
            
            if col_indices.is_empty() {
                return table;
            }
            
            // Создаем новые строки только с выбранными колонками
            let mut new_rows = Vec::new();
            let rr = table.rows_ref().unwrap();
            for row in rr.iter() {
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
            
            Table::from_data(new_rows, Some(new_headers))
        }
        Value::Object(rename_map_rc) => {
            // Переименовываем колонки согласно словарю
            let rename_map = rename_map_rc.borrow();
            let mut new_headers = Vec::new();
            
            for old_header in table.headers() {
                if let Some(new_name_val) = rename_map.get(old_header) {
                    match new_name_val {
                        Value::String(new_name) => {
                            // Переименовываем
                            new_headers.push(new_name.clone());
                        }
                        Value::Null => {
                            // Оставляем оригинальное имя
                            new_headers.push(old_header.clone());
                        }
                        _ => {
                            // Игнорируем некорректные значения, оставляем оригинальное имя
                            new_headers.push(old_header.clone());
                        }
                    }
                } else {
                    // Колонка не указана в словаре - оставляем как есть
                    new_headers.push(old_header.clone());
                }
            }
            
            // Создаем новую таблицу с переименованными заголовками
            // Данные остаются теми же, меняются только заголовки
            Table::from_data(table.rows_ref().unwrap().to_vec(), Some(new_headers))
        }
        _ => {
            // Некорректный тип - возвращаем таблицу без изменений
            table
        }
    }
}

/// Извлекает аргументы для read_file, определяя их по типу, а не только по позиции
/// Возвращает (header_row, sheet_name, header_arg)
fn extract_read_file_args(args: &[Value]) -> (usize, Option<String>, Option<&Value>) {
    let mut header_row = 0;
    let mut sheet_name: Option<String> = None;
    let mut header_arg: Option<&Value> = None;
    
    // Ищем header - это Array или Object (может быть на любой позиции после path)
    for arg in args.iter().skip(1) {
        if matches!(arg, Value::Array(_) | Value::Object(_)) {
            header_arg = Some(arg);
            break;
        }
    }
    
    // Определяем header_row и sheet_name по позиции и типу
    // Исключаем header из проверки
    if args.len() > 1 {
        // Проверяем, является ли args[1] header
        let is_header_1 = matches!(&args[1], Value::Array(_) | Value::Object(_));
        
        if !is_header_1 {
            match &args[1] {
                Value::Number(n) => {
                    // args[1] - это header_row
                    header_row = *n as usize;
                    
                    // Проверяем args[2] для sheet_name
                    if args.len() > 2 {
                        let is_header_2 = matches!(&args[2], Value::Array(_) | Value::Object(_));
                        if !is_header_2 {
                            if let Value::String(s) = &args[2] {
                                sheet_name = Some(s.clone());
                            }
                        }
                    }
                }
                Value::String(s) => {
                    // args[1] - это sheet_name
                    sheet_name = Some(s.clone());
                }
                _ => {}
            }
        } else {
            // args[1] - это header, проверяем args[2] для sheet_name
            if args.len() > 2 {
                let is_header_2 = matches!(&args[2], Value::Array(_) | Value::Object(_));
                if !is_header_2 {
                    if let Value::String(s) = &args[2] {
                        sheet_name = Some(s.clone());
                    }
                }
            }
        }
    }
    
    (header_row, sheet_name, header_arg)
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
    
    // Извлекаем аргументы по типу
    let (header_row, sheet_name, header_arg) = extract_read_file_args(args);
    
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
        
        // Получаем SmbManager из thread-local storage; hold lock only for read_file, then release before processing content.
        if let Some(smb_manager) = crate::vm::file_ops::get_smb_manager() {
            let read_result = {
                let guard = smb_manager.lock().unwrap();
                guard.read_file(share_name, file_path_on_share)
            };
            match read_result {
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
                                            let filtered_table = apply_header_filter(table, header_arg);
                                            return Value::Table(Rc::new(RefCell::new(filtered_table)));
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
                                    match read_xlsx_file(&temp_file, header_row, sheet_name.as_deref()) {
                                        Ok(table) => {
                                            let _ = fs::remove_file(&temp_file);
                                            let filtered_table = apply_header_filter(table, header_arg);
                                            return Value::Table(Rc::new(RefCell::new(filtered_table)));
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
            set_native_error(format!("File does not exist: {}", format_path_for_error(&resolved_path)));
            return Value::Null;
        }
        
        if !resolved_path.is_file() {
            use crate::websocket::set_native_error;
            set_native_error(format!("Path is not a file: {}", format_path_for_error(&resolved_path)));
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
                    Ok(table) => {
                        let filtered_table = apply_header_filter(table, header_arg);
                        Value::Table(Rc::new(RefCell::new(filtered_table)))
                    },
                    Err(e) => {
                        use crate::websocket::set_native_error;
                        set_native_error(format!("Error reading CSV file: {}", e));
                        Value::Null
                    },
                }
            }
            "xlsx" => {
                // Читаем XLSX файл
                match read_xlsx_file(&resolved_path, header_row, sheet_name.as_deref()) {
                    Ok(table) => {
                        let filtered_table = apply_header_filter(table, header_arg);
                        Value::Table(Rc::new(RefCell::new(filtered_table)))
                    },
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

fn col_type_from_value(v: &Value) -> &'static str {
    match v {
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Bool(_) => "bool",
        Value::Array(_) => "array",
        _ => "mixed",
    }
}

pub fn native_table_info(args: &[Value]) -> Value {
    if args.is_empty() {
        return Value::String("Table: empty".to_string());
    }

    match &args[0] {
        Value::Table(table) => {
            let row_count = table.borrow().len();
            let col_count = table.borrow().column_count();
            let headers = table.borrow().headers().clone();
            let mut info = format!("Table: {} rows, {} columns\n", row_count, col_count);
            info.push_str("Columns:\n");

            if table.borrow().is_view() {
                crate::vm::vm::with_current_stores(|store, heap| {
                    let t = table.borrow();
                    for header in &headers {
                        let len = crate::vm::table_ops::column_len(&*t, header).unwrap_or(0);
                        let col_type = (0..len)
                            .find_map(|i| crate::vm::table_ops::get_cell_value(&*t, i, header, store, heap))
                            .map(|v| col_type_from_value(&v).to_string())
                            .unwrap_or_else(|| if len == 0 { "empty".to_string() } else { "mixed".to_string() });
                        info.push_str(&format!("  - {}: {} ({} values)\n", header, col_type, len));
                    }
                });
            } else {
                let mut table_ref = table.borrow_mut();
                for header in &headers {
                    if let Some(column) = table_ref.get_column(header) {
                        let col_type = if column.is_empty() {
                            "empty".to_string()
                        } else {
                            let first_val = column.iter().find(|v| !matches!(v, Value::Null));
                            match first_val {
                                Some(v) => col_type_from_value(v).to_string(),
                                _ => "mixed".to_string(),
                            }
                        };
                        info.push_str(&format!("  - {}: {} ({} values)\n", header, col_type, column.len()));
                    }
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
            let headers = table_ref.headers().clone();

            let new_rows: Vec<Vec<Value>> = if table_ref.is_view() {
                crate::vm::vm::with_current_stores(|store, heap| {
                    let mut rows = Vec::with_capacity(take_n);
                    for i in 0..take_n {
                        if let Some(row) = crate::vm::table_ops::get_row(&*table_ref, i, store, heap) {
                            rows.push(row);
                        }
                    }
                    rows
                })
            } else {
                let mut new_rows = Vec::with_capacity(take_n);
                for i in 0..take_n {
                    if let Some(row) = table_ref.get_row(i) {
                        new_rows.push(row.to_vec());
                    }
                }
                new_rows
            };

            let new_table = Table::from_data(new_rows, Some(headers));
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
            let headers = table_ref.headers().clone();

            let new_rows: Vec<Vec<Value>> = if table_ref.is_view() {
                crate::vm::vm::with_current_stores(|store, heap| {
                    let mut rows = Vec::with_capacity(take_n);
                    for i in start_idx..row_count {
                        if let Some(row) = crate::vm::table_ops::get_row(&*table_ref, i, store, heap) {
                            rows.push(row);
                        }
                    }
                    rows
                })
            } else {
                let mut new_rows = Vec::with_capacity(take_n);
                for i in start_idx..row_count {
                    if let Some(row) = table_ref.get_row(i) {
                        new_rows.push(row.to_vec());
                    }
                }
                new_rows
            };

            let new_table = Table::from_data(new_rows, Some(headers));
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
            let mut col_indices = Vec::new();
            for col_name in &columns_to_select {
                if let Some(idx) = table_ref.headers().iter().position(|h| h == col_name) {
                    col_indices.push(idx);
                } else {
                    return Value::Null; // Колонка не найдена
                }
            }

            let new_rows: Vec<Vec<Value>> = if table_ref.is_view() {
                crate::vm::vm::with_current_stores(|store, heap| {
                    let n_rows = table_ref.len();
                    let mut rows = Vec::with_capacity(n_rows);
                    for i in 0..n_rows {
                        if let Some(row) = crate::vm::table_ops::get_row(&*table_ref, i, store, heap) {
                            let mut new_row = Vec::new();
                            for &idx in &col_indices {
                                new_row.push(row.get(idx).cloned().unwrap_or(Value::Null));
                            }
                            rows.push(new_row);
                        }
                    }
                    rows
                })
            } else {
                let mut new_rows = Vec::new();
                let rr = table_ref.rows_ref().unwrap();
                for row in rr.iter() {
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
                new_rows
            };

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
            let n_rows = table.borrow().len();
            let headers = table.borrow().headers().clone();
            let is_view = table.borrow().is_view();

            let sort_column: Vec<Value> = if is_view {
                crate::vm::vm::with_current_stores(|store, heap| {
                    let mut t = table.borrow_mut();
                    crate::vm::table_ops::get_column(&mut *t, &column_name, store, heap)
                }).unwrap_or_default()
            } else {
                table.borrow_mut().get_column(&column_name).map(|c| c.clone()).unwrap_or_default()
            };
            if sort_column.len() != n_rows {
                return Value::Null;
            }

            let mut indices: Vec<usize> = (0..n_rows).collect();
            indices.sort_by(|&a, &b| {
                let cmp = compare_values(&sort_column[a], &sort_column[b]);
                if ascending { cmp } else { cmp.reverse() }
            });

            let new_rows: Vec<Vec<Value>> = if is_view {
                crate::vm::vm::with_current_stores(|store, heap| {
                    let t = table.borrow();
                    indices.iter()
                        .filter_map(|&idx| crate::vm::table_ops::get_row(&*t, idx, store, heap))
                        .collect()
                })
            } else {
                let table_ref = table.borrow();
                indices.iter()
                    .filter_map(|&idx| table_ref.get_row(idx).map(|r| r.to_vec()))
                    .collect()
            };

            let new_table = Table::from_data(new_rows, Some(headers));
            Value::Table(Rc::new(RefCell::new(new_table)))
        }
        _ => Value::Null,
    }
}

/// Ядро фильтрации таблицы. Используется из native_table_where и из opcode TableFilter в VM.
pub fn table_where_impl(
    table: &Rc<RefCell<Table>>,
    column_name: &str,
    operator: &str,
    filter_value: &Value,
) -> Value {
    let headers = table.borrow().headers().clone();
    let is_view = table.borrow().is_view();

    let filter_column: Vec<Value> = if is_view {
        crate::vm::vm::with_current_stores(|store, heap| {
            let mut t = table.borrow_mut();
            crate::vm::table_ops::get_column(&mut *t, column_name, store, heap)
        })
        .unwrap_or_default()
    } else {
        table
            .borrow_mut()
            .get_column(column_name)
            .map(|c| c.clone())
            .unwrap_or_default()
    };

    let matching_indices: Vec<usize> = filter_column
        .iter()
        .enumerate()
        .filter(|(_, val)| {
            match operator {
                ">" => compare_values(val, filter_value) == std::cmp::Ordering::Greater,
                "<" => compare_values(val, filter_value) == std::cmp::Ordering::Less,
                ">=" => {
                    let cmp = compare_values(val, filter_value);
                    cmp == std::cmp::Ordering::Greater || cmp == std::cmp::Ordering::Equal
                }
                "<=" => {
                    let cmp = compare_values(val, filter_value);
                    cmp == std::cmp::Ordering::Less || cmp == std::cmp::Ordering::Equal
                }
                "==" | "=" => compare_values(val, filter_value) == std::cmp::Ordering::Equal,
                "!=" | "<>" => compare_values(val, filter_value) != std::cmp::Ordering::Equal,
                _ => false,
            }
        })
        .map(|(i, _)| i)
        .collect();

    let new_rows: Vec<Vec<Value>> = if is_view {
        crate::vm::vm::with_current_stores(|store, heap| {
            let t = table.borrow();
            matching_indices
                .iter()
                .filter_map(|&idx| crate::vm::table_ops::get_row(&*t, idx, store, heap))
                .collect()
        })
    } else {
        let table_ref = table.borrow();
        matching_indices
            .iter()
            .filter_map(|&idx| table_ref.get_row(idx).map(|r| r.to_vec()))
            .collect()
    };

    let new_table = Table::from_data(new_rows, Some(headers));
    Value::Table(Rc::new(RefCell::new(new_table)))
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
        Value::Table(table) => table_where_impl(table, &column_name, operator, &filter_value),
        _ => Value::Null,
    }
}

pub fn native_show_table(args: &[Value]) -> Value {
    if args.is_empty() {
        return Value::Null;
    }

    match &args[0] {
        Value::Table(table) => {
            let len = table.borrow().len();
            if len == 0 {
                println!("Empty table");
                return Value::Null;
            }
            let headers: Vec<String> = table.borrow().headers().clone();
            let is_view = table.borrow().is_view();

            let max_show = len.min(20);
            let (col_widths, rows_to_show): (Vec<usize>, usize) = if is_view {
                crate::vm::vm::with_current_stores(|store, heap| {
                    let t = table.borrow();
                    let mut col_widths: Vec<usize> = headers.iter().map(|h| h.len()).collect();
                    for row_idx in 0..max_show {
                        for (col_i, header) in headers.iter().enumerate() {
                            if let Some(v) = crate::vm::table_ops::get_cell_value(&*t, row_idx, header, store, heap) {
                                let w = v.to_string().len();
                                if col_widths[col_i] < w {
                                    col_widths[col_i] = w;
                                }
                            }
                        }
                    }
                    (col_widths.into_iter().map(|w| w.max(3)).collect(), max_show)
                })
            } else {
                let table_ref = table.borrow();
                let mut col_widths: Vec<usize> = headers.iter().map(|h| h.len()).collect();
                let rr = table_ref.rows_ref().unwrap();
                for row_idx in 0..rr.len() {
                    if let Some(row) = rr.row(row_idx) {
                        for (col_i, val) in row.iter().enumerate() {
                            if col_i < col_widths.len() {
                                let w = val.to_string().len();
                                if col_widths[col_i] < w {
                                    col_widths[col_i] = w;
                                }
                            }
                        }
                    }
                }
                (col_widths.into_iter().map(|w| w.max(3)).collect(), rr.len().min(20))
            };

            // Печатаем верхнюю границу
            print!("┌");
            for (i, &width) in col_widths.iter().enumerate() {
                if i > 0 {
                    print!("┬");
                }
                print!("{}", "─".repeat(width + 2));
            }
            println!("┐");
            print!("│");
            for (i, header) in headers.iter().enumerate() {
                if i > 0 {
                    print!("│");
                }
                print!(" {:<width$} ", header, width = col_widths[i]);
            }
            println!("│");
            print!("├");
            for (i, &width) in col_widths.iter().enumerate() {
                if i > 0 {
                    print!("┼");
                }
                print!("{}", "─".repeat(width + 2));
            }
            println!("┤");

            for row_idx in 0..rows_to_show {
                let row_vals: Vec<Value> = if is_view {
                    crate::vm::vm::with_current_stores(|store, heap| {
                        let t = table.borrow();
                        crate::vm::table_ops::get_row(&*t, row_idx, store, heap).unwrap_or_default()
                    })
                } else {
                    table.borrow().get_row(row_idx).map(|r| r.to_vec()).unwrap_or_default()
                };
                print!("│");
                for (i, val) in row_vals.iter().enumerate() {
                    if i > 0 {
                        print!("│");
                    }
                    let w = col_widths.get(i).copied().unwrap_or(3);
                    print!(" {:<width$} ", val.to_string(), width = w);
                }
                println!("│");
            }

            print!("└");
            for (i, &width) in col_widths.iter().enumerate() {
                if i > 0 {
                    print!("┴");
                }
                print!("{}", "─".repeat(width + 2));
            }
            println!("┘");

            if len > max_show {
                println!("... ({} more rows)", len - max_show);
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
        let rows: Vec<Vec<Value>> = if let Some(rr) = table_ref.rows_ref() {
            rr.to_vec()
        } else {
            crate::vm::vm::with_current_stores(|store, heap| {
                (0..table_ref.len())
                    .filter_map(|i| crate::vm::table_ops::get_row(&*table_ref, i, store, heap))
                    .collect()
            })
        };
        let new_table = Table::from_data(rows, Some(table_ref.headers().clone()));
        return Value::Table(Rc::new(RefCell::new(new_table)));
    }

    // Собираем все уникальные колонки
    let mut all_columns_set = HashSet::new();
    let mut column_order = Vec::new();
    
    // Сначала добавляем колонки первой таблицы для сохранения порядка
    let first_table = tables[0].borrow();
    for header in first_table.headers() {
        if all_columns_set.insert(header.clone()) {
            column_order.push(header.clone());
        }
    }
    
    // Затем добавляем колонки из остальных таблиц
    for table_rc in &tables[1..] {
        let table_ref = table_rc.borrow();
        for header in table_ref.headers() {
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
                if !table_ref.headers().contains(col) {
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

        let rows: Vec<Vec<Value>> = if let Some(rr) = table_ref.rows_ref() {
            rr.iter().map(|r| r.to_vec()).collect()
        } else {
            crate::vm::vm::with_current_stores(|store, heap| {
                (0..table_ref.len())
                    .filter_map(|i| crate::vm::table_ops::get_row(&*table_ref, i, store, heap))
                    .collect()
            })
        };

        for row in rows {
            let mut new_row = Vec::new();

            for col_name in &result_columns {
                if let Some(col_idx) = table_ref.headers().iter().position(|h| h == col_name) {
                    if col_idx < row.len() {
                        new_row.push(row[col_idx].clone());
                    } else {
                        new_row.push(Value::Null);
                    }
                } else {
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
