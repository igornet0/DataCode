// Table manipulation native functions

use crate::common::table::Table;
use crate::common::range_args::range_len;
use crate::common::value::{IterableInner, Value};
use std::cell::RefCell;
use std::fs;
use std::io;
use std::path::PathBuf;
use std::rc::Rc;

// Import resolve_path_in_session and format_path_for_error from file module
use super::file::{format_path_for_error, resolve_path_in_session};

pub use super::date_format::try_parse_date;

fn column_source_len(v: &Value) -> Option<usize> {
    match v {
        Value::Array(a) => Some(a.borrow().len()),
        Value::Iterable(rc) => match &*rc.borrow() {
            IterableInner::Range {
                current,
                end,
                step,
            } => Some(range_len(*current, *end, *step)),
            _ => None,
        },
        _ => None,
    }
}

fn column_source_get(v: &Value, row: usize) -> Value {
    match v {
        Value::Array(a) => a
            .borrow()
            .get(row)
            .cloned()
            .unwrap_or(Value::Null),
        Value::Iterable(rc) => match &*rc.borrow() {
            IterableInner::Range {
                current,
                end,
                step,
            } => {
                let val = current + row as i64 * step;
                if (*step > 0 && val >= *end) || (*step < 0 && val <= *end) {
                    Value::Null
                } else {
                    Value::Number(val as f64)
                }
            }
            _ => Value::Null,
        },
        _ => Value::Null,
    }
}

pub fn native_table_add_row(args: &[Value]) -> Value {
    use crate::vm::store_convert::load_value;
    use crate::vm::vm::with_current_stores;

    if args.len() < 2 {
        crate::websocket::set_native_error(
            "TypeError: add_row() expects (table, array)".to_string(),
        );
        return Value::Null;
    }

    let table_rc = match &args[0] {
        Value::Table(t) => Rc::clone(t),
        _ => {
            crate::websocket::set_native_error(
                "TypeError: add_row() expects (table, array)".to_string(),
            );
            return Value::Null;
        }
    };

    let row = match &args[1] {
        Value::Array(arr) => arr.borrow().clone(),
        _ => {
            crate::websocket::set_native_error(
                "TypeError: add_row() expects (table, array)".to_string(),
            );
            return Value::Null;
        }
    };

    if table_rc.borrow().is_view() {
        let materialized = with_current_stores(|store, heap| {
            table_rc
                .borrow()
                .materialize_with(|id| load_value(id, store, heap))
        });
        *table_rc.borrow_mut() = materialized;
    }

    let result = table_rc.borrow_mut().add_row(row);
    match result {
        Ok(()) => Value::Table(table_rc),
        Err(msg) => {
            crate::websocket::set_native_error(msg);
            Value::Null
        }
    }
}

pub fn native_table(args: &[Value]) -> Value {
    if args.is_empty() {
        return Value::Null;
    }

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

    // Column-oriented `table([col1, col2, …], headers)` — transpose to rows.
    if let (Some(ref hdrs), Value::Array(outer)) = (&headers, &args[0]) {
        let outer_ref = outer.borrow();
        if !outer_ref.is_empty() && !hdrs.is_empty() && outer_ref.len() == hdrs.len() {
            let col_lens: Vec<usize> = outer_ref
                .iter()
                .filter_map(|v| column_source_len(v))
                .collect();
            if col_lens.len() == hdrs.len() {
                let n_rows = col_lens[0];
                let column_oriented =
                    n_rows > 0 && col_lens.iter().all(|&l| l == n_rows) && n_rows != hdrs.len();
                if column_oriented {
                    let mut rows = Vec::with_capacity(n_rows);
                    for r in 0..n_rows {
                        let mut row = Vec::with_capacity(hdrs.len());
                        for col in outer_ref.iter() {
                            row.push(column_source_get(col, r));
                        }
                        rows.push(row);
                    }
                    let table = Table::from_data(rows, headers);
                    return Value::Table(Rc::new(RefCell::new(table)));
                }
            }
        }
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

    let table = Table::from_data(data, headers);
    Value::Table(Rc::new(RefCell::new(table)))
}

fn read_csv_file(path: &PathBuf) -> Result<Table, io::Error> {
    use csv::ReaderBuilder;

    let mut reader = ReaderBuilder::new().has_headers(true).from_path(path)?;

    // Читаем заголовки
    let headers: Vec<String> = reader.headers()?.iter().map(|s| s.to_string()).collect();

    // Читаем данные
    let mut rows = Vec::new();
    for result in reader.records() {
        let record = result?;
        let row: Vec<Value> = record
            .iter()
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
                } else if let Some(d) = try_parse_date(field) {
                    Value::Date(d)
                } else {
                    Value::String(field.to_string())
                }
            })
            .collect();
        rows.push(row);
    }

    Ok(Table::from_data(rows, Some(headers)))
}

fn read_xlsx_file(
    path: &PathBuf,
    header_row: usize,
    sheet_name: Option<&str>,
) -> Result<Table, Box<dyn std::error::Error>> {
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
        let values: Vec<Value> = row
            .iter()
            .map(|cell| match cell {
                calamine::Data::Int(n) => Value::Number(*n as f64),
                calamine::Data::Float(n) => Value::Number(*n),
                calamine::Data::String(s) => Value::String(s.clone()),
                calamine::Data::Bool(b) => Value::Bool(*b),
                calamine::Data::DateTime(dt) => try_parse_date(&dt.to_string())
                    .map(Value::Date)
                    .unwrap_or_else(|| Value::String(dt.to_string())),
                calamine::Data::DateTimeIso(s) => try_parse_date(s)
                    .map(Value::Date)
                    .unwrap_or_else(|| Value::String(s.clone())),
                calamine::Data::DurationIso(s) => try_parse_date(s)
                    .map(Value::Date)
                    .unwrap_or_else(|| Value::String(s.clone())),
                calamine::Data::Error(_) => Value::Null,
                calamine::Data::Empty => Value::Null,
            })
            .collect();

        if row_idx == header_row {
            // Это строка заголовков
            headers = values
                .iter()
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
        headers = (0..num_cols).map(|i| format!("Column_{}", i)).collect();
    }

    Ok(Table::from_data(rows, Some(headers)))
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
    match crate::file_io::read_value(args) {
        Ok(v) => v,
        Err(msg) => {
            crate::websocket::set_native_error(msg);
            Value::Null
        }
    }
}

#[allow(dead_code)]
fn native_read_file_legacy(args: &[Value]) -> Value {
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
        let (share_name, file_path_on_share) = match crate::file_io::parse_lib_smb_path(
            &file_path_str,
        ) {
            Ok(parts) => parts,
            Err(err_msg) => {
                use crate::websocket::set_native_error;
                set_native_error(err_msg);
                return Value::Null;
            }
        };

        // Получаем SmbManager из thread-local storage; hold lock only for read_file, then release before processing content.
        if let Some(smb_manager) = crate::vm::file_ops::get_smb_manager() {
            let read_result = {
                let guard = smb_manager.lock().unwrap();
                guard.read_file(&share_name, &file_path_on_share)
            };
            match read_result {
                Ok(content) => {
                    // Определяем тип файла по расширению
                    let extension = std::path::Path::new(&file_path_on_share)
                        .extension()
                        .and_then(|ext| ext.to_str())
                        .unwrap_or("")
                        .to_lowercase();

                    match extension.as_str() {
                        "csv" => {
                            // Парсим CSV из байтов
                            use std::io::Write;
                            let temp_file = std::env::temp_dir()
                                .join(format!("datacode_smb_{}.csv", std::process::id()));
                            if let Ok(mut file) = fs::File::create(&temp_file) {
                                if file.write_all(&content).is_ok() {
                                    match read_csv_file(&temp_file) {
                                        Ok(table) => {
                                            let _ = fs::remove_file(&temp_file);
                                            let filtered_table =
                                                crate::file_io::apply_header_filter(table, header_arg);
                                            return Value::Table(Rc::new(RefCell::new(
                                                filtered_table,
                                            )));
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
                            let temp_file = std::env::temp_dir()
                                .join(format!("datacode_smb_{}.xlsx", std::process::id()));
                            if let Ok(mut file) = fs::File::create(&temp_file) {
                                if file.write_all(&content).is_ok() {
                                    match read_xlsx_file(
                                        &temp_file,
                                        header_row,
                                        sheet_name.as_deref(),
                                    ) {
                                        Ok(table) => {
                                            let _ = fs::remove_file(&temp_file);
                                            let filtered_table =
                                                crate::file_io::apply_header_filter(table, header_arg);
                                            return Value::Table(Rc::new(RefCell::new(
                                                filtered_table,
                                            )));
                                        }
                                        Err(_) => {
                                            let _ = fs::remove_file(&temp_file);
                                        }
                                    }
                                }
                            }
                            Value::Null
                        }
                        "txt" | "text" => match String::from_utf8(content) {
                            Ok(text) => Value::String(text),
                            Err(_) => Value::Null,
                        },
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
            set_native_error(format!(
                "File does not exist: {}",
                format_path_for_error(&resolved_path)
            ));
            return Value::Null;
        }

        if !resolved_path.is_file() {
            use crate::websocket::set_native_error;
            set_native_error(format!(
                "Path is not a file: {}",
                format_path_for_error(&resolved_path)
            ));
            return Value::Null;
        }

        // Проверяем расширение файла
        let extension = resolved_path
            .extension()
            .and_then(|ext| ext.to_str())
            .unwrap_or("")
            .to_lowercase();

        match extension.as_str() {
            "csv" => {
                // Читаем CSV файл
                match read_csv_file(&resolved_path) {
                    Ok(table) => {
                        let filtered_table =
                            crate::file_io::apply_header_filter(table, header_arg);
                        Value::Table(Rc::new(RefCell::new(filtered_table)))
                    }
                    Err(e) => {
                        use crate::websocket::set_native_error;
                        set_native_error(format!("Error reading CSV file: {}", e));
                        Value::Null
                    }
                }
            }
            "xlsx" => {
                // Читаем XLSX файл
                match read_xlsx_file(&resolved_path, header_row, sheet_name.as_deref()) {
                    Ok(table) => {
                        let filtered_table =
                            crate::file_io::apply_header_filter(table, header_arg);
                        Value::Table(Rc::new(RefCell::new(filtered_table)))
                    }
                    Err(e) => {
                        use crate::websocket::set_native_error;
                        set_native_error(format!("Error reading XLSX file: {}", e));
                        Value::Null
                    }
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
                    }
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
                    }
                }
            }
        }
    }
}

pub fn compare_values(a: &Value, b: &Value) -> std::cmp::Ordering {
    match crate::common::value_ord::value_partial_cmp(a, b) {
        Ok(o) => o,
        Err(_) => a.to_string().cmp(&b.to_string()),
    }
}

fn col_type_from_value(v: &Value) -> &'static str {
    match v {
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Bool(_) => "bool",
        Value::Date(_) => "date",
        Value::Duration(_) => "duration",
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
                            .find_map(|i| {
                                crate::vm::table_ops::get_cell_value(&*t, i, header, store, heap)
                            })
                            .map(|v| col_type_from_value(&v).to_string())
                            .unwrap_or_else(|| {
                                if len == 0 {
                                    "empty".to_string()
                                } else {
                                    "mixed".to_string()
                                }
                            });
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
                        info.push_str(&format!(
                            "  - {}: {} ({} values)\n",
                            header,
                            col_type,
                            column.len()
                        ));
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
                        if let Some(row) =
                            crate::vm::table_ops::get_row(&*table_ref, i, store, heap)
                        {
                            rows.push(row);
                        }
                    }
                    rows
                })
            } else if let Some(t) = table_ref.slice_rows_owned(0, take_n) {
                return Value::Table(Rc::new(RefCell::new(t)));
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
            let start_idx = row_count.saturating_sub(take_n);
            let headers = table_ref.headers().clone();

            let new_rows: Vec<Vec<Value>> = if table_ref.is_view() {
                crate::vm::vm::with_current_stores(|store, heap| {
                    let mut rows = Vec::with_capacity(take_n);
                    for i in start_idx..row_count {
                        if let Some(row) =
                            crate::vm::table_ops::get_row(&*table_ref, i, store, heap)
                        {
                            rows.push(row);
                        }
                    }
                    rows
                })
            } else if let Some(t) = table_ref.slice_rows_owned(start_idx, take_n) {
                return Value::Table(Rc::new(RefCell::new(t)));
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

fn table_col_error(msg: impl Into<String>) -> Value {
    crate::websocket::set_native_error(msg.into());
    Value::Null
}

fn materialize_table_rows(table: &Table) -> Vec<Vec<Value>> {
    let n_rows = table.len();
    if table.is_view() {
        crate::vm::vm::with_current_stores(|store, heap| {
            (0..n_rows)
                .filter_map(|i| crate::vm::table_ops::get_row(table, i, store, heap))
                .collect()
        })
    } else {
        table
            .rows_ref()
            .map(|rr| rr.iter().map(|r| r.to_vec()).collect())
            .unwrap_or_default()
    }
}

fn parse_column_name_list(arg: &Value, headers: &[String]) -> Result<Vec<String>, Value> {
    let names: Vec<String> = match arg {
        Value::String(s) => vec![s.clone()],
        Value::Array(arr) => arr
            .borrow()
            .iter()
            .filter_map(|v| match v {
                Value::String(s) => Some(s.clone()),
                _ => None,
            })
            .collect(),
        _ => {
            return Err(table_col_error(
                "TypeError: column must be a string or array of strings",
            ));
        }
    };
    if names.is_empty() {
        return Err(table_col_error(
            "TypeError: column must be a non-empty string or array of strings",
        ));
    }
    for name in &names {
        if !headers.iter().any(|h| h == name) {
            return Err(table_col_error(format!(
                "KeyError: column '{}' not found in table",
                name
            )));
        }
    }
    Ok(names)
}

/// Select columns by name; returns `KeyError` if any column is missing.
pub fn table_select_impl(table: &Table, columns: Vec<String>) -> Result<Table, Value> {
    let headers = table.headers();
    let mut col_indices = Vec::with_capacity(columns.len());
    for col_name in &columns {
        let Some(idx) = headers.iter().position(|h| h == col_name) else {
            return Err(table_col_error(format!(
                "KeyError: column '{}' not found in table",
                col_name
            )));
        };
        col_indices.push(idx);
    }

    if !table.is_view() {
        if let Some(t) = table.select_columns_owned(&col_indices, columns.clone()) {
            return Ok(t);
        }
    }

    let new_rows: Vec<Vec<Value>> = if table.is_view() {
        crate::vm::vm::with_current_stores(|store, heap| {
            let n_rows = table.len();
            let mut rows = Vec::with_capacity(n_rows);
            for i in 0..n_rows {
                if let Some(row) = crate::vm::table_ops::get_row(table, i, store, heap) {
                    let mut new_row = Vec::with_capacity(col_indices.len());
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
        if let Some(rr) = table.rows_ref() {
            for row in rr.iter() {
                let mut new_row = Vec::with_capacity(col_indices.len());
                for &idx in &col_indices {
                    new_row.push(row.get(idx).cloned().unwrap_or(Value::Null));
                }
                new_rows.push(new_row);
            }
        }
        new_rows
    };

    Ok(Table::from_data(new_rows, Some(columns)))
}

/// Rename columns via object map or single old/new pair.
pub fn table_rename_impl(table: &Table, rename_arg: &Value, new_name: Option<&str>) -> Result<Table, Value> {
    let headers = table.headers().clone();

    if let Some(new) = new_name {
        let old = match rename_arg {
            Value::String(s) => s.as_str(),
            _ => {
                return Err(table_col_error(
                    "TypeError: table_rename() expects column names as strings",
                ));
            }
        };
        if !headers.iter().any(|h| h == old) {
            return Err(table_col_error(format!(
                "KeyError: column '{}' not found in table",
                old
            )));
        }
        let new_headers: Vec<String> = headers
            .iter()
            .map(|h| if h == old { new.to_string() } else { h.clone() })
            .collect();
        if !table.is_view() {
            if let Some(t) = table.clone_flat_with_headers(new_headers.clone()) {
                return Ok(t);
            }
        }
        let rows = materialize_table_rows(table);
        return Ok(Table::from_data(rows, Some(new_headers)));
    }

    let rows = materialize_table_rows(table);

    match rename_arg {
        Value::Object(_) => {
            let owned = Table::from_data(rows, Some(headers));
            Ok(crate::file_io::apply_header_filter(
                owned,
                Some(rename_arg),
            ))
        }
        _ => Err(table_col_error(
            "TypeError: table_rename() expects an object map or two string column names",
        )),
    }
}

/// Drop one or more columns by name.
pub fn table_drop_column_impl(table: &Table, drop_arg: &Value) -> Result<Table, Value> {
    let headers = table.headers();
    let to_drop = parse_column_name_list(drop_arg, headers)?;
    let remaining: Vec<String> = headers
        .iter()
        .filter(|h| !to_drop.iter().any(|d| d == *h))
        .cloned()
        .collect();
    if remaining.is_empty() {
        return Err(table_col_error(
            "ValueError: cannot drop all columns from table",
        ));
    }
    table_select_impl(table, remaining)
}

/// Append a column with a scalar value or per-row array.
pub fn table_add_column_impl(
    table: &Table,
    name: &str,
    values_arg: Option<&Value>,
) -> Result<Table, Value> {
    let headers = table.headers();
    if headers.iter().any(|h| h == name) {
        return Err(table_col_error(format!(
            "ValueError: column '{}' already exists",
            name
        )));
    }

    let n_rows = table.len();
    let column_values: Vec<Value> = match values_arg {
        None => vec![Value::Null; n_rows],
        Some(Value::Array(arr)) => {
            let vals: Vec<Value> = arr.borrow().iter().cloned().collect();
            if vals.len() != n_rows {
                return Err(table_col_error(format!(
                    "ValueError: array length {} does not match table row count {}",
                    vals.len(),
                    n_rows
                )));
            }
            vals
        }
        Some(scalar) => vec![scalar.clone(); n_rows],
    };

    if !table.is_view() {
        if let Some(t) = table.append_column_owned(name, &column_values) {
            return Ok(t);
        }
    }

    let rows = materialize_table_rows(table);
    let mut new_headers = headers.to_vec();
    new_headers.push(name.to_string());
    let new_rows: Vec<Vec<Value>> = rows
        .into_iter()
        .zip(column_values)
        .map(|(mut row, val)| {
            row.push(val);
            row
        })
        .collect();

    Ok(Table::from_data(new_rows, Some(new_headers)))
}

fn validate_map_callback(func: &Value, expected_arity: usize) -> Result<(), Value> {
    match func {
        Value::NativeFunction(_) => {
            if expected_arity == 1 {
                Ok(())
            } else {
                Err(table_col_error(format!(
                    "TypeError: map callback must have arity {}, got 1",
                    expected_arity
                )))
            }
        }
        Value::Function(fn_idx) => {
            let Some(vm_ptr) = crate::vm::vm::current_vm_ptr() else {
                return Err(table_col_error("map: VM context not available"));
            };
            unsafe {
                let vm = &*vm_ptr;
                let arity = vm
                    .get_functions()
                    .get(*fn_idx)
                    .map(|fun| fun.arity)
                    .unwrap_or(0);
                if arity != expected_arity {
                    return Err(table_col_error(format!(
                        "TypeError: map callback must have arity {}, got {}",
                        expected_arity, arity
                    )));
                }
            }
            Ok(())
        }
        Value::ModuleFunction { .. } => Ok(()),
        _ => Err(table_col_error(
            "TypeError: table_map() third argument must be a callable",
        )),
    }
}

fn parse_value_map_rules(mappings: &Value) -> Result<Vec<(Value, Value)>, Value> {
    match mappings {
        Value::Object(obj) => {
            let obj_ref = obj.borrow();
            let entries = obj_ref.str_key_entries_cloned();
            if entries.is_empty() {
                return Err(table_col_error(
                    "TypeError: table_value_map() mappings object must not be empty",
                ));
            }
            Ok(entries
                .into_iter()
                .map(|(k, v)| (Value::String(k), v))
                .collect())
        }
        Value::Array(arr) => {
            let arr_ref = arr.borrow();
            if arr_ref.is_empty() {
                return Err(table_col_error(
                    "TypeError: table_value_map() mappings array must not be empty",
                ));
            }
            let mut rules = Vec::with_capacity(arr_ref.len());
            for item in arr_ref.iter() {
                let Value::Object(obj) = item else {
                    return Err(table_col_error(
                        "TypeError: table_value_map() mappings array must contain objects",
                    ));
                };
                let obj_ref = obj.borrow();
                let from = obj_ref
                    .str_key_get("from")
                    .or_else(|| obj_ref.str_key_get("old"))
                    .cloned();
                let to = obj_ref
                    .str_key_get("to")
                    .or_else(|| obj_ref.str_key_get("new"))
                    .cloned();
                match (from, to) {
                    (Some(f), Some(t)) => rules.push((f, t)),
                    _ => {
                        return Err(table_col_error(
                            "TypeError: mapping object must contain from/to or old/new",
                        ));
                    }
                }
            }
            Ok(rules)
        }
        _ => Err(table_col_error(
            "TypeError: table_value_map() mappings must be an object or array of mapping objects",
        )),
    }
}

pub fn table_value_map_impl(table: &Table, column: &str, mappings: &Value) -> Result<Table, Value> {
    let headers = table.headers();
    let Some(col_idx) = headers.iter().position(|h| h == column) else {
        return Err(table_col_error(format!(
            "KeyError: column '{}' not found in table",
            column
        )));
    };
    let rules = parse_value_map_rules(mappings)?;
    if !table.is_view() {
        if let Some(t) = table.map_column_owned(col_idx, |current| {
            rules
                .iter()
                .find_map(|(from, to)| (current == *from).then(|| to.clone()))
                .unwrap_or(current)
        }) {
            return Ok(t);
        }
    }
    let rows = materialize_table_rows(table);
    let mapped_rows: Vec<Vec<Value>> = rows
        .into_iter()
        .map(|mut row| {
            let current = row.get(col_idx).cloned().unwrap_or(Value::Null);
            let mapped = rules
                .iter()
                .find_map(|(from, to)| (current == *from).then(|| to.clone()))
                .unwrap_or(current);
            row[col_idx] = mapped;
            row
        })
        .collect();
    Ok(Table::from_data(mapped_rows, Some(headers.to_vec())))
}

/// Apply `func` to each cell in `column`; returns a new table with that column replaced.
pub fn table_map_impl(table: &Table, column: &str, func: &Value) -> Result<Table, Value> {
    use super::utils::invoke_value_callable;
    use crate::common::error::LangError;
    use crate::vm::vm::{current_vm_ptr, VmExecutionContext, VM_CALL_CONTEXT};

    let headers = table.headers();
    let Some(col_idx) = headers.iter().position(|h| h == column) else {
        return Err(table_col_error(format!(
            "KeyError: column '{}' not found in table",
            column
        )));
    };

    validate_map_callback(func, 1)?;

    let vm_ptr = current_vm_ptr();
    if !table.is_view() {
        if let crate::common::table::TableData::Owned {
            flat,
            num_cols,
            headers,
            ..
        } = &table.data
        {
            let nc = *num_cols;
            let mut out = flat.clone();
            let n_rows = if nc == 0 { 0 } else { out.len() / nc };
            for row in 0..n_rows {
                let idx = row * nc + col_idx;
                let cell = out[idx].clone();
                if let Some(vm_ptr) = vm_ptr {
                    VM_CALL_CONTEXT.with(|ctx| {
                        *ctx.borrow_mut() = Some(VmExecutionContext { vm: vm_ptr });
                    });
                }
                let new_val = match invoke_value_callable(func, &[cell]) {
                    Ok(v) => v,
                    Err(e) => {
                        let message = match e {
                            LangError::LexError { message, .. }
                            | LangError::ParseError { message, .. }
                            | LangError::SemanticError { message, .. }
                            | LangError::RuntimeError { message, .. } => message,
                        };
                        return Err(table_col_error(message));
                    }
                };
                out[idx] = new_val;
            }
            return Ok(Table::from_flat_owned(out, nc, headers.clone()));
        }
    }

    let rows = materialize_table_rows(table);
    let mut mapped_rows = Vec::with_capacity(rows.len());

    for mut row in rows {
        let cell = row[col_idx].clone();
        if let Some(vm_ptr) = vm_ptr {
            VM_CALL_CONTEXT.with(|ctx| {
                *ctx.borrow_mut() = Some(VmExecutionContext { vm: vm_ptr });
            });
        }
        let new_val = match invoke_value_callable(func, &[cell]) {
            Ok(v) => v,
            Err(e) => {
                let message = match e {
                    LangError::LexError { message, .. }
                    | LangError::ParseError { message, .. }
                    | LangError::SemanticError { message, .. }
                    | LangError::RuntimeError { message, .. } => message,
                };
                return Err(table_col_error(message));
            }
        };
        row[col_idx] = new_val;
        mapped_rows.push(row);
    }

    Ok(Table::from_data(mapped_rows, Some(headers.to_vec())))
}

/// Apply `fn` to each cell of a column reference; returns a materialized array.
pub fn column_map_impl(
    table: &Table,
    column_name: &str,
    func: &Value,
) -> Result<Vec<Value>, Value> {
    use super::utils::invoke_value_callable;
    use crate::common::error::LangError;
    use crate::vm::vm::{current_vm_ptr, VmExecutionContext, VM_CALL_CONTEXT};

    if !table.has_column(column_name) {
        return Err(table_col_error(format!(
            "KeyError: column '{}' not found in table",
            column_name
        )));
    }

    validate_map_callback(func, 1)?;

    let rows = materialize_table_rows(table);
    let headers = table.headers();
    let col_idx = headers
        .iter()
        .position(|h| h == column_name)
        .expect("has_column implied index exists");

    let vm_ptr = current_vm_ptr();
    let mut out = Vec::with_capacity(rows.len());

    for row in rows {
        let cell = row[col_idx].clone();
        if let Some(vm_ptr) = vm_ptr {
            VM_CALL_CONTEXT.with(|ctx| {
                *ctx.borrow_mut() = Some(VmExecutionContext { vm: vm_ptr });
            });
        }
        let new_val = match invoke_value_callable(func, &[cell]) {
            Ok(v) => v,
            Err(e) => {
                let message = match e {
                    LangError::LexError { message, .. }
                    | LangError::ParseError { message, .. }
                    | LangError::SemanticError { message, .. }
                    | LangError::RuntimeError { message, .. } => message,
                };
                return Err(table_col_error(message));
            }
        };
        out.push(new_val);
    }

    Ok(out)
}

/// `column.map(fn)` — materialized array (not lazy iterable).
pub fn native_column_map(args: &[Value]) -> Value {
    if args.len() != 2 {
        return table_col_error("TypeError: column.map() expects 1 argument (function)");
    }
    let Value::ColumnReference { table, column_name } = &args[0] else {
        return table_col_error("TypeError: column.map() expects a column reference as receiver");
    };
    match column_map_impl(&table.borrow(), column_name, &args[1]) {
        Ok(vals) => Value::Array(Rc::new(RefCell::new(vals))),
        Err(v) => v,
    }
}

/// Apply `fn` to zipped cells of several columns; returns a materialized array.
pub fn columns_map_impl(
    table: &Table,
    column_names: &[String],
    func: &Value,
) -> Result<Vec<Value>, Value> {
    use super::utils::invoke_value_callable;
    use crate::common::error::LangError;
    use crate::vm::vm::{current_vm_ptr, VmExecutionContext, VM_CALL_CONTEXT};

    if column_names.len() < 2 {
        return Err(table_col_error(
            "TypeError: columns.map() requires at least 2 columns",
        ));
    }
    for name in column_names {
        if !table.has_column(name) {
            return Err(table_col_error(format!(
                "KeyError: column '{}' not found in table",
                name
            )));
        }
    }

    validate_map_callback(func, column_names.len())?;

    let n_rows = table.len();
    let vm_ptr = current_vm_ptr();
    let mut out = Vec::with_capacity(n_rows);

    for row_i in 0..n_rows {
        let args = crate::vm::vm::with_current_stores(|store, heap| {
            crate::vm::table_ops::row_cells(table, row_i, column_names, store, heap)
        })
        .ok_or_else(|| {
            table_col_error(format!(
                "internal: missing cell at row {} for columns {:?}",
                row_i, column_names
            ))
        })?;
        if let Some(vm_ptr) = vm_ptr {
            VM_CALL_CONTEXT.with(|ctx| {
                *ctx.borrow_mut() = Some(VmExecutionContext { vm: vm_ptr });
            });
        }
        let new_val = match invoke_value_callable(func, &args) {
            Ok(v) => v,
            Err(e) => {
                let message = match e {
                    LangError::LexError { message, .. }
                    | LangError::ParseError { message, .. }
                    | LangError::SemanticError { message, .. }
                    | LangError::RuntimeError { message, .. } => message,
                };
                return Err(table_col_error(message));
            }
        };
        out.push(new_val);
    }

    Ok(out)
}

/// `columns.map(fn)` — materialized array from a multi-column reference.
pub fn native_columns_map(args: &[Value]) -> Value {
    if args.len() != 2 {
        return table_col_error("TypeError: columns.map() expects 1 argument (function)");
    }
    let Value::ColumnsReference {
        table,
        column_names,
    } = &args[0]
    else {
        return table_col_error(
            "TypeError: columns.map() expects a columns reference as receiver",
        );
    };
    match columns_map_impl(&table.borrow(), column_names, &args[1]) {
        Ok(vals) => Value::Array(Rc::new(RefCell::new(vals))),
        Err(v) => v,
    }
}

/// Split one column by delimiter string (no VM callback).
pub fn table_split_column_delim_impl(
    table: &Table,
    column: &str,
    delimiter: &str,
    new_names: &[String],
) -> Result<Table, Value> {
    if new_names.is_empty() {
        return Err(table_col_error("ValueError: new_columns must not be empty"));
    }

    let headers = table.headers();
    let Some(col_idx) = headers.iter().position(|h| h == column) else {
        return Err(table_col_error(format!(
            "KeyError: column '{}' not found in table",
            column
        )));
    };

    let mut seen_new = std::collections::HashSet::new();
    for name in new_names {
        if headers.iter().any(|h| h == name) {
            return Err(table_col_error(format!(
                "ValueError: column '{}' already exists",
                name
            )));
        }
        if !seen_new.insert(name.as_str()) {
            return Err(table_col_error(format!(
                "ValueError: duplicate new column name '{}'",
                name
            )));
        }
    }

    let n_cols = new_names.len();
    let n_rows = table.len();

    if !table.is_view() {
        let mut column_values: Vec<Vec<Value>> = vec![Vec::with_capacity(n_rows); n_cols];
        for row_idx in 0..n_rows {
            let owned_storage;
            let text: &str = match table.get_row(row_idx).and_then(|r| r.get(col_idx)) {
                Some(Value::String(s)) => s,
                Some(other) => {
                    owned_storage = other.to_string();
                    &owned_storage
                }
                None => "",
            };
            let parts: Vec<&str> = text.split(delimiter).collect();
            for (j, col) in column_values.iter_mut().enumerate() {
                col.push(
                    parts
                        .get(j)
                        .map(|p| Value::String(p.to_string()))
                        .unwrap_or_else(|| Value::String(String::new())),
                );
            }
        }
        if let Some(t) = table.append_columns_owned(new_names, &column_values) {
            return Ok(t);
        }
    }

    let rows = materialize_table_rows(table);
    let mut column_values: Vec<Vec<Value>> = vec![Vec::with_capacity(rows.len()); n_cols];

    for row in &rows {
        let owned;
        let text: &str = match &row[col_idx] {
            Value::String(s) => s,
            other => {
                owned = other.to_string();
                &owned
            }
        };
        let parts: Vec<&str> = text.split(delimiter).collect();
        for (j, col) in column_values.iter_mut().enumerate() {
            col.push(
                parts
                    .get(j)
                    .map(|p| Value::String(p.to_string()))
                    .unwrap_or_else(|| Value::String(String::new())),
            );
        }
    }

    let mut result = Table::from_data(rows, Some(headers.to_vec()));
    for (name, vals) in new_names.iter().zip(column_values) {
        let arr = Value::Array(Rc::new(RefCell::new(vals)));
        result = table_add_column_impl(&result, name, Some(&arr))?;
    }
    Ok(result)
}

/// Split one column into several new columns via `iter_fn(cell) -> array`.
pub fn table_split_column_impl(
    table: &Table,
    column: &str,
    iter_fn: &Value,
    new_names: &[String],
) -> Result<Table, Value> {
    use super::utils::invoke_value_callable;
    use crate::common::error::LangError;
    use crate::vm::vm::{current_vm_ptr, VmExecutionContext, VM_CALL_CONTEXT};

    if new_names.is_empty() {
        return Err(table_col_error("ValueError: new_columns must not be empty"));
    }

    let headers = table.headers();
    let Some(col_idx) = headers.iter().position(|h| h == column) else {
        return Err(table_col_error(format!(
            "KeyError: column '{}' not found in table",
            column
        )));
    };

    let mut seen_new = std::collections::HashSet::new();
    for name in new_names {
        if headers.iter().any(|h| h == name) {
            return Err(table_col_error(format!(
                "ValueError: column '{}' already exists",
                name
            )));
        }
        if !seen_new.insert(name.as_str()) {
            return Err(table_col_error(format!(
                "ValueError: duplicate new column name '{}'",
                name
            )));
        }
    }

    validate_map_callback(iter_fn, 1)?;

    let rows = materialize_table_rows(table);
    let n_cols = new_names.len();
    let mut column_values: Vec<Vec<Value>> = vec![Vec::with_capacity(rows.len()); n_cols];
    let vm_ptr = current_vm_ptr();

    for row in &rows {
        let cell = row[col_idx].clone();
        if let Some(vm_ptr) = vm_ptr {
            VM_CALL_CONTEXT.with(|ctx| {
                *ctx.borrow_mut() = Some(VmExecutionContext { vm: vm_ptr });
            });
        }
        let parts_val = match invoke_value_callable(iter_fn, &[cell]) {
            Ok(v) => v,
            Err(e) => {
                let message = match e {
                    LangError::LexError { message, .. }
                    | LangError::ParseError { message, .. }
                    | LangError::SemanticError { message, .. }
                    | LangError::RuntimeError { message, .. } => message,
                };
                return Err(table_col_error(message));
            }
        };
        let Value::Array(parts_arr) = parts_val else {
            return Err(table_col_error(
                "TypeError: split_column callback must return an array",
            ));
        };
        let parts_ref = parts_arr.borrow();
        for (j, col) in column_values.iter_mut().enumerate() {
            col.push(
                parts_ref
                    .get(j)
                    .cloned()
                    .unwrap_or_else(|| Value::String(String::new())),
            );
        }
    }

    let mut result = Table::from_data(rows, Some(headers.to_vec()));
    for (name, vals) in new_names.iter().zip(column_values) {
        let arr = Value::Array(Rc::new(RefCell::new(vals)));
        result = table_add_column_impl(&result, name, Some(&arr))?;
    }
    Ok(result)
}

/// Join several columns into one new column with `delimiter`.
pub fn table_join_columns_impl(
    table: &Table,
    source_columns: &[String],
    new_name: &str,
    delimiter: &str,
) -> Result<Table, Value> {
    let headers = table.headers();
    if source_columns.is_empty() {
        return Err(table_col_error(
            "ValueError: source_columns must not be empty",
        ));
    }
    if headers.iter().any(|h| h == new_name) {
        return Err(table_col_error(format!(
            "ValueError: column '{}' already exists",
            new_name
        )));
    }

    let mut col_indices = Vec::with_capacity(source_columns.len());
    for col_name in source_columns {
        let Some(idx) = headers.iter().position(|h| h == col_name) else {
            return Err(table_col_error(format!(
                "KeyError: column '{}' not found in table",
                col_name
            )));
        };
        col_indices.push(idx);
    }

    if !table.is_view() {
        let n_rows = table.len();
        let mut merged: Vec<Value> = Vec::with_capacity(n_rows);
        for row in 0..n_rows {
            let parts: Vec<String> = col_indices
                .iter()
                .filter_map(|&i| {
                    table
                        .get_row(row)
                        .and_then(|r| r.get(i))
                        .map(|v| v.to_string())
                })
                .collect();
            merged.push(Value::String(parts.join(delimiter)));
        }
        if let Some(t) = table.append_column_owned(new_name, &merged) {
            return Ok(t);
        }
    }

    let rows = materialize_table_rows(table);
    let merged: Vec<Value> = rows
        .iter()
        .map(|row| {
            let parts: Vec<String> = col_indices
                .iter()
                .map(|&i| row[i].to_string())
                .collect();
            Value::String(parts.join(delimiter))
        })
        .collect();

    let arr = Value::Array(Rc::new(RefCell::new(merged)));
    table_add_column_impl(table, new_name, Some(&arr))
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

    let Value::Table(table) = &args[0] else {
        return Value::Null;
    };

    let table_ref = table.borrow();
    for col_name in &columns_to_select {
        if !table_ref.headers().iter().any(|h| h == col_name) {
            return Value::Null;
        }
    }

    match table_select_impl(&table_ref, columns_to_select) {
        Ok(t) => Value::Table(Rc::new(RefCell::new(t))),
        Err(v) => v,
    }
}

pub fn native_table_rename(args: &[Value]) -> Value {
    if args.len() < 2 {
        return table_col_error("TypeError: table_rename() expects at least 2 arguments");
    }
    let Value::Table(table) = &args[0] else {
        return table_col_error("TypeError: table_rename() expects a table as the first argument");
    };
    let table_ref = table.borrow();
    let new_name = if args.len() >= 3 {
        match &args[2] {
            Value::String(s) => Some(s.as_str()),
            _ => {
                return table_col_error(
                    "TypeError: table_rename() new column name must be a string",
                );
            }
        }
    } else {
        None
    };
    match table_rename_impl(&table_ref, &args[1], new_name) {
        Ok(t) => Value::Table(Rc::new(RefCell::new(t))),
        Err(v) => v,
    }
}

pub fn native_table_drop_column(args: &[Value]) -> Value {
    if args.len() < 2 {
        return table_col_error("TypeError: table_drop_column() expects at least 2 arguments");
    }
    let Value::Table(table) = &args[0] else {
        return table_col_error(
            "TypeError: table_drop_column() expects a table as the first argument",
        );
    };
    match table_drop_column_impl(&table.borrow(), &args[1]) {
        Ok(t) => Value::Table(Rc::new(RefCell::new(t))),
        Err(v) => v,
    }
}

pub fn native_table_add_column(args: &[Value]) -> Value {
    if args.len() < 2 {
        return table_col_error("TypeError: table_add_column() expects at least 2 arguments");
    }
    let Value::Table(table) = &args[0] else {
        return table_col_error(
            "TypeError: table_add_column() expects a table as the first argument",
        );
    };
    let Value::String(name) = &args[1] else {
        return table_col_error("TypeError: table_add_column() column name must be a string");
    };
    let values_arg = args.get(2);
    match table_add_column_impl(&table.borrow(), name, values_arg) {
        Ok(t) => Value::Table(Rc::new(RefCell::new(t))),
        Err(v) => v,
    }
}

pub fn native_table_map(args: &[Value]) -> Value {
    if args.len() < 3 {
        return table_col_error(
            "TypeError: table_map() expects 3 arguments (table, column, function)",
        );
    }
    let Value::Table(table) = &args[0] else {
        return table_col_error("TypeError: table_map() expects a table as the first argument");
    };
    let Value::String(column) = &args[1] else {
        return table_col_error("TypeError: table_map() column name must be a string");
    };
    match table_map_impl(&table.borrow(), column, &args[2]) {
        Ok(t) => Value::Table(Rc::new(RefCell::new(t))),
        Err(v) => v,
    }
}

/// `table_value_map(table, column, mappings)` / `table.value_map(column, mappings)`
pub fn native_table_value_map(args: &[Value]) -> Value {
    if args.len() != 3 {
        return table_col_error(
            "TypeError: table_value_map() expects 3 arguments (table, column, mappings)",
        );
    }
    let Value::Table(table) = &args[0] else {
        return table_col_error("TypeError: table_value_map() expects a table as the first argument");
    };
    let Value::String(column) = &args[1] else {
        return table_col_error("TypeError: table_value_map() column name must be a string");
    };
    match table_value_map_impl(&table.borrow(), column, &args[2]) {
        Ok(t) => Value::Table(Rc::new(RefCell::new(t))),
        Err(v) => v,
    }
}

pub fn native_table_split_column(args: &[Value]) -> Value {
    if args.len() < 4 {
        return table_col_error(
            "TypeError: table_split_column() expects 4 arguments (table, column, iter_fn, new_columns)",
        );
    }
    let Value::Table(table) = &args[0] else {
        return table_col_error(
            "TypeError: table_split_column() expects a table as the first argument",
        );
    };
    let Value::String(column) = &args[1] else {
        return table_col_error("TypeError: table_split_column() column name must be a string");
    };
    if let Value::String(delim) = &args[2] {
        let new_names = match &args[3] {
            Value::Array(arr) => {
                let arr_ref = arr.borrow();
                if arr_ref.is_empty() {
                    return table_col_error("ValueError: new_columns must not be empty");
                }
                let mut names = Vec::with_capacity(arr_ref.len());
                for val in arr_ref.iter() {
                    match val {
                        Value::String(s) => names.push(s.clone()),
                        _ => {
                            return table_col_error(
                                "TypeError: new_columns must be an array of strings",
                            );
                        }
                    }
                }
                names
            }
            _ => {
                return table_col_error("TypeError: new_columns must be an array of strings");
            }
        };
        return match table_split_column_delim_impl(&table.borrow(), column, delim, &new_names) {
            Ok(t) => Value::Table(Rc::new(RefCell::new(t))),
            Err(v) => v,
        };
    }
    let new_names = match &args[3] {
        Value::Array(arr) => {
            let arr_ref = arr.borrow();
            if arr_ref.is_empty() {
                return table_col_error("ValueError: new_columns must not be empty");
            }
            let mut names = Vec::with_capacity(arr_ref.len());
            for val in arr_ref.iter() {
                match val {
                    Value::String(s) => names.push(s.clone()),
                    _ => {
                        return table_col_error(
                            "TypeError: new_columns must be an array of strings",
                        );
                    }
                }
            }
            names
        }
        _ => {
            return table_col_error("TypeError: new_columns must be an array of strings");
        }
    };
    match table_split_column_impl(&table.borrow(), column, &args[2], &new_names) {
        Ok(t) => Value::Table(Rc::new(RefCell::new(t))),
        Err(v) => v,
    }
}

pub fn native_table_join_columns(args: &[Value]) -> Value {
    if args.len() < 4 {
        return table_col_error(
            "TypeError: table_join_columns() expects 4 arguments (table, source_columns, new_column, delimiter)",
        );
    }
    let Value::Table(table) = &args[0] else {
        return table_col_error(
            "TypeError: table_join_columns() expects a table as the first argument",
        );
    };
    let headers = table.borrow().headers().to_vec();
    let source_columns = match parse_column_name_list(&args[1], &headers) {
        Ok(cols) => cols,
        Err(v) => return v,
    };
    let Value::String(new_name) = &args[2] else {
        return table_col_error("TypeError: table_join_columns() new_column must be a string");
    };
    let Value::String(delimiter) = &args[3] else {
        return table_col_error("TypeError: table_join_columns() delimiter must be a string");
    };
    match table_join_columns_impl(
        &table.borrow(),
        &source_columns,
        new_name,
        delimiter,
    ) {
        Ok(t) => Value::Table(Rc::new(RefCell::new(t))),
        Err(v) => v,
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
                })
                .unwrap_or_default()
            } else if let Some(cv) = table.borrow().column_view(&column_name) {
                (0..cv.len())
                    .filter_map(|i| cv.get_owned(i))
                    .collect()
            } else {
                table
                    .borrow_mut()
                    .get_column(&column_name)
                    .map(|c| c.clone())
                    .unwrap_or_default()
            };
            if sort_column.len() != n_rows {
                return Value::Null;
            }

            let mut indices: Vec<usize> = (0..n_rows).collect();
            indices.sort_by(|&a, &b| {
                let cmp = compare_values(&sort_column[a], &sort_column[b]);
                if ascending {
                    cmp
                } else {
                    cmp.reverse()
                }
            });

            if !is_view {
                let table_ref = table.borrow();
                if let Some(t) = table_ref.gather_rows_owned(&indices) {
                    return Value::Table(Rc::new(RefCell::new(t)));
                }
            }

            let new_rows: Vec<Vec<Value>> = if is_view {
                crate::vm::vm::with_current_stores(|store, heap| {
                    let t = table.borrow();
                    indices
                        .iter()
                        .filter_map(|&idx| crate::vm::table_ops::get_row(&*t, idx, store, heap))
                        .collect()
                })
            } else {
                let table_ref = table.borrow();
                indices
                    .iter()
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
    } else if let Some(cv) = table.borrow().column_view(column_name) {
        (0..cv.len())
            .filter_map(|i| cv.get_owned(i))
            .collect()
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
        .filter(|(_, val)| match operator {
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
        })
        .map(|(i, _)| i)
        .collect();

    if !is_view {
        let table_ref = table.borrow();
        if let Some(t) = table_ref.gather_rows_owned(&matching_indices) {
            return Value::Table(Rc::new(RefCell::new(t)));
        }
    }

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

fn drop_nulls_error(msg: impl Into<String>) -> Value {
    crate::websocket::set_native_error(msg.into());
    Value::Null
}

fn replace_nulls_error(msg: impl Into<String>) -> Value {
    crate::websocket::set_native_error(msg.into());
    Value::Null
}

fn aggregate_error(msg: impl Into<String>) -> Value {
    crate::websocket::set_native_error(msg.into());
    Value::Null
}

#[derive(Clone)]
struct AggregateSpecItem {
    output: String,
    op: String,
    column: Option<String>,
    p: Option<f64>,
    where_fn: Option<Value>,
}

fn validate_agg_op(op: &str) -> bool {
    matches!(
        op,
        "count"
            | "count_distinct"
            | "sum"
            | "avg"
            | "min"
            | "max"
            | "first"
            | "last"
            | "median"
            | "mode"
            | "stddev"
            | "variance"
            | "percentile"
            | "list"
            | "any"
    )
}

fn parse_agg_spec_entries(
    entries: Vec<(String, Value)>,
    empty_error: &str,
) -> Result<Vec<AggregateSpecItem>, Value> {
    if entries.is_empty() {
        return Err(aggregate_error(empty_error));
    }

    let mut out = Vec::with_capacity(entries.len());
    for (output, item) in entries {
        match item {
            Value::String(op) => {
                if !validate_agg_op(&op) {
                    return Err(aggregate_error(format!(
                        "TypeError: unsupported aggregate op '{}'",
                        op
                    )));
                }
                if op != "count" {
                    return Err(aggregate_error(format!(
                        "TypeError: string aggregate spec supports only 'count' (got '{}')",
                        op
                    )));
                }
                out.push(AggregateSpecItem {
                    output,
                    op,
                    column: None,
                    p: None,
                    where_fn: None,
                });
            }
            Value::Object(obj) => {
                let obj_ref = obj.borrow();
                let op = match obj_ref.str_key_get("op") {
                    Some(Value::String(s)) => s.clone(),
                    _ => {
                        return Err(aggregate_error(
                            "TypeError: aggregate spec object must contain string field 'op'",
                        ));
                    }
                };
                if !validate_agg_op(&op) {
                    return Err(aggregate_error(format!(
                        "TypeError: unsupported aggregate op '{}'",
                        op
                    )));
                }

                let column = match obj_ref.str_key_get("column") {
                    Some(Value::String(s)) => Some(s.clone()),
                    Some(_) => {
                        return Err(aggregate_error(
                            "TypeError: aggregate spec field 'column' must be a string",
                        ));
                    }
                    None => None,
                };

                if op != "count" && column.is_none() {
                    return Err(aggregate_error(format!(
                        "TypeError: aggregate op '{}' requires string field 'column'",
                        op
                    )));
                }

                let p = match obj_ref.str_key_get("p") {
                    Some(v) => {
                        if let Some(n) = v.as_ieee_f64() {
                            Some(n)
                        } else {
                            return Err(aggregate_error(
                                "TypeError: aggregate spec field 'p' must be a number",
                            ));
                        }
                    }
                    None => None,
                };

                let where_fn = obj_ref.str_key_get("where").cloned();
                if op == "percentile" && p.is_none() {
                    return Err(aggregate_error(
                        "TypeError: aggregate op 'percentile' requires numeric field 'p'",
                    ));
                }
                if op == "any" {
                    let Some(pred) = where_fn.clone() else {
                        return Err(aggregate_error(
                            "TypeError: aggregate op 'any' requires field 'where'",
                        ));
                    };
                    if !matches!(
                        pred,
                        Value::NativeFunction(_)
                            | Value::Function(_)
                            | Value::ModuleFunction { .. }
                    ) {
                        return Err(aggregate_error(
                            "TypeError: aggregate spec field 'where' must be callable",
                        ));
                    }
                }

                out.push(AggregateSpecItem {
                    output,
                    op,
                    column,
                    p,
                    where_fn,
                });
            }
            _ => {
                return Err(aggregate_error(
                    "TypeError: each aggregate spec value must be a string or object",
                ));
            }
        }
    }

    Ok(out)
}

fn parse_agg_spec(spec: &Value) -> Result<Vec<AggregateSpecItem>, Value> {
    let Value::Object(spec_obj) = spec else {
        return Err(aggregate_error("TypeError: table.aggregate() spec must be an object"));
    };
    parse_agg_spec_entries(
        spec_obj.borrow().str_key_entries_cloned(),
        "TypeError: table.aggregate() spec must not be empty",
    )
}

fn parse_group_columns(group_arg: &Value) -> Result<Vec<String>, Value> {
    match group_arg {
        Value::String(s) => Ok(vec![s.clone()]),
        Value::Array(arr) => {
            let arr_ref = arr.borrow();
            if arr_ref.is_empty() {
                return Err(aggregate_error(
                    "TypeError: aggregate_group field 'group' must be a non-empty string or array of strings",
                ));
            }
            let mut names = Vec::with_capacity(arr_ref.len());
            for item in arr_ref.iter() {
                match item {
                    Value::String(s) => names.push(s.clone()),
                    _ => {
                        return Err(aggregate_error(
                            "TypeError: aggregate_group field 'group' must be a string or array of strings",
                        ));
                    }
                }
            }
            Ok(names)
        }
        _ => Err(aggregate_error(
            "TypeError: aggregate_group field 'group' must be a string or array of strings",
        )),
    }
}

fn parse_aggregate_group_spec(
    spec: &Value,
    headers: &[String],
) -> Result<(Vec<String>, Vec<AggregateSpecItem>), Value> {
    let Value::Object(spec_obj) = spec else {
        return Err(aggregate_error(
            "TypeError: table.aggregate_group() spec must be an object",
        ));
    };
    let entries = spec_obj.borrow().str_key_entries_cloned();
    if entries.is_empty() {
        return Err(aggregate_error(
            "TypeError: table.aggregate_group() spec must not be empty",
        ));
    }

    let Some((_, group_arg)) = entries.iter().find(|(k, _)| k == "group") else {
        return Err(aggregate_error(
            "TypeError: table.aggregate_group() spec must contain field 'group'",
        ));
    };
    let group_columns = parse_group_columns(group_arg)?;
    for name in &group_columns {
        if !headers.iter().any(|h| h == name) {
            return Err(aggregate_error(format!(
                "KeyError: column '{}' not found in table",
                name
            )));
        }
    }

    let agg_entries: Vec<(String, Value)> = entries
        .into_iter()
        .filter(|(k, _)| k != "group")
        .collect();
    let agg_items = parse_agg_spec_entries(
        agg_entries,
        "TypeError: table.aggregate_group() spec must contain at least one aggregation",
    )?;
    Ok((group_columns, agg_items))
}

fn numbers_from_values(values: &[Value]) -> Vec<f64> {
    values.iter().filter_map(|v| v.as_ieee_f64()).collect()
}

fn aggregate_median(values: &[Value]) -> Value {
    let mut nums = numbers_from_values(values);
    if nums.is_empty() {
        return Value::Null;
    }
    nums.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = nums.len() / 2;
    if nums.len() % 2 == 1 {
        Value::Number(nums[mid])
    } else {
        Value::Number((nums[mid - 1] + nums[mid]) / 2.0)
    }
}

fn aggregate_mode(values: &[Value]) -> Value {
    if values.is_empty() {
        return Value::Null;
    }
    let mut counts: std::collections::HashMap<String, (usize, Value)> = std::collections::HashMap::new();
    for v in values {
        let key = v.to_string();
        counts
            .entry(key)
            .and_modify(|(c, _)| *c += 1)
            .or_insert((1, v.clone()));
    }
    counts
        .into_iter()
        .max_by_key(|(_, (count, _))| *count)
        .map(|(_, (_, v))| v)
        .unwrap_or(Value::Null)
}

fn aggregate_variance(values: &[Value]) -> Value {
    let nums = numbers_from_values(values);
    if nums.is_empty() {
        return Value::Null;
    }
    let mean = nums.iter().sum::<f64>() / nums.len() as f64;
    let var = nums.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / nums.len() as f64;
    Value::Number(var)
}

fn aggregate_stddev(values: &[Value]) -> Value {
    match aggregate_variance(values) {
        Value::Number(v) => Value::Number(v.sqrt()),
        _ => Value::Null,
    }
}

fn aggregate_percentile(values: &[Value], p: f64) -> Result<Value, Value> {
    if !(0.0..=1.0).contains(&p) {
        return Err(aggregate_error("TypeError: percentile 'p' must be between 0 and 1"));
    }
    let mut nums = numbers_from_values(values);
    if nums.is_empty() {
        return Ok(Value::Null);
    }
    nums.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let rank = p * (nums.len().saturating_sub(1) as f64);
    let lo = rank.floor() as usize;
    let hi = rank.ceil() as usize;
    if lo == hi {
        return Ok(Value::Number(nums[lo]));
    }
    let w = rank - lo as f64;
    Ok(Value::Number(nums[lo] + (nums[hi] - nums[lo]) * w))
}

fn aggregate_column_values(
    headers: &[String],
    rows: &[Vec<Value>],
    column: &str,
) -> Result<Vec<Value>, Value> {
    let Some(col_idx) = headers.iter().position(|h| h == column) else {
        return Err(aggregate_error(format!(
            "KeyError: column '{}' not found in table",
            column
        )));
    };
    Ok(rows
        .iter()
        .map(|r| r.get(col_idx).cloned().unwrap_or(Value::Null))
        .collect())
}

fn aggregate_column_values_from_table(
    table: &Table,
    column: &str,
    row_indices: Option<&[usize]>,
) -> Result<Vec<Value>, Value> {
    let Some(col_idx) = table.headers().iter().position(|h| h == column) else {
        return Err(aggregate_error(format!(
            "KeyError: column '{}' not found in table",
            column
        )));
    };
    if let Some(idxs) = row_indices {
        Ok(idxs
            .iter()
            .filter_map(|&ri| {
                table
                    .get_row(ri)
                    .and_then(|r| r.get(col_idx).cloned())
            })
            .collect())
    } else if let Some(cv) = table.column_view(column) {
        Ok((0..cv.len()).filter_map(|i| cv.get_owned(i)).collect())
    } else {
        Ok(Vec::new())
    }
}

fn apply_aggregate_op_on_table(
    item: &AggregateSpecItem,
    table: &Table,
    row_indices: Option<&[usize]>,
    vm_ptr: Option<*mut crate::vm::vm::Vm>,
) -> Result<Value, Value> {
    use super::utils::invoke_value_callable;
    use crate::common::error::LangError;
    use crate::vm::vm::{VmExecutionContext, VM_CALL_CONTEXT};

    let row_count = row_indices.map(|i| i.len()).unwrap_or_else(|| table.len());
    let col_values = if let Some(col) = item.column.as_deref() {
        aggregate_column_values_from_table(table, col, row_indices)?
    } else {
        Vec::new()
    };

    match item.op.as_str() {
        "count" => Ok(Value::Number(row_count as f64)),
        "count_distinct" => {
            let mut set = std::collections::HashSet::new();
            for v in &col_values {
                set.insert(v.to_string());
            }
            Ok(Value::Number(set.len() as f64))
        }
        "sum" => Ok(Value::Number(
            numbers_from_values(&col_values).iter().sum::<f64>(),
        )),
        "avg" => {
            let nums = numbers_from_values(&col_values);
            if nums.is_empty() {
                Ok(Value::Number(0.0))
            } else {
                Ok(Value::Number(nums.iter().sum::<f64>() / nums.len() as f64))
            }
        }
        "min" => Ok(col_values
            .iter()
            .cloned()
            .min_by(compare_values)
            .unwrap_or(Value::Null)),
        "max" => Ok(col_values
            .iter()
            .cloned()
            .max_by(compare_values)
            .unwrap_or(Value::Null)),
        "first" => Ok(col_values.first().cloned().unwrap_or(Value::Null)),
        "last" => Ok(col_values.last().cloned().unwrap_or(Value::Null)),
        "median" => Ok(aggregate_median(&col_values)),
        "mode" => Ok(aggregate_mode(&col_values)),
        "stddev" => Ok(aggregate_stddev(&col_values)),
        "variance" => Ok(aggregate_variance(&col_values)),
        "percentile" => aggregate_percentile(&col_values, item.p.unwrap_or(0.5)),
        "list" => Ok(Value::Array(Rc::new(RefCell::new(col_values)))),
        "any" => {
            let where_fn = item.where_fn.as_ref().ok_or_else(|| {
                aggregate_error("TypeError: aggregate op 'any' requires field 'where'")
            })?;
            let mut ok = false;
            for v in &col_values {
                if let Some(vm_ptr) = vm_ptr {
                    VM_CALL_CONTEXT.with(|ctx| {
                        *ctx.borrow_mut() = Some(VmExecutionContext { vm: vm_ptr });
                    });
                }
                let pred_res = invoke_value_callable(where_fn, std::slice::from_ref(v)).map_err(|e| {
                    let message = match e {
                        LangError::LexError { message, .. }
                        | LangError::ParseError { message, .. }
                        | LangError::SemanticError { message, .. }
                        | LangError::RuntimeError { message, .. } => message,
                    };
                    aggregate_error(format!(
                        "TypeError: aggregate any(where) callback failed: {}",
                        message
                    ))
                })?;
                if pred_res.is_truthy() {
                    ok = true;
                    break;
                }
            }
            Ok(Value::Bool(ok))
        }
        _ => Err(aggregate_error(format!(
            "TypeError: unsupported aggregate op '{}'",
            item.op
        ))),
    }
}

fn apply_aggregate_op(
    item: &AggregateSpecItem,
    headers: &[String],
    rows: &[Vec<Value>],
    vm_ptr: Option<*mut crate::vm::vm::Vm>,
) -> Result<Value, Value> {
    use super::utils::invoke_value_callable;
    use crate::common::error::LangError;
    use crate::vm::vm::{VmExecutionContext, VM_CALL_CONTEXT};

    let col_values = if let Some(col) = item.column.as_deref() {
        aggregate_column_values(headers, rows, col)?
    } else {
        Vec::new()
    };

    match item.op.as_str() {
        "count" => Ok(Value::Number(rows.len() as f64)),
        "count_distinct" => {
            let mut set = std::collections::HashSet::new();
            for v in &col_values {
                set.insert(v.to_string());
            }
            Ok(Value::Number(set.len() as f64))
        }
        "sum" => Ok(Value::Number(
            numbers_from_values(&col_values).iter().sum::<f64>(),
        )),
        "avg" => {
            let nums = numbers_from_values(&col_values);
            if nums.is_empty() {
                Ok(Value::Number(0.0))
            } else {
                Ok(Value::Number(nums.iter().sum::<f64>() / nums.len() as f64))
            }
        }
        "min" => Ok(col_values
            .iter()
            .cloned()
            .min_by(compare_values)
            .unwrap_or(Value::Null)),
        "max" => Ok(col_values
            .iter()
            .cloned()
            .max_by(compare_values)
            .unwrap_or(Value::Null)),
        "first" => Ok(col_values.first().cloned().unwrap_or(Value::Null)),
        "last" => Ok(col_values.last().cloned().unwrap_or(Value::Null)),
        "median" => Ok(aggregate_median(&col_values)),
        "mode" => Ok(aggregate_mode(&col_values)),
        "stddev" => Ok(aggregate_stddev(&col_values)),
        "variance" => Ok(aggregate_variance(&col_values)),
        "percentile" => aggregate_percentile(&col_values, item.p.unwrap_or(0.5)),
        "list" => Ok(Value::Array(Rc::new(RefCell::new(col_values)))),
        "any" => {
            let where_fn = item.where_fn.as_ref().ok_or_else(|| {
                aggregate_error("TypeError: aggregate op 'any' requires field 'where'")
            })?;
            let mut ok = false;
            for v in &col_values {
                if let Some(vm_ptr) = vm_ptr {
                    VM_CALL_CONTEXT.with(|ctx| {
                        *ctx.borrow_mut() = Some(VmExecutionContext { vm: vm_ptr });
                    });
                }
                let pred_res = invoke_value_callable(where_fn, std::slice::from_ref(v)).map_err(|e| {
                    let message = match e {
                        LangError::LexError { message, .. }
                        | LangError::ParseError { message, .. }
                        | LangError::SemanticError { message, .. }
                        | LangError::RuntimeError { message, .. } => message,
                    };
                    aggregate_error(format!(
                        "TypeError: aggregate any(where) callback failed: {}",
                        message
                    ))
                })?;
                if pred_res.is_truthy() {
                    ok = true;
                    break;
                }
            }
            Ok(Value::Bool(ok))
        }
        _ => Err(aggregate_error(format!(
            "TypeError: unsupported aggregate op '{}'",
            item.op
        ))),
    }
}

pub fn table_aggregate_impl(table: &Table, spec: &Value) -> Result<Table, Value> {
    use crate::vm::vm::current_vm_ptr;

    let parsed = parse_agg_spec(spec)?;
    let vm_ptr = current_vm_ptr();

    let mut out_headers: Vec<String> = Vec::with_capacity(parsed.len());
    let mut out_row: Vec<Value> = Vec::with_capacity(parsed.len());

    if !table.is_view() {
        for item in parsed {
            out_headers.push(item.output.clone());
            out_row.push(apply_aggregate_op_on_table(
                &item,
                table,
                None,
                vm_ptr,
            )?);
        }
        return Ok(Table::from_data(vec![out_row], Some(out_headers)));
    }

    let headers = table.headers().to_vec();
    let rows = materialize_table_rows(table);

    for item in parsed {
        out_headers.push(item.output.clone());
        out_row.push(apply_aggregate_op(&item, &headers, &rows, vm_ptr)?);
    }

    Ok(Table::from_data(vec![out_row], Some(out_headers)))
}

pub fn table_aggregate_group_impl(table: &Table, spec: &Value) -> Result<Table, Value> {
    use crate::vm::vm::current_vm_ptr;
    use std::collections::HashMap;

    let headers = table.headers().to_vec();
    let (group_columns, agg_items) = parse_aggregate_group_spec(spec, &headers)?;
    let vm_ptr = current_vm_ptr();

    let group_indices: Vec<usize> = group_columns
        .iter()
        .map(|name| {
            headers
                .iter()
                .position(|h| h == name)
                .expect("group column validated")
        })
        .collect();

    let mut buckets: Vec<(Vec<Value>, Vec<usize>)> = Vec::new();
    let mut bucket_index: HashMap<String, usize> = HashMap::new();

    if !table.is_view() {
        for row_idx in 0..table.len() {
            let key_values: Vec<Value> = group_indices
                .iter()
                .map(|&idx| {
                    table
                        .get_row(row_idx)
                        .and_then(|r| r.get(idx).cloned())
                        .unwrap_or(Value::Null)
                })
                .collect();
            let key = key_values
                .iter()
                .map(|v| v.to_string())
                .collect::<Vec<_>>()
                .join("\x1f");
            if let Some(&bucket_idx) = bucket_index.get(&key) {
                buckets[bucket_idx].1.push(row_idx);
            } else {
                let bucket_idx = buckets.len();
                bucket_index.insert(key, bucket_idx);
                buckets.push((key_values, vec![row_idx]));
            }
        }

        let mut out_headers = group_columns.clone();
        out_headers.extend(agg_items.iter().map(|item| item.output.clone()));
        let mut out_rows: Vec<Vec<Value>> = Vec::with_capacity(buckets.len());

        for (key_values, row_indices) in buckets {
            let mut out_row = key_values;
            for item in &agg_items {
                out_row.push(apply_aggregate_op_on_table(
                    item,
                    table,
                    Some(&row_indices),
                    vm_ptr,
                )?);
            }
            out_rows.push(out_row);
        }

        return Ok(Table::from_data(out_rows, Some(out_headers)));
    }

    let rows = materialize_table_rows(table);

    for (row_idx, row) in rows.iter().enumerate() {
        let key_values: Vec<Value> = group_indices
            .iter()
            .map(|&idx| row.get(idx).cloned().unwrap_or(Value::Null))
            .collect();
        let key = key_values
            .iter()
            .map(|v| v.to_string())
            .collect::<Vec<_>>()
            .join("\x1f");
        if let Some(&bucket_idx) = bucket_index.get(&key) {
            buckets[bucket_idx].1.push(row_idx);
        } else {
            let bucket_idx = buckets.len();
            bucket_index.insert(key, bucket_idx);
            buckets.push((key_values, vec![row_idx]));
        }
    }

    let mut out_headers = group_columns.clone();
    out_headers.extend(agg_items.iter().map(|item| item.output.clone()));
    let mut out_rows: Vec<Vec<Value>> = Vec::with_capacity(buckets.len());

    for (key_values, row_indices) in buckets {
        let group_rows: Vec<Vec<Value>> = row_indices
            .iter()
            .map(|&idx| rows[idx].clone())
            .collect();
        let mut out_row = key_values;
        for item in &agg_items {
            out_row.push(apply_aggregate_op(item, &headers, &group_rows, vm_ptr)?);
        }
        out_rows.push(out_row);
    }

    Ok(Table::from_data(out_rows, Some(out_headers)))
}

/// `table_aggregate(table, spec)` / `table.aggregate(spec)`
pub fn native_table_aggregate(args: &[Value]) -> Value {
    if args.len() != 2 {
        return aggregate_error("TypeError: table_aggregate() expects 2 arguments (table, spec)");
    }
    let Value::Table(table) = &args[0] else {
        return aggregate_error("TypeError: table_aggregate() expects a table as the first argument");
    };

    match table_aggregate_impl(&table.borrow(), &args[1]) {
        Ok(t) => Value::Table(Rc::new(RefCell::new(t))),
        Err(v) => v,
    }
}

/// `table_aggregate_group(table, spec)` / `table.aggregate_group(spec)`
pub fn native_table_aggregate_group(args: &[Value]) -> Value {
    if args.len() != 2 {
        return aggregate_error(
            "TypeError: table_aggregate_group() expects 2 arguments (table, spec)",
        );
    }
    let Value::Table(table) = &args[0] else {
        return aggregate_error(
            "TypeError: table_aggregate_group() expects a table as the first argument",
        );
    };

    match table_aggregate_group_impl(&table.borrow(), &args[1]) {
        Ok(t) => Value::Table(Rc::new(RefCell::new(t))),
        Err(v) => v,
    }
}

/// Resolve optional column argument to header indices. `None` = check all columns.
fn parse_drop_nulls_columns(
    arg: Option<&Value>,
    headers: &[String],
) -> Result<Option<Vec<usize>>, Value> {
    let Some(arg) = arg else {
        return Ok(None);
    };
    let names: Vec<String> = match arg {
        Value::String(s) => vec![s.clone()],
        Value::Array(arr) => arr
            .borrow()
            .iter()
            .filter_map(|v| match v {
                Value::String(s) => Some(s.clone()),
                _ => None,
            })
            .collect(),
        _ => {
            return Err(drop_nulls_error(
                "TypeError: column must be a string or array of strings",
            ));
        }
    };
    if names.is_empty() {
        return Err(drop_nulls_error(
            "TypeError: column must be a non-empty string or array of strings",
        ));
    }
    let mut indices = Vec::with_capacity(names.len());
    for name in names {
        let Some(idx) = headers.iter().position(|h| h == &name) else {
            return Err(drop_nulls_error(format!(
                "KeyError: column '{}' not found in table",
                name
            )));
        };
        indices.push(idx);
    }
    Ok(Some(indices))
}

/// Resolve optional column argument to header indices. `None` = all columns.
fn parse_replace_nulls_columns(
    arg: Option<&Value>,
    headers: &[String],
) -> Result<Option<Vec<usize>>, Value> {
    let Some(arg) = arg else {
        return Ok(None);
    };
    let names: Vec<String> = match arg {
        Value::String(s) => vec![s.clone()],
        Value::Array(arr) => arr
            .borrow()
            .iter()
            .filter_map(|v| match v {
                Value::String(s) => Some(s.clone()),
                _ => None,
            })
            .collect(),
        _ => {
            return Err(replace_nulls_error(
                "TypeError: column must be a string or array of strings",
            ));
        }
    };
    if names.is_empty() {
        return Err(replace_nulls_error(
            "TypeError: column must be a non-empty string or array of strings",
        ));
    }
    let mut indices = Vec::with_capacity(names.len());
    for name in names {
        let Some(idx) = headers.iter().position(|h| h == &name) else {
            return Err(replace_nulls_error(format!(
                "KeyError: column '{}' not found in table",
                name
            )));
        };
        indices.push(idx);
    }
    Ok(Some(indices))
}

fn validate_replace_nulls_callback(func: &Value) -> Result<(), Value> {
    match func {
        Value::NativeFunction(_) => Ok(()),
        Value::Function(fn_idx) => {
            let Some(vm_ptr) = crate::vm::vm::current_vm_ptr() else {
                return Err(replace_nulls_error("replace_nulls: VM context not available"));
            };
            unsafe {
                let vm = &*vm_ptr;
                let arity = vm
                    .get_functions()
                    .get(*fn_idx)
                    .map(|fun| fun.arity)
                    .unwrap_or(0);
                if arity != 1 {
                    return Err(replace_nulls_error(format!(
                        "TypeError: replace_nulls callback must have arity 1, got {}",
                        arity
                    )));
                }
            }
            Ok(())
        }
        Value::ModuleFunction { .. } => Ok(()),
        _ => Err(replace_nulls_error(
            "TypeError: replacement must be a value or callback function",
        )),
    }
}

fn row_has_null_in_columns(row: &[Value], col_indices: Option<&[usize]>) -> bool {
    match col_indices {
        None => row.iter().any(|v| matches!(v, Value::Null)),
        Some(idxs) => idxs
            .iter()
            .any(|&i| row.get(i).is_some_and(|v| matches!(v, Value::Null))),
    }
}

/// Drop rows with `null` in `columns` (or any column when `columns` is `None`).
pub fn table_drop_nulls_impl(
    table: &Rc<RefCell<Table>>,
    columns: Option<Vec<usize>>,
) -> Value {
    let headers = table.borrow().headers().clone();
    let n_rows = table.borrow().len();
    let is_view = table.borrow().is_view();

    let matching_indices: Vec<usize> = if is_view {
        crate::vm::vm::with_current_stores(|store, heap| {
            let t = table.borrow();
            (0..n_rows)
                .filter(|&i| {
                    crate::vm::table_ops::get_row(&*t, i, store, heap)
                        .is_some_and(|row| !row_has_null_in_columns(&row, columns.as_deref()))
                })
                .collect()
        })
    } else {
        let table_ref = table.borrow();
        (0..n_rows)
            .filter(|&i| {
                table_ref
                    .get_row(i)
                    .is_some_and(|row| !row_has_null_in_columns(row, columns.as_deref()))
            })
            .collect()
    };

    if !is_view {
        let table_ref = table.borrow();
        if let Some(t) = table_ref.gather_rows_owned(&matching_indices) {
            return Value::Table(Rc::new(RefCell::new(t)));
        }
    }

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

/// `table_drop_nulls(table [, column])` / `table.drop_nulls([column])`
pub fn native_table_drop_nulls(args: &[Value]) -> Value {
    if args.is_empty() {
        return drop_nulls_error("TypeError: table_drop_nulls() expects at least 1 argument");
    }

    let table = match &args[0] {
        Value::Table(t) => t,
        _ => {
            return drop_nulls_error(
                "TypeError: table_drop_nulls() expects a table as the first argument",
            );
        }
    };

    let headers = table.borrow().headers().clone();
    let columns = match parse_drop_nulls_columns(args.get(1), &headers) {
        Ok(c) => c,
        Err(v) => return v,
    };

    table_drop_nulls_impl(table, columns)
}

/// Replace `null` values in selected columns (or all columns when `columns` is `None`).
/// `replacement` can be a scalar value or callback `fn(row)`.
pub fn table_replace_nulls_impl(
    table: &Rc<RefCell<Table>>,
    columns: Option<Vec<usize>>,
    replacement: &Value,
) -> Value {
    use super::utils::invoke_value_callable;
    use crate::common::error::LangError;
    use crate::vm::vm::{current_vm_ptr, VmExecutionContext, VM_CALL_CONTEXT};
    use std::collections::HashMap;

    let headers = table.borrow().headers().clone();
    let rows = {
        let table_ref = table.borrow();
        materialize_table_rows(&table_ref)
    };
    let target_columns: Vec<usize> = columns.unwrap_or_else(|| (0..headers.len()).collect());
    let callback_mode = matches!(
        replacement,
        Value::NativeFunction(_) | Value::Function(_) | Value::ModuleFunction { .. }
    );
    if callback_mode {
        if let Err(v) = validate_replace_nulls_callback(replacement) {
            return v;
        }
    }

    let vm_ptr = current_vm_ptr();
    let mut out_rows = Vec::with_capacity(rows.len());
    for mut row in rows {
        let mut row_object = HashMap::with_capacity(headers.len());
        for (idx, header) in headers.iter().enumerate() {
            row_object.insert(
                header.clone(),
                row.get(idx).cloned().unwrap_or_else(|| Value::Null),
            );
        }
        let row_value = Value::legacy_object(row_object);
        for &col_idx in &target_columns {
            if row.get(col_idx).is_some_and(|v| matches!(v, Value::Null)) {
                let new_value = if callback_mode {
                    if let Some(vm_ptr) = vm_ptr {
                        VM_CALL_CONTEXT.with(|ctx| {
                            *ctx.borrow_mut() = Some(VmExecutionContext { vm: vm_ptr });
                        });
                    }
                    match invoke_value_callable(replacement, std::slice::from_ref(&row_value)) {
                        Ok(v) => v,
                        Err(e) => {
                            let message = match e {
                                LangError::LexError { message, .. }
                                | LangError::ParseError { message, .. }
                                | LangError::SemanticError { message, .. }
                                | LangError::RuntimeError { message, .. } => message,
                            };
                            return replace_nulls_error(message);
                        }
                    }
                } else {
                    replacement.clone()
                };
                if let Some(cell) = row.get_mut(col_idx) {
                    *cell = new_value;
                }
            }
        }
        out_rows.push(row);
    }

    let new_table = Table::from_data(out_rows, Some(headers));
    Value::Table(Rc::new(RefCell::new(new_table)))
}

/// `table_replace_nulls(table, replacement)` /
/// `table_replace_nulls(table, column, replacement)` /
/// `table.replace_nulls(...)`
pub fn native_table_replace_nulls(args: &[Value]) -> Value {
    if args.len() < 2 || args.len() > 3 {
        return replace_nulls_error(
            "TypeError: table_replace_nulls() expects 2 or 3 arguments",
        );
    }

    let table = match &args[0] {
        Value::Table(t) => t,
        _ => {
            return replace_nulls_error(
                "TypeError: table_replace_nulls() expects a table as the first argument",
            );
        }
    };

    let headers = table.borrow().headers().clone();
    let (columns, replacement) = if args.len() == 2 {
        (None, &args[1])
    } else {
        let columns = match parse_replace_nulls_columns(args.get(1), &headers) {
            Ok(c) => c,
            Err(v) => return v,
        };
        (columns, &args[2])
    };

    table_replace_nulls_impl(table, columns, replacement)
}

/// `table_row_number(table [, column_name [, start_from]])` /
/// `table.row_number([column_name [, start_from]])`
pub fn native_table_row_number(args: &[Value]) -> Value {
    if args.is_empty() || args.len() > 3 {
        return table_col_error(
            "TypeError: table_row_number() expects 1 to 3 arguments",
        );
    }
    let Value::Table(table) = &args[0] else {
        return table_col_error("TypeError: table_row_number() expects a table as the first argument");
    };

    let column_name = if args.len() > 1 {
        match &args[1] {
            Value::String(s) => s.as_str(),
            _ => {
                return table_col_error(
                    "TypeError: table_row_number() column name must be a string",
                );
            }
        }
    } else {
        "RowNumber"
    };
    let start_from = if args.len() > 2 {
        match &args[2] {
            Value::Number(n) => *n,
            _ => {
                return table_col_error(
                    "TypeError: table_row_number() start_from must be a number",
                );
            }
        }
    } else {
        1.0
    };

    let n_rows = table.borrow().len();
    let numbers: Vec<Value> = (0..n_rows)
        .map(|i| Value::Number(start_from + i as f64))
        .collect();
    let values = Value::Array(Rc::new(RefCell::new(numbers)));
    match table_add_column_impl(&table.borrow(), column_name, Some(&values)) {
        Ok(t) => Value::Table(Rc::new(RefCell::new(t))),
        Err(v) => v,
    }
}

/// `table_distinct(table [, columns])` / `table.distinct([columns])`
pub fn native_table_distinct(args: &[Value]) -> Value {
    use std::collections::HashSet;

    if args.is_empty() || args.len() > 2 {
        return table_col_error("TypeError: table_distinct() expects 1 or 2 arguments");
    }
    let Value::Table(table) = &args[0] else {
        return table_col_error("TypeError: table_distinct() expects a table as the first argument");
    };

    let table_ref = table.borrow();
    let headers = table_ref.headers().to_vec();
    let selected_indices: Option<Vec<usize>> = if let Some(columns_arg) = args.get(1) {
        let names = match parse_column_name_list(columns_arg, &headers) {
            Ok(n) => n,
            Err(v) => return v,
        };
        let mut indices = Vec::with_capacity(names.len());
        for name in names {
            if let Some(idx) = headers.iter().position(|h| h == &name) {
                indices.push(idx);
            }
        }
        Some(indices)
    } else {
        None
    };

    if !table_ref.is_view() {
        let n_rows = table_ref.len();
        let mut seen: HashSet<Vec<String>> = HashSet::new();
        let mut keep_indices = Vec::with_capacity(n_rows);
        for i in 0..n_rows {
            if let Some(row) = table_ref.get_row(i) {
                let key: Vec<String> = match &selected_indices {
                    Some(indices) => indices
                        .iter()
                        .map(|&ci| row.get(ci).cloned().unwrap_or(Value::Null).to_string())
                        .collect(),
                    None => row.iter().map(|v| v.to_string()).collect(),
                };
                if seen.insert(key) {
                    keep_indices.push(i);
                }
            }
        }
        if let Some(t) = table_ref.gather_rows_owned(&keep_indices) {
            return Value::Table(Rc::new(RefCell::new(t)));
        }
    }

    let rows = materialize_table_rows(&table_ref);
    let mut seen: HashSet<Vec<String>> = HashSet::new();
    let mut out_rows: Vec<Vec<Value>> = Vec::with_capacity(rows.len());
    for row in rows {
        let key: Vec<String> = match &selected_indices {
            Some(indices) => indices
                .iter()
                .map(|&i| row.get(i).cloned().unwrap_or(Value::Null).to_string())
                .collect(),
            None => row.iter().map(|v| v.to_string()).collect(),
        };
        if seen.insert(key) {
            out_rows.push(row);
        }
    }
    Value::Table(Rc::new(RefCell::new(Table::from_data(out_rows, Some(headers)))))
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
                            if let Some(v) = crate::vm::table_ops::get_cell_value(
                                &*t, row_idx, header, store, heap,
                            ) {
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
                (
                    col_widths.into_iter().map(|w| w.max(3)).collect(),
                    rr.len().min(20),
                )
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
                    table
                        .borrow()
                        .get_row(row_idx)
                        .map(|r| r.to_vec())
                        .unwrap_or_default()
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
    use std::collections::{HashMap, HashSet};

    if args.is_empty() {
        return Value::Null;
    }

    let table_handles: Vec<Rc<RefCell<Table>>> = match &args[0] {
        Value::Array(arr) => {
            let arr_ref = arr.borrow();
            if arr_ref.is_empty() {
                return Value::Null;
            }
            let mut out = Vec::with_capacity(arr_ref.len());
            for val in arr_ref.iter() {
                match val {
                    Value::Table(t) => out.push(Rc::clone(t)),
                    _ => return Value::Null,
                }
            }
            out
        }
        _ => return Value::Null,
    };

    let mode = if args.len() > 1 {
        match &args[1] {
            Value::String(s) => s.as_str(),
            _ => "outer",
        }
    } else {
        "outer"
    };

    if table_handles.len() == 1 {
        return Value::Table(Rc::clone(&table_handles[0]));
    }

    let same_schema = {
        let first_headers = table_handles[0].borrow();
        let headers = first_headers.headers();
        table_handles[1..]
            .iter()
            .all(|t| t.borrow().headers() == headers)
    };

    if same_schema {
        let headers = table_handles[0].borrow().headers().clone();
        let total_rows: usize = table_handles.iter().map(|t| t.borrow().len()).sum();
        let mut dest = Table::with_capacity_owned(headers, total_rows);
        for t in &table_handles {
            if merge_append_table(&mut dest, &t.borrow()).is_err() {
                return Value::Null;
            }
        }
        return Value::Table(Rc::new(RefCell::new(dest)));
    }

    let mut all_columns_set = HashSet::new();
    let mut column_order = Vec::new();
    for table_rc in &table_handles {
        let table_ref = table_rc.borrow();
        for header in table_ref.headers() {
            if all_columns_set.insert(header.clone()) {
                column_order.push(header.clone());
            }
        }
    }

    let result_columns = if mode == "inner" {
        column_order
            .into_iter()
            .filter(|col| {
                table_handles
                    .iter()
                    .all(|t| t.borrow().headers().iter().any(|h| h == col))
            })
            .collect::<Vec<_>>()
    } else {
        column_order
    };

    let total_rows: usize = table_handles.iter().map(|t| t.borrow().len()).sum();
    let n_cols = result_columns.len();
    let mut flat = Vec::with_capacity(total_rows.saturating_mul(n_cols));

    for table_rc in &table_handles {
        let src = table_rc.borrow();
        let src_index: HashMap<&str, usize> = src
            .headers()
            .iter()
            .enumerate()
            .map(|(i, h)| (h.as_str(), i))
            .collect();
        let col_map: Vec<Option<usize>> = result_columns
            .iter()
            .map(|c| src_index.get(c.as_str()).copied())
            .collect();

        if let Some(src_flat) = src.owned_flat() {
            let src_cols = src.owned_num_cols().unwrap_or(0);
            let n = src.len();
            for row in 0..n {
                for src_c in &col_map {
                    match src_c {
                        Some(si) if src_cols > 0 => {
                            let idx = row * src_cols + *si;
                            if idx < src_flat.len() {
                                flat.push(src_flat[idx].clone());
                            } else {
                                flat.push(Value::Null);
                            }
                        }
                        _ => flat.push(Value::Null),
                    }
                }
            }
        } else {
            crate::vm::vm::with_current_stores(|store, heap| {
                for i in 0..src.len() {
                    let row = crate::vm::table_ops::get_row(&src, i, store, heap).unwrap_or_default();
                    for src_c in &col_map {
                        match src_c {
                            Some(si) if *si < row.len() => flat.push(row[*si].clone()),
                            _ => flat.push(Value::Null),
                        }
                    }
                }
            });
        }
    }

    Value::Table(Rc::new(RefCell::new(Table::from_flat_owned(
        flat,
        n_cols,
        result_columns,
    ))))
}

fn merge_append_table(dest: &mut Table, src: &Table) -> Result<(), String> {
    if let Some(chunk) = src.owned_flat() {
        dest.append_flat_chunk(chunk).map(|_| ())
    } else {
        crate::vm::vm::with_current_stores(|store, heap| {
            let n = src.len();
            let cols = dest.owned_num_cols().unwrap_or(0);
            let mut chunk = Vec::with_capacity(n.saturating_mul(cols));
            for i in 0..n {
                match crate::vm::table_ops::get_row(src, i, store, heap) {
                    Some(row) => chunk.extend(row),
                    None => return Err("failed to read source row".to_string()),
                }
            }
            dest.append_flat_chunk(&chunk).map(|_| ())
        })
    }
}
