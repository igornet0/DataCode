//! Tabular file readers (CSV, XLSX).

use crate::common::table::Table;
use crate::common::value::Value;
use crate::vm::natives::table::try_parse_date;
use std::io;

pub fn read_csv_bytes(content: &[u8]) -> Result<Table, io::Error> {
    read_csv_bytes_inner(content, true)
}

/// Read CSV with every row as data (no separate header row). Used before transpose.
pub fn read_csv_bytes_raw(content: &[u8]) -> Result<Table, io::Error> {
    read_csv_bytes_inner(content, false)
}

fn read_csv_bytes_inner(content: &[u8], has_headers: bool) -> Result<Table, io::Error> {
    use csv::ReaderBuilder;
    let mut reader = ReaderBuilder::new()
        .has_headers(has_headers)
        .from_reader(content);
    build_csv_table(&mut reader, has_headers)
}

fn build_csv_table<R: std::io::Read>(
    reader: &mut csv::Reader<R>,
    has_headers: bool,
) -> Result<Table, io::Error> {
    let mut rows = Vec::new();
    let mut max_cols = 0usize;
    if has_headers {
        let headers: Vec<String> = reader.headers()?.iter().map(|s| s.to_string()).collect();
        max_cols = headers.len();
        for result in reader.records() {
            let record = result?;
            let row: Vec<Value> = record.iter().map(|field| parse_csv_field(field)).collect();
            max_cols = max_cols.max(row.len());
            rows.push(row);
        }
        pad_rows(&mut rows, max_cols);
        return Ok(Table::from_data(rows, Some(headers)));
    }

    for result in reader.records() {
        let record = result?;
        let row: Vec<Value> = record.iter().map(|field| parse_csv_field(field)).collect();
        max_cols = max_cols.max(row.len());
        rows.push(row);
    }
    pad_rows(&mut rows, max_cols);
    let headers: Vec<String> = (0..max_cols).map(|i| format!("Column_{}", i)).collect();
    Ok(Table::from_data(rows, Some(headers)))
}

fn pad_rows(rows: &mut [Vec<Value>], width: usize) {
    for row in rows.iter_mut() {
        while row.len() < width {
            row.push(Value::Null);
        }
    }
}

fn parse_csv_field(field: &str) -> Value {
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
}

pub fn read_xlsx_bytes(
    content: &[u8],
    header_row: usize,
    sheet_name: Option<&str>,
) -> Result<Table, Box<dyn std::error::Error>> {
    read_xlsx_bytes_inner(content, header_row, sheet_name, false)
}

/// Read XLSX with all rows as data (no header row extraction). Used before transpose.
pub fn read_xlsx_bytes_raw(
    content: &[u8],
    sheet_name: Option<&str>,
) -> Result<Table, Box<dyn std::error::Error>> {
    read_xlsx_bytes_inner(content, 0, sheet_name, true)
}

fn read_xlsx_bytes_inner(
    content: &[u8],
    header_row: usize,
    sheet_name: Option<&str>,
    raw: bool,
) -> Result<Table, Box<dyn std::error::Error>> {
    use calamine::{open_workbook_from_rs, Reader, Xlsx};
    use std::io::Cursor;
    let cursor = Cursor::new(content);
    let mut workbook: Xlsx<_> = open_workbook_from_rs(cursor)?;
    let sheet = if let Some(name) = sheet_name {
        workbook.worksheet_range(name)?
    } else {
        let sheet_names = workbook.sheet_names();
        if sheet_names.is_empty() {
            return Err("No sheets found".into());
        }
        workbook.worksheet_range(&sheet_names[0])?
    };
    if raw {
        read_xlsx_sheet_raw(sheet)
    } else {
        read_xlsx_sheet(sheet, header_row)
    }
}

fn parse_xlsx_row(row: &[calamine::Data]) -> Vec<Value> {
    row.iter()
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
        .collect()
}

fn read_xlsx_sheet_raw(
    sheet: calamine::Range<calamine::Data>,
) -> Result<Table, Box<dyn std::error::Error>> {
    let mut rows = Vec::new();
    let mut max_cols = 0usize;
    for row in sheet.rows() {
        let values = parse_xlsx_row(row);
        max_cols = max_cols.max(values.len());
        rows.push(values);
    }
    pad_rows(&mut rows, max_cols);
    let headers: Vec<String> = (0..max_cols).map(|i| format!("Column_{}", i)).collect();
    Ok(Table::from_data(rows, Some(headers)))
}

fn read_xlsx_sheet(
    sheet: calamine::Range<calamine::Data>,
    header_row: usize,
) -> Result<Table, Box<dyn std::error::Error>> {
    let mut rows = Vec::new();
    let mut headers = Vec::new();
    for (row_idx, row) in sheet.rows().enumerate() {
        let values = parse_xlsx_row(row);
        if row_idx == header_row {
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
    if headers.is_empty() && !rows.is_empty() {
        let num_cols = rows[0].len();
        headers = (0..num_cols).map(|i| format!("Column_{}", i)).collect();
    }
    Ok(Table::from_data(rows, Some(headers)))
}

/// Transpose table: rows become columns and columns become rows.
pub fn transpose_table(table: Table) -> Table {
    let rows = match table.rows_ref() {
        Some(rr) => rr.to_vec(),
        None => return table,
    };
    let num_cols = table.headers().len();
    let num_rows = rows.len();
    if num_cols == 0 || num_rows == 0 {
        return table;
    }

    let mut new_rows = Vec::with_capacity(num_cols);
    for col_idx in 0..num_cols {
        let mut new_row = Vec::with_capacity(num_rows);
        for row in &rows {
            new_row.push(row.get(col_idx).cloned().unwrap_or(Value::Null));
        }
        new_rows.push(new_row);
    }
    let new_headers: Vec<String> = (0..num_rows).map(|i| format!("Column_{}", i)).collect();
    Table::from_data(new_rows, Some(new_headers))
}

/// Pick `header_row` from table rows as column headers; remaining rows become data.
pub fn apply_header_row(table: Table, header_row: usize) -> Table {
    let rows = match table.rows_ref() {
        Some(rr) => rr.to_vec(),
        None => return table,
    };
    if rows.is_empty() {
        return table;
    }
    let header_row = header_row.min(rows.len().saturating_sub(1));
    let headers: Vec<String> = rows[header_row]
        .iter()
        .map(|v| match v {
            Value::String(s) => s.clone(),
            _ => v.to_string(),
        })
        .collect();
    let mut data_rows = Vec::new();
    for (idx, row) in rows.iter().enumerate() {
        if idx != header_row {
            data_rows.push(row.clone());
        }
    }
    if headers.is_empty() && !data_rows.is_empty() {
        let num_cols = data_rows[0].len();
        let generated: Vec<String> = (0..num_cols).map(|i| format!("Column_{}", i)).collect();
        return Table::from_data(data_rows, Some(generated));
    }
    Table::from_data(data_rows, Some(headers))
}

/// Raw read → transpose → header_row on transposed table → headerT filter.
pub fn finish_transposed_read(table: Table, header_row: usize, header_t: Option<&Value>) -> Table {
    let transposed = transpose_table(table);
    let with_headers = apply_header_row(transposed, header_row);
    apply_header_filter(with_headers, header_t)
}

/// Apply column filter/rename from `read(..., header=...)`.
pub fn apply_header_filter(table: Table, header_arg: Option<&Value>) -> Table {
    let header_arg = match header_arg {
        Some(v) => v,
        None => return table,
    };
    match header_arg {
        Value::Array(cols_arr) => {
            let cols_arr_ref = cols_arr.borrow();
            let mut selected_cols = Vec::new();
            for col_val in cols_arr_ref.iter() {
                if let Value::String(s) = col_val {
                    selected_cols.push(s.clone());
                }
            }
            if selected_cols.is_empty() {
                return table;
            }
            let mut col_indices = Vec::new();
            let mut new_headers = Vec::new();
            for col_name in &selected_cols {
                if let Some(idx) = table.headers().iter().position(|h| h == col_name) {
                    col_indices.push(idx);
                    new_headers.push(col_name.clone());
                }
            }
            if col_indices.is_empty() {
                return table;
            }
            let mut new_rows = Vec::new();
            let rr = table.rows_ref().unwrap();
            for row in rr.iter() {
                let mut new_row = Vec::new();
                for &idx in &col_indices {
                    new_row.push(row.get(idx).cloned().unwrap_or(Value::Null));
                }
                new_rows.push(new_row);
            }
            Table::from_data(new_rows, Some(new_headers))
        }
        Value::Object(rename_map_rc) => {
            let rename_map = rename_map_rc.borrow();
            let mut new_headers = Vec::new();
            for old_header in table.headers() {
                if let Some(new_name_val) = rename_map.str_key_get(old_header) {
                    match new_name_val {
                        Value::String(new_name) => new_headers.push(new_name.clone()),
                        Value::Null => new_headers.push(old_header.clone()),
                        _ => new_headers.push(old_header.clone()),
                    }
                } else {
                    new_headers.push(old_header.clone());
                }
            }
            Table::from_data(table.rows_ref().unwrap().to_vec(), Some(new_headers))
        }
        _ => table,
    }
}
