//! Write a [`Table`] to a CSV file.

use crate::common::table::Table;
use crate::common::value::Value;
use std::path::Path;

fn cell_to_csv_string(value: &Value) -> String {
    match value {
        Value::Null => String::new(),
        other => other.to_string(),
    }
}

/// Write `table` to `path` as RFC 4180 CSV (header row + data rows).
pub fn write_table_csv(table: &Table, path: &Path) -> Result<(), String> {
    let mut writer = csv::Writer::from_path(path)
        .map_err(|e| format!("Failed to create CSV file: {}", e))?;

    let headers = table.headers();
    if !headers.is_empty() {
        writer
            .write_record(headers.iter())
            .map_err(|e| format!("Failed to write CSV header: {}", e))?;
    }

    let Some(rows) = table.rows_ref() else {
        return Err(
            "CSV export: view table must be materialized before export".to_string(),
        );
    };

    for row in rows.iter() {
        let record: Vec<String> = row.iter().map(cell_to_csv_string).collect();
        writer
            .write_record(&record)
            .map_err(|e| format!("Failed to write CSV row: {}", e))?;
    }

    writer
        .flush()
        .map_err(|e| format!("Failed to flush CSV file: {}", e))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::table::Table;
    use std::fs;

    #[test]
    fn writes_header_and_rows() {
        let table = Table::from_data(
            vec![
                vec![Value::Number(1.0), Value::String("Alex".to_string())],
                vec![Value::Number(2.0), Value::String("Kate".to_string())],
            ],
            Some(vec!["id".to_string(), "name".to_string()]),
        );
        let dir = std::env::temp_dir().join("dc_csv_export_test");
        let _ = fs::create_dir_all(&dir);
        let path = dir.join("out.csv");
        write_table_csv(&table, &path).unwrap();
        let content = fs::read_to_string(&path).unwrap();
        assert!(content.contains("id,name"));
        assert!(content.contains("1,Alex"));
        assert!(content.contains("2,Kate"));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn empty_table_writes_headers_only() {
        let table = Table::from_data(vec![], Some(vec!["a".to_string(), "b".to_string()]));
        let dir = std::env::temp_dir().join("dc_csv_export_empty_test");
        let _ = fs::create_dir_all(&dir);
        let path = dir.join("empty.csv");
        write_table_csv(&table, &path).unwrap();
        let content = fs::read_to_string(&path).unwrap().trim().to_string();
        assert_eq!(content, "a,b");
        let _ = fs::remove_dir_all(&dir);
    }
}
