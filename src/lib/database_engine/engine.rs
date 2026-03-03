// Database engine abstraction for multiple backend types

use crate::common::value::Value;
use rusqlite::Connection;
use std::collections::HashMap;
use std::path::PathBuf;

/// Backend type - SQLite for MVP, others to be added
#[derive(Debug)]
pub enum DbBackend {
    SQLite(Connection),
}

/// Database engine - holds connection/connection pool and config
#[derive(Debug)]
pub struct DatabaseEngine {
    pub backend: DbBackend,
    pub url: String,
    pub echo: bool,
    pub echo_pool: bool,
    pub pool_size: u32,
    pub max_overflow: u32,
    pub timeout: Option<f64>,
    pub connect_args: HashMap<String, Value>,
}

impl DatabaseEngine {
    pub fn new_sqlite(
        url: String,
        echo: bool,
        echo_pool: bool,
        pool_size: u32,
        max_overflow: u32,
        timeout: Option<f64>,
        connect_args: HashMap<String, Value>,
    ) -> Result<Self, String> {
        let path = parse_sqlite_path(&url)?;
        let conn = Connection::open(&path).map_err(|e| format!("SQLite connection failed: {}", e))?;
        conn.execute("PRAGMA foreign_keys = ON", [])
            .map_err(|e| format!("Failed to enable foreign keys: {}", e))?;
        Ok(Self {
            backend: DbBackend::SQLite(conn),
            url,
            echo,
            echo_pool,
            pool_size,
            max_overflow,
            timeout,
            connect_args,
        })
    }

    pub fn execute(&mut self, sql: &str, params: &[Value]) -> Result<i64, String> {
        match &mut self.backend {
            DbBackend::SQLite(conn) => {
                if self.echo {
                    eprintln!("[SQL] {}", sql);
                }
                let params_vec: Vec<Box<dyn rusqlite::ToSql>> = params
                    .iter()
                    .map(value_to_sql_param)
                    .collect();
                let params_refs: Vec<&dyn rusqlite::ToSql> =
                    params_vec.iter().map(|b| b.as_ref()).collect();
                let count = conn.execute(sql, params_refs.as_slice())
                    .map_err(|e| format!("Execute failed: {}", e))?;
                Ok(count as i64)
            }
        }
    }

    pub fn query(&mut self, sql: &str, params: &[Value]) -> Result<crate::common::table::Table, String> {
        match &mut self.backend {
            DbBackend::SQLite(conn) => {
                if self.echo {
                    eprintln!("[SQL] {}", sql);
                }
                let params_vec: Vec<Box<dyn rusqlite::ToSql>> = params
                    .iter()
                    .map(value_to_sql_param)
                    .collect();
                let params_refs: Vec<&dyn rusqlite::ToSql> =
                    params_vec.iter().map(|b| b.as_ref()).collect();
                let mut stmt = conn
                    .prepare(sql)
                    .map_err(|e| format!("Prepare failed: {}", e))?;
                let column_count = stmt.column_count();
                let headers: Vec<String> = (0..column_count)
                    .map(|i| stmt.column_name(i).unwrap_or("").to_string())
                    .collect();
                let rows_iter = stmt
                    .query_map(params_refs.as_slice(), |row| {
                        let mut r = Vec::with_capacity(column_count);
                        for i in 0..column_count {
                            let v = row_get_value(row, i);
                            r.push(v);
                        }
                        Ok(r)
                    })
                    .map_err(|e| format!("Query failed: {}", e))?;
                let mut rows = Vec::new();
                for row_result in rows_iter {
                    rows.push(row_result.map_err(|e| format!("Row error: {}", e))?);
                }
                let table = crate::common::table::Table::from_data(rows, Some(headers));
                Ok(table)
            }
        }
    }
}

fn parse_sqlite_path(url: &str) -> Result<PathBuf, String> {
    let url = url.trim();
    if url.starts_with("sqlite:///") {
        Ok(PathBuf::from(&url["sqlite:///".len()..]))
    } else if url.starts_with("sqlite:") {
        Ok(PathBuf::from(&url["sqlite:".len()..]))
    } else {
        Err(format!("Invalid SQLite URL: {}", url))
    }
}

fn value_to_sql_param(v: &Value) -> Box<dyn rusqlite::ToSql> {
    match v {
        Value::Number(n) => {
            if n.fract() == 0.0 {
                Box::new(*n as i64) as Box<dyn rusqlite::ToSql>
            } else {
                Box::new(*n) as Box<dyn rusqlite::ToSql>
            }
        }
        Value::Bool(b) => Box::new(if *b { 1i64 } else { 0i64 }) as Box<dyn rusqlite::ToSql>,
        Value::String(s) => Box::new(s.clone()) as Box<dyn rusqlite::ToSql>,
        Value::Null => Box::new(Option::<String>::None) as Box<dyn rusqlite::ToSql>,
        _ => Box::new(v.to_string()) as Box<dyn rusqlite::ToSql>,
    }
}

fn row_get_value(row: &rusqlite::Row, idx: usize) -> Value {
    use rusqlite::types::Value as SqlValue;
    let sql_val = match row.get::<_, SqlValue>(idx) {
        Ok(v) => v,
        Err(_) => return Value::Null,
    };
    match sql_val {
        SqlValue::Integer(i) => Value::Number(i as f64),
        SqlValue::Real(r) => Value::Number(r),
        SqlValue::Text(s) => Value::String(s),
        SqlValue::Blob(_) => Value::Null, // Blob as null for now
        SqlValue::Null => Value::Null,
    }
}
