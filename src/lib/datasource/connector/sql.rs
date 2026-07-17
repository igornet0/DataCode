//! SQLite connector wrapping DatabaseEngine.

use crate::common::table::Table;
use crate::common::value::Value;
use crate::database_engine::engine::DatabaseEngine;
use crate::datasource::config::{sqlite_url_from_config, DataSourceConfig};
use crate::datasource::connector::ConnectorBackend;
use crate::datasource::error::DataSourceError;
use crate::datasource::request::{GetTableSpec, RequestSpec, SendTableSpec};
use crate::datasource::response::DataSourceResponse;
use crate::datasource::send_table::table_rows_batch;
use std::collections::HashMap;
use std::time::Instant;

pub struct SqlConnector {
    config: DataSourceConfig,
    engine: Option<DatabaseEngine>,
    url: String,
}

impl SqlConnector {
    pub fn new(config: DataSourceConfig) -> Result<Self, DataSourceError> {
        let url = sqlite_url_from_config(&config)?;
        Ok(Self {
            config,
            engine: None,
            url,
        })
    }

    fn engine_mut(&mut self) -> Result<&mut DatabaseEngine, DataSourceError> {
        if self.engine.is_none() {
            self.connect()?;
        }
        Ok(self.engine.as_mut().unwrap())
    }
}

impl ConnectorBackend for SqlConnector {
    fn connector_type(&self) -> &str {
        "sqlite"
    }

    fn connect(&mut self) -> Result<(), DataSourceError> {
        if self.engine.is_some() {
            return Ok(());
        }
        let engine = DatabaseEngine::new_sqlite(
            self.url.clone(),
            false,
            false,
            5,
            10,
            self.config.timeout,
            HashMap::new(),
        )
        .map_err(|e| DataSourceError::Connection { message: e })?;
        self.engine = Some(engine);
        Ok(())
    }

    fn disconnect(&mut self) {
        self.engine = None;
    }

    fn ping(&mut self) -> Result<bool, DataSourceError> {
        let engine = self.engine_mut()?;
        engine
            .query("SELECT 1", &[])
            .map(|_| true)
            .map_err(|e| DataSourceError::Connection { message: e })
    }

    fn test(&mut self) -> Result<Value, DataSourceError> {
        match self.ping() {
            Ok(ok) => Ok(sql_diagnostic(ok, &self.url, None)),
            Err(e) => Ok(sql_diagnostic(false, &self.url, Some(e.display()))),
        }
    }

    fn request(&mut self, spec: &RequestSpec) -> Result<DataSourceResponse, DataSourceError> {
        let sql = spec.sql.as_ref().ok_or_else(|| DataSourceError::Validation {
            message: "sql request requires 'sql' field".to_string(),
        })?;
        let start = Instant::now();
        let engine = self.engine_mut()?;
        let sql_upper = sql.trim().to_uppercase();
        if sql_upper.starts_with("SELECT") {
            let table = engine
                .query(sql, &spec.parameters)
                .map_err(|e| DataSourceError::Other { message: e })?;
            let body = format!(
                "{{\"rows\": {}, \"columns\": {}}}",
                table.len(),
                table.column_count()
            );
            Ok(DataSourceResponse::new(
                200,
                self.url.clone(),
                HashMap::from([(
                    "Content-Type".to_string(),
                    "application/json".to_string(),
                )]),
                body.into_bytes(),
                start.elapsed(),
            ))
        } else {
            let count = engine
                .execute(sql, &spec.parameters)
                .map_err(|e| DataSourceError::Other { message: e })?;
            let body = format!("{{\"affected_rows\": {}}}", count);
            Ok(DataSourceResponse::new(
                200,
                self.url.clone(),
                HashMap::from([(
                    "Content-Type".to_string(),
                    "application/json".to_string(),
                )]),
                body.into_bytes(),
                start.elapsed(),
            ))
        }
    }

    fn get_table(&mut self, spec: &GetTableSpec) -> Result<Table, DataSourceError> {
        let sql = spec.sql.as_ref().ok_or_else(|| DataSourceError::Validation {
            message: "get_table for sqlite requires 'sql' field".to_string(),
        })?;
        let engine = self.engine_mut()?;
        engine
            .query(sql, &spec.parameters)
            .map_err(|e| DataSourceError::Other { message: e })
    }

    fn send_table(&mut self, spec: &SendTableSpec) -> Result<(), DataSourceError> {
        if spec.mode != "append" {
            return Err(DataSourceError::Unsupported {
                message: format!("send_table mode '{}' not supported in v1 (use 'append')", spec.mode),
            });
        }
        let table_name = spec.table_name.as_ref().ok_or_else(|| DataSourceError::Validation {
            message: "send_table for sqlite requires 'table_name'".to_string(),
        })?;
        let table_val = spec.table.as_ref().ok_or_else(|| DataSourceError::Validation {
            message: "send_table requires table".to_string(),
        })?;
        let Value::Table(rc) = table_val else {
            return Err(DataSourceError::Validation {
                message: "send_table requires a table value".to_string(),
            });
        };
        let table = rc.borrow();
        let headers = table.headers().clone();
        if headers.is_empty() {
            return Ok(());
        }
        let col_list = headers.join(", ");
        let placeholders = (0..headers.len()).map(|_| "?").collect::<Vec<_>>().join(", ");
        let sql = format!(
            "INSERT INTO {} ({}) VALUES ({})",
            table_name, col_list, placeholders
        );
        let engine = self.engine_mut()?;
        for batch in table_rows_batch(&table, spec.batch_size) {
            for row in batch {
                engine
                    .execute(&sql, &row)
                    .map_err(|e| DataSourceError::Other { message: e })?;
            }
        }
        Ok(())
    }

    fn clone_backend(&self) -> Box<dyn ConnectorBackend> {
        Box::new(Self {
            config: self.config.clone(),
            engine: None,
            url: self.url.clone(),
        })
    }
}

fn sql_diagnostic(ok: bool, url: &str, message: Option<String>) -> Value {
    use crate::common::value::ObjectKind;
    use std::cell::RefCell;
    use std::rc::Rc;
    let mut m = HashMap::new();
    m.insert("ok".to_string(), Value::Bool(ok));
    m.insert("type".to_string(), Value::String("sqlite".to_string()));
    m.insert("url".to_string(), Value::String(url.to_string()));
    if let Some(msg) = message {
        m.insert("message".to_string(), Value::String(msg));
    } else if ok {
        m.insert("message".to_string(), Value::String("ok".to_string()));
    }
    Value::Object(Rc::new(RefCell::new(ObjectKind::legacy(m))))
}
