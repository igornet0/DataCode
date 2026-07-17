//! Connector backends for DataSource.

mod file;
mod http;
mod sql;

pub use file::FileConnector;
pub use http::HttpConnector;
pub use sql::SqlConnector;

use crate::common::table::Table;
use crate::common::value::Value;
use crate::datasource::config::DataSourceConfig;
use crate::datasource::error::DataSourceError;
use crate::datasource::request::{GetTableSpec, RequestSpec, SendTableSpec};
use crate::datasource::response::DataSourceResponse;

pub trait ConnectorBackend {
    fn connector_type(&self) -> &str;
    fn connect(&mut self) -> Result<(), DataSourceError>;
    fn disconnect(&mut self);
    fn ping(&mut self) -> Result<bool, DataSourceError>;
    fn test(&mut self) -> Result<Value, DataSourceError>;
    fn request(&mut self, spec: &RequestSpec) -> Result<DataSourceResponse, DataSourceError>;
    fn get_table(&mut self, spec: &GetTableSpec) -> Result<Table, DataSourceError>;
    fn send_table(&mut self, spec: &SendTableSpec) -> Result<(), DataSourceError>;
    fn clone_backend(&self) -> Box<dyn ConnectorBackend>;
}

pub fn create_backend(cfg: &DataSourceConfig) -> Result<Box<dyn ConnectorBackend>, DataSourceError> {
    match cfg.connector_type.as_str() {
        "http" | "https" => Ok(Box::new(HttpConnector::new(cfg.clone())?)),
        "file" => Ok(Box::new(FileConnector::new(cfg.clone())?)),
        "sqlite" | "sql" => Ok(Box::new(SqlConnector::new(cfg.clone())?)),
        t if t.starts_with("postgres") || t.starts_with("mysql") => {
            Err(DataSourceError::Unsupported {
                message: format!("connector type '{}' is not supported in v1 (SQLite only)", t),
            })
        }
        other => Err(DataSourceError::Unsupported {
            message: format!("unknown datasource type '{}'", other),
        }),
    }
}
