//! Connector backends for DataSource.

mod file;
mod http;
mod mongodb;
mod sql;

pub use file::FileConnector;
pub use http::HttpConnector;
pub use mongodb::MongoConnector;
pub use sql::SqlConnector;

use crate::common::table::Table;
use crate::common::value::Value;
use crate::datasource::capabilities::Capabilities;
use crate::datasource::config::DataSourceConfig;
use crate::datasource::error::DataSourceError;
use crate::datasource::request::{GetTableSpec, RequestSpec, SendTableSpec};
use crate::datasource::response::DataSourceResponse;

pub trait ConnectorBackend {
    fn connector_type(&self) -> &str;
    fn capabilities(&self) -> Capabilities {
        Capabilities::default()
    }
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
        "sqlite" | "sql" | "postgresql" | "postgres" | "mysql" | "mariadb" | "mssql"
        | "sqlserver" => Ok(Box::new(SqlConnector::new(cfg.clone())?)),
        "mongodb" | "mongo" => Ok(Box::new(MongoConnector::new(cfg.clone())?)),
        other => Err(DataSourceError::Unsupported {
            message: format!("unknown datasource type '{}'", other),
        }),
    }
}
