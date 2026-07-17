//! DataSource host object.

use crate::common::table::Table;
use crate::common::value::Value;
use crate::datasource::config::DataSourceConfig;
use crate::datasource::connector::{create_backend, ConnectorBackend};
use crate::datasource::error::DataSourceError;
use crate::datasource::request::{GetTableSpec, RequestSpec, SendTableSpec};
use crate::datasource::response::DataSourceResponse;
use std::cell::RefCell;

pub struct DataSource {
    pub config: DataSourceConfig,
    backend: RefCell<Box<dyn ConnectorBackend>>,
}

impl DataSource {
    pub fn new(config: DataSourceConfig) -> Result<Self, DataSourceError> {
        let backend = create_backend(&config)?;
        Ok(Self {
            config,
            backend: RefCell::new(backend),
        })
    }

    pub fn connector_type(&self) -> &str {
        self.config.connector_type.as_str()
    }

    pub fn connect(&self) -> Result<(), DataSourceError> {
        self.backend.borrow_mut().connect()
    }

    pub fn disconnect(&self) {
        self.backend.borrow_mut().disconnect();
    }

    pub fn ping(&self) -> Result<bool, DataSourceError> {
        self.backend.borrow_mut().ping()
    }

    pub fn test(&self) -> Result<Value, DataSourceError> {
        self.backend.borrow_mut().test()
    }

    pub fn request(&self, spec: &RequestSpec) -> Result<DataSourceResponse, DataSourceError> {
        self.backend.borrow_mut().request(spec)
    }

    pub fn get_table(&self, spec: &GetTableSpec) -> Result<Table, DataSourceError> {
        self.backend.borrow_mut().get_table(spec)
    }

    pub fn send_table(&self, spec: &SendTableSpec) -> Result<(), DataSourceError> {
        self.backend.borrow_mut().send_table(spec)
    }

    pub fn clone_handle(&self) -> Result<Self, DataSourceError> {
        let backend = self.backend.borrow().clone_backend();
        Ok(Self {
            config: self.config.clone(),
            backend: RefCell::new(backend),
        })
    }
}
