//! DataSource module — universal data connectors.

pub mod capabilities;
pub mod config;
pub mod connector;
pub mod datasource;
pub mod error;
pub mod get_table;
pub mod natives;
pub mod normalize;
pub mod request;
pub mod response;
pub mod send_table;

pub use datasource::DataSource;
pub use error::DataSourceError;
pub use response::DataSourceResponse;
