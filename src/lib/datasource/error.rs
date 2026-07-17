//! DataSource error types.

#[derive(Debug)]
pub enum DataSourceError {
    Connection { message: String },
    Timeout { message: String },
    Authentication { message: String },
    NotFound { message: String },
    Permission { message: String },
    Parse { message: String },
    RateLimit { message: String },
    Validation { message: String },
    Unsupported { message: String },
    Other { message: String },
}

impl DataSourceError {
    pub fn display(&self) -> String {
        match self {
            DataSourceError::Connection { message } => format!("ConnectionError: {}", message),
            DataSourceError::Timeout { message } => format!("TimeoutError: {}", message),
            DataSourceError::Authentication { message } => {
                format!("AuthenticationError: {}", message)
            }
            DataSourceError::NotFound { message } => format!("NotFoundError: {}", message),
            DataSourceError::Permission { message } => format!("PermissionError: {}", message),
            DataSourceError::Parse { message } => format!("ParseError: {}", message),
            DataSourceError::RateLimit { message } => format!("RateLimitError: {}", message),
            DataSourceError::Validation { message } => format!("ValidationError: {}", message),
            DataSourceError::Unsupported { message } => format!("DatasourceError: {}", message),
            DataSourceError::Other { message } => format!("DatasourceError: {}", message),
        }
    }
}

impl std::fmt::Display for DataSourceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.display())
    }
}

impl From<DataSourceError> for String {
    fn from(e: DataSourceError) -> String {
        e.display()
    }
}
