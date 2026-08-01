//! Errors for the built-in `web` module.

use crate::websocket::set_native_error;

#[derive(Debug, Clone)]
pub struct WebError {
    pub message: String,
}

impl WebError {
    pub fn value(msg: impl Into<String>) -> Self {
        Self {
            message: format!("ValueError: {}", msg.into()),
        }
    }

    pub fn type_err(msg: impl Into<String>) -> Self {
        Self {
            message: format!("TypeError: {}", msg.into()),
        }
    }

    pub fn io(msg: impl Into<String>) -> Self {
        Self {
            message: format!("IOError: {}", msg.into()),
        }
    }

    pub fn runtime(msg: impl Into<String>) -> Self {
        Self {
            message: format!("RuntimeError: {}", msg.into()),
        }
    }

    pub fn display(&self) -> &str {
        &self.message
    }
}

impl std::fmt::Display for WebError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for WebError {}

pub fn raise(err: WebError) {
    set_native_error(err.message);
}

pub fn raise_msg(msg: impl Into<String>) {
    set_native_error(msg.into());
}
