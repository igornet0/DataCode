//! Built-in `web` module: HTTP client, browser automation, HTML→table extraction.

pub mod args;
pub mod browser;
pub mod data;
pub mod error;
pub mod http;
pub mod natives;
pub mod security;
pub mod values;

pub use browser::cleanup_all;
pub use values::{HttpResponse, WebElement, WebPage};

/// Access the current VM (same pattern as `system` natives).
pub(crate) fn with_vm<T, F: FnOnce(&crate::vm::Vm) -> T>(f: F) -> Option<T> {
    let ptr = crate::vm::vm::current_vm_ptr()?;
    Some(unsafe { f(&*ptr) })
}
