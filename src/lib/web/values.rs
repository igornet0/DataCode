//! Host structs for `web` runtime values.

use crate::common::value::Value;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

static NEXT_PAGE_ID: AtomicU64 = AtomicU64::new(1);

pub fn next_page_id() -> u64 {
    NEXT_PAGE_ID.fetch_add(1, Ordering::Relaxed)
}

/// HTTP response returned by `http.get` / `post` / …
#[derive(Debug, Clone)]
pub struct HttpResponse {
    pub status: u16,
    pub status_text: String,
    pub ok: bool,
    pub url: String,
    pub headers: HashMap<String, String>,
    pub body: Vec<u8>,
    pub json_cache: Option<Value>,
}

impl HttpResponse {
    pub fn new(status: u16, url: String, headers: HashMap<String, String>, body: Vec<u8>) -> Self {
        Self {
            status,
            status_text: status_text(status).to_string(),
            ok: (200..300).contains(&status),
            url,
            headers,
            body,
            json_cache: None,
        }
    }

    pub fn body_string(&self) -> Result<String, String> {
        String::from_utf8(self.body.clone()).map_err(|e| {
            format!("ValueError: response body is not valid UTF-8: {}", e)
        })
    }

    pub fn size(&self) -> usize {
        self.body.len()
    }
}

fn status_text(status: u16) -> &'static str {
    match status {
        100 => "Continue",
        101 => "Switching Protocols",
        200 => "OK",
        201 => "Created",
        202 => "Accepted",
        204 => "No Content",
        301 => "Moved Permanently",
        302 => "Found",
        303 => "See Other",
        304 => "Not Modified",
        307 => "Temporary Redirect",
        308 => "Permanent Redirect",
        400 => "Bad Request",
        401 => "Unauthorized",
        403 => "Forbidden",
        404 => "Not Found",
        405 => "Method Not Allowed",
        408 => "Request Timeout",
        409 => "Conflict",
        410 => "Gone",
        429 => "Too Many Requests",
        500 => "Internal Server Error",
        502 => "Bad Gateway",
        503 => "Service Unavailable",
        504 => "Gateway Timeout",
        _ => "Unknown",
    }
}

/// Live browser page instance.
pub struct WebPage {
    pub id: u64,
    pub closed: bool,
    /// Opaque driver handle shared with elements.
    pub driver: Arc<Mutex<Box<dyn crate::web::browser::driver::BrowserDriver>>>,
}

impl std::fmt::Debug for WebPage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WebPage")
            .field("id", &self.id)
            .field("closed", &self.closed)
            .finish()
    }
}

impl Drop for WebPage {
    fn drop(&mut self) {
        if !self.closed {
            let _ = self.driver.lock().map(|mut d| d.close());
            self.closed = true;
            crate::web::browser::registry::unregister(self.id);
        }
    }
}

/// Element bound to a live page DOM.
#[derive(Clone)]
pub struct WebElement {
    pub page_id: u64,
    pub selector: String,
    pub index: usize,
    pub driver: Arc<Mutex<Box<dyn crate::web::browser::driver::BrowserDriver>>>,
}

impl std::fmt::Debug for WebElement {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WebElement")
            .field("page_id", &self.page_id)
            .field("selector", &self.selector)
            .field("index", &self.index)
            .finish()
    }
}
