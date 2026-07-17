//! HTTP response host object.

use crate::common::value::ByteBuffer;
use std::collections::HashMap;
use std::time::Duration;

#[derive(Debug, Clone)]
pub struct DataSourceResponse {
    pub status: u16,
    pub success: bool,
    pub url: String,
    pub headers: HashMap<String, String>,
    pub content_type: Option<String>,
    pub content_length: Option<u64>,
    pub encoding: Option<String>,
    pub elapsed_ms: f64,
    pub body: Vec<u8>,
}

impl DataSourceResponse {
    pub fn new(
        status: u16,
        url: String,
        headers: HashMap<String, String>,
        body: Vec<u8>,
        elapsed: Duration,
    ) -> Self {
        let content_type = headers
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case("content-type"))
            .map(|(_, v)| v.clone());
        let content_length = headers
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case("content-length"))
            .and_then(|(_, v)| v.parse().ok());
        Self {
            status,
            success: (200..300).contains(&status),
            url,
            headers,
            content_type,
            content_length: content_length.or(Some(body.len() as u64)),
            encoding: None,
            elapsed_ms: elapsed.as_secs_f64() * 1000.0,
            body,
        }
    }

    pub fn text(&self) -> Result<String, String> {
        String::from_utf8(self.body.clone()).map_err(|e| {
            format!(
                "Response body is not valid UTF-8: {}",
                e
            )
        })
    }

    pub fn byte_buffer(&self) -> ByteBuffer {
        ByteBuffer::from_vec(self.body.clone())
    }

    pub fn virtual_path_for_format(&self) -> std::path::PathBuf {
        if let Some(ct) = &self.content_type {
            let ext = content_type_to_ext(ct);
            if !ext.is_empty() {
                return std::path::PathBuf::from(format!("response.{}", ext));
            }
        }
        std::path::PathBuf::from("response.bin")
    }
}

pub fn content_type_to_ext(content_type: &str) -> String {
    let ct = content_type.split(';').next().unwrap_or("").trim().to_lowercase();
    match ct.as_str() {
        "application/json" | "application/json; charset=utf-8" => "json".to_string(),
        "text/csv" => "csv".to_string(),
        "text/plain" => "txt".to_string(),
        "application/xml" | "text/xml" => "xml".to_string(),
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" => "xlsx".to_string(),
        "text/html" => "html".to_string(),
        _ if ct.contains("json") => "json".to_string(),
        _ if ct.contains("csv") => "csv".to_string(),
        _ => String::new(),
    }
}

pub fn format_from_spec_or_content_type(
    format: Option<&str>,
    content_type: Option<&str>,
    path: Option<&str>,
) -> String {
    if let Some(f) = format {
        if !f.is_empty() {
            return f.to_lowercase();
        }
    }
    if let Some(ct) = content_type {
        let ext = content_type_to_ext(ct);
        if !ext.is_empty() {
            return ext;
        }
    }
    if let Some(p) = path {
        if let Some(ext) = std::path::Path::new(p).extension() {
            return ext.to_string_lossy().to_lowercase();
        }
    }
    "txt".to_string()
}
