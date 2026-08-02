//! File connector — local paths, lib://, archives.

use crate::archive::archive::Archive;
use crate::archive::format::detect_format;
use crate::common::table::Table;
use crate::common::value::Value;
use crate::datasource::capabilities::Capabilities;
use crate::datasource::config::DataSourceConfig;
use crate::datasource::connector::ConnectorBackend;
use crate::datasource::error::DataSourceError;
use crate::datasource::get_table::bytes_to_table;
use crate::datasource::request::{GetTableSpec, RequestSpec, SendTableSpec};
use crate::datasource::response::DataSourceResponse;
use crate::datasource::send_table::save_table_to_path;
use crate::file_io::read_bytes_from_path;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

pub struct FileConnector {
    config: DataSourceConfig,
    base_path: PathBuf,
}

impl FileConnector {
    pub fn new(config: DataSourceConfig) -> Result<Self, DataSourceError> {
        let base = config
            .path
            .as_ref()
            .map(PathBuf::from)
            .or_else(|| config.url.as_ref().map(PathBuf::from))
            .unwrap_or_else(|| PathBuf::from("."));
        Ok(Self {
            config,
            base_path: base,
        })
    }

    fn resolve_path(&self, spec_path: Option<&str>) -> PathBuf {
        if let Some(p) = spec_path {
            let pb = PathBuf::from(p);
            if pb.is_absolute() {
                return pb;
            }
            return self.base_path.join(pb);
        }
        self.base_path.clone()
    }

    fn read_bytes_at(&self, path: &PathBuf) -> Result<Vec<u8>, DataSourceError> {
        if let Some((archive_path, inner)) = split_archive_inner(path) {
            let mut arch = Archive::open(archive_path.clone()).map_err(|e| {
                DataSourceError::NotFound { message: e }
            })?;
            let val = arch.read(&inner).map_err(|e| DataSourceError::NotFound {
                message: e,
            })?;
            return value_to_bytes(&val);
        }
        if detect_format(path).is_ok() {
            return Err(DataSourceError::Validation {
                message: format!(
                    "path '{}' is an archive; specify inner file like '{}inner.csv'",
                    path.display(),
                    path.display()
                ),
            });
        }
        read_bytes_from_path(path).map_err(|e| {
            if e.to_lowercase().contains("not found") || e.contains("No such file") {
                DataSourceError::NotFound { message: e }
            } else {
                DataSourceError::Other { message: e }
            }
        })
    }
}

impl ConnectorBackend for FileConnector {
    fn connector_type(&self) -> &str {
        "file"
    }

    fn capabilities(&self) -> Capabilities {
        Capabilities::file_default()
    }

    fn connect(&mut self) -> Result<(), DataSourceError> {
        if !self.base_path.exists() {
            return Err(DataSourceError::NotFound {
                message: format!("path '{}' does not exist", self.base_path.display()),
            });
        }
        Ok(())
    }

    fn disconnect(&mut self) {}

    fn ping(&mut self) -> Result<bool, DataSourceError> {
        Ok(self.base_path.exists())
    }

    fn test(&mut self) -> Result<Value, DataSourceError> {
        let ok = self.base_path.exists();
        Ok(file_diagnostic(ok, &self.base_path))
    }

    fn request(&mut self, spec: &RequestSpec) -> Result<DataSourceResponse, DataSourceError> {
        let path = self.resolve_path(spec.path.as_deref());
        let start = Instant::now();
        let bytes = self.read_bytes_at(&path)?;
        let mut headers = HashMap::new();
        if let Some(ext) = path.extension() {
            let ext = ext.to_string_lossy();
            headers.insert(
                "Content-Type".to_string(),
                extension_to_content_type(&ext),
            );
        }
        Ok(DataSourceResponse::new(
            200,
            path.to_string_lossy().into_owned(),
            headers,
            bytes,
            start.elapsed(),
        ))
    }

    fn get_table(&mut self, spec: &GetTableSpec) -> Result<Table, DataSourceError> {
        let path = self.resolve_path(spec.path.as_deref());
        let bytes = self.read_bytes_at(&path)?;
        bytes_to_table(&bytes, &path, spec, None)
    }

    fn send_table(&mut self, spec: &SendTableSpec) -> Result<(), DataSourceError> {
        let table = spec.table.as_ref().ok_or_else(|| DataSourceError::Validation {
            message: "send_table requires table".to_string(),
        })?;
        let path = spec
            .path
            .as_ref()
            .map(|p| self.resolve_path(Some(p)))
            .unwrap_or_else(|| self.base_path.clone());
        save_table_to_path(table, &path)
    }

    fn clone_backend(&self) -> Box<dyn ConnectorBackend> {
        Box::new(Self {
            config: self.config.clone(),
            base_path: self.base_path.clone(),
        })
    }
}

fn split_archive_inner(path: &Path) -> Option<(PathBuf, String)> {
    let s = path.to_string_lossy();
    for ext in &[".zip", ".7z", ".rar"] {
        if let Some(pos) = s.to_lowercase().find(ext) {
            let end = pos + ext.len();
            if end < s.len() && (s.as_bytes().get(end) == Some(&b'/') || s.as_bytes().get(end) == Some(&b'\\')) {
                let arch = PathBuf::from(&s[..end]);
                let inner = s[end + 1..].to_string();
                return Some((arch, inner));
            }
        }
    }
    None
}

fn value_to_bytes(val: &Value) -> Result<Vec<u8>, DataSourceError> {
    match val {
        Value::String(s) => Ok(s.as_bytes().to_vec()),
        Value::ByteBuffer(b) => {
            let end = b.offset.saturating_add(b.len);
            Ok(b.bytes[b.offset..end.min(b.bytes.len())].to_vec())
        }
        other => {
            let json = crate::file_io::value_serde::value_to_json(other).map_err(|e| {
                DataSourceError::Parse {
                    message: e.message(),
                }
            })?;
            serde_json::to_vec(&json).map_err(|e| DataSourceError::Parse {
                message: format!("JSON serialize: {}", e),
            })
        }
    }
}

fn extension_to_content_type(ext: &str) -> String {
    match ext.to_lowercase().as_str() {
        "json" => "application/json".to_string(),
        "csv" => "text/csv".to_string(),
        "xlsx" => {
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet".to_string()
        }
        "xml" => "application/xml".to_string(),
        "txt" => "text/plain".to_string(),
        _ => "application/octet-stream".to_string(),
    }
}

fn file_diagnostic(ok: bool, path: &Path) -> Value {
    use crate::common::value::ObjectKind;
    use std::cell::RefCell;
    use std::rc::Rc;
    let mut m = HashMap::new();
    m.insert("ok".to_string(), Value::Bool(ok));
    m.insert("type".to_string(), Value::String("file".to_string()));
    m.insert(
        "path".to_string(),
        Value::Path(path.to_path_buf()),
    );
    if ok {
        m.insert("message".to_string(), Value::String("ok".to_string()));
    } else {
        m.insert(
            "message".to_string(),
            Value::String(format!("path not found: {}", path.display())),
        );
    }
    Value::Object(Rc::new(RefCell::new(ObjectKind::legacy(m))))
}
