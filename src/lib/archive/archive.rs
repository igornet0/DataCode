//! Archive host object — lazy reading with O(1) index lookup.

use crate::archive::backend::{open_backend, ArchiveBackend};
use crate::archive::entry::{path_stem_for_lookup, ArchiveEntry};
use crate::archive::error::ArchiveError;
use crate::archive::format::{detect_format, ArchiveFormat};
use crate::common::value::Value;
use crate::file_io::{read_bytes_from_memory, ReadOptions};
use lru::LruCache;
use std::cell::RefCell;
use std::collections::HashMap;
use std::fmt;
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::rc::Rc;

const CACHE_CAPACITY: usize = 32;

pub struct Archive {
    pub path: PathBuf,
    pub format: ArchiveFormat,
    pub entries: HashMap<String, ArchiveEntry>,
    pub count: usize,
    pub size: u64,
    pub compressed_size: u64,
    backend: Box<dyn ArchiveBackend>,
    cache: RefCell<LruCache<String, Value>>,
    closed: bool,
}

impl fmt::Debug for Archive {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Archive")
            .field("path", &self.path)
            .field("format", &self.format)
            .field("count", &self.count)
            .field("closed", &self.closed)
            .finish()
    }
}

impl Archive {
    pub fn open(path: PathBuf) -> Result<Self, String> {
        let format = detect_format(&path)?;
        let compressed_size = std::fs::metadata(&path)
            .map(|m| m.len())
            .unwrap_or(0);
        let mut backend = open_backend(&path, format).map_err(|e| e.display())?;
        let listed = backend.list_entries().map_err(|e| e.display())?;
        let mut entries = HashMap::new();
        let mut size: u64 = 0;
        for entry in listed {
            size = size.saturating_add(entry.size);
            entries.insert(entry.path.clone(), entry);
        }
        let count = entries.len();
        Ok(Self {
            path,
            format,
            entries,
            count,
            size,
            compressed_size,
            backend,
            cache: RefCell::new(LruCache::new(
                NonZeroUsize::new(CACHE_CAPACITY).unwrap(),
            )),
            closed: false,
        })
    }

    fn ensure_open(&self) -> Result<(), String> {
        if self.closed {
            return Err(ArchiveError::Closed {
                path: self.path.display().to_string(),
            }
            .display());
        }
        Ok(())
    }

    pub fn read(&mut self, path: &str) -> Result<Value, String> {
        self.ensure_open()?;
        let key = path_stem_for_lookup(path);
        if let Some(cached) = self.cache.borrow_mut().get(&key) {
            return Ok(cached.clone());
        }
        if !self.entries.contains_key(&key) {
            return Err(
                ArchiveError::EntryNotFound {
                    archive: self.path.display().to_string(),
                    entry: path.to_string(),
                }
                .display(),
            );
        }
        let bytes = self
            .backend
            .read_entry(&key)
            .map_err(|e| e.display())?;
        let virtual_path = PathBuf::from(&key);
        let value = read_bytes_from_memory(&virtual_path, &bytes, &ReadOptions::default())?;
        self.cache.borrow_mut().put(key, value.clone());
        Ok(value)
    }

    pub fn read_text(&mut self, path: &str) -> Result<Value, String> {
        self.ensure_open()?;
        let key = path_stem_for_lookup(path);
        if !self.entries.contains_key(&key) {
            return Err(
                ArchiveError::EntryNotFound {
                    archive: self.path.display().to_string(),
                    entry: path.to_string(),
                }
                .display(),
            );
        }
        let bytes = self
            .backend
            .read_entry(&key)
            .map_err(|e| e.display())?;
        let text = String::from_utf8(bytes).map_err(|e| {
            format!(
                "Cannot read '{}' in archive '{}' as text: invalid UTF-8: {}",
                path,
                self.path.display(),
                e
            )
        })?;
        Ok(Value::String(text))
    }

    pub fn extract(&mut self, dest: &Path) -> Result<(), String> {
        self.ensure_open()?;
        self.backend
            .extract_all(dest)
            .map_err(|e| e.display())
    }

    pub fn close(&mut self) {
        if !self.closed {
            self.backend.close();
            self.cache.borrow_mut().clear();
            self.closed = true;
        }
    }

    pub fn files_as_values(&self) -> Vec<Value> {
        let mut keys: Vec<_> = self.entries.keys().cloned().collect();
        keys.sort();
        keys.into_iter()
            .filter_map(|k| self.entries.get(&k))
            .map(entry_to_object)
            .collect()
    }
}

fn entry_to_object(entry: &ArchiveEntry) -> Value {
    use crate::common::value::ObjectKind;
    use std::collections::HashMap;
    let mut map = HashMap::new();
    map.insert("path".to_string(), Value::String(entry.path.clone()));
    map.insert("name".to_string(), Value::String(entry.name.clone()));
    map.insert(
        "directory".to_string(),
        Value::String(entry.directory.clone()),
    );
    map.insert(
        "extension".to_string(),
        Value::String(entry.extension.clone()),
    );
    map.insert("size".to_string(), Value::Number(entry.size as f64));
    map.insert(
        "compressed_size".to_string(),
        Value::Number(entry.compressed_size as f64),
    );
    Value::Object(Rc::new(RefCell::new(ObjectKind::legacy(map))))
}

impl Drop for Archive {
    fn drop(&mut self) {
        self.close();
    }
}
