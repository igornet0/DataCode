//! ZIP archive backend.

use crate::archive::backend::ArchiveBackend;
use crate::archive::entry::{entry_from_path, normalize_archive_path};
use crate::archive::error::ArchiveError;
use crate::archive::format::ArchiveFormat;
use crate::archive::mmap::ArchiveSource;
use crate::archive::path_safe::safe_output_path;
use std::fs;
use std::io::Read;
use std::path::Path;
use zip::read::ZipArchive;

pub struct ZipBackend {
    source: ArchiveSource,
    indexed: bool,
    entries_cache: Vec<crate::archive::entry::ArchiveEntry>,
}

impl ZipBackend {
    pub fn open(path: &Path) -> Result<Self, ArchiveError> {
        let source = ArchiveSource::open(path).map_err(|e| ArchiveError::Other { message: e })?;
        Ok(Self {
            source,
            indexed: false,
            entries_cache: Vec::new(),
        })
    }

    fn with_archive<F, T>(&self, f: F) -> Result<T, ArchiveError>
    where
        F: FnOnce(
            &mut ZipArchive<&mut crate::archive::mmap::ArchiveCursor>,
        ) -> Result<T, ArchiveError>,
    {
        let mut cursor = self.source.cursor();
        let mut archive = ZipArchive::new(&mut cursor).map_err(|e| {
            ArchiveError::corrupted(&self.source.path, format!("invalid zip: {}", e))
        })?;
        f(&mut archive)
    }

    fn read_entry_inner(&self, key: &str, original_path: &str) -> Result<Vec<u8>, ArchiveError> {
        let path_display = self.source.path.display().to_string();
        self.with_archive(|archive| {
            let mut file = archive.by_name(key).map_err(|_| ArchiveError::EntryNotFound {
                archive: path_display.clone(),
                entry: original_path.to_string(),
            })?;
            if file.encrypted() {
                return Err(ArchiveError::PasswordProtected {
                    archive: path_display,
                    entry: key.to_string(),
                });
            }
            let mut buf = Vec::with_capacity(file.size() as usize);
            file.read_to_end(&mut buf).map_err(|e| ArchiveError::Corrupted {
                path: self.source.path.display().to_string(),
                detail: format!("read '{}': {}", key, e),
            })?;
            Ok(buf)
        })
    }
}

impl ArchiveBackend for ZipBackend {
    fn format(&self) -> ArchiveFormat {
        ArchiveFormat::Zip
    }

    fn list_entries(&mut self) -> Result<Vec<crate::archive::entry::ArchiveEntry>, ArchiveError> {
        if self.indexed {
            return Ok(self.entries_cache.clone());
        }
        let path_display = self.source.path.display().to_string();
        let mut out = Vec::new();
        self.with_archive(|archive| {
            for i in 0..archive.len() {
                let file = archive.by_index(i).map_err(|e| ArchiveError::Corrupted {
                    path: path_display.clone(),
                    detail: format!("zip entry {}: {}", i, e),
                })?;
                let name = file.name().to_string();
                if name.is_empty() {
                    continue;
                }
                let is_dir = name.ends_with('/') || file.is_dir();
                if file.encrypted() {
                    return Err(ArchiveError::PasswordProtected {
                        archive: path_display.clone(),
                        entry: name,
                    });
                }
                let entry = entry_from_path(
                    &name,
                    file.size(),
                    file.compressed_size(),
                    None,
                    is_dir,
                );
                if !is_dir {
                    out.push(entry);
                }
            }
            Ok(())
        })?;
        self.entries_cache = out.clone();
        self.indexed = true;
        Ok(out)
    }

    fn read_entry(&mut self, path: &str) -> Result<Vec<u8>, ArchiveError> {
        let key = normalize_archive_path(path);
        match self.read_entry_inner(&key, path) {
            Ok(v) => Ok(v),
            Err(ArchiveError::EntryNotFound { .. }) => {
                self.read_entry_inner(&format!("{}/", key), path)
            }
            Err(e) => Err(e),
        }
    }

    fn extract_all(&mut self, dest: &Path) -> Result<(), ArchiveError> {
        fs::create_dir_all(dest).map_err(|e| ArchiveError::Io {
            path: self.source.path.display().to_string(),
            detail: e.to_string(),
        })?;
        let path_display = self.source.path.display().to_string();
        self.with_archive(|archive| {
            for i in 0..archive.len() {
                let mut file = archive.by_index(i).map_err(|e| ArchiveError::Corrupted {
                    path: path_display.clone(),
                    detail: format!("zip entry {}: {}", i, e),
                })?;
                let name = file.name().to_string();
                if name.is_empty() {
                    continue;
                }
                if file.encrypted() {
                    return Err(ArchiveError::PasswordProtected {
                        archive: path_display.clone(),
                        entry: name,
                    });
                }
                let out_path = safe_output_path(dest, &name).map_err(|e| ArchiveError::Other {
                    message: e,
                })?;
                if name.ends_with('/') || file.is_dir() {
                    fs::create_dir_all(&out_path).map_err(|e| ArchiveError::Io {
                        path: path_display.clone(),
                        detail: e.to_string(),
                    })?;
                    continue;
                }
                if let Some(parent) = out_path.parent() {
                    fs::create_dir_all(parent).map_err(|e| ArchiveError::Io {
                        path: path_display.clone(),
                        detail: e.to_string(),
                    })?;
                }
                let mut out_file = fs::File::create(&out_path).map_err(|e| ArchiveError::Io {
                    path: path_display.clone(),
                    detail: e.to_string(),
                })?;
                std::io::copy(&mut file, &mut out_file).map_err(|e| ArchiveError::Io {
                    path: path_display.clone(),
                    detail: e.to_string(),
                })?;
            }
            Ok(())
        })
    }

    fn close(&mut self) {
        self.entries_cache.clear();
        self.indexed = false;
    }
}
