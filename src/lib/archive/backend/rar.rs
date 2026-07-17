//! RAR archive backend via unrar-ng (feature `archive-rar`).

use crate::archive::backend::ArchiveBackend;
use crate::archive::entry::{entry_from_path, normalize_archive_path};
use crate::archive::error::ArchiveError;
use crate::archive::format::ArchiveFormat;
use crate::archive::path_safe::safe_output_path;
use std::fs;
use std::path::Path;
use unrar_ng::Archive;

pub struct RarBackend {
    path: std::path::PathBuf,
    indexed: bool,
    entries_cache: Vec<crate::archive::entry::ArchiveEntry>,
}

impl RarBackend {
    pub fn open(path: &Path) -> Result<Self, ArchiveError> {
        Ok(Self {
            path: path.to_path_buf(),
            indexed: false,
            entries_cache: Vec::new(),
        })
    }
}

impl ArchiveBackend for RarBackend {
    fn format(&self) -> ArchiveFormat {
        ArchiveFormat::Rar
    }

    fn list_entries(&mut self) -> Result<Vec<crate::archive::entry::ArchiveEntry>, ArchiveError> {
        if self.indexed {
            return Ok(self.entries_cache.clone());
        }
        let path_display = self.path.display().to_string();
        let archive = Archive::new(&self.path)
            .open_for_listing()
            .map_err(|e| ArchiveError::corrupted(&self.path, format!("invalid rar: {}", e)))?;
        let mut out = Vec::new();
        let mut archive = archive;
        loop {
            let header = match archive.read_header() {
                Ok(Some(h)) => h,
                Ok(None) => break,
                Err(e) => {
                    return Err(ArchiveError::Corrupted {
                        path: path_display.clone(),
                        detail: format!("rar header: {}", e),
                    });
                }
            };
            let entry = header.entry();
            let name = entry.filename.to_string_lossy().into_owned();
            let is_dir = entry.is_directory();
            if entry.is_encrypted() {
                return Err(ArchiveError::PasswordProtected {
                    archive: path_display.clone(),
                    entry: name,
                });
            }
            let meta = entry_from_path(
                &name,
                entry.unpacked_size,
                entry.unpacked_size,
                None,
                is_dir,
            );
            if !is_dir {
                out.push(meta);
            }
            archive = header.skip().map_err(|e| ArchiveError::Corrupted {
                path: path_display.clone(),
                detail: format!("rar skip: {}", e),
            })?;
        }
        self.entries_cache = out.clone();
        self.indexed = true;
        Ok(out)
    }

    fn read_entry(&mut self, path: &str) -> Result<Vec<u8>, ArchiveError> {
        let key = normalize_archive_path(path);
        let path_display = self.path.display().to_string();
        let archive = Archive::new(&self.path)
            .open_for_processing()
            .map_err(|e| ArchiveError::corrupted(&self.path, format!("open rar: {}", e)))?;
        let mut archive = archive;
        loop {
            let header = match archive.read_header() {
                Ok(Some(h)) => h,
                Ok(None) => {
                    return Err(ArchiveError::EntryNotFound {
                        archive: path_display.clone(),
                        entry: path.to_string(),
                    });
                }
                Err(e) => {
                    return Err(ArchiveError::Corrupted {
                        path: path_display.clone(),
                        detail: format!("rar header: {}", e),
                    });
                }
            };
            let entry = header.entry();
            let name = normalize_archive_path(&entry.filename.to_string_lossy());
            if entry.is_encrypted() {
                return Err(ArchiveError::PasswordProtected {
                    archive: path_display.clone(),
                    entry: name,
                });
            }
            if name == key {
                let (data, _) = header.read().map_err(|e| ArchiveError::Corrupted {
                    path: path_display.clone(),
                    detail: format!("read '{}': {}", key, e),
                })?;
                return Ok(data);
            }
            archive = header.skip().map_err(|e| ArchiveError::Corrupted {
                path: path_display.clone(),
                detail: format!("rar skip: {}", e),
            })?;
        }
    }

    fn extract_all(&mut self, dest: &Path) -> Result<(), ArchiveError> {
        fs::create_dir_all(dest).map_err(|e| ArchiveError::Io {
            path: self.path.display().to_string(),
            detail: e.to_string(),
        })?;
        let path_display = self.path.display().to_string();
        let archive = Archive::new(&self.path)
            .open_for_processing()
            .map_err(|e| ArchiveError::corrupted(&self.path, format!("open rar: {}", e)))?;
        let mut archive = archive;
        loop {
            let header = match archive.read_header() {
                Ok(Some(h)) => h,
                Ok(None) => break,
                Err(e) => {
                    return Err(ArchiveError::Corrupted {
                        path: path_display.clone(),
                        detail: format!("rar header: {}", e),
                    });
                }
            };
            let entry = header.entry();
            let name = entry.filename.to_string_lossy().into_owned();
            if entry.is_encrypted() {
                return Err(ArchiveError::PasswordProtected {
                    archive: path_display.clone(),
                    entry: name,
                });
            }
            if entry.is_directory() {
                let out_path = safe_output_path(dest, &name).map_err(|e| ArchiveError::Other {
                    message: e,
                })?;
                fs::create_dir_all(&out_path).map_err(|e| ArchiveError::Io {
                    path: path_display.clone(),
                    detail: e.to_string(),
                })?;
                archive = header.skip().map_err(|e| ArchiveError::Corrupted {
                    path: path_display.clone(),
                    detail: format!("rar skip: {}", e),
                })?;
                continue;
            }
            let out_path = safe_output_path(dest, &name).map_err(|e| ArchiveError::Other {
                message: e,
            })?;
            if let Some(parent) = out_path.parent() {
                fs::create_dir_all(parent).map_err(|e| ArchiveError::Io {
                    path: path_display.clone(),
                    detail: e.to_string(),
                })?;
            }
            let (data, rest) = header.read().map_err(|e| ArchiveError::Corrupted {
                path: path_display.clone(),
                detail: format!("extract '{}': {}", name, e),
            })?;
            fs::write(&out_path, &data).map_err(|e| ArchiveError::Io {
                path: path_display.clone(),
                detail: e.to_string(),
            })?;
            archive = rest;
        }
        Ok(())
    }

    fn close(&mut self) {
        self.entries_cache.clear();
        self.indexed = false;
    }
}
