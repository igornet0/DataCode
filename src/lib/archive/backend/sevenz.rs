//! 7z archive backend via sevenz-rust2.

use crate::archive::backend::ArchiveBackend;
use crate::archive::entry::{entry_from_path, normalize_archive_path};
use crate::archive::error::ArchiveError;
use crate::archive::format::ArchiveFormat;
use crate::archive::path_safe::safe_output_path;
use sevenz_rust2::{Archive, ArchiveReader, Password};
use std::fs::{self, File};
use std::path::Path;

pub struct SevenZipBackend {
    path: std::path::PathBuf,
    archive: Archive,
    indexed: bool,
    entries_cache: Vec<crate::archive::entry::ArchiveEntry>,
}

impl SevenZipBackend {
    pub fn open(path: &Path) -> Result<Self, ArchiveError> {
        let archive = Archive::open(path).map_err(|e| {
            ArchiveError::corrupted(path, format!("invalid 7z: {}", e))
        })?;
        Ok(Self {
            path: path.to_path_buf(),
            archive,
            indexed: false,
            entries_cache: Vec::new(),
        })
    }

    fn reader(&self) -> Result<ArchiveReader<File>, ArchiveError> {
        let file = File::open(&self.path).map_err(|e| ArchiveError::Io {
            path: self.path.display().to_string(),
            detail: e.to_string(),
        })?;
        ArchiveReader::new(file, Password::empty()).map_err(|e| {
            ArchiveError::corrupted(&self.path, format!("7z reader: {}", e))
        })
    }
}

impl ArchiveBackend for SevenZipBackend {
    fn format(&self) -> ArchiveFormat {
        ArchiveFormat::SevenZ
    }

    fn list_entries(&mut self) -> Result<Vec<crate::archive::entry::ArchiveEntry>, ArchiveError> {
        if self.indexed {
            return Ok(self.entries_cache.clone());
        }
        let mut out = Vec::new();
        for file in &self.archive.files {
            let name = file.name().to_string();
            if name.is_empty() {
                continue;
            }
            let is_dir = file.is_directory;
            let entry = entry_from_path(
                &name,
                file.size,
                file.compressed_size,
                None,
                is_dir,
            );
            if !is_dir {
                out.push(entry);
            }
        }
        self.entries_cache = out.clone();
        self.indexed = true;
        Ok(out)
    }

    fn read_entry(&mut self, path: &str) -> Result<Vec<u8>, ArchiveError> {
        let key = normalize_archive_path(path);
        let path_display = self.path.display().to_string();
        let found = self
            .archive
            .files
            .iter()
            .any(|f| normalize_archive_path(f.name()) == key);
        if !found {
            return Err(ArchiveError::EntryNotFound {
                archive: path_display,
                entry: path.to_string(),
            });
        }
        let mut reader = self.reader()?;
        reader.read_file(&key).map_err(|e| {
            let msg = e.to_string();
            if msg.to_lowercase().contains("password") {
                ArchiveError::PasswordProtected {
                    archive: self.path.display().to_string(),
                    entry: key,
                }
            } else {
                ArchiveError::Corrupted {
                    path: self.path.display().to_string(),
                    detail: format!("read '{}': {}", key, msg),
                }
            }
        })
    }

    fn extract_all(&mut self, dest: &Path) -> Result<(), ArchiveError> {
        fs::create_dir_all(dest).map_err(|e| ArchiveError::Io {
            path: self.path.display().to_string(),
            detail: e.to_string(),
        })?;
        let path_display = self.path.display().to_string();
        let mut reader = self.reader()?;
        for file in &self.archive.files {
            let name = file.name().to_string();
            if name.is_empty() {
                continue;
            }
            if file.is_directory {
                let out_path = safe_output_path(dest, &name).map_err(|e| ArchiveError::Other {
                    message: e,
                })?;
                fs::create_dir_all(&out_path).map_err(|e| ArchiveError::Io {
                    path: path_display.clone(),
                    detail: e.to_string(),
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
            let data = reader.read_file(&name).map_err(|e| {
                let msg = e.to_string();
                if msg.to_lowercase().contains("password") {
                    ArchiveError::PasswordProtected {
                        archive: path_display.clone(),
                        entry: name.clone(),
                    }
                } else {
                    ArchiveError::Corrupted {
                        path: path_display.clone(),
                        detail: format!("extract '{}': {}", name, msg),
                    }
                }
            })?;
            fs::write(&out_path, &data).map_err(|e| ArchiveError::Io {
                path: path_display.clone(),
                detail: e.to_string(),
            })?;
        }
        Ok(())
    }

    fn close(&mut self) {
        self.entries_cache.clear();
        self.indexed = false;
    }
}
