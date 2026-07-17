//! Archive error types with user-facing messages.

use std::path::Path;

#[derive(Debug)]
pub enum ArchiveError {
    NotFound { path: String },
    UnsupportedFormat { path: String },
    Corrupted { path: String, detail: String },
    EntryNotFound { archive: String, entry: String },
    PasswordProtected { archive: String, entry: String },
    Closed { path: String },
    RarNotCompiled { path: String },
    Io { path: String, detail: String },
    Other { message: String },
}

impl ArchiveError {
    pub fn display(&self) -> String {
        match self {
            ArchiveError::NotFound { path } => format!("Archive not found: '{}'", path),
            ArchiveError::UnsupportedFormat { path } => {
                format!("Unsupported archive format in '{}'", path)
            }
            ArchiveError::Corrupted { path, detail } => {
                format!("Corrupted archive '{}': {}", path, detail)
            }
            ArchiveError::EntryNotFound { archive, entry } => {
                format!("File '{}' not found in archive '{}'", entry, archive)
            }
            ArchiveError::PasswordProtected { archive, entry } => format!(
                "Password-protected entry '{}' in '{}' is not supported",
                entry, archive
            ),
            ArchiveError::Closed { path } => format!("Archive '{}' is closed", path),
            ArchiveError::RarNotCompiled { path } => format!(
                "RAR support not compiled (enable feature archive-rar) for '{}'",
                path
            ),
            ArchiveError::Io { path, detail } => {
                format!("Archive I/O error for '{}': {}", path, detail)
            }
            ArchiveError::Other { message } => message.clone(),
        }
    }

    pub fn not_found(path: &Path) -> Self {
        ArchiveError::NotFound {
            path: path.display().to_string(),
        }
    }

    pub fn corrupted(path: &Path, detail: impl Into<String>) -> Self {
        ArchiveError::Corrupted {
            path: path.display().to_string(),
            detail: detail.into(),
        }
    }
}

impl std::fmt::Display for ArchiveError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.display())
    }
}

impl From<ArchiveError> for String {
    fn from(e: ArchiveError) -> String {
        e.display()
    }
}
