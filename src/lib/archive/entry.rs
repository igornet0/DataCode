//! Single file entry metadata inside an archive.

use std::path::{Path, PathBuf};

#[derive(Debug, Clone)]
pub struct ArchiveEntry {
    pub path: String,
    pub name: String,
    pub directory: String,
    pub extension: String,
    pub size: u64,
    pub compressed_size: u64,
    pub modified: Option<i64>,
    pub is_directory: bool,
}

pub fn normalize_archive_path(raw: &str) -> String {
    let mut s = raw.replace('\\', "/");
    while s.starts_with("./") {
        s = s[2..].to_string();
    }
    if s.starts_with('/') {
        s = s[1..].to_string();
    }
    s
}

pub fn entry_from_path(
    raw_path: &str,
    size: u64,
    compressed_size: u64,
    modified: Option<i64>,
    is_directory: bool,
) -> ArchiveEntry {
    let path = normalize_archive_path(raw_path);
    let pb = Path::new(&path);
    let name = pb
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();
    let directory = pb
        .parent()
        .map(|p| p.to_string_lossy().replace('\\', "/"))
        .unwrap_or_default();
    let extension = pb
        .extension()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();
    ArchiveEntry {
        path,
        name,
        directory,
        extension,
        size,
        compressed_size,
        modified,
        is_directory,
    }
}

pub fn path_stem_for_lookup(path: &str) -> String {
    normalize_archive_path(path)
}
