//! Zip-slip safe path handling for archive extraction.

use std::path::{Component, Path, PathBuf};

pub fn safe_relative_path(name: &str) -> Option<PathBuf> {
    let p = Path::new(name);
    let mut out = PathBuf::new();
    for c in p.components() {
        match c {
            Component::ParentDir => return None,
            Component::Normal(s) => out.push(s),
            Component::CurDir => {}
            Component::RootDir | Component::Prefix(_) => return None,
        }
    }
    if out.as_os_str().is_empty() {
        None
    } else {
        Some(out)
    }
}

pub fn safe_output_path(base: &Path, entry_name: &str) -> Result<PathBuf, String> {
    let rel = safe_relative_path(entry_name)
        .ok_or_else(|| format!("unsafe archive path: {}", entry_name))?;
    let root = fs::canonicalize(base).map_err(|e| format!("Cannot resolve extract destination: {}", e))?;
    let out_path = root.join(&rel);
    if !out_path.starts_with(&root) {
        return Err(format!("unsafe archive path: {}", entry_name));
    }
    Ok(out_path)
}

use std::fs;
