//! Per-connection virtual environment folder for WebSocket sessions.
//!
//! Writes in a session are denied unless ws_app.dc opts in with
//! `configure({"allow_write": true})`. Then files go to a private folder
//! `<tmp>/datacode-ve/<session>/`, created on the first write only, limited by
//! `write_quota_mb`, shown to client code as `./…` and removed when the client
//! disconnects. Server paths never reach client code.

use std::cell::RefCell;
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

const VE_ROOT_DIR: &str = "datacode-ve";

thread_local! {
    static VE_DIR: RefCell<Option<PathBuf>> = const { RefCell::new(None) };
}

static SESSION_COUNTER: AtomicU64 = AtomicU64::new(0);

/// The session folder if a write already created it.
pub fn ve_dir() -> Option<PathBuf> {
    VE_DIR.with(|d| d.borrow().clone())
}

fn quota_bytes() -> u64 {
    crate::websocket::app::app_config().write_quota_bytes
}

/// Create the session folder on first use and make it the session root, so
/// path display (`./…`) and session path resolution point inside it.
fn ensure_ve_dir() -> Result<PathBuf, String> {
    if let Some(dir) = ve_dir() {
        return Ok(dir);
    }
    let id = format!(
        "{}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0),
        SESSION_COUNTER.fetch_add(1, Ordering::Relaxed)
    );
    let dir = std::env::temp_dir().join(VE_ROOT_DIR).join(id);
    std::fs::create_dir_all(&dir).map_err(|_| "Cannot create session folder".to_string())?;
    let dir = dir.canonicalize().unwrap_or(dir);
    VE_DIR.with(|d| *d.borrow_mut() = Some(dir.clone()));
    crate::websocket::set_user_session_path(Some(dir.clone()));
    Ok(dir)
}

/// Remove the session folder (client disconnected / session finished).
pub fn cleanup_ve_dir() {
    if let Some(dir) = VE_DIR.with(|d| d.borrow_mut().take()) {
        let _ = std::fs::remove_dir_all(&dir);
        crate::websocket::set_user_session_path(None);
    }
}

/// Validate a client path for the session folder: relative, no `..`.
/// Returns the normalized relative path (`out/report.csv`).
fn session_relative(path: &Path) -> Result<PathBuf, String> {
    let raw = path.to_string_lossy();
    if raw.trim().is_empty() {
        return Err("Path must not be empty".to_string());
    }
    if path.is_absolute() || raw.starts_with('/') || raw.starts_with('\\') {
        return Err(format!(
            "Absolute paths are not allowed in a session, use a relative path like ./{}",
            path.file_name().map(|n| n.to_string_lossy()).unwrap_or_default()
        ));
    }
    let mut rel = PathBuf::new();
    for component in path.components() {
        match component {
            Component::Normal(part) => rel.push(part),
            Component::CurDir => {}
            Component::ParentDir => return Err("Path traversal not allowed in a session".to_string()),
            Component::RootDir | Component::Prefix(_) => {
                return Err("Absolute paths are not allowed in a session".to_string())
            }
        }
    }
    if rel.as_os_str().is_empty() {
        return Err("Path must name a file".to_string());
    }
    Ok(rel)
}

/// Real path inside the session folder for a write; creates the folder and
/// parent directories, and rejects the write once the quota is used up.
pub fn resolve_write_path(path: &Path) -> Result<PathBuf, String> {
    let rel = session_relative(path)?;
    let quota = quota_bytes();
    if let Some(dir) = ve_dir() {
        if dir_size(&dir) >= quota {
            return Err(quota_error(quota));
        }
    }
    let dir = ensure_ve_dir()?;
    let target = dir.join(&rel);
    if let Some(parent) = target.parent() {
        std::fs::create_dir_all(parent).map_err(|_| {
            format!("Cannot create directory ./{}", rel.parent().unwrap_or(Path::new("")).display())
        })?;
    }
    Ok(target)
}

/// After a write: enforce the quota, removing what was just written if it
/// pushed the folder over the limit.
pub fn enforce_quota_after_write(written: &Path) -> Result<(), String> {
    let Some(dir) = ve_dir() else {
        return Ok(());
    };
    let quota = quota_bytes();
    if dir_size(&dir) > quota {
        if written.is_dir() {
            let _ = std::fs::remove_dir_all(written);
        } else {
            let _ = std::fs::remove_file(written);
        }
        return Err(quota_error(quota));
    }
    Ok(())
}

/// Real path of a file previously written in this session (`out.csv`).
pub fn resolve_read_path(path: &Path) -> Option<PathBuf> {
    let dir = ve_dir()?;
    let rel = session_relative(path).ok()?;
    let target = dir.join(rel);
    target.is_file().then_some(target)
}

/// `./relative` form when `path` lies inside the session folder.
pub fn display_if_inside(path: &Path) -> Option<String> {
    let dir = ve_dir()?;
    path.strip_prefix(&dir).ok()?;
    Some(display_path(path))
}

/// Logical paths (relative to the session root) of files and folders written
/// under `prefix` (`""` = the whole session folder), for `list_files`.
pub fn list_written(prefix: &str) -> Vec<PathBuf> {
    let Some(dir) = ve_dir() else {
        return Vec::new();
    };
    let start = if prefix.is_empty() { dir.clone() } else { dir.join(prefix) };
    let mut out = Vec::new();
    collect(&dir, &start, &mut out);
    out
}

fn collect(root: &Path, current: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(current) else {
        return;
    };
    let mut entries: Vec<_> = entries.filter_map(Result::ok).map(|e| e.path()).collect();
    entries.sort();
    for path in entries {
        if let Ok(rel) = path.strip_prefix(root) {
            out.push(rel.to_path_buf());
        }
        if path.is_dir() {
            collect(root, &path, out);
        }
    }
}

/// `./relative` form of a path inside the session folder.
pub fn display_path(real: &Path) -> String {
    if let Some(dir) = ve_dir() {
        if let Ok(rel) = real.strip_prefix(&dir) {
            let rel = rel.to_string_lossy().replace('\\', "/");
            return if rel.is_empty() { "./".to_string() } else { format!("./{rel}") };
        }
    }
    real.file_name()
        .map(|n| format!("./{}", n.to_string_lossy()))
        .unwrap_or_else(|| "./".to_string())
}

fn quota_error(quota: u64) -> String {
    format!(
        "Session write quota exceeded ({} MB)",
        (quota as f64 / (1024.0 * 1024.0)).max(0.0).round()
    )
}

fn dir_size(dir: &Path) -> u64 {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return 0;
    };
    entries
        .filter_map(Result::ok)
        .map(|e| match e.metadata() {
            Ok(m) if m.is_dir() => dir_size(&e.path()),
            Ok(m) => m.len(),
            Err(_) => 0,
        })
        .sum()
}
