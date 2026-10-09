//! Path normalization for file I/O (`String`, `Path`, SMB `lib://`).

use crate::common::value::Value;
use std::path::{Path, PathBuf};

pub fn path_from_value(v: &Value) -> Result<PathBuf, String> {
    match v {
        Value::String(s) => Ok(PathBuf::from(s)),
        Value::Path(p) => Ok(p.clone()),
        _ => Err("read/save: expected path or string as file argument".to_string()),
    }
}

pub fn extension_lower(path: &Path) -> String {
    path.extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase()
}

pub fn format_path_for_error(path: &Path) -> String {
    crate::vm::natives::file::format_path_for_error(&path.to_path_buf())
}

pub fn resolve_local_path(path: &PathBuf) -> Result<PathBuf, String> {
    crate::vm::natives::file::resolve_path_in_session(path)
}

const LIB_SMB_PREFIX: &str = "lib://";

/// Parse `lib://share_name/path/on/share` into `(share_name, path_on_share)`.
/// Leading slashes after `lib://` are normalized (`lib:///data/file` → share `data`).
pub fn parse_lib_smb_path(file_path_str: &str) -> Result<(String, String), String> {
    if !file_path_str.starts_with(LIB_SMB_PREFIX) {
        return Err(format!(
            "Invalid SMB path (must start with {}): {}",
            LIB_SMB_PREFIX, file_path_str
        ));
    }
    let mut rest = &file_path_str[LIB_SMB_PREFIX.len()..];
    while rest.starts_with('/') {
        rest = &rest[1..];
    }
    if rest.is_empty() {
        return Err(invalid_lib_smb_path_error(file_path_str));
    }
    let parts: Vec<&str> = rest.splitn(2, '/').collect();
    let share_name = parts[0];
    if share_name.is_empty() {
        return Err(invalid_lib_smb_path_error(file_path_str));
    }
    let path_on_share = if parts.len() > 1 {
        parts[1].to_string()
    } else {
        String::new()
    };
    Ok((share_name.to_string(), path_on_share))
}

fn invalid_lib_smb_path_error(file_path_str: &str) -> String {
    format!(
        "Неверный lib:// путь: '{}'. Используйте lib://share_name/path, например lib://data/sample.csv",
        file_path_str
    )
}

pub fn read_bytes_from_path(path: &PathBuf) -> Result<Vec<u8>, String> {
    let file_path_str = path.to_string_lossy().to_string();
    if file_path_str.starts_with("lib://") {
        return read_smb_bytes(&file_path_str);
    }

    if let Some(vfs) = crate::dcp::get_dcp_vfs() {
        let key = crate::dcp::normalize_vfs_path(path).map_err(|e| e.to_string())?;
        if key.is_empty() {
            return Err("Path is a directory, not a file".to_string());
        }
        if let Some(bytes) = vfs.get(&key) {
            return Ok(bytes.to_vec());
        }
        // A file written earlier in this session (session folder).
        if let Some(real) = crate::websocket::session_ve::resolve_read_path(Path::new(&key)) {
            return std::fs::read(&real).map_err(|e| e.to_string());
        }
        return Err(format!(
            "File does not exist: {}",
            crate::dcp::format_logical_path(&key)
        ));
    }

    if crate::websocket::client_sandbox_active() {
        return match crate::websocket::session_ve::resolve_read_path(path) {
            Some(real) => std::fs::read(&real).map_err(|e| e.to_string()),
            None => Err(format!(
                "File does not exist: ./{}",
                path.to_string_lossy().trim_start_matches("./")
            )),
        };
    }

    let resolved = resolve_local_path(path)?;
    if !resolved.exists() {
        return Err(format!(
            "File does not exist: {}",
            format_path_for_error(&resolved)
        ));
    }
    if !resolved.is_file() {
        return Err(format!(
            "Path is not a file: {}",
            format_path_for_error(&resolved)
        ));
    }
    std::fs::read(&resolved).map_err(|e| {
        format!(
            "Error reading file {}: {}",
            format_path_for_error(&resolved),
            e
        )
    })
}

/// Error returned by every write attempt inside a WebSocket session.
pub const SESSION_WRITE_DENIED: &str =
    "Write not allowed in DCP WebSocket session: files cannot be created on the server";

/// Single gate for creating or writing files. Inside a WebSocket session writes
/// are denied regardless of the permission policy, unless ws_app.dc enabled
/// `allow_write` — then they land in the session folder (see [`resolve_write_path`]).
pub fn ensure_write_allowed() -> Result<(), String> {
    if crate::websocket::client_sandbox_active() && !crate::websocket::app::app_config().allow_write {
        Err(SESSION_WRITE_DENIED.to_string())
    } else {
        Ok(())
    }
}

/// Real destination of a write. Outside a session: the usual resolution.
/// In a session: denied by default, or the session folder when enabled.
pub fn resolve_write_path(path: &Path) -> Result<PathBuf, String> {
    if !crate::websocket::client_sandbox_active() {
        return resolve_local_path(&path.to_path_buf());
    }
    ensure_write_allowed()?;
    crate::websocket::session_ve::resolve_write_path(path)
}

/// Call after writing to a path from [`resolve_write_path`] (session quota).
pub fn finish_write(written: &Path) -> Result<(), String> {
    if crate::websocket::client_sandbox_active() {
        crate::websocket::session_ve::enforce_quota_after_write(written)
    } else {
        Ok(())
    }
}

/// Path to return to client code for a written file (`./out.csv` in a session).
pub fn display_written_path(real: &Path) -> String {
    if crate::websocket::client_sandbox_active() {
        crate::websocket::session_ve::display_path(real)
    } else {
        real.to_string_lossy().into_owned()
    }
}

pub fn write_bytes_to_path(path: &PathBuf, data: &[u8]) -> Result<PathBuf, String> {
    let file_path_str = path.to_string_lossy().to_string();
    if file_path_str.starts_with("lib://") {
        return Err(format!(
            "Cannot write to SMB path via save(): {}",
            file_path_str
        ));
    }
    let resolved = resolve_write_path(path)?;
    if let Some(parent) = resolved.parent() {
        if !parent.as_os_str().is_empty() && parent != Path::new(".") {
            if !parent.exists() {
                return Err(format!(
                    "Directory does not exist: {}",
                    parent.to_string_lossy()
                ));
            }
            if !parent.is_dir() {
                return Err(format!(
                    "Parent path is not a directory: {}",
                    parent.to_string_lossy()
                ));
            }
        }
    }
    std::fs::write(&resolved, data).map_err(|e| {
        format!(
            "Error writing file {}: {}",
            format_path_for_error(&resolved),
            e
        )
    })?;
    finish_write(&resolved)?;
    Ok(resolved)
}

fn read_smb_bytes(file_path_str: &str) -> Result<Vec<u8>, String> {
    let (share_name, file_path_on_share) = parse_lib_smb_path(file_path_str)?;
    let smb_manager = crate::vm::file_ops::get_smb_manager()
        .ok_or_else(|| "SMB manager not available".to_string())?;
    let guard = smb_manager.lock().unwrap();
    guard
        .read_file(&share_name, &file_path_on_share)
        .map_err(|e| format!("SMB read error ({}): {}", file_path_str, e))
}

#[cfg(test)]
mod tests {
    use super::parse_lib_smb_path;

    #[test]
    fn parse_lib_smb_path_standard() {
        let (share, path) = parse_lib_smb_path("lib://data/sample.csv").unwrap();
        assert_eq!(share, "data");
        assert_eq!(path, "sample.csv");
    }

    #[test]
    fn parse_lib_smb_path_leading_slash_normalized() {
        let (share, path) = parse_lib_smb_path("lib:///data/sample.csv").unwrap();
        assert_eq!(share, "data");
        assert_eq!(path, "sample.csv");
    }

    #[test]
    fn parse_lib_smb_path_share_only() {
        let (share, path) = parse_lib_smb_path("lib://data").unwrap();
        assert_eq!(share, "data");
        assert_eq!(path, "");
    }

    #[test]
    fn parse_lib_smb_path_empty_fails() {
        assert!(parse_lib_smb_path("lib://").is_err());
    }

    #[test]
    fn parse_lib_smb_path_nested_path() {
        let (share, path) = parse_lib_smb_path("lib://Stream/my_dir/file.csv").unwrap();
        assert_eq!(share, "Stream");
        assert_eq!(path, "my_dir/file.csv");
    }
}
