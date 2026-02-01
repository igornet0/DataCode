//! Install packages (git clone) into env root

use std::path::Path;
use std::process::Command;

/// Clone a git repo into dest_dir. Supports source like "git+https://github.com/user/repo.git".
pub fn install_package(_name: &str, source: &str, dest_dir: &Path) -> Result<String, String> {
    let url = source
        .strip_prefix("git+")
        .unwrap_or(source)
        .trim();
    if url.is_empty() {
        return Err("Source must be git+<url>".to_string());
    }
    if dest_dir.exists() {
        std::fs::remove_dir_all(dest_dir).map_err(|e| format!("Remove {}: {}", dest_dir.display(), e))?;
    }
    if let Some(parent) = dest_dir.parent() {
        std::fs::create_dir_all(parent).map_err(|e| format!("Create {}: {}", parent.display(), e))?;
    }
    std::fs::create_dir_all(dest_dir).map_err(|e| format!("Create {}: {}", dest_dir.display(), e))?;
    let status = Command::new("git")
        .args(["clone", "--depth", "1", url, "."])
        .current_dir(dest_dir)
        .status()
        .map_err(|e| format!("Run git clone: {}", e))?;
    if !status.success() {
        return Err("git clone failed".to_string());
    }
    let revision = get_head_revision(dest_dir).unwrap_or_else(|| "unknown".to_string());
    Ok(revision)
}

fn get_head_revision(dir: &Path) -> Option<String> {
    let out = Command::new("git")
        .args(["rev-parse", "HEAD"])
        .current_dir(dir)
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    Some(String::from_utf8_lossy(&out.stdout).trim().to_string())
}
