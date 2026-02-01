//! dpm.lock format and read/write

use std::path::Path;

#[derive(Debug, Default, serde::Deserialize, serde::Serialize)]
pub struct DpmLock {
    #[serde(default)]
    pub package: Vec<LockPackage>,
}

#[derive(Debug, serde::Deserialize, serde::Serialize)]
pub struct LockPackage {
    pub name: String,
    pub source: String,
    /// Resolved ref (commit hash or tag) for reproducibility
    pub revision: Option<String>,
}

/// Load dpm.lock from project root (path from manifest [lock] or default "dpm.lock").
pub fn load_lock(project_root: &Path, lock_file_name: &str) -> Result<DpmLock, String> {
    let path = project_root.join(lock_file_name);
    if !path.exists() {
        return Ok(DpmLock::default());
    }
    let s = std::fs::read_to_string(&path).map_err(|e| format!("Read {}: {}", path.display(), e))?;
    toml::from_str(&s).map_err(|e| format!("Parse {}: {}", path.display(), e))
}

/// Write dpm.lock to project root.
pub fn write_lock(project_root: &Path, lock_file_name: &str, lock: &DpmLock) -> Result<(), String> {
    let path = project_root.join(lock_file_name);
    let s = toml::to_string_pretty(lock).map_err(|e| e.to_string())?;
    std::fs::write(&path, s).map_err(|e| e.to_string())?;
    Ok(())
}

/// Lock file name from manifest (default "dpm.lock").
pub fn lock_file_name(manifest: &super::manifest::DpmManifest) -> &str {
    manifest
        .lock
        .as_ref()
        .map(|l| l.file.as_str())
        .unwrap_or("dpm.lock")
}
