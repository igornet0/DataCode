//! Parse dpm.toml manifest

use std::path::{Path, PathBuf};

#[derive(Debug, Default, serde::Deserialize)]
pub struct DpmManifest {
    pub project: Option<ProjectSection>,
    #[serde(default)]
    pub dependencies: std::collections::HashMap<String, String>,
    pub lock: Option<LockSection>,
}

#[derive(Debug, serde::Deserialize)]
pub struct ProjectSection {
    pub name: String,
    pub version: Option<String>,
    pub datacode: Option<String>,
    pub entry: Option<String>,
}

#[derive(Debug, serde::Deserialize)]
pub struct LockSection {
    #[serde(default = "default_lock_file")]
    pub file: String,
    #[serde(default)]
    pub checksum: bool,
}

fn default_lock_file() -> String {
    "dpm.lock".to_string()
}

/// Find project root by walking up from `start` until dpm.toml is found.
pub fn find_project_root(start: &Path) -> Option<PathBuf> {
    let mut current = start.to_path_buf();
    if current.is_file() {
        current = current.parent()?.to_path_buf();
    }
    loop {
        let manifest_path = current.join("dpm.toml");
        if manifest_path.exists() {
            return Some(current);
        }
        current = current.parent()?.to_path_buf();
    }
}

/// Load and parse dpm.toml from project root.
pub fn load_manifest(project_root: &Path) -> Result<DpmManifest, String> {
    let path = project_root.join("dpm.toml");
    let s = std::fs::read_to_string(&path).map_err(|e| format!("Read {}: {}", path.display(), e))?;
    toml::from_str(&s).map_err(|e| format!("Parse dpm.toml: {}", e))
}

/// Check if current datacode version satisfies required constraint (e.g. ">=2.0.0").
pub fn datacode_version_satisfies(required: &str, current: &str) -> bool {
    let required = required.trim();
    if required.is_empty() {
        return true;
    }
    let (op, ver) = if required.starts_with(">=") {
        (">=", required[2..].trim())
    } else if required.starts_with(">") {
        (">", required[1..].trim())
    } else if required.starts_with("<=") {
        ("<=", required[2..].trim())
    } else if required.starts_with("<") {
        ("<", required[1..].trim())
    } else if required.starts_with("==") {
        ("==", required[2..].trim())
    } else {
        ("", required)
    };
    let parse = |s: &str| -> (u32, u32, u32) {
        let parts: Vec<u32> = s
            .split('.')
            .filter_map(|p| p.trim().parse().ok())
            .collect();
        (
            parts.get(0).copied().unwrap_or(0),
            parts.get(1).copied().unwrap_or(0),
            parts.get(2).copied().unwrap_or(0),
        )
    };
    let (ra, rb, rc) = parse(ver);
    let (ca, cb, cc) = parse(current);
    match op {
        ">=" => (ca, cb, cc) >= (ra, rb, rc),
        ">" => (ca, cb, cc) > (ra, rb, rc),
        "<=" => (ca, cb, cc) <= (ra, rb, rc),
        "<" => (ca, cb, cc) < (ra, rb, rc),
        "==" => (ca, cb, cc) == (ra, rb, rc),
        _ => true,
    }
}

/// Project name for env directory (sanitized).
pub fn project_name_for_env(manifest: &DpmManifest) -> String {
    manifest
        .project
        .as_ref()
        .map(|p| {
            p.name
                .chars()
                .map(|c| if c.is_alphanumeric() || c == '-' || c == '_' { c } else { '_' })
                .collect::<String>()
        })
        .unwrap_or_else(|| "project".to_string())
}
