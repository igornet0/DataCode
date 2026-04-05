//! Parse dpm.toml manifest

use std::path::{Path, PathBuf};

#[derive(Debug, Default, serde::Deserialize)]
pub struct DpmManifest {
    pub project: Option<ProjectSection>,
    #[serde(default)]
    pub dependencies: std::collections::HashMap<String, String>,
    pub lock: Option<LockSection>,
    /// Tooling: custom base dir for `<project>-<hash>/` envs (see `env_base_value_for_storage`).
    #[serde(default)]
    pub dpm: Option<DpmSection>,
}

#[derive(Debug, Default, serde::Deserialize)]
pub struct DpmSection {
    /// Relative to project root (e.g. `.`) or absolute path.
    #[serde(default)]
    pub env_base: Option<String>,
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

/// Portable `env_base` value for `dpm.toml`: `.` when base equals project root, else relative path when under project, else absolute.
pub fn env_base_value_for_storage(project_root: &Path, abs_base: &Path) -> String {
    let proj = project_root.canonicalize().unwrap_or_else(|_| project_root.to_path_buf());
    let base = abs_base.canonicalize().unwrap_or_else(|_| abs_base.to_path_buf());
    if base == proj {
        return ".".to_string();
    }
    if let Ok(rel) = base.strip_prefix(&proj) {
        let s = rel.to_string_lossy().replace('\\', "/");
        let t = s.trim_matches('/').trim();
        if t.is_empty() {
            ".".to_string()
        } else {
            t.to_string()
        }
    } else {
        base.to_string_lossy().to_string()
    }
}

fn toml_escape_line_value(s: &str) -> String {
    format!(
        "\"{}\"",
        s.replace('\\', "\\\\").replace('"', "\\\"")
    )
}

/// Merge or update `[dpm]` / `env_base` in `dpm.toml` without dropping other sections.
pub fn set_manifest_env_base(project_root: &Path, abs_base: &Path) -> Result<(), String> {
    let path = project_root.join("dpm.toml");
    let s = std::fs::read_to_string(&path).map_err(|e| e.to_string())?;
    let value = env_base_value_for_storage(project_root, abs_base);
    let escaped = toml_escape_line_value(&value);
    let env_line = format!("env_base = {}", escaped);

    let lines: Vec<String> = s.lines().map(|l| l.to_string()).collect();
    let mut in_dpm = false;
    let mut dpm_start: Option<usize> = None;
    let mut env_base_idx: Option<usize> = None;

    for (i, line) in lines.iter().enumerate() {
        let t = line.trim();
        if t == "[dpm]" {
            in_dpm = true;
            dpm_start = Some(i);
            continue;
        }
        if in_dpm && t.starts_with('[') && t != "[dpm]" {
            break;
        }
        if in_dpm && t.starts_with("env_base") {
            env_base_idx = Some(i);
            break;
        }
    }

    let mut out = lines;
    if let Some(idx) = env_base_idx {
        out[idx] = env_line;
    } else if let Some(start) = dpm_start {
        out.insert(start + 1, env_line);
    } else {
        if !out.is_empty() && !out.last().map(|l| l.trim().is_empty()).unwrap_or(true) {
            out.push(String::new());
        }
        out.push("[dpm]".to_string());
        out.push(env_line);
    }
    std::fs::write(&path, out.join("\n") + "\n").map_err(|e| e.to_string())?;
    Ok(())
}

/// Remove `env_base` from `[dpm]`; remove `[dpm]` if it becomes empty.
pub fn clear_manifest_env_base(project_root: &Path) -> Result<(), String> {
    let path = project_root.join("dpm.toml");
    let s = std::fs::read_to_string(&path).map_err(|e| e.to_string())?;
    let mut lines: Vec<String> = s.lines().map(|l| l.to_string()).collect();
    let mut i = 0;
    let mut in_dpm = false;
    while i < lines.len() {
        let t = lines[i].trim();
        if t == "[dpm]" {
            in_dpm = true;
            i += 1;
            continue;
        }
        if in_dpm && t.starts_with('[') && t != "[dpm]" {
            in_dpm = false;
            i += 1;
            continue;
        }
        if in_dpm && t.starts_with("env_base") {
            lines.remove(i);
            continue;
        }
        i += 1;
    }
    let mut i = 0;
    while i < lines.len() {
        if lines[i].trim() == "[dpm]" {
            let mut j = i + 1;
            while j < lines.len() && lines[j].trim().is_empty() {
                j += 1;
            }
            let next_is_section = j < lines.len() && lines[j].trim().starts_with('[');
            if j >= lines.len() || next_is_section {
                lines.remove(i);
                if i < lines.len() && lines.get(i).map(|l| l.trim().is_empty()).unwrap_or(false) {
                    lines.remove(i);
                }
                continue;
            }
        }
        i += 1;
    }
    std::fs::write(&path, lines.join("\n") + "\n").map_err(|e| e.to_string())?;
    Ok(())
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
