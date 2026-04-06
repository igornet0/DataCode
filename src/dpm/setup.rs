//! Optional `setup.dcmodule` in a cloned package: declarative build steps + optional shell hooks.
//!
//! The file is **JSON** (not a zip). If missing, DPM skips setup silently.
//! Disable auto-run with `DPM_SETUP_AUTO=0`.

use std::path::{Path, PathBuf};
use std::process::Command;

/// Env var: set to `0` to skip running setup after `dpm add` / `dpm init`.
pub const ENV_DPM_SETUP_AUTO: &str = "DPM_SETUP_AUTO";

const SETUP_FILENAME: &str = "setup.dcmodule";

#[derive(Debug, serde::Deserialize)]
pub struct SetupDescriptor {
    pub schema_version: u32,
    /// Import name for `import <name>` (informational).
    #[serde(default)]
    pub module_name: Option<String>,
    #[serde(default)]
    pub package_version: Option<String>,
    #[serde(default)]
    pub hooks: Option<Hooks>,
    #[serde(default)]
    pub build: Vec<BuildStep>,
    #[serde(default)]
    pub install: Vec<InstallRule>,
}

#[derive(Debug, Default, serde::Deserialize)]
pub struct Hooks {
    #[serde(default)]
    pub pre_build: Option<String>,
    #[serde(default)]
    pub post_build: Option<String>,
}

#[derive(Debug, serde::Deserialize)]
pub struct BuildStep {
    #[serde(default)]
    pub when: Option<When>,
    #[serde(default)]
    pub cwd: Option<String>,
    /// Shell one-liner (same as `sh -c` / `cmd /C`).
    pub command: String,
}

#[derive(Debug, Default, serde::Deserialize)]
pub struct When {
    /// If set, step runs only when `cfg!(target_os)` matches one of: `macos`, `linux`, `windows`.
    #[serde(default)]
    pub os: Option<Vec<String>>,
}

#[derive(Debug, serde::Deserialize)]
pub struct InstallRule {
    pub from: String,
    /// Destination path relative to the package root (e.g. `libml.dylib`).
    pub to: String,
    #[serde(default)]
    pub when: Option<When>,
}

fn setup_path(pkg_dir: &Path) -> PathBuf {
    pkg_dir.join(SETUP_FILENAME)
}

pub(crate) fn current_os_tag() -> &'static str {
    if cfg!(target_os = "macos") {
        "macos"
    } else if cfg!(target_os = "linux") {
        "linux"
    } else if cfg!(target_os = "windows") {
        "windows"
    } else {
        "unknown"
    }
}

fn matches_when(when: &Option<When>) -> bool {
    let Some(w) = when else {
        return true;
    };
    let Some(ref list) = w.os else {
        return true;
    };
    if list.is_empty() {
        return true;
    }
    let tag = current_os_tag();
    list.iter().any(|s| s.eq_ignore_ascii_case(tag))
}

fn run_shell(command: &str, cwd: &Path) -> Result<(), String> {
    eprintln!("[dpm setup] cwd={}  {}", cwd.display(), command);
    #[cfg(unix)]
    {
        let status = Command::new("sh")
            .arg("-c")
            .arg(command)
            .current_dir(cwd)
            .status()
            .map_err(|e| format!("spawn sh -c: {}", e))?;
        if !status.success() {
            return Err(format!("command failed with status {:?}", status.code()));
        }
    }
    #[cfg(windows)]
    {
        let status = Command::new("cmd")
            .args(["/C", command])
            .current_dir(cwd)
            .status()
            .map_err(|e| format!("spawn cmd /C: {}", e))?;
        if !status.success() {
            return Err(format!("command failed with status {:?}", status.code()));
        }
    }
    Ok(())
}

/// Load and parse `setup.dcmodule` from `pkg_dir`. Returns `None` if file is missing.
pub fn load_setup_descriptor(pkg_dir: &Path) -> Result<Option<SetupDescriptor>, String> {
    let p = setup_path(pkg_dir);
    if !p.is_file() {
        return Ok(None);
    }
    let raw = std::fs::read_to_string(&p).map_err(|e| format!("Read {}: {}", p.display(), e))?;
    let desc: SetupDescriptor =
        serde_json::from_str(&raw).map_err(|e| format!("Parse {}: {}", p.display(), e))?;
    if desc.schema_version != 1 {
        return Err(format!(
            "Unsupported setup.dcmodule schema_version {} (expected 1)",
            desc.schema_version
        ));
    }
    Ok(Some(desc))
}

/// Run setup hooks + build + install copies into `pkg_dir` (package root under env).
pub fn run_setup(pkg_dir: &Path, desc: &SetupDescriptor) -> Result<(), String> {
    if let Some(ref h) = desc.hooks {
        if let Some(ref cmd) = h.pre_build {
            run_shell(cmd, pkg_dir)?;
        }
    }

    for step in &desc.build {
        if !matches_when(&step.when) {
            continue;
        }
        let cwd = if let Some(ref rel) = step.cwd {
            pkg_dir.join(rel)
        } else {
            pkg_dir.to_path_buf()
        };
        let cwd = cwd.canonicalize().unwrap_or(cwd);
        run_shell(&step.command, &cwd)?;
    }

    for rule in &desc.install {
        if !matches_when(&rule.when) {
            continue;
        }
        let from = pkg_dir.join(&rule.from);
        let to = pkg_dir.join(&rule.to);
        if !from.is_file() {
            return Err(format!(
                "setup install: expected artifact missing: {}",
                from.display()
            ));
        }
        if let Some(parent) = to.parent() {
            std::fs::create_dir_all(parent).map_err(|e| format!("create_dir_all: {}", e))?;
        }
        std::fs::copy(&from, &to)
            .map_err(|e| format!("copy {} -> {}: {}", from.display(), to.display(), e))?;
        eprintln!(
            "[dpm setup] installed {} -> {}",
            from.display(),
            to.display()
        );
    }

    if let Some(ref h) = desc.hooks {
        if let Some(ref cmd) = h.post_build {
            run_shell(cmd, pkg_dir)?;
        }
    }

    Ok(())
}

/// If `setup.dcmodule` exists and `DPM_SETUP_AUTO` is not `0`, run it.
pub fn run_setup_if_present(pkg_dir: &Path) -> Result<(), String> {
    if std::env::var(ENV_DPM_SETUP_AUTO)
        .ok()
        .as_deref()
        .map(|s| s == "0")
        .unwrap_or(false)
    {
        eprintln!("[dpm setup] skipped ({}=0)", ENV_DPM_SETUP_AUTO);
        return Ok(());
    }
    let Some(desc) = load_setup_descriptor(pkg_dir)? else {
        return Ok(());
    };
    eprintln!(
        "[dpm setup] running for {} (module={:?})",
        pkg_dir.display(),
        desc.module_name
    );
    run_setup(pkg_dir, &desc)
}

/// Run `setup.dcmodule` unconditionally (for `dpm setup <pkg>`). Fails if file is missing.
pub fn run_setup_for_package(pkg_dir: &Path) -> Result<(), String> {
    let Some(desc) = load_setup_descriptor(pkg_dir)? else {
        return Err(format!("No {} in {}", SETUP_FILENAME, pkg_dir.display()));
    };
    eprintln!(
        "[dpm setup] (manual) {} (module={:?})",
        pkg_dir.display(),
        desc.module_name
    );
    run_setup(pkg_dir, &desc)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_minimal_setup() {
        let j = r#"{
            "schema_version": 1,
            "module_name": "ml",
            "build": [],
            "install": []
        }"#;
        let d: SetupDescriptor = serde_json::from_str(j).expect("parse");
        assert_eq!(d.schema_version, 1);
        assert_eq!(d.module_name.as_deref(), Some("ml"));
    }

    #[test]
    fn when_matches_current_platform() {
        let w: When =
            serde_json::from_str(&format!(r#"{{"os":["{}"]}}"#, super::current_os_tag())).unwrap();
        assert!(matches_when(&Some(w)));
    }

    #[test]
    fn when_empty_os_list_matches() {
        assert!(matches_when(&Some(When { os: None })));
        assert!(matches_when(&Some(When { os: Some(vec![]) })));
    }
}
