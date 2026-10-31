//! Fetch and parse the public package registry index (JSON).

use serde::Deserialize;
use std::path::PathBuf;

const DEFAULT_REGISTRY_URL: &str =
    "https://raw.githubusercontent.com/igornet0/Datacode-registry-index/main/config.json";

/// On-disk cache when network fails: `<cache_dir>/datacode/registry/config.json`
pub fn registry_cache_path() -> Option<PathBuf> {
    dirs::cache_dir().map(|p| p.join("datacode").join("registry").join("config.json"))
}

#[derive(Debug, Deserialize)]
pub struct RegistryIndex {
    #[serde(default)]
    pub version: u32,
    #[serde(default)]
    pub packages: Vec<RegistryPackage>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RegistryPackage {
    pub name: String,
    pub source: String,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub min_datacode: Option<String>,
    /// Prebuilt `.dcmodule` bundles per target triple; installed instead of building from source.
    #[serde(default)]
    pub artifacts: Vec<RegistryArtifact>,
}

/// Prebuilt native bundle for one target (`aarch64-apple-darwin`, `x86_64-unknown-linux-gnu`, ...).
#[derive(Debug, Clone, Deserialize)]
pub struct RegistryArtifact {
    pub target: String,
    pub url: String,
    pub sha256: String,
    #[serde(default)]
    pub version: Option<String>,
}

/// Target triple of this dpm binary, in the form used by registry `artifacts[].target`.
pub fn current_target() -> String {
    let os = match std::env::consts::OS {
        "macos" => "apple-darwin",
        "linux" => "unknown-linux-gnu",
        "windows" => "pc-windows-msvc",
        other => other,
    };
    format!("{}-{}", std::env::consts::ARCH, os)
}

impl RegistryPackage {
    /// Artifact matching [`current_target`], if the registry publishes one.
    pub fn artifact_for_current_target(&self) -> Option<&RegistryArtifact> {
        let target = current_target();
        self.artifacts.iter().find(|a| a.target == target)
    }
}

/// URL of `config.json` (GET). Override with `DATACODE_REGISTRY_URL`.
pub fn registry_index_url() -> String {
    std::env::var("DATACODE_REGISTRY_URL").unwrap_or_else(|_| DEFAULT_REGISTRY_URL.to_string())
}

/// Download and parse the registry index. On success, writes a copy under [`registry_cache_path`].
/// On network failure, falls back to the last cached file if present.
pub fn fetch_registry_index() -> Result<RegistryIndex, String> {
    let url = registry_index_url();
    match ureq::get(&url).call() {
        Ok(resp) => {
            let body = resp
                .into_string()
                .map_err(|e| format!("Registry body: {}", e))?;
            let idx: RegistryIndex =
                serde_json::from_str(&body).map_err(|e| format!("Registry JSON: {}", e))?;
            if let Some(p) = registry_cache_path() {
                if let Some(parent) = p.parent() {
                    let _ = std::fs::create_dir_all(parent);
                }
                let _ = std::fs::write(&p, &body);
            }
            Ok(idx)
        }
        Err(e) => {
            if let Some(p) = registry_cache_path() {
                if p.exists() {
                    let body = std::fs::read_to_string(&p).map_err(|io| io.to_string())?;
                    eprintln!(
                        "Warning: registry fetch failed ({}), using cached index from {}",
                        e,
                        p.display()
                    );
                    return serde_json::from_str(&body)
                        .map_err(|err| format!("Cached registry JSON: {}", err));
                }
            }
            Err(format!("Registry request {}: {}", url, e))
        }
    }
}

/// Full registry entry for `name`, or error if missing.
pub fn resolve_registry_package(name: &str) -> Result<RegistryPackage, String> {
    let idx = fetch_registry_index()?;
    idx.packages
        .into_iter()
        .find(|p| p.name == name)
        .ok_or_else(|| {
            format!(
                "Package '{}' not found in registry (set DATACODE_REGISTRY_URL or use dpm add {} git+<url>)",
                name, name
            )
        })
}

/// Resolve `git+...` source for a package name from the registry.
pub fn resolve_package_source(name: &str) -> Result<String, String> {
    resolve_registry_package(name).map(|p| p.source)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_cache_path_ends_with_config_json() {
        let p = registry_cache_path().expect("cache dir");
        assert!(p.to_string_lossy().contains("datacode"));
        assert!(p.ends_with("config.json"));
    }

    #[test]
    fn artifacts_are_optional_and_matched_by_target() {
        let json = format!(
            r#"{{"version":1,"packages":[
                {{"name":"a","source":"git+https://x/a.git"}},
                {{"name":"b","source":"git+https://x/b.git","artifacts":[
                    {{"target":"none-none","url":"https://x/n","sha256":"00"}},
                    {{"target":"{}","url":"https://x/b.dcmodule","sha256":"ab","version":"0.1.0"}}
                ]}}
            ]}}"#,
            current_target()
        );
        let idx: RegistryIndex = serde_json::from_str(&json).unwrap();
        assert!(idx.packages[0].artifact_for_current_target().is_none());
        let art = idx.packages[1].artifact_for_current_target().unwrap();
        assert_eq!(art.url, "https://x/b.dcmodule");
        assert_eq!(art.version.as_deref(), Some("0.1.0"));
    }
}
