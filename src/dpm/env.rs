//! Compute env root (cache or in-project) and package paths

use std::path::{Path, PathBuf};

use super::config;
use super::manifest;

/// Short hash of path for unique env directory name (8 hex chars).
fn path_hash(path: &Path) -> String {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut h = DefaultHasher::new();
    path.hash(&mut h);
    format!("{:016x}", h.finish())[..8].to_string()
}

/// Cache base directory for DPM envs.
/// Linux/macOS: ~/.cache/datacode/dpm/envs
/// Windows: %APPDATA%\datacode\Cache\dpm\envs
fn cache_envs_base() -> Option<PathBuf> {
    #[cfg(target_os = "windows")]
    let base = dirs::data_local_dir(); // e.g. C:\Users\...\AppData\Local

    #[cfg(not(target_os = "windows"))]
    let base = dirs::cache_dir(); // ~/.cache

    base.map(|p| p.join("datacode").join("dpm").join("envs"))
}

/// Env root for project: either cache or in-project.
/// - In-project: <project_root>/.dpm/
/// - Cache: ~/.cache/datacode/dpm/envs/<project_name>-<hash>/ (Linux/macOS)
///          or %APPDATA%\datacode\Cache\dpm\envs\<project_name>-<hash>\ (Windows)
pub fn env_root(project_root: &Path, manifest: &manifest::DpmManifest) -> Option<PathBuf> {
    if config::virtualenvs_in_project() {
        return Some(project_root.join(".dpm"));
    }
    let base = cache_envs_base()?;
    let name = manifest::project_name_for_env(manifest);
    let hash = path_hash(project_root);
    let dir_name = format!("{}-{}", name, hash);
    Some(base.join(dir_name))
}

/// Path to packages directory inside env root: <env_root>/packages/
pub fn packages_dir(env_root: &Path) -> PathBuf {
    env_root.join("packages")
}

/// Single search path for import resolution: <env_root>/packages/.
/// Import "foo" will resolve to packages/foo.dc or packages/foo/__lib__.dc.
pub fn package_paths(env_root: &Path, _manifest: &manifest::DpmManifest, _lock: &super::lock::DpmLock) -> Vec<PathBuf> {
    let packages = env_root.join("packages");
    if packages.exists() {
        vec![packages]
    } else {
        Vec::new()
    }
}
