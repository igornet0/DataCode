//! Compute env root (cache or in-project) and package paths

use std::path::{Path, PathBuf};

use super::config;
use super::manifest;

/// If set (e.g. by `dpm --env-path` or the user shell), per-project env dirs are created under
/// this directory as `<project_name>-<path_hash>/` instead of the default cache `.../datacode/dpm/envs/`.
pub const ENV_DPM_ENV_BASE: &str = "DPM_ENV_BASE";

fn env_base_from_env() -> Option<PathBuf> {
    let s = std::env::var(ENV_DPM_ENV_BASE).ok()?;
    let t = s.trim();
    if t.is_empty() {
        return None;
    }
    Some(PathBuf::from(t))
}

/// `[dpm] env_base` from manifest: absolute path or relative to `project_root`.
pub fn env_base_from_manifest(project_root: &Path, manifest: &manifest::DpmManifest) -> Option<PathBuf> {
    let s = manifest.dpm.as_ref()?.env_base.as_ref()?.trim();
    if s.is_empty() {
        return None;
    }
    let p = Path::new(s);
    if p.is_absolute() {
        Some(p.to_path_buf())
    } else {
        Some(project_root.join(p))
    }
}

/// Short hash of path for unique env directory name (8 hex chars).
fn path_hash(path: &Path) -> String {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut h = DefaultHasher::new();
    path.hash(&mut h);
    format!("{:016x}", h.finish())[..8].to_string()
}

/// Cache base directory for DPM envs.
/// Linux ~/.cache/datacode/dpm/envs
/// macOS: ~/Library/Caches/datacode/dpm/envs
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
/// - Cache: ~/.cache/datacode/dpm/envs/<project_name>-<hash>/ (Linux)
///          or ~/Library/Caches/datacode/dpm/envs/<project_name>-<hash>/ (macOS)
///          or %APPDATA%\datacode\Cache\dpm\envs\<project_name>-<hash>\ (Windows)
/// - Override: [`ENV_DPM_ENV_BASE`] (highest), then `[dpm] env_base` in manifest, then cache.
pub fn env_root(project_root: &Path, manifest: &manifest::DpmManifest) -> Option<PathBuf> {
    if config::virtualenvs_in_project() {
        return Some(project_root.join(".dpm"));
    }
    let base = env_base_from_env()
        .or_else(|| env_base_from_manifest(project_root, manifest))
        .or_else(cache_envs_base)?;
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    /// `DPM_ENV_BASE` is process-global; serialize tests that set it.
    static ENV_TEST_LOCK: Mutex<()> = Mutex::new(());

    fn sample_manifest() -> manifest::DpmManifest {
        manifest::DpmManifest {
            project: Some(manifest::ProjectSection {
                name: "p".to_string(),
                version: None,
                datacode: None,
                entry: None,
            }),
            ..Default::default()
        }
    }

    #[test]
    fn env_root_respects_dpm_env_base() {
        let _guard = ENV_TEST_LOCK.lock().expect("env test lock");
        let tmp = std::env::temp_dir().join("dpm_env_test_base");
        std::env::set_var(ENV_DPM_ENV_BASE, &tmp);
        let manifest = sample_manifest();
        let project = Path::new("/tmp/phony_project_root_for_hash");
        let root = env_root(project, &manifest).expect("env root");
        assert!(root.starts_with(&tmp));
        assert!(root.to_string_lossy().contains("-"));
        std::env::remove_var(ENV_DPM_ENV_BASE);
    }

    #[test]
    fn env_root_uses_manifest_env_base_when_env_unset() {
        let _guard = ENV_TEST_LOCK.lock().expect("env test lock");
        let _ = std::env::remove_var(ENV_DPM_ENV_BASE);
        let project_root = std::env::temp_dir().join("dpm_test_manifest_env_base");
        let _ = std::fs::create_dir_all(&project_root);
        let mut manifest = sample_manifest();
        manifest.dpm = Some(manifest::DpmSection {
            env_base: Some(".".to_string()),
            ..Default::default()
        });
        let root = env_root(&project_root, &manifest).expect("env root");
        assert!(root.starts_with(&project_root));
        assert!(root.to_string_lossy().contains("-"));
        let _ = std::fs::remove_dir_all(&project_root);
    }

    #[test]
    fn env_root_env_overrides_manifest() {
        let _guard = ENV_TEST_LOCK.lock().expect("env test lock");
        let tmp = std::env::temp_dir().join("dpm_env_override_test");
        let _ = std::fs::create_dir_all(&tmp);
        std::env::set_var(ENV_DPM_ENV_BASE, &tmp);
        let project_root = std::env::temp_dir().join("dpm_test_override_proj");
        let _ = std::fs::create_dir_all(&project_root);
        let mut manifest = sample_manifest();
        manifest.dpm = Some(manifest::DpmSection {
            env_base: Some(".".to_string()),
            ..Default::default()
        });
        let root = env_root(&project_root, &manifest).expect("env root");
        assert!(root.starts_with(&tmp));
        std::env::remove_var(ENV_DPM_ENV_BASE);
        let _ = std::fs::remove_dir_all(&tmp);
        let _ = std::fs::remove_dir_all(&project_root);
    }
}
