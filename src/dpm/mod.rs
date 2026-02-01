//! DPM: DataCode Package Manager — virtual env from dpm.toml, cache or in-project.

pub mod config;
pub mod env;
pub mod install;
pub mod lock;
pub mod manifest;

pub use config::{virtualenvs_in_project, set_virtualenvs_in_project, config_file_path};
pub use env::{env_root, package_paths, packages_dir};
pub use lock::{load_lock, write_lock, lock_file_name, DpmLock, LockPackage};
pub use manifest::{find_project_root, load_manifest, project_name_for_env, datacode_version_satisfies, DpmManifest};
pub use install::install_package;

use std::path::Path;

/// Resolve project root from file path, load manifest and lock, return env root and package paths.
/// Returns None if no dpm.toml found. Errors only on invalid manifest/lock.
pub fn resolve_env_and_packages(start_path: &Path) -> Result<Option<(std::path::PathBuf, Vec<std::path::PathBuf>)>, String> {
    let project_root = match manifest::find_project_root(start_path) {
        Some(r) => r,
        None => return Ok(None),
    };
    let manifest = load_manifest(&project_root)?;
    let lock_file = lock::lock_file_name(&manifest);
    let lock = load_lock(&project_root, lock_file)?;
    let env_root_path = match env_root(&project_root, &manifest) {
        Some(p) => p,
        None => return Ok(None),
    };
    let paths = package_paths(&env_root_path, &manifest, &lock);
    Ok(Some((env_root_path, paths)))
}
