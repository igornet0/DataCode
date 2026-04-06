//! DPM: DataCode Package Manager — virtual env from dpm.toml, cache or in-project.

pub mod adapters_lib;
pub mod add_database;
pub mod config;
pub mod dcmodule;
pub mod env;
pub mod init_database;
pub mod init_wizard;
pub mod install;
pub mod lock;
pub mod manifest;
pub mod registry;
pub mod setup;

pub use add_database::run_add_database;
pub use config::{config_file_path, set_virtualenvs_in_project, virtualenvs_in_project};
pub use dcmodule::{expected_dcmodule_path, path_in_packages_directory};
pub use env::{env_base_from_manifest, env_root, package_paths, packages_dir, ENV_DPM_ENV_BASE};
pub use init_database::run_init_database;
pub use init_wizard::run_init_wizard;
pub use install::install_package;
pub use lock::{load_lock, lock_file_name, write_lock, DpmLock, LockPackage};
pub use manifest::{
    clear_manifest_env_base, datacode_version_satisfies, env_base_value_for_storage,
    find_project_root, load_manifest, project_name_for_env, set_manifest_env_base, DpmManifest,
};
pub use registry::{
    fetch_registry_index, registry_cache_path, registry_index_url, resolve_package_source,
    resolve_registry_package, RegistryIndex, RegistryPackage,
};
pub use setup::{run_setup_for_package, run_setup_if_present, ENV_DPM_SETUP_AUTO};

use std::path::Path;

/// Resolve project root from file path, load manifest and lock, return env root and package paths.
/// Returns None if no dpm.toml found. Errors only on invalid manifest/lock.
pub fn resolve_env_and_packages(
    start_path: &Path,
) -> Result<Option<(std::path::PathBuf, Vec<std::path::PathBuf>)>, String> {
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
