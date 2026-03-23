//! Conventional paths for `.dcmodule` artifacts next to DPM-installed packages.

use std::path::{Path, PathBuf};

use super::env::packages_dir;

/// `<packages_dir>/<package_name>/<package_name>.dcmodule` (same layout as `lib<name>.dylib` next to the package root).
pub fn path_in_packages_directory(packages_dir: &Path, package_name: &str) -> PathBuf {
    packages_dir
        .join(package_name)
        .join(format!("{}.dcmodule", package_name))
}

/// Expected location of a native module bundle after `dpm add` / `dpm init`:
/// `<env_root>/packages/<package_name>/<package_name>.dcmodule`.
///
/// Registry or install flows may place a downloaded `.dcmodule` here so the VM
/// resolves `import <name>` the same way as a loose dylib under the package root.
pub fn expected_dcmodule_path(env_root: &Path, package_name: &str) -> PathBuf {
    path_in_packages_directory(&packages_dir(env_root), package_name)
}
