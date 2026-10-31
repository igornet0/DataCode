//! Install packages (git clone) into env root.
//!
//! Native bundles: a registry or manual step may place `<packages>/<name>/<name>.dcmodule`
//! (see [`crate::dpm::expected_dcmodule_path`]) next to the clone; the VM loads it like a loose dylib.

use sha2::{Digest, Sha256};
use std::io::Read;
use std::path::Path;
use std::process::Command;

/// Set to `1` to ignore registry artifacts and always build from source via `setup.dcmodule`.
pub const ENV_DPM_FROM_SOURCE: &str = "DPM_FROM_SOURCE";

pub fn prefer_source_build() -> bool {
    std::env::var(ENV_DPM_FROM_SOURCE).ok().as_deref() == Some("1")
}

/// Download a prebuilt `.dcmodule` to `dest_file`, verifying its sha256 (hex, case-insensitive).
/// The file is written only after the checksum matches.
pub fn install_artifact(url: &str, sha256: &str, dest_file: &Path) -> Result<(), String> {
    let resp = ureq::get(url)
        .call()
        .map_err(|e| format!("Download {}: {}", url, e))?;
    let mut bytes = Vec::new();
    resp.into_reader()
        .read_to_end(&mut bytes)
        .map_err(|e| format!("Download {}: {}", url, e))?;
    let actual = sha256_hex(&bytes);
    if !actual.eq_ignore_ascii_case(sha256.trim()) {
        return Err(format!(
            "Checksum mismatch for {}: expected {}, got {}",
            url, sha256, actual
        ));
    }
    if let Some(parent) = dest_file.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("Create {}: {}", parent.display(), e))?;
    }
    std::fs::write(dest_file, &bytes).map_err(|e| format!("Write {}: {}", dest_file.display(), e))
}

pub fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{:02x}", b))
        .collect()
}

/// Clone a git repo into dest_dir. Supports source like "git+https://github.com/user/repo.git".
pub fn install_package(_name: &str, source: &str, dest_dir: &Path) -> Result<String, String> {
    let url = source.strip_prefix("git+").unwrap_or(source).trim();
    if url.is_empty() {
        return Err("Source must be git+<url>".to_string());
    }
    if dest_dir.exists() {
        std::fs::remove_dir_all(dest_dir)
            .map_err(|e| format!("Remove {}: {}", dest_dir.display(), e))?;
    }
    if let Some(parent) = dest_dir.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("Create {}: {}", parent.display(), e))?;
    }
    std::fs::create_dir_all(dest_dir)
        .map_err(|e| format!("Create {}: {}", dest_dir.display(), e))?;
    let status = Command::new("git")
        .args(["clone", "--depth", "1", url, "."])
        .current_dir(dest_dir)
        .status()
        .map_err(|e| format!("Run git clone: {}", e))?;
    if !status.success() {
        return Err("git clone failed".to_string());
    }
    let revision = get_head_revision(dest_dir).unwrap_or_else(|| "unknown".to_string());
    Ok(revision)
}

fn get_head_revision(dir: &Path) -> Option<String> {
    let out = Command::new("git")
        .args(["rev-parse", "HEAD"])
        .current_dir(dir)
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    Some(String::from_utf8_lossy(&out.stdout).trim().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha256_hex_matches_known_digest() {
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }
}
