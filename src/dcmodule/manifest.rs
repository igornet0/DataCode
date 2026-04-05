//! Parsed `manifest.json` for `.dcmodule` archives.

use serde::Deserialize;
use std::collections::HashMap;

use crate::abi::{abi_compatible, AbiVersion, DATACODE_ABI_VERSION};

/// Root manifest inside a `.dcmodule` zip.
#[derive(Debug, Deserialize)]
pub struct DcmoduleManifest {
    pub schema_version: u32,
    pub name: String,
    pub version: String,
    pub abi_version: AbiVersionJson,
    #[serde(default)]
    pub library: Option<String>,
    #[serde(default)]
    pub targets: Option<HashMap<String, String>>,
}

#[derive(Debug, Deserialize)]
pub struct AbiVersionJson {
    pub major: u16,
    pub minor: u16,
}

impl From<&AbiVersionJson> for AbiVersion {
    fn from(v: &AbiVersionJson) -> Self {
        AbiVersion {
            major: v.major,
            minor: v.minor,
        }
    }
}

/// Rust-style host triple for `targets` lookup.
pub fn host_target_key() -> &'static str {
    if cfg!(all(target_os = "macos", target_arch = "aarch64")) {
        "aarch64-apple-darwin"
    } else if cfg!(all(target_os = "macos", target_arch = "x86_64")) {
        "x86_64-apple-darwin"
    } else if cfg!(all(target_os = "linux", target_arch = "x86_64")) {
        "x86_64-unknown-linux-gnu"
    } else if cfg!(all(target_os = "linux", target_arch = "aarch64")) {
        "aarch64-unknown-linux-gnu"
    } else if cfg!(all(target_os = "windows", target_arch = "x86_64")) {
        "x86_64-pc-windows-msvc"
    } else if cfg!(all(target_os = "windows", target_arch = "aarch64")) {
        "aarch64-pc-windows-msvc"
    } else {
        "unknown"
    }
}

impl DcmoduleManifest {
    /// Validates `abi_version` against the running VM.
    pub fn check_abi(&self) -> Result<(), String> {
        let m: AbiVersion = (&self.abi_version).into();
        if !abi_compatible(&m, &DATACODE_ABI_VERSION) {
            return Err(format!(
                "manifest abi_version {}.{} is not compatible with VM {}.{}",
                m.major,
                m.minor,
                DATACODE_ABI_VERSION.major,
                DATACODE_ABI_VERSION.minor
            ));
        }
        Ok(())
    }

    /// Relative path to the native library for this host (forward slashes in JSON).
    pub fn library_relative_path(&self) -> Result<String, String> {
        let key = host_target_key();
        if let Some(ref t) = self.targets {
            if let Some(p) = t.get(key) {
                return Ok(p.clone());
            }
        }
        if let Some(ref lib) = self.library {
            return Ok(lib.clone());
        }
        if key == "unknown" {
            return Err(
                "manifest has no library and no targets entry for this platform".to_string(),
            );
        }
        Err(format!(
            "manifest has no library and no targets entry for {}",
            key
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn host_target_key_is_non_empty() {
        let k = host_target_key();
        assert!(!k.is_empty());
    }
}
