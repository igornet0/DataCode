//! Read `.dcmodule` (zip), validate manifest, extract to cache, return dylib path.

use std::fs;
use std::io::{Cursor, Read};
use std::path::{Component, Path, PathBuf};

use sha2::{Digest, Sha256};
use zip::read::ZipArchive;

use super::manifest::DcmoduleManifest;

fn hex_sha256(bytes: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(bytes);
    h.finalize().iter().map(|b| format!("{:02x}", b)).collect()
}

/// Read and parse `manifest.json` from zip bytes without extracting.
pub fn manifest_from_zip_bytes(bytes: &[u8]) -> Result<DcmoduleManifest, String> {
    let cursor = Cursor::new(bytes);
    let mut archive = ZipArchive::new(cursor).map_err(|e| format!("invalid zip: {}", e))?;
    let mut f = archive
        .by_name("manifest.json")
        .map_err(|_| "zip missing manifest.json at archive root".to_string())?;
    let mut s = String::new();
    f.read_to_string(&mut s)
        .map_err(|e| format!("read manifest.json: {}", e))?;
    let m: DcmoduleManifest =
        serde_json::from_str(&s).map_err(|e| format!("manifest.json: {}", e))?;
    if m.schema_version != 1 {
        return Err(format!(
            "unsupported manifest schema_version {} (expected 1)",
            m.schema_version
        ));
    }
    Ok(m)
}

fn safe_relative_path(name: &str) -> Option<PathBuf> {
    let p = Path::new(name);
    let mut out = PathBuf::new();
    for c in p.components() {
        match c {
            Component::ParentDir => return None,
            Component::Normal(s) => out.push(s),
            Component::CurDir => {}
            Component::RootDir | Component::Prefix(_) => return None,
        }
    }
    if out.as_os_str().is_empty() {
        None
    } else {
        Some(out)
    }
}

/// Extract all entries under `out_dir`, rejecting paths that escape `out_dir` (zip slip).
pub fn extract_zip_bytes(bytes: &[u8], out_dir: &Path) -> Result<(), String> {
    fs::create_dir_all(out_dir).map_err(|e| e.to_string())?;
    let root = fs::canonicalize(out_dir).map_err(|e| e.to_string())?;
    let cursor = Cursor::new(bytes);
    let mut archive = ZipArchive::new(cursor).map_err(|e| format!("invalid zip: {}", e))?;
    for i in 0..archive.len() {
        let mut file = archive
            .by_index(i)
            .map_err(|e| format!("zip entry {}: {}", i, e))?;
        let name = file.name();
        if name.is_empty() {
            continue;
        }
        let rel = safe_relative_path(name).ok_or_else(|| format!("unsafe zip path: {}", name))?;
        let out_path = root.join(&rel);
        if !out_path.starts_with(&root) {
            return Err(format!("unsafe zip path: {}", name));
        }
        if name.ends_with('/') || file.is_dir() {
            fs::create_dir_all(&out_path).map_err(|e| e.to_string())?;
            continue;
        }
        if let Some(p) = out_path.parent() {
            fs::create_dir_all(p).map_err(|e| e.to_string())?;
        }
        let mut out = fs::File::create(&out_path).map_err(|e| e.to_string())?;
        std::io::copy(&mut file, &mut out).map_err(|e| e.to_string())?;
    }
    Ok(())
}

/// Cache root: `<cache>/datacode/dcmodule/`.
pub fn dcmodule_cache_root() -> PathBuf {
    dirs::cache_dir()
        .unwrap_or_else(std::env::temp_dir)
        .join("datacode")
        .join("dcmodule")
}

/// Read manifest, check ABI, extract to `cache_root/name/<sha256>/`, return absolute dylib path.
pub fn resolve_dylib_from_archive(zip_path: &Path, cache_root: &Path) -> Result<PathBuf, String> {
    let bytes = fs::read(zip_path).map_err(|e| e.to_string())?;
    let manifest = manifest_from_zip_bytes(&bytes)?;
    manifest.check_abi()?;
    let rel = manifest.library_relative_path()?;
    let digest = hex_sha256(&bytes);
    let out_dir = cache_root
        .join(&manifest.name)
        .join(&digest);
    let dylib_path = out_dir.join(&rel);
    if dylib_path.is_file() {
        return Ok(dylib_path);
    }
    if out_dir.exists() {
        fs::remove_dir_all(&out_dir).map_err(|e| e.to_string())?;
    }
    fs::create_dir_all(&out_dir).map_err(|e| e.to_string())?;
    extract_zip_bytes(&bytes, &out_dir)?;
    let dylib_path = out_dir.join(&rel);
    if !dylib_path.is_file() {
        return Err(format!(
            "manifest library path {:?} not found after extract (expected a file)",
            rel
        ));
    }
    Ok(dylib_path)
}
