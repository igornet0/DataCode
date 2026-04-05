//! Build a `.dcmodule` zip from a staging directory containing `manifest.json`.

use std::fs::{self, File};
use std::path::{Path, PathBuf};

use zip::write::FileOptions;
use zip::CompressionMethod;
use zip::ZipWriter;

use super::manifest::DcmoduleManifest;

/// Pack `dir` (must contain `manifest.json`) into `out_path` (typically `name.dcmodule`).
pub fn pack_directory(dir: &Path, out_path: &Path) -> Result<(), String> {
    let manifest_path = dir.join("manifest.json");
    if !manifest_path.is_file() {
        return Err(format!(
            "missing {}",
            manifest_path.display()
        ));
    }
    let raw = fs::read_to_string(&manifest_path).map_err(|e| e.to_string())?;
    let manifest: DcmoduleManifest =
        serde_json::from_str(&raw).map_err(|e| format!("manifest.json: {}", e))?;
    manifest.check_abi()?;
    let _ = manifest.library_relative_path()?;

    let file = File::create(out_path).map_err(|e| e.to_string())?;
    let mut zip = ZipWriter::new(file);
    let opts = FileOptions::<()>::default()
        .compression_method(CompressionMethod::Deflated);

    let dir = fs::canonicalize(dir).map_err(|e| e.to_string())?;
    let mut files: Vec<PathBuf> = Vec::new();
    collect_files(&dir, &mut files)?;
    for p in files {
        let rel = p.strip_prefix(&dir).map_err(|e| e.to_string())?;
        let name_in_zip = path_to_zip_name(rel);
        zip.start_file(name_in_zip, opts)
            .map_err(|e| e.to_string())?;
        let mut f = fs::File::open(&p).map_err(|e| e.to_string())?;
        std::io::copy(&mut f, &mut zip).map_err(|e| e.to_string())?;
    }
    zip.finish().map_err(|e| e.to_string())?;
    Ok(())
}

fn collect_files(dir: &Path, out: &mut Vec<PathBuf>) -> Result<(), String> {
    for e in fs::read_dir(dir).map_err(|e| e.to_string())? {
        let e = e.map_err(|e| e.to_string())?;
        let p = e.path();
        if p.is_dir() {
            collect_files(&p, out)?;
        } else if p.is_file() {
            out.push(p);
        }
    }
    Ok(())
}

fn path_to_zip_name(rel: &Path) -> String {
    rel
        .components()
        .map(|c| c.as_os_str().to_string_lossy())
        .collect::<Vec<_>>()
        .join("/")
}

/// Default output path: `<cwd>/<name>.dcmodule` from manifest `name`.
pub fn default_output_path(dir: &Path) -> Result<PathBuf, String> {
    let manifest_path = dir.join("manifest.json");
    let raw = fs::read_to_string(&manifest_path).map_err(|e| e.to_string())?;
    let manifest: DcmoduleManifest =
        serde_json::from_str(&raw).map_err(|e| e.to_string())?;
    let cwd = std::env::current_dir().map_err(|e| e.to_string())?;
    Ok(cwd.join(format!("{}.dcmodule", manifest.name)))
}
