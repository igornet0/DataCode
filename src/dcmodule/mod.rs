//! Datacode native module artifact (`.dcmodule`): zip + `manifest.json` + dylib.

mod extract;
mod manifest;
mod pack;

pub use extract::{dcmodule_cache_root, extract_zip_bytes, manifest_from_zip_bytes, resolve_dylib_from_archive};
pub use manifest::{DcmoduleManifest, host_target_key};
pub use pack::{default_output_path, pack_directory};
