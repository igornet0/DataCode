//! Runtime module cache: compiled bytecode (chunk + functions) keyed by canonical path.
//! We cache the compile stage only; run() is executed on every import to preserve semantics (side effects).

use crate::bytecode::{Chunk, Function};
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Compiled module: chunk (main script bytecode) and functions. Stored in cache by canonical path.
/// On cache hit we run this again to get exports; we do not cache execution result.
#[derive(Debug, Clone)]
pub struct CachedModule {
    pub chunk: Chunk,
    pub functions: Arc<Vec<Function>>,
}

/// Canonical path for use as cache key (in-memory module cache, dependency graph, `.dcb` filename).
/// Always tries [`Path::canonicalize`] so symlink-equivalent spellings (e.g. `/var/...` vs
/// `/private/var/...` on macOS) share one key; on failure falls back to [`Path::to_path_buf`].
/// On Windows, paths are normalized to lowercase for case-insensitive dedup.
#[inline]
pub fn canonical_module_cache_key(path: &Path) -> PathBuf {
    let canonical = path
        .canonicalize()
        .unwrap_or_else(|_| path.to_path_buf());
    #[cfg(windows)]
    {
        PathBuf::from(canonical.to_string_lossy().to_lowercase())
    }
    #[cfg(not(windows))]
    {
        canonical
    }
}
