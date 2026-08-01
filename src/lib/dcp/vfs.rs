//! In-memory virtual filesystem for DCP assets (WebSocket execution).

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::dcp::asset::normalize_asset_path;
use crate::dcp::error::DcpError;

thread_local! {
    static DCP_VFS: RefCell<Option<Arc<DcpVfs>>> = RefCell::new(None);
}

pub fn set_dcp_vfs(vfs: Option<Arc<DcpVfs>>) {
    DCP_VFS.with(|slot| *slot.borrow_mut() = vfs);
}

pub fn get_dcp_vfs() -> Option<Arc<DcpVfs>> {
    DCP_VFS.with(|slot| slot.borrow().clone())
}

pub fn clear_dcp_vfs() {
    set_dcp_vfs(None);
}

pub fn dcp_vfs_active() -> bool {
    get_dcp_vfs().is_some()
}

/// Normalize a VM path to a VFS lookup key (empty string = root).
pub fn normalize_vfs_path(path: &Path) -> Result<String, DcpError> {
    let s = path.to_string_lossy();
    let trimmed = s.trim();
    if trimmed.is_empty() || trimmed == "." {
        return Ok(String::new());
    }
    normalize_asset_path(trimmed)
}

pub fn logical_path_to_pathbuf(key: &str) -> PathBuf {
    if key.is_empty() {
        PathBuf::from(".")
    } else {
        PathBuf::from(key)
    }
}

pub fn format_logical_path(key: &str) -> String {
    if key.is_empty() {
        "./".to_string()
    } else {
        format!("./{key}")
    }
}

#[derive(Debug, Clone)]
pub struct DcpVfs {
    files: HashMap<String, Vec<u8>>,
}

impl DcpVfs {
    pub fn from_assets(assets: Vec<(String, Vec<u8>)>) -> Result<Self, DcpError> {
        let mut files = HashMap::with_capacity(assets.len());
        for (rel_path, bytes) in assets {
            let key = normalize_asset_path(&rel_path)?;
            files.insert(key, bytes);
        }
        Ok(Self { files })
    }

    pub fn asset_paths(&self) -> Vec<String> {
        let mut paths: Vec<String> = self.files.keys().cloned().collect();
        paths.sort();
        paths
    }

    pub fn asset_count(&self) -> usize {
        self.files.len()
    }

    pub fn get(&self, key: &str) -> Option<&[u8]> {
        self.files.get(key).map(|v| v.as_slice())
    }

    pub fn contains_file(&self, key: &str) -> bool {
        self.files.contains_key(key)
    }

    pub fn is_dir(&self, prefix: &str) -> bool {
        if prefix.is_empty() {
            return !self.files.is_empty();
        }
        if self.files.contains_key(prefix) {
            return false;
        }
        let needle = format!("{prefix}/");
        self.files.keys().any(|k| k.starts_with(&needle))
    }

    pub fn exists(&self, key: &str) -> bool {
        self.contains_file(key) || self.is_dir(key)
    }

    /// Direct child names under `prefix` (files and virtual directories).
    pub fn list_dir(&self, prefix: &str) -> Vec<String> {
        let prefix_norm = if prefix.is_empty() {
            String::new()
        } else {
            match normalize_asset_path(prefix) {
                Ok(p) => p,
                Err(_) => return Vec::new(),
            }
        };

        let mut children: HashSet<String> = HashSet::new();

        for key in self.files.keys() {
            if prefix_norm.is_empty() {
                if let Some((first, rest)) = key.split_once('/') {
                    if rest.contains('/') {
                        children.insert(first.to_string());
                    } else {
                        children.insert(key.clone());
                    }
                } else {
                    children.insert(key.clone());
                }
            } else if key == &prefix_norm {
                continue;
            } else if let Some(rest) = key.strip_prefix(&format!("{prefix_norm}/")) {
                if rest.is_empty() {
                    continue;
                }
                if let Some((first, _)) = rest.split_once('/') {
                    children.insert(first.to_string());
                } else {
                    children.insert(rest.to_string());
                }
            }
        }

        let mut out: Vec<String> = children.into_iter().collect();
        out.sort();
        out
    }

    /// Full logical paths of direct children for `list_files`.
    pub fn list_dir_paths(&self, prefix: &str) -> Vec<PathBuf> {
        let prefix_norm = if prefix.is_empty() {
            String::new()
        } else {
            normalize_asset_path(prefix).unwrap_or_default()
        };

        self.list_dir(prefix)
            .into_iter()
            .map(|name| {
                if prefix_norm.is_empty() {
                    PathBuf::from(&name)
                } else {
                    PathBuf::from(format!("{prefix_norm}/{name}"))
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_vfs() -> DcpVfs {
        DcpVfs {
            files: HashMap::from([
                ("sample.csv".to_string(), b"a,b".to_vec()),
                ("data/nested.txt".to_string(), b"nested".to_vec()),
                ("data/sub/deep.bin".to_string(), vec![0, 1]),
            ]),
        }
    }

    #[test]
    fn get_existing_file() {
        let vfs = sample_vfs();
        assert_eq!(vfs.get("sample.csv"), Some(b"a,b".as_slice()));
    }

    #[test]
    fn is_dir_and_list_root() {
        let vfs = sample_vfs();
        assert!(vfs.is_dir(""));
        assert!(vfs.is_dir("data"));
        assert!(!vfs.is_dir("sample.csv"));
        let root = vfs.list_dir("");
        assert!(root.contains(&"sample.csv".to_string()));
        assert!(root.contains(&"data".to_string()));
    }

    #[test]
    fn list_nested_dir() {
        let vfs = sample_vfs();
        let data = vfs.list_dir("data");
        assert!(data.contains(&"nested.txt".to_string()));
        assert!(data.contains(&"sub".to_string()));
    }

    #[test]
    fn from_assets_normalizes() {
        let vfs = DcpVfs::from_assets(vec![(
            "data\\file.txt".to_string(),
            b"ok".to_vec(),
        )])
        .unwrap();
        assert!(vfs.contains_file("data/file.txt"));
    }

    #[test]
    fn normalize_vfs_path_root() {
        assert_eq!(normalize_vfs_path(Path::new("")).unwrap(), "");
        assert_eq!(normalize_vfs_path(Path::new(".")).unwrap(), "");
    }
}
