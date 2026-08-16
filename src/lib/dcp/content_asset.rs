//! Content-addressed DCP assets (`assets/{sha256}`, `asset://` refs).

use std::cell::RefCell;
use std::collections::HashMap;

use serde::{Deserialize, Serialize};

pub const ASSET_SCHEME: &str = "asset://";
pub const CONTENT_ASSET_PREFIX: &str = "assets/";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssetMeta {
    pub id: String,
    pub kind: String,
    #[serde(default = "default_mime")]
    pub mime_type: String,
    pub size: u64,
    #[serde(default)]
    pub filename: Option<String>,
}

fn default_mime() -> String {
    "application/octet-stream".to_string()
}

#[derive(Debug, Clone)]
pub struct ContentAsset {
    pub meta: AssetMeta,
    pub data: Vec<u8>,
}

#[derive(Debug, Clone, Default)]
pub struct ContentAssetStore {
    by_id: HashMap<String, ContentAsset>,
}

impl ContentAssetStore {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, asset: ContentAsset) {
        self.by_id.insert(asset.meta.id.clone(), asset);
    }

    pub fn get(&self, id: &str) -> Option<&ContentAsset> {
        self.by_id.get(id)
    }

    pub fn contains(&self, id: &str) -> bool {
        self.by_id.contains_key(id)
    }

    pub fn ids(&self) -> Vec<String> {
        let mut ids: Vec<String> = self.by_id.keys().cloned().collect();
        ids.sort();
        ids
    }

    pub fn metas(&self) -> Vec<AssetMeta> {
        let mut metas: Vec<AssetMeta> = self.by_id.values().map(|a| a.meta.clone()).collect();
        metas.sort_by(|a, b| a.id.cmp(&b.id));
        metas
    }

    pub fn len(&self) -> usize {
        self.by_id.len()
    }

    pub fn is_empty(&self) -> bool {
        self.by_id.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = (&String, &ContentAsset)> {
        self.by_id.iter()
    }
}

pub fn is_content_asset_section_name(name: &str) -> bool {
    if !name.starts_with(CONTENT_ASSET_PREFIX) {
        return false;
    }
    let rest = &name[CONTENT_ASSET_PREFIX.len()..];
    !rest.is_empty() && !rest.contains('/')
}

pub fn content_asset_id_from_section_name(name: &str) -> Option<&str> {
    if is_content_asset_section_name(name) {
        Some(&name[CONTENT_ASSET_PREFIX.len()..])
    } else {
        None
    }
}

pub fn parse_asset_ref(value: &str) -> Option<&str> {
    let id = value.strip_prefix(ASSET_SCHEME)?;
    if id.is_empty() || id.contains('/') || id.contains('\\') {
        return None;
    }
    Some(id)
}

pub fn decode_asset_index_json(data: &[u8]) -> Result<Vec<AssetMeta>, String> {
    let text = std::str::from_utf8(data).map_err(|e| format!("__asset_index__ utf-8: {e}"))?;
    let metas: Vec<AssetMeta> =
        serde_json::from_str(text).map_err(|e| format!("__asset_index__ json: {e}"))?;
    Ok(metas)
}

thread_local! {
    static DCP_CONTENT_ASSETS: RefCell<Option<ContentAssetStore>> = const { RefCell::new(None) };
}

pub fn set_dcp_content_assets(store: Option<ContentAssetStore>) {
    DCP_CONTENT_ASSETS.with(|slot| *slot.borrow_mut() = store);
}

pub fn get_dcp_content_assets() -> Option<ContentAssetStore> {
    DCP_CONTENT_ASSETS.with(|slot| slot.borrow().clone())
}

pub fn clear_dcp_content_assets() {
    set_dcp_content_assets(None);
}

pub fn dcp_content_assets_active() -> bool {
    DCP_CONTENT_ASSETS.with(|slot| slot.borrow().is_some())
}

/// Validate that every `asset://` string cell in `table` resolves in `store`.
pub fn validate_table_asset_refs(
    table_name: &str,
    table: &crate::common::table::Table,
    store: &ContentAssetStore,
) -> Result<(), String> {
    let headers = table.headers();
    let n = table.len();
    for (col_idx, col_name) in headers.iter().enumerate() {
        for row_idx in 0..n {
            let Some(row) = table.get_row(row_idx) else {
                continue;
            };
            let Some(cell) = row.get(col_idx) else {
                continue;
            };
            let crate::common::value::Value::String(s) = cell else {
                continue;
            };
            let Some(asset_id) = parse_asset_ref(s) else {
                continue;
            };
            if !store.contains(asset_id) {
                return Err(format!(
                    "Asset \"{asset_id}\" referenced by {table_name}.{col_name} does not exist"
                ));
            }
        }
    }
    Ok(())
}
