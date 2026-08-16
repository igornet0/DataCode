//! Datacode Package (DCP) decoder for WebSocket execution.

mod arrow_table;
mod asset;
mod binary;
mod checksum;
mod compression;
mod constants;
mod content_asset;
mod decoder;
mod error;
mod fk_check;
mod header;
mod index;
mod metadata;
mod section;
mod session;
mod string_pool;
mod tables;
mod vfs;

pub use arrow_table::{apply_source_table_columns, arrow_ipc_to_table};
pub use asset::normalize_asset_path;
pub use constants::{SectionType, DCP_MAGIC};
pub use content_asset::{
    clear_dcp_content_assets, dcp_content_assets_active, get_dcp_content_assets, parse_asset_ref,
    set_dcp_content_assets, validate_table_asset_refs, AssetMeta, ContentAsset, ContentAssetStore,
};
pub use decoder::{DecodedPackage, DcpDecoder};
pub use error::DcpError;
pub use fk_check::FkCheckMode;
pub use session::{
    clear_dcp_metadata, clear_dcp_session, dcp_session_active, get_dcp_metadata, set_dcp_metadata,
};
pub use tables::{clear_dcp_tables, get_dcp_tables, set_dcp_tables, DcpTables};
pub use vfs::{
    clear_dcp_vfs, dcp_vfs_active, format_logical_path, get_dcp_vfs, logical_path_to_pathbuf,
    normalize_vfs_path, set_dcp_vfs, DcpVfs,
};
