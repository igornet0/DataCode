//! Datacode Package (DCP) decoder for WebSocket execution.

mod asset;
mod binary;
mod checksum;
mod compression;
mod constants;
mod decoder;
mod error;
mod header;
mod index;
mod metadata;
mod section;
mod string_pool;
mod vfs;

pub use asset::normalize_asset_path;
pub use constants::{SectionType, DCP_MAGIC};
pub use decoder::{DecodedPackage, DcpDecoder};
pub use error::DcpError;
pub use vfs::{
    clear_dcp_vfs, dcp_vfs_active, format_logical_path, get_dcp_vfs, logical_path_to_pathbuf,
    normalize_vfs_path, set_dcp_vfs, DcpVfs,
};
