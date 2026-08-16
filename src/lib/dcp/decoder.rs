use std::collections::HashMap;

use crate::dcp::asset::normalize_asset_path;
use crate::dcp::binary::BinaryReader;
use crate::dcp::constants::{
    ASSET_INDEX_SECTION_NAME, CODE_SECTION_NAME, CONFIG_SECTION_NAME, MAX_PACKAGE_SIZE,
    METADATA_SECTION_NAME, SQL_SECTION_NAME, SQL_TABLE_SECTION_NAME, STRING_POOL_OFFSET,
    SectionType,
};
use crate::dcp::content_asset::{
    content_asset_id_from_section_name, decode_asset_index_json, AssetMeta, ContentAsset,
    ContentAssetStore,
};
use crate::dcp::error::DcpError;
use crate::dcp::fk_check::{parse_config_section, FkCheckMode};
use crate::dcp::header::read_header;
use crate::dcp::index::{SectionIndexEntry, SectionIndexMeta, read_index};
use crate::dcp::metadata::Metadata;
use crate::dcp::section::{read_section_header, read_section_payload};
use crate::dcp::string_pool::StringPool;

#[derive(Debug, Clone)]
pub struct DecodedPackage {
    pub code: String,
    /// Path-based VFS assets (excludes `assets/{sha256}`).
    pub assets: Vec<(String, Vec<u8>)>,
    /// Content-addressed assets keyed by SHA-256 hex id.
    pub content_assets: ContentAssetStore,
    pub tables: Vec<(String, Vec<u8>)>,
    pub metadata: Option<Metadata>,
    pub sql: Option<String>,
    /// Soft SQL (sql_table / table_insert): warn + skip on errors.
    pub sql_table: Option<String>,
    /// From `__config__.fk_check`; default `Strict` when the section is absent.
    pub fk_check: FkCheckMode,
}

pub struct DcpDecoder;

impl DcpDecoder {
    pub fn decode(data: &[u8]) -> Result<DecodedPackage, DcpError> {
        if data.len() > MAX_PACKAGE_SIZE {
            return Err(DcpError::PackageTooLarge {
                size: data.len(),
                limit: MAX_PACKAGE_SIZE,
            });
        }

        let mut reader = BinaryReader::new(data);
        let (header, package_size) = read_header(&mut reader)?;
        if package_size as usize != data.len() {
            return Err(DcpError::InvalidHeader(format!(
                "package size mismatch: header {package_size}, actual {}",
                data.len()
            )));
        }

        let string_pool = if crate::dcp::constants::uses_string_pool(header.version) {
            reader.seek(STRING_POOL_OFFSET)?;
            Some(StringPool::read(&mut reader)?)
        } else {
            None
        };

        reader.seek(header.index_offset as usize)?;
        let index = read_index(
            &mut reader,
            header.section_count,
            header.version,
            string_pool.as_ref(),
        )?;

        let mut code: Option<String> = None;
        let mut metadata: Option<Metadata> = None;
        let mut assets: Vec<(String, Vec<u8>)> = Vec::new();
        let mut content_blobs: Vec<(String, Vec<u8>)> = Vec::new();
        let mut asset_index_metas: Option<Vec<AssetMeta>> = None;
        let mut tables: Vec<(String, Vec<u8>)> = Vec::new();
        let mut sql: Option<String> = None;
        let mut sql_table: Option<String> = None;
        let mut fk_check = FkCheckMode::Strict;

        for (entry, index_meta) in index {
            let payload = load_section_payload(
                data,
                &entry,
                &index_meta,
                string_pool.as_ref(),
                header.version,
            )?;

            match entry.section_type {
                SectionType::Code => {
                    if entry.name == CODE_SECTION_NAME {
                        code = Some(String::from_utf8(payload).map_err(|e| {
                            DcpError::Utf8Error(format!("CODE section: {e}"))
                        })?);
                    }
                }
                SectionType::Metadata => {
                    if entry.name == METADATA_SECTION_NAME {
                        metadata = Some(crate::dcp::metadata::decode_metadata(&payload)?);
                    }
                }
                SectionType::Asset => {
                    if let Some(id) = content_asset_id_from_section_name(&entry.name) {
                        content_blobs.push((id.to_string(), payload));
                    } else {
                        let path = normalize_asset_path(&entry.name)?;
                        assets.push((path, payload));
                    }
                }
                SectionType::AssetIndex => {
                    if entry.name != ASSET_INDEX_SECTION_NAME {
                        return Err(DcpError::InvalidSection(format!(
                            "ASSET_INDEX section must be named '{ASSET_INDEX_SECTION_NAME}', got '{}'",
                            entry.name
                        )));
                    }
                    let metas = decode_asset_index_json(&payload)
                        .map_err(DcpError::InvalidSection)?;
                    asset_index_metas = Some(metas);
                }
                SectionType::ArrowTable => {
                    tables.push((entry.name, payload));
                }
                SectionType::Sql => {
                    if entry.name != SQL_SECTION_NAME {
                        return Err(DcpError::InvalidSection(format!(
                            "SQL section must be named '{SQL_SECTION_NAME}', got '{}'",
                            entry.name
                        )));
                    }
                    sql = Some(String::from_utf8(payload).map_err(|e| {
                        DcpError::Utf8Error(format!("SQL section: {e}"))
                    })?);
                }
                SectionType::SqlTable => {
                    if entry.name != SQL_TABLE_SECTION_NAME {
                        return Err(DcpError::InvalidSection(format!(
                            "SQL_TABLE section must be named '{SQL_TABLE_SECTION_NAME}', got '{}'",
                            entry.name
                        )));
                    }
                    sql_table = Some(String::from_utf8(payload).map_err(|e| {
                        DcpError::Utf8Error(format!("SQL_TABLE section: {e}"))
                    })?);
                }
                SectionType::Config => {
                    if entry.name != CONFIG_SECTION_NAME {
                        return Err(DcpError::InvalidSection(format!(
                            "CONFIG section must be named '{CONFIG_SECTION_NAME}', got '{}'",
                            entry.name
                        )));
                    }
                    fk_check = parse_config_section(&payload)?;
                }
                SectionType::Variables => {}
            }
        }

        let code = code.ok_or(DcpError::CodeNotFound)?;
        assets.sort_by(|a, b| a.0.cmp(&b.0));
        tables.sort_by(|a, b| a.0.cmp(&b.0));

        let mut content_assets = ContentAssetStore::new();
        let meta_by_id: HashMap<String, AssetMeta> = asset_index_metas
            .unwrap_or_default()
            .into_iter()
            .map(|m| (m.id.clone(), m))
            .collect();
        for (id, payload) in content_blobs {
            let meta = meta_by_id.get(&id).cloned().unwrap_or_else(|| AssetMeta {
                id: id.clone(),
                kind: "file".to_string(),
                mime_type: "application/octet-stream".to_string(),
                size: payload.len() as u64,
                filename: None,
            });
            let mut meta = meta;
            meta.size = payload.len() as u64;
            content_assets.insert(ContentAsset {
                meta,
                data: payload,
            });
        }

        Ok(DecodedPackage {
            code,
            assets,
            content_assets,
            tables,
            metadata,
            sql,
            sql_table,
            fk_check,
        })
    }
}

fn load_section_payload(
    data: &[u8],
    entry: &SectionIndexEntry,
    index_meta: &SectionIndexMeta,
    string_pool: Option<&StringPool>,
    version: u16,
) -> Result<Vec<u8>, DcpError> {
    let mut reader = BinaryReader::new(data);
    reader.seek(entry.offset as usize)?;

    let (section_type, header, name, data_offset) =
        read_section_header(&mut reader, version, string_pool)?;

    if section_type != entry.section_type as u16 {
        return Err(DcpError::InvalidSection(format!(
            "section type mismatch at offset {}: index {:?}, header {section_type}",
            entry.offset, entry.section_type
        )));
    }

    if name != entry.name {
        return Err(DcpError::InvalidSection(format!(
            "section name mismatch at offset {}",
            entry.offset
        )));
    }

    if header.data_length != index_meta.data_length {
        return Err(DcpError::InvalidSection(format!(
            "section data length mismatch for '{}': index {}, header {}",
            entry.name, index_meta.data_length, header.data_length
        )));
    }

    if header.compression != index_meta.compression {
        return Err(DcpError::InvalidSection(format!(
            "section compression mismatch for '{}'",
            entry.name
        )));
    }

    if header.checksum_type != index_meta.checksum_type {
        return Err(DcpError::InvalidSection(format!(
            "section checksum type mismatch for '{}'",
            entry.name
        )));
    }

    read_section_payload(
        data,
        data_offset,
        header.data_length,
        header.compression,
        header.checksum_type,
        &header.checksum,
        &name,
    )
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::PathBuf;

    use crate::dcp::DcpDecoder;

    fn fixture(name: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests")
            .join("dcp_fixtures")
            .join(name)
    }

    #[test]
    fn decode_code_only_fixture() {
        let bytes = fs::read(fixture("code_only.dcp")).expect("fixture");
        let package = DcpDecoder::decode(&bytes).expect("decode");
        assert!(package.code.contains("hello dcp"));
        assert!(package.assets.is_empty());
        assert!(package.metadata.is_some());
        assert_eq!(
            package.metadata.as_ref().unwrap().custom.get("author"),
            Some(&"test".to_string())
        );
    }

    #[test]
    fn decode_with_assets_fixture() {
        let bytes = fs::read(fixture("with_assets.dcp")).expect("fixture");
        let package = DcpDecoder::decode(&bytes).expect("decode");
        assert!(!package.assets.is_empty());
    }

    #[test]
    fn decode_with_table_fixture() {
        let bytes = fs::read(fixture("with_table.dcp")).expect("fixture");
        let package = DcpDecoder::decode(&bytes).expect("decode");
        assert_eq!(package.tables.len(), 1);
    }

    #[test]
    fn decode_with_sql_fixture() {
        let bytes = fs::read(fixture("with_sql.dcp")).expect("fixture");
        let package = DcpDecoder::decode(&bytes).expect("decode");
        assert!(package.sql.is_some());
        assert_eq!(package.fk_check, crate::dcp::FkCheckMode::Strict);
    }

    #[test]
    fn decode_with_config_fk_check_warn() {
        let bytes = fs::read(fixture("with_config.dcp")).expect("fixture");
        let package = DcpDecoder::decode(&bytes).expect("decode");
        assert_eq!(package.fk_check, crate::dcp::FkCheckMode::Warn);
    }
}
