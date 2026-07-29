use crate::dcp::asset::normalize_asset_path;
use crate::dcp::binary::BinaryReader;
use crate::dcp::constants::{
    CODE_SECTION_NAME, MAX_PACKAGE_SIZE, METADATA_SECTION_NAME, STRING_POOL_OFFSET,
    SectionType,
};
use crate::dcp::error::DcpError;
use crate::dcp::header::read_header;
use crate::dcp::index::{SectionIndexEntry, read_index};
use crate::dcp::metadata::Metadata;
use crate::dcp::section::{read_section_header, read_section_payload};
use crate::dcp::string_pool::StringPool;

#[derive(Debug, Clone)]
pub struct DecodedPackage {
    pub code: String,
    pub assets: Vec<(String, Vec<u8>)>,
    pub metadata: Option<Metadata>,
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
        let header = read_header(&mut reader)?;

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

        for entry in index {
            let payload = load_section_payload(data, &entry, string_pool.as_ref(), header.version)?;

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
                    let path = normalize_asset_path(&entry.name)?;
                    assets.push((path, payload));
                }
                SectionType::ArrowTable | SectionType::Config | SectionType::Variables => {}
            }
        }

        let code = code.ok_or(DcpError::CodeNotFound)?;
        assets.sort_by(|a, b| a.0.cmp(&b.0));

        Ok(DecodedPackage {
            code,
            assets,
            metadata,
        })
    }
}

fn load_section_payload(
    data: &[u8],
    entry: &SectionIndexEntry,
    string_pool: Option<&StringPool>,
    version: u16,
) -> Result<Vec<u8>, DcpError> {
    let mut reader = BinaryReader::new(data);
    reader.seek(entry.offset as usize)?;

    let (header, name, data_offset) =
        read_section_header(&mut reader, version, string_pool)?;

    if name != entry.name {
        return Err(DcpError::InvalidSection(format!(
            "section name mismatch at offset {}",
            entry.offset
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
        assert_eq!(package.assets.len(), 1);
        assert_eq!(package.assets[0].0, "data/sample.txt");
        assert_eq!(package.assets[0].1, b"asset payload");
    }

    #[test]
    fn decode_large_compressed_code() {
        let bytes = fs::read(fixture("large_code.dcp")).expect("fixture");
        let package = DcpDecoder::decode(&bytes).expect("decode");
        assert_eq!(package.code.len(), 5000);
        assert!(package.code.chars().all(|c| c == 'x'));
    }

    #[test]
    fn reject_invalid_magic() {
        let err = DcpDecoder::decode(b"NOPE").unwrap_err();
        assert!(err.to_string().contains("magic"));
    }
}
