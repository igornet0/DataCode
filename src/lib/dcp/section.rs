use crate::dcp::binary::BinaryReader;
use crate::dcp::checksum::verify_checksum;
use crate::dcp::compression::decompress_data;
use crate::dcp::constants::{ChecksumType, CompressionType, uses_string_pool};
use crate::dcp::error::DcpError;
use crate::dcp::string_pool::StringPool;

pub fn verify_section_checksum(
    checksum_type: ChecksumType,
    data: &[u8],
    expected: &[u8],
    name: &str,
) -> Result<(), DcpError> {
    if verify_checksum(checksum_type, data, expected) {
        Ok(())
    } else {
        Err(DcpError::ChecksumMismatch(format!(
            "section '{name}' checksum mismatch"
        )))
    }
}

pub fn read_section_payload(
    data: &[u8],
    offset: usize,
    data_length: u64,
    compression: CompressionType,
    checksum_type: ChecksumType,
    checksum: &[u8],
    name: &str,
) -> Result<Vec<u8>, DcpError> {
    let start = offset;
    let end = start
        .checked_add(data_length as usize)
        .ok_or(DcpError::InvalidSection(format!("section '{name}' size overflow")))?;
    if end > data.len() {
        return Err(DcpError::UnexpectedEof);
    }

    let payload = &data[start..end];
    verify_section_checksum(checksum_type, payload, checksum, name)?;

    decompress_data(payload, compression)
}

pub fn read_section_header(
    reader: &mut BinaryReader<'_>,
    version: u16,
    string_pool: Option<&StringPool>,
) -> Result<(u16, SectionHeaderInfo, String, usize), DcpError> {
    let section_type = reader.u16()?;
    let _flags = reader.u16()?;
    let identifier = reader.u32()?;
    let data_length = reader.u64()?;
    let compression_value = reader.u8()?;
    let checksum_type_value = reader.u8()?;
    let _reserved = reader.u16()?;

    let checksum_type = ChecksumType::from_u8(checksum_type_value)
        .ok_or_else(|| DcpError::InvalidSection("unknown checksum type".into()))?;
    let checksum_len = checksum_type.size();
    let checksum = reader.bytes(checksum_len)?.to_vec();

    let compression = CompressionType::from_u8(compression_value)
        .ok_or_else(|| DcpError::InvalidSection("unknown compression type".into()))?;

    let name = if uses_string_pool(version) {
        let pool = string_pool.ok_or_else(|| {
            DcpError::InvalidSection("string pool required for v1.1 sections".into())
        })?;
        pool.resolve(identifier as usize)?
    } else {
        let name_length = identifier as usize;
        let name_bytes = reader.bytes(name_length)?;
        String::from_utf8(name_bytes.to_vec())
            .map_err(|e| DcpError::Utf8Error(e.to_string()))?
    };

    let data_offset = reader.position();

    Ok((
        section_type,
        SectionHeaderInfo {
            data_length,
            compression,
            checksum_type,
            checksum,
        },
        name,
        data_offset,
    ))
}

#[derive(Debug, Clone)]
pub struct SectionHeaderInfo {
    pub data_length: u64,
    pub compression: CompressionType,
    pub checksum_type: ChecksumType,
    pub checksum: Vec<u8>,
}
