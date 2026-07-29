use crate::dcp::binary::BinaryReader;
use crate::dcp::constants::{ChecksumType, CompressionType, SectionType, uses_string_pool};
use crate::dcp::error::DcpError;
use crate::dcp::string_pool::StringPool;

#[derive(Debug, Clone)]
pub struct SectionIndexEntry {
    pub section_type: SectionType,
    pub flags: u16,
    pub name: String,
    pub offset: u64,
    pub data_length: u64,
    pub compression: CompressionType,
    pub checksum_type: ChecksumType,
}

pub fn read_index(
    reader: &mut BinaryReader<'_>,
    count: u32,
    version: u16,
    string_pool: Option<&StringPool>,
) -> Result<Vec<SectionIndexEntry>, DcpError> {
    let mut entries = Vec::with_capacity(count as usize);
    let pooled = uses_string_pool(version);

    for _ in 0..count {
        let section_type_value = reader.u16()?;
        let section_type = SectionType::from_u16(section_type_value).ok_or_else(|| {
            DcpError::InvalidSection(format!("unknown section type {section_type_value}"))
        })?;
        let flags = reader.u16()?;

        let name = if pooled {
            let string_id = reader.u32()? as usize;
            let pool = string_pool.ok_or_else(|| {
                DcpError::InvalidSection("string pool required for v1.1 index".into())
            })?;
            pool.resolve(string_id)?
        } else {
            let name_length = reader.u32()? as usize;
            let name_bytes = reader.bytes(name_length)?;
            String::from_utf8(name_bytes.to_vec())
                .map_err(|e| DcpError::Utf8Error(e.to_string()))?
        };

        let offset = reader.u64()?;
        let data_length = reader.u64()?;
        let compression_value = reader.u8()?;
        let checksum_type_value = reader.u8()?;
        reader.skip(2)?;

        let compression = CompressionType::from_u8(compression_value)
            .ok_or_else(|| DcpError::InvalidSection("unknown compression type".into()))?;
        let checksum_type = ChecksumType::from_u8(checksum_type_value)
            .ok_or_else(|| DcpError::InvalidSection("unknown checksum type".into()))?;

        entries.push(SectionIndexEntry {
            section_type,
            flags,
            name,
            offset,
            data_length,
            compression,
            checksum_type,
        });
    }

    Ok(entries)
}
