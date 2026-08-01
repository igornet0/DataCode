use crate::dcp::binary::BinaryReader;
use crate::dcp::constants::{DCP_MAGIC, HEADER_SIZE, is_supported_version};
use crate::dcp::error::DcpError;

#[derive(Debug, Clone)]
pub struct PackageHeader {
    pub version: u16,
    pub section_count: u32,
    pub index_offset: u64,
}

pub fn read_header(reader: &mut BinaryReader<'_>) -> Result<(PackageHeader, u64), DcpError> {
    let start = reader.position();
    let magic = reader.bytes(4)?;
    if magic != DCP_MAGIC {
        return Err(DcpError::InvalidMagic);
    }

    let version = reader.u16()?;
    if !is_supported_version(version) {
        return Err(DcpError::UnsupportedVersion(version));
    }

    let _flags = reader.u16()?;
    let header_size = reader.u16()?;
    if header_size as usize != HEADER_SIZE {
        return Err(DcpError::InvalidHeader(format!(
            "invalid header size {header_size}"
        )));
    }

    let _reserved = reader.u16()?;
    let section_count = reader.u32()?;
    let index_offset = reader.u64()?;
    let package_size = reader.u64()?;
    let _checksum_type = reader.u8()?;

    let consumed = reader.position() - start;
    reader.skip(HEADER_SIZE - consumed)?;

    Ok((
        PackageHeader {
            version,
            section_count,
            index_offset,
        },
        package_size,
    ))
}
