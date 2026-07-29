use std::collections::HashMap;

use crate::dcp::binary::BinaryReader;

#[derive(Debug, Clone, Default)]
pub struct Metadata {
    pub custom: HashMap<String, String>,
}

impl Metadata {
    pub fn to_map(&self) -> HashMap<String, String> {
        self.custom.clone()
    }
}

pub fn decode_metadata(data: &[u8]) -> Result<Metadata, crate::dcp::error::DcpError> {
    let mut reader = BinaryReader::new(data);
    let _version = reader.u16()?;

    let variable_count = reader.u32()?;
    for _ in 0..variable_count {
        let _name = reader.string()?;
        let value_length = reader.u32()? as usize;
        reader.bytes(value_length)?;
    }

    let resource_count = reader.u32()?;
    for _ in 0..resource_count {
        let _name = reader.string()?;
        let _uri = reader.string()?;
    }

    let mut custom = HashMap::new();
    let custom_count = reader.u32()?;
    for _ in 0..custom_count {
        let key = reader.string()?;
        let value_length = reader.u32()? as usize;
        let value_bytes = reader.bytes(value_length)?;
        let value = String::from_utf8(value_bytes.to_vec())
            .map_err(|e| crate::dcp::error::DcpError::Utf8Error(e.to_string()))?;
        custom.insert(key, value);
    }

    Ok(Metadata { custom })
}
