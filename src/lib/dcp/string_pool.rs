use crate::dcp::binary::BinaryReader;
use crate::dcp::error::DcpError;

#[derive(Debug, Clone, Default)]
pub struct StringPool {
    strings: Vec<String>,
}

impl StringPool {
    pub fn read(reader: &mut BinaryReader<'_>) -> Result<Self, DcpError> {
        let count = reader.u32()? as usize;
        let mut strings = Vec::with_capacity(count);
        for _ in 0..count {
            strings.push(reader.string()?);
        }
        Ok(Self { strings })
    }

    pub fn resolve(&self, string_id: usize) -> Result<String, DcpError> {
        self.strings.get(string_id).cloned().ok_or_else(|| {
            DcpError::InvalidSection(format!("invalid string id {string_id}"))
        })
    }
}
