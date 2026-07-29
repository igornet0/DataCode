use crate::dcp::error::DcpError;

pub struct BinaryReader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> BinaryReader<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    pub fn position(&self) -> usize {
        self.pos
    }

    pub fn seek(&mut self, offset: usize) -> Result<(), DcpError> {
        if offset > self.data.len() {
            return Err(DcpError::UnexpectedEof);
        }
        self.pos = offset;
        Ok(())
    }

    fn read(&mut self, size: usize) -> Result<&'a [u8], DcpError> {
        if self.pos + size > self.data.len() {
            return Err(DcpError::UnexpectedEof);
        }
        let chunk = &self.data[self.pos..self.pos + size];
        self.pos += size;
        Ok(chunk)
    }

    pub fn u8(&mut self) -> Result<u8, DcpError> {
        Ok(self.read(1)?[0])
    }

    pub fn u16(&mut self) -> Result<u16, DcpError> {
        Ok(u16::from_le_bytes(self.read(2)?.try_into().unwrap()))
    }

    pub fn u32(&mut self) -> Result<u32, DcpError> {
        Ok(u32::from_le_bytes(self.read(4)?.try_into().unwrap()))
    }

    pub fn u64(&mut self) -> Result<u64, DcpError> {
        Ok(u64::from_le_bytes(self.read(8)?.try_into().unwrap()))
    }

    pub fn bytes(&mut self, size: usize) -> Result<&'a [u8], DcpError> {
        self.read(size)
    }

    pub fn string(&mut self) -> Result<String, DcpError> {
        let length = self.u32()? as usize;
        let data = self.read(length)?;
        String::from_utf8(data.to_vec()).map_err(|e| DcpError::Utf8Error(e.to_string()))
    }

    pub fn skip(&mut self, size: usize) -> Result<(), DcpError> {
        self.read(size)?;
        Ok(())
    }
}
