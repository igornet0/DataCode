use flate2::read::ZlibDecoder;
use std::io::Read;

use crate::dcp::constants::CompressionType;
use crate::dcp::error::DcpError;

pub fn decompress_data(data: &[u8], compression: CompressionType) -> Result<Vec<u8>, DcpError> {
    match compression {
        CompressionType::None => Ok(data.to_vec()),
        CompressionType::Zlib => {
            let mut decoder = ZlibDecoder::new(data);
            let mut out = Vec::new();
            decoder
                .read_to_end(&mut out)
                .map_err(|e| DcpError::DecompressionFailed(e.to_string()))?;
            Ok(out)
        }
    }
}
