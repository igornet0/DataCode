use crc32fast::Hasher as Crc32Hasher;
use sha2::{Digest, Sha256};

use crate::dcp::constants::ChecksumType;

pub fn verify_checksum(checksum_type: ChecksumType, data: &[u8], expected: &[u8]) -> bool {
    match checksum_type {
        ChecksumType::None => expected.is_empty(),
        ChecksumType::Crc32 => {
            if expected.len() != 4 {
                return false;
            }
            let mut hasher = Crc32Hasher::new();
            hasher.update(data);
            let digest = hasher.finalize().to_le_bytes();
            digest == expected
        }
        ChecksumType::Sha256 => {
            if expected.len() != 32 {
                return false;
            }
            let digest = Sha256::digest(data);
            digest.as_slice() == expected
        }
    }
}
