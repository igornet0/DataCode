//! Archive format detection via magic bytes.

use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArchiveFormat {
    Zip,
    SevenZ,
    Rar,
}

impl ArchiveFormat {
    pub fn as_str(self) -> &'static str {
        match self {
            ArchiveFormat::Zip => "zip",
            ArchiveFormat::SevenZ => "7z",
            ArchiveFormat::Rar => "rar",
        }
    }
}

/// Detect archive format from the first bytes of a file (not by extension).
pub fn detect_format_from_header(header: &[u8]) -> Option<ArchiveFormat> {
    if header.starts_with(b"PK\x03\x04")
        || header.starts_with(b"PK\x05\x06")
        || header.starts_with(b"PK\x07\x08")
    {
        return Some(ArchiveFormat::Zip);
    }
    if header.starts_with(b"7z\xBC\xAF\x27\x1C") {
        return Some(ArchiveFormat::SevenZ);
    }
    if header.starts_with(b"Rar!\x1a\x07\x00") || header.starts_with(b"Rar!\x1a\x07\x01\x00") {
        return Some(ArchiveFormat::Rar);
    }
    None
}

pub fn read_format_header(path: &Path) -> Result<Vec<u8>, String> {
    use std::fs::File;
    let mut file = File::open(path).map_err(|e| format!("Archive not found: '{}': {}", path.display(), e))?;
    let mut header = vec![0u8; 12];
    let n = file
        .read(&mut header)
        .map_err(|e| format!("Cannot read archive '{}': {}", path.display(), e))?;
    header.truncate(n);
    Ok(header)
}

pub fn detect_format(path: &Path) -> Result<ArchiveFormat, String> {
    let header = read_format_header(path)?;
    detect_format_from_header(&header).ok_or_else(|| {
        format!(
            "Unsupported archive format in '{}'",
            path.display()
        )
    })
}

/// Seekable reader helper: read header without consuming the stream permanently.
pub fn detect_format_from_reader<R: Read + Seek>(reader: &mut R) -> Result<ArchiveFormat, String> {
    let pos = reader
        .stream_position()
        .map_err(|e| format!("Cannot read archive header: {}", e))?;
    let mut header = vec![0u8; 12];
    let n = reader
        .read(&mut header)
        .map_err(|e| format!("Cannot read archive header: {}", e))?;
    header.truncate(n);
    reader
        .seek(SeekFrom::Start(pos))
        .map_err(|e| format!("Cannot seek archive: {}", e))?;
    detect_format_from_header(&header).ok_or_else(|| "Unsupported archive format".to_string())
}
