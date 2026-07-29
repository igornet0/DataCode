use std::fmt;

#[derive(Debug)]
pub enum DcpError {
    InvalidMagic,
    UnsupportedVersion(u16),
    InvalidHeader(String),
    UnexpectedEof,
    InvalidSection(String),
    ChecksumMismatch(String),
    DecompressionFailed(String),
    CodeNotFound,
    InvalidPath(String),
    PackageTooLarge { size: usize, limit: usize },
    Utf8Error(String),
}

impl fmt::Display for DcpError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidMagic => write!(f, "Invalid DCP package magic"),
            Self::UnsupportedVersion(v) => write!(f, "Unsupported DCP version {v}"),
            Self::InvalidHeader(msg) => write!(f, "Invalid DCP header: {msg}"),
            Self::UnexpectedEof => write!(f, "Unexpected end of DCP package"),
            Self::InvalidSection(msg) => write!(f, "Invalid DCP section: {msg}"),
            Self::ChecksumMismatch(msg) => write!(f, "Checksum mismatch: {msg}"),
            Self::DecompressionFailed(msg) => write!(f, "Decompression failed: {msg}"),
            Self::CodeNotFound => write!(f, "CODE section not found"),
            Self::InvalidPath(msg) => write!(f, "Invalid asset path: {msg}"),
            Self::PackageTooLarge { size, limit } => {
                write!(f, "DCP package too large ({size} bytes, limit {limit})")
            }
            Self::Utf8Error(msg) => write!(f, "Invalid UTF-8 in DCP package: {msg}"),
        }
    }
}

impl std::error::Error for DcpError {}
