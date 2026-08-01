pub const DCP_MAGIC: &[u8; 4] = b"DCPK";
pub const DCP_VERSION_MAJOR: u16 = 1;
pub const DCP_VERSION_1_0: u16 = (DCP_VERSION_MAJOR << 8) | 0;
pub const DCP_VERSION_1_1: u16 = (DCP_VERSION_MAJOR << 8) | 1;
pub const HEADER_SIZE: usize = 64;
pub const STRING_POOL_OFFSET: usize = HEADER_SIZE;

pub const CODE_SECTION_NAME: &str = "__code__";
pub const METADATA_SECTION_NAME: &str = "__metadata__";
pub const CONFIG_SECTION_NAME: &str = "__config__";
pub const VARIABLES_SECTION_NAME: &str = "__variables__";
pub const SQL_SECTION_NAME: &str = "__sql__";
pub const SQL_TABLE_SECTION_NAME: &str = "__sql_table__";

pub const MAX_PACKAGE_SIZE: usize = 100 * 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u16)]
pub enum SectionType {
    Code = 1,
    Metadata = 2,
    ArrowTable = 3,
    Asset = 4,
    Config = 5,
    Variables = 6,
    Sql = 7,
    SqlTable = 8,
}

impl SectionType {
    pub fn from_u16(value: u16) -> Option<Self> {
        match value {
            1 => Some(Self::Code),
            2 => Some(Self::Metadata),
            3 => Some(Self::ArrowTable),
            4 => Some(Self::Asset),
            5 => Some(Self::Config),
            6 => Some(Self::Variables),
            7 => Some(Self::Sql),
            8 => Some(Self::SqlTable),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ChecksumType {
    None = 0,
    Crc32 = 1,
    Sha256 = 2,
}

impl ChecksumType {
    pub fn from_u8(value: u8) -> Option<Self> {
        match value {
            0 => Some(Self::None),
            1 => Some(Self::Crc32),
            2 => Some(Self::Sha256),
            _ => None,
        }
    }

    pub fn size(self) -> usize {
        match self {
            Self::None => 0,
            Self::Crc32 => 4,
            Self::Sha256 => 32,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum CompressionType {
    None = 0,
    Zlib = 1,
}

impl CompressionType {
    pub fn from_u8(value: u8) -> Option<Self> {
        match value {
            0 => Some(Self::None),
            1 => Some(Self::Zlib),
            _ => None,
        }
    }
}

pub fn uses_string_pool(version: u16) -> bool {
    version >= DCP_VERSION_1_1
}

pub fn is_supported_version(version: u16) -> bool {
    matches!(version, DCP_VERSION_1_0 | DCP_VERSION_1_1)
}
