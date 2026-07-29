use crate::dcp::constants::{
    CODE_SECTION_NAME, CONFIG_SECTION_NAME, METADATA_SECTION_NAME, VARIABLES_SECTION_NAME,
};
use crate::dcp::error::DcpError;

const RESERVED: &[&str] = &[
    CODE_SECTION_NAME,
    METADATA_SECTION_NAME,
    CONFIG_SECTION_NAME,
    VARIABLES_SECTION_NAME,
];

pub fn normalize_asset_path(rel_path: &str) -> Result<String, DcpError> {
    let trimmed = rel_path.trim();
    if trimmed.is_empty() {
        return Err(DcpError::InvalidPath("asset path must not be empty".into()));
    }

    let normalized = trimmed.replace('\\', "/");
    let normalized = normalized
        .split('/')
        .fold(String::new(), |mut acc, part| {
            if part.is_empty() || part == "." {
                return acc;
            }
            if !acc.is_empty() {
                acc.push('/');
            }
            acc.push_str(part);
            acc
        });

    if normalized.is_empty() {
        return Err(DcpError::InvalidPath(format!(
            "invalid asset path: {rel_path:?}"
        )));
    }

    if normalized.starts_with('/') {
        return Err(DcpError::InvalidPath(format!(
            "asset path must be relative: {rel_path:?}"
        )));
    }

    if normalized.split('/').any(|part| part == "..") {
        return Err(DcpError::InvalidPath(format!(
            "asset path must not contain '..': {rel_path:?}"
        )));
    }

    if RESERVED.contains(&normalized.as_str()) {
        return Err(DcpError::InvalidPath(format!(
            "asset path {normalized:?} is reserved for DCP sections"
        )));
    }

    Ok(normalized)
}
