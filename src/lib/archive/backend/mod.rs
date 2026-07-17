//! Archive backend trait — format-specific implementations.

use crate::archive::entry::ArchiveEntry;
use crate::archive::error::ArchiveError;
use crate::archive::format::ArchiveFormat;
use std::path::Path;

pub mod sevenz;
pub mod zip;

#[cfg(feature = "archive-rar")]
pub mod rar;

pub trait ArchiveBackend: Send {
    fn format(&self) -> ArchiveFormat;
    fn list_entries(&mut self) -> Result<Vec<ArchiveEntry>, ArchiveError>;
    fn read_entry(&mut self, path: &str) -> Result<Vec<u8>, ArchiveError>;
    fn extract_all(&mut self, dest: &Path) -> Result<(), ArchiveError>;
    fn close(&mut self);
}

pub fn open_backend(
    path: &Path,
    format: ArchiveFormat,
) -> Result<Box<dyn ArchiveBackend>, ArchiveError> {
    match format {
        ArchiveFormat::Zip => Ok(Box::new(zip::ZipBackend::open(path)?)),
        ArchiveFormat::SevenZ => Ok(Box::new(sevenz::SevenZipBackend::open(path)?)),
        ArchiveFormat::Rar => {
            #[cfg(feature = "archive-rar")]
            {
                Ok(Box::new(rar::RarBackend::open(path)?))
            }
            #[cfg(not(feature = "archive-rar"))]
            {
                Err(ArchiveError::RarNotCompiled {
                    path: path.display().to_string(),
                })
            }
        }
    }
}
