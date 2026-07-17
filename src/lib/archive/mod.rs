//! Built-in Archive type for ZIP, 7Z, and RAR (optional) archives.

pub mod archive;
pub mod backend;
pub mod entry;
pub mod error;
pub mod format;
pub mod mmap;
pub mod natives;
pub mod path_safe;

pub use archive::Archive;
pub use format::ArchiveFormat;
