//! Import and `ImportFrom` opcode handlers.
//! Loads built-in, `.dc` file, and native modules; merges into caller's globals.

mod import_from_pipeline;
mod import_ops;

pub(crate) use import_ops::{handle_import, handle_import_from};
