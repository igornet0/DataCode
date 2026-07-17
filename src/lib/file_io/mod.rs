//! Universal file I/O: `read()` and `save()` by extension.

mod path_input;
mod read_dispatch;
mod read_options;
mod save_dispatch;
mod table_readers;
pub mod value_serde;
mod xml_value;

pub use read_dispatch::{read_bytes_from_memory, read_value};
pub use read_options::ReadOptions;
pub use save_dispatch::save_value;
pub use path_input::{parse_lib_smb_path, path_from_value, read_bytes_from_path, write_bytes_to_path};
pub use table_readers::apply_header_filter;
