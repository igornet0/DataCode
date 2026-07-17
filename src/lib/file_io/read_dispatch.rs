//! Universal `read()` dispatch by file extension.

use crate::common::value::{ByteBuffer, Value};
use crate::file_io::path_input::{
    extension_lower, format_path_for_error, path_from_value, read_bytes_from_path,
};
use crate::file_io::read_options::{read_options_from_args, ReadOptions};
use crate::file_io::table_readers::{
    apply_header_filter, finish_transposed_read, read_csv_bytes, read_csv_bytes_raw,
    read_xlsx_bytes, read_xlsx_bytes_raw,
};
use crate::file_io::value_serde::{parse_json_str, parse_toml_str, parse_xml_str, parse_yaml_str};
use crate::plot::Image;
use std::cell::RefCell;
use std::path::{Path, PathBuf};
use std::rc::Rc;

fn err(path: &PathBuf, ext: &str, msg: impl Into<String>) -> String {
    format!(
        "Cannot read file '{}' (extension '{}'): {}",
        format_path_for_error(path),
        ext,
        msg.into()
    )
}

pub fn read_value(args: &[Value]) -> Result<Value, String> {
    if args.is_empty() {
        return Err("read() expects a path argument".to_string());
    }
    let path = path_from_value(&args[0])?;
    let opts = read_options_from_args(args)?;
    let display_path = path.clone();

    let bytes = read_bytes_from_path(&path)?;
    read_bytes_from_memory(&display_path, &bytes, &opts)
}

/// Read bytes in memory with the same extension dispatch as [`read_value`].
pub fn read_bytes_from_memory(
    path: &Path,
    bytes: &[u8],
    opts: &ReadOptions,
) -> Result<Value, String> {
    let ext = extension_lower(path);
    read_bytes_with_ext(&path.to_path_buf(), &ext, bytes, opts)
}

fn read_bytes_with_ext(
    path: &PathBuf,
    ext: &str,
    bytes: &[u8],
    opts: &ReadOptions,
) -> Result<Value, String> {
    match ext {
        "csv" => read_tabular_csv(path, ext, bytes, opts),
        "xlsx" => read_tabular_xlsx(path, ext, bytes, opts),
        "json" => {
            let text = std::str::from_utf8(bytes)
                .map_err(|e| err(path, ext, format!("invalid UTF-8: {}", e)))?;
            parse_json_str(text).map_err(|e| err(path, ext, e.message()))
        }
        "toml" => {
            let text = std::str::from_utf8(bytes)
                .map_err(|e| err(path, ext, format!("invalid UTF-8: {}", e)))?;
            parse_toml_str(text).map_err(|e| err(path, ext, e.message()))
        }
        "yaml" | "yml" => {
            let text = std::str::from_utf8(bytes)
                .map_err(|e| err(path, ext, format!("invalid UTF-8: {}", e)))?;
            parse_yaml_str(text).map_err(|e| err(path, ext, e.message()))
        }
        "txt" | "text" | "md" | "html" => {
            let text = String::from_utf8(bytes.to_vec())
                .map_err(|e| err(path, ext, format!("invalid UTF-8: {}", e)))?;
            Ok(Value::String(text))
        }
        "bin" => Ok(Value::ByteBuffer(ByteBuffer::from_vec(bytes.to_vec()))),
        "png" | "jpg" | "jpeg" | "gif" | "webp" | "bmp" => {
            Image::from_bytes(bytes)
                .map(|img| Value::Image(Rc::new(RefCell::new(img))))
                .map_err(|e| err(path, ext, e))
        }
        "xml" => {
            let text = std::str::from_utf8(bytes)
                .map_err(|e| err(path, ext, format!("invalid UTF-8: {}", e)))?;
            parse_xml_str(text).map_err(|e| err(path, ext, e.message()))
        }
        _ => {
            let text = String::from_utf8(bytes.to_vec())
                .map_err(|e| err(path, ext, format!("invalid UTF-8: {}", e)))?;
            Ok(Value::String(text))
        }
    }
}

fn read_tabular_csv(
    path: &PathBuf,
    ext: &str,
    bytes: &[u8],
    opts: &ReadOptions,
) -> Result<Value, String> {
    let table = if opts.uses_transpose() {
        read_csv_bytes_raw(bytes).map_err(|e| err(path, ext, e.to_string()))?
    } else {
        read_csv_bytes(bytes).map_err(|e| err(path, ext, e.to_string()))?
    };

    let filtered = if opts.uses_transpose() {
        finish_transposed_read(
            table,
            opts.header_row,
            opts.header_t_filter.as_ref(),
        )
    } else {
        apply_header_filter(table, opts.header_filter.as_ref())
    };
    Ok(Value::Table(Rc::new(RefCell::new(filtered))))
}

fn read_tabular_xlsx(
    path: &PathBuf,
    ext: &str,
    bytes: &[u8],
    opts: &ReadOptions,
) -> Result<Value, String> {
    let table = if opts.uses_transpose() {
        read_xlsx_bytes_raw(bytes, opts.sheet_name.as_deref())
            .map_err(|e| err(path, ext, format!("{}", e)))?
    } else {
        read_xlsx_bytes(bytes, opts.header_row, opts.sheet_name.as_deref())
            .map_err(|e| err(path, ext, format!("{}", e)))?
    };

    let filtered = if opts.uses_transpose() {
        finish_transposed_read(
            table,
            opts.header_row,
            opts.header_t_filter.as_ref(),
        )
    } else {
        apply_header_filter(table, opts.header_filter.as_ref())
    };
    Ok(Value::Table(Rc::new(RefCell::new(filtered))))
}
