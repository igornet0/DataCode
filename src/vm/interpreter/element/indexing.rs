//! GetArrayElement for Path, ColumnReference, String, Date, Figure, Axis, DatabaseEngine, DatabaseCluster, NativeFunction(str).

use std::cell::RefCell;
use std::collections::HashMap;
use std::path::PathBuf;
use std::rc::Rc;

use crate::common::array_slice::{contiguous_positive_slice_bounds, slice_indices};
use crate::common::numeric::integer_value_as_i64_if_whole;
use crate::common::{error::ErrorType, error::LangError, value::Value, value_store::ValueStore};
use crate::vm::exceptions::ExceptionHandler;
use crate::vm::frame::CallFrame;
use crate::vm::heavy_store::HeavyStore;
use crate::vm::stack;
use crate::vm::store_convert::store_value;
use crate::vm::table_ops;
use crate::vm::types::VMStatus;

/// Get Path property (is_file, is_dir, extension, name, parent, exists).
pub fn get_path(
    line: usize,
    stack: &mut Vec<crate::common::TaggedValue>,
    frames: &mut Vec<CallFrame>,
    exception_handlers: &mut Vec<ExceptionHandler>,
    value_store: &mut ValueStore,
    heavy_store: &mut HeavyStore,
    path: PathBuf,
    index_value: Value,
) -> Result<VMStatus, LangError> {
    match index_value {
        Value::String(property_name) => match property_name.as_str() {
            "is_file" => {
                stack::push_id(
                    stack,
                    store_value(Value::Bool(path.is_file()), value_store, heavy_store),
                );
            }
            "is_dir" => {
                stack::push_id(
                    stack,
                    store_value(Value::Bool(path.is_dir()), value_store, heavy_store),
                );
            }
            "extension" => {
                if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
                    stack::push_id(
                        stack,
                        store_value(Value::String(ext.to_string()), value_store, heavy_store),
                    );
                } else {
                    stack::push_id(stack, crate::common::value_store::NULL_VALUE_ID);
                }
            }
            "name" => {
                if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                    stack::push_id(
                        stack,
                        store_value(Value::String(name.to_string()), value_store, heavy_store),
                    );
                } else {
                    stack::push_id(stack, crate::common::value_store::NULL_VALUE_ID);
                }
            }
            "parent" => {
                use crate::vm::natives::path::safe_path_parent;
                match safe_path_parent(&path) {
                    Some(parent) => stack::push_id(
                        stack,
                        store_value(Value::Path(parent), value_store, heavy_store),
                    ),
                    None => stack::push_id(stack, crate::common::value_store::NULL_VALUE_ID),
                }
            }
            "exists" => {
                stack::push_id(
                    stack,
                    store_value(Value::Bool(path.exists()), value_store, heavy_store),
                );
            }
            _ => {
                let error = ExceptionHandler::runtime_error(
                    &frames,
                    format!("Property '{}' not found on Path", property_name),
                    line,
                );
                return match ExceptionHandler::handle_exception(
                    stack,
                    frames,
                    exception_handlers,
                    error,
                    value_store,
                    heavy_store,
                ) {
                    Ok(()) => Ok(VMStatus::Continue),
                    Err(e) => Err(e),
                };
            }
        },
        _ => {
            let error = ExceptionHandler::runtime_error(
                &frames,
                "Path property access requires string index".to_string(),
                line,
            );
            return match ExceptionHandler::handle_exception(
                stack,
                frames,
                exception_handlers,
                error,
                value_store,
                heavy_store,
            ) {
                Ok(()) => Ok(VMStatus::Continue),
                Err(e) => Err(e),
            };
        }
    }
    Ok(VMStatus::Continue)
}

/// Get `date` fields (`year`, …, `utc` / `to_utc`) as values — no call needed (`z.year` same as lowered `z.year()`).
pub fn get_date(
    line: usize,
    stack: &mut Vec<crate::common::TaggedValue>,
    frames: &mut Vec<CallFrame>,
    exception_handlers: &mut Vec<ExceptionHandler>,
    value_store: &mut ValueStore,
    heavy_store: &mut HeavyStore,
    dt: chrono::DateTime<chrono::FixedOffset>,
    index_value: Value,
) -> Result<VMStatus, LangError> {
    match index_value {
        Value::String(property_name) => {
            if property_name == "format" {
                stack::push_id(
                    stack,
                    store_value(
                        Value::NativeFunction(crate::vm::native_indices::builtin::FORMAT_DATE),
                        value_store,
                        heavy_store,
                    ),
                );
                return Ok(VMStatus::Continue);
            }
            if let Some(v) =
                crate::vm::natives::basic::date_property_value(dt, property_name.as_str())
            {
                stack::push_id(stack, store_value(v, value_store, heavy_store));
            } else {
                let error = ExceptionHandler::runtime_error(
                    &frames,
                    format!(
                        "Property '{}' not found on date. Available: year, month, quarter, day, hour, minute, second, weekday, to_utc, utc, format",
                        property_name
                    ),
                    line,
                );
                return match ExceptionHandler::handle_exception(
                    stack,
                    frames,
                    exception_handlers,
                    error,
                    value_store,
                    heavy_store,
                ) {
                    Ok(()) => Ok(VMStatus::Continue),
                    Err(e) => Err(e),
                };
            }
        }
        _ => {
            let error = ExceptionHandler::runtime_error(
                &frames,
                "Date field access requires a string key (year, month, quarter, day, hour, minute, second, weekday, to_utc, utc, format); numeric indexing is not supported"
                    .to_string(),
                line,
            );
            return match ExceptionHandler::handle_exception(
                stack,
                frames,
                exception_handlers,
                error,
                value_store,
                heavy_store,
            ) {
                Ok(()) => Ok(VMStatus::Continue),
                Err(e) => Err(e),
            };
        }
    }
    Ok(VMStatus::Continue)
}

/// Get ColumnReference cell by numeric index.
pub fn get_column_reference(
    line: usize,
    stack: &mut Vec<crate::common::TaggedValue>,
    frames: &mut Vec<CallFrame>,
    exception_handlers: &mut Vec<ExceptionHandler>,
    value_store: &mut ValueStore,
    heavy_store: &mut HeavyStore,
    table: Rc<RefCell<crate::common::table::Table>>,
    column_name: String,
    index_value: Value,
) -> Result<VMStatus, LangError> {
    let index = match index_value {
        Value::Number(n) => {
            let idx = n as i64;
            if idx < 0 {
                let error = ExceptionHandler::runtime_error(
                    &frames,
                    "Column index must be non-negative".to_string(),
                    line,
                );
                return match ExceptionHandler::handle_exception(
                    stack,
                    frames,
                    exception_handlers,
                    error,
                    value_store,
                    heavy_store,
                ) {
                    Ok(()) => Ok(VMStatus::Continue),
                    Err(e) => Err(e),
                };
            }
            idx as usize
        }
        _ => {
            let error = ExceptionHandler::runtime_error(
                &frames,
                "Column index must be a number".to_string(),
                line,
            );
            return match ExceptionHandler::handle_exception(
                stack,
                frames,
                exception_handlers,
                error,
                value_store,
                heavy_store,
            ) {
                Ok(()) => Ok(VMStatus::Continue),
                Err(e) => Err(e),
            };
        }
    };
    let pushed = if let Some(column) = table.borrow().get_column_cached(&column_name) {
        if index < column.len() {
            stack::push_id(
                stack,
                store_value(column[index].clone(), value_store, heavy_store),
            );
            true
        } else {
            false
        }
    } else {
        let t = table.borrow();
        if index < t.len() {
            if let Some(cell) =
                table_ops::get_cell_value(&*t, index, &column_name, value_store, heavy_store)
            {
                stack::push_id(stack, store_value(cell, value_store, heavy_store));
                true
            } else {
                false
            }
        } else {
            false
        }
    };
    if !pushed {
        let t = table.borrow();
        let has = t.has_column(&column_name);
        let len_opt = table_ops::column_len(&*t, &column_name);
        let error = if has {
            ExceptionHandler::runtime_error_with_type(
                &frames,
                format!(
                    "Column index {} out of bounds{}",
                    index,
                    len_opt
                        .map(|l| format!(" (length: {})", l))
                        .unwrap_or_default()
                ),
                line,
                ErrorType::IndexError,
            )
        } else {
            ExceptionHandler::runtime_error_with_type(
                &frames,
                format!("Column '{}' not found", column_name),
                line,
                ErrorType::KeyError,
            )
        };
        return match ExceptionHandler::handle_exception(
            stack,
            frames,
            exception_handlers,
            error,
            value_store,
            heavy_store,
        ) {
            Ok(()) => Ok(VMStatus::Continue),
            Err(e) => Err(e),
        };
    }
    Ok(VMStatus::Continue)
}

/// Get String character by index or method by name.
pub fn get_string(
    line: usize,
    stack: &mut Vec<crate::common::TaggedValue>,
    frames: &mut Vec<CallFrame>,
    exception_handlers: &mut Vec<ExceptionHandler>,
    value_store: &mut ValueStore,
    heavy_store: &mut HeavyStore,
    s: String,
    index_value: Value,
) -> Result<VMStatus, LangError> {
    if let Value::String(key) = index_value {
        use crate::vm::native_indices::builtin;
        let method_index = match key.as_str() {
            "upper" => Some(builtin::UPPER),
            "lower" => Some(builtin::LOWER),
            "trim" => Some(builtin::TRIM),
            "split" => Some(builtin::SPLIT),
            "join" => Some(builtin::JOIN),
            "contains" => Some(builtin::CONTAINS),
            "isupper" => Some(builtin::ISUPPER),
            "islower" => Some(builtin::ISLOWER),
            "replace" => Some(builtin::REPLACE),
            "capitalize" => Some(builtin::CAPITALIZE),
            _ => None,
        };
        if let Some(idx) = method_index {
            stack::push_id(
                stack,
                store_value(Value::NativeFunction(idx), value_store, heavy_store),
            );
        } else {
            let error = ExceptionHandler::runtime_error(
                &frames,
                format!("String has no property '{}'. Available: upper, lower, trim, split, join, contains, isupper, islower, replace, capitalize, or use numeric index", key),
                line,
            );
            return match ExceptionHandler::handle_exception(
                stack,
                frames,
                exception_handlers,
                error,
                value_store,
                heavy_store,
            ) {
                Ok(()) => Ok(VMStatus::Continue),
                Err(e) => Err(e),
            };
        }
        return Ok(VMStatus::Continue);
    }

    if let Some(idx) = integer_value_as_i64_if_whole(&index_value) {
        if idx < 0 {
            let error = ExceptionHandler::runtime_error(
                &frames,
                "String index must be non-negative".to_string(),
                line,
            );
            return match ExceptionHandler::handle_exception(
                stack,
                frames,
                exception_handlers,
                error,
                value_store,
                heavy_store,
            ) {
                Ok(()) => Ok(VMStatus::Continue),
                Err(e) => Err(e),
            };
        }
        let idx_usize = idx as usize;
        if let Some(ch) = s.chars().nth(idx_usize) {
            stack::push_id(
                stack,
                store_value(Value::String(ch.to_string()), value_store, heavy_store),
            );
        } else {
            let error = ExceptionHandler::runtime_error_with_type(
                &frames,
                format!(
                    "String index {} out of bounds (length: {})",
                    idx_usize,
                    s.chars().count()
                ),
                line,
                ErrorType::IndexError,
            );
            return match ExceptionHandler::handle_exception(
                stack,
                frames,
                exception_handlers,
                error,
                value_store,
                heavy_store,
            ) {
                Ok(()) => Ok(VMStatus::Continue),
                Err(e) => Err(e),
            };
        }
        return Ok(VMStatus::Continue);
    }

    let error = ExceptionHandler::runtime_error(
        &frames,
        "String index must be a number or a method name (upper, lower, trim, split, join, contains, isupper, islower)".to_string(),
        line,
    );
    match ExceptionHandler::handle_exception(
        stack,
        frames,
        exception_handlers,
        error,
        value_store,
        heavy_store,
    ) {
        Ok(()) => Ok(VMStatus::Continue),
        Err(e) => Err(e),
    }
}

/// Python-style substring slice; indices are Unicode scalar values (same as `s[i]`).
pub fn string_slice_value(
    s: &str,
    start: Option<i64>,
    stop: Option<i64>,
    step: Option<i64>,
) -> Result<Value, String> {
    let char_len = s.chars().count();
    let st = step.unwrap_or(1);
    if st == 1 {
        let (a, b) = contiguous_positive_slice_bounds(char_len, start, stop);
        let result: String = s.chars().skip(a).take(b.saturating_sub(a)).collect();
        return Ok(Value::String(result));
    }
    let indices = slice_indices(char_len, start, stop, st)?;
    let chars: Vec<char> = s.chars().collect();
    let result: String = indices.iter().map(|&i| chars[i]).collect();
    Ok(Value::String(result))
}

/// Get str[N] type descriptor when container is NativeFunction(str).
pub fn get_native_str(
    line: usize,
    stack: &mut Vec<crate::common::TaggedValue>,
    frames: &mut Vec<CallFrame>,
    exception_handlers: &mut Vec<ExceptionHandler>,
    value_store: &mut ValueStore,
    heavy_store: &mut HeavyStore,
    index_value: Value,
) -> Result<VMStatus, LangError> {
    if let Value::Number(n) = index_value {
        let idx = n as i64;
        if idx >= 0 && n.fract() == 0.0 {
            let mut desc = HashMap::new();
            desc.insert("__type".to_string(), Value::String("str".to_string()));
            desc.insert("__length".to_string(), Value::Number(idx as f64));
            stack::push_id(
                stack,
                store_value(
                    Value::legacy_object(desc),
                    value_store,
                    heavy_store,
                ),
            );
            return Ok(VMStatus::Continue);
        }
    }
    let error = ExceptionHandler::runtime_error(
        &frames,
        "Expected array, tuple, column reference, table, object, path, database engine, or database cluster for GetArrayElement".to_string(),
        line,
    );
    match ExceptionHandler::handle_exception(
        stack,
        frames,
        exception_handlers,
        error,
        value_store,
        heavy_store,
    ) {
        Ok(()) => Ok(VMStatus::Continue),
        Err(e) => Err(e),
    }
}

#[cfg(test)]
mod string_slice_tests {
    use super::string_slice_value;
    use crate::common::value::Value;

    fn slice_str(s: &str, start: Option<i64>, stop: Option<i64>, step: Option<i64>) -> String {
        match string_slice_value(s, start, stop, step) {
            Ok(Value::String(out)) => out,
            other => panic!("expected string slice, got {:?}", other),
        }
    }

    #[test]
    fn ascii_basic() {
        assert_eq!(slice_str("hello", Some(1), Some(4), None), "ell");
    }

    #[test]
    fn unicode_char_indices() {
        assert_eq!(slice_str("Привет", Some(1), Some(3), None), "ри");
    }

    #[test]
    fn negative_and_full() {
        assert_eq!(slice_str("hello", Some(-2), None, None), "lo");
        assert_eq!(slice_str("hello", None, None, None), "hello");
    }

    #[test]
    fn step_and_reverse() {
        assert_eq!(slice_str("abcdef", None, None, Some(2)), "ace");
        assert_eq!(slice_str("abc", None, None, Some(-1)), "cba");
    }

    #[test]
    fn empty_string() {
        assert_eq!(slice_str("", Some(0), Some(0), None), "");
    }

    #[test]
    fn step_zero_errors() {
        assert!(string_slice_value("abc", None, None, Some(0)).is_err());
    }
}
