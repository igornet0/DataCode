//! String match helpers for table filter predicates and builtins.

use crate::common::value::Value;

/// Strict string match: both `cell` and `pattern` must be `Value::String`.
pub fn match_cell_string(cell: &Value, op: &str, pattern: &Value) -> bool {
    let Value::String(cell_str) = cell else {
        return false;
    };
    let Value::String(pattern_str) = pattern else {
        return false;
    };
    match op {
        "contains" => cell_str.contains(pattern_str.as_str()),
        "starts_with" => cell_str.starts_with(pattern_str.as_str()),
        "ends_with" => cell_str.ends_with(pattern_str.as_str()),
        _ => false,
    }
}
