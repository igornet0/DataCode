//! Total ordering for `<` / table `compare_values` across numbers, strings, tuples, etc.

use crate::common::numeric::cmp_numeric_values;
use crate::common::value::Value;
use std::cmp::Ordering;

/// Surface type label for ordering errors (aligned with `typeof` where practical).
pub fn value_ord_type_name(v: &Value) -> &'static str {
    match v {
        Value::Int(_) => "int",
        Value::Float(_) => "float",
        Value::Number(n) => {
            if n.fract() == 0.0 {
                "int"
            } else {
                "float"
            }
        }
        Value::Bool(_) => "bool",
        Value::String(_) => "string",
        Value::Tuple(_) => "tuple",
        Value::Array(_) | Value::ArrayView(_) | Value::ObjectFieldList { .. } => {
            "array"
        }
        Value::ByteBuffer(_) => "bytes",
        Value::Date(_) => "date",
        Value::Duration(_) => "duration",
        Value::Null => "null",
        Value::Table(_) => "table",
        Value::Object(_) => "object",
        Value::Set(_) => "set",
        Value::Path(_) => "path",
        Value::Uuid(_, _) => "uuid",
        Value::Function(_) | Value::ModuleFunction { .. } => "function",
        Value::NativeFunction(_) => "function",
        Value::Iterable(_) => "iterable",
        Value::Enumerate { .. } => "enumerate",
        Value::Generator(_) => "generator",
        Value::ColumnReference { .. } => "column",
        Value::ColumnsReference { .. } => "columns",
        Value::PluginOpaque { .. } => "plugin_opaque",
        Value::Window(_) => "window",
        Value::Image(_) => "image",
        Value::Figure(_) => "figure",
        Value::Axis(_) => "axis",
        Value::DatabaseEngine(_) => "database_engine",
        Value::DatabaseCluster(_) => "database_cluster",
        Value::Archive(_) => "archive",
        Value::DataSource(_) => "datasource",
        Value::DataSourceResponse(_) => "response",
        Value::HttpResponse(_) => "http_response",
        Value::WebPage(_) => "web_page",
        Value::WebElement(_) => "web_element",
        Value::Ellipsis => "ellipsis",
    }
}

/// Lexicographic tuple compare; numeric/string/bool/date/duration/null as in Python-style heaps.
///
/// Returns [`Err`] with a message suitable for `TypeError` when types are incomparable.
pub fn value_partial_cmp(a: &Value, b: &Value) -> Result<Ordering, String> {
    if let (Value::Tuple(ta), Value::Tuple(tb)) = (a, b) {
        let va = ta.borrow();
        let vb = tb.borrow();
        let n = va.len().min(vb.len());
        for i in 0..n {
            match value_partial_cmp(&va[i], &vb[i])? {
                Ordering::Equal => {}
                o => return Ok(o),
            }
        }
        return Ok(va.len().cmp(&vb.len()));
    }

    if let Some(o) = cmp_numeric_values(a, b) {
        return Ok(o);
    }

    if let (Value::String(sa), Value::String(sb)) = (a, b) {
        return Ok(sa.cmp(sb));
    }

    if let (Value::Bool(ba), Value::Bool(bb)) = (a, b) {
        return Ok(ba.cmp(bb));
    }

    if let (Value::Date(da), Value::Date(db)) = (a, b) {
        return Ok(da.cmp(db));
    }

    if let (Value::Duration(da), Value::Duration(db)) = (a, b) {
        return Ok(da.cmp(db));
    }

    match (a, b) {
        (Value::Null, Value::Null) => Ok(Ordering::Equal),
        (Value::Null, _) => Ok(Ordering::Less),
        (_, Value::Null) => Ok(Ordering::Greater),
        _ => Err(format!(
            "TypeError: '<' not supported between instances of '{}' and '{}'",
            value_ord_type_name(a),
            value_ord_type_name(b),
        )),
    }
}
