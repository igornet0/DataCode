//! Unified runtime type matching for annotations, isinstance, and error messages.

use crate::common::value::Value;
use crate::parser::ast::TypePart;

/// Language primitive / builtin typeof names (lowercase). User class names are not listed here.
const LANGUAGE_PRIMITIVE_TYPES: &[&str] = &[
    "int",
    "integer",
    "float",
    "num",
    "number",
    "str",
    "string",
    "bool",
    "boolean",
    "array",
    "list",
    "bytes",
    "iterable",
    "set",
    "tuple",
    "object",
    "dict",
    "dictionary",
    "table",
    "null",
    "none",
    "path",
    "function",
    "fn",
    "window",
    "image",
    "figure",
    "axis",
    "database_engine",
    "database_cluster",
    "column",
    "columns",
    "generator",
    "date",
    "duration",
    "uuid",
    "enumerate",
    "ellipsis",
    "plugin_opaque",
    "money",
];

/// True if `type_name` is a built-in language type, not a user-defined class name.
pub fn is_language_primitive_type(type_name: &str) -> bool {
    let lower = type_name.to_ascii_lowercase();
    LANGUAGE_PRIMITIVE_TYPES
        .iter()
        .any(|p| *p == lower.as_str())
}

/// `__class_name` on class instances.
pub fn instance_class_name(value: &Value) -> Option<String> {
    let Value::Object(map_rc) = value else {
        return None;
    };
    map_rc
        .borrow()
        .str_key_get("__class_name")
        .and_then(|v| match v {
            Value::String(s) => Some(s.clone()),
            _ => None,
        })
}

/// Whether `obj` is an instance of `target_class` (direct name, immediate superclass, ORM flags).
pub fn value_matches_class_annotation(obj: &Value, target_class: &str) -> bool {
    let Value::Object(map_rc) = obj else {
        return false;
    };
    let map = map_rc.borrow();
    if let Some(Value::String(ref cn)) = map.str_key_get("__class_name") {
        if cn == target_class {
            return true;
        }
        if let Some(Value::String(ref super_name)) = map.str_key_get("__superclass") {
            if super_name == target_class {
                return true;
            }
        }
        if target_class == "Table" {
            if let Some(Value::Bool(true)) = map.str_key_get("__extends_table") {
                return true;
            }
        }
        if target_class == "SQLEnum" {
            if let Some(Value::Bool(true)) = map.str_key_get("__extends_sqenum") {
                return true;
            }
        }
    }
    false
}

fn check_primitive_value_type(value: &Value, type_name_lower: &str) -> bool {
    if let Value::PluginOpaque { .. } = value {
        return type_name_lower == "plugin_opaque";
    }
    match (value, type_name_lower) {
        (Value::Int(_), "int" | "integer") => true,
        (Value::Int(_), "float" | "num" | "number") => true,
        (Value::Float(_), "float" | "num" | "number") => true,
        (Value::Float(_), "int" | "integer") => false,
        (Value::Number(n), "int" | "integer") => n.fract() == 0.0,
        (Value::Number(_), "float" | "num" | "number") => true,
        (Value::String(_), "str" | "string") => true,
        (Value::Bool(_), "bool" | "boolean") => true,
        (Value::Array(_) | Value::ArrayView(_) | Value::ObjectFieldList { .. }, "array" | "list") => {
            true
        }
        (Value::ByteBuffer(_), "bytes") => true,
        (Value::ByteBuffer(_), "array" | "list") => true,
        (Value::Iterable(_), "iterable") => true,
        (Value::Set(_), "set") => true,
        (Value::Tuple(_), "tuple") => true,
        (Value::Object(_), "object" | "dict" | "dictionary") => true,
        (Value::Object(map_rc), "table") => matches!(
            map_rc.borrow().str_key_get("__extends_table"),
            Some(Value::Bool(true))
        ),
        (Value::Table(_), "table") => true,
        (Value::Null, "null" | "none") => true,
        (Value::Path(_), "path") => true,
        (
            Value::Function(_) | Value::ModuleFunction { .. } | Value::NativeFunction(_),
            "function" | "fn",
        ) => true,
        (Value::Window(_), "window") => true,
        (Value::Image(_), "image") => true,
        (Value::Figure(_), "figure") => true,
        (Value::Axis(_), "axis") => true,
        (Value::DatabaseEngine(_), "database_engine") => true,
        (Value::DatabaseCluster(_), "database_cluster") => true,
        (Value::Archive(_), "archive") => true,
        (Value::DataSource(_), "datasource") => true,
        (Value::DataSourceResponse(_), "response") => true,
        (Value::HttpResponse(_), "http_response") => true,
        (Value::WebPage(_), "web_page") => true,
        (Value::WebElement(_), "web_element") => true,
        (Value::ColumnReference { .. }, "column") => true,
        (Value::ColumnsReference { .. }, "columns") => true,
        (Value::Generator(_), "generator") => true,
        (Value::Date(_), "date") => true,
        (Value::Duration(_), "duration") => true,
        (Value::Uuid(_, _), "uuid") => true,
        (Value::Enumerate { .. }, "enumerate") => true,
        (Value::Ellipsis, "ellipsis") => true,
        _ => false,
    }
}

/// User-defined class name: instance metadata and/or `class_chain` from globals (`get_superclass_chain`).
pub fn value_matches_user_class_name(
    value: &Value,
    type_name: &str,
    class_chain: Option<&[String]>,
) -> bool {
    if value_matches_class_annotation(value, type_name) {
        return true;
    }
    if let Some(chain) = class_chain {
        if chain.iter().any(|n| n == type_name) {
            return true;
        }
    }
    matches!(value, Value::Table(_) if type_name == "Table")
}

/// Match a value against a single type name (primitive or user class).
pub fn value_matches_type_name(
    value: &Value,
    type_name: &str,
    class_chain: Option<&[String]>,
) -> bool {
    if is_language_primitive_type(type_name) {
        check_primitive_value_type(value, &type_name.to_ascii_lowercase())
    } else {
        value_matches_user_class_name(value, type_name, class_chain)
    }
}

/// Does `value` satisfy one `TypePart` annotation?
pub fn value_matches_type_part(
    value: &Value,
    part: &TypePart,
    class_chain: Option<&[String]>,
) -> bool {
    match part {
        TypePart::LiteralStr(s) => matches!(value, Value::String(v) if v == s),
        TypePart::TypeName(n) => value_matches_type_name(value, n.as_str(), class_chain),
        TypePart::Union(alts) => alts
            .iter()
            .any(|p| value_matches_type_part(value, p, class_chain)),
        TypePart::Generic { base, args } => {
            if base.eq_ignore_ascii_case("optional") && args.len() == 1 {
                matches!(value, Value::Null)
                    || value_matches_type_part(value, &args[0], class_chain)
            } else {
                value_matches_type_name(value, base, class_chain)
            }
        }
    }
}

/// Does `value` satisfy a parameter type annotation (union at parameter level)?
pub fn value_matches_type_parts(
    value: &Value,
    type_parts: &[TypePart],
    class_chain: Option<&[String]>,
) -> bool {
    type_parts
        .iter()
        .any(|part| value_matches_type_part(value, part, class_chain))
}

/// Human-readable type for errors and typeof (class name when present on instances).
pub fn display_value_type(value: &Value) -> String {
    if let Some(class_name) = instance_class_name(value) {
        return class_name;
    }
    if let Value::Object(map_rc) = value {
        let map = map_rc.borrow();
        if let Some(Value::String(ns)) = map.str_key_get("__plugin_namespace") {
            return ns.clone();
        }
    }
    primitive_display_value_type(value).to_string()
}

/// Static typeof name for non-class values (legacy `get_type_name_value` behavior).
pub fn primitive_display_value_type(value: &Value) -> &'static str {
    match value {
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
        Value::Date(_) => "date",
        Value::Duration(_) => "duration",
        Value::String(_) => "str",
        Value::Array(_) | Value::ArrayView(_) | Value::ObjectFieldList { .. } => "array",
        Value::ByteBuffer(_) => "bytes",
        Value::Iterable(_) => "iterable",
        Value::Tuple(_) => "tuple",
        Value::Object(_) => "object",
        Value::Set(_) => "set",
        Value::Table(_) => "table",
        Value::Null => "null",
        Value::Path(_) => "path",
        Value::Uuid(_, _) => "uuid",
        Value::Function(_) | Value::ModuleFunction { .. } | Value::NativeFunction(_) => "function",
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
        Value::ColumnReference { .. } => "column",
        Value::ColumnsReference { .. } => "columns",
        Value::Enumerate { .. } => "enumerate",
        Value::Generator(_) => "generator",
        Value::Ellipsis => "ellipsis",
    }
}
