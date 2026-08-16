//! Central Datacode ↔ SQLite declared-type mapping and canonical ser/de.
//!
//! Declared types (`DATE`, `DATETIME`, …) document semantics for external SQLite
//! clients. `_datacode_schema` stores the exact Datacode type. Physical storage
//! still uses SQLite storage classes (often TEXT for dates).

use crate::common::numeric::IntValue;
use crate::common::table::{Table, TableData};
use crate::common::value::{ByteBuffer, Value};
use chrono::{FixedOffset, NaiveDate, NaiveTime, TimeZone, Timelike, Utc};
use rusqlite::types::{ToSqlOutput, Value as SqlValue, ValueRef};
use std::collections::HashMap;

pub const SCHEMA_VERSION: i64 = 1;

pub const TABLE_SCHEMA: &str = "_datacode_schema";
pub const TABLE_VERSION: &str = "_datacode_version";
pub const TABLE_VARIABLES: &str = "_datacode_variables";

/// Datacode logical types stored in `_datacode_schema.datacode_type`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DatacodeSqliteType {
    Int,
    Float,
    String,
    Bool,
    Bytes,
    Date,
    DateTime,
    Time,
    Duration,
}

impl DatacodeSqliteType {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Int => "int",
            Self::Float => "float",
            Self::String => "string",
            Self::Bool => "bool",
            Self::Bytes => "bytes",
            Self::Date => "date",
            Self::DateTime => "datetime",
            Self::Time => "time",
            Self::Duration => "duration",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "int" | "integer" => Some(Self::Int),
            "float" | "real" | "number" | "num" => Some(Self::Float),
            "string" | "str" | "text" => Some(Self::String),
            "bool" | "boolean" => Some(Self::Bool),
            "bytes" | "blob" | "bytebuffer" => Some(Self::Bytes),
            "date" => Some(Self::Date),
            "datetime" => Some(Self::DateTime),
            "time" => Some(Self::Time),
            "duration" => Some(Self::Duration),
            _ => None,
        }
    }

    pub fn sqlite_declared(self) -> &'static str {
        datacode_to_sqlite_declared(self.as_str())
    }
}

/// System tables use the `_datacode_` prefix.
pub fn is_datacode_system_table(name: &str) -> bool {
    name.starts_with("_datacode_")
}

/// Deterministic Datacode type → SQLite declared type.
pub fn datacode_to_sqlite_declared(dc: &str) -> &'static str {
    match DatacodeSqliteType::parse(dc) {
        Some(DatacodeSqliteType::Int) => "INTEGER",
        Some(DatacodeSqliteType::Float) => "REAL",
        Some(DatacodeSqliteType::String) => "TEXT",
        Some(DatacodeSqliteType::Bool) => "INTEGER",
        Some(DatacodeSqliteType::Bytes) => "BLOB",
        Some(DatacodeSqliteType::Date) => "DATE",
        Some(DatacodeSqliteType::DateTime) => "DATETIME",
        Some(DatacodeSqliteType::Time) => "TIME",
        Some(DatacodeSqliteType::Duration) => "INTEGER",
        None => "TEXT",
    }
}

/// Validate that datacode_type and sqlite_type match the central mapping.
pub fn validate_schema_type_pair(datacode_type: &str, sqlite_type: &str) -> Result<(), String> {
    let dc = DatacodeSqliteType::parse(datacode_type).ok_or_else(|| {
        format!("unknown datacode_type '{}'", datacode_type)
    })?;
    let expected = dc.sqlite_declared();
    let got = sqlite_type.trim().to_ascii_uppercase();
    // Allow affinity-compatible aliases (e.g. INT vs INTEGER) only for exact declared map.
    if got != expected {
        return Err(format!(
            "sqlite_type mismatch for datacode_type '{}': expected '{}', got '{}'",
            dc.as_str(),
            expected,
            sqlite_type
        ));
    }
    Ok(())
}

/// Fallback: SQLite declared type → Datacode type (when `_datacode_schema` is absent).
pub fn sqlite_declared_to_datacode(decl: &str) -> &'static str {
    let u = decl.trim().to_ascii_uppercase();
    // Strip length suffixes: VARCHAR(255), DECIMAL(10,2)
    let base = u.split('(').next().unwrap_or(&u).trim();
    match base {
        "INTEGER" | "INT" | "BIGINT" | "SMALLINT" | "TINYINT" | "MEDIUMINT" => "int",
        "REAL" | "DOUBLE" | "FLOAT" | "NUMERIC" | "DECIMAL" => "float",
        "BLOB" => "bytes",
        "DATE" => "date",
        "DATETIME" | "TIMESTAMP" => "datetime",
        "TIME" => "time",
        "BOOLEAN" | "BOOL" => "bool",
        "TEXT" | "VARCHAR" | "CHAR" | "CLOB" | "" => "string",
        _ => {
            // SQLite type affinity rules (simplified)
            if base.contains("INT") {
                "int"
            } else if base.contains("CHAR") || base.contains("CLOB") || base.contains("TEXT") {
                "string"
            } else if base.contains("BLOB") {
                "bytes"
            } else if base.contains("REAL") || base.contains("FLOA") || base.contains("DOUB") {
                "float"
            } else {
                "string"
            }
        }
    }
}

/// Content-based sniff for string cells (export / Arrow Utf8).
///
/// Returns `None` for empty/whitespace (treat as NULL skip) or unrecognized text.
pub fn sniff_string_datacode_type(s: &str) -> Option<&'static str> {
    let s = s.trim();
    if s.is_empty() {
        return None;
    }
    // Calendar date only: exactly YYYY-MM-DD
    if s.len() == 10
        && s.as_bytes().get(4) == Some(&b'-')
        && s.as_bytes().get(7) == Some(&b'-')
        && parse_date(s).is_some()
    {
        return Some("date");
    }
    // Datetime / RFC3339 (must be longer than a bare date)
    if s.len() > 10 && parse_datetime(s).is_some() {
        // Reject if parse_datetime only matched via date fallback on a non-datetime shape
        let has_time_sep = s.contains('T') || s.contains(' ');
        if has_time_sep {
            return Some("datetime");
        }
    }
    // Integer literal (no decimal / exponent)
    if !s.contains('.') && !s.contains('e') && !s.contains('E') {
        if s.parse::<i64>().is_ok() {
            return Some("int");
        }
    }
    // Float literal
    if let Ok(f) = s.parse::<f64>() {
        if f.is_finite() {
            return Some("float");
        }
    }
    None
}

/// Coerce a Utf8 cell into a typed Value when sniff succeeds.
pub fn coerce_utf8_to_value(s: &str) -> Value {
    match sniff_string_datacode_type(s) {
        Some("date") => parse_date(s.trim()).unwrap_or_else(|| Value::String(s.to_string())),
        Some("datetime") => {
            parse_datetime(s.trim()).unwrap_or_else(|| Value::String(s.to_string()))
        }
        Some("int") => s
            .trim()
            .parse::<i64>()
            .map(|i| Value::Number(i as f64))
            .unwrap_or_else(|_| Value::String(s.to_string())),
        Some("float") => s
            .trim()
            .parse::<f64>()
            .map(Value::Number)
            .unwrap_or_else(|_| Value::String(s.to_string())),
        _ => Value::String(s.to_string()),
    }
}

/// Infer Datacode logical type from a single Value.
pub fn infer_datacode_type_from_value(v: &Value) -> &'static str {
    match v {
        Value::Null => "string",
        Value::Bool(_) => "bool",
        Value::Int(_) => "int",
        Value::Float(_) => "float",
        Value::Number(n) => {
            if n.fract() == 0.0 {
                "int"
            } else {
                "float"
            }
        }
        Value::String(s) => sniff_string_datacode_type(s).unwrap_or("string"),
        Value::ByteBuffer(_) => "bytes",
        Value::Date(dt) => {
            let utc = dt.with_timezone(&Utc);
            if utc.time() == NaiveTime::from_hms_opt(0, 0, 0).unwrap()
                && utc.nanosecond() == 0
            {
                "date"
            } else {
                "datetime"
            }
        }
        Value::Duration(_) => "duration",
        _ => "string",
    }
}

fn accumulate_inferred_type(
    flags: &mut InferFlags,
    ty: &str,
) -> bool {
    match ty {
        "date" => {
            if flags.saw_int || flags.saw_float || flags.saw_bool || flags.saw_duration || flags.saw_bytes
            {
                return false;
            }
            flags.saw_date = true;
            true
        }
        "datetime" => {
            if flags.saw_int || flags.saw_float || flags.saw_bool || flags.saw_duration || flags.saw_bytes
            {
                return false;
            }
            flags.saw_datetime = true;
            true
        }
        "int" => {
            if flags.saw_date || flags.saw_datetime || flags.saw_duration || flags.saw_bytes {
                return false;
            }
            flags.saw_int = true;
            true
        }
        "float" => {
            if flags.saw_date || flags.saw_datetime || flags.saw_duration || flags.saw_bytes {
                return false;
            }
            flags.saw_float = true;
            true
        }
        "bool" => {
            if flags.saw_date || flags.saw_datetime || flags.saw_duration || flags.saw_bytes {
                return false;
            }
            flags.saw_bool = true;
            true
        }
        "duration" => {
            if flags.saw_int
                || flags.saw_float
                || flags.saw_bool
                || flags.saw_date
                || flags.saw_datetime
                || flags.saw_bytes
            {
                return false;
            }
            flags.saw_duration = true;
            true
        }
        "bytes" => {
            if flags.saw_int
                || flags.saw_float
                || flags.saw_bool
                || flags.saw_date
                || flags.saw_datetime
                || flags.saw_duration
            {
                return false;
            }
            flags.saw_bytes = true;
            true
        }
        _ => false,
    }
}

#[derive(Default)]
struct InferFlags {
    saw_float: bool,
    saw_int: bool,
    saw_bool: bool,
    saw_date: bool,
    saw_datetime: bool,
    saw_duration: bool,
    saw_bytes: bool,
}

fn resolve_infer_flags(flags: &InferFlags) -> &'static str {
    if flags.saw_bytes
        && !(flags.saw_int
            || flags.saw_float
            || flags.saw_bool
            || flags.saw_date
            || flags.saw_datetime
            || flags.saw_duration)
    {
        return "bytes";
    }
    if flags.saw_duration
        && !(flags.saw_int || flags.saw_float || flags.saw_bool || flags.saw_date || flags.saw_datetime)
    {
        return "duration";
    }
    if flags.saw_datetime {
        return "datetime";
    }
    if flags.saw_date {
        return "date";
    }
    if flags.saw_float {
        return "float";
    }
    if flags.saw_int || flags.saw_bool {
        if flags.saw_bool && !flags.saw_int {
            return "bool";
        }
        return "int";
    }
    "string"
}

/// Infer column Datacode type by scanning table cells (export path).
pub fn infer_datacode_type_for_header(table: &Table, header: &str) -> &'static str {
    let Some(col_idx) = table.headers().iter().position(|h| h == header) else {
        return "string";
    };
    match &table.data {
        TableData::Owned {
            flat,
            num_cols,
            ..
        } => {
            if *num_cols == 0 {
                return "string";
            }
            let nrows = flat.len() / num_cols;
            let mut flags = InferFlags::default();

            for row in 0..nrows {
                let value = flat.get(row * num_cols + col_idx).unwrap_or(&Value::Null);
                match value {
                    Value::Null => {}
                    Value::String(s) => {
                        if s.trim().is_empty() {
                            continue;
                        }
                        match sniff_string_datacode_type(s) {
                            Some(ty) => {
                                if !accumulate_inferred_type(&mut flags, ty) {
                                    return "string";
                                }
                            }
                            None => return "string",
                        }
                    }
                    Value::ByteBuffer(_) => {
                        if !accumulate_inferred_type(&mut flags, "bytes") {
                            return "string";
                        }
                    }
                    Value::Bool(_) => {
                        if !accumulate_inferred_type(&mut flags, "bool") {
                            return "string";
                        }
                    }
                    Value::Duration(_) => {
                        if !accumulate_inferred_type(&mut flags, "duration") {
                            return "string";
                        }
                    }
                    Value::Date(dt) => {
                        let utc = dt.with_timezone(&Utc);
                        let ty = if utc.time() == NaiveTime::from_hms_opt(0, 0, 0).unwrap()
                            && utc.nanosecond() == 0
                        {
                            "date"
                        } else {
                            "datetime"
                        };
                        if !accumulate_inferred_type(&mut flags, ty) {
                            return "string";
                        }
                    }
                    Value::Int(_) => {
                        if !accumulate_inferred_type(&mut flags, "int") {
                            return "string";
                        }
                    }
                    Value::Float(_) => {
                        if !accumulate_inferred_type(&mut flags, "float") {
                            return "string";
                        }
                    }
                    Value::Number(n) => {
                        let ty = if n.fract() == 0.0 { "int" } else { "float" };
                        if !accumulate_inferred_type(&mut flags, ty) {
                            return "string";
                        }
                    }
                    _ => return "string",
                }
            }

            resolve_infer_flags(&flags)
        }
        TableData::View { .. } => "string",
    }
}

/// Declared SQLite type for a table column (export DDL).
pub fn infer_sqlite_declared_for_header(table: &Table, header: &str) -> String {
    datacode_to_sqlite_declared(infer_datacode_type_for_header(table, header)).to_string()
}

/// ORM annotation / type name → declared SQLite type.
/// `date` maps to `DATETIME` because `Value::Date` is always an instant.
pub fn orm_type_name_to_sqlite_declared(type_name: &str) -> String {
    let lower = type_name.trim().to_ascii_lowercase();
    match lower.as_str() {
        "date" => "DATETIME".to_string(),
        "datetime" => "DATETIME".to_string(),
        "time" => "TIME".to_string(),
        "duration" => "INTEGER".to_string(),
        "bytes" | "blob" => "BLOB".to_string(),
        other => datacode_to_sqlite_declared(other).to_string(),
    }
}

/// Datacode type written to `_datacode_schema` for ORM column type name.
pub fn orm_type_name_to_datacode(type_name: &str) -> &'static str {
    let lower = type_name.trim().to_ascii_lowercase();
    match lower.as_str() {
        "date" | "datetime" => "datetime",
        "time" => "time",
        "duration" => "duration",
        "bytes" | "blob" => "bytes",
        other => DatacodeSqliteType::parse(other)
            .map(|t| t.as_str())
            .unwrap_or("string"),
    }
}

// ----- Canonical serialization -----

pub fn serialize_date(dt: &chrono::DateTime<FixedOffset>) -> String {
    let utc = dt.with_timezone(&Utc);
    utc.format("%Y-%m-%d").to_string()
}

pub fn serialize_datetime(dt: &chrono::DateTime<FixedOffset>) -> String {
    let utc = dt.with_timezone(&Utc);
    // Millisecond precision + Z
    let ms = utc.timestamp_subsec_millis();
    format!(
        "{}.{:03}Z",
        utc.format("%Y-%m-%dT%H:%M:%S"),
        ms
    )
}

pub fn serialize_time(dt: &chrono::DateTime<FixedOffset>) -> String {
    let utc = dt.with_timezone(&Utc);
    let ms = utc.timestamp_subsec_millis();
    if ms == 0 {
        utc.format("%H:%M:%S").to_string()
    } else {
        format!("{}.{:03}", utc.format("%H:%M:%S"), ms)
    }
}

pub fn serialize_duration_nanos(d: &chrono::Duration) -> i64 {
    d.num_nanoseconds().unwrap_or_else(|| {
        d.num_seconds().saturating_mul(1_000_000_000)
            + d.subsec_nanos() as i64
    })
}

pub fn parse_date(s: &str) -> Option<Value> {
    let naive = NaiveDate::parse_from_str(s.trim(), "%Y-%m-%d").ok()?;
    let dt = Utc
        .from_utc_datetime(&naive.and_hms_opt(0, 0, 0)?)
        .with_timezone(&FixedOffset::east_opt(0)?);
    Some(Value::Date(dt))
}

pub fn parse_datetime(s: &str) -> Option<Value> {
    let s = s.trim();
    if let Ok(dt) = chrono::DateTime::parse_from_rfc3339(s) {
        return Some(Value::Date(dt));
    }
    // Also accept space separator
    if let Ok(dt) = chrono::DateTime::parse_from_str(s, "%Y-%m-%d %H:%M:%S%.f%z") {
        return Some(Value::Date(dt));
    }
    if let Ok(naive) = chrono::NaiveDateTime::parse_from_str(s, "%Y-%m-%dT%H:%M:%S%.f") {
        let dt = Utc
            .from_utc_datetime(&naive)
            .with_timezone(&FixedOffset::east_opt(0)?);
        return Some(Value::Date(dt));
    }
    if let Ok(naive) = chrono::NaiveDateTime::parse_from_str(s, "%Y-%m-%d %H:%M:%S%.f") {
        let dt = Utc
            .from_utc_datetime(&naive)
            .with_timezone(&FixedOffset::east_opt(0)?);
        return Some(Value::Date(dt));
    }
    parse_date(s)
}

pub fn parse_time(s: &str) -> Option<Value> {
    let s = s.trim();
    let naive = NaiveTime::parse_from_str(s, "%H:%M:%S%.f")
        .or_else(|_| NaiveTime::parse_from_str(s, "%H:%M:%S"))
        .ok()?;
    let dt = Utc
        .from_utc_datetime(
            &NaiveDate::from_ymd_opt(1970, 1, 1)?
                .and_time(naive),
        )
        .with_timezone(&FixedOffset::east_opt(0)?);
    Some(Value::Date(dt))
}

pub fn parse_duration_nanos(n: i64) -> Value {
    Value::Duration(chrono::Duration::nanoseconds(n))
}

/// Bind a Value for INSERT given its Datacode logical type (from schema/inference).
pub fn value_to_sql_output<'a>(
    v: &'a Value,
    dc_type: Option<&str>,
) -> ToSqlOutput<'a> {
    if matches!(v, Value::Null) {
        return ToSqlOutput::Owned(SqlValue::Null);
    }
    let dc = dc_type
        .and_then(DatacodeSqliteType::parse)
        .unwrap_or_else(|| {
            DatacodeSqliteType::parse(infer_datacode_type_from_value(v)).unwrap_or(DatacodeSqliteType::String)
        });

    match (dc, v) {
        (DatacodeSqliteType::Bool, Value::Bool(b)) => {
            ToSqlOutput::Owned(SqlValue::Integer(if *b { 1 } else { 0 }))
        }
        (DatacodeSqliteType::Int, Value::Int(IntValue::Finite(i))) => {
            ToSqlOutput::Owned(SqlValue::Integer(*i))
        }
        (DatacodeSqliteType::Int, Value::Number(n)) => {
            ToSqlOutput::Owned(SqlValue::Integer(*n as i64))
        }
        (DatacodeSqliteType::Float, Value::Float(f)) => match f {
            crate::common::numeric::FloatValue::Finite(n) => {
                ToSqlOutput::Owned(SqlValue::Real(*n))
            }
            _ => ToSqlOutput::Owned(SqlValue::Null),
        },
        (DatacodeSqliteType::Float, Value::Number(n)) => {
            ToSqlOutput::Owned(SqlValue::Real(*n))
        }
        (DatacodeSqliteType::String, Value::String(s)) => {
            ToSqlOutput::Borrowed(ValueRef::Text(s.as_bytes()))
        }
        (DatacodeSqliteType::Bytes, Value::ByteBuffer(b)) => {
            let blob = b.bytes[b.offset..b.offset + b.len].to_vec();
            ToSqlOutput::Owned(SqlValue::Blob(blob))
        }
        (DatacodeSqliteType::Date, Value::Date(d)) => {
            ToSqlOutput::Owned(SqlValue::Text(serialize_date(d)))
        }
        (DatacodeSqliteType::Date, Value::String(s)) => match parse_date(s.trim()) {
            Some(Value::Date(d)) => ToSqlOutput::Owned(SqlValue::Text(serialize_date(&d))),
            _ => ToSqlOutput::Borrowed(ValueRef::Text(s.as_bytes())),
        },
        (DatacodeSqliteType::DateTime, Value::Date(d)) => {
            ToSqlOutput::Owned(SqlValue::Text(serialize_datetime(d)))
        }
        (DatacodeSqliteType::DateTime, Value::String(s)) => match parse_datetime(s.trim()) {
            Some(Value::Date(d)) => ToSqlOutput::Owned(SqlValue::Text(serialize_datetime(&d))),
            _ => ToSqlOutput::Borrowed(ValueRef::Text(s.as_bytes())),
        },
        (DatacodeSqliteType::Time, Value::Date(d)) => {
            ToSqlOutput::Owned(SqlValue::Text(serialize_time(d)))
        }
        (DatacodeSqliteType::Time, Value::String(s)) => match parse_time(s.trim()) {
            Some(Value::Date(d)) => ToSqlOutput::Owned(SqlValue::Text(serialize_time(&d))),
            _ => ToSqlOutput::Borrowed(ValueRef::Text(s.as_bytes())),
        },
        (DatacodeSqliteType::Int, Value::String(s)) => match s.trim().parse::<i64>() {
            Ok(i) => ToSqlOutput::Owned(SqlValue::Integer(i)),
            Err(_) => ToSqlOutput::Borrowed(ValueRef::Text(s.as_bytes())),
        },
        (DatacodeSqliteType::Float, Value::String(s)) => match s.trim().parse::<f64>() {
            Ok(f) if f.is_finite() => ToSqlOutput::Owned(SqlValue::Real(f)),
            _ => ToSqlOutput::Borrowed(ValueRef::Text(s.as_bytes())),
        },
        (DatacodeSqliteType::Duration, Value::Duration(d)) => {
            ToSqlOutput::Owned(SqlValue::Integer(serialize_duration_nanos(d)))
        }
        // Fallbacks by value shape
        (_, Value::Date(d)) => ToSqlOutput::Owned(SqlValue::Text(serialize_datetime(d))),
        (_, Value::Duration(d)) => {
            ToSqlOutput::Owned(SqlValue::Integer(serialize_duration_nanos(d)))
        }
        (_, Value::Bool(b)) => ToSqlOutput::Owned(SqlValue::Integer(if *b { 1 } else { 0 })),
        (_, Value::Number(n)) if n.fract() == 0.0 => {
            ToSqlOutput::Owned(SqlValue::Integer(*n as i64))
        }
        (_, Value::Number(n)) => ToSqlOutput::Owned(SqlValue::Real(*n)),
        (_, Value::Int(IntValue::Finite(i))) => ToSqlOutput::Owned(SqlValue::Integer(*i)),
        (_, Value::String(s)) => ToSqlOutput::Borrowed(ValueRef::Text(s.as_bytes())),
        (_, Value::ByteBuffer(b)) => {
            let blob = b.bytes[b.offset..b.offset + b.len].to_vec();
            ToSqlOutput::Owned(SqlValue::Blob(blob))
        }
        (_, other) => ToSqlOutput::Owned(SqlValue::Text(format!("{:?}", other))),
    }
}

/// Owned bind value for rusqlite (engine path).
pub fn value_to_owned_sql(v: &Value, dc_type: Option<&str>) -> SqlValue {
    match value_to_sql_output(v, dc_type) {
        ToSqlOutput::Owned(v) => v,
        ToSqlOutput::Borrowed(ValueRef::Text(t)) => {
            SqlValue::Text(String::from_utf8_lossy(t).into_owned())
        }
        ToSqlOutput::Borrowed(ValueRef::Blob(b)) => SqlValue::Blob(b.to_vec()),
        ToSqlOutput::Borrowed(ValueRef::Null) => SqlValue::Null,
        ToSqlOutput::Borrowed(ValueRef::Integer(i)) => SqlValue::Integer(i),
        ToSqlOutput::Borrowed(ValueRef::Real(r)) => SqlValue::Real(r),
        _ => SqlValue::Null,
    }
}

/// Restore Datacode Value from SQLite cell using metadata type then declared type.
pub fn sql_value_to_datacode(
    sql_val: SqlValue,
    datacode_type: Option<&str>,
    declared_type: Option<&str>,
) -> Value {
    if matches!(sql_val, SqlValue::Null) {
        return Value::Null;
    }
    let dc = datacode_type
        .and_then(DatacodeSqliteType::parse)
        .or_else(|| {
            declared_type
                .map(sqlite_declared_to_datacode)
                .and_then(DatacodeSqliteType::parse)
        });

    match dc {
        Some(DatacodeSqliteType::Date) => match &sql_val {
            SqlValue::Text(s) => parse_date(s).unwrap_or_else(|| Value::String(s.clone())),
            _ => storage_class_to_value(sql_val),
        },
        Some(DatacodeSqliteType::DateTime) => match &sql_val {
            SqlValue::Text(s) => parse_datetime(s).unwrap_or_else(|| Value::String(s.clone())),
            _ => storage_class_to_value(sql_val),
        },
        Some(DatacodeSqliteType::Time) => match &sql_val {
            SqlValue::Text(s) => parse_time(s).unwrap_or_else(|| Value::String(s.clone())),
            _ => storage_class_to_value(sql_val),
        },
        Some(DatacodeSqliteType::Duration) => match &sql_val {
            SqlValue::Integer(n) => parse_duration_nanos(*n),
            SqlValue::Real(r) => {
                // Legacy REAL seconds from old export
                let nanos = (*r * 1_000_000_000.0) as i64;
                parse_duration_nanos(nanos)
            }
            SqlValue::Text(s) => s
                .parse::<i64>()
                .map(parse_duration_nanos)
                .unwrap_or_else(|_| Value::String(s.clone())),
            _ => storage_class_to_value(sql_val),
        },
        Some(DatacodeSqliteType::Bool) => match &sql_val {
            SqlValue::Integer(i) => Value::Bool(*i != 0),
            SqlValue::Real(r) => Value::Bool(*r != 0.0),
            _ => storage_class_to_value(sql_val),
        },
        Some(DatacodeSqliteType::Int) => match &sql_val {
            SqlValue::Integer(i) => Value::Int(IntValue::Finite(*i)),
            SqlValue::Real(r) if r.fract() == 0.0 => Value::Int(IntValue::Finite(*r as i64)),
            _ => storage_class_to_value(sql_val),
        },
        Some(DatacodeSqliteType::Float) => match &sql_val {
            SqlValue::Real(r) => Value::Number(*r),
            SqlValue::Integer(i) => Value::Number(*i as f64),
            _ => storage_class_to_value(sql_val),
        },
        Some(DatacodeSqliteType::Bytes) => match sql_val {
            SqlValue::Blob(b) => Value::ByteBuffer(ByteBuffer::from_vec(b)),
            other => storage_class_to_value(other),
        },
        Some(DatacodeSqliteType::String) | None => storage_class_to_value(sql_val),
    }
}

fn storage_class_to_value(sql_val: SqlValue) -> Value {
    match sql_val {
        SqlValue::Integer(i) => Value::Number(i as f64),
        SqlValue::Real(r) => Value::Number(r),
        SqlValue::Text(s) => Value::String(s),
        SqlValue::Blob(bytes) => Value::ByteBuffer(ByteBuffer::from_vec(bytes)),
        SqlValue::Null => Value::Null,
    }
}

/// One column entry from `_datacode_schema` or discovery.
#[derive(Debug, Clone)]
pub struct SchemaColumn {
    pub table_name: String,
    pub column_name: String,
    pub datacode_type: String,
    pub sqlite_type: String,
    pub nullable: bool,
}

pub type SchemaCache = HashMap<(String, String), SchemaColumn>;

/// Discover column types from `PRAGMA table_info` when `_datacode_schema` is absent.
pub fn schema_discovery_from_pragma(conn: &rusqlite::Connection, table: &str) -> Vec<SchemaColumn> {
    let mut out = Vec::new();
    // Quote table name safely for PRAGMA (identifiers only).
    let safe: String = table
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() || c == '_' { c } else { '_' })
        .collect();
    let sql = format!("PRAGMA table_info(\"{}\")", safe.replace('"', "\"\""));
    let Ok(mut stmt) = conn.prepare(&sql) else {
        return out;
    };
    let Ok(rows) = stmt.query_map([], |row| {
        let name: String = row.get(1)?;
        let decl: String = row.get::<_, Option<String>>(2)?.unwrap_or_default();
        let notnull: i64 = row.get(3).unwrap_or(0);
        let dc = sqlite_declared_to_datacode(&decl);
        Ok(SchemaColumn {
            table_name: table.to_string(),
            column_name: name,
            datacode_type: dc.to_string(),
            sqlite_type: if decl.is_empty() {
                "TEXT".to_string()
            } else {
                decl.split('(')
                    .next()
                    .unwrap_or(&decl)
                    .trim()
                    .to_ascii_uppercase()
            },
            nullable: notnull == 0,
        })
    }) else {
        return out;
    };
    for row in rows.flatten() {
        out.push(row);
    }
    out
}

/// DDL to create system schema tables.
pub fn ensure_system_tables_sql() -> &'static str {
    "
    CREATE TABLE IF NOT EXISTS _datacode_version (
        version INTEGER NOT NULL
    );
    CREATE TABLE IF NOT EXISTS _datacode_schema (
        table_name TEXT NOT NULL,
        column_name TEXT NOT NULL,
        datacode_type TEXT NOT NULL,
        sqlite_type TEXT NOT NULL,
        nullable INTEGER NOT NULL DEFAULT 1,
        version INTEGER NOT NULL DEFAULT 1,
        PRIMARY KEY (table_name, column_name)
    );
    "
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    #[test]
    fn mapping_date_datetime_duration() {
        assert_eq!(datacode_to_sqlite_declared("date"), "DATE");
        assert_eq!(datacode_to_sqlite_declared("datetime"), "DATETIME");
        assert_eq!(datacode_to_sqlite_declared("time"), "TIME");
        assert_eq!(datacode_to_sqlite_declared("duration"), "INTEGER");
        assert_eq!(datacode_to_sqlite_declared("bytes"), "BLOB");
        assert_eq!(datacode_to_sqlite_declared("bool"), "INTEGER");
    }

    #[test]
    fn declared_fallback() {
        assert_eq!(sqlite_declared_to_datacode("DATE"), "date");
        assert_eq!(sqlite_declared_to_datacode("DATETIME"), "datetime");
        assert_eq!(sqlite_declared_to_datacode("TIME"), "time");
        assert_eq!(sqlite_declared_to_datacode("TEXT"), "string");
    }

    #[test]
    fn validate_pair_ok() {
        assert!(validate_schema_type_pair("date", "DATE").is_ok());
        assert!(validate_schema_type_pair("date", "TEXT").is_err());
    }

    #[test]
    fn system_table_prefix() {
        assert!(is_datacode_system_table("_datacode_schema"));
        assert!(!is_datacode_system_table("users"));
    }

    #[test]
    fn datetime_roundtrip_canonical() {
        let dt = Utc
            .with_ymd_and_hms(2026, 8, 3, 12, 0, 0)
            .unwrap()
            .with_timezone(&FixedOffset::east_opt(0).unwrap());
        let s = serialize_datetime(&dt);
        assert!(s.ends_with('Z'));
        let back = parse_datetime(&s).unwrap();
        match back {
            Value::Date(d) => assert_eq!(d.with_timezone(&Utc), dt.with_timezone(&Utc)),
            _ => panic!("expected Date"),
        }
    }

    #[test]
    fn date_midnight_inference() {
        let dt = Utc
            .with_ymd_and_hms(2026, 8, 3, 0, 0, 0)
            .unwrap()
            .with_timezone(&FixedOffset::east_opt(0).unwrap());
        assert_eq!(infer_datacode_type_from_value(&Value::Date(dt)), "date");
    }

    #[test]
    fn sniff_date_datetime_numbers() {
        assert_eq!(sniff_string_datacode_type("2024-01-15"), Some("date"));
        assert_eq!(
            sniff_string_datacode_type("2026-08-02T13:20:21.571931+00:00"),
            Some("datetime")
        );
        assert_eq!(sniff_string_datacode_type("1"), Some("int"));
        assert_eq!(sniff_string_datacode_type("233.4"), Some("float"));
        assert_eq!(sniff_string_datacode_type("bf96cfeb:power"), None);
        assert_eq!(sniff_string_datacode_type("  "), None);
    }

    #[test]
    fn infer_string_column_types() {
        let table = Table::from_data(
            vec![
                vec![
                    Value::String("1".into()),
                    Value::String("233.4".into()),
                    Value::String("2024-01-15".into()),
                    Value::String("2026-08-02T13:20:21.571Z".into()),
                    Value::String("id:x".into()),
                ],
                vec![
                    Value::String("2".into()),
                    Value::String("26.1".into()),
                    Value::String("2024-01-16".into()),
                    Value::String("2026-08-02T14:00:00.000Z".into()),
                    Value::String("id:y".into()),
                ],
            ],
            Some(vec![
                "row_id".into(),
                "value".into(),
                "day".into(),
                "ts".into(),
                "id".into(),
            ]),
        );
        assert_eq!(infer_datacode_type_for_header(&table, "row_id"), "int");
        assert_eq!(infer_datacode_type_for_header(&table, "value"), "float");
        assert_eq!(infer_datacode_type_for_header(&table, "day"), "date");
        assert_eq!(infer_datacode_type_for_header(&table, "ts"), "datetime");
        assert_eq!(infer_datacode_type_for_header(&table, "id"), "string");
    }

    #[test]
    fn infer_mixed_column_falls_back_to_string() {
        let table = Table::from_data(
            vec![
                vec![Value::String("2024-01-15".into())],
                vec![Value::String("not-a-date".into())],
            ],
            Some(vec!["col".into()]),
        );
        assert_eq!(infer_datacode_type_for_header(&table, "col"), "string");
    }
}
