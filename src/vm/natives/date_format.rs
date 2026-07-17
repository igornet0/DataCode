//! Date parsing and formatting (`try_parse_date`, `parse_with_format`, `format_datetime`).

use crate::common::value::Value;
use chrono::{DateTime, FixedOffset, NaiveDate, NaiveDateTime, TimeZone, Utc};

fn zulu_offset() -> FixedOffset {
    FixedOffset::east_opt(0).expect("offset")
}

/// Parse common date/time strings (RFC3339, RFC2822, several naive formats).
pub fn try_parse_date(s: &str) -> Option<DateTime<FixedOffset>> {
    let s = s.trim();
    if s.len() < 8 || !s.chars().next().is_some_and(|c| c.is_ascii_digit()) {
        return None;
    }
    if let Ok(d) = DateTime::parse_from_rfc3339(s) {
        return Some(d);
    }
    if let Ok(d) = DateTime::parse_from_rfc2822(s) {
        return Some(d);
    }
    let zulu = zulu_offset();
    for fmt in [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y/%m/%d %H:%M:%S",
        "%d.%m.%Y %H:%M:%S",
        "%m/%d/%Y %H:%M:%S",
    ] {
        if let Ok(ndt) = NaiveDateTime::parse_from_str(s, fmt) {
            return Some(Utc.from_utc_datetime(&ndt).with_timezone(&zulu));
        }
    }
    for fmt in ["%Y-%m-%d", "%Y/%m/%d", "%d.%m.%Y", "%m/%d/%Y"] {
        if let Ok(nd) = NaiveDate::parse_from_str(s, fmt) {
            let ndt = nd.and_hms_opt(0, 0, 0)?;
            return Some(Utc.from_utc_datetime(&ndt).with_timezone(&zulu));
        }
    }
    None
}

/// Parse `s` with an explicit chrono/strftime `fmt` string.
pub fn parse_with_format(s: &str, fmt: &str) -> Option<DateTime<FixedOffset>> {
    let s = s.trim();
    if s.is_empty() || fmt.is_empty() {
        return None;
    }
    if let Ok(d) = DateTime::parse_from_str(s, fmt) {
        return Some(d);
    }
    let zulu = zulu_offset();
    if let Ok(ndt) = NaiveDateTime::parse_from_str(s, fmt) {
        return Some(Utc.from_utc_datetime(&ndt).with_timezone(&zulu));
    }
    if let Ok(nd) = NaiveDate::parse_from_str(s, fmt) {
        let ndt = nd.and_hms_opt(0, 0, 0)?;
        return Some(Utc.from_utc_datetime(&ndt).with_timezone(&zulu));
    }
    None
}

/// Format a datetime with chrono/strftime `fmt`.
pub fn format_datetime(dt: &DateTime<FixedOffset>, fmt: &str) -> Result<String, String> {
    Ok(dt.format(fmt).to_string())
}

fn date_format_error(msg: impl Into<String>) -> Value {
    crate::websocket::set_native_error(msg.into());
    Value::Null
}

fn first_arg_datetime(args: &[Value]) -> Option<DateTime<FixedOffset>> {
    match args.first() {
        Some(Value::Date(d)) => Some(*d),
        Some(Value::String(s)) => try_parse_date(s),
        _ => None,
    }
}

/// `parse_date(string, format)` → `date` or `null`.
pub fn native_parse_date(args: &[Value]) -> Value {
    if args.len() < 2 {
        return date_format_error("TypeError: parse_date() expects 2 arguments (string, format)");
    }
    let Value::String(s) = &args[0] else {
        return date_format_error("TypeError: parse_date() first argument must be a string");
    };
    let Value::String(fmt) = &args[1] else {
        return date_format_error("TypeError: parse_date() second argument must be a format string");
    };
    parse_with_format(s, fmt)
        .map(Value::Date)
        .unwrap_or(Value::Null)
}

/// `format_date(date, format)` → string or `null` (also used as `d.format(fmt)`).
pub fn native_format_date(args: &[Value]) -> Value {
    if args.len() < 2 {
        return date_format_error("TypeError: format_date() expects 2 arguments (date, format)");
    }
    let Value::String(fmt) = &args[1] else {
        return date_format_error("TypeError: format_date() second argument must be a format string");
    };
    let Some(dt) = first_arg_datetime(args) else {
        return date_format_error("TypeError: format_date() first argument must be a date or parseable date string");
    };
    match format_datetime(&dt, fmt) {
        Ok(s) => Value::String(s),
        Err(e) => date_format_error(format!("ValueError: {}", e)),
    }
}
