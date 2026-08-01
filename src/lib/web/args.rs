//! Argument helpers for `web` natives.

use crate::common::numeric::{FloatValue, IntValue};
use crate::common::value::{ObjectKind, Value};
use crate::web::error::WebError;
use std::collections::HashMap;

pub fn require_string(args: &[Value], idx: usize, name: &str) -> Result<String, WebError> {
    match args.get(idx) {
        Some(Value::String(s)) => Ok(s.clone()),
        Some(Value::Null) | None => Err(WebError::type_err(format!(
            "expected string {name} at argument {}",
            idx + 1
        ))),
        Some(other) => Err(WebError::type_err(format!(
            "expected string {name}, got {}",
            other.to_string()
        ))),
    }
}

pub fn optional_string(args: &[Value], idx: usize) -> Option<String> {
    match args.get(idx) {
        Some(Value::String(s)) => Some(s.clone()),
        _ => None,
    }
}

fn value_as_f64(v: &Value) -> Option<f64> {
    match v {
        Value::Number(n) => Some(*n),
        Value::Int(IntValue::Finite(n)) => Some(*n as f64),
        Value::Int(IntValue::PosInfinity) => Some(f64::INFINITY),
        Value::Int(IntValue::NegInfinity) => Some(f64::NEG_INFINITY),
        Value::Float(FloatValue::Finite(n)) => Some(*n),
        Value::Float(FloatValue::PosInfinity) => Some(f64::INFINITY),
        Value::Float(FloatValue::NegInfinity) => Some(f64::NEG_INFINITY),
        Value::Float(FloatValue::NaN) => Some(f64::NAN),
        _ => None,
    }
}

pub fn optional_f64(args: &[Value], idx: usize) -> Option<f64> {
    args.get(idx).and_then(value_as_f64)
}

pub fn value_as_string_map(v: &Value) -> Result<HashMap<String, String>, WebError> {
    let Value::Object(rc) = v else {
        return Err(WebError::type_err("expected object for string map"));
    };
    let mut out = HashMap::new();
    match &*rc.borrow() {
        ObjectKind::Legacy(m) => {
            for (k, val) in m {
                out.insert(k.clone(), value_to_plain_string(val));
            }
        }
        ObjectKind::Inline(entries) => {
            for (k, val) in entries {
                if let Value::String(sk) = k {
                    out.insert(sk.clone(), value_to_plain_string(val));
                }
            }
        }
        ObjectKind::Bucket(_) => {
            return Err(WebError::type_err(
                "web options must use a plain object literal (string keys)",
            ));
        }
    }
    Ok(out)
}

pub fn value_to_plain_string(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Null => String::new(),
        other => other.to_string(),
    }
}

/// Read string-keyed entries from a Legacy/Inline object into a HashMap of Values.
pub fn object_entries(v: &Value) -> Result<HashMap<String, Value>, WebError> {
    let Value::Object(rc) = v else {
        return Err(WebError::type_err("expected object"));
    };
    let mut out = HashMap::new();
    match &*rc.borrow() {
        ObjectKind::Legacy(m) => {
            for (k, val) in m {
                out.insert(k.clone(), val.clone());
            }
        }
        ObjectKind::Inline(entries) => {
            for (k, val) in entries {
                if let Value::String(sk) = k {
                    out.insert(sk.clone(), val.clone());
                }
            }
        }
        ObjectKind::Bucket(_) => {
            return Err(WebError::type_err(
                "web options must use a plain object literal (string keys)",
            ));
        }
    }
    Ok(out)
}

/// Parse HTTP call options from remaining args after URL.
///
/// Supports:
/// - single options object: `{ params, headers, body, json, timeout }`
/// - positional slots in order: params, headers, body, json, timeout
pub fn parse_http_options(args: &[Value]) -> Result<HttpOptions, WebError> {
    let mut opts = HttpOptions::default();
    if args.is_empty() {
        return Ok(opts);
    }
    if args.len() == 1 {
        if let Value::Object(_) = &args[0] {
            return parse_http_options_object(&args[0]);
        }
    }
    // Positional optional: params, headers, body, json, timeout
    if let Some(v) = args.get(0) {
        if !matches!(v, Value::Null) {
            opts.params = Some(value_as_string_map(v)?);
        }
    }
    if let Some(v) = args.get(1) {
        if !matches!(v, Value::Null) {
            opts.headers = Some(value_as_string_map(v)?);
        }
    }
    if let Some(v) = args.get(2) {
        if !matches!(v, Value::Null) {
            opts.body = Some(value_to_plain_string(v));
        }
    }
    if let Some(v) = args.get(3) {
        if !matches!(v, Value::Null) {
            opts.json = Some(v.clone());
        }
    }
    if let Some(v) = args.get(4) {
        if !matches!(v, Value::Null) {
            opts.timeout = Some(
                value_as_f64(v).ok_or_else(|| WebError::type_err("timeout must be a number"))?,
            );
        }
    }
    Ok(opts)
}

fn parse_http_options_object(v: &Value) -> Result<HttpOptions, WebError> {
    let entries = object_entries(v)?;
    let mut opts = HttpOptions::default();
    if let Some(p) = entries.get("params") {
        if !matches!(p, Value::Null) {
            opts.params = Some(value_as_string_map(p)?);
        }
    }
    if let Some(h) = entries.get("headers") {
        if !matches!(h, Value::Null) {
            opts.headers = Some(value_as_string_map(h)?);
        }
    }
    if let Some(b) = entries.get("body") {
        if !matches!(b, Value::Null) {
            opts.body = Some(value_to_plain_string(b));
        }
    }
    if let Some(j) = entries.get("json") {
        if !matches!(j, Value::Null) {
            opts.json = Some(j.clone());
        }
    }
    if let Some(t) = entries.get("timeout") {
        if !matches!(t, Value::Null) {
            opts.timeout = Some(
                value_as_f64(t).ok_or_else(|| WebError::type_err("timeout must be a number"))?,
            );
        }
    }
    Ok(opts)
}

#[derive(Debug, Default)]
pub struct HttpOptions {
    pub params: Option<HashMap<String, String>>,
    pub headers: Option<HashMap<String, String>>,
    pub body: Option<String>,
    pub json: Option<Value>,
    pub timeout: Option<f64>,
}

/// Parse `browser.open` options from args after URL.
///
/// Accepts:
/// - `browser.open(url, { stealth: true, profile: "..." })`
/// - `browser.open(url, options={ stealth: true, profile: "..." })`
/// - `browser.open(url, stealth=true)` (when compiler resolves named kwargs)
/// - positional: headless, user_agent, headers, stealth, options, profile
pub fn parse_browser_options(args: &[Value]) -> Result<BrowserOptions, WebError> {
    if args.is_empty() {
        return Ok(BrowserOptions::default());
    }

    // Single object → full options bag (or nested `options`).
    if args.len() == 1 {
        if let Value::Object(_) = &args[0] {
            return parse_browser_options_object(&args[0]);
        }
    }

    // If any non-null arg is an options-like object, merge it.
    for v in args {
        if let Value::Object(_) = v {
            if looks_like_browser_options(v) {
                let mut opts = parse_browser_options_object(v)?;
                // Still allow earlier positional headless bool if present.
                if let Some(Value::Bool(b)) = args.first() {
                    // Only override headless from positional when object didn't set it
                    // and first slot is bool (legacy).
                    if !object_entries(v)?.contains_key("headless") {
                        opts.headless = *b;
                    }
                }
                return Ok(opts);
            }
        }
    }

    // Positional: headless, user_agent, headers, stealth, options, profile
    let mut opts = BrowserOptions::default();
    if let Some(v) = args.get(0) {
        if !matches!(v, Value::Null) {
            match v {
                Value::Bool(b) => opts.headless = *b,
                Value::Object(_) => return parse_browser_options_object(v),
                _ => return Err(WebError::type_err("headless must be a bool or options object")),
            }
        }
    }
    if let Some(v) = args.get(1) {
        if !matches!(v, Value::Null) {
            opts.user_agent = Some(value_to_plain_string(v));
        }
    }
    if let Some(v) = args.get(2) {
        if !matches!(v, Value::Null) {
            opts.headers = Some(value_as_string_map(v)?);
        }
    }
    if let Some(v) = args.get(3) {
        if !matches!(v, Value::Null) {
            match v {
                Value::Bool(b) => opts.stealth = *b,
                _ => return Err(WebError::type_err("stealth must be a bool")),
            }
        }
    }
    if let Some(v) = args.get(4) {
        if !matches!(v, Value::Null) {
            // `options={...}` bag merged on top
            let nested = parse_browser_options_object(v)?;
            opts.merge_from(nested);
        }
    }
    if let Some(v) = args.get(5) {
        if !matches!(v, Value::Null) {
            opts.profile = Some(value_to_plain_string(v));
        }
    }
    Ok(opts)
}

fn looks_like_browser_options(v: &Value) -> bool {
    let Ok(entries) = object_entries(v) else {
        return false;
    };
    entries.contains_key("stealth")
        || entries.contains_key("profile")
        || entries.contains_key("options")
        || entries.contains_key("headless")
        || entries.contains_key("user_agent")
        || entries.contains_key("headers")
        || entries.contains_key("locale")
        || entries.contains_key("timezone")
        || entries.contains_key("viewport")
}

fn parse_browser_options_object(v: &Value) -> Result<BrowserOptions, WebError> {
    let entries = object_entries(v)?;
    let mut opts = BrowserOptions::default();

    // Nested `options={ stealth, profile, ... }`
    if let Some(inner) = entries.get("options") {
        if !matches!(inner, Value::Null) {
            opts.merge_from(parse_browser_options_object(inner)?);
        }
    }

    if let Some(Value::Bool(b)) = entries.get("headless") {
        opts.headless = *b;
    }
    if let Some(Value::Bool(b)) = entries.get("stealth") {
        opts.stealth = *b;
    }
    if let Some(ua) = entries.get("user_agent") {
        if !matches!(ua, Value::Null) {
            opts.user_agent = Some(value_to_plain_string(ua));
        }
    }
    if let Some(h) = entries.get("headers") {
        if !matches!(h, Value::Null) {
            opts.headers = Some(value_as_string_map(h)?);
        }
    }
    if let Some(p) = entries.get("profile") {
        if !matches!(p, Value::Null) {
            opts.profile = Some(value_to_plain_string(p));
        }
    }
    if let Some(l) = entries.get("locale") {
        if !matches!(l, Value::Null) {
            opts.locale = Some(value_to_plain_string(l));
        }
    }
    if let Some(tz) = entries.get("timezone") {
        if !matches!(tz, Value::Null) {
            opts.timezone = Some(value_to_plain_string(tz));
        }
    }
    if let Some(vp) = entries.get("viewport") {
        if !matches!(vp, Value::Null) {
            opts.viewport = Some(parse_viewport(vp)?);
        }
    }
    Ok(opts)
}

fn parse_viewport(v: &Value) -> Result<(u32, u32), WebError> {
    match v {
        Value::Array(rc) => {
            let arr = rc.borrow();
            if arr.len() < 2 {
                return Err(WebError::value("viewport must be [width, height]"));
            }
            let w = value_as_f64(&arr[0])
                .ok_or_else(|| WebError::type_err("viewport width must be a number"))?
                as u32;
            let h = value_as_f64(&arr[1])
                .ok_or_else(|| WebError::type_err("viewport height must be a number"))?
                as u32;
            Ok((w, h))
        }
        Value::Object(_) => {
            let e = object_entries(v)?;
            let w = e
                .get("width")
                .and_then(value_as_f64)
                .ok_or_else(|| WebError::value("viewport.width required"))? as u32;
            let h = e
                .get("height")
                .and_then(value_as_f64)
                .ok_or_else(|| WebError::value("viewport.height required"))? as u32;
            Ok((w, h))
        }
        _ => Err(WebError::type_err(
            "viewport must be [width, height] or { width, height }",
        )),
    }
}

#[derive(Debug, Clone)]
pub struct BrowserOptions {
    pub headless: bool,
    pub stealth: bool,
    pub user_agent: Option<String>,
    pub headers: Option<HashMap<String, String>>,
    /// Persistent Chromium user-data directory (cookies, localStorage, prefs).
    pub profile: Option<String>,
    pub locale: Option<String>,
    pub timezone: Option<String>,
    pub viewport: Option<(u32, u32)>,
    pub hardware_concurrency: Option<u32>,
    pub device_memory_gb: Option<u32>,
}

impl Default for BrowserOptions {
    fn default() -> Self {
        Self {
            headless: false,
            stealth: false,
            user_agent: None,
            headers: None,
            profile: None,
            locale: None,
            timezone: None,
            viewport: None,
            hardware_concurrency: None,
            device_memory_gb: None,
        }
    }
}

impl BrowserOptions {
    fn merge_from(&mut self, other: BrowserOptions) {
        self.stealth |= other.stealth;
        if other.headless {
            self.headless = true;
        }
        if other.user_agent.is_some() {
            self.user_agent = other.user_agent;
        }
        if other.headers.is_some() {
            self.headers = other.headers;
        }
        if other.profile.is_some() {
            self.profile = other.profile;
        }
        if other.locale.is_some() {
            self.locale = other.locale;
        }
        if other.timezone.is_some() {
            self.timezone = other.timezone;
        }
        if other.viewport.is_some() {
            self.viewport = other.viewport;
        }
        if other.hardware_concurrency.is_some() {
            self.hardware_concurrency = other.hardware_concurrency;
        }
        if other.device_memory_gb.is_some() {
            self.device_memory_gb = other.device_memory_gb;
        }
    }
}
