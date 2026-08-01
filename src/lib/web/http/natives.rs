//! Natives for `web.http`.

use crate::common::value::Value;
use crate::datasource::get_table::table_from_value;
use crate::file_io::value_serde::json_to_value;
use crate::web::args::{parse_http_options, require_string};
use crate::web::error::{raise, WebError};
use crate::web::http::client;
use crate::web::security::check_net_http;
use crate::web::values::HttpResponse;
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

fn wrap_response(resp: HttpResponse) -> Value {
    Value::HttpResponse(Rc::new(RefCell::new(resp)))
}

fn do_request(method: &str, args: &[Value]) -> Value {
    if let Err(e) = check_net_http() {
        raise(e);
        return Value::Null;
    }
    let url = match require_string(args, 0, "url") {
        Ok(u) => u,
        Err(e) => {
            raise(e);
            return Value::Null;
        }
    };
    let opts = match parse_http_options(if args.len() > 1 { &args[1..] } else { &[] }) {
        Ok(o) => o,
        Err(e) => {
            raise(e);
            return Value::Null;
        }
    };
    match client::request(method, &url, &opts) {
        Ok(resp) => wrap_response(resp),
        Err(e) => {
            raise(e);
            Value::Null
        }
    }
}

pub fn native_http_get(args: &[Value]) -> Value {
    do_request("GET", args)
}
pub fn native_http_post(args: &[Value]) -> Value {
    do_request("POST", args)
}
pub fn native_http_put(args: &[Value]) -> Value {
    do_request("PUT", args)
}
pub fn native_http_patch(args: &[Value]) -> Value {
    do_request("PATCH", args)
}
pub fn native_http_delete(args: &[Value]) -> Value {
    do_request("DELETE", args)
}
pub fn native_http_head(args: &[Value]) -> Value {
    do_request("HEAD", args)
}
pub fn native_http_options(args: &[Value]) -> Value {
    do_request("OPTIONS", args)
}

/// `http.get_table(url, …)` — GET + convert JSON body to Table.
pub fn native_http_get_table(args: &[Value]) -> Value {
    let resp_val = native_http_get(args);
    let Value::HttpResponse(rc) = resp_val else {
        return Value::Null;
    };
    let mut resp = rc.borrow_mut();
    let json = match ensure_json(&mut resp) {
        Ok(v) => v,
        Err(e) => {
            raise(e);
            return Value::Null;
        }
    };
    match table_from_value(&json) {
        Ok(t) => Value::Table(Rc::new(RefCell::new(t))),
        Err(e) => {
            raise(WebError::value(e.display()));
            Value::Null
        }
    }
}

pub fn ensure_json(resp: &mut HttpResponse) -> Result<Value, WebError> {
    if let Some(cached) = &resp.json_cache {
        return Ok(cached.clone());
    }
    let text = resp.body_string().map_err(WebError::value)?;
    if text.trim().is_empty() {
        return Err(WebError::value("response body is empty; cannot parse JSON"));
    }
    let parsed: serde_json::Value = serde_json::from_str(&text)
        .map_err(|e| WebError::value(format!("invalid JSON: {}", e)))?;
    let value = json_to_value(parsed).map_err(|e| WebError::value(e.message()))?;
    resp.json_cache = Some(value.clone());
    Ok(value)
}

pub fn headers_value(headers: &HashMap<String, String>) -> Value {
    let mut m = HashMap::new();
    for (k, v) in headers {
        m.insert(k.clone(), Value::String(v.clone()));
    }
    Value::legacy_object(m)
}
