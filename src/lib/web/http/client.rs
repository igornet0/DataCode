//! ureq-based HTTP client for `web.http`.

use crate::file_io::value_serde::value_to_json;
use crate::web::args::HttpOptions;
use crate::web::error::WebError;
use crate::web::security::{validate_url, DEFAULT_HTTP_TIMEOUT_SECS, MAX_REDIRECTS};
use crate::web::values::HttpResponse;
use std::collections::HashMap;
use std::time::Duration;

pub fn request(method: &str, url: &str, opts: &HttpOptions) -> Result<HttpResponse, WebError> {
    validate_url(url)?;
    let timeout = opts.timeout.unwrap_or(DEFAULT_HTTP_TIMEOUT_SECS).max(0.1);
    let dur = Duration::from_secs_f64(timeout);
    let agent = ureq::AgentBuilder::new()
        .timeout_connect(dur)
        .timeout_read(dur)
        .redirects(MAX_REDIRECTS)
        .build();

    let final_url = append_query(url, opts.params.as_ref())?;
    let mut req = match method {
        "GET" => agent.get(&final_url),
        "POST" => agent.post(&final_url),
        "PUT" => agent.put(&final_url),
        "PATCH" => agent.request("PATCH", &final_url),
        "DELETE" => agent.delete(&final_url),
        "HEAD" => agent.head(&final_url),
        "OPTIONS" => agent.request("OPTIONS", &final_url),
        other => {
            return Err(WebError::value(format!("unsupported HTTP method '{}'", other)));
        }
    };

    if let Some(headers) = &opts.headers {
        for (k, v) in headers {
            req = req.set(k, v);
        }
    }

    let result = if let Some(json) = &opts.json {
        let body = serde_json::to_string(&value_to_json(json).map_err(|e| {
            WebError::value(format!("invalid JSON body: {}", e.message()))
        })?)
        .map_err(|e| WebError::value(format!("JSON encode: {}", e)))?;
        req.set("Content-Type", "application/json")
            .send_string(&body)
    } else if let Some(body) = &opts.body {
        req.send_string(body)
    } else {
        req.call()
    };

    // ureq returns Err(Status) for 4xx/5xx — still a usable response for HttpResponse.
    let resp = match result {
        Ok(r) => r,
        Err(ureq::Error::Status(_code, r)) => r,
        Err(ureq::Error::Transport(t)) => {
            let msg = t.to_string();
            if msg.to_ascii_lowercase().contains("timeout") {
                return Err(WebError::io(format!("request timed out: {}", msg)));
            }
            return Err(WebError::io(format!("network error: {}", msg)));
        }
    };

    let status = resp.status();
    let mut headers = HashMap::new();
    for name in resp.headers_names() {
        if let Some(v) = resp.header(&name) {
            headers.insert(name, v.to_string());
        }
    }
    let response_url = resp.get_url().to_string();
    let body = if method == "HEAD" {
        Vec::new()
    } else {
        resp.into_string()
            .map_err(|e| WebError::io(format!("response body: {}", e)))?
            .into_bytes()
    };
    Ok(HttpResponse::new(status, response_url, headers, body))
}

fn append_query(url: &str, params: Option<&HashMap<String, String>>) -> Result<String, WebError> {
    let Some(params) = params else {
        return Ok(url.to_string());
    };
    if params.is_empty() {
        return Ok(url.to_string());
    }
    let qs: Vec<String> = params
        .iter()
        .map(|(k, v)| format!("{}={}", urlencoding_encode(k), urlencoding_encode(v)))
        .collect();
    let sep = if url.contains('?') { '&' } else { '?' };
    Ok(format!("{}{}{}", url, sep, qs.join("&")))
}

fn urlencoding_encode(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(b as char);
            }
            _ => out.push_str(&format!("%{:02X}", b)),
        }
    }
    out
}
