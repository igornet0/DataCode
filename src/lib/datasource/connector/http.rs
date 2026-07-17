//! HTTP connector using ureq.

use crate::common::table::Table;
use crate::common::value::Value;
use crate::datasource::config::DataSourceConfig;
use crate::datasource::connector::ConnectorBackend;
use crate::datasource::error::DataSourceError;
use crate::datasource::get_table::response_to_table;
use crate::datasource::request::{GetTableSpec, RequestSpec, SendTableSpec};
use crate::datasource::response::DataSourceResponse;
use crate::datasource::send_table::table_to_json_value;
use crate::file_io::value_serde::value_to_json;
use std::collections::HashMap;
use std::time::{Duration, Instant};

pub struct HttpConnector {
    config: DataSourceConfig,
    agent: Option<ureq::Agent>,
    connected: bool,
}

impl HttpConnector {
    pub fn new(config: DataSourceConfig) -> Result<Self, DataSourceError> {
        if config.url.is_none() {
            return Err(DataSourceError::Validation {
                message: "http datasource requires 'url'".to_string(),
            });
        }
        Ok(Self {
            config,
            agent: None,
            connected: false,
        })
    }

    fn base_url(&self) -> &str {
        self.config.url.as_deref().unwrap_or("")
    }

    fn build_agent(&self) -> ureq::Agent {
        let timeout_secs = self
            .config
            .timeout
            .or(self.config.connect_timeout)
            .unwrap_or(30.0)
            .max(0.1);
        let dur = Duration::from_secs_f64(timeout_secs);
        ureq::AgentBuilder::new()
            .timeout_connect(dur)
            .timeout_read(dur)
            .build()
    }

    fn ensure_agent(&mut self) -> Result<&ureq::Agent, DataSourceError> {
        if self.agent.is_none() {
            self.agent = Some(self.build_agent());
        }
        Ok(self.agent.as_ref().unwrap())
    }

    fn build_url(&self, spec_path: Option<&str>, spec_url: Option<&str>, query: &HashMap<String, String>) -> String {
        let mut url = if let Some(u) = spec_url {
            u.to_string()
        } else if let Some(p) = spec_path {
            join_url(self.base_url(), p)
        } else {
            self.base_url().to_string()
        };
        if !query.is_empty() {
            let qs: Vec<String> = query
                .iter()
                .map(|(k, v)| format!("{}={}", urlencoding_encode(k), urlencoding_encode(v)))
                .collect();
            let sep = if url.contains('?') { '&' } else { '?' };
            url.push(sep);
            url.push_str(&qs.join("&"));
        }
        url
    }

    fn apply_auth(&self, req: ureq::Request) -> ureq::Request {
        let mut req = req;
        if let Some(token) = &self.config.token {
            req = req.set("Authorization", &format!("Bearer {}", token));
        }
        if let Some(key) = &self.config.api_key {
            req = req.set("X-API-Key", key);
        }
        if let (Some(user), Some(pass)) = (&self.config.username, &self.config.password) {
            req = req.set("Authorization", &format_basic_auth(user, pass));
        }
        for (k, v) in &self.config.headers {
            req = req.set(k, v);
        }
        req
    }

    fn apply_spec_headers(&self, req: ureq::Request, headers: &HashMap<String, String>) -> ureq::Request {
        let mut req = req;
        for (k, v) in headers {
            req = req.set(k, v);
        }
        req
    }

    fn execute_request(&mut self, spec: &RequestSpec) -> Result<DataSourceResponse, DataSourceError> {
        let url = self.build_url(spec.path.as_deref(), spec.url.as_deref(), &spec.query);
        let method = spec.method.to_uppercase();
        let agent = self.ensure_agent()?;
        let start = Instant::now();

        let mut req = match method.as_str() {
            "GET" => agent.get(&url),
            "POST" => agent.post(&url),
            "PUT" => agent.put(&url),
            "PATCH" => agent.request("PATCH", &url),
            "DELETE" => agent.delete(&url),
            "HEAD" => agent.head(&url),
            other => {
                return Err(DataSourceError::Validation {
                    message: format!("unsupported HTTP method '{}'", other),
                });
            }
        };
        req = self.apply_auth(req);
        req = self.apply_spec_headers(req, &spec.headers);

        let resp = if let Some(json) = &spec.json {
            let body = serde_json::to_string(&value_to_json(json).map_err(|e| {
                DataSourceError::Parse {
                    message: e.message(),
                }
            })?)
            .map_err(|e| DataSourceError::Parse {
                message: format!("JSON encode: {}", e),
            })?;
            req.set("Content-Type", "application/json")
                .send_string(&body)
        } else if let Some(body) = &spec.body {
            req.send_string(body)
        } else if !spec.form.is_empty() {
            let body: String = spec
                .form
                .iter()
                .map(|(k, v)| format!("{}={}", urlencoding_encode(k), urlencoding_encode(v)))
                .collect::<Vec<_>>()
                .join("&");
            req.set("Content-Type", "application/x-www-form-urlencoded")
                .send_string(&body)
        } else {
            req.call()
        };

        let resp = resp.map_err(map_ureq_err)?;
        let status = resp.status();
        let mut headers = HashMap::new();
        for name in resp.headers_names() {
            if let Some(v) = resp.header(&name) {
                headers.insert(name, v.to_string());
            }
        }
        let body = if method == "HEAD" {
            Vec::new()
        } else {
            resp.into_string()
                .map_err(|e| DataSourceError::Parse {
                    message: format!("response body: {}", e),
                })?
                .into_bytes()
        };
        Ok(DataSourceResponse::new(
            status,
            url,
            headers,
            body,
            start.elapsed(),
        ))
    }
}

impl ConnectorBackend for HttpConnector {
    fn connector_type(&self) -> &str {
        "http"
    }

    fn connect(&mut self) -> Result<(), DataSourceError> {
        self.agent = Some(self.build_agent());
        self.connected = true;
        Ok(())
    }

    fn disconnect(&mut self) {
        self.agent = None;
        self.connected = false;
    }

    fn ping(&mut self) -> Result<bool, DataSourceError> {
        let spec = RequestSpec {
            method: "HEAD".to_string(),
            path: Some("/".to_string()),
            ..Default::default()
        };
        let resp = self.execute_request(&spec)?;
        Ok(resp.success || resp.status < 500)
    }

    fn test(&mut self) -> Result<Value, DataSourceError> {
        match self.ping() {
            Ok(ok) => Ok(diagnostic_object(ok, None)),
            Err(e) => Ok(diagnostic_object(false, Some(e.display()))),
        }
    }

    fn request(&mut self, spec: &RequestSpec) -> Result<DataSourceResponse, DataSourceError> {
        self.execute_request(spec)
    }

    fn get_table(&mut self, spec: &GetTableSpec) -> Result<Table, DataSourceError> {
        let req = RequestSpec {
            method: spec.method.clone(),
            path: spec.path.clone(),
            url: spec.url.clone(),
            headers: spec.headers.clone(),
            query: spec.query.clone(),
            timeout: spec.timeout,
            ..Default::default()
        };
        let response = self.execute_request(&req)?;
        response_to_table(&response, spec)
    }

    fn send_table(&mut self, spec: &SendTableSpec) -> Result<(), DataSourceError> {
        let table_val = spec.table.as_ref().ok_or_else(|| DataSourceError::Validation {
            message: "send_table requires table".to_string(),
        })?;
        let Value::Table(rc) = table_val else {
            return Err(DataSourceError::Validation {
                message: "send_table requires a table value".to_string(),
            });
        };
        let json_val = table_to_json_value(&rc.borrow());
        let body = serde_json::to_string(&value_to_json(&json_val).map_err(|e| {
            DataSourceError::Parse {
                message: e.message(),
            }
        })?)
        .map_err(|e| DataSourceError::Parse {
            message: format!("JSON encode: {}", e),
        })?;
        let mut req_spec = RequestSpec {
            method: spec.method.clone(),
            path: spec.path.clone(),
            url: spec.url.clone(),
            body: Some(body),
            ..Default::default()
        };
        if req_spec.method.is_empty() {
            req_spec.method = "POST".to_string();
        }
        req_spec.headers.insert("Content-Type".to_string(), "application/json".to_string());
        let _ = self.execute_request(&req_spec)?;
        Ok(())
    }

    fn clone_backend(&self) -> Box<dyn ConnectorBackend> {
        Box::new(Self {
            config: self.config.clone(),
            agent: None,
            connected: false,
        })
    }
}

fn join_url(base: &str, path: &str) -> String {
    let base = base.trim_end_matches('/');
    let path = path.trim_start_matches('/');
    if path.is_empty() {
        base.to_string()
    } else {
        format!("{}/{}", base, path)
    }
}

fn urlencoding_encode(s: &str) -> String {
    let mut out = String::new();
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

fn format_basic_auth(user: &str, pass: &str) -> String {
    use base64::Engine;
    let creds = format!("{}:{}", user, pass);
    format!(
        "Basic {}",
        base64::engine::general_purpose::STANDARD.encode(creds.as_bytes())
    )
}

fn map_ureq_err(e: ureq::Error) -> DataSourceError {
    match e {
        ureq::Error::Status(code, resp) => {
            let msg = resp.into_string().unwrap_or_default();
            if code == 401 || code == 403 {
                DataSourceError::Authentication {
                    message: format!("HTTP {}: {}", code, msg),
                }
            } else if code == 404 {
                DataSourceError::NotFound {
                    message: format!("HTTP {}: {}", code, msg),
                }
            } else {
                DataSourceError::Other {
                    message: format!("HTTP {}: {}", code, msg),
                }
            }
        }
        ureq::Error::Transport(t) => {
            let msg = t.to_string();
            if msg.to_lowercase().contains("timeout") {
                DataSourceError::Timeout { message: msg }
            } else {
                DataSourceError::Connection { message: msg }
            }
        }
    }
}

fn diagnostic_object(ok: bool, message: Option<String>) -> Value {
    use crate::common::value::ObjectKind;
    use std::cell::RefCell;
    use std::rc::Rc;
    let mut m = HashMap::new();
    m.insert("ok".to_string(), Value::Bool(ok));
    m.insert(
        "type".to_string(),
        Value::String("http".to_string()),
    );
    if let Some(msg) = message {
        m.insert("message".to_string(), Value::String(msg));
    } else if ok {
        m.insert("message".to_string(), Value::String("ok".to_string()));
    }
    Value::Object(Rc::new(RefCell::new(ObjectKind::legacy(m))))
}
