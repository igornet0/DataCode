//! Module namespaces with own callable `get` must not be compiled/run as plain-dict `.get`.
//! Regression: `http.get(url_var)` / `env.get(k)` used `ObjectGetIntegral` key-lookup.

#[cfg(test)]
mod tests {
    use data_code::{run, Value};
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::thread;

    fn spawn_tiny_http_server() -> (String, thread::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
        let addr = listener.local_addr().expect("addr");
        let handle = thread::spawn(move || {
            if let Ok((mut stream, _)) = listener.accept() {
                let mut buf = [0u8; 1024];
                let _ = stream.read(&mut buf);
                let body = r#"{"ok":true}"#;
                let resp = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    body.len(),
                    body
                );
                let _ = stream.write_all(resp.as_bytes());
            }
        });
        (format!("http://{}", addr), handle)
    }

    #[test]
    fn http_get_with_variable_url() {
        let (base, handle) = spawn_tiny_http_server();
        let code = format!(
            r##"
from web import http
url = "{}"
response = http.get(url)
if typeof(response) != "http_response" {{
    throw "expected http_response, got " + typeof(response)
}}
if response.status != 200 {{
    throw "bad status"
}}
if response.ok != true {{
    throw "not ok"
}}
"##,
            base
        );
        run(&code).expect("http.get(var)");
        let _ = handle.join();
    }

    #[test]
    fn http_get_with_variable_url_and_opts() {
        let (base, handle) = spawn_tiny_http_server();
        let code = format!(
            r##"
from web import http
url = "{}"
opts = {{ "timeout": 10.0 }}
response = http.get(url, opts)
if typeof(response) != "http_response" {{
    throw "expected http_response, got " + typeof(response) + " value=" + str(response)
}}
if response.status != 200 {{
    throw "bad status"
}}
"##,
            base
        );
        run(&code).expect("http.get(var, opts)");
        let _ = handle.join();
    }

    #[test]
    fn http_get_literal_still_works() {
        let (base, handle) = spawn_tiny_http_server();
        let code = format!(
            r##"
from web import http
response = http.get("{}")
if response.status != 200 {{
    throw "bad status"
}}
"##,
            base
        );
        run(&code).expect("http.get(literal)");
        let _ = handle.join();
    }

    #[test]
    fn env_get_with_variable_key() {
        std::env::set_var("DATACODE_SHADOW_GET_TEST", "shadow-ok");
        let code = r#"
from system import env
k = "DATACODE_SHADOW_GET_TEST"
v = env.get(k)
if v != "shadow-ok" {
    throw "env.get(var) failed: " + str(v)
}
"#;
        run(code).expect("env.get(var)");
        std::env::remove_var("DATACODE_SHADOW_GET_TEST");
    }

    #[test]
    fn plain_dict_get_still_uses_default() {
        let code = r#"
obj = { "a": 1 }
if obj.get("missing", 99) != 99 {
    throw "plain dict default broken"
}
if obj.get("a") != 1 {
    throw "plain dict get broken"
}
"#;
        match run(code).expect("plain dict get") {
            Value::Null => {}
            other => panic!("expected null script result, got {:?}", other),
        }
    }
}
