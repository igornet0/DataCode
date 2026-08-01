//! Tests for the built-in `web` module (HTTP / data / import).

use data_code::run;
use data_code::run_with_vm_with_policy;
use data_code::vm::PermissionPolicy;
use std::io::{Read, Write};
use std::net::TcpListener;
use std::thread;
use std::time::Duration;

fn spawn_tiny_http_server() -> (String, thread::JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
    let addr = listener.local_addr().expect("addr");
    let url = format!("http://{}/", addr);
    let handle = thread::spawn(move || {
        if let Ok((mut stream, _)) = listener.accept() {
            let mut buf = [0u8; 1024];
            let _ = stream.read(&mut buf);
            let body = r#"{"users":[{"id":1,"name":"Alex"},{"id":2,"name":"Sam"}]}"#;
            let resp = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                body.len(),
                body
            );
            let _ = stream.write_all(resp.as_bytes());
        }
    });
    thread::sleep(Duration::from_millis(20));
    (url, handle)
}

#[test]
fn web_import_binds_namespaces() {
    let code = r#"
from web import http, browser, data
1
"#;
    run(code).expect("import web namespaces");
}

#[test]
fn data_table_and_extract_from_html_string() {
    let code = r##"
from web import data
html = """
<table id="products">
  <tr><th>name</th><th>price</th></tr>
  <tr><td>Widget</td><td>1200</td></tr>
  <tr><td>Gadget</td><td>800</td></tr>
</table>
<div class="product"><span class="name">Alpha</span><span class="price">100</span><a href="/p/1">x</a></div>
<div class="product"><span class="name">Beta</span><span class="price">200</span><a href="/p/2">x</a></div>
"""
t = data.table(html, "#products")
if len(t) != 2 {
    throw "expected 2 table rows"
}
items = data.extract(html, ".product", {
    "name": ".name",
    "price": ".price",
    "url": { "selector": "a", "attribute": "href" }
})
if len(items) != 2 {
    throw "expected 2 items"
}
if items[0]["name"] != "Alpha" {
    throw "bad name"
}
if items[0]["url"] != "/p/1" {
    throw "bad url"
}
"##;
    run(code).expect("data extract/table");
}

#[test]
fn http_get_local_server() {
    let (base, handle) = spawn_tiny_http_server();
    let code = format!(
        r##"
from web import http
response = http.get("{}")
if response.ok != true {{
    throw "not ok"
}}
if response.status != 200 {{
    throw "bad status"
}}
data = response.json
if data["users"][0]["name"] != "Alex" {{
    throw "bad json"
}}
"##,
        base
    );
    run(&code).expect("http.get");
    let _ = handle.join();
}

#[test]
fn http_invalid_url_errors() {
    let code = r#"
from web import http
try {
    http.get("file:///etc/passwd")
    throw "should have failed"
} catch e {
    1
}
"#;
    run(code).expect("invalid url");
}

#[test]
fn data_table_from_array_of_objects() {
    let code = r#"
from web import data
rows = [
    { "name": "A", "price": "10" },
    { "name": "B", "price": "20" }
]
t = data.table(rows)
if len(t) != 2 {
    throw "expected 2 rows"
}
"#;
    run(code).expect("data.table from array");
}

#[test]
fn restricted_policy_denies_http() {
    let code = r#"
from web import http
http.get("https://example.com")
"#;
    match run_with_vm_with_policy(code, PermissionPolicy::Restricted) {
        Err(e) => {
            let msg = format!("{:?}", e);
            assert!(
                msg.contains("permission denied")
                    || msg.contains("net.http")
                    || msg.contains("IOError"),
                "unexpected error: {}",
                msg
            );
        }
        Ok(_) => panic!("expected permission denial"),
    }
}

#[test]
fn browser_stealth_options_parse() {
    // Compile/runtime: stealth=true and options={...} must bind without type errors.
    // Does not launch Chrome — invalid URL fails after options parse.
    let code = r#"
from web import browser
try {
    browser.open("file:///stealth-options-check", stealth=true)
    throw "should reject file url"
} catch e {
    1
}
try {
    browser.open("file:///stealth-options-check", options={
        stealth: true,
        profile: "./.tmp_web_profile_test"
    })
    throw "should reject file url"
} catch e {
    1
}
"#;
    run(code).expect("stealth options parse");
}

#[test]
#[ignore = "requires local Chrome/Chromium"]
fn browser_open_example_com() {
    let code = r#"
from web import browser
page = browser.open("https://example.com", true)
text = page.text()
page.close()
if len(text) < 1 {
    throw "empty page text"
}
"#;
    run(code).expect("browser open");
}

#[test]
#[ignore = "requires local Chrome/Chromium"]
fn browser_stealth_open_example_com() {
    let code = r#"
from web import browser
page = browser.open("https://example.com", stealth=true, headless=true)
text = page.text()
page.close()
if len(text) < 1 {
    throw "empty page text"
}
"#;
    run(code).expect("stealth browser open");
}

#[test]
fn http_response_typeof() {
    let (base, handle) = spawn_tiny_http_server();
    let code = format!(
        r##"
from web import http
response = http.get("{}")
if typeof(response) != "http_response" {{
    throw "bad typeof"
}}
"##,
        base
    );
    run(&code).expect("typeof");
    let _ = handle.join();
}
