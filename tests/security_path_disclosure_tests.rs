//! Security: code run through the WebSocket server (`--use-ve`) must never see
//! server filesystem paths or reach the server disk.
//!
//! Contract for a DCP session with default server settings (no ws_app.dc):
//! - every path a built-in returns, prints or puts into an error message is the
//!   short form relative to the session's virtual environment (`./…`);
//! - creating and writing files is denied (protects the server disk from being
//!   filled by clients); a developer may widen this only via ws_app.dc;
//! - server environment (home, temp, env variables, processes) is not exposed.
//!
//! Each test sends a real DCP package through `router::dispatch_dcp` — the same
//! path a `datacode --websocket --use-ve` client (py-worm) takes — and scans the
//! whole JSON response for server paths.

mod common;

use common::dcp::{run_session, Response};
use std::path::PathBuf;

const SAMPLE_CSV: &[u8] = b"id,name\n1,alpha\n2,beta\n";

// ---------------------------------------------------------------------------
// Session + leak detection
// ---------------------------------------------------------------------------

/// Absolute locations of this machine that must never reach a client.
fn server_paths() -> Vec<String> {
    let mut paths: Vec<PathBuf> = vec![PathBuf::from(env!("CARGO_MANIFEST_DIR")), std::env::temp_dir()];
    if let Ok(cwd) = std::env::current_dir() {
        paths.push(cwd);
    }
    if let Some(home) = dirs::home_dir() {
        paths.push(home);
    }
    let mut out = Vec::new();
    for p in paths {
        if let Ok(canonical) = p.canonicalize() {
            out.push(canonical.to_string_lossy().into_owned());
        }
        out.push(p.to_string_lossy().into_owned());
    }
    out.retain(|s| s.len() > 1);
    out.iter_mut().for_each(|s| {
        while s.len() > 1 && s.ends_with('/') {
            s.pop();
        }
    });
    out.sort();
    out.dedup();
    out
}

/// OS-level roots that indicate a server path even when not one of ours.
const SERVER_PATH_MARKERS: &[&str] = &["/Users/", "/home/", "/private/var/", "/var/folders/", "/root/"];

fn assert_no_server_path(resp: &Response, what: &str) {
    let mut leaks: Vec<String> = server_paths()
        .into_iter()
        .filter(|p| resp.raw.contains(p.as_str()))
        .collect();
    leaks.extend(
        SERVER_PATH_MARKERS
            .iter()
            .filter(|m| resp.raw.contains(*m))
            .map(|m| m.to_string()),
    );
    assert!(
        leaks.is_empty(),
        "{what}: server path disclosed {leaks:?}\nresponse: {}",
        resp.raw
    );
}

/// Removes probe files from the server working directory even when a test fails
/// (a failing write test means the file *was* created there).
struct CwdProbe(Vec<PathBuf>);

impl CwdProbe {
    fn new(names: &[&str]) -> Self {
        let cwd = std::env::current_dir().unwrap();
        let paths: Vec<PathBuf> = names.iter().map(|n| cwd.join(n)).collect();
        paths.iter().for_each(|p| {
            let _ = std::fs::remove_file(p);
        });
        CwdProbe(paths)
    }

    fn created(&self) -> Vec<String> {
        self.0
            .iter()
            .filter(|p| p.exists())
            .map(|p| p.file_name().unwrap().to_string_lossy().into_owned())
            .collect()
    }
}

impl Drop for CwdProbe {
    fn drop(&mut self) {
        self.0.iter().for_each(|p| {
            let _ = std::fs::remove_file(p);
        });
    }
}

fn lines(output: &str) -> Vec<&str> {
    output.lines().map(str::trim).filter(|l| !l.is_empty()).collect()
}

// ---------------------------------------------------------------------------
// Built-in path functions: short `./` form only
// ---------------------------------------------------------------------------

#[test]
fn getcwd_is_the_virtual_env_root() {
    let resp = run_session("print(getcwd())", &[]);
    assert!(resp.success, "{}", resp.raw);
    assert_no_server_path(&resp, "getcwd()");
    let printed = lines(&resp.output);
    assert!(
        printed.is_empty() || printed == ["./"],
        "getcwd() must be the VE root './' (or empty), got {printed:?}"
    );
}

#[test]
fn path_values_print_relative_to_virtual_env() {
    let code = r#"
p = path("data/input.csv")
print(p)
print(p.parent)
print(p.name)
print(p.exists)
"#;
    let resp = run_session(code, &[("data/input.csv", SAMPLE_CSV)]);
    assert!(resp.success, "{}", resp.raw);
    assert_no_server_path(&resp, "path() display");
    assert_eq!(lines(&resp.output), ["./data/input.csv", "./data", "input.csv", "true"]);
}

#[test]
fn list_files_returns_relative_paths() {
    let code = r#"
for f in list_files(path("data")) {
    print(f)
}
"#;
    let resp = run_session(code, &[("data/input.csv", SAMPLE_CSV), ("data/sub/more.csv", SAMPLE_CSV)]);
    assert!(resp.success, "{}", resp.raw);
    assert_no_server_path(&resp, "list_files()");
    let printed = lines(&resp.output);
    assert!(!printed.is_empty(), "list_files printed nothing: {}", resp.raw);
    for line in &printed {
        assert!(line.starts_with("./data"), "list_files entry must be './data/…', got {line:?}");
    }
}

#[test]
fn reading_an_asset_works_without_disk_access() {
    let resp = run_session(
        "t = read(path(\"data/input.csv\"))\nprint(len(t))",
        &[("data/input.csv", SAMPLE_CSV)],
    );
    assert!(resp.success, "{}", resp.raw);
    assert_no_server_path(&resp, "read(asset)");
    assert_eq!(lines(&resp.output), ["2"]);
}

// ---------------------------------------------------------------------------
// Error messages
// ---------------------------------------------------------------------------

#[test]
fn missing_file_error_uses_relative_path() {
    let resp = run_session("t = read(path(\"missing/report.csv\"))\nprint(t)", &[]);
    assert_no_server_path(&resp, "read(missing file)");
}

#[test]
fn path_traversal_is_rejected_without_disclosure() {
    let resp = run_session("t = read_file(path(\"../../../../etc/hosts\"))\nprint(t)", &[]);
    assert_no_server_path(&resp, "read_file(../../etc/hosts)");
    assert!(!resp.raw.contains("localhost"), "traversal read a server file: {}", resp.raw);
}

#[test]
fn absolute_path_outside_virtual_env_is_rejected() {
    let resp = run_session("t = read_file(path(\"/etc/hosts\"))\nprint(t)", &[]);
    assert_no_server_path(&resp, "read_file(/etc/hosts)");
    assert!(!resp.raw.contains("localhost"), "absolute read reached a server file: {}", resp.raw);
}

#[test]
fn runtime_error_has_no_server_paths() {
    let resp = run_session("x = 1\ny = x + undefined_name_for_test", &[]);
    assert!(!resp.success);
    assert_no_server_path(&resp, "runtime error");
}

#[test]
fn missing_module_import_error_has_no_server_paths() {
    let resp = run_session("import definitely_missing_module_for_test", &[]);
    assert!(!resp.success, "{}", resp.raw);
    assert_no_server_path(&resp, "import error");
}

// ---------------------------------------------------------------------------
// Writes: denied by default in a DCP session, nothing reaches the server disk
// ---------------------------------------------------------------------------

#[test]
fn save_in_session_is_denied_and_creates_nothing() {
    let name = "dc_security_probe_out.csv";
    let probe = CwdProbe::new(&[name]);

    let code = format!(
        "t = table([[1, \"a\"]], [\"id\", \"name\"])\nr = save(t, \"{name}\")\nprint(r)"
    );
    let resp = run_session(&code, &[]);
    assert_no_server_path(&resp, "save()");
    assert!(
        probe.created().is_empty(),
        "save() wrote {:?} into the server working directory; response: {}",
        probe.created(),
        resp.raw
    );
    assert!(!resp.success, "save() must be denied in a DCP session: {}", resp.raw);
}

#[test]
fn table_save_methods_are_denied_and_create_nothing() {
    let name = "dc_security_probe_save";
    let probe = CwdProbe::new(&[&format!("{name}.csv"), &format!("{name}.sqlite")]);
    let code = format!(
        "t = table([[1, \"a\"]], [\"id\", \"name\"])\nprint(t.save_csv(\"{name}.csv\"))\nprint(t.save_sqlite(\"{name}.sqlite\"))"
    );
    let resp = run_session(&code, &[]);
    assert_no_server_path(&resp, "table.save_csv / save_sqlite");
    assert!(
        probe.created().is_empty(),
        "table save wrote {:?} into the server working directory; response: {}",
        probe.created(),
        resp.raw
    );
    assert!(!resp.success, "table save must be denied in a DCP session: {}", resp.raw);
}

#[test]
fn system_fs_write_is_denied_and_creates_nothing() {
    let name = "dc_security_probe_fs_write.txt";
    let probe = CwdProbe::new(&[name]);
    let resp = run_system(&format!("system.fs.write(\"{name}\", \"x\")"));
    assert_no_server_path(&resp, "system.fs.write");
    assert!(
        probe.created().is_empty(),
        "system.fs.write created {:?} on the server; response: {}",
        probe.created(),
        resp.raw
    );
}

#[test]
fn read_only_session_creates_no_virtual_env_folder() {
    let before: Vec<_> = std::fs::read_dir(".").unwrap().filter_map(|e| e.ok()).map(|e| e.file_name()).collect();
    let resp = run_session(
        "t = read(path(\"data/input.csv\"))\nprint(len(t))\nprint(getcwd())",
        &[("data/input.csv", SAMPLE_CSV)],
    );
    assert!(resp.success, "{}", resp.raw);
    let after: Vec<_> = std::fs::read_dir(".").unwrap().filter_map(|e| e.ok()).map(|e| e.file_name()).collect();
    let created: Vec<_> = after.iter().filter(|n| !before.contains(n)).collect();
    assert!(
        created.iter().all(|n| !n.to_string_lossy().contains("session") && !n.to_string_lossy().starts_with(".ve")),
        "read-only session created {created:?}"
    );
}

// ---------------------------------------------------------------------------
// `system` module: no server paths in a WebSocket session
// ---------------------------------------------------------------------------

fn run_system(expr: &str) -> Response {
    run_session(&format!("import system\nprint({expr})"), &[])
}

#[test]
fn system_home_dir_is_not_disclosed() {
    assert_no_server_path(&run_system("system.env.get_home_dir()"), "system.env.get_home_dir()");
}

#[test]
fn system_temp_dir_is_not_disclosed() {
    assert_no_server_path(&run_system("system.env.get_temp_dir()"), "system.env.get_temp_dir()");
}

#[test]
fn system_env_does_not_expose_server_secrets() {
    let key = "DATACODE_SECURITY_TEST_SECRET";
    let secret = "s3cr3t-value-for-security-test";
    std::env::set_var(key, secret);
    let resp = run_system(&format!("system.env.get(\"{key}\")"));
    std::env::remove_var(key);
    assert!(!resp.raw.contains(secret), "server environment variable reached the client: {}", resp.raw);
}

#[test]
fn system_env_variables_do_not_disclose_paths() {
    for key in ["HOME", "PWD", "TMPDIR", "CARGO_MANIFEST_DIR", "PATH"] {
        assert_no_server_path(&run_system(&format!("system.env.get(\"{key}\")")), &format!("system.env.get({key})"));
    }
}

#[test]
fn system_runtime_paths_are_not_disclosed() {
    for f in ["get_module_path", "get_venv_path", "get_dpm_env_base", "get_dpm_env_root"] {
        assert_no_server_path(&run_system(&format!("system.runtime.{f}()")), &format!("system.runtime.{f}()"));
    }
}

#[test]
fn system_process_exec_cannot_reveal_working_directory() {
    let resp = run_system("system.process.exec(\"pwd\")");
    assert_no_server_path(&resp, "system.process.exec(pwd)");
}

#[test]
fn system_fs_read_cannot_reach_server_files() {
    let resp = run_system("system.fs.read(\"/etc/hosts\")");
    assert_no_server_path(&resp, "system.fs.read(/etc/hosts)");
    assert!(!resp.raw.contains("localhost"), "system.fs.read reached a server file: {}", resp.raw);
}

#[test]
fn system_host_identity_is_not_disclosed() {
    let username = std::env::var("USER").or_else(|_| std::env::var("USERNAME")).unwrap_or_default();
    let hostname = hostname_of_this_machine();
    for expr in ["system.env.get_username()", "system.env.get_hostname()"] {
        let resp = run_system(expr);
        assert_no_server_path(&resp, expr);
        if username.len() > 1 {
            assert!(!resp.output.contains(&username), "{expr} disclosed the server user: {}", resp.raw);
        }
        if hostname.len() > 1 {
            assert!(!resp.output.contains(&hostname), "{expr} disclosed the server host name: {}", resp.raw);
        }
    }
}

#[test]
fn system_network_details_are_not_disclosed() {
    for expr in ["system.net.get_ip()", "system.net.get_interfaces()"] {
        let resp = run_system(expr);
        assert_no_server_path(&resp, expr);
        let output = resp.output.trim();
        assert!(
            output.is_empty() || output == "null" || output == "[]" || output.starts_with("permission denied"),
            "{expr} disclosed server network details: {}",
            resp.raw
        );
    }
}

fn hostname_of_this_machine() -> String {
    std::process::Command::new("hostname")
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_default()
}
