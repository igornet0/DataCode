//! Session folder (issue #14): with `configure({"allow_write": true})` in
//! ws_app.dc, client code writes into a private per-connection folder that is
//! created only on the first write, shown as `./…`, limited by a quota and
//! removed when the client disconnects.
//!
//! `bootstrap_app` sets process-wide config, so this binary holds a single
//! sequential test; each "connection" runs on its own thread like the server.

mod common;

use common::dcp::{assert_no_server_path, run_session};
use data_code::websocket::app::bootstrap_app;
use data_code::websocket::session_ve::{cleanup_ve_dir, ve_dir};

fn lines(output: &str) -> Vec<String> {
    output.lines().map(str::trim).filter(|l| !l.is_empty()).map(String::from).collect()
}

/// Run `f` as one client connection; returns its result and whether the session
/// folder existed (and then gets removed, as on disconnect).
fn connection<T: Send + 'static>(f: impl FnOnce() -> T + Send + 'static) -> (T, Option<std::path::PathBuf>) {
    std::thread::spawn(move || {
        let out = f();
        let dir = ve_dir();
        cleanup_ve_dir();
        (out, dir)
    })
    .join()
    .expect("connection thread")
}

#[test]
fn session_folder_writes_are_opt_in_short_and_bounded() {
    let dir = tempfile::tempdir().unwrap();
    let app = dir.path().join("ws_app.dc");
    std::fs::write(
        &app,
        "from websocket import configure\nconfigure({\"allow_write\": true, \"write_quota_mb\": 1})\n",
    )
    .unwrap();
    bootstrap_app(app.to_str().unwrap(), None).expect("bootstrap");

    // 1. Write, read back, list and probe — every path in the short form.
    let code = r#"
t = table([[1, "a"], [2, "b"]], ["id", "name"])
print(save(t, "out/report.csv"))
print(t.save_csv("plain.csv"))
p = path("out/report.csv")
print(p)
print(p.exists)
back = read(p)
print(len(back))
for f in list_files(path("out")) {
    print(f)
}
"#;
    let ((resp, written_exists), folder) = connection(move || {
        let resp = run_session(code, &[]);
        let exists = ve_dir().map(|d| d.join("out/report.csv").is_file()).unwrap_or(false);
        (resp, exists)
    });
    assert!(resp.success, "{}", resp.raw);
    assert_no_server_path(&resp, "session folder write");
    assert_eq!(
        lines(&resp.output),
        ["./out/report.csv", "./plain.csv", "./out/report.csv", "true", "2", "./out/report.csv"]
    );
    let folder = folder.expect("session folder created by the write");
    assert!(written_exists, "file not written into the session folder");
    assert!(!folder.exists(), "session folder not removed on disconnect");

    // 2. Files persist across packages within one connection.
    let ((first, second), _) = connection(|| {
        let first = run_session("t = table([[1]], [\"a\"])\nprint(save(t, \"keep.csv\"))", &[]);
        let second = run_session("print(len(read(path(\"keep.csv\"))))", &[]);
        (first, second)
    });
    assert!(first.success && second.success, "{} / {}", first.raw, second.raw);
    assert_eq!(lines(&second.output), ["1"]);

    // 3. The folder is created only when something is written.
    let (resp, folder) = connection(|| run_session("print(getcwd())\nprint(1 + 1)", &[]));
    assert!(resp.success, "{}", resp.raw);
    assert!(folder.is_none(), "read-only session created a session folder");

    // 4. Absolute paths and traversal stay outside reach.
    let escaped = std::env::temp_dir().join("dc_session_ve_escape.csv");
    let _ = std::fs::remove_file(&escaped);
    let abs = escaped.to_string_lossy().into_owned();
    let (resp, _) = connection(move || {
        run_session(&format!("t = table([[1]], [\"a\"])\nprint(save(t, \"{abs}\"))"), &[])
    });
    assert!(!resp.success, "absolute write must fail: {}", resp.raw);
    assert!(resp.raw.contains("Absolute paths are not allowed"), "{}", resp.raw);
    assert!(!escaped.exists(), "absolute write escaped the session folder");
    assert_no_server_path(&resp, "absolute write error");

    let (resp, _) = connection(|| run_session("t = table([[1]], [\"a\"])\nprint(save(t, \"../up.csv\"))", &[]));
    assert!(!resp.success && resp.raw.contains("Path traversal"), "{}", resp.raw);

    // 5. The quota bounds the folder; the oversized file is removed.
    let (resp, _) = connection(|| {
        run_session("big = \"x\" * 1500000\nprint(save(big, \"big.txt\"))", &[])
    });
    assert!(!resp.success, "write over quota must fail: {}", resp.raw);
    assert!(resp.raw.contains("quota exceeded"), "{}", resp.raw);

    // 6. system.fs.write is confined to the session folder as well.
    let ((resp, note), _) = connection(|| {
        let resp = run_session("import system\nprint(system.fs.write(\"note.txt\", \"hi\"))", &[]);
        let note = ve_dir().map(|d| d.join("note.txt").is_file()).unwrap_or(false);
        (resp, note)
    });
    assert!(resp.success, "{}", resp.raw);
    assert!(note, "system.fs.write did not land in the session folder: {}", resp.raw);
}
