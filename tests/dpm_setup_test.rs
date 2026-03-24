//! Tests for `setup.dcmodule` (DPM post-install hooks).

use data_code::dpm::setup::load_setup_descriptor;

#[test]
fn load_setup_descriptor_parses_minimal_json() {
    let dir = tempfile::tempdir().expect("tempdir");
    std::fs::write(
        dir.path().join("setup.dcmodule"),
        r#"{"schema_version":1,"module_name":"ml","build":[],"install":[]}"#,
    )
    .expect("write");
    let d = load_setup_descriptor(dir.path()).expect("load");
    assert!(d.is_some());
    let desc = d.unwrap();
    assert_eq!(desc.module_name.as_deref(), Some("ml"));
}

#[test]
fn load_setup_descriptor_missing_returns_none() {
    let dir = tempfile::tempdir().expect("tempdir");
    let d = load_setup_descriptor(dir.path()).expect("load");
    assert!(d.is_none());
}
