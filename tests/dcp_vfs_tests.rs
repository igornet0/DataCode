//! Integration tests for DCP in-memory VFS.

use data_code::dcp::{clear_dcp_vfs, set_dcp_vfs, DcpDecoder, DcpVfs};
use data_code::file_io::{read_bytes_from_path, write_bytes_to_path};
use data_code::run_with_vm;
use data_code::websocket::{output_capture::OutputCapture, set_use_ve, take_native_error};
use std::fs;
use std::path::PathBuf;
use std::sync::Arc;

fn with_assets_fixture() -> Vec<u8> {
    fs::read("tests/dcp_fixtures/with_assets.dcp").expect("fixture with_assets.dcp")
}

#[test]
fn vfs_read_bytes_from_decoded_package() {
    let bytes = with_assets_fixture();
    let package = DcpDecoder::decode(&bytes).expect("decode");
    let vfs = Arc::new(DcpVfs::from_assets(package.assets).expect("vfs"));
    set_dcp_vfs(Some(vfs));

    let data = read_bytes_from_path(&PathBuf::from("data/sample.txt")).expect("read asset");
    assert_eq!(data, b"asset payload");

    clear_dcp_vfs();
}

#[test]
fn vfs_blocks_local_write() {
    let vfs = Arc::new(
        DcpVfs::from_assets(vec![("file.txt".to_string(), b"x".to_vec())]).expect("vfs"),
    );
    set_dcp_vfs(Some(vfs));

    let err = write_bytes_to_path(&PathBuf::from("out.txt"), b"nope").unwrap_err();
    assert!(err.contains("Write not allowed"));

    clear_dcp_vfs();
}

#[test]
fn vfs_run_vm_read_asset() {
    let bytes = with_assets_fixture();
    let package = DcpDecoder::decode(&bytes).expect("decode");
    let vfs = Arc::new(DcpVfs::from_assets(package.assets).expect("vfs"));
    set_use_ve(true);
    set_dcp_vfs(Some(vfs));

    let code = r#"
global text = read(path("data/sample.txt"))
print("asset:", text)
"#;

    let capture = OutputCapture::new();
    capture.set_capture(true);

    let result = run_with_vm(code);
    let output = capture.get_output();
    capture.set_capture(false);

    clear_dcp_vfs();
    set_use_ve(false);

    result.expect("vm run");
    assert!(output.contains("asset:"));
    assert!(output.contains("asset payload"));
    assert!(take_native_error().is_none());
}

#[test]
fn vfs_list_files_nested() {
    let vfs = Arc::new(
        DcpVfs::from_assets(vec![
            ("data/a.csv".to_string(), b"1".to_vec()),
            ("data/sub/b.txt".to_string(), b"2".to_vec()),
        ])
        .expect("vfs"),
    );
    set_use_ve(true);
    set_dcp_vfs(Some(vfs));

    let code = r#"
global files = list_files(path("data"))
print("count:", len(files))
for f in files {
    print("file:", f)
}
"#;

    let capture = OutputCapture::new();
    capture.set_capture(true);

    let result = run_with_vm(code);
    let output = capture.get_output();
    capture.set_capture(false);

    clear_dcp_vfs();
    set_use_ve(false);

    result.expect("vm run");
    assert!(output.contains("a.csv") || output.contains("sub"));
}
