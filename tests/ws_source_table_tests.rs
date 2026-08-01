//! Integration tests for `ws` module and DCP Arrow tables.

use data_code::dcp::{
    arrow_ipc_to_table, clear_dcp_session, set_dcp_metadata, set_dcp_tables, set_dcp_vfs,
    DcpDecoder, DcpTables, DcpVfs,
};
use data_code::run_with_vm;
use data_code::websocket::{output_capture::OutputCapture, set_use_ve, take_native_error};
use std::fs;
use std::sync::Arc;

fn with_table_fixture() -> Vec<u8> {
    fs::read("tests/dcp_fixtures/with_table.dcp").expect("fixture with_table.dcp")
}

fn mount_table_session() {
    let bytes = with_table_fixture();
    let package = DcpDecoder::decode(&bytes).expect("decode");
    assert!(!package.tables.is_empty());
    assert_eq!(package.tables[0].0, "orders");

    let vfs = Arc::new(DcpVfs::from_assets(package.assets.clone()).expect("vfs"));
    let tables = Arc::new(DcpTables::from_entries(package.tables.clone()));
    let metadata = package.metadata.map(|m| m.to_map());

    set_use_ve(true);
    set_dcp_vfs(Some(vfs));
    set_dcp_tables(Some(tables));
    set_dcp_metadata(metadata);
}

#[test]
fn decode_with_table_fixture() {
    let bytes = with_table_fixture();
    let package = DcpDecoder::decode(&bytes).expect("decode");
    assert_eq!(package.tables.len(), 1);
    assert_eq!(package.tables[0].0, "orders");
    let table = arrow_ipc_to_table(&package.tables[0].1).expect("arrow");
    assert_eq!(table.headers(), &["id", "date", "value"]);
    assert_eq!(table.rows_ref().unwrap().len(), 2);
}

#[test]
fn ws_source_table_integration() {
    mount_table_session();

    let code = r#"
from ws import source_table, tables, package_info

print("tables:", tables())
global orders = source_table("orders", ["id", "date", "value"])
print("rows:", len(orders))
print("info:", package_info())
"#;

    let capture = OutputCapture::new();
    capture.set_capture(true);
    let result = run_with_vm(code);
    let output = capture.get_output();
    capture.set_capture(false);

    clear_dcp_session();
    set_use_ve(false);

    result.expect("vm run");
    assert!(output.contains("tables:"));
    assert!(output.contains("orders"));
    assert!(output.contains("rows: 2"));
    assert!(output.contains("table_count"));
    assert!(!output.contains("/Users"));
    assert!(!output.contains("temp_sessions"));
    assert!(take_native_error().is_none());
}

#[test]
fn ws_star_import_source_table() {
    mount_table_session();

    let code = r#"
from ws import *
print("tables:", tables())
global orders = source_table("orders", ["id", "date", "value"])
print("rows:", len(orders))
"#;

    let capture = OutputCapture::new();
    capture.set_capture(true);
    let result = run_with_vm(code);
    let output = capture.get_output();
    capture.set_capture(false);

    clear_dcp_session();
    set_use_ve(false);

    result.expect("star import vm run");
    assert!(output.contains("tables:"));
    assert!(output.contains("orders"));
    assert!(output.contains("rows: 2"));
    assert!(take_native_error().is_none());
}

#[test]
fn ws_source_table_missing_column_errors() {
    mount_table_session();

    let code = r#"
from ws import source_table
global t = source_table("orders", ["missing_col"])
"#;

    let result = run_with_vm(code);
    clear_dcp_session();
    set_use_ve(false);

    assert!(result.is_err() || take_native_error().is_some());
}

#[test]
fn ws_no_session_errors() {
    clear_dcp_session();
    set_use_ve(true);

    let code = r#"
from ws import tables
print(tables())
"#;

    let result = run_with_vm(code);
    set_use_ve(false);

    assert!(result.is_err() || take_native_error().is_some());
}

#[test]
fn ws_package_info_no_path_leaks() {
    mount_table_session();

    let code = r#"
from ws import package_info, assets
global info = package_info()
global paths = assets()
print("safe")
"#;

    let capture = OutputCapture::new();
    capture.set_capture(true);
    let result = run_with_vm(code);
    let output = capture.get_output();
    capture.set_capture(false);

    clear_dcp_session();
    set_use_ve(false);

    result.expect("vm run");
    assert!(output.contains("safe"));
    assert!(!output.contains("/Users"));
    assert!(!output.contains("temp_sessions"));
}
