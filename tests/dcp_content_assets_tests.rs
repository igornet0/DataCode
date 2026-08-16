//! Content-addressed DCP assets (`assets/{sha256}`, `asset://`).

use data_code::common::table::Table;
use data_code::common::value::Value;
use data_code::dcp::{
    arrow_ipc_to_table, clear_dcp_session, parse_asset_ref, set_dcp_content_assets, set_dcp_metadata,
    set_dcp_tables, set_dcp_vfs, validate_table_asset_refs, DcpDecoder, DcpTables, DcpVfs,
};
use data_code::run_with_vm;
use data_code::sqlite_export;
use data_code::websocket::{output_capture::OutputCapture, set_use_ve, take_native_error};
use rusqlite::Connection;
use std::fs;
use std::sync::Arc;

fn fixture_bytes() -> Vec<u8> {
    fs::read("tests/dcp_fixtures/with_content_assets.dcp").expect("fixture")
}

fn mount_content_session() {
    let package = DcpDecoder::decode(&fixture_bytes()).expect("decode");
    let vfs = Arc::new(DcpVfs::from_assets(package.assets.clone()).expect("vfs"));
    let tables = Arc::new(DcpTables::from_entries(package.tables.clone()));
    set_use_ve(true);
    set_dcp_vfs(Some(vfs));
    set_dcp_content_assets(Some(package.content_assets));
    set_dcp_tables(Some(tables));
    set_dcp_metadata(package.metadata.map(|m| m.to_map()));
}

#[test]
fn decode_content_assets_fixture() {
    let package = DcpDecoder::decode(&fixture_bytes()).expect("decode");
    assert_eq!(package.content_assets.len(), 2);
    assert!(package.assets.is_empty());
    assert_eq!(package.tables.len(), 1);
    let table = arrow_ipc_to_table(&package.tables[0].1).expect("arrow");
    assert!(table.headers().contains(&"avatar".to_string()));
    let row0 = table.get_row(0).expect("row");
    let Value::String(avatar) = &row0[2] else {
        panic!("avatar not string");
    };
    let id = parse_asset_ref(avatar).expect("asset ref");
    assert!(package.content_assets.contains(id));
}

#[test]
fn validate_missing_asset_ref_errors() {
    let store = data_code::dcp::ContentAssetStore::new();
    let table = Table::from_data(
        vec![vec![
            Value::Number(1.0),
            Value::String("asset://deadbeef".into()),
        ]],
        Some(vec!["id".into(), "avatar".into()]),
    );
    let err = validate_table_asset_refs("employees", &table, &store).unwrap_err();
    assert!(err.contains("deadbeef"));
    assert!(err.contains("employees.avatar"));
}

#[test]
fn ws_content_assets_and_source_table() {
    mount_content_session();

    let code = r#"
from ws import source_table, content_assets, has_content_asset, package_info

ids = content_assets()
print("count", len(ids))
print("has0", has_content_asset(ids[0]))
global employees = source_table("employees")
print("rows", len(employees))
print("info", package_info())
"#;

    let capture = OutputCapture::new();
    capture.set_capture(true);
    let result = run_with_vm(code);
    let output = capture.get_output();
    capture.set_capture(false);
    clear_dcp_session();
    set_use_ve(false);

    result.expect("vm run");
    assert!(output.contains("count 2"), "{output}");
    assert!(output.contains("has0 true"), "{output}");
    assert!(output.contains("rows 3"), "{output}");
    assert!(output.contains("content_asset_count"), "{output}");
    assert!(take_native_error().is_none());
}

#[test]
fn export_content_assets_sqlite() {
    let package = DcpDecoder::decode(&fixture_bytes()).expect("decode");
    let dir = tempfile::tempdir().expect("tempdir");
    let db_path = dir.path().join("model.db");

    // Minimal DB with a business table so export path is realistic.
    {
        let conn = Connection::open(&db_path).unwrap();
        conn.execute_batch("CREATE TABLE employees (id INTEGER); INSERT INTO employees VALUES (1);")
            .unwrap();
    }

    sqlite_export::export_content_assets(&db_path, &package.content_assets).expect("export");

    let conn = Connection::open(&db_path).unwrap();
    let count: i64 = conn
        .query_row("SELECT COUNT(*) FROM __assets", [], |r| r.get(0))
        .unwrap();
    assert_eq!(count, 2);
    let kind: String = conn
        .query_row(
            "SELECT kind FROM __assets WHERE id = ?1",
            [package.content_assets.ids()[0].as_str()],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(kind, "image");
}
