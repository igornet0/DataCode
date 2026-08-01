//! Integration tests for DCP SQL post-model transactions.

use data_code::dcp::DcpDecoder;
use data_code::run_with_vm;
use data_code::sqlite_export::{
    apply_sql_table_soft, apply_sql_transaction, export_to_sqlite,
};
use rusqlite::Connection;
use std::fs;
use tempfile::TempDir;

fn with_sql_fixture() -> Vec<u8> {
    fs::read("tests/dcp_fixtures/with_sql.dcp").expect("fixture with_sql.dcp")
}

#[test]
fn decode_with_sql_fixture() {
    let package = DcpDecoder::decode(&with_sql_fixture()).expect("decode");
    let sql = package.sql.expect("sql section");
    assert!(sql.contains("CREATE VIEW v_names"));
}

#[test]
fn apply_sql_after_export_creates_view() {
    let code = r#"
global t = table([[1, "a"], [2, "b"]], ["id", "name"])
"#;
    let (_, mut vm) = run_with_vm(code).expect("run");

    let dir = TempDir::new().expect("tempdir");
    let db_path = dir.path().join("model.db");
    export_to_sqlite(&mut vm, db_path.to_str().unwrap(), false).expect("export");

    let sql = "CREATE VIEW v_names AS SELECT name FROM t;";
    apply_sql_transaction(&db_path, sql).expect("apply sql");

    let conn = Connection::open(&db_path).expect("open");
    let count: i64 = conn
        .query_row("SELECT COUNT(*) FROM v_names", [], |row| row.get(0))
        .expect("query view");
    assert_eq!(count, 2);
}

#[test]
fn apply_sql_failure_preserves_pre_sql_database() {
    let code = r#"
global t = table([[1, "a"]], ["id", "name"])
"#;
    let (_, mut vm) = run_with_vm(code).expect("run");

    let dir = TempDir::new().expect("tempdir");
    let db_path = dir.path().join("model.db");
    export_to_sqlite(&mut vm, db_path.to_str().unwrap(), false).expect("export");
    let before = fs::read(&db_path).expect("read before");

    let err = apply_sql_transaction(&db_path, "NOT VALID SQL;").unwrap_err();
    assert!(err.starts_with("SQL error:"));

    let after = fs::read(&db_path).expect("read after");
    assert_eq!(before, after);
}

#[test]
fn soft_sql_table_insert_after_export() {
    let code = r#"
global t = table([[1, "a"]], ["id", "name"])
"#;
    let (_, mut vm) = run_with_vm(code).expect("run");

    let dir = TempDir::new().expect("tempdir");
    let db_path = dir.path().join("model.db");
    export_to_sqlite(&mut vm, db_path.to_str().unwrap(), false).expect("export");

    let applied = apply_sql_table_soft(
        &db_path,
        r#"
INSERT INTO "t" ("id", "name") VALUES (2, 'b');
INSERT INTO "missing" ("x") VALUES (1);
"#,
    );
    assert_eq!(applied, 1);

    let conn = Connection::open(&db_path).expect("open");
    let count: i64 = conn
        .query_row("SELECT COUNT(*) FROM t", [], |row| row.get(0))
        .expect("count");
    assert_eq!(count, 2);
}

#[test]
fn dcp_package_sql_matches_fixture_export_flow() {
    let package = DcpDecoder::decode(&with_sql_fixture()).expect("decode");
    let sql = package.sql.expect("sql");

    let (_, mut vm) = run_with_vm(&package.code).expect("run");

    let dir = TempDir::new().expect("tempdir");
    let db_path = dir.path().join("model.db");
    export_to_sqlite(&mut vm, db_path.to_str().unwrap(), false).expect("export");
    apply_sql_transaction(&db_path, &sql).expect("apply fixture sql");

    let conn = Connection::open(&db_path).expect("open");
    let names: Vec<String> = conn
        .prepare("SELECT name FROM v_names ORDER BY name")
        .expect("prepare")
        .query_map([], |row| row.get(0))
        .expect("query")
        .collect::<Result<Vec<_>, _>>()
        .expect("rows");
    assert_eq!(names, vec!["a".to_string()]);
}
