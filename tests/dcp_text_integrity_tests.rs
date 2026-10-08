//! Text values must survive DCP → VM → SQLite export unchanged (issues #6, #7)
//! and `--build_model` publishes only `global` tables (issue #8).

use arrow::array::{ArrayRef, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::ipc::writer::StreamWriter;
use arrow::record_batch::RecordBatch;
use data_code::dcp::{clear_dcp_session, set_dcp_tables, set_dcp_vfs, DcpTables, DcpVfs};
use data_code::run_with_vm;
use data_code::sqlite_export::{export_to_sqlite, get_exported_tables, FkCheckMode};
use data_code::websocket::set_use_ve;
use rusqlite::Connection;
use std::sync::{Arc, Mutex};

/// DCP tables live in process-global session state.
static DCP_SESSION: Mutex<()> = Mutex::new(());

fn arrow_ipc(columns: &[(&str, Vec<Option<&str>>)]) -> Vec<u8> {
    let fields: Vec<Field> = columns
        .iter()
        .map(|(name, _)| Field::new(*name, DataType::Utf8, true))
        .collect();
    let arrays: Vec<ArrayRef> = columns
        .iter()
        .map(|(_, values)| Arc::new(StringArray::from(values.clone())) as ArrayRef)
        .collect();
    let schema = Arc::new(Schema::new(fields));
    let batch = RecordBatch::try_new(schema.clone(), arrays).expect("batch");
    let mut buf = Vec::new();
    {
        let mut writer = StreamWriter::try_new(&mut buf, &schema).expect("writer");
        writer.write(&batch).expect("write");
        writer.finish().expect("finish");
    }
    buf
}

fn session() -> std::sync::MutexGuard<'static, ()> {
    DCP_SESSION.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn mount(tables: Vec<(&str, Vec<u8>)>) {
    set_use_ve(true);
    // A DCP session is active only with a mounted VFS (as in a real package).
    set_dcp_vfs(Some(Arc::new(DcpVfs::from_assets(Vec::new()).expect("vfs"))));
    let entries = tables.into_iter().map(|(n, b)| (n.to_string(), b)).collect();
    set_dcp_tables(Some(Arc::new(DcpTables::from_entries(entries))));
}

/// Run code, export the model to a temp SQLite file, return an open connection.
fn build_model(code: &str) -> (Connection, tempfile::TempDir) {
    let (_value, mut vm) = run_with_vm(code).expect("script runs");
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("model.db");
    export_to_sqlite(&mut vm, path.to_str().unwrap(), false, FkCheckMode::Warn).expect("export");
    (Connection::open(&path).expect("open sqlite"), dir)
}

fn user_tables(conn: &Connection) -> Vec<String> {
    let mut stmt = conn
        .prepare(
            "SELECT name FROM sqlite_master WHERE type='table' \
             AND name NOT LIKE 'sqlite_%' AND name NOT LIKE '_datacode%' AND name <> '__assets' \
             ORDER BY name",
        )
        .unwrap();
    stmt.query_map([], |r| r.get::<_, String>(0))
        .unwrap()
        .map(Result::unwrap)
        .collect()
}

fn column_type(conn: &Connection, table: &str, column: &str) -> String {
    conn.query_row(
        &format!("SELECT type FROM pragma_table_info('{table}') WHERE name = ?1"),
        [column],
        |r| r.get(0),
    )
    .unwrap()
}

#[test]
fn identifier_like_text_survives_dcp_vm_and_export() {
    let _guard = session();
    mount(vec![(
        "t",
        arrow_ipc(&[
            ("code", vec![Some("007"), Some("12")]),
            ("phone", vec![Some("+79161234567"), Some("89161234567")]),
            ("zip", vec![Some("01234"), Some("10001")]),
            ("price", vec![Some("001.20"), Some("3.50")]),
        ]),
    )]);

    let code = r#"
from ws import source_table
global t = source_table("t")
global hits = t["code" == "007"]
"#;
    let (conn, _dir) = build_model(code);
    clear_dcp_session();

    // #6: inside the VM the value is still the string "007".
    let hits: i64 = conn.query_row("SELECT COUNT(*) FROM hits", [], |r| r.get(0)).unwrap();
    assert_eq!(hits, 1, "filter on \"007\" must match the DCP row");

    // #7: exported as TEXT, byte for byte.
    for column in ["code", "phone", "zip", "price"] {
        assert_eq!(column_type(&conn, "t", column), "TEXT", "column {column}");
    }
    let rows: Vec<(String, String, String, String)> = conn
        .prepare("SELECT code, phone, zip, price FROM t ORDER BY rowid")
        .unwrap()
        .query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?)))
        .unwrap()
        .map(Result::unwrap)
        .collect();
    assert_eq!(
        rows,
        vec![
            ("007".into(), "+79161234567".into(), "01234".into(), "001.20".into()),
            ("12".into(), "89161234567".into(), "10001".into(), "3.50".into()),
        ]
    );
}

#[test]
fn numbers_sent_as_text_still_become_numbers() {
    let _guard = session();
    mount(vec![(
        "t",
        arrow_ipc(&[
            ("id", vec![Some("1"), Some("2"), Some("3")]),
            ("amount", vec![Some("250.5"), Some("50"), Some("")]),
        ]),
    )]);

    let code = r#"
from ws import source_table
global t = source_table("t")
global big = t["amount" > 100]
"#;
    let (conn, _dir) = build_model(code);
    clear_dcp_session();

    assert_eq!(column_type(&conn, "t", "id"), "INTEGER");
    assert_eq!(column_type(&conn, "t", "amount"), "REAL");
    let big: i64 = conn.query_row("SELECT COUNT(*) FROM big", [], |r| r.get(0)).unwrap();
    assert_eq!(big, 1);
    let empty_is_null: i64 = conn
        .query_row("SELECT COUNT(*) FROM t WHERE amount IS NULL", [], |r| r.get(0))
        .unwrap();
    assert_eq!(empty_is_null, 1, "empty cell in a numeric column is NULL");
}

#[test]
fn string_literals_in_code_are_exported_as_text() {
    let _guard = session();
    clear_dcp_session();
    let code = r#"
global lit = table([["007", "+79161234567", "01234"], ["12", "89161234567", "10001"]], ["code", "phone", "zip"])
"#;
    let (conn, _dir) = build_model(code);
    assert_eq!(column_type(&conn, "lit", "code"), "TEXT");
    let first: (String, String, String) = conn
        .query_row("SELECT code, phone, zip FROM lit ORDER BY rowid LIMIT 1", [], |r| {
            Ok((r.get(0)?, r.get(1)?, r.get(2)?))
        })
        .unwrap();
    assert_eq!(first, ("007".into(), "+79161234567".into(), "01234".into()));
}

#[test]
fn build_model_publishes_only_global_tables() {
    let _guard = session();
    mount(vec![("t", arrow_ipc(&[("id", vec![Some("1")])]))]);

    let code = r#"
from ws import source_table
global g = table([[1]], ["a"])
tmp = table([[2]], ["b"])
src = source_table("t")
global categories = table([], ["id"])
categories = merge_tables(tables=[categories, source_table("t")])
"#;
    let (_value, mut vm) = run_with_vm(code).expect("script runs");
    let mut exported: Vec<String> = get_exported_tables(&mut vm).unwrap().into_keys().collect();
    exported.sort();
    assert_eq!(exported, vec!["categories".to_string(), "g".to_string()]);

    let (conn, _dir) = build_model(code);
    clear_dcp_session();
    assert_eq!(user_tables(&conn), vec!["categories".to_string(), "g".to_string()]);
    let categories: i64 = conn.query_row("SELECT COUNT(*) FROM categories", [], |r| r.get(0)).unwrap();
    assert_eq!(categories, 1);
}
