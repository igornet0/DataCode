//! Typed SQLite schema + `_datacode_schema` / `_datacode_version` round-trips.

use chrono::{FixedOffset, TimeZone, Utc};
use data_code::common::numeric::IntValue;
use data_code::common::table::Table;
use data_code::common::value::{ByteBuffer, Value};
use data_code::database_engine::engine::DatabaseEngine;
use data_code::sqlite_export::export_single_table;
use data_code::sqlite_export::type_map::{
    datacode_to_sqlite_declared, is_datacode_system_table, serialize_datetime,
    sqlite_declared_to_datacode, SCHEMA_VERSION,
};
use data_code::{run, Value as DcValue};
use rusqlite::Connection;
use std::collections::HashMap;
use tempfile::TempDir;

fn utc_midnight(y: i32, m: u32, d: u32) -> chrono::DateTime<FixedOffset> {
    Utc.with_ymd_and_hms(y, m, d, 0, 0, 0)
        .unwrap()
        .with_timezone(&FixedOffset::east_opt(0).unwrap())
}

fn utc_dt(y: i32, m: u32, d: u32, h: u32, mi: u32, s: u32) -> chrono::DateTime<FixedOffset> {
    Utc.with_ymd_and_hms(y, m, d, h, mi, s)
        .unwrap()
        .with_timezone(&FixedOffset::east_opt(0).unwrap())
}

fn escape_path(p: &std::path::Path) -> String {
    p.to_string_lossy().replace('\\', "\\\\")
}

#[test]
fn mapping_declared_types() {
    assert_eq!(datacode_to_sqlite_declared("date"), "DATE");
    assert_eq!(datacode_to_sqlite_declared("datetime"), "DATETIME");
    assert_eq!(datacode_to_sqlite_declared("time"), "TIME");
    assert_eq!(datacode_to_sqlite_declared("duration"), "INTEGER");
    assert_eq!(datacode_to_sqlite_declared("bytes"), "BLOB");
    assert_eq!(datacode_to_sqlite_declared("bool"), "INTEGER");
    assert_eq!(sqlite_declared_to_datacode("DATE"), "date");
    assert_eq!(sqlite_declared_to_datacode("DATETIME"), "datetime");
    assert!(is_datacode_system_table("_datacode_schema"));
    assert!(!is_datacode_system_table("users"));
}

#[test]
fn export_writes_pragma_and_metadata() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("typed.db");

    let table = Table::from_data(
        vec![vec![
            Value::Int(IntValue::Finite(1)),
            Value::Date(utc_midnight(2026, 8, 3)),
            Value::Date(utc_dt(2026, 8, 3, 12, 30, 0)),
            Value::Duration(chrono::Duration::seconds(90)),
            Value::Bool(true),
            Value::ByteBuffer(ByteBuffer::from_vec(vec![1, 2, 3])),
            Value::Null,
        ]],
        Some(vec![
            "id".into(),
            "day".into(),
            "ts".into(),
            "span".into(),
            "ok".into(),
            "blob".into(),
            "maybe".into(),
        ]),
    );
    export_single_table(&table, &path, "events").expect("export");

    let conn = Connection::open(&path).unwrap();

    let ver: i64 = conn
        .query_row(
            "SELECT version FROM _datacode_version LIMIT 1",
            [],
            |r| r.get(0),
        )
        .expect("version row");
    assert_eq!(ver, SCHEMA_VERSION);

    let mut types: HashMap<String, String> = HashMap::new();
    let mut stmt = conn.prepare("PRAGMA table_info(events)").unwrap();
    let rows = stmt
        .query_map([], |r| {
            Ok((r.get::<_, String>(1)?, r.get::<_, String>(2)?))
        })
        .unwrap();
    for row in rows {
        let (name, ty) = row.unwrap();
        types.insert(name, ty.to_ascii_uppercase());
    }
    assert_eq!(types.get("day").map(|s| s.as_str()), Some("DATE"));
    assert_eq!(types.get("ts").map(|s| s.as_str()), Some("DATETIME"));
    assert_eq!(types.get("span").map(|s| s.as_str()), Some("INTEGER"));
    assert_eq!(types.get("ok").map(|s| s.as_str()), Some("INTEGER"));
    assert_eq!(types.get("blob").map(|s| s.as_str()), Some("BLOB"));

    let mut meta: HashMap<String, (String, String)> = HashMap::new();
    let mut stmt = conn
        .prepare(
            "SELECT column_name, datacode_type, sqlite_type FROM _datacode_schema WHERE table_name = 'events'",
        )
        .unwrap();
    let rows = stmt
        .query_map([], |r| {
            Ok((
                r.get::<_, String>(0)?,
                r.get::<_, String>(1)?,
                r.get::<_, String>(2)?,
            ))
        })
        .unwrap();
    for row in rows {
        let (col, dc, sql) = row.unwrap();
        meta.insert(col, (dc, sql));
    }
    assert_eq!(meta.get("day").map(|(a, b)| (a.as_str(), b.as_str())), Some(("date", "DATE")));
    assert_eq!(
        meta.get("ts").map(|(a, b)| (a.as_str(), b.as_str())),
        Some(("datetime", "DATETIME"))
    );
    assert_eq!(
        meta.get("span").map(|(a, b)| (a.as_str(), b.as_str())),
        Some(("duration", "INTEGER"))
    );
}

#[test]
fn engine_roundtrip_date_duration_null() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("rt.db");
    let day = utc_midnight(2024, 1, 15);
    let ts = utc_dt(2024, 1, 15, 13, 45, 0);
    let table = Table::from_data(
        vec![
            vec![
                Value::Date(day),
                Value::Date(ts),
                Value::Duration(chrono::Duration::nanoseconds(1_500_000_000)),
                Value::Null,
            ],
            vec![
                Value::Date(day),
                Value::Date(ts),
                Value::Duration(chrono::Duration::seconds(1)),
                Value::String("x".into()),
            ],
        ],
        Some(vec![
            "day".into(),
            "ts".into(),
            "span".into(),
            "maybe".into(),
        ]),
    );
    export_single_table(&table, &path, "rt").unwrap();

    let url = format!("sqlite:{}", path.display());
    let mut engine = DatabaseEngine::from_url(
        url,
        false,
        false,
        1,
        0,
        None,
        HashMap::new(),
    )
    .unwrap();
    let out = engine
        .query("SELECT day, ts, span, maybe FROM rt ORDER BY maybe IS NOT NULL, maybe", &[])
        .unwrap();
    assert_eq!(out.len(), 2);

    let row0 = out.get_row(0).expect("row0");
    match &row0[0] {
        Value::Date(d) => {
            assert_eq!(d.with_timezone(&Utc).date_naive(), day.with_timezone(&Utc).date_naive());
        }
        other => panic!("day expected Date, got {:?}", other),
    }
    match &row0[1] {
        Value::Date(d) => {
            assert_eq!(d.with_timezone(&Utc), ts.with_timezone(&Utc));
        }
        other => panic!("ts expected Date, got {:?}", other),
    }
    match &row0[2] {
        Value::Duration(d) => assert_eq!(d.num_nanoseconds(), Some(1_500_000_000)),
        other => panic!("span expected Duration, got {:?}", other),
    }
    assert!(matches!(&row0[3], Value::Null));
}

#[test]
fn datetime_utc_canonical_on_disk() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("canon.db");
    let ts = utc_dt(2026, 8, 3, 12, 0, 0);
    let table = Table::from_data(
        vec![vec![Value::Date(ts)]],
        Some(vec!["ts".into()]),
    );
    export_single_table(&table, &path, "t").unwrap();
    let conn = Connection::open(&path).unwrap();
    let stored: String = conn
        .query_row("SELECT ts FROM t", [], |r| r.get(0))
        .unwrap();
    assert_eq!(stored, serialize_datetime(&ts));
    assert!(stored.ends_with('Z'));
}

#[test]
fn legacy_db_without_metadata_uses_declared_fallback() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("legacy.db");
    {
        let conn = Connection::open(&path).unwrap();
        conn.execute_batch(
            "
            CREATE TABLE legacy (
                label TEXT,
                day DATE,
                ts DATETIME
            );
            INSERT INTO legacy VALUES ('plain', '2026-08-03', '2026-08-03T12:00:00.000Z');
            ",
        )
        .unwrap();
    }

    let url = format!("sqlite:{}", path.display());
    let mut engine = DatabaseEngine::from_url(
        url,
        false,
        false,
        1,
        0,
        None,
        HashMap::new(),
    )
    .unwrap();
    let out = engine
        .query("SELECT label, day, ts FROM legacy", &[])
        .unwrap();
    let row = out.get_row(0).unwrap();
    assert!(matches!(&row[0], Value::String(s) if s == "plain"));
    assert!(matches!(&row[1], Value::Date(_)));
    assert!(matches!(&row[2], Value::Date(_)));
}

#[test]
fn save_sqlite_language_writes_schema_meta() {
    let dir = TempDir::new().unwrap();
    let out = escape_path(&dir.path().join("lang.db"));
    let src = format!(
        r#"
        events = table([
            [date("2024-01-15"), date("2024-01-15T13:45:00Z"), duration(seconds=90)]
        ], ["day", "ts", "span"])
        events.save_sqlite("{}")
        "#,
        out
    );
    let v = run(&src).unwrap_or_else(|e| panic!("{:?}", e));
    let path = match v {
        DcValue::String(s) => s,
        other => panic!("expected path, got {:?}", other),
    };
    let conn = Connection::open(&path).unwrap();
    let count: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM _datacode_schema WHERE table_name = 'events'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert!(count >= 3);
    let day_ty: String = conn
        .query_row(
            "SELECT sqlite_type FROM _datacode_schema WHERE table_name='events' AND column_name='day'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(day_ty, "DATE");
}

#[test]
fn export_sniffs_string_iso_and_numbers() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("sniff.db");
    let table = Table::from_data(
        vec![
            vec![
                Value::String("1".into()),
                Value::String("233.4".into()),
                Value::String("2024-01-15".into()),
                Value::String("2026-08-02T13:20:21.571Z".into()),
                Value::String("bf96:power".into()),
            ],
            vec![
                Value::String("2".into()),
                Value::String("26.1".into()),
                Value::String("2024-01-16".into()),
                Value::String("2026-08-02T14:00:00.000Z".into()),
                Value::String("bf96:temp".into()),
            ],
        ],
        Some(vec![
            "row_id".into(),
            "value".into(),
            "day".into(),
            "ts".into(),
            "id".into(),
        ]),
    );
    export_single_table(&table, &path, "source_data").unwrap();
    let conn = Connection::open(&path).unwrap();

    let mut pragma: HashMap<String, String> = HashMap::new();
    let mut stmt = conn.prepare("PRAGMA table_info(source_data)").unwrap();
    for row in stmt
        .query_map([], |r| Ok((r.get::<_, String>(1)?, r.get::<_, String>(2)?)))
        .unwrap()
    {
        let (n, t) = row.unwrap();
        pragma.insert(n, t.to_ascii_uppercase());
    }
    assert_eq!(pragma.get("row_id").map(String::as_str), Some("INTEGER"));
    assert_eq!(pragma.get("value").map(String::as_str), Some("REAL"));
    assert_eq!(pragma.get("day").map(String::as_str), Some("DATE"));
    assert_eq!(pragma.get("ts").map(String::as_str), Some("DATETIME"));
    assert_eq!(pragma.get("id").map(String::as_str), Some("TEXT"));

    let ts_meta: String = conn
        .query_row(
            "SELECT datacode_type FROM _datacode_schema WHERE table_name='source_data' AND column_name='ts'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(ts_meta, "datetime");

    let stored: String = conn
        .query_row("SELECT ts FROM source_data WHERE row_id = 1", [], |r| r.get(0))
        .unwrap();
    assert!(stored.ends_with('Z'), "canonical UTC: {}", stored);
}

#[test]
#[ignore = "ALTER TABLE API not implemented; schema sync stub only"]
fn alter_sync_schema_stub() {
    let mut engine = DatabaseEngine::from_url(
        "sqlite::memory:".into(),
        false,
        false,
        1,
        0,
        None,
        HashMap::new(),
    )
    .unwrap();
    let err = engine.sync_schema_after_alter("users").unwrap_err();
    assert!(err.contains("stub") || err.contains("not implemented"));
}
