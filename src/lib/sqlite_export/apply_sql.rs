//! Apply post-model SQL scripts to exported SQLite databases.

use std::path::Path;

use rusqlite::Connection;

use super::type_map::{
    datacode_to_sqlite_declared, ensure_system_tables_sql, is_datacode_system_table,
    schema_discovery_from_pragma, sniff_string_datacode_type, validate_schema_type_pair,
    SCHEMA_VERSION, TABLE_SCHEMA, TABLE_VERSION,
};

const TEXT_SAMPLE_LIMIT: usize = 64;

/// Run `sql` against the SQLite file at `db_path` inside one transaction.
/// On error the transaction is rolled back and the file on disk is unchanged.
/// After a successful commit, refreshes `_datacode_schema` from PRAGMA (+ TEXT sniff).
pub fn apply_sql_transaction(db_path: &Path, sql: &str) -> Result<(), String> {
    let trimmed = sql.trim();
    if trimmed.is_empty() {
        return Ok(());
    }

    let conn = Connection::open(db_path)
        .map_err(|e| format!("SQL error: failed to open database: {e}"))?;

    conn.execute_batch("BEGIN IMMEDIATE;")
        .map_err(|e| format!("SQL error: failed to begin transaction: {e}"))?;

    match conn.execute_batch(trimmed) {
        Ok(()) => {
            conn.execute_batch("COMMIT;")
                .map_err(|e| format!("SQL error: failed to commit transaction: {e}"))?;
            let _ = resync_datacode_schema(&conn);
            Ok(())
        }
        Err(e) => {
            let _ = conn.execute_batch("ROLLBACK;");
            Err(format!("SQL error: {e}"))
        }
    }
}

/// Soft-apply SQL from `__sql_table__` / `sql_table` / `table_insert`.
///
/// Each statement is executed independently. On failure (missing table, bad
/// columns, etc.) the statement is skipped and a warning is printed.
/// Never fails the overall package.
pub fn apply_sql_table_soft(db_path: &Path, sql: &str) -> usize {
    let trimmed = sql.trim();
    if trimmed.is_empty() {
        return 0;
    }

    let conn = match Connection::open(db_path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("⚠️  sql_table: failed to open database: {e}");
            return 0;
        }
    };

    let mut applied = 0usize;
    for stmt in split_sql_statements(trimmed) {
        if stmt.is_empty() {
            continue;
        }
        match conn.execute_batch(stmt) {
            Ok(()) => applied += 1,
            Err(e) => {
                let preview: String = stmt.chars().take(120).collect();
                eprintln!("⚠️  sql_table skipped: {e} | SQL: {preview}");
            }
        }
    }
    if applied > 0 {
        let _ = resync_datacode_schema(&conn);
    }
    applied
}

/// Rebuild `_datacode_schema` from user-table PRAGMA info, refining TEXT columns
/// when all sampled values sniff as date/datetime.
pub fn resync_datacode_schema(conn: &Connection) -> Result<(), String> {
    conn.execute_batch(ensure_system_tables_sql())
        .map_err(|e| e.to_string())?;
    let count: i64 = conn
        .query_row(
            &format!("SELECT COUNT(*) FROM {}", TABLE_VERSION),
            [],
            |r| r.get(0),
        )
        .map_err(|e| e.to_string())?;
    if count == 0 {
        conn.execute(
            &format!("INSERT INTO {} (version) VALUES (?1)", TABLE_VERSION),
            [SCHEMA_VERSION],
        )
        .map_err(|e| e.to_string())?;
    }

    let tables: Vec<String> = {
        let mut stmt = conn
            .prepare(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'",
            )
            .map_err(|e| e.to_string())?;
        let rows = stmt
            .query_map([], |r| r.get::<_, String>(0))
            .map_err(|e| e.to_string())?;
        rows.filter_map(|r| r.ok())
            .filter(|n| !is_datacode_system_table(n))
            .collect()
    };

    // Drop stale metadata for tables that no longer exist; rebuild upserts below.
    conn.execute(&format!("DELETE FROM {}", TABLE_SCHEMA), [])
        .map_err(|e| e.to_string())?;

    let mut upsert = conn
        .prepare(&format!(
            "INSERT OR REPLACE INTO {}
            (table_name, column_name, datacode_type, sqlite_type, nullable, version)
            VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            TABLE_SCHEMA
        ))
        .map_err(|e| e.to_string())?;

    for table in tables {
        let mut cols = schema_discovery_from_pragma(conn, &table);
        for col in &mut cols {
            let pragma_upper = col.sqlite_type.to_ascii_uppercase();
            let is_textish = pragma_upper.is_empty()
                || pragma_upper == "TEXT"
                || pragma_upper.starts_with("VARCHAR")
                || pragma_upper.starts_with("CHAR")
                || pragma_upper == "CLOB";
            if is_textish {
                if let Some(refined) = refine_text_column_type(conn, &table, &col.column_name) {
                    col.datacode_type = refined.to_string();
                    col.sqlite_type = datacode_to_sqlite_declared(refined).to_string();
                }
            }
            validate_schema_type_pair(&col.datacode_type, &col.sqlite_type)?;
            upsert
                .execute(rusqlite::params![
                    col.table_name,
                    col.column_name,
                    col.datacode_type,
                    col.sqlite_type,
                    if col.nullable { 1i64 } else { 0i64 },
                    SCHEMA_VERSION
                ])
                .map_err(|e| e.to_string())?;
        }
    }
    Ok(())
}

fn refine_text_column_type(
    conn: &Connection,
    table: &str,
    column: &str,
) -> Option<&'static str> {
    let safe_table: String = table
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect();
    let safe_col: String = column
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect();
    let sql = format!(
        "SELECT \"{}\" FROM \"{}\" WHERE \"{}\" IS NOT NULL LIMIT {}",
        safe_col.replace('"', "\"\""),
        safe_table.replace('"', "\"\""),
        safe_col.replace('"', "\"\""),
        TEXT_SAMPLE_LIMIT
    );
    let mut stmt = conn.prepare(&sql).ok()?;
    let rows = stmt
        .query_map([], |r| r.get::<_, String>(0))
        .ok()?;
    let mut saw_date = false;
    let mut saw_datetime = false;
    let mut any = false;
    for row in rows.flatten() {
        any = true;
        match sniff_string_datacode_type(&row) {
            Some("date") => saw_date = true,
            Some("datetime") => saw_datetime = true,
            _ => return None,
        }
    }
    if !any {
        return None;
    }
    if saw_datetime {
        Some("datetime")
    } else if saw_date {
        Some("date")
    } else {
        None
    }
}

/// Split a SQL script into statements on `;`, ignoring empty parts.
/// Simple splitter (no string-aware parsing); matches generated INSERT scripts.
fn split_sql_statements(sql: &str) -> Vec<&str> {
    sql.split(';')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use rusqlite::Connection;
    use std::fs;

    fn seed_db(path: &Path) {
        let conn = Connection::open(path).expect("open");
        conn.execute_batch(
            "CREATE TABLE t (id INTEGER PRIMARY KEY, name TEXT);
             INSERT INTO t (id, name) VALUES (1, 'a');",
        )
        .expect("seed");
    }

    #[test]
    fn apply_sql_adds_view() {
        let dir = tempfile::tempdir().expect("tempdir");
        let db_path = dir.path().join("test.db");
        seed_db(&db_path);
        let before = fs::read(&db_path).expect("read before");

        apply_sql_transaction(
            &db_path,
            "CREATE VIEW v_names AS SELECT name FROM t;",
        )
        .expect("apply");

        let conn = Connection::open(&db_path).expect("open");
        let count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='view' AND name='v_names'",
                [],
                |row| row.get(0),
            )
            .expect("query");
        assert_eq!(count, 1);
        let after = fs::read(&db_path).expect("read after");
        assert_ne!(before, after);
    }

    #[test]
    fn apply_sql_bad_statement_rolls_back() {
        let dir = tempfile::tempdir().expect("tempdir");
        let db_path = dir.path().join("test.db");
        seed_db(&db_path);
        let before = fs::read(&db_path).expect("read before");

        let err = apply_sql_transaction(&db_path, "NOT VALID SQL;").unwrap_err();
        assert!(err.starts_with("SQL error:"));

        let after = fs::read(&db_path).expect("read after");
        assert_eq!(before, after);

        let conn = Connection::open(&db_path).expect("open");
        let views: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='view'",
                [],
                |row| row.get(0),
            )
            .expect("query");
        assert_eq!(views, 0);
    }

    #[test]
    fn soft_sql_table_inserts_and_skips_bad() {
        let dir = tempfile::tempdir().expect("tempdir");
        let db_path = dir.path().join("test.db");
        seed_db(&db_path);

        let applied = apply_sql_table_soft(
            &db_path,
            r#"
INSERT INTO "t" ("id", "name") VALUES (2, 'b');
INSERT INTO "missing" ("id") VALUES (1);
INSERT INTO "t" ("id", "name") VALUES (3, 'c');
"#,
        );
        assert_eq!(applied, 2);

        let conn = Connection::open(&db_path).expect("open");
        let count: i64 = conn
            .query_row("SELECT COUNT(*) FROM t", [], |row| row.get(0))
            .expect("count");
        assert_eq!(count, 3);
    }

    #[test]
    fn soft_sql_table_skips_column_mismatch() {
        let dir = tempfile::tempdir().expect("tempdir");
        let db_path = dir.path().join("test.db");
        seed_db(&db_path);

        let applied = apply_sql_table_soft(
            &db_path,
            r#"INSERT INTO "t" ("id", "nope") VALUES (9, 'x');"#,
        );
        assert_eq!(applied, 0);

        let conn = Connection::open(&db_path).expect("open");
        let count: i64 = conn
            .query_row("SELECT COUNT(*) FROM t", [], |row| row.get(0))
            .expect("count");
        assert_eq!(count, 1);
    }

    #[test]
    fn resync_after_sql_rebuilds_schema_meta() {
        let dir = tempfile::tempdir().expect("tempdir");
        let db_path = dir.path().join("resync.db");
        {
            let conn = Connection::open(&db_path).unwrap();
            conn.execute_batch(
                "
                CREATE TABLE events (
                    row_id INTEGER,
                    value REAL,
                    ts TEXT,
                    id TEXT
                );
                INSERT INTO events VALUES (1, 233.4, '2026-08-02T13:20:21.571Z', 'a');
                INSERT INTO events VALUES (2, 26.1, '2026-08-02T14:00:00.000Z', 'b');
                ",
            )
            .unwrap();
        }

        apply_sql_transaction(
            &db_path,
            "
            DROP TABLE events;
            CREATE TABLE events (
                row_id INTEGER PRIMARY KEY,
                value REAL,
                ts TEXT,
                id TEXT
            );
            INSERT INTO events VALUES (1, 233.4, '2026-08-02T13:20:21.571Z', 'a');
            INSERT INTO events VALUES (2, 26.1, '2026-08-02T14:00:00.000Z', 'b');
            ",
        )
        .unwrap();

        let conn = Connection::open(&db_path).unwrap();
        let row_id_ty: String = conn
            .query_row(
                "SELECT datacode_type FROM _datacode_schema WHERE table_name='events' AND column_name='row_id'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(row_id_ty, "int");
        let ts_ty: String = conn
            .query_row(
                "SELECT datacode_type || '|' || sqlite_type FROM _datacode_schema WHERE table_name='events' AND column_name='ts'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(ts_ty, "datetime|DATETIME");
    }
}
