//! Apply post-model SQL scripts to exported SQLite databases.

use std::path::Path;

use rusqlite::Connection;

/// Run `sql` against the SQLite file at `db_path` inside one transaction.
/// On error the transaction is rolled back and the file on disk is unchanged.
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
        Ok(()) => conn
            .execute_batch("COMMIT;")
            .map_err(|e| format!("SQL error: failed to commit transaction: {e}")),
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
    applied
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
}
