//! Integration tests for `table.save_csv(path)` and `table.save_sqlite(path)`.

#[cfg(test)]
mod tests {
    use data_code::{run, Value};
    use rusqlite::Connection;
    use std::fs;
    use tempfile::TempDir;

    fn run_ok(source: &str) -> Value {
        run(source).unwrap_or_else(|e| panic!("execution failed: {:?}", e))
    }

    fn run_err(source: &str) -> String {
        match run(source) {
            Ok(v) => panic!("expected error, got {:?}", v),
            Err(e) => format!("{:?}", e),
        }
    }

    fn escape_path(p: &std::path::Path) -> String {
        p.to_string_lossy().replace('\\', "\\\\")
    }

    #[test]
    fn save_csv_creates_file_with_extension() {
        let dir = TempDir::new().expect("tempdir");
        let out_base = escape_path(&dir.path().join("output"));
        let src = format!(
            r#"
            users = table([[1, "Alex"], [2, "Kate"]], ["id", "name"])
            users.save_csv("{}")
        "#,
            out_base
        );
        let v = run_ok(&src);
        let path = match v {
            Value::String(s) => s,
            other => panic!("expected string path, got {:?}", other),
        };
        assert!(path.ends_with("output.csv"));
        let content = fs::read_to_string(&path).expect("read csv");
        assert!(content.contains("id,name"));
        assert!(content.contains("1,Alex"));
        assert!(content.contains("2,Kate"));
    }

    #[test]
    fn save_csv_explicit_extension() {
        let dir = TempDir::new().expect("tempdir");
        let out = escape_path(&dir.path().join("data.csv"));
        let src = format!(
            r#"
            t = table([[1]], ["x"])
            t.save_csv("{}")
        "#,
            out
        );
        let v = run_ok(&src);
        let path = match v {
            Value::String(s) => s,
            _ => panic!("expected string"),
        };
        assert!(path.ends_with("data.csv"));
        assert!(fs::metadata(&path).is_ok());
    }

    #[test]
    fn save_csv_with_path_value() {
        let dir = TempDir::new().expect("tempdir");
        let out_base = escape_path(&dir.path().join("out"));
        let src = format!(
            r#"
            users = table([[1, "A"]], ["id", "name"])
            users.save_csv(path("{}"))
        "#,
            out_base
        );
        let v = run_ok(&src);
        assert!(matches!(v, Value::String(_)));
    }

    #[test]
    fn save_csv_missing_parent_dir_errors() {
        let dir = TempDir::new().expect("tempdir");
        let missing = escape_path(&dir.path().join("no_such_dir_xyz").join("out"));
        let src = format!(
            r#"
            t = table([[1]], ["x"])
            t.save_csv("{}")
        "#,
            missing
        );
        let err = run_err(&src);
        assert!(
            err.contains("Directory does not exist") || err.contains("no_such_dir"),
            "got: {}",
            err
        );
    }

    #[test]
    fn save_csv_invalid_path_type_errors() {
        let err = run_err(
            r#"
            t = table([[1]], ["x"])
            t.save_csv(42)
        "#,
        );
        assert!(
            err.contains("TypeError") || err.contains("path") || err.contains("string"),
            "got: {}",
            err
        );
    }

    #[test]
    fn save_csv_empty_table_writes_headers() {
        let dir = TempDir::new().expect("tempdir");
        let out = escape_path(&dir.path().join("empty"));
        let src = format!(
            r#"
            t = table([], ["a", "b"])
            t.save_csv("{}")
        "#,
            out
        );
        run_ok(&src);
        let path = dir.path().join("empty.csv");
        let content = fs::read_to_string(path).unwrap().trim().to_string();
        assert_eq!(content, "a,b");
    }

    #[test]
    fn save_csv_named_path_kwarg() {
        let dir = TempDir::new().expect("tempdir");
        let out_base = escape_path(&dir.path().join("kw_out"));
        let src = format!(
            r#"
            t = table([[1]], ["x"])
            t.save_csv(path="{}")
        "#,
            out_base
        );
        run_ok(&src);
        assert!(dir.path().join("kw_out.csv").exists());
    }

    #[test]
    fn save_csv_named_filename_alias() {
        let dir = TempDir::new().expect("tempdir");
        let out_base = escape_path(&dir.path().join("fn_out"));
        let src = format!(
            r#"
            t = table([[1]], ["x"])
            t.save_csv(filename="{}")
        "#,
            out_base
        );
        run_ok(&src);
        assert!(dir.path().join("fn_out.csv").exists());
    }

    #[test]
    fn save_csv_subdirectory() {
        let dir = TempDir::new().expect("tempdir");
        let sub = dir.path().join("data");
        fs::create_dir_all(&sub).unwrap();
        let out_base = escape_path(&sub.join("report"));
        let src = format!(
            r#"
            t = table([[1]], ["id"])
            t.save_csv("{}")
        "#,
            out_base
        );
        run_ok(&src);
        assert!(sub.join("report.csv").exists());
    }

    #[test]
    fn save_sqlite_creates_db_with_variable_name() {
        let dir = TempDir::new().expect("tempdir");
        let out_base = escape_path(&dir.path().join("users_db"));
        let src = format!(
            r#"
            users = table([[1, "Alex"], [2, "Kate"]], ["id", "name"])
            users.save_sqlite("{}")
        "#,
            out_base
        );
        let v = run_ok(&src);
        let path = match v {
            Value::String(s) => s,
            _ => panic!("expected path string"),
        };
        assert!(path.ends_with("users_db.sqlite"));
        let conn = Connection::open(&path).expect("open sqlite");
        let count: i64 = conn
            .query_row("SELECT COUNT(*) FROM users", [], |r| r.get(0))
            .expect("count rows");
        assert_eq!(count, 2);
        let name: String = conn
            .query_row("SELECT name FROM users WHERE id = 1", [], |r| r.get(0))
            .expect("row");
        assert_eq!(name, "Alex");
    }

    #[test]
    fn save_sqlite_explicit_db_extension() {
        let dir = TempDir::new().expect("tempdir");
        let out = escape_path(&dir.path().join("report.db"));
        let src = format!(
            r#"
            sales = table([[100]], ["amount"])
            sales.save_sqlite("{}")
        "#,
            out
        );
        run_ok(&src);
        assert!(dir.path().join("report.db").exists());
    }

    #[test]
    fn save_sqlite_unnamed_table_errors() {
        let dir = TempDir::new().expect("tempdir");
        let out = escape_path(&dir.path().join("out"));
        let src = format!(
            r#"
            table([[1]], ["id"]).save_sqlite("{}")
        "#,
            out
        );
        let err = run_err(&src);
        assert!(
            err.contains("no name") || err.contains("variable"),
            "got: {}",
            err
        );
    }

    #[test]
    fn save_sqlite_missing_parent_errors() {
        let dir = TempDir::new().expect("tempdir");
        let missing = escape_path(&dir.path().join("missing_dir_abc").join("db"));
        let src = format!(
            r#"
            users = table([[1]], ["id"])
            users.save_sqlite("{}")
        "#,
            missing
        );
        let err = run_err(&src);
        assert!(err.contains("Directory does not exist"), "got: {}", err);
    }

    #[test]
    fn save_tables_sqlite_multi_table_with_fk() {
        let dir = TempDir::new().expect("tempdir");
        let out_base = escape_path(&dir.path().join("model"));
        let src = format!(
            r#"
            users = table([[1, "Alice"], [2, "Kate"]], ["id", "name"])
            orders = table([[1, 100], [2, 200]], ["user_id", "amount"])
            relate(users["id"], orders["user_id"])
            path = save_tables_sqlite([users, orders], "{}")
            path
        "#,
            out_base
        );
        let v = run_ok(&src);
        let path = match v {
            Value::String(s) => s,
            _ => panic!("expected path string"),
        };
        assert!(path.ends_with("model.sqlite"));
        let conn = Connection::open(&path).expect("open sqlite");
        let users_count: i64 = conn
            .query_row("SELECT COUNT(*) FROM users", [], |r| r.get(0))
            .expect("users count");
        assert_eq!(users_count, 2);
        let orders_count: i64 = conn
            .query_row("SELECT COUNT(*) FROM orders", [], |r| r.get(0))
            .expect("orders count");
        assert_eq!(orders_count, 2);
        let fk_sql: String = conn
            .query_row(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name='orders'",
                [],
                |r| r.get(0),
            )
            .expect("orders ddl");
        assert!(
            fk_sql.contains("FOREIGN KEY") || fk_sql.contains("user_id"),
            "expected FK metadata in orders table: {}",
            fk_sql
        );
    }

    #[test]
    fn save_tables_sqlite_kwargs_in_metadata() {
        let dir = TempDir::new().expect("tempdir");
        let out_base = escape_path(&dir.path().join("meta_db"));
        let src = format!(
            r#"
            users = table([[1]], ["id"])
            save_tables_sqlite([users], filename="{}", env="dev")
        "#,
            out_base
        );
        run_ok(&src);
        let path = dir.path().join("meta_db.sqlite");
        let conn = Connection::open(&path).expect("open sqlite");
        let env: String = conn
            .query_row(
                "SELECT value FROM _datacode_variables WHERE variable_name = 'env'",
                [],
                |r| r.get(0),
            )
            .expect("env row");
        assert_eq!(env, "dev");
    }

    #[test]
    fn save_tables_sqlite_kwargs_spread() {
        let dir = TempDir::new().expect("tempdir");
        let out_base = escape_path(&dir.path().join("spread_db"));
        let src = format!(
            r#"
            users = table([[1]], ["id"])
            meta = {{ "env": "prod", "version": 2 }}
            save_tables_sqlite([users], **meta, filename="{}")
        "#,
            out_base
        );
        run_ok(&src);
        let conn = Connection::open(dir.path().join("spread_db.sqlite")).expect("open");
        let env: String = conn
            .query_row(
                "SELECT value FROM _datacode_variables WHERE variable_name = 'env'",
                [],
                |r| r.get(0),
            )
            .expect("env");
        assert_eq!(env, "prod");
    }
}
