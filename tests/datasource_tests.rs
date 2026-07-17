//! Integration tests for DataSource API.

#[cfg(test)]
mod tests {
    use data_code::{run, Value};
    use std::path::Path;
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

    fn escape_path(p: &Path) -> String {
        p.to_string_lossy().replace('\\', "\\\\")
    }

    #[test]
    fn datasource_file_get_table_csv() {
        let base = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/test_data/datasource");
        let p = escape_path(&base);
        let src = format!(
            r#"
            src = datasource({{ type: "file", path: "{}" }})
            t = src.get_table({{ path: "sample.csv" }})
            len(t)
            "#,
            p
        );
        assert_eq!(run_ok(&src), Value::Number(3.0));
    }

    #[test]
    fn datasource_file_get_table_json() {
        let base = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/test_data/datasource");
        let p = escape_path(&base);
        let src = format!(
            r#"
            src = datasource({{ type: "file", path: "{}" }})
            t = src.get_table({{ path: "sample.json", format: "json" }})
            len(t)
            "#,
            p
        );
        assert_eq!(run_ok(&src), Value::Number(3.0));
    }

    #[test]
    fn datasource_file_request() {
        let base = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/test_data/datasource");
        let p = escape_path(&base);
        let src = format!(
            r#"
            src = datasource({{ type: "file", path: "{}" }})
            resp = src.request({{ path: "sample.json" }})
            resp.status
            "#,
            p
        );
        assert_eq!(run_ok(&src), Value::Number(200.0));
    }

    #[test]
    fn datasource_sqlite_memory() {
        let src = r#"
            db = datasource({ type: "sqlite", url: "sqlite:///:memory:" })
            db.connect()
            db.get_table({ sql: "SELECT 1 AS n" })
        "#;
        let v = run_ok(src);
        assert!(matches!(v, Value::Table(_)));
    }

    #[test]
    fn datasource_sqlite_send_table_append() {
        let dir = TempDir::new().expect("tempdir");
        let db_path = dir.path().join("test.db");
        let p = escape_path(&db_path);
        let src = format!(
            r#"
            db = datasource({{ type: "sqlite", url: "sqlite:///{}" }})
            db.connect()
            db.request({{ sql: "CREATE TABLE users (id INT, name TEXT)" }})
            rows = table([[1, "Alice"], [2, "Bob"]], ["id", "name"])
            db.send_table({{ table: rows, mode: "append", table_name: "users" }})
            result = db.get_table({{ sql: "SELECT COUNT(*) AS cnt FROM users" }})
            result.rows[0][0]
            "#,
            p
        );
        assert_eq!(run_ok(&src), Value::Number(2.0));
    }

    #[test]
    fn datasource_typeof_isinstance() {
        let base = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/test_data/datasource");
        let p = escape_path(&base);
        let src = format!(
            r#"
            src = datasource({{ type: "file", path: "{}" }})
            resp = src.request({{ path: "sample.json" }})
            typeof(src)
            "#,
            p
        );
        assert_eq!(run_ok(&src), Value::String("datasource".into()));

        let src2 = format!(
            r#"
            src = datasource({{ type: "file", path: "{}" }})
            resp = src.request({{ path: "sample.json" }})
            isinstance(resp, "response")
            "#,
            p
        );
        assert_eq!(run_ok(&src2), Value::Bool(true));
    }

    #[test]
    fn datasource_response_json() {
        let base = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/test_data/datasource");
        let p = escape_path(&base);
        let src = format!(
            r#"
            src = datasource({{ type: "file", path: "{}" }})
            resp = src.request({{ path: "sample.json" }})
            data = resp.json()
            isinstance(data, "array")
            "#,
            p
        );
        assert_eq!(run_ok(&src), Value::Bool(true));
    }

    #[test]
    fn datasource_unsupported_type() {
        let err = run_err(r#"datasource({ type: "kafka", url: "localhost:9092" })"#);
        assert!(err.contains("unknown") || err.contains("unsupported") || err.contains("DatasourceError"));
    }

    #[test]
    fn datasource_validation_missing_type() {
        let err = run_err(r#"datasource({ url: "http://example.com" })"#);
        assert!(err.contains("type") || err.contains("ValidationError"));
    }

    #[test]
    fn datasource_ping_file() {
        let base = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/test_data/datasource");
        let p = escape_path(&base);
        let src = format!(
            r#"
            src = datasource({{ type: "file", path: "{}" }})
            src.ping()
            "#,
            p
        );
        assert_eq!(run_ok(&src), Value::Bool(true));
    }
}
