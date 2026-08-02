//! Env-gated integration tests for multi-DB DataSource connectors.
//!
//! Set one or more of:
//! - DATACODE_MONGO_URI (e.g. mongodb://localhost:27017)
//! - DATACODE_POSTGRES_URL (e.g. postgres://user:pass@localhost:5432/datacode_test)
//! - DATACODE_MYSQL_URL (e.g. mysql://user:pass@localhost:3306/datacode_test)
//! - DATACODE_MSSQL_URL (e.g. mssql://user:pass@localhost:1433/datacode_test)

#[cfg(test)]
mod tests {
    use data_code::{run, Value};

    fn run_ok(source: &str) -> Value {
        run(source).unwrap_or_else(|e| panic!("execution failed: {:?}", e))
    }

    fn env_or_skip(key: &str) -> Option<String> {
        match std::env::var(key) {
            Ok(v) if !v.trim().is_empty() => Some(v),
            _ => {
                eprintln!("skipping: {} not set", key);
                None
            }
        }
    }

    fn escape(s: &str) -> String {
        s.replace('\\', "\\\\").replace('"', "\\\"")
    }

    #[test]
    fn capabilities_sqlite() {
        let v = run_ok(
            r#"
            db = datasource({ type: "sqlite", url: "sqlite:///:memory:" })
            db.capabilities.supports_sql
            "#,
        );
        assert_eq!(v, Value::Bool(true));
    }

    #[test]
    fn capabilities_mongodb_declared() {
        // Creating the handle does not require a live server.
        let v = run_ok(
            r#"
            db = datasource({
                type: "mongodb",
                url: "mongodb://localhost:27017",
                database: "testdb",
                collection: "users"
            })
            db.capabilities.supports_aggregation
            "#,
        );
        assert_eq!(v, Value::Bool(true));
    }

    #[test]
    fn flatten_json_file_nested() {
        let base = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/test_data/datasource");
        // Write a nested JSON temp file next to fixtures.
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("nested.json");
        std::fs::write(
            &path,
            r#"[{"_id":"1","name":"John","profile":{"age":25,"city":"Helsinki"}}]"#,
        )
        .unwrap();
        let p = escape(&dir.path().to_string_lossy());
        let src = format!(
            r#"
            src = datasource({{ type: "file", path: "{}" }})
            t = src.get_table({{ path: "nested.json", format: "json", flatten: true }})
            t.columns
            "#,
            p
        );
        let v = run_ok(&src);
        match v {
            Value::Array(rc) => {
                let headers = rc.borrow();
                let as_str: Vec<String> = headers
                    .iter()
                    .filter_map(|h| match h {
                        Value::String(s) => Some(s.clone()),
                        _ => None,
                    })
                    .collect();
                assert!(as_str.iter().any(|h| h == "profile.age"), "got {:?}", as_str);
                assert!(as_str.iter().any(|h| h == "profile.city"), "got {:?}", as_str);
            }
            other => panic!("expected columns array, got {:?}", other),
        }
        let _ = base;
    }

    #[test]
    fn mongodb_roundtrip_when_env_set() {
        let Some(uri) = env_or_skip("DATACODE_MONGO_URI") else {
            return;
        };
        let uri = escape(&uri);
        let src = format!(
            r#"
            db = datasource({{
                type: "mongodb",
                url: "{}",
                database: "datacode_ds_test",
                collection: "users_it"
            }})
            db.connect()
            # wipe via aggregation-free delete not available; insert fresh docs
            rows = table([["Alice", 30], ["Bob", 22]], ["name", "age"])
            db.send_table({{ table: rows, mode: "append", collection: "users_it" }})
            t = db.get_table({{
                filter: {{ age: {{ gt: 18 }} }},
                limit: 10,
                array_mode: "keep"
            }})
            n = len(t)
            cnt_resp = db.request({{ op: "count", filter: {{}} }})
            db.disconnect()
            n > 0
            "#,
            uri
        );
        assert_eq!(run_ok(&src), Value::Bool(true));
    }

    #[test]
    fn postgres_select_when_env_set() {
        let Some(url) = env_or_skip("DATACODE_POSTGRES_URL") else {
            return;
        };
        let url = escape(&url);
        let src = format!(
            r#"
            db = datasource({{ type: "postgresql", url: "{}" }})
            db.connect()
            db.request({{ sql: "CREATE TABLE IF NOT EXISTS ds_users (id INT, name TEXT)" }})
            db.request({{ sql: "DELETE FROM ds_users" }})
            rows = table([[1, "Alice"], [2, "Bob"]], ["id", "name"])
            db.send_table({{ table: rows, mode: "append", table_name: "ds_users" }})
            t = db.get_table({{ sql: "SELECT * FROM ds_users ORDER BY id" }})
            n = len(t)
            db.disconnect()
            n
            "#,
            url
        );
        assert_eq!(run_ok(&src), Value::Number(2.0));
    }

    #[test]
    fn mysql_select_when_env_set() {
        let Some(url) = env_or_skip("DATACODE_MYSQL_URL") else {
            return;
        };
        let url = escape(&url);
        let src = format!(
            r#"
            db = datasource({{ type: "mysql", url: "{}" }})
            db.connect()
            db.request({{ sql: "CREATE TABLE IF NOT EXISTS ds_users (id INT, name TEXT)" }})
            db.request({{ sql: "DELETE FROM ds_users" }})
            rows = table([[1, "Alice"], [2, "Bob"]], ["id", "name"])
            db.send_table({{ table: rows, mode: "append", table_name: "ds_users" }})
            t = db.get_table({{ sql: "SELECT * FROM ds_users ORDER BY id" }})
            n = len(t)
            db.disconnect()
            n
            "#,
            url
        );
        assert_eq!(run_ok(&src), Value::Number(2.0));
    }

    #[test]
    fn mssql_select_when_env_set() {
        let Some(url) = env_or_skip("DATACODE_MSSQL_URL") else {
            return;
        };
        let url = escape(&url);
        let src = format!(
            r#"
            db = datasource({{ type: "mssql", url: "{}" }})
            db.connect()
            db.request({{ sql: "IF OBJECT_ID('ds_users', 'U') IS NULL CREATE TABLE ds_users (id INT, name NVARCHAR(100))" }})
            db.request({{ sql: "DELETE FROM ds_users" }})
            rows = table([[1, "Alice"], [2, "Bob"]], ["id", "name"])
            db.send_table({{ table: rows, mode: "append", table_name: "ds_users" }})
            t = db.get_table({{ sql: "SELECT * FROM ds_users ORDER BY id" }})
            n = len(t)
            db.disconnect()
            n
            "#,
            url
        );
        assert_eq!(run_ok(&src), Value::Number(2.0));
    }
}
