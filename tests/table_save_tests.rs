//! Integration tests for `table.save_csv(path)` and `table.save_sqlite(path)`.

#[cfg(test)]
mod tests {
    use data_code::sqlite_export::FkCheckMode;
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

    /// `(from_col, referenced_table, to_col)` from `PRAGMA foreign_key_list`.
    fn pragma_foreign_keys(conn: &Connection, table: &str) -> Vec<(String, String, String)> {
        let mut stmt = conn
            .prepare(&format!("PRAGMA foreign_key_list(\"{table}\")"))
            .expect("pragma prepare");
        let rows = stmt
            .query_map([], |r| {
                Ok((
                    r.get::<_, String>(3)?, // from
                    r.get::<_, String>(2)?, // table
                    r.get::<_, String>(4)?, // to
                ))
            })
            .expect("pragma query");
        rows.collect::<Result<Vec<_>, _>>().expect("pragma rows")
    }

    fn assert_fk(
        conn: &Connection,
        table: &str,
        from_col: &str,
        ref_table: &str,
        to_col: &str,
    ) {
        let fks = pragma_foreign_keys(conn, table);
        assert!(
            fks.iter().any(|(from, rt, to)| from == from_col && rt == ref_table && to == to_col),
            "expected FK {table}.{from_col} -> {ref_table}({to_col}), got {fks:?}",
        );
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
    fn relate_varargs_star_creates_multiple_fks() {
        let dir = TempDir::new().expect("tempdir");
        let out_base = escape_path(&dir.path().join("star_model"));
        let src = format!(
            r#"
            users = table([[1, "Alice"], [2, "Kate"]], ["id", "name"])
            orders = table([[1, 100], [2, 200]], ["user_id", "amount"])
            invoices = table([[1, 10], [2, 20]], ["user_id", "total"])
            relate(users["id"], orders["user_id"], invoices["user_id"])
            path = save_tables_sqlite([users, orders, invoices], "{}")
            path
        "#,
            out_base
        );
        let v = run_ok(&src);
        let path = match v {
            Value::String(s) => s,
            _ => panic!("expected path string"),
        };
        let conn = Connection::open(&path).expect("open sqlite");
        let orders_sql: String = conn
            .query_row(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name='orders'",
                [],
                |r| r.get(0),
            )
            .expect("orders ddl");
        let invoices_sql: String = conn
            .query_row(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name='invoices'",
                [],
                |r| r.get(0),
            )
            .expect("invoices ddl");
        assert!(
            orders_sql.contains("FOREIGN KEY") && orders_sql.contains("user_id"),
            "orders FK missing: {}",
            orders_sql
        );
        assert!(
            invoices_sql.contains("FOREIGN KEY") && invoices_sql.contains("user_id"),
            "invoices FK missing: {}",
            invoices_sql
        );
    }

    #[test]
    fn relate_array_star_creates_multiple_fks() {
        let dir = TempDir::new().expect("tempdir");
        let out_base = escape_path(&dir.path().join("array_model"));
        let src = format!(
            r#"
            users = table([[1, "Alice"], [2, "Kate"]], ["id", "name"])
            orders = table([[1, 100], [2, 200]], ["user_id", "amount"])
            invoices = table([[1, 10], [2, 20]], ["user_id", "total"])
            relate([users["id"], orders["user_id"], invoices["user_id"]])
            path = save_tables_sqlite([users, orders, invoices], "{}")
            path
        "#,
            out_base
        );
        let v = run_ok(&src);
        let path = match v {
            Value::String(s) => s,
            _ => panic!("expected path string"),
        };
        let conn = Connection::open(&path).expect("open sqlite");
        let orders_sql: String = conn
            .query_row(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name='orders'",
                [],
                |r| r.get(0),
            )
            .expect("orders ddl");
        let invoices_sql: String = conn
            .query_row(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name='invoices'",
                [],
                |r| r.get(0),
            )
            .expect("invoices ddl");
        assert!(
            orders_sql.contains("FOREIGN KEY") && orders_sql.contains("user_id"),
            "orders FK missing: {}",
            orders_sql
        );
        assert!(
            invoices_sql.contains("FOREIGN KEY") && invoices_sql.contains("user_id"),
            "invoices FK missing: {}",
            invoices_sql
        );
    }

    #[test]
    fn relate_too_few_args_errors() {
        let err = run_err(
            r#"
            users = table([[1]], ["id"])
            relate(users["id"])
        "#,
        );
        assert!(
            err.contains("at least 2 column"),
            "expected arity error, got: {}",
            err
        );
    }

    #[test]
    fn relate_empty_array_errors() {
        let err = run_err(
            r#"
            relate([])
        "#,
        );
        assert!(
            err.contains("at least 2 column"),
            "expected arity error, got: {}",
            err
        );
    }

    #[test]
    fn relate_non_column_errors() {
        let err = run_err(
            r#"
            users = table([[1]], ["id"])
            relate(users["id"], 123)
        "#,
        );
        assert!(
            err.contains("column reference"),
            "expected TypeError for non-column, got: {}",
            err
        );
    }

    /// Mirrors `xx.dc`: two `relate` calls (star on documents + employees → documents.author_id).
    /// Asserts SQLite `PRAGMA foreign_key_list`, not just CREATE TABLE text.
    fn platform_documents_model_source(out_base: &str) -> String {
        format!(
            r#"
            global document_field_values = table(
                [["d1", "title", "str", "A"]],
                ["document_id", "field_name", "field_type", "value"]
            )
            global document_table_cells = table(
                [["d1", "grid", "s1", "Sheet", 0, "c0", "Col", "x"]],
                ["document_id", "field_name", "sheet_id", "sheet_name", "row_index", "column_key", "column_label", "value"]
            )
            global document_tables = table(
                [["d1", "grid", "s1", "Sheet", "file", 1, 1]],
                ["document_id", "field_name", "sheet_id", "sheet_name", "source_type", "row_count", "col_count"]
            )
            global documents = table(
                [["d1", "Doc", "N-1", "e1"]],
                ["id", "title", "number", "author_id"]
            )
            global employees = table(
                [["e1", "alice", "Alice"]],
                ["id", "login", "name"]
            )

            primary_key(documents["id"])
            relate(documents["id"], document_tables["document_id"], document_table_cells["document_id"], document_field_values["document_id"])

            primary_key(employees["id"])
            relate(employees["id"], documents["author_id"])

            path = save_tables_sqlite(
                [document_field_values, document_table_cells, document_tables, documents, employees],
                "{}"
            )
            path
        "#,
            out_base
        )
    }

    fn assert_platform_document_fks(conn: &Connection) {
        assert_fk(conn, "document_tables", "document_id", "documents", "id");
        assert_fk(conn, "document_table_cells", "document_id", "documents", "id");
        assert_fk(conn, "document_field_values", "document_id", "documents", "id");
        assert_fk(conn, "documents", "author_id", "employees", "id");
    }

    #[test]
    fn relate_two_calls_employees_documents_fk_in_sqlite() {
        let dir = TempDir::new().expect("tempdir");
        let out_base = escape_path(&dir.path().join("platform_model"));
        let v = run_ok(&platform_documents_model_source(&out_base));
        let path = match v {
            Value::String(s) => s,
            _ => panic!("expected path string"),
        };
        let conn = Connection::open(&path).expect("open sqlite");
        assert_platform_document_fks(&conn);
    }

    /// Mirrors xx.dc: filter parent `orders` to active, then relate unfiltered children.
    /// Export must fail FK check (orphan payments / order_items).
    #[test]
    fn relate_after_filter_parent_orphans_children_fk_check() {
        let dir = TempDir::new().expect("tempdir");
        let out_base = escape_path(&dir.path().join("orphan_fk"));
        // Distinct headers so save_tables_sqlite can name tables (payments vs order_items).
        let src = format!(
            r#"
            global orders = table(
                [["o1", "active"], ["o2", "cancelled"], ["o3", "active"]],
                ["id", "status"]
            )
            global payments = table(
                [["p1", "o1", 10], ["p2", "o2", 20], ["p3", "o3", 30]],
                ["id", "order_id", "amount"]
            )
            global order_items = table(
                [["i1", "o1", "sku"], ["i2", "o2", "sku"]],
                ["id", "order_id", "sku"]
            )

            orders = orders["status" == "active"]

            primary_key(orders["id"])
            relate(orders["id"], payments["order_id"])
            relate(orders["id"], order_items["order_id"])

            save_tables_sqlite([orders, payments, order_items], "{}")
        "#,
            out_base
        );
        let err = run_err(&src);
        assert!(
            err.contains("foreign_key_check"),
            "expected FK integrity error, got: {err}"
        );
        assert!(
            err.contains("payments.order_id") || err.contains("order_items.order_id"),
            "error should name orphan FK, got: {err}"
        );
    }

    #[test]
    fn relate_after_filter_parent_then_filter_children_ok() {
        let dir = TempDir::new().expect("tempdir");
        let out_base = escape_path(&dir.path().join("orphan_fk_fixed"));
        let src = format!(
            r#"
            global orders = table(
                [["o1", "active"], ["o2", "cancelled"], ["o3", "active"]],
                ["id", "status"]
            )
            global payments = table(
                [["p1", "o1"], ["p2", "o2"], ["p3", "o3"]],
                ["id", "order_id"]
            )

            orders = orders["status" == "active"]
            payments = payments["order_id" in orders["id"]]

            primary_key(orders["id"])
            relate(orders["id"], payments["order_id"])

            path = save_tables_sqlite([orders, payments], "{}")
            path
        "#,
            out_base
        );
        let v = run_ok(&src);
        let path = match v {
            Value::String(s) => s,
            _ => panic!("expected path string"),
        };
        let conn = Connection::open(&path).expect("open sqlite");
        assert_fk(&conn, "payments", "order_id", "orders", "id");
        let n_orders: i64 = conn
            .query_row("SELECT COUNT(*) FROM orders", [], |r| r.get(0))
            .unwrap();
        let n_payments: i64 = conn
            .query_row("SELECT COUNT(*) FROM payments", [], |r| r.get(0))
            .unwrap();
        assert_eq!(n_orders, 2);
        assert_eq!(n_payments, 2);
    }

    fn orphan_fk_model_source() -> &'static str {
        r#"
            global customers = table(
                [["c1", "Ann"]],
                ["id", "name"]
            )
            global orders = table(
                [["o1", "active", "c1"], ["o2", "cancelled", "c1"], ["o3", "active", "c1"]],
                ["id", "status", "customer_id"]
            )
            global payments = table(
                [["p1", "o1", 10], ["p2", "o2", 20], ["p3", "o3", 30]],
                ["id", "order_id", "amount"]
            )

            orders = orders["status" == "active"]
            primary_key(customers["id"])
            relate(customers["id"], orders["customer_id"])
            primary_key(orders["id"])
            relate(orders["id"], payments["order_id"])
        "#
    }

    #[test]
    fn fk_check_warn_drops_orphan_fk_keeps_valid() {
        let dir = TempDir::new().expect("tempdir");
        let db_path = dir.path().join("warn.sqlite");
        let (_v, mut vm) = data_code::run_with_vm(orphan_fk_model_source()).expect("run");
        let outcome = data_code::sqlite_export::export_to_sqlite(
            &mut vm,
            db_path.to_str().expect("utf8"),
            false,
            FkCheckMode::Warn,
        )
        .expect("warn export should succeed");
        let warning = outcome.warning.expect("warn should set warning");
        assert!(
            warning.contains("payments.order_id"),
            "warning should mention skipped FK, got {warning}"
        );

        let conn = Connection::open(&db_path).expect("open");
        let payment_fks = pragma_foreign_keys(&conn, "payments");
        assert!(
            !payment_fks
                .iter()
                .any(|(from, rt, to)| from == "order_id" && rt == "orders" && to == "id"),
            "warn must omit broken payments.order_id FK, got {payment_fks:?}"
        );
        assert_fk(&conn, "orders", "customer_id", "customers", "id");
    }

    #[test]
    fn fk_check_skip_keeps_orphan_fk_in_ddl() {
        let dir = TempDir::new().expect("tempdir");
        let db_path = dir.path().join("skip.sqlite");
        let (_v, mut vm) = data_code::run_with_vm(orphan_fk_model_source()).expect("run");
        let outcome = data_code::sqlite_export::export_to_sqlite(
            &mut vm,
            db_path.to_str().expect("utf8"),
            false,
            FkCheckMode::Skip,
        )
        .expect("skip export should succeed");
        let warning = outcome.warning.expect("skip should set warning");
        assert!(
            warning.contains("payments.order_id"),
            "warning should mention orphan FK, got {warning}"
        );

        let conn = Connection::open(&db_path).expect("open");
        assert_fk(&conn, "payments", "order_id", "orders", "id");
        assert_fk(&conn, "orders", "customer_id", "customers", "id");
        let mut stmt = conn.prepare("PRAGMA foreign_key_check").unwrap();
        let leftover: Vec<String> = stmt
            .query_map([], |r| r.get::<_, String>(0))
            .unwrap()
            .filter_map(|r| r.ok())
            .collect();
        assert!(
            !leftover.is_empty(),
            "skip must leave FK violations in the database"
        );
    }

    #[test]
    fn fk_check_strict_fails_on_orphans() {
        let dir = TempDir::new().expect("tempdir");
        let db_path = dir.path().join("strict.sqlite");
        let (_v, mut vm) = data_code::run_with_vm(orphan_fk_model_source()).expect("run");
        let err = data_code::sqlite_export::export_to_sqlite(
            &mut vm,
            db_path.to_str().expect("utf8"),
            false,
            FkCheckMode::Strict,
        )
        .expect_err("strict export should fail");
        assert!(
            err.contains("foreign_key_check"),
            "expected FK integrity error, got: {err}"
        );
    }

    #[test]
    fn relate_build_model_export_keeps_employees_documents_fk() {
        let dir = TempDir::new().expect("tempdir");
        let db_path = dir.path().join("build_model.sqlite");
        let source = r#"
            global document_field_values = table(
                [["d1", "title", "str", "A"]],
                ["document_id", "field_name", "field_type", "value"]
            )
            global document_table_cells = table(
                [["d1", "grid", "s1", "Sheet", 0, "c0", "Col", "x"]],
                ["document_id", "field_name", "sheet_id", "sheet_name", "row_index", "column_key", "column_label", "value"]
            )
            global document_tables = table(
                [["d1", "grid", "s1", "Sheet", "file", 1, 1]],
                ["document_id", "field_name", "sheet_id", "sheet_name", "source_type", "row_count", "col_count"]
            )
            global documents = table(
                [["d1", "Doc", "N-1", "e1"]],
                ["id", "title", "number", "author_id"]
            )
            global employees = table(
                [["e1", "alice", "Alice"]],
                ["id", "login", "name"]
            )

            primary_key(documents["id"])
            relate(documents["id"], document_tables["document_id"], document_table_cells["document_id"], document_field_values["document_id"])

            primary_key(employees["id"])
            relate(employees["id"], documents["author_id"])
        "#;
        let (_v, mut vm) = data_code::run_with_vm(source).expect("run_with_vm");
        data_code::sqlite_export::export_to_sqlite(
            &mut vm,
            db_path.to_str().expect("utf8 path"),
            false,
            data_code::sqlite_export::FkCheckMode::Strict,
        )
        .expect("export_to_sqlite (build_model path)");
        let conn = Connection::open(&db_path).expect("open sqlite");
        assert_platform_document_fks(&conn);
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
