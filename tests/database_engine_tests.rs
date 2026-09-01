// Full tests for database_engine module: engine, connect, execute, query,
// MetaData, Column, select, DatabaseCluster, and engine.run (create_all, insert, select)

#[cfg(test)]
mod tests {
    use data_code::{run, Value};

    fn run_plain(source: &str) -> Result<Value, data_code::LangError> {
        run(source)
    }

    fn assert_number_result(source: &str, expected: f64) {
        let result = run_plain(source);
        match result {
            Ok(Value::Number(n)) => assert!(
                (n - expected).abs() < 1e-10,
                "expected {}, got {}",
                expected,
                n
            ),
            Ok(v) => panic!("expected Number({}), got {:?}", expected, v),
            Err(e) => panic!("error: {:?}", e),
        }
    }

    fn assert_bool_result(source: &str, expected: bool) {
        let result = run_plain(source);
        match result {
            Ok(Value::Bool(b)) => assert_eq!(b, expected, "expected {}, got {}", expected, b),
            Ok(v) => panic!("expected Bool({}), got {:?}", expected, v),
            Err(e) => panic!("error: {:?}", e),
        }
    }

    fn assert_string_result(source: &str, expected: &str) {
        let result = run_plain(source);
        match result {
            Ok(Value::String(s)) => assert_eq!(s, expected, "expected '{}', got '{}'", expected, s),
            Ok(v) => panic!("expected String('{}'), got {:?}", expected, v),
            Err(e) => panic!("error: {:?}", e),
        }
    }

    /// Path to a temp SQLite file for tests that need persistence (unique per test).
    fn temp_db_path() -> String {
        let mut path = std::env::temp_dir();
        path.push(format!("datacode_test_{}.db", std::process::id()));
        path.to_string_lossy()
            .replace('\\', "\\\\")
            .replace('"', "\\\"")
    }

    // ========== Import ==========

    #[test]
    fn test_import_engine() {
        let source = r#"
            from database_engine import engine
            let e = engine("sqlite:///:memory:")
            typeof(e) == "database_engine"
        "#;
        assert_bool_result(source, true);
    }

    #[test]
    fn test_import_engine_database_cluster_meta_column_select() {
        let source = r#"
            from database_engine import engine, DatabaseCluster, MetaData, Column, select
            1
        "#;
        match run_plain(source) {
            Ok(Value::Number(n)) => assert!((n - 1.0).abs() < 1e-10),
            Ok(v) => panic!("expected Number(1), got {:?}", v),
            Err(e) => panic!("error: {:?}", e),
        }
    }

    // ========== engine(url) ==========

    #[test]
    fn test_engine_in_memory_creates_engine() {
        let source = r#"
            from database_engine import engine
            let e = engine("sqlite:///:memory:")
            e != null and typeof(e) == "database_engine"
        "#;
        assert_bool_result(source, true);
    }

    #[test]
    fn test_engine_file_path_creates_engine() {
        let path = temp_db_path();
        let source = format!(
            r#"
            from database_engine import engine
            let e = engine("sqlite:///{}")
            typeof(e) == "database_engine"
            "#,
            path
        );
        assert_bool_result(&source, true);
    }

    #[test]
    fn test_engine_invalid_url_raises_error() {
        let source = r#"
            from database_engine import engine
            engine("unknowndb://localhost/db")
        "#;
        assert!(
            run_plain(source).is_err(),
            "expected error for unsupported URL scheme"
        );
    }

    #[test]
    fn test_engine_missing_url_raises_error() {
        let source = r#"
            from database_engine import engine
            engine()
        "#;
        assert!(run_plain(source).is_err(), "expected error for missing url");
    }

    // ========== engine.connect() ==========

    #[test]
    fn test_connect_returns_engine() {
        let source = r#"
            from database_engine import engine
            let e = engine("sqlite:///:memory:")
            let c = e.connect()
            c != null and typeof(c) == "database_engine"
        "#;
        assert_bool_result(source, true);
    }

    #[test]
    fn test_connect_same_as_engine_for_sqlite() {
        let source = r#"
            from database_engine import engine
            let e = engine("sqlite:///:memory:")
            let c = e.connect()
            c == e
        "#;
        assert_bool_result(source, true);
    }

    // ========== engine.execute() ==========

    #[test]
    fn test_execute_create_table_returns_row_count() {
        let source = r#"
            from database_engine import engine
            let e = engine("sqlite:///:memory:")
            e.execute("CREATE TABLE t (id INT, name TEXT)", [])
        "#;
        assert_number_result(source, 0.0);
    }

    #[test]
    fn test_execute_insert_returns_row_count() {
        let source = r#"
            from database_engine import engine
            let e = engine("sqlite:///:memory:")
            e.execute("CREATE TABLE t (id INT, name TEXT)", [])
            e.execute("INSERT INTO t (id, name) VALUES (1, 'a')", [])
        "#;
        assert_number_result(source, 1.0);
    }

    #[test]
    fn test_execute_insert_with_params() {
        let source = r#"
            from database_engine import engine
            let e = engine("sqlite:///:memory:")
            e.execute("CREATE TABLE t (id INT, name TEXT)", [])
            e.execute("INSERT INTO t (id, name) VALUES (?, ?)", [2, "b"])
        "#;
        assert_number_result(source, 1.0);
    }

    #[test]
    fn test_execute_multiple_inserts() {
        let source = r#"
            from database_engine import engine
            let e = engine("sqlite:///:memory:")
            e.execute("CREATE TABLE t (id INT)", [])
            e.execute("INSERT INTO t (id) VALUES (1)", [])
            e.execute("INSERT INTO t (id) VALUES (2)", [])
            e.execute("INSERT INTO t (id) VALUES (3)", [])
        "#;
        assert_number_result(source, 1.0);
    }

    // ========== engine.query() ==========

    #[test]
    fn test_query_select_returns_table() {
        let source = r#"
            from database_engine import engine
            let e = engine("sqlite:///:memory:")
            e.execute("CREATE TABLE t (id INT, name TEXT)", [])
            e.execute("INSERT INTO t (id, name) VALUES (1, 'a')", [])
            let tbl = e.query("SELECT * FROM t", [])
            typeof(tbl) == "table"
        "#;
        assert_bool_result(source, true);
    }

    #[test]
    fn test_query_returns_correct_row_count() {
        let source = r#"
            from database_engine import engine
            let e = engine("sqlite:///:memory:")
            e.execute("CREATE TABLE t (id INT)", [])
            e.execute("INSERT INTO t (id) VALUES (1)", [])
            e.execute("INSERT INTO t (id) VALUES (2)", [])
            let tbl = e.query("SELECT * FROM t", [])
            len(tbl.rows)
        "#;
        assert_number_result(source, 2.0);
    }

    #[test]
    fn test_query_returns_correct_columns() {
        let source = r#"
            from database_engine import engine
            let e = engine("sqlite:///:memory:")
            e.execute("CREATE TABLE t (id INT, name TEXT)", [])
            e.execute("INSERT INTO t (id, name) VALUES (1, 'alice')", [])
            let tbl = e.query("SELECT id, name FROM t", [])
            tbl.columns
        "#;
        let result = run_plain(source);
        match result {
            Ok(Value::Array(rc)) => {
                let arr = rc.borrow();
                assert_eq!(arr.len(), 2, "expected 2 columns");
                match (&arr[0], &arr[1]) {
                    (Value::String(a), Value::String(b)) => {
                        assert_eq!(a, "id");
                        assert_eq!(b, "name");
                    }
                    _ => panic!("columns should be strings"),
                }
            }
            Ok(v) => panic!("expected Array (columns), got {:?}", v),
            Err(e) => panic!("error: {:?}", e),
        }
    }

    #[test]
    fn test_query_empty_result() {
        let source = r#"
            from database_engine import engine
            let e = engine("sqlite:///:memory:")
            e.execute("CREATE TABLE t (id INT)", [])
            let tbl = e.query("SELECT * FROM t", [])
            len(tbl.rows)
        "#;
        assert_number_result(source, 0.0);
    }

    #[test]
    fn test_query_with_params() {
        let source = r#"
            from database_engine import engine
            let e = engine("sqlite:///:memory:")
            e.execute("CREATE TABLE t (id INT, name TEXT)", [])
            e.execute("INSERT INTO t (id, name) VALUES (1, 'a')", [])
            e.execute("INSERT INTO t (id, name) VALUES (2, 'b')", [])
            let tbl = e.query("SELECT * FROM t WHERE id = ?", [2])
            len(tbl.rows) == 1 and tbl.rows[0][1] == "b"
        "#;
        assert_bool_result(source, true);
    }

    // ========== MetaData ==========

    #[test]
    fn test_metadata_creates_object() {
        let source = r#"
            from database_engine import MetaData
            let m = MetaData()
            typeof(m) == "object"
        "#;
        assert_bool_result(source, true);
    }

    #[test]
    fn test_metadata_default_schema() {
        let source = r#"
            from database_engine import MetaData
            let m = MetaData()
            m["schema"]
        "#;
        assert_string_result(source, "public");
    }

    #[test]
    fn test_metadata_custom_schema() {
        let source = r#"
            from database_engine import MetaData
            let m = MetaData("myschema")
            m["schema"]
        "#;
        assert_string_result(source, "myschema");
    }

    #[test]
    fn test_metadata_has_create_all() {
        let source = r#"
            from database_engine import MetaData
            let m = MetaData()
            typeof(m["create_all"]) == "object"
        "#;
        assert_bool_result(source, true);
    }

    // ========== Column ==========

    #[test]
    fn test_column_creates_descriptor() {
        let source = r#"
            from database_engine import Column
            let c = Column()
            typeof(c) == "object"
        "#;
        assert_bool_result(source, true);
    }

    #[test]
    fn test_column_primary_key() {
        let source = r#"
            from database_engine import Column
            let c = Column(primary_key=true)
            c["primary_key"]
        "#;
        assert_bool_result(source, true);
    }

    #[test]
    fn test_column_autoincrement() {
        let source = r#"
            from database_engine import Column
            let c = Column(autoincrement=true)
            c["autoincrement"]
        "#;
        assert_bool_result(source, true);
    }

    #[test]
    fn test_column_unique() {
        let source = r#"
            from database_engine import Column
            let c = Column(unique=true)
            c["unique"]
        "#;
        assert_bool_result(source, true);
    }

    #[test]
    fn test_column_nullable() {
        let source = r#"
            from database_engine import Column
            let c = Column(nullable=true)
            c["nullable"]
        "#;
        assert_bool_result(source, true);
    }

    #[test]
    fn test_column_default() {
        let source = r#"
            from database_engine import Column
            let c = Column(default="x")
            c["default"]
        "#;
        assert_string_result(source, "x");
    }

    // ========== select ==========

    #[test]
    fn test_select_exists() {
        // select is imported and callable; full ORM flow tested via run(select(Model))
        let source = r#"
            from database_engine import select
            typeof(select) == "function"
        "#;
        assert_bool_result(source, true);
    }

    #[test]
    fn test_select_requires_argument() {
        let source = r#"
            from database_engine import select
            select()
        "#;
        assert!(
            run_plain(source).is_err(),
            "expected error for select() without arg"
        );
    }

    // ========== now_call ==========

    #[test]
    fn test_now_call_returns_string() {
        let source = r#"
            from database_engine import now_call
            let t = now_call()
            typeof(t) == "string" and len(t) > 0
        "#;
        assert_bool_result(source, true);
    }

    #[test]
    fn test_now_call_iso_format() {
        let source = r#"
            from database_engine import now_call
            let t = now_call()
            contains(t, "T") and contains(t, "Z")
        "#;
        assert_bool_result(source, true);
    }

    // ========== DatabaseCluster ==========

    #[test]
    fn test_cluster_creates_empty_cluster() {
        let source = r#"
            from database_engine import DatabaseCluster
            let c = DatabaseCluster()
            typeof(c) == "database_cluster"
        "#;
        assert_bool_result(source, true);
    }

    #[test]
    fn test_cluster_add_named_engine() {
        let source = r#"
            from database_engine import engine, DatabaseCluster
            let cluster = DatabaseCluster()
            cluster.add("main", engine("sqlite:///:memory:"))
            cluster.names()
        "#;
        let result = run_plain(source);
        match result {
            Ok(Value::Array(rc)) => {
                let arr = rc.borrow();
                assert_eq!(arr.len(), 1);
                assert_eq!(arr[0], Value::String("main".to_string()));
            }
            Ok(v) => panic!("expected Array, got {:?}", v),
            Err(e) => panic!("error: {:?}", e),
        }
    }

    #[test]
    fn test_cluster_add_multiple() {
        let source = r#"
            from database_engine import engine, DatabaseCluster
            let cluster = DatabaseCluster()
            cluster.add("primary", engine("sqlite:///:memory:"))
            cluster.add("secondary", engine("sqlite:///:memory:"))
            len(cluster.names())
        "#;
        assert_number_result(source, 2.0);
    }

    #[test]
    fn test_cluster_get_returns_engine() {
        let source = r#"
            from database_engine import engine, DatabaseCluster
            let cluster = DatabaseCluster()
            cluster.add("main", engine("sqlite:///:memory:"))
            let e = cluster.get("main")
            e != null and typeof(e) == "database_engine"
        "#;
        assert_bool_result(source, true);
    }

    #[test]
    fn test_cluster_get_nonexistent_returns_null() {
        let source = r#"
            from database_engine import engine, DatabaseCluster
            let cluster = DatabaseCluster()
            cluster.add("main", engine("sqlite:///:memory:"))
            cluster.get("nonexistent") == null
        "#;
        assert_bool_result(source, true);
    }

    #[test]
    fn test_cluster_names_returns_array() {
        let source = r#"
            from database_engine import engine, DatabaseCluster
            let cluster = DatabaseCluster()
            cluster.add("a", engine("sqlite:///:memory:"))
            cluster.add("b", engine("sqlite:///:memory:"))
            let names = cluster.names()
            typeof(names) == "array" and len(names) == 2
        "#;
        assert_bool_result(source, true);
    }

    #[test]
    fn test_cluster_use_engine_via_get() {
        let source = r#"
            from database_engine import engine, DatabaseCluster
            let cluster = DatabaseCluster()
            cluster.add("main", engine("sqlite:///:memory:"))
            let conn = cluster.get("main")
            conn.execute("CREATE TABLE t (id INT)", [])
            conn.execute("INSERT INTO t (id) VALUES (1)", [])
            let tbl = conn.query("SELECT * FROM t", [])
            len(tbl.rows)
        "#;
        assert_number_result(source, 1.0);
    }

    #[test]
    fn test_cluster_add_engine_only_uses_url_as_name() {
        // cluster.add(name, engine) - add with explicit name
        let source = r#"
            from database_engine import engine, DatabaseCluster
            let cluster = DatabaseCluster()
            let eng = engine("sqlite:///:memory:")
            cluster.add("default", eng)
            len(cluster.names()) == 1 and cluster.names()[0] == "default"
        "#;
        assert_bool_result(source, true);
    }

    // ========== engine.run: create_all (metadata.create_all) ==========

    /// Ensure User.metadata.create_all yields an object with __create_all (used by engine.run).
    #[test]
    fn test_create_all_object_has_marker() {
        let source = r#"
            from database_engine import MetaData, Column, int, str
            cls User(Table) {
                metadata = MetaData()
                public:
                    id: int = Column(primary_key=true, autoincrement=true)
                    name: str = Column()
            }
            let create_all = User.metadata.create_all
            create_all != null and create_all.__create_all == true
        "#;
        assert_bool_result(source, true);
    }

    /// Ensure class is registered in metadata.tables when setting metadata = MetaData() in class body.
    #[test]
    fn test_metadata_tables_registers_class() {
        let source = r#"
            from database_engine import MetaData, Column, int, str
            cls User(Table) {
                metadata = MetaData()
                public:
                    id: int = Column(primary_key=true, autoincrement=true)
                    name: str = Column()
            }
            len(User.metadata.tables)
        "#;
        assert_number_result(source, 1.0);
    }

    #[test]
    fn test_run_create_all_creates_tables() {
        let source = r#"
            from database_engine import engine, MetaData, Column, int, str

            cls User(Table) {
                metadata = MetaData()
                public:
                    id: int = Column(primary_key=true, autoincrement=true)
                    name: str = Column()
                fn __tablename__(@class) -> str { return "users" }
            }

            let e = engine("sqlite:///:memory:")
            e.run(User.metadata.create_all)
            e.execute("INSERT INTO users (name) VALUES (?)", ["alice"])
            let tbl = e.query("SELECT * FROM users", [])
            len(tbl.rows)
        "#;
        assert_number_result(source, 1.0);
    }

    /// Abstract classes must not get a table created by create_all; only concrete subclasses do.
    #[test]
    fn test_run_create_all_skips_abstract_classes() {
        let source = r#"
            from database_engine import engine, MetaData, Column, int, str

            @Abstract
            cls Base(Table) {
                metadata = MetaData()
                fn __tablename__(@class) -> str { return "users" }
            }

            cls User(Base) {
                public:
                    id: int = Column(primary_key=true, autoincrement=true)
                    name: str = Column()
            }

            let e = engine("sqlite:///:memory:")
            e.run(Base.metadata.create_all)
            e.execute("INSERT INTO users (name) VALUES (?)", ["alice"])
            let tbl = e.query("SELECT * FROM users", [])
            len(tbl.rows) == 1
            "#;
        assert_bool_result(source, true);
    }

    #[test]
    fn test_run_insert_model_instance() {
        let source = r#"
            from database_engine import engine, MetaData, Column, int, str

            cls User(Table) {
                metadata = MetaData()
                public:
                    id: int = Column(primary_key=true, autoincrement=true)
                    name: str = Column()
                fn __tablename__(@class) -> str { return "users" }
            }

            let e = engine("sqlite:///:memory:")

            e.run(User.metadata.create_all)
            let u = User(name = "bob")
            e.run(u)
            let tbl = e.query("SELECT * FROM users", [])
            len(tbl.rows) == 1 and tbl.rows[0][1] == "bob"
        "#;
        assert_bool_result(source, true);
    }

    #[test]
    fn test_run_insert_with_column_transform_sha256() {
        let source = r#"
            from database_engine import engine, MetaData, Column, int, str

            cls User(Table) {
                metadata = MetaData()
                public:
                    id: int = Column(primary_key=true, autoincrement=true)
                    pw: str = Column(str, transform=sha256)
                fn __tablename__(@class) -> str { return "users" }
            }

            let e = engine("sqlite:///:memory:")
            e.run(User.metadata.create_all)
            let u = User(pw = "secret")
            e.run(u)
            let tbl = e.query("SELECT pw FROM users", [])
            len(tbl.rows) == 1 and len(tbl.rows[0][0]) == 32
        "#;
        assert_bool_result(source, true);
    }

    #[test]
    fn test_column_transform_skips_string_that_looks_like_bcrypt() {
        let source = r#"
            from database_engine import engine, MetaData, Column, int, str

            cls User(Table) {
                metadata = MetaData()
                public:
                    id: int = Column(primary_key=true, autoincrement=true)
                    pw: str = Column(str, transform=sha256)
                fn __tablename__(@class) -> str { return "users" }
            }

            let e = engine("sqlite:///:memory:")
            e.run(User.metadata.create_all)
            let existing = "$2b$12$xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
            let u = User(pw = existing)
            e.run(u)
            let tbl = e.query("SELECT pw FROM users", [])
            tbl.rows[0][0] == existing
        "#;
        assert_bool_result(source, true);
    }

    #[test]
    fn test_column_validators_min_length_ok() {
        let source = r#"
            from database_engine import engine, MetaData, Column, int, str
            from database_engine.validators import min_length

            cls User(Table) {
                metadata = MetaData()
                public:
                    id: int = Column(primary_key=true, autoincrement=true)
                    name: str = Column(str, validators=[min_length(1)])
                fn __tablename__(@class) -> str { return "users" }
            }

            let e = engine("sqlite:///:memory:")
            e.run(User.metadata.create_all)
            let u = User(name = "a")
            e.run(u)
            let tbl = e.query("SELECT * FROM users", [])
            len(tbl.rows) == 1
        "#;
        assert_bool_result(source, true);
    }

    #[test]
    fn test_column_validators_min_length_fails() {
        let source = r#"
            from database_engine import engine, MetaData, Column, int, str
            from database_engine.validators import min_length

            cls User(Table) {
                metadata = MetaData()
                public:
                    id: int = Column(primary_key=true, autoincrement=true)
                    name: str = Column(str, validators=[min_length(5)])
                fn __tablename__(@class) -> str { return "users" }
            }

            let e = engine("sqlite:///:memory:")
            e.run(User.metadata.create_all)
            let u = User(name = "ab")
            e.run(u)
        "#;
        assert!(
            run_plain(source).is_err(),
            "expected validation error on insert"
        );
    }

    #[test]
    fn test_column_validators_custom_fn() {
        let source = r#"
            from database_engine import engine, MetaData, Column, int, str
            from database_engine.validators import custom

            fn not_weak(v) {
                return v != "123456"
            }

            cls User(Table) {
                metadata = MetaData()
                public:
                    id: int = Column(primary_key=true, autoincrement=true)
                    pw: str = Column(str, validators=[custom(not_weak)])
                fn __tablename__(@class) -> str { return "users" }
            }

            let e = engine("sqlite:///:memory:")
            e.run(User.metadata.create_all)
            e.run(User(pw = "secretok"))
            let tbl = e.query("SELECT * FROM users", [])
            len(tbl.rows) == 1
        "#;
        assert_bool_result(source, true);
    }

    #[test]
    fn test_column_validators_one_of() {
        let source = r#"
            from database_engine import engine, MetaData, Column, int, str
            from database_engine.validators import one_of

            cls User(Table) {
                metadata = MetaData()
                public:
                    id: int = Column(primary_key=true, autoincrement=true)
                    role: str = Column(str, validators=[one_of(["user", "admin"])])
                fn __tablename__(@class) -> str { return "users" }
            }

            let e = engine("sqlite:///:memory:")
            e.run(User.metadata.create_all)
            e.run(User(role = "admin"))
            let tbl = e.query("SELECT * FROM users", [])
            len(tbl.rows) == 1
        "#;
        assert_bool_result(source, true);
    }

    #[test]
    fn test_column_validators_min_value() {
        let source = r#"
            from database_engine import engine, MetaData, Column, int, str
            from database_engine.validators import min_value

            cls Score(Table) {
                metadata = MetaData()
                public:
                    id: int = Column(primary_key=true, autoincrement=true)
                    points: int = Column(int, validators=[min_value(0)])
                fn __tablename__(@class) -> str { return "scores" }
            }

            let e = engine("sqlite:///:memory:")
            e.run(Score.metadata.create_all)
            e.run(Score(points = 10))
            let tbl = e.query("SELECT * FROM scores", [])
            len(tbl.rows) == 1
        "#;
        assert_bool_result(source, true);
    }

    #[test]
    fn test_run_select_model() {
        let source = r#"
            from database_engine import engine, MetaData, Column, select, int, str

            cls User(Table) {
                metadata = MetaData()
                public:
                    id: int = Column(primary_key=true, autoincrement=true)
                    name: str = Column()
                fn __tablename__(@class) -> str { return "users" }
            }

            let e = engine("sqlite:///:memory:")

            e.run(User.metadata.create_all)
            e.execute("INSERT INTO users (name) VALUES (?)", ["charlie"])
            let tbl = e.run(select(User))
            tbl.__result and len(tbl.rows) >= 1 and tbl.row_count >= 1
        "#;
        assert_bool_result(source, true);
    }

    // ========== Error cases ==========

    #[test]
    fn test_execute_invalid_sql_raises_error() {
        let source = r#"
            from database_engine import engine
            let e = engine("sqlite:///:memory:")
            e.execute("INVALID SQL XYZ", [])
        "#;
        assert!(run_plain(source).is_err(), "expected error for invalid SQL");
    }

    #[test]
    fn test_query_invalid_sql_raises_error() {
        let source = r#"
            from database_engine import engine
            let e = engine("sqlite:///:memory:")
            e.query("NOT A SELECT", [])
        "#;
        assert!(
            run_plain(source).is_err(),
            "expected error for invalid query"
        );
    }

    #[test]
    fn test_cluster_add_requires_engine() {
        let source = r#"
            from database_engine import DatabaseCluster
            let cluster = DatabaseCluster()
            cluster.add("x", 123)
        "#;
        assert!(
            run_plain(source).is_err(),
            "expected error when add receives non-engine"
        );
    }

    #[test]
    fn test_cluster_get_requires_string_name() {
        let source = r#"
            from database_engine import engine, DatabaseCluster
            let cluster = DatabaseCluster()
            cluster.add("main", engine("sqlite:///:memory:"))
            cluster.get(42)
        "#;
        assert!(
            run_plain(source).is_err(),
            "expected error when get receives non-string"
        );
    }

    #[test]
    fn test_user_model_full_flow_multiple_inserts_and_select() {
        let source = r#"
            from database_engine import engine, MetaData, Column, select, int, str

            cls User(Table) {
                metadata = MetaData()
                public:
                    id: int = Column(primary_key=true, autoincrement=true)
                    name: str = Column()
            }

            let e = engine("sqlite:///:memory:")

            # 1. create tables
            e.run(User.metadata.create_all)

            # 2. insert multiple users (semicolon after User() so parser does not take next id as argument)
            let u1 = User(name = "alice")
            e.run(u1)
            let u2 = User(name = "bob")
            e.run(u2)
            let u3 = User(name = "charlie")
            e.run(u3)
            # 3. select via ORM
            let tbl = e.run(select(User))
            # 4. verify (Result has .rows column 1 is name)
            len((tbl.rows)) == 3 and (tbl.rows[0])[1] == "alice" and (tbl.rows[1])[1] == "bob" and (tbl.rows[2])[1] == "charlie"
            "#;

        assert_bool_result(source, true);
    }

    #[test]
    fn test_unique_constraint_on_login() {
        let source = r#"
            from database_engine import engine, MetaData, Column, int, str

            @Abstract
            cls Base(Table) {
                metadata = MetaData()
            }

            cls User(Base) {
                public:
                    id: Column[int] = Column(int, primary_key=true, autoincrement=true)
                    login: Column[str] = Column(str[50], unique=true)
            }

            let e = engine("sqlite:///:memory:")

            e.run(User.metadata.create_all)

            # insert first user
            let u1 = User(login = "alice")
            e.run(u1)

            let u2 = User(login = "alice")
            e.run(u2)
            "#;

        match run_plain(source) {
            Ok(Value::Null) => panic!("expected error for unique constraint on login"),
            Ok(v) => panic!("expected error, got Ok({:?})", v),
            Err(e) => assert!(
                e.to_string().contains("UNIQUE") || e.to_string().to_lowercase().contains("unique"),
                "expected UNIQUE constraint error, got: {}",
                e
            ),
        }
    }

    #[test]
    fn test_table_name_is_snake_case() {
        let source = r#"
            from database_engine import engine, MetaData, Column, now_call

            fn camel_case_to_snake_case(input_str: str) -> str {
                """
                >>> camel_case_to_snake_case("SomeSDK")
                'some_sdk'
                >>> camel_case_to_snake_case("RServoDrive")
                'r_servo_drive'
                >>> camel_case_to_snake_case("SDKDemo")
                'sdk_demo'
                """
                chars = []
                for c_idx, char in enum(input_str) {
                    if c_idx and char.isupper() {
                        nxt_idx = c_idx + 1
                        # idea of the flag is to separate abbreviations
                        # as new words, show them in lower case
                        flag = nxt_idx >= len(input_str) or input_str[nxt_idx].isupper()
                        prev_char = input_str[c_idx - 1]
                        if !(prev_char.isupper() and flag) { 
                            chars.push("_")
                        }
                    }
                    chars.push(char.lower())    
                } 
                return "".join(chars)

            }

                   
            @Abstract
            cls Base(Table) {

                created: Column[date] = Column(date, default=now_call)
                updated: Column[date] = Column(date, default=now_call, onupdate=now_call)

                metadata = MetaData(
                    schema = "public",
                    quote_schema=null,
                    info={
                        "app": "billing",
                        "version": 1
                    }
                )

                fn __tablename__(@class) -> str {
                    return "${camel_case_to_snake_case(@class.name)}s"
                }
            }

            cls User(Base) {
                id: Column[int] = Column(int, primary_key=true, autoincrement=true)
                login: Column[str] = Column(str[50], unique=true)
            }

            cls RServoDrive(Base) {
                id: Column[int] = Column(int, primary_key=true, autoincrement=true)
                login: Column[str] = Column(str[50], unique=true)
            }

            cls SDKDemo(Base) {
                id: Column[int] = Column(int, primary_key=true, autoincrement=true)
                login: Column[str] = Column(str[50], unique=true)
            }

            sqlite_url = "sqlite:///:memory:"
            sqlite_engine = engine(sqlite_url, echo=false, echo_pool=false, pool_size=5, max_overflow=10)

            # Connect (for SQLite, returns same engine)
            conn = sqlite_engine.connect()

            conn.run(Base.metadata.create_all)

            rows = conn.query("SELECT name FROM sqlite_master WHERE type = 'table';")

            table_name = []

            for r in rows.rows {
                if r[0] == "sqlite_sequence" {
                    continue
                }
                table_name.push(r[0])
            }

            print(table_name)
            
            len(table_name) == 3 and "users" in table_name and "sdk_demos" in table_name and "r_servo_drives" in table_name

        "#;

        assert_bool_result(source, true);
    }

    #[test]
    fn test_sqenum_import() {
        let source = r#"
            from database_engine import SQLEnum
            typeof(SQLEnum) == "object"
        "#;
        assert_bool_result(source, true);
    }

    #[test]
    fn test_sqenum_column_insert_select_and_string_equality() {
        let source = r#"
            from database_engine import engine, MetaData, Column, select, SQLEnum

            cls Role(SQLEnum) {
                MEMBER = "member"
                ADMIN = "admin"
            }

            @Abstract
            cls Base(Table) {
                metadata = MetaData()
            }

            cls User(Base) {
                public:
                    id: Column[int] = Column(int, primary_key=true, autoincrement=true)
                    role: Column[Role] = Column(Role, default=Role.MEMBER)
                fn __tablename__(@class) -> str { return "users" }
            }

            let e = engine("sqlite:///:memory:")
            e.run(User.metadata.create_all)

            let u = User(role = Role.ADMIN)
            e.run(u)

            let tbl = e.run(select(User))
            let inst = tbl.data[0]
            inst.role == "admin" and str(inst.role) == "admin"
        "#;
        assert_bool_result(source, true);
    }

    #[test]
    fn test_sqenum_insert_rejects_invalid_scalar() {
        let source = r#"
            from database_engine import engine, MetaData, Column, SQLEnum

            cls Role(SQLEnum) {
                MEMBER = "member"
                ADMIN = "admin"
            }

            @Abstract
            cls Base(Table) {
                metadata = MetaData()
            }

            cls User(Base) {
                public:
                    id: Column[int] = Column(int, primary_key=true, autoincrement=true)
                    role: Column[Role] = Column(Role)
                fn __tablename__(@class) -> str { return "users" }
            }

            let e = engine("sqlite:///:memory:")
            e.run(User.metadata.create_all)
            e.run(User(role = "not_a_role"))
        "#;
        assert!(
            run_plain(source).is_err(),
            "expected error when enum column value is not in the set"
        );
    }

    // ========== Introspection API ==========

    fn introspection_fixture() -> &'static str {
        r#"
            from database_engine import engine
            let db = engine("sqlite:///:memory:")
            let conn = db.connect()
            conn.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)", [])
            conn.execute("CREATE TABLE orders (id INTEGER PRIMARY KEY, user_id INTEGER, amount REAL, FOREIGN KEY (user_id) REFERENCES users(id))", [])
            conn.execute("INSERT INTO users (name) VALUES (?)", ["Alice"])
            conn.execute("INSERT INTO users (name) VALUES (?)", ["Bob"])
            conn.execute("CREATE VIEW user_names AS SELECT name FROM users", [])
            conn.execute("CREATE INDEX idx_users_name ON users(name)", [])
        "#
    }

    #[test]
    fn test_schemas_includes_main() {
        let source = format!(
            r#"
            {}
            let names = []
            for s in conn.schemas() {{
                names.push(s.name)
            }}
            "main" in names and "temp" in names
            "#,
            introspection_fixture()
        );
        assert_bool_result(&source, true);
    }

    #[test]
    fn test_tables_lists_user_tables_and_hides_system() {
        let source = format!(
            r#"
            {}
            let names = []
            for t in conn.tables() {{
                names.push(t.name)
            }}
            "users" in names and "orders" in names and ("sqlite_sequence" in names) == false and ("_datacode_schema" in names) == false
            "#,
            introspection_fixture()
        );
        assert_bool_result(&source, true);
    }

    #[test]
    fn test_tables_have_schema_and_type() {
        let source = format!(
            r#"
            {}
            let t = conn.tables()[0]
            t.schema == "main" and t["type"] == "table"
            "#,
            introspection_fixture()
        );
        assert_bool_result(&source, true);
    }

    #[test]
    fn test_views_lists_user_names() {
        let source = format!(
            r#"
            {}
            let names = []
            for v in conn.views() {{
                names.push(v.name)
            }}
            "user_names" in names
            "#,
            introspection_fixture()
        );
        assert_bool_result(&source, true);
    }

    #[test]
    fn test_columns_users() {
        let source = format!(
            r#"
            {}
            let cols = conn.columns("users")
            let names = []
            for c in cols {{
                names.push(c.name)
            }}
            "id" in names and "name" in names and cols[0].name == "id"
            "#,
            introspection_fixture()
        );
        assert_bool_result(&source, true);
    }

    #[test]
    fn test_primary_key_users() {
        let source = format!(
            r#"
            {}
            let pk = conn.primary_key("users")
            pk != null and pk.primary == true and pk.columns[0] == "id"
            "#,
            introspection_fixture()
        );
        assert_bool_result(&source, true);
    }

    #[test]
    fn test_indexes_include_name_index() {
        let source = format!(
            r#"
            {}
            let names = []
            for i in conn.indexes("users") {{
                names.push(i.name)
            }}
            "idx_users_name" in names
            "#,
            introspection_fixture()
        );
        assert_bool_result(&source, true);
    }

    #[test]
    fn test_foreign_keys_orders() {
        let source = format!(
            r#"
            {}
            let fks = conn.foreign_keys("orders")
            len(fks) >= 1 and fks[0].referenced_table == "users" and fks[0].columns[0] == "user_id"
            "#,
            introspection_fixture()
        );
        assert_bool_result(&source, true);
    }

    #[test]
    fn test_inspect_nested_tree() {
        let source = format!(
            r#"
            {}
            let info = conn.inspect()
            let found_users = false
            for s in info.schemas {{
                if s.name == "main" {{
                    for t in s.tables {{
                        if t.name == "users" {{
                            found_users = len(t.columns) >= 2
                        }}
                    }}
                }}
            }}
            found_users
            "#,
            introspection_fixture()
        );
        assert_bool_result(&source, true);
    }

    #[test]
    fn test_table_loads_rows() {
        let source = format!(
            r#"
            {}
            let users = conn.table("users")
            typeof(users) == "table" and len(users.rows) == 2
            "#,
            introspection_fixture()
        );
        assert_bool_result(&source, true);
    }

    #[test]
    #[ignore = "requires DATACODE_TEST_PG_URL"]
    fn test_postgres_introspection() {
        let url = std::env::var("DATACODE_TEST_PG_URL").expect("DATACODE_TEST_PG_URL");
        let url = url.replace('\\', "\\\\").replace('"', "\\\"");
        let source = format!(
            r#"
            from database_engine import engine
            let conn = engine("{url}")
            len(conn.schemas()) >= 1
            "#
        );
        assert_bool_result(&source, true);
    }

    #[test]
    #[ignore = "requires DATACODE_TEST_MYSQL_URL"]
    fn test_mysql_introspection() {
        let url = std::env::var("DATACODE_TEST_MYSQL_URL").expect("DATACODE_TEST_MYSQL_URL");
        let url = url.replace('\\', "\\\\").replace('"', "\\\"");
        let source = format!(
            r#"
            from database_engine import engine
            let conn = engine("{url}")
            len(conn.schemas()) >= 1
            "#
        );
        assert_bool_result(&source, true);
    }

    #[test]
    #[ignore = "requires DATACODE_TEST_MSSQL_URL"]
    fn test_mssql_introspection() {
        let url = std::env::var("DATACODE_TEST_MSSQL_URL").expect("DATACODE_TEST_MSSQL_URL");
        let url = url.replace('\\', "\\\\").replace('"', "\\\"");
        let source = format!(
            r#"
            from database_engine import engine
            let conn = engine("{url}")
            len(conn.schemas()) >= 1
            "#
        );
        assert_bool_result(&source, true);
    }
}
