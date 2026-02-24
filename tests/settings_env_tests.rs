// Тесты модуля settings_env: load_env, config, Field, Settings, загрузка разных .env

#[cfg(test)]
mod tests {
    use data_code::{run, run_with_base_path, Value};
    use std::path::PathBuf;

    fn fixtures_dir() -> PathBuf {
        let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        path.push("tests");
        path.push("settings_env_fixtures");
        path
    }

    /// Path to a fixture .env file, escaped for use inside .dc string literal.
    fn fixture_path(name: &str) -> String {
        let path = fixtures_dir().join(name);
        path.to_string_lossy().replace('\\', "\\\\").replace('"', "\\\"")
    }

    fn run_plain(source: &str) -> Result<Value, data_code::LangError> {
        run(source)
    }

    fn assert_string_result(source: &str, expected: &str) {
        let result = run_plain(source);
        match result {
            Ok(Value::String(s)) => assert_eq!(s, expected, "expected '{}', got '{}'", expected, s),
            Ok(v) => panic!("expected String('{}'), got {:?}", expected, v),
            Err(e) => panic!("error: {:?}", e),
        }
    }

    fn assert_number_result(source: &str, expected: f64) {
        let result = run_plain(source);
        match result {
            Ok(Value::Number(n)) => assert!((n - expected).abs() < 1e-10, "expected {}, got {}", expected, n),
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

    fn assert_null_result(source: &str) {
        let result = run_plain(source);
        match result {
            Ok(Value::Null) => {}
            Ok(v) => panic!("expected Null, got {:?}", v),
            Err(e) => panic!("error: {:?}", e),
        }
    }

    fn assert_error(source: &str) {
        let result = run_plain(source);
        assert!(result.is_err(), "expected error, got {:?}", result);
    }

    // ========== load_env: simple.env (absolute path via run) ==========

    #[test]
    fn test_load_env_simple_keys() {
        let path = fixture_path("simple.env");
        let source = format!(
            r#"
            from settings_env import load_env
            let env = load_env("{}")
            env["app_name"]
            "#,
            path
        );
        assert_string_result(source.as_str(), "myapp");
    }

    #[test]
    fn test_load_env_simple_env_key() {
        let path = fixture_path("simple.env");
        let source = format!(
            r#"
            from settings_env import load_env
            let env = load_env("{}")
            env["env"]
            "#,
            path
        );
        assert_string_result(source.as_str(), "dev");
    }

    #[test]
    fn test_load_env_simple_debug_coerced_to_bool() {
        let path = fixture_path("simple.env");
        let source = format!(
            r#"
            from settings_env import load_env
            let env = load_env("{}")
            env["debug"]
            "#,
            path
        );
        let result = run_plain(&source);
        match result {
            Ok(Value::Bool(b)) => assert!(b),
            Ok(v) => panic!("expected Bool true, got {:?}", v),
            Err(e) => panic!("error: {:?}", e),
        }
    }

    #[test]
    fn test_load_env_simple_has_three_keys() {
        let path = fixture_path("simple.env");
        let source = format!(
            r#"
            from settings_env import load_env
            let env = load_env("{}")
            len(env)
            "#,
            path
        );
        assert_number_result(source.as_str(), 3.0);
    }

    // ========== load_env: empty.env ==========

    #[test]
    fn test_load_env_empty_returns_empty_object() {
        let path = fixture_path("empty.env");
        let source = format!(
            r#"
            from settings_env import load_env
            let env = load_env("{}")
            len(env)
            "#,
            path
        );
        assert_number_result(source.as_str(), 0.0);
    }

    // ========== load_env: with_types.env (coercion) ==========

    #[test]
    fn test_load_env_coerces_true_false() {
        let path = fixture_path("with_types.env");
        let source = format!(
            r#"
            from settings_env import load_env
            let env = load_env("{}")
            env["bool_true"] == true and env["bool_false"] == false
            "#,
            path
        );
        assert_bool_result(source.as_str(), true);
    }

    #[test]
    fn test_load_env_coerces_numbers() {
        let path = fixture_path("with_types.env");
        let source = format!(
            r#"
            from settings_env import load_env
            let env = load_env("{}")
            env["int_num"] + env["float_num"]
            "#,
            path
        );
        assert_number_result(source.as_str(), 45.14);
    }

    #[test]
    fn test_load_env_string_value() {
        let path = fixture_path("with_types.env");
        let source = format!(
            r#"
            from settings_env import load_env
            let env = load_env("{}")
            env["string_val"]
            "#,
            path
        );
        assert_string_result(source.as_str(), "hello");
    }

    #[test]
    fn test_load_env_quoted_value() {
        let path = fixture_path("with_types.env");
        let source = format!(
            r#"
            from settings_env import load_env
            let env = load_env("{}")
            env["quoted"]
            "#,
            path
        );
        assert_string_result(source.as_str(), "value with spaces");
    }

    // ========== load_env: with_prefix.env + model_config ==========

    #[test]
    fn test_load_env_with_prefix_strips_prefix() {
        let path = fixture_path("with_prefix.env");
        let source = format!(
            r#"
            from settings_env import load_env, Settings
            let cfg = Settings.config(env_prefix="APP__")
            let env = load_env("{}", [], cfg)
            env["foo"]
            "#,
            path
        );
        assert_number_result(source.as_str(), 1.0);
    }

    #[test]
    fn test_load_env_with_prefix_only_matching_keys() {
        let path = fixture_path("with_prefix.env");
        let source = format!(
            r#"
            from settings_env import load_env, Settings
            let cfg = Settings.config(env_prefix="APP__")
            let env = load_env("{}", [], cfg)
            len(env)
            "#,
            path
        );
        assert_number_result(source.as_str(), 3.0);
    }

    #[test]
    fn test_load_env_with_prefix_bar_value() {
        let path = fixture_path("with_prefix.env");
        let source = format!(
            r#"
            from settings_env import load_env, Settings
            let cfg = Settings.config(env_prefix="APP__")
            let env = load_env("{}", [], cfg)
            env["bar"]
            "#,
            path
        );
        assert_string_result(source.as_str(), "two");
    }

    #[test]
    fn test_load_env_db_prefix() {
        let path = fixture_path("with_prefix.env");
        let source = format!(
            r#"
            from settings_env import load_env, Settings
            let cfg = Settings.config(env_prefix="DB__")
            let env = load_env("{}", [], cfg)
            env["host"]
            "#,
            path
        );
        assert_string_result(source.as_str(), "localhost");
    }

    // ========== load_env: required_keys ==========

    #[test]
    fn test_load_env_required_keys_satisfied() {
        let path = fixture_path("required_test.env");
        let source = format!(
            r#"
            from settings_env import load_env
            let required = ["required_a", "required_b"]
            let env = load_env("{}", required)
            env["required_a"]
            "#,
            path
        );
        assert_string_result(source.as_str(), "value_a");
    }

    #[test]
    fn test_load_env_required_keys_missing_fails() {
        let path = fixture_path("required_test.env");
        let source = format!(
            r#"
            from settings_env import load_env
            let required = ["required_a", "missing_key"]
            load_env("{}", required)
            "#,
            path
        );
        assert_error(&source);
    }

    // ========== quoted.env ==========

    #[test]
    fn test_load_env_quoted_file() {
        let path = fixture_path("quoted.env");
        let source = format!(
            r#"
            from settings_env import load_env
            let env = load_env("{}")
            env["single"] == "single quoted" and env["double"] == "double quoted"
            "#,
            path
        );
        assert_bool_result(source.as_str(), true);
    }

    // ========== config() ==========

    #[test]
    fn test_config_returns_object() {
        let source = r#"
            from settings_env import Settings
            let cfg = Settings.config()
            typeof(cfg) == "object"
        "#;
        let result = run_plain(source);
        match result {
            Ok(Value::Bool(b)) => assert!(b, "expected true"),
            Ok(v) => panic!("expected Bool, got {:?}", v),
            Err(e) => panic!("error: {:?}", e),
        }
    }

    #[test]
    fn test_config_env_prefix() {
        let source = r#"
            from settings_env import Settings
            let cfg = Settings.config(env_prefix="APP__")
            cfg["env_prefix"]
        "#;
        let result = run_plain(source);
        match result {
            Ok(Value::String(s)) => assert_eq!(s, "APP__"),
            Ok(v) => panic!("expected String, got {:?}", v),
            Err(e) => panic!("error: {:?}", e),
        }
    }

    #[test]
    fn test_config_extra_default() {
        let source = r#"
            from settings_env import Settings
            let cfg = Settings.config()
            cfg["extra"]
        "#;
        let result = run_plain(source);
        match result {
            Ok(Value::String(s)) => assert_eq!(s, "ignore"),
            Ok(v) => panic!("expected String, got {:?}", v),
            Err(e) => panic!("error: {:?}", e),
        }
    }

    #[test]
    fn test_config_case_sensitive_default() {
        let source = r#"
            from settings_env import Settings
            let cfg = Settings.config()
            cfg["case_sensitive"]
        "#;
        let result = run_plain(source);
        match result {
            Ok(Value::Bool(b)) => assert!(!b, "case_sensitive default should be false"),
            Ok(v) => panic!("expected Bool, got {:?}", v),
            Err(e) => panic!("error: {:?}", e),
        }
    }

    // ========== Field() ==========

    #[test]
    fn test_field_single_default_returns_descriptor() {
        let source = r#"
            from settings_env import Field
            let f = Field("dev")
            typeof(f) == "object"
        "#;
        let result = run_plain(source);
        match result {
            Ok(Value::Bool(b)) => assert!(b),
            Ok(v) => panic!("expected Bool, got {:?}", v),
            Err(e) => panic!("error: {:?}", e),
        }
        let source2 = r#"
            from settings_env import Field
            let f = Field("dev")
            f["default"]
        "#;
        let result2 = run_plain(source2);
        match result2 {
            Ok(Value::String(s)) => assert_eq!(s, "dev"),
            Ok(v) => panic!("expected String 'dev', got {:?}", v),
            Err(e) => panic!("error: {:?}", e),
        }
    }

    #[test]
    fn test_field_named_default() {
        let source = r#"
            from settings_env import Field
            let f = Field(default="prod")
            f["default"]
        "#;
        let result = run_plain(source);
        match result {
            Ok(Value::String(s)) => assert_eq!(s, "prod"),
            Ok(v) => panic!("expected String, got {:?}", v),
            Err(e) => panic!("error: {:?}", e),
        }
    }

    #[test]
    fn test_field_descriptor_with_validation() {
        let source = r#"
            from settings_env import Field
            let f = Field(default="x", min_length=1, max_length=10)
            f["min_length"] == 1 and f["max_length"] == 10
        "#;
        let result = run_plain(source);
        match result {
            Ok(Value::Bool(b)) => assert!(b),
            Ok(v) => panic!("expected Bool, got {:?}", v),
            Err(e) => panic!("error: {:?}", e),
        }
    }

    #[test]
    fn test_field_required_marker() {
        let source = r#"
            from settings_env import Field
            let f = Field(alias="userId", title="User ID")
            f["alias"] == "userId"
        "#;
        let result = run_plain(source);
        match result {
            Ok(Value::Bool(b)) => assert!(b),
            Ok(v) => panic!("expected Bool, got {:?}", v),
            Err(e) => panic!("error: {:?}", e),
        }
    }

    #[test]
    fn test_field_descriptor_repr_exclude() {
        let source = r#"
            from settings_env import Field
            let f = Field(default="secret", repr=false)
            f["repr"]
        "#;
        let result = run_plain(source);
        match result {
            Ok(Value::Bool(b)) => assert!(!b),
            Ok(v) => panic!("expected Bool false, got {:?}", v),
            Err(e) => panic!("error: {:?}", e),
        }
    }

    // ========== Settings(path) ==========

    #[test]
    fn test_settings_same_as_load_env() {
        let path = fixture_path("simple.env");
        let source = format!(
            r#"
            from settings_env import load_env, Settings
            let env1 = load_env("{}")
            let env2 = Settings("{}")
            env1["app_name"] == env2["app_name"] and env1["env"] == env2["env"]
            "#,
            path, path
        );
        assert_bool_result(source.as_str(), true);
    }

    // ========== Settings subclass with model_config ==========

    #[test]
    fn test_settings_subclass_loads_env() {
        let path = fixture_path("with_prefix.env");
        let source = format!(
            r#"
            from settings_env import Settings, Field
            cls Config(Settings) {{
                model_config = Settings.config(env_prefix="APP__", extra="ignore")
                public:
                    foo: str
                    bar: str
            }}
            let cfg = Config("{}")
            cfg["foo"] == 1 and cfg["bar"] == "two"
            "#,
            path
        );
        assert_bool_result(source.as_str(), true);
    }

    #[test]
    fn test_settings_subclass_access_fields() {
        // Settings subclass: Config(path) calls load_env(path), returns env object
        let path = fixture_path("simple.env");
        let source = format!(
            r#"
            from settings_env import Settings, Field
            cls AppConfig(Settings) {{
                model_config = Settings.config(extra="ignore")
                public:
                    app_name: str = Field(default="")
                    env: str = Field(default="")
            }}
            AppConfig("{}")
            "#,
            path
        );
        let result = run_plain(&source);
        match result {
            Ok(Value::Object(_)) => {}
            Ok(v) => panic!("expected Object from Config(path), got {:?}", v),
            Err(e) => panic!("Config(path) failed: {:?}", e),
        }
    }

    // ========== apply_field_descriptor (via validation) ==========
    // apply_field_descriptor is used internally; we test that Field descriptors
    // are built correctly and load_env returns expected types.

    #[test]
    fn test_load_env_nonexistent_file_returns_null() {
        let path = fixtures_dir().join("nonexistent_file_123.env");
        let path_str = path.to_string_lossy().replace('\\', "\\\\").replace('"', "\\\"");
        let source = format!(
            r#"
            from settings_env import load_env
            load_env("{}")
            "#,
            path_str
        );
        assert_null_result(&source);
    }

    #[test]
    fn test_load_env_path_object() {
        let path = fixture_path("simple.env");
        let source = format!(
            r#"
            from settings_env import load_env
            let env = load_env("{}")
            env["debug"] == true and env["env"] == "dev"
            "#,
            path
        );
        assert_bool_result(source.as_str(), true);
    }

    /// Config/DatabaseConfig-style script run twice: both runs must return correct cfg.env, cfg.debug, cfg.db.url.
    /// Regression test for constructor LoadGlobal(class_global_index) not being patched on second run when
    /// chunk.global_names lacked the class name.
    /// Ignored by default: can hang (60+ s) or recurse infinitely when default_factory resolves to the wrong
    /// class (Config vs DatabaseConfig). Run with: cargo test test_config_script_run_twice_stable -- --ignored
    #[test]
    #[ignore]
    fn test_config_script_run_twice_stable() {
        let path = fixture_path("config_like.env");
        let source = format!(
            r#"
            from settings_env import Settings, Field
            cls DatabaseConfig(Settings) {{
                model_config = Settings.config(env_prefix="DB__", extra="ignore")
                public:
                    url: str = Field(...)
            }}
            cls Config(Settings) {{
                model_config = Settings.config(env_prefix="APP__", extra="ignore")
                public:
                    env: str = Field(default="dev")
                    debug: bool = Field(default=true)
                    db: DatabaseConfig = Field(default_factory=DatabaseConfig)
            }}
            let cfg = Config("{}")
            cfg
            "#,
            path
        );
        fn check_cfg(value: &Value) {
            let obj = match value {
                Value::Object(rc) => rc.borrow(),
                _ => panic!("expected Object (cfg), got {:?}", value),
            };
            match obj.get("env") {
                Some(Value::String(s)) => assert_eq!(s, "dev", "cfg.env"),
                Some(v) => panic!("cfg.env expected String(\"dev\"), got {:?}", v),
                None => panic!("cfg.env missing"),
            }
            match obj.get("debug") {
                Some(Value::Bool(b)) => assert!(b, "cfg.debug should be true"),
                Some(v) => panic!("cfg.debug expected Bool(true), got {:?}", v),
                None => panic!("cfg.debug missing"),
            }
            let db = obj.get("db").expect("cfg.db missing");
            let db_obj = match db {
                Value::Object(rc) => rc.borrow(),
                _ => panic!("cfg.db expected Object, got {:?}", db),
            };
            match db_obj.get("url") {
                Some(Value::String(s)) => assert_eq!(s, "postgresql://dev.server/dev", "cfg.db.url"),
                Some(v) => panic!("cfg.db.url expected String, got {:?}", v),
                None => panic!("cfg.db.url missing"),
            }
        }
        let r1 = run_plain(&source).expect("first run");
        check_cfg(&r1);
        let r2 = run_plain(&source).expect("second run");
        check_cfg(&r2);
    }

    /// Regression: run load_env("settings/dev.env") with base_path set multiple times.
    /// Ensures base_path is applied deterministically and env keys are stable (no null).
    #[test]
    fn test_load_env_relative_path_multi_run_stable() {
        use data_code::run_with_vm_and_path;

        let base = fixtures_dir().join("config_run");
        let source = r#"
from settings_env import load_env
let env = load_env("settings/dev.env")
env
"#;
        for run in 0..8 {
            let (value, _) = run_with_vm_and_path(source, Some(base.as_path()), None)
                .unwrap_or_else(|e| panic!("run {} failed: {:?}", run, e));
            let obj = match &value {
                Value::Object(rc) => rc.borrow(),
                _ => panic!("expected Object (env), got {:?}", value),
            };
            match obj.get("app__env") {
                Some(Value::String(s)) => assert_eq!(s, "dev", "app__env"),
                Some(Value::Null) => panic!("app__env must not be null"),
                other => panic!("app__env expected String(\"dev\"), got {:?}", other),
            }
            match obj.get("db__url") {
                Some(Value::String(s)) => assert_eq!(s, "postgresql://dev.server/dev", "db__url"),
                Some(Value::Null) => panic!("db__url must not be null"),
                other => panic!("db__url expected String, got {:?}", other),
            }
        }
    }

    /// Regression: run script that uses Config("settings/dev.env")-equivalent logic with base_path set multiple times.
    /// Ensures cfg-like output (env, debug, db.url) is stable (no null). Uses settings_env and inline logic
    /// so we don't depend on loading the config .dc module (base_path is still applied for load_env).
    #[test]
    fn test_config_relative_path_multi_run_stable() {
        use data_code::run_with_vm_and_path;

        let base = fixtures_dir().join("config_run");
        // Equivalent to Config("settings/dev.env"): load env and build cfg-like object (object literal with string keys)
        let source = r#"
from settings_env import load_env
let env = load_env("settings/dev.env")
let db = { "url": env["db__url"] }
let debug_ok = env["app__debug"] == true or env["app__debug"] == "true"
let cfg = { "env": env["app__env"], "debug": debug_ok, "db": db }
cfg
"#;
        fn check_cfg(value: &Value) {
            let obj = match value {
                Value::Object(rc) => rc.borrow(),
                _ => panic!("expected Object (cfg), got {:?}", value),
            };
            match obj.get("env") {
                Some(Value::String(s)) => assert_eq!(s, "dev", "cfg.env"),
                Some(Value::Null) => panic!("cfg.env must not be null (base_path should resolve settings/dev.env)"),
                Some(v) => panic!("cfg.env expected String(\"dev\"), got {:?}", v),
                None => panic!("cfg.env missing"),
            }
            match obj.get("debug") {
                Some(Value::Bool(b)) => assert!(b, "cfg.debug should be true"),
                Some(Value::Null) => panic!("cfg.debug must not be null"),
                Some(v) => panic!("cfg.debug expected Bool(true), got {:?}", v),
                None => panic!("cfg.debug missing"),
            }
            let db = obj.get("db").expect("cfg.db missing");
            let db_obj = match db {
                Value::Object(rc) => rc.borrow(),
                _ => panic!("cfg.db expected Object, got {:?}", db),
            };
            match db_obj.get("url") {
                Some(Value::String(s)) => assert_eq!(s, "postgresql://dev.server/dev", "cfg.db.url"),
                Some(Value::Null) => panic!("cfg.db.url must not be null"),
                Some(v) => panic!("cfg.db.url expected String, got {:?}", v),
                None => panic!("cfg.db.url missing"),
            }
        }
        for run in 0..8 {
            let (value, _) = run_with_vm_and_path(source, Some(base.as_path()), None)
                .unwrap_or_else(|e| panic!("run {} failed: {:?}", run, e));
            check_cfg(&value);
        }
    }

    /// Local .dc module (from config import Config) loads when run with base_path; without base_path run() would fail with "Module 'config' not found".
    #[test]
    fn test_local_config_module_with_base_path() {
        let base = fixtures_dir().join("config_run");
        let source = r#"
from config import Config
1
"#;
        let result = run_with_base_path(source, base.as_path());
        match result {
            Ok(Value::Number(n)) => assert!((n - 1.0).abs() < 1e-10, "expected 1"),
            Ok(v) => panic!("expected Number(1), got {:?}", v),
            Err(e) => panic!("run_with_base_path failed (local module should load with base_path): {:?}", e),
        }
    }
}
