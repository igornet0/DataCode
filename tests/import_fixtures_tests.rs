// Тесты модулей в tests/import_fixtures: pkg, nested, core.config, core.database.
// Проверяют структуру пакетов, импорты, и совместную работу модулей.

#[cfg(test)]
mod tests {
    use data_code::{run_with_base_path, run_with_vm_with_args_and_lib, Value};
    use std::path::PathBuf;

    fn fixtures_dir() -> PathBuf {
        let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        path.push("tests");
        path.push("import_fixtures");
        path
    }

    fn assert_number(result: Result<Value, data_code::LangError>, expected: f64) {
        match result {
            Ok(Value::Number(n)) => assert!((n - expected).abs() < 1e-10, "expected {}, got {}", expected, n),
            Ok(v) => panic!("expected Number({}), got {:?}", expected, v),
            Err(e) => panic!("error: {:?}", e),
        }
    }

    fn assert_string(result: Result<Value, data_code::LangError>, expected: &str) {
        match result {
            Ok(Value::String(s)) => assert_eq!(s, expected),
            Ok(v) => panic!("expected String('{}'), got {:?}", expected, v),
            Err(e) => panic!("error: {:?}", e),
        }
    }

    fn assert_bool(result: Result<Value, data_code::LangError>, expected: bool) {
        match result {
            Ok(Value::Bool(b)) => assert_eq!(b, expected),
            Ok(v) => panic!("expected Bool({}), got {:?}", expected, v),
            Err(e) => panic!("error: {:?}", e),
        }
    }

    // ========== pkg (простой пакет с __lib__.dc) ==========

    #[test]
    fn test_pkg_get_x() {
        let base = fixtures_dir();
        let result = run_with_base_path(
            r#"from pkg import get_x
get_x()"#,
            base.as_path(),
        );
        assert_number(result, 42.0);
    }

    #[test]
    fn test_pkg_variable_x() {
        let base = fixtures_dir();
        let result = run_with_base_path(
            r#"from pkg import x
x"#,
            base.as_path(),
        );
        assert_number(result, 42.0);
    }

    // ========== nested (__lib__.dc + sub.dc) ==========

    #[test]
    fn test_nested_one_and_two() {
        let base = fixtures_dir();
        let result = run_with_base_path(
            r#"from nested import one, two
one() + two()"#,
            base.as_path(),
        );
        assert_number(result, 3.0);
    }

    #[test]
    fn test_nested_two_only() {
        let base = fixtures_dir();
        let result = run_with_base_path(
            r#"from nested import two
two()"#,
            base.as_path(),
        );
        assert_number(result, 2.0);
    }

    // ========== core.config (base, config, dev_config, prod_config, __lib__) ==========

    #[test]
    fn test_core_config_load_dev() {
        let base = fixtures_dir();
        let result = run_with_base_path(
            r#"
            from core.config import load_settings, get_settings
            load_settings("dev")
            get_settings().env
            "#,
            base.as_path(),
        );
        assert_string(result, "dev");
    }

    #[test]
    fn test_core_config_load_prod() {
        let base = fixtures_dir();
        let result = run_with_vm_with_args_and_lib(
            r#"
            from core.config import load_settings, get_settings
            load_settings("prod")
            get_settings().env
            "#,
            Some(vec!["prod".to_string()]),
            None,
            Some(base.as_path()),
            None,
        );
        let (v, _) = result.expect("run ok");
        assert_string(Ok(v), "prod");
    }

    #[test]
    fn test_core_config_debug_dev() {
        let base = fixtures_dir();
        let result = run_with_base_path(
            r#"
            from core.config import load_settings, get_settings
            load_settings("dev")
            get_settings().debug
            "#,
            base.as_path(),
        );
        assert_bool(result, true);
    }

    #[test]
    fn test_core_config_prod_returns_object() {
        let base = fixtures_dir();
        let result = run_with_vm_with_args_and_lib(
            r#"
            from core.config import load_settings, get_settings
            load_settings("prod")
            let s = get_settings()
            s.env == "prod" and typeof(s) == "object"
            "#,
            Some(vec!["prod".to_string()]),
            None,
            Some(base.as_path()),
            None,
        );
        let (v, _) = result.expect("run ok");
        assert_bool(Ok(v), true);
    }

    #[test]
    fn test_core_config_settings_before_load_is_null() {
        let base = fixtures_dir();
        let result = run_with_base_path(
            r#"
            from core.config import settings
            settings == null
            "#,
            base.as_path(),
        );
        assert_bool(result, true);
    }

    #[test]
    fn test_core_config_get_settings_before_load_throws() {
        let base = fixtures_dir();
        let result = run_with_base_path(
            r#"
            from core.config import get_settings
            get_settings()
            "#,
            base.as_path(),
        );
        assert!(result.is_err());
        let msg = format!("{:?}", result.unwrap_err());
        assert!(msg.contains("not loaded") || msg.contains("Settings") || msg.contains("load_settings"), "msg: {}", msg);
    }

    #[test]
    fn test_core_config_load_invalid_env_throws() {
        let base = fixtures_dir();
        let result = run_with_base_path(
            r#"
            from core.config import load_settings
            load_settings("staging")
            "#,
            base.as_path(),
        );
        assert!(result.is_err());
        let msg = format!("{:?}", result.unwrap_err());
        assert!(msg.contains("Invalid") || msg.contains("staging"), "msg: {}", msg);
    }

    // ========== core.database (импортирует core.config) ==========

    #[test]
    fn test_core_database_get_config_env_dev() {
        let base = fixtures_dir();
        let result = run_with_base_path(
            r#"
            from core.database import get_config_env
            get_config_env("dev")
            "#,
            base.as_path(),
        );
        assert_string(result, "dev");
    }

    #[test]
    fn test_core_database_get_config_env_prod() {
        let base = fixtures_dir();
        let result = run_with_vm_with_args_and_lib(
            r#"
            from core.database import get_config_env
            get_config_env("prod")
            "#,
            Some(vec!["prod".to_string()]),
            None,
            Some(base.as_path()),
            None,
        );
        let (v, _) = result.expect("run ok");
        assert_string(Ok(v), "prod");
    }

    // ========== Комбинированные сценарии ==========

    #[test]
    fn test_import_pkg_and_core_config() {
        let base = fixtures_dir();
        let result = run_with_base_path(
            r#"
            from pkg import get_x
            from core.config import load_settings, get_settings
            load_settings("dev")
            let cfg = get_settings()
            get_x() + len(cfg.env)
            "#,
            base.as_path(),
        );
        
        // 42 + len("dev") = 42 + 3 = 45
        assert_number(result, 45.0);
    }

    #[test]
    fn test_main_with_core_config() {
        let base = fixtures_dir();
        let result = run_with_vm_with_args_and_lib(
            r#"
            from core.config import get_settings, load_settings
            fn main(env) {
                load_settings(env)
                return get_settings().env
            }
            fn __main__(env: "dev" | "prod" = "dev") {
                return main(env)
            }
            "#,
            Some(vec!["prod".to_string()]),
            None,
            Some(base.as_path()),
            None,
        );
        let (v, _) = result.expect("run ok");
        assert_string(Ok(v), "prod");
    }
}
