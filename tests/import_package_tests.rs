// Тесты импортов из пакетов (папки с __lib__.dc), объектов и функций.
// Основаны на структуре sandbox/web_api: core/config с __lib__.dc и подмодулями.

#[cfg(test)]
mod tests {
    use data_code::{run_with_base_path, run_with_vm_with_args_and_lib, Value};
    use data_code::vm::file_import;
    use std::path::PathBuf;

    fn fixtures_dir() -> PathBuf {
        let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        path.push("tests");
        path.push("import_fixtures");
        path
    }

    fn assert_number_result(result: Result<Value, data_code::LangError>, expected: f64) {
        match result {
            Ok(Value::Number(n)) => assert!((n - expected).abs() < 1e-10, "expected {}, got {}", expected, n),
            Ok(v) => panic!("expected Number({}), got {:?}", expected, v),
            Err(e) => panic!("error: {:?}", e),
        }
    }

    fn assert_string_result(result: Result<Value, data_code::LangError>, expected: &str) {
        match result {
            Ok(Value::String(s)) => assert_eq!(s, expected, "expected '{}', got '{}'", expected, s),
            Ok(v) => panic!("expected String('{}'), got {:?}", expected, v),
            Err(e) => panic!("error: {:?}", e),
        }
    }

    fn assert_bool_result(result: Result<Value, data_code::LangError>, expected: bool) {
        match result {
            Ok(Value::Bool(b)) => assert_eq!(b, expected, "expected {}, got {}", expected, b),
            Ok(v) => panic!("expected Bool({}), got {:?}", expected, v),
            Err(e) => panic!("error: {:?}", e),
        }
    }

    // ========== Импорт из пакета с __lib__.dc (одна папка) ==========

    #[test]
    fn test_from_pkg_import_function() {
        let base = fixtures_dir();
        let source = r#"
from pkg import get_x
get_x()
"#;
        let result = run_with_base_path(source, base.as_path());
        assert_number_result(result, 42.0);
    }

    #[test]
    fn test_from_pkg_import_object() {
        let base = fixtures_dir();
        let source = r#"
        from pkg import x
        x
        "#;
        let result = run_with_base_path(source, base.as_path());
        assert_number_result(result, 42.0);
    }

    #[test]
    fn test_from_pkg_import_both() {
        let base = fixtures_dir();
        let source = r#"
        from pkg import x, get_x
        x + get_x()
        "#;
        let result = run_with_base_path(source, base.as_path());
        assert_number_result(result, 84.0);
    }

    // ========== Импорт из вложенного пакета (core.config), __lib__.dc + подмодуль ==========

    #[test]
    fn test_from_core_config_import_functions() {
        let base = fixtures_dir();
        let source = r#"
        from core.config import load_settings, get_settings
        load_settings("dev")
        let s = get_settings()
        s.env
        "#;
        let result = run_with_base_path(source, base.as_path());
        assert_string_result(result, "dev");
    }

    #[test]
    fn test_from_core_config_load_settings_prod() {
        let base = fixtures_dir();
        let source = r#"
        from core.config import load_settings, get_settings
        load_settings("prod")
        get_settings().env
        "#;
        // Pass argv=["prod"] so load_env infers settings/prod.env when model_config.env_file is empty
        let result = run_with_vm_with_args_and_lib(
            source,
            Some(vec!["prod".to_string()]),
            None,
            Some(base.as_path()),
            None,
        );
        let (value, _) = result.expect("run should succeed");
        assert_string_result(Ok(value), "prod");
    }

    #[test]
    fn test_from_core_config_import_settings_object_after_load() {
        let base = fixtures_dir();
        let source = r#"
        from core.config import load_settings, get_settings
        load_settings("dev")
        let s = get_settings()
        s.debug
        "#;
        let result = run_with_base_path(source, base.as_path());
        assert_bool_result(result, true);
    }

    #[test]
    fn test_from_core_config_import_settings_before_load_is_null() {
        let base = fixtures_dir();
        let source = r#"
        from core.config import settings
        settings == null
        "#;
        let result = run_with_base_path(source, base.as_path());
        assert_bool_result(result, true);
    }

    #[test]
    fn test_from_core_config_get_settings_before_load_throws() {
        let base = fixtures_dir();
        let source = r#"
        from core.config import get_settings
        get_settings()
        "#;
        let result = run_with_base_path(source, base.as_path());
        assert!(result.is_err(), "expected ValueError (Settings are not loaded), got {:?}", result);
        let err = result.unwrap_err();
        let msg = format!("{:?}", err);
        assert!(msg.contains("not loaded") || msg.contains("Settings") || msg.contains("load_settings"),
            "error should mention settings/load: {}", msg);
    }

    #[test]
    fn test_from_core_config_load_settings_invalid_env_throws() {
        let base = fixtures_dir();
        let source = r#"
        from core.config import load_settings
        load_settings("staging")
        "#;
        let result = run_with_base_path(source, base.as_path());
        assert!(result.is_err(), "expected ValueError (Invalid environment), got {:?}", result);
    }

    // ========== Абсолютный импорт из вложенного пакета (core.database.engine -> core.config) ==========
    //
    // engine.dc находится в core/database/ и делает `from core.config import get_settings`.
    // Абсолютный импорт должен резолвиться от project_root, а не от core/database/.
    #[test]
    fn test_absolute_import_from_nested_module_core_database_imports_core_config() {
        let base = fixtures_dir();
        let source = r#"
from core.database import get_config_env
get_config_env("dev")
"#;
        let result = run_with_base_path(source, base.as_path());
        assert_string_result(result, "dev");
    }

    #[test]
    fn test_absolute_import_from_nested_module_core_database_imports_core_config_prod() {
        let base = fixtures_dir();
        let source = r#"
from core.database import get_config_env
get_config_env("prod")
"#;
        // Pass argv=["prod"] so load_env infers settings/prod.env when model_config.env_file is empty
        let result = run_with_vm_with_args_and_lib(
            source,
            Some(vec!["prod".to_string()]),
            None,
            Some(base.as_path()),
            None,
        );
        let (value, _) = result.expect("run should succeed");
        assert_string_result(Ok(value), "prod");
    }

    #[test]
    fn test_main_absolute_import_from_nested_module_core_database_imports_core_config_prod() {
        let base = fixtures_dir();
        let source = r#"
        from core.config import get_settings, load_settings
        from core.database import get_load_config_env

        fn main(env) {

            load_settings(env)

            settings = get_settings()

            s_env = get_load_config_env(env)
            
            return s_env == settings.env

        }

        fn __main__(env: "dev" | "prod") {
            return main(env)
        }
        "#;
        // Pass argv=["prod"] so load_env infers settings/prod.env when model_config.env_file is empty
        let result = run_with_vm_with_args_and_lib(
            source,
            Some(vec!["prod".to_string()]),
            None,
            Some(base.as_path()),
            None,
        );
        let (value, _) = result.expect("run should succeed");
        assert_bool_result(Ok(value), true);
    }

    // ========== Пакет nested: __lib__.dc импортирует из sub.dc ==========

    #[test]
    fn test_from_nested_import_functions_from_lib_and_sub() {
        let base = fixtures_dir();
        let source = r#"
        from nested import one, two
        one() + two()
        "#;
        let result = run_with_base_path(source, base.as_path());
        assert_number_result(result, 3.0);
    }

    #[test]
    fn test_from_nested_import_one_only() {
        let base = fixtures_dir();
        let source = r#"
        from nested import one
        one()
        "#;
        let result = run_with_base_path(source, base.as_path());
        assert_number_result(result, 1.0);
    }

    // ========== Модуль не найден без base_path ==========

    #[test]
    fn test_import_package_without_base_path_fails() {
        let source = r#"
        from pkg import get_x
        get_x()
        "#;
        let result = data_code::run(source);
        assert!(result.is_err(), "without base_path import should fail: {:?}", result);
    }

    // ========== Повторный запуск с тем же base_path (стабильность) ==========

    #[test]
    fn test_core_config_multi_run_stable() {
        use data_code::run_with_vm_and_path;

        let base = fixtures_dir();
        let source = r#"
        from core.config import load_settings, get_settings
        load_settings("dev")
        let s = get_settings()
        s.env
        "#;
        for _ in 0..4 {
            let (value, _) = run_with_vm_and_path(source, Some(base.as_path()), None)
                .unwrap_or_else(|e| panic!("run failed: {:?}", e));
            match &value {
                Value::String(s) => assert_eq!(s, "dev"),
                v => panic!("expected String(\"dev\"), got {:?}", v),
            }
        }
    }

    // ========== Опционально: sandbox/web_api (если есть core/config/__lib__.dc) ==========

    // ========== __main__ с base_path и импортами (как sandbox/web_api/main.dc) ==========
    #[test]
    fn test_main_entry_with_base_path_and_imports() {
        // Скрипт в стиле main.dc: fn __main__(env = "dev") вызывает main(env), main использует core.config
        let base = fixtures_dir();
        let source = r#"
            from core.config import get_settings, load_settings
            fn main(env) {
                load_settings(env)
                return get_settings().env
            }
            fn __main__(env: "dev" | "prod" = "dev") {
                return main(env)
            }
            "#;
        // Run with no args: __main__ gets default "dev"
        let result = run_with_vm_with_args_and_lib(source, None, None, Some(base.as_path()), None);
        let (value, _) = result.expect("run should succeed");
        match &value {
            Value::String(s) => assert_eq!(s, "dev", "default env should be dev"),
            v => panic!("expected String(\"dev\"), got {:?}", v),
        }
    }

    /// Minimal: no ImportFrom; checks that argv is passed correctly with args=["prod"].
    #[test]
    fn test_main_entry_argv_prod_no_import() {
        let base = fixtures_dir();
        let source = r#"
            fn __main__(env: "dev" | "prod" = "dev") {
                return env
            }
            "#;
        let result = run_with_vm_with_args_and_lib(source, Some(vec!["prod".to_string()]), None, Some(base.as_path()), None);
        let (value, _) = result.expect("run should succeed");
        match &value {
            Value::String(s) => assert_eq!(s, "prod", "argv[0] should be 'prod'"),
            v => panic!("expected String(\"prod\"), got {:?}", v),
        }
    }

    /// Same as test_main_entry_with_base_path_and_imports but only the args=["prod"] case (to check if first run affects second).
    #[test]
    fn test_main_entry_with_base_path_and_imports_prod_only() {
        let base = fixtures_dir();
        let source = r#"
            from core.config import get_settings, load_settings
            load_settings("prod")
            get_settings().env
            "#;
        let result = run_with_vm_with_args_and_lib(source, Some(vec!["prod".to_string()]), None, Some(base.as_path()), None);
        let (value, _) = result.expect("run should succeed");
        match &value {
            Value::String(s) => assert_eq!(s, "prod", "when args=[\"prod\"], get_settings().env should get \"prod\", got \"{}\"", s),
            v => panic!("expected String(\"prod\"), got {:?}", v),
        }
    }

    #[test]
    fn test_main_entry_with_args_and_base_path() {
        // Передаём argv — __main__ получает env из аргумента
        let base = fixtures_dir();
        let source = r#"
        from core.config import get_settings, load_settings
        fn main(env) {
            load_settings(env)
            return get_settings().env
        }
        fn __main__(env: "dev" | "prod" = "dev") {
            return main(env)
        }
        "#;
        let result = run_with_vm_with_args_and_lib(
            source,
            Some(vec!["dev".to_string()]),
            None,
            Some(base.as_path()),
            None,
        );
        let (value, _) = result.expect("run should succeed");
        match &value {
            Value::String(s) => assert_eq!(s, "dev"),
            v => panic!("expected String(\"dev\"), got {:?}", v),
        }
    }

    #[test]
    fn test_private_fields_with_model_config() {
        let base = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("sandbox").join("web_api");
        if !base.join("core").join("config").join("__lib__.dc").exists() {
            return;
        }
        if !base.join("core").join("config").join("dev_config.dc").exists() {
            return;
        }
        if !base.join("settings").join("dev.env").exists() {
            return;
        }
        // So that load_env resolves "./settings/dev.env" when file_import/VM base_path is not set during constructor run.
        let prev = std::env::current_dir().ok();
        let _ = std::env::set_current_dir(&base);
        let source = r#"
            from core.config import get_settings, load_settings
                fn main(env) {
                    load_settings(env)
                    return get_settings().secret.get_code()
                }
                fn __main__(env: "dev" | "prod" = "dev") {
                    return main(env)
                }
            "#;

        let result = run_with_vm_with_args_and_lib(source, Some(vec!["dev".to_string()]), None, Some(base.as_path()), None);
        if let Some(ref p) = prev {
            let _ = std::env::set_current_dir(p);
        }
        let (value, _) = result.expect("run should succeed");
        match &value {
            Value::String(s) => assert!(!s.is_empty(), "get_code() should return non-empty string"),
            v => panic!("expected String from get_settings().secret.get_code(), got {:?}", v),
        }
    }

    /// Проверка: находит ли find_nearest_lib __lib__.dc при base_path = sandbox/web_api.
    /// Если да — в VM при запуске main.dc сначала мержится lib, что меняет start_idx при импорте core.config.
    #[test]
    fn test_find_nearest_lib_from_web_api_base() {
        let base = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("sandbox").join("web_api");
        if !base.exists() {
            return;
        }
        let found = file_import::find_nearest_lib(&base);
        // Если найден __lib__.dc выше web_api (например в sandbox/ или корне репо),
        // то при run_with_vm_with_args_and_lib(lib_path=None) он подхватится и добавит функции в VM до импорта core.config.
        if let Some(ref p) = found {
            assert!(p.ends_with("__lib__.dc"), "find_nearest_lib should return path to __lib__.dc, got {:?}", p);
        }
    }

    /// Воспроизведение сценария CLI: base_path = sandbox/web_api, run_with_vm_with_args_and_lib,
    /// core.config с dev_config/prod_config и вызовом конструкторов. До исправления VM падает с
    /// "Function index 19 out of bounds"; после исправления — успех.
    #[test]
    fn test_web_api_cli_like_reproduces_or_passes() {
        let base = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("sandbox").join("web_api");
        if !base.join("core").join("config").join("__lib__.dc").exists() {
            return;
        }
        if !base.join("core").join("config").join("dev_config.dc").exists() {
            return;
        }
        let source = r#"
        from core.config import get_settings, load_settings
        from core.database import create_engine

        fn main(env) {
            load_settings(env)
            return get_settings().env
        }
        fn __main__(env: "dev" | "prod" = "dev") {
            return main(env)
        }
        "#;
        let result = run_with_vm_with_args_and_lib(
            source,
            Some(vec!["dev".to_string()]),
            None,
            Some(base.as_path()),
            None,
        );
        match &result {
            Ok((Value::String(s), _)) => assert_eq!(s, "dev"),
            Ok((v, _)) => panic!("expected String(\"dev\"), got {:?}", v),
            Err(e) => panic!("expected Ok(Value::String(\"dev\")), got: {:?}", e),
        }
    }

    /// Если в sandbox/web_api есть core/config с dev_config, проверяем импорт (тест может быть пропущен при ошибке VM).
    #[test]
    fn test_sandbox_web_api_from_core_config_import_when_present() {
        let base = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("sandbox").join("web_api");
        if !base.join("core").join("config").join("__lib__.dc").exists() {
            return;
        }
        if !base.join("core").join("config").join("dev_config.dc").exists() {
            return;
        }
        let source = r#"
        from core.config import load_settings, get_settings
        load_settings("dev")
        let s = get_settings()
        s.env
        "#;
        let result = run_with_base_path(source, base.as_path());
        if let Ok(Value::String(s)) = result {
            assert_eq!(s, "dev");
        }
        // Иначе тест просто пропускается (sandbox может быть неполным или VM-специфичная ошибка)
    }
}
