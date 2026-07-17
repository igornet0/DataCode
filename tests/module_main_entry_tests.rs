// __main__ runs only for script entry, not when a .dc file is loaded as an importable module.

#[cfg(test)]
mod tests {
    use data_code::{run_with_base_path, Value};
    use std::fs;
    use std::path::PathBuf;

    fn fixtures_dir() -> PathBuf {
        let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        path.push("tests");
        path.push("import_fixtures");
        path
    }

    fn assert_number(result: Result<Value, data_code::LangError>, expected: f64) {
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

    #[test]
    fn import_does_not_invoke_module_main() {
        let base = fixtures_dir();
        let result = run_with_base_path(
            r#"from mod_with_main import Foo, get_marker
get_marker()"#,
            base.as_path(),
        );
        assert_number(result, 0.0);
    }

    #[test]
    fn explicit_script_run_invokes_main() {
        let base = fixtures_dir();
        let path = base.join("mod_with_main.dc");
        let source = fs::read_to_string(&path).expect("read mod_with_main.dc");
        let result = run_with_base_path(&source, base.as_path());
        assert_number(result, 1.0);
    }

    #[test]
    fn module_main_can_be_called_manually_after_import() {
        let base = fixtures_dir();
        let result = run_with_base_path(
            r#"from mod_with_main import Foo, get_marker, __main__
__main__()
get_marker()"#,
            base.as_path(),
        );
        assert_number(result, 1.0);
    }
}
