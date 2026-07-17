// Imported classes with typed constructors (`Class::new_N_int`) must be callable from other modules.

#[cfg(test)]
mod tests {
    use data_code::{run_with_base_path, Value};
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
    fn imported_typed_ctor_direct_call() {
        let base = fixtures_dir();
        let result = run_with_base_path(
            r#"from typed_ctor_mod import HashMap
m = HashMap(16)
m.size_"#,
            base.as_path(),
        );
        assert_number(result, 16.0);
    }

    #[test]
    fn imported_class_zero_arg_default_ctor() {
        let base = fixtures_dir();
        let result = run_with_base_path(
            r#"from typed_ctor_mod import HashMap
m = HashMap()
m.size_"#,
            base.as_path(),
        );
        assert_number(result, 4.0);
    }

    #[test]
    fn imported_typed_ctor_via_enclosing_param() {
        let base = fixtures_dir();
        let result = run_with_base_path(
            r#"from typed_ctor_mod import Box
b = Box(12)
b.inner.size_"#,
            base.as_path(),
        );
        assert_number(result, 12.0);
    }
}
