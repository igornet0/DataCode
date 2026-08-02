//! Imported free-fn / nested-package class constructors with default parameters.

#[cfg(test)]
mod tests {
    use data_code::common::numeric::IntValue;
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
            Ok(Value::Int(IntValue::Finite(n))) => assert_eq!(n as f64, expected),
            Ok(v) => panic!("expected Number({}), got {:?}", expected, v),
            Err(e) => panic!("error: {:?}", e),
        }
    }

    fn assert_str(result: Result<Value, data_code::LangError>, expected: &str) {
        match result {
            Ok(Value::String(s)) => assert_eq!(s, expected),
            Ok(v) => panic!("expected String({:?}), got {:?}", expected, v),
            Err(e) => panic!("error: {:?}", e),
        }
    }

    #[test]
    fn imported_package_ctor_partial_defaults() {
        // Nested re-export duplicates Client::new_3 in host functions; must still resolve.
        let base = fixtures_dir();
        let result = run_with_base_path(
            r#"from defaults_pkg import Client
c = Client("k")
c.timeout"#,
            base.as_path(),
        );
        assert_number(result, 60.0);
    }

    #[test]
    fn imported_package_ctor_explicit_override() {
        let base = fixtures_dir();
        let result = run_with_base_path(
            r#"from defaults_pkg import Client
c = Client("k", null, 5.0)
c.timeout"#,
            base.as_path(),
        );
        assert_number(result, 5.0);
    }

    #[test]
    fn imported_free_fn_defaults() {
        let base = fixtures_dir();
        let result = run_with_base_path(
            r#"from defaults_pkg import make_client
c = make_client()
c.timeout"#,
            base.as_path(),
        );
        assert_number(result, 15.0);
    }

    #[test]
    fn imported_method_defaults() {
        let base = fixtures_dir();
        let result = run_with_base_path(
            r#"from defaults_pkg import Client
c = Client("Ada")
c.greet("Hello, ")"#,
            base.as_path(),
        );
        assert_str(result, "Hello, Ada!");
    }

    #[test]
    fn imported_flat_module_zero_arg_default_still_works() {
        let base = fixtures_dir();
        let result = run_with_base_path(
            r#"from typed_ctor_mod import HashMap
m = HashMap()
m.size_"#,
            base.as_path(),
        );
        assert_number(result, 4.0);
    }
}
