//! Tests for replace / capitalize globals and string methods.

#[cfg(test)]
mod tests {
    use data_code::{run, Value};

    fn run_ok(source: &str) -> Value {
        run(source).unwrap_or_else(|e| panic!("execution failed: {:?}", e))
    }

    fn assert_string(v: &Value, expected: &str) {
        match v {
            Value::String(s) => assert_eq!(s.as_str(), expected),
            other => panic!("expected string '{}', got {:?}", expected, other),
        }
    }

    #[test]
    fn replace_global_all_occurrences() {
        let v = run_ok(r#"replace("a-old-b-old", "old", "new")"#);
        assert_string(&v, "a-new-b-new");
    }

    #[test]
    fn replace_method() {
        let v = run_ok(r#""x-y-x".replace("x", "z")"#);
        assert_string(&v, "z-y-z");
    }

    #[test]
    fn capitalize_global() {
        let v = run_ok(r#"capitalize("hello WORLD")"#);
        assert_string(&v, "Hello world");
    }

    #[test]
    fn capitalize_method() {
        let v = run_ok(r#""tag".capitalize()"#);
        assert_string(&v, "Tag");
    }

    #[test]
    fn capitalize_empty_string() {
        let v = run_ok(r#"capitalize("")"#);
        assert_string(&v, "");
    }
}
