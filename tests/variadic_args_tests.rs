//! *args / **kwargs in function definitions and calls.

#[cfg(test)]
mod tests {
    use data_code::{run, Value};

    fn run_ok(source: &str) -> Value {
        run(source).expect("expected Ok")
    }

    fn run_err(source: &str) -> data_code::LangError {
        run(source).expect_err("expected Err")
    }

    #[test]
    fn variadic_positional_rest() {
        let src = r#"
            fn pick_first(first, *rest) {
                return { "first": first, "rest_len": len(rest) }
            }
            pick_first(1, 2, 3)
        "#;
        let v = run_ok(src);
        let Value::Object(rc) = &v else { panic!("expected object") };
        let m = rc.borrow();
        assert_eq!(
            m.str_key_get("first").and_then(|v| v.as_finite_f64()),
            Some(1.0)
        );
        assert_eq!(
            m.str_key_get("rest_len").and_then(|v| v.as_finite_f64()),
            Some(2.0)
        );
    }

    #[test]
    fn variadic_kwargs() {
        let src = r#"
            fn bag(**kwargs) {
                return kwargs
            }
            bag(a=1, b="x")
        "#;
        let v = run_ok(src);
        let Value::Object(rc) = &v else { panic!("expected object") };
        let m = rc.borrow();
        assert!(m.str_key_contains("a"));
        assert!(m.str_key_contains("b"));
    }

    #[test]
    fn call_star_unpack_array() {
        let src = r#"
            fn sum3(a, b, c) { return a + b + c }
            extra = [2, 3]
            sum3(1, *extra)
        "#;
        let v = run_ok(src);
        assert_eq!(v.as_finite_f64(), Some(6.0));
    }

    #[test]
    fn call_kwargs_unpack_object() {
        let src = r#"
            fn f(a, b) { return a + b }
            opts = { "a": 10, "b": 5 }
            f(**opts)
        "#;
        let v = run_ok(src);
        assert_eq!(v.as_finite_f64(), Some(15.0));
    }

    #[test]
    fn unexpected_keyword_error() {
        let src = r#"
            fn f(a) { return a }
            f(b=1)
        "#;
        let err = run_err(src);
        let msg = err.to_string();
        assert!(msg.contains("unexpected keyword argument"));
    }
}
