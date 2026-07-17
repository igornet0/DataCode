//! Runtime checks for user class type annotations (`fn f(x: MyClass)`).

#[cfg(test)]
mod tests {
    use data_code::common::error::ErrorType;
    use data_code::common::numeric::IntValue;
    use data_code::{run, LangError, Value};

    fn run_ok(source: &str) -> Value {
        run(source).unwrap_or_else(|e| panic!("unexpected error: {:?}", e))
    }

    fn assert_int(source: &str, expected: i64) {
        match run_ok(source) {
            Value::Int(IntValue::Finite(n)) => assert_eq!(n, expected),
            Value::Number(n) if n == expected as f64 => {}
            other => panic!("expected int {}, got {:?}", expected, other),
        }
    }

    fn assert_type_error_contains(source: &str, substring: &str) {
        let err = run(source).unwrap_err();
        let msg = format!("{:?}", err);
        match err {
            LangError::RuntimeError {
                error_type: Some(ErrorType::TypeError),
                ..
            } => {}
            _ => panic!("expected TypeError, got {:?}", err),
        }
        assert!(
            msg.contains(substring),
            "expected {:?} in error: {}",
            substring,
            msg
        );
    }

    #[test]
    fn class_annotation_accepts_instance() {
        let src = r#"
cls A {
    new A() {}
}
fn f(x: A) { return 1 }
f(A())
"#;
        assert_int(src, 1);
    }

    #[test]
    fn class_annotation_rejects_plain_object() {
        let src = r#"
cls A {
    new A() {}
}
fn f(x: A) { return 1 }
f({"k": 1})
"#;
        assert_type_error_contains(src, "expected type");
        assert_type_error_contains(src, "got 'object'");
    }

    #[test]
    fn class_annotation_accepts_subclass_via_superclass() {
        let src = r#"
cls Parent {
    new Parent() {}
}
cls Child(Parent) {
    new Child() { super() }
}
fn f(p: Parent) { return 2 }
f(Child())
"#;
        assert_int(src, 2);
    }

    #[test]
    fn intersection_typed_params_accept_hashset_instances() {
        let src = r#"
cls HashSet {
    new HashSet() {}
    fn size() -> int { return 0 }
    fn to_list() -> list { return [] }
    fn contains(_item) -> bool { return false }
}
fn intersection(a: HashSet, b: HashSet) -> HashSet {
    return HashSet()
}
a = HashSet()
b = HashSet()
intersection(a, b).size()
"#;
        assert_int(src, 0);
    }

    #[test]
    fn typeof_returns_class_name() {
        let src = r#"
cls Widget {
    new Widget() {}
}
typeof(Widget())
"#;
        match run_ok(src) {
            Value::String(s) => assert_eq!(s, "Widget"),
            other => panic!("expected string Widget, got {:?}", other),
        }
    }

    #[test]
    fn isinstance_string_class_name() {
        let src = r#"
cls Box {
    new Box() {}
}
isinstance(Box(), "Box")
"#;
        match run_ok(src) {
            Value::Bool(true) => {}
            other => panic!("expected true, got {:?}", other),
        }
    }
}
