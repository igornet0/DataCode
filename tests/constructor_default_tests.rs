//! Constructor calls with default parameter values (`ClassName()` when only `new ClassName(x: T = …)` exists).

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

    fn assert_error_type(source: &str, expected: ErrorType) {
        let err = run(source).unwrap_err();
        match err {
            LangError::RuntimeError {
                error_type: Some(et),
                ..
            } => assert_eq!(et, expected),
            LangError::ParseError { .. } if expected == ErrorType::TypeError => {}
            e => panic!("expected {:?}, got {:?}", expected, e),
        }
    }

    fn assert_compile_error_contains(source: &str, expected_substr: &str) {
        let err = run(source).unwrap_err();
        let msg = format!("{:?}", err);
        assert!(
            msg.contains(expected_substr),
            "expected error containing {:?}, got {:?}",
            expected_substr,
            err
        );
    }

    #[test]
    fn ctor_default_module_const() {
        let src = r#"
RED = 0
cls Foo {
    new Foo(x: int = RED) {
        this.x = x
    }
    public:
        x: int
}
Foo().x
"#;
        assert_int(src, 0);
    }

    #[test]
    fn ctor_default_named_const_like_rb_tree() {
        let src = r#"
RED = 0
BLACK = 1
cls Node {
    new Node(value: int, color: int = RED) {
        this.value = value
        this.color = color
    }
    public:
        value: int
        color: int
}
n = Node(5)
n.color
"#;
        assert_int(src, 0);
    }

    #[test]
    fn ctor_default_expr_from_const() {
        let src = r#"
RED = 0
cls Foo {
    new Foo(x: int = RED + 1) {
        this.x = x
    }
    public:
        x: int
}
Foo().x
"#;
        assert_int(src, 1);
    }

    #[test]
    fn fn_default_frozen_after_reassign() {
        let src = r#"
RED = 0
fn f(x: int = RED) {
    return x
}
RED = 5
f()
"#;
        assert_int(src, 0);
    }

    #[test]
    fn ctor_default_undefined_var_errors() {
        let src = r#"
cls Foo {
    new Foo(x: int = MISSING) {}
}
"#;
        assert_compile_error_contains(src, "compile-time constant");
    }

    #[test]
    fn ctor_default_runtime_var_errors() {
        let src = r#"
fn foo() { return 1 }
RED = foo()
cls Foo {
    new Foo(x: int = RED) {}
}
"#;
        assert_compile_error_contains(src, "compile-time constant");
    }

    #[test]
    fn ctor_default_zero_args() {
        let src = r#"
cls Foo {
    new Foo(x: int = 8) {
        this.x = x
    }
    public:
        x: int
}
f = Foo()
f.x
"#;
        assert_int(src, 8);
    }

    #[test]
    fn ctor_default_explicit_arg() {
        let src = r#"
cls Foo {
    new Foo(x: int = 8) {
        this.x = x
    }
    public:
        x: int
}
Foo(3).x
"#;
        assert_int(src, 3);
    }

    #[test]
    fn ctor_default_named_arg() {
        let src = r#"
cls Foo {
    new Foo(x: int = 8) {
        this.x = x
    }
    public:
        x: int
}
Foo(x=5).x
"#;
        assert_int(src, 5);
    }

    #[test]
    fn ctor_missing_required_param_errors() {
        let src = r#"
cls Bar {
    new Bar(a: int, b: int = 2) {}
}
Bar()
"#;
        assert!(run(src).is_err());
    }

    #[test]
    fn hashset_style_ctor_default() {
        let src = r#"
cls HashMap {
    new HashMap(cap: int) {
        this.cap = cap
    }
    public:
        cap: int
}
cls HashSet {
    new HashSet(initial_capacity: int = 8) {
        this.map = HashMap(initial_capacity)
    }
    public:
        map: HashMap
}
s = HashSet()
s.map.cap
"#;
        assert_int(src, 8);
    }
}
