//! Class method default parameters and local under-arity Calls.

#[cfg(test)]
mod tests {
    use data_code::common::numeric::IntValue;
    use data_code::{run, Value};

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

    fn assert_str(source: &str, expected: &str) {
        match run_ok(source) {
            Value::String(s) => assert_eq!(s, expected),
            other => panic!("expected string {:?}, got {:?}", expected, other),
        }
    }

    #[test]
    fn method_default_trailing_arg() {
        let src = r#"
cls C {
    new C() {}
    fn add(a, b = 10) {
        return a + b
    }
}
c = C()
c.add(5)
"#;
        assert_int(src, 15);
    }

    #[test]
    fn method_default_multi_trailing() {
        let src = r#"
cls C {
    new C() {}
    fn f(a, b = 2, c = 3) {
        return a + b + c
    }
}
c = C()
c.f(1)
"#;
        assert_int(src, 6);
    }

    #[test]
    fn method_default_override() {
        let src = r#"
cls C {
    new C() {}
    fn add(a, b = 10) {
        return a + b
    }
}
c = C()
c.add(5, 1)
"#;
        assert_int(src, 6);
    }

    #[test]
    fn method_default_string() {
        let src = r#"
cls Greeter {
    new Greeter(name) {
        this.name = name
    }
    public:
        name: str
    fn hello(prefix = "Hi, ") {
        return prefix + this.name
    }
}
g = Greeter("Ada")
g.hello()
"#;
        assert_str(src, "Hi, Ada");
    }

    #[test]
    fn free_fn_local_defaults_still_work() {
        let src = r#"
fn f(a = 1, b = 2) {
    return a + b
}
f()
"#;
        assert_int(src, 3);
    }
}
