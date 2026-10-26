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

    fn assert_array_numbers(v: Value, expected: &[f64]) {
        let Value::Array(rc) = &v else { panic!("expected array, got {:?}", v) };
        let got: Vec<Option<f64>> = rc.borrow().iter().map(|x| x.as_finite_f64()).collect();
        let want: Vec<Option<f64>> = expected.iter().map(|x| Some(*x)).collect();
        assert_eq!(got, want);
    }

    #[test]
    fn defaults_before_variadic_positional() {
        // Раньше: «Non-default argument follows default argument».
        let v = run_ok(r#"
            fn f(a, b = 2, *rest) { return [a, b, len(rest)] }
            f(1)
        "#);
        assert_array_numbers(v, &[1.0, 2.0, 0.0]);
        let v = run_ok(r#"
            fn f(a, b = 2, *rest) { return [a, b, len(rest)] }
            f(1, 5, 6, 7)
        "#);
        assert_array_numbers(v, &[1.0, 5.0, 2.0]);
    }

    #[test]
    fn defaults_before_variadic_keyword() {
        let v = run_ok(r#"
            fn f(a, b = 2, **kw) { return [a, b, len(kw)] }
            f(b = 8, a = 0, z = 1)
        "#);
        assert_array_numbers(v, &[0.0, 8.0, 1.0]);
    }

    #[test]
    fn several_star_unpackings_keep_order() {
        // Раньше: [3, 4, 1, 2].
        let v = run_ok(r#"
            fn s(a, b, c, d) { return [a, b, c, d] }
            x = [1, 2]
            y = [3, 4]
            s(*x, *y)
        "#);
        assert_array_numbers(v, &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn positional_and_unpacked_named_conflict() {
        let err = run_err(r#"
            fn f(a, **kw) { return a }
            f(1, **{"a": 2})
        "#);
        assert!(err.to_string().contains("multiple values for argument 'a'"), "{}", err);
    }

    #[test]
    fn positional_and_named_conflict_with_variadic() {
        // Раньше `b` молча получал 5.
        let err = run_err(r#"
            fn f(a, b, *rest) { return a }
            f(5, a = 2)
        "#);
        assert!(err.to_string().contains("multiple values for argument 'a'"), "{}", err);
    }

    #[test]
    fn lambda_named_args_bind_by_name() {
        // Раньше: «Named arguments are not supported for built-in function 'h'».
        let v = run_ok(r#"
            h = fn(a, b) => a - b
            h(b = 1, a = 10)
        "#);
        assert_eq!(v.as_finite_f64(), Some(9.0));
    }

    #[test]
    fn lambda_kwargs() {
        let v = run_ok(r#"
            g = fn(a, **kw) => len(kw) * 10 + a
            g(1, z = 2, y = 3)
        "#);
        assert_eq!(v.as_finite_f64(), Some(21.0));
    }

    #[test]
    fn runtime_binding_error_is_catchable() {
        let v = run_ok(r#"
            g = fn(a, b) => a
            r = "none"
            try {
                g(1, c = 2)
            } catch e {
                r = "caught"
            }
            r
        "#);
        assert_eq!(v, Value::String("caught".to_string()));
    }

    // ========== #20: методы и конструкторы классов ==========

    fn num(v: Value) -> f64 {
        v.as_finite_f64().unwrap_or_else(|| panic!("expected number, got {:?}", v))
    }

    #[test]
    fn method_with_variadic_parameters() {
        let src = r#"
            cls A {
                public:
                new A() {
                }
                fn m(*args) {
                    return len(args)
                }
                fn mk(x, *rest, **kw) {
                    return x * 100 + len(rest) * 10 + len(kw)
                }
            }
            let a = A()
            a.m(1, 2, 3) * 10000 + a.m() * 1000 + a.mk(1, 2, 3, k = 4) + a.m(*[1, 2]) * 0
        "#;
        assert_eq!(num(run_ok(src)), 30121.0);
    }

    #[test]
    fn method_with_class_param_and_variadic() {
        let src = r#"
            cls F {
                public:
                new F() {
                }
                fn s(@class, *xs) {
                    return len(xs)
                }
            }
            let f = F()
            f.s() * 100 + f.s(1) * 10 + f.s(*[1, 2, 3])
        "#;
        assert_eq!(num(run_ok(src)), 13.0);
    }

    #[test]
    fn constructor_with_variadic_and_exact_overload() {
        let src = r#"
            cls C {
                public:
                s: int
                new C(a) {
                    this.s = a
                }
                new C(a, *rest) {
                    this.s = a + len(rest) * 100
                }
            }
            cls B {
                public:
                n: int
                new B(*args) {
                    this.n = len(args)
                }
            }
            C(1).s * 10000 + C(1, 2, 3).s * 10 + B().n + B(*[1, 2, 3]).n * 1000
        "#;
        // C(1) -> 1 (точная перегрузка), C(1, 2, 3) -> 201, B() -> 0, B(*[1,2,3]) -> 3
        assert_eq!(num(run_ok(src)), 10000.0 + 2010.0 + 0.0 + 3000.0);
    }

    #[test]
    fn constructor_with_kwargs_named_and_spread() {
        let src = r#"
            cls D {
                public:
                a: int
                k: int
                new D(a, **kw) {
                    this.a = a
                    this.k = len(kw)
                }
            }
            D(1, x = 2, y = 3).k * 1000 + D(1, **{"q": 1}).k * 100 + D(a = 5).a
        "#;
        assert_eq!(num(run_ok(src)), 2105.0);
    }

    #[test]
    fn super_and_delegation_into_variadic_constructor() {
        let src = r#"
            cls P {
                public:
                total: int
                new P(*nums) {
                    this.total = len(nums)
                }
            }
            cls Q(P) {
                public:
                new Q(a, b, c) {
                    super(a, b, c)
                }
            }
            cls R {
                public:
                n: int
                new R(*xs) {
                    this.n = len(xs)
                }
                new R(a, b) : this(a, b, 0, 0) {
                }
            }
            Q(1, 2, 3).total * 100 + R(1, 2).n * 10 + R(1, 2, 3).n
        "#;
        assert_eq!(num(run_ok(src)), 343.0);
    }

    #[test]
    fn variadic_annotations_are_checked_per_element() {
        let ok = r#"
            fn f(*nums: int) {
                return len(nums)
            }
            fn g(**kw: int) {
                return len(kw)
            }
            f(1, 2) * 10 + g(a = 1)
        "#;
        assert_eq!(num(run_ok(ok)), 21.0);
        let err = run_err("fn f(*nums: int) {\n    return len(nums)\n}\nf(1, \"x\")");
        assert!(format!("{}", err).contains("nums[1]"), "{}", err);
        let err = run_err("fn g(**kw: int) {\n    return len(kw)\n}\ng(a = \"s\")");
        assert!(format!("{}", err).contains("kw['a']"), "{}", err);
    }
}
