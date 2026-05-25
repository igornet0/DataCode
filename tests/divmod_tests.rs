//! `divmod()` builtin: Python-style floor division + remainder; `%` / `//` compatibility.

#[cfg(test)]
mod tests {
    use data_code::common::error::ErrorType;
    use data_code::common::numeric::{FloatValue, IntValue};
    use data_code::{run, LangError, Value};

    fn run_ok(source: &str) -> Value {
        run(source).unwrap_or_else(|e| panic!("unexpected error: {:?}", e))
    }

    fn assert_divmod_int(source: &str, q_exp: i64, r_exp: i64) {
        let v = run_ok(source);
        let Value::Tuple(rc) = v else {
            panic!("expected tuple, got {:?}", v);
        };
        let t = rc.borrow();
        assert_eq!(t.len(), 2);
        match (&t[0], &t[1]) {
            (Value::Int(IntValue::Finite(q)), Value::Int(IntValue::Finite(r))) => {
                assert_eq!(*q, q_exp);
                assert_eq!(*r, r_exp);
            }
            other => panic!("expected int pair, got {:?}", other),
        }
    }

    fn assert_divmod_float_approx(source: &str, q_exp: f64, r_exp: f64) {
        let v = run_ok(source);
        let Value::Tuple(rc) = v else {
            panic!("expected tuple, got {:?}", v);
        };
        let t = rc.borrow();
        assert_eq!(t.len(), 2);
        match (&t[0], &t[1]) {
            (Value::Float(FloatValue::Finite(q)), Value::Float(FloatValue::Finite(r))) => {
                assert!((q - q_exp).abs() < 1e-9, "q: expected {}, got {}", q_exp, q);
                assert!((r - r_exp).abs() < 1e-9, "r: expected {}, got {}", r_exp, r);
            }
            other => panic!("expected float pair, got {:?}", other),
        }
    }

    fn assert_bool_result(source: &str, expected: bool) {
        match run(source) {
            Ok(Value::Bool(b)) => assert_eq!(b, expected),
            Ok(v) => panic!("expected bool, got {:?}", v),
            Err(e) => panic!("error: {:?}", e),
        }
    }

    #[test]
    fn divmod_basic_int_literals() {
        assert_divmod_int("divmod(10, 3)", 3, 1);
        assert_divmod_int("divmod(20, 5)", 4, 0);
        assert_divmod_int("divmod(9, 2)", 4, 1);
    }

    #[test]
    fn divmod_negative_int() {
        assert_divmod_int("divmod(-10, 3)", -4, 2);
        assert_divmod_int("divmod(10, -3)", -4, -2);
        assert_divmod_int("divmod(-10, -3)", 3, -1);
        assert_divmod_int("divmod(-1, 5)", -1, 4);
        assert_divmod_int("divmod(1, -5)", -1, -4);
        assert_divmod_int("divmod(-1, -5)", 0, -1);
    }

    #[test]
    fn divmod_float_mixed() {
        assert_divmod_float_approx("divmod(10.5, 3)", 3.0, 1.5);
        assert_divmod_float_approx("divmod(7.25, 2.5)", 2.0, 2.25);
    }

    #[test]
    fn divmod_zero_is_error() {
        let err = run("divmod(10, 0)").unwrap_err();
        match err {
            LangError::RuntimeError {
                error_type: Some(et),
                ..
            } => assert_eq!(et, ErrorType::ZeroDivisionError),
            e => panic!("expected ZeroDivisionError, got {:?}", e),
        }
    }

    #[test]
    fn divmod_zero_caught_as_zero_division_error() {
        let source = r#"
            let ok = false
            try {
                divmod(10, 0)
            } catch ZeroDivisionError e {
                ok = true
            }
            ok
        "#;
        assert_bool_result(source, true);
    }

    #[test]
    fn divmod_unpack_two_vars() {
        let v = run_ok(
            r#"
            let r1 = 0
            let c1 = 0
            r1, c1 = divmod(17, 5)
            r1 * 100 + c1
        "#,
        );
        match v {
            Value::Number(n) => assert_eq!(n, 302.0),
            Value::Int(IntValue::Finite(n)) => assert_eq!(n, 302),
            other => panic!("expected number or int, got {:?}", other),
        }
    }

    #[test]
    fn divmod_matches_floor_div_and_mod() {
        let source = r#"
            let pairs = [[10, 3], [-10, 3], [10, -3], [-10, -3]]
            let ok = true
            for pair in pairs {
                let a = pair[0]
                let b = pair[1]
                let q, r = divmod(a, b)
                if !(q == a // b and r == a % b) {
                    ok = false
                }
            }
            ok
        "#;
        assert_bool_result(source, true);
    }

    #[test]
    fn divmod_type_error_wrong_arity() {
        let err = run("divmod(1)").unwrap_err();
        match err {
            LangError::RuntimeError {
                error_type: Some(et),
                ..
            } => assert_eq!(et, ErrorType::TypeError),
            e => panic!("expected TypeError, got {:?}", e),
        }
    }
}
