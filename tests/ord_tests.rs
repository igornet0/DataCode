//! `ord()` builtin: Unicode code point of a single-character string.

#[cfg(test)]
mod tests {
    use data_code::common::error::ErrorType;
    use data_code::common::numeric::IntValue;
    use data_code::{run, LangError, Value};

    fn run_ok(source: &str) -> Value {
        run(source).unwrap_or_else(|e| panic!("unexpected error: {:?}", e))
    }

    fn assert_ord_int(source: &str, expected: i64) {
        let v = run_ok(source);
        match v {
            Value::Int(IntValue::Finite(n)) => assert_eq!(n, expected),
            other => panic!("expected int, got {:?}", other),
        }
    }

    fn assert_bool_result(source: &str, expected: bool) {
        match run(source) {
            Ok(Value::Bool(b)) => assert_eq!(b, expected),
            Ok(v) => panic!("expected bool, got {:?}", v),
            Err(e) => panic!("error: {:?}", e),
        }
    }

    fn assert_error_type(source: &str, expected: ErrorType) {
        let err = run(source).unwrap_err();
        match err {
            LangError::RuntimeError {
                error_type: Some(et),
                ..
            } => assert_eq!(et, expected),
            e => panic!("expected {:?}, got {:?}", expected, e),
        }
    }

    #[test]
    fn ord_ascii() {
        assert_ord_int("ord(\"A\")", 65);
        assert_ord_int("ord(\"a\")", 97);
        assert_ord_int("ord(\"0\")", 48);
    }

    #[test]
    fn ord_cyrillic_cjk_emoji() {
        assert_ord_int("ord(\"Я\")", 1071);
        assert_ord_int("ord(\"中\")", 20013);
        assert_ord_int("ord(\"😀\")", 128512);
        assert_ord_int("ord(\"🚀\")", 128640);
    }

    #[test]
    fn ord_eq_number_literal() {
        assert_bool_result("ord(\"A\") == 65", true);
    }

    #[test]
    fn ord_wrong_arity() {
        assert_error_type("ord()", ErrorType::TypeError);
        assert_error_type("ord(\"A\", \"B\")", ErrorType::TypeError);
    }

    #[test]
    fn ord_empty_or_multi_char() {
        assert_error_type("ord(\"\")", ErrorType::TypeError);
        assert_error_type("ord(\"AB\")", ErrorType::TypeError);
        assert_error_type("ord(\"Пр\")", ErrorType::TypeError);
        assert_error_type("ord(\"😀😀\")", ErrorType::TypeError);
    }

    #[test]
    fn ord_wrong_type() {
        assert_error_type("ord(123)", ErrorType::RuntimeError);
        assert_error_type("ord(true)", ErrorType::RuntimeError);
        assert_error_type("ord([])", ErrorType::RuntimeError);
    }
}
