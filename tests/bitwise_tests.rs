//! Bitwise operators: &, |, ^, ~, <<, >>

#[cfg(test)]
mod tests {
    use data_code::common::error::ErrorType;
    use data_code::common::numeric::IntValue;
    use data_code::{run, LangError, Value};

    fn run_ok(source: &str) -> Value {
        run(source).unwrap_or_else(|e| panic!("unexpected error for {:?}: {:?}", source, e))
    }

    fn assert_int(source: &str, expected: i64) {
        match run_ok(source) {
            Value::Int(IntValue::Finite(n)) => assert_eq!(n, expected, "expr: {}", source),
            other => panic!("expected int {}, got {:?} for {}", expected, other, source),
        }
    }

    fn assert_error_type(source: &str, expected: ErrorType) {
        let err = run(source).unwrap_err();
        match err {
            LangError::RuntimeError {
                error_type: Some(et),
                message,
                ..
            } => {
                assert_eq!(et, expected);
                assert!(
                    message.contains("Bitwise operator is supported only for int values"),
                    "message: {}",
                    message
                );
            }
            e => panic!("expected TypeError, got {:?}", e),
        }
    }

    #[test]
    fn bit_and_or_xor() {
        assert_int("5 & 3", 1);
        assert_int("5 | 3", 7);
        assert_int("5 ^ 3", 6);
    }

    #[test]
    fn shifts() {
        assert_int("1 << 5", 32);
        assert_int("32 >> 2", 8);
        assert_int("8 >> 3", 1);
        assert_int("1 << 10", 1024);
    }

    #[test]
    fn bit_not() {
        assert_int("~5", -6);
    }

    #[test]
    fn shift_precedence_over_add() {
        assert_int("1 << 2 + 1", 8);
    }

    #[test]
    fn lowbit_bit_and_with_negation() {
        assert_int("8 & (-8)", 8);
        assert_int("1 & (-1)", 1);
    }

    #[test]
    fn logical_and_is_not_bitwise_and() {
        // `and` is short-circuit logical; `&` is bitwise (lowbit pattern).
        match run_ok("1 and (-1)") {
            Value::Int(IntValue::Finite(n)) => {
                assert_eq!(n, -1, "logical and returns rhs when lhs truthy")
            }
            Value::Number(n) => assert_eq!(n, -1.0, "logical and returns rhs when lhs truthy"),
            other => panic!("expected -1 from logical and, got {:?}", other),
        }
        assert_int("1 & (-1)", 1);
    }

    #[test]
    fn bitwise_type_errors() {
        assert_error_type("5.0 & 3", ErrorType::TypeError);
        assert_error_type("true & 1", ErrorType::TypeError);
        assert_error_type("\"x\" | 1", ErrorType::TypeError);
    }

    #[test]
    fn range_accepts_int_from_bit_shift() {
        match run_ok("len(range(1 << 3))") {
            Value::Number(n) => assert_eq!(n, 8.0),
            other => panic!("expected len 8, got {:?}", other),
        }
    }
}
