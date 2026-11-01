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

    /// Edge operands around the i32 immediate boundary (inline vs boxed result).
    const EDGE: [i64; 14] = [
        0, 1, -1, 7, -8,
        i32::MAX as i64, i32::MIN as i64,
        i32::MAX as i64 + 1, i32::MIN as i64 - 1,
        1 << 40, -(1 << 40),
        0x7fff_ffff_ffff, i64::MAX, i64::MIN + 1,
    ];

    // Integer literals are parsed through f64, so |n| > 2^53 cannot be written
    // exactly; build such values from exact bit operations instead.
    fn lit(n: i64) -> String {
        const EXACT: i64 = 1 << 53;
        if n > EXACT || n < -EXACT {
            let hi = n >> 32;
            let lo = n & 0xffff_ffff;
            return format!("((({}) << 32) | ({}))", hi, lo);
        }
        format!("({})", n)
    }

    #[test]
    fn binary_ops_match_i64_reference_at_i32_boundary() {
        for &a in &EDGE {
            for &b in &EDGE {
                assert_int(&format!("{} & {}", lit(a), lit(b)), a & b);
                assert_int(&format!("{} | {}", lit(a), lit(b)), a | b);
                assert_int(&format!("{} ^ {}", lit(a), lit(b)), a ^ b);
            }
            assert_int(&format!("~{}", lit(a)), !a);
        }
    }

    #[test]
    fn shifts_match_i64_reference_including_overflow() {
        for &a in &EDGE {
            for s in [0u32, 1, 31, 32, 33, 62, 63] {
                assert_int(&format!("{} << {}", lit(a), s), a.wrapping_shl(s));
                assert_int(&format!("{} >> {}", lit(a), s), a >> s);
            }
            // shift amounts beyond 63 clamp to 63 (existing semantics)
            assert_int(&format!("{} >> 100", lit(a)), a >> 63);
        }
        assert_int("1 << 31", 1i64 << 31);
        assert_int("(-1) << 31", -1i64 << 31);
        assert_int("1 << 63", i64::MIN);
    }

    #[test]
    fn inline_and_boxed_results_mix_in_loops() {
        // Results crossing the i32 boundary are reused as operands, compared,
        // stored in arrays and accumulated — inline and boxed ints must agree.
        let src = "
            acc = 0
            xs = []
            i = 0
            while i < 40 {
                v = (1 << i) ^ (i & 5)
                xs.push(v & 2147483647)
                acc = acc + (v >> 1)
                i += 1
            }
            acc + len(xs) + xs[39]
        ";
        let mut acc: i64 = 0;
        let mut last = 0;
        for i in 0..40i64 {
            let v = (1i64 << i) ^ (i & 5);
            last = v & 2147483647;
            acc += v >> 1;
        }
        assert_int(src, acc + 40 + last);
    }
}
