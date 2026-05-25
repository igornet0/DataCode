//! Полный набор интеграционных проверок для ±∞ в доменах `int` и `float`.
//!
//! Литералы `inf` и унарный `-` дают IEEE ±∞ в домене `float` (`typeof(inf) == "float"`).
//! Выражение `-inf` сворачивается в литерал при компиляции. Ниже отмечены расхождения со
//! «строгим int»-спеком (IEEE через `Number` в бинопах, `sort` по строкам).

use data_code::common::numeric::{hash_float_value, hash_int_value, FloatValue, IntValue};
use data_code::{run, Value};

fn eval(source: &str) -> Value {
    run(source).unwrap_or_else(|e| panic!("run failed: {:?}\n{}", e, source))
}

fn eval_bool(source: &str) -> bool {
    match eval(source) {
        Value::Bool(b) => b,
        v => panic!("expected bool, got {:?} for {}", v, source),
    }
}

fn assert_number_nan(v: Value) {
    match v.as_ieee_f64() {
        Some(n) if n.is_nan() => {}
        Some(n) => panic!("expected nan, got {}", n),
        None => panic!("expected numeric nan, got {:?}", v),
    }
}

fn array_strings(v: Value) -> Vec<String> {
    match v {
        Value::Array(rc) => rc.borrow().iter().map(|x| x.to_string()).collect(),
        other => panic!("expected array, got {:?}", other),
    }
}

// --- 1–2. Parse / typing ---

#[test]
fn typeof_bare_inf_is_float() {
    assert_eq!(eval("typeof(inf)"), Value::String("float".into()));
}

#[test]
fn int_infinity_typeof_and_self_eq() {
    assert_eq!(eval("typeof(int(inf))"), Value::String("int".into()));
    assert!(eval_bool("int(inf) == int(inf)"));
}

#[test]
fn int_neg_infinity_typeof_and_self_eq() {
    assert_eq!(eval("typeof(int(-inf))"), Value::String("int".into()));
    assert!(eval_bool("int(-inf) == int(-inf)"));
}

#[test]
fn float_infinity_typeof_and_self_eq() {
    assert_eq!(eval("typeof(float(inf))"), Value::String("float".into()));
    assert!(eval_bool("float(inf) == float(inf)"));
}

#[test]
fn float_neg_infinity_typeof_and_self_eq() {
    assert_eq!(
        eval("typeof(float(-inf))"),
        Value::String("float".into())
    );
    assert!(eval_bool("float(-inf) == float(-inf)"));
}

// --- 3. Equality ---

#[test]
fn opposite_integer_infinities_not_equal() {
    assert!(!eval_bool("int(inf) == int(-inf)"));
}

#[test]
fn finite_int_not_equal_pos_infinity() {
    assert!(!eval_bool("999999 == int(inf)"));
}

#[test]
fn int_infinity_equals_float_infinity() {
    assert!(eval_bool("int(inf) == float(inf)"));
}

// --- 4. Comparisons ---

#[test]
fn int_inf_gt_large_finite() {
    assert!(eval_bool("int(inf) > 999999"));
}

#[test]
fn int_neg_inf_lt_large_negative_finite() {
    assert!(eval_bool("int(-inf) < -999999"));
}

#[test]
fn int_infinities_ordered() {
    assert!(eval_bool("int(inf) > int(-inf)"));
}

#[test]
fn float_inf_gt_one() {
    assert!(eval_bool("float(inf) > 1.0"));
}

#[test]
fn int_inf_not_lt_small_finite() {
    assert!(!eval_bool("int(inf) < 100"));
}

// --- 5–8. Arithmetic ---

#[test]
fn add_finite_does_not_change_int_infinity() {
    assert_eq!(eval("int(inf) + 1"), Value::Number(f64::INFINITY));
    assert_eq!(
        eval("int(-inf) + 1"),
        Value::Number(f64::NEG_INFINITY)
    );
}

#[test]
fn float_inf_plus_finite_stays_inf() {
    assert_eq!(
        eval("float(inf) + 5.5"),
        Value::Number(f64::INFINITY)
    );
}

#[test]
fn int_inf_plus_int_inf() {
    assert_eq!(eval("int(inf) + int(inf)"), Value::Number(f64::INFINITY));
}

#[test]
fn int_inf_minus_finite() {
    assert_eq!(eval("int(inf) - 10"), Value::Number(f64::INFINITY));
}

#[test]
fn int_neg_inf_minus_finite() {
    assert_eq!(
        eval("int(-inf) - 10"),
        Value::Number(f64::NEG_INFINITY)
    );
}

/// Строгий int: мог бы быть RuntimeError; здесь — IEEE NaN в `Number`.
#[test]
fn int_inf_minus_int_inf_is_nan_number() {
    assert_number_nan(eval("int(inf) - int(inf)"));
}

#[test]
fn float_inf_minus_float_inf_is_nan_number() {
    assert_number_nan(eval("float(inf) - float(inf)"));
}

#[test]
fn int_inf_times_two() {
    assert_eq!(eval("int(inf) * 2"), Value::Number(f64::INFINITY));
}

#[test]
fn int_inf_times_neg_one_flips_sign() {
    assert_eq!(
        eval("int(inf) * -1"),
        Value::Number(f64::NEG_INFINITY)
    );
}

#[test]
fn int_neg_inf_times_neg_one() {
    assert_eq!(eval("int(-inf) * -1"), Value::Number(f64::INFINITY));
}

#[test]
fn int_inf_times_zero_is_nan_number() {
    assert_number_nan(eval("int(inf) * 0"));
}

#[test]
fn int_inf_div_two() {
    assert_eq!(eval("int(inf) / 2"), Value::Number(f64::INFINITY));
}

#[test]
fn int_neg_inf_div_two() {
    assert_eq!(
        eval("int(-inf) / 2"),
        Value::Number(f64::NEG_INFINITY)
    );
}

#[test]
fn int_inf_div_int_inf_nan() {
    assert_number_nan(eval("int(inf) / int(inf)"));
}

#[test]
fn float_inf_div_float_inf_nan() {
    assert_number_nan(eval("float(inf) / float(inf)"));
}

// --- 9. Large finite ---

#[test]
fn int_inf_plus_huge_finite() {
    assert_eq!(
        eval("int(inf) + 999999999999999999"),
        Value::Number(f64::INFINITY)
    );
}

#[test]
fn int_neg_inf_minus_huge_finite() {
    assert_eq!(
        eval("int(-inf) - 999999999999999999"),
        Value::Number(f64::NEG_INFINITY)
    );
}

// --- 10. sort (числовой total order для scalar IEEE; иначе — строковый порядок) ---

#[test]
fn sort_mixed_integers_and_infinities_total_order() {
    let s = array_strings(eval("sort([5, int(inf), -1, int(-inf)])"));
    assert_eq!(
        s,
        vec!["-inf".to_string(), "-1".to_string(), "5".to_string(), "inf".to_string()],
    );
}

#[test]
fn sort_floats_with_infinity() {
    let sorted = eval("sort([1.5, float(inf), -3.0])");
    let s = array_strings(sorted);
    assert_eq!(
        s,
        vec!["-3".to_string(), "1.5".to_string(), "inf".to_string()]
    );
}

// --- 11. Hashing (common::numeric) ---

#[test]
fn hash_int_pos_inf_stable_and_distinct_from_neg_inf() {
    let a = hash_int_value(IntValue::PosInfinity);
    let b = hash_int_value(IntValue::PosInfinity);
    let n = hash_int_value(IntValue::NegInfinity);
    assert_eq!(a, b);
    assert_ne!(a, n);
}

#[test]
fn hash_float_infinities_align_with_int_sentinels() {
    assert_eq!(
        hash_int_value(IntValue::PosInfinity),
        hash_float_value(FloatValue::PosInfinity)
    );
}

// --- 12–13. Objects & arrays ---

#[test]
fn dict_key_positive_infinity() {
    let src = r#"
        x = { int(inf): "value" }
        x[int(inf)]
    "#;
    assert_eq!(eval(src), Value::String("value".into()));
}

#[test]
fn dict_key_negative_infinity() {
    let src = r#"
        x = { int(-inf): "neg" }
        x[int(-inf)]
    "#;
    assert_eq!(eval(src), Value::String("neg".into()));
}

#[test]
fn array_with_infinity_length() {
    assert_eq!(eval("len([1, int(inf), 3])"), Value::Number(3.0));
}

// --- 15. str ---

#[test]
fn str_int_infinities() {
    assert_eq!(eval("str(int(inf))"), Value::String("inf".into()));
    assert_eq!(eval("str(int(-inf))"), Value::String("-inf".into()));
}

// --- 16. Casts ---

#[test]
fn float_widens_int_infinity() {
    assert_eq!(
        eval("float(int(inf))"),
        Value::Float(FloatValue::PosInfinity)
    );
}

#[test]
fn int_cast_from_float_infinity_is_typed_int_infinity() {
    assert_eq!(
        eval("int(float(inf))"),
        Value::Int(IntValue::PosInfinity)
    );
}

// --- 17. Algorithms ---

#[test]
fn dijkstra_style_dist_init_comparison() {
    let src = r#"
        dist = { 1: 0, 2: int(inf), 3: int(inf) }
        dist[2] > dist[1]
    "#;
    assert!(eval_bool(src));
}

#[test]
fn min_max_with_int_infinity() {
    assert_eq!(eval("min(10, int(inf))"), Value::Number(10.0));
    assert_eq!(eval("min(10, int(-inf))"), Value::Number(f64::NEG_INFINITY));
    assert_eq!(eval("max(10, int(inf))"), Value::Number(f64::INFINITY));
    assert_eq!(eval("max(int(-inf), 10)"), Value::Number(10.0));
}

// --- 18. Compile / equality ---

#[test]
fn compile_int_infinity_snippet_succeeds() {
    data_code::compile("let x = int(inf)").unwrap();
}

#[test]
fn equality_deep_compare_int_infinities_not_pointer_artifact() {
    let src = r#"
        a = int(inf)
        b = int(inf)
        a == b
    "#;
    assert!(eval_bool(src));
}

// --- 19. Unary / abs ---

#[test]
fn double_negation_int_neg_inf() {
    assert_eq!(
        eval("-int(-inf)"),
        Value::Int(IntValue::PosInfinity)
    );
}

#[test]
fn unary_minus_int_inf() {
    assert_eq!(eval("-int(inf)"), Value::Int(IntValue::NegInfinity));
}

#[test]
fn bare_neg_inf_literal_folded() {
    assert_eq!(
        eval("-inf"),
        Value::Float(FloatValue::NegInfinity)
    );
}

#[test]
fn abs_int_neg_infinity() {
    assert_eq!(eval("abs(int(-inf))"), Value::Number(f64::INFINITY));
}

// --- 20. Stability ---

#[test]
fn repeated_add_small_finite_to_int_infinity_unchanged() {
    let src = r#"
        x = int(inf)
        for i in range(20000) {
            x = x + 1
        }
        x == int(inf)
    "#;
    assert!(eval_bool(src));
}

// --- 21. Indeterminate ops → NaN, не RuntimeError ---

#[test]
fn invalid_arithmetic_indeterminate_as_nan() {
    assert_number_nan(eval("int(inf) - int(inf)"));
    assert_number_nan(eval("int(inf) * 0"));
    assert_number_nan(eval("int(inf) / int(inf)"));
}

#[test]
fn runtime_error_not_used_for_int_inf_indeterminate_ops() {
    for expr in [
        "int(inf) - int(inf)",
        "int(inf) * 0",
        "int(inf) / int(inf)",
    ] {
        let r = run(expr);
        assert!(
            matches!(r, Ok(Value::Number(n)) if n.is_nan()),
            "expected Ok(Number(nan)) for {}, got {:?}",
            expr,
            r
        );
    }
}

#[test]
#[ignore = "Нет поверхностного API to_json/from_json для скаляров."]
fn json_inf_roundtrip_placeholder() {}
