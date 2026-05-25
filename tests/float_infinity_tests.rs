//! `float(inf)` / `float(-inf)` / `float(nan)` and `isinf()` — литерал `inf`, IEEE 754.

use data_code::common::numeric::FloatValue;
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

#[test]
fn float_inf_literal_cast_and_print() {
    assert_eq!(
        eval("typeof(float(inf))"),
        Value::String("float".into())
    );
    assert_eq!(
        eval("float(inf)"),
        Value::Float(FloatValue::PosInfinity)
    );
    assert_eq!(eval("str(float(inf))"), Value::String("inf".into()));
}

#[test]
fn float_neg_inf_literal_cast() {
    assert_eq!(
        eval("float(-inf)"),
        Value::Float(FloatValue::NegInfinity)
    );
    assert_eq!(eval("str(float(-inf))"), Value::String("-inf".into()));
}

#[test]
fn float_nan_literal_cast() {
    match eval("float(nan)") {
        Value::Float(f) => assert!(f.is_nan()),
        v => panic!("expected Float(nan), got {:?}", v),
    }
}

#[test]
fn float_inf_arithmetic_and_compare() {
    assert!(eval_bool("float(inf) > 0"));
    assert!(eval_bool("float(-inf) < 0"));
    assert!(eval_bool("float(inf) + 1 == float(inf)"));
    assert!(eval_bool("float(-inf) - 1 == float(-inf)"));
    assert!(!eval_bool("float(inf) == float(-inf)"));
    assert!(eval_bool("float(inf) > 1000000"));
    assert!(eval_bool("float(-inf) < -999999"));
    assert!(eval_bool("float(inf) == float(inf)"));
}

#[test]
fn float_int_inf_cross_cast() {
    assert_eq!(
        eval("float(int(inf))"),
        Value::Float(FloatValue::PosInfinity)
    );
    assert_eq!(
        eval("float(int(-inf))"),
        Value::Float(FloatValue::NegInfinity)
    );
}

#[test]
fn isinf_builtin() {
    assert!(eval_bool("isinf(float(inf))"));
    assert!(eval_bool("isinf(float(-inf))"));
    assert!(eval_bool("isinf(int(inf))"));
    assert!(!eval_bool("isinf(1.0)"));
    assert!(!eval_bool("isinf(float(nan))"));
}

#[test]
fn interpolation_float_inf_display() {
    let src = r#"
        x = float(inf)
        "${x=}"
    "#;
    assert_eq!(eval(src), Value::String("x=inf".into()));
}

#[test]
fn grid_bounds_style_condition() {
    let src = r#"
        fn inside(rows, cols, nr, nc) {
            if !(0 <= nr < rows and 0 <= nc < cols): return false
            return true
        }
        inside(5, 5, 2, 3)
    "#;
    assert!(eval_bool(src));
}
