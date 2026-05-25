//! `int()` / `float()` over ±inf via `Number` (`1.0/0.0`) and cross-type equality.

use data_code::common::numeric::{FloatValue, IntValue};
use data_code::{run, Value};

fn run_last(source: &str) -> Value {
    run(source).unwrap_or_else(|e| panic!("run failed: {:?}", e))
}

#[test]
fn int_cast_pos_inf_div() {
    assert_eq!(
        run_last("int(1.0/0.0)"),
        Value::Int(IntValue::PosInfinity)
    );
}

#[test]
fn int_cast_neg_inf_unary_div() {
    assert_eq!(
        run_last("int(-(1.0/0.0))"),
        Value::Int(IntValue::NegInfinity)
    );
}

#[test]
fn float_cast_inf_div() {
    assert_eq!(
        run_last("float(1.0/0.0)"),
        Value::Float(FloatValue::PosInfinity)
    );
}

#[test]
fn float_cast_neg_inf() {
    assert_eq!(
        run_last("float(-(1.0/0.0))"),
        Value::Float(FloatValue::NegInfinity)
    );
}

#[test]
fn typeof_int_float_after_inf_cast() {
    assert_eq!(
        run_last("typeof(int(1.0/0.0))"),
        Value::String("int".to_string())
    );
    assert_eq!(
        run_last("typeof(float(1.0/0.0))"),
        Value::String("float".to_string())
    );
}

#[test]
fn int_inf_eq_float_inf() {
    assert_eq!(
        run_last("int(1.0/0.0) == float(1.0/0.0)"),
        Value::Bool(true)
    );
}

#[test]
fn empty_int_float_returns_typed_zero() {
    assert_eq!(run_last("int()"), Value::Int(IntValue::Finite(0)));
    assert_eq!(
        run_last("float()"),
        Value::Float(FloatValue::Finite(0.0))
    );
}
