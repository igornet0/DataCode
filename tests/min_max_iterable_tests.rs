//! Streaming min/max over iterables with optional key.

use data_code::{run, Value};

fn assert_number_near(source: &str, expected: f64) {
    match run(source) {
        Ok(Value::Number(n)) => assert!(
            (n - expected).abs() < 1e-9,
            "expected {}, got {}",
            expected,
            n
        ),
        Ok(v) => panic!("expected number, got {:?}", v),
        Err(e) => panic!("unexpected error {:?}", e),
    }
}

fn assert_error_contains(source: &str, substr: &str) {
    match run(source) {
        Ok(v) => panic!("expected error, got {:?}", v),
        Err(e) => {
            let msg = format!("{}", e);
            assert!(
                msg.contains(substr),
                "expected error containing {:?}, got {:?}",
                substr,
                msg
            );
        }
    }
}

#[test]
fn min_basic_array() {
    assert_number_near("min([5, 2, 9])", 2.0);
}

#[test]
fn max_basic_array() {
    assert_number_near("max([5, 2, 9])", 9.0);
}

#[test]
fn min_empty_array_value_error() {
    assert_error_contains("min([])", "min() arg is empty");
    assert_error_contains("min([])", "ValueError:");
}

#[test]
fn max_empty_array_value_error() {
    assert_error_contains("max([])", "max() arg is empty");
}

#[test]
fn min_empty_set_value_error() {
    assert_error_contains("min(set([]))", "min() arg is empty");
}

#[test]
fn min_set_of_numbers() {
    assert_number_near("min(set([7, 1, 4]))", 1.0);
}

#[test]
fn min_with_key_returns_element_with_min_key() {
    assert_number_near(
        r#"min([{"priority": 10}, {"priority": 3}], fn(x)=>x.priority).priority"#,
        3.0,
    );
}

#[test]
fn min_named_key_kwarg() {
    assert_number_near(
        r#"min([-2, -1, -3], key=fn(y)=>abs(y))"#,
        -1.0,
    );
}

#[test]
fn min_lazy_mapped_iterable_by_key_abs() {
    assert_number_near(
        r#"min(map([-2, -1, -3], fn(x)=>x), fn(y)=>abs(y))"#,
        -1.0,
    );
}

#[test]
fn min_incompatible_element_types_type_error() {
    assert_error_contains(r#"min([1, "a"])"#, "TypeError:");
}

#[test]
fn key_function_returns_null_type_error() {
    assert_error_contains(
        r#"min([1, 2, 3], fn(_)=>null)"#,
        "key function returned null",
    );
}

#[test]
fn zero_arg_min_expects_argument() {
    assert_error_contains("min()", "expects at least 1 argument");
}
