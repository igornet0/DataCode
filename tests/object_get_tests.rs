//! Plain dict `.get(key, default=null)` and `KeyError` on missing `obj[key]`.

use data_code::{run, Value};

fn assert_number(source: &str, expected: f64) {
    match run(source) {
        Ok(Value::Number(n)) => assert!(
            (n - expected).abs() < 1e-9,
            "expected {}, got {}",
            expected,
            n
        ),
        Ok(v) => panic!("expected Number, got {:?}", v),
        Err(e) => panic!("error: {:?}", e),
    }
}

fn assert_null(source: &str) {
    match run(source) {
        Ok(Value::Null) => {}
        Ok(v) => panic!("expected Null, got {:?}", v),
        Err(e) => panic!("error: {:?}", e),
    }
}

fn assert_string(source: &str, expected: &str) {
    match run(source) {
        Ok(Value::String(s)) => assert_eq!(s, expected),
        Ok(v) => panic!("expected String, got {:?}", v),
        Err(e) => panic!("error: {:?}", e),
    }
}

fn assert_error_contains(source: &str, substr: &str) {
    let r = run(source);
    assert!(r.is_err(), "expected error, got {:?}", r);
    let msg = format!("{:?}", r.unwrap_err());
    assert!(
        msg.contains(substr),
        "expected error containing '{}', got {}",
        substr,
        msg
    );
}

#[test]
fn object_get_existing_and_default() {
    assert_number(
        r#"obj = { "a": 1 }
obj.get("a")"#,
        1.0,
    );
    assert_null(
        r#"obj = { "a": 1 }
obj.get("b")"#,
    );
    assert_number(
        r#"obj = { "a": 1 }
obj.get("b", 100)"#,
        100.0,
    );
}

#[test]
fn object_get_on_bound_dict_not_only_literal() {
    assert_string(
        r#"obj = { "name": "Alex" }
obj.get("name")"#,
        "Alex",
    );
    assert_null(
        r#"obj = { "name": "Alex" }
obj.get("age")"#,
    );
    assert_number(
        r#"obj = { "name": "Alex" }
obj.get("age", 18)"#,
        18.0,
    );
}

#[test]
fn object_get_numeric_and_bool_keys() {
    assert_string(
        r#"data = { 1: "one", 2: "two" }
data.get(1)"#,
        "one",
    );
    assert_null(
        r#"data = { 1: "one", 2: "two" }
data.get(3)"#,
    );
    assert_string(
        r#"flags = { true: "enabled", false: "disabled" }
flags.get(true)"#,
        "enabled",
    );
}

#[test]
fn object_get_unhashable_key_type_error() {
    assert_error_contains(
        r#"obj = {}
obj.get([])"#,
        "unhashable type",
    );
}

#[test]
fn object_subscript_missing_key_keyerror() {
    assert_error_contains(
        r#"obj = { "a": 1 }
obj["missing"]"#,
        "KeyError",
    );
}

#[test]
fn object_subscript_unhashable_type_error() {
    assert_error_contains(
        r#"obj = {}
obj[[]]"#,
        "unhashable type",
    );
}

#[test]
fn object_get_missing_no_keyerror() {
    assert_null(
        r#"obj = { "x": 1 }
obj.get("missing")"#,
    );
}

#[test]
fn object_get_default_null_means_missing_distinct() {
    // key present with null value — get should return null, not default
    assert_null(
        r#"obj = { "k": null }
obj.get("k", 99)"#,
    );
}
