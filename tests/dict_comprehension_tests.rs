//! Dict comprehension `{ k: v for x in it [if c] }` and hashable object keys.

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

fn assert_bool(source: &str, expected: bool) {
    match run(source) {
        Ok(Value::Bool(b)) => assert_eq!(b, expected),
        Ok(v) => panic!("expected Bool, got {:?}", v),
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
fn dict_comp_identity() {
    assert_number(
        r#"o = { x: x for x in [1, 2, 3] }
o[2]"#,
        2.0,
    );
}

#[test]
fn dict_comp_double() {
    assert_number(
        r#"o = { x: x * 2 for x in [1, 2, 3] }
o[2]"#,
        4.0,
    );
}

#[test]
fn dict_comp_if_filter() {
    assert_number(
        r#"o = { x: x for x in [1, 2, 3] if x > 1 }
o[2] + o[3] + o.get(1, 10)"#,
        15.0,
    );
}

#[test]
fn object_literal_numeric_key_bool_val() {
    assert_bool(
        r#"o = { 1: true }
o[1]"#,
        true,
    );
}

#[test]
fn object_literal_bool_key_string_val() {
    assert_string(
        r#"o = { true: "yes" }
o[true]"#,
        "yes",
    );
}

#[test]
fn object_literal_string_key() {
    assert_number(
        r#"o = { "a": 123 }
o["a"]"#,
        123.0,
    );
}

#[test]
fn dict_comp_scope_outer_x_unchanged() {
    assert_number(
        r#"x = 100
o = { x: x for x in [1, 2] }
x"#,
        100.0,
    );
}

#[test]
fn dict_comp_unhashable_key_runtime_error() {
    assert_error_contains("{ []: 1 for i in [1] }", "unhashable type");
}

/// `{ x for x in it }` — set comprehension (не dict); раньше ожидали ошибку «нет `:`».
#[test]
fn brace_without_colon_is_set_comprehension() {
    match run("{ x for x in [1] }") {
        Ok(Value::Set(s)) => assert_eq!(s.borrow().len(), 1),
        Ok(v) => panic!("expected Set, got {:?}", v),
        Err(e) => panic!("expected ok set comp, got {:?}", e),
    }
}

#[test]
fn parse_error_missing_colon() {
    assert_error_contains("{ x 1 for x in [1] }", "Expect ':'");
}
