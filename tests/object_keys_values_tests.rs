//! Plain dict `.keys` / `.values`: read-only views, iteration, mutation errors.

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
fn keys_len() {
    assert_number(
        r#"o = { 1: 10, 2: 20 }
len(o.keys)"#,
        2.0,
    );
}

#[test]
fn values_len_and_live_cell() {
    assert_number(
        r#"o = { 1: 100 }
v = o.values
o[1] = 999
v[0]"#,
        999.0,
    );
}

#[test]
fn push_on_keys_view_readonly_error() {
    assert_error_contains(
        r#"k = { 1: 2 }.keys
push(k, 3)"#,
        "ReadOnlyError",
    );
}

#[test]
fn index_assign_on_keys_view_errors() {
    assert_error_contains(
        r#"k = { 1: 2 }.keys
k[0] = 99"#,
        "read-only object keys view",
    );
}

#[test]
fn for_in_over_keys() {
    assert_number(
        r#"o = { 1: 0, 2: 0 }
s = 0
for x in o.keys {
  s = s + x
}
s"#,
        3.0,
    );
}

#[test]
fn typeof_keys_view_is_array() {
    match run(r#"typeof({ 1: 1 }.keys)"#) {
        Ok(Value::String(s)) => assert_eq!(s, "array"),
        Ok(v) => panic!("expected string, got {:?}", v),
        Err(e) => panic!("{:?}", e),
    }
}

#[test]
fn plain_dict_keys_property_not_data_field() {
    // `obj.keys` is always the keys view for plain bucket dicts, not a user field named "keys".
    assert_number(
        r#"o = { "x": 1 }
len(o.keys)"#,
        1.0,
    );
}

#[test]
fn class_instance_field_named_keys_not_dict_projection() {
    assert_number(
        r#"
            cls Node {
                public:
                    keys: list[int]
                new Node() { this.keys = [1, 2, 3] }
            }
            len(Node().keys)
        "#,
        3.0,
    );
}
