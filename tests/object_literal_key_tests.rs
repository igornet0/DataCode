//! Scope-aware object literal keys: bound identifier → computed key, unbound → string field name.

#[cfg(test)]
mod tests {
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

    fn assert_string(source: &str, expected: &str) {
        match run(source) {
            Ok(Value::String(s)) => assert_eq!(s, expected),
            Ok(v) => panic!("expected String, got {:?}", v),
            Err(e) => panic!("error: {:?}", e),
        }
    }

    #[test]
    fn computed_key_from_bound_variable() {
        assert_number(
            r#"
start_id = 0
f_score = {start_id: 100}
f_score[start_id]
"#,
            100.0,
        );
    }

    #[test]
    fn unbound_identifier_is_string_field_name() {
        assert_number(
            r#"
fn row() {
    return {id: 2, name: "Bob"}
}
row()["id"]
"#,
            2.0,
        );
    }

    #[test]
    fn unbound_identifier_name_field() {
        assert_string(
            r#"
fn row() {
    return {id: 2, name: "Bob"}
}
row()["name"]
"#,
            "Bob",
        );
    }

    #[test]
    fn bound_identifier_uses_computed_key_not_string() {
        assert_number(
            r#"
id = 5
{id: 99}[id]
"#,
            99.0,
        );
    }

    #[test]
    fn explicit_string_key_when_variable_shadows() {
        assert_number(
            r#"
id = 5
{"id": 42}["id"]
"#,
            42.0,
        );
    }

    #[test]
    fn script_level_global_bound_in_object_literal() {
        assert_number(
            r#"
g = 42
{g: 1}[g]
"#,
            1.0,
        );
    }

    #[test]
    fn computed_key_inside_function() {
        assert_number(
            r#"
fn test() {
    start_id = 0
    f_score = {start_id: 100}
    return f_score[start_id]
}
test()
"#,
            100.0,
        );
    }
}
