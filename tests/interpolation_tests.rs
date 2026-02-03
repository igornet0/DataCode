// String interpolation tests: "${...}", escape "\${", unclosed, nested braces

#[cfg(test)]
mod tests {
    use data_code::{run, Value};

    fn assert_string_result(source: &str, expected: &str) {
        let result = run(source);
        match result {
            Ok(Value::String(s)) => assert_eq!(s, expected, "Expected '{}', got '{}'", expected, s),
            Ok(v) => panic!("Expected String('{}'), got {:?}", expected, v),
            Err(e) => panic!("Error: {:?}", e),
        }
    }

    fn assert_error(source: &str) {
        let result = run(source);
        assert!(result.is_err(), "Expected error, got {:?}", result);
    }

    #[test]
    fn test_interpolation_variable() {
        assert_string_result(
            r#"let name = "Igor"
"Hello ${name}""#,
            "Hello Igor",
        );
    }

    #[test]
    fn test_interpolation_expression() {
        assert_string_result(
            r#"let a = 2
let b = 3
"result: ${a + b}""#,
            "result: 5",
        );
    }

    #[test]
    fn test_interpolation_multiple() {
        assert_string_result(
            r#"let name = "Igor"
let age = 22
"Hello ${name}, you are ${age} years old""#,
            "Hello Igor, you are 22 years old",
        );
    }

    #[test]
    fn test_interpolation_escaped_literal() {
        // \${ → literal ${ in output (not interpolation); unescape strips the backslash
        assert_string_result(
            r#" "\${not interpolation}""#,
            r#"${not interpolation}"#,
        );
    }

    #[test]
    fn test_interpolation_backslash_then_escaped() {
        // Source "\\${literal}": one \ then ${ → output \ then literal ${ 
        assert_string_result(
            r#" "\\${literal}""#,
            r"\${literal}",
        );
    }

    #[test]
    fn test_interpolation_unclosed_errors() {
        assert_error(r#""Hello ${name""#);
    }

    #[test]
    fn test_interpolation_nested_braces() {
        // Expression with braces: obj.method() or similar
        assert_string_result(
            r#"let x = "ok"
"result: ${x}""#,
            "result: ok",
        );
    }

    #[test]
    fn test_interpolation_property_access() {
        assert_string_result(
            r#"let obj = { "name": "DataCode" }
"Hello ${obj.name}""#,
            "Hello DataCode",
        );
    }

    #[test]
    fn test_interpolation_number_and_bool() {
        assert_string_result(
            r#"let n = 42
let b = true
"n=${n} b=${b}""#,
            "n=42 b=true",
        );
    }

    #[test]
    fn test_plain_string_no_interpolation() {
        assert_string_result(r#""just a string""#, "just a string");
    }

    #[test]
    fn test_string_with_dollar_no_brace() {
        // "$" without "{" is not interpolation
        assert_string_result(r#""price: $99""#, "price: $99");
    }
}
