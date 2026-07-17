//! Block-scoped `for` pattern variables vs function-scoped body bindings.

#[cfg(test)]
mod tests {
    use data_code::{compile, run, LangError};

    fn assert_number_result(source: &str, expected: f64) {
        let result = run(source);
        match result {
            Ok(v) if v.as_ieee_f64() == Some(expected) => {}
            Ok(v) => panic!("Expected numeric({}), got {:?}", expected, v),
            Err(e) => panic!("Error: {:?}", e),
        }
    }

    fn assert_runtime_error(source: &str) {
        assert!(run(source).is_err(), "expected runtime error");
    }

    fn assert_compile_ok(source: &str) {
        compile(source).expect("expected compile ok");
    }

    fn assert_outer_assign_compile_error(source: &str) {
        match compile(source) {
            Err(LangError::ParseError { message, .. }) => {
                assert!(
                    message.contains("cannot assign to outer variable"),
                    "unexpected message: {}",
                    message
                );
            }
            Ok(_) => panic!("expected compile error"),
            Err(e) => panic!("expected ParseError, got {:?}", e),
        }
    }

    #[test]
    fn for_body_binding_visible_after_loop() {
        assert_number_result(
            r#"
fn test() {
    for i in range(3) {
        elapsed = i
    }
    return elapsed
}
test()
"#,
            2.0,
        );
    }

    #[test]
    fn for_pattern_not_visible_after_loop() {
        assert_runtime_error(
            r#"
fn test() {
    for i in range(3) {}
    return i
}
test()
"#,
        );
    }

    #[test]
    fn nested_fn_after_for_can_assign_same_name_as_pattern() {
        assert_compile_ok(
            r#"
fn build() {
    items = ["a", "b"]
    for ch in items {
        x = ch
    }
    fn walk() {
        ch = 42
        return ch
    }
    return walk()
}
build()
"#,
        );
        assert_number_result(
            r#"
fn build() {
    items = ["a", "b"]
    for ch in items {
        x = ch
    }
    fn walk() {
        ch = 42
        return ch
    }
    return walk()
}
build()
"#,
            42.0,
        );
    }

    #[test]
    fn nested_fn_cannot_assign_body_binding_from_for_without_let() {
        assert_outer_assign_compile_error(
            r#"
fn outer() {
    for i in range(1) {
        x = 10
    }
    fn inner() {
        x = 20
    }
    inner()
}
outer()
"#,
        );
    }

    #[test]
    fn nested_for_pattern_scopes() {
        assert_runtime_error(
            r#"
fn test() {
    for i in range(2) {
        for j in range(2) {}
        return j
    }
    return 0
}
test()
"#,
        );
        assert_runtime_error(
            r#"
fn test() {
    for i in range(2) {
        for j in range(2) {}
    }
    return i
}
test()
"#,
        );
    }

    #[test]
    fn huffman_walk_pattern_compiles() {
        assert_compile_ok(
            r#"
fn build_huffman_codes(freq) {
    for ch in freq {
        uid = 1
    }
    fn walk(node) {
        ch = node
        return ch
    }
    return walk("x")
}
build_huffman_codes({"a": 1})
"#,
        );
    }

    #[test]
    fn for_body_binding_after_nested_loops_no_slot_collision() {
        assert_number_result(
            r#"
fn lps_string(s) {
    n = len(s)
    dp = [[0 for _ in range(n)] for _ in range(n)]
    for i in range(n) { dp[i][i] = 1 }
    for length in range(2, n + 1) {
        for i in range(n - length + 1) {
            j = i + length - 1
            if s[i] == s[j] {
                inner = 0 if length == 2 else dp[i + 1][j - 1]
                dp[i][j] = inner + 2
            } else {
                dp[i][j] = max(dp[i + 1][j], dp[i][j - 1])
            }
        }
    }
    i = 0
    j = n - 1
    left = []
    right = []
    return j
}
lps_string("abc")
"#,
            2.0,
        );
    }
}
