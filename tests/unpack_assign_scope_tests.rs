//! Unpack assignment at statement level inside functions and compound index assign.

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

#[test]
fn unpack_inside_function_without_let() {
    assert_number(
        r#"
        fn f(a: str, b: str) -> int {
            m, n = len(a), len(b)
            return m + n
        }
        f("ab", "cde")
        "#,
        5.0,
    );
}

#[test]
fn parallel_swap_in_loop() {
    assert_number(
        r#"
        fn fib(n: int) -> int {
            if n <= 1: return n
            prev, curr = 0, 1
            for _ in range(2, n + 1) {
                prev, curr = curr, prev + curr
            }
            return curr
        }
        fib(10)
        "#,
        55.0,
    );
}

#[test]
fn compound_index_assign_in_loop() {
    assert_number(
        r#"
        dp = [0, 10, 0]
        dp[1] += 5
        dp[1]
        "#,
        15.0,
    );
}

#[test]
fn named_arg_not_unpack() {
    assert_number(
        r#"
        fn g(x: int, y: int = 0) -> int { return x + y }
        g(3, y=4)
        "#,
        7.0,
    );
}

#[test]
fn n_var_unpack_from_indexed_tuple_in_for() {
    assert_number(
        r#"
        sum = 0
        tests = [[1, 3, 4], 6, 2]
        for t in [tests] {
            c, a, expected = t[0], t[1], t[2]
            sum = sum + len(c) + a + expected
        }
        sum
        "#,
        11.0,
    );
}
