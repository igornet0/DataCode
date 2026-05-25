//! Conditional expressions: Python ternary and block if-expr.

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
        Ok(Value::String(s)) => assert_eq!(s, expected, "string mismatch"),
        Ok(v) => panic!("expected String, got {:?}", v),
        Err(e) => panic!("error: {:?}", e),
    }
}

#[test]
fn python_ternary_return() {
    assert_number(
        r#"
        fn f() -> int { return 1 if true else 2 }
        f()
        "#,
        1.0,
    );
}

#[test]
fn python_ternary_assign() {
    assert_number(
        r#"
        x = 1 if false else 2
        x
        "#,
        2.0,
    );
}

#[test]
fn python_ternary_short_circuit() {
    assert_number("1 if true else (1 / 0)", 1.0);
}

#[test]
fn nested_ternary() {
    assert_number("1 if false else 2 if true else 3", 2.0);
}

#[test]
fn block_if_expr() {
    assert_number(
        r#"
        x = if true { 10 } else { 20 }
        x
        "#,
        10.0,
    );
}

#[test]
fn precedence() {
    assert_number("1 + 2 if false else 3", 3.0);
}

#[test]
fn coin_change_smoke() {
    assert_number(
        r#"
        INF = float(inf)
        dp = [INF, INF, INF, INF, INF, INF, INF]
        dp[0] = 0
        coins = [1, 3, 4]
        amount = 6
        for a in range(1, amount + 1) {
            for coin in coins {
                if coin <= a and dp[a - coin] + 1 < dp[a] {
                    dp[a] = dp[a - coin] + 1
                }
            }
        }
        dp[amount] if dp[amount] != INF else -1
        "#,
        2.0,
    );
}

#[test]
fn block_if_string() {
    assert_string(
        r#"
        status = if true { "OK" } else { "FAIL" }
        status
        "#,
        "OK",
    );
}

#[test]
fn ternary_not_across_newline() {
    assert_number(
        r#"
        x = 1
        if false { x = 2 }
        x
        "#,
        1.0,
    );
}

#[test]
fn ternary_string() {
    assert_string(
        r#"
        got = 15125
        expected = 15125
        status = if got == expected { "OK" } else { "FAIL" }
        status
        "#,
        "OK",
    );
}
