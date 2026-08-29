//! Tests for global `copy()` and unified deep copy via `.clone()` / `set.copy()`.

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
fn copy_scalar_unchanged() {
    assert_number(
        r#"
        copy(42)
        "#,
        42.0,
    );
    assert_number(
        r#"
        x = copy("hi")
        len(x)
        "#,
        2.0,
    );
}

#[test]
fn copy_nested_array_independence() {
    assert_number(
        r#"
        a = [[1, 2], [3, 4]]
        b = copy(a)
        b[0][0] = 99
        a[0][0]
        "#,
        1.0,
    );
}

#[test]
fn reference_vs_copy_array() {
    assert_number(
        r#"
        a = [1, 2, 3]
        b = a
        push(b, 4)
        len(a)
        "#,
        4.0,
    );
    assert_number(
        r#"
        a = [1, 2, 3]
        b = copy(a)
        push(b, 4)
        len(a)
        "#,
        3.0,
    );
}

#[test]
fn copy_matches_clone_method() {
    assert_number(
        r#"
        a = [[1], [2]]
        b = copy(a)
        c = a.clone()
        b[0][0] = 10
        c[1][0] = 20
        a[0][0] + a[1][0]
        "#,
        3.0,
    );
}

#[test]
fn copy_legacy_object_nested() {
    assert_number(
        r#"
        o = {"items": [1, 2]}
        c = copy(o)
        c["items"][0] = 99
        o["items"][0]
        "#,
        1.0,
    );
}

#[test]
fn copy_bucket_object_integral_keys() {
    assert_number(
        r#"
        o = {1: [10], 2: [20]}
        c = copy(o)
        c[1][0] = 99
        o[1][0]
        "#,
        10.0,
    );
}

#[test]
fn copy_set_deep_new_container() {
    assert_number(
        r#"
        s = set([1, 2, 3])
        t = copy(s)
        t.add(4)
        len(s)
        "#,
        3.0,
    );
}

#[test]
fn copy_table_independence() {
    assert_number(
        r#"
        t1 = table([[1, "Alice"]], ["id", "name"])
        t2 = copy(t1)
        t2.add_row([2, "Bob"])
        len(t1)
        "#,
        1.0,
    );
    assert_number(
        r#"
        t1 = table([[1, "Alice"]], ["id", "name"])
        t2 = copy(t1)
        t2.add_row([2, "Bob"])
        len(t2)
        "#,
        2.0,
    );
}

#[test]
fn copy_table_vs_alias() {
    assert_number(
        r#"
        t1 = table([[1, "Alice"]], ["id", "name"])
        alias = t1
        cloned = copy(t1)
        alias.add_row([2, "Bob"])
        cloned.add_row([3, "Charlie"])
        len(t1)
        "#,
        1.0,
    );
    assert_number(
        r#"
        t1 = table([[1, "Alice"]], ["id", "name"])
        alias = t1
        cloned = copy(t1)
        alias.add_row([2, "Bob"])
        cloned.add_row([3, "Charlie"])
        len(cloned)
        "#,
        2.0,
    );
}

#[test]
fn set_copy_matches_global_copy() {
    assert_number(
        r#"
        s = set([1, 2])
        a = s.copy()
        b = copy(s)
        push(a, 3)
        push(b, 4)
        len(s)
        "#,
        2.0,
    );
}

#[test]
fn copy_function_type_error() {
    match run(
        r#"
        copy(print)
        "#,
    ) {
        Err(_) => {}
        Ok(v) => panic!("expected error, got {:?}", v),
    }
}
