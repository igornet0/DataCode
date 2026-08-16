//! Builtin `set()`, `.add`, `len`, `typeof`, and `in` for sets.

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
fn set_empty_len_zero() {
    assert_number(
        r#"s = set()
len(s)"#,
        0.0,
    );
}

#[test]
fn set_from_empty_array() {
    assert_number(
        r#"s = set([])
len(s)"#,
        0.0,
    );
}

#[test]
fn set_typeof_is_set() {
    assert_string(
        r#"s = set()
typeof(s)"#,
        "set",
    );
}

#[test]
fn set_from_array_uniqs_elements() {
    assert_number(
        r#"s = set([1, 2, 2, 3, 1])
len(s)"#,
        3.0,
    );
}

#[test]
fn set_add_mutates_and_uniques() {
    assert_number(
        r#"s = set()
s.add(10)
s.add(10)
s.add(20)
len(s)"#,
        2.0,
    );
}

#[test]
fn set_in_true() {
    assert_bool(
        r#"s = set([1, 2, 3])
2 in s"#,
        true,
    );
}

#[test]
fn set_in_false() {
    assert_bool(
        r#"s = set([1, 2, 3])
99 in s"#,
        false,
    );
}

#[test]
fn set_not_in() {
    assert_bool(
        r#"s = set([1])
!5 in s"#,
        true,
    );
}

#[test]
fn set_string_elements_hashable() {
    assert_number(
        r#"s = set(["a", "b", "a"])
len(s)"#,
        2.0,
    );
}

#[test]
fn set_non_iterable_single_arg_is_type_error() {
    assert_error_contains(
        r#"x = set(123)"#,
        "TypeError: set() expected an iterable",
    );
}

#[test]
fn set_too_many_args_is_type_error() {
    assert_error_contains(
        r#"x = set([], [])"#,
        "TypeError: set() expected at most 1 argument",
    );
}

#[test]
fn set_from_table_column() {
    assert_number(
        r#"
orders = table([
    [1, "west"],
    [2, "east"],
    [3, "west"],
], ["id", "region_id"])
s = set(orders.region_id)
len(s)
"#,
        2.0,
    );
}

#[test]
fn set_from_table_column_for_in() {
    assert_number(
        r#"
orders = table([
    [1, "west"],
    [2, "east"],
    [3, "west"],
], ["id", "region_id"])
c = 0
for key in set(orders.region_id) {
    c = c + 1
}
c
"#,
        2.0,
    );
}

#[test]
fn set_from_table_column_membership() {
    assert_bool(
        r#"
orders = table([
    [1, "west"],
    [2, "east"],
    [3, "west"],
], ["id", "region_id"])
"west" in set(orders.region_id)
"#,
        true,
    );
}

#[test]
fn set_from_tuple() {
    assert_number(
        r#"s = set((1, 2, 2, 3))
len(s)"#,
        3.0,
    );
}

#[test]
fn set_from_string() {
    assert_number(
        r#"s = set("aab")
len(s)"#,
        2.0,
    );
}

#[test]
fn set_from_existing_set() {
    assert_number(
        r#"a = set([1, 2, 2])
b = set(a)
len(b)"#,
        2.0,
    );
}

#[test]
fn set_from_table_rows_is_unhashable() {
    assert_error_contains(
        r#"
t = table([[1, "a"], [2, "b"]], ["id", "name"])
set(t)
"#,
        "unhashable type",
    );
}

#[test]
fn set_from_array_with_unhashable_is_runtime_error() {
    assert_error_contains(
        r#"x = set([[1]])"#,
        "unhashable type",
    );
}

#[test]
fn nested_fn_closure_shares_set_capture() {
    // Regression: recursive inner fn must read the same `visited` set (not another slot).
    assert_number(
        r#"
            fn runit() {
                seen = set()
                fn walk(n) {
                    seen.add(n)
                    if n < 3 {
                        walk(n + 1)
                    }
                }
                walk(0)
                return len(seen)
            }
            runit()
        "#,
        4.0,
    );
}

#[test]
fn set_add_duplicate_len_one() {
    assert_number(
        r#"s = set()
s.add(1)
s.add(1)
s.add(1)
len(s)"#,
        1.0,
    );
}

#[test]
fn set_remove_existing() {
    assert_bool(
        r#"s = set([1, 2, 3])
s.remove(2)
2 in s"#,
        false,
    );
}

#[test]
fn set_remove_key_error() {
    assert_error_contains(
        r#"s = set([1, 2, 3])
s.remove(100)"#,
        "KeyError",
    );
}

#[test]
fn set_discard_missing() {
    assert_number(
        r#"s = set([1, 2])
s.discard(100)
len(s)"#,
        2.0,
    );
}

#[test]
fn set_pop_single() {
    assert_number(
        r#"s = set([5])
s.pop()
len(s)"#,
        0.0,
    );
}

#[test]
fn set_pop_empty_key_error() {
    assert_error_contains(
        r#"s = set()
s.pop()"#,
        "pop from empty set",
    );
}

#[test]
fn set_clear() {
    assert_number(
        r#"s = set([1, 2, 3])
s.clear()
len(s)"#,
        0.0,
    );
}

#[test]
fn set_copy_independent() {
    assert_number(
        r#"a = set([1, 2])
b = a.copy()
b.add(3)
len(a)"#,
        2.0,
    );
}

#[test]
fn set_update_values() {
    assert_number(
        r#"s = set([1, 2])
s.update([2, 3, 4])
len(s)"#,
        4.0,
    );
}

#[test]
fn set_tuple_keys_dedup() {
    assert_number(
        r#"s = set()
s.add((1, 2))
s.add((1, 2))
len(s)"#,
        1.0,
    );
}

#[test]
fn set_equals_order_insensitive() {
    assert_bool(
        r#"set([1, 2]) == set([2, 1])"#,
        true,
    );
}

#[test]
fn set_not_equals_different_members() {
    assert_bool(
        r#"set([1, 2]) != set([1, 3])"#,
        true,
    );
}

#[test]
fn set_contains_method() {
    assert_bool(
        r#"s = set([1, 2, 3])
s.contains(2)"#,
        true,
    );
}

#[test]
fn set_for_in_two_iterations() {
    assert_number(
        r#"s = set([1, 2])
c = 0
for x in s {
    c = c + 1
}
c"#,
        2.0,
    );
}
