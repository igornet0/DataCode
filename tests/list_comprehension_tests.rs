//! List comprehension `[ e for ... ]` — nested for/if, scope, unpack.

use data_code::{run, Value};

fn assert_numbers_array(source: &str, expected: &[f64]) {
    match run(source) {
        Ok(Value::Array(arr)) => {
            let v = arr.borrow();
            assert_eq!(
                v.len(),
                expected.len(),
                "length mismatch: {:?}",
                v.iter().map(|x| format!("{:?}", x)).collect::<Vec<_>>()
            );
            for i in 0..v.len() {
                match &v[i] {
                    Value::Number(n) => assert!(
                        (n - expected[i]).abs() < 1e-9,
                        "idx {} expected {} got {}",
                        i,
                        expected[i],
                        n
                    ),
                    other => panic!("expected number at {}, got {:?}", i, other),
                }
            }
        }
        Ok(v) => panic!("expected Array, got {:?}", v),
        Err(e) => panic!("error: {:?}", e),
    }
}

fn assert_number_line(source: &str, expected: f64) {
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
fn list_comp_identity() {
    assert_numbers_array("[ x for x in [1, 2, 3] ]", &[1.0, 2.0, 3.0]);
}

#[test]
fn list_comp_squares() {
    assert_numbers_array("[ x * x for x in [1, 2, 3] ]", &[1.0, 4.0, 9.0]);
}

#[test]
fn list_comp_if_filter() {
    assert_numbers_array("[ x for x in [1, 2, 3, 4] if x % 2 == 0 ]", &[2.0, 4.0]);
}

#[test]
fn list_comp_nested_two_fors() {
    assert_numbers_array(
        "[ x * y for x in [1, 2] for y in [10, 20] ]",
        &[10.0, 20.0, 20.0, 40.0],
    );
}

#[test]
fn list_comp_if_between_fors() {
    // for x; if; for y  — if filters on x before inner y loop
    assert_numbers_array(
        "[ 100 * x + y for x in [1, 2] if x == 2 for y in [1, 2] ]",
        &[201.0, 202.0],
    );
}

#[test]
fn list_comp_unpack_pair() {
    assert_numbers_array(
        "[ a + b for a, b in [ [1, 2], [3, 4] ] ]",
        &[3.0, 7.0],
    );
}

#[test]
fn list_comp_outer_scope_unchanged() {
    assert_number_line(
        r#"x = 100
let a = [ x for x in [1, 2] ]
x"#,
        100.0,
    );
}

#[test]
fn list_comp_in_set() {
    assert_number_line(r#"len(set([ x for x in [1, 2, 2, 3] ]))"#, 3.0);
}

#[test]
fn list_comp_undefined_in_elt() {
    assert_error_contains("[ y for x in [1] ]", "Undefined variable");
}

#[test]
fn list_comp_unpack_count_mismatch() {
    assert_error_contains(
        "[ a + b for a, b in [1, 2, 3] ]",
        "Expected array",
    );
}
