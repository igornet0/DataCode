//! `in` on set/dict must be an O(1) hash lookup and agree with writes made through the
//! integral fast paths (`d[i] = v`, `s.add(i)`), including after the dict has been passed to
//! a native function such as `print` / `str` (regression for igornet0/DataCode#22).

use data_code::{run, Value};

fn assert_bool(source: &str, expected: bool) {
    match run(source) {
        Ok(Value::Bool(b)) => assert_eq!(b, expected, "expr:\n{}", source),
        Ok(v) => panic!("expected Bool, got {:?}\n{}", v, source),
        Err(e) => panic!("error {:?}\n{}", e, source),
    }
}

fn assert_number(source: &str, expected: f64) {
    match run(source) {
        Ok(Value::Number(n)) => assert!(
            (n - expected).abs() < 1e-9,
            "expected {}, got {}\n{}",
            expected,
            n,
            source
        ),
        Ok(v) => panic!("expected Number, got {:?}\n{}", v, source),
        Err(e) => panic!("error {:?}\n{}", e, source),
    }
}

fn assert_string(source: &str, expected: &str) {
    match run(source) {
        Ok(Value::String(s)) => assert_eq!(s, expected, "expr:\n{}", source),
        Ok(v) => panic!("expected String, got {:?}\n{}", v, source),
        Err(e) => panic!("error {:?}\n{}", e, source),
    }
}

#[test]
fn dict_in_sees_integral_index_write() {
    assert_bool("d = {}\nd[7] = 2\n7 in d", true);
    assert_bool("d = {}\nd[7] = 2\n8 in d", false);
}

#[test]
fn dict_in_inside_function_param() {
    assert_bool(
        r#"
fn walk(came_from) {
    came_from[5] = 3
    came_from[3] = 1
    cur = 5
    n = 0
    while cur in came_from { cur = came_from[cur]; n += 1 }
    return n == 2
}
walk({})
"#,
        true,
    );
}

#[test]
fn dict_integral_keys_survive_native_call() {
    assert_number("d = {}\nd[7] = 1\ns = str(d)\nlen(d)", 1.0);
    assert_bool("d = {}\nd[7] = 1\ns = str(d)\n7 in d", true);
    assert_string("d = {}\nd[7] = 1\nstr(d)", "{7: 1}");
}

#[test]
fn dict_len_counts_integral_and_string_keys() {
    assert_number("y = {1: 5, \"a\": 2}\ny[1] = 9\ny[2] = 3\nlen(y)", 3.0);
    assert_number("y = {1: 5, \"a\": 2}\ny[1] = 9\ns = str(y)\ny[1]", 9.0);
}

#[test]
fn dict_in_string_key() {
    assert_bool("d = {}\nd[\"a\"] = 1\n\"a\" in d", true);
    assert_bool("d = {}\nd[\"a\"] = 1\n\"b\" in d", false);
}

#[test]
fn set_in_integral_and_tuple_members() {
    assert_bool("s = set()\ns.add(3)\n3 in s", true);
    assert_bool("s = set()\ns.add(3)\ns.discard(3)\n3 in s", false);
    assert_bool("s = set()\ns.add((1, 2))\nt = (1, 2)\nt in s", true);
    assert_bool("s = set()\ns.add((1, 2))\nt = (2, 1)\nt in s", false);
    assert_bool("s = set()\ns.add((1, 2))\ns.discard((1, 2))\nt = (1, 2)\nt in s", false);
    assert_number("s = set()\nfor i in range(100) { s.add((i, i)) }\nlen(s)", 100.0);
}

#[test]
fn set_discard_keeps_other_members_reachable() {
    // Removing from the middle of an open-addressing probe chain must not hide later keys.
    assert_number(
        r#"
s = set()
for i in range(2000) { s.add(i * 500) }
for i in range(0, 2000, 2) { s.discard(i * 500) }
c = 0
for i in range(2000) { if i * 500 in s { c += 1 } }
c
"#,
        1000.0,
    );
}

#[test]
fn set_in_large_is_fast() {
    // Was O(len) per `in` (whole set materialized); 40k lookups took seconds.
    let t0 = std::time::Instant::now();
    assert_number(
        r#"
s = set()
for i in range(40000) { s.add(i) }
c = 0
for i in range(40000) { if i in s { c += 1 } }
c
"#,
        40000.0,
    );
    assert!(t0.elapsed().as_secs_f64() < 2.0, "set `in` too slow: {:?}", t0.elapsed());
}
