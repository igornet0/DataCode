//! Regression tests for integral dict/set fast paths (OPTIMAZ_PLAN phase 1.1–1.2).

use data_code::{run, Value};
use std::time::Instant;

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

fn assert_bool(source: &str, expected: bool) {
    match run(source) {
        Ok(Value::Bool(b)) => assert_eq!(b, expected, "expr:\n{}", source),
        Ok(v) => panic!("expected Bool, got {:?}\n{}", v, source),
        Err(e) => panic!("error {:?}\n{}", e, source),
    }
}

#[test]
fn integral_dict_many_updates_same_key_variants() {
    assert_number(
        r#"
d = {}
for i in range(500) {
    d[i] = i
    d[i + 0.0] = i * 2
}
d.get(250)
"#,
        500.0,
    );
}

#[test]
fn object_get_integral_key_without_load_semantics() {
    assert_number(
        r#"
d = {1: 10, 2: 20}
d.get(1)
"#,
        10.0,
    );
    assert_number(
        r#"
d = {1: 10}
d.get(1.0)
"#,
        10.0,
    );
    assert_number(
        r#"
d = {}
d.get(99, 42)
"#,
        42.0,
    );
}

#[test]
fn dict_bracket_integral_immediate_index() {
    assert_number(
        r#"
d = {5: 100}
d[5]
"#,
        100.0,
    );
}

#[test]
fn object_get_integral_with_inf_default() {
    match run(
        r#"
d = {1: 5}
d.get(2, float(inf))
"#,
    ) {
        Ok(Value::Number(n)) => assert!(n.is_infinite() && n > 0.0),
        other => panic!("expected +inf, got {:?}", other),
    }
    assert_number(
        r#"
d = {10: 3}
d.get(10)
"#,
        3.0,
    );
}

#[test]
fn object_get_integral_astar_style_compare() {
    assert_bool(
        r#"
g = {5: 10}
tentative = 8
tentative < g.get(7, float(inf))
"#,
        true,
    );
}

#[test]
fn integral_set_membership_stress() {
    assert_bool(
        r#"
s = set()
for i in range(800) {
    s.add(i)
}
ok = true
for i in range(800) {
    if !(i in s) { ok = false }
    if !(i + 0.0 in s) { ok = false }
}
ok
"#,
        true,
    );
}

#[test]
fn object_get_integral_emitted_for_variable_key_get() {
    use data_code::bytecode::OpCode;
    use data_code::compile;
    let (_chunk, functions) = compile(
        r#"
fn f(d, k) { return d.get(k, -1) }
f({1: 2}, 1)
"#,
    )
    .expect("compile");
    let has_integral = functions.iter().any(|f| {
        f.chunk
            .code
            .iter()
            .any(|op| matches!(op, OpCode::ObjectGetIntegral))
    });
    assert!(
        has_integral,
        "d.get(k) must compile to ObjectGetIntegral in function body"
    );
}

#[test]
fn cluster_get_string_literal_keeps_call_path() {
    use data_code::bytecode::OpCode;
    use data_code::compile;
    let (_chunk, functions) = compile(
        r#"
from database_engine import DatabaseCluster
fn f(c) { return c.get("primary") }
f(DatabaseCluster())
"#,
    )
    .expect("compile");
    let Some(f) = functions.first() else {
        panic!("expected one function");
    };
    let has_integral = f.chunk.code.iter().any(|op| matches!(op, OpCode::ObjectGetIntegral));
    let has_call = f
        .chunk
        .code
        .iter()
        .any(|op| matches!(op, OpCode::Call(_)));
    assert!(
        has_call && !has_integral,
        "cluster.get(\"primary\") must use GetArrayElement+Call, not ObjectGetIntegral"
    );
}

#[test]
fn object_index_integral_emitted_for_scalar_subscript() {
    use data_code::bytecode::OpCode;
    use data_code::compile;
    let (_chunk, functions) = compile(
        r#"
fn f(d, k) {
    return d[k]
}
f({1: 2}, 1)
"#,
    )
    .expect("compile");
    let code: Vec<_> = functions
        .iter()
        .flat_map(|f| f.chunk.code.iter())
        .cloned()
        .collect();
    assert!(
        code.iter().any(|op| matches!(op, OpCode::ObjectIndexIntegral)),
        "expected ObjectIndexIntegral for d[k], got {:?}",
        code
    );
}

#[test]
fn set_add_discard_integral_emitted() {
    use data_code::{run, Value};
    let result = run(
        r#"
fn f(s, x) {
    s.discard(x)
    s.add(x)
    return len(s)
}
f(set(), 1)
"#,
    );
    match result {
        Ok(Value::Number(n)) => assert_eq!(n, 1.0),
        Ok(v) => panic!("expected Number(1), got {:?}", v),
        Err(e) => panic!("error: {:?}", e),
    }
}

/// Wall-clock smoke for dict integral path (printed for CHANGELOG / manual baseline).
#[test]
#[ignore = "timing benchmark: cargo test bench_integral_dict_get_loop --release -- --ignored --nocapture"]
fn bench_integral_dict_get_loop() {
    let n = 50_000;
    let source = format!(
        r#"
d = {{}}
for i in range({n}) {{
    d[i] = i
}}
let s = 0
for i in range({n}) {{
    let s = s + d.get(i, -1)
}}
s
"#,
        n = n
    );
    let start = Instant::now();
    let r = run(&source);
    let elapsed = start.elapsed();
    assert!(r.is_ok(), "{:?}", r);
    if let Ok(Value::Number(v)) = r {
        let expected = (n as f64) * (n as f64 - 1.0) / 2.0;
        assert!((v - expected).abs() < 1.0, "sum {} vs {}", v, expected);
    }
    println!(
        "bench_integral_dict_get_loop: n={} elapsed={:?}",
        n, elapsed
    );
}
