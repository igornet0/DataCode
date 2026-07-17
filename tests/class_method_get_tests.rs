//! Class instance `.get(key)` must invoke user `fn get`, not plain-dict `ObjectGetIntegral`.

use data_code::{compile, run, Value};
use data_code::bytecode::OpCode;
use data_code::common::numeric::IntValue;

fn assert_int(source: &str, expected: i64) {
    match run(source) {
        Ok(Value::Int(IntValue::Finite(n))) => assert_eq!(n, expected, "{}", source),
        Ok(Value::Number(n)) if n.fract() == 0.0 && n.is_finite() => {
            assert_eq!(n as i64, expected, "{}", source)
        }
        Ok(v) => panic!("expected int {}, got {:?}\n{}", expected, v, source),
        Err(e) => panic!("expected int {}, got {:?}\n{}", expected, e, source),
    }
}

fn assert_ok(source: &str) {
    match run(source) {
        Ok(_) => {}
        Err(e) => panic!("expected success, got {:?}\n{}", e, source),
    }
}

#[test]
fn class_get_int_literal() {
    assert_int(
        r#"
cls Box {
    new Box() {}
    fn get(key) { return key + 100 }
}
fn __main__() {
    b = Box()
    return b.get(42)
}
"#,
        142,
    );
}

#[test]
fn class_get_string_literal() {
    assert_int(
        r#"
cls Box {
    new Box() {}
    fn get(key) { return 7 }
}
fn __main__() {
    b = Box()
    return b.get("x")
}
"#,
        7,
    );
}

#[test]
fn class_get_variable_key_in_loop() {
    assert_int(
        r#"
cls Counter {
    new Counter() {
        this.counts = {}
    }
    fn get(key) {
        if key in this.counts {
            return this.counts[key]
        }
        return 0
    }
    fn put(key, value) {
        this.counts[key] = value
    }
}
fn __main__() {
    c = Counter()
    words = ["a", "b", "a", "c", "b", "a"]
    for w in words {
        count = c.get(w)
        c.put(w, count + 1)
    }
    return c.get("a")
}
"#,
        3,
    );
}

#[test]
fn class_get_no_object_get_integral_in_bytecode() {
    let (_chunk, functions) = compile(
        r#"
cls HashMap {
    new HashMap() {}
    fn get(key) { return key }
}
fn __main__() {
    m = HashMap()
    m.get(1)
}
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
        !has_integral,
        "class instance .get() must not compile to ObjectGetIntegral"
    );
}

#[test]
fn plain_dict_get_still_uses_object_get_integral() {
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
        "plain dict d.get(k) must still emit ObjectGetIntegral"
    );
}

#[test]
fn hash_map_example_integration() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/examples/ru/09-продвинутые/алгоритмы/массивы/hash_map.dc"
    );
    let source = std::fs::read_to_string(path).expect("hash_map.dc");
    assert_ok(&source);
}
