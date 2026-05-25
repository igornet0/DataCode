//! Tests for `system.time.perf_counter()` (monotonic high-resolution timer).

use data_code::common::numeric::FloatValue;
use data_code::{run, Value};

fn assert_float_finite(source: &str) {
    match run(source) {
        Ok(Value::Float(FloatValue::Finite(_))) => {}
        Ok(v) => panic!("expected Float(Finite), got {:?}", v),
        Err(e) => panic!("run failed: {:?}", e),
    }
}

fn assert_bool(source: &str, expected: bool) {
    match run(source) {
        Ok(Value::Bool(b)) => assert_eq!(b, expected),
        Ok(v) => panic!("expected Bool({}), got {:?}", expected, v),
        Err(e) => panic!("run failed: {:?}", e),
    }
}

fn assert_string(source: &str, expected: &str) {
    match run(source) {
        Ok(Value::String(s)) => assert_eq!(s, expected),
        Ok(v) => panic!("expected String('{}'), got {:?}", expected, v),
        Err(e) => panic!("run failed: {:?}", e),
    }
}

#[test]
fn perf_counter_returns_float() {
    let source = r#"
from system import time
t = time.perf_counter()
typeof(t)
"#;
    assert_string(source, "float");
}

#[test]
fn perf_counter_is_monotonic() {
    let source = r#"
from system import time
a = time.perf_counter()
b = time.perf_counter()
b >= a
"#;
    assert_bool(source, true);
}

#[test]
fn perf_counter_measures_positive_interval() {
    let source = r#"
from system import time
start = time.perf_counter()
for i in range(1000000) {
}
end = time.perf_counter()
end - start > 0
"#;
    assert_bool(source, true);
}

#[test]
fn perf_counter_rejects_arguments() {
    let source = r#"
from system import time
time.perf_counter(1)
"#;
    let result = run(source);
    assert!(result.is_err(), "expected TypeError, got {:?}", result);
    let msg = format!("{}", result.unwrap_err());
    assert!(
        msg.contains("TypeError") && msg.contains("perf_counter"),
        "unexpected error: {}",
        msg
    );
}

#[test]
fn perf_counter_via_import_system() {
    assert_float_finite(
        r#"
import system
system.time.perf_counter()
"#,
    );
}
