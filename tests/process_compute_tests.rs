//! Tests for `system.process` compute device API.

use data_code::{run, Value};

#[test]
fn process_device_constants_and_cpu() {
    match run(
        r#"
from system import process
process.set_device(process.cpu)
d = process.get_device()
process.cpu == d
"#,
    ) {
        Ok(Value::Bool(true)) => {}
        Ok(v) => panic!("expected true, got {:?}", v),
        Err(e) => panic!("{:#?}", e),
    }
}

#[test]
fn process_gpu_min_size() {
    match run(
        r#"
from system import process
process.set_gpu_min_size(1000)
process.get_gpu_min_size()
"#,
    ) {
        Ok(Value::Number(n)) => assert_eq!(n, 1000.0),
        Ok(v) => panic!("expected 1000, got {:?}", v),
        Err(e) => panic!("{:#?}", e),
    }
}

#[test]
fn process_vector_add_cpu() {
    match run(
        r#"
from system import process
process.set_device(process.cpu)
a = [1.0, 2.0, 3.0]
b = [4.0, 5.0, 6.0]
process.vector_add(a, b)[0]
"#,
    ) {
        Ok(Value::Number(n)) => assert!((n - 5.0).abs() < 1e-9),
        Ok(v) => panic!("expected 5.0, got {:?}", v),
        Err(e) => panic!("{:#?}", e),
    }
}

#[test]
fn process_run_sum() {
    match run(
        r#"
from system import process
process.set_device(process.cpu)
process.run("sum", [1, 2, 3, 4])
"#,
    ) {
        Ok(Value::Number(n)) => assert!((n - 10.0).abs() < 1e-9),
        Ok(v) => panic!("expected 10, got {:?}", v),
        Err(e) => panic!("{:#?}", e),
    }
}

#[test]
fn process_info_cpu_backend() {
    match run(
        r#"
from system import process
process.set_device(process.cpu)
info = process.info()
info["backend"]
"#,
    ) {
        Ok(Value::String(s)) => assert_eq!(s, "cpu"),
        Ok(v) => panic!("expected cpu backend, got {:?}", v),
        Err(e) => panic!("{:#?}", e),
    }
}

#[test]
fn process_has_metal_reports_bool() {
    match run(
        r#"
from system import process
process.has_metal()
"#,
    ) {
        Ok(Value::Bool(_)) => {}
        Ok(v) => panic!("expected bool, got {:?}", v),
        Err(e) => panic!("{:#?}", e),
    }
}

#[cfg(feature = "metal")]
#[test]
#[ignore = "requires --features metal on macOS with GPU"]
fn process_metal_vector_add_large() {
    match run(
        r#"
from system import process
process.set_device(process.auto)
process.set_gpu_min_size(1000)
let a = []
let b = []
for i in range(5000) {
    a.push(i * 1.0)
    b.push(i * 2.0)
}
out = process.vector_add(a, b)
out[100]
"#,
    ) {
        Ok(Value::Number(n)) => assert!((n - 300.0).abs() < 1e-6),
        Ok(v) => panic!("expected 300, got {:?}", v),
        Err(e) => panic!("{:#?}", e),
    }
}
