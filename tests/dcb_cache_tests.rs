//! Tests for disk bytecode cache (.dcb), fingerprint invalidation, executed_modules singleton, and dependency order.

use data_code::{run_with_base_path, Value};
use std::fs;
use std::path::PathBuf;

fn temp_dcb_test_dir() -> PathBuf {
    let base = std::env::temp_dir().join("datacode_dcb_tests");
    let _ = fs::create_dir_all(&base);
    base
}

/// run_with_base_path returns the last expression value; simple expression works.
#[test]
fn test_run_with_base_path_returns_last_value() {
    let dir = temp_dcb_test_dir();
    let result = run_with_base_path("1 + 1", &dir);
    assert!(result.is_ok(), "{:?}", result);
    assert_eq!(result.unwrap(), Value::Number(2.0));
}

/// After first run that imports a module, .dcb file should exist for that module.
#[test]
fn test_dcb_file_created_after_import() {
    let dir = temp_dcb_test_dir();
    let mod_path = dir.join("mymod.dc");
    fs::write(&mod_path, "fn one() { return 1 }\n").expect("write mymod.dc");
    // Use intermediate variable so last expression is definitely the call result
    let main_source = "import mymod\nlet r = mymod.one()\nr\n";

    let result = run_with_base_path(main_source, &dir);
    assert!(result.is_ok(), "first run should succeed: {:?}", result);
    assert_eq!(result.unwrap(), Value::Number(1.0));

    let canonical = mod_path.canonicalize().unwrap_or_else(|_| mod_path.clone());
    let dcb_path = data_code::vm::dcb::dcb_cache_path(&canonical);
    assert!(dcb_path.exists(), ".dcb should exist after first run: {}", dcb_path.display());
}

/// Changing source invalidates cache: different source produces different result after re-run.
#[test]
fn test_dcb_invalidated_on_source_change() {
    let dir = temp_dcb_test_dir();
    let mod_path = dir.join("varmod.dc");
    fs::write(&mod_path, "let val = 42\n").expect("write varmod.dc");
    let main_source = "import varmod\nvarmod.val\n";

    let r1 = run_with_base_path(main_source, &dir).unwrap();
    assert_eq!(r1, Value::Number(42.0));

    fs::write(&mod_path, "let val = 99\n").expect("change varmod.dc");
    let r2 = run_with_base_path(main_source, &dir).unwrap();
    assert_eq!(r2, Value::Number(99.0), "changed source should be recompiled and return new value");
}

/// import A; from A import x — module runs once (executed_modules singleton).
#[test]
fn test_singleton_module_import_and_from() {
    let dir = temp_dcb_test_dir();
    let mod_path = dir.join("sing.dc");
    fs::write(&mod_path, "let x = 10\n").expect("write sing.dc");
    let main_source = r#"
import sing
from sing import x
x + 1
"#;
    let result = run_with_base_path(main_source, &dir);
    assert!(result.is_ok(), "singleton import + from should succeed: {:?}", result);
    assert_eq!(result.unwrap(), Value::Number(11.0));
}

/// A imports B and C, B imports C — topological order; constants flow through.
#[test]
fn test_dependency_order_a_b_c() {
    let dir = temp_dcb_test_dir();
    fs::write(dir.join("c.dc"), "let x = 3\n").expect("write c.dc");
    fs::write(dir.join("b.dc"), "import c\nlet y = c.x + 1\n").expect("write b.dc");
    fs::write(dir.join("a.dc"), "import b\nimport c\nlet z = b.y + c.x\n").expect("write a.dc");
    let main_source = "import a\na.z\n";
    let result = run_with_base_path(main_source, &dir);
    assert!(result.is_ok(), "A->B,C and B->C should run: {:?}", result);
    assert_eq!(result.unwrap(), Value::Number(7.0)); // b.y=4, c.x=3 => z=7
}
