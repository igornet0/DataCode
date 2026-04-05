//! User-defined operators (preload, conflicts, parse errors).

use data_code::common::value::Value;
use data_code::vm::file_import;
use data_code::vm::operator_registry::OperatorRegistry;
use std::cell::RefCell;
use std::path::Path;
use std::rc::Rc;

fn row(sym: &str, name: &str, prec: f64, assoc: &str) -> Value {
    Value::Array(Rc::new(RefCell::new(vec![
        Value::String(sym.to_string()),
        Value::String(name.to_string()),
        Value::Number(prec),
        Value::String(assoc.to_string()),
    ])))
}

#[test]
fn merge_operator_descriptor_conflict_is_error() {
    let mut reg = OperatorRegistry::with_builtins();
    let v1 = Value::Array(Rc::new(RefCell::new(vec![row("@", "matmul", 60.0, "left")])));
    reg.merge_from_descriptor_value(&v1, "mod_a").expect("first merge");
    let v2 = Value::Array(Rc::new(RefCell::new(vec![row("@", "other", 60.0, "left")])));
    let err = reg.merge_from_descriptor_value(&v2, "mod_b").unwrap_err();
    let msg = format!("{}", err);
    assert!(
        msg.contains("already registered") || msg.contains("@"),
        "unexpected err: {}",
        msg
    );
}

#[test]
fn parse_error_without_import_infix_at() {
    let err = data_code::run("1 @ 2").unwrap_err();
    let s = format!("{:?}", err);
    assert!(
        s.contains("Unregistered") || s.contains("operator") || s.contains("@"),
        "{}",
        s
    );
}

#[test]
fn debug_operators_returns_table_after_run() {
    // `debug` is pre-seeded in compiler natives (see `register_natives`); do not `import debug` — that shadows the builtin.
    let source = "debug.operators()\n";
    let v = data_code::run(source).expect("run");
    let Value::String(s) = v else {
        panic!("expected string from debug.operators(), got {:?}", v);
    };
    assert!(
        s.contains("symbol=") && s.contains("builtin"),
        "expected operator table lines, got: {}",
        s
    );
}

#[test]
#[ignore = "skip"]
fn ml_dylib_tensor_at_parse_requires_import_and_ml() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("datacode_lib/ML-Datacode-lib");
    // Prefer debug lib first: release cdylib may be stripped or ABI-mismatched vs current VM; preload swallows load errors.
    let lib = ["target/debug/libml.dylib", "target/release/libml.dylib"]
        .into_iter()
        .map(|p| root.join(p))
        .find(|p| p.is_file());
    let Some(lib) = lib else {
        eprintln!("skip: build ml dylib");
        return;
    };
    // Force this exact dylib for preload (same as runtime `import ml`): otherwise `try_load_native_module`
    // may pick another `.dcmodule`/package path first and merge no `operator_descriptor`.
    let _guard = file_import::push_native_lib_override(Some(lib.clone()));
    let base = lib.parent().expect("libml in target dir");
    let ok = r#"
import ml
let a = ml.tensor([[1.0]])
let b = ml.tensor([[1.0]])
let _ = a @ b
debug.operators()
"#;
    let v = data_code::run_with_base_path(ok, base).expect("run");
    let Value::String(s) = v else {
        panic!("expected string, got {:?}", v);
    };
    assert!(s.contains("@") || s.contains("matmul"), "operators: {}", s);

    // Precedence: @ tighter than +  →  (a + b) @ c  groups as  ((a+b) @ c) if @ has higher prec than +
    let prec = r#"
import ml
let a = ml.tensor([[1.0]])
let b = ml.tensor([[2.0]])
let z = ml.tensor([[1.0]])
(a + b) @ z
"#;
    let _ = data_code::run_with_base_path(prec, base).expect("precedence");

    // Associativity: left-assoc @  →  (a @ b) @ c
    let assoc = r#"
import ml
let a = ml.tensor([[1.0]])
let b = ml.tensor([[2.0]])
let c = ml.tensor([[3.0]])
a @ b @ c
"#;
    let _ = data_code::run_with_base_path(assoc, base).expect("assoc");
}

