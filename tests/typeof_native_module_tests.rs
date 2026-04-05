//! typeof() for objects with __plugin_namespace (native module root / ml.layer).

use data_code::common::value::Value;
use data_code::common::value_store::ValueStore;
use data_code::vm::heavy_store::HeavyStore;
use data_code::vm::module_object::BUILTIN_END;
use data_code::extract_globals_from_vm;
use data_code::vm::file_import;
use data_code::vm::native_loader::try_load_native_module;
use data_code::vm::natives::basic::native_typeof;
use data_code::vm::store_convert::{load_value, store_value};
use std::cell::RefCell;
use std::collections::HashMap;
use std::path::Path;
use std::rc::Rc;

#[test]
fn typeof_native_module_after_store_roundtrip() {
    let mut hm = HashMap::new();
    hm.insert(
        "__plugin_namespace".to_string(),
        Value::String("module".to_string()),
    );
    hm.insert("x".to_string(), Value::Number(1.0));
    let v = Value::Object(Rc::new(RefCell::new(hm)));
    let mut vs = ValueStore::new();
    let mut hs = HeavyStore::new();
    let id = store_value(v, &mut vs, &mut hs);
    let v2 = load_value(id, &vs, &hs);
    let out = native_typeof(&[v2]);
    assert_eq!(out, Value::String("module".to_string()));
}

#[test]
fn try_load_ml_hashmap_roundtrip_typeof_is_module() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("datacode_lib/ML-Datacode-lib");
    let debug_lib = root.join("target/debug/libml.dylib");
    let release_lib = root.join("target/release/libml.dylib");
    let base = if debug_lib.is_file() {
        debug_lib.parent().unwrap()
    } else if release_lib.is_file() {
        release_lib.parent().unwrap()
    } else {
        return;
    };
    let mut abi = Vec::new();
    let mut libs = Vec::new();
    let (m, _) = try_load_native_module("ml", Some(base), BUILTIN_END, &mut abi, &mut libs, None)
        .expect("try_load ml");
    let mut vs = ValueStore::new();
    let mut hs = HeavyStore::new();
    let id = store_value(
        Value::Object(Rc::new(RefCell::new(m))),
        &mut vs,
        &mut hs,
    );
    let v = load_value(id, &vs, &hs);
    let out = native_typeof(&[v]);
    assert_eq!(out, Value::String("module".to_string()));
}

#[test]
fn import_ml_via_vm_keeps_plugin_namespace_on_global() {
    let lib = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("datacode_lib/ML-Datacode-lib/target/release/libml.dylib");
    if !lib.is_file() {
        return;
    }
    let tmp = tempfile::tempdir().expect("tempdir");
    let base = tmp.path();
    let _guard = file_import::push_native_lib_override(Some(lib));
    let source = "import ml\n";
    let (_, mut vm) = data_code::run_with_vm_and_path(source, Some(base), None).expect("run");
    let gm = extract_globals_from_vm(&mut vm);
    let ml = gm.get("ml").expect("global ml");
    let Value::Object(rc) = ml else {
        panic!("ml should be Object, got {:?}", ml);
    };
    assert!(
        rc.borrow().get("__plugin_namespace").is_some(),
        "ml global keys: {:?}",
        rc.borrow().keys().collect::<Vec<_>>()
    );
}

#[test]
fn isinstance_ml_dataset_true_when_plugin_opaque_matches_namespace() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("datacode_lib/ML-Datacode-lib");
    let debug_lib = root.join("target/debug/libml.dylib");
    let release_lib = root.join("target/release/libml.dylib");
    let lib = if debug_lib.is_file() {
        debug_lib
    } else if release_lib.is_file() {
        release_lib
    } else {
        return;
    };
    let tmp = tempfile::tempdir().expect("tempdir");
    let base = tmp.path();
    let _guard = file_import::push_native_lib_override(Some(lib));
    let source = r#"
import ml
let X = ml.tensor([1.0, 2.0, 3.0, 4.0], [2, 2])
let y = ml.tensor([0.0, 1.0], [2, 1])
let ds = ml.dataset(X, y)
isinstance(ds, ml.dataset)
"#;
    let (v, _) = data_code::run_with_vm_and_path(source, Some(base), None).expect("run");
    assert_eq!(v, Value::Bool(true));
}

#[test]
fn typeof_ml_global_after_import_is_module_string() {
    let lib = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("datacode_lib/ML-Datacode-lib/target/release/libml.dylib");
    if !lib.is_file() {
        return;
    }
    let tmp = tempfile::tempdir().expect("tempdir");
    let base = tmp.path();
    let _guard = file_import::push_native_lib_override(Some(lib));
    let source = "import ml\nv = typeof(ml)\n";
    let (_, mut vm) = data_code::run_with_vm_and_path(source, Some(base), None).expect("run");
    let gm = extract_globals_from_vm(&mut vm);
    let v = gm.get("v").expect("v");
    assert_eq!(
        v,
        &Value::String("module".to_string()),
        "typeof(ml) should be module, got {:?}",
        v
    );
}
