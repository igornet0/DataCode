//! Builds `tests/test_lib/ml_native` (cdylib `libml`), copies it into DPM-like `packages/ml/`,
//! then smoke-tests `import ml` with PluginOpaque method / layer / `native_plugin_call` paths.

use std::fs;
use std::path::PathBuf;
use std::process::Command;
use std::sync::Mutex;

/// `file_import::set_dpm_package_paths` is process-global (thread_local); run tests that touch it serially.
static DPM_PATHS_TEST_LOCK: Mutex<()> = Mutex::new(());

use data_code::common::value::Value;
use data_code::run_with_base_path;
use data_code::vm::file_import;

fn datacode_sdk_submodule_present() -> bool {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("datacode_sdk")
        .join("Cargo.toml")
        .is_file()
}

fn skip_without_datacode_sdk() -> bool {
    if datacode_sdk_submodule_present() {
        return false;
    }
    eprintln!("skip: datacode_sdk submodule absent (main branch layout)");
    true
}

fn manifest_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("test_lib")
        .join("ml_native")
}

fn packages_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("test_lib")
        .join("fixtures")
        .join("env")
        .join("packages")
}

fn release_dylib_path() -> PathBuf {
    let dir = manifest_dir().join("target").join("release");
    if cfg!(target_os = "macos") {
        dir.join("libml.dylib")
    } else if cfg!(target_os = "windows") {
        dir.join("ml.dll")
    } else {
        dir.join("libml.so")
    }
}

fn target_dir_for_ml_native() -> PathBuf {
    manifest_dir().join("target")
}

fn copy_dylib_to_packages_ml() -> Result<PathBuf, String> {
    let src = release_dylib_path();
    if !src.is_file() {
        return Err(format!("expected dylib at {}", src.display()));
    }
    let dest_dir = packages_root().join("ml");
    fs::create_dir_all(&dest_dir).map_err(|e| e.to_string())?;
    let name = src.file_name().ok_or("dylib has no name")?.to_owned();
    let dest = dest_dir.join(&name);
    fs::copy(&src, &dest).map_err(|e| e.to_string())?;
    Ok(dest)
}

fn build_ml_native_cdylib() -> Result<(), String> {
    let manifest = manifest_dir().join("Cargo.toml");
    let status = Command::new("cargo")
        .arg("build")
        .arg("--release")
        .arg("--manifest-path")
        .arg(&manifest)
        .arg("--target-dir")
        .arg(target_dir_for_ml_native())
        .status()
        .map_err(|e| format!("cargo: {}", e))?;
    if !status.success() {
        return Err("cargo build --release for ml_native (libml) failed".to_string());
    }
    Ok(())
}

fn assert_bool(v: Result<Value, data_code::LangError>, expected: bool) {
    match v {
        Ok(Value::Bool(b)) => assert_eq!(b, expected, "expected bool {}", expected),
        Ok(o) => panic!("expected Bool, got {:?}", o),
        Err(e) => panic!("unexpected error: {:?}", e),
    }
}

fn assert_number(v: Result<Value, data_code::LangError>, n: f64) {
    match v {
        Ok(Value::Number(x)) => assert!((x - n).abs() < 1e-9, "expected number {}, got {}", n, x),
        Ok(o) => panic!("expected Number, got {:?}", o),
        Err(e) => panic!("unexpected error: {:?}", e),
    }
}

#[test]
fn vm_smoke_without_ml_native() {
    let _g = DPM_PATHS_TEST_LOCK.lock().expect("lock");
    let tmp = tempfile::tempdir().expect("tempdir");
    let base = tmp.path();
    let r = run_with_base_path("2 + 2", base);
    assert_number(r, 4.0);
}

#[test]
#[ignore = "datacode_sdk submodule is not present in the main branch layout"]
fn native_ml_module_plugin_opaque_smoke() {
    if skip_without_datacode_sdk() {
        return;
    }
    let _g = DPM_PATHS_TEST_LOCK.lock().expect("lock");
    build_ml_native_cdylib().expect("build ml cdylib");
    let copied = copy_dylib_to_packages_ml().expect("copy dylib to packages/ml");
    assert!(copied.is_file());

    let tmp = tempfile::tempdir().expect("tempdir");
    let base = tmp.path();

    file_import::set_dpm_package_paths(vec![packages_root()]);
    let source = r#"
import ml
model = ml.neural_network(3)
model.device("cpu")
d = model.get_device()
layer0 = model.layers[0]
layer0.freeze()
layer0.unfreeze()
loss = model.train(1)
d == "cpu" and loss[0] == 0.42
"#;
    let result = run_with_base_path(source, base);
    file_import::set_dpm_package_paths(vec![]);

    assert_bool(result, true);
}
