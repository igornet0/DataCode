//! Builds the `tests/test_lib/math_native` cdylib, copies it into a DPM-like `packages/<name>/`
//! layout, runs DataCode `import`, and checks the VM still runs plain code.

use std::fs;
use std::path::PathBuf;
use std::process::Command;
use std::sync::Mutex;

/// `file_import::set_dpm_package_paths` is process-global (thread_local); run tests that touch it serially.
static DPM_PATHS_TEST_LOCK: Mutex<()> = Mutex::new(());

use data_code::common::value::Value;
use data_code::run_with_base_path;
use data_code::vm::file_import;
use data_code::vm::module_object::BUILTIN_END;
use data_code::vm::native_loader::try_load_native_module;
use libloading::Library;

fn manifest_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("test_lib")
        .join("math_native")
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
        dir.join("libmath_native.dylib")
    } else if cfg!(target_os = "windows") {
        dir.join("math_native.dll")
    } else {
        dir.join("libmath_native.so")
    }
}

fn target_dir_for_math_native() -> PathBuf {
    manifest_dir().join("target")
}

fn copy_dylib_to_packages_layout() -> Result<PathBuf, String> {
    let src = release_dylib_path();
    if !src.is_file() {
        return Err(format!("expected dylib at {}", src.display()));
    }
    let dest_dir = packages_root().join("math_native");
    fs::create_dir_all(&dest_dir).map_err(|e| e.to_string())?;
    let name = src.file_name().ok_or("dylib has no name")?.to_owned();
    let dest = dest_dir.join(&name);
    fs::copy(&src, &dest).map_err(|e| e.to_string())?;
    Ok(dest)
}

fn build_math_native_cdylib() -> Result<(), String> {
    let manifest = manifest_dir().join("Cargo.toml");
    let status = Command::new("cargo")
        .arg("build")
        .arg("--release")
        .arg("--manifest-path")
        .arg(&manifest)
        .arg("--target-dir")
        .arg(target_dir_for_math_native())
        .status()
        .map_err(|e| format!("cargo: {}", e))?;
    if !status.success() {
        return Err("cargo build --release for math_native failed".to_string());
    }
    Ok(())
}

#[test]
fn try_load_native_module_direct_smoke() {
    let _g = DPM_PATHS_TEST_LOCK.lock().expect("lock");
    build_math_native_cdylib().expect("build");
    let tmp = tempfile::tempdir().expect("tempdir");
    let base = tmp.path();
    let src = release_dylib_path();
    fs::copy(&src, base.join(src.file_name().unwrap())).expect("copy");
    let mut abi_natives = Vec::new();
    let mut loaded_libs = Vec::new();
    let r = try_load_native_module(
        "math_native",
        Some(base),
        BUILTIN_END,
        &mut abi_natives,
        &mut loaded_libs,
        None,
    );
    assert!(
        r.is_ok(),
        "try_load_native_module: {:?}",
        r.as_ref().err()
    );
}

fn assert_number(v: Result<Value, data_code::LangError>, n: f64) {
    match v {
        Ok(Value::Number(x)) => assert!(
            (x - n).abs() < 1e-9,
            "expected number {}, got {}",
            n,
            x
        ),
        Ok(o) => panic!("expected Number, got {:?}", o),
        Err(e) => panic!("unexpected error: {:?}", e),
    }
}

#[test]
fn vm_smoke_without_native_module_path() {
    let _g = DPM_PATHS_TEST_LOCK.lock().expect("lock");
    let tmp = tempfile::tempdir().expect("tempdir");
    let base = tmp.path();
    let r = run_with_base_path("2 + 2", base);
    assert_number(r, 4.0);
}

#[test]
fn native_math_module_via_dpm_packages_layout() {
    let _g = DPM_PATHS_TEST_LOCK.lock().expect("lock");
    build_math_native_cdylib().expect("build math_native cdylib");
    let copied = copy_dylib_to_packages_layout().expect("copy dylib to packages layout");
    assert!(copied.is_file());

    let tmp = tempfile::tempdir().expect("tempdir");
    let base = tmp.path();

    file_import::set_dpm_package_paths(vec![packages_root()]);
    let source = r#"
from math_native import add, mul
add(2, 3) + mul(2, 5)
"#;
    let result = run_with_base_path(source, base);
    file_import::set_dpm_package_paths(vec![]);

    assert_number(result, 15.0);
}

#[test]
fn native_math_module_iterable_arg_materializes_to_array() {
    let _g = DPM_PATHS_TEST_LOCK.lock().expect("lock");
    build_math_native_cdylib().expect("build math_native cdylib");
    let copied = copy_dylib_to_packages_layout().expect("copy dylib to packages layout");
    assert!(copied.is_file());

    let tmp = tempfile::tempdir().expect("tempdir");
    let base = tmp.path();

    file_import::set_dpm_package_paths(vec![packages_root()]);
    let source = r#"
from math_native import sum_array
sum_array(map([1, 2, 3], fn(x) => x * 2))
"#;
    let result = run_with_base_path(source, base);
    file_import::set_dpm_package_paths(vec![]);

    assert_number(result, 12.0);
}

#[test]
fn native_math_module_loose_dylib_next_to_base() {
    let _g = DPM_PATHS_TEST_LOCK.lock().expect("lock");
    build_math_native_cdylib().expect("build math_native cdylib");
    let src = release_dylib_path();
    assert!(src.is_file(), "dylib missing at {}", src.display());
    unsafe { Library::new(&src) }.expect("dlopen built math_native (sanity)");

    let tmp = tempfile::tempdir().expect("tempdir");
    let base = tmp.path();
    let dest = base.join(src.file_name().unwrap());
    fs::copy(&src, &dest).expect("copy dylib beside base path");
    assert!(dest.is_file(), "copy failed: {:?}", dest);
    unsafe { Library::new(&dest) }.expect("dlopen copied dylib beside base_path");

    let source = r#"
from math_native import add
add(10, -3)
"#;
    let result = run_with_base_path(source, base);
    assert_number(result, 7.0);
}
