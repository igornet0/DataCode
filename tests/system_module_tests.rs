//! Smoke tests for the built-in `system` module.

use data_code::{run, PermissionPolicy, Value};
use data_code::vm::permission_policy::is_permission_allowed;

#[test]
fn permission_restricted_denies_unsafe() {
    assert!(!is_permission_allowed(
        PermissionPolicy::Restricted,
        PermissionPolicy::FS_READ,
    ));
    assert!(!is_permission_allowed(
        PermissionPolicy::Restricted,
        PermissionPolicy::FS_WRITE,
    ));
    assert!(!is_permission_allowed(
        PermissionPolicy::Restricted,
        PermissionPolicy::PROCESS_EXEC,
    ));
    assert!(!is_permission_allowed(
        PermissionPolicy::Restricted,
        PermissionPolicy::ENV_WRITE,
    ));
    assert!(is_permission_allowed(
        PermissionPolicy::Restricted,
        "other.capability",
    ));
    assert!(is_permission_allowed(
        PermissionPolicy::AllowAll,
        PermissionPolicy::FS_READ,
    ));
}

#[test]
fn system_env_get_os_returns_known_string() {
    let source = r#"
import system
system.env.get_os()
"#;
    let v = run(source).expect("run");
    match v {
        Value::String(s) => assert!(
            s == "linux" || s == "macos" || s == "windows",
            "unexpected OS: {}",
            s
        ),
        other => panic!("expected String, got {:?}", other),
    }
}

#[test]
fn system_runtime_datacode_version() {
    let source = r#"
import system
system.runtime.get_datacode_version()
"#;
    let v = run(source).expect("run");
    let expected = env!("CARGO_PKG_VERSION");
    match v {
        Value::String(s) => assert_eq!(s, expected),
        other => panic!("expected String, got {:?}", other),
    }
}

#[test]
fn system_hardware_cpu_positive() {
    let source = r#"
import system
system.hardware.cpu_count()
"#;
    let v = run(source).expect("run");
    match v {
        Value::Number(n) => assert!(n >= 1.0, "cpu_count {}", n),
        other => panic!("expected Number, got {:?}", other),
    }
}

#[test]
fn system_time_now_rfc3339() {
    let source = r#"
import system
system.time.now()
"#;
    let v = run(source).expect("run");
    match v {
        Value::String(s) => assert!(
            s.contains('T') || s.len() >= 10,
            "expected ISO-like string: {}",
            s
        ),
        other => panic!("expected String, got {:?}", other),
    }
}

#[test]
fn system_net_get_interfaces_is_array() {
    let source = r#"
import system
system.net.get_interfaces()
"#;
    let v = run(source).expect("run");
    match v {
        Value::Array(_) => {}
        other => panic!("expected Array, got {:?}", other),
    }
}
