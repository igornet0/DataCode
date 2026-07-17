//! Lightweight architecture checks (no new magic native indices drift, ABI policy present).

use data_code::abi_policy;
use data_code::vm::native_indices;

#[test]
fn host_abi_version_is_nonzero() {
    let v = abi_policy::host_abi_version();
    assert!(v.major > 0 || v.minor > 0);
}

#[test]
fn abi_policy_rejects_absurd_version() {
    use data_code::abi::AbiVersion;
    let ancient = AbiVersion {
        major: 65535,
        minor: 65535,
    };
    assert!(abi_policy::ensure_module_abi_compatible(&ancient).is_err());
}

#[test]
fn builtin_native_indices_stable_entrypoints() {
    assert_eq!(native_indices::builtin::PRINT, 0);
    assert_eq!(native_indices::builtin::RANGE, 2);
    assert_eq!(native_indices::builtin::LEN, 1);
    assert_eq!(native_indices::builtin::PUSH, 41);
    assert!(native_indices::builtin::PRINT < native_indices::builtin::LEN);
}
