//! Minimal native module for DataCode integration tests (`tests/test_lib_native_math.rs`).
//!
//! Implemented without `define_module_entry!` so static `AbiExport` tables stay compatible
//! with strict `Sync` rules on `static` in this toolchain (see `datacode_sdk` macros).

use std::ffi::CString;
use std::os::raw::c_char;
use std::sync::Once;

use datacode_sdk::abi::{AbiExport, AbiModuleDescriptor, AbiVersion};

fn add(args: &[datacode_sdk::abi::AbiValue]) -> datacode_sdk::abi::AbiValue {
    use datacode_sdk::types::{abi_int, get_int};
    let a = get_int(args, 0).unwrap_or(0);
    let b = get_int(args, 1).unwrap_or(0);
    abi_int(a + b)
}

fn mul(args: &[datacode_sdk::abi::AbiValue]) -> datacode_sdk::abi::AbiValue {
    use datacode_sdk::types::{abi_int, get_int};
    let a = get_int(args, 0).unwrap_or(0);
    let b = get_int(args, 1).unwrap_or(0);
    abi_int(a * b)
}

extern "C" fn trampoline_add(
    _ctx: *mut datacode_sdk::abi::VmContext,
    args_ptr: *const datacode_sdk::abi::AbiValue,
    argc: usize,
) -> datacode_sdk::abi::AbiValue {
    let args: &[datacode_sdk::abi::AbiValue] = if args_ptr.is_null() || argc == 0 {
        &[]
    } else {
        unsafe { std::slice::from_raw_parts(args_ptr, argc) }
    };
    add(args)
}

extern "C" fn trampoline_mul(
    _ctx: *mut datacode_sdk::abi::VmContext,
    args_ptr: *const datacode_sdk::abi::AbiValue,
    argc: usize,
) -> datacode_sdk::abi::AbiValue {
    let args: &[datacode_sdk::abi::AbiValue] = if args_ptr.is_null() || argc == 0 {
        &[]
    } else {
        unsafe { std::slice::from_raw_parts(args_ptr, argc) }
    };
    mul(args)
}

static INIT: Once = Once::new();
static mut MODULE_PTR: *const AbiModuleDescriptor = std::ptr::null();

#[no_mangle]
pub extern "C" fn datacode_module_entry() -> *const AbiModuleDescriptor {
    unsafe {
        INIT.call_once(|| {
            let name = CString::new("math_native").expect("module name");
            let name_ptr = name.as_ptr();
            let exports: Vec<AbiExport> = vec![
                AbiExport {
                    name: concat!("add", "\0").as_ptr() as *const c_char,
                    func: trampoline_add,
                    arity: 0,
                    flags: 0,
                },
                AbiExport {
                    name: concat!("mul", "\0").as_ptr() as *const c_char,
                    func: trampoline_mul,
                    arity: 0,
                    flags: 0,
                },
            ];
            let exports_box = exports.into_boxed_slice();
            let exports_ptr = exports_box.as_ptr();
            let exports_len = exports_box.len();
            std::mem::forget(exports_box);
            std::mem::forget(name);

            let desc = Box::new(AbiModuleDescriptor {
                abi_version: AbiVersion { major: 1, minor: 3 },
                name: name_ptr,
                functions: exports_ptr,
                functions_len: exports_len,
                classes: std::ptr::null(),
                classes_len: 0,
                globals: std::ptr::null(),
                globals_len: 0,
            });
            MODULE_PTR = Box::into_raw(desc);
        });
        MODULE_PTR
    }
}
