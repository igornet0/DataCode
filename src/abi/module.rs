//! Точка входа нативного модуля и дескриптор модуля.

use std::ffi::c_char;

use crate::abi::version::AbiVersion;
use crate::abi::vm_context::{NativeAbiFn, VmContext};

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct AbiExport {
    pub name: *const c_char,
    pub func: NativeAbiFn,
    pub arity: usize,
    pub flags: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct AbiExportTable {
    pub exports: *const AbiExport,
    pub exports_len: usize,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct AbiClassDescriptor {
    pub name: *const c_char,
    pub methods: *const AbiExport,
    pub methods_len: usize,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct AbiGlobalDescriptor {
    pub name: *const c_char,
    pub getter: NativeAbiFn,
    pub flags: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct AbiModuleDescriptor {
    pub abi_version: AbiVersion,
    pub name: *const c_char,
    pub functions: *const AbiExport,
    pub functions_len: usize,
    pub classes: *const AbiClassDescriptor,
    pub classes_len: usize,
    pub globals: *const AbiGlobalDescriptor,
    pub globals_len: usize,
}

#[repr(C)]
pub struct DatacodeModuleLegacy {
    pub abi_version: AbiVersion,
    pub name: *const c_char,
    pub register: extern "C" fn(*mut VmContext),
}

#[repr(C)]
pub struct DatacodeModule {
    pub abi_version: AbiVersion,
    pub name: *const c_char,
    pub export_table: *const AbiExportTable,
    pub register: Option<extern "C" fn(*mut VmContext)>,
}

pub const DATACODE_MODULE_SYMBOL: &str = "datacode_module";
pub const DATACODE_MODULE_ENTRY_SYMBOL: &str = "datacode_module_entry";

pub type DatacodeModuleFn = extern "C" fn() -> *const DatacodeModule;
pub type DatacodeModuleEntryFn = extern "C" fn() -> *const AbiModuleDescriptor;
