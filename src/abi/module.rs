//! Точка входа нативного модуля и дескриптор модуля.

use std::ffi::c_char;

use crate::abi::version::AbiVersion;
use crate::abi::vm_context::VmContext;

/// Дескриптор модуля, возвращаемый из `datacode_module()`.
#[repr(C)]
pub struct DatacodeModule {
    /// Версия ABI модуля; должна быть совместима с VM (см. `abi_compatible`).
    pub abi_version: AbiVersion,
    /// Имя модуля (UTF-8, null-terminated), например "telegram".
    pub name: *const c_char,
    /// VM вызывает эту функцию после проверки ABI; модуль регистрирует функции/классы/константы.
    pub register: extern "C" fn(*mut VmContext),
}

/// Имя символа точки входа нативного модуля.
pub const DATACODE_MODULE_SYMBOL: &str = "datacode_module";

/// Сигнатура точки входа: модуль экспортирует функцию с этим именем.
/// VM загружает .so/.dylib, вызывает `datacode_module()`, проверяет
/// `abi_compatible(&(*module).abi_version, &DATACODE_ABI_VERSION)`, затем вызывает
/// `(*module).register(&mut vm_context)`.
///
/// Реализация — в коде модуля, не в VM.
pub type DatacodeModuleFn = extern "C" fn() -> *const DatacodeModule;
