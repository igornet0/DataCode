//! Единственный мост модуля к VM.
//!
//! Модуль не имеет прямого доступа к VM: только alloc, throw_error и register_native.

use std::ffi::c_char;

use crate::abi::error::DatacodeError;
use crate::abi::value::Value as AbiValue;

/// Сигнатура нативной функции, экспортируемой модулем по ABI.
pub type NativeAbiFn = extern "C" fn(*mut VmContext, *const AbiValue, usize) -> AbiValue;

/// Контекст, передаваемый в `register`. Модуль использует только эти коллбэки.
#[repr(C)]
pub struct VmContext {
    /// Выделить память через аллокатор VM. Возвращает null при ошибке.
    pub alloc: extern "C" fn(size: usize) -> *mut u8,
    /// Сообщить VM об ошибке; VM переведёт в try/catch Datacode.
    /// `msg` — UTF-8, null-terminated; может быть null.
    pub throw_error: extern "C" fn(code: DatacodeError, msg: *const c_char),
    /// Зарегистрировать нативную функцию в VM. Вызывается модулем из register().
    pub register_native: extern "C" fn(*mut VmContext, name: *const c_char, func: NativeAbiFn),
}
