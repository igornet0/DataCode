//! Универсальный тип значений на границе VM ↔ модуль.
//!
//! Единственный тип, который VM понимает извне. `#[repr(C)]` обязателен для FFI.

use std::ffi::c_char;
use std::ffi::c_void;

/// Handle непрозрачного объекта (словарь/объект), управляемого VM или модулем.
pub type NativeHandle = *mut c_void;

/// Значение в формате ABI. Все указатели и строки — с точки зрения модуля:
/// строки, переданные из VM, действительны на время вызова нативной функции.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub enum Value {
    Int(i64),
    Float(f64),
    Bool(bool),
    /// UTF-8, null-terminated. Владение не передаётся.
    Str(*const c_char),
    Null,
    /// Массив элементов. Указатель действителен на время вызова.
    Array(*mut Value, usize),
    /// Непрозрачный handle объекта (словарь в VM).
    Object(NativeHandle),
}
