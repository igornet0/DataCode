//! Коды ошибок на границе VM ↔ модуль.
//!
//! VM переводит их в try/catch и типы ошибок Datacode.
//! Модуль сообщает об ошибке через VmContext::throw_error(code, msg).

/// Коды ошибок, передаваемые из модуля в VM.
/// VM маппит их на внутренние типы (LangError / try-catch).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DatacodeError {
    Ok = 0,
    TypeError = 1,
    RuntimeError = 2,
    Panic = 3,
}
