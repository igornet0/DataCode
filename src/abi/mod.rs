//! # datacode-abi (VM boundary)
//!
//! Единый контракт VM ↔ нативные модули: типы, версия, граница FFI.
//! **Источник правды** — крейт [`datacode_abi`] в `datacode_sdk/datacode_abi`.
//! Этот модуль только реэкспортирует его для `data-code`, чтобы не дублировать layout и версию.

pub use datacode_abi::*;
