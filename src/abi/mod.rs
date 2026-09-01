//! # datacode-abi (VM boundary)
//!
//! Единый контракт VM ↔ нативные модули: типы, версия, граница FFI.
//! **Источник правды** — крейт [`datacode_abi`] (git submodule: `datacode_abi/` на main, `datacode_sdk/datacode_abi/` на dev).
//! Этот модуль только реэкспортирует его для `data-code`, чтобы не дублировать layout и версию.

pub use datacode_abi::*;
