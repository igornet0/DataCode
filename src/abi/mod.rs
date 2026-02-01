//! # datacode-abi
//!
//! Минимальный, стабильный, C-подобный контракт между Datacode VM и внешними модулями.
//!
//! Не VM, не SDK, не stdlib — только спецификация в коде: типы, версия, граница VM ↔ модуль.
//!
//! Аналоги: Python C-API, Node N-API, Java JNI, Rust `extern "C"`.
//!
//! ## ABI boundary
//!
//! - **Граница:** ABI — контракт между **Datacode VM и нативными плагинами** (внешние модули .so/.dylib).
//!   Не между «Rust и DataCode» как таковыми: VM реализует контракт, плагины компилируются против него.
//! - **Режим:** вызовы по контракту **синхронные** (`extern "C" fn`, без async).
//! - **Стабильность:** версия ABI (major/minor) фиксируется; при несовместимой смене контракта поднимается major.
//!   VM при загрузке проверяет совместимость (`abi_compatible`: same major, module.minor <= vm.minor) и отказывает несовместимым модулям.

pub mod version;
pub mod value;
pub mod error;
pub mod vm_context;
pub mod module;

pub use version::{AbiVersion, DATACODE_ABI_VERSION, abi_compatible};
pub use value::{Value as AbiValue, NativeHandle};
pub use error::DatacodeError;
pub use vm_context::{VmContext, NativeAbiFn};
pub use module::{DatacodeModule, DatacodeModuleFn, DATACODE_MODULE_SYMBOL};
