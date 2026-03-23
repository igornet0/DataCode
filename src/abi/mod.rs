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
//!
//! ## Descriptor vs `register` (ABI 1.1+)
//!
//! См. зеркальные типы в `datacode_abi`: при `abi_version.minor == 0` используется только
//! [`DatacodeModuleLegacy`] и вызывается `register`. При `minor >= 1` — [`DatacodeModule`]:
//! если `export_table != null`, экспорты читаются только из [`AbiExportTable`]; иначе
//! при непустом `register` — путь с callback. Предпочтительная точка входа — `datacode_module_entry`
//! → [`AbiModuleDescriptor`]. Источник правды: крейт `datacode_abi`.

pub mod version;
pub mod value;
pub mod error;
pub mod vm_context;
pub mod module;

pub use version::{AbiVersion, DATACODE_ABI_VERSION, abi_compatible};
pub use value::{Value as AbiValue, NativeHandle};
pub use error::DatacodeError;
pub use vm_context::{VmContext, NativeAbiFn};
pub use module::{
    AbiClassDescriptor, AbiExport, AbiExportTable, AbiGlobalDescriptor, AbiModuleDescriptor,
    DatacodeModule, DatacodeModuleEntryFn, DatacodeModuleFn, DatacodeModuleLegacy,
    DATACODE_MODULE_ENTRY_SYMBOL, DATACODE_MODULE_SYMBOL,
};
