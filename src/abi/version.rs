//! ABI version — версионирование контракта совместимости.
//!
//! VM проверяет совместимость при загрузке модуля через `abi_compatible(module.abi_version, DATACODE_ABI_VERSION)`.

/// Версия контракта datacode-abi (major.minor).
/// Layout из двух u16 — C/FFI-совместим; для C-модулей заполнять `major` и `minor`.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AbiVersion {
    pub major: u16,
    pub minor: u16,
}

/// Текущая версия контракта datacode-abi, поддерживаемая VM.
/// Менять major при несовместимых изменениях ABI; minor — при обратно-совместимых уточнениях.
/// Minor 4+: `Value::Table` for VM Table → native modules.
/// Minor 5+: `AbiModuleDescriptor` native param metadata, plugin hook names, opaque type tables.
pub const DATACODE_ABI_VERSION: AbiVersion = AbiVersion { major: 1, minor: 5 };

/// Проверка совместимости версии модуля с версией VM.
/// Правило: одинаковый major, у модуля minor <= vm.minor (см. `datacode_abi::DATACODE_ABI_VERSION`).
#[inline]
pub fn abi_compatible(module: &AbiVersion, vm: &AbiVersion) -> bool {
    module.major == vm.major && module.minor <= vm.minor
}
