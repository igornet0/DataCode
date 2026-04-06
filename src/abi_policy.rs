//! Central ABI compatibility checks against the running VM host.
//! Use this from `.dcmodule` manifest validation and native module loading so messages stay consistent.

use crate::abi::{abi_compatible, AbiVersion, DATACODE_ABI_VERSION};

/// Host ABI version embedded in this build (re-exported for diagnostics).
pub fn host_abi_version() -> &'static AbiVersion {
    &DATACODE_ABI_VERSION
}

/// Returns `Err` if `module` is not compatible with [`DATACODE_ABI_VERSION`].
pub fn ensure_module_abi_compatible(module: &AbiVersion) -> Result<(), String> {
    if !abi_compatible(module, &DATACODE_ABI_VERSION) {
        return Err(format!(
            "abi_version {}.{} is not compatible with VM {}.{}",
            module.major,
            module.minor,
            DATACODE_ABI_VERSION.major,
            DATACODE_ABI_VERSION.minor
        ));
    }
    Ok(())
}
