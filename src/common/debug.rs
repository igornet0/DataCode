// Debug mode management

use std::sync::atomic::{AtomicBool, Ordering};

static DEBUG_ENABLED: AtomicBool = AtomicBool::new(false);

/// Enable debug mode
pub fn set_debug(enabled: bool) {
    DEBUG_ENABLED.store(enabled, Ordering::Relaxed);
}

/// Check if debug mode is enabled
pub fn is_debug_enabled() -> bool {
    DEBUG_ENABLED.load(Ordering::Relaxed)
}

/// True when constructor/merge debug logs should be printed (only when env DATACODE_DEBUG is set).
pub fn verbose_constructor_debug() -> bool {
    std::env::var("DATACODE_DEBUG").is_ok()
}

/// Print debug message if debug mode is enabled
#[macro_export]
macro_rules! debug_println {
    ($($arg:tt)*) => {
        if $crate::common::debug::is_debug_enabled() {
            eprintln!($($arg)*);
        }
    };
}





