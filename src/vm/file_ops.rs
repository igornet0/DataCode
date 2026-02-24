// Модуль для работы с файловыми операциями и SMB
//
// SMB_MANAGER is thread-local RefCell<Option<Arc<Mutex<SmbManager>>>>. Lock is held only for
// the duration of a single SMB op (e.g. read_file/list_files). See docs/gil_bottlenecks.md.

use std::sync::{Arc, Mutex};
use crate::websocket::smb::SmbManager;

thread_local! {
    static SMB_MANAGER: std::cell::RefCell<Option<Arc<Mutex<SmbManager>>>> = std::cell::RefCell::new(None);
}

/// Установить SmbManager для текущего потока
pub fn set_smb_manager(manager: Arc<Mutex<SmbManager>>) {
    SMB_MANAGER.with(|m| {
        *m.borrow_mut() = Some(manager);
    });
}

/// Очистить SmbManager для текущего потока
pub fn clear_smb_manager() {
    SMB_MANAGER.with(|m| {
        *m.borrow_mut() = None;
    });
}

/// Получить SmbManager: из RunContext во время run() (предпочтительно), иначе из thread_local.
pub fn get_smb_manager() -> Option<Arc<Mutex<SmbManager>>> {
    if crate::vm::run_context::RunContext::is_set() {
        if let Some(m) = crate::vm::run_context::RunContext::get_smb_manager() {
            return Some(m);
        }
    }
    SMB_MANAGER.with(|m| m.borrow().clone())
}

