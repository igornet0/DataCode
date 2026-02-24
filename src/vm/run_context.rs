// Single thread-local context for a VM run (replaces multiple RefCells for base_path, executing_lib, dpm_package_paths).
// Set at run() start, taken at run() end. See docs/gil_bottlenecks.md.
// SMB manager is read from thread_local at run() start so file_ops prefer RunContext (no global RefCell on each SMB op).

use std::cell::RefCell;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

/// Context installed for the duration of VM::run(). Holds path/config and optional SMB manager.
#[derive(Clone, Default)]
pub struct RunContext {
    pub base_path: Option<PathBuf>,
    pub executing_lib: bool,
    pub dpm_package_paths: Vec<PathBuf>,
    /// SMB manager for this run (set from thread_local at run() start to avoid global RefCell on each op).
    pub smb_manager: Option<Arc<Mutex<crate::websocket::smb::SmbManager>>>,
}

thread_local! {
    static RUN_CONTEXT: RefCell<Option<RunContext>> = RefCell::new(None);
}

impl RunContext {
    /// Install context for the current thread (called by VM at run() start).
    pub fn set_current(ctx: RunContext) {
        RUN_CONTEXT.with(|cell| *cell.borrow_mut() = Some(ctx));
    }

    /// Take context back from the current thread (called by VM at run() end).
    pub fn take_current() -> Option<RunContext> {
        RUN_CONTEXT.with(|cell| cell.borrow_mut().take())
    }

    /// Run closure with the current RunContext (panics if not set).
    pub fn with_current<F, R>(f: F) -> R
    where
        F: FnOnce(&mut RunContext) -> R,
    {
        RUN_CONTEXT.with(|cell| {
            let mut borrow = cell.borrow_mut();
            let ctx = borrow.as_mut().expect("RunContext not set. VM must set context before run().");
            f(ctx)
        })
    }

    /// Run closure with the current RunContext if set; otherwise do nothing.
    pub fn with_current_opt<F>(f: F)
    where
        F: FnOnce(&mut RunContext),
    {
        RUN_CONTEXT.with(|cell| {
            if let Some(ref mut ctx) = *cell.borrow_mut() {
                f(ctx);
            }
        });
    }

    /// Returns true if RunContext is set (e.g. during run()).
    pub fn is_set() -> bool {
        RUN_CONTEXT.with(|cell| cell.borrow().is_some())
    }

    /// Get base_path from current context if set; otherwise None.
    pub fn get_base_path() -> Option<PathBuf> {
        RUN_CONTEXT.with(|cell| cell.borrow().as_ref().map(|r| r.base_path.clone()).flatten())
    }

    /// Get executing_lib from current context if set; otherwise false.
    pub fn get_executing_lib() -> bool {
        RUN_CONTEXT.with(|cell| cell.borrow().as_ref().map(|r| r.executing_lib).unwrap_or(false))
    }

    /// Get dpm_package_paths from current context if set; otherwise empty vec.
    pub fn get_dpm_package_paths() -> Vec<PathBuf> {
        RUN_CONTEXT.with(|cell| cell.borrow().as_ref().map(|r| r.dpm_package_paths.clone()).unwrap_or_default())
    }

    /// Get SMB manager from current context if set (preferred over thread_local during run()).
    pub fn get_smb_manager() -> Option<Arc<Mutex<crate::websocket::smb::SmbManager>>> {
        RUN_CONTEXT.with(|cell| cell.borrow().as_ref().and_then(|r| r.smb_manager.clone()))
    }
}
