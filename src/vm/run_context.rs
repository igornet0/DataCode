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
    /// Root directory of the project (entry script dir). Never overwritten; used for absolute imports.
    pub project_root: Option<PathBuf>,
    pub executing_lib: bool,
    pub dpm_package_paths: Vec<PathBuf>,
    /// SMB manager for this run (set from thread_local at run() start to avoid global RefCell on each op).
    pub smb_manager: Option<Arc<Mutex<crate::websocket::smb::SmbManager>>>,
    /// Canonical argv value id for this run (script args); set when run() is called with argv_patch.
    pub argv_value_id: Option<crate::common::value_store::ValueId>,
}

thread_local! {
    static RUN_CONTEXT: RefCell<Option<RunContext>> = RefCell::new(None);
    /// Script argv value id set at run() start when argv_patch is present; not overwritten by nested module run().
    static SCRIPT_ARGV_VALUE_ID: RefCell<Option<crate::common::value_store::ValueId>> = RefCell::new(None);
    /// Set by executor when re-establishing argv slot after ImportFrom; used by LoadGlobal(argv) in main so script args are correct.
    static RESTORED_SCRIPT_ARGV_AFTER_IMPORT: RefCell<Option<crate::common::value_store::ValueId>> = RefCell::new(None);
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

    /// Get project_root from current context if set; otherwise None.
    pub fn get_project_root() -> Option<PathBuf> {
        RUN_CONTEXT.with(|cell| cell.borrow().as_ref().map(|r| r.project_root.clone()).flatten())
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

    /// Get canonical argv value id for this run (script args) if set.
    /// Prefers RunContext; if that was cleared by nested module run(), uses SCRIPT_ARGV_VALUE_ID set at outer run() start.
    pub fn get_argv_value_id() -> Option<crate::common::value_store::ValueId> {
        RUN_CONTEXT
            .with(|cell| cell.borrow().as_ref().and_then(|r| r.argv_value_id))
            .or_else(|| SCRIPT_ARGV_VALUE_ID.with(|cell| *cell.borrow()))
    }

    /// Set the script argv value id for the current run (called by VM::run() when argv_patch is present).
    /// Survives nested module run() which overwrites RunContext.
    pub fn set_script_argv_value_id(id: Option<crate::common::value_store::ValueId>) {
        SCRIPT_ARGV_VALUE_ID.with(|cell| *cell.borrow_mut() = id);
    }

    /// Get script argv value id only from thread-local (not from RunContext). Used when loading argv in main chunk after ImportFrom.
    pub fn get_script_argv_value_id() -> Option<crate::common::value_store::ValueId> {
        SCRIPT_ARGV_VALUE_ID.with(|cell| *cell.borrow())
    }

    /// Set the script argv id that was restored after ImportFrom (executor sets this when re-establishing the argv slot).
    pub fn set_restored_script_argv_after_import(id: Option<crate::common::value_store::ValueId>) {
        RESTORED_SCRIPT_ARGV_AFTER_IMPORT.with(|cell| *cell.borrow_mut() = id);
    }

    /// Get the script argv id restored after ImportFrom; use this when loading argv in main chunk so script args are correct.
    pub fn get_restored_script_argv_after_import() -> Option<crate::common::value_store::ValueId> {
        RESTORED_SCRIPT_ARGV_AFTER_IMPORT.with(|cell| *cell.borrow())
    }
}
