//! Guards for run() context: restore thread-locals and VM state on drop.
//! RestoreArgvIdGuard stays in vm.rs (needs direct Vm reference).

/// Clears RunContext::SCRIPT_ARGV_VALUE_ID on drop only if this run set it (had argv_patch).
pub struct ClearScriptArgvGuard(pub(crate) bool);
impl Drop for ClearScriptArgvGuard {
    fn drop(&mut self) {
        if self.0 {
            crate::vm::run_context::RunContext::set_script_argv_value_id(None);
        }
    }
}

/// Restores Plot context from thread-local on drop (any exit path from run()).
pub struct PlotContextGuard(pub(crate) *mut Option<crate::plot::PlotContext>);
impl Drop for PlotContextGuard {
    fn drop(&mut self) {
        unsafe {
            *self.0 = crate::plot::PlotContext::take_current();
        }
    }
}

/// Restores VM base_path from RunContext on drop (any exit path from run()).
pub struct RunContextGuard(pub(crate) *mut Option<std::path::PathBuf>);
impl Drop for RunContextGuard {
    fn drop(&mut self) {
        if let Some(ctx) = crate::vm::run_context::RunContext::take_current() {
            unsafe {
                *self.0 = ctx.base_path;
            }
        }
    }
}

/// Closes any open `web.browser` pages left after a run.
pub struct WebBrowserCleanupGuard;
impl Drop for WebBrowserCleanupGuard {
    fn drop(&mut self) {
        crate::web::cleanup_all();
    }
}
