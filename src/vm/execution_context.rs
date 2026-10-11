//! Thread-local execution context for the VM on the current OS thread.
//!
//! Natives and interpreter helpers read the active [`crate::vm::vm::Vm`] from here. Future fields
//! (script path, import roots, session id) can be added without new `thread_local!` keys.

use crate::vm::vm::Vm;
use std::cell::Cell;

/// Active VM pointer plus room for future embedder-facing metadata.
#[derive(Clone, Copy)]
pub struct VmExecutionContext {
    pub vm: *mut Vm,
}

// `Cell` + `const` init: read on every stack push/pop, so no borrow flag and no lazy-init check.
thread_local! {
    pub(crate) static VM_CALL_CONTEXT: Cell<Option<VmExecutionContext>> = const { Cell::new(None) };
}

#[inline]
pub(crate) fn current_vm_ptr() -> Option<*mut Vm> {
    VM_CALL_CONTEXT.with(|c| c.get().map(|ctx| ctx.vm))
}

/// Nested native/table helpers: restore [`VM_CALL_CONTEXT`] on drop (do not set to `None` mid-opcode).
pub(crate) struct RestoreVmCallContextGuard {
    previous: Option<VmExecutionContext>,
}

impl RestoreVmCallContextGuard {
    pub(crate) fn push(vm_ptr: *mut Vm) -> Self {
        let previous = VM_CALL_CONTEXT.with(|ctx| {
            let prev = ctx.get();
            ctx.set(Some(VmExecutionContext { vm: vm_ptr }));
            prev
        });
        Self { previous }
    }
}

impl Drop for RestoreVmCallContextGuard {
    fn drop(&mut self) {
        VM_CALL_CONTEXT.with(|ctx| {
            ctx.set(self.previous);
        });
    }
}
