//! Thread-local execution context for the VM on the current OS thread.
//!
//! Natives and interpreter helpers read the active [`crate::vm::vm::Vm`] from here. Future fields
//! (script path, import roots, session id) can be added without new `thread_local!` keys.

use crate::vm::vm::Vm;
use std::cell::RefCell;

/// Active VM pointer plus room for future embedder-facing metadata.
#[derive(Clone, Copy)]
pub struct VmExecutionContext {
    pub vm: *mut Vm,
}

thread_local! {
    pub(crate) static VM_CALL_CONTEXT: RefCell<Option<VmExecutionContext>> = RefCell::new(None);
}

#[inline]
pub(crate) fn current_vm_ptr() -> Option<*mut Vm> {
    VM_CALL_CONTEXT.with(|c| c.borrow().as_ref().map(|ctx| ctx.vm))
}
