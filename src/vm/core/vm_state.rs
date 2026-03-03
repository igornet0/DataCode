// VmState: execution state only (stack, frames).
// Used by interpreter; Phase 7 will refactor Vm to delegate to VmState.

use crate::common::TaggedValue;

use super::frame::CallFrame;

/// Execution state of the VM — stack and call frames.
/// No bytecode logic, no module cache, no value store.
#[derive(Default)]
pub struct VmState {
    pub stack: Vec<TaggedValue>,
    pub frames: Vec<CallFrame>,
}

impl VmState {
    pub fn new() -> Self {
        Self::default()
    }
}
