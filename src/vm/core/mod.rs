// VM Core: execution state (stack, frames).
// No bytecode logic — only push/pop, frame access.

pub mod frame;
pub mod stack;
pub mod vm_state;

pub use frame::CallFrame;
pub use frame::ForRangeState;
pub use stack::{peek, pop, push, push_id};
pub use vm_state::VmState;
