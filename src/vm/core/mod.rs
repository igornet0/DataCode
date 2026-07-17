// VM Core: execution state (stack, frames).
// No bytecode logic — only push/pop, frame access.

pub mod frame;
pub mod stack;
pub mod vm_state;

pub use frame::{CallFrame, ForRangeState, CALL_FRAME_FUNCTION_INDEX_MAIN};
pub use stack::{
    active_sp_for, available_in_frame, compact_after_native_result,
    compact_caller_stack_after_return, insert_at_frame_start, pop_direct, pop_return_value,
    push_direct, discard_top, frame_len, logical_len, peek, peek_with_sp, pop, pop_unchecked,
    pop_with_sp, push, push_id, push_unchecked, truncate_to, with_sp,
};
pub use vm_state::VmState;
