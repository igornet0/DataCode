pub mod chunk;
pub mod function;
pub mod opcode;

pub use chunk::{Chunk, ExceptionHandlerInfo};
pub use function::{CapturedVar, Function};
pub use opcode::OpCode;

/// Global indices at or above this value are compile-time placeholders for names not defined yet
/// when referenced (`fn f() { return B }` before `B = 5`). Each name gets its own placeholder
/// (`usize::MAX`, `usize::MAX - 1`, ...) so `chunk.global_names` keeps every name; the VM resolves
/// them by name at run time. They are always >= the real globals count.
pub const UNDEFINED_GLOBAL_SENTINEL_BASE: usize = usize::MAX - (1 << 24);

/// True for an undefined-name placeholder index (see [`UNDEFINED_GLOBAL_SENTINEL_BASE`]).
#[inline]
pub fn is_undefined_global_sentinel(index: usize) -> bool {
    index >= UNDEFINED_GLOBAL_SENTINEL_BASE
}
