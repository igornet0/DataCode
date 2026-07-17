// Stack operations for VM: logical top via `Vm::stack_sp` (watermark), storage in `Vec`.

use crate::common::TaggedValue;
use crate::common::{
    error::LangError,
    value_store::{ValueId, ValueStore},
};
use crate::vm::exceptions::ExceptionHandler;
use crate::vm::heavy_store::HeavyStore;
use crate::vm::vm::current_vm_ptr;

use super::frame::CallFrame;

#[inline]
fn clamp_sp_to_len(sp: &mut usize, stack_len: usize) {
    if *sp > stack_len {
        *sp = stack_len;
    }
}

/// Mutable reference to the active VM stack watermark (for safe-point compaction).
#[inline]
pub fn active_sp_for<'a>(stack: &mut Vec<TaggedValue>) -> Option<&'a mut usize> {
    active_sp(stack.len())
}

#[inline]
fn active_sp<'a>(stack_len: usize) -> Option<&'a mut usize> {
    current_vm_ptr().map(|vm_ptr| {
        let vm = unsafe { &mut *vm_ptr };
        clamp_sp_to_len(&mut vm.stack_sp, stack_len);
        &mut vm.stack_sp
    })
}

/// Run `f` with the active VM's [`stack_sp`](crate::vm::vm::Vm::stack_sp), or a local fallback when no VM context.
pub fn with_sp<R>(stack: &mut Vec<TaggedValue>, f: impl FnOnce(&mut usize) -> R) -> R {
    if let Some(sp) = active_sp(stack.len()) {
        return f(sp);
    }
    let mut sp = stack.len();
    let r = f(&mut sp);
    if sp < stack.len() {
        stack.truncate(sp);
    }
    r
}

/// Logical stack height above `stack_start` for the active frame.
#[inline]
pub fn frame_len(sp: usize, stack_start: usize) -> usize {
    sp.saturating_sub(stack_start)
}

#[inline]
fn push_at(stack: &mut Vec<TaggedValue>, sp: &mut usize, tv: TaggedValue) {
    if *sp < stack.len() {
        stack[*sp] = tv;
    } else {
        stack.push(tv);
    }
    *sp += 1;
}

/// Push a heap value by id (converts to TaggedValue on stack).
#[inline]
pub fn push_id(stack: &mut Vec<TaggedValue>, id: ValueId) {
    if let Some(sp) = active_sp(stack.len()) {
        push_at(stack, sp, TaggedValue::from_heap(id));
    } else {
        stack.push(TaggedValue::from_heap(id));
    }
}

#[inline]
pub fn push(stack: &mut Vec<TaggedValue>, tv: TaggedValue) {
    if let Some(sp) = active_sp(stack.len()) {
        push_at(stack, sp, tv);
    } else {
        stack.push(tv);
    }
}

/// Drop the top logical slot without returning it (`OpCode::Pop`).
#[inline]
pub fn discard_top(sp: &mut usize, stack_start: usize) -> bool {
    if *sp > stack_start {
        *sp -= 1;
        true
    } else {
        false
    }
}

/// After a statement `Pop` returns the stack to `stack_start`, drop dead `Vec` slots.
#[inline]
pub fn truncate_if_at_frame(stack: &mut Vec<TaggedValue>, sp: usize, stack_start: usize) {
    if sp == stack_start && stack.len() > stack_start {
        stack.truncate(stack_start);
    }
}

/// Truncate backing storage to the current watermark (e.g. before pushing a new call frame).
pub fn truncate_to_current_sp(stack: &mut Vec<TaggedValue>) -> usize {
    if let Some(sp) = active_sp_for(stack) {
        let n = *sp;
        truncate_to(stack, sp, n);
        n
    } else {
        let n = stack.len();
        stack.truncate(n);
        n
    }
}

/// Set logical top and shrink backing storage (safe point after call/return).
pub fn truncate_to(stack: &mut Vec<TaggedValue>, sp: &mut usize, new_sp: usize) {
    let _old_sp = *sp;
    *sp = new_sp;
    if stack.len() > new_sp {
        stack.truncate(new_sp);
    }
}

/// After `Return`: drop callee stack slots `[callee_stack_start..sp)`, push return value.
/// Caller expression temps below `callee_stack_start` are preserved.
pub fn compact_caller_stack_after_return(
    stack: &mut Vec<TaggedValue>,
    sp: &mut usize,
    callee_stack_start: usize,
    return_tv: TaggedValue,
) {
    truncate_to(stack, sp, callee_stack_start);
    push_at(stack, sp, return_tv);
}

/// Compact stack after native call: one result above `stack_start`.
#[inline]
pub fn compact_after_native_result(
    stack: &mut Vec<TaggedValue>,
    sp: &mut usize,
    stack_start: usize,
) {
    truncate_to(stack, sp, stack_start + 1);
}

fn stack_underflow_line(frames: &[CallFrame]) -> usize {
    if let Some(frame) = frames.last() {
        if frame.ip > 0 {
            frame.function.chunk.get_line(frame.ip - 1)
        } else {
            0
        }
    } else {
        0
    }
}

fn stack_underflow_error(frames: &[CallFrame]) -> LangError {
    ExceptionHandler::runtime_error(
        frames,
        "Stack underflow".to_string(),
        stack_underflow_line(frames),
    )
}

pub fn pop(
    stack: &mut Vec<TaggedValue>,
    frames: &mut Vec<CallFrame>,
    exception_handlers: &mut Vec<ExceptionHandler>,
    value_store: &mut ValueStore,
    heavy_store: &mut HeavyStore,
) -> Result<TaggedValue, LangError> {
    if let Some(sp) = active_sp(stack.len()) {
        return pop_with_sp(stack, sp, frames, exception_handlers, value_store, heavy_store);
    }
    stack.pop().ok_or_else(|| stack_underflow_error(frames))
}

pub fn pop_with_sp(
    stack: &mut Vec<TaggedValue>,
    sp: &mut usize,
    frames: &mut Vec<CallFrame>,
    exception_handlers: &mut Vec<ExceptionHandler>,
    value_store: &mut ValueStore,
    heavy_store: &mut HeavyStore,
) -> Result<TaggedValue, LangError> {
    let line = stack_underflow_line(frames);

    if let Some(frame) = frames.last() {
        if *sp <= frame.stack_start {
            let error =
                ExceptionHandler::runtime_error(frames, "Stack underflow".to_string(), line);
            return match ExceptionHandler::handle_exception(
                stack,
                frames,
                exception_handlers,
                error.clone(),
                value_store,
                heavy_store,
            ) {
                Ok(_) => Err(error),
                Err(e) => Err(e),
            };
        }
    } else if *sp == 0 {
        return Err(stack_underflow_error(frames));
    }

    *sp -= 1;
    Ok(stack[*sp])
}

pub fn peek(
    stack: &[TaggedValue],
    distance: usize,
    frames: &[CallFrame],
) -> Result<TaggedValue, LangError> {
    let sp = if let Some(vm_ptr) = current_vm_ptr() {
        let vm = unsafe { &*vm_ptr };
        vm.stack_sp
    } else {
        stack.len()
    };
    peek_with_sp(stack, sp, distance, frames)
}

pub fn peek_with_sp(
    stack: &[TaggedValue],
    sp: usize,
    distance: usize,
    frames: &[CallFrame],
) -> Result<TaggedValue, LangError> {
    let stack_start = frames.last().map(|f| f.stack_start).unwrap_or(0);
    if distance >= sp.saturating_sub(stack_start) {
        return Err(stack_underflow_error(frames));
    }
    Ok(stack[sp - 1 - distance])
}

/// Unchecked pop for native fast paths that already validated arity.
#[inline]
pub fn pop_unchecked(stack: &[TaggedValue], sp: &mut usize) -> TaggedValue {
    *sp -= 1;
    stack[*sp]
}

/// Unchecked push for native fast paths.
#[inline]
pub fn push_unchecked(stack: &mut Vec<TaggedValue>, sp: &mut usize, tv: TaggedValue) {
    push_at(stack, sp, tv);
}

/// Current logical height (for exception handlers and generator setup).
#[inline]
pub fn logical_len(_stack: &[TaggedValue]) -> usize {
    if let Some(vm_ptr) = current_vm_ptr() {
        let vm = unsafe { &*vm_ptr };
        vm.stack_sp
    } else {
        _stack.len()
    }
}

/// Args available for the active frame (replaces `stack.len() - frame.stack_start`).
#[inline]
pub fn available_in_frame(stack: &[TaggedValue], stack_start: usize) -> usize {
    logical_len(stack).saturating_sub(stack_start)
}

/// Pop one slot (native fast paths); decrements watermark when VM context is active.
#[inline]
pub fn pop_direct(stack: &mut Vec<TaggedValue>) -> Option<TaggedValue> {
    if let Some(sp) = active_sp(stack.len()) {
        if *sp == 0 {
            return None;
        }
        *sp -= 1;
        Some(stack[*sp])
    } else {
        stack.pop()
    }
}

/// Push one slot (native fast paths that bypassed `stack::push`).
#[inline]
pub fn push_direct(stack: &mut Vec<TaggedValue>, tv: TaggedValue) {
    if let Some(sp) = active_sp(stack.len()) {
        push_at(stack, sp, tv);
    } else {
        stack.push(tv);
    }
}

/// Insert a slot at `stack_start` and bump the logical watermark (for `@call` receiver injection).
pub fn insert_at_frame_start(
    stack: &mut Vec<TaggedValue>,
    stack_start: usize,
    tv: TaggedValue,
) {
    if let Some(sp) = active_sp(stack.len()) {
        if *sp < stack_start {
            *sp = stack_start;
        }
        stack.insert(stack_start, tv);
        *sp += 1;
    } else {
        stack.insert(stack_start, tv);
    }
}

/// Pop return value for `Return` when stack may be empty (no underflow error).
#[inline]
pub fn pop_return_value(stack: &mut Vec<TaggedValue>, stack_start: usize) -> TaggedValue {
    if let Some(sp) = active_sp(stack.len()) {
        if *sp > stack_start {
            *sp -= 1;
            return stack[*sp];
        }
        return TaggedValue::null();
    }
    if stack.len() > stack_start {
        stack.pop().unwrap_or(TaggedValue::null())
    } else {
        TaggedValue::null()
    }
}
