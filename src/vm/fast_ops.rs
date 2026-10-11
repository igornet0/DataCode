//! Inline fast path for the most frequent simple opcodes on immediate values.
//!
//! [`run_fast_ops`] executes `LoadLocal` / `StoreLocal` / `Constant` / `Pop` / `Dup` / jumps /
//! integer `Add`/`Sub` straight on the VM stack and frame, without the per-instruction
//! `execute_instruction` call (22 arguments), the thread-local VM context and the `Result` plumbing.
//! Every handler first checks that the operands are the trivial case it knows; anything else
//! (heap values, overflow, stack underflow, end of chunk) leaves the instruction untouched for the
//! general path, so results are the same as the regular opcode handlers produce.

use crate::bytecode::OpCode;
use crate::common::TaggedValue;
use crate::vm::frame::CallFrame;

/// Same rule as the arithmetic handlers: a finite `f64` with no fractional part that converts
/// to `i64` exactly.
#[inline(always)]
fn f64_whole_to_i64_exact(f: f64) -> Option<i64> {
    if !f.is_finite() || f.fract() != 0.0 {
        return None;
    }
    let i = f as i64;
    if (i as f64) == f {
        Some(i)
    } else {
        None
    }
}

/// Integer operand for `+`/`-`: `int` immediate or whole float-domain number.
/// `None` means "not a simple integer" (fractional number, heap value, ±int∞, …).
#[inline(always)]
fn int_operand(tv: TaggedValue) -> Option<i64> {
    if tv.is_int() {
        return Some(tv.get_i32() as i64);
    }
    if tv.is_number() {
        return f64_whole_to_i64_exact(tv.get_f64());
    }
    None
}

/// Result of `a + b` / `a - b` for immediates, or `None` to defer to the full handler.
#[inline(always)]
fn add_sub_immediates(a: TaggedValue, b: TaggedValue, sub: bool) -> Option<TaggedValue> {
    if a.is_number() && b.is_number() {
        let (x, y) = (a.get_f64(), b.get_f64());
        return match (f64_whole_to_i64_exact(x), f64_whole_to_i64_exact(y)) {
            (Some(i), Some(j)) => int_result(if sub { i.wrapping_sub(j) } else { i.wrapping_add(j) }),
            _ => Some(TaggedValue::from_f64(if sub { x - y } else { x + y })),
        };
    }
    let i = int_operand(a)?;
    let j = int_operand(b)?;
    int_result(if sub { i.wrapping_sub(j) } else { i.wrapping_add(j) })
}

#[inline(always)]
fn int_result(n: i64) -> Option<TaggedValue> {
    if n >= i32::MIN as i64 && n <= i32::MAX as i64 {
        Some(TaggedValue::from_i32(n as i32))
    } else {
        None
    }
}

/// Truthiness of an immediate condition (`None` for heap values).
#[inline(always)]
fn immediate_truthy(tv: TaggedValue) -> Option<bool> {
    if tv.is_heap() {
        return None;
    }
    if tv.is_null() {
        return Some(false);
    }
    if tv.is_bool() {
        return Some(tv.get_bool());
    }
    if tv.is_number() {
        return Some(tv.get_f64() != 0.0);
    }
    if tv.is_int() {
        return Some(tv.get_i32() != 0);
    }
    Some(true)
}

#[inline(always)]
fn push(stack: &mut Vec<TaggedValue>, sp: &mut usize, tv: TaggedValue) {
    if *sp < stack.len() {
        stack[*sp] = tv;
    } else {
        stack.push(tv);
    }
    *sp += 1;
}

#[inline(always)]
fn jump(frame: &mut CallFrame, next_ip: usize, offset: i64) {
    frame.ip = (next_ip as i64 + offset) as usize;
}

/// Execute simple opcodes of the current frame until one needs the general path.
/// `stack_sp` is the VM's logical stack top ([`crate::vm::vm::Vm::stack_sp`]).
pub(crate) fn run_fast_ops(stack: &mut Vec<TaggedValue>, stack_sp: &mut usize, frames: &mut [CallFrame]) {
    let Some(frame) = frames.last_mut() else {
        return;
    };
    // Same clamp the stack helpers apply before every access.
    let mut sp = (*stack_sp).min(stack.len());
    let stack_start = frame.stack_start;
    loop {
        let ip = frame.ip;
        let Some(op) = frame.function.chunk.code.get(ip) else {
            break;
        };
        let next = ip + 1;
        match *op {
            OpCode::LoadLocal(index) => {
                let Some(&tv) = frame.slots.get(index) else {
                    break;
                };
                push(stack, &mut sp, tv);
            }
            OpCode::StoreLocal(index) => {
                // Immediate → immediate slot only: heap values need array-view materialization,
                // table alias forks and heap-pair recycling done by `op_store_local`.
                if sp <= stack_start || index >= frame.slots.len() {
                    break;
                }
                let tv = stack[sp - 1];
                if tv.is_heap() || frame.slots[index].is_heap() {
                    break;
                }
                sp -= 1;
                frame.slots[index] = tv;
            }
            OpCode::Constant(index) => {
                let Some(Some(tv)) = frame.constant_tagged.get(index).copied() else {
                    break;
                };
                push(stack, &mut sp, tv);
            }
            OpCode::Pop => {
                if sp <= stack_start {
                    break;
                }
                sp -= 1;
                if sp == stack_start && stack.len() > stack_start {
                    stack.truncate(stack_start);
                }
            }
            OpCode::Dup => {
                if sp <= stack_start {
                    break;
                }
                let tv = stack[sp - 1];
                if tv.is_heap() {
                    break;
                }
                push(stack, &mut sp, tv);
            }
            OpCode::Jump8(offset) => {
                jump(frame, next, offset as i64);
                continue;
            }
            OpCode::Jump16(offset) => {
                jump(frame, next, offset as i64);
                continue;
            }
            OpCode::Jump32(offset) => {
                jump(frame, next, offset as i64);
                continue;
            }
            OpCode::JumpIfFalse8(offset) => {
                if sp <= stack_start {
                    break;
                }
                let Some(truthy) = immediate_truthy(stack[sp - 1]) else {
                    break;
                };
                sp -= 1;
                if !truthy {
                    jump(frame, next, offset as i64);
                    continue;
                }
            }
            OpCode::JumpIfFalse16(offset) => {
                if sp <= stack_start {
                    break;
                }
                let Some(truthy) = immediate_truthy(stack[sp - 1]) else {
                    break;
                };
                sp -= 1;
                if !truthy {
                    jump(frame, next, offset as i64);
                    continue;
                }
            }
            OpCode::Add | OpCode::Sub => {
                if sp < stack_start + 2 {
                    break;
                }
                let (a, b) = (stack[sp - 2], stack[sp - 1]);
                let Some(r) = add_sub_immediates(a, b, matches!(op, OpCode::Sub)) else {
                    break;
                };
                sp -= 1;
                stack[sp - 1] = r;
            }
            _ => break,
        }
        frame.ip = next;
    }
    *stack_sp = sp;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_sub_rules() {
        let i = TaggedValue::from_i32;
        let f = TaggedValue::from_f64;
        assert_eq!(add_sub_immediates(i(2), i(3), false), Some(i(5)));
        assert_eq!(add_sub_immediates(f(2.0), f(3.0), false), Some(i(5)));
        assert_eq!(add_sub_immediates(f(2.5), f(3.0), false), Some(f(5.5)));
        assert_eq!(add_sub_immediates(i(2), f(3.0), true), Some(i(-1)));
        assert_eq!(add_sub_immediates(i(2), f(0.5), false), None);
        assert_eq!(add_sub_immediates(i(i32::MAX), i(1), false), None);
        assert_eq!(add_sub_immediates(TaggedValue::null(), i(1), false), None);
    }
}
