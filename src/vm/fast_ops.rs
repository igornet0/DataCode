//! Inline fast path for the most frequent simple opcodes on immediate values.
//!
//! [`run_fast_ops`] executes `LoadLocal` / `StoreLocal` / `Constant` / `Pop` / `Dup` / jumps /
//! `Add`/`Sub`/`Mul` / `AbsI32` / `InGridBoundsOut` straight on the VM stack and frame, without the per-instruction
//! `execute_instruction` call (22 arguments), the thread-local VM context and the `Result` plumbing.
//! Every handler first checks that the operands are the trivial case it knows; anything else
//! (heap values, overflow, stack underflow, end of chunk) leaves the instruction untouched for the
//! general path, so results are the same as the regular opcode handlers produce.

use crate::bytecode::OpCode;
use crate::common::integral_map::IntegralSlot;
use crate::common::numeric::tagged_integral_canonical_if_whole;
use crate::common::object_map::ObjectMap;
use crate::common::value_store::{ValueCell, ValueId, ValueStore, NULL_VALUE_ID};
use crate::common::TaggedValue;
use crate::vm::frame::CallFrame;

/// Same rule as the arithmetic handlers: a finite `f64` with no fractional part that converts
/// to `i64` exactly.
#[inline(always)]
fn f64_whole_to_i64_exact(f: f64) -> Option<i64> {
    // Equivalent to `is_finite && fract == 0` followed by the exactness check: a saturated or
    // truncated cast never round-trips for non-finite or fractional values.
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

#[derive(Clone, Copy)]
enum ArithOp {
    Add,
    Sub,
    Mul,
}

/// Result of `a + b` / `a - b` / `a * b` for immediates, or `None` to defer to the full
/// handler. Same rules as `op_add`/`op_sub`/`op_mul`: two whole operands give an `int`,
/// two float-domain numbers otherwise give a float, anything else goes to the handler.
#[inline(always)]
fn arith_immediates(a: TaggedValue, b: TaggedValue, op: ArithOp) -> Option<TaggedValue> {
    let int_op = |i: i64, j: i64| match op {
        ArithOp::Add => i.wrapping_add(j),
        ArithOp::Sub => i.wrapping_sub(j),
        ArithOp::Mul => i.wrapping_mul(j),
    };
    if a.is_number() && b.is_number() {
        let (x, y) = (a.get_f64(), b.get_f64());
        return match (f64_whole_to_i64_exact(x), f64_whole_to_i64_exact(y)) {
            (Some(i), Some(j)) => int_result(int_op(i, j)),
            _ => Some(TaggedValue::from_f64(match op {
                ArithOp::Add => x + y,
                ArithOp::Sub => x - y,
                ArithOp::Mul => x * y,
            })),
        };
    }
    let i = int_operand(a)?;
    let j = int_operand(b)?;
    int_result(int_op(i, j))
}

/// Whole-number key of an immediate member / subscript (`None` for heap values and
/// non-integral numbers, which go to the general handlers).
#[inline(always)]
fn immediate_integral_key(tv: TaggedValue) -> Option<i64> {
    if tv.is_heap() {
        return None;
    }
    tagged_integral_canonical_if_whole(tv)
}

/// Dict cell the integral fast paths may touch without class-visibility checks: the same
/// condition the general handlers use (plain dict, or no string keys that could hold class
/// metadata).
#[inline(always)]
fn is_plain_dict(map: &ObjectMap, id: ValueId, store: &ValueStore) -> bool {
    map.is_plain() || !map.has_bucket_entries() || store.is_plain_object(id)
}

/// `member in container` for a whole-number member and a set / plain dict container
/// (whole-number keys live only in the integral index). `None` → general handler.
#[inline(always)]
fn integral_membership(member: TaggedValue, container: TaggedValue, store: &ValueStore) -> Option<bool> {
    let key = immediate_integral_key(member)?;
    if !container.is_heap() {
        return None;
    }
    let id = container.get_heap_id();
    match store.get(id)? {
        ValueCell::Set(set) => Some(set.contains_integral(key)),
        ValueCell::Object(map) if is_plain_dict(map, id, store) => Some(map.contains_integral(key)),
        _ => None,
    }
}

/// `container[key] = value` for a plain dict, whole-number key and immediate score value,
/// mirroring `op_object_set_integral`. Returns false (nothing changed) when the general handler
/// is needed (heap values / existing heap slot / other containers).
#[inline(always)]
fn dict_set_integral(container: TaggedValue, key: TaggedValue, value: TaggedValue, store: &mut ValueStore) -> bool {
    if !container.is_heap() || value.is_heap() || !(value.is_number() || value.is_int() || value.is_null()) {
        return false;
    }
    let Some(key) = immediate_integral_key(key) else {
        return false;
    };
    let id = container.get_heap_id();
    let plain_object = store.is_plain_object(id);
    let Some(ValueCell::Object(map)) = store.get_mut(id) else {
        return false;
    };
    if map.is_frozen() || !(plain_object || map.is_plain() || !map.has_bucket_entries()) {
        return false;
    }
    if map.try_write_integral_immediate(key, value) {
        return true;
    }
    if matches!(map.find_integral_slot(key), Some(IntegralSlot::Heap(_))) {
        return false;
    }
    map.upsert_integral_tagged(key, NULL_VALUE_ID, value);
    true
}

/// Grid coordinate operand of `InGridBoundsOut` for immediates (heap values → `Err`).
#[inline(always)]
fn grid_bound(tv: TaggedValue) -> Result<Option<i64>, ()> {
    if tv.is_heap() {
        return Err(());
    }
    Ok(crate::common::numeric::tagged_integral_canonical_if_whole(tv))
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

/// Execute simple opcodes of the current frame until one needs the general path.
/// `stack_sp` is the VM's logical stack top ([`crate::vm::vm::Vm::stack_sp`]).
pub(crate) fn run_fast_ops(
    stack: &mut Vec<TaggedValue>,
    stack_sp: &mut usize,
    frames: &mut [CallFrame],
    store: &mut ValueStore,
) {
    let Some(frame) = frames.last_mut() else {
        return;
    };
    // Split borrows so `ip`, the code slice and the slot vector stay in registers across
    // iterations (writes through `slots` / `stack` cannot alias the code or each other).
    let CallFrame {
        function,
        ip: frame_ip,
        slots,
        stack_start,
        constant_tagged,
        ..
    } = frame;
    let code: &[OpCode] = &function.chunk.code;
    let constants: &[Option<TaggedValue>] = constant_tagged;
    let stack_start = *stack_start;
    let mut ip = *frame_ip;
    // Same clamp the stack helpers apply before every access; afterwards `sp <= stack.len()`
    // holds throughout (pushes extend the vector when `sp` reaches its length).
    let mut sp = (*stack_sp).min(stack.len());

    macro_rules! push {
        ($tv:expr) => {{
            let tv = $tv;
            if sp < stack.len() {
                stack[sp] = tv;
            } else {
                stack.push(tv);
            }
            sp += 1;
        }};
    }

    while let Some(op) = code.get(ip) {
        let next = ip + 1;
        match *op {
            OpCode::LoadLocal(index) => {
                let Some(&tv) = slots.get(index) else {
                    break;
                };
                push!(tv);
            }
            OpCode::LoadLocalPair(a, b) => {
                // Second `LoadLocal(b)` sits at `next`; execute both and skip it.
                let (Some(&x), Some(&y)) = (slots.get(a), slots.get(b)) else {
                    break;
                };
                push!(x);
                push!(y);
                ip = next + 1;
                continue;
            }
            OpCode::StoreLocal(index) => {
                // Immediate → immediate slot only: heap values need array-view materialization,
                // table alias forks and heap-pair recycling done by `op_store_local`.
                if sp <= stack_start {
                    break;
                }
                let tv = stack[sp - 1];
                let Some(slot) = slots.get_mut(index) else {
                    break;
                };
                if tv.is_heap() || slot.is_heap() {
                    break;
                }
                *slot = tv;
                sp -= 1;
                // Same frame-boundary truncation as `Pop` (statement `x = v` has no `Pop`).
                if sp == stack_start && stack.len() > stack_start {
                    stack.truncate(stack_start);
                }
            }
            OpCode::Constant(index) => {
                let Some(Some(tv)) = constants.get(index).copied() else {
                    break;
                };
                push!(tv);
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
                push!(tv);
            }
            OpCode::Jump8(offset) => {
                ip = (next as i64 + offset as i64) as usize;
                continue;
            }
            OpCode::Jump16(offset) => {
                ip = (next as i64 + offset as i64) as usize;
                continue;
            }
            OpCode::Jump32(offset) => {
                ip = (next as i64 + offset as i64) as usize;
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
                    ip = (next as i64 + offset as i64) as usize;
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
                    ip = (next as i64 + offset as i64) as usize;
                    continue;
                }
            }
            OpCode::Add | OpCode::Sub | OpCode::Mul => {
                if sp < stack_start + 2 {
                    break;
                }
                let kind = match op {
                    OpCode::Add => ArithOp::Add,
                    OpCode::Sub => ArithOp::Sub,
                    _ => ArithOp::Mul,
                };
                let (a, b) = (stack[sp - 2], stack[sp - 1]);
                let Some(r) = arith_immediates(a, b, kind) else {
                    break;
                };
                sp -= 1;
                stack[sp - 1] = r;
            }
            OpCode::AbsI32 => {
                if sp <= stack_start {
                    break;
                }
                let tv = stack[sp - 1];
                // Same as `op_abs_i32`: the result is always float-domain.
                let r = if tv.is_int() {
                    (tv.get_i32() as i64).unsigned_abs() as f64
                } else if tv.is_number() && tv.get_f64().is_finite() {
                    tv.get_f64().abs()
                } else {
                    break;
                };
                stack[sp - 1] = TaggedValue::from_f64(r);
            }
            OpCode::InGridBoundsOut => {
                if sp < stack_start + 4 {
                    break;
                }
                let bounds = (
                    grid_bound(stack[sp - 4]),
                    grid_bound(stack[sp - 3]),
                    grid_bound(stack[sp - 2]),
                    grid_bound(stack[sp - 1]),
                );
                let (Ok(nr), Ok(nc), Ok(rows), Ok(cols)) = bounds else {
                    break;
                };
                let in_bounds = match (nr, nc, rows, cols) {
                    (Some(nr), Some(nc), Some(rows), Some(cols)) => {
                        nr >= 0 && nr < rows && nc >= 0 && nc < cols
                    }
                    _ => false,
                };
                sp -= 3;
                stack[sp - 1] = TaggedValue::from_bool(!in_bounds);
            }
            OpCode::In | OpCode::InIntegral | OpCode::NotInIntegral => {
                if sp < stack_start + 2 {
                    break;
                }
                let Some(found) = integral_membership(stack[sp - 2], stack[sp - 1], store) else {
                    break;
                };
                sp -= 1;
                stack[sp - 1] = TaggedValue::from_bool(found != matches!(op, OpCode::NotInIntegral));
            }
            OpCode::ObjectSetIntegral => {
                // Stack `[value, key, dict]` → `[dict]`.
                if sp < stack_start + 3 {
                    break;
                }
                let container = stack[sp - 1];
                if !dict_set_integral(container, stack[sp - 2], stack[sp - 3], store) {
                    break;
                }
                sp -= 2;
                stack[sp - 1] = container;
            }
            OpCode::SetAddIntegral => {
                // Stack `[set, item]` → `[set]`.
                if sp < stack_start + 2 {
                    break;
                }
                let (set_tv, item) = (stack[sp - 2], stack[sp - 1]);
                let Some(key) = immediate_integral_key(item) else {
                    break;
                };
                if !set_tv.is_heap() {
                    break;
                }
                let Some(ValueCell::Set(set)) = store.get_mut(set_tv.get_heap_id()) else {
                    break;
                };
                set.insert_integral_only(key, NULL_VALUE_ID);
                sp -= 1;
            }
            _ => break,
        }
        ip = next;
    }
    *frame_ip = ip;
    *stack_sp = sp;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_sub_rules() {
        let i = TaggedValue::from_i32;
        let f = TaggedValue::from_f64;
        assert_eq!(arith_immediates(i(2), i(3), ArithOp::Add), Some(i(5)));
        assert_eq!(arith_immediates(f(2.0), f(3.0), ArithOp::Add), Some(i(5)));
        assert_eq!(arith_immediates(f(2.5), f(3.0), ArithOp::Add), Some(f(5.5)));
        assert_eq!(arith_immediates(i(2), f(3.0), ArithOp::Sub), Some(i(-1)));
        assert_eq!(arith_immediates(i(2), f(0.5), ArithOp::Add), None);
        assert_eq!(arith_immediates(i(i32::MAX), i(1), ArithOp::Add), None);
        assert_eq!(arith_immediates(i(1 << 20), i(1 << 20), ArithOp::Mul), None);
        assert_eq!(arith_immediates(f(1.5), f(2.0), ArithOp::Mul), Some(f(3.0)));
        assert_eq!(arith_immediates(TaggedValue::null(), i(1), ArithOp::Add), None);
    }
}
