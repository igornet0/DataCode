// Control flow opcodes: Jump8, Jump16, Jump32, JumpIfFalse8/16/32, JumpLabel, ForRange, ForRangeNext, PopForRange.
// Logic preserved 1:1 from executor.rs — no semantic changes.

use crate::common::{error::LangError, value_store::{ValueCell, ValueStore, NULL_VALUE_ID}};
use crate::vm::frame::CallFrame;
use crate::vm::heavy_store::HeavyStore;
use crate::vm::store_convert::load_value;
use crate::vm::types::VMStatus;
use crate::common::TaggedValue;

use super::helpers::pop_to_value_id;

pub fn op_jump8(offset: i8, frames: &mut Vec<CallFrame>) -> Result<VMStatus, LangError> {
    let frame = frames.last_mut().unwrap();
    frame.ip = (frame.ip as i32 + offset as i32) as usize;
    Ok(VMStatus::Continue)
}

pub fn op_jump16(offset: i16, frames: &mut Vec<CallFrame>) -> Result<VMStatus, LangError> {
    let frame = frames.last_mut().unwrap();
    frame.ip = (frame.ip as i32 + offset as i32) as usize;
    Ok(VMStatus::Continue)
}

pub fn op_jump32(offset: i32, frames: &mut Vec<CallFrame>) -> Result<VMStatus, LangError> {
    let frame = frames.last_mut().unwrap();
    frame.ip = (frame.ip as i64 + offset as i64) as usize;
    Ok(VMStatus::Continue)
}

pub fn op_jump_if_false8(
    offset: i8,
    stack: &mut Vec<TaggedValue>,
    frames: &mut Vec<CallFrame>,
    exception_handlers: &mut Vec<crate::vm::exceptions::ExceptionHandler>,
    value_store: &mut ValueStore,
    heavy_store: &mut HeavyStore,
) -> Result<VMStatus, LangError> {
    let cond_id = pop_to_value_id(stack, frames, exception_handlers, value_store, heavy_store)?;
    let condition = load_value(cond_id, value_store, heavy_store);
    let frame = frames.last_mut().unwrap();
    if !condition.is_truthy() {
        frame.ip = (frame.ip as i32 + offset as i32) as usize;
    }
    Ok(VMStatus::Continue)
}

pub fn op_jump_if_false16(
    offset: i16,
    stack: &mut Vec<TaggedValue>,
    frames: &mut Vec<CallFrame>,
    exception_handlers: &mut Vec<crate::vm::exceptions::ExceptionHandler>,
    value_store: &mut ValueStore,
    heavy_store: &mut HeavyStore,
) -> Result<VMStatus, LangError> {
    let cond_id = pop_to_value_id(stack, frames, exception_handlers, value_store, heavy_store)?;
    let condition = load_value(cond_id, value_store, heavy_store);
    let frame = frames.last_mut().unwrap();
    if !condition.is_truthy() {
        frame.ip = (frame.ip as i32 + offset as i32) as usize;
    }
    Ok(VMStatus::Continue)
}

pub fn op_jump_if_false32(
    offset: i32,
    stack: &mut Vec<TaggedValue>,
    frames: &mut Vec<CallFrame>,
    exception_handlers: &mut Vec<crate::vm::exceptions::ExceptionHandler>,
    value_store: &mut ValueStore,
    heavy_store: &mut HeavyStore,
) -> Result<VMStatus, LangError> {
    let cond_id = pop_to_value_id(stack, frames, exception_handlers, value_store, heavy_store)?;
    let condition = load_value(cond_id, value_store, heavy_store);
    let frame = frames.last_mut().unwrap();
    if !condition.is_truthy() {
        frame.ip = (frame.ip as i64 + offset as i64) as usize;
    }
    Ok(VMStatus::Continue)
}

pub fn op_jump_label(line: usize) -> Result<VMStatus, LangError> {
    Err(crate::common::error::LangError::runtime_error(
        "JumpLabel found in VM - compilation not finalized".to_string(),
        line,
    ))
}

pub fn op_for_range(
    var_slot: usize,
    start_const: usize,
    end_const: usize,
    step_const: usize,
    end_offset: i32,
    frames: &mut Vec<CallFrame>,
    value_store: &mut ValueStore,
) -> Result<VMStatus, LangError> {
    let frame = frames.last_mut().unwrap();
    let read_const = |store: &ValueStore, f: &CallFrame, idx: usize| -> i64 {
        let id = f.constant_ids.get(idx).copied().unwrap_or(NULL_VALUE_ID);
        store.get(id).and_then(|c| match c {
            ValueCell::Number(n) => Some(*n as i64),
            _ => None,
        }).unwrap_or(0)
    };
    let (current, end, step, _continued) = match frame.for_range_stack.last() {
        Some((_, _, _, slot)) if *slot == var_slot => {
            let s = frame.for_range_stack.pop().unwrap();
            (s.0, s.1, s.2, true)
        },
        _ => {
            let start = read_const(value_store, frame, start_const);
            let end_val = read_const(value_store, frame, end_const);
            let step_val = read_const(value_store, frame, step_const);
            (start, end_val, step_val, false)
        },
    };
    let done = if step > 0 { current >= end } else { current <= end };
    if done {
        frame.ip = (frame.ip as i32 + end_offset) as usize;
    } else {
        frame.for_range_stack.push((current, end, step, var_slot));
        frame.ensure_slot(var_slot);
        frame.slots[var_slot] = TaggedValue::from_f64(current as f64);
        if frame.load_local_cache_slot == Some(var_slot) {
            frame.load_local_cache_slot = None;
        }
    }
    Ok(VMStatus::Continue)
}

pub fn op_for_range_next(back_offset: i32, frames: &mut Vec<CallFrame>) -> Result<VMStatus, LangError> {
    let frame = frames.last_mut().unwrap();
    let Some((cur, end, step, vslot)) = frame.for_range_stack.pop() else {
        return Ok(VMStatus::Continue);
    };
    let next_val = cur + step;
    let done = if step > 0 { next_val >= end } else { next_val <= end };
    if done {
        return Ok(VMStatus::Continue);
    }
    frame.for_range_stack.push((next_val, end, step, vslot));
    frame.ip = (frame.ip as i32 - back_offset) as usize;
    Ok(VMStatus::Continue)
}

pub fn op_pop_for_range(frames: &mut Vec<CallFrame>) -> Result<VMStatus, LangError> {
    let frame = frames.last_mut().unwrap();
    let _ = frame.for_range_stack.pop();
    Ok(VMStatus::Continue)
}
