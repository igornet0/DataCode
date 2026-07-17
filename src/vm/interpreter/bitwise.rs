//! Bitwise operators (`&`, `|`, `^`, `~`, `<<`, `>>`) — [`Value::Int`] finite operands only.

use crate::common::error::ErrorType;
use crate::common::error::LangError;
use crate::common::numeric::{f64_trunc_to_i64_clamped, IntValue};
use crate::common::value::Value;
use crate::common::value_store::{ValueCell, ValueStore};
use crate::common::TaggedValue;
use crate::vm::exceptions::ExceptionHandler;
use crate::vm::frame::CallFrame;
use crate::vm::heavy_store::HeavyStore;
use crate::vm::store_convert::load_value;
use crate::vm::stack;
use crate::vm::types::VMStatus;

const BITWISE_ERR: &str = "Bitwise operator is supported only for int values";

fn line_at(frames: &[CallFrame], ip: usize) -> usize {
    frames
        .last()
        .map(|f| f.function.chunk.get_line(ip.saturating_sub(1)))
        .unwrap_or(0)
}

fn runtime_type_err(
    line: usize,
    stack: &mut Vec<TaggedValue>,
    frames: &mut Vec<CallFrame>,
    exception_handlers: &mut Vec<ExceptionHandler>,
    value_store: &mut ValueStore,
    heavy_store: &mut HeavyStore,
) -> Result<VMStatus, LangError> {
    let error = ExceptionHandler::runtime_error_with_type(
        frames,
        BITWISE_ERR.to_string(),
        line,
        ErrorType::TypeError,
    );
    ExceptionHandler::handle_exception_vm(
        stack,
        frames,
        exception_handlers,
        error,
        value_store,
        heavy_store,
    )
}

/// Finite `int` or whole `number` literal — not `float` / `5.0`.
fn value_as_bitwise_int(v: &Value) -> Option<i64> {
    match v {
        Value::Int(IntValue::Finite(n)) => Some(*n),
        Value::Number(n) if n.is_finite() && n.fract() == 0.0 => {
            Some(f64_trunc_to_i64_clamped(*n))
        }
        _ => None,
    }
}

fn tagged_as_int_i64(
    tv: TaggedValue,
    value_store: &ValueStore,
    heavy_store: &HeavyStore,
) -> Option<i64> {
    if tv.is_int() {
        return Some(tv.get_i32() as i64);
    }
    if tv.is_number() {
        let n = tv.get_f64();
        if n.is_finite() && n.fract() == 0.0 {
            return Some(f64_trunc_to_i64_clamped(n));
        }
        return None;
    }
    if tv.is_heap() {
        let v = load_value(tv.get_heap_id(), value_store, heavy_store);
        return value_as_bitwise_int(&v);
    }
    None
}

fn push_int(
    stack: &mut Vec<TaggedValue>,
    n: i64,
    value_store: &mut ValueStore,
    _heavy_store: &mut HeavyStore,
) {
    let id = value_store.allocate_ephemeral(ValueCell::Int(IntValue::Finite(n)));
    stack::push_id(stack, id);
}

fn shift_amount(b: i64) -> Option<u32> {
    if b < 0 {
        return None;
    }
    Some((b as u32).min(63))
}

fn binary_bitwise<F>(
    current_ip: usize,
    stack: &mut Vec<TaggedValue>,
    frames: &mut Vec<CallFrame>,
    exception_handlers: &mut Vec<ExceptionHandler>,
    value_store: &mut ValueStore,
    heavy_store: &mut HeavyStore,
    op: F,
) -> Result<VMStatus, LangError>
where
    F: FnOnce(i64, i64) -> i64,
{
    let line = line_at(frames, current_ip);
    let b_tv = stack::pop(stack, frames, exception_handlers, value_store, heavy_store)?;
    let a_tv = stack::pop(stack, frames, exception_handlers, value_store, heavy_store)?;
    let (a, b) = match (
        tagged_as_int_i64(a_tv, value_store, heavy_store),
        tagged_as_int_i64(b_tv, value_store, heavy_store),
    ) {
        (Some(a), Some(b)) => (a, b),
        _ => return runtime_type_err(line, stack, frames, exception_handlers, value_store, heavy_store),
    };
    push_int(stack, op(a, b), value_store, heavy_store);
    Ok(VMStatus::Continue)
}

fn shift_op<F>(
    current_ip: usize,
    stack: &mut Vec<TaggedValue>,
    frames: &mut Vec<CallFrame>,
    exception_handlers: &mut Vec<ExceptionHandler>,
    value_store: &mut ValueStore,
    heavy_store: &mut HeavyStore,
    op: F,
) -> Result<VMStatus, LangError>
where
    F: FnOnce(i64, u32) -> i64,
{
    let line = line_at(frames, current_ip);
    let b_tv = stack::pop(stack, frames, exception_handlers, value_store, heavy_store)?;
    let a_tv = stack::pop(stack, frames, exception_handlers, value_store, heavy_store)?;
    let a = match tagged_as_int_i64(a_tv, value_store, heavy_store) {
        Some(n) => n,
        None => return runtime_type_err(line, stack, frames, exception_handlers, value_store, heavy_store),
    };
    let b = match tagged_as_int_i64(b_tv, value_store, heavy_store) {
        Some(n) => n,
        None => return runtime_type_err(line, stack, frames, exception_handlers, value_store, heavy_store),
    };
    let shift = match shift_amount(b) {
        Some(s) => s,
        None => return runtime_type_err(line, stack, frames, exception_handlers, value_store, heavy_store),
    };
    push_int(stack, op(a, shift), value_store, heavy_store);
    Ok(VMStatus::Continue)
}

pub fn op_bit_and(
    current_ip: usize,
    stack: &mut Vec<TaggedValue>,
    frames: &mut Vec<CallFrame>,
    exception_handlers: &mut Vec<ExceptionHandler>,
    value_store: &mut ValueStore,
    heavy_store: &mut HeavyStore,
) -> Result<VMStatus, LangError> {
    binary_bitwise(current_ip, stack, frames, exception_handlers, value_store, heavy_store, |a, b| a & b)
}

pub fn op_bit_or(
    current_ip: usize,
    stack: &mut Vec<TaggedValue>,
    frames: &mut Vec<CallFrame>,
    exception_handlers: &mut Vec<ExceptionHandler>,
    value_store: &mut ValueStore,
    heavy_store: &mut HeavyStore,
) -> Result<VMStatus, LangError> {
    binary_bitwise(current_ip, stack, frames, exception_handlers, value_store, heavy_store, |a, b| a | b)
}

pub fn op_bit_xor(
    current_ip: usize,
    stack: &mut Vec<TaggedValue>,
    frames: &mut Vec<CallFrame>,
    exception_handlers: &mut Vec<ExceptionHandler>,
    value_store: &mut ValueStore,
    heavy_store: &mut HeavyStore,
) -> Result<VMStatus, LangError> {
    binary_bitwise(current_ip, stack, frames, exception_handlers, value_store, heavy_store, |a, b| a ^ b)
}

pub fn op_shift_left(
    current_ip: usize,
    stack: &mut Vec<TaggedValue>,
    frames: &mut Vec<CallFrame>,
    exception_handlers: &mut Vec<ExceptionHandler>,
    value_store: &mut ValueStore,
    heavy_store: &mut HeavyStore,
) -> Result<VMStatus, LangError> {
    shift_op(
        current_ip,
        stack,
        frames,
        exception_handlers,
        value_store,
        heavy_store,
        |a, s| a.wrapping_shl(s),
    )
}

pub fn op_shift_right(
    current_ip: usize,
    stack: &mut Vec<TaggedValue>,
    frames: &mut Vec<CallFrame>,
    exception_handlers: &mut Vec<ExceptionHandler>,
    value_store: &mut ValueStore,
    heavy_store: &mut HeavyStore,
) -> Result<VMStatus, LangError> {
    shift_op(current_ip, stack, frames, exception_handlers, value_store, heavy_store, |a, s| a >> s)
}

pub fn op_bit_not(
    current_ip: usize,
    stack: &mut Vec<TaggedValue>,
    frames: &mut Vec<CallFrame>,
    exception_handlers: &mut Vec<ExceptionHandler>,
    value_store: &mut ValueStore,
    heavy_store: &mut HeavyStore,
) -> Result<VMStatus, LangError> {
    let line = line_at(frames, current_ip);
    let a_tv = stack::pop(stack, frames, exception_handlers, value_store, heavy_store)?;
    let a = match tagged_as_int_i64(a_tv, value_store, heavy_store) {
        Some(n) => n,
        None => return runtime_type_err(line, stack, frames, exception_handlers, value_store, heavy_store),
    };
    push_int(stack, !a, value_store, heavy_store);
    Ok(VMStatus::Continue)
}
