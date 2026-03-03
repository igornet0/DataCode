// Arithmetic opcodes: Add, Sub, Mul, Div, IntDiv, Mod, Pow, Negate, Not, Or, And, RegAdd.
// Logic preserved 1:1 from executor.rs — no semantic changes.

use crate::common::{error::LangError, value::Value, value_store::{ValueCell, ValueId, ValueStore}, TaggedValue};
use crate::vm::exceptions::ExceptionHandler;
use crate::vm::frame::CallFrame;
use crate::vm::heavy_store::HeavyStore;
use crate::vm::operations;
use crate::vm::stack;
use crate::vm::store_convert::{load_value, store_value, tagged_to_value_id};
use crate::vm::types::VMStatus;
use std::fmt::Write;

fn pop_to_value_id(
    stack: &mut Vec<TaggedValue>,
    frames: &mut Vec<CallFrame>,
    exception_handlers: &mut Vec<ExceptionHandler>,
    value_store: &mut ValueStore,
    heavy_store: &mut HeavyStore,
) -> Result<ValueId, LangError> {
    let tv = stack::pop(stack, frames, exception_handlers, value_store, heavy_store)?;
    Ok(tagged_to_value_id(tv, value_store))
}

pub fn op_add(
    current_ip: usize,
    stack: &mut Vec<TaggedValue>,
    frames: &mut Vec<CallFrame>,
    exception_handlers: &mut Vec<ExceptionHandler>,
    value_store: &mut ValueStore,
    heavy_store: &mut HeavyStore,
) -> Result<VMStatus, LangError> {
    let b_tv = stack::pop(stack, frames, exception_handlers, value_store, heavy_store)?;
    let a_tv = stack::pop(stack, frames, exception_handlers, value_store, heavy_store)?;
    {
        let frame = frames.last_mut().unwrap();
        let cache_hit = frame.add_cache_ip == Some(current_ip) && frame.add_cache_both_number;
        if cache_hit && a_tv.is_number() && b_tv.is_number() {
            stack::push(stack, TaggedValue::from_f64(a_tv.get_f64() + b_tv.get_f64()));
            return Ok(VMStatus::Continue);
        }
        if cache_hit {
            frame.add_cache_both_number = false;
        }
        if a_tv.is_number() && b_tv.is_number() {
            frame.add_cache_ip = Some(current_ip);
            frame.add_cache_both_number = true;
            stack::push(stack, TaggedValue::from_f64(a_tv.get_f64() + b_tv.get_f64()));
            return Ok(VMStatus::Continue);
        }
        frame.add_cache_both_number = false;
    }
    let a_id = tagged_to_value_id(a_tv, value_store);
    let b_id = tagged_to_value_id(b_tv, value_store);
    if let (Some(ValueCell::String(sid)), Some(ValueCell::Number(n))) =
        (value_store.get(a_id), value_store.get(b_id))
    {
        if let Some(prefix) = value_store.get_string(*sid) {
            let mut buf = String::with_capacity(prefix.len() + 24);
            buf.push_str(prefix);
            let _ = write!(buf, "{}", n);
            let new_sid = value_store.intern_string(buf);
            let result_id = value_store.allocate(ValueCell::String(new_sid));
            stack::push_id(stack, result_id);
            return Ok(VMStatus::Continue);
        }
    }
    if let (Some(ValueCell::Number(n)), Some(ValueCell::String(sid))) =
        (value_store.get(a_id), value_store.get(b_id))
    {
        if let Some(suffix) = value_store.get_string(*sid) {
            let mut buf = String::with_capacity(24 + suffix.len());
            let _ = write!(buf, "{}", n);
            buf.push_str(suffix);
            let new_sid = value_store.intern_string(buf);
            let result_id = value_store.allocate(ValueCell::String(new_sid));
            stack::push_id(stack, result_id);
            return Ok(VMStatus::Continue);
        }
    }
    let a = load_value(a_id, value_store, heavy_store);
    let b = load_value(b_id, value_store, heavy_store);
    let result = match (&a, &b) {
        (Value::Number(n1), Value::Number(n2)) => Value::Number(n1 + n2),
        _ => operations::binary_add(&a, &b, frames, stack, exception_handlers, value_store, heavy_store)?,
    };
    stack::push_id(stack, store_value(result, value_store, heavy_store));
    Ok(VMStatus::Continue)
}

pub fn op_reg_add(rd: u8, r1: u8, r2: u8, frames: &mut Vec<CallFrame>) -> Result<VMStatus, LangError> {
    let frame = frames.last_mut().unwrap();
    let (rd, r1, r2) = (rd as usize, r1 as usize, r2 as usize);
    let n = 1 + rd.max(r1).max(r2);
    if frame.regs.len() < n {
        frame.regs.resize(n, TaggedValue::null());
    }
    let a = frame.regs[r1];
    let b = frame.regs[r2];
    if a.is_number() && b.is_number() {
        frame.regs[rd] = TaggedValue::from_f64(a.get_f64() + b.get_f64());
    }
    Ok(VMStatus::Continue)
}

pub fn op_sub(
    current_ip: usize,
    stack: &mut Vec<TaggedValue>,
    frames: &mut Vec<CallFrame>,
    exception_handlers: &mut Vec<ExceptionHandler>,
    value_store: &mut ValueStore,
    heavy_store: &mut HeavyStore,
) -> Result<VMStatus, LangError> {
    let b_tv = stack::pop(stack, frames, exception_handlers, value_store, heavy_store)?;
    let a_tv = stack::pop(stack, frames, exception_handlers, value_store, heavy_store)?;
    if a_tv.is_number() && b_tv.is_number() {
        stack::push(stack, TaggedValue::from_f64(a_tv.get_f64() - b_tv.get_f64()));
        return Ok(VMStatus::Continue);
    }
    {
        let frame = frames.last_mut().unwrap();
        if frame.sub_cache_ip == Some(current_ip) {
            frame.sub_cache_both_number = false;
        }
    }
    let a_id = tagged_to_value_id(a_tv, value_store);
    let b_id = tagged_to_value_id(b_tv, value_store);
    let a = load_value(a_id, value_store, heavy_store);
    let b = load_value(b_id, value_store, heavy_store);
    let result = match (&a, &b) {
        (Value::Number(n1), Value::Number(n2)) => Value::Number(n1 - n2),
        (Value::Null, Value::Number(n2)) => Value::Number(-n2),
        (Value::Number(n1), Value::Null) => Value::Number(*n1),
        _ => operations::binary_sub(&a, &b, frames, stack, exception_handlers, value_store, heavy_store)?,
    };
    stack::push_id(stack, store_value(result, value_store, heavy_store));
    Ok(VMStatus::Continue)
}

pub fn op_mul(
    current_ip: usize,
    stack: &mut Vec<TaggedValue>,
    frames: &mut Vec<CallFrame>,
    exception_handlers: &mut Vec<ExceptionHandler>,
    value_store: &mut ValueStore,
    heavy_store: &mut HeavyStore,
) -> Result<VMStatus, LangError> {
    let b_tv = stack::pop(stack, frames, exception_handlers, value_store, heavy_store)?;
    let a_tv = stack::pop(stack, frames, exception_handlers, value_store, heavy_store)?;
    if a_tv.is_number() && b_tv.is_number() {
        stack::push(stack, TaggedValue::from_f64(a_tv.get_f64() * b_tv.get_f64()));
        return Ok(VMStatus::Continue);
    }
    {
        let frame = frames.last_mut().unwrap();
        if frame.mul_cache_ip == Some(current_ip) {
            frame.mul_cache_both_number = false;
        }
    }
    let a_id = tagged_to_value_id(a_tv, value_store);
    let b_id = tagged_to_value_id(b_tv, value_store);
    let a = load_value(a_id, value_store, heavy_store);
    let b = load_value(b_id, value_store, heavy_store);
    let result = match (&a, &b) {
        (Value::Number(n1), Value::Number(n2)) => Value::Number(n1 * n2),
        _ => operations::binary_mul(&a, &b, frames, stack, exception_handlers, value_store, heavy_store)?,
    };
    stack::push_id(stack, store_value(result, value_store, heavy_store));
    Ok(VMStatus::Continue)
}

pub fn op_div(
    current_ip: usize,
    stack: &mut Vec<TaggedValue>,
    frames: &mut Vec<CallFrame>,
    exception_handlers: &mut Vec<ExceptionHandler>,
    value_store: &mut ValueStore,
    heavy_store: &mut HeavyStore,
) -> Result<VMStatus, LangError> {
    let b_tv = stack::pop(stack, frames, exception_handlers, value_store, heavy_store)?;
    let a_tv = stack::pop(stack, frames, exception_handlers, value_store, heavy_store)?;
    if a_tv.is_number() && b_tv.is_number() {
        let n2 = b_tv.get_f64();
        if n2 != 0.0 {
            stack::push(stack, TaggedValue::from_f64(a_tv.get_f64() / n2));
            return Ok(VMStatus::Continue);
        }
    }
    {
        let frame = frames.last_mut().unwrap();
        if frame.div_cache_ip == Some(current_ip) {
            frame.div_cache_both_number = false;
        }
    }
    let a_id = tagged_to_value_id(a_tv, value_store);
    let b_id = tagged_to_value_id(b_tv, value_store);
    let a = load_value(a_id, value_store, heavy_store);
    let b = load_value(b_id, value_store, heavy_store);
    let result = match (&a, &b) {
        (Value::Number(n1), Value::Number(n2)) if *n2 != 0.0 => Value::Number(n1 / n2),
        (Value::Number(_), Value::Number(_)) | _ => operations::binary_div(&a, &b, frames, stack, exception_handlers, value_store, heavy_store)?,
    };
    stack::push_id(stack, store_value(result, value_store, heavy_store));
    Ok(VMStatus::Continue)
}

pub fn op_int_div(
    current_ip: usize,
    stack: &mut Vec<TaggedValue>,
    frames: &mut Vec<CallFrame>,
    exception_handlers: &mut Vec<ExceptionHandler>,
    value_store: &mut ValueStore,
    heavy_store: &mut HeavyStore,
) -> Result<VMStatus, LangError> {
    let b_tv = stack::pop(stack, frames, exception_handlers, value_store, heavy_store)?;
    let a_tv = stack::pop(stack, frames, exception_handlers, value_store, heavy_store)?;
    if a_tv.is_number() && b_tv.is_number() {
        let n2 = b_tv.get_f64();
        if n2 != 0.0 {
            stack::push(stack, TaggedValue::from_f64((a_tv.get_f64() / n2).floor()));
            return Ok(VMStatus::Continue);
        }
    }
    {
        let frame = frames.last_mut().unwrap();
        if frame.intdiv_cache_ip == Some(current_ip) {
            frame.intdiv_cache_both_number = false;
        }
    }
    let a_id = tagged_to_value_id(a_tv, value_store);
    let b_id = tagged_to_value_id(b_tv, value_store);
    let a = load_value(a_id, value_store, heavy_store);
    let b = load_value(b_id, value_store, heavy_store);
    let result = operations::binary_int_div(&a, &b, frames, stack, exception_handlers, value_store, heavy_store)?;
    stack::push_id(stack, store_value(result, value_store, heavy_store));
    Ok(VMStatus::Continue)
}

pub fn op_mod(
    current_ip: usize,
    stack: &mut Vec<TaggedValue>,
    frames: &mut Vec<CallFrame>,
    exception_handlers: &mut Vec<ExceptionHandler>,
    value_store: &mut ValueStore,
    heavy_store: &mut HeavyStore,
) -> Result<VMStatus, LangError> {
    let b_tv = stack::pop(stack, frames, exception_handlers, value_store, heavy_store)?;
    let a_tv = stack::pop(stack, frames, exception_handlers, value_store, heavy_store)?;
    if a_tv.is_number() && b_tv.is_number() {
        let n2 = b_tv.get_f64();
        if n2 != 0.0 {
            stack::push(stack, TaggedValue::from_f64(a_tv.get_f64() % n2));
            return Ok(VMStatus::Continue);
        }
    }
    {
        let frame = frames.last_mut().unwrap();
        if frame.mod_cache_ip == Some(current_ip) {
            frame.mod_cache_both_number = false;
        }
    }
    let a_id = tagged_to_value_id(a_tv, value_store);
    let b_id = tagged_to_value_id(b_tv, value_store);
    let a = load_value(a_id, value_store, heavy_store);
    let b = load_value(b_id, value_store, heavy_store);
    let result = operations::binary_mod(&a, &b, frames, stack, exception_handlers, value_store, heavy_store)?;
    stack::push_id(stack, store_value(result, value_store, heavy_store));
    Ok(VMStatus::Continue)
}

pub fn op_pow(
    stack: &mut Vec<TaggedValue>,
    frames: &mut Vec<CallFrame>,
    exception_handlers: &mut Vec<ExceptionHandler>,
    value_store: &mut ValueStore,
    heavy_store: &mut HeavyStore,
) -> Result<VMStatus, LangError> {
    let b_tv = stack::pop(stack, frames, exception_handlers, value_store, heavy_store)?;
    let a_tv = stack::pop(stack, frames, exception_handlers, value_store, heavy_store)?;
    if a_tv.is_number() && b_tv.is_number() {
        stack::push(stack, TaggedValue::from_f64(a_tv.get_f64().powf(b_tv.get_f64())));
        return Ok(VMStatus::Continue);
    }
    let a_id = tagged_to_value_id(a_tv, value_store);
    let b_id = tagged_to_value_id(b_tv, value_store);
    let a = load_value(a_id, value_store, heavy_store);
    let b = load_value(b_id, value_store, heavy_store);
    let result = operations::binary_pow(&a, &b, frames, stack, exception_handlers, value_store, heavy_store)?;
    stack::push_id(stack, store_value(result, value_store, heavy_store));
    Ok(VMStatus::Continue)
}

pub fn op_negate(
    stack: &mut Vec<TaggedValue>,
    frames: &mut Vec<CallFrame>,
    exception_handlers: &mut Vec<ExceptionHandler>,
    value_store: &mut ValueStore,
    heavy_store: &mut HeavyStore,
) -> Result<VMStatus, LangError> {
    let val_tv = stack::pop(stack, frames, exception_handlers, value_store, heavy_store)?;
    if val_tv.is_number() {
        stack::push(stack, TaggedValue::from_f64(-val_tv.get_f64()));
        return Ok(VMStatus::Continue);
    }
    let val_id = tagged_to_value_id(val_tv, value_store);
    let value = load_value(val_id, value_store, heavy_store);
    let result = operations::unary_negate(&value, frames, stack, exception_handlers, value_store, heavy_store)?;
    stack::push_id(stack, store_value(result, value_store, heavy_store));
    Ok(VMStatus::Continue)
}

pub fn op_not(
    stack: &mut Vec<TaggedValue>,
    frames: &mut Vec<CallFrame>,
    exception_handlers: &mut Vec<ExceptionHandler>,
    value_store: &mut ValueStore,
    heavy_store: &mut HeavyStore,
) -> Result<VMStatus, LangError> {
    let val_tv = stack::pop(stack, frames, exception_handlers, value_store, heavy_store)?;
    if val_tv.is_bool() {
        stack::push(stack, TaggedValue::from_bool(!val_tv.get_bool()));
        return Ok(VMStatus::Continue);
    }
    let val_id = tagged_to_value_id(val_tv, value_store);
    let value = load_value(val_id, value_store, heavy_store);
    let result = operations::unary_not(&value);
    stack::push_id(stack, store_value(result, value_store, heavy_store));
    Ok(VMStatus::Continue)
}

pub fn op_or(
    stack: &mut Vec<TaggedValue>,
    frames: &mut Vec<CallFrame>,
    exception_handlers: &mut Vec<ExceptionHandler>,
    value_store: &mut ValueStore,
    heavy_store: &mut HeavyStore,
) -> Result<VMStatus, LangError> {
    let b_id = pop_to_value_id(stack, frames, exception_handlers, value_store, heavy_store)?;
    let a_id = pop_to_value_id(stack, frames, exception_handlers, value_store, heavy_store)?;
    if let Some(ValueCell::Bool(a)) = value_store.get(a_id) {
        if *a {
            stack::push_id(stack, a_id);
            return Ok(VMStatus::Continue);
        }
        stack::push_id(stack, b_id);
        return Ok(VMStatus::Continue);
    }
    let a = load_value(a_id, value_store, heavy_store);
    let b = load_value(b_id, value_store, heavy_store);
    let result = operations::binary_or(&a, &b);
    stack::push_id(stack, store_value(result, value_store, heavy_store));
    Ok(VMStatus::Continue)
}

pub fn op_and(
    stack: &mut Vec<TaggedValue>,
    frames: &mut Vec<CallFrame>,
    exception_handlers: &mut Vec<ExceptionHandler>,
    value_store: &mut ValueStore,
    heavy_store: &mut HeavyStore,
) -> Result<VMStatus, LangError> {
    let b_id = pop_to_value_id(stack, frames, exception_handlers, value_store, heavy_store)?;
    let a_id = pop_to_value_id(stack, frames, exception_handlers, value_store, heavy_store)?;
    if let Some(ValueCell::Bool(a)) = value_store.get(a_id) {
        if !*a {
            stack::push_id(stack, a_id);
            return Ok(VMStatus::Continue);
        }
        stack::push_id(stack, b_id);
        return Ok(VMStatus::Continue);
    }
    let a = load_value(a_id, value_store, heavy_store);
    let b = load_value(b_id, value_store, heavy_store);
    let result = operations::binary_and(&a, &b);
    stack::push_id(stack, store_value(result, value_store, heavy_store));
    Ok(VMStatus::Continue)
}
