// Arithmetic opcodes: Add, Sub, Mul, Div, IntDiv, Mod, Pow, Negate, Not, Or, And, RegAdd.
// Logic preserved 1:1 from executor.rs — no semantic changes.

use crate::common::{
    error::LangError,
    numeric::{divide_by_zero_raises, floor_mod_i64, ieee_div_quotient_value, integer_value_as_i64_if_whole, number_is_int_surface, IntValue},
    value::Value,
    value_store::{ValueCell, ValueId, ValueStore},
    TaggedValue,
};
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

fn try_special_binary_op(
    a: &Value,
    b: &Value,
    method: &str,
    stack: &mut Vec<TaggedValue>,
    value_store: &mut ValueStore,
    heavy_store: &mut HeavyStore,
) -> Result<Option<VMStatus>, LangError> {
    if !crate::vm::special_methods::is_class_instance(a) {
        return Ok(None);
    }
    if let Some(result) =
        crate::vm::special_methods::try_dispatch_binary_arithmetic(a, method, b)?
    {
        stack::push_id(stack, store_value(result, value_store, heavy_store));
        return Ok(Some(VMStatus::Continue));
    }
    Ok(None)
}

/// `null` behaves as 0 for `+`/`-` with typed numerics.
#[inline]
fn as_ieee_null_as_zero(v: &Value) -> Option<f64> {
    match v {
        Value::Null => Some(0.0),
        _ => v.as_ieee_f64(),
    }
}

#[inline]
fn ieee_binop_f64(a: &Value, b: &Value, op: impl Fn(f64, f64) -> f64) -> Option<Value> {
    let x = a.as_ieee_f64()?;
    let y = b.as_ieee_f64()?;
    Some(Value::Number(op(x, y)))
}

#[inline]
fn ieee_binop_f64_null_zero(a: &Value, b: &Value, op: impl Fn(f64, f64) -> f64) -> Option<Value> {
    let x = as_ieee_null_as_zero(a)?;
    let y = as_ieee_null_as_zero(b)?;
    Some(Value::Number(op(x, y)))
}

#[inline]
fn int_i64_from_tv(tv: TaggedValue) -> Option<i64> {
    match tv.int_value_domain()? {
        IntValue::Finite(n) => Some(n),
        _ => None,
    }
}

#[inline]
fn f64_whole_to_i64_exact(f: f64) -> Option<i64> {
    if !f.is_finite() || !number_is_int_surface(f) {
        return None;
    }
    let i = f as i64;
    if (i as f64) == f { Some(i) } else { None }
}

#[inline]
fn tagged_is_non_whole_number(tv: TaggedValue) -> bool {
    tv.is_number() && f64_whole_to_i64_exact(tv.get_f64()).is_none()
}

/// Int fast-path for `+`/`-`: skip only when either side is a non-whole float surface.
#[inline]
fn allows_int_add_sub(a_tv: TaggedValue, b_tv: TaggedValue) -> bool {
    !tagged_is_non_whole_number(a_tv) && !tagged_is_non_whole_number(b_tv)
}

/// Int fast-path for `*`/`%`: allow whole float + int as long as neither side is a fractional float.
#[inline]
fn allows_int_mul_mod(a_tv: TaggedValue, b_tv: TaggedValue) -> bool {
    !tagged_is_non_whole_number(a_tv) && !tagged_is_non_whole_number(b_tv)
}

#[inline]
fn int_i64_from_tv_or_value(
    tv: TaggedValue,
    value_store: &ValueStore,
    heavy_store: &HeavyStore,
) -> Option<i64> {
    if let Some(n) = int_i64_from_tv(tv) {
        return Some(n);
    }
    if tv.is_number() {
        return f64_whole_to_i64_exact(tv.get_f64());
    }
    if tv.is_heap() {
        let v = load_value(tv.get_heap_id(), value_store, heavy_store);
        return integer_value_as_i64_if_whole(&v);
    }
    None
}

#[inline]
fn push_int_i64_result(
    stack: &mut Vec<TaggedValue>,
    result: i64,
    value_store: &mut ValueStore,
    heavy_store: &mut HeavyStore,
) {
    if let Some(tv) = TaggedValue::try_from_int_iv(IntValue::Finite(result)) {
        stack::push(stack, tv);
    } else {
        stack::push_id(
            stack,
            store_value(Value::Int(IntValue::Finite(result)), value_store, heavy_store),
        );
    }
}

#[inline]
fn try_int_binop_values(a: &Value, b: &Value, op: fn(i64, i64) -> i64) -> Option<Value> {
    let ai = match a {
        Value::Int(IntValue::Finite(n)) => *n,
        _ => return None,
    };
    let bi = match b {
        Value::Int(IntValue::Finite(n)) => *n,
        _ => return None,
    };
    Some(Value::Int(IntValue::Finite(op(ai, bi))))
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
            stack::push(
                stack,
                TaggedValue::from_f64(a_tv.get_f64() + b_tv.get_f64()),
            );
            return Ok(VMStatus::Continue);
        }
        if cache_hit {
            frame.add_cache_both_number = false;
        }
        if a_tv.is_number() && b_tv.is_number() {
            frame.add_cache_ip = Some(current_ip);
            frame.add_cache_both_number = true;
            if let (Some(a), Some(b)) = (
                f64_whole_to_i64_exact(a_tv.get_f64()),
                f64_whole_to_i64_exact(b_tv.get_f64()),
            ) {
                frame.add_cache_both_number = false;
                push_int_i64_result(stack, a.wrapping_add(b), value_store, heavy_store);
                return Ok(VMStatus::Continue);
            }
            stack::push(
                stack,
                TaggedValue::from_f64(a_tv.get_f64() + b_tv.get_f64()),
            );
            return Ok(VMStatus::Continue);
        }
        if allows_int_add_sub(a_tv, b_tv) {
            if let (Some(a), Some(b)) = (
                int_i64_from_tv_or_value(a_tv, value_store, heavy_store),
                int_i64_from_tv_or_value(b_tv, value_store, heavy_store),
            ) {
                frame.add_cache_both_number = false;
                push_int_i64_result(stack, a.wrapping_add(b), value_store, heavy_store);
                return Ok(VMStatus::Continue);
            }
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
    if let Some(status) = try_special_binary_op(&a, &b, "@add", stack, value_store, heavy_store)? {
        return Ok(status);
    }
    if let Some(v) = try_int_binop_values(&a, &b, |x, y| x.wrapping_add(y)) {
        stack::push_id(stack, store_value(v, value_store, heavy_store));
        return Ok(VMStatus::Continue);
    }
    let mixed_float_int = (matches!(a, Value::Number(_)) && matches!(b, Value::Int(_)))
        || (matches!(b, Value::Number(_)) && matches!(a, Value::Int(_)));
    if !mixed_float_int {
        if let (Some(ai), Some(bi)) = (
            integer_value_as_i64_if_whole(&a),
            integer_value_as_i64_if_whole(&b),
        ) {
            stack::push_id(
                stack,
                store_value(
                    Value::Int(IntValue::Finite(ai.wrapping_add(bi))),
                    value_store,
                    heavy_store,
                ),
            );
            return Ok(VMStatus::Continue);
        }
    }
    if let Some(v) = ieee_binop_f64_null_zero(&a, &b, |x, y| x + y) {
        stack::push_id(stack, store_value(v, value_store, heavy_store));
        return Ok(VMStatus::Continue);
    }
    let result = operations::binary_add(
        &a,
        &b,
        frames,
        stack,
        exception_handlers,
        value_store,
        heavy_store,
    )?;
    stack::push_id(stack, store_value(result, value_store, heavy_store));
    Ok(VMStatus::Continue)
}

pub fn op_reg_add(
    rd: u8,
    r1: u8,
    r2: u8,
    frames: &mut Vec<CallFrame>,
) -> Result<VMStatus, LangError> {
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
        if let (Some(a), Some(b)) = (
            f64_whole_to_i64_exact(a_tv.get_f64()),
            f64_whole_to_i64_exact(b_tv.get_f64()),
        ) {
            push_int_i64_result(stack, a.wrapping_sub(b), value_store, heavy_store);
            return Ok(VMStatus::Continue);
        }
        stack::push(
            stack,
            TaggedValue::from_f64(a_tv.get_f64() - b_tv.get_f64()),
        );
        return Ok(VMStatus::Continue);
    }
    if allows_int_add_sub(a_tv, b_tv) {
        if let (Some(a), Some(b)) = (
            int_i64_from_tv_or_value(a_tv, value_store, heavy_store),
            int_i64_from_tv_or_value(b_tv, value_store, heavy_store),
        ) {
            push_int_i64_result(stack, a.wrapping_sub(b), value_store, heavy_store);
            return Ok(VMStatus::Continue);
        }
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
    if let Some(status) = try_special_binary_op(&a, &b, "@sub", stack, value_store, heavy_store)? {
        return Ok(status);
    }
    if let Some(v) = try_int_binop_values(&a, &b, |x, y| x.wrapping_sub(y)) {
        stack::push_id(stack, store_value(v, value_store, heavy_store));
        return Ok(VMStatus::Continue);
    }
    let result = if let Some(v) = ieee_binop_f64_null_zero(&a, &b, |x, y| x - y) {
        v
    } else {
        operations::binary_sub(
            &a,
            &b,
            frames,
            stack,
            exception_handlers,
            value_store,
            heavy_store,
        )?
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
        if let (Some(a), Some(b)) = (
            f64_whole_to_i64_exact(a_tv.get_f64()),
            f64_whole_to_i64_exact(b_tv.get_f64()),
        ) {
            push_int_i64_result(stack, a.wrapping_mul(b), value_store, heavy_store);
            return Ok(VMStatus::Continue);
        }
        stack::push(
            stack,
            TaggedValue::from_f64(a_tv.get_f64() * b_tv.get_f64()),
        );
        return Ok(VMStatus::Continue);
    }
    if allows_int_mul_mod(a_tv, b_tv) {
        if let (Some(a), Some(b)) = (
            int_i64_from_tv_or_value(a_tv, value_store, heavy_store),
            int_i64_from_tv_or_value(b_tv, value_store, heavy_store),
        ) {
            push_int_i64_result(stack, a.wrapping_mul(b), value_store, heavy_store);
            return Ok(VMStatus::Continue);
        }
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
    if let Some(status) = try_special_binary_op(&a, &b, "@mul", stack, value_store, heavy_store)? {
        return Ok(status);
    }
    if let Some(v) = try_int_binop_values(&a, &b, |x, y| x.wrapping_mul(y)) {
        stack::push_id(stack, store_value(v, value_store, heavy_store));
        return Ok(VMStatus::Continue);
    }
    if let Some(v) = ieee_binop_f64(&a, &b, |x, y| x * y) {
        stack::push_id(stack, store_value(v, value_store, heavy_store));
        return Ok(VMStatus::Continue);
    }
    let result = operations::binary_mul(
            &a,
            &b,
            frames,
            stack,
            exception_handlers,
            value_store,
            heavy_store,
        )?;
    stack::push_id(stack, store_value(result, value_store, heavy_store));
    Ok(VMStatus::Continue)
}

pub fn op_binary_op(
    const_idx: usize,
    current_ip: usize,
    stack: &mut Vec<TaggedValue>,
    frames: &mut Vec<CallFrame>,
    exception_handlers: &mut Vec<ExceptionHandler>,
    value_store: &mut ValueStore,
    heavy_store: &mut HeavyStore,
) -> Result<VMStatus, LangError> {
    let b_tv = stack::pop(stack, frames, exception_handlers, value_store, heavy_store)?;
    let a_tv = stack::pop(stack, frames, exception_handlers, value_store, heavy_store)?;
    let op_name = {
        let chunk = &frames.last().unwrap().function.chunk;
        match chunk.constants.get(const_idx) {
            Some(Value::String(s)) => s.clone(),
            _ => {
                let line = chunk.get_line(current_ip.saturating_sub(1));
                return Err(LangError::runtime_error(
                    format!("BinaryOp: bad constant index {}", const_idx),
                    line,
                ));
            }
        }
    };
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
    let result = operations::exec_binary_op_by_name(
        op_name.as_str(),
        &a,
        &b,
        frames,
        stack,
        exception_handlers,
        value_store,
        heavy_store,
    )?;
    stack::push_id(stack, store_value(result, value_store, heavy_store));
    Ok(VMStatus::Continue)
}

pub fn op_matmul(
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
        stack::push(
            stack,
            TaggedValue::from_f64(a_tv.get_f64() * b_tv.get_f64()),
        );
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
    if let Some(v) = ieee_binop_f64(&a, &b, |x, y| x * y) {
        stack::push_id(stack, store_value(v, value_store, heavy_store));
        return Ok(VMStatus::Continue);
    }
    let result = operations::binary_matmul(
        &a,
        &b,
        frames,
        stack,
        exception_handlers,
        value_store,
        heavy_store,
    )?;
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
        let n1 = a_tv.get_f64();
        let n2 = b_tv.get_f64();
        if n2 == 0.0 {
            let a_val = Value::Number(n1);
            let b_val = Value::Number(n2);
            if divide_by_zero_raises(&a_val, &b_val) {
                let line = frames
                    .last()
                    .map(|f| {
                        if f.ip > 0 {
                            f.function.chunk.get_line(f.ip - 1)
                        } else {
                            0
                        }
                    })
                    .unwrap_or(0);
                let error = ExceptionHandler::runtime_error(
                    frames,
                    "Division by zero".to_string(),
                    line,
                );
                return ExceptionHandler::handle_exception_vm(
                    stack,
                    frames,
                    exception_handlers,
                    error,
                    value_store,
                    heavy_store,
                );
            }
            stack::push(stack, TaggedValue::from_f64(n1 / n2));
            return Ok(VMStatus::Continue);
        }
        stack::push(stack, TaggedValue::from_f64(n1 / n2));
        return Ok(VMStatus::Continue);
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
    if let Some(status) = try_special_binary_op(&a, &b, "@div", stack, value_store, heavy_store)? {
        return Ok(status);
    }
    if let (Some(x), Some(y)) = (a.as_ieee_f64(), b.as_ieee_f64()) {
        if y == 0.0 && divide_by_zero_raises(&a, &b) {
            let line = frames
                .last()
                .map(|f| {
                    if f.ip > 0 {
                        f.function.chunk.get_line(f.ip - 1)
                    } else {
                        0
                    }
                })
                .unwrap_or(0);
            let error = ExceptionHandler::runtime_error(
                frames,
                "Division by zero".to_string(),
                line,
            );
            return ExceptionHandler::handle_exception_vm(
                stack,
                frames,
                exception_handlers,
                error,
                value_store,
                heavy_store,
            );
        }
        stack::push_id(
            stack,
            store_value(ieee_div_quotient_value(&a, &b, x, y), value_store, heavy_store),
        );
        return Ok(VMStatus::Continue);
    }
    let result = operations::binary_div(
            &a,
            &b,
            frames,
            stack,
            exception_handlers,
            value_store,
            heavy_store,
        )?;
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
    if let (Some(x), Some(y)) = (a.as_ieee_f64(), b.as_ieee_f64()) {
        if y == 0.0 {
            let line = frames
                .last()
                .map(|f| {
                    if f.ip > 0 {
                        f.function.chunk.get_line(f.ip - 1)
                    } else {
                        0
                    }
                })
                .unwrap_or(0);
            let error = ExceptionHandler::runtime_error(
                frames,
                "Division by zero".to_string(),
                line,
            );
            return ExceptionHandler::handle_exception_vm(
                stack,
                frames,
                exception_handlers,
                error,
                value_store,
                heavy_store,
            );
        }
        stack::push_id(
            stack,
            store_value(Value::Number((x / y).floor()), value_store, heavy_store),
        );
        return Ok(VMStatus::Continue);
    }
    let result = operations::binary_int_div(
        &a,
        &b,
        frames,
        stack,
        exception_handlers,
        value_store,
        heavy_store,
    )?;
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
            if let (Some(a), Some(b)) = (
                f64_whole_to_i64_exact(a_tv.get_f64()),
                f64_whole_to_i64_exact(n2),
            ) {
                push_int_i64_result(stack, floor_mod_i64(a, b), value_store, heavy_store);
                return Ok(VMStatus::Continue);
            }
            let x = a_tv.get_f64();
            let q = (x / n2).floor();
            stack::push(stack, TaggedValue::from_f64(x - q * n2));
            return Ok(VMStatus::Continue);
        }
    }
    if allows_int_mul_mod(a_tv, b_tv) {
        if let (Some(a), Some(b)) = (
            int_i64_from_tv_or_value(a_tv, value_store, heavy_store),
            int_i64_from_tv_or_value(b_tv, value_store, heavy_store),
        ) {
            if b == 0 {
                let line = frames
                    .last()
                    .map(|f| if f.ip > 0 { f.function.chunk.get_line(f.ip - 1) } else { 0 })
                    .unwrap_or(0);
                let error = ExceptionHandler::runtime_error(frames, "Modulo by zero".to_string(), line);
                return ExceptionHandler::handle_exception_vm(
                    stack,
                    frames,
                    exception_handlers,
                    error,
                    value_store,
                    heavy_store,
                );
            }
            push_int_i64_result(stack, floor_mod_i64(a, b), value_store, heavy_store);
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
    if let Some(status) = try_special_binary_op(&a, &b, "@mod", stack, value_store, heavy_store)? {
        return Ok(status);
    }
    if let (Some(ai), Some(bi)) = (
        match &a {
            Value::Int(IntValue::Finite(n)) => Some(*n),
            _ => None,
        },
        match &b {
            Value::Int(IntValue::Finite(n)) => Some(*n),
            _ => None,
        },
    ) {
        if bi == 0 {
            let line = frames
                .last()
                .map(|f| if f.ip > 0 { f.function.chunk.get_line(f.ip - 1) } else { 0 })
                .unwrap_or(0);
            let error = ExceptionHandler::runtime_error(frames, "Modulo by zero".to_string(), line);
            return ExceptionHandler::handle_exception_vm(
                stack,
                frames,
                exception_handlers,
                error,
                value_store,
                heavy_store,
            );
        }
        stack::push_id(
            stack,
            store_value(
                Value::Int(IntValue::Finite(floor_mod_i64(ai, bi))),
                value_store,
                heavy_store,
            ),
        );
        return Ok(VMStatus::Continue);
    }
    if let (Some(x), Some(y)) = (a.as_ieee_f64(), b.as_ieee_f64()) {
        if y == 0.0 {
            let line = frames
                .last()
                .map(|f| {
                    if f.ip > 0 {
                        f.function.chunk.get_line(f.ip - 1)
                    } else {
                        0
                    }
                })
                .unwrap_or(0);
            let error = ExceptionHandler::runtime_error(frames, "Modulo by zero".to_string(), line);
            return ExceptionHandler::handle_exception_vm(
                stack,
                frames,
                exception_handlers,
                error,
                value_store,
                heavy_store,
            );
        }
        let q = (x / y).floor();
        stack::push_id(
            stack,
            store_value(Value::Number(x - q * y), value_store, heavy_store),
        );
        return Ok(VMStatus::Continue);
    }
    let result = operations::binary_mod(
        &a,
        &b,
        frames,
        stack,
        exception_handlers,
        value_store,
        heavy_store,
    )?;
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
        stack::push(
            stack,
            TaggedValue::from_f64(a_tv.get_f64().powf(b_tv.get_f64())),
        );
        return Ok(VMStatus::Continue);
    }
    let a_id = tagged_to_value_id(a_tv, value_store);
    let b_id = tagged_to_value_id(b_tv, value_store);
    let a = load_value(a_id, value_store, heavy_store);
    let b = load_value(b_id, value_store, heavy_store);
    if let Some(status) = try_special_binary_op(&a, &b, "@pow", stack, value_store, heavy_store)? {
        return Ok(status);
    }
    if let Some(v) = ieee_binop_f64(&a, &b, |x, y| x.powf(y)) {
        stack::push_id(stack, store_value(v, value_store, heavy_store));
        return Ok(VMStatus::Continue);
    }
    let result = operations::binary_pow(
        &a,
        &b,
        frames,
        stack,
        exception_handlers,
        value_store,
        heavy_store,
    )?;
    stack::push_id(stack, store_value(result, value_store, heavy_store));
    Ok(VMStatus::Continue)
}

pub fn op_abs_i32(
    stack: &mut Vec<TaggedValue>,
    frames: &mut Vec<CallFrame>,
    exception_handlers: &mut Vec<ExceptionHandler>,
    value_store: &mut ValueStore,
    heavy_store: &mut HeavyStore,
) -> Result<VMStatus, LangError> {
    let val_tv = stack::pop(stack, frames, exception_handlers, value_store, heavy_store)?;
    if val_tv.is_int() {
        let n = val_tv.get_i32();
        let abs_n = if n == i32::MIN {
            i64::from(n).unsigned_abs() as f64
        } else {
            n.unsigned_abs() as f64
        };
        stack::push(stack, TaggedValue::from_f64(abs_n));
        return Ok(VMStatus::Continue);
    }
    if val_tv.is_number() {
        let n = val_tv.get_f64();
        if n.is_finite() {
            stack::push(stack, TaggedValue::from_f64(n.abs()));
            return Ok(VMStatus::Continue);
        }
    }
    let val_id = tagged_to_value_id(val_tv, value_store);
    let value = load_value(val_id, value_store, heavy_store);
    if let Some(n) = value.as_ieee_f64() {
        stack::push(stack, TaggedValue::from_f64(n.abs()));
        return Ok(VMStatus::Continue);
    }
    stack::push(stack, val_tv);
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
    let result = operations::unary_negate(
        &value,
        frames,
        stack,
        exception_handlers,
        value_store,
        heavy_store,
    )?;
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
