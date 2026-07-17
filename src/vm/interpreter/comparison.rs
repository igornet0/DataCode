// Comparison opcodes: Equal, NotEqual, Greater, Less, GreaterEqual, LessEqual, In.
// Logic preserved 1:1 from executor.rs — no semantic changes.

use crate::common::{
    error::LangError,
    numeric::{
        numeric_eq_int_float, tagged_integral_canonical_if_whole, FloatValue, IntValue,
    },
    value::Value,
    value_store::{ValueCell, ValueStore},
    TaggedValue,
};
use crate::vm::exceptions::ExceptionHandler;
use crate::vm::frame::CallFrame;
use crate::vm::heavy_store::HeavyStore;
use crate::vm::memory::{
    load_value, store_value, tagged_to_value_id,
};
use crate::vm::operations;
use crate::vm::set_ops::sets_equal;
use crate::vm::stack;
use crate::vm::types::VMStatus;

/// `int` / `number` / ±∞ on stack tags — same rules as [`Value`] `PartialEq` (A* stale-check `current_f != f_score.get(current)`).
fn try_tagged_numeric_equal(a: TaggedValue, b: TaggedValue) -> Option<bool> {
    let int_from = |tv: TaggedValue| -> Option<IntValue> {
        if tv.is_int() {
            Some(IntValue::Finite(tv.get_i32() as i64))
        } else if tv.is_int_pos_inf() {
            Some(IntValue::PosInfinity)
        } else if tv.is_int_neg_inf() {
            Some(IntValue::NegInfinity)
        } else {
            None
        }
    };
    let float_from = |tv: TaggedValue| -> Option<FloatValue> {
        if tv.is_number() {
            Some(FloatValue::from_f64_for_stack(tv.get_f64()))
        } else {
            None
        }
    };
    match (int_from(a), float_from(a), int_from(b), float_from(b)) {
        (None, Some(af), None, Some(bf)) => Some(af == bf),
        (Some(ai), None, Some(bi), None) => Some(ai == bi),
        (Some(ai), None, None, Some(bf)) => Some(numeric_eq_int_float(ai, bf)),
        (None, Some(af), Some(bi), None) => Some(numeric_eq_int_float(bi, af)),
        _ => None,
    }
}

fn try_special_compare(
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
        crate::vm::special_methods::try_dispatch_binary_compare(a, method, b)?
    {
        stack::push_id(stack, store_value(result, value_store, heavy_store));
        return Ok(Some(VMStatus::Continue));
    }
    Ok(None)
}

pub fn op_equal(
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
            TaggedValue::from_bool(a_tv.get_f64() == b_tv.get_f64()),
        );
        return Ok(VMStatus::Continue);
    }
    if a_tv.is_bool() && b_tv.is_bool() {
        stack::push(
            stack,
            TaggedValue::from_bool(a_tv.get_bool() == b_tv.get_bool()),
        );
        return Ok(VMStatus::Continue);
    }
    if a_tv.is_null() && b_tv.is_null() {
        stack::push(stack, TaggedValue::from_bool(true));
        return Ok(VMStatus::Continue);
    }
    if let Some(eq) = try_tagged_numeric_equal(a_tv, b_tv) {
        stack::push(stack, TaggedValue::from_bool(eq));
        return Ok(VMStatus::Continue);
    }
    let a_id = tagged_to_value_id(a_tv, value_store);
    let b_id = tagged_to_value_id(b_tv, value_store);
    let a = load_value(a_id, value_store, heavy_store);
    let b = load_value(b_id, value_store, heavy_store);
    if let Some(status) = try_special_compare(&a, &b, "@eq", stack, value_store, heavy_store)? {
        return Ok(status);
    }
    let result = match (&a, &b) {
        (Value::Number(n1), Value::Number(n2)) => Value::Bool(n1 == n2),
        (Value::Set(sa), Value::Set(sb)) => Value::Bool(sets_equal(
            &sa.borrow(),
            &sb.borrow(),
            value_store,
            heavy_store,
        )),
        _ => operations::binary_equal(&a, &b),
    };
    stack::push_id(stack, store_value(result, value_store, heavy_store));
    Ok(VMStatus::Continue)
}

pub fn op_not_equal(
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
            TaggedValue::from_bool(a_tv.get_f64() != b_tv.get_f64()),
        );
        return Ok(VMStatus::Continue);
    }
    if a_tv.is_bool() && b_tv.is_bool() {
        stack::push(
            stack,
            TaggedValue::from_bool(a_tv.get_bool() != b_tv.get_bool()),
        );
        return Ok(VMStatus::Continue);
    }
    if a_tv.is_null() && b_tv.is_null() {
        stack::push(stack, TaggedValue::from_bool(false));
        return Ok(VMStatus::Continue);
    }
    if let Some(eq) = try_tagged_numeric_equal(a_tv, b_tv) {
        stack::push(stack, TaggedValue::from_bool(!eq));
        return Ok(VMStatus::Continue);
    }
    let a_id = tagged_to_value_id(a_tv, value_store);
    let b_id = tagged_to_value_id(b_tv, value_store);
    let a = load_value(a_id, value_store, heavy_store);
    let b = load_value(b_id, value_store, heavy_store);
    if let Some(status) = try_special_compare(&a, &b, "@neq", stack, value_store, heavy_store)? {
        return Ok(status);
    }
    let result = match (&a, &b) {
        (Value::Number(n1), Value::Number(n2)) => Value::Bool(n1 != n2),
        (Value::Set(sa), Value::Set(sb)) => Value::Bool(!sets_equal(
            &sa.borrow(),
            &sb.borrow(),
            value_store,
            heavy_store,
        )),
        _ => operations::binary_not_equal(&a, &b),
    };
    stack::push_id(stack, store_value(result, value_store, heavy_store));
    Ok(VMStatus::Continue)
}

pub fn op_greater(
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
            TaggedValue::from_bool(a_tv.get_f64() > b_tv.get_f64()),
        );
        return Ok(VMStatus::Continue);
    }
    let a_id = tagged_to_value_id(a_tv, value_store);
    let b_id = tagged_to_value_id(b_tv, value_store);
    let a = load_value(a_id, value_store, heavy_store);
    let b = load_value(b_id, value_store, heavy_store);
    if let Some(status) = try_special_compare(&a, &b, "@gt", stack, value_store, heavy_store)? {
        return Ok(status);
    }
    let result = match (&a, &b) {
        (Value::Number(n1), Value::Number(n2)) => Value::Bool(n1 > n2),
        _ => operations::binary_greater(
            &a,
            &b,
            frames,
            stack,
            exception_handlers,
            value_store,
            heavy_store,
        )?,
    };
    stack::push_id(stack, store_value(result, value_store, heavy_store));
    Ok(VMStatus::Continue)
}

fn tagged_as_f64(tv: TaggedValue, store: &ValueStore) -> Option<f64> {
    if tv.is_number() {
        return Some(tv.get_f64());
    }
    if tv.is_int() {
        return Some(tv.get_i32() as f64);
    }
    if tv.is_int_pos_inf() {
        return Some(f64::INFINITY);
    }
    if tv.is_int_neg_inf() {
        return Some(f64::NEG_INFINITY);
    }
    if tv.is_heap() {
        return match store.get(tv.get_heap_id()) {
            Some(ValueCell::Number(n)) => Some(*n),
            Some(ValueCell::Int(iv)) => match iv {
                IntValue::Finite(n) => Some(*n as f64),
                IntValue::PosInfinity => Some(f64::INFINITY),
                IntValue::NegInfinity => Some(f64::NEG_INFINITY),
            },
            _ => None,
        };
    }
    None
}

fn try_tagged_numeric_less(a: TaggedValue, b: TaggedValue, store: &ValueStore) -> Option<bool> {
    let af = tagged_as_f64(a, store)?;
    let bf = tagged_as_f64(b, store)?;
    Some(af < bf)
}

pub fn op_less(
    stack: &mut Vec<TaggedValue>,
    frames: &mut Vec<CallFrame>,
    exception_handlers: &mut Vec<ExceptionHandler>,
    value_store: &mut ValueStore,
    heavy_store: &mut HeavyStore,
) -> Result<VMStatus, LangError> {
    let b_tv = stack::pop(stack, frames, exception_handlers, value_store, heavy_store)?;
    let a_tv = stack::pop(stack, frames, exception_handlers, value_store, heavy_store)?;
    if let Some(cmp) = try_tagged_numeric_less(a_tv, b_tv, value_store) {
        stack::push(stack, TaggedValue::from_bool(cmp));
        return Ok(VMStatus::Continue);
    }
    let a_id = tagged_to_value_id(a_tv, value_store);
    let b_id = tagged_to_value_id(b_tv, value_store);
    let a = load_value(a_id, value_store, heavy_store);
    let b = load_value(b_id, value_store, heavy_store);
    if let Some(status) = try_special_compare(&a, &b, "@lt", stack, value_store, heavy_store)? {
        return Ok(status);
    }
    let result = match (&a, &b) {
        (Value::Number(n1), Value::Number(n2)) => Value::Bool(n1 < n2),
        _ => operations::binary_less(
            &a,
            &b,
            frames,
            stack,
            exception_handlers,
            value_store,
            heavy_store,
        )?,
    };
    stack::push_id(stack, store_value(result, value_store, heavy_store));
    Ok(VMStatus::Continue)
}

pub fn op_greater_equal(
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
            TaggedValue::from_bool(a_tv.get_f64() >= b_tv.get_f64()),
        );
        return Ok(VMStatus::Continue);
    }
    let a_id = tagged_to_value_id(a_tv, value_store);
    let b_id = tagged_to_value_id(b_tv, value_store);
    let a = load_value(a_id, value_store, heavy_store);
    let b = load_value(b_id, value_store, heavy_store);
    if let Some(status) = try_special_compare(&a, &b, "@gte", stack, value_store, heavy_store)? {
        return Ok(status);
    }
    let result = match (&a, &b) {
        (Value::Number(n1), Value::Number(n2)) => Value::Bool(n1 >= n2),
        _ => operations::binary_greater_equal(
            &a,
            &b,
            frames,
            stack,
            exception_handlers,
            value_store,
            heavy_store,
        )?,
    };
    stack::push_id(stack, store_value(result, value_store, heavy_store));
    Ok(VMStatus::Continue)
}

pub fn op_less_equal(
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
            TaggedValue::from_bool(a_tv.get_f64() <= b_tv.get_f64()),
        );
        return Ok(VMStatus::Continue);
    }
    let a_id = tagged_to_value_id(a_tv, value_store);
    let b_id = tagged_to_value_id(b_tv, value_store);
    let a = load_value(a_id, value_store, heavy_store);
    let b = load_value(b_id, value_store, heavy_store);
    if let Some(status) = try_special_compare(&a, &b, "@lte", stack, value_store, heavy_store)? {
        return Ok(status);
    }
    let result = match (&a, &b) {
        (Value::Number(n1), Value::Number(n2)) => Value::Bool(n1 <= n2),
        _ => operations::binary_less_equal(
            &a,
            &b,
            frames,
            stack,
            exception_handlers,
            value_store,
            heavy_store,
        )?,
    };
    stack::push_id(stack, store_value(result, value_store, heavy_store));
    Ok(VMStatus::Continue)
}

pub fn op_in(
    line: usize,
    stack: &mut Vec<TaggedValue>,
    frames: &mut Vec<CallFrame>,
    exception_handlers: &mut Vec<ExceptionHandler>,
    value_store: &mut ValueStore,
    heavy_store: &mut HeavyStore,
) -> Result<VMStatus, LangError> {
    let container_tv =
        stack::pop(stack, frames, exception_handlers, value_store, heavy_store)?;
    let value_tv = stack::pop(stack, frames, exception_handlers, value_store, heavy_store)?;

    let container_id = tagged_to_value_id(container_tv, value_store);
    let value_id = tagged_to_value_id(value_tv, value_store);
    let container = load_value(container_id, value_store, heavy_store);
    let member = load_value(value_id, value_store, heavy_store);

    if crate::vm::special_methods::is_class_instance(&container)
        && crate::vm::special_methods::class_has_special(&container, "@contains")
    {
        if let Ok(Some(result)) = crate::vm::special_methods::dispatch_special(
            &container,
            "@contains",
            &[member.clone()],
        ) {
            stack::push_id(
                stack,
                store_value(result, value_store, heavy_store),
            );
            return Ok(VMStatus::Continue);
        }
    }

    match crate::vm::membership::value_in_container(&member, &container, line) {
        Ok(found) => {
            stack::push(stack, TaggedValue::from_bool(found));
            Ok(VMStatus::Continue)
        }
        Err(error) => match ExceptionHandler::handle_exception(
            stack,
            frames,
            exception_handlers,
            error,
            value_store,
            heavy_store,
        ) {
            Ok(()) => Ok(VMStatus::Continue),
            Err(e) => Err(e),
        },
    }
}

/// Integral-member `in` for plain set/dict (compiler peephole); falls back to [`op_in`].
pub fn op_in_integral(
    line: usize,
    stack: &mut Vec<TaggedValue>,
    frames: &mut Vec<CallFrame>,
    exception_handlers: &mut Vec<ExceptionHandler>,
    value_store: &mut ValueStore,
    heavy_store: &mut HeavyStore,
) -> Result<VMStatus, LangError> {
    op_in(
        line,
        stack,
        frames,
        exception_handlers,
        value_store,
        heavy_store,
    )
}

/// Integral `not in` for plain set/dict (`!x in s` peephole).
pub fn op_not_in_integral(
    line: usize,
    stack: &mut Vec<TaggedValue>,
    frames: &mut Vec<CallFrame>,
    exception_handlers: &mut Vec<ExceptionHandler>,
    value_store: &mut ValueStore,
    heavy_store: &mut HeavyStore,
) -> Result<VMStatus, LangError> {
    op_in(
        line,
        stack,
        frames,
        exception_handlers,
        value_store,
        heavy_store,
    )?;
    if let Some(sp) = stack::active_sp_for(stack) {
        if *sp > 0 {
            let idx = *sp - 1;
            if stack[idx].is_bool() {
                stack[idx] = TaggedValue::from_bool(!stack[idx].get_bool());
            }
        }
    } else if let Some(b) = stack.last_mut() {
        if b.is_bool() {
            *b = TaggedValue::from_bool(!b.get_bool());
        }
    }
    Ok(VMStatus::Continue)
}

/// Stack `[nr, nc, rows, cols]` (cols on top) → bool: `0 <= nr < rows and 0 <= nc < cols`.
pub fn op_in_grid_bounds(
    stack: &mut Vec<TaggedValue>,
    frames: &mut Vec<CallFrame>,
    exception_handlers: &mut Vec<ExceptionHandler>,
    value_store: &mut ValueStore,
    heavy_store: &mut HeavyStore,
) -> Result<VMStatus, LangError> {
    let cols_tv = stack::pop(stack, frames, exception_handlers, value_store, heavy_store)?;
    let rows_tv = stack::pop(stack, frames, exception_handlers, value_store, heavy_store)?;
    let nc_tv = stack::pop(stack, frames, exception_handlers, value_store, heavy_store)?;
    let nr_tv = stack::pop(stack, frames, exception_handlers, value_store, heavy_store)?;
    let in_bounds = match (
        grid_bound_to_i64(nr_tv, value_store),
        grid_bound_to_i64(nc_tv, value_store),
        grid_bound_to_i64(rows_tv, value_store),
        grid_bound_to_i64(cols_tv, value_store),
    ) {
        (Some(nr), Some(nc), Some(rows), Some(cols)) => {
            nr >= 0 && nr < rows && nc >= 0 && nc < cols
        }
        _ => false,
    };
    stack::push(stack, TaggedValue::from_bool(in_bounds));
    Ok(VMStatus::Continue)
}

/// Stack `[nr, nc, rows, cols]` → bool: out of grid (`!(in bounds)`).
pub fn op_in_grid_bounds_out(
    stack: &mut Vec<TaggedValue>,
    frames: &mut Vec<CallFrame>,
    exception_handlers: &mut Vec<ExceptionHandler>,
    value_store: &mut ValueStore,
    heavy_store: &mut HeavyStore,
) -> Result<VMStatus, LangError> {
    let cols_tv = stack::pop(stack, frames, exception_handlers, value_store, heavy_store)?;
    let rows_tv = stack::pop(stack, frames, exception_handlers, value_store, heavy_store)?;
    let nc_tv = stack::pop(stack, frames, exception_handlers, value_store, heavy_store)?;
    let nr_tv = stack::pop(stack, frames, exception_handlers, value_store, heavy_store)?;
    let in_bounds = match (
        grid_bound_to_i64(nr_tv, value_store),
        grid_bound_to_i64(nc_tv, value_store),
        grid_bound_to_i64(rows_tv, value_store),
        grid_bound_to_i64(cols_tv, value_store),
    ) {
        (Some(nr), Some(nc), Some(rows), Some(cols)) => {
            nr >= 0 && nr < rows && nc >= 0 && nc < cols
        }
        _ => false,
    };
    stack::push(stack, TaggedValue::from_bool(!in_bounds));
    Ok(VMStatus::Continue)
}

/// Stack `[current_f, dict, key]` → bool: stale heap entry (`current_f != dict.get(key)`).
pub fn op_f_score_stale_check(
    stack: &mut Vec<TaggedValue>,
    frames: &mut Vec<CallFrame>,
    exception_handlers: &mut Vec<ExceptionHandler>,
    value_store: &mut ValueStore,
    heavy_store: &mut HeavyStore,
) -> Result<VMStatus, LangError> {
    let key_tv = stack::pop(stack, frames, exception_handlers, value_store, heavy_store)?;
    let dict_tv = stack::pop(stack, frames, exception_handlers, value_store, heavy_store)?;
    let current_f = stack::pop(stack, frames, exception_handlers, value_store, heavy_store)?;
    let stored = grid_bound_to_i64(key_tv, value_store)
        .and_then(|key| integral_dict_lookup_tagged(dict_tv, key, value_store, heavy_store));
    let stale = match stored {
        None => true,
        Some(stored_tv) => {
            !tagged_numeric_eq(current_f, stored_tv, value_store, heavy_store)
        }
    };
    stack::push(stack, TaggedValue::from_bool(stale));
    Ok(VMStatus::Continue)
}

/// Stack `[tentative_g, dict, key]` → bool: `tentative_g < dict.get(key, +inf)`.
pub fn op_dict_get_integral_lt(
    stack: &mut Vec<TaggedValue>,
    frames: &mut Vec<CallFrame>,
    exception_handlers: &mut Vec<ExceptionHandler>,
    value_store: &mut ValueStore,
    heavy_store: &mut HeavyStore,
) -> Result<VMStatus, LangError> {
    let key_tv = stack::pop(stack, frames, exception_handlers, value_store, heavy_store)?;
    let dict_tv = stack::pop(stack, frames, exception_handlers, value_store, heavy_store)?;
    let tentative_tv = stack::pop(stack, frames, exception_handlers, value_store, heavy_store)?;
    let stored_tv = grid_bound_to_i64(key_tv, value_store)
        .and_then(|key| integral_dict_lookup_tagged(dict_tv, key, value_store, heavy_store))
        .unwrap_or_else(|| TaggedValue::from_f64(f64::INFINITY));
    let lt = tagged_numeric_lt(tentative_tv, stored_tv, value_store, heavy_store);
    stack::push(stack, TaggedValue::from_bool(lt));
    Ok(VMStatus::Continue)
}

/// Stack `[container, key]` → `container[key] + addend` (array or plain dict; missing → null+addend).
pub fn op_dict_index_integral_add_imm(
    addend: i8,
    stack: &mut Vec<TaggedValue>,
    frames: &mut Vec<CallFrame>,
    exception_handlers: &mut Vec<ExceptionHandler>,
    value_store: &mut ValueStore,
    heavy_store: &mut HeavyStore,
) -> Result<VMStatus, LangError> {
    let key_tv = stack::pop(stack, frames, exception_handlers, value_store, heavy_store)?;
    let container_tv = stack::pop(stack, frames, exception_handlers, value_store, heavy_store)?;
    let base = integral_subscript_lookup_tagged(
        container_tv,
        key_tv,
        value_store,
        heavy_store,
    )
    .unwrap_or(TaggedValue::null());
    stack::push(
        stack,
        tagged_add_i64(base, i64::from(addend), value_store, heavy_store),
    );
    Ok(VMStatus::Continue)
}

#[inline]
fn tagged_add_i64(
    tv: TaggedValue,
    addend: i64,
    store: &ValueStore,
    heap: &HeavyStore,
) -> TaggedValue {
    if tv.is_int() {
        return TaggedValue::from_i32((tv.get_i32() as i64).saturating_add(addend) as i32);
    }
    if tv.is_number() {
        return TaggedValue::from_f64(tv.get_f64() + addend as f64);
    }
    if tv.is_null() {
        return if (-(i32::MAX as i64)..=i32::MAX as i64).contains(&addend) {
            TaggedValue::from_i32(addend as i32)
        } else {
            TaggedValue::from_f64(addend as f64)
        };
    }
    if tv.is_heap() {
        if let Some(n) = load_value(tv.get_heap_id(), store, heap).as_ieee_f64() {
            return TaggedValue::from_f64(n + addend as f64);
        }
    }
    TaggedValue::from_i32(addend as i32)
}

#[inline]
fn grid_bound_to_i64(tv: TaggedValue, store: &mut ValueStore) -> Option<i64> {
    tagged_integral_canonical_if_whole(tv).or_else(|| {
        if tv.is_heap() {
            crate::vm::memory::canonical_integral_from_key_id_mut(tv.get_heap_id(), store)
        } else {
            None
        }
    })
}

/// Integral subscript for peephole `arr[i]+k` / `dict[i]+k` (compiler emits one opcode for both).
#[inline]
fn integral_subscript_lookup_tagged(
    container_tv: TaggedValue,
    key_tv: TaggedValue,
    store: &mut ValueStore,
    heap: &HeavyStore,
) -> Option<TaggedValue> {
    let key = grid_bound_to_i64(key_tv, store)?;
    if !container_tv.is_heap() {
        return None;
    }
    let container_id = container_tv.get_heap_id();
    if key >= 0 {
        if let Some(ValueCell::Array(ref vec)) = store.get(container_id) {
            let u = key as usize;
            if u < vec.len() {
                return Some(vec[u]);
            }
            return None;
        }
    }
    integral_dict_lookup_tagged(container_tv, key, store, heap)
}

#[inline]
fn integral_dict_lookup_tagged(
    dict_tv: TaggedValue,
    key: i64,
    store: &ValueStore,
    heap: &HeavyStore,
) -> Option<TaggedValue> {
    let _ = heap;
    if !dict_tv.is_heap() {
        return None;
    }
    let obj_id = dict_tv.get_heap_id();
    let ValueCell::Object(omap) = store.get(obj_id)? else {
        return None;
    };
    let slot = omap.find_integral_slot(key)?;
    match slot {
        crate::common::integral_map::IntegralSlot::Immediate(tv) => Some(tv),
        crate::common::integral_map::IntegralSlot::Heap(id) => Some(TaggedValue::from_heap(id)),
    }
}

#[inline]
fn tagged_f64(
    tv: TaggedValue,
    store: &ValueStore,
    heap: &HeavyStore,
) -> Option<f64> {
    if tv.is_number() {
        return Some(tv.get_f64());
    }
    if tv.is_int() {
        return Some(tv.get_i32() as f64);
    }
    if tv.is_heap() {
        return load_value(tv.get_heap_id(), store, heap).as_ieee_f64();
    }
    None
}

#[inline]
fn tagged_numeric_eq(
    a: TaggedValue,
    b: TaggedValue,
    store: &ValueStore,
    heap: &HeavyStore,
) -> bool {
    if let (Some(x), Some(y)) = (tagged_f64(a, store, heap), tagged_f64(b, store, heap)) {
        return x == y;
    }
    a.0 == b.0
}

#[inline]
fn tagged_numeric_lt(
    a: TaggedValue,
    b: TaggedValue,
    store: &ValueStore,
    heap: &HeavyStore,
) -> bool {
    match (
        tagged_f64(a, store, heap),
        tagged_f64(b, store, heap),
    ) {
        (Some(x), Some(y)) => x < y,
        _ => false,
    }
}

