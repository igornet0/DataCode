// Stack opcodes: Constant, LoadLocal, StoreLocal, Pop, Dup, FormatInterp.
// Logic preserved 1:1 from executor.rs — no semantic changes.

use crate::common::error::{ErrorType, LangError};
use crate::common::numeric::{divmod_f64, divmod_i64, integer_value_as_i64_if_whole, tagged_integral_canonical_if_whole};
use crate::common::value::Value;
use crate::common::value_store::{ValueCell, ValueStore};
use crate::common::TaggedValue;
use crate::vm::exceptions::ExceptionHandler;
use crate::vm::frame::CallFrame;
use crate::vm::heavy_store::HeavyStore;
use crate::vm::stack;
use crate::vm::array_view::materialize_array_view;
use crate::vm::store_convert::{load_value, slot_to_value, store_value};
use crate::vm::types::VMStatus;

use super::helpers::pop_to_value_id;

/// If `tv` is a table already bound in another local slot, store a new Heavy
/// handle that shares the same `Rc` so later CoW mutations do not alias `t1`.
fn fork_table_if_slot_aliased(
    tv: TaggedValue,
    slots: &[TaggedValue],
    dest_index: usize,
    value_store: &mut ValueStore,
    heavy_store: &mut HeavyStore,
) -> TaggedValue {
    if !tv.is_heap() {
        return tv;
    }
    let id = tv.get_heap_id();
    let is_table = matches!(
        value_store.get(id),
        Some(ValueCell::Heavy(h)) if matches!(heavy_store.get(*h), Some(Value::Table(_)))
    );
    if !is_table {
        return tv;
    }
    let aliased = slots
        .iter()
        .enumerate()
        .any(|(i, s)| i != dest_index && s.is_heap() && s.get_heap_id() == id);
    if !aliased {
        return tv;
    }
    let val = load_value(id, value_store, heavy_store);
    TaggedValue::from_heap(store_value(val, value_store, heavy_store))
}

/// Format a value for string interpolation with a spec like ".2f" or ".0f".
fn format_value_interp(value: &Value, spec: &str) -> String {
    let spec = spec.trim();
    if let Some(rest) = spec.strip_prefix('.') {
        if let Some(dot_f) = rest.find('f') {
            let prec_str = &rest[..dot_f];
            if prec_str.chars().all(|c| c.is_ascii_digit()) {
                let prec: usize = prec_str.parse().unwrap_or(6);
                if let Some(n) = value.as_ieee_f64() {
                    return format!("{:.*}", prec, n);
                }
            }
        }
    }
    value.to_string()
}

pub fn op_constant(
    index: usize,
    stack: &mut Vec<TaggedValue>,
    frame: &mut CallFrame,
) -> Result<VMStatus, LangError> {
    let len = frame.constant_ids.len();
    if index >= len {
        return Err(LangError::ParseError {
            message: format!(
                "Constant index {} out of bounds (chunk has {} constants); function '{}'",
                index, len, frame.function.name
            ),
            line: 0,
            file: frame.function.chunk.source_name.clone(),
        });
    }
    let tv = frame
        .constant_tagged
        .get(index)
        .and_then(|opt| *opt)
        .unwrap_or_else(|| TaggedValue::from_heap(frame.constant_ids[index]));
    stack::push(stack, tv);
    Ok(VMStatus::Continue)
}

pub fn op_load_local(
    index: usize,
    current_ip: usize,
    stack: &mut Vec<TaggedValue>,
    frames: &mut Vec<CallFrame>,
) -> Result<VMStatus, LangError> {
    let frame = frames.last_mut().unwrap();
    if frame.load_local_cache_ip == Some(current_ip) && frame.load_local_cache_slot == Some(index) {
        if let Some(tv) = frame.load_local_cache_tagged {
            stack::push(stack, tv);
            return Ok(VMStatus::Continue);
        }
    }
    let frame = frames.last_mut().unwrap();
    if index >= frame.slots.len() {
        frame.ensure_slot(index);
    }
    let tv = frame.slots[index];
    stack::push(stack, tv);
    {
        let frame = frames.last_mut().unwrap();
        frame.load_local_cache_ip = Some(current_ip);
        frame.load_local_cache_slot = Some(index);
        frame.load_local_cache_tagged = Some(tv);
    }
    Ok(VMStatus::Continue)
}

pub fn op_store_local(
    index: usize,
    stack: &mut Vec<TaggedValue>,
    frames: &mut Vec<CallFrame>,
    exception_handlers: &mut Vec<ExceptionHandler>,
    value_store: &mut ValueStore,
    heavy_store: &mut HeavyStore,
) -> Result<VMStatus, LangError> {
    let tv = stack::pop(stack, frames, exception_handlers, value_store, heavy_store)?;
    let tv = crate::vm::array_view::materialize_tagged_if_array_view(tv, value_store, heavy_store);
    let tv = {
        let slots = frames.last().map(|f| f.slots.as_slice()).unwrap_or(&[]);
        fork_table_if_slot_aliased(tv, slots, index, value_store, heavy_store)
    };
    let frame = frames.last_mut().unwrap();
    if frame.load_local_cache_slot == Some(index) {
        frame.load_local_cache_slot = None;
    }
    if index >= frame.slots.len() {
        frame.slots.resize(index + 1, TaggedValue::null());
    } else {
        let old = frame.slots[index];
        if old.is_heap() {
            let old_id = old.get_heap_id();
            if value_store.is_heapq_owned_pair(old_id) && !value_store.holds_scratch_heap_pair(old_id) {
                value_store.recycle_heap_pair(old_id);
            }
        }
    }
    frame.slots[index] = tv;
    frame.invalidate_inline_caches();
    // StoreLocal may allocate new heap ids (push, etc.); GetArrayElement inline cache is keyed only by IP — clear on all frames.
    for fr in frames.iter_mut() {
        fr.get_array_element_cache_ip = None;
        fr.get_array_element_cache_array_number = false;
        fr.get_array_element_cache_object_string = false;
        fr.get_array_element_cache_object_integral = false;
        fr.get_array_element_cache_container_tv = None;
        fr.get_array_element_cache_index_tv = None;
        fr.get_array_element_cache_integral_key = None;
    }

    if cfg!(debug_assertions) {
        let frame = frames.last().unwrap();
        if frame.function.name.contains("::new_") {
            let current_ip = frame.ip - 1;
            let key_count = if tv.is_heap() {
                match value_store.get(tv.get_heap_id()) {
                    Some(crate::common::value_store::ValueCell::Object(omap)) => Some(omap.len()),
                    Some(crate::common::value_store::ValueCell::Heavy(idx)) => {
                        heavy_store.get(*idx).and_then(|v| {
                            if let crate::common::value::Value::Object(obj_rc) = v {
                                Some(obj_rc.borrow().len())
                            } else {
                                None
                            }
                        })
                    }
                    _ => None,
                }
            } else {
                None
            };
            if let Some(n) = key_count {
                crate::debug_println!(
                    "[DEBUG StoreLocal] constructor '{}' IP {} slot {}: Object ({} keys)",
                    frame.function.name,
                    current_ip,
                    index,
                    n
                );
            }
        }
    }
    Ok(VMStatus::Continue)
}

pub fn op_pop(
    stack: &mut Vec<TaggedValue>,
    frames: &mut Vec<CallFrame>,
) -> Result<VMStatus, LangError> {
    let frame_stack_start = frames.last().map(|f| f.stack_start);
    let sp_after = crate::vm::stack::with_sp(stack, |sp| {
        if let Some(stack_start) = frame_stack_start {
            crate::vm::stack::discard_top(sp, stack_start);
        } else if *sp > 0 {
            *sp -= 1;
        }
        *sp
    });
    let sync_start = frame_stack_start.unwrap_or(0);
    crate::vm::stack::truncate_if_at_frame(stack, sp_after, sync_start);
    Ok(VMStatus::Continue)
}

pub fn op_dup(
    stack: &mut Vec<TaggedValue>,
    frames: &mut Vec<CallFrame>,
    exception_handlers: &mut Vec<ExceptionHandler>,
    value_store: &mut ValueStore,
    heavy_store: &mut HeavyStore,
) -> Result<VMStatus, LangError> {
    let top = pop_to_value_id(stack, frames, exception_handlers, value_store, heavy_store)?;
    stack::push_id(stack, top);
    stack::push_id(stack, top);
    Ok(VMStatus::Continue)
}

/// FormatInterp(format_const_index): pop value, format with spec from constant, push string.
pub fn op_format_interp(
    format_index: usize,
    stack: &mut Vec<TaggedValue>,
    frames: &mut Vec<CallFrame>,
    exception_handlers: &mut Vec<ExceptionHandler>,
    value_store: &mut ValueStore,
    heavy_store: &mut HeavyStore,
) -> Result<VMStatus, LangError> {
    let value_id = pop_to_value_id(stack, frames, exception_handlers, value_store, heavy_store)?;
    let frame = frames.last_mut().unwrap();
    let format_id = frame
        .constant_ids
        .get(format_index)
        .copied()
        .ok_or_else(|| LangError::ParseError {
            message: format!("FormatInterp: constant index {} out of range", format_index),
            line: 0,
            file: None,
        })?;
    let value = load_value(value_id, value_store, heavy_store);
    let value = match &value {
        Value::ArrayView(av) => materialize_array_view(av, value_store, heavy_store),
        _ => value,
    };
    let format_spec = match load_value(format_id, value_store, heavy_store) {
        Value::String(s) => s,
        _ => String::new(),
    };
    let result = format_value_interp(&value, &format_spec);
    let result_id = store_value(Value::String(result), value_store, heavy_store);
    stack::push_id(stack, result_id);
    Ok(VMStatus::Continue)
}

pub fn op_heappush_flat(
    line: usize,
    stack: &mut Vec<TaggedValue>,
    frames: &mut Vec<CallFrame>,
    exception_handlers: &mut Vec<ExceptionHandler>,
    value_store: &mut ValueStore,
    heavy_store: &mut HeavyStore,
) -> Result<VMStatus, LangError> {
    let b = stack::pop(stack, frames, exception_handlers, value_store, heavy_store)?;
    let a = stack::pop(stack, frames, exception_handlers, value_store, heavy_store)?;
    let heap_tv = stack::pop(stack, frames, exception_handlers, value_store, heavy_store)?;
    let heap_id = if heap_tv.is_heap() {
        heap_tv.get_heap_id()
    } else {
        stack::push(stack, heap_tv);
        stack::push(stack, a);
        stack::push(stack, b);
        return Err(LangError::runtime_error(
            "TypeError: heappush expects heap array".to_string(),
            line,
        ));
    };
    match crate::vm::runtime::call_engine::heappush_two_slot_pair(
        value_store,
        heavy_store,
        heap_id,
        a,
        b,
    ) {
        Ok(()) => Ok(VMStatus::Continue),
        Err(msg) => {
            let error = ExceptionHandler::runtime_error(&frames, msg, line);
            match ExceptionHandler::handle_exception(
                stack,
                frames,
                exception_handlers,
                error,
                value_store,
                heavy_store,
            ) {
                Ok(()) => Ok(VMStatus::Continue),
                Err(e) => Err(e),
            }
        }
    }
}

pub fn op_heappop_unpack2(
    f_slot: usize,
    n_slot: usize,
    line: usize,
    stack: &mut Vec<TaggedValue>,
    frames: &mut Vec<CallFrame>,
    exception_handlers: &mut Vec<ExceptionHandler>,
    value_store: &mut ValueStore,
    heavy_store: &mut HeavyStore,
) -> Result<VMStatus, LangError> {
    let heap_tv = stack::pop(stack, frames, exception_handlers, value_store, heavy_store)?;
    let heap_id = if heap_tv.is_heap() {
        heap_tv.get_heap_id()
    } else {
        stack::push(stack, heap_tv);
        return Err(LangError::runtime_error(
            "TypeError: heappop expects heap array".to_string(),
            line,
        ));
    };
    match crate::vm::runtime::call_engine::heappop_unpack2_locals(
        value_store,
        heavy_store,
        heap_id,
    ) {
        Ok((f_tv, n_tv)) => {
            let frame = frames.last_mut().unwrap();
            if f_slot >= frame.slots.len() {
                frame.ensure_slot(f_slot);
            }
            if n_slot >= frame.slots.len() {
                frame.ensure_slot(n_slot);
            }
            if let Some(old) = frame.slots.get(f_slot).copied() {
                if old.is_heap() {
                    let old_id = old.get_heap_id();
                    if value_store.is_heapq_owned_pair(old_id) {
                        value_store.recycle_heap_pair(old_id);
                    }
                }
            }
            frame.slots[f_slot] = f_tv;
            frame.slots[n_slot] = n_tv;
            Ok(VMStatus::Continue)
        }
        Err(msg) => {
            let err_type = if msg.contains("IndexError") {
                ErrorType::IndexError
            } else {
                ErrorType::TypeError
            };
            let error = ExceptionHandler::runtime_error_with_type(
                frames,
                msg,
                line,
                err_type,
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
    }
}

pub fn op_divmod_unpack2(
    q_slot: usize,
    r_slot: usize,
    line: usize,
    stack: &mut Vec<TaggedValue>,
    frames: &mut Vec<CallFrame>,
    exception_handlers: &mut Vec<ExceptionHandler>,
    value_store: &mut ValueStore,
    heavy_store: &mut HeavyStore,
) -> Result<VMStatus, LangError> {
    let b_tv = stack::pop(stack, frames, exception_handlers, value_store, heavy_store)?;
    let a_tv = stack::pop(stack, frames, exception_handlers, value_store, heavy_store)?;

    let integral_operand = |tv: TaggedValue| -> Option<i64> {
        tagged_integral_canonical_if_whole(tv).or_else(|| {
            let v = slot_to_value(tv, value_store, heavy_store);
            integer_value_as_i64_if_whole(&v)
        })
    };

    let (q_tv, r_tv) = if let (Some(ai), Some(bi)) = (integral_operand(a_tv), integral_operand(b_tv)) {
        if bi == 0 {
            return Err(LangError::runtime_error(
                "ZeroDivisionError: integer division or modulo by zero".to_string(),
                line,
            ));
        }
        let (q, r) = divmod_i64(ai, bi);
        (
            TaggedValue::from_f64(q as f64),
            TaggedValue::from_f64(r as f64),
        )
    } else if a_tv.is_number() && b_tv.is_number() {
        let y = b_tv.get_f64();
        if y == 0.0 {
            return Err(LangError::runtime_error(
                "ZeroDivisionError: float division or modulo by zero".to_string(),
                line,
            ));
        }
        let (q, r) = divmod_f64(a_tv.get_f64(), y);
        (TaggedValue::from_f64(q), TaggedValue::from_f64(r))
    } else {
        return Err(LangError::runtime_error(
            "TypeError: divmod() expected numeric operands".to_string(),
            line,
        ));
    };

    let frame = frames.last_mut().unwrap();
    if q_slot >= frame.slots.len() {
        frame.ensure_slot(q_slot);
    }
    if r_slot >= frame.slots.len() {
        frame.ensure_slot(r_slot);
    }
    frame.slots[q_slot] = q_tv;
    frame.slots[r_slot] = r_tv;
    Ok(VMStatus::Continue)
}

pub fn op_object_get_integral(
    line: usize,
    stack: &mut Vec<TaggedValue>,
    frames: &mut Vec<CallFrame>,
    exception_handlers: &mut Vec<ExceptionHandler>,
    value_store: &mut ValueStore,
    heavy_store: &mut HeavyStore,
) -> Result<VMStatus, LangError> {
    crate::vm::runtime::call_engine::object_get_from_stack(
        line,
        stack,
        frames,
        exception_handlers,
        value_store,
        heavy_store,
    )
}

pub fn op_set_discard_integral(
    line: usize,
    stack: &mut Vec<TaggedValue>,
    frames: &mut Vec<CallFrame>,
    exception_handlers: &mut Vec<ExceptionHandler>,
    value_store: &mut ValueStore,
    heavy_store: &mut HeavyStore,
) -> Result<VMStatus, LangError> {
    crate::vm::runtime::call_engine::set_mut_integral_from_stack(
        crate::vm::runtime::call_engine::SetIntegralMut::Discard,
        line,
        stack,
        frames,
        exception_handlers,
        value_store,
        heavy_store,
    )
}

pub fn op_object_clear(
    line: usize,
    stack: &mut Vec<TaggedValue>,
    frames: &mut Vec<CallFrame>,
    exception_handlers: &mut Vec<ExceptionHandler>,
    value_store: &mut ValueStore,
    heavy_store: &mut HeavyStore,
) -> Result<VMStatus, LangError> {
    crate::vm::runtime::call_engine::object_clear_from_stack(
        line,
        stack,
        frames,
        exception_handlers,
        value_store,
        heavy_store,
    )
}

pub fn op_set_add_integral(
    line: usize,
    stack: &mut Vec<TaggedValue>,
    frames: &mut Vec<CallFrame>,
    exception_handlers: &mut Vec<ExceptionHandler>,
    value_store: &mut ValueStore,
    heavy_store: &mut HeavyStore,
) -> Result<VMStatus, LangError> {
    crate::vm::runtime::call_engine::set_mut_integral_from_stack(
        crate::vm::runtime::call_engine::SetIntegralMut::Add,
        line,
        stack,
        frames,
        exception_handlers,
        value_store,
        heavy_store,
    )
}

pub fn op_heappop_flat(
    line: usize,
    stack: &mut Vec<TaggedValue>,
    frames: &mut Vec<CallFrame>,
    exception_handlers: &mut Vec<ExceptionHandler>,
    value_store: &mut ValueStore,
    heavy_store: &mut HeavyStore,
) -> Result<VMStatus, LangError> {
    let heap_tv = stack::pop(stack, frames, exception_handlers, value_store, heavy_store)?;
    let heap_id = if heap_tv.is_heap() {
        heap_tv.get_heap_id()
    } else {
        stack::push(stack, heap_tv);
        return Err(LangError::runtime_error(
            "TypeError: heappop expects heap array".to_string(),
            line,
        ));
    };
    match crate::vm::runtime::call_engine::heappop_store(
        value_store,
        heavy_store,
        heap_id,
    ) {
        Ok(root) => {
            stack::push(stack, root);
            Ok(VMStatus::Continue)
        }
        Err(msg) => {
            let err_type = if msg.contains("IndexError") {
                ErrorType::IndexError
            } else {
                ErrorType::TypeError
            };
            let error = ExceptionHandler::runtime_error_with_type(
                frames,
                msg,
                line,
                err_type,
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
    }
}
