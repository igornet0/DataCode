// Stack opcodes: Constant, LoadLocal, StoreLocal, Pop, Dup, FormatInterp.
// Logic preserved 1:1 from executor.rs — no semantic changes.

use crate::common::error::LangError;
use crate::common::value::Value;
use crate::common::value_store::ValueStore;
use crate::common::TaggedValue;
use crate::vm::exceptions::ExceptionHandler;
use crate::vm::frame::CallFrame;
use crate::vm::heavy_store::HeavyStore;
use crate::vm::stack;
use crate::vm::store_convert::{load_value, store_value, slot_to_value};
use crate::vm::types::VMStatus;
use std::rc::Rc;

use super::helpers::pop_to_value_id;

/// Format a value for string interpolation with a spec like ".2f" or ".0f".
fn format_value_interp(value: &Value, spec: &str) -> String {
    let spec = spec.trim();
    if let Some(rest) = spec.strip_prefix('.') {
        if let Some(dot_f) = rest.find('f') {
            let prec_str = &rest[..dot_f];
            if prec_str.chars().all(|c| c.is_ascii_digit()) {
                let prec: usize = prec_str.parse().unwrap_or(6);
                if let Value::Number(n) = value {
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
    let tv = frame.constant_tagged.get(index)
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
    let frame = frames.last_mut().unwrap();
    if frame.load_local_cache_slot == Some(index) {
        frame.load_local_cache_slot = None;
    }
    if index >= frame.slots.len() {
        frame.slots.resize(index + 1, TaggedValue::null());
    }
    frame.slots[index] = tv;
    if cfg!(debug_assertions) {
        let val = slot_to_value(tv, value_store, heavy_store);
        if let crate::common::value::Value::Object(obj_rc) = &val {
            let _obj_ptr = Rc::as_ptr(obj_rc);
            let f = frames.last().unwrap();
            let is_constructor = f.function.name.contains("::new_");
            let current_ip = f.ip - 1;
            if is_constructor {
                let map = obj_rc.borrow();
                crate::debug_println!("[DEBUG StoreLocal] constructor '{}' IP {} slot {}: Object ({} keys)", f.function.name, current_ip, index, map.len());
            }
        }
    }
    Ok(VMStatus::Continue)
}

pub fn op_pop(
    stack: &mut Vec<TaggedValue>,
    frames: &mut Vec<CallFrame>,
) -> Result<VMStatus, LangError> {
    if let Some(frame) = frames.last() {
        if stack.len() > frame.stack_start {
            stack.pop();
        }
    } else {
        if !stack.is_empty() {
            stack.pop();
        }
    }
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
    let format_id = frame.constant_ids.get(format_index).copied().ok_or_else(|| LangError::ParseError {
        message: format!("FormatInterp: constant index {} out of range", format_index),
        line: 0,
        file: None,
    })?;
    let value = load_value(value_id, value_store, heavy_store);
    let format_spec = match load_value(format_id, value_store, heavy_store) {
        Value::String(s) => s,
        _ => value.to_string(),
    };
    let result = format_value_interp(&value, &format_spec);
    let result_id = store_value(Value::String(result), value_store, heavy_store);
    stack::push_id(stack, result_id);
    Ok(VMStatus::Continue)
}
