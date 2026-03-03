// Stack opcodes: Constant, LoadLocal, StoreLocal, Pop, Dup.
// Logic preserved 1:1 from executor.rs — no semantic changes.

use crate::common::{error::LangError, value_store::ValueStore};
use crate::common::TaggedValue;
use crate::vm::exceptions::ExceptionHandler;
use crate::vm::frame::CallFrame;
use crate::vm::heavy_store::HeavyStore;
use crate::vm::stack;
use crate::vm::store_convert::slot_to_value;
use crate::vm::types::VMStatus;
use std::rc::Rc;

use super::helpers::pop_to_value_id;

pub fn op_constant(
    index: usize,
    stack: &mut Vec<TaggedValue>,
    frame: &mut CallFrame,
) -> Result<VMStatus, LangError> {
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
