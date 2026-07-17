//! Grid buffer opcode fast paths (`grid.get_i32`, `grid.heap_push`, …).

use crate::common::astar_grid_core::bitmap_blocked;
use crate::common::error::ErrorType;
use crate::common::numeric::tagged_integral_canonical_if_whole;
use crate::common::value_store::{ValueCell, ValueId, ValueStore};
use crate::common::TaggedValue;
use crate::grid::buffer::{GRID_HEAP_TAG, GRID_I32_TAG, GRID_U8_TAG, i32_slice, i32_slice_mut, u8_slice, u8_slice_mut};
use crate::grid::heap::{heap_len, heap_pop, heap_push_key};
use crate::vm::exceptions::ExceptionHandler;
use crate::vm::frame::CallFrame;
use crate::vm::heavy_store::HeavyStore;
use crate::vm::stack;
use crate::vm::types::VMStatus;

fn resolve_grid_cell(store: &ValueStore, tv: TaggedValue, tag: u8) -> Option<ValueId> {
    if !tv.is_heap() {
        return None;
    }
    let wrapper_id = tv.get_heap_id();
    match store.get(wrapper_id) {
        Some(ValueCell::PluginOpaque { tag: t, id }) if *t == tag => Some(*id as ValueId),
        Some(ValueCell::GridBufferI32(_)) if tag == GRID_I32_TAG => Some(wrapper_id),
        Some(ValueCell::GridBufferU8(_)) if tag == GRID_U8_TAG => Some(wrapper_id),
        Some(ValueCell::GridHeapU32(_)) if tag == GRID_HEAP_TAG => Some(wrapper_id),
        _ => None,
    }
}

fn slot_usize(tv: TaggedValue) -> Option<usize> {
    let n = tagged_integral_canonical_if_whole(tv)?;
    if n < 0 {
        return None;
    }
    Some(n as usize)
}

fn runtime_err(
    msg: impl Into<String>,
    line: usize,
    stack: &mut Vec<TaggedValue>,
    frames: &mut Vec<CallFrame>,
    exception_handlers: &mut Vec<ExceptionHandler>,
    value_store: &mut ValueStore,
    heavy_store: &mut HeavyStore,
) -> Result<VMStatus, crate::common::error::LangError> {
    let error = ExceptionHandler::runtime_error_with_type(
        frames,
        msg.into(),
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

pub fn op_grid_get_i32(
    buf_slot: usize,
    idx_slot: usize,
    line: usize,
    stack: &mut Vec<TaggedValue>,
    frames: &mut Vec<CallFrame>,
    exception_handlers: &mut Vec<ExceptionHandler>,
    value_store: &mut ValueStore,
    heavy_store: &mut HeavyStore,
) -> Result<VMStatus, crate::common::error::LangError> {
    let frame = frames.last().unwrap();
    let buf_tv = *frame.slots.get(buf_slot).unwrap_or(&TaggedValue::null());
    let idx_tv = *frame.slots.get(idx_slot).unwrap_or(&TaggedValue::null());
    let buf_id = match resolve_grid_cell(value_store, buf_tv, GRID_I32_TAG) {
        Some(id) => id,
        None => {
            return runtime_err(
                "TypeError: grid.get_i32 expects i32 buffer handle",
                line,
                stack,
                frames,
                exception_handlers,
                value_store,
                heavy_store,
            );
        }
    };
    let idx = match slot_usize(idx_tv) {
        Some(i) => i,
        None => {
            return runtime_err(
                "TypeError: grid.get_i32 index must be integral",
                line,
                stack,
                frames,
                exception_handlers,
                value_store,
                heavy_store,
            );
        }
    };
    let v = i32_slice(value_store, buf_id)
        .and_then(|s| s.get(idx))
        .copied()
        .unwrap_or(0);
    stack::push(stack, TaggedValue::from_i32(v));
    Ok(VMStatus::Continue)
}

pub fn op_grid_set_i32(
    buf_slot: usize,
    idx_slot: usize,
    val_slot: usize,
    line: usize,
    stack: &mut Vec<TaggedValue>,
    frames: &mut Vec<CallFrame>,
    exception_handlers: &mut Vec<ExceptionHandler>,
    value_store: &mut ValueStore,
    heavy_store: &mut HeavyStore,
) -> Result<VMStatus, crate::common::error::LangError> {
    let frame = frames.last().unwrap();
    let buf_tv = *frame.slots.get(buf_slot).unwrap_or(&TaggedValue::null());
    let idx_tv = *frame.slots.get(idx_slot).unwrap_or(&TaggedValue::null());
    let val_tv = *frame.slots.get(val_slot).unwrap_or(&TaggedValue::null());
    let buf_id = match resolve_grid_cell(value_store, buf_tv, GRID_I32_TAG) {
        Some(id) => id,
        None => {
            return runtime_err(
                "TypeError: grid.set_i32 expects i32 buffer handle",
                line,
                stack,
                frames,
                exception_handlers,
                value_store,
                heavy_store,
            );
        }
    };
    let idx = match slot_usize(idx_tv) {
        Some(i) => i,
        None => {
            return runtime_err(
                "TypeError: grid.set_i32 index must be integral",
                line,
                stack,
                frames,
                exception_handlers,
                value_store,
                heavy_store,
            );
        }
    };
    let val = match tagged_integral_canonical_if_whole(val_tv) {
        Some(v) if v >= i32::MIN as i64 && v <= i32::MAX as i64 => v as i32,
        _ => {
            return runtime_err(
                "TypeError: grid.set_i32 value must be integral",
                line,
                stack,
                frames,
                exception_handlers,
                value_store,
                heavy_store,
            );
        }
    };
    let ok = i32_slice_mut(value_store, buf_id)
        .and_then(|s| s.get_mut(idx))
        .map(|slot| {
            *slot = val;
            true
        })
        .unwrap_or(false);
    stack::push(stack, TaggedValue::from_bool(ok));
    Ok(VMStatus::Continue)
}

pub fn op_grid_get_u8(
    buf_slot: usize,
    idx_slot: usize,
    line: usize,
    stack: &mut Vec<TaggedValue>,
    frames: &mut Vec<CallFrame>,
    exception_handlers: &mut Vec<ExceptionHandler>,
    value_store: &mut ValueStore,
    heavy_store: &mut HeavyStore,
) -> Result<VMStatus, crate::common::error::LangError> {
    let frame = frames.last().unwrap();
    let buf_tv = *frame.slots.get(buf_slot).unwrap_or(&TaggedValue::null());
    let idx_tv = *frame.slots.get(idx_slot).unwrap_or(&TaggedValue::null());
    let buf_id = match resolve_grid_cell(value_store, buf_tv, GRID_U8_TAG) {
        Some(id) => id,
        None => {
            return runtime_err(
                "TypeError: grid.get_u8 expects u8 buffer handle",
                line,
                stack,
                frames,
                exception_handlers,
                value_store,
                heavy_store,
            );
        }
    };
    let idx = match slot_usize(idx_tv) {
        Some(i) => i,
        None => {
            return runtime_err(
                "TypeError: grid.get_u8 index must be integral",
                line,
                stack,
                frames,
                exception_handlers,
                value_store,
                heavy_store,
            );
        }
    };
    let v = u8_slice(value_store, buf_id)
        .and_then(|s| s.get(idx))
        .copied()
        .unwrap_or(0);
    stack::push(stack, TaggedValue::from_i32(v as i32));
    Ok(VMStatus::Continue)
}

pub fn op_grid_set_u8(
    buf_slot: usize,
    idx_slot: usize,
    val_slot: usize,
    line: usize,
    stack: &mut Vec<TaggedValue>,
    frames: &mut Vec<CallFrame>,
    exception_handlers: &mut Vec<ExceptionHandler>,
    value_store: &mut ValueStore,
    heavy_store: &mut HeavyStore,
) -> Result<VMStatus, crate::common::error::LangError> {
    let frame = frames.last().unwrap();
    let buf_tv = *frame.slots.get(buf_slot).unwrap_or(&TaggedValue::null());
    let idx_tv = *frame.slots.get(idx_slot).unwrap_or(&TaggedValue::null());
    let val_tv = *frame.slots.get(val_slot).unwrap_or(&TaggedValue::null());
    let buf_id = match resolve_grid_cell(value_store, buf_tv, GRID_U8_TAG) {
        Some(id) => id,
        None => {
            return runtime_err(
                "TypeError: grid.set_u8 expects u8 buffer handle",
                line,
                stack,
                frames,
                exception_handlers,
                value_store,
                heavy_store,
            );
        }
    };
    let idx = match slot_usize(idx_tv) {
        Some(i) => i,
        None => {
            return runtime_err(
                "TypeError: grid.set_u8 index must be integral",
                line,
                stack,
                frames,
                exception_handlers,
                value_store,
                heavy_store,
            );
        }
    };
    let val = match tagged_integral_canonical_if_whole(val_tv) {
        Some(v) if (0..=255).contains(&v) => v as u8,
        _ => {
            return runtime_err(
                "TypeError: grid.set_u8 value must be 0..255",
                line,
                stack,
                frames,
                exception_handlers,
                value_store,
                heavy_store,
            );
        }
    };
    let ok = u8_slice_mut(value_store, buf_id)
        .and_then(|s| s.get_mut(idx))
        .map(|slot| {
            *slot = val;
            true
        })
        .unwrap_or(false);
    stack::push(stack, TaggedValue::from_bool(ok));
    Ok(VMStatus::Continue)
}

pub fn op_grid_test_blocked(
    bitmap_slot: usize,
    idx_slot: usize,
    line: usize,
    stack: &mut Vec<TaggedValue>,
    frames: &mut Vec<CallFrame>,
    exception_handlers: &mut Vec<ExceptionHandler>,
    value_store: &mut ValueStore,
    heavy_store: &mut HeavyStore,
) -> Result<VMStatus, crate::common::error::LangError> {
    let frame = frames.last().unwrap();
    let buf_tv = *frame.slots.get(bitmap_slot).unwrap_or(&TaggedValue::null());
    let idx_tv = *frame.slots.get(idx_slot).unwrap_or(&TaggedValue::null());
    let buf_id = match resolve_grid_cell(value_store, buf_tv, GRID_U8_TAG) {
        Some(id) => id,
        None => {
            return runtime_err(
                "TypeError: grid.test_blocked expects u8 bitmap handle",
                line,
                stack,
                frames,
                exception_handlers,
                value_store,
                heavy_store,
            );
        }
    };
    let idx = match slot_usize(idx_tv) {
        Some(i) => i,
        None => {
            return runtime_err(
                "TypeError: grid.test_blocked index must be integral",
                line,
                stack,
                frames,
                exception_handlers,
                value_store,
                heavy_store,
            );
        }
    };
    let blocked = u8_slice(value_store, buf_id)
        .map(|bits| bitmap_blocked(bits, idx as u32))
        .unwrap_or(false);
    stack::push(stack, TaggedValue::from_bool(blocked));
    Ok(VMStatus::Continue)
}

pub fn op_grid_heap_push(
    heap_slot: usize,
    node_slot: usize,
    f_buf_slot: usize,
    line: usize,
    stack: &mut Vec<TaggedValue>,
    frames: &mut Vec<CallFrame>,
    exception_handlers: &mut Vec<ExceptionHandler>,
    value_store: &mut ValueStore,
    heavy_store: &mut HeavyStore,
) -> Result<VMStatus, crate::common::error::LangError> {
    let frame = frames.last().unwrap();
    let heap_tv = *frame.slots.get(heap_slot).unwrap_or(&TaggedValue::null());
    let node_tv = *frame.slots.get(node_slot).unwrap_or(&TaggedValue::null());
    let f_tv = *frame.slots.get(f_buf_slot).unwrap_or(&TaggedValue::null());
    let heap_id = match resolve_grid_cell(value_store, heap_tv, GRID_HEAP_TAG) {
        Some(id) => id,
        None => {
            return runtime_err(
                "TypeError: grid.heap_push expects heap handle",
                line,
                stack,
                frames,
                exception_handlers,
                value_store,
                heavy_store,
            );
        }
    };
    let f_id = match resolve_grid_cell(value_store, f_tv, GRID_I32_TAG) {
        Some(id) => id,
        None => {
            return runtime_err(
                "TypeError: grid.heap_push expects f_score i32 buffer",
                line,
                stack,
                frames,
                exception_handlers,
                value_store,
                heavy_store,
            );
        }
    };
    let node = match slot_usize(node_tv) {
        Some(i) if i <= u32::MAX as usize => i as u32,
        _ => {
            return runtime_err(
                "TypeError: grid.heap_push node must be non-negative integral",
                line,
                stack,
                frames,
                exception_handlers,
                value_store,
                heavy_store,
            );
        }
    };
    let key = i32_slice(value_store, f_id)
        .and_then(|s| s.get(node as usize).copied())
        .unwrap_or(i32::MAX);
    if let Some(entries) = value_store.get_mut(heap_id).and_then(|c| {
        if let ValueCell::GridHeapU32(v) = c {
            Some(v)
        } else {
            None
        }
    }) {
        heap_push_key(entries, node, key);
        stack::push(stack, TaggedValue::null());
        Ok(VMStatus::Continue)
    } else {
        runtime_err(
            "TypeError: invalid heap handle",
            line,
            stack,
            frames,
            exception_handlers,
            value_store,
            heavy_store,
        )
    }
}

pub fn op_grid_heap_pop_unpack2(
    f_slot: usize,
    node_slot: usize,
    heap_slot: usize,
    line: usize,
    stack: &mut Vec<TaggedValue>,
    frames: &mut Vec<CallFrame>,
    exception_handlers: &mut Vec<ExceptionHandler>,
    value_store: &mut ValueStore,
    heavy_store: &mut HeavyStore,
) -> Result<VMStatus, crate::common::error::LangError> {
    let frame = frames.last_mut().unwrap();
    let heap_tv = *frame.slots.get(heap_slot).unwrap_or(&TaggedValue::null());
    let heap_id = match resolve_grid_cell(value_store, heap_tv, GRID_HEAP_TAG) {
        Some(id) => id,
        None => {
            return runtime_err(
                "TypeError: grid.heap_pop expects heap handle",
                line,
                stack,
                frames,
                exception_handlers,
                value_store,
                heavy_store,
            );
        }
    };
    let popped = value_store.get_mut(heap_id).and_then(|c| {
        if let ValueCell::GridHeapU32(v) = c {
            heap_pop(v)
        } else {
            None
        }
    });
    match popped {
        Some((f_at_push, node)) => {
            if f_slot >= frame.slots.len() {
                frame.slots.resize(f_slot + 1, TaggedValue::null());
            }
            if node_slot >= frame.slots.len() {
                frame.slots.resize(node_slot + 1, TaggedValue::null());
            }
            frame.slots[f_slot] = TaggedValue::from_i32(f_at_push);
            frame.slots[node_slot] = TaggedValue::from_i32(node as i32);
            stack::push(stack, TaggedValue::null());
            Ok(VMStatus::Continue)
        }
        None => {
            return runtime_err(
                "IndexError: heap_pop from empty heap",
                line,
                stack,
                frames,
                exception_handlers,
                value_store,
                heavy_store,
            );
        }
    }
}

pub fn op_grid_heap_len(
    heap_slot: usize,
    line: usize,
    stack: &mut Vec<TaggedValue>,
    frames: &mut Vec<CallFrame>,
    exception_handlers: &mut Vec<ExceptionHandler>,
    value_store: &mut ValueStore,
    heavy_store: &mut HeavyStore,
) -> Result<VMStatus, crate::common::error::LangError> {
    let frame = frames.last().unwrap();
    let heap_tv = *frame.slots.get(heap_slot).unwrap_or(&TaggedValue::null());
    let heap_id = match resolve_grid_cell(value_store, heap_tv, GRID_HEAP_TAG) {
        Some(id) => id,
        None => {
            return runtime_err(
                "TypeError: grid.heap_len expects heap handle",
                line,
                stack,
                frames,
                exception_handlers,
                value_store,
                heavy_store,
            );
        }
    };
    let len = value_store
        .get(heap_id)
        .and_then(|c| match c {
            ValueCell::GridHeapU32(v) => Some(heap_len(v)),
            _ => None,
        })
        .unwrap_or(0);
    stack::push(stack, TaggedValue::from_i32(len as i32));
    Ok(VMStatus::Continue)
}
