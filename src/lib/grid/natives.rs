//! Native `grid` API: dense buffers + A* on flat memory.

use crate::common::astar_grid_core::{
    astar_grid_on_buffers, bitmap_blocked, bitmap_bytes_for_cells, bitmap_set,
};
use crate::common::value::{ObjectKind, Value};
use crate::common::value_store::ValueStore;
use crate::grid::buffer::{
    alloc_heap, alloc_i32, alloc_u8, fill_i32, fill_u8, heap_slice_mut, i32_slice, i32_slice_mut,
    shrink_all_buffers, shrink_buffer, u8_slice, u8_slice_mut, with_astar_buffers,
    with_astar_buffers_and_heap, GRID_HEAP_TAG, GRID_I32_TAG, GRID_U8_TAG,
};
use crate::grid::heap::{heap_clear, heap_len, heap_pop, heap_push, heap_push_key};
use crate::vm::store_convert::load_value;
use crate::vm::vm::current_vm_ptr;
use crate::websocket::set_native_error;
use std::cell::RefCell;
use std::collections::BinaryHeap;
use std::rc::Rc;

fn data_args<'a>(args: &'a [Value]) -> &'a [Value] {
    if args.len() > 1 {
        if let Value::Object(o) = &args[0] {
            let map = o.borrow();
            let native_count = match &*map {
                ObjectKind::Legacy(hm) => hm
                    .values()
                    .filter(|v| matches!(v, Value::NativeFunction(_)))
                    .count(),
                ObjectKind::Inline(pairs) => pairs
                    .iter()
                    .filter(|(_, v)| matches!(v, Value::NativeFunction(_)))
                    .count(),
                _ => 0,
            };
            if native_count >= 3 {
                return &args[1..];
            }
        }
    }
    args
}

fn arg_usize(v: &Value, name: &str) -> Option<usize> {
    let Some(n) = v.as_finite_f64() else {
        set_native_error(format!("TypeError: {} must be a non-negative integer", name));
        return None;
    };
    if n >= 0.0 && n.fract() == 0.0 {
        Some(n as usize)
    } else {
        set_native_error(format!("TypeError: {} must be a non-negative integer", name));
        None
    }
}

fn arg_i32(v: &Value, name: &str) -> Option<i32> {
    let Some(n) = v.as_finite_f64() else {
        set_native_error(format!("TypeError: {} must be an integer", name));
        return None;
    };
    if n >= i32::MIN as f64 && n <= i32::MAX as f64 && n.fract() == 0.0 {
        Some(n as i32)
    } else {
        set_native_error(format!("TypeError: {} must be an integer", name));
        None
    }
}

fn arg_u32(v: &Value, name: &str) -> Option<u32> {
    let Some(n) = v.as_finite_f64() else {
        set_native_error(format!("TypeError: {} must be a non-negative integer", name));
        return None;
    };
    if n >= 0.0 && n.fract() == 0.0 && n <= u32::MAX as f64 {
        Some(n as u32)
    } else {
        set_native_error(format!("TypeError: {} must be a non-negative integer", name));
        None
    }
}

fn arg_grid_i32(v: &Value) -> Option<u32> {
    match v {
        Value::PluginOpaque {
            tag: GRID_I32_TAG,
            id,
        } => Some(*id as u32),
        _ => {
            set_native_error("TypeError: expected grid i32 buffer handle".to_string());
            None
        }
    }
}

fn arg_grid_u8(v: &Value) -> Option<u32> {
    match v {
        Value::PluginOpaque {
            tag: GRID_U8_TAG,
            id,
        } => Some(*id as u32),
        _ => {
            set_native_error("TypeError: expected grid u8 buffer handle".to_string());
            None
        }
    }
}

fn handle_i32(id: u32) -> Value {
    Value::PluginOpaque {
        tag: GRID_I32_TAG,
        id: id as u64,
    }
}

fn arg_grid_heap(v: &Value) -> Option<u32> {
    match v {
        Value::PluginOpaque {
            tag: GRID_HEAP_TAG,
            id,
        } => Some(*id as u32),
        _ => {
            set_native_error("TypeError: expected grid heap handle".to_string());
            None
        }
    }
}

fn handle_u8(id: u32) -> Value {
    Value::PluginOpaque {
        tag: GRID_U8_TAG,
        id: id as u64,
    }
}

fn handle_heap(id: u32) -> Value {
    Value::PluginOpaque {
        tag: GRID_HEAP_TAG,
        id: id as u64,
    }
}

fn with_store<F, R>(f: F) -> Option<R>
where
    F: FnOnce(&mut ValueStore) -> R,
{
    let vm_ptr = current_vm_ptr()?;
    Some(unsafe { (*vm_ptr).with_stores_mut(|store, _| f(store)) })
}

fn arg_coord(v: &Value, name: &str) -> Option<(u32, u32)> {
    match v {
        Value::Tuple(t) => {
            let tup = t.borrow();
            if tup.len() != 2 {
                set_native_error(format!("TypeError: {} must be (row, col)", name));
                return None;
            }
            let r = arg_u32(&tup[0], &format!("{name}[0]"))?;
            let c = arg_u32(&tup[1], &format!("{name}[1]"))?;
            Some((r, c))
        }
        Value::Array(a) => {
            let arr = a.borrow();
            if arr.len() != 2 {
                set_native_error(format!("TypeError: {} must be [row, col]", name));
                return None;
            }
            let r = arg_u32(&arr[0], &format!("{name}[0]"))?;
            let c = arg_u32(&arr[1], &format!("{name}[1]"))?;
            Some((r, c))
        }
        _ => {
            set_native_error(format!("TypeError: {} must be (row, col)", name));
            None
        }
    }
}

fn path_to_value(path: Vec<(u32, u32)>) -> Value {
    let tuples: Vec<Value> = path
        .into_iter()
        .map(|(r, c)| {
            Value::Tuple(Rc::new(RefCell::new(vec![
                Value::Number(r as f64),
                Value::Number(c as f64),
            ])))
        })
        .collect();
    Value::Array(Rc::new(RefCell::new(tuples)))
}

/// `alloc_i32(n, fill)` → opaque handle
pub fn native_grid_alloc_i32(args: &[Value]) -> Value {
    if args.len() < 2 {
        set_native_error("TypeError: alloc_i32(n, fill)".to_string());
        return Value::Null;
    }
    let n = match arg_usize(&args[0], "n") {
        Some(v) => v,
        None => return Value::Null,
    };
    let fill = match arg_i32(&args[1], "fill") {
        Some(v) => v,
        None => return Value::Null,
    };
    with_store(|store| handle_i32(alloc_i32(store, n, fill))).unwrap_or(Value::Null)
}

/// `alloc_u8(n, fill)` → opaque handle
pub fn native_grid_alloc_u8(args: &[Value]) -> Value {
    if args.len() < 2 {
        set_native_error("TypeError: alloc_u8(n, fill)".to_string());
        return Value::Null;
    }
    let n = match arg_usize(&args[0], "n") {
        Some(v) => v,
        None => return Value::Null,
    };
    let fill = match args[1] {
        Value::Number(n) if n.is_finite() && n >= 0.0 && n <= 255.0 => n as u8,
        _ => {
            set_native_error("TypeError: fill must be 0..255".to_string());
            return Value::Null;
        }
    };
    with_store(|store| handle_u8(alloc_u8(store, n, fill))).unwrap_or(Value::Null)
}

pub fn native_grid_fill_i32(args: &[Value]) -> Value {
    if args.len() < 2 {
        set_native_error("TypeError: fill_i32(buf, fill)".to_string());
        return Value::Null;
    }
    let id = match arg_grid_i32(&args[0]) {
        Some(v) => v,
        None => return Value::Null,
    };
    let fill = match arg_i32(&args[1], "fill") {
        Some(v) => v,
        None => return Value::Null,
    };
    with_store(|store| fill_i32(store, id, fill))
        .map(|ok| Value::Bool(ok))
        .unwrap_or(Value::Null)
}

pub fn native_grid_fill_u8(args: &[Value]) -> Value {
    if args.len() < 2 {
        set_native_error("TypeError: fill_u8(buf, fill)".to_string());
        return Value::Null;
    }
    let id = match arg_grid_u8(&args[0]) {
        Some(v) => v,
        None => return Value::Null,
    };
    let fill = match args[1] {
        Value::Number(n) if n.is_finite() && n >= 0.0 && n <= 255.0 => n as u8,
        _ => {
            set_native_error("TypeError: fill must be 0..255".to_string());
            return Value::Null;
        }
    };
    with_store(|store| fill_u8(store, id, fill))
        .map(|ok| Value::Bool(ok))
        .unwrap_or(Value::Null)
}

pub fn native_grid_get_i32(args: &[Value]) -> Value {
    if args.len() < 2 {
        set_native_error("TypeError: get_i32(buf, index)".to_string());
        return Value::Null;
    }
    let id = match arg_grid_i32(&args[0]) {
        Some(v) => v,
        None => return Value::Null,
    };
    let idx = match arg_usize(&args[1], "index") {
        Some(v) => v,
        None => return Value::Null,
    };
    with_store(|store| {
        i32_slice(store, id)
            .and_then(|s| s.get(idx))
            .map(|&v| Value::Number(v as f64))
            .unwrap_or(Value::Null)
    })
    .unwrap_or(Value::Null)
}

pub fn native_grid_set_i32(args: &[Value]) -> Value {
    if args.len() < 3 {
        set_native_error("TypeError: set_i32(buf, index, value)".to_string());
        return Value::Null;
    }
    let id = match arg_grid_i32(&args[0]) {
        Some(v) => v,
        None => return Value::Null,
    };
    let idx = match arg_usize(&args[1], "index") {
        Some(v) => v,
        None => return Value::Null,
    };
    let val = match arg_i32(&args[2], "value") {
        Some(v) => v,
        None => return Value::Null,
    };
    with_store(|store| {
        if let Some(sl) = i32_slice_mut(store, id) {
            if let Some(slot) = sl.get_mut(idx) {
                *slot = val;
                return Value::Bool(true);
            }
        }
        Value::Bool(false)
    })
    .unwrap_or(Value::Null)
}

pub fn native_grid_get_u8(args: &[Value]) -> Value {
    if args.len() < 2 {
        set_native_error("TypeError: get_u8(buf, index)".to_string());
        return Value::Null;
    }
    let id = match arg_grid_u8(&args[0]) {
        Some(v) => v,
        None => return Value::Null,
    };
    let idx = match arg_usize(&args[1], "index") {
        Some(v) => v,
        None => return Value::Null,
    };
    with_store(|store| {
        u8_slice(store, id)
            .and_then(|s| s.get(idx))
            .map(|&v| Value::Number(v as f64))
            .unwrap_or(Value::Null)
    })
    .unwrap_or(Value::Null)
}

pub fn native_grid_set_u8(args: &[Value]) -> Value {
    if args.len() < 3 {
        set_native_error("TypeError: set_u8(buf, index, value)".to_string());
        return Value::Null;
    }
    let id = match arg_grid_u8(&args[0]) {
        Some(v) => v,
        None => return Value::Null,
    };
    let idx = match arg_usize(&args[1], "index") {
        Some(v) => v,
        None => return Value::Null,
    };
    let val = match args[2] {
        Value::Number(n) if n.is_finite() && n >= 0.0 && n <= 255.0 => n as u8,
        _ => {
            set_native_error("TypeError: value must be 0..255".to_string());
            return Value::Null;
        }
    };
    with_store(|store| {
        if let Some(sl) = u8_slice_mut(store, id) {
            if let Some(slot) = sl.get_mut(idx) {
                *slot = val;
                return Value::Bool(true);
            }
        }
        Value::Bool(false)
    })
    .unwrap_or(Value::Null)
}

pub fn native_grid_test_blocked(args: &[Value]) -> Value {
    if args.len() < 2 {
        set_native_error("TypeError: test_blocked(bitmap, cell_id)".to_string());
        return Value::Null;
    }
    let id = match arg_grid_u8(&args[0]) {
        Some(v) => v,
        None => return Value::Null,
    };
    let cell = match arg_u32(&args[1], "cell_id") {
        Some(v) => v,
        None => return Value::Null,
    };
    with_store(|store| {
        u8_slice(store, id)
            .map(|bits| Value::Bool(bitmap_blocked(bits, cell)))
            .unwrap_or(Value::Null)
    })
    .unwrap_or(Value::Null)
}

pub fn native_grid_set_blocked(args: &[Value]) -> Value {
    if args.len() < 2 {
        set_native_error("TypeError: set_blocked(bitmap, cell_id)".to_string());
        return Value::Null;
    }
    let id = match arg_grid_u8(&args[0]) {
        Some(v) => v,
        None => return Value::Null,
    };
    let cell = match arg_u32(&args[1], "cell_id") {
        Some(v) => v,
        None => return Value::Null,
    };
    with_store(|store| {
        if let Some(bits) = u8_slice_mut(store, id) {
            bitmap_set(bits, cell);
            Value::Bool(true)
        } else {
            Value::Bool(false)
        }
    })
    .unwrap_or(Value::Null)
}

pub fn native_grid_shrink(args: &[Value]) -> Value {
    if args.is_empty() {
        set_native_error("TypeError: shrink(buf)".to_string());
        return Value::Null;
    }
    let id = match &args[0] {
        Value::PluginOpaque { tag: GRID_I32_TAG, id } => *id as u32,
        Value::PluginOpaque { tag: GRID_U8_TAG, id } => *id as u32,
        _ => {
            set_native_error("TypeError: shrink() expects grid buffer handle".to_string());
            return Value::Null;
        }
    };
    with_store(|store| Value::Bool(shrink_buffer(store, id))).unwrap_or(Value::Null)
}

/// `bitmap_bytes(n_cells)` — byte length for a blocked bitmap.
pub fn native_grid_bitmap_bytes(args: &[Value]) -> Value {
    if args.is_empty() {
        set_native_error("TypeError: bitmap_bytes(n_cells)".to_string());
        return Value::Null;
    }
    let n = match arg_usize(&args[0], "n_cells") {
        Some(v) => v,
        None => return Value::Null,
    };
    Value::Number(bitmap_bytes_for_cells(n) as f64)
}

/// Native A* using grid buffers + blocked bitmap (<1s on 1000×5000).
pub fn native_grid_astar(args: &[Value]) -> Value {
    if args.len() < 8 {
        set_native_error(
            "TypeError: astar(rows, cols, start, goal, blocked_bitmap, g, f, parent, closed)"
                .to_string(),
        );
        return Value::Null;
    }
    let rows = match arg_u32(&args[0], "rows") {
        Some(v) => v,
        None => return Value::Null,
    };
    let cols = match arg_u32(&args[1], "cols") {
        Some(v) => v,
        None => return Value::Null,
    };
    let start = match arg_coord(&args[2], "start") {
        Some(v) => v,
        None => return Value::Null,
    };
    let goal = match arg_coord(&args[3], "goal") {
        Some(v) => v,
        None => return Value::Null,
    };
    let blocked_id = match arg_grid_u8(&args[4]) {
        Some(v) => v,
        None => return Value::Null,
    };
    let g_id = match arg_grid_i32(&args[5]) {
        Some(v) => v,
        None => return Value::Null,
    };
    let f_id = match arg_grid_i32(&args[6]) {
        Some(v) => v,
        None => return Value::Null,
    };
    let parent_id = match arg_grid_i32(&args[7]) {
        Some(v) => v,
        None => return Value::Null,
    };
    let closed_id = if args.len() >= 9 {
        match arg_grid_u8(&args[8]) {
            Some(v) => v,
            None => return Value::Null,
        }
    } else {
        set_native_error("TypeError: astar requires closed u8 buffer".to_string());
        return Value::Null;
    };

    let vm_ptr = match current_vm_ptr() {
        Some(p) => p,
        None => return Value::Null,
    };

    unsafe {
        (*vm_ptr).with_stores_mut(|store, _heap| {
            let mut heap_q = BinaryHeap::new();
            with_astar_buffers(
                store,
                blocked_id,
                g_id,
                f_id,
                parent_id,
                closed_id,
                |blocked_bits, g, f, parent, closed| {
                    match astar_grid_on_buffers(
                        rows,
                        cols,
                        start,
                        goal,
                        Some(blocked_bits),
                        g,
                        f,
                        parent,
                        closed,
                        &mut heap_q,
                    ) {
                        Some(path) => path_to_value(path),
                        None => Value::Null,
                    }
                },
            )
            .unwrap_or(Value::Null)
        })
    }
}

/// Build bitmap from linear cell ids in a set-like array (for tests).
pub fn native_grid_bitmap_from_ids(args: &[Value]) -> Value {
    if args.len() < 2 {
        set_native_error("TypeError: bitmap_from_ids(n_cells, ids_array)".to_string());
        return Value::Null;
    }
    let n_cells = match arg_usize(&args[0], "n_cells") {
        Some(v) => v,
        None => return Value::Null,
    };
    let ids = match &args[1] {
        Value::Array(a) => a.borrow().clone(),
        _ => {
            set_native_error("TypeError: ids must be array".to_string());
            return Value::Null;
        }
    };
    with_store(|store| {
        let id = alloc_u8(store, bitmap_bytes_for_cells(n_cells), 0);
        if let Some(bits) = u8_slice_mut(store, id) {
            for v in ids {
                if let Value::Number(n) = v {
                    bitmap_set(bits, n as u32);
                }
            }
        }
        handle_u8(id)
    })
    .unwrap_or(Value::Null)
}

/// VM store length (diagnostics).
pub fn native_grid_store_len(_args: &[Value]) -> Value {
    current_vm_ptr()
        .map(|p| unsafe { Value::Number((*p).value_store_len() as f64) })
        .unwrap_or(Value::Null)
}

/// Convenience: `astar_from_set(rows, cols, start, goal, blocked_set)` using internal bitmap.
pub fn native_grid_astar_from_set(args: &[Value]) -> Value {
    let args = data_args(args);
    if args.len() < 5 {
        set_native_error(
            "TypeError: astar_from_set(rows, cols, start, goal, blocked_set)".to_string(),
        );
        return Value::Null;
    }
    let rows = match arg_u32(&args[0], "rows") {
        Some(v) => v,
        None => return Value::Null,
    };
    let cols = match arg_u32(&args[1], "cols") {
        Some(v) => v,
        None => return Value::Null,
    };
    let start = match arg_coord(&args[2], "start") {
        Some(v) => v,
        None => return Value::Null,
    };
    let goal = match arg_coord(&args[3], "goal") {
        Some(v) => v,
        None => return Value::Null,
    };

    let vm_ptr = match current_vm_ptr() {
        Some(p) => p,
        None => return Value::Null,
    };

    unsafe {
        (*vm_ptr).with_stores_mut(|store, heap| {
            let n_cells = rows as usize * cols as usize;
            let mut bits = vec![0u8; bitmap_bytes_for_cells(n_cells)];
            if let Value::Set(s) = &args[4] {
                for key_id in s.borrow().iter_key_ids() {
                    let v = load_value(key_id, store, heap);
                    if let Value::Number(n) = v {
                        bitmap_set(&mut bits, n as u32);
                    } else if let Value::Tuple(t) = v {
                        let tup = t.borrow();
                        if tup.len() == 2 {
                            if let (Value::Number(r), Value::Number(c)) = (&tup[0], &tup[1]) {
                                let id = *r as u32 * cols + *c as u32;
                                bitmap_set(&mut bits, id);
                            }
                        }
                    }
                }
            } else if !matches!(&args[4], Value::Null) {
                set_native_error("TypeError: blocked must be a set".to_string());
                return Value::Null;
            }
            use crate::common::astar_grid_core::astar_grid_blocked;
            match astar_grid_blocked(rows, cols, start, goal, Some(&bits), None) {
                Some(path) => path_to_value(path),
                None => Value::Null,
            }
        })
    }
}

/// `heap_alloc()` → opaque heap handle.
pub fn native_grid_heap_alloc(_args: &[Value]) -> Value {
    with_store(|store| handle_heap(alloc_heap(store)))
        .unwrap_or(Value::Null)
}

pub fn native_grid_heap_clear(args: &[Value]) -> Value {
    if args.is_empty() {
        set_native_error("TypeError: heap_clear(heap)".to_string());
        return Value::Null;
    }
    let id = match arg_grid_heap(&args[0]) {
        Some(v) => v,
        None => return Value::Null,
    };
    with_store(|store| {
        if let Some(entries) = heap_slice_mut(store, id) {
            heap_clear(entries);
            Value::Bool(true)
        } else {
            Value::Bool(false)
        }
    })
    .unwrap_or(Value::Null)
}

pub fn native_grid_heap_push(args: &[Value]) -> Value {
    if args.len() < 3 {
        set_native_error("TypeError: heap_push(heap, node, f_buf)".to_string());
        return Value::Null;
    }
    let heap_id = match arg_grid_heap(&args[0]) {
        Some(v) => v,
        None => return Value::Null,
    };
    let node = match arg_u32(&args[1], "node") {
        Some(v) => v,
        None => return Value::Null,
    };
    let f_id = match arg_grid_i32(&args[2]) {
        Some(v) => v,
        None => return Value::Null,
    };
    with_store(|store| {
        let key = i32_slice(store, f_id)
            .and_then(|s| s.get(node as usize).copied())
            .unwrap_or(i32::MAX);
        if let Some(entries) = heap_slice_mut(store, heap_id) {
            heap_push_key(entries, node, key);
            Value::Null
        } else {
            Value::Null
        }
    })
    .unwrap_or(Value::Null)
}

/// Pop min entry; returns `(f_at_push, node)` tuple for stale-check loops.
pub fn native_grid_heap_pop(args: &[Value]) -> Value {
    if args.is_empty() {
        set_native_error("TypeError: heap_pop(heap)".to_string());
        return Value::Null;
    }
    let heap_id = match arg_grid_heap(&args[0]) {
        Some(v) => v,
        None => return Value::Null,
    };
    with_store(|store| {
        if let Some(entries) = heap_slice_mut(store, heap_id) {
            if let Some((f, node)) = heap_pop(entries) {
                Value::Tuple(Rc::new(RefCell::new(vec![
                    Value::Number(f as f64),
                    Value::Number(node as f64),
                ])))
            } else {
                set_native_error("IndexError: heap_pop from empty heap".to_string());
                Value::Null
            }
        } else {
            Value::Null
        }
    })
    .unwrap_or(Value::Null)
}

pub fn native_grid_heap_len(args: &[Value]) -> Value {
    if args.is_empty() {
        set_native_error("TypeError: heap_len(heap)".to_string());
        return Value::Null;
    }
    let heap_id = match arg_grid_heap(&args[0]) {
        Some(v) => v,
        None => return Value::Null,
    };
    with_store(|store| {
        heap_slice_mut(store, heap_id)
            .map(|e| Value::Number(heap_len(e) as f64))
            .unwrap_or(Value::Null)
    })
    .unwrap_or(Value::Null)
}

pub fn native_grid_shrink_all(args: &[Value]) -> Value {
    let ids: Vec<u32> = args
        .iter()
        .filter_map(|v| match v {
            Value::PluginOpaque { tag: GRID_I32_TAG, id } => Some(*id as u32),
            Value::PluginOpaque { tag: GRID_U8_TAG, id } => Some(*id as u32),
            Value::PluginOpaque { tag: GRID_HEAP_TAG, id } => Some(*id as u32),
            _ => None,
        })
        .collect();
    with_store(|store| {
        let vids: Vec<_> = ids.iter().map(|&i| i as crate::common::value_store::ValueId).collect();
        shrink_all_buffers(store, &vids);
        Value::Bool(true)
    })
    .unwrap_or(Value::Null)
}

/// Run up to `max_expansions` A* steps on grid buffers (fused Rust loop for DC stress).
pub fn native_grid_astar_step(args: &[Value]) -> Value {
    if args.len() < 10 {
        set_native_error(
            "TypeError: astar_step(rows, cols, goal, blocked, g, f, parent, closed, heap, max_expansions)"
                .to_string(),
        );
        return Value::Null;
    }
    let rows = match arg_u32(&args[0], "rows") {
        Some(v) => v,
        None => return Value::Null,
    };
    let cols = match arg_u32(&args[1], "cols") {
        Some(v) => v,
        None => return Value::Null,
    };
    let goal = match arg_coord(&args[2], "goal") {
        Some(v) => v,
        None => return Value::Null,
    };
    let blocked_id = match arg_grid_u8(&args[3]) {
        Some(v) => v,
        None => return Value::Null,
    };
    let g_id = match arg_grid_i32(&args[4]) {
        Some(v) => v,
        None => return Value::Null,
    };
    let f_id = match arg_grid_i32(&args[5]) {
        Some(v) => v,
        None => return Value::Null,
    };
    let parent_id = match arg_grid_i32(&args[6]) {
        Some(v) => v,
        None => return Value::Null,
    };
    let closed_id = match arg_grid_u8(&args[7]) {
        Some(v) => v,
        None => return Value::Null,
    };
    let heap_id = match arg_grid_heap(&args[8]) {
        Some(v) => v,
        None => return Value::Null,
    };
    let max_exp = match arg_usize(&args[9], "max_expansions") {
        Some(v) => v,
        None => return Value::Null,
    };

    let goal_id = goal.0 * cols + goal.1;
    let goal_r = goal.0;
    let goal_c = goal.1;
    let neighbors_delta: [(i32, i32); 4] = [(-1, 0), (1, 0), (0, -1), (0, 1)];

    with_store(|store| {
        let mut expansions = 0usize;
        let mut found = false;
        let run = with_astar_buffers_and_heap(
            store,
            blocked_id,
            g_id,
            f_id,
            parent_id,
            closed_id,
            heap_id,
            |blocked_bits, g_score, f_score, came_from, closed, heap| {
                while expansions < max_exp {
                    let Some((current_f, current)) = heap_pop(heap) else {
                        break;
                    };
                    if current_f != f_score.get(current as usize).copied().unwrap_or(i32::MAX) {
                        continue;
                    }
                    if closed.get(current as usize).copied().unwrap_or(0) != 0 {
                        continue;
                    }
                    expansions += 1;
                    if current == goal_id {
                        found = true;
                        break;
                    }
                    closed[current as usize] = 1;
                    let r = current / cols;
                    let c = current % cols;
                    let g_cur = g_score[current as usize];
                    for (dr, dc) in neighbors_delta {
                        let nr = r as i32 + dr;
                        let nc = c as i32 + dc;
                        if nr < 0 || nc < 0 || nr >= rows as i32 || nc >= cols as i32 {
                            continue;
                        }
                        let neighbor = (nr as u32) * cols + (nc as u32);
                        if bitmap_blocked(blocked_bits, neighbor) {
                            continue;
                        }
                        if closed.get(neighbor as usize).copied().unwrap_or(0) != 0 {
                            continue;
                        }
                        let tentative_g = g_cur.saturating_add(1);
                        if tentative_g < g_score[neighbor as usize] {
                            came_from[neighbor as usize] = current as i32;
                            g_score[neighbor as usize] = tentative_g;
                            let h = (nr as u32).abs_diff(goal_r) + (nc as u32).abs_diff(goal_c);
                            let f = tentative_g.saturating_add(h as i32);
                            f_score[neighbor as usize] = f;
                            heap_push(heap, neighbor, f_score);
                        }
                    }
                }
                Value::Tuple(Rc::new(RefCell::new(vec![
                    Value::Number(expansions as f64),
                    Value::Bool(found),
                ])))
            },
        );
        run.unwrap_or(Value::Null)
    })
    .unwrap_or(Value::Null)
}
