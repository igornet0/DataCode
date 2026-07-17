//! Dense grid buffers stored in [`ValueCell::GridBufferI32`] / [`GridBufferU8`].

use crate::common::value_store::{ValueCell, ValueId, ValueStore};

/// [`crate::common::value::Value::PluginOpaque`] tag for i32 grid buffers.
pub const GRID_I32_TAG: u8 = 240;
/// [`crate::common::value::Value::PluginOpaque`] tag for u8 grid buffers (bytes or bitmap).
pub const GRID_U8_TAG: u8 = 241;
/// [`crate::common::value::Value::PluginOpaque`] tag for index-only min-heaps.
pub const GRID_HEAP_TAG: u8 = 242;

pub fn alloc_i32(store: &mut ValueStore, n: usize, fill: i32) -> ValueId {
    let data = vec![fill; n];
    store.allocate_arena(ValueCell::GridBufferI32(data))
}

pub fn alloc_u8(store: &mut ValueStore, n: usize, fill: u8) -> ValueId {
    let data = vec![fill; n];
    store.allocate_arena(ValueCell::GridBufferU8(data))
}

pub fn i32_slice_mut(store: &mut ValueStore, id: ValueId) -> Option<&mut [i32]> {
    match store.get_mut(id) {
        Some(ValueCell::GridBufferI32(v)) => Some(v.as_mut_slice()),
        _ => None,
    }
}

pub fn i32_slice(store: &ValueStore, id: ValueId) -> Option<&[i32]> {
    match store.get(id) {
        Some(ValueCell::GridBufferI32(v)) => Some(v.as_slice()),
        _ => None,
    }
}

pub fn u8_slice_mut(store: &mut ValueStore, id: ValueId) -> Option<&mut [u8]> {
    match store.get_mut(id) {
        Some(ValueCell::GridBufferU8(v)) => Some(v.as_mut_slice()),
        _ => None,
    }
}

pub fn u8_slice(store: &ValueStore, id: ValueId) -> Option<&[u8]> {
    match store.get(id) {
        Some(ValueCell::GridBufferU8(v)) => Some(v.as_slice()),
        _ => None,
    }
}

pub fn fill_i32(store: &mut ValueStore, id: ValueId, fill: i32) -> bool {
    if let Some(sl) = i32_slice_mut(store, id) {
        sl.fill(fill);
        true
    } else {
        false
    }
}

pub fn fill_u8(store: &mut ValueStore, id: ValueId, fill: u8) -> bool {
    if let Some(sl) = u8_slice_mut(store, id) {
        sl.fill(fill);
        true
    } else {
        false
    }
}

pub fn shrink_buffer(store: &mut ValueStore, id: ValueId) -> bool {
    match store.get_mut(id) {
        Some(ValueCell::GridBufferI32(v)) => {
            v.shrink_to_fit();
            true
        }
        Some(ValueCell::GridBufferU8(v)) => {
            v.shrink_to_fit();
            true
        }
        Some(ValueCell::GridHeapU32(v)) => {
            v.shrink_to_fit();
            true
        }
        _ => false,
    }
}

pub fn alloc_heap(store: &mut ValueStore) -> ValueId {
    store.allocate_arena(ValueCell::GridHeapU32(Vec::new()))
}

pub fn heap_slice_mut(store: &mut ValueStore, id: ValueId) -> Option<&mut Vec<(i32, u32)>> {
    match store.get_mut(id) {
        Some(ValueCell::GridHeapU32(v)) => Some(v),
        _ => None,
    }
}

pub fn heap_slice(store: &ValueStore, id: ValueId) -> Option<&[(i32, u32)]> {
    match store.get(id) {
        Some(ValueCell::GridHeapU32(v)) => Some(v.as_slice()),
        _ => None,
    }
}

/// Release RSS for large grid buffers (`shrink_to_fit` on all handles).
pub fn shrink_all_buffers(store: &mut ValueStore, ids: &[ValueId]) {
    for &id in ids {
        shrink_buffer(store, id);
    }
}

fn i32_raw(store: &mut ValueStore, id: ValueId) -> Option<(*mut i32, usize)> {
    match store.get_mut(id) {
        Some(ValueCell::GridBufferI32(v)) => Some((v.as_mut_ptr(), v.len())),
        _ => None,
    }
}

fn u8_raw(store: &mut ValueStore, id: ValueId) -> Option<(*mut u8, usize)> {
    match store.get_mut(id) {
        Some(ValueCell::GridBufferU8(v)) => Some((v.as_mut_ptr(), v.len())),
        _ => None,
    }
}

/// Run A* with multiple grid buffers from the same store (distinct ids only).
pub fn with_astar_buffers<R>(
    store: &mut ValueStore,
    blocked_id: ValueId,
    g_id: ValueId,
    f_id: ValueId,
    parent_id: ValueId,
    closed_id: ValueId,
    run: impl FnOnce(&[u8], &mut [i32], &mut [i32], &mut [i32], &mut [u8]) -> R,
) -> Option<R> {
    let blocked = u8_slice(store, blocked_id)?.to_vec();
    let (g_ptr, g_len) = i32_raw(store, g_id)?;
    let (f_ptr, f_len) = i32_raw(store, f_id)?;
    let (p_ptr, p_len) = i32_raw(store, parent_id)?;
    let (c_ptr, c_len) = u8_raw(store, closed_id)?;
    // SAFETY: each buffer lives in a distinct store cell; ids must differ.
    Some(unsafe {
        let g = std::slice::from_raw_parts_mut(g_ptr, g_len);
        let f = std::slice::from_raw_parts_mut(f_ptr, f_len);
        let parent = std::slice::from_raw_parts_mut(p_ptr, p_len);
        let closed = std::slice::from_raw_parts_mut(c_ptr, c_len);
        run(&blocked, g, f, parent, closed)
    })
}

/// Same as [`with_astar_buffers`] plus mutable grid heap (distinct cell id).
pub fn with_astar_buffers_and_heap<R>(
    store: &mut ValueStore,
    blocked_id: ValueId,
    g_id: ValueId,
    f_id: ValueId,
    parent_id: ValueId,
    closed_id: ValueId,
    heap_id: ValueId,
    run: impl FnOnce(
        &[u8],
        &mut [i32],
        &mut [i32],
        &mut [i32],
        &mut [u8],
        &mut Vec<(i32, u32)>,
    ) -> R,
) -> Option<R> {
    let blocked = u8_slice(store, blocked_id)?.to_vec();
    let (g_ptr, g_len) = i32_raw(store, g_id)?;
    let (f_ptr, f_len) = i32_raw(store, f_id)?;
    let (p_ptr, p_len) = i32_raw(store, parent_id)?;
    let (c_ptr, c_len) = u8_raw(store, closed_id)?;
    let heap_ptr = match store.get_mut(heap_id) {
        Some(ValueCell::GridHeapU32(v)) => v as *mut Vec<(i32, u32)>,
        _ => return None,
    };
    // SAFETY: heap cell is distinct from g/f/parent/closed/blocked ids.
    Some(unsafe {
        let g = std::slice::from_raw_parts_mut(g_ptr, g_len);
        let f = std::slice::from_raw_parts_mut(f_ptr, f_len);
        let parent = std::slice::from_raw_parts_mut(p_ptr, p_len);
        let closed = std::slice::from_raw_parts_mut(c_ptr, c_len);
        let heap = &mut *heap_ptr;
        run(&blocked, g, f, parent, closed, heap)
    })
}
