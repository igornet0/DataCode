//! Optional sweep of heap ids in dead stack slots at VM safe points.

#[cfg(feature = "deferred_stack_gc")]
use std::collections::HashSet;

use crate::common::TaggedValue;
use crate::common::value_store::ValueStore;
use crate::vm::frame::CallFrame;

/// Recycle recyclable heap-pair ids found only in `stack[dead_from..dead_to)` and not in live VM roots.
#[cfg(feature = "deferred_stack_gc")]
pub fn sweep_dead_stack_range(
    stack: &[TaggedValue],
    dead_from: usize,
    dead_to: usize,
    frames: &[CallFrame],
    live_stack_sp: usize,
    store: &mut ValueStore,
) {
    if dead_from >= dead_to || dead_from >= stack.len() {
        return;
    }
    let end = dead_to.min(stack.len());
    let mut candidates = HashSet::new();
    for tv in &stack[dead_from..end] {
        if tv.is_heap() {
            let id = tv.get_heap_id();
            if store.holds_scratch_heap_pair(id) {
                continue;
            }
            if store.is_recyclable_heap_pair(id) {
                candidates.insert(id);
            }
        }
    }
    if candidates.is_empty() {
        return;
    }
    let mut live = HashSet::new();
    for tv in &stack[..live_stack_sp.min(stack.len())] {
        if tv.is_heap() {
            live.insert(tv.get_heap_id());
        }
    }
    for frame in frames {
        for slot in &frame.slots {
            if slot.is_heap() {
                live.insert(slot.get_heap_id());
            }
        }
    }
    for id in candidates {
        if !live.contains(&id) {
            store.recycle_heap_pair(id);
        }
    }
}

#[cfg(feature = "deferred_stack_gc")]
pub fn sweep_before_truncate(
    stack: &[TaggedValue],
    old_sp: usize,
    new_sp: usize,
    frames: &[CallFrame],
    store: &mut ValueStore,
) {
    if old_sp > new_sp {
        sweep_dead_stack_range(stack, new_sp, old_sp, frames, new_sp, store);
    }
}

#[cfg(not(feature = "deferred_stack_gc"))]
pub fn sweep_before_truncate(
    _stack: &[TaggedValue],
    _old_sp: usize,
    _new_sp: usize,
    _frames: &[CallFrame],
    _store: &mut ValueStore,
) {
}
