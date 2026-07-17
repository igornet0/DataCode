//! Inline `heapq.heappush` / `heapq.heappop` / `heapq.heappeek` on [`ValueCell::Array`] without full native invoke.
//!
//! Priority pairs `(f, item)` for A* use **flat** storage: `[f0, n0, f1, n1, …]` in one array (one cell
//! for the whole heap) instead of a nested 2-slot cell per entry.

use std::cmp::Ordering;

use crate::common::error::ErrorType;
use crate::common::numeric::{cmp_numeric_values, IntValue};
use crate::common::value::Value;
use crate::common::value_ord::value_partial_cmp;
use crate::common::value_store::{ValueCell, ValueId, ValueStore};
use crate::common::TaggedValue;
use crate::heapq::natives::{native_heapq_heappop, native_heapq_heappush, native_heapq_heappeek};
use crate::vm::exceptions::ExceptionHandler;
use crate::vm::frame::CallFrame;
use crate::vm::heavy_store::HeavyStore;
use crate::vm::memory::{push_stack_value_id, slot_to_value, value_cell_to_tagged};
use crate::vm::stack;
use crate::vm::store_convert::tagged_to_value_id;
use crate::common::error::LangError;
use crate::vm::types::VMStatus;

struct SlotCmpCtx<'a> {
    store: &'a ValueStore,
    heavy: &'a HeavyStore,
}

impl<'a> SlotCmpCtx<'a> {
    fn compare(&self, a: TaggedValue, b: TaggedValue) -> Result<Ordering, String> {
        if let Some(ord) = compare_heap_pair_slots(a, b, self)? {
            return Ok(ord);
        }
        if let Some(ord) = try_tagged_partial_cmp(a, b) {
            return Ok(ord);
        }
        value_partial_cmp(
            &slot_to_value(a, self.store, self.heavy),
            &slot_to_value(b, self.store, self.heavy),
        )
    }

}

/// Lexicographic compare for 2-slot heap items (`Array` or `Tuple` of two numerics) — A* `(f, node)`.
fn compare_heap_pair_slots(
    a: TaggedValue,
    b: TaggedValue,
    ctx: &SlotCmpCtx<'_>,
) -> Result<Option<Ordering>, String> {
    let Some((a0, a1)) = heap_pair_slots(a, ctx)? else {
        return Ok(None);
    };
    let Some((b0, b1)) = heap_pair_slots(b, ctx)? else {
        return Ok(None);
    };
    match ctx.compare(a0, b0)? {
        Ordering::Equal => Ok(Some(ctx.compare(a1, b1)?)),
        ord => Ok(Some(ord)),
    }
}

fn heap_pair_slots(
    tv: TaggedValue,
    ctx: &SlotCmpCtx<'_>,
) -> Result<Option<(TaggedValue, TaggedValue)>, String> {
    if !tv.is_heap() {
        return Ok(None);
    }
    let id = tv.get_heap_id();
    match ctx.store.get(id) {
        Some(ValueCell::Array(slots)) if slots.len() == 2 => Ok(Some((slots[0], slots[1]))),
        Some(ValueCell::Tuple(ids)) if ids.len() == 2 => {
            let mut out = [TaggedValue::null(); 2];
            for (i, &kid) in ids.iter().enumerate() {
                out[i] = match ctx.store.get(kid) {
                    Some(cell) => {
                        value_cell_to_tagged(cell).unwrap_or_else(|| TaggedValue::from_heap(kid))
                    }
                    None => return Ok(None),
                };
            }
            Ok(Some((out[0], out[1])))
        }
        _ => Ok(None),
    }
}

fn pair_from_item(
    item: TaggedValue,
    store: &ValueStore,
    heavy: &HeavyStore,
) -> Option<(TaggedValue, TaggedValue)> {
    let ctx = SlotCmpCtx { store, heavy };
    if let Some(pair) = heap_pair_slots(item, &ctx).ok().flatten() {
        return Some(pair);
    }
    None
}

/// Heap stores nested 2-slot cells (legacy) rather than flat `[f,n,f,n,…]`.
fn is_nested_pair_heap(slots: &[TaggedValue], store: &ValueStore) -> bool {
    slots.first().is_some_and(|tv| {
        if !tv.is_heap() {
            return false;
        }
        let id = tv.get_heap_id();
        match store.get(id) {
            Some(ValueCell::Array(v)) => v.len() == 2,
            Some(ValueCell::Tuple(v)) => v.len() == 2,
            _ => store.is_heapq_owned_pair(id),
        }
    })
}

pub(crate) fn can_use_flat_pair_heap(heap_id: ValueId, store: &ValueStore) -> bool {
    store.is_flat_heap(heap_id)
}

/// Module object passed as implicit receiver on `heapq.heappush` / `heappop` Call sites.
fn is_heapq_module_receiver(tv: TaggedValue, store: &ValueStore) -> bool {
    tv.is_heap() && matches!(store.get(tv.get_heap_id()), Some(ValueCell::Object(_)))
}

fn pop_heappush_args(
    stack: &mut Vec<TaggedValue>,
    arity: usize,
    available: usize,
    store: &ValueStore,
) -> Option<(TaggedValue, TaggedValue)> {
    if available < 2 {
        return None;
    }
    if arity == 3 && available >= 3 {
        let item = crate::vm::stack::pop_direct(stack)?;
        let heap = crate::vm::stack::pop_direct(stack)?;
        let receiver = crate::vm::stack::pop_direct(stack)?;
        if !is_heapq_module_receiver(receiver, store) {
            crate::vm::stack::push_direct(stack, receiver);
            crate::vm::stack::push_direct(stack, heap);
            crate::vm::stack::push_direct(stack, item);
            return None;
        }
        return Some((heap, item));
    }
    if arity == 2 {
        let item = crate::vm::stack::pop_direct(stack)?;
        let heap = crate::vm::stack::pop_direct(stack)?;
        return Some((heap, item));
    }
    None
}

fn pop_heappop_arg(
    stack: &mut Vec<TaggedValue>,
    arity: usize,
    available: usize,
    store: &ValueStore,
) -> Option<TaggedValue> {
    if available < 1 {
        return None;
    }
    if arity == 2 && available >= 2 {
        let heap = crate::vm::stack::pop_direct(stack)?;
        let receiver = crate::vm::stack::pop_direct(stack)?;
        if !is_heapq_module_receiver(receiver, store) {
            crate::vm::stack::push_direct(stack, receiver);
            crate::vm::stack::push_direct(stack, heap);
            return None;
        }
        return Some(heap);
    }
    if arity == 1 {
        return crate::vm::stack::pop_direct(stack);
    }
    None
}

fn maybe_migrate_heap_id_to_flat(store: &mut ValueStore, heap_id: ValueId, heavy: &HeavyStore) {
    if store.is_flat_heap(heap_id) {
        return;
    }
    let nested = match store.get(heap_id) {
        Some(ValueCell::Array(s)) => is_nested_pair_heap(s, store),
        _ => false,
    };
    if !nested {
        return;
    }
    let mut slots = match store.get_mut(heap_id) {
        Some(ValueCell::Array(v)) => std::mem::take(v),
        _ => return,
    };
    migrate_nested_to_flat(&mut slots, store, heavy);
    if let Some(ValueCell::Array(v)) = store.get_mut(heap_id) {
        *v = slots;
        store.mark_flat_heap(heap_id);
    }
}

fn migrate_nested_to_flat(slots: &mut Vec<TaggedValue>, store: &mut ValueStore, heavy: &HeavyStore) {
    if slots.is_empty() || !is_nested_pair_heap(slots, store) {
        return;
    }
    let ctx = SlotCmpCtx { store, heavy };
    let drained: Vec<TaggedValue> = slots.drain(..).collect();
    let mut flat = Vec::with_capacity(drained.len() * 2);
    let mut recycle = Vec::new();
    for tv in drained {
        if let Ok(Some((a, b))) = heap_pair_slots(tv, &ctx) {
            flat.push(a);
            flat.push(b);
            if tv.is_heap() {
                recycle.push(tv.get_heap_id());
            }
        }
    }
    *slots = flat;
    for id in recycle {
        store.recycle_heap_pair(id);
    }
}

/// Owned 2-slot pair for `heappop` expression results (not the scratch cell reused by unpack).
fn heap_pair_return_tv(a: TaggedValue, b: TaggedValue, store: &mut ValueStore) -> TaggedValue {
    TaggedValue::from_heap(store.alloc_heap_pair(a, b))
}

/// Coerce stack/local slot to an inline numeric [`TaggedValue`] for flat `[f,n,…]` heap storage.
fn coerce_flat_numeric(tv: TaggedValue, store: &ValueStore) -> Option<TaggedValue> {
    if tv.is_number() || tv.is_int() || tv.is_int_pos_inf() || tv.is_int_neg_inf() {
        return Some(tv);
    }
    if tv.is_heap() {
        return store
            .get(tv.get_heap_id())
            .and_then(value_cell_to_tagged);
    }
    None
}

/// Store `(f, node)` as a 2-slot [`ValueCell::Array`] (reuse id when already compact; convert `Tuple` in place).
fn compact_heap_push_item(tv: TaggedValue, store: &mut ValueStore) -> TaggedValue {
    if !tv.is_heap() {
        return tv;
    }
    let id = tv.get_heap_id();
    if matches!(store.get(id), Some(ValueCell::Array(v)) if v.len() == 2) {
        return tv;
    }
    let Some(ValueCell::Tuple(pair)) = store.get(id) else {
        return tv;
    };
    if pair.len() != 2 {
        return tv;
    }
    let mut slots = [TaggedValue::null(); 2];
    for (i, kid) in pair.iter().enumerate() {
        match store.get(*kid) {
            Some(cell) => {
                slots[i] = value_cell_to_tagged(cell).unwrap_or_else(|| TaggedValue::from_heap(*kid));
            }
            None => return tv,
        }
    }
    if let Some(cell) = store.get_mut(id) {
        *cell = ValueCell::Array(vec![slots[0], slots[1]]);
        store.mark_heapq_owned_pair(id);
        return TaggedValue::from_heap(id);
    }
    let arr_id = store.alloc_heap_pair(slots[0], slots[1]);
    TaggedValue::from_heap(arr_id)
}

fn try_tagged_partial_cmp(a: TaggedValue, b: TaggedValue) -> Option<Ordering> {
    if a.is_number() && b.is_number() {
        return a.get_f64().partial_cmp(&b.get_f64());
    }
    if a.is_bool() && b.is_bool() {
        return Some(a.get_bool().cmp(&b.get_bool()));
    }
    if a.is_null() && b.is_null() {
        return Some(Ordering::Equal);
    }
    if a.is_null() {
        return Some(Ordering::Less);
    }
    if b.is_null() {
        return Some(Ordering::Greater);
    }
    let to_num = |tv: TaggedValue| -> Option<Value> {
        if tv.is_number() {
            Some(Value::Number(tv.get_f64()))
        } else if tv.is_int() {
            Some(Value::Int(IntValue::Finite(tv.get_i32() as i64)))
        } else if tv.is_int_pos_inf() {
            Some(Value::Int(IntValue::PosInfinity))
        } else if tv.is_int_neg_inf() {
            Some(Value::Int(IntValue::NegInfinity))
        } else {
            None
        }
    };
    match (to_num(a), to_num(b)) {
        (Some(va), Some(vb)) => cmp_numeric_values(&va, &vb),
        _ => None,
    }
}

fn array_slots<'a>(store: &'a ValueStore, id: ValueId) -> Option<&'a [TaggedValue]> {
    match store.get(id)? {
        ValueCell::Array(v) => Some(v.as_slice()),
        _ => None,
    }
}

fn array_slots_mut<'a>(store: &'a mut ValueStore, id: ValueId) -> Option<&'a mut Vec<TaggedValue>> {
    match store.get_mut(id)? {
        ValueCell::Array(v) => Some(v),
        _ => None,
    }
}

fn heap_eligible_for_flat_push(heap_id: ValueId, store: &ValueStore) -> bool {
    if store.is_flat_heap(heap_id) {
        return true;
    }
    match store.get(heap_id) {
        Some(ValueCell::Array(slots)) if slots.is_empty() => true,
        Some(ValueCell::Array(slots)) if is_nested_pair_heap(slots, store) => true,
        _ => false,
    }
}

/// Flat `(f, node)` push for [`crate::bytecode::OpCode::HeappushFlat`].
pub(crate) fn heappush_flat_pair(
    store: &mut ValueStore,
    heavy: &HeavyStore,
    heap_id: ValueId,
    a: TaggedValue,
    b: TaggedValue,
) -> Result<(), String> {
    maybe_migrate_heap_id_to_flat(store, heap_id, heavy);
    store.mark_flat_heap(heap_id);
    let slots = array_slots_mut(store, heap_id)
        .ok_or_else(|| "TypeError: heappush() argument 1 must be an array".to_string())?;
    slots.push(a);
    slots.push(b);
    let pair_idx = slots.len() / 2 - 1;
    sift_up_flat_pairs(slots, pair_idx);
    Ok(())
}

/// Opcode/runtime path: flat push only for numeric pairs; otherwise nested tuple heap.
pub(crate) fn heappush_two_slot_pair(
    store: &mut ValueStore,
    heavy: &HeavyStore,
    heap_id: ValueId,
    a: TaggedValue,
    b: TaggedValue,
) -> Result<(), String> {
    if let (Some(a), Some(b)) = (coerce_flat_numeric(a, store), coerce_flat_numeric(b, store)) {
        if heap_eligible_for_flat_push(heap_id, store) {
            return heappush_flat_pair(store, heavy, heap_id, a, b);
        }
    }
    let item = heap_pair_return_tv(a, b, store);
    heappush_nested(store, heavy, heap_id, item)
}

fn heappop_flat(
    store: &mut ValueStore,
    heavy: &HeavyStore,
    heap_id: ValueId,
) -> Result<TaggedValue, String> {
    maybe_migrate_heap_id_to_flat(store, heap_id, heavy);
    let slots = array_slots_mut(store, heap_id)
        .ok_or_else(|| "TypeError: heappop() argument 1 must be an array".to_string())?;
    let (root_a, root_b) = pop_root_flat_pairs(slots)?;
    Ok(heap_pair_return_tv(root_a, root_b, store))
}

fn heappush_nested(
    store: &mut ValueStore,
    heavy: &HeavyStore,
    heap_id: ValueId,
    item: TaggedValue,
) -> Result<(), String> {
    let item = compact_heap_push_item(item, store);
    {
        let slots = array_slots_mut(store, heap_id)
            .ok_or_else(|| "TypeError: heappush() argument 1 must be an array".to_string())?;
        slots.push(item);
    }
    let mut index = array_slots(store, heap_id).map(|s| s.len().saturating_sub(1)).unwrap_or(0);
    while index > 0 {
        let parent = (index - 1) / 2;
        let ord = {
            let slots = array_slots(store, heap_id).expect("heap just pushed");
            let ctx = SlotCmpCtx { store, heavy };
            ctx.compare(slots[parent], slots[index])?
        };
        if ord != Ordering::Greater {
            break;
        }
        {
            let slots = array_slots_mut(store, heap_id).expect("heap");
            slots.swap(parent, index);
        }
        index = parent;
    }
    Ok(())
}

fn heappop_nested(
    store: &mut ValueStore,
    heavy: &HeavyStore,
    heap_id: ValueId,
) -> Result<TaggedValue, String> {
    let n = array_slots(store, heap_id)
        .ok_or_else(|| "TypeError: heappop() argument 1 must be an array".to_string())?
        .len();
    if n == 0 {
        return Err("IndexError: heappop from empty heap".to_string());
    }
    if n == 1 {
        let slots = array_slots_mut(store, heap_id).expect("heap");
        return Ok(slots.pop().unwrap_or(TaggedValue::null()));
    }
    let root = array_slots(store, heap_id).expect("heap")[0];
    {
        let slots = array_slots_mut(store, heap_id).expect("heap");
        slots[0] = slots.pop().unwrap();
    }
    let mut index = 0usize;
    let n = array_slots(store, heap_id).expect("heap").len();
    loop {
        let left = 2 * index + 1;
        let right = 2 * index + 2;
        let mut smallest = index;
        let ctx = SlotCmpCtx { store, heavy };
        if left < n {
            let slots = array_slots(store, heap_id).expect("heap");
            if ctx.compare(slots[left], slots[smallest])? == Ordering::Less {
                smallest = left;
            }
        }
        if right < n {
            let slots = array_slots(store, heap_id).expect("heap");
            if ctx.compare(slots[right], slots[smallest])? == Ordering::Less {
                smallest = right;
            }
        }
        if smallest == index {
            break;
        }
        {
            let slots = array_slots_mut(store, heap_id).expect("heap");
            slots.swap(index, smallest);
        }
        index = smallest;
    }
    Ok(root)
}

pub(crate) fn heap_clear_store(store: &mut ValueStore, heap_id: ValueId) -> Result<(), String> {
    if store.plain_array_clear_shrink(heap_id) {
        Ok(())
    } else {
        Err("TypeError: heap_clear() argument must be an array".to_string())
    }
}

pub(crate) fn heappush_store(
    store: &mut ValueStore,
    heavy: &HeavyStore,
    heap_id: ValueId,
    item: TaggedValue,
) -> Result<(), String> {
    if let Some((a, b)) = pair_from_item(item, store, heavy) {
        if let (Some(a), Some(b)) = (coerce_flat_numeric(a, store), coerce_flat_numeric(b, store)) {
            if heap_eligible_for_flat_push(heap_id, store) {
                return heappush_flat_pair(store, heavy, heap_id, a, b);
            }
        }
    }
    heappush_nested(store, heavy, heap_id, item)
}

pub(crate) fn heappop_store(
    store: &mut ValueStore,
    heavy: &HeavyStore,
    heap_id: ValueId,
) -> Result<TaggedValue, String> {
    let use_flat = can_use_flat_pair_heap(heap_id, store);
    if use_flat {
        heappop_flat(store, heavy, heap_id)
    } else {
        heappop_nested(store, heavy, heap_id)
    }
}

fn compare_flat_pairs_in_slice(slots: &[TaggedValue], p: usize, q: usize) -> Ordering {
    let a0 = slots[2 * p];
    let a1 = slots[2 * p + 1];
    let b0 = slots[2 * q];
    let b1 = slots[2 * q + 1];
    match try_tagged_partial_cmp(a0, b0) {
        Some(Ordering::Equal) | None => {
            try_tagged_partial_cmp(a1, b1).unwrap_or(Ordering::Equal)
        }
        Some(ord) => ord,
    }
}

fn swap_flat_pairs_in_slice(slots: &mut [TaggedValue], a: usize, b: usize) {
    let (left, right) = slots.split_at_mut(2 * b);
    let a_base = 2 * a;
    left[a_base..a_base + 2].swap_with_slice(&mut right[..2]);
}

fn sift_up_flat_pairs(slots: &mut [TaggedValue], mut pair_idx: usize) {
    while pair_idx > 0 {
        let parent = (pair_idx - 1) / 2;
        if compare_flat_pairs_in_slice(slots, parent, pair_idx) != Ordering::Greater {
            break;
        }
        swap_flat_pairs_in_slice(slots, parent, pair_idx);
        pair_idx = parent;
    }
}

fn sift_down_flat_pairs(slots: &mut [TaggedValue], mut pair_idx: usize, n_pairs: usize) {
    loop {
        let left = 2 * pair_idx + 1;
        let right = 2 * pair_idx + 2;
        let mut smallest = pair_idx;
        if left < n_pairs && compare_flat_pairs_in_slice(slots, left, smallest) == Ordering::Less {
            smallest = left;
        }
        if right < n_pairs && compare_flat_pairs_in_slice(slots, right, smallest) == Ordering::Less {
            smallest = right;
        }
        if smallest == pair_idx {
            break;
        }
        swap_flat_pairs_in_slice(slots, pair_idx, smallest);
        pair_idx = smallest;
    }
}

/// Pop root pair from flat `[f,n,…]` storage; `sift_down` runs on `slots` without repeated `store.get`.
fn pop_root_flat_pairs(slots: &mut Vec<TaggedValue>) -> Result<(TaggedValue, TaggedValue), String> {
    let n_pairs = slots.len() / 2;
    if n_pairs == 0 {
        return Err("IndexError: heappop from empty heap".to_string());
    }
    let (root_a, root_b) = (slots[0], slots[1]);
    if n_pairs == 1 {
        slots.clear();
        return Ok((root_a, root_b));
    }
    let last_n = slots.pop().expect("pair");
    let last_f = slots.pop().expect("pair");
    slots[0] = last_f;
    slots[1] = last_n;
    let n_pairs = slots.len() / 2;
    sift_down_flat_pairs(slots, 0, n_pairs);
    Ok((root_a, root_b))
}

/// `HeappopUnpack2`: pop from heap local into two locals without tuple temp / pair shell alloc.
pub(crate) fn heappop_unpack2_locals(
    store: &mut ValueStore,
    heavy: &HeavyStore,
    heap_id: ValueId,
) -> Result<(TaggedValue, TaggedValue), String> {
    let use_flat = can_use_flat_pair_heap(heap_id, store);
    if use_flat {
        maybe_migrate_heap_id_to_flat(store, heap_id, heavy);
        let slots = match store.get_mut(heap_id) {
            Some(ValueCell::Array(v)) => v,
            _ => return Err("TypeError: heappop expects array heap".to_string()),
        };
        return pop_root_flat_pairs(slots);
    }
    let root = heappop_nested(store, heavy, heap_id)?;
    if let Some((a, b)) = pair_from_item(root, store, heavy) {
        return Ok((a, b));
    }
    if root.is_heap() {
        if let Some(ValueCell::Array(slots)) = store.get(root.get_heap_id()) {
            if slots.len() == 2 {
                return Ok((slots[0], slots[1]));
            }
        }
    }
    Err("TypeError: heappop unpack expects (f, node) pair".to_string())
}

fn runtime_err(
    stack: &mut Vec<TaggedValue>,
    frames: &mut Vec<CallFrame>,
    exception_handlers: &mut Vec<ExceptionHandler>,
    value_store: &mut ValueStore,
    heavy_store: &mut HeavyStore,
    line: usize,
    msg: String,
    err_type: ErrorType,
) -> Result<VMStatus, LangError> {
    let error = ExceptionHandler::runtime_error_with_type(&frames, msg, line, err_type);
    ExceptionHandler::handle_exception_vm(
        stack,
        frames,
        exception_handlers,
        error,
        value_store,
        heavy_store,
    )
}

fn err_type_for_heapq_msg(msg: &str) -> ErrorType {
    if msg.starts_with("IndexError") {
        ErrorType::IndexError
    } else if msg.starts_with("TypeError") {
        ErrorType::TypeError
    } else {
        ErrorType::RuntimeError
    }
}

/// Match by native function pointer (stable across module registration order).
pub(super) fn try_heapq_fast_path(
    native_ptr: Option<*const ()>,
    arity: usize,
    line: usize,
    stack: &mut Vec<TaggedValue>,
    frames: &mut Vec<CallFrame>,
    exception_handlers: &mut Vec<ExceptionHandler>,
    value_store: &mut ValueStore,
    heavy_store: &mut HeavyStore,
) -> Option<Result<VMStatus, LangError>> {
    let is_heappush = native_ptr == Some(native_heapq_heappush as *const ());
    let is_heappop = native_ptr == Some(native_heapq_heappop as *const ());
    let is_heappeek = native_ptr == Some(native_heapq_heappeek as *const ());
    let is_heap_clear = native_ptr == Some(crate::heapq::natives::native_heapq_heap_clear as *const ());
    if !is_heappush && !is_heappop && !is_heappeek && !is_heap_clear {
        return None;
    }

    let frame = frames.last()?;
    let available = crate::vm::stack::available_in_frame(stack, frame.stack_start);

    if is_heap_clear {
        let heap_tv = pop_heappop_arg(stack, arity, available, value_store)?;
        let heap_id = tagged_to_value_id(heap_tv, value_store);
        match heap_clear_store(value_store, heap_id) {
            Ok(()) => {
                stack::push(stack, TaggedValue::null());
                Some(Ok(VMStatus::Continue))
            }
            Err(msg) => {
                stack::push(stack, heap_tv);
                Some(runtime_err(
                    stack,
                    frames,
                    exception_handlers,
                    value_store,
                    heavy_store,
                    line,
                    msg,
                    ErrorType::TypeError,
                ))
            }
        }
    } else if is_heappush {
        let (heap_tv, item_tv) = pop_heappush_args(stack, arity, available, value_store)?;
        let heap_id = tagged_to_value_id(heap_tv, value_store);
        if value_store.get(heap_id).is_none() {
            stack::push(stack, heap_tv);
            stack::push(stack, item_tv);
            return None;
        }
        match heappush_store(value_store, heavy_store, heap_id, item_tv) {
            Ok(()) => {
                stack::push(stack, TaggedValue::null());
                Some(Ok(VMStatus::Continue))
            }
            Err(msg) => {
                stack::push(stack, heap_tv);
                stack::push(stack, item_tv);
                let err_type = err_type_for_heapq_msg(&msg);
                Some(runtime_err(
                    stack,
                    frames,
                    exception_handlers,
                    value_store,
                    heavy_store,
                    line,
                    msg,
                    err_type,
                ))
            }
        }
    } else if is_heappop {
        let heap_tv = pop_heappop_arg(stack, arity, available, value_store)?;
        let heap_id = tagged_to_value_id(heap_tv, value_store);
        if !matches!(value_store.get(heap_id), Some(ValueCell::Array(_))) {
            stack::push(stack, heap_tv);
            return None;
        }
        match heappop_store(value_store, heavy_store, heap_id) {
            Ok(root) => {
                stack::push(stack, root);
                Some(Ok(VMStatus::Continue))
            }
            Err(msg) => {
                stack::push(stack, heap_tv);
                let err_type = err_type_for_heapq_msg(&msg);
                Some(runtime_err(
                    stack,
                    frames,
                    exception_handlers,
                    value_store,
                    heavy_store,
                    line,
                    msg,
                    err_type,
                ))
            }
        }
    } else if is_heappeek {
        // heappeek
        let heap_tv = pop_heappop_arg(stack, arity, available, value_store)?;
        let heap_id = tagged_to_value_id(heap_tv, value_store);
        let slots = match array_slots(value_store, heap_id) {
            Some(s) => s,
            None => {
                stack::push(stack, heap_tv);
                return None;
            }
        };
        if slots.is_empty() {
            stack::push(stack, heap_tv);
            return Some(runtime_err(
                stack,
                frames,
                exception_handlers,
                value_store,
                heavy_store,
                line,
                "IndexError: peek from empty heap".to_string(),
                ErrorType::IndexError,
            ));
        }
        if can_use_flat_pair_heap(heap_id, value_store) {
            let out = heap_pair_return_tv(slots[0], slots[1], value_store);
            if out.is_heap() {
                push_stack_value_id(stack, value_store, out.get_heap_id());
            } else {
                stack::push(stack, out);
            }
        } else {
            let root = slots[0];
            if root.is_heap() {
                push_stack_value_id(stack, value_store, root.get_heap_id());
            } else {
                stack::push(stack, root);
            }
        }
        Some(Ok(VMStatus::Continue))
    } else {
        None
    }
}
