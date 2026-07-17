//! Zero-copy array views: slice and chunk share backing storage (ValueStore array or Rc<Vec>).

use crate::common::value::{ArrayViewData, ArrayViewSource, Value};
use crate::common::value_store::{ValueCell, ValueId, ValueStore};
use crate::common::TaggedValue;
use crate::vm::heavy_store::HeavyStore;
use crate::vm::store_convert::slot_to_value;
use std::cell::RefCell;
use std::rc::Rc;

/// Resolved backing for slice/length (array cell or heap view).
#[derive(Clone)]
pub enum SliceOrigin {
    Store {
        base_id: ValueId,
        offset: usize,
        length: usize,
    },
    Heap {
        vec: Rc<RefCell<Vec<Value>>>,
        offset: usize,
        length: usize,
    },
}

impl SliceOrigin {
    pub fn len(&self) -> usize {
        match self {
            SliceOrigin::Store { length, .. } | SliceOrigin::Heap { length, .. } => *length,
        }
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Resolve container to a contiguous logical range for slicing/indexing.
pub fn resolve_slice_origin(
    container_id: ValueId,
    store: &ValueStore,
    heap: &HeavyStore,
) -> Option<SliceOrigin> {
    match store.get(container_id)? {
        ValueCell::Array(slots) => Some(SliceOrigin::Store {
            base_id: container_id,
            offset: 0,
            length: slots.len(),
        }),
        ValueCell::ArrayView {
            base_id,
            offset,
            length,
        } => Some(SliceOrigin::Store {
            base_id: *base_id,
            offset: *offset,
            length: *length,
        }),
        ValueCell::Heavy(h) => match heap.get(*h)? {
            Value::ArrayView(av) => match &av.source {
                ArrayViewSource::Store { base_id } => Some(SliceOrigin::Store {
                    base_id: *base_id,
                    offset: av.offset,
                    length: av.length,
                }),
                ArrayViewSource::Heap(rc) => Some(SliceOrigin::Heap {
                    vec: rc.clone(),
                    offset: av.offset,
                    length: av.length,
                }),
            },
            _ => None,
        },
        _ => None,
    }
}

/// Narrow a view to `[local_offset .. local_offset + local_length)` within the view (zero-copy).
pub fn subview(
    av: &ArrayViewData,
    local_offset: usize,
    local_length: usize,
) -> Option<ArrayViewData> {
    if local_length == 0 {
        return Some(ArrayViewData {
            source: match &av.source {
                ArrayViewSource::Store { base_id } => ArrayViewSource::Store { base_id: *base_id },
                ArrayViewSource::Heap(rc) => ArrayViewSource::Heap(Rc::clone(rc)),
            },
            offset: av.offset + local_offset,
            length: 0,
        });
    }
    local_offset
        .checked_add(local_length)
        .filter(|&end| end <= av.length)
        .map(|_| ArrayViewData {
            source: match &av.source {
                ArrayViewSource::Store { base_id } => ArrayViewSource::Store { base_id: *base_id },
                ArrayViewSource::Heap(rc) => ArrayViewSource::Heap(Rc::clone(rc)),
            },
            offset: av.offset + local_offset,
            length: local_length,
        })
}

/// Build [`ArrayViewData`] from origin (shared helper).
pub fn origin_to_view_data(origin: &SliceOrigin) -> ArrayViewData {
    match origin {
        SliceOrigin::Store {
            base_id,
            offset,
            length,
        } => ArrayViewData {
            source: ArrayViewSource::Store { base_id: *base_id },
            offset: *offset,
            length: *length,
        },
        SliceOrigin::Heap {
            vec,
            offset,
            length,
        } => ArrayViewData {
            source: ArrayViewSource::Heap(vec.clone()),
            offset: *offset,
            length: *length,
        },
    }
}

/// Read one element from a view (may clone [`Value`] from backing).
pub fn view_get_element(
    av: &ArrayViewData,
    i: usize,
    store: &ValueStore,
    heap: &HeavyStore,
) -> Option<Value> {
    if i >= av.length {
        return None;
    }
    let abs = av.offset + i;
    match &av.source {
        ArrayViewSource::Store { base_id } => {
            let cell = store.get(*base_id)?;
            match cell {
                ValueCell::Array(slots) => {
                    if abs >= slots.len() {
                        return None;
                    }
                    Some(slot_to_value(slots[abs], store, heap))
                }
                _ => None,
            }
        }
        ArrayViewSource::Heap(rc) => {
            let b = rc.borrow();
            b.get(abs).cloned()
        }
    }
}

/// Copy view contents into a new owned [`Value::Array`].
pub fn materialize_array_view(av: &ArrayViewData, store: &ValueStore, heap: &HeavyStore) -> Value {
    let mut out: Vec<Value> = Vec::with_capacity(av.length);
    for i in 0..av.length {
        if let Some(v) = view_get_element(av, i, store, heap) {
            out.push(v);
        } else {
            out.push(Value::Null);
        }
    }
    Value::Array(Rc::new(RefCell::new(out)))
}

/// When assigning a slice view, store an owned copy so later mutations do not alias the parent.
pub fn materialize_value_if_array_view(
    value: Value,
    store: &ValueStore,
    heap: &HeavyStore,
) -> Value {
    match value {
        Value::ArrayView(av) => materialize_array_view(&av, store, heap),
        other => other,
    }
}

/// Materialize heap-backed array views before storing into locals / fields.
pub fn materialize_tagged_if_array_view(
    tv: TaggedValue,
    store: &mut ValueStore,
    heap: &mut HeavyStore,
) -> TaggedValue {
    if !tv.is_heap() {
        return tv;
    }
    let id = tv.get_heap_id();
    let v = crate::vm::store_convert::load_value(id, store, heap);
    match v {
        Value::ArrayView(av) => {
            let mat = materialize_array_view(&av, store, heap);
            TaggedValue::from_heap(crate::vm::store_convert::store_value(mat, store, heap))
        }
        _ => tv,
    }
}

/// Store an [`ArrayViewData`] as [`ValueCell::ArrayView`] (store-backed) or [`ValueCell::Heavy`] (heap-backed).
pub fn store_array_view_data(
    av: ArrayViewData,
    store: &mut ValueStore,
    heap: &mut HeavyStore,
) -> ValueId {
    match &av.source {
        ArrayViewSource::Store { base_id } => store.allocate(ValueCell::ArrayView {
            base_id: *base_id,
            offset: av.offset,
            length: av.length,
        }),
        ArrayViewSource::Heap(_) => {
            let idx = heap.push(Value::ArrayView(av));
            store.allocate(ValueCell::Heavy(idx))
        }
    }
}

/// Check that `[offset .. offset + length)` lies within the physical backing.
pub fn validate_view_physical(av: &ArrayViewData, store: &ValueStore) -> Result<(), String> {
    match &av.source {
        ArrayViewSource::Store { base_id } => {
            if let Some(ValueCell::Array(slots)) = store.get(*base_id) {
                if av.offset + av.length <= slots.len() {
                    return Ok(());
                }
            }
            Err("array view out of bounds".to_string())
        }
        ArrayViewSource::Heap(rc) => {
            let b = rc.borrow();
            if av.offset + av.length <= b.len() {
                Ok(())
            } else {
                Err("array view out of bounds".to_string())
            }
        }
    }
}

/// Push a [`Value`] built from `ArrayViewData` (used after constructing view).
pub fn push_array_view_value(
    av: ArrayViewData,
    stack: &mut Vec<crate::common::TaggedValue>,
    store: &mut ValueStore,
    heap: &mut HeavyStore,
) {
    let id = store_array_view_data(av, store, heap);
    crate::vm::stack::push_id(stack, id);
}
