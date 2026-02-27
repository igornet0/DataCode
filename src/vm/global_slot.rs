// Global variable slot: Inline(TaggedValue) for primitives (no store alloc/get), Heap(ValueId) for the rest.
// Minimizes store_alloc and store_get for mass data scenarios.

use crate::common::value_store::{ValueId, ValueStore, NULL_VALUE_ID};
use crate::common::TaggedValue;
use crate::vm::store_convert::tagged_to_value_id_arena;

/// One global slot: either an inline primitive (number, bool, null, int) or a heap reference.
#[derive(Clone, Copy, Debug)]
pub enum GlobalSlot {
    Inline(TaggedValue),
    Heap(ValueId),
}

impl GlobalSlot {
    /// Default uninitialized slot (null in store semantics).
    pub fn null() -> Self {
        GlobalSlot::Heap(NULL_VALUE_ID)
    }

    /// Resolve to ValueId for reading. For Inline: materialize once, then switch slot to Heap(id) so
    /// subsequent calls return the same id without allocate. For Heap returns the id.
    pub fn resolve_to_value_id(&mut self, store: &mut ValueStore) -> ValueId {
        match self {
            GlobalSlot::Inline(tv) => {
                let id = tagged_to_value_id_arena(*tv, store);
                *self = GlobalSlot::Heap(id);
                id
            }
            GlobalSlot::Heap(id) => *id,
        }
    }
}

/// Resize/initialization fill value for globals vec.
pub fn default_global_slot() -> GlobalSlot {
    GlobalSlot::null()
}
