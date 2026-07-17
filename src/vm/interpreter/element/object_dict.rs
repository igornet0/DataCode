//! `.keys` / `.values` on plain bucket dicts ([`ValueCell::Object`] without class metadata).

use crate::common::object_map::ObjectMap;
use crate::common::TaggedValue;
use crate::common::value_store::{
    ObjectProjectionKind, ValueCell, ValueId, ValueStore,
};
use crate::vm::heavy_store::HeavyStore;
use crate::vm::memory::{object_map_try_lookup_by_key_id, push_integral_slot, push_stack_value_id};

/// If `index_value_id` refers to interned `"keys"` or `"values"`, return the corresponding projection.
pub(crate) fn projection_for_keys_values_property(
    index_value_id: ValueId,
    store: &ValueStore,
) -> Option<ObjectProjectionKind> {
    let ValueCell::String(sid) = store.get(index_value_id)? else {
        return None;
    };
    match store.get_string(*sid)? {
        "keys" => Some(ObjectProjectionKind::Keys),
        "values" => Some(ObjectProjectionKind::Values),
        _ => None,
    }
}

/// Build [`ValueCell::ObjectFieldList`] (snapshot of key or value ids). Caller must ensure a plain dict.
pub(crate) fn allocate_object_field_list(
    source_object_id: ValueId,
    omap: &ObjectMap,
    projection: ObjectProjectionKind,
    store: &mut ValueStore,
) -> ValueId {
    let element_ids: Vec<ValueId> = if omap.integral_in_use() {
        omap.iter_integral_canonicals()
            .filter_map(|canonical| {
                let slot = omap.find_integral_slot(canonical)?;
                match projection {
                    ObjectProjectionKind::Keys => Some(canonical_to_key_id(canonical, store)),
                    ObjectProjectionKind::Values => Some(integral_slot_to_value_id_for_list(
                        slot,
                        store,
                    )),
                }
            })
            .collect()
    } else {
        omap.iter_entries()
            .map(|(_, kid, vid)| match projection {
                ObjectProjectionKind::Keys => kid,
                ObjectProjectionKind::Values => vid,
            })
            .collect()
    };
    store.allocate(ValueCell::ObjectFieldList {
        source_object_id,
        projection,
        element_ids,
    })
}

#[inline]
fn canonical_to_key_id(canonical: i64, store: &mut ValueStore) -> ValueId {
    if (-(i32::MAX as i64)..=i32::MAX as i64).contains(&canonical) {
        return store.intern_whole_i64(canonical);
    }
    store.intern_number_f64(canonical as f64)
}

#[inline]
fn integral_slot_to_value_id_for_list(
    slot: crate::common::integral_map::IntegralSlot,
    store: &mut ValueStore,
) -> ValueId {
    use crate::common::integral_map::IntegralSlot;
    match slot {
        IntegralSlot::Heap(id) => id,
        IntegralSlot::Immediate(tv) => crate::vm::memory::tagged_to_value_id(tv, store),
    }
}

/// Push the key or **live** value at `index` (`.values` reads the current integral/bucket cell).
pub(crate) fn push_object_projection_element(
    source_object_id: ValueId,
    projection: ObjectProjectionKind,
    index: usize,
    stack: &mut Vec<TaggedValue>,
    store: &mut ValueStore,
    heap: &HeavyStore,
) -> bool {
    let omap = match store.get(source_object_id) {
        Some(ValueCell::Object(omap)) => omap.clone(),
        _ => return false,
    };
    push_object_projection_element_from_map(&omap, projection, index, stack, store, heap)
}

fn push_object_projection_element_from_map(
    omap: &ObjectMap,
    projection: ObjectProjectionKind,
    index: usize,
    stack: &mut Vec<TaggedValue>,
    store: &mut ValueStore,
    heap: &HeavyStore,
) -> bool {
    if index >= omap.len() {
        return false;
    }
    if omap.integral_in_use() {
        let Some(canonical) = omap.iter_integral_canonicals().nth(index) else {
            return false;
        };
        return match projection {
            ObjectProjectionKind::Keys => {
                let kid = canonical_to_key_id(canonical, store);
                push_stack_value_id(stack, store, kid);
                true
            }
            ObjectProjectionKind::Values => {
                let Some(slot) = omap.find_integral_slot(canonical) else {
                    return false;
                };
                push_integral_slot(stack, store, slot);
                true
            }
        };
    }
    let Some((_, kid, _)) = omap.iter_entries().nth(index) else {
        return false;
    };
    match projection {
        ObjectProjectionKind::Keys => {
            push_stack_value_id(stack, store, kid);
            true
        }
        ObjectProjectionKind::Values => {
            let Some(live) = object_map_try_lookup_by_key_id(omap, kid, store, heap) else {
                return false;
            };
            push_stack_value_id(stack, store, live);
            true
        }
    }
}
