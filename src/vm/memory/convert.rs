// Conversion between Value and ValueId for Stage 1 ValueStore migration.
// Used at native-call boundaries: materialize Value from ValueId for natives, store Value result back as ValueId.

use crate::common::numeric::{
    f64_trunc_to_i64_clamped, integer_cell_as_i64_if_whole, integer_value_as_i64_if_whole,
    FloatValue, IntValue,
};
use crate::common::integral_map::IntegralSlot;
use crate::common::object_map::{ObjectEntry, ObjectMap};
use crate::common::value::{ArrayViewData, ArrayViewSource, ObjectKind, Value};
use crate::common::value_store::{ValueCell, ValueId, ValueStore, CALL_ARENA_BASE, NULL_VALUE_ID};
use crate::common::TaggedValue;
use crate::vm::stack;
use chrono::DateTime;
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::rc::Rc;

use super::store::HeavyStore;

/// Insert/replace into bucket map comparing stored keys to `key_material` via [`load_value`].
/// Numeric keys (`int` / whole `number` / whole `float`) share one hash bucket ([`numeric::hash_numeric_key_value`]);
/// equality uses [`PartialEq`] so `1` and `1.0` update the same entry.
pub fn object_map_upsert(
    omap: &mut ObjectMap,
    store: &ValueStore,
    heap: &HeavyStore,
    key_material: &Value,
    kid: ValueId,
    vid: ValueId,
) {
    if let Some(canonical) = integer_value_as_i64_if_whole(key_material) {
        let had = omap.contains_integral(canonical);
        omap.upsert_integral(canonical, kid, vid);
        // Rewrite of existing int key (A* g_score/f_score): skip bucket chain sync.
        if had {
            return;
        }
    }
    let h = crate::common::type_model::object_key_hash_value(key_material).expect("hashable key");
    let bucket = omap.bucket_mut_or_insert(h);
    for e in bucket.iter_mut() {
        if load_value(e.key_id, store, heap) == *key_material {
            e.value_id = vid;
            return;
        }
    }
    bucket.push(ObjectEntry {
        key_id: kid,
        value_id: vid,
    });
}

/// Update a plain dict in place without cloning the whole [`ObjectMap`].
/// Temporarily moves the map out of [`ValueStore`] so bucket equality can use `load_value`.
#[inline]
pub fn object_map_upsert_in_place(
    value_store: &mut ValueStore,
    container_id: ValueId,
    heap: &HeavyStore,
    key_material: &Value,
    kid: ValueId,
    vid: ValueId,
) -> bool {
    if let Some(canonical) = crate::common::numeric::integer_value_as_i64_if_whole(key_material)
        .or_else(|| canonical_integral_from_key_id(kid, value_store))
    {
        if value_store.plain_object_upsert_integral(container_id, canonical, kid, vid) {
            return true;
        }
    }
    let mut omap = match value_store.get_mut(container_id) {
        Some(ValueCell::Object(omap)) => std::mem::take(omap),
        _ => return false,
    };
    object_map_upsert(&mut omap, value_store, heap, key_material, kid, vid);
    if let Some(ValueCell::Object(slot)) = value_store.get_mut(container_id) {
        *slot = omap;
    }
    true
}

/// Push dict integral lookup result without materializing [`Value`].
#[inline]
pub fn push_integral_slot(
    stack: &mut Vec<TaggedValue>,
    value_store: &mut ValueStore,
    slot: IntegralSlot,
) {
    match slot {
        IntegralSlot::Immediate(tv) => stack::push(stack, tv),
        IntegralSlot::Heap(id) => push_stack_value_id(stack, value_store, id),
    }
}

/// Push a store value: immediates as [`TaggedValue`], heap refs as tagged heap id.
#[inline]
pub fn push_stack_value_id(
    stack: &mut Vec<TaggedValue>,
    value_store: &mut ValueStore,
    id: ValueId,
) {
    if id == NULL_VALUE_ID {
        stack::push(stack, TaggedValue::null());
        return;
    }
    if let Some(cell) = value_store.get_mut(id) {
        if let Some(tv) = value_cell_to_tagged(cell) {
            stack::push(stack, tv);
            return;
        }
    }
    stack::push_id(stack, id);
}

/// Truthiness for control flow (`while`, `if`) without materializing large [`ValueCell::Array`] / dict / set.
pub fn value_id_is_truthy(id: ValueId, store: &ValueStore, heap: &HeavyStore) -> bool {
    if id == NULL_VALUE_ID {
        return false;
    }
    let Some(cell) = store.get(id) else {
        return false;
    };
    match cell {
        ValueCell::Null => false,
        ValueCell::Bool(b) => *b,
        ValueCell::Number(n) => *n != 0.0,
        ValueCell::Int(IntValue::Finite(n)) => *n != 0,
        ValueCell::Int(IntValue::PosInfinity | IntValue::NegInfinity) => true,
        ValueCell::Float(f) => match f {
            FloatValue::Finite(n) => *n != 0.0,
            FloatValue::NaN | FloatValue::PosInfinity | FloatValue::NegInfinity => true,
        },
        ValueCell::String(sid) => store
            .get_string(*sid)
            .map(|s| !s.is_empty())
            .unwrap_or(false),
        ValueCell::Array(slots) => {
            if store.is_flat_heap(id) {
                slots.len() >= 2
            } else {
                !slots.is_empty()
            }
        }
        ValueCell::Set(smap) => !smap.is_empty(),
        ValueCell::Object(omap) => !omap.is_empty(),
        ValueCell::Tuple(ids) => !ids.is_empty(),
        ValueCell::Heavy(idx) => heap
            .get(*idx)
            .map(|v| v.is_truthy())
            .unwrap_or(false),
        _ => load_value(id, store, heap).is_truthy(),
    }
}

/// Same as [`value_id_is_truthy`] via [`ValueStore::get_mut`] for common cell kinds (no profile `get` tick).
pub fn value_id_is_truthy_mut(id: ValueId, store: &mut ValueStore, heap: &HeavyStore) -> bool {
    if id == NULL_VALUE_ID {
        return false;
    }
    let is_flat = store.is_flat_heap(id);
    match store.get_mut(id) {
        Some(ValueCell::Null) => false,
        Some(ValueCell::Bool(b)) => *b,
        Some(ValueCell::Number(n)) => *n != 0.0,
        Some(ValueCell::Int(IntValue::Finite(n))) => *n != 0,
        Some(ValueCell::Int(IntValue::PosInfinity | IntValue::NegInfinity)) => true,
        Some(ValueCell::Float(f)) => match f {
            FloatValue::Finite(n) => *n != 0.0,
            FloatValue::NaN | FloatValue::PosInfinity | FloatValue::NegInfinity => true,
        },
        Some(ValueCell::Array(slots)) => {
            if is_flat {
                slots.len() >= 2
            } else {
                !slots.is_empty()
            }
        }
        Some(ValueCell::Set(smap)) => !smap.is_empty(),
        Some(ValueCell::Object(omap)) => !omap.is_empty(),
        Some(ValueCell::Tuple(ids)) => !ids.is_empty(),
        _ => value_id_is_truthy(id, store, heap),
    }
}

/// Overwrite an existing numeric store cell from a stack [`TaggedValue`] (no new [`ValueId`]).
#[inline]
pub fn try_write_cell_from_tagged(
    id: ValueId,
    tv: TaggedValue,
    store: &mut ValueStore,
) -> bool {
    if id == NULL_VALUE_ID || store.is_interned_scalar(id) {
        return false;
    }
    let Some(cell) = store.get_mut(id) else {
        return false;
    };
    match cell {
        ValueCell::Number(n) => {
            if tv.is_number() {
                *n = tv.get_f64();
                return true;
            }
            if tv.is_int() {
                *n = tv.get_i32() as f64;
                return true;
            }
        }
        ValueCell::Int(iv @ IntValue::Finite(_)) => {
            if tv.is_int() {
                *iv = IntValue::Finite(tv.get_i32() as i64);
                return true;
            }
            if tv.is_number() {
                let f = tv.get_f64();
                if f.is_finite() && f.fract() == 0.0 {
                    *iv = IntValue::Finite(f64_trunc_to_i64_clamped(f));
                    return true;
                }
            }
        }
        _ => {}
    }
    false
}

/// Canonical integral key from a store cell id without materializing [`Value`].
#[inline]
pub fn canonical_integral_from_key_id(key_id: ValueId, store: &ValueStore) -> Option<i64> {
    if key_id == NULL_VALUE_ID {
        return None;
    }
    store
        .get(key_id)
        .and_then(integer_cell_as_i64_if_whole)
}

/// Same as [`canonical_integral_from_key_id`] via [`ValueStore::get_mut`] (no profile `get` tick).
#[inline]
pub fn canonical_integral_from_key_id_mut(key_id: ValueId, store: &mut ValueStore) -> Option<i64> {
    if key_id == NULL_VALUE_ID {
        return None;
    }
    store
        .get_mut(key_id)
        .as_deref()
        .and_then(integer_cell_as_i64_if_whole)
}

/// Materialize an integral dict slot to a [`ValueId`] (inline immediates interned in `store`).
#[inline]
pub fn integral_slot_to_value_id(slot: IntegralSlot, store: &mut ValueStore) -> ValueId {
    match slot {
        IntegralSlot::Heap(id) => id,
        IntegralSlot::Immediate(tv) if tv.is_null() => NULL_VALUE_ID,
        IntegralSlot::Immediate(tv) => tagged_to_value_id(tv, store),
    }
}

/// Lookup integral side table by key id (whole-number keys only).
pub fn object_map_find_integral_slot_by_key_id(
    map: &ObjectMap,
    key_id: ValueId,
    store: &ValueStore,
    heap: &HeavyStore,
) -> Option<IntegralSlot> {
    let canonical = canonical_integral_from_key_id(key_id, store).or_else(|| {
        let key_material = load_value(key_id, store, heap);
        integer_value_as_i64_if_whole(&key_material)
    })?;
    map.find_integral_slot(canonical)
}

/// Lookup `key_id` in a bucket [`ObjectMap`] using [`object_key_hash_value`] + structural equality.
/// Returns [`None`] if the key is absent or unhashable; [`Some`] if present (value id may be
/// [`NULL_VALUE_ID`] when the stored value is null).
/// Immediate integral values are returned only via [`object_map_find_integral_slot_by_key_id`].
pub fn object_map_try_lookup_by_key_id(
    map: &ObjectMap,
    key_id: ValueId,
    store: &mut ValueStore,
    heap: &HeavyStore,
) -> Option<ValueId> {
    if let Some(slot) = object_map_find_integral_slot_by_key_id(map, key_id, store, heap) {
        return Some(integral_slot_to_value_id(slot, store));
    }
    let key_material = load_value(key_id, store, heap);
    let h = crate::common::type_model::object_key_hash_value(&key_material)?;
    map.find_in_bucket(h, |k_id| load_value(k_id, store, heap) == key_material)
}

/// Resolve a dict key to [`Value`] without mutating the store (integral immediates via [`slot_to_value`]).
pub fn object_map_lookup_value(
    omap: &ObjectMap,
    key_id: ValueId,
    store: &ValueStore,
    heap: &HeavyStore,
) -> Option<Value> {
    if let Some(slot) = object_map_find_integral_slot_by_key_id(omap, key_id, store, heap) {
        return Some(match slot {
            IntegralSlot::Heap(id) => load_value(id, store, heap),
            IntegralSlot::Immediate(tv) => slot_to_value(tv, store, heap),
        });
    }
    let key_material = load_value(key_id, store, heap);
    let h = crate::common::type_model::object_key_hash_value(&key_material)?;
    let id = omap.find_in_bucket(h, |k_id| load_value(k_id, store, heap) == key_material)?;
    Some(load_value(id, store, heap))
}

/// Lookup `obj_id[key_id]` when `obj_id` is a plain [`ValueCell::Object`].
pub fn object_cell_try_lookup_by_key_id(
    obj_id: ValueId,
    key_id: ValueId,
    store: &mut ValueStore,
    heap: &HeavyStore,
) -> Option<ValueId> {
    let integral_slot = match store.get(obj_id) {
        Some(ValueCell::Object(omap)) => {
            object_map_find_integral_slot_by_key_id(omap, key_id, store, heap)
        }
        _ => None,
    };
    if let Some(slot) = integral_slot {
        return Some(integral_slot_to_value_id(slot, store));
    }
    let key_material = load_value(key_id, store, heap);
    let h = crate::common::type_model::object_key_hash_value(&key_material)?;
    match store.get(obj_id) {
        Some(ValueCell::Object(omap)) => {
            omap.find_in_bucket(h, |k_id| load_value(k_id, store, heap) == key_material)
        }
        _ => None,
    }
}

/// Legacy convenience: absent or unhashable key → [`NULL_VALUE_ID`]. Prefer
/// [`object_map_try_lookup_by_key_id`] when `null` values must stay distinct from missing keys.
pub fn object_map_lookup_by_key_id(
    map: &ObjectMap,
    key_id: ValueId,
    store: &mut ValueStore,
    heap: &HeavyStore,
) -> ValueId {
    object_map_try_lookup_by_key_id(map, key_id, store, heap).unwrap_or(NULL_VALUE_ID)
}

/// When assigning `dict[key] = rhs`, reuse the existing value [`ValueId`] if the key is already
/// present, overwriting that cell with the new payload. This keeps ids stored in
/// [`ValueCell::ObjectFieldList`] (`.values` views) pointing at live cells.
pub fn value_id_for_object_field_update(
    omap: &ObjectMap,
    key_id: ValueId,
    new_value_tv: TaggedValue,
    store: &mut ValueStore,
    heap: &HeavyStore,
) -> ValueId {
    let new_vid = tagged_to_value_id(new_value_tv, store);
    match object_map_try_lookup_by_key_id(omap, key_id, store, heap) {
        None => new_vid,
        Some(existing) => {
            if existing == new_vid {
                return existing;
            }
            // The shared null sentinel (id 0) must never be overwritten in place; remap the slot to new_vid.
            if existing == NULL_VALUE_ID {
                return new_vid;
            }
            if let Some(new_cell) = store.get(new_vid).cloned() {
                if let Some(dest) = store.get_mut(existing) {
                    *dest = new_cell;
                    return existing;
                }
            }
            // Cannot update the existing cell in place (e.g. unmapped id): remap the key to new_vid.
            new_vid
        }
    }
}

/// Store a Value into ValueStore and HeavyStore; returns its ValueId.
/// Recursive for Array and Object; heavy variants go to HeavyStore.
fn alloc_store_cell(store: &mut ValueStore, cell: ValueCell) -> ValueId {
    if store.in_ephemeral_scope() {
        store.allocate_ephemeral(cell)
    } else {
        store.allocate(cell)
    }
}

pub fn store_value(v: Value, store: &mut ValueStore, heap: &mut HeavyStore) -> ValueId {
    match v {
        Value::Null => NULL_VALUE_ID,
        Value::Int(i) => alloc_store_cell(store, ValueCell::Int(i)),
        Value::Float(fv) => alloc_store_cell(store, ValueCell::Float(fv)),
        Value::Number(n) => {
            if store.in_ephemeral_scope() {
                alloc_store_cell(store, ValueCell::Number(n))
            } else {
                store.intern_number_f64(n)
            }
        }
        Value::Bool(b) => alloc_store_cell(store, ValueCell::Bool(b)),
        // Native→VM: s is moved into pool; intern_string dedup avoids duplicate storage.
        Value::String(s) => {
            let sid = store.intern_string(s);
            store.allocate(ValueCell::String(sid))
        }
        Value::Array(rc) => {
            let b = rc.borrow();
            let cap = b.capacity().max(b.len());
            let mut slots = Vec::with_capacity(cap);
            for x in b.iter() {
                slots.push(value_to_slot(x, store, heap));
            }
            store.allocate(ValueCell::Array(slots))
        }
        Value::ArrayView(av) => match av.source {
            ArrayViewSource::Store { base_id } => store.allocate(ValueCell::ArrayView {
                base_id,
                offset: av.offset,
                length: av.length,
            }),
            ArrayViewSource::Heap(_) => {
                let idx = heap.push(Value::ArrayView(av));
                store.allocate(ValueCell::Heavy(idx))
            }
        },
        Value::Tuple(rc) => {
            let arr: Vec<ValueId> = rc
                .borrow()
                .iter()
                .map(|x| store_value(x.clone(), store, heap))
                .collect();
            alloc_store_cell(store, ValueCell::Tuple(arr))
        }
        Value::Function(i) => store.allocate(ValueCell::Function(i)),
        Value::ModuleFunction {
            module_uid,
            local_index,
        } => store.allocate(ValueCell::ModuleFunction {
            module_uid,
            local_index,
        }),
        Value::NativeFunction(i) => store.allocate(ValueCell::NativeFunction(i)),
        Value::Path(p) => store.allocate(ValueCell::Path(p)),
        Value::Uuid(hi, lo) => store.allocate(ValueCell::Uuid(hi, lo)),
        Value::Date(d) => store.allocate(ValueCell::Date {
            secs: d.timestamp(),
            nanos: d.timestamp_subsec_nanos(),
            offset_secs: d.offset().local_minus_utc(),
        }),
        Value::Duration(d) => store.allocate(ValueCell::Duration {
            secs: d.num_seconds(),
            nanos: d.subsec_nanos() as u32,
        }),
        Value::Table(rc) => {
            let idx = heap.push(Value::Table(rc));
            store.allocate(ValueCell::Heavy(idx))
        }
        Value::Set(rc) => {
            let smap = rc.borrow().clone();
            let plain = smap.is_plain();
            let id = alloc_store_cell(store, ValueCell::Set(smap));
            if plain {
                store.mark_plain_set(id);
            }
            id
        }
        Value::Object(rc) => {
            let snap = rc.borrow().clone();
            match snap {
                ObjectKind::Legacy(snap) => {
                    if snap
                        .get("__meta")
                        .and_then(|v| {
                            if let Value::Bool(b) = v {
                                Some(*b)
                            } else {
                                None
                            }
                        })
                        .unwrap_or(false)
                        || snap
                            .get("__create_all")
                            .and_then(|v| {
                                if let Value::Bool(b) = v {
                                    Some(*b)
                                } else {
                                    None
                                }
                            })
                            .unwrap_or(false)
                        || snap.contains_key("__class_name")
                    {
                        let idx = heap.push(Value::Object(rc.clone()));
                        return store.allocate(ValueCell::Heavy(idx));
                    }
                    let mut omap = ObjectMap::with_capacity(snap.len());
                    for (ks, val) in snap {
                        let key_material = Value::String(ks.clone());
                        let kid = store_value(key_material.clone(), store, heap);
                        let vid = store_value(val, store, heap);
                        object_map_upsert(&mut omap, store, heap, &key_material, kid, vid);
                    }
                    alloc_store_cell(store, ValueCell::Object(omap))
                }
                ObjectKind::Bucket(omap) => alloc_store_cell(store, ValueCell::Object(omap)),
                ObjectKind::Inline(pairs) => {
                    let mut omap = ObjectMap::with_capacity(pairs.len());
                    for (k, val) in pairs {
                        let kid = store_value(k.clone(), store, heap);
                        let vid = store_value(val, store, heap);
                        object_map_upsert(&mut omap, store, heap, &k, kid, vid);
                    }
                    alloc_store_cell(store, ValueCell::Object(omap))
                }
            }
        }
        Value::ColumnReference { table, column_name } => {
            let table_val = Value::Table(table);
            let idx = heap.push(table_val);
            store.allocate(ValueCell::ColumnReference {
                table_handle: idx,
                column_name,
            })
        }
        Value::ColumnsReference {
            table,
            column_names,
        } => {
            let table_val = Value::Table(table);
            let idx = heap.push(table_val);
            store.allocate(ValueCell::ColumnsReference {
                table_handle: idx,
                column_names,
            })
        }
        Value::PluginOpaque { tag, id } => {
            if tag == crate::grid::GRID_I32_TAG || tag == crate::grid::GRID_U8_TAG {
                return id as ValueId;
            }
            store.allocate(ValueCell::PluginOpaque { tag, id })
        }
        Value::Window(h) => store.allocate(ValueCell::Window(h)),
        Value::Image(rc) => {
            let idx = heap.push(Value::Image(rc));
            store.allocate(ValueCell::Heavy(idx))
        }
        Value::Figure(rc) => {
            let idx = heap.push(Value::Figure(rc));
            store.allocate(ValueCell::Heavy(idx))
        }
        Value::Axis(rc) => {
            let idx = heap.push(Value::Axis(rc));
            store.allocate(ValueCell::Heavy(idx))
        }
        Value::DatabaseEngine(rc) => {
            let idx = heap.push(Value::DatabaseEngine(rc));
            store.allocate(ValueCell::Heavy(idx))
        }
        Value::DatabaseCluster(rc) => {
            let idx = heap.push(Value::DatabaseCluster(rc));
            store.allocate(ValueCell::Heavy(idx))
        }
        Value::Archive(rc) => {
            let idx = heap.push(Value::Archive(rc));
            store.allocate(ValueCell::Heavy(idx))
        }
        Value::DataSource(rc) => {
            let idx = heap.push(Value::DataSource(rc));
            store.allocate(ValueCell::Heavy(idx))
        }
        Value::DataSourceResponse(rc) => {
            let idx = heap.push(Value::DataSourceResponse(rc));
            store.allocate(ValueCell::Heavy(idx))
        }
        Value::HttpResponse(rc) => {
            let idx = heap.push(Value::HttpResponse(rc));
            store.allocate(ValueCell::Heavy(idx))
        }
        Value::WebPage(rc) => {
            let idx = heap.push(Value::WebPage(rc));
            store.allocate(ValueCell::Heavy(idx))
        }
        Value::WebElement(rc) => {
            let idx = heap.push(Value::WebElement(rc));
            store.allocate(ValueCell::Heavy(idx))
        }
        Value::Enumerate { data, start } => {
            let data_id = store_value(Value::Array(data), store, heap);
            store.allocate(ValueCell::Enumerate { data_id, start })
        }
        Value::Iterable(rc) => {
            let idx = heap.push(Value::Iterable(rc.clone()));
            store.allocate(ValueCell::Heavy(idx))
        }
        Value::Generator(rc) => {
            let idx = heap.push(Value::Generator(rc.clone()));
            store.allocate(ValueCell::Heavy(idx))
        }
        Value::ByteBuffer(b) => {
            let idx = heap.push(Value::ByteBuffer(b));
            store.allocate(ValueCell::Heavy(idx))
        }
        Value::ObjectFieldList {
            source_object_id,
            projection,
            element_ids,
        } => store.allocate(ValueCell::ObjectFieldList {
            source_object_id,
            projection,
            element_ids: element_ids.as_ref().clone(),
        }),
        Value::Ellipsis => store.allocate(ValueCell::Ellipsis),
    }
}

/// After a native call that may mutate Array/Object args, write back the Value to the store cell at id (in place).
pub fn update_cell_if_mutable(
    id: ValueId,
    value: &Value,
    store: &mut ValueStore,
    heap: &mut HeavyStore,
) {
    if id == NULL_VALUE_ID {
        return;
    }
    match value {
        Value::Array(rc) => {
            let slots: Vec<TaggedValue> = rc
                .borrow()
                .iter()
                .map(|x| value_to_slot(x, store, heap))
                .collect();
            if let Some(ValueCell::Array(s)) = store.get_mut(id) {
                *s = slots;
            }
        }
        Value::Object(rc) => {
            let snap = rc.borrow().clone();
            match snap {
                ObjectKind::Legacy(map_ref) => {
                    if map_ref
                        .get("__meta")
                        .and_then(|v| {
                            if let Value::Bool(b) = v {
                                Some(*b)
                            } else {
                                None
                            }
                        })
                        .unwrap_or(false)
                        || map_ref
                            .get("__create_all")
                            .and_then(|v| {
                                if let Value::Bool(b) = v {
                                    Some(*b)
                                } else {
                                    None
                                }
                            })
                            .unwrap_or(false)
                        || map_ref.contains_key("__class_name")
                    {
                        let idx = heap.push(Value::Object(rc.clone()));
                        if let Some(ValueCell::Heavy(h)) = store.get_mut(id) {
                            *h = idx;
                        }
                    } else {
                        let mut omap = ObjectMap::with_capacity(map_ref.len());
                        for (ks, val) in map_ref.into_iter() {
                            let key_material = Value::String(ks.clone());
                            let kid = store_value(key_material.clone(), store, heap);
                            let vid = store_value(val, store, heap);
                            object_map_upsert(&mut omap, store, heap, &key_material, kid, vid);
                        }
                        if let Some(ValueCell::Object(m)) = store.get_mut(id) {
                            *m = omap;
                        }
                    }
                }
                ObjectKind::Inline(pairs) => {
                    let mut omap = ObjectMap::with_capacity(pairs.len());
                    for (k, val) in pairs.into_iter() {
                        let kid = store_value(k.clone(), store, heap);
                        let vid = store_value(val, store, heap);
                        object_map_upsert(&mut omap, store, heap, &k, kid, vid);
                    }
                    if let Some(ValueCell::Object(m)) = store.get_mut(id) {
                        *m = omap;
                    }
                }
                ObjectKind::Bucket(omap) => {
                    if let Some(ValueCell::Object(m)) = store.get_mut(id) {
                        *m = omap.clone();
                    }
                }
            }
        }
        Value::Set(rc) => {
            if let Some(ValueCell::Set(m)) = store.get_mut(id) {
                *m = rc.borrow().clone();
            }
        }
        Value::Table(rc) => {
            if let Some(ValueCell::Heavy(h)) = store.get_mut(id) {
                if let Some(slot) = heap.get_mut(*h) {
                    *slot = Value::Table(Rc::clone(rc));
                } else {
                    *h = heap.push(Value::Table(Rc::clone(rc)));
                }
            }
        }
        _ => {}
    }
}

/// Load a Value from ValueStore and HeavyStore by ValueId.
pub fn load_value(id: ValueId, store: &ValueStore, heap: &HeavyStore) -> Value {
    load_value_inner(id, store, heap, &mut HashSet::new())
}

fn load_value_cycle_placeholder() -> Value {
    Value::Object(Rc::new(RefCell::new(ObjectKind::Legacy(HashMap::new()))))
}

fn load_value_inner(
    id: ValueId,
    store: &ValueStore,
    heap: &HeavyStore,
    visiting: &mut HashSet<ValueId>,
) -> Value {
    if id == NULL_VALUE_ID {
        return Value::Null;
    }
    let cell = match store.get(id) {
        Some(c) => c,
        None => return Value::Null,
    };
    match cell {
        ValueCell::Int(i) => Value::Int(*i),
        ValueCell::Float(f) => Value::Float(*f),
        ValueCell::Number(n) => Value::Number(*n),
        ValueCell::Bool(b) => Value::Bool(*b),
        ValueCell::Null => Value::Null,
        // Value::String is owned; one .to_string() at VM→native boundary is required (no extra clone).
        ValueCell::String(sid) => Value::String(store.get_string(*sid).unwrap_or("").to_string()),
        ValueCell::Array(slots) => {
            let arr: Vec<Value> = slots
                .iter()
                .map(|slot| slot_to_value_inner(*slot, store, heap, visiting))
                .collect();
            Value::Array(Rc::new(RefCell::new(arr)))
        }
        ValueCell::ArrayView {
            base_id,
            offset,
            length,
        } => Value::ArrayView(ArrayViewData {
            source: ArrayViewSource::Store { base_id: *base_id },
            offset: *offset,
            length: *length,
        }),
        ValueCell::Tuple(ids) => {
            let arr: Vec<Value> = ids
                .iter()
                .map(|&i| load_value_inner(i, store, heap, visiting))
                .collect();
            Value::Tuple(Rc::new(RefCell::new(arr)))
        }
        ValueCell::Object(omap) => {
            if !visiting.insert(id) {
                return load_value_cycle_placeholder();
            }
            let pairs: Vec<(Value, Value)> = omap
                .iter_entries()
                .map(|(_, kid, vid)| {
                    (
                        load_value_inner(kid, store, heap, visiting),
                        load_value_inner(vid, store, heap, visiting),
                    )
                })
                .collect();
            visiting.remove(&id);
            Value::Object(Rc::new(RefCell::new(ObjectKind::Inline(pairs))))
        }
        ValueCell::Function(i) => Value::Function(*i),
        ValueCell::ModuleFunction {
            module_uid,
            local_index,
        } => Value::ModuleFunction {
            module_uid: *module_uid,
            local_index: *local_index,
        },
        ValueCell::NativeFunction(i) => Value::NativeFunction(*i),
        ValueCell::Path(p) => Value::Path(p.clone()),
        ValueCell::Uuid(hi, lo) => Value::Uuid(*hi, *lo),
        ValueCell::Date {
            secs,
            nanos,
            offset_secs,
        } => {
            use chrono::FixedOffset;
            let offset = FixedOffset::east_opt(*offset_secs)
                .unwrap_or_else(|| FixedOffset::east_opt(0).expect("offset 0"));
            DateTime::from_timestamp(*secs, *nanos)
                .map(|utc| Value::Date(utc.with_timezone(&offset)))
                .unwrap_or(Value::Null)
        }
        ValueCell::Duration { secs, nanos } => {
            use chrono::Duration as ChronoDuration;
            Value::Duration(
                ChronoDuration::new(*secs, *nanos).unwrap_or_else(ChronoDuration::zero),
            )
        }
        ValueCell::Heavy(idx) => heap.get(*idx).cloned().unwrap_or(Value::Null),
        ValueCell::ColumnReference {
            table_handle,
            column_name,
        } => {
            let table_val = heap.get(*table_handle).cloned().unwrap_or(Value::Null);
            if let Value::Table(rc) = table_val {
                Value::ColumnReference {
                    table: rc,
                    column_name: column_name.clone(),
                }
            } else {
                Value::Null
            }
        }
        ValueCell::ColumnsReference {
            table_handle,
            column_names,
        } => {
            let table_val = heap.get(*table_handle).cloned().unwrap_or(Value::Null);
            if let Value::Table(rc) = table_val {
                Value::ColumnsReference {
                    table: rc,
                    column_names: column_names.clone(),
                }
            } else {
                Value::Null
            }
        }
        ValueCell::PluginOpaque { tag, id } => Value::PluginOpaque { tag: *tag, id: *id },
        ValueCell::Window(h) => Value::Window(*h),
        ValueCell::Enumerate { data_id, start } => {
            let data_val = load_value_inner(*data_id, store, heap, visiting);
            if let Value::Array(rc) = data_val {
                Value::Enumerate {
                    data: rc,
                    start: *start,
                }
            } else {
                Value::Null
            }
        }
        ValueCell::ObjectFieldList {
            source_object_id,
            projection,
            element_ids,
        } => Value::ObjectFieldList {
            source_object_id: *source_object_id,
            projection: *projection,
            element_ids: Rc::new(element_ids.clone()),
        },
        ValueCell::Set(smap) => Value::Set(Rc::new(RefCell::new(smap.clone()))),
        ValueCell::GridBufferI32(_) => Value::PluginOpaque {
            tag: crate::grid::GRID_I32_TAG,
            id: id as u64,
        },
        ValueCell::GridBufferU8(_) => Value::PluginOpaque {
            tag: crate::grid::GRID_U8_TAG,
            id: id as u64,
        },
        ValueCell::GridHeapU32(_) => Value::PluginOpaque {
            tag: crate::grid::GRID_HEAP_TAG,
            id: id as u64,
        },
        ValueCell::Ellipsis => Value::Ellipsis,
    }
}

/// If the cell is an immediate (Number, Bool, Null), return its TaggedValue; else None (use heap id).
pub fn value_cell_to_tagged(cell: &ValueCell) -> Option<TaggedValue> {
    match cell {
        ValueCell::Int(i) => TaggedValue::try_from_int_iv(*i),
        ValueCell::Float(f) => Some(TaggedValue::from_float_value(*f)),
        ValueCell::Number(n) => Some(TaggedValue::from_f64(*n)),
        ValueCell::Bool(b) => Some(TaggedValue::from_bool(*b)),
        ValueCell::Null => Some(TaggedValue::null()),
        _ => None,
    }
}

/// Convert Value to TaggedValue for array slots. Inline for number/bool/null; heap id for rest.
fn value_to_slot(v: &Value, store: &mut ValueStore, heap: &mut HeavyStore) -> TaggedValue {
    match v {
        Value::Int(iv) => match TaggedValue::try_from_int_iv(*iv) {
            Some(tv) => tv,
            None => TaggedValue::from_heap(store_value(Value::Int(*iv), store, heap)),
        },
        Value::Float(fv) => TaggedValue::from_float_value(*fv),
        Value::Number(n) => TaggedValue::from_f64(*n),
        Value::Bool(b) => TaggedValue::from_bool(*b),
        Value::Null => TaggedValue::null(),
        _ => TaggedValue::from_heap(store_value(v.clone(), store, heap)),
    }
}

/// Convert array slot (TaggedValue) to Value for load_value. No store access for inline.
pub fn slot_to_value(slot: TaggedValue, store: &ValueStore, heap: &HeavyStore) -> Value {
    slot_to_value_inner(slot, store, heap, &mut HashSet::new())
}

fn slot_to_value_inner(
    slot: TaggedValue,
    store: &ValueStore,
    heap: &HeavyStore,
    visiting: &mut HashSet<ValueId>,
) -> Value {
    if slot.is_number() {
        Value::Number(slot.get_f64())
    } else if slot.is_bool() {
        Value::Bool(slot.get_bool())
    } else if slot.is_null() {
        Value::Null
    } else if let Some(iv) = slot.int_value_domain() {
        Value::Int(iv)
    } else if slot.is_heap() {
        load_value_inner(slot.get_heap_id(), store, heap, visiting)
    } else {
        Value::Null
    }
}

/// Store into a mutable container (dict entry, etc.): never reuse interned scalar cells.
pub fn tagged_to_mutable_value_id(tv: TaggedValue, store: &mut ValueStore) -> ValueId {
    if tv.is_number() {
        return store.allocate_ephemeral(ValueCell::Number(tv.get_f64()));
    }
    if tv.is_int() {
        return store.allocate_ephemeral(ValueCell::Int(IntValue::Finite(tv.get_i32() as i64)));
    }
    if tv.is_null() {
        return NULL_VALUE_ID;
    }
    if tv.is_bool() {
        return store.allocate_arena(ValueCell::Bool(tv.get_bool()));
    }
    if tv.is_int_pos_inf() {
        return store.allocate_ephemeral(ValueCell::Int(IntValue::PosInfinity));
    }
    if tv.is_int_neg_inf() {
        return store.allocate_ephemeral(ValueCell::Int(IntValue::NegInfinity));
    }
    if tv.is_heap() {
        let id = tv.get_heap_id();
        if store.is_interned_scalar(id) {
            return store.copy_scalar_ephemeral(id);
        }
        return id;
    }
    NULL_VALUE_ID
}

/// When storing into a container: inline heap scalars; promote call-arena values only when the
/// container lives outside the call arena (escape / write barrier).
pub fn prepare_value_for_container_store(
    container_id: ValueId,
    value_tv: TaggedValue,
    store: &mut ValueStore,
) -> TaggedValue {
    if value_tv.is_heap() {
        let id = value_tv.get_heap_id();
        if let Some(inline) = store.get(id).and_then(value_cell_to_tagged) {
            return inline;
        }
        if id >= CALL_ARENA_BASE && container_id < CALL_ARENA_BASE {
            let promoted = store.promote_from_call_arena(id);
            return TaggedValue::from_heap(promoted);
        }
    }
    value_tv
}

/// Promote a call-arena [`ValueId`] only when assigning into a non-ephemeral container.
pub fn promote_value_id_if_escapes(
    container_id: ValueId,
    value_id: ValueId,
    store: &mut ValueStore,
) -> ValueId {
    if value_id >= CALL_ARENA_BASE && container_id < CALL_ARENA_BASE {
        store.promote_from_call_arena(value_id)
    } else {
        value_id
    }
}

/// Convert a TaggedValue to ValueId. Allocates a cell for immediates (number, bool, null, int); returns existing id for heap.
pub fn tagged_to_value_id(tv: TaggedValue, store: &mut ValueStore) -> ValueId {
    if tv.is_number() {
        if store.in_ephemeral_scope() {
            return store.allocate_ephemeral(ValueCell::Number(tv.get_f64()));
        }
        return store.intern_number_f64(tv.get_f64());
    } else if tv.is_null() {
        NULL_VALUE_ID
    } else if tv.is_bool() {
        store.allocate_ephemeral(ValueCell::Bool(tv.get_bool()))
    } else if tv.is_int() {
        if store.in_ephemeral_scope() {
            return store.allocate_ephemeral(ValueCell::Int(IntValue::Finite(tv.get_i32() as i64)));
        }
        store.intern_whole_i64(tv.get_i32() as i64)
    } else if tv.is_int_pos_inf() {
        store.allocate_ephemeral(ValueCell::Int(crate::common::numeric::IntValue::PosInfinity))
    } else if tv.is_int_neg_inf() {
        store.allocate_ephemeral(ValueCell::Int(crate::common::numeric::IntValue::NegInfinity))
    } else if tv.is_heap() {
        // NOTE: do NOT promote here — most reads/temps inside a user call should stay ephemeral.
        // Escape/promotion is handled at write barriers (StoreGlobal / object field set / etc.)
        tv.get_heap_id()
    } else {
        NULL_VALUE_ID
    }
}

/// Like tagged_to_value_id but allocates in the heap arena (for globals/slots). Use for StoreGlobal and resolve_to_value_id.
pub fn tagged_to_value_id_arena(tv: TaggedValue, store: &mut ValueStore) -> ValueId {
    if tv.is_number() {
        store.allocate_arena(ValueCell::Number(tv.get_f64()))
    } else if tv.is_null() {
        NULL_VALUE_ID
    } else if tv.is_bool() {
        store.allocate_arena(ValueCell::Bool(tv.get_bool()))
    } else if tv.is_int() {
        store.allocate_arena(ValueCell::Int(crate::common::numeric::IntValue::Finite(
            tv.get_i32() as i64,
        )))
    } else if tv.is_int_pos_inf() {
        store.allocate_arena(ValueCell::Int(crate::common::numeric::IntValue::PosInfinity))
    } else if tv.is_int_neg_inf() {
        store.allocate_arena(ValueCell::Int(crate::common::numeric::IntValue::NegInfinity))
    } else if tv.is_heap() {
        // Globals/arena slots must never reference ephemeral ids.
        store.promote_from_call_arena(tv.get_heap_id())
    } else {
        NULL_VALUE_ID
    }
}

/// Store a Value into the heap arena (for globals / ephemeral heap). Same semantics as store_value but allocate_arena.
pub fn store_value_arena(v: Value, store: &mut ValueStore, heap: &mut HeavyStore) -> ValueId {
    match v {
        Value::Null => NULL_VALUE_ID,
        Value::Int(i) => store.allocate_arena(ValueCell::Int(i)),
        Value::Float(fv) => store.allocate_arena(ValueCell::Float(fv)),
        Value::Number(n) => store.allocate_arena(ValueCell::Number(n)),
        Value::Bool(b) => store.allocate_arena(ValueCell::Bool(b)),
        Value::String(s) => {
            let sid = store.intern_string(s);
            store.allocate_arena(ValueCell::String(sid))
        }
        Value::Array(rc) => {
            let b = rc.borrow();
            let cap = b.capacity().max(b.len());
            let mut slots = Vec::with_capacity(cap);
            for x in b.iter() {
                slots.push(value_to_slot_arena(x, store, heap));
            }
            store.allocate_arena(ValueCell::Array(slots))
        }
        Value::Tuple(rc) => {
            let arr: Vec<ValueId> = rc
                .borrow()
                .iter()
                .map(|x| store_value_arena(x.clone(), store, heap))
                .collect();
            store.allocate_arena(ValueCell::Tuple(arr))
        }
        Value::Function(i) => store.allocate_arena(ValueCell::Function(i)),
        Value::ModuleFunction {
            module_uid,
            local_index,
        } => store.allocate_arena(ValueCell::ModuleFunction {
            module_uid,
            local_index,
        }),
        Value::NativeFunction(i) => store.allocate_arena(ValueCell::NativeFunction(i)),
        Value::Path(p) => store.allocate_arena(ValueCell::Path(p)),
        Value::Uuid(hi, lo) => store.allocate_arena(ValueCell::Uuid(hi, lo)),
        Value::Date(d) => store.allocate_arena(ValueCell::Date {
            secs: d.timestamp(),
            nanos: d.timestamp_subsec_nanos(),
            offset_secs: d.offset().local_minus_utc(),
        }),
        Value::Duration(d) => store.allocate_arena(ValueCell::Duration {
            secs: d.num_seconds(),
            nanos: d.subsec_nanos() as u32,
        }),
        Value::Table(rc) => {
            let idx = heap.push(Value::Table(rc));
            store.allocate_arena(ValueCell::Heavy(idx))
        }
        Value::Set(rc) => {
            let smap = rc.borrow().clone();
            let plain = smap.is_plain();
            let id = store.allocate_arena(ValueCell::Set(smap));
            if plain {
                store.mark_plain_set(id);
            }
            id
        }
        Value::Object(rc) => {
            let snap = rc.borrow().clone();
            match snap {
                ObjectKind::Legacy(map) => {
                    if map
                        .get("__meta")
                        .and_then(|v| {
                            if let Value::Bool(b) = v {
                                Some(*b)
                            } else {
                                None
                            }
                        })
                        .unwrap_or(false)
                        || map
                            .get("__create_all")
                            .and_then(|v| {
                                if let Value::Bool(b) = v {
                                    Some(*b)
                                } else {
                                    None
                                }
                            })
                            .unwrap_or(false)
                    {
                        let idx = heap.push(Value::Object(rc.clone()));
                        return store.allocate_arena(ValueCell::Heavy(idx));
                    }
                    if map.contains_key("__class_name") {
                        let idx = heap.push(Value::Object(rc.clone()));
                        return store.allocate_arena(ValueCell::Heavy(idx));
                    }
                    let mut omap = ObjectMap::with_capacity(map.len());
                    for (ks, val) in map.into_iter() {
                        let key_material = Value::String(ks.clone());
                        let kid = store_value_arena(key_material.clone(), store, heap);
                        let vid = store_value_arena(val, store, heap);
                        object_map_upsert(&mut omap, store, heap, &key_material, kid, vid);
                    }
                    store.allocate_arena(ValueCell::Object(omap))
                }
                ObjectKind::Bucket(omap) => store.allocate_arena(ValueCell::Object(omap)),
                ObjectKind::Inline(pairs) => {
                    let mut omap = ObjectMap::with_capacity(pairs.len());
                    for (k, val) in pairs.into_iter() {
                        let kid = store_value_arena(k.clone(), store, heap);
                        let vid = store_value_arena(val, store, heap);
                        object_map_upsert(&mut omap, store, heap, &k, kid, vid);
                    }
                    store.allocate_arena(ValueCell::Object(omap))
                }
            }
        }
        Value::ColumnReference { table, column_name } => {
            let table_val = Value::Table(table);
            let idx = heap.push(table_val);
            store.allocate_arena(ValueCell::ColumnReference {
                table_handle: idx,
                column_name,
            })
        }
        Value::ColumnsReference {
            table,
            column_names,
        } => {
            let table_val = Value::Table(table);
            let idx = heap.push(table_val);
            store.allocate_arena(ValueCell::ColumnsReference {
                table_handle: idx,
                column_names,
            })
        }
        Value::PluginOpaque { tag, id } => {
            store.allocate_arena(ValueCell::PluginOpaque { tag, id })
        }
        Value::Window(h) => store.allocate_arena(ValueCell::Window(h)),
        Value::Image(rc) => {
            let idx = heap.push(Value::Image(rc));
            store.allocate_arena(ValueCell::Heavy(idx))
        }
        Value::Figure(rc) => {
            let idx = heap.push(Value::Figure(rc));
            store.allocate_arena(ValueCell::Heavy(idx))
        }
        Value::Axis(rc) => {
            let idx = heap.push(Value::Axis(rc));
            store.allocate_arena(ValueCell::Heavy(idx))
        }
        Value::DatabaseEngine(rc) => {
            let idx = heap.push(Value::DatabaseEngine(rc));
            store.allocate_arena(ValueCell::Heavy(idx))
        }
        Value::DatabaseCluster(rc) => {
            let idx = heap.push(Value::DatabaseCluster(rc));
            store.allocate_arena(ValueCell::Heavy(idx))
        }
        Value::Archive(rc) => {
            let idx = heap.push(Value::Archive(rc));
            store.allocate_arena(ValueCell::Heavy(idx))
        }
        Value::DataSource(rc) => {
            let idx = heap.push(Value::DataSource(rc));
            store.allocate_arena(ValueCell::Heavy(idx))
        }
        Value::DataSourceResponse(rc) => {
            let idx = heap.push(Value::DataSourceResponse(rc));
            store.allocate_arena(ValueCell::Heavy(idx))
        }
        Value::HttpResponse(rc) => {
            let idx = heap.push(Value::HttpResponse(rc));
            store.allocate_arena(ValueCell::Heavy(idx))
        }
        Value::WebPage(rc) => {
            let idx = heap.push(Value::WebPage(rc));
            store.allocate_arena(ValueCell::Heavy(idx))
        }
        Value::WebElement(rc) => {
            let idx = heap.push(Value::WebElement(rc));
            store.allocate_arena(ValueCell::Heavy(idx))
        }
        Value::Enumerate { data, start } => {
            let data_id = store_value_arena(Value::Array(data), store, heap);
            store.allocate_arena(ValueCell::Enumerate { data_id, start })
        }
        Value::Iterable(rc) => {
            let idx = heap.push(Value::Iterable(rc.clone()));
            store.allocate_arena(ValueCell::Heavy(idx))
        }
        Value::Generator(rc) => {
            let idx = heap.push(Value::Generator(rc.clone()));
            store.allocate_arena(ValueCell::Heavy(idx))
        }
        Value::ByteBuffer(b) => {
            let idx = heap.push(Value::ByteBuffer(b));
            store.allocate_arena(ValueCell::Heavy(idx))
        }
        Value::ObjectFieldList {
            source_object_id,
            projection,
            element_ids,
        } => store.allocate_arena(ValueCell::ObjectFieldList {
            source_object_id,
            projection,
            element_ids: element_ids.as_ref().clone(),
        }),
        Value::ArrayView(av) => match av.source {
            ArrayViewSource::Store { base_id } => store.allocate_arena(ValueCell::ArrayView {
                base_id,
                offset: av.offset,
                length: av.length,
            }),
            ArrayViewSource::Heap(_) => {
                let idx = heap.push(Value::ArrayView(av));
                store.allocate_arena(ValueCell::Heavy(idx))
            }
        },
        Value::Ellipsis => store.allocate_arena(ValueCell::Ellipsis),
    }
}

fn value_to_slot_arena(v: &Value, store: &mut ValueStore, heap: &mut HeavyStore) -> TaggedValue {
    match v {
        Value::Int(iv) => match TaggedValue::try_from_int_iv(*iv) {
            Some(tv) => tv,
            None => TaggedValue::from_heap(store_value_arena(Value::Int(*iv), store, heap)),
        },
        Value::Float(fv) => TaggedValue::from_float_value(*fv),
        Value::Number(n) => TaggedValue::from_f64(*n),
        Value::Bool(b) => TaggedValue::from_bool(*b),
        Value::Null => TaggedValue::null(),
        _ => TaggedValue::from_heap(store_value_arena(v.clone(), store, heap)),
    }
}
