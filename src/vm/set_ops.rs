//! Set runtime helpers (membership, equality, algebra).

use crate::common::numeric::{hash_integral_key, integer_value_as_i64_if_whole};
use std::collections::HashSet;
use crate::common::set_map::SetMap;
use crate::common::type_model::{is_hashable_value, object_key_hash_value};
use crate::common::value::Value;
use crate::common::value_store::{ValueId, ValueStore};
use crate::vm::heavy_store::HeavyStore;
use crate::vm::store_convert::{load_value, store_value};

/// All member key ids: bucket chains plus integral-only entries when the side table is in use.
pub fn set_member_key_ids(map: &SetMap, store: &mut ValueStore) -> Vec<ValueId> {
    let inner = map.inner_ref();
    if inner.integral_in_use() && inner.iter_entries().next().is_none() {
        return inner
            .iter_integral_canonicals()
            .map(|c| store.intern_whole_i64(c))
            .collect();
    }
    let mut ids: Vec<ValueId> = map.iter_key_ids().collect();
    if !inner.integral_in_use() {
        return ids;
    }
    let mut present: HashSet<ValueId> = ids.iter().copied().collect();
    for c in inner.iter_integral_canonicals() {
        let kid = store.intern_whole_i64(c);
        let h = hash_integral_key(c);
        if inner.find_in_bucket(h, |id| id == kid).is_none() && present.insert(kid) {
            ids.push(kid);
        }
    }
    ids
}

pub fn sets_equal(
    a: &SetMap,
    b: &SetMap,
    store: &mut ValueStore,
    heap: &HeavyStore,
) -> bool {
    if a.len() != b.len() {
        return false;
    }
    for kid in set_member_key_ids(a, store) {
        let kv = load_value(kid, store, heap);
        if !set_contains_value(b, &kv, store, heap) {
            return false;
        }
    }
    true
}

pub fn set_contains_value(
    map: &SetMap,
    elem: &Value,
    store: &ValueStore,
    heap: &HeavyStore,
) -> bool {
    if let Some(canonical) = integer_value_as_i64_if_whole(elem) {
        if map.contains_integral(canonical) {
            return true;
        }
    }
    let Some(h) = crate::vm::special_methods::try_instance_hash(elem)
        .or_else(|| object_key_hash_value(elem))
    else {
        return false;
    };
    map.contains(h, |id| load_value(id, store, heap) == *elem)
}

pub fn set_insert_material(
    map: &mut SetMap,
    elem: &Value,
    store: &mut ValueStore,
    heap: &mut HeavyStore,
) -> Result<bool, String> {
    if !is_hashable_value(elem) && !crate::vm::special_methods::class_instance_supports_hash(elem) {
        return Err(format!(
            "unhashable type: {}",
            crate::vm::calls::get_type_name_value(elem)
        ));
    }
    // Whole-number keys: integral side table only (same as fast `set.add`); buckets + heap cells
    // are reserved for non-integral members (tuples, strings, …). Iteration/str/copy use
    // [`set_member_key_ids`].
    if let Some(canonical) = integer_value_as_i64_if_whole(elem) {
        let kid = store.intern_whole_i64(canonical);
        return Ok(map.insert_integral_only(canonical, kid));
    }
    let h = crate::vm::special_methods::try_instance_hash(elem)
        .or_else(|| object_key_hash_value(elem))
        .expect("hashable");
    let kid = store_value(elem.clone(), store, heap);
    Ok(map.insert(
        h,
        kid,
        |id| load_value(id, store, heap) == *elem,
        None,
    ))
}

pub fn set_remove_material(
    map: &mut SetMap,
    elem: &Value,
    store: &ValueStore,
    heap: &HeavyStore,
) -> bool {
    if let Some(canonical) = integer_value_as_i64_if_whole(elem) {
        if map.contains_integral(canonical) {
            return map.discard_integral(canonical);
        }
    }
    let Some(h) = crate::vm::special_methods::try_instance_hash(elem)
        .or_else(|| object_key_hash_value(elem))
    else {
        return false;
    };
    map.remove(
        h,
        |id| load_value(id, store, heap) == *elem,
        None,
    )
}

/// Like [`set_remove_material`] but named for discard semantics (same implementation).
#[inline]
pub fn set_discard_material(
    map: &mut SetMap,
    elem: &Value,
    store: &ValueStore,
    heap: &HeavyStore,
) -> bool {
    set_remove_material(map, elem, store, heap)
}

/// Remove and return an arbitrary element id from the set, if non-empty.
pub fn set_pop_key_id(
    map: &mut SetMap,
    store: &mut ValueStore,
) -> Option<ValueId> {
    map.pop_arbitrary(|canonical| store.intern_whole_i64(canonical))
}

pub fn set_copy(map: &SetMap, store: &mut ValueStore, heap: &mut HeavyStore) -> ValueId {
    let mut out = SetMap::with_capacity(map.len());
    for kid in set_member_key_ids(map, store) {
        let v = load_value(kid, store, heap);
        let _ = set_insert_material(&mut out, &v, store, heap);
    }
    store_value(Value::Set(Rc::new(RefCell::new(out))), store, heap)
}

pub fn set_union(
    a: &SetMap,
    b: &SetMap,
    store: &mut ValueStore,
    heap: &mut HeavyStore,
) -> ValueId {
    let mut out = SetMap::with_capacity(a.len() + b.len());
    for kid in set_member_key_ids(a, store)
        .into_iter()
        .chain(set_member_key_ids(b, store))
    {
        let v = load_value(kid, store, heap);
        let _ = set_insert_material(&mut out, &v, store, heap);
    }
    store_value(
        Value::Set(std::rc::Rc::new(std::cell::RefCell::new(out))),
        store,
        heap,
    )
}

pub fn set_intersection(
    a: &SetMap,
    b: &SetMap,
    store: &mut ValueStore,
    heap: &mut HeavyStore,
) -> ValueId {
    let mut out = SetMap::new();
    for kid in set_member_key_ids(a, store) {
        let v = load_value(kid, store, heap);
        if set_contains_value(b, &v, store, heap) {
            let _ = set_insert_material(&mut out, &v, store, heap);
        }
    }
    store_value(
        Value::Set(std::rc::Rc::new(std::cell::RefCell::new(out))),
        store,
        heap,
    )
}

pub fn set_difference(
    a: &SetMap,
    b: &SetMap,
    store: &mut ValueStore,
    heap: &mut HeavyStore,
) -> ValueId {
    let mut out = SetMap::new();
    for kid in set_member_key_ids(a, store) {
        let v = load_value(kid, store, heap);
        if !set_contains_value(b, &v, store, heap) {
            let _ = set_insert_material(&mut out, &v, store, heap);
        }
    }
    store_value(
        Value::Set(std::rc::Rc::new(std::cell::RefCell::new(out))),
        store,
        heap,
    )
}

pub fn set_symmetric_difference(
    a: &SetMap,
    b: &SetMap,
    store: &mut ValueStore,
    heap: &mut HeavyStore,
) -> ValueId {
    let mut out = SetMap::new();
    for kid in set_member_key_ids(a, store) {
        let v = load_value(kid, store, heap);
        if !set_contains_value(b, &v, store, heap) {
            let _ = set_insert_material(&mut out, &v, store, heap);
        }
    }
    for kid in set_member_key_ids(b, store) {
        let v = load_value(kid, store, heap);
        if !set_contains_value(a, &v, store, heap) {
            let _ = set_insert_material(&mut out, &v, store, heap);
        }
    }
    store_value(
        Value::Set(std::rc::Rc::new(std::cell::RefCell::new(out))),
        store,
        heap,
    )
}

pub fn set_is_subset(
    a: &SetMap,
    b: &SetMap,
    store: &mut ValueStore,
    heap: &HeavyStore,
) -> bool {
    for kid in set_member_key_ids(a, store) {
        let v = load_value(kid, store, heap);
        if !set_contains_value(b, &v, store, heap) {
            return false;
        }
    }
    true
}

pub fn format_set_display(
    map: &SetMap,
    store: &mut ValueStore,
    heap: &HeavyStore,
) -> String {
    if map.is_empty() {
        return "set()".to_string();
    }
    let key_ids = set_member_key_ids(map, store);
    let mut parts: Vec<String> = key_ids
        .iter()
        .map(|&id| load_value(id, store, heap).to_string())
        .collect();
    parts.sort();
    format!("set({})", parts.join(", "))
}

use std::cell::RefCell;
use std::rc::Rc;

pub fn set_from_iterable_values(
    values: impl IntoIterator<Item = Value>,
    store: &mut ValueStore,
    heap: &mut HeavyStore,
) -> Result<ValueId, String> {
    let mut map = SetMap::new();
    for v in values {
        set_insert_material(&mut map, &v, store, heap)?;
    }
    Ok(store_value(
        Value::Set(Rc::new(RefCell::new(map))),
        store,
        heap,
    ))
}

pub fn set_cell_from_store(id: ValueId, store: &ValueStore) -> Option<&SetMap> {
    match store.get(id)? {
        crate::common::value_store::ValueCell::Set(m) => Some(m),
        _ => None,
    }
}

pub fn set_cell_from_store_mut(id: ValueId, store: &mut ValueStore) -> Option<&mut SetMap> {
    match store.get_mut(id)? {
        crate::common::value_store::ValueCell::Set(m) => Some(m),
        _ => None,
    }
}

pub fn load_set_map(id: ValueId, store: &ValueStore, heap: &HeavyStore) -> Option<SetMap> {
    match load_value(id, store, heap) {
        Value::Set(rc) => Some(rc.borrow().clone()),
        _ => None,
    }
}

pub fn check_iter_not_modified(
    map: &SetMap,
    start_generation: u64,
) -> Result<(), &'static str> {
    if map.generation() != start_generation {
        Err("set modified during iteration")
    } else {
        Ok(())
    }
}

pub fn snapshot_set_element_ids(map: &SetMap, store: &mut ValueStore) -> Vec<ValueId> {
    set_member_key_ids(map, store)
}

/// `str(set)` — привычное представление как у кортежа: `(1, 2, 3)`; пустое множество → `()`.
/// Элементы сортируются по строковому представлению для стабильного вывода.
pub fn set_to_repr_string(map: &SetMap, store: &mut ValueStore, heap: &HeavyStore) -> String {
    let mut items: Vec<Value> = set_member_key_ids(map, store)
        .into_iter()
        .map(|id| load_value(id, store, heap))
        .collect();
    items.sort_by(|a, b| a.to_string().cmp(&b.to_string()));
    if items.is_empty() {
        return "()".to_string();
    }
    let elements: Vec<String> = items.iter().map(|v| v.to_string()).collect();
    format!("({})", elements.join(", "))
}
