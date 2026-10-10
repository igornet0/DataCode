//! Live bindings of `.dc` module data (`ITEMS = [...]`, `COUNT = 0`) inside one VM.
//!
//! A module's export namespace stores `Value`s, and turning a `Value` into a store id copies arrays.
//! Each data name therefore gets one slot per VM ([`ModuleDataSlots`]): module functions read and
//! write it like a regular global, the importer's `m` namespace object points its fields at the
//! same cells, and `from m import ITEMS` binds the same array. So `m.COUNT` / `m.ITEMS` and a
//! repeated `from m import ...` see the module's current values, and in-place changes (`push`,
//! `arr[i] = v`) are shared, as with Python modules.

use crate::common::value::{ObjectKind, Value};
use crate::common::value_store::{ValueCell, ValueId, ValueStore};
use crate::vm::global_slot::GlobalSlot;
use crate::vm::heavy_store::HeavyStore;
use crate::vm::module_object::{
    is_module_namespace, namespace_key, ModuleDataSlots, MODULE_MARKER_KEY,
};
use crate::vm::store_convert::{store_value, tagged_to_value_id_arena};
use std::cell::RefCell;
use std::rc::Rc;

/// Module-owned data that lives in the module namespace (never in the importer's globals):
/// primitives, strings, arrays, tuples, sets, paths, dates and plain dicts. Functions, classes,
/// class instances and other objects keep the merged host-slot path.
pub(crate) fn is_module_owned_data(value: &Value) -> bool {
    match value {
        Value::Null
        | Value::Int(_)
        | Value::Float(_)
        | Value::Number(_)
        | Value::Bool(_)
        | Value::String(_)
        | Value::Array(_)
        | Value::Tuple(_)
        | Value::Set(_)
        | Value::Path(_)
        | Value::Uuid(_, _)
        | Value::Date(_)
        | Value::Duration(_) => true,
        Value::Object(rc) => is_plain_dict(&rc.borrow()),
        _ => false,
    }
}

/// A dict literal / user dict: no `__`-prefixed string keys (classes, instances, module
/// namespaces and plugin objects all carry such service keys).
fn is_plain_dict(obj: &ObjectKind) -> bool {
    !obj.str_key_pairs().iter().any(|(k, _)| k.starts_with("__"))
}

fn slot_value_id(slot: GlobalSlot, store: &mut ValueStore) -> ValueId {
    match slot {
        GlobalSlot::Inline(tv) => tagged_to_value_id_arena(tv, store),
        GlobalSlot::Heap(id) => id,
    }
}

/// Point field `name` of every namespace object bound to this module at `slot`.
fn publish(
    entry: &ModuleDataSlots,
    name: &str,
    slot: GlobalSlot,
    store: &mut ValueStore,
    heap: &HeavyStore,
) {
    if entry.objects.is_empty() {
        return;
    }
    let vid = slot_value_id(slot, store);
    let key = Value::String(name.to_string());
    for &obj in &entry.objects {
        let kid = store_value(key.clone(), store, &mut HeavyStore::new());
        crate::vm::store_convert::object_map_upsert_in_place(store, obj, heap, &key, kid, vid);
    }
}

/// Live slot of module data `name`. On first access the export is materialized into the main store
/// once; later accesses return the same id. None when `name` is not module data.
pub(crate) fn data_slot(
    namespace: &Rc<RefCell<ObjectKind>>,
    name: &str,
    store: &mut ValueStore,
    heap: &mut HeavyStore,
    vm_ptr: *mut crate::vm::vm::Vm,
) -> Option<GlobalSlot> {
    let key = namespace_key(namespace);
    let mut all = unsafe { (*vm_ptr).get_module_data_slots_mut() };
    if let Some(slot) = all.get(&key).and_then(|m| m.slots.get(name)) {
        return Some(*slot);
    }
    let value = namespace
        .borrow()
        .str_key_get(name)
        .cloned()
        .filter(is_module_owned_data)?;
    let id = store_value(value, store, heap);
    // Inside a call nested values may land in the call arena (freed on return): keep the slot in the main store.
    let slot = GlobalSlot::Heap(store.promote_from_call_arena(id));
    let entry = all
        .entry(key)
        .or_insert_with(|| ModuleDataSlots::new(Rc::clone(namespace)));
    entry.slots.insert(name.to_string(), slot);
    publish(entry, name, slot, store, heap);
    Some(slot)
}

/// Rebind module data `name` (StoreGlobal in a module frame) and update bound namespace objects.
pub(crate) fn set_data_slot(
    namespace: &Rc<RefCell<ObjectKind>>,
    name: &str,
    slot: GlobalSlot,
    store: &mut ValueStore,
    heap: &HeavyStore,
    vm_ptr: *mut crate::vm::vm::Vm,
) {
    let mut all = unsafe { (*vm_ptr).get_module_data_slots_mut() };
    let entry = all
        .entry(namespace_key(namespace))
        .or_insert_with(|| ModuleDataSlots::new(Rc::clone(namespace)));
    entry.slots.insert(name.to_string(), slot);
    publish(entry, name, slot, store, heap);
}

/// `name` was rebound to a function/class/object: it is no longer module data.
pub(crate) fn remove_data_slot(
    namespace: &Rc<RefCell<ObjectKind>>,
    name: &str,
    vm_ptr: *mut crate::vm::vm::Vm,
) {
    let mut all = unsafe { (*vm_ptr).get_module_data_slots_mut() };
    if let Some(entry) = all.get_mut(&namespace_key(namespace)) {
        entry.slots.remove(name);
    }
}

/// `import m` stored the namespace as store object `obj_id`: make its data fields the module's live
/// cells (adopting the freshly stored cells when the module has no slot for a name yet), and keep
/// them updated on later rebinding.
pub(crate) fn bind_namespace_object(
    module_object: &Value,
    obj_id: ValueId,
    store: &mut ValueStore,
    heap: &HeavyStore,
    vm_ptr: *mut crate::vm::vm::Vm,
) {
    if !is_module_namespace(module_object) {
        return;
    }
    let Value::Object(namespace) = module_object else {
        return;
    };
    let names: Vec<String> = namespace
        .borrow()
        .str_key_pairs()
        .into_iter()
        .filter(|(k, v)| k != MODULE_MARKER_KEY && !k.starts_with("__") && is_module_owned_data(v))
        .map(|(k, _)| k)
        .collect();
    let mut all = unsafe { (*vm_ptr).get_module_data_slots_mut() };
    let entry = all
        .entry(namespace_key(namespace))
        .or_insert_with(|| ModuleDataSlots::new(Rc::clone(namespace)));
    if !entry.objects.contains(&obj_id) {
        entry.objects.push(obj_id);
    }
    let mut scratch = HeavyStore::new();
    for name in names {
        let key = Value::String(name.clone());
        let kid = store_value(key.clone(), store, &mut scratch);
        if let Some(&slot) = entry.slots.get(&name) {
            let vid = slot_value_id(slot, store);
            crate::vm::store_convert::object_map_upsert_in_place(store, obj_id, heap, &key, kid, vid);
        } else if let Some(field_id) =
            crate::vm::store_convert::object_cell_try_lookup_by_key_id(obj_id, kid, store, heap)
        {
            entry.slots.insert(name, GlobalSlot::Heap(field_id));
        }
    }
}

/// Value id for `from m import name` when `name` is module data: the module's live cell for
/// mutable containers (arrays, sets, dicts: shared like Python lists), a copy of the current value
/// otherwise (rebinding the importer's name must not change the module). None when not data.
pub(crate) fn imported_data_value_id(
    namespace: &Rc<RefCell<ObjectKind>>,
    name: &str,
    store: &mut ValueStore,
    heap: &mut HeavyStore,
    vm_ptr: *mut crate::vm::vm::Vm,
) -> Option<ValueId> {
    let slot = data_slot(namespace, name, store, heap, vm_ptr)?;
    let id = slot_value_id(slot, store);
    if matches!(
        store.get(id),
        Some(ValueCell::Array(_) | ValueCell::Set(_) | ValueCell::Object(_))
    ) {
        return Some(id);
    }
    let value = crate::vm::store_convert::load_value(id, store, heap);
    Some(store_value(value, store, heap))
}
