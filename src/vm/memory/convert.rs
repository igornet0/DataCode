// Conversion between Value and ValueId for Stage 1 ValueStore migration.
// Used at native-call boundaries: materialize Value from ValueId for natives, store Value result back as ValueId.

use crate::common::value::Value;
use crate::common::value_store::{ValueCell, ValueId, ValueStore, NULL_VALUE_ID};
use crate::common::TaggedValue;
use std::cell::RefCell;
use std::rc::Rc;
use std::collections::HashMap;

use super::store::HeavyStore;

/// Store a Value into ValueStore and HeavyStore; returns its ValueId.
/// Recursive for Array and Object; heavy variants go to HeavyStore.
pub fn store_value(
    v: Value,
    store: &mut ValueStore,
    heap: &mut HeavyStore,
) -> ValueId {
    match v {
        Value::Null => NULL_VALUE_ID,
        Value::Number(n) => store.allocate(ValueCell::Number(n)),
        Value::Bool(b) => store.allocate(ValueCell::Bool(b)),
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
        Value::Tuple(rc) => {
            let arr: Vec<ValueId> = rc
                .borrow()
                .iter()
                .map(|x| store_value(x.clone(), store, heap))
                .collect();
            store.allocate(ValueCell::Tuple(arr))
        }
        Value::Function(i) => store.allocate(ValueCell::Function(i)),
        Value::ModuleFunction { module_id, local_index } => store.allocate(ValueCell::ModuleFunction { module_id: module_id, local_index: local_index }),
        Value::NativeFunction(i) => store.allocate(ValueCell::NativeFunction(i)),
        Value::Path(p) => store.allocate(ValueCell::Path(p)),
        Value::Uuid(hi, lo) => store.allocate(ValueCell::Uuid(hi, lo)),
        Value::Table(rc) => {
            let idx = heap.push(Value::Table(rc));
            store.allocate(ValueCell::Heavy(idx))
        }
        Value::Object(rc) => {
            let map = rc.borrow();
            // MetaData and create_all have circular refs; store in HeavyStore without decomposing
            // Class objects (ORM models) must stay as one Rc so SetArrayElement mutations are visible to run_create_all
            if map.get("__meta").and_then(|v| if let Value::Bool(b) = v { Some(*b) } else { None }).unwrap_or(false)
                || map.get("__create_all").and_then(|v| if let Value::Bool(b) = v { Some(*b) } else { None }).unwrap_or(false)
                || map.contains_key("__class_name")
            {
                let idx = heap.push(Value::Object(rc.clone()));
                return store.allocate(ValueCell::Heavy(idx));
            }
            let map: HashMap<String, ValueId> = map
                .iter()
                .map(|(k, v)| (k.clone(), store_value(v.clone(), store, heap)))
                .collect();
            store.allocate(ValueCell::Object(map))
        }
        Value::ColumnReference { table, column_name } => {
            let table_val = Value::Table(table);
            let idx = heap.push(table_val);
            store.allocate(ValueCell::ColumnReference {
                table_handle: idx,
                column_name,
            })
        }
        Value::Tensor(rc) => {
            let idx = heap.push(Value::Tensor(rc));
            store.allocate(ValueCell::Heavy(idx))
        }
        Value::Graph(rc) => {
            let idx = heap.push(Value::Graph(rc));
            store.allocate(ValueCell::Heavy(idx))
        }
        Value::LinearRegression(rc) => {
            let idx = heap.push(Value::LinearRegression(rc));
            store.allocate(ValueCell::Heavy(idx))
        }
        Value::SGD(rc) => {
            let idx = heap.push(Value::SGD(rc));
            store.allocate(ValueCell::Heavy(idx))
        }
        Value::Momentum(rc) => {
            let idx = heap.push(Value::Momentum(rc));
            store.allocate(ValueCell::Heavy(idx))
        }
        Value::NAG(rc) => {
            let idx = heap.push(Value::NAG(rc));
            store.allocate(ValueCell::Heavy(idx))
        }
        Value::Adagrad(rc) => {
            let idx = heap.push(Value::Adagrad(rc));
            store.allocate(ValueCell::Heavy(idx))
        }
        Value::RMSprop(rc) => {
            let idx = heap.push(Value::RMSprop(rc));
            store.allocate(ValueCell::Heavy(idx))
        }
        Value::Adam(rc) => {
            let idx = heap.push(Value::Adam(rc));
            store.allocate(ValueCell::Heavy(idx))
        }
        Value::AdamW(rc) => {
            let idx = heap.push(Value::AdamW(rc));
            store.allocate(ValueCell::Heavy(idx))
        }
        Value::Dataset(rc) => {
            let idx = heap.push(Value::Dataset(rc));
            store.allocate(ValueCell::Heavy(idx))
        }
        Value::NeuralNetwork(rc) => {
            let idx = heap.push(Value::NeuralNetwork(rc));
            store.allocate(ValueCell::Heavy(idx))
        }
        Value::Sequential(rc) => {
            let idx = heap.push(Value::Sequential(rc));
            store.allocate(ValueCell::Heavy(idx))
        }
        Value::Layer(id) => store.allocate(ValueCell::Layer(id)),
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
        Value::Enumerate { data, start } => {
            let data_id = store_value(Value::Array(data), store, heap);
            store.allocate(ValueCell::Enumerate { data_id, start })
        }
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
            let map_ref = rc.borrow();
            // MetaData and create_all have circular refs; store in HeavyStore without decomposing
            if map_ref.get("__meta").and_then(|v| if let Value::Bool(b) = v { Some(*b) } else { None }).unwrap_or(false)
                || map_ref.get("__create_all").and_then(|v| if let Value::Bool(b) = v { Some(*b) } else { None }).unwrap_or(false)
            {
                let idx = heap.push(Value::Object(rc.clone()));
                if let Some(ValueCell::Heavy(h)) = store.get_mut(id) {
                    *h = idx;
                }
            } else {
                let map: HashMap<String, ValueId> = map_ref
                    .iter()
                    .map(|(k, v)| (k.clone(), store_value(v.clone(), store, heap)))
                    .collect();
                if let Some(ValueCell::Object(m)) = store.get_mut(id) {
                    *m = map;
                }
            }
        }
        _ => {}
    }
}

/// Load a Value from ValueStore and HeavyStore by ValueId.
pub fn load_value(
    id: ValueId,
    store: &ValueStore,
    heap: &HeavyStore,
) -> Value {
    if id == NULL_VALUE_ID {
        return Value::Null;
    }
    let cell = match store.get(id) {
        Some(c) => c,
        None => return Value::Null,
    };
    match cell {
        ValueCell::Number(n) => Value::Number(*n),
        ValueCell::Bool(b) => Value::Bool(*b),
        ValueCell::Null => Value::Null,
        // Value::String is owned; one .to_string() at VM→native boundary is required (no extra clone).
        ValueCell::String(sid) => Value::String(
            store.get_string(*sid).unwrap_or("").to_string(),
        ),
        ValueCell::Array(slots) => {
            let arr: Vec<Value> = slots
                .iter()
                .map(|slot| slot_to_value(*slot, store, heap))
                .collect();
            Value::Array(Rc::new(RefCell::new(arr)))
        }
        ValueCell::Tuple(ids) => {
            let arr: Vec<Value> = ids
                .iter()
                .map(|&i| load_value(i, store, heap))
                .collect();
            Value::Tuple(Rc::new(RefCell::new(arr)))
        }
        ValueCell::Object(map) => {
            let hm: HashMap<String, Value> = map
                .iter()
                .map(|(k, v)| (k.clone(), load_value(*v, store, heap)))
                .collect();
            Value::Object(Rc::new(RefCell::new(hm)))
        }
        ValueCell::Function(i) => Value::Function(*i),
        ValueCell::ModuleFunction { module_id, local_index } => Value::ModuleFunction { module_id: *module_id, local_index: *local_index },
        ValueCell::NativeFunction(i) => Value::NativeFunction(*i),
        ValueCell::Path(p) => Value::Path(p.clone()),
        ValueCell::Uuid(hi, lo) => Value::Uuid(*hi, *lo),
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
        ValueCell::Layer(id) => Value::Layer(*id),
        ValueCell::Window(h) => Value::Window(*h),
        ValueCell::Enumerate { data_id, start } => {
            let data_val = load_value(*data_id, store, heap);
            if let Value::Array(rc) = data_val {
                Value::Enumerate { data: rc, start: *start }
            } else {
                Value::Null
            }
        }
        ValueCell::Ellipsis => Value::Ellipsis,
    }
}

/// If the cell is an immediate (Number, Bool, Null), return its TaggedValue; else None (use heap id).
pub fn value_cell_to_tagged(cell: &ValueCell) -> Option<TaggedValue> {
    match cell {
        ValueCell::Number(n) => Some(TaggedValue::from_f64(*n)),
        ValueCell::Bool(b) => Some(TaggedValue::from_bool(*b)),
        ValueCell::Null => Some(TaggedValue::null()),
        _ => None,
    }
}

/// Convert Value to TaggedValue for array slots. Inline for number/bool/null; heap id for rest.
fn value_to_slot(v: &Value, store: &mut ValueStore, heap: &mut HeavyStore) -> TaggedValue {
    match v {
        Value::Number(n) => TaggedValue::from_f64(*n),
        Value::Bool(b) => TaggedValue::from_bool(*b),
        Value::Null => TaggedValue::null(),
        _ => TaggedValue::from_heap(store_value(v.clone(), store, heap)),
    }
}

/// Convert array slot (TaggedValue) to Value for load_value. No store access for inline.
pub fn slot_to_value(slot: TaggedValue, store: &ValueStore, heap: &HeavyStore) -> Value {
    if slot.is_number() {
        Value::Number(slot.get_f64())
    } else if slot.is_bool() {
        Value::Bool(slot.get_bool())
    } else if slot.is_null() {
        Value::Null
    } else if slot.is_int() {
        Value::Number(slot.get_i32() as f64)
    } else if slot.is_heap() {
        load_value(slot.get_heap_id(), store, heap)
    } else {
        Value::Null
    }
}

/// Convert a TaggedValue to ValueId. Allocates a cell for immediates (number, bool, null, int); returns existing id for heap.
pub fn tagged_to_value_id(tv: TaggedValue, store: &mut ValueStore) -> ValueId {
    if tv.is_number() {
        store.allocate(ValueCell::Number(tv.get_f64()))
    } else if tv.is_null() {
        NULL_VALUE_ID
    } else if tv.is_bool() {
        store.allocate(ValueCell::Bool(tv.get_bool()))
    } else if tv.is_int() {
        store.allocate(ValueCell::Number(tv.get_i32() as f64))
    } else if tv.is_heap() {
        tv.get_heap_id()
    } else {
        // Unknown tag (e.g. f64::NAN bit pattern mistaken for tagged): treat as null
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
        store.allocate_arena(ValueCell::Number(tv.get_i32() as f64))
    } else if tv.is_heap() {
        tv.get_heap_id()
    } else {
        NULL_VALUE_ID
    }
}

/// Store a Value into the heap arena (for globals / ephemeral heap). Same semantics as store_value but allocate_arena.
pub fn store_value_arena(
    v: Value,
    store: &mut ValueStore,
    heap: &mut HeavyStore,
) -> ValueId {
    match v {
        Value::Null => NULL_VALUE_ID,
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
        Value::ModuleFunction { module_id, local_index } => store.allocate_arena(ValueCell::ModuleFunction { module_id: module_id, local_index: local_index }),
        Value::NativeFunction(i) => store.allocate_arena(ValueCell::NativeFunction(i)),
        Value::Path(p) => store.allocate_arena(ValueCell::Path(p)),
        Value::Uuid(hi, lo) => store.allocate_arena(ValueCell::Uuid(hi, lo)),
        Value::Table(rc) => {
            let idx = heap.push(Value::Table(rc));
            store.allocate_arena(ValueCell::Heavy(idx))
        }
        Value::Object(rc) => {
            let map = rc.borrow();
            // MetaData and create_all have circular refs; store in HeavyStore without decomposing
            if map.get("__meta").and_then(|v| if let Value::Bool(b) = v { Some(*b) } else { None }).unwrap_or(false)
                || map.get("__create_all").and_then(|v| if let Value::Bool(b) = v { Some(*b) } else { None }).unwrap_or(false)
            {
                let idx = heap.push(Value::Object(rc.clone()));
                return store.allocate_arena(ValueCell::Heavy(idx));
            }
            // Class objects (ORM models) stay as one Rc so __col_* set by class body is visible to metadata.classes and run_create_all
            if map.contains_key("__class_name") {
                let idx = heap.push(Value::Object(rc.clone()));
                return store.allocate_arena(ValueCell::Heavy(idx));
            }
            let map: HashMap<String, ValueId> = map
                .iter()
                .map(|(k, v)| (k.clone(), store_value_arena(v.clone(), store, heap)))
                .collect();
            store.allocate_arena(ValueCell::Object(map))
        }
        Value::ColumnReference { table, column_name } => {
            let table_val = Value::Table(table);
            let idx = heap.push(table_val);
            store.allocate_arena(ValueCell::ColumnReference {
                table_handle: idx,
                column_name,
            })
        }
        Value::Tensor(rc) => {
            let idx = heap.push(Value::Tensor(rc));
            store.allocate_arena(ValueCell::Heavy(idx))
        }
        Value::Graph(rc) => {
            let idx = heap.push(Value::Graph(rc));
            store.allocate_arena(ValueCell::Heavy(idx))
        }
        Value::LinearRegression(rc) => {
            let idx = heap.push(Value::LinearRegression(rc));
            store.allocate_arena(ValueCell::Heavy(idx))
        }
        Value::SGD(rc) => {
            let idx = heap.push(Value::SGD(rc));
            store.allocate_arena(ValueCell::Heavy(idx))
        }
        Value::Momentum(rc) => {
            let idx = heap.push(Value::Momentum(rc));
            store.allocate_arena(ValueCell::Heavy(idx))
        }
        Value::NAG(rc) => {
            let idx = heap.push(Value::NAG(rc));
            store.allocate_arena(ValueCell::Heavy(idx))
        }
        Value::Adagrad(rc) => {
            let idx = heap.push(Value::Adagrad(rc));
            store.allocate_arena(ValueCell::Heavy(idx))
        }
        Value::RMSprop(rc) => {
            let idx = heap.push(Value::RMSprop(rc));
            store.allocate_arena(ValueCell::Heavy(idx))
        }
        Value::Adam(rc) => {
            let idx = heap.push(Value::Adam(rc));
            store.allocate_arena(ValueCell::Heavy(idx))
        }
        Value::AdamW(rc) => {
            let idx = heap.push(Value::AdamW(rc));
            store.allocate_arena(ValueCell::Heavy(idx))
        }
        Value::Dataset(rc) => {
            let idx = heap.push(Value::Dataset(rc));
            store.allocate_arena(ValueCell::Heavy(idx))
        }
        Value::NeuralNetwork(rc) => {
            let idx = heap.push(Value::NeuralNetwork(rc));
            store.allocate_arena(ValueCell::Heavy(idx))
        }
        Value::Sequential(rc) => {
            let idx = heap.push(Value::Sequential(rc));
            store.allocate_arena(ValueCell::Heavy(idx))
        }
        Value::Layer(id) => store.allocate_arena(ValueCell::Layer(id)),
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
        Value::Enumerate { data, start } => {
            let data_id = store_value_arena(Value::Array(data), store, heap);
            store.allocate_arena(ValueCell::Enumerate { data_id, start })
        }
        Value::Ellipsis => store.allocate_arena(ValueCell::Ellipsis),
    }
}

fn value_to_slot_arena(v: &Value, store: &mut ValueStore, heap: &mut HeavyStore) -> TaggedValue {
    match v {
        Value::Number(n) => TaggedValue::from_f64(*n),
        Value::Bool(b) => TaggedValue::from_bool(*b),
        Value::Null => TaggedValue::null(),
        _ => TaggedValue::from_heap(store_value_arena(v.clone(), store, heap)),
    }
}
