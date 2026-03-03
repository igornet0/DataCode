// Shared VM utilities used by executor and interpreter modules.
// Extracted to avoid circular dependencies (e.g. element_ops -> executor -> interpreter -> element_ops).

use crate::common::value::Value;
use crate::common::value_store::ValueStore;
use crate::vm::global_slot::GlobalSlot;
use crate::vm::heavy_store::HeavyStore;
use crate::vm::store_convert::load_value;

/// Deterministic global slot by name (min index when multiple; stable across HashMap iteration).
pub(crate) fn global_index_by_name(global_names: &std::collections::BTreeMap<usize, String>, name: &str) -> Option<usize> {
    global_names
        .iter()
        .filter(|(_, n)| n.as_str() == name)
        .map(|(idx, _)| *idx)
        .min()
}

/// All global slot indices for a name (for updating every binding of the same name on import).
pub(crate) fn global_indices_by_name(global_names: &std::collections::BTreeMap<usize, String>, name: &str) -> Vec<usize> {
    let mut indices: Vec<usize> = global_names
        .iter()
        .filter(|(_, n)| n.as_str() == name)
        .map(|(idx, _)| *idx)
        .collect();
    indices.sort_unstable();
    indices
}

/// Returns [class_name, superclass, ...] for VM protected access checks (uses load_value for Object).
/// Stops at cycle or missing __superclass to avoid infinite loop.
pub(crate) fn get_superclass_chain(
    globals: &mut [GlobalSlot],
    global_names: &std::collections::BTreeMap<usize, String>,
    class_name: &str,
    store: &mut ValueStore,
    heap: &HeavyStore,
) -> Vec<String> {
    use std::collections::HashSet;
    let mut chain = vec![class_name.to_string()];
    let mut seen = HashSet::new();
    seen.insert(class_name.to_string());
    let mut current = class_name.to_string();
    loop {
        let super_name_opt = global_names
            .iter()
            .find(|(_, name)| name.as_str() == current)
            .and_then(|(idx, _)| {
                if *idx < globals.len() {
                    let id = globals[*idx].resolve_to_value_id(store);
                    let v = load_value(id, store, heap);
                    if let Value::Object(rc) = &v {
                        let map = rc.borrow();
                        map.get("__superclass").cloned()
                    } else {
                        None
                    }
                } else {
                    None
                }
            });
        let super_name = match super_name_opt {
            Some(Value::String(s)) => s,
            _ => break,
        };
        if !seen.insert(super_name.clone()) {
            // Cycle in class hierarchy — stop to avoid infinite loop
            break;
        }
        chain.push(super_name.clone());
        current = super_name;
    }
    chain
}
