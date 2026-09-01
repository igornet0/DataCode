//! GetArrayElement and SetArrayElement opcodes (large bodies, kept separate).

mod array_ops;
mod indexing;
mod indexing_lib;
mod object_dict;
mod object_fields;
mod table_ops;

use crate::common::{
    error::{ErrorType, LangError},
    value::{IterableInner, Value},
    value_store::{ObjectProjectionKind, ValueCell, ValueId, ValueStore, NULL_VALUE_ID},
    TaggedValue,
};
use crate::vm::memory::{push_integral_slot, push_stack_value_id};
use crate::debug_println;
use crate::vm::exceptions::ExceptionHandler;
use crate::vm::frame::CallFrame;
use crate::vm::heavy_store::HeavyStore;
use crate::vm::stack;
use crate::vm::store_convert::{
    load_value, object_cell_try_lookup_by_key_id,
    object_map_upsert_in_place, prepare_value_for_container_store, promote_value_id_if_escapes,
    store_value,
    tagged_to_mutable_value_id, tagged_to_value_id, tagged_to_value_id_arena,
    try_write_cell_from_tagged,
};
use crate::vm::types::VMStatus;

use super::helpers::pop_to_value_id;
use crate::vm::array_view::resolve_slice_origin;
use crate::vm::iterable::{chunk_source_count, materialize_chunk_at};
use std::sync::OnceLock;

/// Precomputed [`ObjectMap`] bucket hashes for visibility metadata keys (see `store_value` / private fields).
static VISIBILITY_METADATA_KEY_HASH_ENTRIES: OnceLock<[(u64, &'static str); 7]> = OnceLock::new();

#[inline]
fn visibility_metadata_key_hash_entries() -> &'static [(u64, &'static str); 7] {
    VISIBILITY_METADATA_KEY_HASH_ENTRIES.get_or_init(|| {
        const KEYS: [&str; 7] = [
            "__class_name",
            "__private_fields",
            "__protected_fields",
            "__private_methods",
            "__protected_methods",
            "__class_private_vars",
            "__class_protected_vars",
        ];
        KEYS.map(|key_str| {
            let h = crate::common::type_model::object_key_hash_value(&Value::String(
                key_str.to_string(),
            ))
            .expect("visibility metadata key must be hashable");
            (h, key_str)
        })
    })
}

/// True if this bucket map may carry class instance / class visibility rules (must not use raw element lookup).
pub(crate) fn object_map_needs_visibility_checks(
    omap: &crate::common::object_map::ObjectMap,
    store: &ValueStore,
    heap: &HeavyStore,
) -> bool {
    // Class instances are stored in plain [`ObjectMap`]s (`MakeObject`); check visibility metadata
    // keys before `is_plain()` — otherwise private/protected enforcement is skipped on set/get fast paths.
    for &(h, key_str) in visibility_metadata_key_hash_entries() {
        if omap
            .find_in_bucket(h, |k_id| match load_value(k_id, store, heap) {
                Value::String(s) => s.as_str() == key_str,
                _ => false,
            })
            .is_some()
        {
            return true;
        }
    }
    false
}

/// True when the map has an own callable property `prop` (`NativeFunction` / `Function` / `ModuleFunction`).
#[inline]
fn object_map_has_own_callable_prop(
    omap: &crate::common::object_map::ObjectMap,
    store: &ValueStore,
    heap: &HeavyStore,
    prop: &str,
) -> bool {
    let prop_key = Value::String(prop.to_string());
    let Some(h) = crate::common::type_model::object_key_hash_value(&prop_key) else {
        return false;
    };
    let Some(value_id) = omap.find_in_bucket(h, |k_id| {
        matches!(load_value(k_id, store, heap), Value::String(ref s) if s == prop)
    }) else {
        return false;
    };
    matches!(
        load_value(value_id, store, heap),
        Value::NativeFunction(_) | Value::Function(_) | Value::ModuleFunction { .. }
    )
}

/// Whether to inject plain-dict builtin for property `prop` (`"get"` → OBJECT_GET, `"clear"` → OBJECT_CLEAR).
/// Class-like maps and maps with an own callable `prop` keep their real member.
#[inline]
fn should_inject_dict_builtin(
    omap: &crate::common::object_map::ObjectMap,
    store: &ValueStore,
    heap: &HeavyStore,
    prop: &str,
) -> bool {
    if object_map_needs_visibility_checks(omap, store, heap) {
        return false;
    }
    !object_map_has_own_callable_prop(omap, store, heap, prop)
}

/// Plain bucket dicts (no own callable `get`) expose builtin `.get(key [, default])`.
#[inline]
fn object_map_use_builtin_get(
    omap: &crate::common::object_map::ObjectMap,
    store: &ValueStore,
    heap: &HeavyStore,
) -> bool {
    should_inject_dict_builtin(omap, store, heap, "get")
}

/// Plain dict `obj[n]` when `n` is numeric and missing → `null` (adjacency / parent maps in graph examples).
/// String keys still raise `KeyError` (see `object_get_tests`).
#[inline]
fn plain_dict_numeric_missing_key_is_null(key: &Value) -> bool {
    matches!(key, Value::Number(_) | Value::Int(_))
}

/// Native index for `set.<method>` when `index` is a string property name (loads only the key, not the set).
#[inline]
fn set_method_native_index(
    index_value_id: ValueId,
    store: &ValueStore,
    heap: &HeavyStore,
) -> Option<usize> {
    match load_value(index_value_id, store, heap) {
        Value::String(ref key) => match key.as_str() {
            "add" => Some(crate::vm::native_indices::builtin::SET_ADD),
            "remove" => Some(crate::vm::native_indices::builtin::SET_REMOVE),
            "discard" => Some(crate::vm::native_indices::builtin::SET_DISCARD),
            "pop" => Some(crate::vm::native_indices::builtin::SET_POP),
            "clear" => Some(crate::vm::native_indices::builtin::SET_CLEAR),
            "copy" => Some(crate::vm::native_indices::builtin::SET_COPY),
            "update" => Some(crate::vm::native_indices::builtin::SET_UPDATE),
            "contains" => Some(crate::vm::native_indices::builtin::SET_CONTAINS),
            _ => None,
        },
        _ => None,
    }
}

#[inline]
fn push_set_method_native(
    stack: &mut Vec<TaggedValue>,
    value_store: &mut ValueStore,
    heavy_store: &mut HeavyStore,
    native_idx: usize,
) {
    stack::push_id(
        stack,
        store_value(
            Value::NativeFunction(native_idx),
            value_store,
            heavy_store,
        ),
    );
}

/// `set.add` / `.discard` property access on [`ValueCell::Set`] without cloning the whole set.
#[inline]
fn try_store_set_method_get(
    container_id: ValueId,
    index_value_id: ValueId,
    stack: &mut Vec<TaggedValue>,
    value_store: &mut ValueStore,
    heavy_store: &mut HeavyStore,
) -> bool {
    if !matches!(
        value_store.get(container_id),
        Some(ValueCell::Set(_))
    ) {
        return false;
    }
    let Some(native_idx) = set_method_native_index(index_value_id, value_store, heavy_store) else {
        return false;
    };
    push_set_method_native(stack, value_store, heavy_store, native_idx);
    true
}

/// Resolve canonical `i64` dict key from stack index (immediate or heap whole-number cell).
#[inline]
fn integral_lookup_key_from_index(index_tv: TaggedValue, store: &mut ValueStore) -> Option<i64> {
    crate::common::numeric::tagged_integral_canonical_if_whole(index_tv).or_else(|| {
        if index_tv.is_heap() {
            crate::vm::memory::canonical_integral_from_key_id_mut(index_tv.get_heap_id(), store)
        } else {
            None
        }
    })
}

#[inline]
fn set_get_element_object_integral_cache(
    frame: &mut CallFrame,
    current_ip: usize,
    container_tv: TaggedValue,
    key_canonical: i64,
) {
    frame.get_array_element_cache_ip = Some(current_ip);
    frame.get_array_element_cache_array_number = false;
    frame.get_array_element_cache_object_string = false;
    frame.get_array_element_cache_object_integral = true;
    frame.get_array_element_cache_container_tv = Some(container_tv);
    frame.get_array_element_cache_index_tv = None;
    frame.get_array_element_cache_integral_key = Some(key_canonical);
}

/// Plain dict `obj[integral]` / repeated access at same IP (no [`ObjectMap`] clone).
#[inline]
fn try_plain_dict_integral_element_get(
    omap: &crate::common::object_map::ObjectMap,
    key_canonical: i64,
    store: &ValueStore,
    heap: &HeavyStore,
) -> Option<crate::common::integral_map::IntegralSlot> {
    if !object_map_use_builtin_get(omap, store, heap)
        || object_map_needs_visibility_checks(omap, store, heap)
    {
        return None;
    }
    Some(
        omap.find_integral_slot(key_canonical)
            .unwrap_or(crate::common::integral_map::IntegralSlot::Immediate(
                TaggedValue::null(),
            )),
    )
}

/// Class instances may declare fields named `keys` / `values` (e.g. B-tree nodes). Those members
/// must resolve via [`object_fields::get_object`], not plain-dict `.keys` / `.values` projections.
/// Store-backed tuple `t[i]` with integral index — no full [`Value`] materialization.
#[inline]
fn try_store_tuple_integral_get(
    container_id: ValueId,
    index_tv: TaggedValue,
    stack: &mut Vec<TaggedValue>,
    value_store: &mut ValueStore,
) -> bool {
    let idx = if index_tv.is_int() {
        index_tv.get_i32() as i64
    } else if index_tv.is_number() {
        let n = index_tv.get_f64();
        if n.fract() != 0.0 || n < 0.0 {
            return false;
        }
        n as i64
    } else {
        return false;
    };
    if idx < 0 {
        return false;
    }
    let u = idx as usize;
    if let Some(ValueCell::Tuple(ids)) = value_store.get(container_id) {
        if u < ids.len() {
            push_stack_value_id(stack, value_store, ids[u]);
            return true;
        }
    }
    false
}

enum ScalarIndexFastHit {
    Dict(i64),
    Array,
}

/// Plain dict `obj[integral]` or array `arr[i]` without IP-cache / full GetArrayElement dispatch.
#[inline]
fn try_scalar_integral_index_get(
    container_tv: TaggedValue,
    index_tv: TaggedValue,
    stack: &mut Vec<TaggedValue>,
    value_store: &mut ValueStore,
    heavy_store: &HeavyStore,
) -> Option<ScalarIndexFastHit> {
    if !container_tv.is_heap() {
        return None;
    }
    let container_id = container_tv.get_heap_id();
    if try_store_tuple_integral_get(container_id, index_tv, stack, value_store) {
        return Some(ScalarIndexFastHit::Array);
    }
    if let Some(key_canonical) = integral_lookup_key_from_index(index_tv, value_store) {
        let plain_dict = value_store.is_plain_object(container_id);
        if let Some(ValueCell::Object(omap)) = value_store.get_mut(container_id) {
            if plain_dict || omap.is_plain() {
                let slot = omap
                    .find_integral_slot(key_canonical)
                    .unwrap_or(crate::common::integral_map::IntegralSlot::Immediate(
                        TaggedValue::null(),
                    ));
                push_integral_slot(stack, value_store, slot);
                return Some(ScalarIndexFastHit::Dict(key_canonical));
            }
        } else if let Some(ValueCell::Object(omap)) = value_store.get(container_id) {
            if let Some(slot) = try_plain_dict_integral_element_get(
                omap,
                key_canonical,
                value_store,
                heavy_store,
            ) {
                push_integral_slot(stack, value_store, slot);
                return Some(ScalarIndexFastHit::Dict(key_canonical));
            }
        }
    }
    if index_tv.is_int() || index_tv.is_number() {
        let idx = if index_tv.is_int() {
            index_tv.get_i32() as i64
        } else {
            let n = index_tv.get_f64();
            if n.fract() != 0.0 || n < 0.0 {
                return None;
            }
            n as i64
        };
        if idx >= 0 {
            if let Some(ValueCell::Array(ref vec)) = value_store.get_mut(container_id) {
                let u = idx as usize;
                if u < vec.len() {
                    stack::push(stack, vec[u]);
                    return Some(ScalarIndexFastHit::Array);
                }
            }
        }
    }
    None
}

/// Integral subscript fast path + [`CallFrame`] inline cache (shared by [`OpCode::ObjectIndexIntegral`]).
#[inline]
fn object_index_integral_fast(
    current_ip: usize,
    container_tv: TaggedValue,
    index_tv: TaggedValue,
    stack: &mut Vec<TaggedValue>,
    frame: &mut CallFrame,
    value_store: &mut ValueStore,
    heavy_store: &HeavyStore,
) -> bool {
    let key_canonical_early = integral_lookup_key_from_index(index_tv, value_store);
    let cache_obj_int = frame.get_array_element_cache_ip == Some(current_ip)
        && frame.get_array_element_cache_object_integral
        && frame.get_array_element_cache_container_tv == Some(container_tv)
        && key_canonical_early.is_some()
        && frame.get_array_element_cache_integral_key == key_canonical_early;
    if cache_obj_int && container_tv.is_heap() {
        if let Some(key_canonical) = key_canonical_early {
            let container_id = container_tv.get_heap_id();
            if let Some(ValueCell::Object(omap)) = value_store.get_mut(container_id) {
                let slot = omap
                    .find_integral_slot(key_canonical)
                    .unwrap_or(crate::common::integral_map::IntegralSlot::Immediate(
                        TaggedValue::null(),
                    ));
                push_integral_slot(stack, value_store, slot);
                return true;
            }
            if let Some(ValueCell::Object(omap)) = value_store.get(container_id) {
                if let Some(slot) = try_plain_dict_integral_element_get(
                    omap,
                    key_canonical,
                    value_store,
                    heavy_store,
                ) {
                    push_integral_slot(stack, value_store, slot);
                    set_get_element_object_integral_cache(
                        frame,
                        current_ip,
                        container_tv,
                        key_canonical,
                    );
                    return true;
                }
            }
        }
        frame.get_array_element_cache_object_integral = false;
        frame.get_array_element_cache_container_tv = None;
        frame.get_array_element_cache_integral_key = None;
    }
    match try_scalar_integral_index_get(
        container_tv,
        index_tv,
        stack,
        value_store,
        heavy_store,
    ) {
        Some(ScalarIndexFastHit::Dict(key_canonical)) => {
            set_get_element_object_integral_cache(
                frame,
                current_ip,
                container_tv,
                key_canonical,
            );
            true
        }
        Some(ScalarIndexFastHit::Array) => true,
        None => false,
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn op_object_index_integral(
    line: usize,
    stack: &mut Vec<TaggedValue>,
    frames: &mut Vec<CallFrame>,
    globals: &mut Vec<crate::vm::global_slot::GlobalSlot>,
    global_names: &std::collections::BTreeMap<usize, String>,
    functions: &[crate::bytecode::Function],
    natives: &[crate::vm::host::HostEntry],
    exception_handlers: &mut Vec<ExceptionHandler>,
    value_store: &mut ValueStore,
    heavy_store: &mut HeavyStore,
    vm_ptr: *mut crate::vm::vm::Vm,
) -> Result<VMStatus, LangError> {
    let index_tv = stack::pop(stack, frames, exception_handlers, value_store, heavy_store)?;
    let container_tv = stack::pop(stack, frames, exception_handlers, value_store, heavy_store)?;
    let current_ip = frames.last().map(|f| f.ip.saturating_sub(1)).unwrap_or(0);
    if let Some(frame) = frames.last_mut() {
        if object_index_integral_fast(
            current_ip,
            container_tv,
            index_tv,
            stack,
            frame,
            value_store,
            heavy_store,
        ) {
            return Ok(VMStatus::Continue);
        }
    }
    stack::push(stack, container_tv);
    stack::push(stack, index_tv);
    op_get_array_element(
        line,
        stack,
        frames,
        globals,
        global_names,
        functions,
        natives,
        exception_handlers,
        value_store,
        heavy_store,
        vm_ptr,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn op_object_set_integral(
    line: usize,
    stack: &mut Vec<TaggedValue>,
    frames: &mut Vec<CallFrame>,
    globals: &mut Vec<crate::vm::global_slot::GlobalSlot>,
    global_names: &std::collections::BTreeMap<usize, String>,
    functions: &[crate::bytecode::Function],
    natives: &[crate::vm::host::HostEntry],
    exception_handlers: &mut Vec<ExceptionHandler>,
    value_store: &mut ValueStore,
    heavy_store: &mut HeavyStore,
) -> Result<VMStatus, LangError> {
    let container_id =
        pop_to_value_id(stack, frames, exception_handlers, value_store, heavy_store)?;
    let index_tv = stack::pop(stack, frames, exception_handlers, value_store, heavy_store)?;
    let mut value_tv = stack::pop(stack, frames, exception_handlers, value_store, heavy_store)?;
    value_tv = prepare_value_for_container_store(container_id, value_tv, value_store);
    let key_canonical = integral_lookup_key_from_index(index_tv, value_store);
    let index_key_id = if key_canonical.is_some() {
        NULL_VALUE_ID
    } else {
        tagged_to_value_id(index_tv, value_store)
    };

    if let Some(canonical) = key_canonical {
        let immediate_score = !value_tv.is_heap()
            && (value_tv.is_number() || value_tv.is_int() || value_tv.is_null());

        if value_store.is_plain_object(container_id) {
            if let Some(ValueCell::Object(omap)) = value_store.get_mut(container_id) {
                if omap.try_write_integral_immediate(canonical, value_tv) {
                    stack::push_id(stack, container_id);
                    return Ok(VMStatus::Continue);
                }
            }
            let existing_heap_id = match value_store.get_mut(container_id) {
                Some(ValueCell::Object(omap)) => {
                    omap.find_integral_slot(canonical).and_then(|slot| match slot {
                        crate::common::integral_map::IntegralSlot::Heap(id) => Some(id),
                        _ => None,
                    })
                }
                _ => None,
            };
            if existing_heap_id.is_some_and(|existing| {
                existing != NULL_VALUE_ID
                    && try_write_cell_from_tagged(existing, value_tv, value_store)
            }) {
                stack::push_id(stack, container_id);
                return Ok(VMStatus::Continue);
            }
            if immediate_score {
                value_store.plain_object_upsert_integral_tagged(
                    container_id,
                    canonical,
                    index_key_id,
                    value_tv,
                );
            } else {
                let value_id = promote_value_id_if_escapes(
                    container_id,
                    tagged_to_mutable_value_id(value_tv, value_store),
                    value_store,
                );
                value_store.plain_object_upsert_integral(
                    container_id,
                    canonical,
                    index_key_id,
                    value_id,
                );
            }
            stack::push_id(stack, container_id);
            return Ok(VMStatus::Continue);
        }

        if let Some(ValueCell::Object(omap)) = value_store.get_mut(container_id) {
            if omap.try_write_integral_immediate(canonical, value_tv) {
                stack::push_id(stack, container_id);
                return Ok(VMStatus::Continue);
            }
        }
        let existing_heap_id = match value_store.get_mut(container_id) {
            Some(ValueCell::Object(omap)) if omap.is_plain() => {
                omap.find_integral_slot(canonical).and_then(|slot| match slot {
                    crate::common::integral_map::IntegralSlot::Heap(id) => Some(id),
                    _ => None,
                })
            }
            _ => None,
        };
        if existing_heap_id.is_some_and(|existing| {
            existing != NULL_VALUE_ID
                && try_write_cell_from_tagged(existing, value_tv, value_store)
        }) {
            stack::push_id(stack, container_id);
            return Ok(VMStatus::Continue);
        }

        if value_store.is_plain_object(container_id) {
            if immediate_score {
                value_store.plain_object_upsert_integral_tagged(
                    container_id,
                    canonical,
                    index_key_id,
                    value_tv,
                );
            } else {
                let value_id = promote_value_id_if_escapes(
                    container_id,
                    tagged_to_mutable_value_id(value_tv, value_store),
                    value_store,
                );
                value_store.plain_object_upsert_integral(
                    container_id,
                    canonical,
                    index_key_id,
                    value_id,
                );
            }
            stack::push_id(stack, container_id);
            return Ok(VMStatus::Continue);
        }
        if let Some(ValueCell::Object(omap)) = value_store.get_mut(container_id) {
            if omap.is_plain() {
                if immediate_score {
                    value_store.plain_object_upsert_integral_tagged(
                        container_id,
                        canonical,
                        index_key_id,
                        value_tv,
                    );
                } else {
                    let value_id = promote_value_id_if_escapes(
                        container_id,
                        tagged_to_mutable_value_id(value_tv, value_store),
                        value_store,
                    );
                    value_store.plain_object_upsert_integral(
                        container_id,
                        canonical,
                        index_key_id,
                        value_id,
                    );
                }
                stack::push_id(stack, container_id);
                return Ok(VMStatus::Continue);
            }
        }
        let visibility_free = match value_store.get(container_id) {
            Some(ValueCell::Object(omap)) => {
                !object_map_needs_visibility_checks(omap, value_store, heavy_store)
            }
            _ => false,
        };
        if visibility_free {
            if let Some(ValueCell::Object(omap)) = value_store.get_mut(container_id) {
                if omap.try_write_integral_immediate(canonical, value_tv) {
                    stack::push_id(stack, container_id);
                    return Ok(VMStatus::Continue);
                }
            }
            if immediate_score {
                value_store.plain_object_upsert_integral_tagged(
                    container_id,
                    canonical,
                    index_key_id,
                    value_tv,
                );
            } else {
                let value_id = promote_value_id_if_escapes(
                    container_id,
                    tagged_to_mutable_value_id(value_tv, value_store),
                    value_store,
                );
                value_store.plain_object_upsert_integral(
                    container_id,
                    canonical,
                    index_key_id,
                    value_id,
                );
            }
            stack::push_id(stack, container_id);
            return Ok(VMStatus::Continue);
        }
    }

    stack::push(stack, value_tv);
    stack::push(stack, index_tv);
    stack::push_id(stack, container_id);
    op_set_array_element(
        line,
        stack,
        frames,
        globals,
        global_names,
        functions,
        natives,
        exception_handlers,
        value_store,
        heavy_store,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn op_get_array_element(
    line: usize,
    stack: &mut Vec<TaggedValue>,
    frames: &mut Vec<CallFrame>,
    globals: &mut Vec<crate::vm::global_slot::GlobalSlot>,
    global_names: &std::collections::BTreeMap<usize, String>,
    _functions: &[crate::bytecode::Function],
    natives: &[crate::vm::host::HostEntry],
    exception_handlers: &mut Vec<ExceptionHandler>,
    value_store: &mut ValueStore,
    heavy_store: &mut HeavyStore,
    vm_ptr: *mut crate::vm::vm::Vm,
) -> Result<VMStatus, LangError> {
    let index_tv = stack::pop(stack, frames, exception_handlers, value_store, heavy_store)?;
    let container_tv = stack::pop(stack, frames, exception_handlers, value_store, heavy_store)?;
    if let Some(hit) = try_scalar_integral_index_get(
        container_tv,
        index_tv,
        stack,
        value_store,
        heavy_store,
    ) {
        let current_ip = frames.last().unwrap().ip - 1;
        let frame = frames.last_mut().unwrap();
        match hit {
            ScalarIndexFastHit::Dict(key_canonical) => {
                set_get_element_object_integral_cache(
                    frame,
                    current_ip,
                    container_tv,
                    key_canonical,
                );
            }
            ScalarIndexFastHit::Array => {
                frame.get_array_element_cache_ip = Some(current_ip);
                frame.get_array_element_cache_array_number = true;
                frame.get_array_element_cache_object_string = false;
                frame.get_array_element_cache_object_integral = false;
                frame.get_array_element_cache_container_tv = Some(container_tv);
                frame.get_array_element_cache_index_tv = Some(index_tv);
            }
        }
        return Ok(VMStatus::Continue);
    }
    let current_ip = {
        let frame = frames.last().unwrap();
        frame.ip - 1
    };
    {
        let frame = frames.last_mut().unwrap();
        let cache_array = frame.get_array_element_cache_ip == Some(current_ip)
            && frame.get_array_element_cache_array_number
            && frame.get_array_element_cache_container_tv == Some(container_tv)
            && frame.get_array_element_cache_index_tv == Some(index_tv);
        let cache_obj = frame.get_array_element_cache_ip == Some(current_ip)
            && frame.get_array_element_cache_object_string
            && frame.get_array_element_cache_container_tv == Some(container_tv)
            && frame.get_array_element_cache_index_tv == Some(index_tv);
        let key_canonical_early = integral_lookup_key_from_index(index_tv, value_store);
        let cache_obj_int = frame.get_array_element_cache_ip == Some(current_ip)
            && frame.get_array_element_cache_object_integral
            && frame.get_array_element_cache_container_tv == Some(container_tv)
            && key_canonical_early.is_some()
            && frame.get_array_element_cache_integral_key == key_canonical_early;
        if cache_obj_int && container_tv.is_heap() {
            if let Some(key_canonical) = key_canonical_early {
                let container_id = container_tv.get_heap_id();
                if let Some(ValueCell::Object(omap)) = value_store.get_mut(container_id) {
                    if omap.is_plain() {
                        let slot = omap
                            .find_integral_slot(key_canonical)
                            .unwrap_or(crate::common::integral_map::IntegralSlot::Immediate(
                                TaggedValue::null(),
                            ));
                        push_integral_slot(stack, value_store, slot);
                        set_get_element_object_integral_cache(
                            frame,
                            current_ip,
                            container_tv,
                            key_canonical,
                        );
                        return Ok(VMStatus::Continue);
                    }
                }
                if let Some(ValueCell::Object(omap)) = value_store.get(container_id) {
                    if let Some(slot) =
                        try_plain_dict_integral_element_get(omap, key_canonical, value_store, heavy_store)
                    {
                        push_integral_slot(stack, value_store, slot);
                        set_get_element_object_integral_cache(
                            frame,
                            current_ip,
                            container_tv,
                            key_canonical,
                        );
                        return Ok(VMStatus::Continue);
                    }
                }
            }
            frame.get_array_element_cache_object_integral = false;
            frame.get_array_element_cache_container_tv = None;
            frame.get_array_element_cache_integral_key = None;
        }
        if cache_array && container_tv.is_heap() && index_tv.is_number() {
            let container_id = container_tv.get_heap_id();
            let idx = index_tv.get_f64() as i64;
            if idx >= 0 {
                if let Some(ValueCell::Array(ref vec)) = value_store.get(container_id) {
                    let u = idx as usize;
                    if u < vec.len() {
                        stack::push(stack, vec[u]);
                        frame.get_array_element_cache_ip = Some(current_ip);
                        frame.get_array_element_cache_array_number = true;
                        frame.get_array_element_cache_object_string = false;
                        frame.get_array_element_cache_object_integral = false;
                        frame.get_array_element_cache_container_tv = Some(container_tv);
                        frame.get_array_element_cache_index_tv = Some(index_tv);
                        return Ok(VMStatus::Continue);
                    }
                }
            }
            frame.get_array_element_cache_array_number = false;
            frame.get_array_element_cache_container_tv = None;
            frame.get_array_element_cache_index_tv = None;
        }
        if container_tv.is_heap() && index_tv.is_heap() {
            let container_id = container_tv.get_heap_id();
            let index_value_id = index_tv.get_heap_id();
            if try_store_set_method_get(
                container_id,
                index_value_id,
                stack,
                value_store,
                heavy_store,
            ) {
                let frame = frames.last_mut().unwrap();
                frame.get_array_element_cache_ip = Some(current_ip);
                frame.get_array_element_cache_array_number = false;
                frame.get_array_element_cache_object_string = true;
                frame.get_array_element_cache_object_integral = false;
                frame.get_array_element_cache_container_tv = Some(container_tv);
                frame.get_array_element_cache_index_tv = Some(index_tv);
                return Ok(VMStatus::Continue);
            }
        }
        if cache_obj && container_tv.is_heap() && index_tv.is_heap() {
            let container_id = container_tv.get_heap_id();
            let index_value_id = index_tv.get_heap_id();
            if let Some(proj) =
                object_dict::projection_for_keys_values_property(index_value_id, value_store)
            {
                if let Some(ValueCell::Object(omap)) = value_store.get(container_id).cloned() {
                    if !object_map_needs_visibility_checks(&omap, value_store, heavy_store) {
                        let list_id = object_dict::allocate_object_field_list(
                            container_id,
                            &omap,
                            proj,
                            value_store,
                        );
                        stack::push_id(stack, list_id);
                        let frame = frames.last_mut().unwrap();
                        frame.get_array_element_cache_ip = Some(current_ip);
                        frame.get_array_element_cache_array_number = false;
                        frame.get_array_element_cache_object_string = true;
                        frame.get_array_element_cache_object_integral = false;
                        frame.get_array_element_cache_container_tv = Some(container_tv);
                        frame.get_array_element_cache_index_tv = Some(index_tv);
                        return Ok(VMStatus::Continue);
                    }
                }
            } else if let Some(ValueCell::Object(omap)) = value_store.get(container_id) {
                if object_map_use_builtin_get(&omap, value_store, heavy_store)
                    && matches!(
                        load_value(index_value_id, value_store, heavy_store),
                        Value::String(ref s) if s == "get"
                    )
                {
                    stack::push_id(
                        stack,
                        store_value(
                            Value::NativeFunction(
                                crate::vm::native_indices::builtin::OBJECT_GET,
                            ),
                            value_store,
                            heavy_store,
                        ),
                    );
                    frame.get_array_element_cache_ip = Some(current_ip);
                    frame.get_array_element_cache_array_number = false;
                    frame.get_array_element_cache_object_string = true;
                    frame.get_array_element_cache_object_integral = false;
                    frame.get_array_element_cache_container_tv = Some(container_tv);
                    frame.get_array_element_cache_index_tv = Some(index_tv);
                    return Ok(VMStatus::Continue);
                }
                if object_map_needs_visibility_checks(&omap, value_store, heavy_store) {
                    // Class-like maps must go through `object_fields::get_object` — no raw bypass.
                    frame.get_array_element_cache_object_string = false;
                    frame.get_array_element_cache_object_integral = false;
                    frame.get_array_element_cache_container_tv = None;
                    frame.get_array_element_cache_index_tv = None;
                } else if let Some(key_canonical) =
                    crate::vm::memory::canonical_integral_from_key_id(index_value_id, value_store)
                {
                    if let Some(slot) = try_plain_dict_integral_element_get(
                        &omap,
                        key_canonical,
                        value_store,
                        heavy_store,
                    ) {
                        push_integral_slot(stack, value_store, slot);
                        set_get_element_object_integral_cache(
                            frame,
                            current_ip,
                            container_tv,
                            key_canonical,
                        );
                        return Ok(VMStatus::Continue);
                    }
                } else {
                    let key_material =
                        load_value(index_value_id, value_store, heavy_store);
                    if !crate::common::type_model::is_hashable_value(&key_material) {
                        let tn = crate::vm::calls::get_type_name_value(&key_material);
                        let error = ExceptionHandler::runtime_error_with_type(
                            &frames,
                            format!("unhashable type: {}", tn),
                            line,
                            ErrorType::TypeError,
                        );
                        return match ExceptionHandler::handle_exception(
                            stack,
                            frames,
                            exception_handlers,
                            error,
                            value_store,
                            heavy_store,
                        ) {
                            Ok(()) => Ok(VMStatus::Continue),
                            Err(e) => Err(e),
                        };
                    }
                    match object_cell_try_lookup_by_key_id(
                        container_id,
                        index_value_id,
                        value_store,
                        heavy_store,
                    ) {
                        None if plain_dict_numeric_missing_key_is_null(&key_material) => {
                            push_stack_value_id(stack, value_store, NULL_VALUE_ID);
                            frame.get_array_element_cache_ip = Some(current_ip);
                            frame.get_array_element_cache_array_number = false;
                            frame.get_array_element_cache_object_string = true;
                            frame.get_array_element_cache_object_integral = false;
                            frame.get_array_element_cache_container_tv = Some(container_tv);
                            frame.get_array_element_cache_index_tv = Some(index_tv);
                            return Ok(VMStatus::Continue);
                        }
                        None => {
                            let error = ExceptionHandler::runtime_error_with_type(
                                &frames,
                                format!("KeyError: {}", key_material.to_string()),
                                line,
                                ErrorType::KeyError,
                            );
                            return match ExceptionHandler::handle_exception(
                                stack,
                                frames,
                                exception_handlers,
                                error,
                                value_store,
                                heavy_store,
                            ) {
                                Ok(()) => Ok(VMStatus::Continue),
                                Err(e) => Err(e),
                            };
                        }
                        Some(element_id) => {
                            push_stack_value_id(stack, value_store, element_id);
                            frame.get_array_element_cache_ip = Some(current_ip);
                            frame.get_array_element_cache_array_number = false;
                            frame.get_array_element_cache_object_string = true;
                            frame.get_array_element_cache_object_integral = false;
                            frame.get_array_element_cache_container_tv = Some(container_tv);
                            frame.get_array_element_cache_index_tv = Some(index_tv);
                            return Ok(VMStatus::Continue);
                        }
                    }
                }
            } else {
                frame.get_array_element_cache_object_string = false;
                frame.get_array_element_cache_object_integral = false;
                frame.get_array_element_cache_container_tv = None;
                frame.get_array_element_cache_index_tv = None;
            }
        }
    }
    let index_value_id = tagged_to_value_id(index_tv, value_store);
    let container_id = tagged_to_value_id(container_tv, value_store);
    if let Some(ValueCell::Number(n)) = value_store.get(index_value_id) {
        let idx = *n as i64;
        if idx >= 0 {
            let u = idx as usize;
            if let Some(ValueCell::ObjectFieldList {
                element_ids,
                source_object_id,
                projection,
            }) = value_store.get(container_id)
            {
                if u < element_ids.len() {
                    let src = *source_object_id;
                    let proj = *projection;
                    let fallback_id = element_ids[u];
                    let pushed = object_dict::push_object_projection_element(
                        src,
                        proj,
                        u,
                        stack,
                        value_store,
                        heavy_store,
                    );
                    if !pushed {
                        stack::push_id(stack, fallback_id);
                    }
                    let frame = frames.last_mut().unwrap();
                    frame.get_array_element_cache_ip = Some(current_ip);
                    frame.get_array_element_cache_array_number = true;
                    frame.get_array_element_cache_object_string = false;
                    frame.get_array_element_cache_container_tv = Some(container_tv);
                    frame.get_array_element_cache_index_tv = Some(index_tv);
                    return Ok(VMStatus::Continue);
                }
            }
        }
    }
    if let (Some(ValueCell::Array(ref vec)), Some(ValueCell::Number(n))) = (
        value_store.get(container_id),
        value_store.get(index_value_id),
    ) {
        let idx = *n as i64;
        if idx >= 0 {
            let u = idx as usize;
            if u < vec.len() {
                stack::push(stack, vec[u]);
                let frame = frames.last_mut().unwrap();
                frame.get_array_element_cache_ip = Some(current_ip);
                frame.get_array_element_cache_array_number = true;
                frame.get_array_element_cache_object_string = false;
                frame.get_array_element_cache_container_tv = Some(container_tv);
                frame.get_array_element_cache_index_tv = Some(index_tv);
                return Ok(VMStatus::Continue);
            }
        }
    }
    // Plain dict by integral key (including heap `Number` index cells) — never clone the whole map.
    if let Some(key_canonical) = integral_lookup_key_from_index(index_tv, value_store) {
        if let Some(ValueCell::Object(omap)) = value_store.get(container_id) {
            if let Some(slot) = try_plain_dict_integral_element_get(
                omap,
                key_canonical,
                value_store,
                heavy_store,
            ) {
                push_integral_slot(stack, value_store, slot);
                let frame = frames.last_mut().unwrap();
                set_get_element_object_integral_cache(
                    frame,
                    current_ip,
                    container_tv,
                    key_canonical,
                );
                return Ok(VMStatus::Continue);
            }
        }
    }

    // Fast path: [`ValueCell::Object`] bucket map — `.keys` / `.values` or hash/equality lookup.
    // Skip class-like maps for raw lookup (they use `object_fields::get_object`).
    if let Some(proj) =
        object_dict::projection_for_keys_values_property(index_value_id, value_store)
    {
        if let Some(ValueCell::Object(omap)) = value_store.get(container_id).cloned() {
            if !object_map_needs_visibility_checks(&omap, value_store, heavy_store) {
                let list_id =
                    object_dict::allocate_object_field_list(container_id, &omap, proj, value_store);
                stack::push_id(stack, list_id);
                let frame = frames.last_mut().unwrap();
                frame.get_array_element_cache_ip = Some(current_ip);
                frame.get_array_element_cache_array_number = false;
                frame.get_array_element_cache_object_string = true;
                frame.get_array_element_cache_object_integral = false;
                frame.get_array_element_cache_container_tv = Some(container_tv);
                frame.get_array_element_cache_index_tv = Some(index_tv);
                return Ok(VMStatus::Continue);
            }
        }
    } else if let Some(ValueCell::Object(omap)) = value_store.get(container_id) {
        if object_map_use_builtin_get(&omap, value_store, heavy_store)
            && matches!(
                load_value(index_value_id, value_store, heavy_store),
                Value::String(ref s) if s == "get"
            )
        {
            stack::push_id(
                stack,
                store_value(
                    Value::NativeFunction(crate::vm::native_indices::builtin::OBJECT_GET),
                    value_store,
                    heavy_store,
                ),
            );
            let frame = frames.last_mut().unwrap();
            frame.get_array_element_cache_ip = Some(current_ip);
            frame.get_array_element_cache_array_number = false;
            frame.get_array_element_cache_object_string = true;
            frame.get_array_element_cache_object_integral = false;
            frame.get_array_element_cache_container_tv = Some(container_tv);
            frame.get_array_element_cache_index_tv = Some(index_tv);
            return Ok(VMStatus::Continue);
        }
        if let Some(key_canonical) =
            crate::vm::memory::canonical_integral_from_key_id(index_value_id, value_store)
        {
            if let Some(slot) = try_plain_dict_integral_element_get(
                &omap,
                key_canonical,
                value_store,
                heavy_store,
            ) {
                push_integral_slot(stack, value_store, slot);
                let frame = frames.last_mut().unwrap();
                set_get_element_object_integral_cache(
                    frame,
                    current_ip,
                    container_tv,
                    key_canonical,
                );
                return Ok(VMStatus::Continue);
            }
        } else if !object_map_needs_visibility_checks(&omap, value_store, heavy_store) {
            let key_material = load_value(index_value_id, value_store, heavy_store);
            if !crate::common::type_model::is_hashable_value(&key_material) {
                let tn = crate::vm::calls::get_type_name_value(&key_material);
                let error = ExceptionHandler::runtime_error_with_type(
                    &frames,
                    format!("unhashable type: {}", tn),
                    line,
                    ErrorType::TypeError,
                );
                return match ExceptionHandler::handle_exception(
                    stack,
                    frames,
                    exception_handlers,
                    error,
                    value_store,
                    heavy_store,
                ) {
                    Ok(()) => Ok(VMStatus::Continue),
                    Err(e) => Err(e),
                };
            }
            match object_cell_try_lookup_by_key_id(container_id, index_value_id, value_store, heavy_store) {
                None if plain_dict_numeric_missing_key_is_null(&key_material) => {
                    push_stack_value_id(stack, value_store, NULL_VALUE_ID);
                    let frame = frames.last_mut().unwrap();
                    frame.get_array_element_cache_ip = Some(current_ip);
                    frame.get_array_element_cache_array_number = false;
                    frame.get_array_element_cache_object_string = true;
                    frame.get_array_element_cache_object_integral = false;
                    frame.get_array_element_cache_container_tv = Some(container_tv);
                    frame.get_array_element_cache_index_tv = Some(index_tv);
                    return Ok(VMStatus::Continue);
                }
                None => {
                    let error = ExceptionHandler::runtime_error_with_type(
                        &frames,
                        format!("KeyError: {}", key_material.to_string()),
                        line,
                        ErrorType::KeyError,
                    );
                    return match ExceptionHandler::handle_exception(
                        stack,
                        frames,
                        exception_handlers,
                        error,
                        value_store,
                        heavy_store,
                    ) {
                        Ok(()) => Ok(VMStatus::Continue),
                        Err(e) => Err(e),
                    };
                }
                Some(element_id) => {
                    push_stack_value_id(stack, value_store, element_id);
                    let frame = frames.last_mut().unwrap();
                    frame.get_array_element_cache_ip = Some(current_ip);
                    frame.get_array_element_cache_array_number = false;
                    frame.get_array_element_cache_object_string = true;
                    frame.get_array_element_cache_object_integral = false;
                    frame.get_array_element_cache_container_tv = Some(container_tv);
                    frame.get_array_element_cache_index_tv = Some(index_tv);
                    return Ok(VMStatus::Continue);
                }
            }
        }
    }
    let frame = frames.last_mut().unwrap();
    frame.get_array_element_cache_array_number = false;
    frame.get_array_element_cache_object_string = false;
    frame.get_array_element_cache_object_integral = false;
    frame.get_array_element_cache_container_tv = None;
    frame.get_array_element_cache_index_tv = None;
    frame.get_array_element_cache_integral_key = None;

    if let Some(ValueCell::Set(_)) = value_store.get(container_id) {
        if try_store_set_method_get(
            container_id,
            index_value_id,
            stack,
            value_store,
            heavy_store,
        ) {
            return Ok(VMStatus::Continue);
        }
    }
    let key_canonical = integral_lookup_key_from_index(index_tv, value_store).or_else(|| {
        crate::vm::memory::canonical_integral_from_key_id_mut(index_value_id, value_store)
    });
    if let Some(key_canonical) = key_canonical {
        if value_store.is_plain_object(container_id) {
            if let Some(ValueCell::Object(omap)) = value_store.get_mut(container_id) {
                let slot = omap
                    .find_integral_slot(key_canonical)
                    .unwrap_or(crate::common::integral_map::IntegralSlot::Immediate(
                        TaggedValue::null(),
                    ));
                push_integral_slot(stack, value_store, slot);
                let frame = frames.last_mut().unwrap();
                set_get_element_object_integral_cache(
                    frame,
                    current_ip,
                    container_tv,
                    key_canonical,
                );
                return Ok(VMStatus::Continue);
            }
        } else if let Some(ValueCell::Object(omap)) = value_store.get(container_id) {
            if let Some(slot) = try_plain_dict_integral_element_get(
                omap,
                key_canonical,
                value_store,
                heavy_store,
            ) {
                push_integral_slot(stack, value_store, slot);
                let frame = frames.last_mut().unwrap();
                set_get_element_object_integral_cache(
                    frame,
                    current_ip,
                    container_tv,
                    key_canonical,
                );
                return Ok(VMStatus::Continue);
            }
        }
    }
    if let Some(ValueCell::Object(omap)) = value_store.get(container_id) {
        if matches!(
            load_value(index_value_id, value_store, heavy_store),
            Value::String(ref s) if s == "get"
        ) && should_inject_dict_builtin(omap, value_store, heavy_store, "get")
        {
            stack::push_id(
                stack,
                store_value(
                    Value::NativeFunction(crate::vm::native_indices::builtin::OBJECT_GET),
                    value_store,
                    heavy_store,
                ),
            );
            return Ok(VMStatus::Continue);
        }
        if matches!(
            load_value(index_value_id, value_store, heavy_store),
            Value::String(ref s) if s == "clear"
        ) && should_inject_dict_builtin(omap, value_store, heavy_store, "clear")
        {
            stack::push_id(
                stack,
                store_value(
                    Value::NativeFunction(crate::vm::native_indices::builtin::OBJECT_CLEAR),
                    value_store,
                    heavy_store,
                ),
            );
            return Ok(VMStatus::Continue);
        }
        if !object_map_needs_visibility_checks(omap, value_store, heavy_store) {
            if let Some(vid) = object_cell_try_lookup_by_key_id(
                container_id,
                index_value_id,
                value_store,
                heavy_store,
            ) {
                push_stack_value_id(stack, value_store, vid);
                return Ok(VMStatus::Continue);
            }
            let key_material = load_value(index_value_id, value_store, heavy_store);
            if plain_dict_numeric_missing_key_is_null(&key_material) {
                push_stack_value_id(stack, value_store, NULL_VALUE_ID);
                return Ok(VMStatus::Continue);
            }
        }
    }

    let key_canonical = integral_lookup_key_from_index(index_tv, value_store).or_else(|| {
        crate::vm::memory::canonical_integral_from_key_id_mut(index_value_id, value_store)
    });
    if let Some(key_canonical) = key_canonical {
        if let Some(ValueCell::Object(omap)) = value_store.get_mut(container_id) {
            if omap.is_plain() {
                let slot = omap
                    .find_integral_slot(key_canonical)
                    .unwrap_or(crate::common::integral_map::IntegralSlot::Immediate(
                        TaggedValue::null(),
                    ));
                push_integral_slot(stack, value_store, slot);
                return Ok(VMStatus::Continue);
            }
        }
        if let Some(ValueCell::Object(omap)) = value_store.get(container_id) {
            if !object_map_needs_visibility_checks(omap, value_store, heavy_store) {
                if let Some(slot) = try_plain_dict_integral_element_get(
                    omap,
                    key_canonical,
                    value_store,
                    heavy_store,
                ) {
                    push_integral_slot(stack, value_store, slot);
                } else {
                    push_stack_value_id(stack, value_store, NULL_VALUE_ID);
                }
                return Ok(VMStatus::Continue);
            }
        }
    }

    let index_value = load_value(index_value_id, value_store, heavy_store);
    let container = load_value(container_id, value_store, heavy_store);
    if crate::common::debug::is_debug_enabled() {
        let container_type = match &container {
            Value::Array(_) => "Array",
            Value::ArrayView(_) => "ArrayView",
            Value::Enumerate { .. } => "Enumerate",
            Value::Iterable(_) => "Iterable",
            Value::Generator(_) => "Generator",
            Value::Object(_) => "Object",
            Value::Table(_) => "Table",
            Value::Path(_) => "Path",
            Value::Uuid(_, _) => "UUID",
            Value::Date(_) => "Date",
            _ => "Other",
        };
        let key_str = match &index_value {
            Value::String(k) => k.clone(),
            Value::Number(n) => format!("{}", n),
            _ => crate::vm::calls::get_type_name_value(&index_value).to_string(),
        };
        debug_println!(
            "[DEBUG GetArrayElement] line {} IP {}: {} key '{}'",
            line,
            current_ip,
            container_type,
            key_str
        );
    }

    match container {
        Value::ObjectFieldList {
            element_ids,
            source_object_id,
            projection,
        } => {
            return array_ops::get_object_field_list(
                line,
                stack,
                frames,
                exception_handlers,
                value_store,
                heavy_store,
                source_object_id,
                projection,
                &element_ids,
                index_value,
            );
        }
        Value::Array(arr) => {
            return array_ops::get_array(
                line,
                stack,
                frames,
                exception_handlers,
                value_store,
                heavy_store,
                container_id,
                arr,
                index_value,
            );
        }
        Value::ArrayView(av) => {
            return array_ops::get_array_view(
                line,
                stack,
                frames,
                exception_handlers,
                value_store,
                heavy_store,
                container_id,
                &av,
                index_value,
            );
        }
        Value::Tuple(tuple) => {
            return array_ops::get_tuple(
                line,
                stack,
                frames,
                exception_handlers,
                value_store,
                heavy_store,
                tuple,
                index_value,
            );
        }
        Value::Set(_) => {
            if let Value::String(key) = &index_value {
                let method_index = match key.as_str() {
                    "add" => Some(crate::vm::native_indices::builtin::SET_ADD),
                    "remove" => Some(crate::vm::native_indices::builtin::SET_REMOVE),
                    "discard" => Some(crate::vm::native_indices::builtin::SET_DISCARD),
                    "pop" => Some(crate::vm::native_indices::builtin::SET_POP),
                    "clear" => Some(crate::vm::native_indices::builtin::SET_CLEAR),
                    "copy" => Some(crate::vm::native_indices::builtin::SET_COPY),
                    "update" => Some(crate::vm::native_indices::builtin::SET_UPDATE),
                    "contains" => Some(crate::vm::native_indices::builtin::SET_CONTAINS),
                    _ => None,
                };
                if let Some(idx) = method_index {
                    stack::push_id(
                        stack,
                        store_value(
                            Value::NativeFunction(idx),
                            value_store,
                            heavy_store,
                        ),
                    );
                    return Ok(VMStatus::Continue);
                }
                let error = ExceptionHandler::runtime_error(
                    &frames,
                    format!(
                        "Set has no property '{}'. Available: add, remove, discard, pop, clear, copy, update, contains",
                        key
                    ),
                    line,
                );
                return match ExceptionHandler::handle_exception(
                    stack,
                    frames,
                    exception_handlers,
                    error,
                    value_store,
                    heavy_store,
                ) {
                    Ok(()) => Ok(VMStatus::Continue),
                    Err(e) => Err(e),
                };
            }
            let error = ExceptionHandler::runtime_error(
                &frames,
                "Set property access requires string name (e.g. add)".to_string(),
                line,
            );
            return match ExceptionHandler::handle_exception(
                stack,
                frames,
                exception_handlers,
                error,
                value_store,
                heavy_store,
            ) {
                Ok(()) => Ok(VMStatus::Continue),
                Err(e) => Err(e),
            };
        }
        Value::Enumerate { data, start } => {
            return array_ops::get_enumerate(
                line,
                stack,
                frames,
                exception_handlers,
                value_store,
                heavy_store,
                data,
                start,
                index_value,
            );
        }
        Value::ByteBuffer(bb) => {
            if let Value::String(key) = &index_value {
                if key.as_str() == "chunk" {
                    stack::push_id(
                        stack,
                        store_value(
                            Value::NativeFunction(crate::vm::native_indices::builtin::CHUNK),
                            value_store,
                            heavy_store,
                        ),
                    );
                    return Ok(VMStatus::Continue);
                }
                let error = ExceptionHandler::runtime_error(
                    &frames,
                    format!(
                        "ByteBuffer has no property '{}'. Available: chunk, or numeric index",
                        key
                    ),
                    line,
                );
                return match ExceptionHandler::handle_exception(
                    stack,
                    frames,
                    exception_handlers,
                    error,
                    value_store,
                    heavy_store,
                ) {
                    Ok(()) => Ok(VMStatus::Continue),
                    Err(e) => Err(e),
                };
            }
            if let Value::Number(n) = &index_value {
                if n.fract() == 0.0 && *n >= 0.0 {
                    let idx = *n as usize;
                    if idx < bb.len {
                        let byte = bb.bytes[bb.offset + idx];
                        stack::push_id(
                            stack,
                            store_value(Value::Number(byte as f64), value_store, heavy_store),
                        );
                        return Ok(VMStatus::Continue);
                    }
                }
            }
            let error = ExceptionHandler::runtime_error(
                &frames,
                "ByteBuffer index out of range or invalid".to_string(),
                line,
            );
            return match ExceptionHandler::handle_exception(
                stack,
                frames,
                exception_handlers,
                error,
                value_store,
                heavy_store,
            ) {
                Ok(()) => Ok(VMStatus::Continue),
                Err(e) => Err(e),
            };
        }
        Value::Generator(rc) => {
            if let Value::String(key) = &index_value {
                if key.as_str() == "live" {
                    let live = !rc.borrow().finished;
                    stack::push_id(
                        stack,
                        store_value(Value::Bool(live), value_store, heavy_store),
                    );
                    return Ok(VMStatus::Continue);
                }
                if key.as_str() == "final" {
                    stack::push_id(
                        stack,
                        store_value(
                            Value::NativeFunction(crate::vm::native_indices::builtin::GENERATOR_FINAL),
                            value_store,
                            heavy_store,
                        ),
                    );
                    return Ok(VMStatus::Continue);
                }
                if key.as_str() == "next" {
                    stack::push_id(
                        stack,
                        store_value(
                            Value::NativeFunction(crate::vm::native_indices::builtin::GENERATOR_NEXT),
                            value_store,
                            heavy_store,
                        ),
                    );
                    return Ok(VMStatus::Continue);
                }
                if key.as_str() == "send" {
                    stack::push_id(
                        stack,
                        store_value(
                            Value::NativeFunction(crate::vm::native_indices::builtin::GENERATOR_SEND),
                            value_store,
                            heavy_store,
                        ),
                    );
                    return Ok(VMStatus::Continue);
                }
            }
            let error = ExceptionHandler::runtime_error(
                &frames,
                "Generator supports .live, .final(), .next(), .send() (string keys)".to_string(),
                line,
            );
            return match ExceptionHandler::handle_exception(
                stack,
                frames,
                exception_handlers,
                error,
                value_store,
                heavy_store,
            ) {
                Ok(()) => Ok(VMStatus::Continue),
                Err(e) => Err(e),
            };
        }
        Value::Iterable(rc) => {
            let chunk_info = {
                let inner = rc.borrow();
                if let IterableInner::Chunks {
                    source, chunk_size, ..
                } = &*inner
                {
                    if let Value::Number(n) = &index_value {
                        if n.fract() == 0.0 && *n >= 0.0 {
                            let k = *n as usize;
                            if k < chunk_source_count(source, *chunk_size) {
                                Some((source.clone(), *chunk_size, k))
                            } else {
                                None
                            }
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                } else {
                    None
                }
            };
            if let Some((source, chunk_size, k)) = chunk_info {
                let vm = unsafe { &mut *vm_ptr };
                match materialize_chunk_at(vm, &source, chunk_size, k) {
                    Ok(v) => {
                        stack::push_id(stack, store_value(v, value_store, heavy_store));
                        return Ok(VMStatus::Continue);
                    }
                    Err(e) => {
                        let error = ExceptionHandler::runtime_error(&frames, e.to_string(), line);
                        return match ExceptionHandler::handle_exception(
                            stack,
                            frames,
                            exception_handlers,
                            error,
                            value_store,
                            heavy_store,
                        ) {
                            Ok(()) => Ok(VMStatus::Continue),
                            Err(err) => Err(err),
                        };
                    }
                }
            }
            let error = ExceptionHandler::runtime_error(
                &frames,
                "GetArrayElement: only chunk(...) iterables support numeric indexing (chunk index)"
                    .to_string(),
                line,
            );
            return match ExceptionHandler::handle_exception(
                stack,
                frames,
                exception_handlers,
                error,
                value_store,
                heavy_store,
            ) {
                Ok(()) => Ok(VMStatus::Continue),
                Err(e) => Err(e),
            };
        }
        Value::Table(table) => {
            return table_ops::get_table(
                line,
                stack,
                frames,
                exception_handlers,
                value_store,
                heavy_store,
                table,
                index_value,
            );
        }
        Value::Object(map_rc) => {
            return object_fields::get_object(
                line,
                stack,
                frames,
                globals,
                global_names,
                exception_handlers,
                value_store,
                heavy_store,
                vm_ptr,
                container_tv,
                map_rc,
                index_value,
            );
        }
        Value::Figure(figure_rc) => {
            return indexing_lib::get_figure(
                line,
                stack,
                frames,
                exception_handlers,
                value_store,
                heavy_store,
                figure_rc,
                index_value,
            );
        }
        Value::Axis(_axis_rc) => {
            return indexing_lib::get_axis(
                line,
                stack,
                frames,
                globals,
                exception_handlers,
                value_store,
                heavy_store,
                index_value,
            );
        }
        Value::ColumnReference { table, column_name } => {
            if let Value::String(prop) = &index_value {
                if prop == "map" {
                    stack::push_id(
                        stack,
                        store_value(
                            Value::NativeFunction(crate::vm::native_indices::builtin::COLUMN_MAP),
                            value_store,
                            heavy_store,
                        ),
                    );
                    return Ok(VMStatus::Continue);
                }
            }
            return indexing::get_column_reference(
                line,
                stack,
                frames,
                exception_handlers,
                value_store,
                heavy_store,
                table,
                column_name,
                index_value,
            );
        }
        Value::ColumnsReference { .. } => {
            if let Value::String(prop) = &index_value {
                if prop == "map" {
                    stack::push_id(
                        stack,
                        store_value(
                            Value::NativeFunction(crate::vm::native_indices::builtin::COLUMNS_MAP),
                            value_store,
                            heavy_store,
                        ),
                    );
                    return Ok(VMStatus::Continue);
                }
            }
            let error = ExceptionHandler::runtime_error_with_type(
                &frames,
                "TypeError: columns reference does not support cell indexing; use .map(fn)".to_string(),
                line,
                crate::common::error::ErrorType::TypeError,
            );
            return match ExceptionHandler::handle_exception(
                stack,
                frames,
                exception_handlers,
                error,
                value_store,
                heavy_store,
            ) {
                Ok(()) => Ok(VMStatus::Continue),
                Err(e) => Err(e),
            };
        }
        Value::Path(path) => {
            return indexing::get_path(
                line,
                stack,
                frames,
                exception_handlers,
                value_store,
                heavy_store,
                path,
                index_value,
            );
        }
        Value::PluginOpaque { .. } => {
            let Some(native_idx) = (unsafe { (*vm_ptr).plugin_call_native }) else {
                let error = ExceptionHandler::runtime_error(
                    &frames,
                    "GetArrayElement on plugin opaque values requires native_plugin_call (import a native module that exports it)".to_string(),
                    line,
                );
                return match ExceptionHandler::handle_exception(
                    stack,
                    frames,
                    exception_handlers,
                    error,
                    value_store,
                    heavy_store,
                ) {
                    Ok(()) => Ok(VMStatus::Continue),
                    Err(e) => Err(e),
                };
            };
            let builtin_count = natives.len();
            let abi_slice = unsafe { (*vm_ptr).get_abi_natives() };
            if native_idx < builtin_count || native_idx >= builtin_count + abi_slice.len() {
                let error = ExceptionHandler::runtime_error(
                    &frames,
                    "native_plugin_call index is invalid (reload native module)".to_string(),
                    line,
                );
                return match ExceptionHandler::handle_exception(
                    stack,
                    frames,
                    exception_handlers,
                    error,
                    value_store,
                    heavy_store,
                ) {
                    Ok(()) => Ok(VMStatus::Continue),
                    Err(e) => Err(e),
                };
            }
            let args = [container.clone(), index_value.clone()];
            let result = crate::vm::native_loader::call_abi_native(
                abi_slice[native_idx - builtin_count],
                &args,
                Some((value_store, heavy_store)),
            );
            if let Some(abi_err) = crate::vm::native_loader::take_last_abi_error() {
                return match ExceptionHandler::handle_exception(
                    stack,
                    frames,
                    exception_handlers,
                    abi_err,
                    value_store,
                    heavy_store,
                ) {
                    Ok(()) => Ok(VMStatus::Continue),
                    Err(e) => Err(e),
                };
            }
            stack::push_id(stack, store_value(result, value_store, heavy_store));
            return Ok(VMStatus::Continue);
        }
        Value::DatabaseEngine(_engine_rc) => {
            return indexing_lib::get_database_engine(
                line,
                stack,
                frames,
                exception_handlers,
                value_store,
                heavy_store,
                natives,
                index_value,
            );
        }
        Value::DatabaseCluster(cluster_rc) => {
            return indexing_lib::get_database_cluster(
                line,
                stack,
                frames,
                exception_handlers,
                value_store,
                heavy_store,
                natives,
                cluster_rc,
                index_value,
            );
        }
        Value::Archive(archive_rc) => {
            return indexing_lib::get_archive(
                line,
                stack,
                frames,
                exception_handlers,
                value_store,
                heavy_store,
                archive_rc,
                index_value,
            );
        }
        Value::DataSource(ds_rc) => {
            return indexing_lib::get_datasource(
                line,
                stack,
                frames,
                exception_handlers,
                value_store,
                heavy_store,
                ds_rc,
                index_value,
            );
        }
        Value::DataSourceResponse(resp_rc) => {
            return indexing_lib::get_datasource_response(
                line,
                stack,
                frames,
                exception_handlers,
                value_store,
                heavy_store,
                resp_rc,
                index_value,
            );
        }
        Value::HttpResponse(resp_rc) => {
            return indexing_lib::get_http_response(
                line,
                stack,
                frames,
                exception_handlers,
                value_store,
                heavy_store,
                resp_rc,
                index_value,
            );
        }
        Value::WebPage(page_rc) => {
            return indexing_lib::get_web_page(
                line,
                stack,
                frames,
                exception_handlers,
                value_store,
                heavy_store,
                natives,
                page_rc,
                index_value,
            );
        }
        Value::WebElement(el_rc) => {
            return indexing_lib::get_web_element(
                line,
                stack,
                frames,
                exception_handlers,
                value_store,
                heavy_store,
                natives,
                el_rc,
                index_value,
            );
        }
        Value::String(s) => {
            return indexing::get_string(
                line,
                stack,
                frames,
                exception_handlers,
                value_store,
                heavy_store,
                s,
                index_value,
            );
        }
        Value::Date(d) => {
            return indexing::get_date(
                line,
                stack,
                frames,
                exception_handlers,
                value_store,
                heavy_store,
                d,
                index_value,
            );
        }
        Value::NativeFunction(native_index) => {
            use crate::vm::natives::basic::native_str;
            if native_index < natives.len()
                && (natives[native_index].as_fn_ptr() == Some(native_str as *const ())
                    || native_index == crate::vm::native_indices::builtin::STR)
            {
                return indexing::get_native_str(
                    line,
                    stack,
                    frames,
                    exception_handlers,
                    value_store,
                    heavy_store,
                    index_value,
                );
            }
            let error = ExceptionHandler::runtime_error(
                &frames,
                "Expected array, tuple, column reference, table, object, path, database engine, or database cluster for GetArrayElement".to_string(),
                line,
            );
            match ExceptionHandler::handle_exception(
                stack,
                frames,
                exception_handlers,
                error,
                value_store,
                heavy_store,
            ) {
                Ok(()) => return Ok(VMStatus::Continue),
                Err(e) => return Err(e),
            }
        }
        Value::Null => {
            let error = ExceptionHandler::runtime_error(
                &frames,
                "Cannot access element of null value".to_string(),
                line,
            );
            match ExceptionHandler::handle_exception(
                stack,
                frames,
                exception_handlers,
                error,
                value_store,
                heavy_store,
            ) {
                Ok(()) => return Ok(VMStatus::Continue),
                Err(e) => return Err(e),
            }
        }
        _ => {
            let error = ExceptionHandler::runtime_error(
            &frames,
                "Expected array, tuple, column reference, table, object, path, database engine, or database cluster for GetArrayElement".to_string(),
                line,
            );
            match ExceptionHandler::handle_exception(
                stack,
                frames,
                exception_handlers,
                error,
                value_store,
                heavy_store,
            ) {
                Ok(()) => return Ok(VMStatus::Continue),
                Err(e) => return Err(e),
            }
        }
    }
}

fn slice_bound_from_tagged(
    tv: TaggedValue,
    line: usize,
    value_store: &mut ValueStore,
    heavy_store: &mut HeavyStore,
) -> Result<Option<i64>, LangError> {
    let v = load_value(
        tagged_to_value_id(tv, value_store),
        value_store,
        heavy_store,
    );
    if matches!(v, Value::Null) {
        return Ok(None);
    }
    if let Some(n) = crate::common::numeric::integer_value_as_i64_if_whole(&v) {
        return Ok(Some(n));
    }
    if let Value::Number(n) = &v {
        if n.fract() != 0.0 && (n - n.round()).abs() > 1e-9 {
            return Err(LangError::runtime_error(
                "Slice bound must be an integer or null".to_string(),
                line,
            ));
        }
    }
    Err(LangError::runtime_error(
        "Slice bound must be a number or null".to_string(),
        line,
    ))
}

#[allow(clippy::too_many_arguments)]
pub fn op_get_array_slice(
    line: usize,
    stack: &mut Vec<TaggedValue>,
    frames: &mut Vec<CallFrame>,
    exception_handlers: &mut Vec<ExceptionHandler>,
    value_store: &mut ValueStore,
    heavy_store: &mut HeavyStore,
) -> Result<VMStatus, LangError> {
    let step_tv = stack::pop(stack, frames, exception_handlers, value_store, heavy_store)?;
    let stop_tv = stack::pop(stack, frames, exception_handlers, value_store, heavy_store)?;
    let start_tv = stack::pop(stack, frames, exception_handlers, value_store, heavy_store)?;
    let container_id =
        pop_to_value_id(stack, frames, exception_handlers, value_store, heavy_store)?;
    let start = slice_bound_from_tagged(start_tv, line, value_store, heavy_store)?;
    let stop = slice_bound_from_tagged(stop_tv, line, value_store, heavy_store)?;
    let step = slice_bound_from_tagged(step_tv, line, value_store, heavy_store)?;
    let container_v = load_value(container_id, value_store, heavy_store);
    if let Value::ByteBuffer(ref bb) = container_v {
        match array_ops::byte_buffer_slice_value(bb, start, stop, step) {
            Ok(v) => {
                stack::push_id(stack, store_value(v, value_store, heavy_store));
                return Ok(VMStatus::Continue);
            }
            Err(msg) => {
                let error = ExceptionHandler::runtime_error(frames, msg, line);
                return match ExceptionHandler::handle_exception(
                    stack,
                    frames,
                    exception_handlers,
                    error,
                    value_store,
                    heavy_store,
                ) {
                    Ok(()) => Ok(VMStatus::Continue),
                    Err(e) => Err(e),
                };
            }
        }
    }
    if let Value::String(ref s) = container_v {
        match indexing::string_slice_value(s, start, stop, step) {
            Ok(v) => {
                stack::push_id(stack, store_value(v, value_store, heavy_store));
                return Ok(VMStatus::Continue);
            }
            Err(msg) => {
                let error = ExceptionHandler::runtime_error(frames, msg, line);
                return match ExceptionHandler::handle_exception(
                    stack,
                    frames,
                    exception_handlers,
                    error,
                    value_store,
                    heavy_store,
                ) {
                    Ok(()) => Ok(VMStatus::Continue),
                    Err(e) => Err(e),
                };
            }
        }
    }
    if resolve_slice_origin(container_id, value_store, heavy_store).is_some() {
        return array_ops::get_array_slice_from_container(
            line,
            stack,
            frames,
            exception_handlers,
            value_store,
            heavy_store,
            container_id,
            start,
            stop,
            step,
        );
    }
    let error = ExceptionHandler::runtime_error(
        frames,
        "GetArraySlice requires an array or string".to_string(),
        line,
    );
    match ExceptionHandler::handle_exception(
        stack,
        frames,
        exception_handlers,
        error,
        value_store,
        heavy_store,
    ) {
        Ok(()) => Ok(VMStatus::Continue),
        Err(e) => Err(e),
    }
}

#[allow(clippy::too_many_arguments)]
pub fn op_set_array_slice(
    line: usize,
    stack: &mut Vec<TaggedValue>,
    frames: &mut Vec<CallFrame>,
    exception_handlers: &mut Vec<ExceptionHandler>,
    value_store: &mut ValueStore,
    heavy_store: &mut HeavyStore,
) -> Result<VMStatus, LangError> {
    let container_id =
        pop_to_value_id(stack, frames, exception_handlers, value_store, heavy_store)?;
    let step_tv = stack::pop(stack, frames, exception_handlers, value_store, heavy_store)?;
    let stop_tv = stack::pop(stack, frames, exception_handlers, value_store, heavy_store)?;
    let start_tv = stack::pop(stack, frames, exception_handlers, value_store, heavy_store)?;
    let value_tv = stack::pop(stack, frames, exception_handlers, value_store, heavy_store)?;
    let mut value_id = tagged_to_value_id(value_tv, value_store);
    value_id = promote_value_id_if_escapes(container_id, value_id, value_store);
    let value = load_value(value_id, value_store, heavy_store);
    let start = slice_bound_from_tagged(start_tv, line, value_store, heavy_store)?;
    let stop = slice_bound_from_tagged(stop_tv, line, value_store, heavy_store)?;
    let step = slice_bound_from_tagged(step_tv, line, value_store, heavy_store)?;
    array_ops::set_array_slice_splice(
        line,
        stack,
        frames,
        exception_handlers,
        value_store,
        heavy_store,
        container_id,
        start,
        stop,
        step,
        value,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn op_set_array_element(
    line: usize,
    stack: &mut Vec<TaggedValue>,
    frames: &mut Vec<CallFrame>,
    globals: &mut Vec<crate::vm::global_slot::GlobalSlot>,
    global_names: &std::collections::BTreeMap<usize, String>,
    functions: &[crate::bytecode::Function],
    _natives: &[crate::vm::host::HostEntry],
    exception_handlers: &mut Vec<ExceptionHandler>,
    value_store: &mut ValueStore,
    heavy_store: &mut HeavyStore,
) -> Result<VMStatus, LangError> {
    // Stack order from compiler: [value, index, container] with container on top.
    let container_id =
        pop_to_value_id(stack, frames, exception_handlers, value_store, heavy_store)?;
    if let Some(ValueCell::ObjectFieldList { projection, .. }) = value_store.get(container_id) {
        let msg = match projection {
            ObjectProjectionKind::Keys => "cannot modify read-only object keys view",
            ObjectProjectionKind::Values => "cannot modify read-only object values view",
        };
        let error = ExceptionHandler::runtime_error(&frames, msg.to_string(), line);
        return match ExceptionHandler::handle_exception(
            stack,
            frames,
            exception_handlers,
            error,
            value_store,
            heavy_store,
        ) {
            Ok(()) => Ok(VMStatus::Continue),
            Err(e) => Err(e),
        };
    }
    let index_tv = stack::pop(stack, frames, exception_handlers, value_store, heavy_store)?;
    let mut value_tv = stack::pop(stack, frames, exception_handlers, value_store, heavy_store)?;
    value_tv = prepare_value_for_container_store(container_id, value_tv, value_store);
    let mut value_id = if value_tv.is_bool() {
        tagged_to_value_id_arena(value_tv, value_store)
    } else {
        tagged_to_value_id(value_tv, value_store)
    };
    value_id = promote_value_id_if_escapes(container_id, value_id, value_store);
    // Fast path: ValueCell::Array + number index — store TaggedValue slot
    if index_tv.is_number() {
        let idx = index_tv.get_f64() as i64;
        if idx >= 0 && index_tv.get_f64().fract() == 0.0 {
            let u = idx as usize;
            if let Some(ValueCell::Array(slots)) = value_store.get_mut(container_id) {
                if u >= slots.len() {
                    slots.resize(u + 1, TaggedValue::null());
                }
                slots[u] = value_tv;
                stack::push_id(stack, container_id);
                return Ok(VMStatus::Continue);
            }
        }
    }
    let key_canonical = integral_lookup_key_from_index(index_tv, value_store);
    if let Some(canonical) = key_canonical {
        if canonical >= 0 {
            let u = canonical as usize;
            if let Some(ValueCell::Array(slots)) = value_store.get_mut(container_id) {
                if u >= slots.len() {
                    slots.resize(u + 1, TaggedValue::null());
                }
                slots[u] = value_tv;
                stack::push_id(stack, container_id);
                return Ok(VMStatus::Continue);
            }
        }
    }
    let index_key_id = if key_canonical.is_some() {
        NULL_VALUE_ID
    } else {
        tagged_to_value_id(index_tv, value_store)
    };

    let plain_object_fast = match value_store.get(container_id) {
        Some(ValueCell::Object(omap)) => {
            !object_map_needs_visibility_checks(omap, value_store, heavy_store)
        }
        _ => false,
    };

    if let Some(canonical) = key_canonical {
        if plain_object_fast {
            let immediate_score = !value_tv.is_heap()
                && (value_tv.is_number() || value_tv.is_int() || value_tv.is_null());
            let existing_heap_id = value_store.get(container_id).and_then(|c| match c {
                ValueCell::Object(omap) => omap.find_integral_slot(canonical).and_then(|slot| {
                    match slot {
                        crate::common::integral_map::IntegralSlot::Heap(id) => Some(id),
                        _ => None,
                    }
                }),
                _ => None,
            });
            let wrote_in_place = value_store
                .get_mut(container_id)
                .and_then(|c| match c {
                    ValueCell::Object(omap) => {
                        omap.try_write_integral_immediate(canonical, value_tv)
                            .then_some(())
                    }
                    _ => None,
                })
                .is_some()
                || existing_heap_id.is_some_and(|existing| {
                    existing != NULL_VALUE_ID
                        && try_write_cell_from_tagged(existing, value_tv, value_store)
                });
            if wrote_in_place {
                stack::push_id(stack, container_id);
                return Ok(VMStatus::Continue);
            }
            if immediate_score {
                value_store.plain_object_upsert_integral_tagged(
                    container_id,
                    canonical,
                    index_key_id,
                    value_tv,
                );
            } else {
                let value_id = promote_value_id_if_escapes(
                    container_id,
                    tagged_to_mutable_value_id(value_tv, value_store),
                    value_store,
                );
                value_store.plain_object_upsert_integral(
                    container_id,
                    canonical,
                    index_key_id,
                    value_id,
                );
            }
            stack::push_id(stack, container_id);
            return Ok(VMStatus::Continue);
        }
        if matches!(value_store.get(container_id), Some(ValueCell::Set(_))) {
            value_store.plain_set_insert_integral(container_id, canonical, index_key_id);
            stack::push_id(stack, container_id);
            return Ok(VMStatus::Continue);
        }
    }

    let index_value = load_value(index_key_id, value_store, heavy_store);

    if let Some(ValueCell::Set(_)) = value_store.get(container_id) {
        if !crate::common::type_model::is_hashable_value(&index_value) {
            let tn = crate::vm::calls::get_type_name_value(&index_value);
            let error = ExceptionHandler::runtime_error(
                &frames,
                format!("unhashable type: {}", tn),
                line,
            );
            return match ExceptionHandler::handle_exception(
                stack,
                frames,
                exception_handlers,
                error,
                value_store,
                heavy_store,
            ) {
                Ok(()) => Ok(VMStatus::Continue),
                Err(e) => Err(e),
            };
        }
        if let Some(canonical) = crate::common::numeric::integer_value_as_i64_if_whole(&index_value)
        {
            value_store.plain_set_insert_integral(container_id, canonical, index_key_id);
            stack::push_id(stack, container_id);
            return Ok(VMStatus::Continue);
        }
        let h = crate::common::type_model::object_key_hash_value(&index_value).expect("hashable");
        let elem_for_eq = index_value.clone();
        let already = value_store
            .get(container_id)
            .and_then(|c| match c {
                ValueCell::Set(smap) => Some(smap.contains(
                    h,
                    |id| load_value(id, value_store, heavy_store) == elem_for_eq,
                )),
                _ => None,
            })
            .unwrap_or(false);
        if !already {
            if let Some(ValueCell::Set(smap)) = value_store.get_mut(container_id) {
                smap.insert(h, index_key_id, |_| false, None);
            }
        }
        stack::push_id(stack, container_id);
        return Ok(VMStatus::Continue);
    }

    if plain_object_fast {
        if !crate::common::type_model::is_hashable_value(&index_value) {
            let tn = crate::vm::calls::get_type_name_value(&index_value);
            let error = ExceptionHandler::runtime_error(
                &frames,
                format!("unhashable type: {}", tn),
                line,
            );
            return match ExceptionHandler::handle_exception(
                stack,
                frames,
                exception_handlers,
                error,
                value_store,
                heavy_store,
            ) {
                Ok(()) => Ok(VMStatus::Continue),
                Err(e) => Err(e),
            };
        }
        let new_vid = promote_value_id_if_escapes(
            container_id,
            tagged_to_mutable_value_id(value_tv, value_store),
            value_store,
        );
        let existing =
            object_cell_try_lookup_by_key_id(container_id, index_key_id, value_store, heavy_store);
        let value_id = promote_value_id_if_escapes(
            container_id,
            match existing {
                None | Some(NULL_VALUE_ID) => new_vid,
                Some(existing) if existing == new_vid => existing,
                Some(existing) if try_write_cell_from_tagged(existing, value_tv, value_store) => {
                    existing
                }
                Some(_) => new_vid,
            },
            value_store,
        );
        object_map_upsert_in_place(
            value_store,
            container_id,
            heavy_store,
            &index_value,
            index_key_id,
            value_id,
        );
        stack::push_id(stack, container_id);
        return Ok(VMStatus::Continue);
    }

    let container = load_value(container_id, value_store, heavy_store);
    let value = load_value(value_id, value_store, heavy_store);
    let (value, value_id) = if let Value::ArrayView(av) = &value {
        let mat = crate::vm::array_view::materialize_array_view(av, value_store, heavy_store);
        let id = store_value(mat.clone(), value_store, heavy_store);
        (mat, id)
    } else {
        (value, value_id)
    };

    if crate::common::debug::is_debug_enabled() {
        let container_type = match &container {
            Value::Array(_) => "Array",
            Value::Object(_) => "Object",
            Value::Table(_) => "Table",
            _ => "Other",
        };
        let key_str = match &index_value {
            Value::String(k) => k.clone(),
            Value::Number(n) => format!("{}", n),
            _ => crate::vm::calls::get_type_name_value(&index_value).to_string(),
        };
        let value_type_str = match &value {
            Value::Function(fn_idx) => {
                if *fn_idx < functions.len() {
                    format!("Function({}, имя: '{}')", fn_idx, functions[*fn_idx].name)
                } else {
                    format!("Function({}, OUT OF BOUNDS!)", fn_idx)
                }
            }
            Value::NativeFunction(_) => "NativeFunction".to_string(),
            Value::Object(_) => "Object".to_string(),
            other => crate::vm::calls::get_type_name_value(other).to_string(),
        };
        debug_println!(
            "[DEBUG SetArrayElement] line {}, {} key='{}' value={}",
            line,
            container_type,
            key_str,
            value_type_str
        );
    }

    match container {
        Value::Array(_) => {
            return array_ops::set_array(
                line,
                stack,
                frames,
                exception_handlers,
                value_store,
                heavy_store,
                container_id,
                index_value,
                value,
            );
        }
        Value::Object(obj_rc) => {
            let key = match index_value {
                Value::String(key) => key,
                _ => {
                    let instance = load_value(container_id, value_store, heavy_store);
                    if crate::vm::special_methods::is_class_instance(&instance)
                        && crate::vm::special_methods::class_has_special(&instance, "@set")
                        && !object_fields::in_user_constructor(frames)
                        && !object_fields::in_special_method(frames, "set")
                    {
                        let index_id =
                            store_value(index_value.clone(), value_store, heavy_store);
                        if let Ok(Some(_)) = crate::vm::special_methods::dispatch_special_by_id(
                            container_id,
                            "@set",
                            &[index_id, value_id],
                        ) {
                            stack::push_id(stack, container_id);
                            return Ok(VMStatus::Continue);
                        }
                    }
                    let error = ExceptionHandler::runtime_error(
                        &frames,
                        "Object key must be a string".to_string(),
                        line,
                    );
                    match ExceptionHandler::handle_exception(
                        stack,
                        frames,
                        exception_handlers,
                        error,
                        value_store,
                        heavy_store,
                    ) {
                        Ok(()) => return Ok(VMStatus::Continue),
                        Err(e) => return Err(e),
                    }
                }
            };
            return object_fields::set_object(
                line,
                stack,
                frames,
                globals,
                global_names,
                exception_handlers,
                value_store,
                heavy_store,
                container_id,
                obj_rc,
                key,
                value_id,
                value,
            );
        }
        _ => {
            let error = ExceptionHandler::runtime_error(
                &frames,
                format!(
                    "SetArrayElement only supports arrays and objects, got: {:?}",
                    container
                ),
                line,
            );
            match ExceptionHandler::handle_exception(
                stack,
                frames,
                exception_handlers,
                error,
                value_store,
                heavy_store,
            ) {
                Ok(()) => return Ok(VMStatus::Continue),
                Err(e) => return Err(e),
            }
        }
    }
}
