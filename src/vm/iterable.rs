//! Lazy iterable pipeline (`map` / `filter` / `reduce`): one pass, no intermediate arrays.

use crate::common::error::LangError;
use crate::common::table::TableData;
use crate::common::value::{CallableSlot, ChunkSource, IterableInner, Value};
use std::collections::HashMap;
use crate::common::value_store::{ValueId, ValueStore};
use crate::vm::array_view::{materialize_array_view, subview, view_get_element};
use crate::vm::generator::run_generator_next;
use crate::vm::heavy_store::HeavyStore;
use crate::vm::natives::utils::call_user_function;
use crate::vm::store_convert::{load_value, store_value};
use crate::vm::set_ops::set_member_key_ids;
use crate::vm::vm::{current_vm_ptr, Vm};
use std::cell::RefCell;
use std::rc::Rc;

fn runtime(line: usize, msg: impl Into<String>) -> LangError {
    LangError::runtime_error(msg.into(), line)
}

/// Prepare `for x in v` without resetting an existing lazy iterator (Python-style: same `it` stays exhausted).
/// Arrays / tuples / enumerate still get a fresh iterator wrapper with index 0.
pub fn prepare_for_in_iterable(v: Value) -> Result<Value, LangError> {
    match v {
        Value::Iterable(rc) => Ok(Value::Iterable(Rc::clone(&rc))),
        _ => coerce_to_iterable_value_from_id(None, v),
    }
}

/// Like [`prepare_for_in_iterable`], but preserves the canonical instance [`ValueId`] for `@iter`/`@next`.
pub fn prepare_for_in_iterable_from_id(source_id: ValueId, v: Value) -> Result<Value, LangError> {
    match v {
        Value::Iterable(rc) => Ok(Value::Iterable(Rc::clone(&rc))),
        _ => coerce_to_iterable_value_from_id(Some(source_id), v),
    }
}

fn build_special_instance_iterable(
    receiver_id: ValueId,
    instance: &Value,
) -> Result<Value, LangError> {
    if crate::vm::special_methods::class_has_special(instance, "@iter") {
        crate::vm::special_methods::dispatch_special_by_id(receiver_id, "@iter", &[])?;
    }
    Ok(Value::Iterable(Rc::new(RefCell::new(
        IterableInner::SpecialInstance { receiver_id },
    ))))
}

/// Wrap `Value` as [`Value::Iterable`] with iterator state at the start (for `for`-`in` and `reduce`).
pub fn coerce_to_iterable_value(v: Value) -> Result<Value, LangError> {
    coerce_to_iterable_value_from_id(None, v)
}

/// Wrap `Value` as [`Value::Iterable`]; when `source_id` is known, class `@next` mutates the canonical instance.
pub fn coerce_to_iterable_value_from_id(
    source_id: Option<ValueId>,
    v: Value,
) -> Result<Value, LangError> {
    match v {
        Value::Iterable(rc) => Ok(Value::Iterable(Rc::new(RefCell::new(rc.borrow().clone())))),
        Value::Array(a) => Ok(Value::Iterable(Rc::new(RefCell::new(IterableInner::Array {
            array: a,
            index: 0,
        })))),
        Value::ArrayView(av) => Ok(Value::Iterable(Rc::new(RefCell::new(IterableInner::ArrayView {
            view: av,
            index: 0,
        })))),
        Value::Tuple(t) => Ok(Value::Iterable(Rc::new(RefCell::new(IterableInner::Array {
            array: Rc::new(RefCell::new(t.borrow().clone())),
            index: 0,
        })))),
        Value::Enumerate { data, start } => Ok(Value::Iterable(Rc::new(RefCell::new(
            IterableInner::Enumerate {
                data,
                start,
                index: 0,
            },
        )))),
        Value::Generator(g) => Ok(Value::Iterable(Rc::new(RefCell::new(
            IterableInner::StreamGenerator {
                state: Rc::clone(&g),
            },
        )))),
        Value::Table(t) => Ok(Value::Iterable(Rc::new(RefCell::new(
            IterableInner::TableRows { table: t, index: 0 },
        )))),
        Value::ObjectFieldList { element_ids, .. } => Ok(Value::Iterable(Rc::new(RefCell::new(
            IterableInner::ObjectFieldList {
                element_ids: Rc::clone(&element_ids),
                index: 0,
            },
        )))),
        Value::Set(rc) => {
            let map = rc.borrow();
            let Some(vm_ptr) = current_vm_ptr() else {
                return Err(runtime(0, "internal: VM unavailable for set iteration"));
            };
            let ids = unsafe {
                (*vm_ptr).with_stores_mut(|store, _heap| set_member_key_ids(&map, store))
            };
            let gen = map.generation();
            Ok(Value::Iterable(Rc::new(RefCell::new(IterableInner::Set {
                set: Rc::clone(&rc),
                element_ids: Rc::new(ids),
                index: 0,
                start_generation: gen,
            }))))
        }
        Value::String(s) => Ok(Value::Iterable(Rc::new(RefCell::new(IterableInner::String {
            text: s,
            index: 0,
        })))),
        other if crate::vm::special_methods::is_class_instance(&other)
            && crate::vm::special_methods::class_has_special(&other, "@iter")
            && crate::vm::special_methods::class_has_special(&other, "@next") =>
        {
            let receiver_id = if let Some(id) = source_id {
                id
            } else {
                let Some(vm_ptr) = current_vm_ptr() else {
                    return Err(runtime(0, "internal: VM unavailable for class instance iteration"));
                };
                unsafe {
                    (*vm_ptr).with_stores_mut(|store, heap| store_value(other.clone(), store, heap))
                }
            };
            build_special_instance_iterable(receiver_id, &other)
        }
        other => Err(runtime(
            0,
            format!(
                "for-in / iterable: expected array, array view, tuple, set, string, enumerate, iterable, or generator, got {}",
                crate::vm::calls::get_type_name_value(&other)
            ),
        )),
    }
}

/// Build a fresh iterator state from a collection or nested iterable.
pub fn iterable_from_value(coll: &Value) -> Result<Rc<RefCell<IterableInner>>, LangError> {
    match coerce_to_iterable_value(coll.clone())? {
        Value::Iterable(rc) => Ok(rc),
        _ => Err(runtime(0, "internal: coerce did not produce iterable")),
    }
}

/// Membership `needle in rhs` when [`coerce_to_iterable_value`] accepts `rhs` (arrays, iterable views, etc.).
///
/// Returns [`Ok(Some(true/false))`] when iteration finishes (finite source or match found early).
/// Returns [`Ok(None)]` when `rhs` cannot be wrapped as iterable — callers report `TypeError`.
/// Returns [`Err`] if advancing the iterator fails (user callbacks in lazy `map` / `filter`, etc.).
///
/// Uses a fresh coerced iterator (see [`coerce_to_iterable_value`]) so the user's standalone
/// iterable handle is not advanced. Non‑terminating iterators never return [`Some(false)`].
pub fn membership_via_iterable(
    vm: &mut Vm,
    rhs: Value,
    needle: &Value,
) -> Result<Option<bool>, LangError> {
    let coerced = match coerce_to_iterable_value(rhs) {
        Ok(v) => v,
        Err(_) => return Ok(None),
    };
    let Value::Iterable(rc) = coerced else {
        return Err(runtime(
            0,
            "internal: membership_via_iterable expected Iterable after coerce",
        ));
    };

    loop {
        let next = {
            let mut inner = rc.borrow_mut();
            iterable_next(&mut *inner, vm)?
        };
        match next {
            None => return Ok(Some(false)),
            Some(el) if el == *needle => return Ok(Some(true)),
            Some(_) => {}
        }
    }
}

pub fn value_to_callable_slot(f: &Value, vm: &Vm) -> Result<(CallableSlot, u8), LangError> {
    match f {
        Value::Function(fn_idx) => {
            let arity = vm
                .get_functions()
                .get(*fn_idx)
                .map(|fun| fun.arity)
                .unwrap_or(0);
            Ok((CallableSlot::UserFunction(*fn_idx), arity as u8))
        }
        Value::NativeFunction(nidx) => Ok((CallableSlot::NativeFunction(*nidx), 1)),
        Value::ModuleFunction { .. } => Err(runtime(
            0,
            "module functions as callbacks are not supported in this context",
        )),
        _ => Err(runtime(0, "expected a function as callback")),
    }
}

/// Invoke a callable slot (user/native) from VM-native code (`map`-style callbacks).
pub(crate) fn dispatch_callable(slot: &CallableSlot, args: &[Value], vm: &mut Vm) -> Result<Value, LangError> {
    match slot {
        CallableSlot::UserFunction(fn_idx) => call_user_function(*fn_idx, args),
        CallableSlot::NativeFunction(nidx) => {
            if *nidx >= vm.get_natives().len() {
                return Err(runtime(0, "invalid native function index"));
            }
            vm.get_natives()[*nidx].invoke(args)
        }
    }
}

/// Upper bound on how many elements `array(iterable)` will produce when the iterable is finite.
/// `None` if unknown (e.g. [`IterableInner::Filter`] may drop elements).
pub fn iterable_materialize_capacity_hint(inner: &IterableInner) -> Option<usize> {
    match inner {
        IterableInner::Array { array, .. } => Some(array.borrow().len()),
        IterableInner::ArrayView { view, .. } => Some(view.length),
        IterableInner::Map { source, .. } => iterable_materialize_capacity_hint(&source.borrow()),
        IterableInner::Filter { .. } => None,
        IterableInner::Enumerate { data, .. } => Some(data.borrow().len()),
        IterableInner::TableRows { table, .. } => Some(table.borrow().len()),
        IterableInner::ObjectFieldList { element_ids, .. } => Some(element_ids.len()),
        IterableInner::EnumerateIter { source, .. } => {
            iterable_materialize_capacity_hint(&source.borrow())
        }
        IterableInner::Chunks {
            source, chunk_size, ..
        } => Some(chunk_source_count(source, *chunk_size)),
        IterableInner::StreamGenerator { .. } => None,
        IterableInner::Set { element_ids, .. } => Some(element_ids.len()),
        IterableInner::String { text, .. } => Some(text.chars().count()),
        IterableInner::Range {
            current,
            end,
            step,
        } => Some(crate::common::range_args::range_len(*current, *end, *step)),
        IterableInner::SpecialInstance { .. } => None,
    }
}

pub fn chunk_source_len(source: &ChunkSource) -> usize {
    match source {
        ChunkSource::Array(a) => a.borrow().len(),
        ChunkSource::ArrayView(av) => av.length,
        ChunkSource::Bytes(bb) => bb.len,
    }
}

pub fn chunk_source_count(source: &ChunkSource, chunk_size: usize) -> usize {
    if chunk_size == 0 {
        return 0;
    }
    let n = chunk_source_len(source);
    if n == 0 {
        0
    } else {
        n.div_ceil(chunk_size)
    }
}

/// Materialize the `k`-th chunk (0-based) without advancing iterator state.
pub fn materialize_chunk_at(
    vm: &mut Vm,
    source: &ChunkSource,
    chunk_size: usize,
    k: usize,
) -> Result<Value, LangError> {
    match source {
        ChunkSource::Array(a) => {
            let borrow = a.borrow();
            let len = borrow.len();
            let start = k.saturating_mul(chunk_size);
            if start >= len {
                return Err(runtime(0, "chunk index out of range"));
            }
            let end = (start + chunk_size).min(len);
            let chunk: Vec<Value> = borrow[start..end].to_vec();
            Ok(Value::Array(Rc::new(RefCell::new(chunk))))
        }
        ChunkSource::ArrayView(av) => {
            let len = av.length;
            let start = k.saturating_mul(chunk_size);
            if start >= len {
                return Err(runtime(0, "chunk index out of range"));
            }
            let chunk_len = chunk_size.min(len - start);
            let sub = subview(av, start, chunk_len)
                .ok_or_else(|| runtime(0, "chunk subview out of range"))?;
            let store = vm.value_store();
            let heap = vm.heavy_store();
            Ok(materialize_array_view(&sub, store, heap))
        }
        ChunkSource::Bytes(bb) => {
            let start = k.saturating_mul(chunk_size);
            if start >= bb.len {
                return Err(runtime(0, "chunk index out of range"));
            }
            let end = (start + chunk_size).min(bb.len);
            bb.slice_range(start, end)
                .map(Value::ByteBuffer)
                .ok_or_else(|| runtime(0, "chunk byte slice out of range"))
        }
    }
}

/// Pull the next element from a lazy iterator (single pass).
pub fn iterable_next(inner: &mut IterableInner, vm: &mut Vm) -> Result<Option<Value>, LangError> {
    match inner {
        IterableInner::Array { array, index } => {
            let borrow = array.borrow();
            if *index >= borrow.len() {
                return Ok(None);
            }
            let v = borrow[*index].clone();
            *index += 1;
            Ok(Some(v))
        }
        IterableInner::ArrayView { view, index } => {
            if *index >= view.length {
                return Ok(None);
            }
            let v = {
                let store = vm.value_store();
                let heap = vm.heavy_store();
                view_get_element(view, *index, store, heap).unwrap_or(Value::Null)
            };
            *index += 1;
            Ok(Some(v))
        }
        IterableInner::Map {
            source,
            func,
            fn_arity,
            index,
        } => {
            let next = {
                let mut src = source.borrow_mut();
                iterable_next(&mut *src, vm)?
            };
            let Some(val) = next else {
                return Ok(None);
            };
            let i = *index;
            *index += 1;
            let mapped = if *fn_arity == 2 {
                dispatch_callable(func, &[val, Value::Number(i as f64)], vm)?
            } else {
                dispatch_callable(func, &[val], vm)?
            };
            Ok(Some(mapped))
        }
        IterableInner::Filter {
            source,
            pred,
            fn_arity,
            index,
        } => loop {
            let next = {
                let mut src = source.borrow_mut();
                iterable_next(&mut *src, vm)?
            };
            let Some(val) = next else {
                return Ok(None);
            };
            let i = *index;
            *index += 1;
            let keep = if *fn_arity == 2 {
                dispatch_callable(pred, &[val.clone(), Value::Number(i as f64)], vm)?.is_truthy()
            } else {
                dispatch_callable(pred, std::slice::from_ref(&val), vm)?.is_truthy()
            };
            if keep {
                return Ok(Some(val));
            }
        },
        IterableInner::Enumerate { data, start, index } => {
            let data_ref = data.borrow();
            if *index >= data_ref.len() {
                return Ok(None);
            }
            let element = &data_ref[*index];
            let value = match element {
                Value::Array(arr_rc) => Value::Array(Rc::clone(arr_rc)),
                Value::Tuple(tuple_rc) => Value::Tuple(Rc::clone(tuple_rc)),
                Value::Table(table_rc) => Value::Table(Rc::clone(table_rc)),
                Value::Axis(axis_rc) => Value::Axis(Rc::clone(axis_rc)),
                Value::Figure(fig_rc) => Value::Figure(Rc::clone(fig_rc)),
                Value::Image(img_rc) => Value::Image(Rc::clone(img_rc)),
                Value::Window(handle) => Value::Window(*handle),
                Value::PluginOpaque { tag, id } => Value::PluginOpaque { tag: *tag, id: *id },
                Value::Object(obj_rc) => Value::Object(obj_rc.clone()),
                Value::DatabaseEngine(engine_rc) => Value::DatabaseEngine(Rc::clone(engine_rc)),
                Value::DatabaseCluster(cluster_rc) => Value::DatabaseCluster(Rc::clone(cluster_rc)),
                _ => element.clone(),
            };
            let pair = Value::Tuple(Rc::new(RefCell::new(vec![
                Value::Number((*start + *index as i64) as f64),
                value,
            ])));
            *index += 1;
            Ok(Some(pair))
        }
        IterableInner::Chunks {
            source,
            chunk_size,
            chunk_index,
        } => {
            let total = chunk_source_count(source, *chunk_size);
            if *chunk_index >= total {
                return Ok(None);
            }
            let v = materialize_chunk_at(vm, source, *chunk_size, *chunk_index)?;
            *chunk_index += 1;
            Ok(Some(v))
        }
        IterableInner::TableRows { table, index } => {
            let t = table.borrow();
            let len = t.len();
            if *index >= len {
                return Ok(None);
            }
            let row = if t.is_view() {
                crate::vm::table_ops::get_row(&*t, *index, vm.value_store(), vm.heavy_store())
            } else {
                t.get_row(*index).map(|r| r.to_vec())
            };
            drop(t);
            let Some(row) = row else {
                return Err(runtime(0, "internal: table row missing"));
            };
            let t = table.borrow();
            let mut row_dict = HashMap::new();
            for (i, header) in t.headers().iter().enumerate() {
                if i < row.len() {
                    row_dict.insert(header.clone(), row[i].clone());
                }
            }
            drop(t);
            *index += 1;
            Ok(Some(Value::legacy_object(row_dict)))
        }
        IterableInner::EnumerateIter {
            source,
            start,
            next_index,
        } => {
            let next = {
                let mut src = source.borrow_mut();
                iterable_next(&mut *src, vm)?
            };
            let Some(val) = next else {
                return Ok(None);
            };
            let i = *next_index;
            *next_index += 1;
            let pair = Value::Tuple(Rc::new(RefCell::new(vec![
                Value::Number((*start + i as i64) as f64),
                val,
            ])));
            Ok(Some(pair))
        }
        IterableInner::StreamGenerator { state } => run_generator_next(vm, &mut state.borrow_mut()),
        IterableInner::ObjectFieldList { element_ids, index } => {
            if *index >= element_ids.len() {
                return Ok(None);
            }
            let id = element_ids[*index];
            *index += 1;
            let store = vm.value_store();
            let heap = vm.heavy_store();
            Ok(Some(load_value(id, store, heap)))
        }
        IterableInner::Set {
            set,
            element_ids,
            index,
            start_generation,
        } => {
            if set.borrow().generation() != *start_generation {
                return Err(runtime(0, "RuntimeError: set modified during iteration"));
            }
            if *index >= element_ids.len() {
                return Ok(None);
            }
            let id = element_ids[*index];
            *index += 1;
            let store = vm.value_store();
            let heap = vm.heavy_store();
            Ok(Some(load_value(id, store, heap)))
        }
        IterableInner::String { text, index } => {
            if let Some(ch) = text.chars().nth(*index) {
                *index += 1;
                Ok(Some(Value::String(ch.to_string())))
            } else {
                Ok(None)
            }
        }
        IterableInner::Range {
            current,
            end,
            step,
        } => {
            let done = if *step > 0 {
                *current >= *end
            } else {
                *current <= *end
            };
            if done {
                return Ok(None);
            }
            let v = Value::Number(*current as f64);
            *current += *step;
            Ok(Some(v))
        }
        IterableInner::SpecialInstance { receiver_id } => {
            match crate::vm::special_methods::dispatch_special_by_id(*receiver_id, "@next", &[]) {
                Ok(Some(v)) if matches!(v, Value::Null) => Ok(None),
                Ok(Some(v)) => Ok(Some(v)),
                Ok(None) => Err(runtime(0, "missing `@next` during iteration")),
                Err(e) => Err(e),
            }
        }
    }
}

/// Recursively replace [`Value::Iterable`] with owned [`Value::Array`] (eager, one pass per iterator)
/// so values can cross the native-module FFI (`AbiValue` has no iterable variant).
/// Also walks arrays, tuples, table cells, and materialized array views.
pub fn materialize_iterables_in_value(
    vm: &mut Vm,
    v: &Value,
    store: &ValueStore,
    heap: &HeavyStore,
) -> Result<Value, LangError> {
    match v {
        Value::Generator(g) => {
            let rc = Rc::new(RefCell::new(IterableInner::StreamGenerator {
                state: Rc::clone(g),
            }));
            materialize_iterables_in_value(vm, &Value::Iterable(rc), store, heap)
        }
        Value::ObjectFieldList { .. } => Ok(v.clone()),
        Value::Iterable(rc) => {
            let mut inner = rc.borrow().clone();
            let mut out: Vec<Value> = Vec::new();
            loop {
                match iterable_next(&mut inner, vm)? {
                    None => break,
                    Some(elem) => out.push(materialize_iterables_in_value(vm, &elem, store, heap)?),
                }
            }
            Ok(Value::Array(Rc::new(RefCell::new(out))))
        }
        Value::ArrayView(av) => {
            let arr = materialize_array_view(av, store, heap);
            materialize_iterables_in_value(vm, &arr, store, heap)
        }
        Value::Array(rc) => {
            let b = rc.borrow();
            let mut out = Vec::with_capacity(b.len());
            for x in b.iter() {
                out.push(materialize_iterables_in_value(vm, x, store, heap)?);
            }
            drop(b);
            Ok(Value::Array(Rc::new(RefCell::new(out))))
        }
        Value::Tuple(rc) => {
            let b = rc.borrow();
            let mut out = Vec::with_capacity(b.len());
            for x in b.iter() {
                out.push(materialize_iterables_in_value(vm, x, store, heap)?);
            }
            drop(b);
            Ok(Value::Tuple(Rc::new(RefCell::new(out))))
        }
        Value::Table(rc) => {
            let mut table = rc.borrow().clone();
            if let TableData::Owned { ref mut flat, .. } = table.data {
                for cell in flat.iter_mut() {
                    let c = std::mem::replace(cell, Value::Null);
                    *cell = materialize_iterables_in_value(vm, &c, store, heap)?;
                }
            }
            Ok(Value::Table(Rc::new(RefCell::new(table))))
        }
        _ => Ok(v.clone()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::table::Table;

    #[test]
    fn coerce_accepts_table() {
        let t = Table::new();
        let v = Value::Table(Rc::new(RefCell::new(t)));
        let out = coerce_to_iterable_value(v).expect("coerce");
        match out {
            Value::Iterable(rc) => assert!(matches!(
                *rc.borrow(),
                IterableInner::TableRows { .. }
            )),
            _ => panic!("expected iterable"),
        }
    }
}
