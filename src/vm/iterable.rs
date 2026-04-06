//! Lazy iterable pipeline (`map` / `filter` / `reduce`): one pass, no intermediate arrays.

use crate::common::error::LangError;
use crate::common::table::TableData;
use crate::common::value::{CallableSlot, ChunkSource, IterableInner, Value};
use crate::common::value_store::ValueStore;
use crate::vm::array_view::{materialize_array_view, subview, view_get_element};
use crate::vm::generator::run_generator_next;
use crate::vm::heavy_store::HeavyStore;
use crate::vm::natives::utils::call_user_function;
use crate::vm::vm::Vm;
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
        _ => coerce_to_iterable_value(v),
    }
}

/// Wrap `Value` as [`Value::Iterable`] with iterator state at the start (for `for`-`in` and `reduce`).
pub fn coerce_to_iterable_value(v: Value) -> Result<Value, LangError> {
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
        _ => Err(runtime(
            0,
            "for-in / iterable: expected array, array view, tuple, enumerate, iterable, or generator",
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

fn dispatch_slot(slot: &CallableSlot, args: &[Value], vm: &mut Vm) -> Result<Value, LangError> {
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
        IterableInner::Chunks {
            source, chunk_size, ..
        } => Some(chunk_source_count(source, *chunk_size)),
        IterableInner::StreamGenerator { .. } => None,
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
                dispatch_slot(func, &[val, Value::Number(i as f64)], vm)?
            } else {
                dispatch_slot(func, &[val], vm)?
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
                dispatch_slot(pred, &[val.clone(), Value::Number(i as f64)], vm)?.is_truthy()
            } else {
                dispatch_slot(pred, std::slice::from_ref(&val), vm)?.is_truthy()
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
        IterableInner::StreamGenerator { state } => run_generator_next(vm, &mut state.borrow_mut()),
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
