// Array manipulation native functions

use crate::common::error::LangError;
use crate::common::numeric::{integer_value_as_i64_if_whole, numeric_sort_class, SortAtom};
use crate::common::value::{ChunkSource, IterableInner, Value};
use crate::vm::array_view::materialize_array_view;
use crate::vm::host::HostFunction;
use crate::vm::iterable::{coerce_to_iterable_value, iterable_materialize_capacity_hint, iterable_next};
use crate::vm::native_loader::{call_abi_native, clear_last_abi_error, take_last_abi_error};
use crate::vm::vm::{current_vm_ptr, with_current_stores};
use std::cell::RefCell;
use std::collections::HashSet;
use std::rc::Rc;

/// `sum` / `average` on plugin tensors: delegate to `ml` via `native_plugin_call(_, "sum"|"mean")`.
fn plugin_opaque_sum_or_mean_via_abi(arg: &Value, op: &str) -> Option<Value> {
    let Value::PluginOpaque { .. } = arg else {
        return None;
    };
    let vm_ptr = current_vm_ptr()?;
    unsafe {
        let vm = &*vm_ptr;
        let native_idx = vm.plugin_call_native?;
        let builtin_count = vm.builtin_natives_count();
        let abi = vm.get_abi_natives();
        if native_idx < builtin_count || native_idx >= builtin_count + abi.len() {
            return None;
        }
        let args = [arg.clone(), Value::String(op.to_string())];
        clear_last_abi_error();
        let v = call_abi_native(
            abi[native_idx - builtin_count],
            &args,
            Some((vm.value_store(), vm.heavy_store())),
        );
        if take_last_abi_error().is_some() {
            return None;
        }
        Some(v)
    }
}

/// Returns an empty array with pre-allocated capacity to avoid reallocations when using push() in a loop.
pub fn native_array_with_capacity(args: &[Value]) -> Value {
    if args.is_empty() {
        return Value::Array(Rc::new(RefCell::new(Vec::new())));
    }
    let n = match integer_value_as_i64_if_whole(&args[0]) {
        Some(idx) if (0..=1_000_000_000).contains(&idx) => idx as usize,
        _ => return Value::Null,
    };
    Value::Array(Rc::new(RefCell::new(Vec::with_capacity(n))))
}

pub fn native_push(args: &[Value]) -> Value {
    if args.len() < 2 {
        return Value::Null;
    }

    if matches!(args.first(), Some(Value::ObjectFieldList { .. })) {
        crate::websocket::set_native_error(
            "ReadOnlyError: cannot modify read-only dict keys/values view".to_string(),
        );
        return Value::Null;
    }

    let item = args[1].clone();

    match &args[0] {
        Value::Array(a) => {
            // Мутируем массив in-place, сохраняя ссылочную семантику
            a.borrow_mut().push(item);
            Value::Array(Rc::clone(a))
        }
        Value::ArrayView(av) => {
            // Срез (`queue[1:]`) — ArrayView; материализуем, дописываем элемент, возвращаем новый Array
            let Value::Array(a) = with_current_stores(|store, heap| {
                materialize_array_view(av, store, heap)
            }) else {
                return Value::Null;
            };
            a.borrow_mut().push(item);
            Value::Array(Rc::clone(&a))
        }
        _ => Value::Null,
    }
}

/// `pop(array [, idx])` — remove element at `idx` (default `-1`, Python-style negative indices).
pub fn native_pop(args: &[Value]) -> Value {
    if args.is_empty() {
        return Value::Null;
    }

    if matches!(args.first(), Some(Value::ObjectFieldList { .. })) {
        crate::websocket::set_native_error(
            "ReadOnlyError: cannot modify read-only dict keys/values view".to_string(),
        );
        return Value::Null;
    }

    let arr = match &args[0] {
        Value::Array(a) => a,
        _ => return Value::Null,
    };

    let mut idx: i64 = -1;
    if args.len() >= 2 && !matches!(args[1], Value::Null) {
        match integer_value_as_i64_if_whole(&args[1]) {
            Some(i) => idx = i,
            None => {
                crate::websocket::set_native_error(
                    "TypeError: pop() index must be an integer".to_string(),
                );
                return Value::Null;
            }
        }
    }

    // Copy-on-Write: если массив используется в нескольких местах, клонируем
    let arr = if Rc::strong_count(arr) > 1 {
        let cloned_vec: Vec<Value> = arr.borrow().iter().map(|v| v.clone()).collect();
        Rc::new(RefCell::new(cloned_vec))
    } else {
        Rc::clone(arr)
    };

    let mut arr_ref = arr.borrow_mut();
    let n = arr_ref.len();
    if n == 0 {
        return Value::Null;
    }

    if idx < 0 {
        idx += n as i64;
    }
    if idx < 0 || idx >= n as i64 {
        crate::websocket::set_native_error("IndexError: pop index out of range".to_string());
        return Value::Null;
    }

    arr_ref.remove(idx as usize)
}

/// Splits an array into consecutive chunks of length `n` (positive integer). Last chunk may be shorter.
/// Returns a lazy [`Value::Iterable`] that yields each chunk as an owned [`Value::Array`] one at a time
/// (no upfront allocation for the outer list of all chunks). Indexing and `len()` are supported.
pub fn native_chunk(args: &[Value]) -> Value {
    if args.len() < 2 {
        return Value::Null;
    }
    let size = match &args[1] {
        Value::Number(n) => {
            if n.fract() != 0.0 || *n <= 0.0 {
                return Value::Null;
            }
            if *n > usize::MAX as f64 {
                return Value::Null;
            }
            *n as usize
        }
        _ => return Value::Null,
    };
    match &args[0] {
        Value::Array(arr) => Value::Iterable(Rc::new(RefCell::new(IterableInner::Chunks {
            source: ChunkSource::Array(Rc::clone(arr)),
            chunk_size: size,
            chunk_index: 0,
        }))),
        Value::ArrayView(av) => Value::Iterable(Rc::new(RefCell::new(IterableInner::Chunks {
            source: ChunkSource::ArrayView(av.clone()),
            chunk_size: size,
            chunk_index: 0,
        }))),
        Value::ByteBuffer(bb) => Value::Iterable(Rc::new(RefCell::new(IterableInner::Chunks {
            source: ChunkSource::Bytes(bb.clone()),
            chunk_size: size,
            chunk_index: 0,
        }))),
        _ => Value::Null,
    }
}

pub fn native_unique(args: &[Value]) -> Value {
    if args.is_empty() {
        return Value::Array(Rc::new(RefCell::new(Vec::new())));
    }
    let mut seen = HashSet::new();
    let mut result = Vec::new();
    match &args[0] {
        Value::Array(a) => {
            for item in a.borrow().iter() {
                let item_str = item.to_string();
                if !seen.contains(&item_str) {
                    seen.insert(item_str);
                    result.push(item.clone());
                }
            }
        }
        Value::ColumnReference { table, column_name } => {
            crate::vm::vm::with_current_stores(|store, heap| {
                let t = table.borrow();
                for i in 0..t.len() {
                    if let Some(item) =
                        crate::vm::table_ops::get_cell_value(&*t, i, column_name, store, heap)
                    {
                        let item_str = item.to_string();
                        if !seen.contains(&item_str) {
                            seen.insert(item_str);
                            result.push(item);
                        }
                    }
                }
            });
        }
        _ => return Value::Null,
    }
    Value::Array(Rc::new(RefCell::new(result)))
}

pub fn native_reverse(args: &[Value]) -> Value {
    if args.is_empty() {
        return Value::Array(Rc::new(RefCell::new(Vec::new())));
    }

    let arr = match &args[0] {
        Value::Array(a) => a,
        _ => return Value::Null,
    };

    // Copy-on-Write: если массив используется в нескольких местах, клонируем
    let arr = if Rc::strong_count(arr) > 1 {
        let cloned_vec: Vec<Value> = arr.borrow().iter().map(|v| v.clone()).collect();
        Rc::new(RefCell::new(cloned_vec))
    } else {
        arr.clone()
    };

    // Мутируем in-place
    arr.borrow_mut().reverse();

    Value::Array(arr)
}

#[inline]
fn sort_values_slice(values: &mut [Value]) {
    values.sort_by(|a, b| {
        match (value_numeric_sort_atom(a), value_numeric_sort_atom(b)) {
            (Some(sa), Some(sb)) => numeric_sort_class(sa, sb),
            _ => a.to_string().cmp(&b.to_string()),
        }
    });
}

#[inline]
fn value_numeric_sort_atom(v: &Value) -> Option<SortAtom> {
    match v {
        Value::Int(i) => Some(i.into_sort_atom()),
        Value::Float(f) => Some(f.sort_atom()),
        Value::Number(n) => {
            if n.is_nan() {
                Some(SortAtom::Nan)
            } else if *n == f64::INFINITY {
                Some(SortAtom::PosInf)
            } else if *n == f64::NEG_INFINITY {
                Some(SortAtom::NegInf)
            } else {
                Some(SortAtom::Finite(*n))
            }
        }
        _ => None,
    }
}

pub fn native_sort(args: &[Value]) -> Value {
    if args.is_empty() {
        return Value::Array(Rc::new(RefCell::new(Vec::new())));
    }

    match &args[0] {
        Value::Array(a) => {
            // Copy-on-Write: если массив используется в нескольких местах, клонируем
            let arr = if Rc::strong_count(a) > 1 {
                let cloned_vec: Vec<Value> = a.borrow().iter().map(|v| v.clone()).collect();
                Rc::new(RefCell::new(cloned_vec))
            } else {
                a.clone()
            };
            sort_values_slice(&mut *arr.borrow_mut());
            Value::Array(arr)
        }
        other => {
            let Some(vm_ptr) = current_vm_ptr() else {
                return Value::Null;
            };
            let coerced = match coerce_to_iterable_value(other.clone()) {
                Ok(v) => v,
                Err(_) => return Value::Null,
            };
            let Value::Iterable(rc) = coerced else {
                return Value::Null;
            };
            unsafe {
                let vm = &mut *vm_ptr;
                let mut inner = rc.borrow().clone();
                let cap = iterable_materialize_capacity_hint(&inner);
                let mut out = cap.map_or_else(Vec::new, Vec::with_capacity);
                loop {
                    match iterable_next(&mut inner, vm) {
                        Ok(None) => break,
                        Ok(Some(v)) => out.push(v),
                        Err(_) => return Value::Null,
                    }
                }
                sort_values_slice(&mut out);
                Value::Array(Rc::new(RefCell::new(out)))
            }
        }
    }
}

pub fn native_sum(args: &[Value]) -> Value {
    if args.is_empty() {
        return Value::Number(0.0);
    }
    if args.len() == 1 {
        if let Some(v) = plugin_opaque_sum_or_mean_via_abi(&args[0], "sum") {
            return v;
        }
        if matches!(&args[0], Value::PluginOpaque { .. }) {
            return Value::Number(0.0);
        }
    }
    let (mut sum, mut has_numbers) = (0.0, false);
    match &args[0] {
        Value::Array(a) => {
            if let Some(vm_ptr) = current_vm_ptr() {
                unsafe {
                    if let Some(n) = (*vm_ptr).compute_mut().try_sum_value(&args[0]) {
                        return Value::Number(n);
                    }
                }
            }
            for item in a.borrow().iter() {
                if let Some(n) = item.as_ieee_f64() {
                    sum += n;
                    has_numbers = true;
                }
            }
        }
        Value::ColumnReference { table, column_name } => {
            crate::vm::vm::with_current_stores(|store, heap| {
                let t = table.borrow();
                for i in 0..t.len() {
                    if let Some(item) =
                        crate::vm::table_ops::get_cell_value(&*t, i, column_name, store, heap)
                    {
                        if let Some(n) = item.as_ieee_f64() {
                            sum += n;
                            has_numbers = true;
                        }
                    }
                }
            });
        }
        _ => {}
    }
    if has_numbers {
        Value::Number(sum)
    } else {
        Value::Number(0.0)
    }
}

pub fn native_average(args: &[Value]) -> Value {
    if args.is_empty() {
        return Value::Number(0.0);
    }
    if args.len() == 1 {
        if let Some(v) = plugin_opaque_sum_or_mean_via_abi(&args[0], "mean") {
            return v;
        }
        if matches!(&args[0], Value::PluginOpaque { .. }) {
            return Value::Number(0.0);
        }
    }
    let (mut sum, mut count) = (0.0, 0);
    match &args[0] {
        Value::Array(a) => {
            for item in a.borrow().iter() {
                if let Some(n) = item.as_ieee_f64() {
                    sum += n;
                    count += 1;
                }
            }
        }
        Value::ColumnReference { table, column_name } => {
            crate::vm::vm::with_current_stores(|store, heap| {
                let t = table.borrow();
                for i in 0..t.len() {
                    if let Some(item) =
                        crate::vm::table_ops::get_cell_value(&*t, i, column_name, store, heap)
                    {
                        if let Some(n) = item.as_ieee_f64() {
                            sum += n;
                            count += 1;
                        }
                    }
                }
            });
        }
        _ => {}
    }
    if count > 0 {
        Value::Number(sum / count as f64)
    } else {
        Value::Number(0.0)
    }
}

pub fn native_count(args: &[Value]) -> Value {
    if args.is_empty() {
        return Value::Number(0.0);
    }

    // Handle array and column reference (lazy: no materialization)
    match &args[0] {
        Value::Array(arr) => Value::Number(arr.borrow().len() as f64),
        Value::ColumnReference { table, column_name } => {
            crate::vm::vm::with_current_stores(|_store, _heap| {
                let t = table.borrow();
                crate::vm::table_ops::column_len(&*t, column_name)
                    .map(|len| Value::Number(len as f64))
                    .unwrap_or(Value::Number(0.0))
            })
        }
        _ => Value::Number(0.0),
    }
}

/// Built-in `any(...)`: истина, если есть хотя бы один truthy-элемент.
/// Поддерживает [`Value::Array`] и ленивый [`Value::Iterable`] (например результат `map()`).
pub struct AnyHostFunction;

impl HostFunction for AnyHostFunction {
    fn call(&self, args: &[Value]) -> Result<Value, LangError> {
        if args.is_empty() {
            return Ok(Value::Bool(false));
        }

        match &args[0] {
            Value::Array(arr) => {
                let arr_ref = arr.borrow();
                if arr_ref.is_empty() {
                    return Ok(Value::Bool(false));
                }
                for item in arr_ref.iter() {
                    if item.is_truthy() {
                        return Ok(Value::Bool(true));
                    }
                }
                Ok(Value::Bool(false))
            }
            Value::Iterable(it) => {
                let vm_ptr = current_vm_ptr().ok_or_else(|| {
                    LangError::runtime_error("any(iterable): VM context not available".to_string(), 0)
                })?;
                unsafe {
                    let vm = &mut *vm_ptr;
                    let mut inner = it.borrow().clone();
                    loop {
                        match iterable_next(&mut inner, vm)? {
                            None => return Ok(Value::Bool(false)),
                            Some(item) => {
                                if item.is_truthy() {
                                    return Ok(Value::Bool(true));
                                }
                            }
                        }
                    }
                }
            }
            _ => Ok(Value::Bool(false)),
        }
    }
}

/// Built-in `all(...)`: истина, если все элементы truthy и коллекция не пуста.
/// Поддерживает [`Value::Array`] и ленивый [`Value::Iterable`].
pub struct AllHostFunction;

impl HostFunction for AllHostFunction {
    fn call(&self, args: &[Value]) -> Result<Value, LangError> {
        if args.is_empty() {
            return Ok(Value::Bool(false));
        }

        match &args[0] {
            Value::Array(arr) => {
                let arr_ref = arr.borrow();
                if arr_ref.is_empty() {
                    return Ok(Value::Bool(false));
                }
                for item in arr_ref.iter() {
                    if !item.is_truthy() {
                        return Ok(Value::Bool(false));
                    }
                }
                Ok(Value::Bool(true))
            }
            Value::Iterable(it) => {
                let vm_ptr = current_vm_ptr().ok_or_else(|| {
                    LangError::runtime_error("all(iterable): VM context not available".to_string(), 0)
                })?;
                unsafe {
                    let vm = &mut *vm_ptr;
                    let mut inner = it.borrow().clone();
                    let mut saw_any = false;
                    loop {
                        match iterable_next(&mut inner, vm)? {
                            None => {
                                return Ok(Value::Bool(saw_any));
                            }
                            Some(item) => {
                                saw_any = true;
                                if !item.is_truthy() {
                                    return Ok(Value::Bool(false));
                                }
                            }
                        }
                    }
                }
            }
            _ => Ok(Value::Bool(false)),
        }
    }
}

#[cfg(test)]
mod sort_inf_tests {
    use super::native_sort;
    use crate::common::numeric::FloatValue;
    use crate::common::value::Value;
    use std::cell::RefCell;
    use std::rc::Rc;

    #[test]
    fn native_sort_single_positive_infinity() {
        let a = Value::Array(Rc::new(RefCell::new(vec![Value::Float(
            FloatValue::PosInfinity,
        )])));
        let out = native_sort(&[a]);
        match out {
            Value::Array(rc) => {
                let b = rc.borrow();
                assert_eq!(b.len(), 1);
                assert!(matches!(b[0], Value::Float(FloatValue::PosInfinity)));
            }
            other => panic!("expected array, got {:?}", other),
        }
    }
}
