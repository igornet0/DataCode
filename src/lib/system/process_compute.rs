//! `system.process` compute device API (CPU / GPU / Metal).

use crate::common::value::{ObjectKind, Value};
use crate::compute::device::DeviceKind;
use crate::websocket::set_native_error;
use crate::vm::vm::current_vm_ptr;
use std::cell::RefCell;
use std::rc::Rc;

fn is_module_receiver(v: &Value) -> bool {
    let Value::Object(o) = v else {
        return false;
    };
    let map = o.borrow();
    let native_count = match &*map {
        ObjectKind::Legacy(hm) => hm
            .values()
            .filter(|v| matches!(v, Value::NativeFunction(_)))
            .count(),
        ObjectKind::Inline(pairs) => pairs
            .iter()
            .filter(|(_, v)| matches!(v, Value::NativeFunction(_)))
            .count(),
        _ => 0,
    };
    native_count >= 3
}

fn data_args<'a>(args: &'a [Value]) -> Vec<&'a Value> {
    args.iter().filter(|v| !is_module_receiver(v)).collect()
}

fn with_compute_mut<F, T>(f: F) -> Option<T>
where
    F: FnOnce(&mut crate::compute::runtime::ComputeRuntime) -> T,
{
    let ptr = current_vm_ptr()?;
    Some(unsafe { f(&mut (*ptr).compute_mut()) })
}

fn with_compute<F, T>(f: F) -> Option<T>
where
    F: FnOnce(&crate::compute::runtime::ComputeRuntime) -> T,
{
    let ptr = current_vm_ptr()?;
    Some(unsafe { f((*ptr).compute()) })
}

fn arg_device(args: &[Value], i: usize) -> Option<DeviceKind> {
    let n = args.get(i)?.as_ieee_f64()?;
    DeviceKind::from_f64(n).or_else(|| {
        set_native_error(format!(
            "ValueError: invalid device constant at argument {}",
            i + 1
        ));
        None
    })
}

pub fn device_constant(kind: DeviceKind) -> Value {
    Value::Number(kind.as_f64())
}

pub fn native_process_get_device(_args: &[Value]) -> Value {
    with_compute(|c| device_constant(c.get_device())).unwrap_or(device_constant(DeviceKind::Cpu))
}

pub fn native_process_set_device(args: &[Value]) -> Value {
    let Some(kind) = args.iter().find_map(|v| DeviceKind::from_f64(v.as_ieee_f64()?)) else {
        set_native_error("TypeError: set_device(device_constant)".to_string());
        return Value::Null;
    };
    if with_compute_mut(|c| c.set_device(kind)).is_some() {
        device_constant(kind)
    } else {
        Value::Null
    }
}

pub fn native_process_get_gpu_min_size(_args: &[Value]) -> Value {
    with_compute(|c| Value::Number(c.gpu_min_size() as f64))
        .unwrap_or(Value::Number(20_000.0))
}

pub fn native_process_set_gpu_min_size(args: &[Value]) -> Value {
    let n = match data_args(args)
        .first()
        .and_then(|v| v.as_finite_f64())
    {
        Some(x) if x >= 1.0 => x as usize,
        _ => {
            set_native_error("TypeError: gpu_min_size must be >= 1".to_string());
            return Value::Null;
        }
    };
    if with_compute_mut(|c| c.set_gpu_min_size(n)).is_some() {
        Value::Number(n as f64)
    } else {
        Value::Null
    }
}

pub fn native_process_auto_device(args: &[Value]) -> Value {
    let cb = match args.first() {
        Some(v @ (Value::Function(_) | Value::NativeFunction(_) | Value::ModuleFunction { .. })) => {
            Some(v.clone())
        }
        Some(Value::Null) => None,
        _ => {
            set_native_error("TypeError: auto_device(callback_fn)".to_string());
            return Value::Null;
        }
    };
    if with_compute_mut(|c| c.set_auto_selector(cb)).is_some() {
        Value::Bool(true)
    } else {
        Value::Null
    }
}

pub fn native_process_has_gpu(_args: &[Value]) -> Value {
    Value::Bool(with_compute(|c| c.has_gpu()).unwrap_or(false))
}

pub fn native_process_has_cuda(_args: &[Value]) -> Value {
    Value::Bool(with_compute(|c| c.has_cuda()).unwrap_or(false))
}

pub fn native_process_has_metal(_args: &[Value]) -> Value {
    Value::Bool(with_compute(|c| c.has_metal()).unwrap_or(false))
}

pub fn native_process_info(_args: &[Value]) -> Value {
    let info = with_compute(|c| c.info()).unwrap_or_default();
    let mut m = std::collections::HashMap::new();
    m.insert("backend".to_string(), Value::String(info.backend));
    m.insert("gpu_name".to_string(), Value::String(info.gpu_name));
    m.insert("memory".to_string(), Value::Number(info.memory_mb as f64));
    m.insert("cores".to_string(), Value::Number(info.cores as f64));
    Value::Object(Rc::new(RefCell::new(ObjectKind::Legacy(m))))
}

/// Element-wise add on equal-length numeric arrays (uses compute runtime / GPU when enabled).
pub fn native_process_vector_add(args: &[Value]) -> Value {
    let data = data_args(args);
    if data.len() < 2 {
        set_native_error("TypeError: vector_add(a, b)".to_string());
        return Value::Null;
    }
    let ptr = match current_vm_ptr() {
        Some(p) => p,
        None => return Value::Null,
    };
    unsafe {
        if let Some(out) = (*ptr).compute_mut().try_add_values(data[0], data[1]) {
            return out;
        }
    }
    set_native_error("TypeError: vector_add requires equal-length numeric arrays".to_string());
    Value::Null
}

/// Element-wise mul on equal-length numeric arrays.
pub fn native_process_vector_mul(args: &[Value]) -> Value {
    let data = data_args(args);
    if data.len() < 2 {
        set_native_error("TypeError: vector_mul(a, b)".to_string());
        return Value::Null;
    }
    let ptr = match current_vm_ptr() {
        Some(p) => p,
        None => return Value::Null,
    };
    unsafe {
        if let Some(out) = (*ptr).compute_mut().try_mul_values(data[0], data[1]) {
            return out;
        }
    }
    set_native_error("TypeError: vector_mul requires equal-length numeric arrays".to_string());
    Value::Null
}

/// `run("add"|"mul"|"sum", ...)` helper for benchmarks.
pub fn native_process_run(args: &[Value]) -> Value {
    let data = data_args(args);
    if data.is_empty() {
        set_native_error("TypeError: run(op, ...)".to_string());
        return Value::Null;
    }
    let op = match data[0] {
        Value::String(s) => s.as_str(),
        _ => {
            set_native_error("TypeError: run(op, ...) op must be string".to_string());
            return Value::Null;
        }
    };
    let ptr = match current_vm_ptr() {
        Some(p) => p,
        None => return Value::Null,
    };
    unsafe {
        let c = &mut *ptr;
        match op {
            "add" if data.len() >= 3 => {
                if let Some(v) = c.compute_mut().try_add_values(data[1], data[2]) {
                    return v;
                }
            }
            "mul" if data.len() >= 3 => {
                if let Some(v) = c.compute_mut().try_mul_values(data[1], data[2]) {
                    return v;
                }
            }
            "sum" if data.len() >= 2 => {
                if let Some(n) = c.compute_mut().try_sum_value(data[1]) {
                    return Value::Number(n);
                }
            }
            _ => {}
        }
    }
    set_native_error(format!("TypeError: unknown or invalid run({})", op));
    Value::Null
}

#[allow(dead_code)]
pub fn parse_device_arg(args: &[Value], i: usize) -> Option<DeviceKind> {
    arg_device(args, i)
}
