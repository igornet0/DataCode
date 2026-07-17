// Binary and unary operations for VM (stack as Vec<TaggedValue>; handle_exception needs value_store/heavy_store)

use crate::common::{
    error::{ErrorType, LangError},
    numeric::{divide_by_zero_raises, ieee_div_quotient_value, IntValue},
    value::Value,
    value_ord::value_partial_cmp,
    value_store::ValueStore,
    TaggedValue,
};
use crate::vm::exceptions::ExceptionHandler;
use crate::vm::frame::CallFrame;
use crate::vm::heavy_store::HeavyStore;
use crate::vm::iterable::iterable_next;
use crate::vm::native_loader::{call_abi_native, take_last_abi_error};
use crate::vm::vm::current_vm_ptr;
use std::cell::RefCell;
use std::cmp::Ordering;
use std::fmt::Write;
use std::rc::Rc;

/// Two `PluginOpaque` values and loaded `ml.opaque_binop`: delegate to libml (`add`/`sub`/…).
pub(crate) fn try_plugin_opaque_binop(a: &Value, b: &Value, op: &str) -> Option<Value> {
    let (Value::PluginOpaque { .. }, Value::PluginOpaque { .. }) = (a, b) else {
        return None;
    };
    let vm_ptr = current_vm_ptr()?;
    unsafe {
        let vm = &*vm_ptr;
        let native_idx = vm.plugin_opaque_binop?;
        let builtin_count = vm.builtin_natives_count();
        let abi = vm.get_abi_natives();
        if native_idx < builtin_count || native_idx >= builtin_count + abi.len() {
            return None;
        }
        let f = abi[native_idx - builtin_count];
        let out = call_abi_native(
            f,
            &[a.clone(), b.clone(), Value::String(op.to_string())],
            Some((vm.value_store(), vm.heavy_store())),
        );
        if take_last_abi_error().is_some() {
            return None;
        }
        Some(out)
    }
}

/// Get the current line number from the frames
fn get_line(frames: &mut Vec<CallFrame>) -> usize {
    if let Some(frame) = frames.last() {
        if frame.ip > 0 {
            frame.function.chunk.get_line(frame.ip - 1)
        } else {
            0
        }
    } else {
        0
    }
}

/// Binary addition operation
pub fn binary_add(
    a: &Value,
    b: &Value,
    frames: &mut Vec<CallFrame>,
    stack: &mut Vec<TaggedValue>,
    exception_handlers: &mut Vec<ExceptionHandler>,
    value_store: &mut ValueStore,
    heavy_store: &mut HeavyStore,
) -> Result<Value, LangError> {
    let line = get_line(frames);
    if let Some(v) = try_plugin_opaque_binop(a, b, "add") {
        return Ok(v);
    }
    // Convert null to 0 for arithmetic operations (useful for class fields with default values)
    let a = if matches!(a, Value::Null) {
        &Value::Number(0.0)
    } else {
        a
    };
    let b = if matches!(b, Value::Null) {
        &Value::Number(0.0)
    } else {
        b
    };

    match (a, b) {
        (Value::Number(n1), Value::Number(n2)) => Ok(Value::Number(n1 + n2)),
        (Value::String(s1), Value::String(s2)) => {
            let mut buf = String::with_capacity(s1.len() + s2.len());
            buf.push_str(s1);
            buf.push_str(s2);
            Ok(Value::String(buf))
        }
        (Value::String(s), Value::Number(n)) => {
            let mut buf = String::with_capacity(s.len() + 24);
            buf.push_str(s);
            let _ = write!(buf, "{}", n);
            Ok(Value::String(buf))
        }
        (Value::String(s), v) if v.as_ieee_f64().is_some() => {
            let mut buf = String::with_capacity(s.len() + 24);
            buf.push_str(s);
            buf.push_str(&v.to_string());
            Ok(Value::String(buf))
        }
        (Value::String(s), Value::Date(d)) => {
            let mut buf = String::with_capacity(s.len() + 40);
            buf.push_str(s);
            buf.push_str(&d.to_rfc3339());
            Ok(Value::String(buf))
        }
        (Value::Date(d), Value::String(s)) => {
            let mut buf = String::with_capacity(40 + s.len());
            buf.push_str(&d.to_rfc3339());
            buf.push_str(s);
            Ok(Value::String(buf))
        }
        (Value::Number(n), Value::String(s)) => {
            let mut buf = String::with_capacity(24 + s.len());
            let _ = write!(buf, "{}", n);
            buf.push_str(s);
            Ok(Value::String(buf))
        }
        (v, Value::String(s)) if v.as_ieee_f64().is_some() => {
            let mut buf = String::with_capacity(24 + s.len());
            buf.push_str(&v.to_string());
            buf.push_str(s);
            Ok(Value::String(buf))
        }
        (Value::Array(arr1), Value::Array(arr2)) => {
            // Array concatenation
            let mut result = arr1.borrow().clone();
            result.extend_from_slice(&arr2.borrow());
            Ok(Value::Array(Rc::new(RefCell::new(result))))
        }
        (Value::String(s), Value::Bool(b)) => Ok(Value::String(format!("{}{}", s, b))),
        (Value::Bool(b), Value::String(s)) => Ok(Value::String(format!("{}{}", b, s))),
        (Value::String(s), Value::Array(arr)) => {
            let inner = arr
                .borrow()
                .iter()
                .map(|v| v.to_string())
                .collect::<Vec<_>>()
                .join(", ");
            Ok(Value::String(format!("{}[{}]", s, inner)))
        }
        (Value::Array(arr), Value::String(s)) => {
            let inner = arr
                .borrow()
                .iter()
                .map(|v| v.to_string())
                .collect::<Vec<_>>()
                .join(", ");
            Ok(Value::String(format!("[{}]{}", inner, s)))
        }
        (Value::String(s), Value::Iterable(it)) => {
            let Some(vm_ptr) = current_vm_ptr() else {
                let error = ExceptionHandler::runtime_error(
                    frames,
                    "Operands must be numbers or strings".to_string(),
                    line,
                );
                return ExceptionHandler::handle_exception_null_value(
                    stack,
                    frames,
                    exception_handlers,
                    error,
                    value_store,
                    heavy_store,
                );
            };
            unsafe {
                let vm = &mut *vm_ptr;
                let mut inner = it.borrow().clone();
                let mut parts = Vec::new();
                loop {
                    match iterable_next(&mut inner, vm) {
                        Ok(None) => break,
                        Ok(Some(elem)) => parts.push(elem.to_string()),
                        Err(e) => return Err(e),
                    }
                }
                Ok(Value::String(format!(
                    "{}[{}]",
                    s,
                    parts.join(", ")
                )))
            }
        }
        (Value::Iterable(it), Value::String(s)) => {
            let Some(vm_ptr) = current_vm_ptr() else {
                let error = ExceptionHandler::runtime_error(
                    frames,
                    "Operands must be numbers or strings".to_string(),
                    line,
                );
                return ExceptionHandler::handle_exception_null_value(
                    stack,
                    frames,
                    exception_handlers,
                    error,
                    value_store,
                    heavy_store,
                );
            };
            unsafe {
                let vm = &mut *vm_ptr;
                let mut inner = it.borrow().clone();
                let mut parts = Vec::new();
                loop {
                    match iterable_next(&mut inner, vm) {
                        Ok(None) => break,
                        Ok(Some(elem)) => parts.push(elem.to_string()),
                        Err(e) => return Err(e),
                    }
                }
                Ok(Value::String(format!(
                    "[{}]{}",
                    parts.join(", "),
                    s
                )))
            }
        }
        (Value::Date(d), Value::Duration(dur)) | (Value::Duration(dur), Value::Date(d)) => {
            match d.checked_add_signed(*dur) {
                Some(nd) => Ok(Value::Date(nd)),
                None => {
                    let error = ExceptionHandler::runtime_error(
                        frames,
                        "date arithmetic overflow".to_string(),
                        line,
                    );
                    ExceptionHandler::handle_exception_null_value(
                        stack,
                        frames,
                        exception_handlers,
                        error,
                        value_store,
                        heavy_store,
                    )
                }
            }
        }
        _ => {
            let error = ExceptionHandler::runtime_error(
                frames,
                "Operands must be numbers or strings".to_string(),
                line,
            );
            ExceptionHandler::handle_exception_null_value(
                stack,
                frames,
                exception_handlers,
                error,
                value_store,
                heavy_store,
            )
        }
    }
}

/// Binary subtraction operation
pub fn binary_sub(
    a: &Value,
    b: &Value,
    frames: &mut Vec<CallFrame>,
    stack: &mut Vec<TaggedValue>,
    exception_handlers: &mut Vec<ExceptionHandler>,
    value_store: &mut ValueStore,
    heavy_store: &mut HeavyStore,
) -> Result<Value, LangError> {
    let line = get_line(frames);
    if let Some(v) = try_plugin_opaque_binop(a, b, "sub") {
        return Ok(v);
    }
    // Convert null to 0 for arithmetic operations
    let a = if matches!(a, Value::Null) {
        &Value::Number(0.0)
    } else {
        a
    };
    let b = if matches!(b, Value::Null) {
        &Value::Number(0.0)
    } else {
        b
    };

    match (a, b) {
        (Value::Number(n1), Value::Number(n2)) => Ok(Value::Number(n1 - n2)),
        (Value::Date(d), Value::Duration(dur)) => match d.checked_sub_signed(*dur) {
            Some(nd) => Ok(Value::Date(nd)),
            None => {
                let error = ExceptionHandler::runtime_error(
                    frames,
                    "date arithmetic overflow".to_string(),
                    line,
                );
                ExceptionHandler::handle_exception_null_value(
                    stack,
                    frames,
                    exception_handlers,
                    error,
                    value_store,
                    heavy_store,
                )
            }
        },
        (Value::Date(a), Value::Date(b)) => Ok(Value::Duration(a.signed_duration_since(*b))),
        _ => {
            let error = ExceptionHandler::runtime_error(
                frames,
                "Operands must be numbers, dates, or durations".to_string(),
                line,
            );
            ExceptionHandler::handle_exception_null_value(
                stack,
                frames,
                exception_handlers,
                error,
                value_store,
                heavy_store,
            )
        }
    }
}

/// Binary multiplication operation
pub fn binary_mul(
    a: &Value,
    b: &Value,
    frames: &mut Vec<CallFrame>,
    stack: &mut Vec<TaggedValue>,
    exception_handlers: &mut Vec<ExceptionHandler>,
    value_store: &mut ValueStore,
    heavy_store: &mut HeavyStore,
) -> Result<Value, LangError> {
    let line = get_line(frames);
    if let Some(v) = try_plugin_opaque_binop(a, b, "mul") {
        return Ok(v);
    }
    match (a, b) {
        (Value::Number(n1), Value::Number(n2)) => Ok(Value::Number(n1 * n2)),
        (Value::String(s), Value::Number(n)) => {
            let count = *n as i64;
            if count <= 0 {
                Ok(Value::String(String::new()))
            } else {
                Ok(Value::String(s.repeat(count as usize)))
            }
        }
        (Value::String(s), Value::Int(iv)) => {
            let count = match *iv {
                IntValue::Finite(n) => n,
                _ => 0,
            };
            if count <= 0 {
                Ok(Value::String(String::new()))
            } else {
                Ok(Value::String(s.repeat(count as usize)))
            }
        }
        (Value::Number(n), Value::String(s)) => {
            let count = *n as i64;
            if count <= 0 {
                Ok(Value::String(String::new()))
            } else {
                Ok(Value::String(s.repeat(count as usize)))
            }
        }
        (Value::Int(iv), Value::String(s)) => {
            let count = match *iv {
                IntValue::Finite(n) => n,
                _ => 0,
            };
            if count <= 0 {
                Ok(Value::String(String::new()))
            } else {
                Ok(Value::String(s.repeat(count as usize)))
            }
        }
        (Value::Array(a1), Value::Array(a2)) => {
            if let Some(vm_ptr) = current_vm_ptr() {
                unsafe {
                    let va = Value::Array(Rc::clone(a1));
                    let vb = Value::Array(Rc::clone(a2));
                    if let Some(out) = (*vm_ptr)
                        .compute_mut()
                        .try_mul_values(&va, &vb)
                    {
                        return Ok(out);
                    }
                }
            }
            let error = ExceptionHandler::runtime_error(
                frames,
                "Operands must be numbers, equal-length numeric arrays for element-wise *, or string and number for repetition".to_string(),
                line,
            );
            ExceptionHandler::handle_exception_null_value(
                stack,
                frames,
                exception_handlers,
                error,
                value_store,
                heavy_store,
            )
        }
        _ => {
            let error = ExceptionHandler::runtime_error(
                frames,
                "Operands must be numbers, or string and number for repetition".to_string(),
                line,
            );
            ExceptionHandler::handle_exception_null_value(
                stack,
                frames,
                exception_handlers,
                error,
                value_store,
                heavy_store,
            )
        }
    }
}

/// Matrix multiply (`@`) for tensors via `opaque_binop`; numbers multiply as scalars.
pub fn binary_matmul(
    a: &Value,
    b: &Value,
    frames: &mut Vec<CallFrame>,
    stack: &mut Vec<TaggedValue>,
    exception_handlers: &mut Vec<ExceptionHandler>,
    value_store: &mut ValueStore,
    heavy_store: &mut HeavyStore,
) -> Result<Value, LangError> {
    let line = get_line(frames);
    if let Some(v) = try_plugin_opaque_binop(a, b, "matmul") {
        return Ok(v);
    }
    match (a, b) {
        (Value::Number(n1), Value::Number(n2)) => Ok(Value::Number(n1 * n2)),
        (a, b) => {
            if let (Some(x), Some(y)) = (a.as_ieee_f64(), b.as_ieee_f64()) {
                Ok(Value::Number(x * y))
            } else {
                let error = ExceptionHandler::runtime_error(
                    frames,
                    "Operands must be numbers or tensors for @".to_string(),
                    line,
                );
                ExceptionHandler::handle_exception_null_value(
                    stack,
                    frames,
                    exception_handlers,
                    error,
                    value_store,
                    heavy_store,
                )
            }
        }
    }
}

/// Dispatch for [`crate::bytecode::OpCode::BinaryOp`]: logical `op_name` from chunk constants (fast path + `opaque_binop`).
pub fn exec_binary_op_by_name(
    op_name: &str,
    a: &Value,
    b: &Value,
    frames: &mut Vec<CallFrame>,
    stack: &mut Vec<TaggedValue>,
    exception_handlers: &mut Vec<ExceptionHandler>,
    value_store: &mut ValueStore,
    heavy_store: &mut HeavyStore,
) -> Result<Value, LangError> {
    match op_name {
        "add" => binary_add(
            a,
            b,
            frames,
            stack,
            exception_handlers,
            value_store,
            heavy_store,
        ),
        "sub" => binary_sub(
            a,
            b,
            frames,
            stack,
            exception_handlers,
            value_store,
            heavy_store,
        ),
        "mul" => binary_mul(
            a,
            b,
            frames,
            stack,
            exception_handlers,
            value_store,
            heavy_store,
        ),
        "matmul" => binary_matmul(
            a,
            b,
            frames,
            stack,
            exception_handlers,
            value_store,
            heavy_store,
        ),
        "div" => binary_div(
            a,
            b,
            frames,
            stack,
            exception_handlers,
            value_store,
            heavy_store,
        ),
        "idiv" => binary_int_div(
            a,
            b,
            frames,
            stack,
            exception_handlers,
            value_store,
            heavy_store,
        ),
        "mod" => binary_mod(
            a,
            b,
            frames,
            stack,
            exception_handlers,
            value_store,
            heavy_store,
        ),
        "pow" => binary_pow(
            a,
            b,
            frames,
            stack,
            exception_handlers,
            value_store,
            heavy_store,
        ),
        _ => {
            if let Some(v) = try_plugin_opaque_binop(a, b, op_name) {
                return Ok(v);
            }
            let line = get_line(frames);
            let error = ExceptionHandler::runtime_error(
                frames,
                format!("Unknown binary op '{}'", op_name),
                line,
            );
            ExceptionHandler::handle_exception_null_value(
                stack,
                frames,
                exception_handlers,
                error,
                value_store,
                heavy_store,
            )
        }
    }
}

/// Binary division operation
pub fn binary_div(
    a: &Value,
    b: &Value,
    frames: &mut Vec<CallFrame>,
    stack: &mut Vec<TaggedValue>,
    exception_handlers: &mut Vec<ExceptionHandler>,
    value_store: &mut ValueStore,
    heavy_store: &mut HeavyStore,
) -> Result<Value, LangError> {
    let line = get_line(frames);
    if let (Some(x), Some(y)) = (a.as_ieee_f64(), b.as_ieee_f64()) {
        if y == 0.0 && divide_by_zero_raises(a, b) {
            let error =
                ExceptionHandler::runtime_error(frames, "Division by zero".to_string(), line);
            return ExceptionHandler::handle_exception_null_value(
                stack,
                frames,
                exception_handlers,
                error,
                value_store,
                heavy_store,
            );
        }
        return Ok(ieee_div_quotient_value(a, b, x, y));
    }
    match (a, b) {
        (Value::Number(n1), Value::Number(n2)) => {
            if *n2 == 0.0 {
                let error =
                    ExceptionHandler::runtime_error(frames, "Division by zero".to_string(), line);
                ExceptionHandler::handle_exception_null_value(
                    stack,
                    frames,
                    exception_handlers,
                    error,
                    value_store,
                    heavy_store,
                )
            } else {
                Ok(Value::Number(n1 / n2))
            }
        }
        // Конкатенация путей: Path / String -> Path
        (Value::Path(p), Value::String(s)) => {
            let mut new_path = p.clone();
            new_path.push(s);
            Ok(Value::Path(new_path))
        }
        // Конкатенация путей: Path / Path -> Path
        (Value::Path(p1), Value::Path(p2)) => Ok(Value::Path(p1.join(p2))),
        // Конкатенация путей: String / Path -> Path
        (Value::String(s), Value::Path(p)) => {
            use std::path::PathBuf;
            Ok(Value::Path(PathBuf::from(s).join(p)))
        }
        // Конкатенация путей: String / String -> Path (если контекст предполагает путь)
        (Value::String(s1), Value::String(s2)) => {
            use std::path::PathBuf;
            let mut path = PathBuf::from(s1);
            path.push(s2);
            Ok(Value::Path(path))
        }
        _ => {
            let error = ExceptionHandler::runtime_error(
                frames,
                "Operands must be numbers or paths".to_string(),
                line,
            );
            ExceptionHandler::handle_exception_null_value(
                stack,
                frames,
                exception_handlers,
                error,
                value_store,
                heavy_store,
            )
        }
    }
}

/// Binary integer division operation
pub fn binary_int_div(
    a: &Value,
    b: &Value,
    frames: &mut Vec<CallFrame>,
    stack: &mut Vec<TaggedValue>,
    exception_handlers: &mut Vec<ExceptionHandler>,
    value_store: &mut ValueStore,
    heavy_store: &mut HeavyStore,
) -> Result<Value, LangError> {
    let line = get_line(frames);
    if let (Some(x), Some(y)) = (a.as_ieee_f64(), b.as_ieee_f64()) {
        if y == 0.0 {
            let error =
                ExceptionHandler::runtime_error(frames, "Division by zero".to_string(), line);
            return ExceptionHandler::handle_exception_null_value(
                stack,
                frames,
                exception_handlers,
                error,
                value_store,
                heavy_store,
            );
        }
        return Ok(Value::Number((x / y).floor()));
    }
    match (a, b) {
        (Value::Number(n1), Value::Number(n2)) => {
            if *n2 == 0.0 {
                let error =
                    ExceptionHandler::runtime_error(frames, "Division by zero".to_string(), line);
                ExceptionHandler::handle_exception_null_value(
                    stack,
                    frames,
                    exception_handlers,
                    error,
                    value_store,
                    heavy_store,
                )
            } else {
                // Целочисленное деление: отбрасываем дробную часть
                Ok(Value::Number((n1 / n2).floor()))
            }
        }
        _ => {
            let error = ExceptionHandler::runtime_error(
                frames,
                "Operands must be numbers".to_string(),
                line,
            );
            ExceptionHandler::handle_exception_null_value(
                stack,
                frames,
                exception_handlers,
                error,
                value_store,
                heavy_store,
            )
        }
    }
}

/// Binary modulo operation
pub fn binary_mod(
    a: &Value,
    b: &Value,
    frames: &mut Vec<CallFrame>,
    stack: &mut Vec<TaggedValue>,
    exception_handlers: &mut Vec<ExceptionHandler>,
    value_store: &mut ValueStore,
    heavy_store: &mut HeavyStore,
) -> Result<Value, LangError> {
    let line = get_line(frames);
    if let (Some(x), Some(y)) = (a.as_ieee_f64(), b.as_ieee_f64()) {
        if y == 0.0 {
            let error =
                ExceptionHandler::runtime_error(frames, "Modulo by zero".to_string(), line);
            return ExceptionHandler::handle_exception_null_value(
                stack,
                frames,
                exception_handlers,
                error,
                value_store,
                heavy_store,
            );
        }
        let q = (x / y).floor();
        return Ok(Value::Number(x - q * y));
    }
    match (a, b) {
        (Value::Number(n1), Value::Number(n2)) => {
            if *n2 == 0.0 {
                let error =
                    ExceptionHandler::runtime_error(frames, "Modulo by zero".to_string(), line);
                ExceptionHandler::handle_exception_null_value(
                    stack,
                    frames,
                    exception_handlers,
                    error,
                    value_store,
                    heavy_store,
                )
            } else {
                let q = (n1 / n2).floor();
                Ok(Value::Number(n1 - q * n2))
            }
        }
        _ => {
            let error = ExceptionHandler::runtime_error(
                frames,
                "Operands must be numbers".to_string(),
                line,
            );
            ExceptionHandler::handle_exception_null_value(
                stack,
                frames,
                exception_handlers,
                error,
                value_store,
                heavy_store,
            )
        }
    }
}

/// Binary power operation
pub fn binary_pow(
    a: &Value,
    b: &Value,
    frames: &mut Vec<CallFrame>,
    stack: &mut Vec<TaggedValue>,
    exception_handlers: &mut Vec<ExceptionHandler>,
    value_store: &mut ValueStore,
    heavy_store: &mut HeavyStore,
) -> Result<Value, LangError> {
    let line = get_line(frames);
    if let (Some(x), Some(y)) = (a.as_ieee_f64(), b.as_ieee_f64()) {
        return Ok(Value::Number(x.powf(y)));
    }
    match (a, b) {
        (Value::Number(n1), Value::Number(n2)) => Ok(Value::Number(n1.powf(*n2))),
        _ => {
            let error = ExceptionHandler::runtime_error(
                frames,
                "Operands must be numbers".to_string(),
                line,
            );
            ExceptionHandler::handle_exception_null_value(
                stack,
                frames,
                exception_handlers,
                error,
                value_store,
                heavy_store,
            )
        }
    }
}

/// Binary greater than operation
pub fn binary_greater(
    a: &Value,
    b: &Value,
    frames: &mut Vec<CallFrame>,
    stack: &mut Vec<TaggedValue>,
    exception_handlers: &mut Vec<ExceptionHandler>,
    value_store: &mut ValueStore,
    heavy_store: &mut HeavyStore,
) -> Result<Value, LangError> {
    let line = get_line(frames);
    match value_partial_cmp(a, b) {
        Ok(ord) => Ok(Value::Bool(ord == Ordering::Greater)),
        Err(msg) => {
            let error = ExceptionHandler::runtime_error_with_type(
                &frames,
                msg,
                line,
                ErrorType::TypeError,
            );
            ExceptionHandler::handle_exception_null_value(
                stack,
                frames,
                exception_handlers,
                error,
                value_store,
                heavy_store,
            )
        }
    }
}

/// Binary less than operation
pub fn binary_less(
    a: &Value,
    b: &Value,
    frames: &mut Vec<CallFrame>,
    stack: &mut Vec<TaggedValue>,
    exception_handlers: &mut Vec<ExceptionHandler>,
    value_store: &mut ValueStore,
    heavy_store: &mut HeavyStore,
) -> Result<Value, LangError> {
    let line = get_line(frames);
    match value_partial_cmp(a, b) {
        Ok(ord) => Ok(Value::Bool(ord == Ordering::Less)),
        Err(msg) => {
            let error = ExceptionHandler::runtime_error_with_type(
                &frames,
                msg,
                line,
                ErrorType::TypeError,
            );
            ExceptionHandler::handle_exception_null_value(
                stack,
                frames,
                exception_handlers,
                error,
                value_store,
                heavy_store,
            )
        }
    }
}

/// Binary greater than or equal operation
pub fn binary_greater_equal(
    a: &Value,
    b: &Value,
    frames: &mut Vec<CallFrame>,
    stack: &mut Vec<TaggedValue>,
    exception_handlers: &mut Vec<ExceptionHandler>,
    value_store: &mut ValueStore,
    heavy_store: &mut HeavyStore,
) -> Result<Value, LangError> {
    let line = get_line(frames);
    match value_partial_cmp(a, b) {
        Ok(ord) => Ok(Value::Bool(matches!(
            ord,
            Ordering::Greater | Ordering::Equal
        ))),
        Err(msg) => {
            let error = ExceptionHandler::runtime_error_with_type(
                &frames,
                msg,
                line,
                ErrorType::TypeError,
            );
            ExceptionHandler::handle_exception_null_value(
                stack,
                frames,
                exception_handlers,
                error,
                value_store,
                heavy_store,
            )
        }
    }
}

/// Binary less than or equal operation
pub fn binary_less_equal(
    a: &Value,
    b: &Value,
    frames: &mut Vec<CallFrame>,
    stack: &mut Vec<TaggedValue>,
    exception_handlers: &mut Vec<ExceptionHandler>,
    value_store: &mut ValueStore,
    heavy_store: &mut HeavyStore,
) -> Result<Value, LangError> {
    let line = get_line(frames);
    match value_partial_cmp(a, b) {
        Ok(ord) => Ok(Value::Bool(matches!(ord, Ordering::Less | Ordering::Equal))),
        Err(msg) => {
            let error = ExceptionHandler::runtime_error_with_type(
                &frames,
                msg,
                line,
                ErrorType::TypeError,
            );
            ExceptionHandler::handle_exception_null_value(
                stack,
                frames,
                exception_handlers,
                error,
                value_store,
                heavy_store,
            )
        }
    }
}

/// Unary negate operation
pub fn unary_negate(
    value: &Value,
    frames: &mut Vec<CallFrame>,
    stack: &mut Vec<TaggedValue>,
    exception_handlers: &mut Vec<ExceptionHandler>,
    value_store: &mut ValueStore,
    heavy_store: &mut HeavyStore,
) -> Result<Value, LangError> {
    let line = get_line(frames);
    match value {
        Value::Number(n) => Ok(Value::Number(-n)),
        Value::Int(i) => Ok(Value::Int(i.neg())),
        Value::Float(f) => Ok(Value::Float(f.neg())),
        _ => {
            let error = ExceptionHandler::runtime_error(
                frames,
                "Operand must be a number".to_string(),
                line,
            );
            ExceptionHandler::handle_exception_null_value(
                stack,
                frames,
                exception_handlers,
                error,
                value_store,
                heavy_store,
            )
        }
    }
}

/// Unary not operation
pub fn unary_not(value: &Value) -> Value {
    Value::Bool(!value.is_truthy())
}

/// Binary equal operation
pub fn binary_equal(a: &Value, b: &Value) -> Value {
    Value::Bool(a == b)
}

/// Binary not equal operation
pub fn binary_not_equal(a: &Value, b: &Value) -> Value {
    Value::Bool(a != b)
}

/// Binary or operation (short-circuit)
pub fn binary_or(a: &Value, b: &Value) -> Value {
    // Если a истинно, возвращаем a, иначе возвращаем b
    if a.is_truthy() {
        a.clone()
    } else {
        b.clone()
    }
}

/// Binary and operation (short-circuit)
pub fn binary_and(a: &Value, b: &Value) -> Value {
    // Если a ложно, возвращаем a, иначе возвращаем b
    if !a.is_truthy() {
        a.clone()
    } else {
        b.clone()
    }
}
