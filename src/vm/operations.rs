// Binary and unary operations for VM (stack as Vec<TaggedValue>; handle_exception needs value_store/heavy_store)

use crate::common::{
    error::{ErrorType, LangError},
    numeric::{divide_by_zero_raises, ieee_div_quotient_value, IntValue},
    value::Value,
    value_ord::value_partial_cmp,
    value_store::ValueStore,
    typed_column::F64Kind,
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
    if let Some(r) = column_arith(a, b, ColumnOp::Add, value_store, heavy_store) {
        return r.or_else(|e| {
            raise_column_arith_error(e, line, frames, stack, exception_handlers, value_store, heavy_store)
        });
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
    if let Some(r) = column_arith(a, b, ColumnOp::Sub, value_store, heavy_store) {
        return r.or_else(|e| {
            raise_column_arith_error(e, line, frames, stack, exception_handlers, value_store, heavy_store)
        });
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
    if let Some(r) = column_arith(a, b, ColumnOp::Mul, value_store, heavy_store) {
        return r.or_else(|e| {
            raise_column_arith_error(e, line, frames, stack, exception_handlers, value_store, heavy_store)
        });
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
    if let Some(r) = column_arith(a, b, ColumnOp::Div, value_store, heavy_store) {
        return r.or_else(|e| {
            raise_column_arith_error(e, line, frames, stack, exception_handlers, value_store, heavy_store)
        });
    }
    if let (Some(x), Some(y)) = (a.as_ieee_f64(), b.as_ieee_f64()) {
        if y == 0.0 && divide_by_zero_raises(a, b) {
            let error =
                ExceptionHandler::runtime_error_with_type(
                frames,
                "Division by zero".to_string(),
                line,
                crate::common::error::ErrorType::ZeroDivisionError,
            );
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
                    ExceptionHandler::runtime_error_with_type(
                frames,
                "Division by zero".to_string(),
                line,
                crate::common::error::ErrorType::ZeroDivisionError,
            );
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
                ExceptionHandler::runtime_error_with_type(
                frames,
                "Division by zero".to_string(),
                line,
                crate::common::error::ErrorType::ZeroDivisionError,
            );
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
                    ExceptionHandler::runtime_error_with_type(
                frames,
                "Division by zero".to_string(),
                line,
                crate::common::error::ErrorType::ZeroDivisionError,
            );
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
                ExceptionHandler::runtime_error_with_type(
                frames,
                "Modulo by zero".to_string(),
                line,
                crate::common::error::ErrorType::ZeroDivisionError,
            );
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
                    ExceptionHandler::runtime_error_with_type(
                frames,
                "Modulo by zero".to_string(),
                line,
                crate::common::error::ErrorType::ZeroDivisionError,
            );
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

#[derive(Clone, Copy, PartialEq)]
enum ColumnOp {
    Add,
    Sub,
    Mul,
    Div,
}

enum ColumnOperand<'a> {
    Values(Vec<Value>),
    /// Column of an owned table read in place from its row-major buffer.
    Strided {
        flat: &'a [Value],
        stride: usize,
        offset: usize,
        len: usize,
    },
    Slice(&'a [Value]),
    /// Typed column read without building its `Value` cells.
    F64(&'a [f64], Option<&'a [bool]>, F64Kind),
    I64(&'a [i64], Option<&'a [bool]>),
    Scalar(&'a Value),
}

/// One operand cell: borrowed, or built on the fly from a typed column.
enum Cell<'a> {
    Ref(&'a Value),
    Owned(Value),
}

impl std::ops::Deref for Cell<'_> {
    type Target = Value;
    #[inline]
    fn deref(&self) -> &Value {
        match self {
            Cell::Ref(v) => v,
            Cell::Owned(v) => v,
        }
    }
}

#[inline]
fn is_null_at(nulls: Option<&[bool]>, i: usize) -> bool {
    nulls.is_some_and(|n| n[i])
}

impl ColumnOperand<'_> {
    #[inline]
    fn get(&self, i: usize) -> Cell<'_> {
        match self {
            ColumnOperand::Values(v) => Cell::Ref(&v[i]),
            ColumnOperand::Strided { flat, stride, offset, .. } => Cell::Ref(&flat[i * stride + offset]),
            ColumnOperand::Slice(v) => Cell::Ref(&v[i]),
            ColumnOperand::F64(d, n, kind) => Cell::Owned(if is_null_at(*n, i) {
                Value::Null
            } else {
                kind.cell(d[i])
            }),
            ColumnOperand::I64(d, n) => Cell::Owned(if is_null_at(*n, i) {
                Value::Null
            } else {
                Value::Int(IntValue::Finite(d[i]))
            }),
            ColumnOperand::Scalar(s) => Cell::Ref(s),
        }
    }

    #[inline]
    fn len(&self) -> Option<usize> {
        match self {
            ColumnOperand::Values(v) => Some(v.len()),
            ColumnOperand::Strided { len, .. } => Some(*len),
            ColumnOperand::Slice(v) => Some(v.len()),
            ColumnOperand::F64(d, ..) => Some(d.len()),
            ColumnOperand::I64(d, _) => Some(d.len()),
            ColumnOperand::Scalar(_) => None,
        }
    }
}

/// Borrow held for the duration of an element-wise op so operands can be read without copying.
enum OperandBorrow<'a> {
    Table(std::cell::Ref<'a, crate::common::table::Table>),
    Array(std::cell::Ref<'a, Vec<Value>>),
    None,
}

fn borrow_operand(v: &Value) -> OperandBorrow<'_> {
    match v {
        // Views are materialized through get_column (needs borrow_mut), so only owned tables are held.
        Value::ColumnReference { table, .. } if !table.borrow().is_view() => OperandBorrow::Table(table.borrow()),
        Value::Array(arr) => OperandBorrow::Array(arr.borrow()),
        _ => OperandBorrow::None,
    }
}

fn column_operand<'a>(
    v: &'a Value,
    held: &'a OperandBorrow<'a>,
    value_store: &ValueStore,
    heavy_store: &HeavyStore,
) -> Result<ColumnOperand<'a>, String> {
    match (v, held) {
        (Value::ColumnReference { column_name, .. }, OperandBorrow::Table(t)) => {
            let t: &'a crate::common::table::Table = t;
            if let Some(col) = t.typed_column(column_name) {
                for kind in [F64Kind::Float, F64Kind::Number] {
                    if let Some((d, n)) = col.as_f64_of(kind) {
                        return Ok(ColumnOperand::F64(d, n, kind));
                    }
                }
                if let Some((d, n)) = col.as_i64() {
                    return Ok(ColumnOperand::I64(d, n));
                }
            }
            match t.column_view(column_name) {
                Some(crate::common::table::ColumnView::Owned { flat, num_cols, col_index, len }) => {
                    Ok(ColumnOperand::Strided { flat, stride: num_cols, offset: col_index, len })
                }
                Some(_) => unreachable!("borrow_operand holds only owned tables"),
                None => Err(format!("Column '{}' not found", column_name)),
            }
        }
        (Value::ColumnReference { table, column_name }, _) => {
            let mut t = table.borrow_mut();
            crate::vm::table_ops::get_column(&mut t, column_name, value_store, heavy_store)
                .map(ColumnOperand::Values)
                .ok_or_else(|| format!("Column '{}' not found", column_name))
        }
        (_, OperandBorrow::Array(arr)) => Ok(ColumnOperand::Slice(arr.as_slice())),
        (other, _) => Ok(ColumnOperand::Scalar(other)),
    }
}

/// Element-wise `+ - * /` when either operand is a table column (`t["a"] * t["b"]`, `t["a"] * 2`).
/// The other side is a column, a scalar or an equal-length array. Null in either operand gives
/// null in that row (like a pandas NaN). Returns None when neither side is a column.
/// Int ⊕ Int stays Int (falling back to float on overflow); `/` follows scalar division.
fn column_arith(
    a: &Value,
    b: &Value,
    op: ColumnOp,
    value_store: &ValueStore,
    heavy_store: &HeavyStore,
) -> Option<Result<Value, (String, ErrorType)>> {
    if !matches!(a, Value::ColumnReference { .. }) && !matches!(b, Value::ColumnReference { .. }) {
        return None;
    }
    Some(column_arith_values(a, b, op, value_store, heavy_store))
}

fn column_arith_values(
    a: &Value,
    b: &Value,
    op: ColumnOp,
    value_store: &ValueStore,
    heavy_store: &HeavyStore,
) -> Result<Value, (String, ErrorType)> {
    let runtime = |msg: String| (msg, ErrorType::RuntimeError);
    let held_a = borrow_operand(a);
    let held_b = borrow_operand(b);
    let left = column_operand(a, &held_a, value_store, heavy_store).map_err(runtime)?;
    let right = column_operand(b, &held_b, value_store, heavy_store).map_err(runtime)?;
    let len = match (left.len(), right.len()) {
        (Some(l), Some(r)) => {
            if l != r {
                return Err(runtime(format!(
                    "Column operands must have equal length, got {} and {}",
                    l, r
                )));
            }
            l
        }
        (Some(n), None) | (None, Some(n)) => n,
        _ => 0,
    };
    // Rows run on f64 while every result is of one kind (all Float or all Number); the first
    // other result (Int, the other kind, error) hands the rest to `column_elem`, so the output
    // matches it cell for cell.
    let (data, nulls, kind, start) = match op {
        ColumnOp::Add => float_prefix(&left, &right, len, |p, q| p + q),
        ColumnOp::Sub => float_prefix(&left, &right, len, |p, q| p - q),
        ColumnOp::Mul => float_prefix(&left, &right, len, |p, q| p * q),
        ColumnOp::Div => (Vec::new(), None, None, 0),
    };
    let column = if let (true, Some(kind)) = (start == len, kind) {
        // Float cells are Float(classify(r)) and Number cells Number(r); both round-trip every f64.
        crate::common::typed_column::TypedColumn::f64_of(kind, Rc::from(data), nulls.map(Rc::from))
    } else {
        let mut out = crate::common::typed_column::ColumnBuilder::with_capacity(len);
        for (k, &d) in data.iter().enumerate().take(start) {
            out.push(match kind {
                Some(kind) if !is_null_at(nulls.as_deref(), k) => kind.cell(d),
                _ => Value::Null,
            });
        }
        drop(data);
        for i in start..len {
            out.push(column_elem(&left.get(i), &right.get(i), op)?);
        }
        out.finish()
    };
    // The result is itself a column (of a fresh one-column table) so expressions chain:
    // `t["a"] * t["b"] + 1`, and `t!["c"] = ...` accepts it like any other column.
    let column_name = match (a, b) {
        (Value::ColumnReference { column_name, .. }, _) | (_, Value::ColumnReference { column_name, .. }) => {
            column_name.clone()
        }
        _ => "value".to_string(),
    };
    let table = crate::common::table::Table::from_typed_column(column_name.clone(), column);
    Ok(Value::ColumnReference {
        table: Rc::new(RefCell::new(table)),
        column_name,
    })
}

/// Numeric class of one operand cell, as `column_elem` sees it.
#[derive(Clone, Copy)]
enum Num {
    Null,
    Float(f64),
    /// Finite Int: Int ⊕ Int takes the checked integer path.
    Int(f64),
    /// Number, or a non-finite Int: numeric, gives Number unless the other side is Float.
    Number(f64),
    Other,
}

#[inline]
fn num_of(v: &Value) -> Num {
    match v {
        Value::Null => Num::Null,
        Value::Float(f) => Num::Float(f.as_raw_f64()),
        Value::Int(IntValue::Finite(_)) => v.as_ieee_f64().map_or(Num::Other, Num::Int),
        Value::Number(_) | Value::Int(_) => v.as_ieee_f64().map_or(Num::Other, Num::Number),
        _ => Num::Other,
    }
}

/// Result kind and operands of a non-null `column_elem` result that is computed on f64;
/// None for Int ⊕ Int, a null or a non-numeric operand.
#[inline]
fn f64_result(x: Num, y: Num) -> Option<(F64Kind, f64, f64)> {
    match (x, y) {
        (Num::Float(p), Num::Float(q) | Num::Int(q) | Num::Number(q)) | (Num::Int(p) | Num::Number(p), Num::Float(q)) => {
            Some((F64Kind::Float, p, q))
        }
        (Num::Number(p), Num::Int(q) | Num::Number(q)) | (Num::Int(p), Num::Number(q)) => Some((F64Kind::Number, p, q)),
        _ => None,
    }
}

/// Kind of an F64 column ⊕ a non-null numeric value of class `other`.
#[inline]
fn f64_with(kind: F64Kind, other: Num) -> Option<(F64Kind, f64)> {
    let x = match kind {
        F64Kind::Float => Num::Float(0.0),
        F64Kind::Number => Num::Number(0.0),
    };
    let q = match other {
        Num::Float(q) | Num::Int(q) | Num::Number(q) => q,
        _ => return None,
    };
    f64_result(x, other).map(|(k, ..)| (k, q))
}

impl ColumnOperand<'_> {
    #[inline]
    fn num_at(&self, i: usize) -> Num {
        match self {
            ColumnOperand::Values(v) => num_of(&v[i]),
            ColumnOperand::Slice(v) => num_of(&v[i]),
            ColumnOperand::Strided { flat, stride, offset, .. } => num_of(&flat[i * stride + offset]),
            ColumnOperand::F64(d, n, kind) => {
                if is_null_at(*n, i) {
                    Num::Null
                } else if *kind == F64Kind::Float {
                    Num::Float(d[i])
                } else {
                    Num::Number(d[i])
                }
            }
            ColumnOperand::I64(d, n) => {
                if is_null_at(*n, i) {
                    Num::Null
                } else {
                    Num::Int(d[i] as f64)
                }
            }
            ColumnOperand::Scalar(s) => num_of(s),
        }
    }
}

/// Computes rows `0..start` on f64 while each result is null or of one kind: Float (a Float
/// operand, the other numeric) or Number (Number with Int or Number). Returns (data, nulls,
/// kind of the non-null results, start); `start < len` marks the first row the f64 path cannot
/// represent.
fn float_prefix<F: Fn(f64, f64) -> f64 + Copy>(
    left: &ColumnOperand<'_>,
    right: &ColumnOperand<'_>,
    len: usize,
    f: F,
) -> (Vec<f64>, Option<Vec<bool>>, Option<F64Kind>, usize) {
    let scalar = |o: &ColumnOperand<'_>| match o {
        ColumnOperand::Scalar(s) => Some(num_of(s)),
        _ => None,
    };
    let (ls, rs) = (scalar(left), scalar(right));
    let some = |k: F64Kind| (len > 0).then_some(k);
    // Null-free typed columns against a typed column or a numeric scalar: plain slice loops.
    match (left, right, ls, rs) {
        (ColumnOperand::F64(a, None, ka), ColumnOperand::F64(b, None, kb), ..) => {
            let kind = if *ka == F64Kind::Float || *kb == F64Kind::Float { F64Kind::Float } else { F64Kind::Number };
            let data: Vec<f64> = a.iter().zip(b.iter()).map(|(&p, &q)| f(p, q)).collect();
            return (data, None, some(kind), len);
        }
        (ColumnOperand::F64(a, None, ka), _, _, Some(s)) => {
            if let Some((kind, q)) = f64_with(*ka, s) {
                let data: Vec<f64> = a.iter().map(|&p| f(p, q)).collect();
                return (data, None, some(kind), len);
            }
        }
        (_, ColumnOperand::F64(b, None, kb), Some(s), _) => {
            if let Some((kind, p)) = f64_with(*kb, s) {
                let data: Vec<f64> = b.iter().map(|&q| f(p, q)).collect();
                return (data, None, some(kind), len);
            }
        }
        (ColumnOperand::I64(a, None), _, _, Some(s)) => {
            if let Some((kind, _, q)) = f64_result(Num::Int(0.0), s) {
                let data: Vec<f64> = a.iter().map(|&p| f(p as f64, q)).collect();
                return (data, None, some(kind), len);
            }
        }
        (_, ColumnOperand::I64(b, None), Some(s), _) => {
            if let Some((kind, p, _)) = f64_result(s, Num::Int(0.0)) {
                let data: Vec<f64> = b.iter().map(|&q| f(p, q as f64)).collect();
                return (data, None, some(kind), len);
            }
        }
        _ => {}
    }
    let mut data = Vec::with_capacity(len);
    let mut nulls: Option<Vec<bool>> = None;
    let mut kind: Option<F64Kind> = None;
    for i in 0..len {
        let x = ls.unwrap_or_else(|| left.num_at(i));
        let y = rs.unwrap_or_else(|| right.num_at(i));
        if matches!(x, Num::Null) || matches!(y, Num::Null) {
            nulls.get_or_insert_with(|| vec![false; i]).push(true);
            data.push(0.0);
            continue;
        }
        let Some((k, p, q)) = f64_result(x, y) else {
            return (data, nulls, kind, i);
        };
        if *kind.get_or_insert(k) != k {
            return (data, nulls, kind, i);
        }
        data.push(f(p, q));
        if let Some(n) = &mut nulls {
            n.push(false);
        }
    }
    (data, nulls, kind, len)
}

#[inline]
fn column_elem(x: &Value, y: &Value, op: ColumnOp) -> Result<Value, (String, ErrorType)> {
    if matches!(x, Value::Null) || matches!(y, Value::Null) {
        return Ok(Value::Null);
    }
    if let (Value::Int(IntValue::Finite(p)), Value::Int(IntValue::Finite(q))) = (x, y) {
        let r = match op {
            ColumnOp::Add => p.checked_add(*q),
            ColumnOp::Sub => p.checked_sub(*q),
            ColumnOp::Mul => p.checked_mul(*q),
            ColumnOp::Div => None,
        };
        if let Some(r) = r {
            return Ok(Value::Int(IntValue::Finite(r)));
        }
    }
    let (Some(p), Some(q)) = (x.as_ieee_f64(), y.as_ieee_f64()) else {
        return Err((
            format!(
                "Column arithmetic needs numeric values, got {} and {}",
                crate::vm::calls::get_type_name_value(x),
                crate::vm::calls::get_type_name_value(y)
            ),
            ErrorType::TypeError,
        ));
    };
    let r = match op {
        ColumnOp::Add => p + q,
        ColumnOp::Sub => p - q,
        ColumnOp::Mul => p * q,
        ColumnOp::Div => {
            if q == 0.0 && divide_by_zero_raises(x, y) {
                return Err(("Division by zero".to_string(), ErrorType::ZeroDivisionError));
            }
            return Ok(ieee_div_quotient_value(x, y, p, q));
        }
    };
    if matches!(x, Value::Float(_)) || matches!(y, Value::Float(_)) {
        Ok(Value::Float(crate::common::numeric::FloatValue::classify_f64(r)))
    } else {
        Ok(Value::Number(r))
    }
}

fn raise_column_arith_error(
    (msg, ty): (String, ErrorType),
    line: usize,
    frames: &mut Vec<CallFrame>,
    stack: &mut Vec<TaggedValue>,
    exception_handlers: &mut Vec<ExceptionHandler>,
    value_store: &mut ValueStore,
    heavy_store: &mut HeavyStore,
) -> Result<Value, LangError> {
    let error = ExceptionHandler::runtime_error_with_type(frames, msg, line, ty);
    ExceptionHandler::handle_exception_null_value(
        stack,
        frames,
        exception_handlers,
        error,
        value_store,
        heavy_store,
    )
}
