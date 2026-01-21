// Binary and unary operations for VM

use crate::common::{error::LangError, value::Value};
use crate::vm::frame::CallFrame;
use crate::vm::exceptions::ExceptionHandler;
use std::rc::Rc;
use std::cell::RefCell;

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
    stack: &mut Vec<Value>,
    exception_handlers: &mut Vec<ExceptionHandler>,
) -> Result<Value, LangError> {
    let line = get_line(frames);
    match (a, b) {
        (Value::Number(n1), Value::Number(n2)) => Ok(Value::Number(n1 + n2)),
        (Value::String(s1), Value::String(s2)) => Ok(Value::String(format!("{}{}", s1, s2))),
        (Value::String(s), Value::Number(n)) => Ok(Value::String(format!("{}{}", s, n))),
        (Value::Number(n), Value::String(s)) => Ok(Value::String(format!("{}{}", n, s))),
        (Value::Array(arr1), Value::Array(arr2)) => {
            // Array concatenation
            let mut result = arr1.borrow().clone();
            result.extend_from_slice(&arr2.borrow());
            Ok(Value::Array(Rc::new(RefCell::new(result))))
        },
        _ => {
            let error = ExceptionHandler::runtime_error(
                frames,
                "Operands must be numbers or strings".to_string(),
                line,
            );
            match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error) {
                Ok(()) => Ok(Value::Null),
                Err(e) => Err(e),
            }
        }
    }
}

/// Binary subtraction operation
pub fn binary_sub(
    a: &Value,
    b: &Value,
    frames: &mut Vec<CallFrame>,
    stack: &mut Vec<Value>,
    exception_handlers: &mut Vec<ExceptionHandler>,
) -> Result<Value, LangError> {
    let line = get_line(frames);
    match (a, b) {
        (Value::Number(n1), Value::Number(n2)) => Ok(Value::Number(n1 - n2)),
        _ => {
            let error = ExceptionHandler::runtime_error(
                frames,
                "Operands must be numbers".to_string(),
                line,
            );
            match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error) {
                Ok(()) => Ok(Value::Null),
                Err(e) => Err(e),
            }
        }
    }
}

/// Binary multiplication operation
pub fn binary_mul(
    a: &Value,
    b: &Value,
    frames: &mut Vec<CallFrame>,
    stack: &mut Vec<Value>,
    exception_handlers: &mut Vec<ExceptionHandler>,
) -> Result<Value, LangError> {
    let line = get_line(frames);
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
        (Value::Number(n), Value::String(s)) => {
            let count = *n as i64;
            if count <= 0 {
                Ok(Value::String(String::new()))
            } else {
                Ok(Value::String(s.repeat(count as usize)))
            }
        }
        _ => {
            let error = ExceptionHandler::runtime_error(
                frames,
                "Operands must be numbers, or string and number for repetition".to_string(),
                line,
            );
            match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error) {
                Ok(()) => Ok(Value::Null),
                Err(e) => Err(e),
            }
        }
    }
}

/// Binary division operation
pub fn binary_div(
    a: &Value,
    b: &Value,
    frames: &mut Vec<CallFrame>,
    stack: &mut Vec<Value>,
    exception_handlers: &mut Vec<ExceptionHandler>,
) -> Result<Value, LangError> {
    let line = get_line(frames);
    match (a, b) {
        (Value::Number(n1), Value::Number(n2)) => {
            if *n2 == 0.0 {
                let error = ExceptionHandler::runtime_error(
                    frames,
                    "Division by zero".to_string(),
                    line,
                );
                match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error) {
                    Ok(()) => Ok(Value::Null),
                    Err(e) => Err(e),
                }
            } else {
                Ok(Value::Number(n1 / n2))
            }
        }
        // Tensor / Number
        (Value::Tensor(t1), Value::Number(n2)) => {
            if *n2 == 0.0 {
                let error = ExceptionHandler::runtime_error(
                    frames,
                    "Division by zero".to_string(),
                    line,
                );
                match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error) {
                    Ok(()) => Ok(Value::Null),
                    Err(e) => Err(e),
                }
            } else {
                match t1.borrow().div_scalar(*n2 as f32) {
                    Ok(result) => Ok(Value::Tensor(std::rc::Rc::new(std::cell::RefCell::new(result)))),
                    Err(e) => {
                        let error = ExceptionHandler::runtime_error(
                            frames,
                            e,
                            line,
                        );
                        match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error) {
                            Ok(()) => Ok(Value::Null),
                            Err(e) => Err(e),
                        }
                    }
                }
            }
        }
        // Number / Tensor
        (Value::Number(n1), Value::Tensor(t2)) => {
            let tensor_ref = t2.borrow();
            match tensor_ref.to_cpu() {
                Ok(cpu_tensor) => {
                    // Check for division by zero
                    if cpu_tensor.data.iter().any(|&x| x == 0.0) {
                        let error = ExceptionHandler::runtime_error(
                            frames,
                            "Division by zero".to_string(),
                            line,
                        );
                        match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error) {
                            Ok(()) => Ok(Value::Null),
                            Err(e) => Err(e),
                        }
                    } else {
                        // Create a tensor filled with n1 and divide element-wise
                        let scalar = *n1 as f32;
                        // OPTIMIZATION: Pre-allocate vector with known capacity to avoid reallocations
                        let len = cpu_tensor.data.len();
                        let mut data = Vec::with_capacity(len);
                        data.extend(cpu_tensor.data.iter().map(|&x| scalar / x));
                        Ok(Value::Tensor(std::rc::Rc::new(std::cell::RefCell::new(
                            crate::ml::tensor::Tensor::from_slice(&data, &cpu_tensor.shape)
                        ))))
                    }
                }
                Err(e) => {
                    let error = ExceptionHandler::runtime_error(
                        frames,
                        e,
                        line,
                    );
                    match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error) {
                        Ok(()) => Ok(Value::Null),
                        Err(e) => Err(e),
                    }
                }
            }
        }
        // Tensor / Tensor
        (Value::Tensor(t1), Value::Tensor(t2)) => {
            match t1.borrow().div(&t2.borrow()) {
                Ok(result) => Ok(Value::Tensor(std::rc::Rc::new(std::cell::RefCell::new(result)))),
                Err(e) => {
                    let error = ExceptionHandler::runtime_error(
                        frames,
                        e,
                        line,
                    );
                    match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error) {
                        Ok(()) => Ok(Value::Null),
                        Err(e) => Err(e),
                    }
                }
            }
        }
        // Конкатенация путей: Path / String -> Path
        (Value::Path(p), Value::String(s)) => {
            let mut new_path = p.clone();
            new_path.push(s);
            Ok(Value::Path(new_path))
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
                "Operands must be numbers, tensors, or paths".to_string(),
                line,
            );
            match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error) {
                Ok(()) => Ok(Value::Null),
                Err(e) => Err(e),
            }
        }
    }
}

/// Binary integer division operation
pub fn binary_int_div(
    a: &Value,
    b: &Value,
    frames: &mut Vec<CallFrame>,
    stack: &mut Vec<Value>,
    exception_handlers: &mut Vec<ExceptionHandler>,
) -> Result<Value, LangError> {
    let line = get_line(frames);
    match (a, b) {
        (Value::Number(n1), Value::Number(n2)) => {
            if *n2 == 0.0 {
                let error = ExceptionHandler::runtime_error(
                    frames,
                    "Division by zero".to_string(),
                    line,
                );
                match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error) {
                    Ok(()) => Ok(Value::Null),
                    Err(e) => Err(e),
                }
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
            match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error) {
                Ok(()) => Ok(Value::Null),
                Err(e) => Err(e),
            }
        }
    }
}

/// Binary modulo operation
pub fn binary_mod(
    a: &Value,
    b: &Value,
    frames: &mut Vec<CallFrame>,
    stack: &mut Vec<Value>,
    exception_handlers: &mut Vec<ExceptionHandler>,
) -> Result<Value, LangError> {
    let line = get_line(frames);
    match (a, b) {
        (Value::Number(n1), Value::Number(n2)) => {
            if *n2 == 0.0 {
                let error = ExceptionHandler::runtime_error(
                    frames,
                    "Modulo by zero".to_string(),
                    line,
                );
                match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error) {
                    Ok(()) => Ok(Value::Null),
                    Err(e) => Err(e),
                }
            } else {
                Ok(Value::Number(n1 % n2))
            }
        }
        _ => {
            let error = ExceptionHandler::runtime_error(
                frames,
                "Operands must be numbers".to_string(),
                line,
            );
            match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error) {
                Ok(()) => Ok(Value::Null),
                Err(e) => Err(e),
            }
        }
    }
}

/// Binary power operation
pub fn binary_pow(
    a: &Value,
    b: &Value,
    frames: &mut Vec<CallFrame>,
    stack: &mut Vec<Value>,
    exception_handlers: &mut Vec<ExceptionHandler>,
) -> Result<Value, LangError> {
    let line = get_line(frames);
    match (a, b) {
        (Value::Number(n1), Value::Number(n2)) => Ok(Value::Number(n1.powf(*n2))),
        _ => {
            let error = ExceptionHandler::runtime_error(
                frames,
                "Operands must be numbers".to_string(),
                line,
            );
            match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error) {
                Ok(()) => Ok(Value::Null),
                Err(e) => Err(e),
            }
        }
    }
}

/// Binary greater than operation
pub fn binary_greater(
    a: &Value,
    b: &Value,
    frames: &mut Vec<CallFrame>,
    stack: &mut Vec<Value>,
    exception_handlers: &mut Vec<ExceptionHandler>,
) -> Result<Value, LangError> {
    let line = get_line(frames);
    match (a, b) {
        (Value::Number(n1), Value::Number(n2)) => Ok(Value::Bool(n1 > n2)),
        (Value::String(s1), Value::String(s2)) => Ok(Value::Bool(s1 > s2)),
        _ => {
            let error = ExceptionHandler::runtime_error(
                frames,
                "Operands must be numbers or strings".to_string(),
                line,
            );
            match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error) {
                Ok(()) => Ok(Value::Null),
                Err(e) => Err(e),
            }
        }
    }
}

/// Binary less than operation
pub fn binary_less(
    a: &Value,
    b: &Value,
    frames: &mut Vec<CallFrame>,
    stack: &mut Vec<Value>,
    exception_handlers: &mut Vec<ExceptionHandler>,
) -> Result<Value, LangError> {
    let line = get_line(frames);
    match (a, b) {
        (Value::Number(n1), Value::Number(n2)) => Ok(Value::Bool(n1 < n2)),
        (Value::String(s1), Value::String(s2)) => Ok(Value::Bool(s1 < s2)),
        _ => {
            let error = ExceptionHandler::runtime_error(
                frames,
                "Operands must be numbers or strings".to_string(),
                line,
            );
            match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error) {
                Ok(()) => Ok(Value::Null),
                Err(e) => Err(e),
            }
        }
    }
}

/// Binary greater than or equal operation
pub fn binary_greater_equal(
    a: &Value,
    b: &Value,
    frames: &mut Vec<CallFrame>,
    stack: &mut Vec<Value>,
    exception_handlers: &mut Vec<ExceptionHandler>,
) -> Result<Value, LangError> {
    let line = get_line(frames);
    match (a, b) {
        (Value::Number(n1), Value::Number(n2)) => Ok(Value::Bool(n1 >= n2)),
        (Value::String(s1), Value::String(s2)) => Ok(Value::Bool(s1 >= s2)),
        _ => {
            let error = ExceptionHandler::runtime_error(
                frames,
                "Operands must be numbers or strings".to_string(),
                line,
            );
            match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error) {
                Ok(()) => Ok(Value::Null),
                Err(e) => Err(e),
            }
        }
    }
}

/// Binary less than or equal operation
pub fn binary_less_equal(
    a: &Value,
    b: &Value,
    frames: &mut Vec<CallFrame>,
    stack: &mut Vec<Value>,
    exception_handlers: &mut Vec<ExceptionHandler>,
) -> Result<Value, LangError> {
    let line = get_line(frames);
    match (a, b) {
        (Value::Number(n1), Value::Number(n2)) => Ok(Value::Bool(n1 <= n2)),
        (Value::String(s1), Value::String(s2)) => Ok(Value::Bool(s1 <= s2)),
        _ => {
            let error = ExceptionHandler::runtime_error(
                frames,
                "Operands must be numbers or strings".to_string(),
                line,
            );
            match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error) {
                Ok(()) => Ok(Value::Null),
                Err(e) => Err(e),
            }
        }
    }
}

/// Unary negate operation
pub fn unary_negate(
    value: &Value,
    frames: &mut Vec<CallFrame>,
    stack: &mut Vec<Value>,
    exception_handlers: &mut Vec<ExceptionHandler>,
) -> Result<Value, LangError> {
    let line = get_line(frames);
    match value {
        Value::Number(n) => Ok(Value::Number(-n)),
        _ => {
            let error = ExceptionHandler::runtime_error(
                frames,
                "Operand must be a number".to_string(),
                line,
            );
            match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error) {
                Ok(()) => Ok(Value::Null),
                Err(e) => Err(e),
            }
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
