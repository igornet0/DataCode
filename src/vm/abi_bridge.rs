//! Мост между внутренним Value и ABI Value.
//!
//! Конвертеры только для типов, представимых в ABI (Number↔Int/Float, Bool,
//! String↔Str, Null, Array, Object как handle). Сложные типы (Tensor, Figure и т.д.)
//! во внешних ABI-модулях не экспонируются.

use std::ffi::{CStr, CString, c_void};
use std::rc::Rc;
use std::cell::RefCell;
use std::collections::HashMap;

use crate::common::value::Value;
use crate::abi::AbiValue;

/// Ошибка конвертации: тип не представим в ABI.
#[derive(Debug)]
pub enum BridgeError {
    Unrepresentable(&'static str),
    InvalidUtf8,
    InvalidHandle,
}

/// Контекст конвертации: хранит временные данные (C-строки, буферы массивов,
/// ссылки на объекты), чтобы указатели в AbiValue оставались валидными на время вызова.
pub struct AbiBridgeContext {
    cstrings: Vec<CString>,
    array_buffers: Vec<Vec<AbiValue>>,
    object_refs: Vec<Rc<RefCell<HashMap<String, Value>>>>,
}

impl AbiBridgeContext {
    pub fn new() -> Self {
        Self {
            cstrings: Vec::new(),
            array_buffers: Vec::new(),
            object_refs: Vec::new(),
        }
    }

    /// Конвертирует внутреннее Value в ABI Value.
    /// Непредставимые типы (Function, Tensor, Table, Path и т.д.) возвращают Err.
    pub fn value_to_abi(&mut self, v: &Value) -> Result<AbiValue, BridgeError> {
        match v {
            Value::Number(n) => {
                if n.fract() == 0.0 && *n >= (i64::MIN as f64) && *n <= (i64::MAX as f64) {
                    Ok(AbiValue::Int(*n as i64))
                } else {
                    Ok(AbiValue::Float(*n))
                }
            }
            Value::Bool(b) => Ok(AbiValue::Bool(*b)),
            Value::String(s) => {
                let cstr = CString::new(s.as_str()).map_err(|_| BridgeError::InvalidUtf8)?;
                self.cstrings.push(cstr);
                Ok(AbiValue::Str(self.cstrings.last().unwrap().as_ptr()))
            }
            Value::Null => Ok(AbiValue::Null),
            Value::Array(rc) => {
                let arr = rc.borrow();
                let mut abi_elems = Vec::with_capacity(arr.len());
                for elem in arr.iter() {
                    abi_elems.push(self.value_to_abi(elem)?);
                }
                self.array_buffers.push(abi_elems);
                let buf = self.array_buffers.last().unwrap();
                Ok(AbiValue::Array(buf.as_ptr() as *mut AbiValue, buf.len()))
            }
            Value::Object(rc) => {
                self.object_refs.push(Rc::clone(rc));
                let ptr = Rc::as_ptr(self.object_refs.last().unwrap()) as *mut c_void;
                Ok(AbiValue::Object(ptr))
            }
            _ => Err(BridgeError::Unrepresentable(
                "Function, NativeFunction, Path, Table, Tensor, Figure and other VM-only types are not representable in ABI",
            )),
        }
    }

    /// Конвертирует ABI Value обратно во внутреннее Value.
    /// Handle Object должен был быть получен из value_to_abi в том же контексте.
    pub fn abi_to_value(&self, a: AbiValue) -> Result<Value, BridgeError> {
        match a {
            AbiValue::Int(i) => Ok(Value::Number(i as f64)),
            AbiValue::Float(f) => Ok(Value::Number(f)),
            AbiValue::Bool(b) => Ok(Value::Bool(b)),
            AbiValue::Str(p) => {
                if p.is_null() {
                    Ok(Value::String(String::new()))
                } else {
                    let s = unsafe { CStr::from_ptr(p) }
                        .to_str()
                        .map_err(|_| BridgeError::InvalidUtf8)?;
                    Ok(Value::String(s.to_string()))
                }
            }
            AbiValue::Null => Ok(Value::Null),
            AbiValue::Array(ptr, len) => {
                if ptr.is_null() && len == 0 {
                    return Ok(Value::Array(Rc::new(RefCell::new(Vec::new()))));
                }
                if ptr.is_null() {
                    return Err(BridgeError::InvalidHandle);
                }
                let slice = unsafe { std::slice::from_raw_parts(ptr, len) };
                let mut inner = Vec::with_capacity(len);
                for &av in slice {
                    inner.push(self.abi_to_value(av)?);
                }
                Ok(Value::Array(Rc::new(RefCell::new(inner))))
            }
            AbiValue::Object(handle) => {
                if handle.is_null() {
                    return Err(BridgeError::InvalidHandle);
                }
                let ptr = handle as *const c_void;
                for rc in &self.object_refs {
                    if Rc::as_ptr(rc) as *const c_void == ptr {
                        return Ok(Value::Object(Rc::clone(rc)));
                    }
                }
                Err(BridgeError::InvalidHandle)
            }
        }
    }
}

impl Default for AbiBridgeContext {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::rc::Rc;
    use std::cell::RefCell;
    use std::collections::HashMap;

    #[test]
    fn bridge_number_bool_null() {
        let mut ctx = AbiBridgeContext::new();
        assert!(matches!(ctx.value_to_abi(&Value::Number(42.0)), Ok(AbiValue::Int(42))));
        assert!(matches!(ctx.value_to_abi(&Value::Number(3.14)), Ok(AbiValue::Float(_))));
        assert!(matches!(ctx.value_to_abi(&Value::Bool(true)), Ok(AbiValue::Bool(true))));
        assert!(matches!(ctx.value_to_abi(&Value::Null), Ok(AbiValue::Null)));
    }

    #[test]
    fn bridge_string() {
        let mut ctx = AbiBridgeContext::new();
        let abi = ctx.value_to_abi(&Value::String("hello".into())).unwrap();
        match abi {
            AbiValue::Str(p) => {
                assert!(!p.is_null());
                let s = unsafe { CStr::from_ptr(p).to_str().unwrap() };
                assert_eq!(s, "hello");
            }
            _ => panic!("expected Str"),
        }
    }

    #[test]
    fn bridge_roundtrip() {
        let mut ctx = AbiBridgeContext::new();
        let v = Value::Number(1.0);
        let a = ctx.value_to_abi(&v).unwrap();
        let v2 = ctx.abi_to_value(a).unwrap();
        assert!(matches!((&v, &v2), (Value::Number(x), Value::Number(y)) if x == y));
    }

    #[test]
    fn bridge_object_handle() {
        let mut ctx = AbiBridgeContext::new();
        let obj = Value::Object(Rc::new(RefCell::new(HashMap::new())));
        let a = ctx.value_to_abi(&obj).unwrap();
        match a {
            AbiValue::Object(h) => {
                assert!(!h.is_null());
                let v2 = ctx.abi_to_value(AbiValue::Object(h)).unwrap();
                assert!(matches!(v2, Value::Object(_)));
            }
            _ => panic!("expected Object"),
        }
    }
}
