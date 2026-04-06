//! Мост между внутренним Value и ABI Value.
//!
//! Конвертеры только для типов, представимых в ABI (Number↔Int/Float, Bool,
//! String↔Str, Null, Array, Object как handle, ByteBuffer↔Bytes). Сложные типы (Figure и т.д.)
//! во внешних ABI-модулях не экспонируются.

use std::cell::RefCell;
use std::collections::HashMap;
use std::ffi::{c_void, CStr, CString};
use std::rc::Rc;

use crate::abi::AbiValue;
use crate::common::table::TableData;
use crate::common::value::{ByteBuffer, Value};
use crate::common::value_store::ValueStore;
use crate::vm::array_view::materialize_array_view;
use crate::vm::heavy_store::HeavyStore;
use crate::vm::store_convert::load_value;

/// Recursively replace [`Value::ArrayView`] with owned [`Value::Array`] and walk [`Value::Array`],
/// [`Value::Tuple`], and [`Value::Table`] cells so ABI serialization can represent all nested data.
pub fn materialize_value_for_abi(v: &Value, store: &ValueStore, heap: &HeavyStore) -> Value {
    match v {
        Value::ByteBuffer(bb) => Value::ByteBuffer(bb.clone()),
        Value::ArrayView(av) => {
            let m = materialize_array_view(av, store, heap);
            materialize_value_for_abi(&m, store, heap)
        }
        Value::Array(rc) => {
            let b = rc.borrow();
            let mut out = Vec::with_capacity(b.len());
            for x in b.iter() {
                out.push(materialize_value_for_abi(x, store, heap));
            }
            Value::Array(Rc::new(RefCell::new(out)))
        }
        Value::Tuple(rc) => {
            let b = rc.borrow();
            let mut out = Vec::with_capacity(b.len());
            for x in b.iter() {
                out.push(materialize_value_for_abi(x, store, heap));
            }
            Value::Tuple(Rc::new(RefCell::new(out)))
        }
        Value::Table(rc) => {
            let t = rc.borrow();
            let mut table = if t.is_view() {
                t.materialize_with(|id| load_value(id, store, heap))
            } else {
                (*t).clone()
            };
            drop(t);
            if let TableData::Owned { ref mut flat, .. } = table.data {
                for cell in flat.iter_mut() {
                    *cell = materialize_value_for_abi(cell, store, heap);
                }
            }
            Value::Table(Rc::new(RefCell::new(table)))
        }
        _ => v.clone(),
    }
}

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
    /// Keeps header + cell buffers alive for `AbiValue::Table` for the duration of the call.
    table_buffers: Vec<(Vec<AbiValue>, Vec<AbiValue>)>,
    /// Keeps `Rc<Vec<u8>>` alive for `AbiValue::Bytes` pointers into `ByteBuffer` storage.
    byte_keepalive: Vec<Rc<Vec<u8>>>,
}

impl AbiBridgeContext {
    pub fn new() -> Self {
        Self {
            cstrings: Vec::new(),
            array_buffers: Vec::new(),
            object_refs: Vec::new(),
            table_buffers: Vec::new(),
            byte_keepalive: Vec::new(),
        }
    }

    /// Конвертирует внутреннее Value в ABI Value.
    /// Непредставимые типы (Function, Table, Path и т.д.) возвращают Err.
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
            Value::ByteBuffer(bb) => {
                let ptr = if bb.len == 0 {
                    std::ptr::null()
                } else {
                    bb.bytes.as_ptr().wrapping_add(bb.offset)
                };
                self.byte_keepalive.push(Rc::clone(&bb.bytes));
                Ok(AbiValue::Bytes {
                    ptr,
                    len: bb.len,
                })
            }
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
            Value::PluginOpaque { tag, id } => Ok(AbiValue::PluginOpaque {
                tag: *tag,
                id: *id,
            }),
            Value::Path(p) => {
                let s = p.to_string_lossy();
                let cstr = CString::new(s.as_ref()).map_err(|_| BridgeError::InvalidUtf8)?;
                self.cstrings.push(cstr);
                Ok(AbiValue::Str(self.cstrings.last().unwrap().as_ptr()))
            }
            Value::Table(rc) => {
                let t = rc.borrow();
                match &t.data {
                    TableData::View { .. } => Err(BridgeError::Unrepresentable(
                        "Table View must be materialized to Owned before ABI conversion",
                    )),
                    TableData::Owned {
                        flat,
                        num_cols,
                        headers,
                        ..
                    } => {
                        let cols = *num_cols;
                        let rows = if cols == 0 {
                            0
                        } else {
                            flat.len() / cols
                        };
                        let mut header_abi = Vec::with_capacity(headers.len());
                        for h in headers.iter() {
                            let cstr = CString::new(h.as_str()).map_err(|_| BridgeError::InvalidUtf8)?;
                            self.cstrings.push(cstr);
                            header_abi.push(AbiValue::Str(self.cstrings.last().unwrap().as_ptr()));
                        }
                        let mut cells_abi = Vec::with_capacity(flat.len());
                        for cell in flat.iter() {
                            cells_abi.push(self.value_to_abi(cell)?);
                        }
                        self.table_buffers.push((header_abi, cells_abi));
                        let (h_buf, c_buf) = self.table_buffers.last().unwrap();
                        Ok(AbiValue::Table {
                            headers: h_buf.as_ptr() as *mut AbiValue,
                            headers_len: h_buf.len(),
                            cells: c_buf.as_ptr() as *mut AbiValue,
                            rows,
                            cols,
                        })
                    }
                }
            }
            _ => Err(BridgeError::Unrepresentable(
                "Function, NativeFunction, Figure and other VM-only types are not representable in ABI",
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
            AbiValue::PluginOpaque { tag, id } => Ok(Value::PluginOpaque { tag, id }),
            AbiValue::Bytes { ptr, len } => {
                if len == 0 {
                    return Ok(Value::ByteBuffer(ByteBuffer::from_vec(Vec::new())));
                }
                if ptr.is_null() {
                    return Err(BridgeError::InvalidHandle);
                }
                let slice = unsafe { std::slice::from_raw_parts(ptr, len) };
                Ok(Value::ByteBuffer(ByteBuffer::from_vec(slice.to_vec())))
            }
            AbiValue::Table { .. } => Err(BridgeError::Unrepresentable(
                "Table return values are not supported on VM←module ABI path yet",
            )),
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
    use crate::common::value::{ArrayViewData, ArrayViewSource, ByteBuffer};
    use crate::common::value_store::ValueStore;
    use crate::vm::heavy_store::HeavyStore;
    use std::cell::RefCell;
    use std::collections::HashMap;
    use std::rc::Rc;

    #[test]
    fn materialize_nested_array_view_converts_to_abi() {
        let store = ValueStore::new();
        let heap = HeavyStore::new();
        let inner = Rc::new(RefCell::new(vec![
            Value::Number(1.0),
            Value::Number(2.0),
            Value::Number(3.0),
        ]));
        let view = Value::ArrayView(ArrayViewData {
            source: ArrayViewSource::Heap(Rc::clone(&inner)),
            offset: 1,
            length: 2,
        });
        let nested = Value::Array(Rc::new(RefCell::new(vec![view])));
        let mat = materialize_value_for_abi(&nested, &store, &heap);
        let mut ctx = AbiBridgeContext::new();
        let abi = ctx.value_to_abi(&mat).expect("ABI after materialize");
        match abi {
            AbiValue::Array(ptr, len) => {
                assert_eq!(len, 1);
                let sl = unsafe { std::slice::from_raw_parts(ptr, len) };
                match sl[0] {
                    AbiValue::Array(p2, n2) => {
                        assert_eq!(n2, 2);
                        let inner_sl = unsafe { std::slice::from_raw_parts(p2, n2) };
                        assert!(matches!(inner_sl[0], AbiValue::Int(2)));
                        assert!(matches!(inner_sl[1], AbiValue::Int(3)));
                    }
                    _ => panic!("expected inner Array"),
                }
            }
            _ => panic!("expected Array"),
        }
    }

    #[test]
    fn bridge_number_bool_null() {
        let mut ctx = AbiBridgeContext::new();
        assert!(matches!(
            ctx.value_to_abi(&Value::Number(42.0)),
            Ok(AbiValue::Int(42))
        ));
        assert!(matches!(
            ctx.value_to_abi(&Value::Number(3.14)),
            Ok(AbiValue::Float(_))
        ));
        assert!(matches!(
            ctx.value_to_abi(&Value::Bool(true)),
            Ok(AbiValue::Bool(true))
        ));
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
    fn bridge_byte_buffer_roundtrip() {
        let mut ctx = AbiBridgeContext::new();
        let v = Value::ByteBuffer(ByteBuffer::from_vec(vec![1u8, 2, 3]));
        let a = ctx.value_to_abi(&v).unwrap();
        match a {
            AbiValue::Bytes { ptr, len } => {
                assert_eq!(len, 3);
                let sl = unsafe { std::slice::from_raw_parts(ptr, len) };
                assert_eq!(sl, &[1, 2, 3]);
            }
            _ => panic!("expected Bytes"),
        }
        let v2 = ctx.abi_to_value(a).unwrap();
        match v2 {
            Value::ByteBuffer(b2) => {
                assert_eq!(b2.len, 3);
                assert_eq!(&b2.bytes[b2.offset..b2.offset + b2.len], &[1, 2, 3]);
            }
            _ => panic!("expected ByteBuffer"),
        }
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
