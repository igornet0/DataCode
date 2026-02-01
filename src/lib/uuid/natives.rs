// Native functions for uuid module

use crate::common::value::Value;
use std::rc::Rc;
use std::cell::RefCell;
use uuid::Uuid;
use uuid::Variant;

fn value_to_uuid(v: &Value) -> Option<Uuid> {
    let (hi, lo) = match v {
        Value::Uuid(hi, lo) => (*hi, *lo),
        _ => return None,
    };
    Some(Uuid::from_u64_pair(hi, lo))
}

fn uuid_to_value(u: Uuid) -> Value {
    let (hi, lo) = u.as_u64_pair();
    Value::Uuid(hi, lo)
}

fn variant_to_number(v: Variant) -> f64 {
    match v {
        Variant::NCS => 0.0,
        Variant::RFC4122 => 1.0,
        Variant::Microsoft => 2.0,
        Variant::Future => 3.0,
        _ => 0.0,
    }
}

/// v4() -> UUID (random)
pub fn native_uuid_v4(_args: &[Value]) -> Value {
    uuid_to_value(Uuid::new_v4())
}

/// v7() -> UUID (time-ordered)
pub fn native_uuid_v7(_args: &[Value]) -> Value {
    uuid_to_value(Uuid::now_v7())
}

/// new() -> UUID (alias for v7)
pub fn native_uuid_new(_args: &[Value]) -> Value {
    native_uuid_v7(_args)
}

/// random() -> UUID (alias for v4)
pub fn native_uuid_random(_args: &[Value]) -> Value {
    native_uuid_v4(_args)
}

/// parse(s: string) -> UUID or null
pub fn native_uuid_parse(args: &[Value]) -> Value {
    let s = match args.first() {
        Some(Value::String(s)) => s.as_str(),
        _ => {
            crate::websocket::set_native_error("uuid.parse requires a string argument".to_string());
            return Value::Null;
        }
    };
    match Uuid::parse_str(s) {
        Ok(u) => uuid_to_value(u),
        Err(_) => Value::Null,
    }
}

/// to_string(u: UUID) -> string
pub fn native_uuid_to_string(args: &[Value]) -> Value {
    let u = match args.first() {
        Some(v) => match value_to_uuid(v) {
            Some(u) => u,
            None => {
                crate::websocket::set_native_error("uuid.to_string requires a UUID argument".to_string());
                return Value::Null;
            }
        },
        None => {
            crate::websocket::set_native_error("uuid.to_string requires one argument".to_string());
            return Value::Null;
        }
    };
    Value::String(u.hyphenated().to_string())
}

/// to_bytes(u: UUID) -> array of 16 numbers (0-255)
pub fn native_uuid_to_bytes(args: &[Value]) -> Value {
    let u = match args.first().and_then(value_to_uuid) {
        Some(u) => u,
        None => {
            crate::websocket::set_native_error("uuid.to_bytes requires a UUID argument".to_string());
            return Value::Null;
        }
    };
    let bytes = u.as_bytes();
    let arr: Vec<Value> = bytes.iter().map(|&b| Value::Number(b as f64)).collect();
    Value::Array(Rc::new(RefCell::new(arr)))
}

/// from_bytes(arr: array of 16 numbers) -> UUID or null
pub fn native_uuid_from_bytes(args: &[Value]) -> Value {
    let arr = match args.first() {
        Some(Value::Array(rc)) => rc.borrow().clone(),
        _ => {
            crate::websocket::set_native_error("uuid.from_bytes requires an array of 16 numbers".to_string());
            return Value::Null;
        }
    };
    if arr.len() != 16 {
        crate::websocket::set_native_error(format!("uuid.from_bytes: array must have 16 elements, got {}", arr.len()));
        return Value::Null;
    }
    let mut bytes = [0u8; 16];
    for (i, v) in arr.iter().enumerate() {
        let n = match v {
            Value::Number(x) if *x >= 0.0 && *x <= 255.0 && x.fract() == 0.0 => *x as u8,
            _ => {
                crate::websocket::set_native_error("uuid.from_bytes: each element must be integer 0-255".to_string());
                return Value::Null;
            }
        };
        bytes[i] = n;
    }
    uuid_to_value(Uuid::from_bytes(bytes))
}

/// version(u: UUID) -> number (1-7)
pub fn native_uuid_version(args: &[Value]) -> Value {
    let u = match args.first().and_then(value_to_uuid) {
        Some(u) => u,
        None => {
            crate::websocket::set_native_error("uuid.version requires a UUID argument".to_string());
            return Value::Null;
        }
    };
    Value::Number(u.get_version_num() as f64)
}

/// variant(u: UUID) -> number (0=NCS, 1=RFC4122, 2=Microsoft, 3=Future)
pub fn native_uuid_variant(args: &[Value]) -> Value {
    let u = match args.first().and_then(value_to_uuid) {
        Some(u) => u,
        None => {
            crate::websocket::set_native_error("uuid.variant requires a UUID argument".to_string());
            return Value::Null;
        }
    };
    Value::Number(variant_to_number(u.get_variant()))
}

/// timestamp(u: UUID) -> number (unix seconds for v1/v7) or null
pub fn native_uuid_timestamp(args: &[Value]) -> Value {
    let u = match args.first().and_then(value_to_uuid) {
        Some(u) => u,
        None => {
            crate::websocket::set_native_error("uuid.timestamp requires a UUID argument".to_string());
            return Value::Null;
        }
    };
    match u.get_timestamp() {
        Some(ts) => {
            let (secs, subsec_nanos) = ts.to_unix();
            Value::Number(secs as f64 + (subsec_nanos as f64) / 1e9)
        }
        None => Value::Null,
    }
}

/// v3(namespace: UUID, name: string) -> UUID
pub fn native_uuid_v3(args: &[Value]) -> Value {
    let (ns, name) = match (args.get(0), args.get(1)) {
        (Some(ns_v), Some(Value::String(name))) => {
            let ns = match value_to_uuid(ns_v) {
                Some(u) => u,
                None => {
                    crate::websocket::set_native_error("uuid.v3: first argument must be a UUID (namespace)".to_string());
                    return Value::Null;
                }
            };
            (ns, name.as_bytes())
        }
        _ => {
            crate::websocket::set_native_error("uuid.v3 requires (namespace: UUID, name: string)".to_string());
            return Value::Null;
        }
    };
    uuid_to_value(Uuid::new_v3(&ns, name))
}

/// v5(namespace: UUID, name: string) -> UUID
pub fn native_uuid_v5(args: &[Value]) -> Value {
    let (ns, name) = match (args.get(0), args.get(1)) {
        (Some(ns_v), Some(Value::String(name))) => {
            let ns = match value_to_uuid(ns_v) {
                Some(u) => u,
                None => {
                    crate::websocket::set_native_error("uuid.v5: first argument must be a UUID (namespace)".to_string());
                    return Value::Null;
                }
            };
            (ns, name.as_bytes())
        }
        _ => {
            crate::websocket::set_native_error("uuid.v5 requires (namespace: UUID, name: string)".to_string());
            return Value::Null;
        }
    };
    uuid_to_value(Uuid::new_v5(&ns, name))
}

/// DNS namespace constant
pub fn uuid_namespace_dns() -> Value {
    uuid_to_value(Uuid::NAMESPACE_DNS)
}

/// URL namespace constant
pub fn uuid_namespace_url() -> Value {
    uuid_to_value(Uuid::NAMESPACE_URL)
}

/// OID namespace constant
pub fn uuid_namespace_oid() -> Value {
    uuid_to_value(Uuid::NAMESPACE_OID)
}
