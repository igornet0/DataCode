//! Immutable / hashable classification and deterministic object-key hashing (no VM dependency).

use std::hash::{Hash, Hasher};

use crate::common::numeric;
use crate::common::object_map::ObjectMap;
use crate::common::value::{ObjectKind, Value};
use crate::common::value_store::{ValueCell, ValueId};

/// Metadata for a Datacode type surface (immutable ⟺ not mutable).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DcType {
    pub mutable: bool,
    pub hashable: bool,
}

impl DcType {
    #[inline]
    pub fn immutable(self) -> bool {
        !self.mutable
    }
}

pub fn type_of_value(v: &Value) -> DcType {
    let mutable = is_mutable_value(v);
    let hashable = is_hashable_value(v);
    DcType { mutable, hashable }
}

/// Whether `v` is treated as mutable (forbidden as object key unless frozen object).
pub fn is_mutable_value(v: &Value) -> bool {
    match v {
        Value::Int(_)
        | Value::Float(_)
        | Value::Number(_)
        | Value::Bool(_)
        | Value::String(_)
        | Value::Uuid(_, _)
        | Value::Date(_)
        | Value::Duration(_)
        | Value::Null
        | Value::Ellipsis
        | Value::Path(_) => false,
        Value::ObjectFieldList { .. } => false,
        Value::Tuple(rc) => rc.borrow().iter().any(is_mutable_value),
        Value::Object(rc) => match &*rc.borrow() {
            ObjectKind::Legacy(_) | ObjectKind::Inline(_) => true,
            ObjectKind::Bucket(map) => !map.is_frozen(),
        },
        Value::Set(_) => true,
        Value::Array(_)
        | Value::ArrayView(_)
        | Value::Table(_)
        | Value::Function(_)
        | Value::ModuleFunction { .. }
        | Value::NativeFunction(_)
        | Value::ColumnReference { .. }
        | Value::PluginOpaque { .. }
        | Value::Window(_)
        | Value::Image(_)
        | Value::Figure(_)
        | Value::Axis(_)
        | Value::DatabaseEngine(_)
        | Value::DatabaseCluster(_)
        | Value::Enumerate { .. }
        | Value::Iterable(_)
        | Value::Generator(_)
        | Value::ByteBuffer(_) => true,
    }
}

/// Fast hashability for dict/set keys stored as [`ValueCell`] (no [`Value`] materialize).
#[inline]
pub fn value_cell_is_hashable_key(cell: &ValueCell) -> bool {
    match cell {
        ValueCell::Int(_)
        | ValueCell::Float(_)
        | ValueCell::Number(_)
        | ValueCell::Bool(_)
        | ValueCell::String(_)
        | ValueCell::Null
        | ValueCell::Path(_)
        | ValueCell::Uuid(_, _)
        | ValueCell::Date { .. }
        | ValueCell::Duration { .. }
        | ValueCell::Ellipsis => true,
        ValueCell::Object(map) => map.is_frozen(),
        ValueCell::Tuple(_) => false,
        _ => false,
    }
}

pub fn is_hashable_value(v: &Value) -> bool {
    match v {
        Value::Int(_)
        | Value::Float(_)
        | Value::Number(_)
        | Value::Bool(_)
        | Value::String(_)
        | Value::Uuid(_, _)
        | Value::Date(_)
        | Value::Duration(_)
        | Value::Null
        | Value::Ellipsis
        | Value::Path(_) => true,
        Value::Tuple(rc) => rc.borrow().iter().all(is_hashable_value),
        Value::Object(rc) => match &*rc.borrow() {
            ObjectKind::Legacy(_) | ObjectKind::Inline(_) => false,
            ObjectKind::Bucket(map) => map.is_frozen(),
        },
        Value::Set(_) => false,
        _ => false,
    }
}

#[inline]
pub fn hash_mix(mut h: u64, x: u64) -> u64 {
    h ^= x.wrapping_mul(0x100000001b3);
    h.wrapping_mul(0xc759065465161607)
}

pub fn hash_finish(tag: u8, payload: u64) -> u64 {
    hash_mix(0xcbf29ce484222325 ^ tag as u64, payload)
}

/// Deterministic hash for object keys from a materialized [`Value`].
pub fn object_key_hash_value(v: &Value) -> Option<u64> {
    if !is_hashable_value(v) {
        return None;
    }
    Some(hash_value_tree(v))
}

fn hash_value_tree(v: &Value) -> u64 {
    match v {
        Value::Number(_) | Value::Int(_) | Value::Float(_) => {
            numeric::hash_numeric_key_value(v).expect("numeric key must be hashable")
        }
        Value::Bool(b) => hash_finish(1, if *b { 1 } else { 0 }),
        Value::String(s) => {
            let mut st = std::collections::hash_map::DefaultHasher::new();
            s.hash(&mut st);
            hash_finish(2, st.finish())
        }
        Value::Null => hash_finish(3, 0),
        Value::Ellipsis => hash_finish(4, 0),
        Value::Uuid(hi, lo) => hash_mix(hash_finish(5, *hi), *lo),
        Value::Date(d) => {
            let h = hash_mix(
                d.timestamp() as u64,
                d.timestamp_subsec_nanos() as u64,
            );
            hash_mix(h, d.offset().local_minus_utc() as u64)
        }
        Value::Duration(d) => hash_mix(d.num_seconds() as u64, d.subsec_nanos() as u64),
        Value::Path(p) => {
            let mut st = std::collections::hash_map::DefaultHasher::new();
            p.hash(&mut st);
            hash_finish(8, st.finish())
        }
        Value::Tuple(rc) => {
            let slice = rc.borrow();
            let mut h = hash_finish(9, slice.len() as u64);
            for e in slice.iter() {
                h = hash_mix(h, hash_value_tree(e));
            }
            h
        }
        Value::Object(rc) => match &*rc.borrow() {
            ObjectKind::Bucket(map) if map.is_frozen() => hash_object_map_entries(map),
            ObjectKind::Bucket(_) => 0,
            ObjectKind::Legacy(_) | ObjectKind::Inline(_) => 0,
        },
        _ => 0,
    }
}

fn hash_object_map_entries(map: &ObjectMap) -> u64 {
    let mut pairs: Vec<(u64, ValueId, ValueId)> = map.iter_entries().collect();
    pairs.sort();
    let mut h = hash_finish(10, pairs.len() as u64);
    for (kh, kid, vid) in pairs {
        h = hash_mix(h, kh);
        h = hash_mix(h, kid as u64);
        h = hash_mix(h, vid as u64);
    }
    h
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;
    use std::rc::Rc;

    #[test]
    fn number_hash_stable() {
        let n = Value::Number(1.0);
        assert!(!is_mutable_value(&n));
        let h1 = object_key_hash_value(&n).unwrap();
        let h2 = object_key_hash_value(&n).unwrap();
        assert_eq!(h1, h2);
    }

    #[test]
    fn tuple_hash() {
        let t = Value::Tuple(Rc::new(RefCell::new(vec![
            Value::Number(1.0),
            Value::String("x".into()),
        ])));
        assert!(object_key_hash_value(&t).is_some());
    }

    #[test]
    fn numeric_dict_key_hash_int_number_alias() {
        use crate::common::numeric::{FloatValue, IntValue};
        let int_key = Value::Int(IntValue::Finite(5));
        let num_key = Value::Number(5.0);
        let float_key = Value::Float(FloatValue::Finite(5.0));
        let h_int = object_key_hash_value(&int_key).unwrap();
        let h_num = object_key_hash_value(&num_key).unwrap();
        let h_float = object_key_hash_value(&float_key).unwrap();
        assert_eq!(h_int, h_num);
        assert_eq!(h_int, h_float);
    }
}
