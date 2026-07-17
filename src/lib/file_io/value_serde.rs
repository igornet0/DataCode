//! Convert [`Value`] to/from JSON, TOML, YAML, and XML for file I/O.

pub use super::xml_value::{parse_xml_str, value_to_xml_string};

use crate::common::numeric::{FloatValue, IntValue};
use crate::common::value::{ObjectKind, Value};
use std::cell::RefCell;
use std::rc::Rc;

#[derive(Debug)]
pub enum SerdeError {
    Unsupported(String),
    Parse(String),
    Serialize(String),
}

impl SerdeError {
    pub fn message(self) -> String {
        match self {
            SerdeError::Unsupported(m) => m,
            SerdeError::Parse(m) => m,
            SerdeError::Serialize(m) => m,
        }
    }
}

pub fn value_to_json(v: &Value) -> Result<serde_json::Value, SerdeError> {
    use serde_json::json;
    Ok(match v {
        Value::Null => json!(null),
        Value::Bool(b) => json!(b),
        Value::Int(i) => match i {
            IntValue::Finite(n) => json!(n),
            IntValue::PosInfinity => json!("inf"),
            IntValue::NegInfinity => json!("-inf"),
        },
        Value::Float(f) => match f {
            FloatValue::Finite(n) => json!(n),
            FloatValue::PosInfinity => json!("inf"),
            FloatValue::NegInfinity => json!("-inf"),
            FloatValue::NaN => json!(null),
        },
        Value::Number(n) => json!(n),
        Value::String(s) => json!(s),
        Value::Array(arr) => {
            let out: Result<Vec<_>, _> = arr.borrow().iter().map(value_to_json).collect();
            json!(out?)
        }
        Value::Object(obj) => {
            let map = obj.borrow();
            let mut out = serde_json::Map::new();
            for (k, v) in object_str_pairs(&map) {
                out.insert(k, value_to_json(&v)?);
            }
            serde_json::Value::Object(out)
        }
        Value::ByteBuffer(b) => json!(b.bytes.as_ref()),
        other => json!(other.to_string()),
    })
}

pub fn json_to_value(v: serde_json::Value) -> Result<Value, SerdeError> {
    Ok(match v {
        serde_json::Value::Null => Value::Null,
        serde_json::Value::Bool(b) => Value::Bool(b),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Value::Number(i as f64)
            } else if let Some(f) = n.as_f64() {
                Value::Number(f)
            } else {
                Value::String(n.to_string())
            }
        }
        serde_json::Value::String(s) => Value::String(s),
        serde_json::Value::Array(arr) => {
            let items: Result<Vec<_>, _> = arr.into_iter().map(json_to_value).collect();
            Value::Array(Rc::new(RefCell::new(items?)))
        }
        serde_json::Value::Object(map) => {
            let pairs: Result<Vec<_>, _> = map
                .into_iter()
                .map(|(k, v)| json_to_value(v).map(|val| (Value::String(k), val)))
                .collect();
            Value::Object(Rc::new(RefCell::new(ObjectKind::Inline(pairs?))))
        }
    })
}

pub fn parse_json_str(s: &str) -> Result<Value, SerdeError> {
    let j: serde_json::Value =
        serde_json::from_str(s).map_err(|e| SerdeError::Parse(format!("JSON: {}", e)))?;
    json_to_value(j)
}

pub fn value_to_json_string_pretty(v: &Value) -> Result<String, SerdeError> {
    let j = value_to_json(v)?;
    serde_json::to_string_pretty(&j).map_err(|e| SerdeError::Serialize(format!("JSON: {}", e)))
}

pub fn value_to_toml(v: &Value) -> Result<toml::Value, SerdeError> {
    Ok(match v {
        Value::Null => toml::Value::String(String::new()),
        Value::Bool(b) => toml::Value::Boolean(*b),
        Value::Int(i) => match i {
            IntValue::Finite(n) => toml::Value::Integer(*n),
            IntValue::PosInfinity | IntValue::NegInfinity => {
                toml::Value::String(i.to_display_string())
            }
        },
        Value::Float(f) => match f {
            FloatValue::Finite(n) => toml::Value::Float(*n),
            _ => toml::Value::String(f.to_display_string()),
        },
        Value::Number(n) => {
            if n.fract() == 0.0 && n.is_finite() {
                toml::Value::Integer(*n as i64)
            } else {
                toml::Value::Float(*n)
            }
        }
        Value::String(s) => toml::Value::String(s.clone()),
        Value::Array(arr) => {
            let out: Result<Vec<_>, _> = arr.borrow().iter().map(value_to_toml).collect();
            toml::Value::Array(out?)
        }
        Value::Object(obj) => {
            let mut table = toml::map::Map::new();
            for (k, v) in object_str_pairs(&obj.borrow()) {
                table.insert(k, value_to_toml(&v)?);
            }
            toml::Value::Table(table)
        }
        other => toml::Value::String(other.to_string()),
    })
}

pub fn toml_to_value(v: toml::Value) -> Result<Value, SerdeError> {
    Ok(match v {
        toml::Value::String(s) => Value::String(s),
        toml::Value::Integer(i) => Value::Number(i as f64),
        toml::Value::Float(f) => Value::Number(f),
        toml::Value::Boolean(b) => Value::Bool(b),
        toml::Value::Datetime(dt) => Value::String(dt.to_string()),
        toml::Value::Array(arr) => {
            let items: Result<Vec<_>, _> = arr.into_iter().map(toml_to_value).collect();
            Value::Array(Rc::new(RefCell::new(items?)))
        }
        toml::Value::Table(map) => {
            let pairs: Result<Vec<_>, _> = map
                .into_iter()
                .map(|(k, v)| toml_to_value(v).map(|val| (Value::String(k), val)))
                .collect();
            Value::Object(Rc::new(RefCell::new(ObjectKind::Inline(pairs?))))
        }
    })
}

pub fn parse_toml_str(s: &str) -> Result<Value, SerdeError> {
    let t: toml::Value =
        toml::from_str(s).map_err(|e| SerdeError::Parse(format!("TOML: {}", e)))?;
    toml_to_value(t)
}

pub fn value_to_toml_string_pretty(v: &Value) -> Result<String, SerdeError> {
    let t = value_to_toml(v)?;
    toml::to_string_pretty(&t).map_err(|e| SerdeError::Serialize(format!("TOML: {}", e)))
}

pub fn parse_yaml_str(s: &str) -> Result<Value, SerdeError> {
    let y: serde_yaml::Value =
        serde_yaml::from_str(s).map_err(|e| SerdeError::Parse(format!("YAML: {}", e)))?;
    yaml_to_value(y)
}

pub fn yaml_to_value(v: serde_yaml::Value) -> Result<Value, SerdeError> {
    Ok(match v {
        serde_yaml::Value::Null => Value::Null,
        serde_yaml::Value::Bool(b) => Value::Bool(b),
        serde_yaml::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Value::Number(i as f64)
            } else if let Some(f) = n.as_f64() {
                Value::Number(f)
            } else {
                Value::String(n.to_string())
            }
        }
        serde_yaml::Value::String(s) => Value::String(s),
        serde_yaml::Value::Sequence(seq) => {
            let items: Result<Vec<_>, _> = seq.into_iter().map(yaml_to_value).collect();
            Value::Array(Rc::new(RefCell::new(items?)))
        }
        serde_yaml::Value::Mapping(map) => {
            let pairs: Result<Vec<_>, _> = map
                .into_iter()
                .map(|(k, v)| {
                    let key = yaml_to_value(k)?;
                    let val = yaml_to_value(v)?;
                    Ok((key, val))
                })
                .collect();
            Value::Object(Rc::new(RefCell::new(ObjectKind::Inline(pairs?))))
        }
        serde_yaml::Value::Tagged(t) => yaml_to_value(t.value)?,
    })
}

pub fn value_to_yaml_string(v: &Value) -> Result<String, SerdeError> {
    let j = value_to_json(v)?;
    serde_yaml::to_string(&j).map_err(|e| SerdeError::Serialize(format!("YAML: {}", e)))
}

fn object_str_pairs(map: &ObjectKind) -> Vec<(String, Value)> {
    match map {
        ObjectKind::Inline(pairs) => pairs
            .iter()
            .filter_map(|(k, v)| match k {
                Value::String(s) => Some((s.clone(), v.clone())),
                _ => Some((k.to_string(), v.clone())),
            })
            .collect(),
        ObjectKind::Bucket(b) => b
            .iter_entries()
            .map(|(_, k_id, _v_id)| {
                (format!("key_{}", k_id), Value::Null)
            })
            .collect(),
        ObjectKind::Legacy(map) => map
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn json_round_trip_object() {
        let v = Value::Object(Rc::new(RefCell::new(ObjectKind::Inline(vec![
            (Value::String("a".to_string()), Value::Number(1.0)),
            (Value::String("b".to_string()), Value::String("x".to_string())),
        ]))));
        let j = value_to_json(&v).unwrap();
        let back = json_to_value(j).unwrap();
        match &back {
            Value::Object(obj) => {
                assert_eq!(obj.borrow().str_key_get("a"), Some(&Value::Number(1.0)));
            }
            _ => panic!("expected object"),
        }
    }

    #[test]
    fn toml_round_trip() {
        let src = r#"
        name = "test"
        count = 3
        "#;
        let v = parse_toml_str(src).unwrap();
        let out = value_to_toml_string_pretty(&v).unwrap();
        assert!(out.contains("name"));
        assert!(out.contains("test"));
    }
}
