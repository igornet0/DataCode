//! Convert [`Value`] to/from XML for file I/O (xmltodict-style: `@attr`, `#text`).

use crate::common::value::{ObjectKind, Value};
use crate::file_io::value_serde::SerdeError;
use quick_xml::events::{BytesEnd, BytesStart, BytesText, Event};
use quick_xml::Writer;
use roxmltree::Node;
use std::cell::RefCell;
use std::collections::HashMap;
use std::io::Cursor;
use std::rc::Rc;

pub fn parse_xml_str(s: &str) -> Result<Value, SerdeError> {
    let doc = roxmltree::Document::parse(s)
        .map_err(|e| SerdeError::Parse(format!("XML: {}", e)))?;
    let root = doc.root_element();
    let name = root.tag_name().name().to_string();
    let val = element_to_value(root);
    Ok(Value::Object(Rc::new(RefCell::new(ObjectKind::Inline(vec![(
        Value::String(name),
        val,
    )])))))
}

fn element_to_value(node: Node<'_, '_>) -> Value {
    let mut map: HashMap<String, Value> = HashMap::new();

    for attr in node.attributes() {
        map.insert(
            format!("@{}", attr.name()),
            Value::String(attr.value().to_string()),
        );
    }

    for child in node.children().filter(|n| n.is_element()) {
        let name = child.tag_name().name().to_string();
        let val = element_to_value(child);
        merge_map_entry(&mut map, name, val);
    }

    let text = node
        .text()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string);

    if let Some(t) = text {
        if map.is_empty() {
            return scalar_from_str(&t);
        }
        map.insert("#text".to_string(), Value::String(t));
    }

    if map.is_empty() {
        return Value::Null;
    }

    let pairs: Vec<(Value, Value)> = map
        .into_iter()
        .map(|(k, v)| (Value::String(k), v))
        .collect();
    Value::Object(Rc::new(RefCell::new(ObjectKind::Inline(pairs))))
}

fn merge_map_entry(map: &mut HashMap<String, Value>, key: String, val: Value) {
    match map.remove(&key) {
        None => {
            map.insert(key, val);
        }
        Some(existing) => {
            let arr = match existing {
                Value::Array(rc) => rc,
                other => Rc::new(RefCell::new(vec![other])),
            };
            arr.borrow_mut().push(val);
            map.insert(key, Value::Array(arr));
        }
    }
}

fn scalar_from_str(s: &str) -> Value {
    if s == "true" || s == "True" {
        Value::Bool(true)
    } else if s == "false" || s == "False" {
        Value::Bool(false)
    } else if let Ok(n) = s.parse::<f64>() {
        Value::Number(n)
    } else {
        Value::String(s.to_string())
    }
}

pub fn value_to_xml_string(v: &Value) -> Result<String, SerdeError> {
    let io_err = |e: std::io::Error| SerdeError::Serialize(format!("XML: {}", e));
    let mut writer = Writer::new_with_indent(Cursor::new(Vec::new()), b' ', 2);
    writer
        .write_event(Event::Decl(quick_xml::events::BytesDecl::new(
            "1.0",
            Some("UTF-8"),
            None,
        )))
        .map_err(io_err)?;

    match v {
        Value::Object(obj) => {
            let map = obj.borrow();
            let pairs = inline_pairs(&map);
            if pairs.len() == 1 {
                let (root_name, root_val) = &pairs[0];
                write_value_as_element(&mut writer, root_name, root_val)?;
            } else if pairs.is_empty() {
                write_value_as_element(&mut writer, "root", &Value::Null)?;
            } else {
                let pairs: Vec<(Value, Value)> = pairs
                    .into_iter()
                    .map(|(k, v)| (Value::String(k), v))
                    .collect();
                let inner = Value::Object(Rc::new(RefCell::new(ObjectKind::Inline(pairs))));
                write_value_as_element(&mut writer, "root", &inner)?;
            }
        }
        other => write_value_as_element(&mut writer, "root", other)?,
    }

    let bytes = writer.into_inner().into_inner();
    String::from_utf8(bytes).map_err(|e| SerdeError::Serialize(format!("XML: {}", e)))
}

fn inline_pairs(map: &ObjectKind) -> Vec<(String, Value)> {
    match map {
        ObjectKind::Inline(pairs) => pairs
            .iter()
            .filter_map(|(k, v)| match k {
                Value::String(s) => Some((s.clone(), v.clone())),
                _ => Some((k.to_string(), v.clone())),
            })
            .collect(),
        ObjectKind::Legacy(hm) => hm
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect(),
        ObjectKind::Bucket(_) => Vec::new(),
    }
}

fn write_value_as_element(
    writer: &mut Writer<Cursor<Vec<u8>>>,
    tag: &str,
    val: &Value,
) -> Result<(), SerdeError> {
    let io_err = |e: std::io::Error| SerdeError::Serialize(format!("XML: {}", e));

    match val {
        Value::Null => {
            let elem = BytesStart::new(tag);
            writer.write_event(Event::Empty(elem)).map_err(io_err)?;
        }
        Value::Object(obj) => {
            let pairs = inline_pairs(&obj.borrow());
            let mut attrs = Vec::new();
            let mut children = Vec::new();
            let mut text: Option<String> = None;

            for (k, v) in pairs {
                if let Some(attr) = k.strip_prefix('@') {
                    attrs.push((attr.to_string(), value_to_xml_attr(&v)?));
                } else if k == "#text" {
                    text = Some(value_to_xml_text(&v)?);
                } else {
                    children.push((k, v));
                }
            }

            let mut elem = BytesStart::new(tag);
            for (ak, av) in &attrs {
                elem.push_attribute((ak.as_str(), av.as_str()));
            }
            writer.write_event(Event::Start(elem)).map_err(io_err)?;

            if let Some(t) = text {
                writer
                    .write_event(Event::Text(BytesText::new(&t)))
                    .map_err(io_err)?;
            }
            for (child_tag, child_val) in children {
                match &child_val {
                    Value::Array(arr) => {
                        for item in arr.borrow().iter() {
                            write_value_as_element(writer, &child_tag, item)?;
                        }
                    }
                    _ => write_value_as_element(writer, &child_tag, &child_val)?,
                }
            }
            writer
                .write_event(Event::End(BytesEnd::new(tag)))
                .map_err(io_err)?;
        }
        scalar => {
            let elem = BytesStart::new(tag);
            writer.write_event(Event::Start(elem)).map_err(io_err)?;
            let text = value_to_xml_text(scalar)?;
            writer
                .write_event(Event::Text(BytesText::new(&text)))
                .map_err(io_err)?;
            writer
                .write_event(Event::End(BytesEnd::new(tag)))
                .map_err(io_err)?;
        }
    }
    Ok(())
}

fn value_to_xml_text(v: &Value) -> Result<String, SerdeError> {
    Ok(match v {
        Value::String(s) => s.clone(),
        Value::Bool(b) => b.to_string(),
        Value::Number(n) => {
            if n.fract() == 0.0 && n.is_finite() {
                format!("{}", *n as i64)
            } else {
                n.to_string()
            }
        }
        Value::Null => String::new(),
        other => other.to_string(),
    })
}

fn value_to_xml_attr(v: &Value) -> Result<String, SerdeError> {
    value_to_xml_text(v)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn xml_round_trip_config_like() {
        let src = r#"<?xml version="1.0" encoding="UTF-8"?>
<config>
  <name>demo</name>
  <version>1</version>
  <debug>true</debug>
  <paths>
    <path>input/</path>
    <path>output/</path>
  </paths>
</config>"#;
        let v = parse_xml_str(src).unwrap();
        let out = value_to_xml_string(&v).unwrap();
        assert!(out.contains("<config>"));
        assert!(out.contains("<name>demo</name>"));
        let back = parse_xml_str(&out).unwrap();
        match &back {
            Value::Object(obj) => {
                let obj_ref = obj.borrow();
                let inner = obj_ref.str_key_get("config").expect("config key");
                match inner {
                    Value::Object(cfg) => {
                        assert_eq!(
                            cfg.borrow().str_key_get("name"),
                            Some(&Value::String("demo".to_string()))
                        );
                    }
                    _ => panic!("expected config object"),
                }
            }
            _ => panic!("expected root object"),
        }
    }
}
