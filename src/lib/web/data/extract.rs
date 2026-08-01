//! HTML → DataCode structures (tables / extract schemas).

use crate::common::table::Table;
use crate::common::value::Value;
use crate::web::args::object_entries;
use crate::web::error::WebError;
use scraper::{Html, Selector};
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

#[derive(Debug, Clone)]
pub enum FieldSpec {
    Selector(String),
    Attr { selector: String, attribute: String },
}

pub fn parse_field_schema(schema: &Value) -> Result<HashMap<String, FieldSpec>, WebError> {
    let entries = object_entries(schema)?;
    let mut out = HashMap::new();
    for (name, spec) in entries {
        out.insert(name, parse_field_spec(&spec)?);
    }
    Ok(out)
}

fn parse_field_spec(spec: &Value) -> Result<FieldSpec, WebError> {
    match spec {
        Value::String(s) => Ok(FieldSpec::Selector(s.clone())),
        Value::Object(_) => {
            let e = object_entries(spec)?;
            let selector = e
                .get("selector")
                .and_then(|v| match v {
                    Value::String(s) => Some(s.clone()),
                    _ => None,
                })
                .ok_or_else(|| WebError::value("field spec requires string 'selector'"))?;
            let attribute = e
                .get("attribute")
                .and_then(|v| match v {
                    Value::String(s) => Some(s.clone()),
                    _ => None,
                })
                .ok_or_else(|| WebError::value("field spec object requires string 'attribute'"))?;
            Ok(FieldSpec::Attr {
                selector,
                attribute,
            })
        }
        _ => Err(WebError::type_err(
            "field spec must be a selector string or { selector, attribute }",
        )),
    }
}

pub fn table_from_html(html: &str, selector: Option<&str>) -> Result<Table, WebError> {
    let document = Html::parse_document(html);
    let table_sel = if let Some(sel) = selector {
        Selector::parse(sel).map_err(|e| WebError::value(format!("invalid selector '{}': {:?}", sel, e)))?
    } else {
        Selector::parse("table").map_err(|e| WebError::value(format!("invalid selector: {:?}", e)))?
    };
    let table_el = document
        .select(&table_sel)
        .next()
        .ok_or_else(|| WebError::value("no <table> found for selector"))?;

    let th_sel = Selector::parse("th").unwrap();
    let tr_sel = Selector::parse("tr").unwrap();
    let td_sel = Selector::parse("td").unwrap();

    let mut headers: Vec<String> = table_el
        .select(&th_sel)
        .map(|th| th.text().collect::<String>().trim().to_string())
        .collect();

    let mut rows: Vec<Vec<Value>> = Vec::new();
    for tr in table_el.select(&tr_sel) {
        let cells: Vec<Value> = tr
            .select(&td_sel)
            .map(|td| Value::String(td.text().collect::<String>().trim().to_string()))
            .collect();
        if cells.is_empty() {
            continue;
        }
        rows.push(cells);
    }

    if headers.is_empty() {
        let col_count = rows.first().map(|r| r.len()).unwrap_or(0);
        headers = (0..col_count).map(|i| format!("col{}", i)).collect();
    }
    // Normalize row widths
    let n = headers.len();
    for row in &mut rows {
        row.resize(n, Value::Null);
    }
    Ok(Table::from_data(rows, Some(headers)))
}

/// Extract one record from the document root using field schema.
pub fn extract_one(html: &str, schema: &HashMap<String, FieldSpec>) -> Result<Value, WebError> {
    let document = Html::parse_document(html);
    let root = document.root_element();
    Ok(extract_from_element(&root, schema)?)
}

/// Extract all items matching `item_selector`, each with relative field schema.
pub fn extract_many(
    html: &str,
    item_selector: &str,
    schema: &HashMap<String, FieldSpec>,
) -> Result<Value, WebError> {
    let document = Html::parse_document(html);
    let sel = Selector::parse(item_selector)
        .map_err(|e| WebError::value(format!("invalid selector '{}': {:?}", item_selector, e)))?;
    let mut items = Vec::new();
    for el in document.select(&sel) {
        items.push(extract_from_element(&el, schema)?);
    }
    Ok(Value::Array(Rc::new(RefCell::new(items))))
}

fn extract_from_element(
    el: &scraper::ElementRef<'_>,
    schema: &HashMap<String, FieldSpec>,
) -> Result<Value, WebError> {
    let mut map = HashMap::new();
    for (name, spec) in schema {
        let value = match spec {
            FieldSpec::Selector(sel_str) => {
                let sel = Selector::parse(sel_str).map_err(|e| {
                    WebError::value(format!("invalid selector '{}': {:?}", sel_str, e))
                })?;
                el.select(&sel)
                    .next()
                    .map(|n| n.text().collect::<String>().trim().to_string())
                    .unwrap_or_default()
            }
            FieldSpec::Attr {
                selector,
                attribute,
            } => {
                let sel = Selector::parse(selector).map_err(|e| {
                    WebError::value(format!("invalid selector '{}': {:?}", selector, e))
                })?;
                el.select(&sel)
                    .next()
                    .and_then(|n| n.value().attr(attribute).map(|s| s.to_string()))
                    .unwrap_or_default()
            }
        };
        map.insert(name.clone(), Value::String(value));
    }
    Ok(Value::legacy_object(map))
}
