//! Document → Table normalizer (JSON / MongoDB documents).
//!
//! Nested objects become dotted columns (`profile.age`). Arrays default to cell
//! values; `ArrayMode::Explode` expands array-of-objects into multiple rows.

use crate::common::numeric::IntValue;
use crate::common::table::Table;
use crate::common::value::{ObjectKind, Value};
use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::rc::Rc;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ArrayMode {
    /// Keep arrays as cell values (`Value::Array`).
    #[default]
    Keep,
    /// Explode arrays of objects into multiple rows with `field.key` columns.
    Explode,
}

#[derive(Debug, Clone)]
pub struct FlattenOptions {
    pub array_mode: ArrayMode,
    pub max_depth: usize,
    pub separator: String,
}

impl Default for FlattenOptions {
    fn default() -> Self {
        Self {
            array_mode: ArrayMode::Keep,
            max_depth: 32,
            separator: ".".to_string(),
        }
    }
}

impl FlattenOptions {
    pub fn from_array_mode_str(s: Option<&str>) -> Self {
        let mut opts = Self::default();
        if let Some(mode) = s {
            opts.array_mode = match mode.to_ascii_lowercase().as_str() {
                "explode" => ArrayMode::Explode,
                _ => ArrayMode::Keep,
            };
        }
        opts
    }
}

/// Flatten one document into dotted key → value map (arrays kept as cells in Keep mode).
pub fn flatten_document(value: &Value, opts: &FlattenOptions) -> HashMap<String, Value> {
    let mut out = HashMap::new();
    flatten_into(value, "", 0, opts, &mut out);
    out
}

fn flatten_into(
    value: &Value,
    prefix: &str,
    depth: usize,
    opts: &FlattenOptions,
    out: &mut HashMap<String, Value>,
) {
    if depth > opts.max_depth {
        if !prefix.is_empty() {
            out.insert(prefix.to_string(), value.clone());
        }
        return;
    }
    match value {
        Value::Object(rc) => {
            let kind = rc.borrow();
            let entries = object_entries(&kind);
            if entries.is_empty() && !prefix.is_empty() {
                out.insert(prefix.to_string(), Value::Null);
                return;
            }
            for (k, v) in entries {
                let key = if prefix.is_empty() {
                    k
                } else {
                    format!("{}{}{}", prefix, opts.separator, k)
                };
                match &v {
                    Value::Object(_) => flatten_into(&v, &key, depth + 1, opts, out),
                    Value::Array(arr_rc) if opts.array_mode == ArrayMode::Keep => {
                        out.insert(key, Value::Array(Rc::clone(arr_rc)));
                    }
                    Value::Array(_) if opts.array_mode == ArrayMode::Explode => {
                        // Leave for explode pass; store raw array under key.
                        out.insert(key, v);
                    }
                    other => {
                        out.insert(key, other.clone());
                    }
                }
            }
        }
        other => {
            if prefix.is_empty() {
                out.insert("_value".to_string(), other.clone());
            } else {
                out.insert(prefix.to_string(), other.clone());
            }
        }
    }
}

fn object_entries(kind: &ObjectKind) -> Vec<(String, Value)> {
    match kind {
        ObjectKind::Legacy(m) => m.iter().map(|(k, v)| (k.clone(), v.clone())).collect(),
        ObjectKind::Inline(entries) => entries
            .iter()
            .filter_map(|(k, v)| match k {
                Value::String(sk) => Some((sk.clone(), v.clone())),
                _ => None,
            })
            .collect(),
        ObjectKind::Bucket(_) => vec![],
    }
}

/// Infer unified headers: `_id` first (if present), then remaining keys sorted.
pub fn infer_headers(docs: &[HashMap<String, Value>]) -> Vec<String> {
    let mut keys = BTreeSet::new();
    let mut has_id = false;
    for doc in docs {
        for k in doc.keys() {
            if k == "_id" {
                has_id = true;
            } else {
                keys.insert(k.clone());
            }
        }
    }
    let mut headers = Vec::with_capacity(keys.len() + usize::from(has_id));
    if has_id {
        headers.push("_id".to_string());
    }
    headers.extend(keys);
    headers
}

/// Convert documents (Value::Object or already-flat maps) into a Table.
pub fn documents_to_table(docs: &[Value], opts: &FlattenOptions) -> Table {
    match opts.array_mode {
        ArrayMode::Keep => documents_to_table_keep(docs, opts),
        ArrayMode::Explode => documents_to_table_explode(docs, opts),
    }
}

fn documents_to_table_keep(docs: &[Value], opts: &FlattenOptions) -> Table {
    let flat: Vec<HashMap<String, Value>> = docs
        .iter()
        .map(|d| flatten_document(d, opts))
        .collect();
    let headers = infer_headers(&flat);
    let rows: Vec<Vec<Value>> = flat
        .into_iter()
        .map(|row| {
            headers
                .iter()
                .map(|h| row.get(h).cloned().unwrap_or(Value::Null))
                .collect()
        })
        .collect();
    Table::from_data(rows, Some(headers))
}

fn documents_to_table_explode(docs: &[Value], opts: &FlattenOptions) -> Table {
    let mut keep_opts = opts.clone();
    keep_opts.array_mode = ArrayMode::Keep;
    // First flatten with Keep so nested objects become dotted keys; arrays remain cells.
    let mut all_rows: Vec<HashMap<String, Value>> = Vec::new();
    for doc in docs {
        let flat = flatten_document(doc, &keep_opts);
        all_rows.extend(explode_row(flat, &opts.separator));
    }
    let headers = infer_headers(&all_rows);
    let rows: Vec<Vec<Value>> = all_rows
        .into_iter()
        .map(|row| {
            headers
                .iter()
                .map(|h| row.get(h).cloned().unwrap_or(Value::Null))
                .collect()
        })
        .collect();
    Table::from_data(rows, Some(headers))
}

/// Explode array-of-object fields; non-object arrays stay as cells.
fn explode_row(row: HashMap<String, Value>, separator: &str) -> Vec<HashMap<String, Value>> {
    let mut base = HashMap::new();
    let mut explode_fields: Vec<(String, Vec<HashMap<String, Value>>)> = Vec::new();

    for (k, v) in row {
        match v {
            Value::Array(rc) => {
                let arr = rc.borrow();
                if arr.is_empty() {
                    base.insert(k, Value::Array(Rc::new(RefCell::new(vec![]))));
                    continue;
                }
                if arr.iter().all(|x| matches!(x, Value::Object(_))) {
                    let mut nested_rows = Vec::new();
                    for item in arr.iter() {
                        let mut nested_opts = FlattenOptions::default();
                        nested_opts.separator = separator.to_string();
                        let nested = flatten_document(item, &nested_opts);
                        let mut prefixed = HashMap::new();
                        for (nk, nv) in nested {
                            prefixed.insert(format!("{}{}{}", k, separator, nk), nv);
                        }
                        nested_rows.push(prefixed);
                    }
                    explode_fields.push((k, nested_rows));
                } else {
                    base.insert(k, Value::Array(Rc::clone(&rc)));
                }
            }
            other => {
                base.insert(k, other);
            }
        }
    }

    if explode_fields.is_empty() {
        return vec![base];
    }

    // Cartesian product of exploded array fields.
    let mut rows = vec![base];
    for (_field, variants) in explode_fields {
        let mut next = Vec::new();
        for row in &rows {
            for variant in &variants {
                let mut merged = row.clone();
                for (k, v) in variant {
                    merged.insert(k.clone(), v.clone());
                }
                next.push(merged);
            }
        }
        rows = next;
    }
    rows
}

/// Parse a simple filter object into a map of field → (op, value).
/// Supports:
/// - `{ age: 18 }` → eq
/// - `{ age: { "$gt": 18 } }` / `{ age: { gt: 18 } }`
/// - `{ name: { contains: "jo" } }`
pub fn parse_filter_ops(filter: &Value) -> Result<BTreeMap<String, FilterOp>, String> {
    let Value::Object(rc) = filter else {
        return Err("filter must be an object".to_string());
    };
    let kind = rc.borrow();
    let mut out = BTreeMap::new();
    for (field, val) in object_entries(&kind) {
        match val {
            Value::Object(op_rc) => {
                let op_kind = op_rc.borrow();
                let entries = object_entries(&op_kind);
                if entries.is_empty() {
                    out.insert(field, FilterOp::Eq(Value::Null));
                    continue;
                }
                // Take first operator entry (and also support multiple via And later).
                for (op_name, op_val) in entries {
                    let op = filter_op_from_name(&op_name, op_val)?;
                    out.insert(field.clone(), op);
                    break;
                }
            }
            other => {
                out.insert(field, FilterOp::Eq(other));
            }
        }
    }
    Ok(out)
}

#[derive(Debug, Clone)]
pub enum FilterOp {
    Eq(Value),
    Ne(Value),
    Gt(Value),
    Gte(Value),
    Lt(Value),
    Lte(Value),
    Contains(Value),
    StartsWith(Value),
    EndsWith(Value),
    In(Vec<Value>),
    NotIn(Vec<Value>),
    IsNull,
    IsNotNull,
}

fn filter_op_from_name(name: &str, val: Value) -> Result<FilterOp, String> {
    let n = name.trim_start_matches('$').to_ascii_lowercase();
    match n.as_str() {
        "eq" | "=" => Ok(FilterOp::Eq(val)),
        "ne" | "!=" | "neq" => Ok(FilterOp::Ne(val)),
        "gt" | ">" => Ok(FilterOp::Gt(val)),
        "gte" | ">=" => Ok(FilterOp::Gte(val)),
        "lt" | "<" => Ok(FilterOp::Lt(val)),
        "lte" | "<=" => Ok(FilterOp::Lte(val)),
        "contains" => Ok(FilterOp::Contains(val)),
        "starts_with" | "startswith" => Ok(FilterOp::StartsWith(val)),
        "ends_with" | "endswith" => Ok(FilterOp::EndsWith(val)),
        "in" => match val {
            Value::Array(rc) => Ok(FilterOp::In(rc.borrow().clone())),
            other => Ok(FilterOp::In(vec![other])),
        },
        "not_in" | "nin" => match val {
            Value::Array(rc) => Ok(FilterOp::NotIn(rc.borrow().clone())),
            other => Ok(FilterOp::NotIn(vec![other])),
        },
        "is_null" | "isnull" => Ok(FilterOp::IsNull),
        "is_not_null" | "isnotnull" => Ok(FilterOp::IsNotNull),
        other => Err(format!("unsupported filter operator '{}'", other)),
    }
}

/// Convert FilterOp map to MongoDB BSON filter document (as serde_json Value for flexibility).
pub fn filter_ops_to_mongo_json(ops: &BTreeMap<String, FilterOp>) -> serde_json::Value {
    let mut map = serde_json::Map::new();
    for (field, op) in ops {
        let clause = match op {
            FilterOp::Eq(v) => value_to_json_simple(v),
            FilterOp::Ne(v) => serde_json::json!({ "$ne": value_to_json_simple(v) }),
            FilterOp::Gt(v) => serde_json::json!({ "$gt": value_to_json_simple(v) }),
            FilterOp::Gte(v) => serde_json::json!({ "$gte": value_to_json_simple(v) }),
            FilterOp::Lt(v) => serde_json::json!({ "$lt": value_to_json_simple(v) }),
            FilterOp::Lte(v) => serde_json::json!({ "$lte": value_to_json_simple(v) }),
            FilterOp::Contains(v) => {
                let s = value_as_string(v);
                serde_json::json!({ "$regex": regex_escape(&s), "$options": "i" })
            }
            FilterOp::StartsWith(v) => {
                let s = value_as_string(v);
                serde_json::json!({ "$regex": format!("^{}", regex_escape(&s)), "$options": "i" })
            }
            FilterOp::EndsWith(v) => {
                let s = value_as_string(v);
                serde_json::json!({ "$regex": format!("{}$", regex_escape(&s)), "$options": "i" })
            }
            FilterOp::In(vals) => {
                serde_json::json!({ "$in": vals.iter().map(value_to_json_simple).collect::<Vec<_>>() })
            }
            FilterOp::NotIn(vals) => {
                serde_json::json!({ "$nin": vals.iter().map(value_to_json_simple).collect::<Vec<_>>() })
            }
            FilterOp::IsNull => serde_json::Value::Null,
            FilterOp::IsNotNull => serde_json::json!({ "$ne": null }),
        };
        map.insert(field.clone(), clause);
    }
    serde_json::Value::Object(map)
}

fn value_as_string(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

fn regex_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '\\' | '.' | '+' | '*' | '?' | '(' | ')' | '[' | ']' | '{' | '}' | '^' | '$'
            | '|' => {
                out.push('\\');
                out.push(c);
            }
            _ => out.push(c),
        }
    }
    out
}

fn value_to_json_simple(v: &Value) -> serde_json::Value {
    match v {
        Value::Null => serde_json::Value::Null,
        Value::Bool(b) => serde_json::Value::Bool(*b),
        Value::Number(n) => serde_json::json!(*n),
        Value::Float(f) => match f {
            crate::common::numeric::FloatValue::Finite(n) => serde_json::json!(*n),
            _ => serde_json::Value::Null,
        },
        Value::Int(IntValue::Finite(n)) => serde_json::json!(*n),
        Value::Int(_) => serde_json::Value::Null,
        Value::String(s) => serde_json::Value::String(s.clone()),
        Value::Array(rc) => {
            serde_json::Value::Array(rc.borrow().iter().map(value_to_json_simple).collect())
        }
        other => serde_json::Value::String(other.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn obj(pairs: Vec<(&str, Value)>) -> Value {
        let mut m = HashMap::new();
        for (k, v) in pairs {
            m.insert(k.to_string(), v);
        }
        Value::Object(Rc::new(RefCell::new(ObjectKind::legacy(m))))
    }

    #[test]
    fn flatten_nested_object() {
        let doc = obj(vec![
            ("_id", Value::String("1".into())),
            ("name", Value::String("John".into())),
            (
                "profile",
                obj(vec![
                    ("age", Value::Number(25.0)),
                    ("city", Value::String("Helsinki".into())),
                ]),
            ),
        ]);
        let flat = flatten_document(&doc, &FlattenOptions::default());
        assert_eq!(flat.get("_id"), Some(&Value::String("1".into())));
        assert_eq!(flat.get("name"), Some(&Value::String("John".into())));
        assert_eq!(flat.get("profile.age"), Some(&Value::Number(25.0)));
        assert_eq!(
            flat.get("profile.city"),
            Some(&Value::String("Helsinki".into()))
        );
        assert!(!flat.contains_key("profile"));
    }

    #[test]
    fn documents_missing_fields_become_null() {
        let docs = vec![
            obj(vec![
                ("name", Value::String("John".into())),
                ("age", Value::Number(20.0)),
            ]),
            obj(vec![
                ("name", Value::String("Alice".into())),
                ("email", Value::String("a@example.com".into())),
            ]),
        ];
        let table = documents_to_table(&docs, &FlattenOptions::default());
        let headers = table.headers().clone();
        assert!(headers.contains(&"name".to_string()));
        assert!(headers.contains(&"age".to_string()));
        assert!(headers.contains(&"email".to_string()));
        assert_eq!(table.len(), 2);
        let rows = table.rows_ref().unwrap().to_vec();
        let name_idx = headers.iter().position(|h| h == "name").unwrap();
        let age_idx = headers.iter().position(|h| h == "age").unwrap();
        let email_idx = headers.iter().position(|h| h == "email").unwrap();
        assert_eq!(rows[0][name_idx], Value::String("John".into()));
        assert_eq!(rows[0][age_idx], Value::Number(20.0));
        assert_eq!(rows[0][email_idx], Value::Null);
        assert_eq!(rows[1][age_idx], Value::Null);
        assert_eq!(rows[1][email_idx], Value::String("a@example.com".into()));
    }

    #[test]
    fn array_keep_mode() {
        let orders = Value::Array(Rc::new(RefCell::new(vec![obj(vec![
            ("id", Value::Number(1.0)),
            ("price", Value::Number(100.0)),
        ])])));
        let docs = vec![obj(vec![
            ("name", Value::String("John".into())),
            ("orders", orders.clone()),
        ])];
        let table = documents_to_table(&docs, &FlattenOptions::default());
        let headers = table.headers().clone();
        let orders_idx = headers.iter().position(|h| h == "orders").unwrap();
        let rows = table.rows_ref().unwrap().to_vec();
        assert!(matches!(rows[0][orders_idx], Value::Array(_)));
    }

    #[test]
    fn array_explode_mode() {
        let orders = Value::Array(Rc::new(RefCell::new(vec![
            obj(vec![
                ("id", Value::Number(1.0)),
                ("price", Value::Number(100.0)),
            ]),
            obj(vec![
                ("id", Value::Number(2.0)),
                ("price", Value::Number(200.0)),
            ]),
        ])));
        let docs = vec![obj(vec![
            ("name", Value::String("John".into())),
            ("orders", orders),
        ])];
        let mut opts = FlattenOptions::default();
        opts.array_mode = ArrayMode::Explode;
        let table = documents_to_table(&docs, &opts);
        assert_eq!(table.len(), 2);
        let headers = table.headers().clone();
        assert!(headers.contains(&"orders.id".to_string()));
        assert!(headers.contains(&"orders.price".to_string()));
        let id_idx = headers.iter().position(|h| h == "orders.id").unwrap();
        let rows = table.rows_ref().unwrap().to_vec();
        assert_eq!(rows[0][id_idx], Value::Number(1.0));
        assert_eq!(rows[1][id_idx], Value::Number(2.0));
    }

    #[test]
    fn filter_gt_to_mongo() {
        let filter = obj(vec![(
            "age",
            obj(vec![("gt", Value::Number(18.0))]),
        )]);
        let ops = parse_filter_ops(&filter).unwrap();
        let json = filter_ops_to_mongo_json(&ops);
        assert_eq!(json["age"]["$gt"], 18.0);
    }

    #[test]
    fn id_header_first() {
        let docs = vec![obj(vec![
            ("name", Value::String("A".into())),
            ("_id", Value::String("x".into())),
            ("age", Value::Number(1.0)),
        ])];
        let table = documents_to_table(&docs, &FlattenOptions::default());
        assert_eq!(table.headers()[0], "_id");
    }
}
