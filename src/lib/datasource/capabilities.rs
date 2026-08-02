//! DataSource capability flags exposed as a Value object.

use crate::common::value::{ObjectKind, Value};
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

#[derive(Debug, Clone, Default)]
pub struct Capabilities {
    pub supports_filter: bool,
    pub supports_sort: bool,
    pub supports_limit: bool,
    pub supports_offset: bool,
    pub supports_count: bool,
    pub supports_streaming: bool,
    pub supports_nested_objects: bool,
    pub supports_arrays: bool,
    pub supports_aggregation: bool,
    pub supports_native_query: bool,
    pub supports_transactions: bool,
    pub supports_sql: bool,
}

impl Capabilities {
    pub fn sql_default() -> Self {
        Self {
            supports_filter: false, // via SQL string
            supports_sort: false,
            supports_limit: true,
            supports_offset: true,
            supports_count: true,
            supports_streaming: false,
            supports_nested_objects: false,
            supports_arrays: false,
            supports_aggregation: false,
            supports_native_query: true, // raw SQL
            supports_transactions: false,
            supports_sql: true,
        }
    }

    pub fn mongodb_default() -> Self {
        Self {
            supports_filter: true,
            supports_sort: true,
            supports_limit: true,
            supports_offset: true,
            supports_count: true,
            supports_streaming: true,
            supports_nested_objects: true,
            supports_arrays: true,
            supports_aggregation: true,
            supports_native_query: true,
            supports_transactions: false,
            supports_sql: false,
        }
    }

    pub fn http_default() -> Self {
        Self {
            supports_filter: false,
            supports_sort: false,
            supports_limit: false,
            supports_offset: false,
            supports_count: false,
            supports_streaming: false,
            supports_nested_objects: true,
            supports_arrays: true,
            supports_aggregation: false,
            supports_native_query: false,
            supports_transactions: false,
            supports_sql: false,
        }
    }

    pub fn file_default() -> Self {
        Self {
            supports_filter: false,
            supports_sort: false,
            supports_limit: false,
            supports_offset: false,
            supports_count: false,
            supports_streaming: false,
            supports_nested_objects: true,
            supports_arrays: true,
            supports_aggregation: false,
            supports_native_query: false,
            supports_transactions: false,
            supports_sql: false,
        }
    }

    pub fn to_value(&self) -> Value {
        let mut m = HashMap::new();
        m.insert("supports_filter".into(), Value::Bool(self.supports_filter));
        m.insert("supports_sort".into(), Value::Bool(self.supports_sort));
        m.insert("supports_limit".into(), Value::Bool(self.supports_limit));
        m.insert("supports_offset".into(), Value::Bool(self.supports_offset));
        m.insert("supports_count".into(), Value::Bool(self.supports_count));
        m.insert(
            "supports_streaming".into(),
            Value::Bool(self.supports_streaming),
        );
        m.insert(
            "supports_nested_objects".into(),
            Value::Bool(self.supports_nested_objects),
        );
        m.insert("supports_arrays".into(), Value::Bool(self.supports_arrays));
        m.insert(
            "supports_aggregation".into(),
            Value::Bool(self.supports_aggregation),
        );
        m.insert(
            "supports_native_query".into(),
            Value::Bool(self.supports_native_query),
        );
        m.insert(
            "supports_transactions".into(),
            Value::Bool(self.supports_transactions),
        );
        m.insert("supports_sql".into(), Value::Bool(self.supports_sql));
        Value::Object(Rc::new(RefCell::new(ObjectKind::legacy(m))))
    }
}
