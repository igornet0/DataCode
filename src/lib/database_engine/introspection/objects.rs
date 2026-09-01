//! Convert introspection structs into Datacode `legacy_object` values.

use super::types::{
    ColumnInfo, ForeignKeyInfo, IndexInfo, InspectInfo, InspectedSchema, InspectedTable,
    SchemaInfo, TableInfo,
};
use crate::common::value::Value;
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

pub const KEY_SCHEMA: &str = "__db_schema";
pub const KEY_TABLE: &str = "__db_table";
pub const KEY_COLUMN: &str = "__db_column";
pub const KEY_INDEX: &str = "__db_index";
pub const KEY_FOREIGN_KEY: &str = "__db_foreign_key";
pub const KEY_INSPECT: &str = "__db_inspect";

fn array_of(items: Vec<Value>) -> Value {
    Value::Array(Rc::new(RefCell::new(items)))
}

fn opt_string(v: Option<String>) -> Value {
    match v {
        Some(s) => Value::String(s),
        None => Value::Null,
    }
}

pub fn schema_object(info: SchemaInfo) -> Value {
    let mut map = HashMap::new();
    map.insert(KEY_SCHEMA.to_string(), Value::Bool(true));
    map.insert("name".to_string(), Value::String(info.name));
    Value::legacy_object(map)
}

pub fn table_object(info: TableInfo) -> Value {
    let mut map = HashMap::new();
    map.insert(KEY_TABLE.to_string(), Value::Bool(true));
    map.insert("name".to_string(), Value::String(info.name));
    map.insert("schema".to_string(), Value::String(info.schema));
    map.insert("type".to_string(), Value::String(info.kind.as_str().to_string()));
    Value::legacy_object(map)
}

pub fn column_object(info: ColumnInfo) -> Value {
    let mut map = HashMap::new();
    map.insert(KEY_COLUMN.to_string(), Value::Bool(true));
    map.insert("name".to_string(), Value::String(info.name));
    map.insert("type".to_string(), Value::String(info.sql_type));
    map.insert("nullable".to_string(), Value::Bool(info.nullable));
    map.insert("default".to_string(), opt_string(info.default));
    map.insert("datacode_type".to_string(), opt_string(info.datacode_type));
    Value::legacy_object(map)
}

pub fn index_object(info: IndexInfo) -> Value {
    let mut map = HashMap::new();
    map.insert(KEY_INDEX.to_string(), Value::Bool(true));
    map.insert("name".to_string(), Value::String(info.name));
    map.insert(
        "columns".to_string(),
        array_of(info.columns.into_iter().map(Value::String).collect()),
    );
    map.insert("unique".to_string(), Value::Bool(info.unique));
    map.insert("primary".to_string(), Value::Bool(info.primary));
    Value::legacy_object(map)
}

pub fn foreign_key_object(info: ForeignKeyInfo) -> Value {
    let mut map = HashMap::new();
    map.insert(KEY_FOREIGN_KEY.to_string(), Value::Bool(true));
    map.insert("name".to_string(), Value::String(info.name));
    map.insert(
        "columns".to_string(),
        array_of(info.columns.into_iter().map(Value::String).collect()),
    );
    map.insert(
        "referenced_table".to_string(),
        Value::String(info.referenced_table),
    );
    map.insert(
        "referenced_schema".to_string(),
        opt_string(info.referenced_schema),
    );
    map.insert(
        "referenced_columns".to_string(),
        array_of(
            info.referenced_columns
                .into_iter()
                .map(Value::String)
                .collect(),
        ),
    );
    Value::legacy_object(map)
}

fn inspected_table_object(info: InspectedTable) -> Value {
    let mut map = HashMap::new();
    map.insert(KEY_TABLE.to_string(), Value::Bool(true));
    map.insert("name".to_string(), Value::String(info.meta.name));
    map.insert("schema".to_string(), Value::String(info.meta.schema));
    map.insert(
        "type".to_string(),
        Value::String(info.meta.kind.as_str().to_string()),
    );
    map.insert(
        "columns".to_string(),
        array_of(info.columns.into_iter().map(column_object).collect()),
    );
    map.insert(
        "indexes".to_string(),
        array_of(info.indexes.into_iter().map(index_object).collect()),
    );
    map.insert(
        "primary_key".to_string(),
        match info.primary_key {
            Some(pk) => index_object(pk),
            None => Value::Null,
        },
    );
    map.insert(
        "foreign_keys".to_string(),
        array_of(
            info.foreign_keys
                .into_iter()
                .map(foreign_key_object)
                .collect(),
        ),
    );
    Value::legacy_object(map)
}

fn inspected_schema_object(info: InspectedSchema) -> Value {
    let mut map = HashMap::new();
    map.insert(KEY_SCHEMA.to_string(), Value::Bool(true));
    map.insert("name".to_string(), Value::String(info.name));
    map.insert(
        "tables".to_string(),
        array_of(
            info.tables
                .into_iter()
                .map(inspected_table_object)
                .collect(),
        ),
    );
    map.insert(
        "views".to_string(),
        array_of(info.views.into_iter().map(inspected_table_object).collect()),
    );
    Value::legacy_object(map)
}

pub fn inspect_object(info: InspectInfo) -> Value {
    let mut map = HashMap::new();
    map.insert(KEY_INSPECT.to_string(), Value::Bool(true));
    map.insert(
        "schemas".to_string(),
        array_of(
            info.schemas
                .into_iter()
                .map(inspected_schema_object)
                .collect(),
        ),
    );
    Value::legacy_object(map)
}

pub fn schema_array(items: Vec<SchemaInfo>) -> Value {
    array_of(items.into_iter().map(schema_object).collect())
}

pub fn table_array(items: Vec<TableInfo>) -> Value {
    array_of(items.into_iter().map(table_object).collect())
}

pub fn column_array(items: Vec<ColumnInfo>) -> Value {
    array_of(items.into_iter().map(column_object).collect())
}

pub fn index_array(items: Vec<IndexInfo>) -> Value {
    array_of(items.into_iter().map(index_object).collect())
}

pub fn foreign_key_array(items: Vec<ForeignKeyInfo>) -> Value {
    array_of(items.into_iter().map(foreign_key_object).collect())
}

pub fn optional_index(info: Option<IndexInfo>) -> Value {
    match info {
        Some(idx) => index_object(idx),
        None => Value::Null,
    }
}
