//! Shared introspection types used by every SQL backend.

use crate::sqlite_export::type_map::is_datacode_system_table;

/// Catalog schema (SQLite attached database, PostgreSQL schema, MySQL database, MSSQL schema).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SchemaInfo {
    pub name: String,
}

/// Table or view catalog entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TableInfo {
    pub name: String,
    pub schema: String,
    pub kind: RelationKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RelationKind {
    Table,
    View,
}

impl RelationKind {
    pub fn as_str(self) -> &'static str {
        match self {
            RelationKind::Table => "table",
            RelationKind::View => "view",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ColumnInfo {
    pub name: String,
    pub sql_type: String,
    pub nullable: bool,
    pub default: Option<String>,
    pub datacode_type: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndexInfo {
    pub name: String,
    pub columns: Vec<String>,
    pub unique: bool,
    pub primary: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ForeignKeyInfo {
    pub name: String,
    pub columns: Vec<String>,
    pub referenced_table: String,
    pub referenced_schema: Option<String>,
    pub referenced_columns: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct InspectedTable {
    pub meta: TableInfo,
    pub columns: Vec<ColumnInfo>,
    pub indexes: Vec<IndexInfo>,
    pub primary_key: Option<IndexInfo>,
    pub foreign_keys: Vec<ForeignKeyInfo>,
}

#[derive(Debug, Clone)]
pub struct InspectedSchema {
    pub name: String,
    pub tables: Vec<InspectedTable>,
    pub views: Vec<InspectedTable>,
}

#[derive(Debug, Clone)]
pub struct InspectInfo {
    pub schemas: Vec<InspectedSchema>,
}

/// Hide SQLite internals and Datacode system tables from user-facing lists.
pub fn is_hidden_table(name: &str) -> bool {
    name.eq_ignore_ascii_case("sqlite_sequence")
        || name.starts_with("sqlite_")
        || is_datacode_system_table(name)
}

pub fn append_index_column(
    out: &mut Vec<IndexInfo>,
    name: String,
    unique: bool,
    primary: bool,
    column: String,
) {
    if let Some(idx) = out.iter_mut().find(|i| i.name == name) {
        if !column.is_empty() {
            idx.columns.push(column);
        }
        return;
    }
    out.push(IndexInfo {
        name,
        unique,
        primary,
        columns: if column.is_empty() {
            Vec::new()
        } else {
            vec![column]
        },
    });
}

pub fn append_fk_column(
    out: &mut Vec<ForeignKeyInfo>,
    name: String,
    column: String,
    referenced_table: String,
    referenced_schema: Option<String>,
    referenced_column: String,
) {
    if let Some(fk) = out.iter_mut().find(|f| f.name == name) {
        if !column.is_empty() {
            fk.columns.push(column);
        }
        if !referenced_column.is_empty() {
            fk.referenced_columns.push(referenced_column);
        }
        return;
    }
    out.push(ForeignKeyInfo {
        name,
        columns: if column.is_empty() {
            Vec::new()
        } else {
            vec![column]
        },
        referenced_table,
        referenced_schema,
        referenced_columns: if referenced_column.is_empty() {
            Vec::new()
        } else {
            vec![referenced_column]
        },
    });
}
