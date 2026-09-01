//! Universal database introspection API for `DatabaseEngine`.

mod mssql;
mod mysql;
mod objects;
mod postgres;
mod sqlite;
pub mod types;

pub use objects::{
    column_array, foreign_key_array, index_array, inspect_object, optional_index, schema_array,
    table_array,
};
pub use types::{
    ColumnInfo, ForeignKeyInfo, IndexInfo, InspectInfo, InspectedSchema, InspectedTable,
    RelationKind, SchemaInfo, TableInfo,
};

use crate::common::table::Table;
use crate::database_engine::engine::{DatabaseEngine, DbBackend, SqlDialect};

/// Backend-agnostic catalog queries. Each SQL dialect implements this with its own
/// system catalogs (`sqlite_schema`, `information_schema`, `sys.*`).
pub trait IntrospectionBackend {
    fn schemas(&mut self) -> Result<Vec<SchemaInfo>, String>;
    fn tables(&mut self, schema: Option<&str>) -> Result<Vec<TableInfo>, String>;
    fn views(&mut self, schema: Option<&str>) -> Result<Vec<TableInfo>, String>;
    fn columns(&mut self, table: &str, schema: Option<&str>) -> Result<Vec<ColumnInfo>, String>;
    fn indexes(&mut self, table: &str, schema: Option<&str>) -> Result<Vec<IndexInfo>, String>;
    fn primary_key(
        &mut self,
        table: &str,
        schema: Option<&str>,
    ) -> Result<Option<IndexInfo>, String>;
    fn foreign_keys(
        &mut self,
        table: &str,
        schema: Option<&str>,
    ) -> Result<Vec<ForeignKeyInfo>, String>;
}

impl IntrospectionBackend for DbBackend {
    fn schemas(&mut self) -> Result<Vec<SchemaInfo>, String> {
        match self {
            DbBackend::SQLite(conn) => sqlite::schemas(conn),
            DbBackend::Postgres(client) => postgres::schemas(client),
            DbBackend::Mysql(conn) => mysql::schemas(conn),
            DbBackend::Mssql(client) => mssql::schemas(client),
        }
    }

    fn tables(&mut self, schema: Option<&str>) -> Result<Vec<TableInfo>, String> {
        match self {
            DbBackend::SQLite(conn) => sqlite::tables(conn, schema),
            DbBackend::Postgres(client) => postgres::tables(client, schema),
            DbBackend::Mysql(conn) => mysql::tables(conn, schema),
            DbBackend::Mssql(client) => mssql::tables(client, schema),
        }
    }

    fn views(&mut self, schema: Option<&str>) -> Result<Vec<TableInfo>, String> {
        match self {
            DbBackend::SQLite(conn) => sqlite::views(conn, schema),
            DbBackend::Postgres(client) => postgres::views(client, schema),
            DbBackend::Mysql(conn) => mysql::views(conn, schema),
            DbBackend::Mssql(client) => mssql::views(client, schema),
        }
    }

    fn columns(&mut self, table: &str, schema: Option<&str>) -> Result<Vec<ColumnInfo>, String> {
        match self {
            DbBackend::SQLite(conn) => sqlite::columns(conn, table, schema),
            DbBackend::Postgres(client) => postgres::columns(client, table, schema),
            DbBackend::Mysql(conn) => mysql::columns(conn, table, schema),
            DbBackend::Mssql(client) => mssql::columns(client, table, schema),
        }
    }

    fn indexes(&mut self, table: &str, schema: Option<&str>) -> Result<Vec<IndexInfo>, String> {
        match self {
            DbBackend::SQLite(conn) => sqlite::indexes(conn, table, schema),
            DbBackend::Postgres(client) => postgres::indexes(client, table, schema),
            DbBackend::Mysql(conn) => mysql::indexes(conn, table, schema),
            DbBackend::Mssql(client) => mssql::indexes(client, table, schema),
        }
    }

    fn primary_key(
        &mut self,
        table: &str,
        schema: Option<&str>,
    ) -> Result<Option<IndexInfo>, String> {
        match self {
            DbBackend::SQLite(conn) => sqlite::primary_key(conn, table, schema),
            DbBackend::Postgres(client) => postgres::primary_key(client, table, schema),
            DbBackend::Mysql(conn) => mysql::primary_key(conn, table, schema),
            DbBackend::Mssql(client) => mssql::primary_key(client, table, schema),
        }
    }

    fn foreign_keys(
        &mut self,
        table: &str,
        schema: Option<&str>,
    ) -> Result<Vec<ForeignKeyInfo>, String> {
        match self {
            DbBackend::SQLite(conn) => sqlite::foreign_keys(conn, table, schema),
            DbBackend::Postgres(client) => postgres::foreign_keys(client, table, schema),
            DbBackend::Mysql(conn) => mysql::foreign_keys(conn, table, schema),
            DbBackend::Mssql(client) => mssql::foreign_keys(client, table, schema),
        }
    }
}

pub fn quote_ident(name: &str, dialect: SqlDialect) -> String {
    match dialect {
        SqlDialect::Mysql => format!("`{}`", name.replace('`', "``")),
        SqlDialect::Mssql => format!("[{}]", name.replace(']', "]]")),
        SqlDialect::Sqlite | SqlDialect::Postgres => {
            format!("\"{}\"", name.replace('"', "\"\""))
        }
    }
}

pub fn qualify_table(name: &str, schema: Option<&str>, dialect: SqlDialect) -> String {
    let table = quote_ident(name, dialect);
    match schema {
        Some(s) if !s.is_empty() => format!("{}.{}", quote_ident(s, dialect), table),
        _ => table,
    }
}

impl DatabaseEngine {
    pub fn schemas(&mut self) -> Result<Vec<SchemaInfo>, String> {
        self.backend.schemas()
    }

    pub fn tables(&mut self, schema: Option<&str>) -> Result<Vec<TableInfo>, String> {
        self.backend.tables(schema)
    }

    pub fn views(&mut self, schema: Option<&str>) -> Result<Vec<TableInfo>, String> {
        self.backend.views(schema)
    }

    pub fn columns(&mut self, table: &str, schema: Option<&str>) -> Result<Vec<ColumnInfo>, String> {
        self.backend.columns(table, schema)
    }

    pub fn indexes(&mut self, table: &str, schema: Option<&str>) -> Result<Vec<IndexInfo>, String> {
        self.backend.indexes(table, schema)
    }

    pub fn primary_key(
        &mut self,
        table: &str,
        schema: Option<&str>,
    ) -> Result<Option<IndexInfo>, String> {
        self.backend.primary_key(table, schema)
    }

    pub fn foreign_keys(
        &mut self,
        table: &str,
        schema: Option<&str>,
    ) -> Result<Vec<ForeignKeyInfo>, String> {
        self.backend.foreign_keys(table, schema)
    }

    pub fn inspect(&mut self) -> Result<InspectInfo, String> {
        let schemas = self.schemas()?;
        let mut inspected = Vec::new();
        for schema in schemas {
            let schema_name = schema.name.clone();
            let table_metas = self.tables(Some(&schema_name))?;
            let view_metas = self.views(Some(&schema_name))?;
            let tables = self.inspect_relations(&schema_name, table_metas)?;
            let views = self.inspect_relations(&schema_name, view_metas)?;
            inspected.push(InspectedSchema {
                name: schema_name,
                tables,
                views,
            });
        }
        Ok(InspectInfo { schemas: inspected })
    }

    fn inspect_relations(
        &mut self,
        schema: &str,
        metas: Vec<TableInfo>,
    ) -> Result<Vec<InspectedTable>, String> {
        let mut result = Vec::new();
        for meta in metas {
            let columns = self.columns(&meta.name, Some(schema))?;
            let indexes = self.indexes(&meta.name, Some(schema))?;
            let primary_key = self.primary_key(&meta.name, Some(schema))?;
            let foreign_keys = self.foreign_keys(&meta.name, Some(schema))?;
            result.push(InspectedTable {
                meta,
                columns,
                indexes,
                primary_key,
                foreign_keys,
            });
        }
        Ok(result)
    }

    /// Load a relation as a Datacode `Table` (`SELECT *`).
    pub fn table(&mut self, name: &str, schema: Option<&str>) -> Result<Table, String> {
        let dialect = self.dialect;
        let schema_owned = match schema {
            Some(s) if !s.is_empty() => Some(s.to_string()),
            _ => match dialect {
                // Keep a simple name so SQLite typed `query()` can resolve `_datacode_schema`.
                SqlDialect::Sqlite => None,
                SqlDialect::Postgres => Some(postgres::DEFAULT_SCHEMA.to_string()),
                SqlDialect::Mysql => mysql::current_database(match &mut self.backend {
                    DbBackend::Mysql(conn) => conn,
                    _ => unreachable!(),
                })
                .ok(),
                SqlDialect::Mssql => Some(mssql::DEFAULT_SCHEMA.to_string()),
            },
        };
        let qualified = qualify_table(name, schema_owned.as_deref(), dialect);
        self.query(&format!("SELECT * FROM {}", qualified), &[])
    }
}
