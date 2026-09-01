//! MySQL catalog introspection (`information_schema`).

use super::types::{
    append_fk_column, append_index_column, ColumnInfo, ForeignKeyInfo, IndexInfo, RelationKind,
    SchemaInfo, TableInfo,
};
use mysql::prelude::Queryable;
use mysql::Conn;

pub fn current_database(conn: &mut Conn) -> Result<String, String> {
    let db: Option<String> = conn
        .query_first("SELECT DATABASE()")
        .map_err(|e| e.to_string())?;
    db.filter(|s| !s.is_empty())
        .ok_or_else(|| "MySQL has no current database selected".to_string())
}

fn schema_name(conn: &mut Conn, schema: Option<&str>) -> Result<String, String> {
    match schema {
        Some(s) if !s.is_empty() => Ok(s.to_string()),
        _ => current_database(conn),
    }
}

pub fn schemas(conn: &mut Conn) -> Result<Vec<SchemaInfo>, String> {
    let names: Vec<String> = conn
        .query_map(
            "SELECT SCHEMA_NAME FROM information_schema.SCHEMATA
             WHERE SCHEMA_NAME NOT IN ('information_schema', 'mysql', 'performance_schema', 'sys')
             ORDER BY SCHEMA_NAME",
            |name: String| name,
        )
        .map_err(|e| e.to_string())?;
    Ok(names
        .into_iter()
        .map(|name| SchemaInfo { name })
        .collect())
}

fn list_relations(
    conn: &mut Conn,
    schema: &str,
    kind: RelationKind,
) -> Result<Vec<TableInfo>, String> {
    let table_type = match kind {
        RelationKind::Table => "BASE TABLE",
        RelationKind::View => "VIEW",
    };
    let names: Vec<String> = conn
        .exec_map(
            "SELECT TABLE_NAME FROM information_schema.TABLES
             WHERE TABLE_SCHEMA = ? AND TABLE_TYPE = ?
             ORDER BY TABLE_NAME",
            (schema, table_type),
            |name: String| name,
        )
        .map_err(|e| e.to_string())?;
    Ok(names
        .into_iter()
        .map(|name| TableInfo {
            name,
            schema: schema.to_string(),
            kind,
        })
        .collect())
}

pub fn tables(conn: &mut Conn, schema: Option<&str>) -> Result<Vec<TableInfo>, String> {
    let schema = schema_name(conn, schema)?;
    list_relations(conn, &schema, RelationKind::Table)
}

pub fn views(conn: &mut Conn, schema: Option<&str>) -> Result<Vec<TableInfo>, String> {
    let schema = schema_name(conn, schema)?;
    list_relations(conn, &schema, RelationKind::View)
}

pub fn columns(
    conn: &mut Conn,
    table: &str,
    schema: Option<&str>,
) -> Result<Vec<ColumnInfo>, String> {
    let schema = schema_name(conn, schema)?;
    conn.exec_map(
        "SELECT COLUMN_NAME, COLUMN_TYPE, IS_NULLABLE, COLUMN_DEFAULT
         FROM information_schema.COLUMNS
         WHERE TABLE_SCHEMA = ? AND TABLE_NAME = ?
         ORDER BY ORDINAL_POSITION",
        (&schema, table),
        |(name, sql_type, nullable, default): (String, String, String, Option<String>)| {
            ColumnInfo {
                name,
                sql_type: sql_type.to_ascii_uppercase(),
                nullable: nullable.eq_ignore_ascii_case("YES"),
                default,
                datacode_type: None,
            }
        },
    )
    .map_err(|e| e.to_string())
}

pub fn indexes(
    conn: &mut Conn,
    table: &str,
    schema: Option<&str>,
) -> Result<Vec<IndexInfo>, String> {
    let schema = schema_name(conn, schema)?;
    let rows: Vec<(String, i64, String)> = conn
        .exec_map(
            "SELECT INDEX_NAME, NON_UNIQUE, COLUMN_NAME
             FROM information_schema.STATISTICS
             WHERE TABLE_SCHEMA = ? AND TABLE_NAME = ?
             ORDER BY INDEX_NAME, SEQ_IN_INDEX",
            (&schema, table),
            |(name, non_unique, column): (String, i64, String)| (name, non_unique, column),
        )
        .map_err(|e| e.to_string())?;
    let mut out = Vec::new();
    for (name, non_unique, column) in rows {
        let primary = name.eq_ignore_ascii_case("PRIMARY");
        append_index_column(&mut out, name, non_unique == 0, primary, column);
    }
    Ok(out)
}

pub fn primary_key(
    conn: &mut Conn,
    table: &str,
    schema: Option<&str>,
) -> Result<Option<IndexInfo>, String> {
    let schema = schema_name(conn, schema)?;
    let cols: Vec<String> = conn
        .exec_map(
            "SELECT COLUMN_NAME FROM information_schema.COLUMNS
             WHERE TABLE_SCHEMA = ? AND TABLE_NAME = ? AND COLUMN_KEY = 'PRI'
             ORDER BY ORDINAL_POSITION",
            (&schema, table),
            |name: String| name,
        )
        .map_err(|e| e.to_string())?;
    if cols.is_empty() {
        return Ok(None);
    }
    Ok(Some(IndexInfo {
        name: "PRIMARY".to_string(),
        columns: cols,
        unique: true,
        primary: true,
    }))
}

pub fn foreign_keys(
    conn: &mut Conn,
    table: &str,
    schema: Option<&str>,
) -> Result<Vec<ForeignKeyInfo>, String> {
    let schema = schema_name(conn, schema)?;
    let rows: Vec<(String, String, String, Option<String>, String)> = conn
        .exec_map(
            "SELECT CONSTRAINT_NAME, COLUMN_NAME, REFERENCED_TABLE_NAME,
                    REFERENCED_TABLE_SCHEMA, REFERENCED_COLUMN_NAME
             FROM information_schema.KEY_COLUMN_USAGE
             WHERE TABLE_SCHEMA = ? AND TABLE_NAME = ?
               AND REFERENCED_TABLE_NAME IS NOT NULL
             ORDER BY CONSTRAINT_NAME, ORDINAL_POSITION",
            (&schema, table),
            |(name, col, ref_table, ref_schema, ref_col): (
                String,
                String,
                String,
                Option<String>,
                String,
            )| { (name, col, ref_table, ref_schema, ref_col) },
        )
        .map_err(|e| e.to_string())?;
    let mut out = Vec::new();
    for (name, col, ref_table, ref_schema, ref_col) in rows {
        append_fk_column(&mut out, name, col, ref_table, ref_schema, ref_col);
    }
    Ok(out)
}
