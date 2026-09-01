//! SQLite catalog introspection (`sqlite_schema` + PRAGMA).

use super::types::{
    append_fk_column, append_index_column, is_hidden_table, ColumnInfo, ForeignKeyInfo, IndexInfo,
    RelationKind, SchemaInfo, TableInfo,
};
use crate::sqlite_export::type_map::{sqlite_declared_to_datacode, TABLE_SCHEMA};
use rusqlite::Connection;

fn quote_ident(name: &str) -> String {
    format!("\"{}\"", name.replace('"', "\"\""))
}

fn schema_name(schema: Option<&str>) -> &str {
    schema.unwrap_or("main")
}

fn pragma_sql(schema: &str, pragma: &str, arg: &str) -> String {
    let arg_q = quote_ident(arg);
    if schema == "main" {
        format!("PRAGMA {}({})", pragma, arg_q)
    } else {
        format!("PRAGMA {}.{}({})", quote_ident(schema), pragma, arg_q)
    }
}

fn list_relations(
    conn: &Connection,
    schema: &str,
    kind: RelationKind,
) -> Result<Vec<TableInfo>, String> {
    let sql = format!(
        "SELECT name FROM {}.sqlite_schema WHERE type = ?1 AND name NOT LIKE 'sqlite_%' ORDER BY name",
        quote_ident(schema)
    );
    let type_name = kind.as_str();
    let mut stmt = conn.prepare(&sql).map_err(|e| e.to_string())?;
    let rows = stmt
        .query_map([type_name], |row| row.get::<_, String>(0))
        .map_err(|e| e.to_string())?;
    let mut out = Vec::new();
    for row in rows {
        let name = row.map_err(|e| e.to_string())?;
        if is_hidden_table(&name) {
            continue;
        }
        out.push(TableInfo {
            name,
            schema: schema.to_string(),
            kind,
        });
    }
    Ok(out)
}

pub fn schemas(conn: &Connection) -> Result<Vec<SchemaInfo>, String> {
    let mut stmt = conn
        .prepare("PRAGMA database_list")
        .map_err(|e| e.to_string())?;
    let rows = stmt
        .query_map([], |row| row.get::<_, String>(1))
        .map_err(|e| e.to_string())?;
    let mut out = Vec::new();
    for row in rows {
        out.push(SchemaInfo {
            name: row.map_err(|e| e.to_string())?,
        });
    }
    if !out.iter().any(|s| s.name == "temp") {
        out.push(SchemaInfo {
            name: "temp".to_string(),
        });
    }
    Ok(out)
}

pub fn tables(conn: &Connection, schema: Option<&str>) -> Result<Vec<TableInfo>, String> {
    list_relations(conn, schema_name(schema), RelationKind::Table)
}

pub fn views(conn: &Connection, schema: Option<&str>) -> Result<Vec<TableInfo>, String> {
    list_relations(conn, schema_name(schema), RelationKind::View)
}

pub fn columns(
    conn: &Connection,
    table: &str,
    schema: Option<&str>,
) -> Result<Vec<ColumnInfo>, String> {
    let schema = schema_name(schema);
    let sql = pragma_sql(schema, "table_info", table);
    let mut stmt = conn.prepare(&sql).map_err(|e| e.to_string())?;
    let rows = stmt
        .query_map([], |row| {
            let name: String = row.get(1)?;
            let decl: String = row.get::<_, Option<String>>(2)?.unwrap_or_default();
            let notnull: i64 = row.get(3).unwrap_or(0);
            let default: Option<String> = match row.get::<_, rusqlite::types::Value>(4)? {
                rusqlite::types::Value::Null => None,
                rusqlite::types::Value::Text(s) => Some(s),
                rusqlite::types::Value::Integer(i) => Some(i.to_string()),
                rusqlite::types::Value::Real(r) => Some(r.to_string()),
                rusqlite::types::Value::Blob(_) => None,
            };
            Ok((name, decl, notnull, default))
        })
        .map_err(|e| e.to_string())?;

    let mut out = Vec::new();
    for row in rows {
        let (name, decl, notnull, default) = row.map_err(|e| e.to_string())?;
        let sql_type = if decl.is_empty() {
            "TEXT".to_string()
        } else {
            decl.split('(')
                .next()
                .unwrap_or(&decl)
                .trim()
                .to_ascii_uppercase()
        };
        let mut col = ColumnInfo {
            name,
            sql_type: sql_type.clone(),
            nullable: notnull == 0,
            default,
            datacode_type: Some(sqlite_declared_to_datacode(&sql_type).to_string()),
        };
        merge_datacode_type(conn, table, &mut col);
        out.push(col);
    }
    Ok(out)
}

fn merge_datacode_type(conn: &Connection, table: &str, col: &mut ColumnInfo) {
    let sql = format!(
        "SELECT datacode_type FROM {} WHERE table_name = ?1 AND column_name = ?2",
        TABLE_SCHEMA
    );
    if let Ok(dc) = conn.query_row(&sql, rusqlite::params![table, &col.name], |r| {
        r.get::<_, String>(0)
    }) {
        col.datacode_type = Some(dc);
    }
}

pub fn indexes(
    conn: &Connection,
    table: &str,
    schema: Option<&str>,
) -> Result<Vec<IndexInfo>, String> {
    let schema = schema_name(schema);
    let sql = pragma_sql(schema, "index_list", table);
    let mut stmt = conn.prepare(&sql).map_err(|e| e.to_string())?;
    let listed = stmt
        .query_map([], |row| {
            let name: String = row.get(1)?;
            let unique: i64 = row.get(2).unwrap_or(0);
            let origin: String = row.get::<_, Option<String>>(3)?.unwrap_or_default();
            Ok((name, unique != 0, origin))
        })
        .map_err(|e| e.to_string())?;

    let mut out = Vec::new();
    let listed: Vec<(String, bool, String)> = listed
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| e.to_string())?;
    for (name, unique, origin) in listed {
        let primary = origin.eq_ignore_ascii_case("pk");
        let info_sql = pragma_sql(schema, "index_info", &name);
        let mut info_stmt = conn.prepare(&info_sql).map_err(|e| e.to_string())?;
        let cols = info_stmt
            .query_map([], |row| row.get::<_, Option<String>>(2))
            .map_err(|e| e.to_string())?;
        for col in cols {
            let col_name = col.map_err(|e| e.to_string())?.unwrap_or_default();
            append_index_column(&mut out, name.clone(), unique, primary, col_name);
        }
        if !out.iter().any(|i| i.name == name) {
            out.push(IndexInfo {
                name,
                columns: Vec::new(),
                unique,
                primary,
            });
        }
    }
    Ok(out)
}

pub fn primary_key(
    conn: &Connection,
    table: &str,
    schema: Option<&str>,
) -> Result<Option<IndexInfo>, String> {
    let schema = schema_name(schema);
    let sql = pragma_sql(schema, "table_info", table);
    let mut stmt = conn.prepare(&sql).map_err(|e| e.to_string())?;
    let rows = stmt
        .query_map([], |row| {
            let name: String = row.get(1)?;
            let pk: i64 = row.get(5).unwrap_or(0);
            Ok((name, pk))
        })
        .map_err(|e| e.to_string())?;
    let mut pk_cols: Vec<(i64, String)> = Vec::new();
    for row in rows {
        let (name, pk) = row.map_err(|e| e.to_string())?;
        if pk > 0 {
            pk_cols.push((pk, name));
        }
    }
    if pk_cols.is_empty() {
        return Ok(None);
    }
    pk_cols.sort_by_key(|(order, _)| *order);
    Ok(Some(IndexInfo {
        name: "PRIMARY".to_string(),
        columns: pk_cols.into_iter().map(|(_, n)| n).collect(),
        unique: true,
        primary: true,
    }))
}

pub fn foreign_keys(
    conn: &Connection,
    table: &str,
    schema: Option<&str>,
) -> Result<Vec<ForeignKeyInfo>, String> {
    let schema = schema_name(schema);
    let sql = pragma_sql(schema, "foreign_key_list", table);
    let mut stmt = conn.prepare(&sql).map_err(|e| e.to_string())?;
    let rows = stmt
        .query_map([], |row| {
            let id: i64 = row.get(0)?;
            let referenced_table: String = row.get(2)?;
            let from_col: String = row.get(3)?;
            let to_col: Option<String> = row.get(4)?;
            Ok((id, referenced_table, from_col, to_col.unwrap_or_default()))
        })
        .map_err(|e| e.to_string())?;
    let mut out = Vec::new();
    for row in rows {
        let (id, referenced_table, from_col, to_col) = row.map_err(|e| e.to_string())?;
        append_fk_column(
            &mut out,
            format!("fk_{}", id),
            from_col,
            referenced_table,
            Some(schema.to_string()),
            to_col,
        );
    }
    Ok(out)
}
