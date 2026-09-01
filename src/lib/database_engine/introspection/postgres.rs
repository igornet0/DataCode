//! PostgreSQL catalog introspection (`information_schema` / `pg_catalog`).

use super::types::{
    append_fk_column, append_index_column, ColumnInfo, ForeignKeyInfo, IndexInfo, RelationKind,
    SchemaInfo, TableInfo,
};
use postgres::Client;

pub const DEFAULT_SCHEMA: &str = "public";

fn schema_name(schema: Option<&str>) -> &str {
    schema.unwrap_or(DEFAULT_SCHEMA)
}

pub fn schemas(client: &mut Client) -> Result<Vec<SchemaInfo>, String> {
    let rows = client
        .query(
            "SELECT schema_name FROM information_schema.schemata
             WHERE schema_name NOT IN ('information_schema', 'pg_catalog', 'pg_toast')
               AND schema_name NOT LIKE 'pg_toast%'
               AND schema_name NOT LIKE 'pg_temp%'
             ORDER BY schema_name",
            &[],
        )
        .map_err(|e| e.to_string())?;
    Ok(rows
        .iter()
        .map(|r| SchemaInfo {
            name: r.get::<_, String>(0),
        })
        .collect())
}

fn list_relations(
    client: &mut Client,
    schema: &str,
    kind: RelationKind,
) -> Result<Vec<TableInfo>, String> {
    let table_type = match kind {
        RelationKind::Table => "BASE TABLE",
        RelationKind::View => "VIEW",
    };
    let schema_s = schema.to_string();
    let rows = client
        .query(
            "SELECT table_name FROM information_schema.tables
             WHERE table_schema = $1 AND table_type = $2
             ORDER BY table_name",
            &[&schema_s, &table_type],
        )
        .map_err(|e| e.to_string())?;
    Ok(rows
        .iter()
        .map(|r| TableInfo {
            name: r.get::<_, String>(0),
            schema: schema.to_string(),
            kind,
        })
        .collect())
}

pub fn tables(client: &mut Client, schema: Option<&str>) -> Result<Vec<TableInfo>, String> {
    list_relations(client, schema_name(schema), RelationKind::Table)
}

pub fn views(client: &mut Client, schema: Option<&str>) -> Result<Vec<TableInfo>, String> {
    list_relations(client, schema_name(schema), RelationKind::View)
}

pub fn columns(
    client: &mut Client,
    table: &str,
    schema: Option<&str>,
) -> Result<Vec<ColumnInfo>, String> {
    let schema_s = schema_name(schema).to_string();
    let table_s = table.to_string();
    let rows = client
        .query(
            "SELECT column_name, udt_name, is_nullable, column_default
             FROM information_schema.columns
             WHERE table_name = $1 AND table_schema = $2
             ORDER BY ordinal_position",
            &[&table_s, &schema_s],
        )
        .map_err(|e| e.to_string())?;
    Ok(rows
        .iter()
        .map(|r| {
            let nullable: String = r.get(2);
            ColumnInfo {
                name: r.get(0),
                sql_type: r.get::<_, String>(1).to_ascii_uppercase(),
                nullable: nullable.eq_ignore_ascii_case("YES"),
                default: r.get(3),
                datacode_type: None,
            }
        })
        .collect())
}

pub fn indexes(
    client: &mut Client,
    table: &str,
    schema: Option<&str>,
) -> Result<Vec<IndexInfo>, String> {
    let schema_s = schema_name(schema).to_string();
    let table_s = table.to_string();
    let rows = client
        .query(
            "SELECT i.relname AS index_name,
                    ix.indisunique,
                    ix.indisprimary,
                    a.attname AS column_name,
                    array_position(ix.indkey, a.attnum) AS pos
             FROM pg_index ix
             JOIN pg_class t ON t.oid = ix.indrelid
             JOIN pg_class i ON i.oid = ix.indexrelid
             JOIN pg_namespace n ON n.oid = t.relnamespace
             JOIN pg_attribute a ON a.attrelid = t.oid AND a.attnum = ANY (ix.indkey)
             WHERE t.relname = $1 AND n.nspname = $2
             ORDER BY i.relname, pos",
            &[&table_s, &schema_s],
        )
        .map_err(|e| e.to_string())?;
    let mut out = Vec::new();
    for r in &rows {
        let name: String = r.get(0);
        let unique: bool = r.get(1);
        let primary: bool = r.get(2);
        let column: String = r.get(3);
        append_index_column(&mut out, name, unique, primary, column);
    }
    Ok(out)
}

pub fn primary_key(
    client: &mut Client,
    table: &str,
    schema: Option<&str>,
) -> Result<Option<IndexInfo>, String> {
    let schema_s = schema_name(schema).to_string();
    let table_s = table.to_string();
    let rows = client
        .query(
            "SELECT kcu.column_name, tc.constraint_name
             FROM information_schema.table_constraints tc
             JOIN information_schema.key_column_usage kcu
               ON tc.constraint_name = kcu.constraint_name
              AND tc.table_schema = kcu.table_schema
             WHERE tc.constraint_type = 'PRIMARY KEY'
               AND tc.table_name = $1
               AND tc.table_schema = $2
             ORDER BY kcu.ordinal_position",
            &[&table_s, &schema_s],
        )
        .map_err(|e| e.to_string())?;
    if rows.is_empty() {
        return Ok(None);
    }
    let name: String = rows[0].get(1);
    Ok(Some(IndexInfo {
        name,
        columns: rows.iter().map(|r| r.get::<_, String>(0)).collect(),
        unique: true,
        primary: true,
    }))
}

pub fn foreign_keys(
    client: &mut Client,
    table: &str,
    schema: Option<&str>,
) -> Result<Vec<ForeignKeyInfo>, String> {
    let schema_s = schema_name(schema).to_string();
    let table_s = table.to_string();
    let rows = client
        .query(
            "SELECT
                con.conname,
                att.attname AS column_name,
                nsp.nspname AS ref_schema,
                rel.relname AS ref_table,
                att2.attname AS ref_column
             FROM pg_constraint con
             JOIN pg_class t ON t.oid = con.conrelid
             JOIN pg_namespace n ON n.oid = t.relnamespace
             JOIN LATERAL unnest(con.conkey) WITH ORDINALITY AS ord(attnum, ordinality) ON true
             JOIN pg_attribute att ON att.attrelid = t.oid AND att.attnum = ord.attnum
             JOIN pg_class rel ON rel.oid = con.confrelid
             JOIN pg_namespace nsp ON nsp.oid = rel.relnamespace
             JOIN LATERAL unnest(con.confkey) WITH ORDINALITY AS ord2(attnum, ordinality)
               ON ord2.ordinality = ord.ordinality
             JOIN pg_attribute att2 ON att2.attrelid = rel.oid AND att2.attnum = ord2.attnum
             WHERE con.contype = 'f' AND t.relname = $1 AND n.nspname = $2
             ORDER BY con.conname, ord.ordinality",
            &[&table_s, &schema_s],
        )
        .map_err(|e| e.to_string())?;
    let mut out = Vec::new();
    for r in &rows {
        append_fk_column(
            &mut out,
            r.get(0),
            r.get(1),
            r.get(3),
            Some(r.get(2)),
            r.get(4),
        );
    }
    Ok(out)
}
