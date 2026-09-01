//! MSSQL catalog introspection (`sys.*`).

use super::types::{
    append_fk_column, append_index_column, ColumnInfo, ForeignKeyInfo, IndexInfo, RelationKind,
    SchemaInfo, TableInfo,
};
use crate::web::browser::runtime::block_on;
use tiberius::{Client as MssqlClient, Query};
use tokio::net::TcpStream;
use tokio_util::compat::Compat;

pub const DEFAULT_SCHEMA: &str = "dbo";

type MssqlConn = MssqlClient<Compat<TcpStream>>;

fn schema_name(schema: Option<&str>) -> &str {
    schema.unwrap_or(DEFAULT_SCHEMA)
}

fn query_string_col(
    client: &mut MssqlConn,
    sql: &str,
    binds: &[&str],
) -> Result<Vec<Vec<Option<String>>>, String> {
    let sql_owned = sql.to_string();
    let binds_owned: Vec<String> = binds.iter().map(|s| (*s).to_string()).collect();
    block_on(async {
        let mut query = Query::new(&sql_owned);
        for b in &binds_owned {
            query.bind(b.as_str());
        }
        let stream = query
            .query(client)
            .await
            .map_err(|e| format!("MSSQL query failed: {}", e))?;
        let rows = stream
            .into_first_result()
            .await
            .map_err(|e| format!("MSSQL query failed: {}", e))?;
        let mut out = Vec::new();
        for row in rows {
            let mut cols = Vec::new();
            for i in 0..row.columns().len() {
                let v = row
                    .try_get::<&str, _>(i)
                    .ok()
                    .flatten()
                    .map(|s| s.to_string());
                cols.push(v);
            }
            out.push(cols);
        }
        Ok(out)
    })
}

pub fn schemas(client: &mut MssqlConn) -> Result<Vec<SchemaInfo>, String> {
    let rows = query_string_col(
        client,
        "SELECT name FROM sys.schemas
         WHERE name NOT IN ('sys', 'INFORMATION_SCHEMA', 'guest')
           AND name NOT LIKE 'db_%'
         ORDER BY name",
        &[],
    )?;
    Ok(rows
        .into_iter()
        .filter_map(|r| r.into_iter().next().flatten())
        .map(|name| SchemaInfo { name })
        .collect())
}

fn list_relations(
    client: &mut MssqlConn,
    schema: &str,
    kind: RelationKind,
) -> Result<Vec<TableInfo>, String> {
    let sql = match kind {
        RelationKind::Table => {
            "SELECT t.name FROM sys.tables t
             JOIN sys.schemas s ON s.schema_id = t.schema_id
             WHERE s.name = @P1
             ORDER BY t.name"
        }
        RelationKind::View => {
            "SELECT v.name FROM sys.views v
             JOIN sys.schemas s ON s.schema_id = v.schema_id
             WHERE s.name = @P1
             ORDER BY v.name"
        }
    };
    let rows = query_string_col(client, sql, &[schema])?;
    Ok(rows
        .into_iter()
        .filter_map(|r| r.into_iter().next().flatten())
        .map(|name| TableInfo {
            name,
            schema: schema.to_string(),
            kind,
        })
        .collect())
}

pub fn tables(client: &mut MssqlConn, schema: Option<&str>) -> Result<Vec<TableInfo>, String> {
    list_relations(client, schema_name(schema), RelationKind::Table)
}

pub fn views(client: &mut MssqlConn, schema: Option<&str>) -> Result<Vec<TableInfo>, String> {
    list_relations(client, schema_name(schema), RelationKind::View)
}

pub fn columns(
    client: &mut MssqlConn,
    table: &str,
    schema: Option<&str>,
) -> Result<Vec<ColumnInfo>, String> {
    let schema = schema_name(schema);
    // Mix of string and bit columns — query with a dedicated mapper.
    let table_s = table.to_string();
    let schema_s = schema.to_string();
    block_on(async {
        let mut query = Query::new(
            "SELECT c.name, ty.name, c.is_nullable, dc.definition
             FROM sys.columns c
             JOIN sys.types ty ON ty.user_type_id = c.user_type_id
             JOIN sys.objects o ON o.object_id = c.object_id
             JOIN sys.schemas s ON s.schema_id = o.schema_id
             LEFT JOIN sys.default_constraints dc
               ON dc.parent_object_id = c.object_id AND dc.parent_column_id = c.column_id
             WHERE o.name = @P1 AND s.name = @P2
             ORDER BY c.column_id",
        );
        query.bind(table_s.as_str());
        query.bind(schema_s.as_str());
        let stream = query
            .query(client)
            .await
            .map_err(|e| format!("MSSQL query failed: {}", e))?;
        let rows = stream
            .into_first_result()
            .await
            .map_err(|e| format!("MSSQL query failed: {}", e))?;
        let mut out = Vec::new();
        for row in rows {
            let name = row
                .try_get::<&str, _>(0)
                .ok()
                .flatten()
                .unwrap_or("")
                .to_string();
            let sql_type = row
                .try_get::<&str, _>(1)
                .ok()
                .flatten()
                .unwrap_or("nvarchar")
                .to_ascii_uppercase();
            let nullable = row.try_get::<bool, _>(2).ok().flatten().unwrap_or(true);
            let default = row.try_get::<&str, _>(3).ok().flatten().map(|s| s.to_string());
            out.push(ColumnInfo {
                name,
                sql_type,
                nullable,
                default,
                datacode_type: None,
            });
        }
        Ok(out)
    })
}

pub fn indexes(
    client: &mut MssqlConn,
    table: &str,
    schema: Option<&str>,
) -> Result<Vec<IndexInfo>, String> {
    let schema = schema_name(schema);
    let table_s = table.to_string();
    let schema_s = schema.to_string();
    block_on(async {
        let mut query = Query::new(
            "SELECT i.name, i.is_unique, i.is_primary_key, col.name
             FROM sys.indexes i
             JOIN sys.index_columns ic
               ON ic.object_id = i.object_id AND ic.index_id = i.index_id
             JOIN sys.columns col
               ON col.object_id = ic.object_id AND col.column_id = ic.column_id
             JOIN sys.objects o ON o.object_id = i.object_id
             JOIN sys.schemas s ON s.schema_id = o.schema_id
             WHERE o.name = @P1 AND s.name = @P2 AND i.is_hypothetical = 0
               AND i.name IS NOT NULL
             ORDER BY i.name, ic.key_ordinal",
        );
        query.bind(table_s.as_str());
        query.bind(schema_s.as_str());
        let stream = query
            .query(client)
            .await
            .map_err(|e| format!("MSSQL query failed: {}", e))?;
        let rows = stream
            .into_first_result()
            .await
            .map_err(|e| format!("MSSQL query failed: {}", e))?;
        let mut out = Vec::new();
        for row in rows {
            let name = row
                .try_get::<&str, _>(0)
                .ok()
                .flatten()
                .unwrap_or("")
                .to_string();
            let unique = row.try_get::<bool, _>(1).ok().flatten().unwrap_or(false);
            let primary = row.try_get::<bool, _>(2).ok().flatten().unwrap_or(false);
            let column = row
                .try_get::<&str, _>(3)
                .ok()
                .flatten()
                .unwrap_or("")
                .to_string();
            if !name.is_empty() {
                append_index_column(&mut out, name, unique, primary, column);
            }
        }
        Ok(out)
    })
}

pub fn primary_key(
    client: &mut MssqlConn,
    table: &str,
    schema: Option<&str>,
) -> Result<Option<IndexInfo>, String> {
    let schema = schema_name(schema);
    let table_s = table.to_string();
    let schema_s = schema.to_string();
    block_on(async {
        let mut query = Query::new(
            "SELECT kc.name, col.name
             FROM sys.key_constraints kc
             JOIN sys.index_columns ic
               ON ic.object_id = kc.parent_object_id AND ic.index_id = kc.unique_index_id
             JOIN sys.columns col
               ON col.object_id = ic.object_id AND col.column_id = ic.column_id
             JOIN sys.objects o ON o.object_id = kc.parent_object_id
             JOIN sys.schemas s ON s.schema_id = o.schema_id
             WHERE kc.type = 'PK' AND o.name = @P1 AND s.name = @P2
             ORDER BY ic.key_ordinal",
        );
        query.bind(table_s.as_str());
        query.bind(schema_s.as_str());
        let stream = query
            .query(client)
            .await
            .map_err(|e| format!("MSSQL query failed: {}", e))?;
        let rows = stream
            .into_first_result()
            .await
            .map_err(|e| format!("MSSQL query failed: {}", e))?;
        let mut columns = Vec::new();
        let mut name = String::from("PRIMARY");
        for row in rows {
            if let Some(n) = row.try_get::<&str, _>(0).ok().flatten() {
                name = n.to_string();
            }
            if let Some(c) = row.try_get::<&str, _>(1).ok().flatten() {
                columns.push(c.to_string());
            }
        }
        if columns.is_empty() {
            Ok(None)
        } else {
            Ok(Some(IndexInfo {
                name,
                columns,
                unique: true,
                primary: true,
            }))
        }
    })
}

pub fn foreign_keys(
    client: &mut MssqlConn,
    table: &str,
    schema: Option<&str>,
) -> Result<Vec<ForeignKeyInfo>, String> {
    let schema = schema_name(schema);
    let table_s = table.to_string();
    let schema_s = schema.to_string();
    block_on(async {
        let mut query = Query::new(
            "SELECT fk.name, pc.name, rs.name, rt.name, rc.name
             FROM sys.foreign_keys fk
             JOIN sys.foreign_key_columns fkc ON fkc.constraint_object_id = fk.object_id
             JOIN sys.objects t ON t.object_id = fk.parent_object_id
             JOIN sys.schemas s ON s.schema_id = t.schema_id
             JOIN sys.columns pc
               ON pc.object_id = fkc.parent_object_id AND pc.column_id = fkc.parent_column_id
             JOIN sys.objects rt ON rt.object_id = fk.referenced_object_id
             JOIN sys.schemas rs ON rs.schema_id = rt.schema_id
             JOIN sys.columns rc
               ON rc.object_id = fkc.referenced_object_id AND rc.column_id = fkc.referenced_column_id
             WHERE t.name = @P1 AND s.name = @P2
             ORDER BY fk.name, fkc.constraint_column_id",
        );
        query.bind(table_s.as_str());
        query.bind(schema_s.as_str());
        let stream = query
            .query(client)
            .await
            .map_err(|e| format!("MSSQL query failed: {}", e))?;
        let rows = stream
            .into_first_result()
            .await
            .map_err(|e| format!("MSSQL query failed: {}", e))?;
        let mut out = Vec::new();
        for row in rows {
            let name = row
                .try_get::<&str, _>(0)
                .ok()
                .flatten()
                .unwrap_or("fk")
                .to_string();
            let col = row
                .try_get::<&str, _>(1)
                .ok()
                .flatten()
                .unwrap_or("")
                .to_string();
            let ref_schema = row.try_get::<&str, _>(2).ok().flatten().map(|s| s.to_string());
            let ref_table = row
                .try_get::<&str, _>(3)
                .ok()
                .flatten()
                .unwrap_or("")
                .to_string();
            let ref_col = row
                .try_get::<&str, _>(4)
                .ok()
                .flatten()
                .unwrap_or("")
                .to_string();
            append_fk_column(&mut out, name, col, ref_table, ref_schema, ref_col);
        }
        Ok(out)
    })
}
