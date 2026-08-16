// Database engine abstraction for multiple backend types

use crate::common::table::Table;
use crate::common::value::ByteBuffer;
use crate::common::value::Value;
use crate::sqlite_export::type_map::{
    ensure_system_tables_sql, schema_discovery_from_pragma, sql_value_to_datacode,
    validate_schema_type_pair, value_to_owned_sql, SchemaCache, SchemaColumn, SCHEMA_VERSION,
    TABLE_SCHEMA, TABLE_VERSION,
};
use crate::web::browser::runtime::block_on;
use postgres::types::ToSql as PgToSql;
use postgres::{Client as PgClient, NoTls};
use rusqlite::Connection;
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Duration;
use tiberius::{AuthMethod, Client as MssqlClient, Config as MssqlConfig, EncryptionLevel};
use tokio::net::TcpStream;
use tokio_util::compat::{Compat, TokioAsyncWriteCompatExt};

/// SQL dialect for placeholder style and type mapping.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SqlDialect {
    Sqlite,
    Postgres,
    Mysql,
    Mssql,
}

/// Backend type
pub enum DbBackend {
    SQLite(Connection),
    Postgres(PgClient),
    Mysql(mysql::Conn),
    Mssql(MssqlClient<Compat<TcpStream>>),
}

impl std::fmt::Debug for DbBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DbBackend::SQLite(_) => write!(f, "SQLite(..)"),
            DbBackend::Postgres(_) => write!(f, "Postgres(..)"),
            DbBackend::Mysql(_) => write!(f, "Mysql(..)"),
            DbBackend::Mssql(_) => write!(f, "Mssql(..)"),
        }
    }
}

/// Database engine - holds connection/connection pool and config
#[derive(Debug)]
pub struct DatabaseEngine {
    pub backend: DbBackend,
    pub url: String,
    pub dialect: SqlDialect,
    pub echo: bool,
    pub echo_pool: bool,
    pub pool_size: u32,
    pub max_overflow: u32,
    pub timeout: Option<f64>,
    pub connect_args: HashMap<String, Value>,
    /// Lazy-loaded `_datacode_schema` (SQLite only). `None` = not yet attempted.
    schema_cache: Option<SchemaCache>,
    /// Declared types from `PRAGMA table_info` keyed by (table, column).
    declared_cache: HashMap<(String, String), String>,
}

impl DatabaseEngine {
    pub fn new_sqlite(
        url: String,
        echo: bool,
        echo_pool: bool,
        pool_size: u32,
        max_overflow: u32,
        timeout: Option<f64>,
        connect_args: HashMap<String, Value>,
    ) -> Result<Self, String> {
        Self::from_url(
            url,
            echo,
            echo_pool,
            pool_size,
            max_overflow,
            timeout,
            connect_args,
        )
    }

    pub fn from_url(
        url: String,
        echo: bool,
        echo_pool: bool,
        pool_size: u32,
        max_overflow: u32,
        timeout: Option<f64>,
        connect_args: HashMap<String, Value>,
    ) -> Result<Self, String> {
        let lower = url.to_ascii_lowercase();
        let (backend, dialect) = if lower.starts_with("sqlite:") {
            let path = parse_sqlite_path(&url)?;
            let conn =
                Connection::open(&path).map_err(|e| format!("SQLite connection failed: {}", e))?;
            conn.execute("PRAGMA foreign_keys = ON", [])
                .map_err(|e| format!("Failed to enable foreign keys: {}", e))?;
            (DbBackend::SQLite(conn), SqlDialect::Sqlite)
        } else if lower.starts_with("postgres://") || lower.starts_with("postgresql://") {
            let mut client =
                PgClient::connect(&url, NoTls).map_err(|e| format!("PostgreSQL connection failed: {}", e))?;
            if let Some(t) = timeout {
                let _ = client.execute(
                    &format!("SET statement_timeout = {}", (t * 1000.0) as i64),
                    &[],
                );
            }
            (DbBackend::Postgres(client), SqlDialect::Postgres)
        } else if lower.starts_with("mysql://") {
            let opts = mysql::Opts::from_url(&url)
                .map_err(|e| format!("MySQL URL parse failed: {}", e))?;
            let mut builder = mysql::OptsBuilder::from_opts(opts);
            if let Some(t) = timeout {
                builder = builder.read_timeout(Some(Duration::from_secs_f64(t.max(0.1))));
                builder = builder.write_timeout(Some(Duration::from_secs_f64(t.max(0.1))));
            }
            let conn = mysql::Conn::new(builder)
                .map_err(|e| format!("MySQL connection failed: {}", e))?;
            (DbBackend::Mysql(conn), SqlDialect::Mysql)
        } else if lower.starts_with("mssql://") || lower.starts_with("sqlserver://") {
            let client = connect_mssql(&url, timeout)?;
            (DbBackend::Mssql(client), SqlDialect::Mssql)
        } else {
            return Err(format!(
                "Unsupported database URL scheme (expected sqlite/postgres/mysql/mssql): {}",
                url
            ));
        };
        Ok(Self {
            backend,
            url,
            dialect,
            echo,
            echo_pool,
            pool_size,
            max_overflow,
            timeout,
            connect_args,
            schema_cache: None,
            declared_cache: HashMap::new(),
        })
    }

    pub fn dialect(&self) -> SqlDialect {
        self.dialect
    }

    /// Ensure `_datacode_version` + `_datacode_schema` exist (SQLite only).
    pub fn ensure_datacode_system_tables(&mut self) -> Result<(), String> {
        let DbBackend::SQLite(conn) = &mut self.backend else {
            return Ok(());
        };
        conn.execute_batch(ensure_system_tables_sql())
            .map_err(|e| e.to_string())?;
        let count: i64 = conn
            .query_row(
                &format!("SELECT COUNT(*) FROM {}", TABLE_VERSION),
                [],
                |r| r.get(0),
            )
            .map_err(|e| e.to_string())?;
        if count == 0 {
            conn.execute(
                &format!("INSERT INTO {} (version) VALUES (?1)", TABLE_VERSION),
                [SCHEMA_VERSION],
            )
            .map_err(|e| e.to_string())?;
        }
        // Invalidate cache so next read reloads.
        self.schema_cache = None;
        Ok(())
    }

    /// Register column types in the in-memory schema cache only (no disk I/O).
    ///
    /// Used by ORM `create_all` so typed reads work in-session without creating
    /// `_datacode_*` system tables (those are written by export / explicit upsert).
    pub fn register_datacode_schema_cache(
        &mut self,
        table_name: &str,
        columns: &[(String, String, String)],
    ) -> Result<(), String> {
        for (_col, dc, sql_ty) in columns {
            validate_schema_type_pair(dc, sql_ty)?;
        }
        if self.schema_cache.is_none() {
            self.schema_cache = Some(HashMap::new());
        }
        for (col, dc, sql_ty) in columns {
            if let Some(cache) = self.schema_cache.as_mut() {
                cache.insert(
                    (table_name.to_string(), col.clone()),
                    SchemaColumn {
                        table_name: table_name.to_string(),
                        column_name: col.clone(),
                        datacode_type: dc.clone(),
                        sqlite_type: sql_ty.clone(),
                        nullable: true,
                    },
                );
            }
            self.declared_cache
                .insert((table_name.to_string(), col.clone()), sql_ty.clone());
        }
        Ok(())
    }

    /// Upsert column metadata into `_datacode_schema` for one table.
    /// Each entry is `(column_name, datacode_type, sqlite_type)`.
    pub fn upsert_datacode_schema(
        &mut self,
        table_name: &str,
        columns: &[(String, String, String)],
    ) -> Result<(), String> {
        self.ensure_datacode_system_tables()?;
        {
            let DbBackend::SQLite(conn) = &mut self.backend else {
                return Ok(());
            };
            let mut stmt = conn
                .prepare(&format!(
                    "INSERT OR REPLACE INTO {}
                    (table_name, column_name, datacode_type, sqlite_type, nullable, version)
                    VALUES (?1, ?2, ?3, ?4, 1, ?5)",
                    TABLE_SCHEMA
                ))
                .map_err(|e| e.to_string())?;
            for (col, dc, sql_ty) in columns {
                validate_schema_type_pair(dc, sql_ty)?;
                stmt.execute(rusqlite::params![
                    table_name,
                    col,
                    dc,
                    sql_ty,
                    SCHEMA_VERSION
                ])
                .map_err(|e| e.to_string())?;
            }
        }
        self.register_datacode_schema_cache(table_name, columns)
    }

    /// Stub: sync `_datacode_schema` after ALTER TABLE (no ALTER API yet).
    pub fn sync_schema_after_alter(
        &mut self,
        _table_name: &str,
    ) -> Result<(), String> {
        // When ALTER TABLE is added, update `_datacode_schema` in the same transaction.
        Err(
            "sync_schema_after_alter: ALTER TABLE is not implemented; schema sync is a stub"
                .to_string(),
        )
    }

    fn ensure_schema_cache_loaded(&mut self) {
        if self.schema_cache.is_some() {
            return;
        }
        let cache = match &self.backend {
            DbBackend::SQLite(conn) => load_datacode_schema(conn),
            _ => HashMap::new(),
        };
        self.schema_cache = Some(cache);
    }

    fn ensure_declared_for_table(&mut self, table: &str) {
        if self
            .declared_cache
            .keys()
            .any(|(t, _)| t.eq_ignore_ascii_case(table))
        {
            return;
        }
        let discovered = match &self.backend {
            DbBackend::SQLite(conn) => schema_discovery_from_pragma(conn, table),
            _ => Vec::new(),
        };
        for col in discovered {
            self.declared_cache.insert(
                (col.table_name.clone(), col.column_name.clone()),
                col.sqlite_type.clone(),
            );
            if let Some(cache) = self.schema_cache.as_mut() {
                let key = (col.table_name.clone(), col.column_name.clone());
                cache.entry(key).or_insert(col);
            }
        }
    }

    fn column_type_hints(
        &mut self,
        table_hint: Option<&str>,
        col_name: &str,
    ) -> (Option<String>, Option<String>) {
        self.ensure_schema_cache_loaded();
        if let Some(t) = table_hint {
            self.ensure_declared_for_table(t);
        }
        let cache = self.schema_cache.as_ref().unwrap();
        if let Some(t) = table_hint {
            let key = (t.to_string(), col_name.to_string());
            if let Some(sc) = cache.get(&key) {
                return (
                    Some(sc.datacode_type.clone()),
                    Some(sc.sqlite_type.clone()),
                );
            }
            if let Some(decl) = self.declared_cache.get(&key) {
                return (None, Some(decl.clone()));
            }
        }
        // Unique column-name match across schema
        let matches: Vec<_> = cache
            .values()
            .filter(|c| c.column_name == col_name)
            .collect();
        if matches.len() == 1 {
            return (
                Some(matches[0].datacode_type.clone()),
                Some(matches[0].sqlite_type.clone()),
            );
        }
        (None, None)
    }

    pub fn execute(&mut self, sql: &str, params: &[Value]) -> Result<i64, String> {
        if self.echo {
            eprintln!("[SQL] {}", sql);
        }
        match &mut self.backend {
            DbBackend::SQLite(conn) => {
                let params_vec: Vec<Box<dyn rusqlite::ToSql>> =
                    params.iter().map(value_to_sqlite_param).collect();
                let params_refs: Vec<&dyn rusqlite::ToSql> =
                    params_vec.iter().map(|b| b.as_ref()).collect();
                let count = conn
                    .execute(sql, params_refs.as_slice())
                    .map_err(|e| format!("Execute failed: {}", e))?;
                Ok(count as i64)
            }
            DbBackend::Postgres(client) => {
                let pg_params = values_to_pg_params(params);
                let refs: Vec<&(dyn PgToSql + Sync)> = pg_params
                    .iter()
                    .map(|p| p as &(dyn PgToSql + Sync))
                    .collect();
                let count = client
                    .execute(sql, &refs)
                    .map_err(|e| format!("Execute failed: {}", e))?;
                Ok(count as i64)
            }
            DbBackend::Mysql(conn) => {
                use mysql::prelude::Queryable;
                let params = values_to_mysql_params(params);
                conn.exec_drop(sql, params)
                    .map_err(|e| format!("Execute failed: {}", e))?;
                Ok(conn.affected_rows() as i64)
            }
            DbBackend::Mssql(client) => {
                let sql_owned = sql.to_string();
                let params_owned = params.to_vec();
                block_on(async {
                    let mut query = tiberius::Query::new(&sql_owned);
                    for p in &params_owned {
                        bind_mssql_param(&mut query, p);
                    }
                    let result = query
                        .execute(client)
                        .await
                        .map_err(|e| format!("Execute failed: {}", e))?;
                    Ok(result.rows_affected().iter().sum::<u64>() as i64)
                })
            }
        }
    }

    pub fn query(&mut self, sql: &str, params: &[Value]) -> Result<Table, String> {
        if self.echo {
            eprintln!("[SQL] {}", sql);
        }
        if matches!(self.backend, DbBackend::SQLite(_)) {
            let table_hint = extract_simple_from_table(sql);
            let (headers, sql_rows) = {
                let DbBackend::SQLite(conn) = &self.backend else {
                    unreachable!()
                };
                let params_vec: Vec<Box<dyn rusqlite::ToSql>> =
                    params.iter().map(value_to_sqlite_param).collect();
                let params_refs: Vec<&dyn rusqlite::ToSql> =
                    params_vec.iter().map(|b| b.as_ref()).collect();
                let mut stmt = conn
                    .prepare(sql)
                    .map_err(|e| format!("Prepare failed: {}", e))?;
                let column_count = stmt.column_count();
                let headers: Vec<String> = (0..column_count)
                    .map(|i| stmt.column_name(i).unwrap_or("").to_string())
                    .collect();
                let rows_iter = stmt
                    .query_map(params_refs.as_slice(), |row| {
                        let mut r = Vec::with_capacity(column_count);
                        for i in 0..column_count {
                            use rusqlite::types::Value as SqlValue;
                            let sql_val = row.get::<_, SqlValue>(i).unwrap_or(SqlValue::Null);
                            r.push(sql_val);
                        }
                        Ok(r)
                    })
                    .map_err(|e| format!("Query failed: {}", e))?;
                let mut sql_rows = Vec::new();
                for row_result in rows_iter {
                    sql_rows.push(row_result.map_err(|e| format!("Row error: {}", e))?);
                }
                (headers, sql_rows)
            };
            let mut hints = Vec::with_capacity(headers.len());
            for h in &headers {
                hints.push(self.column_type_hints(table_hint.as_deref(), h));
            }
            let mut rows = Vec::with_capacity(sql_rows.len());
            for sql_row in sql_rows {
                let mut r = Vec::with_capacity(headers.len());
                for (i, sql_val) in sql_row.into_iter().enumerate() {
                    let (dc, decl) = &hints[i];
                    r.push(sql_value_to_datacode(
                        sql_val,
                        dc.as_deref(),
                        decl.as_deref(),
                    ));
                }
                rows.push(r);
            }
            return Ok(Table::from_data(rows, Some(headers)));
        }
        match &mut self.backend {
            DbBackend::SQLite(_) => unreachable!(),
            DbBackend::Postgres(client) => {
                let pg_params = values_to_pg_params(params);
                let refs: Vec<&(dyn PgToSql + Sync)> = pg_params
                    .iter()
                    .map(|p| p as &(dyn PgToSql + Sync))
                    .collect();
                let rows = client
                    .query(sql, &refs)
                    .map_err(|e| format!("Query failed: {}", e))?;
                if rows.is_empty() {
                    // Still need headers — run a describe via empty result workaround.
                    return Ok(Table::from_data(vec![], Some(vec![])));
                }
                let headers: Vec<String> = rows[0]
                    .columns()
                    .iter()
                    .map(|c| c.name().to_string())
                    .collect();
                let mut out = Vec::new();
                for row in &rows {
                    let mut r = Vec::with_capacity(headers.len());
                    for i in 0..headers.len() {
                        r.push(pg_row_get_value(row, i));
                    }
                    out.push(r);
                }
                Ok(Table::from_data(out, Some(headers)))
            }
            DbBackend::Mysql(conn) => {
                use mysql::prelude::Queryable;
                let params = values_to_mysql_params(params);
                let result = conn
                    .exec_iter(sql, params)
                    .map_err(|e| format!("Query failed: {}", e))?;
                let columns = result.columns();
                let headers: Vec<String> = columns
                    .as_ref()
                    .iter()
                    .map(|c| c.name_str().into_owned())
                    .collect();
                let mut out = Vec::new();
                for row_result in result {
                    let row = row_result.map_err(|e| format!("Row error: {}", e))?;
                    let mut r = Vec::with_capacity(headers.len());
                    for i in 0..headers.len() {
                        r.push(mysql_row_get_value(&row, i));
                    }
                    out.push(r);
                }
                Ok(Table::from_data(out, Some(headers)))
            }
            DbBackend::Mssql(client) => {
                let sql_owned = sql.to_string();
                let params_owned = params.to_vec();
                block_on(async {
                    let mut query = tiberius::Query::new(&sql_owned);
                    for p in &params_owned {
                        bind_mssql_param(&mut query, p);
                    }
                    let stream = query
                        .query(client)
                        .await
                        .map_err(|e| format!("Query failed: {}", e))?;
                    let rows = stream
                        .into_results()
                        .await
                        .map_err(|e| format!("Query failed: {}", e))?;
                    let mut headers = Vec::new();
                    let mut out = Vec::new();
                    if let Some(first_set) = rows.first() {
                        if let Some(first_row) = first_set.first() {
                            headers = first_row
                                .columns()
                                .iter()
                                .map(|c| c.name().to_string())
                                .collect();
                        }
                        for row in first_set {
                            if headers.is_empty() {
                                headers = row
                                    .columns()
                                    .iter()
                                    .map(|c| c.name().to_string())
                                    .collect();
                            }
                            let mut r = Vec::with_capacity(headers.len());
                            for i in 0..headers.len() {
                                r.push(mssql_row_get_value(row, i));
                            }
                            out.push(r);
                        }
                    }
                    Ok(Table::from_data(out, Some(headers)))
                })
            }
        }
    }
}

fn connect_mssql(
    url: &str,
    timeout: Option<f64>,
) -> Result<MssqlClient<Compat<TcpStream>>, String> {
    let parsed = parse_mssql_url(url)?;
    let mut config = MssqlConfig::new();
    config.host(&parsed.host);
    config.port(parsed.port);
    config.database(&parsed.database);
    config.authentication(AuthMethod::sql_server(&parsed.user, &parsed.password));
    config.encryption(EncryptionLevel::NotSupported);
    if let Some(t) = timeout {
        let _ = t; // tiberius uses connection timeout via TCP
    }
    block_on(async {
        let tcp = TcpStream::connect(parsed.addr())
            .await
            .map_err(|e| format!("MSSQL TCP connect failed: {}", e))?;
        tcp.set_nodelay(true)
            .map_err(|e| format!("MSSQL set_nodelay failed: {}", e))?;
        let client = MssqlClient::connect(config, tcp.compat_write())
            .await
            .map_err(|e| format!("MSSQL connection failed: {}", e))?;
        Ok(client)
    })
}

struct MssqlUrl {
    host: String,
    port: u16,
    database: String,
    user: String,
    password: String,
}

impl MssqlUrl {
    fn addr(&self) -> String {
        format!("{}:{}", self.host, self.port)
    }
}

fn parse_mssql_url(url: &str) -> Result<MssqlUrl, String> {
    let rest = url
        .strip_prefix("mssql://")
        .or_else(|| url.strip_prefix("sqlserver://"))
        .ok_or_else(|| format!("Invalid MSSQL URL: {}", url))?;
    // user:pass@host:port/db
    let (auth, hostpart) = rest
        .split_once('@')
        .ok_or_else(|| "MSSQL URL requires user:pass@host".to_string())?;
    let (user, password) = auth
        .split_once(':')
        .map(|(u, p)| (u.to_string(), p.to_string()))
        .unwrap_or((auth.to_string(), String::new()));
    let (hostport, database) = hostpart
        .split_once('/')
        .map(|(h, d)| (h, d.split('?').next().unwrap_or(d).to_string()))
        .unwrap_or((hostpart, String::new()));
    let (host, port) = if let Some((h, p)) = hostport.split_once(':') {
        (
            h.to_string(),
            p.parse::<u16>()
                .map_err(|_| format!("Invalid MSSQL port: {}", p))?,
        )
    } else {
        (hostport.to_string(), 1433u16)
    };
    Ok(MssqlUrl {
        host,
        port,
        database,
        user,
        password,
    })
}

fn value_to_sqlite_param(v: &Value) -> Box<dyn rusqlite::ToSql> {
    Box::new(value_to_owned_sql(v, None)) as Box<dyn rusqlite::ToSql>
}

fn load_datacode_schema(conn: &Connection) -> SchemaCache {
    let mut cache = HashMap::new();
    let Ok(mut stmt) = conn.prepare(&format!(
        "SELECT table_name, column_name, datacode_type, sqlite_type, nullable
         FROM {}",
        TABLE_SCHEMA
    )) else {
        return cache;
    };
    let Ok(rows) = stmt.query_map([], |row| {
        Ok(SchemaColumn {
            table_name: row.get::<_, String>(0)?,
            column_name: row.get::<_, String>(1)?,
            datacode_type: row.get::<_, String>(2)?,
            sqlite_type: row.get::<_, String>(3)?,
            nullable: row.get::<_, i64>(4).unwrap_or(1) != 0,
        })
    }) else {
        return cache;
    };
    for row in rows.flatten() {
        cache.insert((row.table_name.clone(), row.column_name.clone()), row);
    }
    cache
}

/// Best-effort extract of a single table name from simple `FROM table` SQL.
fn extract_simple_from_table(sql: &str) -> Option<String> {
    let lower = sql.to_ascii_lowercase();
    let idx = lower.find(" from ")?;
    let after = sql[idx + 6..].trim_start();
    let token = after
        .split(|c: char| c.is_whitespace() || c == ',' || c == ';' || c == ')')
        .next()?
        .trim_matches(|c| c == '"' || c == '`' || c == '[' || c == ']');
    if token.is_empty() || token.eq_ignore_ascii_case("select") {
        return None;
    }
    // Skip subquery
    if token.starts_with('(') {
        return None;
    }
    Some(token.to_string())
}

fn parse_sqlite_path(url: &str) -> Result<PathBuf, String> {
    let url = url.trim();
    if url.starts_with("sqlite:///") {
        Ok(PathBuf::from(&url["sqlite:///".len()..]))
    } else if url.starts_with("sqlite:") {
        Ok(PathBuf::from(&url["sqlite:".len()..]))
    } else {
        Err(format!("Invalid SQLite URL: {}", url))
    }
}

#[derive(Debug)]
enum PgParam {
    Null,
    Bool(bool),
    I64(i64),
    F64(f64),
    String(String),
    Bytes(Vec<u8>),
}

impl PgToSql for PgParam {
    fn to_sql(
        &self,
        ty: &postgres::types::Type,
        out: &mut bytes::BytesMut,
    ) -> Result<postgres::types::IsNull, Box<dyn std::error::Error + Sync + Send>> {
        match self {
            PgParam::Null => Ok(postgres::types::IsNull::Yes),
            PgParam::Bool(b) => b.to_sql(ty, out),
            PgParam::I64(i) => i.to_sql(ty, out),
            PgParam::F64(f) => f.to_sql(ty, out),
            PgParam::String(s) => s.to_sql(ty, out),
            PgParam::Bytes(b) => b.to_sql(ty, out),
        }
    }

    fn accepts(ty: &postgres::types::Type) -> bool {
        true || ty == &postgres::types::Type::TEXT
    }

    postgres::types::to_sql_checked!();
}

fn values_to_pg_params(params: &[Value]) -> Vec<PgParam> {
    params
        .iter()
        .map(|v| match v {
            Value::Null => PgParam::Null,
            Value::Bool(b) => PgParam::Bool(*b),
            Value::Number(n) => {
                if n.fract() == 0.0 {
                    PgParam::I64(*n as i64)
                } else {
                    PgParam::F64(*n)
                }
            }
            Value::Int(crate::common::numeric::IntValue::Finite(i)) => PgParam::I64(*i),
            Value::String(s) => PgParam::String(s.clone()),
            Value::ByteBuffer(b) => {
                PgParam::Bytes(b.bytes[b.offset..b.offset + b.len].to_vec())
            }
            other => PgParam::String(other.to_string()),
        })
        .collect()
}

fn pg_row_get_value(row: &postgres::Row, idx: usize) -> Value {
    if let Ok(v) = row.try_get::<_, Option<i64>>(idx) {
        return match v {
            Some(i) => Value::Number(i as f64),
            None => Value::Null,
        };
    }
    if let Ok(v) = row.try_get::<_, Option<f64>>(idx) {
        return match v {
            Some(n) => Value::Number(n),
            None => Value::Null,
        };
    }
    if let Ok(v) = row.try_get::<_, Option<bool>>(idx) {
        return match v {
            Some(b) => Value::Bool(b),
            None => Value::Null,
        };
    }
    if let Ok(v) = row.try_get::<_, Option<String>>(idx) {
        return match v {
            Some(s) => Value::String(s),
            None => Value::Null,
        };
    }
    if let Ok(v) = row.try_get::<_, Option<Vec<u8>>>(idx) {
        return match v {
            Some(b) => Value::ByteBuffer(ByteBuffer::from_vec(b)),
            None => Value::Null,
        };
    }
    Value::Null
}

fn values_to_mysql_params(params: &[Value]) -> mysql::Params {
    use mysql::Value as Mv;
    let vals: Vec<Mv> = params
        .iter()
        .map(|v| match v {
            Value::Null => Mv::NULL,
            Value::Bool(b) => Mv::Int(if *b { 1 } else { 0 }),
            Value::Number(n) => {
                if n.fract() == 0.0 {
                    Mv::Int(*n as i64)
                } else {
                    Mv::Double(*n)
                }
            }
            Value::Int(crate::common::numeric::IntValue::Finite(i)) => Mv::Int(*i),
            Value::String(s) => Mv::Bytes(s.as_bytes().to_vec()),
            Value::ByteBuffer(b) => {
                Mv::Bytes(b.bytes[b.offset..b.offset + b.len].to_vec())
            }
            other => Mv::Bytes(other.to_string().into_bytes()),
        })
        .collect();
    mysql::Params::Positional(vals)
}

fn mysql_row_get_value(row: &mysql::Row, idx: usize) -> Value {
    use mysql::Value as Mv;
    match row.get_opt(idx) {
        Some(Ok(Mv::NULL)) | None => Value::Null,
        Some(Ok(Mv::Int(i))) => Value::Number(i as f64),
        Some(Ok(Mv::UInt(u))) => Value::Number(u as f64),
        Some(Ok(Mv::Float(f))) => Value::Number(f as f64),
        Some(Ok(Mv::Double(d))) => Value::Number(d),
        Some(Ok(Mv::Bytes(b))) => match String::from_utf8(b.clone()) {
            Ok(s) => Value::String(s),
            Err(_) => Value::ByteBuffer(ByteBuffer::from_vec(b)),
        },
        Some(Ok(Mv::Date(..))) => Value::String(format!("{:?}", row.get::<Mv, _>(idx))),
        Some(Ok(Mv::Time(..))) => Value::String(format!("{:?}", row.get::<Mv, _>(idx))),
        Some(Err(_)) => Value::Null,
    }
}

fn bind_mssql_param(query: &mut tiberius::Query<'_>, v: &Value) {
    match v {
        Value::Null => {
            query.bind(Option::<String>::None);
        }
        Value::Bool(b) => {
            query.bind(*b);
        }
        Value::Number(n) => {
            if n.fract() == 0.0 {
                query.bind(*n as i64);
            } else {
                query.bind(*n);
            }
        }
        Value::Int(crate::common::numeric::IntValue::Finite(i)) => {
            query.bind(*i);
        }
        Value::String(s) => {
            query.bind(s.clone());
        }
        Value::ByteBuffer(b) => {
            let bytes = b.bytes[b.offset..b.offset + b.len].to_vec();
            query.bind(bytes);
        }
        other => {
            query.bind(other.to_string());
        }
    }
}

fn mssql_row_get_value(row: &tiberius::Row, idx: usize) -> Value {
    if let Ok(v) = row.try_get::<i64, _>(idx) {
        return match v {
            Some(i) => Value::Number(i as f64),
            None => Value::Null,
        };
    }
    if let Ok(v) = row.try_get::<f64, _>(idx) {
        return match v {
            Some(n) => Value::Number(n),
            None => Value::Null,
        };
    }
    if let Ok(v) = row.try_get::<bool, _>(idx) {
        return match v {
            Some(b) => Value::Bool(b),
            None => Value::Null,
        };
    }
    if let Ok(v) = row.try_get::<&str, _>(idx) {
        return match v {
            Some(s) => Value::String(s.to_string()),
            None => Value::Null,
        };
    }
    if let Ok(v) = row.try_get::<&[u8], _>(idx) {
        return match v {
            Some(b) => Value::ByteBuffer(ByteBuffer::from_vec(b.to_vec())),
            None => Value::Null,
        };
    }
    Value::Null
}

/// Build INSERT placeholders for dialect: `?` / `$1,$2` / `@P1,@P2`.
pub fn insert_placeholders(dialect: SqlDialect, n: usize) -> String {
    match dialect {
        SqlDialect::Sqlite | SqlDialect::Mysql => (0..n)
            .map(|_| "?".to_string())
            .collect::<Vec<_>>()
            .join(", "),
        SqlDialect::Postgres => (1..=n)
            .map(|i| format!("${}", i))
            .collect::<Vec<_>>()
            .join(", "),
        SqlDialect::Mssql => (1..=n)
            .map(|i| format!("@P{}", i))
            .collect::<Vec<_>>()
            .join(", "),
    }
}
