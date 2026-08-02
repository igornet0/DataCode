// Database engine abstraction for multiple backend types

use crate::common::table::Table;
use crate::common::value::ByteBuffer;
use crate::common::value::Value;
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
        })
    }

    pub fn dialect(&self) -> SqlDialect {
        self.dialect
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
        match &mut self.backend {
            DbBackend::SQLite(conn) => {
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
                            r.push(sqlite_row_get_value(row, i));
                        }
                        Ok(r)
                    })
                    .map_err(|e| format!("Query failed: {}", e))?;
                let mut rows = Vec::new();
                for row_result in rows_iter {
                    rows.push(row_result.map_err(|e| format!("Row error: {}", e))?);
                }
                Ok(Table::from_data(rows, Some(headers)))
            }
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

fn value_to_sqlite_param(v: &Value) -> Box<dyn rusqlite::ToSql> {
    match v {
        Value::Number(n) => {
            if n.fract() == 0.0 {
                Box::new(*n as i64) as Box<dyn rusqlite::ToSql>
            } else {
                Box::new(*n) as Box<dyn rusqlite::ToSql>
            }
        }
        Value::Bool(b) => Box::new(if *b { 1i64 } else { 0i64 }) as Box<dyn rusqlite::ToSql>,
        Value::String(s) => Box::new(s.clone()) as Box<dyn rusqlite::ToSql>,
        Value::Null => Box::new(Option::<String>::None) as Box<dyn rusqlite::ToSql>,
        Value::ByteBuffer(b) => {
            let blob = b.bytes[b.offset..b.offset + b.len].to_vec();
            Box::new(blob) as Box<dyn rusqlite::ToSql>
        }
        _ => Box::new(v.to_string()) as Box<dyn rusqlite::ToSql>,
    }
}

fn sqlite_row_get_value(row: &rusqlite::Row, idx: usize) -> Value {
    use rusqlite::types::Value as SqlValue;
    let sql_val = match row.get::<_, SqlValue>(idx) {
        Ok(v) => v,
        Err(_) => return Value::Null,
    };
    match sql_val {
        SqlValue::Integer(i) => Value::Number(i as f64),
        SqlValue::Real(r) => Value::Number(r),
        SqlValue::Text(s) => Value::String(s),
        SqlValue::Blob(bytes) => Value::ByteBuffer(ByteBuffer::from_vec(bytes)),
        SqlValue::Null => Value::Null,
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
