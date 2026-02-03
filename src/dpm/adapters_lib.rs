//! Generate adapters/__lib__.dc from a list of used DB types (only those adapters are imported and loaded).
//! Also provides DB type metadata (port, driver, pool, prompts) for init_database and add_database.

/// (display label, db_type) for the selector. Same order as init_database DB_TYPES.
pub const DB_TYPE_LABELS: &[(&str, &str)] = &[
    ("SQLite", "sqlite"),
    ("PostgreSQL", "postgresql"),
    ("MySQL", "mysql"),
    ("MariaDB", "mariadb"),
    ("Microsoft SQL Server", "mssql"),
    ("Oracle", "oracle"),
    ("MongoDB", "mongodb"),
    ("CouchDB", "couchdb"),
    ("Redis", "redis"),
    ("Memcached", "memcached"),
    ("Neo4j", "neo4j"),
    ("ArangoDB", "arangodb"),
    ("ClickHouse", "clickhouse"),
    ("Cassandra", "cassandra"),
    ("InfluxDB", "influxdb"),
];

pub fn default_port(db_type: &str) -> u16 {
    match db_type {
        "sqlite" => 0,
        "postgresql" => 5432,
        "mysql" | "mariadb" => 3306,
        "mssql" => 1433,
        "oracle" => 1521,
        "mongodb" => 27017,
        "couchdb" => 5984,
        "redis" => 6379,
        "memcached" => 11211,
        "neo4j" => 7687,
        "arangodb" => 8529,
        "clickhouse" => 9000,
        "cassandra" => 9042,
        "influxdb" => 8086,
        _ => 5432,
    }
}

pub fn default_driver(db_type: &str) -> &'static str {
    match db_type {
        "sqlite" => "sqlite3",
        "postgresql" => "psycopg",
        "mysql" | "mariadb" => "pymysql",
        "mssql" => "pyodbc",
        "oracle" => "cx_oracle",
        "mongodb" => "pymongo",
        "couchdb" => "httpx",
        "redis" => "redis",
        "memcached" => "pymemcache",
        "neo4j" => "neo4j",
        "arangodb" => "python-arango",
        "clickhouse" => "clickhouse-connect",
        "cassandra" => "cassandra-driver",
        "influxdb" => "influxdb-client",
        _ => "psycopg",
    }
}

pub fn default_pool_size(db_type: &str) -> u32 {
    match db_type {
        "sqlite" => 5,
        "postgresql" => 10,
        "mysql" | "mariadb" | "mssql" | "oracle" | "couchdb" | "memcached" | "neo4j" | "arangodb" | "cassandra" | "influxdb" => 5,
        "mongodb" | "redis" => 10,
        "clickhouse" => 10,
        _ => 10,
    }
}

pub fn default_pool_max_overflow(db_type: &str) -> u32 {
    match db_type {
        "sqlite" => 10,
        "postgresql" | "clickhouse" => 20,
        "mysql" | "mariadb" | "mssql" | "couchdb" | "memcached" | "neo4j" | "arangodb" | "cassandra" | "influxdb" => 10,
        "oracle" => 5,
        "mongodb" => 50,
        "redis" => 20,
        _ => 20,
    }
}

/// (prompt text, default value) for the name field.
pub fn name_prompt(db_type: &str) -> (&'static str, &'static str) {
    match db_type {
        "sqlite" => ("Database file path (e.g. mydb.sqlite)", "mydb.sqlite"),
        "oracle" => ("Service name", "orclpdb1"),
        "cassandra" => ("Keyspace", "mykeyspace"),
        _ => ("Database name", "my_app_db"),
    }
}

pub fn asks_user_password(db_type: &str) -> bool {
    !matches!(db_type, "sqlite" | "redis" | "memcached")
}

pub fn default_username(db_type: &str) -> &'static str {
    match db_type {
        "postgresql" => "postgres",
        "oracle" => "system",
        _ => "user",
    }
}

/// Map db_type (config key) -> (module file name without .dc, adapter class name).
pub fn adapter_module_and_class(db_type: &str) -> Option<(&'static str, &'static str)> {
    match db_type {
        "postgresql" => Some(("postgres", "PostgresAdapter")),
        "sqlite" => Some(("sqlite", "SqliteAdapter")),
        "mysql" => Some(("mysql", "MysqlAdapter")),
        "mariadb" => Some(("mariadb", "MariadbAdapter")),
        "mongodb" => Some(("mongodb", "MongodbAdapter")),
        "mssql" => Some(("mssql", "MssqlAdapter")),
        "oracle" => Some(("oracle", "OracleAdapter")),
        "couchdb" => Some(("couchdb", "CouchdbAdapter")),
        "redis" => Some(("redis", "RedisAdapter")),
        "memcached" => Some(("memcached", "MemcachedAdapter")),
        "neo4j" => Some(("neo4j", "Neo4jAdapter")),
        "arangodb" => Some(("arangodb", "ArangodbAdapter")),
        "clickhouse" => Some(("clickhouse", "ClickhouseAdapter")),
        "cassandra" => Some(("cassandra", "CassandraAdapter")),
        "influxdb" => Some(("influxdb", "InfluxdbAdapter")),
        _ => None,
    }
}

/// Generate content of core/database/adapters/__lib__.dc for the given list of used DB types.
/// Only these adapters are imported and exposed in get_adapter, so only they are loaded at runtime.
pub fn render_adapters_lib(used_types: &[&str]) -> String {
    let mut imports = String::from("from core.database.adapters.base import BaseAdapter\n");
    let mut branches = String::new();

    for db_type in used_types {
        if let Some((module, class)) = adapter_module_and_class(db_type) {
            imports.push_str(&format!(
                "from core.database.adapters.{} import {}\n",
                module, class
            ));
            branches.push_str(&format!(
                "    if db_type == \"{}\" {{\n        return {}()\n    }}\n",
                db_type, class
            ));
        }
    }

    format!(
        "{}fn get_adapter(db_type, driver) {{\n{}    raise NotImplementedError()\n}}\n",
        imports, branches
    )
}
