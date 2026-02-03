//! Interactive wizard for `dpm init database`: creates core/database/ with config, engine, adapters, migrations.

use std::collections::HashMap;
use std::io::{self, IsTerminal, Write};
use std::path::Path;

use dialoguer::Select;

use super::adapters_lib;

/// Database type choice with default port, driver, and pool settings.
#[derive(Clone, Copy, PartialEq)]
enum DbType {
    Sqlite,
    PostgreSQL,
    MySQL,
    MariaDB,
    Mssql,
    Oracle,
    MongoDB,
    CouchDB,
    Redis,
    Memcached,
    Neo4j,
    ArangoDB,
    ClickHouse,
    Cassandra,
    InfluxDB,
}

const DB_TYPES: &[(DbType, &str)] = &[
    (DbType::Sqlite, "SQLite"),
    (DbType::PostgreSQL, "PostgreSQL"),
    (DbType::MySQL, "MySQL"),
    (DbType::MariaDB, "MariaDB"),
    (DbType::Mssql, "Microsoft SQL Server"),
    (DbType::Oracle, "Oracle"),
    (DbType::MongoDB, "MongoDB"),
    (DbType::CouchDB, "CouchDB"),
    (DbType::Redis, "Redis"),
    (DbType::Memcached, "Memcached"),
    (DbType::Neo4j, "Neo4j"),
    (DbType::ArangoDB, "ArangoDB"),
    (DbType::ClickHouse, "ClickHouse"),
    (DbType::Cassandra, "Cassandra"),
    (DbType::InfluxDB, "InfluxDB"),
];

impl DbType {
    fn default_port(self) -> u16 {
        match self {
            DbType::Sqlite => 0,
            DbType::PostgreSQL => 5432,
            DbType::MySQL | DbType::MariaDB => 3306,
            DbType::Mssql => 1433,
            DbType::Oracle => 1521,
            DbType::MongoDB => 27017,
            DbType::CouchDB => 5984,
            DbType::Redis => 6379,
            DbType::Memcached => 11211,
            DbType::Neo4j => 7687,
            DbType::ArangoDB => 8529,
            DbType::ClickHouse => 9000,
            DbType::Cassandra => 9042,
            DbType::InfluxDB => 8086,
        }
    }
    fn as_str(self) -> &'static str {
        match self {
            DbType::Sqlite => "sqlite",
            DbType::PostgreSQL => "postgresql",
            DbType::MySQL => "mysql",
            DbType::MariaDB => "mariadb",
            DbType::Mssql => "mssql",
            DbType::Oracle => "oracle",
            DbType::MongoDB => "mongodb",
            DbType::CouchDB => "couchdb",
            DbType::Redis => "redis",
            DbType::Memcached => "memcached",
            DbType::Neo4j => "neo4j",
            DbType::ArangoDB => "arangodb",
            DbType::ClickHouse => "clickhouse",
            DbType::Cassandra => "cassandra",
            DbType::InfluxDB => "influxdb",
        }
    }
    fn default_driver(self) -> &'static str {
        match self {
            DbType::Sqlite => "sqlite3",
            DbType::PostgreSQL => "psycopg",
            DbType::MySQL | DbType::MariaDB => "pymysql",
            DbType::Mssql => "pyodbc",
            DbType::Oracle => "cx_oracle",
            DbType::MongoDB => "pymongo",
            DbType::CouchDB => "httpx",
            DbType::Redis => "redis",
            DbType::Memcached => "pymemcache",
            DbType::Neo4j => "neo4j",
            DbType::ArangoDB => "python-arango",
            DbType::ClickHouse => "clickhouse-connect",
            DbType::Cassandra => "cassandra-driver",
            DbType::InfluxDB => "influxdb-client",
        }
    }
    fn default_pool_size(self) -> u32 {
        match self {
            DbType::Sqlite => 5,
            DbType::PostgreSQL => 10,
            DbType::MySQL | DbType::MariaDB => 5,
            DbType::Mssql => 5,
            DbType::Oracle => 5,
            DbType::MongoDB => 10,
            DbType::CouchDB => 5,
            DbType::Redis => 10,
            DbType::Memcached => 5,
            DbType::Neo4j => 5,
            DbType::ArangoDB => 5,
            DbType::ClickHouse => 10,
            DbType::Cassandra => 5,
            DbType::InfluxDB => 5,
        }
    }
    fn default_pool_max_overflow(self) -> u32 {
        match self {
            DbType::Sqlite => 10,
            DbType::PostgreSQL => 20,
            DbType::MySQL | DbType::MariaDB => 10,
            DbType::Mssql => 10,
            DbType::Oracle => 5,
            DbType::MongoDB => 50,
            DbType::CouchDB => 10,
            DbType::Redis => 20,
            DbType::Memcached => 5,
            DbType::Neo4j => 10,
            DbType::ArangoDB => 10,
            DbType::ClickHouse => 20,
            DbType::Cassandra => 10,
            DbType::InfluxDB => 10,
        }
    }
    /// (prompt text, default value) for the "name" field (database name, file path, keyspace, service name).
    fn name_prompt(self) -> (&'static str, &'static str) {
        match self {
            DbType::Sqlite => ("Database file path (e.g. mydb.sqlite)", "mydb.sqlite"),
            DbType::Oracle => ("Service name", "orclpdb1"),
            DbType::Cassandra => ("Keyspace", "mykeyspace"),
            _ => ("Database name", "my_app_db"),
        }
    }
    /// Whether to ask for username and password (false for SQLite, Redis, Memcached).
    fn asks_user_password(self) -> bool {
        match self {
            DbType::Sqlite | DbType::Redis | DbType::Memcached => false,
            _ => true,
        }
    }
}

const BLUE: &str = "\x1b[34m";
const GREEN: &str = "\x1b[32m";
const RESET: &str = "\x1b[0m";

fn colored_prompt(question: &str, bracket_content: &str) {
    let display = if bracket_content.is_empty() {
        ""
    } else {
        bracket_content
    };
    if io::stdout().is_terminal() {
        print!("{}{}{} [{}{}{}]: ", BLUE, question, RESET, GREEN, display, RESET);
    } else {
        print!("{} [{}]: ", question, display);
    }
}

fn prompt(default: &str) -> Result<String, String> {
    io::stdout().flush().map_err(|e| e.to_string())?;
    let mut line = String::new();
    io::stdin().read_line(&mut line).map_err(|e| e.to_string())?;
    let s = line.trim().to_string();
    if s.is_empty() {
        Ok(default.to_string())
    } else {
        Ok(s)
    }
}

fn prompt_yes_no(default_yes: bool) -> Result<bool, String> {
    let default = if default_yes { "yes" } else { "no" };
    let s = prompt(default)?;
    let y = s.is_empty()
        || s.eq_ignore_ascii_case("y")
        || s.eq_ignore_ascii_case("yes")
        || s == "1";
    Ok(y)
}

fn prompt_password() -> Result<String, String> {
    if io::stdout().is_terminal() {
        // Try to hide input (disable echo). On Unix we could use rpassword crate; for now just prompt.
        print!("{}Password{}: ", BLUE, RESET);
        io::stdout().flush().map_err(|e| e.to_string())?;
    } else {
        print!("Password: ");
        io::stdout().flush().map_err(|e| e.to_string())?;
    }
    let mut line = String::new();
    io::stdin().read_line(&mut line).map_err(|e| e.to_string())?;
    Ok(line.trim().to_string())
}

/// Replace {{ key }} in template with values from map.
fn render_template(tpl: &str, vars: &HashMap<String, String>) -> String {
    let mut out = tpl.to_string();
    for (key, value) in vars {
        let placeholder = format!("{{{{ {} }}}}", key);
        out = out.replace(&placeholder, value);
    }
    out
}

/// Load embedded template (built from repo root).
macro_rules! embed_tpl {
    ($path:literal) => {
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), $path))
    };
}

/// Return adapter template content for the given db_type (e.g. "postgresql", "redis").
/// Used by init_database and add_database when writing a single adapter file.
pub fn get_adapter_template(db_type: &str) -> Result<&'static str, String> {
    let tpl = match db_type {
        "postgresql" => embed_tpl!("/templates/database/adapters/postgres.dc.tpl"),
        "sqlite" => embed_tpl!("/templates/database/adapters/sqlite.dc.tpl"),
        "mysql" => embed_tpl!("/templates/database/adapters/mysql.dc.tpl"),
        "mariadb" => embed_tpl!("/templates/database/adapters/mariadb.dc.tpl"),
        "mongodb" => embed_tpl!("/templates/database/adapters/mongodb.dc.tpl"),
        "mssql" => embed_tpl!("/templates/database/adapters/mssql.dc.tpl"),
        "oracle" => embed_tpl!("/templates/database/adapters/oracle.dc.tpl"),
        "couchdb" => embed_tpl!("/templates/database/adapters/couchdb.dc.tpl"),
        "redis" => embed_tpl!("/templates/database/adapters/redis.dc.tpl"),
        "memcached" => embed_tpl!("/templates/database/adapters/memcached.dc.tpl"),
        "neo4j" => embed_tpl!("/templates/database/adapters/neo4j.dc.tpl"),
        "arangodb" => embed_tpl!("/templates/database/adapters/arangodb.dc.tpl"),
        "clickhouse" => embed_tpl!("/templates/database/adapters/clickhouse.dc.tpl"),
        "cassandra" => embed_tpl!("/templates/database/adapters/cassandra.dc.tpl"),
        "influxdb" => embed_tpl!("/templates/database/adapters/influxdb.dc.tpl"),
        _ => return Err(format!("Unknown db_type for adapter template: {}", db_type)),
    };
    Ok(tpl)
}

/// Run the database init wizard and create core/database/.
/// flags: e.g. ["--async", "--no-migrations"] for non-interactive overrides.
pub fn run_init_database(project_root: &Path, flags: &[String]) -> Result<(), String> {
    let flag_async = flags.iter().any(|s| s == "--async");
    let flag_no_migrations = flags.iter().any(|s| s == "--no-migrations");
    let flag_no_pooling = flags.iter().any(|s| s == "--no-pooling");

    println!("Initializing database module...");
    println!();

    // Step 1: database type (selector in TTY, else numeric prompt)
    let db_type = if io::stdout().is_terminal() {
        let items: Vec<&str> = DB_TYPES.iter().map(|(_, label)| *label).collect();
        let idx = Select::with_theme(&dialoguer::theme::ColorfulTheme::default())
            .with_prompt("? Select database type")
            .items(&items)
            .default(0)
            .interact()
            .map_err(|e| e.to_string())?;
        DB_TYPES[idx].0
    } else {
        println!("? Select database type:");
        for (i, (_, label)) in DB_TYPES.iter().enumerate() {
            println!("  {}) {}", i + 1, label);
        }
        colored_prompt("Enter number", "1");
        let num_str = prompt("1")?;
        let idx: usize = num_str
            .trim()
            .parse()
            .map_err(|_| "Invalid number")?;
        DB_TYPES
            .get(idx.wrapping_sub(1))
            .copied()
            .map(|(t, _)| t)
            .ok_or("Invalid selection")?
    };

    let db_type_str = db_type.as_str().to_string();
    let default_port = db_type.default_port();
    let driver = db_type.default_driver().to_string();
    let (name_prompt_text, name_default) = db_type.name_prompt();

    // Step 2: connection params (SQLite: only file path; others: host, port, name)
    let name = if db_type == DbType::Sqlite {
        colored_prompt(name_prompt_text, name_default);
        prompt(name_default)?
    } else {
        colored_prompt(name_prompt_text, name_default);
        prompt(name_default)?
    };

    let (user, password_expr) = if !db_type.asks_user_password() {
        (String::new(), "\"\"".to_string())
    } else {
        let default_user = match db_type {
            DbType::PostgreSQL => "postgres",
            DbType::Oracle => "system",
            _ => "user",
        };
        colored_prompt("Username", default_user);
        let user = prompt(default_user)?;
        let pass = prompt_password()?;
        let password_expr = if pass.is_empty() {
            "env(\"DB_PASSWORD\")".to_string()
        } else {
            format!("\"{}\"", pass.replace('\\', "\\\\").replace('"', "\\\""))
        };
        (user, password_expr)
    };

    let (host, port) = if db_type == DbType::Sqlite {
        ("localhost".to_string(), 0u16)
    } else {
        colored_prompt("Host", "localhost");
        let host = prompt("localhost")?;
        colored_prompt("Port", &default_port.to_string());
        let port_str = prompt(&default_port.to_string())?;
        let port: u16 = port_str.trim().parse().unwrap_or(default_port);
        (host, port)
    };

    // Step 3: features
    let migrations = if flag_no_migrations {
        false
    } else {
        colored_prompt("Enable migrations? (Y/n)", "yes");
        prompt_yes_no(true)?
    };
    let pool_enabled = if flag_no_pooling {
        false
    } else {
        colored_prompt("Enable connection pooling? (Y/n)", "yes");
        prompt_yes_no(true)?
    };
    let async_support = if flag_async {
        true
    } else {
        colored_prompt("Enable async support? (y/N)", "no");
        prompt_yes_no(false)?
    };

    println!();

    // Build template vars
    let mut vars = HashMap::<String, String>::new();
    vars.insert("db_type".to_string(), db_type_str.clone());
    vars.insert("driver".to_string(), driver);
    vars.insert("name".to_string(), name);
    vars.insert("user".to_string(), user);
    vars.insert("password_expr".to_string(), password_expr);
    vars.insert("host".to_string(), host);
    vars.insert("port".to_string(), port.to_string());
    vars.insert("pool_enabled".to_string(), pool_enabled.to_string());
    vars.insert(
        "pool_size".to_string(),
        db_type.default_pool_size().to_string(),
    );
    vars.insert(
        "pool_max_overflow".to_string(),
        db_type.default_pool_max_overflow().to_string(),
    );
    vars.insert("async".to_string(), async_support.to_string());

    let base = project_root.join("core").join("database");
    let adapters_dir = base.join("adapters");
    let migrations_dir = base.join("migrations");

    std::fs::create_dir_all(&base).map_err(|e| e.to_string())?;
    std::fs::create_dir_all(&adapters_dir).map_err(|e| e.to_string())?;
    std::fs::create_dir_all(&migrations_dir).map_err(|e| e.to_string())?;

    // Common templates
    let config_tpl = embed_tpl!("/templates/database/common/config.dc.tpl");
    let engine_tpl = embed_tpl!("/templates/database/common/engine.dc.tpl");
    let connection_tpl = embed_tpl!("/templates/database/common/connection.dc.tpl");
    let models_tpl = embed_tpl!("/templates/database/common/models.dc.tpl");
    let lib_tpl = embed_tpl!("/templates/database/common/__lib__.dc.tpl");

    std::fs::write(base.join("config.dc"), render_template(config_tpl, &vars))
        .map_err(|e| e.to_string())?;
    std::fs::write(base.join("engine.dc"), render_template(engine_tpl, &vars))
        .map_err(|e| e.to_string())?;
    std::fs::write(base.join("connection.dc"), render_template(connection_tpl, &vars))
        .map_err(|e| e.to_string())?;
    std::fs::write(base.join("models.dc"), render_template(models_tpl, &vars))
        .map_err(|e| e.to_string())?;
    std::fs::write(base.join("__lib__.dc"), render_template(lib_tpl, &vars))
        .map_err(|e| e.to_string())?;

    // Adapters: only base.dc + the one adapter for selected type; .dpm-adapters and generated __lib__.dc
    let db_type_str = db_type.as_str();
    let dpm_adapters_path = adapters_dir.join(".dpm-adapters");
    std::fs::write(&dpm_adapters_path, format!("{}\n", db_type_str)).map_err(|e| e.to_string())?;

    let base_adapter_tpl = embed_tpl!("/templates/database/adapters/base.dc.tpl");
    std::fs::write(adapters_dir.join("base.dc"), base_adapter_tpl).map_err(|e| e.to_string())?;

    let adapter_tpl = get_adapter_template(db_type_str)?;
    let (adapter_module, _) = adapters_lib::adapter_module_and_class(db_type_str)
        .ok_or_else(|| format!("Unknown adapter for db_type: {}", db_type_str))?;
    std::fs::write(
        adapters_dir.join(format!("{}.dc", adapter_module)),
        adapter_tpl,
    )
    .map_err(|e| e.to_string())?;

    let adapters_lib_content = adapters_lib::render_adapters_lib(&[db_type_str]);
    std::fs::write(adapters_dir.join("__lib__.dc"), adapters_lib_content).map_err(|e| e.to_string())?;

    let migrations_readme_tpl = embed_tpl!("/templates/database/migrations/README.md.tpl");
    std::fs::write(
        migrations_dir.join("README.md"),
        render_template(migrations_readme_tpl, &HashMap::new()),
    )
    .map_err(|e| e.to_string())?;

    // dpm.toml [migrations]: if migrations enabled and dpm.toml exists and no [migrations], append
    if migrations {
        let manifest_path = project_root.join("dpm.toml");
        if manifest_path.exists() {
            let content = std::fs::read_to_string(&manifest_path).map_err(|e| e.to_string())?;
            if !content.contains("[migrations]") {
                let migration_section = "\n[migrations]\nfolder = \"core/database/migrations\"\ntable = \"dc_migrations\"\n";
                let new_content = content.trim_end().to_string() + migration_section;
                std::fs::write(&manifest_path, new_content).map_err(|e| e.to_string())?;
            }
        }
    }

    // Final output
    println!("Database module created");
    println!("Configuration saved to core/database/config.dc");
    println!("Connection helper created");
    println!();
    println!("Next steps:");
    println!("  - Edit core/database/config.dc");
    if migrations {
        println!("  - Configure folder/table in dpm.toml [migrations] if needed");
    }
    println!("  - Run `dpm db test` to verify connection");

    Ok(())
}
