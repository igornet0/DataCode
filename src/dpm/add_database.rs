//! Add another database connection: creates core/database/<connection_name>/ with config, engine, connection.
//! Requires core/database/ to exist (from dpm init database). Updates adapters if the new type is not yet in .dpm-adapters.

use std::collections::HashMap;
use std::io::{self, IsTerminal, Write};
use std::path::Path;

use dialoguer::Select;

use super::adapters_lib;
use super::init_database::get_adapter_template;

macro_rules! embed_tpl {
    ($path:literal) => {
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), $path))
    };
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

fn render_template(tpl: &str, vars: &HashMap<String, String>) -> String {
    let mut out = tpl.to_string();
    for (key, value) in vars {
        let placeholder = format!("{{{{ {} }}}}", key);
        out = out.replace(&placeholder, value);
    }
    out
}

/// Sanitize connection name for folder: only alphanumeric, underscore.
fn sanitize_connection_name(s: &str) -> String {
    s.chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect::<String>()
}

/// Default connection folder name from db_type (e.g. redis -> redis_cache, postgresql -> analytics).
fn default_connection_name(db_type: &str) -> &'static str {
    match db_type {
        "redis" => "redis_cache",
        "memcached" => "memcache",
        _ => "analytics",
    }
}

/// Run the add database wizard and create core/database/<connection_name>/.
pub fn run_add_database(project_root: &Path, _flags: &[String]) -> Result<(), String> {
    let base = project_root.join("core").join("database");
    if !base.exists() {
        return Err("core/database/ not found. Run `dpm init database` first.".to_string());
    }

    let adapters_dir = base.join("adapters");
    let dpm_adapters_path = adapters_dir.join(".dpm-adapters");
    if !dpm_adapters_path.exists() {
        return Err("core/database/adapters/.dpm-adapters not found. Run `dpm init database` first.".to_string());
    }

    println!("Adding database connection...");
    println!();

    // Step 1: database type
    let db_type_str = if io::stdout().is_terminal() {
        let items: Vec<&str> = adapters_lib::DB_TYPE_LABELS.iter().map(|(label, _)| *label).collect();
        let idx = Select::with_theme(&dialoguer::theme::ColorfulTheme::default())
            .with_prompt("? Select database type")
            .items(&items)
            .default(0)
            .interact()
            .map_err(|e| e.to_string())?;
        adapters_lib::DB_TYPE_LABELS[idx].1.to_string()
    } else {
        println!("? Select database type:");
        for (i, (label, _)) in adapters_lib::DB_TYPE_LABELS.iter().enumerate() {
            println!("  {}) {}", i + 1, label);
        }
        colored_prompt("Enter number", "1");
        let num_str = prompt("1")?;
        let idx: usize = num_str.trim().parse().map_err(|_| "Invalid number")?;
        adapters_lib::DB_TYPE_LABELS
            .get(idx.wrapping_sub(1))
            .map(|(_, t)| t.to_string())
            .ok_or("Invalid selection")?
    };

    // Step 2: connection name (folder under core/database)
    let default_conn = default_connection_name(&db_type_str);
    colored_prompt("Connection name (folder under core/database)", default_conn);
    let connection_name = sanitize_connection_name(&prompt(default_conn)?);
    if connection_name.is_empty() {
        return Err("Connection name cannot be empty.".to_string());
    }
    let connection_dir = base.join(&connection_name);
    if connection_dir.exists() {
        return Err(format!(
            "core/database/{} already exists. Choose another name.",
            connection_name
        ));
    }

    // Step 3: connection params
    let (name_prompt_text, name_default) = adapters_lib::name_prompt(&db_type_str);
    colored_prompt(name_prompt_text, name_default);
    let name = prompt(name_default)?;

    let (user, password_expr) = if !adapters_lib::asks_user_password(&db_type_str) {
        (String::new(), "\"\"".to_string())
    } else {
        let default_user = adapters_lib::default_username(&db_type_str);
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

    let default_port = adapters_lib::default_port(&db_type_str);
    let (host, port) = if db_type_str == "sqlite" {
        ("localhost".to_string(), 0u16)
    } else {
        colored_prompt("Host", "localhost");
        let host = prompt("localhost")?;
        colored_prompt("Port", &default_port.to_string());
        let port_str = prompt(&default_port.to_string())?;
        let port: u16 = port_str.trim().parse().unwrap_or(default_port);
        (host, port)
    };

    // Step 4: features
    colored_prompt("Enable migrations? (Y/n)", "yes");
    let _migrations = prompt_yes_no(true)?;
    colored_prompt("Enable connection pooling? (Y/n)", "yes");
    let pool_enabled = prompt_yes_no(true)?;
    colored_prompt("Enable async support? (y/N)", "no");
    let async_support = prompt_yes_no(false)?;

    println!();

    let driver = adapters_lib::default_driver(&db_type_str).to_string();
    let pool_size = adapters_lib::default_pool_size(&db_type_str);
    let pool_max_overflow = adapters_lib::default_pool_max_overflow(&db_type_str);

    let mut vars = HashMap::<String, String>::new();
    vars.insert("db_type".to_string(), db_type_str.clone());
    vars.insert("driver".to_string(), driver);
    vars.insert("name".to_string(), name);
    vars.insert("user".to_string(), user.clone());
    vars.insert("password_expr".to_string(), password_expr);
    vars.insert("host".to_string(), host);
    vars.insert("port".to_string(), port.to_string());
    vars.insert("pool_enabled".to_string(), pool_enabled.to_string());
    vars.insert("pool_size".to_string(), pool_size.to_string());
    vars.insert("pool_max_overflow".to_string(), pool_max_overflow.to_string());
    vars.insert("async".to_string(), async_support.to_string());
    vars.insert("connection_name".to_string(), connection_name.clone());
    vars.insert("connection_module".to_string(), connection_name.clone());

    std::fs::create_dir_all(&connection_dir).map_err(|e| e.to_string())?;

    let config_tpl = embed_tpl!("/templates/database/connection_folder/config.dc.tpl");
    let engine_tpl = embed_tpl!("/templates/database/connection_folder/engine.dc.tpl");
    let connection_tpl = embed_tpl!("/templates/database/connection_folder/connection.dc.tpl");
    let lib_tpl = embed_tpl!("/templates/database/connection_folder/__lib__.dc.tpl");

    std::fs::write(connection_dir.join("config.dc"), render_template(config_tpl, &vars))
        .map_err(|e| e.to_string())?;
    std::fs::write(connection_dir.join("engine.dc"), render_template(engine_tpl, &vars))
        .map_err(|e| e.to_string())?;
    std::fs::write(connection_dir.join("connection.dc"), render_template(connection_tpl, &vars))
        .map_err(|e| e.to_string())?;
    std::fs::write(connection_dir.join("__lib__.dc"), render_template(lib_tpl, &vars))
        .map_err(|e| e.to_string())?;

    // Update .dpm-adapters and adapters/ if this type is new
    let current = std::fs::read_to_string(&dpm_adapters_path).unwrap_or_default();
    let used_types: Vec<String> = current
        .lines()
        .filter(|s| !s.is_empty())
        .map(String::from)
        .collect();
    if !used_types.iter().any(|s| s == &db_type_str) {
        let mut new_list = used_types;
        new_list.push(db_type_str.clone());
        std::fs::write(&dpm_adapters_path, new_list.join("\n") + "\n")
            .map_err(|e| e.to_string())?;

        let adapter_tpl = get_adapter_template(&db_type_str)?;
        let (adapter_module, _) = adapters_lib::adapter_module_and_class(&db_type_str)
            .ok_or_else(|| format!("Unknown adapter for db_type: {}", db_type_str))?;
        std::fs::write(
            adapters_dir.join(format!("{}.dc", adapter_module)),
            adapter_tpl,
        )
        .map_err(|e| e.to_string())?;

        let used_types_ref: Vec<&str> = new_list.iter().map(String::as_str).collect();
        let adapters_lib_content = adapters_lib::render_adapters_lib(&used_types_ref);
        std::fs::write(adapters_dir.join("__lib__.dc"), adapters_lib_content)
            .map_err(|e| e.to_string())?;
    }

    println!("Connection '{}' created.", connection_name);
    println!("  config: core/database/{}/config.dc", connection_name);
    println!();
    println!("Usage: from core.database.{} import get_connection", connection_name);

    Ok(())
}
