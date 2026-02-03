//! Interactive wizard to create dpm.toml (like poetry init).

use std::io::{self, IsTerminal, Write};
use std::path::Path;
use std::process::Command;

/// ANSI: blue (questions)
const BLUE: &str = "\x1b[34m";
/// ANSI: green (defaults in brackets)
const GREEN: &str = "\x1b[32m";
/// ANSI: reset
const RESET: &str = "\x1b[0m";

fn colored_prompt(question: &str, bracket_content: &str) {
    let display = if bracket_content.is_empty() { "" } else { bracket_content };
    if io::stdout().is_terminal() {
        print!("{}{}{} [{}{}{}]: ", BLUE, question, RESET, GREEN, display, RESET);
    } else {
        print!("{} [{}]: ", question, display);
    }
}

fn print_question(question: &str) {
    if io::stdout().is_terminal() {
        print!("{}{}{}: ", BLUE, question, RESET);
    } else {
        print!("{}: ", question);
    }
}

/// Get default author: git config (local then global), then USER env, else empty.
pub fn default_author() -> String {
    let run_git = |args: &[&str]| -> Option<String> {
        let out = Command::new("git").args(args).output().ok()?;
        if !out.status.success() {
            return None;
        }
        let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
        if s.is_empty() {
            None
        } else {
            Some(s)
        }
    };
    let name = run_git(&["config", "user.name"])
        .or_else(|| run_git(&["config", "--global", "user.name"]));
    let email = run_git(&["config", "user.email"])
        .or_else(|| run_git(&["config", "--global", "user.email"]));
    if let Some(n) = name {
        let n = n.trim().to_string();
        if let Some(e) = email {
            let e = e.trim().to_string();
            if e.is_empty() {
                return n;
            }
            return format!("{} <{}>", n, e);
        }
        return n;
    }
    #[cfg(unix)]
    if let Ok(u) = std::env::var("USER") {
        return u.trim().to_string();
    }
    #[cfg(windows)]
    if let Ok(u) = std::env::var("USERNAME") {
        return u.trim().to_string();
    }
    String::new()
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

/// Sanitize directory name for package name: only alphanumeric, -, _.
fn sanitize_package_name(s: &str) -> String {
    s.chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect::<String>()
}

/// Escape a string for TOML double-quoted value (backslash and quote).
fn toml_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            _ => out.push(c),
        }
    }
    out.push('"');
    out
}

/// Run interactive wizard and write dpm.toml if confirmed. Returns Ok(()) on success.
pub fn run_init_wizard(project_root: &Path) -> Result<(), String> {
    println!("This command will guide you through creating your dpm.toml config.");
    println!();

    let default_name = project_root
        .file_name()
        .and_then(|n| n.to_str())
        .map(sanitize_package_name)
        .unwrap_or_else(|| "project".to_string());
    let default_name = if default_name.is_empty() {
        "project".to_string()
    } else {
        default_name
    };

    colored_prompt("Package name", &default_name);
    let name = prompt(&default_name)?;

    colored_prompt("Version", "0.1.0");
    let version = prompt("0.1.0")?;

    colored_prompt("Description", "");
    let description = prompt("")?;

    let author_default = default_author();
    colored_prompt("Author", if author_default.is_empty() { "" } else { &author_default });
    let author = prompt(&author_default)?;

    colored_prompt("License", "");
    let license = prompt("")?;

    colored_prompt("datacode version", ">=2.0.0");
    let datacode = prompt(">=2.0.0")?;

    colored_prompt("Entry point (e.g. main.dc or src/main.dc)", "");
    let entry = prompt("")?;

    colored_prompt("Would you like to define your main dependencies interactively? (yes/no)", "yes");
    let deps_interactive = prompt_yes_no(true)?;

    let mut dependencies: Vec<(String, String)> = Vec::new();
    if deps_interactive {
        println!("You can specify a package in the following forms:");
        println!("  - A single name (my_package)");
        println!("  - A name and source (my_package git+https://github.com/user/repo.git)");
        println!();
        loop {
            print_question("Package to add or search for (leave blank to skip)");
            io::stdout().flush().map_err(|e| e.to_string())?;
            let mut line = String::new();
            io::stdin().read_line(&mut line).map_err(|e| e.to_string())?;
            let spec = line.trim().to_string();
            if spec.is_empty() {
                break;
            }
            let parts: Vec<&str> = spec.split_whitespace().collect();
            match parts.as_slice() {
                [pkg, source] => {
                    dependencies.push(((*pkg).to_string(), (*source).to_string()));
                }
                [pkg] => {
                    println!("  (skipped: specify source as '{} git+<url>')", pkg);
                }
                _ => {
                    println!("  (skipped: use 'name git+<url>')");
                }
            }
        }
    }

    let toml_content = build_toml(&name, &version, &description, &author, &license, &datacode, &entry, &dependencies);

    println!();
    println!("Generated file");
    println!();
    println!("{}", toml_content);
    println!();

    colored_prompt("Do you confirm generation? (yes/no)", "yes");
    let confirm = prompt_yes_no(true)?;
    if !confirm {
        println!("Aborted.");
        return Ok(());
    }

    let manifest_path = project_root.join("dpm.toml");
    std::fs::write(&manifest_path, toml_content).map_err(|e| format!("Write {}: {}", manifest_path.display(), e))?;
    println!("Created {}", manifest_path.display());

    Ok(())
}

fn build_toml(
    name: &str,
    version: &str,
    description: &str,
    author: &str,
    license: &str,
    datacode: &str,
    entry: &str,
    dependencies: &[(String, String)],
) -> String {
    let mut out = String::new();

    out.push_str("[project]\n");
    out.push_str("name = ");
    out.push_str(&toml_escape(name));
    out.push('\n');
    out.push_str("version = ");
    out.push_str(&toml_escape(version));
    out.push('\n');
    if !description.is_empty() {
        out.push_str("description = ");
        out.push_str(&toml_escape(description));
        out.push('\n');
    }
    if !author.is_empty() {
        out.push_str("authors = [");
        out.push_str(&toml_escape(author));
        out.push_str("]\n");
    }
    if !license.is_empty() {
        out.push_str("license = ");
        out.push_str(&toml_escape(license));
        out.push('\n');
    }
    if !datacode.is_empty() {
        out.push_str("datacode = ");
        out.push_str(&toml_escape(datacode));
        out.push('\n');
    }
    if !entry.is_empty() {
        out.push_str("entry = ");
        out.push_str(&toml_escape(entry));
        out.push('\n');
    }

    out.push_str("\n[environment]\n");
    out.push_str("python = false\n");
    out.push_str("wasm = false\n");
    out.push_str("target = \"native\"\n");
    out.push_str("profile = \"release\"\n");

    out.push_str("\n[dependencies]\n");
    for (dep_name, source) in dependencies {
        out.push_str(&format!("{} = {}", dep_name, toml_escape(source)));
        out.push('\n');
    }

    out.push_str("\n[lock]\n");
    out.push_str("file = \"dpm.lock\"\n");
    out.push_str("checksum = true\n");

    out
}
