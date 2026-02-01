//! DPM binary: init, add, config

use data_code::dpm::{
    env_root, find_project_root, install_package, load_lock, load_manifest, lock_file_name,
    packages_dir, set_virtualenvs_in_project, write_lock, LockPackage,
};
use std::path::Path;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        print_help();
        std::process::exit(1);
    }
    let cmd = args[1].as_str();
    let result = match cmd {
        "init" => cmd_init(&args[2..]),
        "add" => cmd_add(&args[2..]),
        "config" => cmd_config(&args[2..]),
        "-h" | "--help" => {
            print_help();
            Ok(())
        }
        _ => {
            eprintln!("Unknown command: {}", cmd);
            print_help();
            std::process::exit(1);
        }
    };
    if let Err(e) = result {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }
}

fn print_help() {
    println!("DPM - DataCode Package Manager");
    println!();
    println!("Usage:");
    println!("  dpm init              Initialize virtual env from dpm.toml (in current directory)");
    println!("  dpm add <name> <source>  Add dependency (source: git+https://...)");
    println!("  dpm config virtualenvs.in-project <true|false>  Use .dpm in project (default: cache)");
    println!();
    println!("Virtual env location:");
    println!("  Default: ~/.cache/datacode/dpm/envs/<project>-<hash>/ (Linux/macOS)");
    println!("           %APPDATA%\\datacode\\Cache\\dpm\\envs\\... (Windows)");
    println!("  In-project: <project_root>/.dpm/  (set config or DPM_IN_PROJECT=1)");
}

fn cmd_init(args: &[String]) -> Result<(), String> {
    let cwd = std::env::current_dir().map_err(|e| e.to_string())?;
    let project_root = if let Some(dir) = args.first() {
        let p = Path::new(dir);
        if p.is_absolute() {
            p.to_path_buf()
        } else {
            cwd.join(p)
        }
    } else {
        cwd.clone()
    };
    let manifest_path = project_root.join("dpm.toml");
    if !manifest_path.exists() {
        return Err(format!(
            "dpm.toml not found in {} (run from project root or create dpm.toml)",
            project_root.display()
        ));
    }
    let manifest = load_manifest(&project_root)?;
    let lock_file = lock_file_name(&manifest);
    let env_root_path = env_root(&project_root, &manifest).ok_or("Could not determine env root")?;
    std::fs::create_dir_all(packages_dir(&env_root_path)).map_err(|e| e.to_string())?;
    let mut lock = load_lock(&project_root, lock_file)?;
    lock.package.clear();
    for (name, source) in &manifest.dependencies {
        let dest = packages_dir(&env_root_path).join(name);
        println!("Installing {} from {}...", name, source);
        let revision = install_package(name, source, &dest)?;
        lock.package.push(LockPackage {
            name: name.clone(),
            source: source.clone(),
            revision: Some(revision),
        });
    }
    write_lock(&project_root, lock_file, &lock)?;
    println!("Lock file written: {}", project_root.join(lock_file).display());
    Ok(())
}

fn cmd_add(args: &[String]) -> Result<(), String> {
    let (name, source) = match args {
        [n, s, ..] => (n.as_str(), s.as_str()),
        [n] => return Err(format!("Source required: dpm add {} git+<url>", n)),
        _ => return Err("Usage: dpm add <package_name> <source> (e.g. git+https://...)".to_string()),
    };
    let cwd = std::env::current_dir().map_err(|e| e.to_string())?;
    let project_root = find_project_root(&cwd).ok_or("No dpm.toml found (run from project with dpm.toml)")?;
    let mut manifest = load_manifest(&project_root)?;
    manifest.dependencies.insert(name.to_string(), source.to_string());
    // Write back dpm.toml (preserve other sections via raw edit or re-serialize; for simplicity we append to [dependencies])
    let manifest_path = project_root.join("dpm.toml");
    let content = std::fs::read_to_string(&manifest_path).map_err(|e| e.to_string())?;
    let new_dep_line = format!("{} = \"{}\"", name, source);
    let has_dep = content.lines().any(|l| l.trim().starts_with(&format!("{} = ", name)) || l.trim() == name);
    if !has_dep {
        let new_content = if content.contains("[dependencies]") {
            let mut out = String::new();
            for line in content.lines() {
                out.push_str(line);
                out.push('\n');
                if line.trim() == "[dependencies]" {
                    out.push_str(&format!("{}\n", new_dep_line));
                }
            }
            out
        } else {
            format!("{}\n\n[dependencies]\n{}\n", content.trim_end(), new_dep_line)
        };
        std::fs::write(&manifest_path, new_content).map_err(|e| e.to_string())?;
    }
    let lock_file = lock_file_name(&manifest);
    let env_root_path = env_root(&project_root, &manifest).ok_or("Could not determine env root")?;
    std::fs::create_dir_all(packages_dir(&env_root_path)).map_err(|e| e.to_string())?;
    let dest = packages_dir(&env_root_path).join(name);
    println!("Installing {} from {}...", name, source);
    let revision = install_package(name, source, &dest)?;
    let mut lock = load_lock(&project_root, lock_file)?;
    if let Some(p) = lock.package.iter_mut().find(|p| p.name == name) {
        p.source = source.to_string();
        p.revision = Some(revision);
    } else {
        lock.package.push(LockPackage {
            name: name.to_string(),
            source: source.to_string(),
            revision: Some(revision),
        });
    }
    write_lock(&project_root, lock_file, &lock)?;
    println!("Added {} and updated lock file.", name);
    Ok(())
}

fn cmd_config(args: &[String]) -> Result<(), String> {
    if args.len() >= 2 && args[0] == "virtualenvs.in-project" {
        let value = &args[1];
        let on = value == "true" || value == "1" || value.eq_ignore_ascii_case("yes");
        set_virtualenvs_in_project(on)?;
        println!("virtualenvs.in-project = {}", on);
        Ok(())
    } else {
        Err("Usage: dpm config virtualenvs.in-project true|false".to_string())
    }
}
