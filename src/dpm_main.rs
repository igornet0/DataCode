//! DPM binary: init, add, config

use data_code::dcmodule::{default_output_path, pack_directory};
use data_code::dpm::{
    clear_manifest_env_base, current_target, datacode_version_satisfies, env_root,
    fetch_registry_index, find_project_root, install_artifact, install_package,
    path_in_packages_directory, prefer_source_build, RegistryPackage, load_lock, load_manifest, lock_file_name, packages_dir,
    resolve_registry_package, run_add_database, run_init_database, run_init_wizard,
    run_setup_for_package, run_setup_if_present, set_manifest_env_base, set_virtualenvs_in_project,
    virtualenvs_in_project, write_lock, LockPackage, ENV_DPM_ENV_BASE,
};
use std::path::{Path, PathBuf};

/// Clone `source` into `dest`, then either install the registry's prebuilt `.dcmodule` for this
/// target (verified by sha256) or fall back to building via `setup.dcmodule`.
/// Returns `(revision, artifact_sha256, artifact_version)`.
fn install_dependency(
    name: &str,
    source: &str,
    dest: &Path,
    registry_pkg: Option<&RegistryPackage>,
) -> Result<(String, Option<String>, Option<String>), String> {
    let revision = install_package(name, source, dest)?;
    let artifact = registry_pkg
        .filter(|p| p.source == source && !prefer_source_build())
        .and_then(|p| p.artifact_for_current_target());
    if let Some(art) = artifact {
        // `<packages>/<name>/<name>.dcmodule`, where the VM looks for bundles.
        let dest_file = path_in_packages_directory(dest.parent().unwrap_or(dest), name);
        println!("Downloading prebuilt {} for {}...", name, art.target);
        match install_artifact(&art.url, &art.sha256, &dest_file) {
            Ok(()) => {
                return Ok((
                    revision,
                    Some(art.sha256.to_ascii_lowercase()),
                    art.version.clone(),
                ))
            }
            Err(e) => eprintln!("Warning: {}; building {} from source", e, name),
        }
    } else if registry_pkg.is_some_and(|p| !p.artifacts.is_empty()) && !prefer_source_build() {
        eprintln!(
            "No prebuilt {} for {}; building from source",
            name,
            current_target()
        );
    }
    run_setup_if_present(dest)?;
    Ok((revision, None, None))
}

fn resolve_env_base_flag(path: &str) -> Result<PathBuf, String> {
    let p = Path::new(path);
    let abs = if p.is_absolute() {
        p.to_path_buf()
    } else {
        std::env::current_dir().map_err(|e| e.to_string())?.join(p)
    };
    Ok(abs)
}

/// Strips leading `dpm --env-path <dir>` / `--env-path=<dir>` and sets [`ENV_DPM_ENV_BASE`].
fn apply_leading_env_path(args: &mut Vec<String>) -> Result<(), String> {
    let i = 1usize;
    while i < args.len() {
        let arg = args[i].as_str();
        if arg == "--env-path" {
            let path = args
                .get(i + 1)
                .ok_or_else(|| "--env-path requires a directory".to_string())?;
            let abs = resolve_env_base_flag(path)?;
            std::env::set_var(ENV_DPM_ENV_BASE, abs);
            args.remove(i);
            args.remove(i);
            continue;
        }
        if let Some(rest) = arg.strip_prefix("--env-path=") {
            if rest.is_empty() {
                return Err("--env-path= requires a directory".to_string());
            }
            let abs = resolve_env_base_flag(rest)?;
            std::env::set_var(ENV_DPM_ENV_BASE, abs);
            args.remove(i);
            continue;
        }
        break;
    }
    Ok(())
}

fn main() {
    let mut args: Vec<String> = std::env::args().collect();
    if let Err(e) = apply_leading_env_path(&mut args) {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }
    if args.len() < 2 {
        print_help();
        std::process::exit(1);
    }
    let cmd = args[1].as_str();
    let result = match cmd {
        "init" => cmd_init(&args[2..]),
        "add" => cmd_add(&args[2..]),
        "update" => cmd_update(&args[2..]),
        "setup" => cmd_setup(&args[2..]),
        "config" => cmd_config(&args[2..]),
        "pack" => cmd_pack(&args[2..]),
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
    println!("  dpm [--env-path <dir>] <command> ...");
    println!("  dpm init              Create dpm.toml interactively (if missing) or install deps from existing dpm.toml");
    println!("  dpm init database     Create core/database module interactively");
    println!(
        "  dpm add database       Add another database connection (new folder under core/database)"
    );
    println!("  dpm add <name> [<source>]  Add dependency; source from registry if omitted, or git+https://...");
    println!("  dpm update [<name>]    Update all dependencies (or one) that have a newer version");
    println!("  dpm setup <package_name>  Run setup.dcmodule in an installed package (see docs)");
    println!(
        "  dpm config virtualenvs.in-project <true|false>  Use .dpm in project (default: cache)"
    );
    println!("  dpm config env-base <dir>|clear  Store env base in dpm.toml (or clear)");
    println!("  dpm pack <dir> [-o|--output <path.dcmodule>]  Zip dir (manifest.json + libs) → .dcmodule");
    println!();
    println!("Virtual env location:");
    println!("  Default: ~/.cache/datacode/dpm/envs/<project>-<hash>/ (Linux)");
    println!("           ~/Library/Caches/datacode/dpm/envs/<project>-<hash>/ (macOS)");
    println!("           %APPDATA%\\datacode\\Cache\\dpm\\envs\\... (Windows)");
    println!("  In-project: <project_root>/.dpm/  (set config or DPM_IN_PROJECT=1)");
    println!("  Custom base: dpm --env-path <dir> init  saves [dpm] env_base in dpm.toml;");
    println!("               later dpm/datacode use it without repeating the flag.");
    println!(
        "               One-off: {}=<dir> overrides manifest for that process.",
        ENV_DPM_ENV_BASE
    );
    println!("  After `dpm add` / `dpm init`, if a package contains setup.dcmodule, DPM runs it.");
    println!("  Disable: DPM_SETUP_AUTO=0");
    println!();
    println!("Registry (package index JSON):");
    println!("  GET uses the raw GitHub URL (not the blob HTML page), e.g.:");
    println!(
        "  https://raw.githubusercontent.com/igornet0/Datacode-registry-index/main/config.json"
    );
    println!("  Override: DATACODE_REGISTRY_URL=<url>");
    println!(
        "  Offline: last successful fetch is cached under <cache>/datacode/registry/config.json"
    );
}

fn cmd_init(args: &[String]) -> Result<(), String> {
    let cwd = std::env::current_dir().map_err(|e| e.to_string())?;
    if args.first().map(|s| s.as_str()) == Some("database") {
        let project_root = if let Some(dir) = args.get(1) {
            let p = Path::new(dir);
            if p.is_absolute() {
                p.to_path_buf()
            } else {
                cwd.join(p)
            }
        } else {
            cwd.clone()
        };
        let flags = args.get(2..).unwrap_or(&[]);
        return run_init_database(&project_root, flags);
    }
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
        run_init_wizard(&project_root)?;
    }
    // Same path as `dpm init` when manifest already existed: create env dir, install deps, write lock.
    let mut manifest = load_manifest(&project_root)?;
    if !virtualenvs_in_project() {
        if let Ok(base) = std::env::var(ENV_DPM_ENV_BASE) {
            let t = base.trim();
            if !t.is_empty() {
                set_manifest_env_base(&project_root, &PathBuf::from(t))?;
                manifest = load_manifest(&project_root)?;
            }
        }
    }
    let lock_file = lock_file_name(&manifest);
    let env_root_path = env_root(&project_root, &manifest).ok_or("Could not determine env root")?;
    std::fs::create_dir_all(packages_dir(&env_root_path)).map_err(|e| e.to_string())?;
    let mut lock = load_lock(&project_root, lock_file)?;
    lock.package.clear();
    // Registry is optional here: offline installs still work by building from source.
    let registry = if manifest.dependencies.is_empty() {
        None
    } else {
        fetch_registry_index().ok()
    };
    for (name, source) in &manifest.dependencies {
        let dest = packages_dir(&env_root_path).join(name);
        println!("Installing {} from {}...", name, source);
        let registry_pkg = registry
            .as_ref()
            .and_then(|r| r.packages.iter().find(|p| &p.name == name));
        let (revision, artifact_sha256, version) =
            install_dependency(name, source, &dest, registry_pkg)?;
        lock.package.push(LockPackage {
            name: name.clone(),
            source: source.clone(),
            revision: Some(revision),
            artifact_sha256,
            version,
        });
    }
    write_lock(&project_root, lock_file, &lock)?;
    println!(
        "Lock file written: {}",
        project_root.join(lock_file).display()
    );
    Ok(())
}

fn cmd_add(args: &[String]) -> Result<(), String> {
    let cwd = std::env::current_dir().map_err(|e| e.to_string())?;
    if args.first().map(|s| s.as_str()) == Some("database") {
        let project_root = if let Some(dir) = args.get(1) {
            let p = Path::new(dir);
            if p.is_absolute() {
                p.to_path_buf()
            } else {
                cwd.join(p)
            }
        } else {
            find_project_root(&cwd).ok_or("No dpm.toml found (run from project with dpm.toml)")?
        };
        let flags = args.get(2..).unwrap_or(&[]);
        return run_add_database(&project_root, flags);
    }
    let mut registry_pkg: Option<RegistryPackage> = None;
    let (name, source_owned): (&str, String) = match args {
        [n, s, ..] => (n.as_str(), s.clone()),
        [n] => {
            let pkg = resolve_registry_package(n)?;
            if let Some(ref min) = pkg.min_datacode {
                let current = env!("CARGO_PKG_VERSION");
                if !datacode_version_satisfies(min, current) {
                    return Err(format!(
                        "Package '{}' requires datacode {} but this binary is {}",
                        n, min, current
                    ));
                }
            }
            let source = pkg.source.clone();
            registry_pkg = Some(pkg);
            (n.as_str(), source)
        }
        _ => return Err("Usage: dpm add <package_name> [<source>]".to_string()),
    };
    let source = source_owned.as_str();
    let project_root =
        find_project_root(&cwd).ok_or("No dpm.toml found (run from project with dpm.toml)")?;
    let mut manifest = load_manifest(&project_root)?;
    manifest
        .dependencies
        .insert(name.to_string(), source.to_string());
    // Write back dpm.toml (preserve other sections via raw edit or re-serialize; for simplicity we append to [dependencies])
    let manifest_path = project_root.join("dpm.toml");
    let content = std::fs::read_to_string(&manifest_path).map_err(|e| e.to_string())?;
    let new_dep_line = format!("{} = \"{}\"", name, source);
    let has_dep = content
        .lines()
        .any(|l| l.trim().starts_with(&format!("{} = ", name)) || l.trim() == name);
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
            format!(
                "{}\n\n[dependencies]\n{}\n",
                content.trim_end(),
                new_dep_line
            )
        };
        std::fs::write(&manifest_path, new_content).map_err(|e| e.to_string())?;
    }
    let lock_file = lock_file_name(&manifest);
    let env_root_path = env_root(&project_root, &manifest).ok_or("Could not determine env root")?;
    std::fs::create_dir_all(packages_dir(&env_root_path)).map_err(|e| e.to_string())?;
    let dest = packages_dir(&env_root_path).join(name);
    println!("Installing {} from {}...", name, source);
    let (revision, artifact_sha256, version) =
        install_dependency(name, source, &dest, registry_pkg.as_ref())?;
    let mut lock = load_lock(&project_root, lock_file)?;
    if let Some(p) = lock.package.iter_mut().find(|p| p.name == name) {
        p.source = source.to_string();
        p.revision = Some(revision);
        p.artifact_sha256 = artifact_sha256;
        p.version = version;
    } else {
        lock.package.push(LockPackage {
            name: name.to_string(),
            source: source.to_string(),
            revision: Some(revision),
            artifact_sha256,
            version,
        });
    }
    write_lock(&project_root, lock_file, &lock)?;
    if !virtualenvs_in_project() {
        if let Ok(base) = std::env::var(ENV_DPM_ENV_BASE) {
            let t = base.trim();
            if !t.is_empty() {
                set_manifest_env_base(&project_root, &PathBuf::from(t))?;
            }
        }
    }
    println!("Added {} and updated lock file.", name);
    Ok(())
}

/// Commit hash `HEAD` points to in the remote repo of `source` (`git ls-remote`).
fn remote_head_revision(source: &str) -> Result<String, String> {
    let url = source.strip_prefix("git+").unwrap_or(source).trim();
    let out = std::process::Command::new("git")
        .args(["ls-remote", url, "HEAD"])
        .output()
        .map_err(|e| format!("Run git ls-remote: {}", e))?;
    if !out.status.success() {
        return Err(format!(
            "git ls-remote {} failed: {}",
            url,
            String::from_utf8_lossy(&out.stderr).trim()
        ));
    }
    String::from_utf8_lossy(&out.stdout)
        .split_whitespace()
        .next()
        .map(str::to_string)
        .ok_or_else(|| format!("git ls-remote {}: no HEAD", url))
}

/// Why `pkg` needs reinstalling, or `None` if it is current.
/// Prebuilt installs compare the registry artifact sha256; source installs compare the git HEAD.
fn update_reason(
    pkg: &LockPackage,
    registry_pkg: Option<&RegistryPackage>,
    installed: bool,
) -> Result<Option<String>, String> {
    if !installed {
        return Ok(Some("not installed".to_string()));
    }
    let artifact = registry_pkg
        .filter(|p| p.source == pkg.source && !prefer_source_build())
        .and_then(|p| p.artifact_for_current_target());
    if let Some(art) = artifact {
        let same = pkg
            .artifact_sha256
            .as_deref()
            .is_some_and(|s| s.eq_ignore_ascii_case(art.sha256.trim()));
        if same {
            return Ok(None);
        }
        let from = pkg
            .version
            .as_deref()
            .or(pkg.revision.as_deref().map(|r| &r[..r.len().min(12)]))
            .unwrap_or("?");
        let to = art.version.as_deref().unwrap_or("new build");
        return Ok(Some(format!("{} -> {}", from, to)));
    }
    let remote = remote_head_revision(&pkg.source)?;
    match pkg.revision.as_deref() {
        Some(r) if r == remote && pkg.artifact_sha256.is_none() => Ok(None),
        r => Ok(Some(format!(
            "{} -> {}",
            r.map(|r| &r[..r.len().min(12)]).unwrap_or("?"),
            &remote[..remote.len().min(12)]
        ))),
    }
}

fn cmd_update(args: &[String]) -> Result<(), String> {
    let only = match args {
        [] => None,
        [n] => Some(n.as_str()),
        _ => return Err("Usage: dpm update [<package_name>]".to_string()),
    };
    let cwd = std::env::current_dir().map_err(|e| e.to_string())?;
    let project_root =
        find_project_root(&cwd).ok_or("No dpm.toml found (run from project with dpm.toml)")?;
    let manifest = load_manifest(&project_root)?;
    let lock_file = lock_file_name(&manifest);
    let mut lock = load_lock(&project_root, lock_file)?;
    // Dependencies added to dpm.toml by hand but never installed get a lock entry too.
    for (name, source) in &manifest.dependencies {
        if !lock.package.iter().any(|p| &p.name == name) {
            lock.package.push(LockPackage {
                name: name.clone(),
                source: source.clone(),
                revision: None,
                artifact_sha256: None,
                version: None,
            });
        }
    }
    if let Some(n) = only {
        if !lock.package.iter().any(|p| p.name == n) {
            return Err(format!(
                "Package '{}' is not a dependency of this project (use dpm add {})",
                n, n
            ));
        }
    }
    let env_root_path = env_root(&project_root, &manifest).ok_or("Could not determine env root")?;
    std::fs::create_dir_all(packages_dir(&env_root_path)).map_err(|e| e.to_string())?;
    let registry = match fetch_registry_index() {
        Ok(r) => Some(r),
        Err(e) => {
            eprintln!("Warning: {}; checking git sources only", e);
            None
        }
    };
    let mut updated = 0usize;
    for pkg in lock
        .package
        .iter_mut()
        .filter(|p| only.map_or(true, |n| p.name == n))
    {
        let registry_pkg = registry
            .as_ref()
            .and_then(|r| r.packages.iter().find(|p| p.name == pkg.name));
        if let Some(min) = registry_pkg.and_then(|p| p.min_datacode.as_deref()) {
            let current = env!("CARGO_PKG_VERSION");
            if !datacode_version_satisfies(min, current) {
                eprintln!(
                    "Skipping {}: latest version requires datacode {} but this binary is {}",
                    pkg.name, min, current
                );
                continue;
            }
        }
        let dest = packages_dir(&env_root_path).join(&pkg.name);
        let reason = match update_reason(pkg, registry_pkg, dest.exists()) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("Warning: cannot check {}: {}", pkg.name, e);
                continue;
            }
        };
        let Some(reason) = reason else {
            println!("{} is up to date.", pkg.name);
            continue;
        };
        println!("Updating {} ({})...", pkg.name, reason);
        let (revision, artifact_sha256, version) =
            install_dependency(&pkg.name, &pkg.source, &dest, registry_pkg)?;
        pkg.revision = Some(revision);
        pkg.artifact_sha256 = artifact_sha256;
        pkg.version = version;
        updated += 1;
    }
    write_lock(&project_root, lock_file, &lock)?;
    if updated > 0 {
        println!("Updated {} package(s) and the lock file.", updated);
    }
    Ok(())
}

fn cmd_setup(args: &[String]) -> Result<(), String> {
    let cwd = std::env::current_dir().map_err(|e| e.to_string())?;
    let name = args
        .first()
        .ok_or_else(|| "Usage: dpm setup <package_name>".to_string())?;
    let project_root =
        find_project_root(&cwd).ok_or("No dpm.toml found (run from project with dpm.toml)")?;
    let manifest = load_manifest(&project_root)?;
    let env_root_path = env_root(&project_root, &manifest).ok_or("Could not determine env root")?;
    let dest = packages_dir(&env_root_path).join(name);
    if !dest.is_dir() {
        return Err(format!(
            "Package directory not found: {} (run dpm add first)",
            dest.display()
        ));
    }
    run_setup_for_package(&dest)
}

fn cmd_pack(args: &[String]) -> Result<(), String> {
    let mut out: Option<PathBuf> = None;
    let mut dir: Option<PathBuf> = None;
    let mut i = 0usize;
    while i < args.len() {
        match args[i].as_str() {
            "-o" | "--output" => {
                let p = args
                    .get(i + 1)
                    .ok_or_else(|| "--output requires a path".to_string())?;
                out = Some(PathBuf::from(p));
                i += 2;
            }
            s if !s.starts_with('-') && dir.is_none() => {
                dir = Some(PathBuf::from(s));
                i += 1;
            }
            s => return Err(format!("unexpected argument: {}", s)),
        }
    }
    let dir =
        dir.ok_or_else(|| "Usage: dpm pack <directory> [-o|--output <path.dcmodule>]".to_string())?;
    let cwd = std::env::current_dir().map_err(|e| e.to_string())?;
    let dir_abs = if dir.is_absolute() {
        dir
    } else {
        cwd.join(&dir)
    };
    let out_path = if let Some(o) = out {
        if o.is_absolute() {
            o
        } else {
            cwd.join(o)
        }
    } else {
        default_output_path(&dir_abs)?
    };
    pack_directory(&dir_abs, &out_path)?;
    println!("Wrote {}", out_path.display());
    Ok(())
}

fn cmd_config(args: &[String]) -> Result<(), String> {
    let cwd = std::env::current_dir().map_err(|e| e.to_string())?;
    if args.len() >= 2 && args[0] == "virtualenvs.in-project" {
        let value = &args[1];
        let on = value == "true" || value == "1" || value.eq_ignore_ascii_case("yes");
        set_virtualenvs_in_project(on)?;
        println!("virtualenvs.in-project = {}", on);
        return Ok(());
    }
    if args.len() >= 2 && args[0] == "env-base" {
        let project_root =
            find_project_root(&cwd).ok_or("No dpm.toml found (run from project with dpm.toml)")?;
        if args[1].eq_ignore_ascii_case("clear") {
            clear_manifest_env_base(&project_root)?;
            println!(
                "Cleared [dpm] env_base in {}",
                project_root.join("dpm.toml").display()
            );
            return Ok(());
        }
        let abs = resolve_env_base_flag(&args[1])?;
        set_manifest_env_base(&project_root, &abs)?;
        println!(
            "Wrote env_base to {} (resolved: {})",
            project_root.join("dpm.toml").display(),
            abs.display()
        );
        return Ok(());
    }
    Err(
        "Usage: dpm config virtualenvs.in-project true|false | dpm config env-base <dir>|clear"
            .to_string(),
    )
}
