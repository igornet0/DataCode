//! Самообновление: `datacode --update-version` (последний релиз GitHub → замена бинарников).

use crate::dcmodule::extract_zip_bytes;
use std::fs::{self, File};
use std::path::{Path, PathBuf};
use std::process::Command;

const GITHUB_LATEST: &str = "https://api.github.com/repos/igornet0/DataCode/releases/latest";
const ASSET_PREFIX: &str = "data-code-";
const BINS: &[&str] = &["datacode", "dpm", "datacode-server"];

/// Запуск `datacode --update-version`.
pub fn run() -> Result<(), String> {
    eprintln!("Проверка: {GITHUB_LATEST}");
    let current = semver::Version::parse(env!("CARGO_PKG_VERSION"))
        .map_err(|e| format!("CARGO_PKG_VERSION: {e}"))?;
    let body = ureq::get(GITHUB_LATEST)
        .set("User-Agent", &format!("DataCode/{}", env!("CARGO_PKG_VERSION")))
        .set("Accept", "application/vnd.github+json")
        .call()
        .map_err(|e| format!("GitHub: {e}"))?
        .into_string()
        .map_err(|e| e.to_string())?;
    let v: serde_json::Value =
        serde_json::from_str(&body).map_err(|e| format!("Ожидаем JSON: {e}"))?;
    let tag = v
        .get("tag_name")
        .and_then(|t| t.as_str())
        .ok_or("В ответе нет tag_name")?;
    let rel_ver = tag
        .strip_prefix('v')
        .map(|s| s.to_string())
        .unwrap_or_else(|| tag.to_string());
    let tag_ver = semver::Version::parse(&rel_ver).map_err(|e| format!("Версия релиза: {e}"))?;

    if tag_ver <= current {
        println!("Уже установлена актуальная или более новая версия: {current} (релиз: {rel_ver})");
        return Ok(());
    }
    println!("Доступно: {current} -> {rel_ver}");
    let (os_key, arch_key) = runtime_artifact_keys()?;
    let ext = if cfg!(target_os = "windows") { "zip" } else { "tar.gz" };
    let want_name = format!("{ASSET_PREFIX}{rel_ver}-{os_key}-{arch_key}.{ext}");
    let url = find_asset_url(&v, &want_name).ok_or_else(|| {
        format!(
            "В релизе нет ассета «{want_name}». Загрузите пакет на GitHub (см. `make -f Makefile.build package-all`)."
        )
    })?;

    let install_dir = install_dir_from_current_exe()?;
    eprintln!("Каталог установки: {}", install_dir.display());
    let temp = std::env::temp_dir().join(format!("dc-up-{}", now_ms()));
    fs::create_dir_all(&temp).map_err(|e| e.to_string())?;
    let arch_path = temp.join("download");
    download_to_file(&url, &arch_path)?;

    let extracted = temp.join("out");
    fs::create_dir_all(&extracted).map_err(|e| e.to_string())?;
    if ext == "zip" {
        let b = fs::read(&arch_path).map_err(|e| e.to_string())?;
        extract_zip_bytes(&b, &extracted)?;
    } else {
        extract_tgz(&arch_path, &extracted)?;
    }

    let bin_in = if extracted.join("bin").is_dir() {
        extracted.join("bin")
    } else {
        first_level_with_bin(&extracted)?
    };

    let this_exe = std::env::current_exe().map_err(|e| e.to_string())?;

    for &base in BINS {
        let from = bin_in.join(exe_name(base));
        if !from.is_file() {
            return Err(format!("В архиве нет: {}", from.display()));
        }
        let to = install_dir.join(exe_name(base));
        if cfg!(target_os = "windows") {
            if same_path(&this_exe, &to)? {
                let staging = install_dir.join(format!("{base}.new.exe"));
                fs::copy(&from, &staging).map_err(|e| e.to_string())?;
                win_schedule_replace(&staging, &to)?;
                eprintln!("{} обновлён после перезапуска процесса датакод.", to.display());
            } else {
                try_copy(&from, &to)?;
            }
        } else {
            try_copy(&from, &to)?;
        }
    }

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        for &base in BINS {
            let p = install_dir.join(base);
            if p.is_file() {
                let mut m = fs::metadata(&p).map_err(|e| e.to_string())?.permissions();
                m.set_mode(0o755);
                fs::set_permissions(&p, m).map_err(|e| e.to_string())?;
            }
        }
    }

    let _ = fs::remove_dir_all(&temp);
    println!("Готово. DataCode {rel_ver} (перезапустите оболочку, если путь в кэше).");
    Ok(())
}

fn now_ms() -> u128 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0)
}

fn runtime_artifact_keys() -> Result<(&'static str, &'static str), String> {
    use std::env::consts::{ARCH, OS};
    match (OS, ARCH) {
        ("linux", "x86_64") => Ok(("linux", "amd64")),
        ("linux", "aarch64") => Ok(("linux", "arm64")),
        ("macos", "x86_64") => Ok(("macos", "amd64")),
        ("macos", "aarch64") => Ok(("macos", "arm64")),
        ("windows", "x86_64") => Ok(("windows", "amd64")),
        ("windows", "aarch64") => Ok(("windows", "arm64")),
        _ => Err(format!(
            "Самообновление: неподдерживаемая плата: {OS} {ARCH}. Скачайте архив вручную с GitHub."
        )),
    }
}

fn find_asset_url(v: &serde_json::Value, name: &str) -> Option<String> {
    v.get("assets")
        .and_then(|a| a.as_array())?
        .iter()
        .find_map(|a| {
            if a.get("name").and_then(|n| n.as_str()) == Some(name) {
                a.get("browser_download_url")
                    .and_then(|u| u.as_str())
                    .map(String::from)
            } else {
                None
            }
        })
}

fn install_dir_from_current_exe() -> Result<PathBuf, String> {
    let p = std::env::current_exe().map_err(|e| e.to_string())?;
    p.parent()
        .map(Path::to_path_buf)
        .ok_or_else(|| "невозможно определить каталог бинарника".to_string())
}

fn download_to_file(url: &str, to: &Path) -> Result<(), String> {
    eprintln!("Скачивание…");
    let r = ureq::get(url)
        .set("User-Agent", &format!("DataCode/{}", env!("CARGO_PKG_VERSION")))
        .set("Accept", "application/octet-stream")
        .call()
        .map_err(|e| format!("скачивание: {e}"))?;
    if r.status() != 200 {
        return Err(format!("HTTP {} при скачивании", r.status()));
    }
    let mut w = File::create(to).map_err(|e| e.to_string())?;
    let mut r = r.into_reader();
    std::io::copy(&mut r, &mut w).map_err(|e| e.to_string())?;
    Ok(())
}

fn extract_tgz(path: &Path, out: &Path) -> Result<(), String> {
    let s = path.to_str().ok_or("путь: UTF-8")?;
    let o = out.to_str().ok_or("путь: UTF-8")?;
    let st = Command::new("tar")
        .arg("-xzf")
        .arg(s)
        .arg("-C")
        .arg(o)
        .status()
        .map_err(|e| e.to_string())?;
    if !st.success() {
        return Err(format!("tar: статус {st} (нужен `tar` в PATH)"));
    }
    Ok(())
}

fn first_level_with_bin(extracted: &Path) -> Result<PathBuf, String> {
    let r = fs::read_dir(extracted).map_err(|e| e.to_string())?;
    for e in r.filter_map(|e| e.ok()) {
        if e.file_type().map_err(|e| e.to_string())?.is_dir() {
            let b = e.path().join("bin");
            if b.is_dir() {
                return Ok(b);
            }
        }
    }
    Err("В распаковке нет папки bin/".to_string())
}

fn exe_name(base: &str) -> String {
    if cfg!(target_os = "windows") {
        format!("{base}.exe")
    } else {
        base.to_string()
    }
}

fn try_copy(from: &Path, to: &Path) -> Result<(), String> {
    fs::copy(from, to).map_err(|e| {
        format!(
            "копирование {} -> {}: {e} (нужны права на запись; для /usr/local/bin — sudo)",
            from.display(),
            to.display()
        )
    })?;
    eprintln!("Обновлено: {}", to.display());
    Ok(())
}

fn same_path(a: &Path, b: &Path) -> Result<bool, String> {
    let a = fs::canonicalize(a).map_err(|e| e.to_string())?;
    let b = fs::canonicalize(b).map_err(|e| e.to_string())?;
    Ok(a == b)
}

/// Windows: `datacode.exe` в памяти держит файл; замена после задержки (когда этот процесс выйдет).
#[cfg(windows)]
fn win_schedule_replace(staging: &Path, final_dst: &Path) -> Result<(), String> {
    let st = staging.to_str().ok_or("UTF-8: staging")?;
    let de = final_dst.to_str().ok_or("UTF-8: final")?;
    let ps = format!(
        "Start-Sleep -Seconds 2; Copy-Item -LiteralPath '{}' -Destination '{}' -Force; Remove-Item -LiteralPath '{}' -ErrorAction SilentlyContinue",
        st.replace('\'', "''"),
        de.replace('\'', "''"),
        st.replace('\'', "''")
    );
    std::process::Command::new("powershell")
        .args(["-NoProfile", "-WindowStyle", "Hidden", "-Command", &ps])
        .spawn()
        .map_err(|e| e.to_string())?;
    Ok(())
}

#[cfg(not(windows))]
#[allow(dead_code)]
fn win_schedule_replace(_staging: &Path, _final_dst: &Path) -> Result<(), String> {
    Ok(())
}
