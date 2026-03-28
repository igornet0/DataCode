//! Native implementations for the built-in `system` module.

use crate::common::debug;
use crate::common::value::Value;
use crate::dpm::registry::registry_index_url;
use crate::vm::permission_policy::PermissionPolicy;
use crate::vm::vm::VM_CALL_CONTEXT;
use std::io::Write;
use std::rc::Rc;
use std::cell::RefCell;
use std::time::Duration;

fn with_vm<T, F: FnOnce(&crate::vm::Vm) -> T>(f: F) -> Option<T> {
    VM_CALL_CONTEXT.with(|ctx| {
        let ptr = (*ctx.borrow())?;
        Some(unsafe { f(&*ptr) })
    })
}

fn check_perm(perm: &str) -> bool {
    with_vm(|vm| vm.can_system_permission(perm)).unwrap_or(true)
}

fn deny_value(perm: &str) -> Value {
    Value::String(format!("permission denied: {}", perm))
}

fn arg_string(args: &[Value], i: usize) -> Option<String> {
    match args.get(i)? {
        Value::String(s) => Some(s.clone()),
        Value::Number(n) => Some(n.to_string()),
        _ => None,
    }
}

// --- env (0..9) ---

pub fn native_system_get_os(_args: &[Value]) -> Value {
    let raw = std::env::consts::OS;
    let s = match raw {
        "linux" => "linux",
        "macos" => "macos",
        "windows" => "windows",
        _ => raw,
    };
    Value::String(s.to_string())
}

pub fn native_system_get_arch(_args: &[Value]) -> Value {
    let a = std::env::consts::ARCH;
    let s = if a == "aarch64" { "arm64" } else { a };
    Value::String(s.to_string())
}

pub fn native_system_get_os_version(_args: &[Value]) -> Value {
    let v = sysinfo::System::long_os_version()
        .or_else(|| sysinfo::System::os_version())
        .unwrap_or_else(|| "unknown".to_string());
    Value::String(v)
}

pub fn native_system_get_hostname(_args: &[Value]) -> Value {
    let h = hostname::get()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|_| String::new());
    Value::String(h)
}

pub fn native_system_get_username(_args: &[Value]) -> Value {
    let u = if cfg!(windows) {
        std::env::var("USERNAME")
    } else {
        std::env::var("USER")
    }
    .unwrap_or_else(|_| String::new());
    Value::String(u)
}

pub fn native_system_get_home_dir(_args: &[Value]) -> Value {
    let h = dirs::home_dir()
        .map(|p| p.to_string_lossy().into_owned())
        .unwrap_or_else(|| String::new());
    Value::String(h)
}

pub fn native_system_get_temp_dir(_args: &[Value]) -> Value {
    let t = std::env::temp_dir().to_string_lossy().into_owned();
    Value::String(t)
}

pub fn native_system_env_get(args: &[Value]) -> Value {
    let key = match arg_string(args, 0) {
        Some(k) => k,
        None => return Value::Null,
    };
    match std::env::var(&key) {
        Ok(v) => Value::String(v),
        Err(_) => Value::Null,
    }
}

pub fn native_system_env_set(args: &[Value]) -> Value {
    if !check_perm(PermissionPolicy::ENV_WRITE) {
        return deny_value(PermissionPolicy::ENV_WRITE);
    }
    let key = match arg_string(args, 0) {
        Some(k) => k,
        None => return Value::String("system.env.set: expected key".to_string()),
    };
    let val = match arg_string(args, 1) {
        Some(v) => v,
        None => return Value::String("system.env.set: expected value".to_string()),
    };
    std::env::set_var(key, val);
    Value::Null
}

// --- runtime (10..15) ---

pub fn native_system_get_datacode_version(_args: &[Value]) -> Value {
    Value::String(env!("CARGO_PKG_VERSION").to_string())
}

pub fn native_system_get_vm_version(_args: &[Value]) -> Value {
    Value::String(env!("CARGO_PKG_VERSION").to_string())
}

pub fn native_system_get_module_path(_args: &[Value]) -> Value {
    let path = with_vm(|vm| {
        vm.get_base_path()
            .or_else(|| vm.get_project_root())
            .map(|p| p.to_string_lossy().into_owned())
    })
    .flatten();
    match path {
        Some(s) => Value::String(s),
        None => Value::Null,
    }
}

pub fn native_system_get_venv_path(_args: &[Value]) -> Value {
    let v = std::env::var("VIRTUAL_ENV")
        .ok()
        .or_else(|| std::env::var("DATACODE_VENV").ok())
        .unwrap_or_default();
    if v.is_empty() {
        Value::Null
    } else {
        Value::String(v)
    }
}

/// `DPM_ENV_BASE` — базовый каталог, задаётся `dpm --env-path=...` (или вручную в окружении).
pub fn native_system_get_dpm_env_base(_args: &[Value]) -> Value {
    match std::env::var(crate::dpm::ENV_DPM_ENV_BASE) {
        Ok(s) if !s.trim().is_empty() => Value::String(s),
        _ => Value::Null,
    }
}

/// Реальный корень окружения DPM для проекта (`.../name-hash/`, `<project>/.dpm/` и т.д.), по `dpm.toml`.
pub fn native_system_get_dpm_env_root(_args: &[Value]) -> Value {
    let start = with_vm(|vm| vm.get_project_root().or_else(|| vm.get_base_path())).flatten();
    let start = start.or_else(|| std::env::current_dir().ok());
    let Some(start_path) = start else {
        return Value::Null;
    };
    match crate::dpm::resolve_env_and_packages(&start_path) {
        Ok(Some((env_root, _))) => Value::String(env_root.to_string_lossy().into_owned()),
        Ok(None) | Err(_) => Value::Null,
    }
}

pub fn native_system_get_loaded_modules(_args: &[Value]) -> Value {
    let names: Vec<Value> = with_vm(|vm| {
        let mut v: Vec<String> = vm.get_loaded_module_names();
        v.sort();
        v.into_iter().map(Value::String).collect()
    })
    .unwrap_or_default();
    Value::Array(Rc::new(RefCell::new(names)))
}

pub fn native_system_get_registry_url(_args: &[Value]) -> Value {
    Value::String(registry_index_url())
}

// --- hardware (16..20) ---

fn hardware_system() -> sysinfo::System {
    let mut sys = sysinfo::System::new();
    sys.refresh_all();
    sys
}

pub fn native_system_cpu_count(_args: &[Value]) -> Value {
    let sys = hardware_system();
    let n = sys.cpus().len().max(1);
    Value::Number(n as f64)
}

pub fn native_system_memory_total(_args: &[Value]) -> Value {
    let sys = hardware_system();
    Value::Number(sys.total_memory() as f64)
}

pub fn native_system_memory_free(_args: &[Value]) -> Value {
    let sys = hardware_system();
    Value::Number(sys.available_memory() as f64)
}

pub fn native_system_gpu_count(_args: &[Value]) -> Value {
    Value::Number(0.0)
}

/// Array of objects `{ name, backend, detail }` (v1 stub when no GPU enumeration).
pub fn native_system_gpu_info(_args: &[Value]) -> Value {
    Value::Array(Rc::new(RefCell::new(Vec::new())))
}

// --- time (21..23) ---

pub fn native_system_time_now(_args: &[Value]) -> Value {
    let now = chrono::Utc::now();
    Value::String(now.to_rfc3339())
}

pub fn native_system_sleep_ms(args: &[Value]) -> Value {
    let ms = match args.first() {
        Some(Value::Number(n)) if *n >= 0.0 => *n as u64,
        _ => return Value::Null,
    };
    std::thread::sleep(Duration::from_millis(ms));
    Value::Null
}

pub fn native_system_uptime(_args: &[Value]) -> Value {
    let s = sysinfo::System::uptime();
    Value::Number(s as f64)
}

// --- permissions (24..25) ---

pub fn native_system_has_permission(args: &[Value]) -> Value {
    let key = match arg_string(args, 0) {
        Some(k) => k,
        None => return Value::Bool(false),
    };
    let ok = with_vm(|vm| vm.can_system_permission(&key)).unwrap_or(true);
    Value::Bool(ok)
}

pub fn native_system_request_permission(args: &[Value]) -> Value {
    let _key = arg_string(args, 0);
    // Phase A: no interactive prompt; policy unchanged.
    Value::Bool(true)
}

// --- log (26..29) ---

pub fn native_system_log_info(args: &[Value]) -> Value {
    let msg = args
        .iter()
        .map(|v| v.to_string())
        .collect::<Vec<_>>()
        .join(" ");
    eprintln!("[system] INFO {}", msg);
    let _ = std::io::stderr().flush();
    Value::Null
}

pub fn native_system_log_warn(args: &[Value]) -> Value {
    let msg = args
        .iter()
        .map(|v| v.to_string())
        .collect::<Vec<_>>()
        .join(" ");
    eprintln!("[system] WARN {}", msg);
    let _ = std::io::stderr().flush();
    Value::Null
}

pub fn native_system_log_error(args: &[Value]) -> Value {
    let msg = args
        .iter()
        .map(|v| v.to_string())
        .collect::<Vec<_>>()
        .join(" ");
    eprintln!("[system] ERROR {}", msg);
    let _ = std::io::stderr().flush();
    Value::Null
}

pub fn native_system_log_debug(args: &[Value]) -> Value {
    if !debug::is_debug_enabled() {
        return Value::Null;
    }
    let msg = args
        .iter()
        .map(|v| v.to_string())
        .collect::<Vec<_>>()
        .join(" ");
    eprintln!("[system] DEBUG {}", msg);
    let _ = std::io::stderr().flush();
    Value::Null
}

// --- net (30..31) ---

pub fn native_system_net_get_ip(_args: &[Value]) -> Value {
    match local_ip_address::local_ip() {
        Ok(ip) => Value::String(ip.to_string()),
        Err(_) => Value::Null,
    }
}

pub fn native_system_net_get_interfaces(_args: &[Value]) -> Value {
    let mut rows: Vec<Value> = Vec::new();
    if let Ok(ifaces) = if_addrs::get_if_addrs() {
        for iface in ifaces {
            let mut m = std::collections::HashMap::new();
            m.insert("name".to_string(), Value::String(iface.name));
            m.insert("ip".to_string(), Value::String(iface.addr.ip().to_string()));
            m.insert(
                "is_loopback".to_string(),
                Value::Bool(iface.addr.ip().is_loopback()),
            );
            rows.push(Value::Object(Rc::new(RefCell::new(m))));
        }
    }
    Value::Array(Rc::new(RefCell::new(rows)))
}

// --- process (32) ---

pub fn native_system_process_exec(args: &[Value]) -> Value {
    if !check_perm(PermissionPolicy::PROCESS_EXEC) {
        return deny_value(PermissionPolicy::PROCESS_EXEC);
    }
    let cmdline = match arg_string(args, 0) {
        Some(s) => s,
        None => return Value::String("system.process.exec: expected command string".to_string()),
    };
    let out = if cfg!(windows) {
        std::process::Command::new("cmd")
            .args(["/C", &cmdline])
            .output()
    } else {
        std::process::Command::new("sh")
            .args(["-c", &cmdline])
            .output()
    };
    match out {
        Ok(o) => {
            let mut s = String::new();
            if !o.stdout.is_empty() {
                s.push_str(&String::from_utf8_lossy(&o.stdout));
            }
            if !o.stderr.is_empty() {
                if !s.is_empty() {
                    s.push('\n');
                }
                s.push_str(&String::from_utf8_lossy(&o.stderr));
            }
            Value::String(s)
        }
        Err(e) => Value::String(format!("system.process.exec: {}", e)),
    }
}

// --- fs (33..34) ---

pub fn native_system_fs_read(args: &[Value]) -> Value {
    if !check_perm(PermissionPolicy::FS_READ) {
        return deny_value(PermissionPolicy::FS_READ);
    }
    let path = match arg_string(args, 0) {
        Some(p) => std::path::PathBuf::from(p),
        None => return Value::String("system.fs.read: expected path".to_string()),
    };
    match std::fs::read_to_string(&path) {
        Ok(s) => Value::String(s),
        Err(e) => Value::String(format!("system.fs.read: {}", e)),
    }
}

pub fn native_system_fs_write(args: &[Value]) -> Value {
    if !check_perm(PermissionPolicy::FS_WRITE) {
        return deny_value(PermissionPolicy::FS_WRITE);
    }
    let path = match arg_string(args, 0) {
        Some(p) => std::path::PathBuf::from(p),
        None => return Value::String("system.fs.write: expected path".to_string()),
    };
    let content = match arg_string(args, 1) {
        Some(c) => c,
        None => return Value::String("system.fs.write: expected content".to_string()),
    };
    match std::fs::write(&path, content.as_bytes()) {
        Ok(()) => Value::Null,
        Err(e) => Value::String(format!("system.fs.write: {}", e)),
    }
}
