# `system` Module — Environment, Runtime, OS, and Sandbox

The **`system`** module is a **VM built-in** (implemented in the interpreter core; it is not loaded as an ABI `.dylib`/`.so`). It exposes a hierarchical API such as `system.env.get_os()`, `system.fs.read(path)`, and is the single entry point for OS details, paths, DataCode version, hardware, and networking. Sensitive operations (`fs`, `process.exec`, `set_env`) can be gated by `PermissionPolicy` on the host (CLI/embedder); see `data_code::PermissionPolicy` and `Vm::set_permission_policy` / `can_system_permission`.

**Russian version:** [модуль system (русский)](../../ru/2-язык/модули/system/README.md)

## Import

```datacode
import system
```

## Namespace overview

| Namespace | Purpose |
|-----------|---------|
| `system.env` | OS, arch, OS version, host, user, dirs, environment variables |
| `system.runtime` | DataCode/VM version, paths, loaded modules, package registry URL |
| `system.hardware` | CPU, memory, GPU (v1 stub) |
| `system.time` | Current time (ISO 8601 string), sleep, system uptime |
| `system.permissions` | Permission checks and requests (string keys) |
| `system.log` | Logging to stderr |
| `system.net` | Local IP, network interfaces |
| `system.process` | Shell (`exec`) and **CPU/GPU control** ([process-compute.md](process-compute.md)) |
| `system.fs` | File read/write |

## `system.env`

| Method | Description |
|--------|-------------|
| `get_os()` | `"linux"`, `"macos"`, or `"windows"` (or raw OS name if other) |
| `get_arch()` | `"x86_64"`, `"arm64"` (for `aarch64`), etc. |
| `get_version()` | OS version string (via sysinfo) |
| `get_hostname()` | Host name |
| `get_username()` | User name (`USER` / `USERNAME`) |
| `get_home_dir()` | Home directory path |
| `get_temp_dir()` | Temporary directory path |
| `get(key)` | Environment variable value or `null` |
| `set_env(key, value)` | Set environment variable (**requires `env.write` under Restricted policy**) |

Example:

```datacode
import system
print(system.env.get_os())
print(system.env.get("PATH"))
```

## `system.runtime`

| Method | Description |
|--------|-------------|
| `get_datacode_version()` | DataCode package version (`CARGO_PKG_VERSION`) |
| `get_vm_version()` | VM version (currently same as package version) |
| `get_module_path()` | Script/project base path or `null` |
| `get_venv_path()` | `VIRTUAL_ENV` or `DATACODE_VENV`, else `null` |
| `get_dpm_env_base()` | Directory from **`DPM_ENV_BASE`** (what `dpm --env-path=...` sets for that process), or `null` if unset |
| `get_dpm_env_root()` | Resolved **DPM environment root** for the project (`dpm.toml`): where `packages/` lives (`…/name-hash/`, or `<project>/.dpm/`), or `null` if no DPM project / error |
| `get_loaded_modules()` | Array of strings — loaded module names |
| `get_registry_url()` | Package registry index URL (`DATACODE_REGISTRY_URL` or default) |

## `system.hardware`

| Method | Description |
|--------|-------------|
| `cpu_count()` | Logical CPU count (at least 1) |
| `memory_total()` | Total RAM in bytes |
| `memory_free()` | Available memory in bytes |
| `gpu_count()` | GPU count (Metal when built with `--features metal`) |
| `gpu_info()` | Array of `{name, backend, detail}` |

## `system.time`

| Method | Description |
|--------|-------------|
| `now()` | Current UTC instant as **RFC 3339** string (the language does not expose a dedicated Date object type here) |
| `sleep(ms)` | Sleep for milliseconds |
| `uptime()` | System uptime since boot in seconds |

## `system.permissions`

| Method | Description |
|--------|-------------|
| `has_permission(key)` | `true`/`false` according to VM policy |
| `request_permission(key)` | Stub: always `true` for now (interactive prompts later) |

Keys denied under **Restricted** policy: `"fs.read"`, `"fs.write"`, `"process.exec"`, `"env.write"`.

## `system.log`

Messages go to **stderr** with a `[system]` prefix.

| Method | Description |
|--------|-------------|
| `log_info(...)` | INFO level |
| `log_warn(...)` | WARN level |
| `log_error(...)` | ERROR level |
| `debug(...)` | DEBUG; only when debug mode is enabled (`DATACODE_DEBUG`) |

## `system.net`

| Method | Description |
|--------|-------------|
| `get_ip()` | Local IPv4 if available, else `null` |
| `get_interfaces()` | Array of objects with `name`, `ip`, `is_loopback` |

## `system.process`

| Method | Description |
|--------|-------------|
| `exec(command)` | Single command string: Unix — `sh -c`, Windows — `cmd /C`. Returns combined stdout/stderr. (**`process.exec`** is checked under Restricted) |

## `system.fs`

| Method | Description |
|--------|-------------|
| `read(path)` | File contents as UTF-8 text or an error string |
| `write(path, content)` | Write string to file |

On policy denial, a string like `permission denied: fs.read` is returned.

### DPM and `--env-path`

The `dpm --env-path=/base/path …` invocation sets the **`DPM_ENV_BASE`** environment variable (see `src/dpm_main.rs`). That applies to the same process; if you run **`datacode script.dc`** separately, **`DPM_ENV_BASE` is not set automatically** unless you export it in the shell or store `env_base` in `dpm.toml`.

- Read the base from code: `system.runtime.get_dpm_env_base()` or `system.env.get("DPM_ENV_BASE")`.
- Read the **actual** package env directory for the project (after `dpm init`): `system.runtime.get_dpm_env_root()` — works from `dpm.toml` + project root even when `--env-path` was not passed to the current `datacode` process.

## Implementation and security

- Natives are implemented in Rust in-tree (`src/lib/system/natives.rs`), not via the SDK ABI.
- Policy: `data_code::PermissionPolicy`; VM methods `set_permission_policy` / `permission_policy` / `can_system_permission`.
- Built-in modules and imports: [Modules and Imports](../../0-syntax/11-modules-and-imports.md).
