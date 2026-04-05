//! Sandbox policy for the built-in `system` module (Phase A: default allow-all; Restricted blocks unsafe ops).

/// Policy applied by [`crate::vm::Vm::can_system_permission`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PermissionPolicy {
    /// All `system` capabilities allowed (default, matches pre-sandbox behavior).
    #[default]
    AllowAll,
    /// Denies `fs.read`, `fs.write`, `process.exec`, and `env.write` (`set_env`).
    Restricted,
}

impl PermissionPolicy {
    /// Known permission keys used by `system.permissions` and internal checks.
    pub const FS_READ: &'static str = "fs.read";
    pub const FS_WRITE: &'static str = "fs.write";
    pub const PROCESS_EXEC: &'static str = "process.exec";
    pub const ENV_WRITE: &'static str = "env.write";
}

const RESTRICTED_DENIED: &[&str] = &[
    PermissionPolicy::FS_READ,
    PermissionPolicy::FS_WRITE,
    PermissionPolicy::PROCESS_EXEC,
    PermissionPolicy::ENV_WRITE,
];

/// True if `perm` is allowed under `policy` (used by [`crate::vm::Vm::can_system_permission`]).
pub fn is_permission_allowed(policy: PermissionPolicy, perm: &str) -> bool {
    match policy {
        PermissionPolicy::AllowAll => true,
        PermissionPolicy::Restricted => !RESTRICTED_DENIED.iter().any(|&p| p == perm),
    }
}
