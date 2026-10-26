//! WebSocket app configuration (ws_app.dc bootstrap).

use crate::vm::PermissionPolicy;
use std::cell::RefCell;
use std::collections::HashSet;

/// Configuration set by `websocket.configure()` during ws_app.dc bootstrap.
#[derive(Clone, Debug)]
pub struct WsAppConfig {
    pub execute_permission_policy: PermissionPolicy,
    pub disabled_builtins: HashSet<String>,
    /// Client code may write files into its session folder (`./…`).
    pub allow_write: bool,
    /// Size limit of one session folder.
    pub write_quota_bytes: u64,
}

/// Default size limit of a session folder when writes are enabled.
pub const DEFAULT_WRITE_QUOTA_MB: u64 = 50;

impl Default for WsAppConfig {
    /// Client code is sandboxed unless the developer opts in with
    /// `configure({"execute_policy": "allow_all"})` in ws_app.dc.
    fn default() -> Self {
        Self {
            execute_permission_policy: PermissionPolicy::Restricted,
            disabled_builtins: HashSet::new(),
            allow_write: false,
            write_quota_bytes: DEFAULT_WRITE_QUOTA_MB * 1024 * 1024,
        }
    }
}

thread_local! {
    /// Mutable config while ws_app.dc runs at startup (natives write here).
    static WS_BOOTSTRAP_CONFIG: RefCell<WsAppConfig> = RefCell::new(WsAppConfig::default());
}

pub fn reset_bootstrap_config() {
    WS_BOOTSTRAP_CONFIG.with(|c| *c.borrow_mut() = WsAppConfig::default());
}

pub fn take_bootstrap_config() -> WsAppConfig {
    WS_BOOTSTRAP_CONFIG.with(|c| std::mem::take(&mut *c.borrow_mut()))
}

pub fn with_bootstrap_config<F, R>(f: F) -> R
where
    F: FnOnce(&mut WsAppConfig) -> R,
{
    WS_BOOTSTRAP_CONFIG.with(|c| f(&mut c.borrow_mut()))
}

pub fn is_builtin_disabled(config: &WsAppConfig, msg_type: &str) -> bool {
    config.disabled_builtins.contains(msg_type)
}
