//! WebSocket app configuration (ws_app.dc bootstrap).

use crate::vm::PermissionPolicy;
use std::cell::RefCell;
use std::collections::HashSet;

/// Configuration set by `websocket.configure()` during ws_app.dc bootstrap.
#[derive(Clone, Debug, Default)]
pub struct WsAppConfig {
    pub execute_permission_policy: PermissionPolicy,
    pub disabled_builtins: HashSet<String>,
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
