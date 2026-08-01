//! Network / URL guards for the `web` module.

use crate::vm::permission_policy::PermissionPolicy;
use crate::web::error::WebError;
use crate::web::with_vm;

pub const MAX_REDIRECTS: u32 = 10;
pub const DEFAULT_HTTP_TIMEOUT_SECS: f64 = 30.0;
pub const DEFAULT_WAIT_TIMEOUT_SECS: f64 = 30.0;

/// Validate that URL uses http/https (file: and others denied by default).
pub fn validate_url(url: &str) -> Result<(), WebError> {
    let trimmed = url.trim();
    if trimmed.is_empty() {
        return Err(WebError::value("URL must not be empty"));
    }
    let lower = trimmed.to_ascii_lowercase();
    if lower.starts_with("http://") || lower.starts_with("https://") {
        // Future: domain allowlist hook
        let _ = domain_allowed(trimmed);
        return Ok(());
    }
    if lower.starts_with("file:") {
        return Err(WebError::value(
            "file: URLs are not allowed; use http:// or https://",
        ));
    }
    Err(WebError::value(format!(
        "invalid URL scheme (expected http/https): {}",
        trimmed
    )))
}

/// Stub for future domain allowlist. Always true in v1.
pub fn domain_allowed(_url: &str) -> bool {
    true
}

pub fn check_net_http() -> Result<(), WebError> {
    let ok = with_vm(|vm| vm.can_system_permission(PermissionPolicy::NET_HTTP)).unwrap_or(true);
    if ok {
        Ok(())
    } else {
        Err(WebError::runtime(format!(
            "permission denied: {}",
            PermissionPolicy::NET_HTTP
        )))
    }
}

pub fn check_net_browser() -> Result<(), WebError> {
    let ok = with_vm(|vm| vm.can_system_permission(PermissionPolicy::NET_BROWSER)).unwrap_or(true);
    if ok {
        Ok(())
    } else {
        Err(WebError::runtime(format!(
            "permission denied: {}",
            PermissionPolicy::NET_BROWSER
        )))
    }
}
