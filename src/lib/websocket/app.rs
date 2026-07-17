//! Bootstrap ws_app.dc: load VM, route table, and app config.

use crate::vm::Vm;
use crate::websocket::config::{reset_bootstrap_config, take_bootstrap_config, WsAppConfig};
use std::cell::RefCell;
use std::path::{Path, PathBuf};

pub type WsRouteEntry = (String, usize);

thread_local! {
    static WS_APP_VM: RefCell<Option<Vm>> = RefCell::new(None);
    static WS_ROUTE_TABLE: RefCell<Option<Vec<WsRouteEntry>>> = RefCell::new(None);
    static WS_APP_CONFIG: RefCell<Option<WsAppConfig>> = RefCell::new(None);
}

/// Load and run ws_app.dc before the server accepts connections.
pub fn bootstrap_app(app_path: &str, base_dir: Option<&str>) -> Result<(), String> {
    reset_bootstrap_config();

    let source = std::fs::read_to_string(app_path)
        .map_err(|e| format!("Failed to read ws_app {}: {}", app_path, e))?;

    let app_path_buf = PathBuf::from(app_path);
    let base_path = if let Some(b) = base_dir {
        PathBuf::from(b)
    } else {
        app_path_buf
            .parent()
            .map(|p| p.to_path_buf())
            .ok_or_else(|| "Invalid ws_app path".to_string())?
    };

    let lib_path = base_path.join("__lib__.dc");
    let lib_path_opt = if lib_path.exists() {
        Some(lib_path.as_path())
    } else {
        None
    };

    crate::vm::file_import::set_base_path(Some(base_path.clone()));

    let (_, vm) = crate::run_with_vm_with_args_and_lib(
        &source,
        None,
        lib_path_opt,
        Some(base_path.as_path()),
        Some(app_path_buf.as_path()),
    )
    .map_err(|e| format!("Failed to run ws_app: {}", e))?;

    let mut route_table: Vec<WsRouteEntry> = Vec::new();
    for (i, f) in vm.get_functions().iter().enumerate() {
        if let Some(ref t) = f.ws_route_type {
            route_table.push((t.clone(), i));
        }
    }

    let config = take_bootstrap_config();

    WS_APP_VM.with(|cell| *cell.borrow_mut() = Some(vm));
    WS_ROUTE_TABLE.with(|cell| *cell.borrow_mut() = Some(route_table.clone()));
    WS_APP_CONFIG.with(|cell| *cell.borrow_mut() = Some(config));

    println!("📜 WebSocket app loaded: {}", app_path);
    if route_table.is_empty() {
        println!("   (no @ws_route handlers — built-in message types only)");
    } else {
        for (t, _) in &route_table {
            println!("   @ws_route(\"{}\")", t);
        }
    }
    println!();

    Ok(())
}

pub fn has_app() -> bool {
    WS_APP_VM.with(|cell| cell.borrow().is_some())
}

pub fn app_config() -> WsAppConfig {
    WS_APP_CONFIG
        .with(|cell| cell.borrow().clone())
        .unwrap_or_default()
}

/// Resolve handler index for a message type, if registered via @ws_route.
pub fn route_handler_index(msg_type: &str) -> Option<usize> {
    WS_ROUTE_TABLE.with(|cell| {
        let table = cell.borrow();
        let routes = table.as_ref()?;
        routes
            .iter()
            .find(|(t, _)| t == msg_type)
            .map(|(_, idx)| *idx)
    })
}

/// Call a @ws_route handler on the app VM.
pub fn call_route_handler(
    handler_idx: usize,
    request_value: crate::common::value::Value,
) -> Result<crate::common::value::Value, String> {
    WS_APP_VM.with(|vm_cell| {
        let mut vm_opt = vm_cell.borrow_mut();
        let vm = vm_opt.as_mut().ok_or("WebSocket app VM not loaded")?;
        let result = vm
            .call_function_by_index(handler_idx, &[request_value])
            .map_err(|e| e.to_string());
        vm.reset_stores_and_globals_for_stateless();
        result
    })
}

/// Resolve ws_app path relative to base_dir when provided.
pub fn resolve_app_path(app_file: &str, base_dir: Option<&str>) -> PathBuf {
    if let Some(b) = base_dir {
        Path::new(b).join(app_file)
    } else {
        PathBuf::from(app_file)
    }
}
