//! DCP session metadata and unified cleanup for WebSocket execution.

use std::cell::RefCell;
use std::collections::HashMap;

use crate::dcp::{
    clear_dcp_content_assets, clear_dcp_tables, clear_dcp_vfs, dcp_content_assets_active,
    dcp_vfs_active,
};

thread_local! {
    static DCP_METADATA: RefCell<Option<HashMap<String, String>>> = RefCell::new(None);
}

pub fn set_dcp_metadata(metadata: Option<HashMap<String, String>>) {
    DCP_METADATA.with(|slot| *slot.borrow_mut() = metadata);
}

pub fn get_dcp_metadata() -> Option<HashMap<String, String>> {
    DCP_METADATA.with(|slot| slot.borrow().clone())
}

pub fn clear_dcp_metadata() {
    set_dcp_metadata(None);
}

pub fn dcp_session_active() -> bool {
    dcp_vfs_active() || dcp_content_assets_active()
}

pub fn clear_dcp_session() {
    clear_dcp_vfs();
    clear_dcp_content_assets();
    clear_dcp_tables();
    clear_dcp_metadata();
}
