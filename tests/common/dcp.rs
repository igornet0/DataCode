//! Helpers shared by WebSocket / DCP integration tests.

use data_code::websocket::router::{dispatch_dcp, ClientContext};
use data_code::websocket::set_use_ve;
use data_code::websocket::smb::SmbManager;
use std::sync::{Arc, Mutex};

// ---------------------------------------------------------------------------
// DCP v1.1 package (code + path assets), no checksums — mirrors
// tests/dcp_fixtures/build_*.py.
// ---------------------------------------------------------------------------

fn put_str(buf: &mut Vec<u8>, s: &str) {
    buf.extend((s.len() as u32).to_le_bytes());
    buf.extend(s.as_bytes());
}

pub fn dcp_package(code: &str, assets: &[(&str, &[u8])]) -> Vec<u8> {
    const HEADER_SIZE: usize = 64;
    const INDEX_ENTRY_SIZE: usize = 2 + 2 + 4 + 8 + 8 + 1 + 1 + 2;
    const SECTION_HEADER_SIZE: usize = 2 + 2 + 4 + 8 + 1 + 1 + 2;

    let mut sections: Vec<(u16, &str, &[u8])> = vec![(1, "__code__", code.as_bytes())];
    sections.extend(assets.iter().map(|(name, data)| (4u16, *name, *data)));

    let mut pool = Vec::new();
    pool.extend((sections.len() as u32).to_le_bytes());
    for (_, name, _) in &sections {
        put_str(&mut pool, name);
    }

    let index_offset = HEADER_SIZE + pool.len();
    let mut cursor = index_offset + INDEX_ENTRY_SIZE * sections.len();
    let mut index = Vec::new();
    let mut blobs = Vec::new();
    for (string_id, (section_type, _, payload)) in sections.iter().enumerate() {
        index.extend(section_type.to_le_bytes());
        index.extend(0u16.to_le_bytes()); // flags
        index.extend((string_id as u32).to_le_bytes());
        index.extend((cursor as u64).to_le_bytes());
        index.extend((payload.len() as u64).to_le_bytes());
        index.push(0); // compression: none
        index.push(0); // checksum: none
        index.extend(0u16.to_le_bytes());

        blobs.extend(section_type.to_le_bytes());
        blobs.extend(0u16.to_le_bytes());
        blobs.extend((string_id as u32).to_le_bytes());
        blobs.extend((payload.len() as u64).to_le_bytes());
        blobs.push(0);
        blobs.push(0);
        blobs.extend(0u16.to_le_bytes());
        blobs.extend(*payload);
        cursor += SECTION_HEADER_SIZE + payload.len();
    }

    let mut out = Vec::with_capacity(cursor);
    out.extend(b"DCPK");
    out.extend(0x0101u16.to_le_bytes()); // version 1.1
    out.extend(0u16.to_le_bytes());
    out.extend((HEADER_SIZE as u16).to_le_bytes());
    out.extend(0u16.to_le_bytes());
    out.extend((sections.len() as u32).to_le_bytes());
    out.extend((index_offset as u64).to_le_bytes());
    out.extend((cursor as u64).to_le_bytes());
    out.push(1);
    out.resize(HEADER_SIZE, 0);
    out.extend(pool);
    out.extend(index);
    out.extend(blobs);
    out
}

pub struct Response {
    pub raw: String,
    pub success: bool,
    pub output: String,
    pub error: String,
}

/// Run `code` exactly like a WebSocket client with `--use-ve` would.
pub fn run_session(code: &str, assets: &[(&str, &[u8])]) -> Response {
    let ctx = ClientContext {
        smb_manager: Arc::new(Mutex::new(SmbManager::new())),
        build_model: false,
    };
    set_use_ve(true);
    let raw = dispatch_dcp(&dcp_package(code, assets), &ctx);
    set_use_ve(false);
    let json: serde_json::Value = serde_json::from_str(&raw).expect("JSON response");
    let text = |key: &str| json.get(key).and_then(|v| v.as_str()).unwrap_or("").to_string();
    Response {
        success: json.get("success").and_then(|v| v.as_bool()).unwrap_or(false),
        output: text("output"),
        error: text("error"),
        raw,
    }
}

