//! Thread-local storage for DCP Arrow table sections (WebSocket execution).

use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;

thread_local! {
    static DCP_TABLES: RefCell<Option<Arc<DcpTables>>> = RefCell::new(None);
}

pub fn set_dcp_tables(tables: Option<Arc<DcpTables>>) {
    DCP_TABLES.with(|slot| *slot.borrow_mut() = tables);
}

pub fn get_dcp_tables() -> Option<Arc<DcpTables>> {
    DCP_TABLES.with(|slot| slot.borrow().clone())
}

pub fn clear_dcp_tables() {
    set_dcp_tables(None);
}

#[derive(Debug, Clone)]
pub struct DcpTables {
    tables: HashMap<String, Vec<u8>>,
}

impl DcpTables {
    pub fn from_entries(entries: Vec<(String, Vec<u8>)>) -> Self {
        let mut tables = HashMap::with_capacity(entries.len());
        for (name, bytes) in entries {
            tables.insert(name, bytes);
        }
        Self { tables }
    }

    pub fn count(&self) -> usize {
        self.tables.len()
    }

    pub fn names(&self) -> Vec<String> {
        let mut names: Vec<String> = self.tables.keys().cloned().collect();
        names.sort();
        names
    }

    pub fn has(&self, name: &str) -> bool {
        self.tables.contains_key(name)
    }

    pub fn get(&self, name: &str) -> Option<&[u8]> {
        self.tables.get(name).map(|v| v.as_slice())
    }
}
