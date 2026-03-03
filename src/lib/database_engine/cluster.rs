// Database cluster: named collection of database engines

use crate::database_engine::engine::DatabaseEngine;
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

/// Cluster of named database connections (engines).
#[derive(Debug)]
pub struct DatabaseCluster {
    pub connections: HashMap<String, Rc<RefCell<DatabaseEngine>>>,
}

impl DatabaseCluster {
    /// Create an empty cluster.
    pub fn new() -> Self {
        Self {
            connections: HashMap::new(),
        }
    }

    /// Add or replace a named connection.
    pub fn add(&mut self, name: String, engine: Rc<RefCell<DatabaseEngine>>) {
        self.connections.insert(name, engine);
    }

    /// Get connection by name.
    pub fn get(&self, name: &str) -> Option<Rc<RefCell<DatabaseEngine>>> {
        self.connections.get(name).cloned()
    }

    /// Return list of connection names.
    pub fn names(&self) -> Vec<String> {
        self.connections.keys().cloned().collect()
    }
}

impl Default for DatabaseCluster {
    fn default() -> Self {
        Self::new()
    }
}
