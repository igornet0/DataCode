//! Module namespace: own globals and global_names. Used for module isolation.

use crate::common::value::Value;
use crate::vm::global_slot::{GlobalSlot, default_global_slot};
use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap};
use std::rc::Rc;

/// Builtin indices 0..BUILTIN_END are in VM.builtins; module globals start at BUILTIN_END.
pub const BUILTIN_END: usize = 75;

/// Per-module namespace: globals (slot indices >= BUILTIN_END) and their names.
/// When loaded from a .dc file, namespace holds the export map (name -> Value) for "from X import a" lookup.
#[derive(Clone)]
pub struct ModuleObject {
    pub name: String,
    /// Module globals only (indices 0..len correspond to bytecode indices BUILTIN_END, BUILTIN_END+1, ...).
    pub globals: Vec<GlobalSlot>,
    /// Map bytecode global index (>= BUILTIN_END) -> name.
    pub global_names: BTreeMap<usize, String>,
    /// Export namespace (name -> Value). Set when module is loaded from .dc; used for from X import a.
    pub namespace: Option<Rc<RefCell<HashMap<String, Value>>>>,
}

impl ModuleObject {
    pub fn new(name: String) -> Self {
        Self {
            name,
            globals: Vec::new(),
            global_names: BTreeMap::new(),
            namespace: None,
        }
    }

    /// Create a module from an export map (e.g. after loading a .dc module). Registers in vm.modules for isolation.
    pub fn from_exports(name: String, exports: HashMap<String, Value>) -> Self {
        Self {
            name: name.clone(),
            globals: Vec::new(),
            global_names: BTreeMap::new(),
            namespace: Some(Rc::new(RefCell::new(exports))),
        }
    }

    /// Create a module that shares the same namespace as the given Rc (avoids cloning the export map).
    pub fn from_namespace(name: String, namespace: Rc<RefCell<HashMap<String, Value>>>) -> Self {
        Self {
            name,
            globals: Vec::new(),
            global_names: BTreeMap::new(),
            namespace: Some(namespace),
        }
    }

    /// Get export by name (from namespace). Used for "from X import a" and LoadGlobal in module context.
    pub fn get_export(&self, name: &str) -> Option<Value> {
        self.namespace
            .as_ref()
            .and_then(|rc| rc.borrow().get(name).cloned())
    }

    /// Set export by name (for StoreGlobal in module context). Updates shared namespace.
    pub fn set_export(&self, name: &str, value: Value) {
        if let Some(ref rc) = self.namespace {
            rc.borrow_mut().insert(name.to_string(), value);
        }
    }

    /// Ensure globals vec has a slot for the given bytecode index (>= BUILTIN_END).
    pub fn ensure_slot(&mut self, index: usize) {
        if index < BUILTIN_END {
            return;
        }
        let slot_index = index - BUILTIN_END;
        if self.globals.len() <= slot_index {
            self.globals.resize(slot_index + 1, default_global_slot());
        }
    }

    /// Get slot for bytecode global index (>= BUILTIN_END). Returns None if index is builtin or out of range.
    pub fn get_slot(&self, index: usize) -> Option<&GlobalSlot> {
        if index < BUILTIN_END {
            return None;
        }
        let slot_index = index - BUILTIN_END;
        self.globals.get(slot_index)
    }

    /// Get mutable slot for bytecode global index (>= BUILTIN_END). Grows vec if needed.
    pub fn get_slot_mut(&mut self, index: usize) -> Option<&mut GlobalSlot> {
        if index < BUILTIN_END {
            return None;
        }
        self.ensure_slot(index);
        let slot_index = index - BUILTIN_END;
        self.globals.get_mut(slot_index)
    }

    /// Set name for a global index (>= BUILTIN_END).
    pub fn set_global_name(&mut self, index: usize, name: String) {
        if index >= BUILTIN_END {
            self.global_names.insert(index, name);
        }
    }

    /// Get name for a global index.
    pub fn get_global_name(&self, index: usize) -> Option<&String> {
        self.global_names.get(&index)
    }

    /// Resolve global index to slot index in self.globals (index - BUILTIN_END), or None if builtin.
    pub fn slot_index_for_bytecode_index(&self, index: usize) -> Option<usize> {
        if index < BUILTIN_END {
            None
        } else {
            Some(index - BUILTIN_END)
        }
    }
}

impl std::fmt::Debug for ModuleObject {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ModuleObject")
            .field("name", &self.name)
            .field("globals_len", &self.globals.len())
            .field("global_names_len", &self.global_names.len())
            .field("has_namespace", &self.namespace.is_some())
            .finish()
    }
}
