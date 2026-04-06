// Области видимости переменных

use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct Scope {
    pub locals: HashMap<String, usize>, // имя переменной -> индекс в локальном стеке
    pub parent: Option<Box<Scope>>,
}

impl Scope {
    pub fn new() -> Self {
        Self {
            locals: HashMap::new(),
            parent: None,
        }
    }

    pub fn with_parent(parent: Box<Scope>) -> Self {
        Self {
            locals: HashMap::new(),
            parent: Some(parent),
        }
    }

    pub fn define(&mut self, name: String, index: usize) {
        self.locals.insert(name, index);
    }

    pub fn resolve(&self, name: &str) -> Option<usize> {
        if let Some(&index) = self.locals.get(name) {
            Some(index)
        } else if let Some(parent) = &self.parent {
            parent.resolve(name)
        } else {
            None
        }
    }
}
