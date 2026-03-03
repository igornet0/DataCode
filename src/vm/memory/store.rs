// Heavy value storage for Stage 1 ValueStore migration.
// Holds Value variants that are "heavy" (Table, Tensor, Image, etc.) so that
// stack/globals/slots only hold ValueId; one place to materialize Value at native boundaries.

use crate::common::value::Value;

/// Stores heavy Value variants (Table, Tensor, Image, etc.); indexed by ValueCell::Heavy(usize).
#[derive(Debug, Default)]
pub struct HeavyStore {
    cells: Vec<Value>,
}

impl HeavyStore {
    pub fn new() -> Self {
        HeavyStore { cells: Vec::new() }
    }

    /// Push a heavy value and return its index.
    #[inline]
    pub fn push(&mut self, v: Value) -> usize {
        let id = self.cells.len();
        self.cells.push(v);
        id
    }

    #[inline]
    pub fn get(&self, index: usize) -> Option<&Value> {
        self.cells.get(index)
    }

    #[inline]
    pub fn get_mut(&mut self, index: usize) -> Option<&mut Value> {
        self.cells.get_mut(index)
    }

    pub fn len(&self) -> usize {
        self.cells.len()
    }

    /// Clear all heavy values. Used when reusing VM for stateless runs (e.g. HTTP requests).
    pub fn clear(&mut self) {
        self.cells.clear();
    }
}
