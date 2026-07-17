//! HashMap-backed object storage: `HashMap<KeyHash, Bucket>` with chaining by `(key_id, value_id)`.
//!
//! Finite mathematical integer keys (`int`, whole `number`/`float`) also live in
//! [`ObjectMap::integral_index`] for O(1) lookup by canonical `i64` (Python `hash(1)==hash(1.0)`).

use std::collections::HashMap;

use crate::common::integral_map::{IntegralMap, IntegralSlot};
use crate::common::numeric::hash_integral_key;
use crate::common::tagged_value::TaggedValue;
use crate::common::value_store::ValueId;

/// Stable bucket index derived from [`crate::common::type_model::object_key_hash64`].
pub type KeyHash = u64;

#[derive(Debug, Clone)]
pub struct ObjectEntry {
    pub key_id: ValueId,
    pub value_id: ValueId,
}

/// Flat dictionary using bucket chaining (no string coercion).
#[derive(Debug, Clone)]
pub struct ObjectMap {
    buckets: HashMap<KeyHash, Vec<ObjectEntry>>,
    /// Canonical `i64` → value; authoritative for integral-only fast paths (see [`Self::len`]).
    integral_index: IntegralMap,
    /// When true: mutation APIs return errors at VM boundary.
    pub frozen: bool,
    /// User `{}` / `dict()` literals and plain stress-test maps — skip class visibility scans.
    plain: bool,
}

impl Default for ObjectMap {
    fn default() -> Self {
        Self::new()
    }
}

impl ObjectMap {
    pub fn new() -> Self {
        Self {
            buckets: HashMap::new(),
            integral_index: IntegralMap::new(),
            frozen: false,
            plain: false,
        }
    }

    pub fn with_capacity(cap: usize) -> Self {
        Self {
            buckets: HashMap::with_capacity(cap),
            integral_index: IntegralMap::with_capacity(cap),
            frozen: false,
            plain: false,
        }
    }

    /// Plain dict literal / `{}` / reused A* score maps (no class visibility metadata).
    pub fn new_plain() -> Self {
        Self {
            buckets: HashMap::new(),
            integral_index: IntegralMap::new(),
            frozen: false,
            plain: true,
        }
    }

    pub fn with_capacity_plain(cap: usize) -> Self {
        Self {
            buckets: HashMap::with_capacity(cap),
            integral_index: IntegralMap::with_capacity(cap),
            frozen: false,
            plain: true,
        }
    }

    #[inline]
    pub fn is_plain(&self) -> bool {
        self.plain
    }

    #[inline]
    pub fn mark_plain(&mut self) {
        self.plain = true;
    }

    #[inline]
    pub fn is_frozen(&self) -> bool {
        self.frozen
    }

    /// Distinct keys count (same logical size as Python dict len).
    /// When the integral side table is in use, it is authoritative (see fast upsert path).
    pub fn len(&self) -> usize {
        if !self.integral_index.is_empty() {
            return self.integral_index.len();
        }
        self.buckets.values().map(|v| v.len()).sum()
    }

    pub fn is_empty(&self) -> bool {
        if !self.integral_index.is_empty() {
            return false;
        }
        self.buckets.values().all(|v| v.is_empty())
    }

    pub fn clear(&mut self) {
        self.buckets.clear();
        if self.buckets.capacity() > 4096 {
            self.buckets.shrink_to_fit();
        }
        self.integral_index.clear();
    }

    /// Release bucket / integral side-table capacity (stress-test reuse between A* runs).
    pub fn shrink_to_fit(&mut self) {
        self.buckets.shrink_to_fit();
        self.integral_index.shrink_to_fit();
    }

    /// O(1) lookup by canonical integral key.
    #[inline]
    pub fn find_integral_slot(&self, canonical: i64) -> Option<IntegralSlot> {
        self.integral_index.get(canonical)
    }

    #[inline]
    pub fn contains_integral(&self, canonical: i64) -> bool {
        self.integral_index.contains_key(canonical)
    }

    /// First whole-number key in the integral side table (order unspecified).
    #[inline]
    pub fn first_integral_canonical(&self) -> Option<i64> {
        self.integral_index.first_canonical()
    }

    /// True when the integral side table is populated (authoritative for [`Self::len`]).
    #[inline]
    pub fn integral_in_use(&self) -> bool {
        !self.integral_index.is_empty()
    }

    /// All canonical integral keys in the side table (order unspecified).
    #[inline]
    pub fn iter_integral_canonicals(&self) -> impl Iterator<Item = i64> + '_ {
        self.integral_index.iter_canonicals()
    }

    /// Legacy: heap [`ValueId`] only (immediate slots → [`NULL_VALUE_ID`] sentinel for set compat).
    #[inline]
    pub fn find_integral(&self, canonical: i64) -> Option<ValueId> {
        self.integral_index.get_heap_id(canonical)
    }

    /// Insert or replace the integral side table (bucket chain must be updated separately).
    pub fn upsert_integral(&mut self, canonical: i64, _key_id: ValueId, value_id: ValueId) {
        self.integral_index.insert_heap(canonical, value_id);
    }

    /// Rewrite an immediate integral value in place (no new heap cell).
    pub fn try_write_integral_immediate(&mut self, canonical: i64, value_tv: TaggedValue) -> bool {
        if let Some(slot) = self.integral_index.get_mut(canonical) {
            return slot.try_write_immediate(value_tv);
        }
        false
    }

    /// Store an immediate value without allocating a [`ValueCell`].
    pub fn upsert_integral_tagged(
        &mut self,
        canonical: i64,
        _key_id: ValueId,
        value_tv: TaggedValue,
    ) {
        self.integral_index.insert_tagged(canonical, value_tv);
    }

    /// Integral upsert + bucket chain sync without [`ValueStore`] reads (safe alongside in-place mutation).
    pub fn upsert_integral_with_bucket_sync(
        &mut self,
        canonical: i64,
        key_id: ValueId,
        value_id: ValueId,
    ) {
        let had = self.integral_index.contains_key(canonical);
        self.upsert_integral(canonical, key_id, value_id);
        let h = hash_integral_key(canonical);
        let bucket = self.bucket_mut_or_insert(h);
        if had {
            if let Some(e) = bucket.iter_mut().find(|e| e.key_id == key_id) {
                e.value_id = value_id;
                return;
            }
            if bucket.len() == 1 {
                bucket[0].key_id = key_id;
                bucket[0].value_id = value_id;
                return;
            }
        }
        bucket.push(ObjectEntry {
            key_id,
            value_id,
        });
    }

    /// Remove integral slot; returns previous heap id if any (immediate → sentinel).
    pub fn remove_integral(&mut self, canonical: i64) -> Option<ValueId> {
        self.integral_index
            .remove(canonical)
            .map(IntegralSlot::heap_value_id)
    }

    pub fn buckets_ref(&self) -> &HashMap<KeyHash, Vec<ObjectEntry>> {
        &self.buckets
    }

    pub fn iter_entries(&self) -> impl Iterator<Item = (KeyHash, ValueId, ValueId)> + '_ {
        self.buckets.iter().flat_map(|(&h, bucket)| {
            bucket
                .iter()
                .map(move |e| (h, e.key_id, e.value_id))
        })
    }

    /// Rewrite every stored [`ValueId`] (bucket entries and integral heap slots).
    /// Used when promoting a call-arena object so nested field values survive `leave_ephemeral`.
    pub fn remap_stored_ids<F>(&mut self, mut f: F)
    where
        F: FnMut(ValueId) -> ValueId,
    {
        for bucket in self.buckets.values_mut() {
            for e in bucket.iter_mut() {
                e.key_id = f(e.key_id);
                e.value_id = f(e.value_id);
            }
        }
        let canonicals: Vec<i64> = self.iter_integral_canonicals().collect();
        for c in canonicals {
            if let Some(IntegralSlot::Heap(id)) = self.integral_index.get(c) {
                let new_id = f(id);
                if new_id != id {
                    self.integral_index.insert_heap(c, new_id);
                }
            }
        }
    }

    /// Lookup by hash + equality predicate on key ids.
    pub fn find_in_bucket<F>(&self, hash: KeyHash, eq: F) -> Option<ValueId>
    where
        F: Fn(ValueId) -> bool,
    {
        let bucket = self.buckets.get(&hash)?;
        for e in bucket {
            if eq(e.key_id) {
                return Some(e.value_id);
            }
        }
        None
    }

    /// Replace existing entry equal under `eq`, or append.
    pub fn upsert<F>(&mut self, hash: KeyHash, key_id: ValueId, value_id: ValueId, eq: F)
    where
        F: Fn(ValueId) -> bool,
    {
        let bucket = self.buckets.entry(hash).or_default();
        for e in bucket.iter_mut() {
            if eq(e.key_id) {
                e.value_id = value_id;
                return;
            }
        }
        bucket.push(ObjectEntry {
            key_id,
            value_id,
        });
    }

    /// Remove first entry in bucket matching `eq`. Returns removed value id if any.
    pub fn remove<F>(&mut self, hash: KeyHash, eq: F, canonical: Option<i64>) -> Option<ValueId>
    where
        F: Fn(ValueId) -> bool,
    {
        let bucket = self.buckets.get_mut(&hash)?;
        let pos = bucket.iter().position(|e| eq(e.key_id))?;
        let removed = bucket.remove(pos).value_id;
        if bucket.is_empty() {
            self.buckets.remove(&hash);
        }
        if let Some(c) = canonical {
            self.integral_index.remove(c);
        }
        Some(removed)
    }

    pub fn insert_with_hash(
        &mut self,
        hash: KeyHash,
        key_id: ValueId,
        value_id: ValueId,
        eq_key: impl Fn(ValueId) -> bool,
    ) {
        self.upsert(hash, key_id, value_id, eq_key);
    }

    pub(crate) fn bucket_mut_or_insert(&mut self, h: KeyHash) -> &mut Vec<ObjectEntry> {
        self.buckets.entry(h).or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn upsert_replaces_same_predicate() {
        let mut m = ObjectMap::new();
        let h = 42u64;
        m.upsert(h, 1, 100, |k| k == 1);
        m.upsert(h, 2, 200, |k| k == 2);
        assert_eq!(m.len(), 2);
        m.upsert(h, 1, 999, |k| k == 1);
        assert_eq!(m.len(), 2);
        assert_eq!(m.find_in_bucket(h, |k| k == 1), Some(999));
    }

    #[test]
    fn remove_empties_bucket() {
        let mut m = ObjectMap::new();
        let h = 7u64;
        m.upsert(h, 1, 10, |k| k == 1);
        assert_eq!(m.remove(h, |k| k == 1, None), Some(10));
        assert!(m.is_empty());
    }

    #[test]
    fn integral_index_lookup_and_replace() {
        let mut m = ObjectMap::new();
        m.upsert_integral(42, 1, 100);
        assert_eq!(m.find_integral(42), Some(100));
        m.upsert_integral(42, 2, 200);
        assert_eq!(m.find_integral(42), Some(200));
        assert_eq!(m.remove_integral(42), Some(200));
        assert_eq!(m.find_integral(42), None);
    }

    #[test]
    fn integral_tagged_scores() {
        let mut m = ObjectMap::new();
        m.upsert_integral_tagged(7, 1, TaggedValue::from_i32(42));
        assert_eq!(
            m.find_integral_slot(7),
            Some(IntegralSlot::Immediate(TaggedValue::from_i32(42)))
        );
        m.upsert_integral_tagged(7, 1, TaggedValue::from_i32(99));
        assert_eq!(
            m.find_integral_slot(7),
            Some(IntegralSlot::Immediate(TaggedValue::from_i32(99)))
        );
    }

    #[test]
    fn integral_only_side_table_lookup_without_buckets() {
        let mut m = ObjectMap::new();
        m.upsert_integral(0, 1, 100);
        m.upsert_integral(1, 2, 200);
        m.upsert_integral(2, 3, 300);
        assert!(m.iter_entries().next().is_none());
        assert!(m.find_integral_slot(1).is_some());
        assert_eq!(m.find_integral(1), Some(200));
    }

    /// Sparse high cell ids (grid 1000×5000 style) must stay on open-addressing map, not a 5M-slot `Vec`.
    #[test]
    fn integral_sparse_high_cell_ids() {
        let mut m = ObjectMap::new();
        for i in 0..8000i64 {
            let cell = i * 5000 + (i % 500);
            m.upsert_integral(cell, cell as ValueId + 1, cell as ValueId + 100);
        }
        assert_eq!(m.len(), 8000);
        assert_eq!(m.find_integral(7 * 5000 + 7), Some((7 * 5000 + 7) as ValueId + 100));
    }
}
