//! Hash-set storage: keys only via [`ObjectMap`] with a null sentinel value id.

use crate::common::object_map::{KeyHash, ObjectMap};
use crate::common::tagged_value::TaggedValue;
use crate::common::value_store::ValueId;
use crate::common::value_store::NULL_VALUE_ID;

/// Unique-value set backed by the same bucket map as plain dicts.
#[derive(Debug, Clone)]
pub struct SetMap {
    inner: ObjectMap,
    /// Incremented on every mutating operation; iterators snapshot this at start.
    pub generation: u64,
}

impl Default for SetMap {
    fn default() -> Self {
        Self::new()
    }
}

impl SetMap {
    pub fn new() -> Self {
        Self {
            inner: ObjectMap::new(),
            generation: 0,
        }
    }

    pub fn with_capacity(cap: usize) -> Self {
        Self {
            inner: ObjectMap::with_capacity(cap),
            generation: 0,
        }
    }

    fn bump_generation(&mut self) {
        self.generation = self.generation.wrapping_add(1);
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    pub fn generation(&self) -> u64 {
        self.generation
    }

    pub fn inner_ref(&self) -> &ObjectMap {
        &self.inner
    }

    pub fn inner_mut(&mut self) -> &mut ObjectMap {
        &mut self.inner
    }

    pub fn clear(&mut self) {
        if !self.inner.is_empty() {
            self.inner.clear();
            self.bump_generation();
        }
    }

    /// Insert whole-number key into [`ObjectMap::integral_index`] only (A* `closed_set` / `open_set` RAM).
    /// Membership uses [`Self::contains_integral`]; bucket chains are not needed for canonical keys.
    pub fn insert_integral_only(&mut self, canonical: i64, key_id: ValueId) -> bool {
        if self.contains_integral(canonical) {
            return false;
        }
        self.inner
            .upsert_integral_tagged(canonical, key_id, TaggedValue::null());
        self.bump_generation();
        true
    }

    /// Insert whole-number key with bucket + integral index in sync (non-integral fallback paths).
    pub fn insert_integral_sync(&mut self, canonical: i64, key_id: ValueId) -> bool {
        if self.contains_integral(canonical) {
            return false;
        }
        let h = crate::common::numeric::hash_integral_key(canonical);
        self.insert(h, key_id, |_| false, Some(canonical))
    }

    /// Insert key if not present. Returns true when a new element was added.
    /// Pass `canonical` for whole numeric keys (maintains [`ObjectMap::integral_index`]).
    pub fn insert<F>(&mut self, hash: KeyHash, key_id: ValueId, eq: F, canonical: Option<i64>) -> bool
    where
        F: Fn(ValueId) -> bool,
    {
        if let Some(c) = canonical {
            if self.contains_integral(c) {
                return false;
            }
        } else if self.inner.find_in_bucket(hash, &eq).is_some() {
            return false;
        }
        self.inner.upsert(hash, key_id, NULL_VALUE_ID, eq);
        if let Some(c) = canonical {
            self.inner.upsert_integral(c, key_id, NULL_VALUE_ID);
        }
        self.bump_generation();
        true
    }

    pub fn contains<F>(&self, hash: KeyHash, eq: F) -> bool
    where
        F: Fn(ValueId) -> bool,
    {
        self.inner.find_in_bucket(hash, eq).is_some()
    }

    /// Membership by canonical integral key (no `load_value` on bucket chain).
    #[inline]
    pub fn contains_integral(&self, canonical: i64) -> bool {
        self.inner.contains_integral(canonical)
    }

    /// Remove by canonical integral key only (O(1); no bucket equality).
    pub fn discard_integral(&mut self, canonical: i64) -> bool {
        if self.inner.remove_integral(canonical).is_some() {
            self.bump_generation();
            true
        } else {
            false
        }
    }

    /// Remove matching key. Returns true if removed.
    /// Pass `canonical` when the key is a whole numeric value (keeps [`ObjectMap::integral_index`] in sync).
    pub fn remove<F>(&mut self, hash: KeyHash, eq: F, canonical: Option<i64>) -> bool
    where
        F: Fn(ValueId) -> bool,
    {
        if self.inner.remove(hash, eq, canonical).is_some() {
            self.bump_generation();
            true
        } else {
            false
        }
    }

    pub fn iter_key_ids(&self) -> impl Iterator<Item = ValueId> + '_ {
        self.inner
            .iter_entries()
            .map(|(_, key_id, _)| key_id)
    }

    /// Removes and returns one arbitrary stored key id. Order is unspecified (depends on map iteration).
    /// `intern_integral_key` rebuilds the element id for integral-only entries (no bucket chain).
    pub fn pop_arbitrary<F>(&mut self, intern_integral_key: F) -> Option<ValueId>
    where
        F: FnOnce(i64) -> ValueId,
    {
        use crate::common::numeric::hash_integral_key;

        if let Some(canonical) = self.inner.first_integral_canonical() {
            let h = hash_integral_key(canonical);
            if let Some(kid) = self
                .inner
                .buckets_ref()
                .get(&h)
                .and_then(|b| b.first().map(|e| e.key_id))
            {
                if self.remove(h, |id| id == kid, Some(canonical)) {
                    return Some(kid);
                }
            }
            if self.discard_integral(canonical) {
                return Some(intern_integral_key(canonical));
            }
            return None;
        }

        let (h, kid, _) = self.inner.iter_entries().next()?;
        if self.remove(h, |id| id == kid, None) {
            Some(kid)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tagged_f64_canonicals_unique_through_2000() {
        use crate::common::numeric::tagged_integral_canonical_if_whole;
        use crate::common::TaggedValue;
        let mut seen = std::collections::HashSet::new();
        for i in 0..2000i64 {
            let tv = TaggedValue::from_f64(i as f64);
            let c = tagged_integral_canonical_if_whole(tv).expect("whole");
            assert!(seen.insert(c), "canonical collision at i={}", i);
        }
    }

    #[test]
    fn insert_integral_only_2000_distinct_keys() {
        let mut s = SetMap::new();
        let mut added = 0usize;
        for i in 0..2000i64 {
            let key_id = i as ValueId + 1;
            if s.insert_integral_only(i, key_id) {
                added += 1;
            }
        }
        assert_eq!(added, 2000, "each canonical key should insert once");
        assert_eq!(s.len(), 2000);
        assert_eq!(s.inner_ref().buckets_ref().values().map(|v| v.len()).sum::<usize>(), 0);
        for i in 0..2000i64 {
            assert!(s.contains_integral(i), "missing {}", i);
        }
    }

    #[test]
    fn pop_arbitrary_whole_number_with_bucket() {
        use crate::common::numeric::hash_integral_key;
        let mut s = SetMap::new();
        let canonical = 5i64;
        let key_id = 42u32;
        let h = hash_integral_key(canonical);
        s.insert(h, key_id, |_| false, Some(canonical));
        assert_eq!(s.len(), 1);
        let popped = s.pop_arbitrary(|_| key_id);
        assert_eq!(popped, Some(key_id));
        assert_eq!(s.len(), 0);
    }

    #[test]
    fn pop_arbitrary_integral_only() {
        let mut s = SetMap::new();
        s.insert_integral_only(5, 42);
        assert_eq!(s.len(), 1);
        let popped = s.pop_arbitrary(|c| {
            assert_eq!(c, 5);
            42
        });
        assert_eq!(popped, Some(42));
        assert_eq!(s.len(), 0);
    }
}
