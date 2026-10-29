//! Open-addressing map for canonical integral keys (`i64`).
//! Used by [`crate::common::object_map::ObjectMap`] and set membership side tables.

use crate::common::tagged_value::TaggedValue;
use crate::common::value_store::{ValueId, NULL_VALUE_ID};

/// Dict/set value stored without a heap [`ValueCell`] when possible (A* scores).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum IntegralSlot {
    /// Inline stack word (number, int, bool, null).
    Immediate(TaggedValue),
    /// Non-immediate or shared heap cell.
    Heap(ValueId),
}

impl IntegralSlot {
    #[inline]
    pub fn from_tagged(tv: TaggedValue) -> Self {
        if tv.is_heap() {
            IntegralSlot::Heap(tv.get_heap_id())
        } else {
            IntegralSlot::Immediate(tv)
        }
    }

    #[inline]
    pub fn from_value_id(id: ValueId) -> Self {
        if id == NULL_VALUE_ID {
            IntegralSlot::Immediate(TaggedValue::null())
        } else {
            IntegralSlot::Heap(id)
        }
    }

    /// Legacy API: heap id only; immediate slots map to [`NULL_VALUE_ID`] for set sentinel.
    #[inline]
    pub fn heap_value_id(self) -> ValueId {
        match self {
            IntegralSlot::Heap(id) => id,
            IntegralSlot::Immediate(tv) if tv.is_null() => NULL_VALUE_ID,
            IntegralSlot::Immediate(_) => NULL_VALUE_ID,
        }
    }

    #[inline]
    pub fn as_tagged(self) -> TaggedValue {
        match self {
            IntegralSlot::Immediate(tv) => tv,
            IntegralSlot::Heap(id) => TaggedValue::from_heap(id),
        }
    }

    /// Update an immediate numeric slot in place (A* score rewrite).
    #[inline]
    pub fn try_write_immediate(&mut self, tv: TaggedValue) -> bool {
        match self {
            IntegralSlot::Immediate(slot_tv) => {
                if tv.is_number() && slot_tv.is_number() {
                    *slot_tv = tv;
                    return true;
                }
                if tv.is_int() && slot_tv.is_int() {
                    *slot_tv = tv;
                    return true;
                }
                if tv.is_number() && slot_tv.is_int() {
                    *slot_tv = tv;
                    return true;
                }
                if tv.is_int() && slot_tv.is_number() {
                    *slot_tv = TaggedValue::from_f64(tv.get_i32() as f64);
                    return true;
                }
                false
            }
            _ => false,
        }
    }
}

const EMPTY_KEY: i64 = i64::MIN;

/// Robin-Hood-style open addressing on canonical `i64` keys.
#[derive(Debug, Clone)]
pub struct IntegralMap {
    keys: Vec<i64>,
    slots: Vec<IntegralSlot>,
    len: usize,
}

impl Default for IntegralMap {
    fn default() -> Self {
        Self::new()
    }
}

impl IntegralMap {
    pub fn new() -> Self {
        Self {
            keys: Vec::new(),
            slots: Vec::new(),
            len: 0,
        }
    }

    pub fn with_capacity(n: usize) -> Self {
        let cap = capacity_for(n);
        Self {
            keys: vec![EMPTY_KEY; cap],
            slots: vec![IntegralSlot::Immediate(TaggedValue::null()); cap],
            len: 0,
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// First occupied canonical key (order unspecified; for set `pop`).
    pub fn first_canonical(&self) -> Option<i64> {
        if self.len == 0 {
            return None;
        }
        for &k in &self.keys {
            if k != EMPTY_KEY {
                return Some(k);
            }
        }
        None
    }

    /// All occupied canonical keys (order unspecified).
    pub fn iter_canonicals(&self) -> impl Iterator<Item = i64> + '_ {
        self.keys
            .iter()
            .filter(|&&k| k != EMPTY_KEY)
            .copied()
    }

    /// All occupied `(canonical, slot)` pairs (order unspecified).
    pub fn iter_entries(&self) -> impl Iterator<Item = (i64, IntegralSlot)> + '_ {
        self.keys
            .iter()
            .zip(self.slots.iter())
            .filter(|(&k, _)| k != EMPTY_KEY)
            .map(|(&k, &slot)| (k, slot))
    }

    /// Drop occupied entries. When the table grew large, release backing storage (A* reuse + `.clear()`).
    pub fn clear(&mut self) {
        if self.keys.is_empty() {
            return;
        }
        const SHRINK_THRESHOLD: usize = 4096;
        if self.keys.len() > SHRINK_THRESHOLD {
            self.keys = Vec::new();
            self.slots = Vec::new();
        } else {
            for k in self.keys.iter_mut() {
                *k = EMPTY_KEY;
            }
        }
        self.len = 0;
    }

    /// Release unused capacity after [`Self::clear`] (optional explicit shrink).
    pub fn shrink_to_fit(&mut self) {
        if self.len == 0 && !self.keys.is_empty() {
            self.keys = Vec::new();
            self.slots = Vec::new();
        }
    }

    #[inline]
    pub fn contains_key(&self, canonical: i64) -> bool {
        self.get(canonical).is_some()
    }

    #[inline]
    pub fn get_mut(&mut self, canonical: i64) -> Option<&mut IntegralSlot> {
        if self.len == 0 {
            return None;
        }
        let cap = self.keys.len();
        let mut idx = hash_slot(canonical, cap);
        for _ in 0..cap {
            let k = self.keys[idx];
            if k == EMPTY_KEY {
                return None;
            }
            if k == canonical {
                return Some(&mut self.slots[idx]);
            }
            idx = (idx + 1) & (cap - 1);
        }
        None
    }

    #[inline]
    pub fn get(&self, canonical: i64) -> Option<IntegralSlot> {
        if self.len == 0 {
            return None;
        }
        let cap = self.keys.len();
        let mut idx = hash_slot(canonical, cap);
        for _ in 0..cap {
            let k = self.keys[idx];
            if k == EMPTY_KEY {
                return None;
            }
            if k == canonical {
                return Some(self.slots[idx]);
            }
            idx = (idx + 1) & (cap - 1);
        }
        None
    }

    /// Legacy: returns heap id; immediate non-null values still need [`Self::get`].
    #[inline]
    pub fn get_heap_id(&self, canonical: i64) -> Option<ValueId> {
        self.get(canonical).map(IntegralSlot::heap_value_id)
    }

    #[inline]
    pub fn insert(&mut self, canonical: i64, slot: IntegralSlot) {
        if self.len * 10 >= self.keys.len() * 7 {
            self.grow((self.keys.len() * 2).max(8));
        }
        let mask = self.keys.len() - 1;
        let mut idx = hash_slot(canonical, mask + 1);
        // Load factor < 0.7 guarantees an empty slot, so the probe terminates.
        loop {
            let k = self.keys[idx];
            if k == canonical {
                self.slots[idx] = slot;
                return;
            }
            if k == EMPTY_KEY {
                self.len += 1;
                self.keys[idx] = canonical;
                self.slots[idx] = slot;
                return;
            }
            idx = (idx + 1) & mask;
        }
    }

    pub fn insert_heap(&mut self, canonical: i64, value_id: ValueId) {
        self.insert(canonical, IntegralSlot::from_value_id(value_id));
    }

    pub fn insert_tagged(&mut self, canonical: i64, tv: TaggedValue) {
        self.insert(canonical, IntegralSlot::from_tagged(tv));
    }

    #[inline]
    pub fn remove(&mut self, canonical: i64) -> Option<IntegralSlot> {
        if self.len == 0 {
            return None;
        }
        let cap = self.keys.len();
        let mut idx = hash_slot(canonical, cap);
        for _ in 0..cap {
            let k = self.keys[idx];
            if k == EMPTY_KEY {
                return None;
            }
            if k == canonical {
                let removed = self.slots[idx];
                self.backward_shift_delete(idx);
                self.len = self.len.saturating_sub(1);
                return Some(removed);
            }
            idx = (idx + 1) & (cap - 1);
        }
        None
    }

    /// Linear-probing delete: pull later entries of the probe run back into the hole so
    /// lookups that stop at the first [`EMPTY_KEY`] still find them (no tombstones needed).
    fn backward_shift_delete(&mut self, mut hole: usize) {
        let mask = self.keys.len() - 1;
        let mut j = hole;
        loop {
            j = (j + 1) & mask;
            let k = self.keys[j];
            if k == EMPTY_KEY {
                break;
            }
            let home = hash_slot(k, mask + 1);
            // Entry at `j` may move to `hole` only if its home slot is not inside (hole, j].
            if (j.wrapping_sub(home) & mask) >= (j.wrapping_sub(hole) & mask) {
                self.keys[hole] = k;
                self.slots[hole] = self.slots[j];
                hole = j;
            }
        }
        self.keys[hole] = EMPTY_KEY;
    }
}

#[inline]
fn capacity_for(n: usize) -> usize {
    let mut cap = 8usize;
    while cap < n.max(1) * 2 {
        cap *= 2;
    }
    cap
}

#[inline]
fn hash_slot(key: i64, cap: usize) -> usize {
    // Fibonacci hashing: the high bits of the product are well mixed, the low ones are not
    // (low `k` bits of the product depend only on the low `k` bits of the key).
    let h = (key as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    (h >> (64 - cap.trailing_zeros())) as usize
}

impl IntegralMap {
    #[cold]
    #[inline(never)]
    fn grow(&mut self, new_cap: usize) {
        let new_cap = new_cap.max(8).next_power_of_two();
        let old_keys = std::mem::replace(&mut self.keys, vec![EMPTY_KEY; new_cap]);
        let old_slots = std::mem::replace(
            &mut self.slots,
            vec![IntegralSlot::Immediate(TaggedValue::null()); new_cap],
        );
        // Keys are unique, so each one goes to the first empty slot of its probe run.
        let mask = new_cap - 1;
        for (k, slot) in old_keys.into_iter().zip(old_slots) {
            if k != EMPTY_KEY {
                let mut idx = hash_slot(k, new_cap);
                while self.keys[idx] != EMPTY_KEY {
                    idx = (idx + 1) & mask;
                }
                self.keys[idx] = k;
                self.slots[idx] = slot;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_get_replace() {
        let mut m = IntegralMap::new();
        m.insert_tagged(42, TaggedValue::from_i32(100));
        assert_eq!(
            m.get(42),
            Some(IntegralSlot::Immediate(TaggedValue::from_i32(100)))
        );
        m.insert_tagged(42, TaggedValue::from_i32(200));
        assert_eq!(
            m.get(42),
            Some(IntegralSlot::Immediate(TaggedValue::from_i32(200)))
        );
        assert_eq!(m.len(), 1);
    }

    #[test]
    fn sparse_high_cell_ids() {
        let mut m = IntegralMap::with_capacity(8192);
        for i in 0..8000i64 {
            let cell = i * 5000 + (i % 500);
            m.insert_tagged(cell, TaggedValue::from_i32((cell % 100_000) as i32));
        }
        assert_eq!(m.len(), 8000);
        let cell = 7 * 5000 + 7;
        assert!(m.get(cell).is_some());
    }

    #[test]
    fn float_ten_is_number_not_heap_tag() {
        // 10.0 raw bits have nibble 4 at bits 48–51 (same as Heap tag); must not classify as heap.
        let tv = TaggedValue::from_f64(10.0);
        assert!(tv.is_number());
        assert!(!tv.is_heap());
    }

    #[test]
    fn remove_keeps_rest_of_probe_chain_reachable() {
        // Small table forces long probe chains; remove from the middle and check the rest.
        let mut m = IntegralMap::with_capacity(4);
        let keys: Vec<i64> = (0..6).map(|i| i * 1024).collect();
        for &k in &keys {
            m.insert_tagged(k, TaggedValue::from_i32(k as i32));
        }
        for (n, &k) in keys.iter().enumerate() {
            assert!(m.remove(k).is_some());
            for &rest in &keys[n + 1..] {
                assert!(m.contains_key(rest), "lost {} after removing {}", rest, k);
            }
            assert_eq!(m.len(), keys.len() - n - 1);
        }
    }

    #[test]
    fn random_insert_remove_matches_hashset() {
        let mut m = IntegralMap::new();
        let mut reference = std::collections::HashSet::new();
        let mut x: u64 = 12345;
        for _ in 0..20000 {
            x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let key = ((x >> 33) % 512) as i64 * 500;
            if (x >> 20) & 1 == 0 {
                m.insert_tagged(key, TaggedValue::null());
                reference.insert(key);
            } else {
                assert_eq!(m.remove(key).is_some(), reference.remove(&key));
            }
            assert_eq!(m.len(), reference.len());
        }
        for k in 0..512i64 {
            assert_eq!(m.contains_key(k * 500), reference.contains(&(k * 500)));
        }
    }

    #[test]
    fn remove_and_reinsert() {
        let mut m = IntegralMap::new();
        m.insert_heap(1, 10);
        assert_eq!(m.remove(1), Some(IntegralSlot::Heap(10)));
        assert!(m.get(1).is_none());
        m.insert_tagged(1, TaggedValue::from_f64(3.0));
        assert!(m.get(1).unwrap().as_tagged().is_number());
    }
}
