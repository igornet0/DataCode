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

    pub fn clear(&mut self) {
        if self.keys.is_empty() {
            return;
        }
        for k in self.keys.iter_mut() {
            *k = EMPTY_KEY;
        }
        self.len = 0;
    }

    #[inline]
    pub fn contains_key(&self, canonical: i64) -> bool {
        self.get(canonical).is_some()
    }

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

    pub fn insert(&mut self, canonical: i64, slot: IntegralSlot) {
        if self.keys.is_empty() {
            self.grow(8);
        }
        if self.len * 10 >= self.keys.len() * 7 {
            self.grow(self.keys.len() * 2);
        }
        let cap = self.keys.len();
        let mut idx = hash_slot(canonical, cap);
        for _ in 0..cap {
            let k = self.keys[idx];
            if k == EMPTY_KEY || k == canonical {
                if k == EMPTY_KEY {
                    self.len += 1;
                }
                self.keys[idx] = canonical;
                self.slots[idx] = slot;
                return;
            }
            idx = (idx + 1) & (cap - 1);
        }
        self.grow(cap * 2);
        self.insert(canonical, slot);
    }

    pub fn insert_heap(&mut self, canonical: i64, value_id: ValueId) {
        self.insert(canonical, IntegralSlot::from_value_id(value_id));
    }

    pub fn insert_tagged(&mut self, canonical: i64, tv: TaggedValue) {
        self.insert(canonical, IntegralSlot::from_tagged(tv));
    }

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
                self.keys[idx] = EMPTY_KEY;
                self.len = self.len.saturating_sub(1);
                return Some(self.slots[idx]);
            }
            idx = (idx + 1) & (cap - 1);
        }
        None
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
    let h = (key as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    (h as usize) & (cap - 1)
}

impl IntegralMap {
    fn grow(&mut self, new_cap: usize) {
        let new_cap = new_cap.max(8).next_power_of_two();
        let old_keys = std::mem::replace(&mut self.keys, vec![EMPTY_KEY; new_cap]);
        let old_slots = std::mem::replace(
            &mut self.slots,
            vec![IntegralSlot::Immediate(TaggedValue::null()); new_cap],
        );
        let old_len = self.len;
        self.len = 0;
        for (k, slot) in old_keys.into_iter().zip(old_slots) {
            if k != EMPTY_KEY {
                self.insert(k, slot);
            }
        }
        debug_assert_eq!(self.len, old_len);
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
    fn remove_and_reinsert() {
        let mut m = IntegralMap::new();
        m.insert_heap(1, 10);
        assert_eq!(m.remove(1), Some(IntegralSlot::Heap(10)));
        assert!(m.get(1).is_none());
        m.insert_tagged(1, TaggedValue::from_f64(3.0));
        assert!(m.get(1).unwrap().as_tagged().is_number());
    }
}
