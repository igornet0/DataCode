// Tagged value representation (NaN-boxing): one u64 for numbers, bool, null, or heap reference.
// Avoids ValueStore lookup for primitives on the hot path.

use crate::common::value_store::ValueId;

/// Single-word value: either an f64 (using NaN-boxing) or a tagged immediate/heap ref.
#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(transparent)]
pub struct TaggedValue(pub u64);

// NaN-boxing: use a quiet NaN range for tagged values so any non-tagged bit pattern is a valid f64.
// TAG_VALUE must leave low 4 bits of high word (bits 48-51) for the tag so (v>>48)&0xF == tag.
const TAG_MASK: u64 = 0x7FF0_0000_0000_0000;
const TAG_VALUE: u64 = 0x7FF0_0000_0000_0000;
#[allow(dead_code)]
const PAYLOAD_MASK: u64 = 0x0000_FFFF_FFFF_FFFF;

#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Tag {
    Null = 0,
    False = 1,
    True = 2,
    Int = 3,   // signed 32-bit in payload
    Heap = 4,  // ValueId in low 32 bits
}

impl TaggedValue {
    #[inline]
    pub fn from_f64(n: f64) -> Self {
        TaggedValue(n.to_bits())
    }

    #[inline]
    pub fn from_bool(b: bool) -> Self {
        TaggedValue(TAG_VALUE | ((if b { Tag::True } else { Tag::False }) as u64) << 48)
    }

    #[inline]
    pub fn null() -> Self {
        TaggedValue(TAG_VALUE | ((Tag::Null as u64) << 48))
    }

    #[inline]
    pub fn from_i32(n: i32) -> Self {
        TaggedValue(TAG_VALUE | ((Tag::Int as u64) << 48) | (n as u32 as u64))
    }

    #[inline]
    pub fn from_heap(id: ValueId) -> Self {
        TaggedValue(TAG_VALUE | ((Tag::Heap as u64) << 48) | (id as u64))
    }

    /// True if this word is a number (including Inf, -0); false if tagged.
    #[inline]
    pub fn is_number(self) -> bool {
        (self.0 & TAG_MASK) != TAG_VALUE
    }

    #[inline]
    pub fn is_null(self) -> bool {
        self.0 == (TAG_VALUE | ((Tag::Null as u64) << 48))
    }

    #[inline]
    pub fn is_bool(self) -> bool {
        let tag = (self.0 >> 48) & 0xF;
        tag == Tag::False as u64 || tag == Tag::True as u64
    }

    #[inline]
    pub fn is_heap(self) -> bool {
        ((self.0 >> 48) & 0xF) == Tag::Heap as u64
    }

    #[inline]
    pub fn is_int(self) -> bool {
        ((self.0 >> 48) & 0xF) == Tag::Int as u64
    }

    #[inline]
    pub fn get_f64(self) -> f64 {
        debug_assert!(self.is_number());
        f64::from_bits(self.0)
    }

    #[inline]
    pub fn get_bool(self) -> bool {
        debug_assert!(self.is_bool());
        ((self.0 >> 48) & 0xF) == Tag::True as u64
    }

    #[inline]
    pub fn get_heap_id(self) -> ValueId {
        debug_assert!(self.is_heap());
        (self.0 & 0xFFFF_FFFF) as ValueId
    }

    #[inline]
    pub fn get_i32(self) -> i32 {
        debug_assert!(self.is_int());
        (self.0 & 0xFFFF_FFFF) as i32
    }

    /// Returns true if this is an immediate (number, bool, null, int) that does not need store lookup.
    #[inline]
    pub fn is_immediate(self) -> bool {
        if self.is_number() {
            return true;
        }
        let tag = (self.0 >> 48) & 0xF;
        tag <= Tag::Int as u64
    }
}

impl std::fmt::Debug for TaggedValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.is_number() {
            write!(f, "TaggedValue::Number({})", self.get_f64())
        } else if self.is_null() {
            write!(f, "TaggedValue::Null")
        } else if self.is_bool() {
            write!(f, "TaggedValue::Bool({})", self.get_bool())
        } else if self.is_int() {
            write!(f, "TaggedValue::Int({})", self.get_i32())
        } else if self.is_heap() {
            write!(f, "TaggedValue::Heap({})", self.get_heap_id())
        } else {
            write!(f, "TaggedValue(0x{:016x})", self.0)
        }
    }
}
