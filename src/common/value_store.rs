// Value storage / arena for hot path (Stage 1: kill GIL in VM).
// Executor works with ValueId; one mutable borrow of ValueStore per instruction.
// Stack, globals, frame slots are Vec<ValueId>; no Rc/RefCell in hot path.
// Strings are interned in StringPool (per ValueStore) to cut heap fragmentation and duplicate allocations.

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

use super::numeric::{FloatValue, IntValue};
use super::tagged_value::TaggedValue;

/// Handle into ValueStore; executor uses only ids in hot path (no Rc/RefCell/borrow per value).
pub type ValueId = u32;

/// Handle into StringPool; stored in ValueCell::String instead of String to deduplicate and reduce allocations.
pub type StringId = u32;

/// Reserved id for null (allocated once at VM creation).
pub const NULL_VALUE_ID: ValueId = 0;

/// Chunk size for the cell arena; each chunk is one allocation. Enables growth without realloc of a single huge Vec.
const CHUNK_SIZE: usize = 65536;

/// ValueIds >= ARENA_BASE refer to the heap arena (globals/slots). Main store uses 0..ARENA_BASE.
pub const ARENA_BASE: ValueId = 0x8000_0000;

/// Max recycled chunks to keep in free list (lazy shrink: avoid unbounded retention after many resets).
const MAX_FREE_CHUNKS: usize = 4;

/// Soft cap on recycled `(f, node)` shells; overflow drops oldest ids (heappop uses [`scratch_heap_pair_id`]).
const MAX_HEAP_PAIR_FREE: usize = 65536;

/// Bump-style arena for heap globals and frame-slot values. Reduces fragmentation and allows bulk free on reset.
/// Supports partial chunk recycling and configurable chunk size for large arrays/tables.
#[derive(Debug)]
pub struct HeapArena {
    /// Active chunks; ValueId maps to chunks[c][i] for local index = c*chunk_size + i.
    chunks: Vec<Vec<ValueCell>>,
    /// Recycled chunks from clear(); reused in allocate(). Capped at MAX_FREE_CHUNKS (lazy shrink).
    free_chunks: Vec<Vec<ValueCell>>,
    /// Cells per chunk; configurable for large-data scenarios (default 64K).
    chunk_size: usize,
}

impl Default for HeapArena {
    fn default() -> Self {
        HeapArena::new()
    }
}

impl HeapArena {
    pub fn new() -> Self {
        HeapArena {
            chunks: Vec::new(),
            free_chunks: Vec::new(),
            chunk_size: CHUNK_SIZE,
        }
    }

    /// Arena with larger chunks (e.g. for >100k rows). Reduces chunk count and fragmentation.
    pub fn new_with_chunk_size(chunk_size: usize) -> Self {
        let size = chunk_size.max(1024);
        HeapArena {
            chunks: Vec::new(),
            free_chunks: Vec::new(),
            chunk_size: size,
        }
    }

    /// Allocate a cell in the arena; returns ValueId in arena range (>= ARENA_BASE).
    /// When a new chunk is needed, reuses one from free_chunks if available (recycling).
    #[inline]
    pub fn allocate(&mut self, cell: ValueCell) -> ValueId {
        let sz = self.chunk_size;
        let need_new = self.chunks.last().map(|c| c.len() >= sz).unwrap_or(true);
        if need_new {
            let mut chunk = match self.free_chunks.pop() {
                Some(mut reused) => {
                    reused.clear();
                    if reused.capacity() < sz {
                        reused.reserve(sz.saturating_sub(reused.capacity()));
                    }
                    reused
                }
                None => Vec::with_capacity(sz),
            };
            chunk.push(cell);
            let local_idx = self.chunks.len() * sz;
            self.chunks.push(chunk);
            ARENA_BASE.saturating_add(local_idx as ValueId)
        } else {
            let chunk_idx = self.chunks.len() - 1;
            let last = self.chunks.last_mut().unwrap();
            let offset = last.len();
            last.push(cell);
            ARENA_BASE.saturating_add((chunk_idx * sz + offset) as ValueId)
        }
    }

    #[inline]
    pub fn get(&self, id: ValueId) -> Option<&ValueCell> {
        let local = (id as usize).saturating_sub(ARENA_BASE as usize);
        let sz = self.chunk_size;
        let c = local / sz;
        let i = local % sz;
        self.chunks.get(c).and_then(|ch| ch.get(i))
    }

    #[inline]
    pub fn get_mut(&mut self, id: ValueId) -> Option<&mut ValueCell> {
        let local = (id as usize).saturating_sub(ARENA_BASE as usize);
        let sz = self.chunk_size;
        let c = local / sz;
        let i = local % sz;
        self.chunks.get_mut(c).and_then(|ch| ch.get_mut(i))
    }

    /// Clear all active chunks; arena ValueIds become invalid. Chunks move to free_chunks (recycling);
    /// free_chunks is then capped at MAX_FREE_CHUNKS (lazy shrink) to limit memory retention.
    pub fn clear(&mut self) {
        for ch in &mut self.chunks {
            ch.clear();
            self.free_chunks.push(std::mem::take(ch));
        }
        self.chunks.clear();
        if self.free_chunks.len() > MAX_FREE_CHUNKS {
            self.free_chunks.truncate(MAX_FREE_CHUNKS);
        }
    }
}

/// Per-VM string pool: one canonical copy per distinct string. Reduces heap fragmentation and malloc count.
#[derive(Debug, Default)]
pub struct StringPool {
    /// vec[id] = canonical string
    vec: Vec<String>,
    /// map[string] = id for dedup
    map: HashMap<String, StringId>,
}

impl StringPool {
    pub fn new() -> Self {
        StringPool {
            vec: Vec::new(),
            map: HashMap::new(),
        }
    }

    /// Returns StringId for the string; reuses existing id if already interned.
    /// For new strings: one clone for map key, original moved into vec (no extra clone for vec).
    pub fn intern(&mut self, s: String) -> StringId {
        if let Some(&id) = self.map.get(&s) {
            return id;
        }
        let id = self.vec.len() as StringId;
        self.map.insert(s.clone(), id);
        self.vec.push(s);
        id
    }

    /// Returns the string for a StringId; None if id is out of range.
    #[inline]
    pub fn get(&self, id: StringId) -> Option<&str> {
        self.vec.get(id as usize).map(String::as_str)
    }

    pub fn clear(&mut self) {
        self.vec.clear();
        self.map.clear();
    }
}

/// Which cells are listed by [`ValueCell::ObjectFieldList`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ObjectProjectionKind {
    Keys,
    Values,
}

/// One cell in the store; composite types refer to other cells by ValueId.
/// Heavy types (Table, etc.) are stored in HeavyStore and referenced by index.
/// Array and Object are index-only here (no Rc<RefCell> in hot path); Value materialization
/// with Rc<RefCell<...>> happens only at native boundaries (store_convert).
/// Strings are stored as StringId (index into ValueStore's StringPool).
#[derive(Debug, Clone)]
pub enum ValueCell {
    Int(IntValue),
    Float(FloatValue),
    /// Raw IEEE heap cell (distinct from typed [`FloatValue`] on [`Value`]).
    Number(f64),
    Bool(bool),
    Null,
    String(StringId),
    /// Elements as TaggedValue: inline (number/bool/null) without allocate; heap refs as ValueId in tag.
    Array(Vec<TaggedValue>),
    /// Zero-copy subrange of [`ValueCell::Array`] at `base_id` (see [`crate::common::value::ArrayViewData`]).
    ArrayView {
        base_id: ValueId,
        offset: usize,
        length: usize,
    },
    Tuple(Vec<ValueId>),
    Object(crate::common::object_map::ObjectMap),
    Set(crate::common::set_map::SetMap),
    Function(usize),
    ModuleFunction {
        module_uid: u64,
        local_index: usize,
    },
    NativeFunction(usize),
    Path(PathBuf),
    Uuid(u64, u64),
    /// Instants in [`chrono::DateTime<chrono::FixedOffset>`] as (unix secs, subsec nanos, offset from UTC in seconds).
    Date {
        secs: i64,
        nanos: u32,
        offset_secs: i32,
    },
    /// Signed span as whole seconds + subsecond nanoseconds (chrono [`chrono::Duration`]).
    Duration {
        secs: i64,
        nanos: u32,
    },
    /// Index into HeavyStore (Table, Image, etc.)
    Heavy(usize),
    ColumnReference {
        table_handle: usize,
        column_name: String,
    },
    /// Opaque plugin object (tag + id)
    PluginOpaque {
        tag: u8,
        id: u64,
    },
    Window(crate::plot::PlotWindowHandle),
    Enumerate {
        data_id: ValueId,
        start: i64,
    },
    /// Read-only snapshot view over a plain bucket dict's key or value cell ids.
    ObjectFieldList {
        source_object_id: ValueId,
        projection: ObjectProjectionKind,
        element_ids: Vec<ValueId>,
    },
    Ellipsis,
}

/// Arena of value cells in fixed-size chunks; one mutable borrow per instruction.
/// Growth adds new chunks instead of reallocating a single Vec, reducing sys time and fragmentation.
/// Holds a StringPool for interned strings (ValueCell::String stores StringId).
/// Heap globals and frame-slot heap values use the separate HeapArena (ids >= ARENA_BASE) for bulk free on reset.
#[derive(Debug)]
pub struct ValueStore {
    /// Chunks of cells; each chunk has capacity CHUNK_SIZE. ValueId i => chunks[i / CHUNK_SIZE][i % CHUNK_SIZE].
    chunks: Vec<Vec<ValueCell>>,
    string_pool: StringPool,
    /// Bump arena for heap globals and ephemeral heap values; ids >= ARENA_BASE.
    arena: HeapArena,
    /// Canonical whole numbers → one [`ValueCell`] per `i64` (avoids millions of duplicate `Number` allocs).
    whole_i64_cache: HashMap<i64, ValueId>,
    /// Non-whole `f64` bit patterns (`inf`, `nan`, …) → one cell each (A* `float(inf)` defaults).
    f64_bits_cache: HashMap<u64, ValueId>,
    /// Recycled 2-slot heap items (`Array` len 2 / compacted tuple pairs) for `heapq`.
    heap_pair_free_list: Vec<ValueId>,
    /// Single reused 2-slot cell for `heappop` → tuple-unpack (`current_f, current = …`) hot path.
    scratch_heap_pair: Option<ValueId>,
    /// Heap arrays using flat `[f,n,f,n,…]` storage (A* open_heap); not inferred from slot shape alone.
    flat_heap_ids: HashSet<ValueId>,
}

impl Default for ValueStore {
    fn default() -> Self {
        ValueStore::new()
    }
}

impl ValueStore {
    pub fn new() -> Self {
        let mut first = Vec::with_capacity(CHUNK_SIZE);
        first.push(ValueCell::Null);
        ValueStore {
            chunks: vec![first],
            string_pool: StringPool::new(),
            arena: HeapArena::new(),
            whole_i64_cache: HashMap::new(),
            f64_bits_cache: HashMap::new(),
            heap_pair_free_list: Vec::new(),
            scratch_heap_pair: None,
            flat_heap_ids: HashSet::new(),
        }
    }

    /// Ensure capacity for at least `min_capacity` cells (adds chunks if needed).
    /// Call before bulk allocations (e.g. before loading a large table) to avoid repeated chunk growth.
    pub fn reserve_min(&mut self, min_capacity: usize) {
        let required_chunks = min_capacity.div_ceil(CHUNK_SIZE);
        while self.chunks.len() < required_chunks {
            self.chunks.push(Vec::with_capacity(CHUNK_SIZE));
        }
    }

    /// Intern a string and return its StringId. Use before allocate(ValueCell::String(id)).
    #[inline]
    pub fn intern_string(&mut self, s: String) -> StringId {
        self.string_pool.intern(s)
    }

    /// Resolve StringId to &str. Used in load_value and executor fast paths.
    #[inline]
    pub fn get_string(&self, id: StringId) -> Option<&str> {
        self.string_pool.get(id)
    }

    /// Allocate a new cell and return its id. Uses current chunk; adds a new chunk when full.
    #[inline]
    pub fn allocate(&mut self, cell: ValueCell) -> ValueId {
        #[cfg(feature = "profile")]
        crate::vm::profile::record_allocate();
        let need_new = self
            .chunks
            .last()
            .map(|c| c.len() >= CHUNK_SIZE)
            .unwrap_or(false);
        if need_new {
            let mut new_chunk = Vec::with_capacity(CHUNK_SIZE);
            new_chunk.push(cell);
            let id = (self.chunks.len() * CHUNK_SIZE) as ValueId;
            self.chunks.push(new_chunk);
            id
        } else {
            let chunk_idx = self.chunks.len() - 1;
            let last = self.chunks.last_mut().unwrap();
            let offset = last.len();
            last.push(cell);
            (chunk_idx * CHUNK_SIZE + offset) as ValueId
        }
    }

    /// Allocate in the heap arena (globals/slots). Use for StoreGlobal, materialization, MakeArray, etc.
    #[inline]
    pub fn allocate_arena(&mut self, cell: ValueCell) -> ValueId {
        #[cfg(feature = "profile")]
        crate::vm::profile::record_allocate();
        self.arena.allocate(cell)
    }

    #[inline]
    pub fn get(&self, id: ValueId) -> Option<&ValueCell> {
        #[cfg(feature = "profile")]
        crate::vm::profile::record_store_get();
        if id >= ARENA_BASE {
            return self.arena.get(id);
        }
        let u = id as usize;
        let c = u / CHUNK_SIZE;
        let i = u % CHUNK_SIZE;
        self.chunks.get(c).and_then(|ch| ch.get(i))
    }

    /// Plain dict write for whole-number keys: O(1) integral map + bucket sync, no whole-map `take`/`clone`.
    pub fn plain_object_upsert_integral(
        &mut self,
        container_id: ValueId,
        canonical: i64,
        key_id: ValueId,
        value_id: ValueId,
    ) -> bool {
        let Some(ValueCell::Object(omap)) = self.get_mut(container_id) else {
            return false;
        };
        omap.upsert_integral(canonical, key_id, value_id);
        true
    }

    /// Plain dict write storing an immediate score (no new [`ValueCell`] when `value_tv` is inline).
    pub fn plain_object_upsert_integral_tagged(
        &mut self,
        container_id: ValueId,
        canonical: i64,
        key_id: ValueId,
        value_tv: crate::common::TaggedValue,
    ) -> bool {
        let Some(ValueCell::Object(omap)) = self.get_mut(container_id) else {
            return false;
        };
        omap.upsert_integral_tagged(canonical, key_id, value_tv);
        true
    }

    /// Plain set insert for whole-number keys (see [`Self::plain_object_upsert_integral`]).
    pub fn plain_set_insert_integral(
        &mut self,
        container_id: ValueId,
        canonical: i64,
        key_id: ValueId,
    ) -> bool {
        let Some(ValueCell::Set(smap)) = self.get_mut(container_id) else {
            return false;
        };
        smap.insert_integral_only(canonical, key_id)
    }

    /// Plain set `discard` for whole-number keys — no whole-set clone.
    pub fn plain_set_discard_integral(&mut self, container_id: ValueId, canonical: i64) -> bool {
        let Some(ValueCell::Set(smap)) = self.get_mut(container_id) else {
            return false;
        };
        smap.discard_integral(canonical)
    }

    #[inline]
    pub fn get_mut(&mut self, id: ValueId) -> Option<&mut ValueCell> {
        if id >= ARENA_BASE {
            return self.arena.get_mut(id);
        }
        let u = id as usize;
        let c = u / CHUNK_SIZE;
        let i = u % CHUNK_SIZE;
        self.chunks.get_mut(c).and_then(|ch| ch.get_mut(i))
    }

    pub fn len(&self) -> usize {
        let n = self.chunks.len();
        if n == 0 {
            0
        } else {
            (n - 1) * CHUNK_SIZE + self.chunks[n - 1].len()
        }
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Clear all cells and reset to initial state (single Null at index 0).
    /// Drops main chunks and arena; also clears the string pool.
    /// Used when reusing VM for stateless runs (e.g. HTTP requests).
    pub fn clear(&mut self) {
        let mut first = Vec::with_capacity(CHUNK_SIZE);
        first.push(ValueCell::Null);
        self.chunks = vec![first];
        self.string_pool.clear();
        self.arena.clear();
        self.whole_i64_cache.clear();
        self.f64_bits_cache.clear();
        self.heap_pair_free_list.clear();
        self.scratch_heap_pair = None;
        self.flat_heap_ids.clear();
    }

    #[inline]
    pub fn mark_flat_heap(&mut self, id: ValueId) {
        if id != NULL_VALUE_ID {
            self.flat_heap_ids.insert(id);
        }
    }

    #[inline]
    pub fn is_flat_heap(&self, id: ValueId) -> bool {
        self.flat_heap_ids.contains(&id)
    }

    #[inline]
    pub fn holds_scratch_heap_pair(&self, id: ValueId) -> bool {
        self.scratch_heap_pair == Some(id)
    }

    /// One persistent 2-slot array for `heappop` results (avoids millions of alloc/free in A* loops).
    pub fn scratch_heap_pair_id(&mut self, a: TaggedValue, b: TaggedValue) -> ValueId {
        if let Some(id) = self.scratch_heap_pair {
            if let Some(ValueCell::Array(slots)) = self.get_mut(id) {
                slots[0] = a;
                slots[1] = b;
                return id;
            }
        }
        let id = self.allocate(ValueCell::Array(vec![a, b]));
        self.scratch_heap_pair = Some(id);
        id
    }

    /// One [`ValueCell::Number`] per distinct `f64` bit pattern (whole ints, `inf`, `nan`, …).
    pub fn intern_number_f64(&mut self, n: f64) -> ValueId {
        if n.is_finite() && n.fract() == 0.0 {
            return self.intern_whole_i64(super::numeric::f64_trunc_to_i64_clamped(n));
        }
        let bits = n.to_bits();
        if let Some(&id) = self.f64_bits_cache.get(&bits) {
            return id;
        }
        let id = self.allocate(ValueCell::Number(n));
        self.f64_bits_cache.insert(bits, id);
        id
    }

    /// One store cell per canonical whole number (`int` / whole `number` / whole `float` key).
    pub fn intern_whole_i64(&mut self, canonical: i64) -> ValueId {
        if let Some(&id) = self.whole_i64_cache.get(&canonical) {
            return id;
        }
        let id = self.allocate(ValueCell::Number(canonical as f64));
        self.whole_i64_cache.insert(canonical, id);
        id
    }

    /// Reuse or allocate a 2-slot heap item for `heapq` `(priority, item)` pairs.
    pub fn alloc_heap_pair(&mut self, a: TaggedValue, b: TaggedValue) -> ValueId {
        while let Some(id) = self.heap_pair_free_list.pop() {
            if let Some(ValueCell::Array(slots)) = self.get_mut(id) {
                slots.clear();
                slots.push(a);
                slots.push(b);
                return id;
            }
        }
        self.allocate(ValueCell::Array(vec![a, b]))
    }

    /// True when `id` is a compact 2-slot [`ValueCell::Array`] used by `heapq` `(priority, item)` pairs.
    #[inline]
    pub fn is_recyclable_heap_pair(&self, id: ValueId) -> bool {
        if id == NULL_VALUE_ID {
            return false;
        }
        matches!(self.get(id), Some(ValueCell::Array(v)) if v.len() == 2)
    }

    /// Return a compact heap-pair cell to the free list (best-effort).
    pub fn recycle_heap_pair(&mut self, id: ValueId) {
        if id == NULL_VALUE_ID || !self.is_recyclable_heap_pair(id) {
            return;
        }
        if self.scratch_heap_pair == Some(id) {
            return;
        }
        if self.heap_pair_free_list.len() >= MAX_HEAP_PAIR_FREE {
            let _ = self.heap_pair_free_list.remove(0);
        }
        self.heap_pair_free_list.push(id);
    }
}
