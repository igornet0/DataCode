// Value storage / arena for hot path (Stage 1: kill GIL in VM).
// Executor works with ValueId; one mutable borrow of ValueStore per instruction.
// Stack, globals, frame slots are Vec<ValueId>; no Rc/RefCell in hot path.
// Strings are interned in StringPool (per ValueStore) to cut heap fragmentation and duplicate allocations.

use std::collections::HashMap;
use std::path::PathBuf;

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

/// One cell in the store; composite types refer to other cells by ValueId.
/// Heavy types (Table, Tensor, etc.) are stored in HeavyStore and referenced by index.
/// Array and Object are index-only here (no Rc<RefCell> in hot path); Value materialization
/// with Rc<RefCell<...>> happens only at native boundaries (store_convert).
/// Strings are stored as StringId (index into ValueStore's StringPool).
#[derive(Debug, Clone)]
pub enum ValueCell {
    Number(f64),
    Bool(bool),
    Null,
    String(StringId),
    /// Elements as TaggedValue: inline (number/bool/null) without allocate; heap refs as ValueId in tag.
    Array(Vec<TaggedValue>),
    Tuple(Vec<ValueId>),
    Object(HashMap<String, ValueId>),
    Function(usize),
    ModuleFunction { module_id: usize, local_index: usize },
    NativeFunction(usize),
    Path(PathBuf),
    Uuid(u64, u64),
    /// Index into HeavyStore (Table, Tensor, Image, etc.)
    Heavy(usize),
    ColumnReference { table_handle: usize, column_name: String },
    /// Layer registry id (ml::layer::LayerId)
    Layer(usize),
    Window(crate::plot::PlotWindowHandle),
    Enumerate { data_id: ValueId, start: i64 },
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
        }
    }

    /// Ensure capacity for at least `min_capacity` cells (adds chunks if needed).
    /// Call before bulk allocations (e.g. before loading a large table) to avoid repeated chunk growth.
    pub fn reserve_min(&mut self, min_capacity: usize) {
        let required_chunks = (min_capacity + CHUNK_SIZE - 1) / CHUNK_SIZE;
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
        let need_new = self.chunks.last().map(|c| c.len() >= CHUNK_SIZE).unwrap_or(false);
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

    /// Clear all cells and reset to initial state (single Null at index 0).
    /// Drops main chunks and arena; also clears the string pool.
    /// Used when reusing VM for stateless runs (e.g. HTTP requests).
    pub fn clear(&mut self) {
        let mut first = Vec::with_capacity(CHUNK_SIZE);
        first.push(ValueCell::Null);
        self.chunks = vec![first];
        self.string_pool.clear();
        self.arena.clear();
    }
}
