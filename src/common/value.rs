// Единый тип значений для VM

use crate::common::numeric::{self, FloatValue, IntValue};
use crate::common::object_map::ObjectMap;
use crate::common::set_map::SetMap;
use crate::common::table::Table;
use crate::common::value_host_types::{
    Archive, Axis, DataSource, DataSourceResponse, DatabaseCluster, DatabaseEngine, Figure,
    HttpResponse, Image, PlotWindowHandle, WebElement, WebPage,
};
use chrono::{DateTime, Duration, FixedOffset};
use crate::common::value_store::{ObjectProjectionKind, ValueId};
use crate::common::TaggedValue;
use std::cell::RefCell;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::path::PathBuf;
use std::rc::Rc;

/// Backing for [`Value::ArrayView`]: store cell or shared heap vec (zero-copy slice / chunk).
#[derive(Debug, Clone)]
pub enum ArrayViewSource {
    /// `base_id` refers to [`crate::common::value_store::ValueCell::Array`] in the VM store.
    Store { base_id: ValueId },
    /// Same `Rc` as [`Value::Array`] after materialization from store.
    Heap(Rc<RefCell<Vec<Value>>>),
}

/// Dense byte payload (e.g. `read_file_bin`): one `Vec<u8>` shared by slice views.
#[derive(Debug, Clone)]
pub struct ByteBuffer {
    pub bytes: Rc<Vec<u8>>,
    pub offset: usize,
    pub len: usize,
    /// When true, `str()` / `print()` show lowercase hex (crypto digests).
    pub display_hex: bool,
}

impl ByteBuffer {
    pub fn from_vec(v: Vec<u8>) -> Self {
        let len = v.len();
        Self {
            bytes: Rc::new(v),
            offset: 0,
            len,
            display_hex: false,
        }
    }

    /// Binary payload with hex display in `str()` / `print()` (sha256, hmac, etc.).
    pub fn from_vec_hex(v: Vec<u8>) -> Self {
        let len = v.len();
        Self {
            bytes: Rc::new(v),
            offset: 0,
            len,
            display_hex: true,
        }
    }

    /// Lowercase hex of bytes in `[offset .. offset + len)`.
    pub fn hex_string(&self) -> String {
        self.bytes[self.offset..self.offset + self.len]
            .iter()
            .map(|b| format!("{:02x}", b))
            .collect()
    }

    pub fn slice_range(&self, start: usize, end: usize) -> Option<Self> {
        if start > end || end > self.len {
            return None;
        }
        Some(Self {
            bytes: Rc::clone(&self.bytes),
            offset: self.offset + start,
            len: end - start,
            display_hex: self.display_hex,
        })
    }
}

/// Contiguous view `[offset .. offset + length)` into a backing array.
#[derive(Debug, Clone)]
pub struct ArrayViewData {
    pub source: ArrayViewSource,
    pub offset: usize,
    pub length: usize,
}

impl PartialEq for ArrayViewData {
    fn eq(&self, other: &Self) -> bool {
        self.offset == other.offset
            && self.length == other.length
            && match (&self.source, &other.source) {
                (ArrayViewSource::Store { base_id: a }, ArrayViewSource::Store { base_id: b }) => {
                    a == b
                }
                (ArrayViewSource::Heap(ra), ArrayViewSource::Heap(rb)) => Rc::ptr_eq(ra, rb),
                _ => false,
            }
    }
}

/// Состояние `stream fn` между yield (сохраняется между вызовами `next` / шагами `for`).
#[derive(Debug)]
pub struct GeneratorState {
    pub fn_index: usize,
    pub ip: usize,
    pub slots: Vec<TaggedValue>,
    pub finished: bool,
    /// Установлено только при завершении через `ereturn expr` (не входит в поток yield).
    pub final_value: Option<Value>,
    /// Первый шаг: аргументы вызова до инициализации слотов.
    pub pending_args: Option<Vec<Value>>,
    /// После `YieldAwaitInput` без полученного `.send()` / до следующего `.next()` (который подставляет `null`): нужен ввод.
    pub waiting_for_input: bool,
    /// Значение из холодного `send(v)` до первого yield-await (подставляется после первого yield).
    pub pending_first_send: Option<Value>,
    /// Первый yield при холодном `send(v)` — возвращается из `send()` при паузе на следующем yield или при `ereturn`.
    pub cold_send_first_yield: Option<Value>,
    /// После `send()` на yield-await: следующий yield (например `return x*2`) отдаётся следующему `.next()`, не `send()`.
    pub pending_deferred_yield: Option<Value>,
}

impl Clone for GeneratorState {
    fn clone(&self) -> Self {
        Self {
            fn_index: self.fn_index,
            ip: self.ip,
            slots: self.slots.clone(),
            finished: self.finished,
            final_value: self.final_value.clone(),
            pending_args: self.pending_args.clone(),
            waiting_for_input: self.waiting_for_input,
            pending_first_send: self.pending_first_send.clone(),
            cold_send_first_yield: self.cold_send_first_yield.clone(),
            pending_deferred_yield: self.pending_deferred_yield.clone(),
        }
    }
}

pub enum Value {
    Int(IntValue),
    Float(FloatValue),
    /// IEEE `f64` scalar from literals / stack / FFI (±inf, NaN); distinct from typed [`IntValue`] ∞.
    Number(f64),
    Bool(bool),
    String(String),
    Array(Rc<RefCell<Vec<Value>>>),
    Tuple(Rc<RefCell<Vec<Value>>>),
    Function(usize), // Индекс функции в массиве функций (main chunk или legacy)
    /// Функция из импортированного модуля: разрешается в момент Call через module_registry.
    /// module_uid = hash(module_path), stable across VMs so shared cached objects work.
    ModuleFunction {
        module_uid: u64,
        local_index: usize,
    },
    NativeFunction(usize), // Индекс нативной функции
    Path(PathBuf),         // Путь к файлу или директории
    Uuid(u64, u64),        // 128-bit UUID (hi, lo), value-type, ABI-friendly
    /// Calendar instant with fixed UTC offset (RFC3339 `to_rfc3339` / print).
    Date(DateTime<FixedOffset>),
    /// Signed time span (`Date ± Duration`, `Date - Date`).
    Duration(Duration),
    Table(Rc<RefCell<Table>>),
    /// Dict (bucket map) or legacy string map for classes / metadata.
    Object(Rc<RefCell<ObjectKind>>),
    /// Unique hashable values (unordered).
    Set(Rc<RefCell<SetMap>>),
    ColumnReference {
        table: Rc<RefCell<Table>>,
        column_name: String,
    },
    /// Opaque plugin-owned object (`tag` + `id`); semantics defined by the plugin (e.g. dylib).
    PluginOpaque {
        tag: u8,
        id: u64,
    },
    Window(PlotWindowHandle), // Runtime only holds WindowId - Window lives in GUI thread
    Image(Rc<RefCell<Image>>),
    Figure(Rc<RefCell<Figure>>),
    Axis(Rc<RefCell<Axis>>),
    DatabaseEngine(Rc<RefCell<DatabaseEngine>>),
    DatabaseCluster(Rc<RefCell<DatabaseCluster>>),
    Archive(Rc<RefCell<Archive>>),
    DataSource(Rc<RefCell<DataSource>>),
    DataSourceResponse(Rc<RefCell<DataSourceResponse>>),
    HttpResponse(Rc<RefCell<HttpResponse>>),
    WebPage(Rc<RefCell<WebPage>>),
    WebElement(Rc<RefCell<WebElement>>),
    Enumerate {
        data: Rc<RefCell<Vec<Value>>>,
        start: i64,
    }, // enum(iterable): lazy (idx, element) wrapper
    /// Zero-copy view; see [`ArrayViewData`].
    ArrayView(ArrayViewData),
    /// Raw bytes from a file or similar; slice with `ByteBuffer::slice_range` / VM slice ops.
    ByteBuffer(ByteBuffer),
    /// Read-only view over a plain dict's keys or value cells ([`ValueCell::ObjectFieldList`]).
    ObjectFieldList {
        source_object_id: ValueId,
        projection: ObjectProjectionKind,
        element_ids: Rc<Vec<ValueId>>,
    },
    /// Lazy functional pipeline (`map` / `filter`); single-pass iteration, no intermediate array.
    Iterable(Rc<RefCell<IterableInner>>),
    /// Результат вызова `stream fn`: ленивый генератор с фиксированным состоянием.
    Generator(Rc<RefCell<GeneratorState>>),
    Null,
    Ellipsis, // ... (e.g. Field(...) for required field)
}

/// Result of lexer `parse::<f64>()`: prefers [`Value::Int`] for finite whole numbers in `i64` range.
#[inline]
pub fn value_from_lex_number(n: f64) -> Value {
    if !n.is_finite() {
        return Value::Number(n);
    }
    if n.fract() == 0.0 && n >= i64::MIN as f64 && n <= i64::MAX as f64 {
        Value::Int(IntValue::Finite(n as i64))
    } else {
        Value::Number(n)
    }
}

/// Class / ORM metadata uses string-key [`HashMap`]; plain dicts use bucket [`ObjectMap`].
#[derive(Debug, Clone)]
pub enum ObjectKind {
    /// Class metadata, modules, ORM: string keys only.
    Legacy(HashMap<String, Value>),
    /// VM store bucket map (keys/values are [`ValueId`]).
    Bucket(ObjectMap),
    /// Materialized entries after [`crate::vm::memory::convert::load_value`] (host/tests).
    Inline(Vec<(Value, Value)>),
}

pub type ObjectHandle = Rc<RefCell<ObjectKind>>;

impl ObjectKind {
    #[inline]
    pub fn legacy(map: HashMap<String, Value>) -> Self {
        ObjectKind::Legacy(map)
    }

    #[inline]
    pub fn bucket(map: ObjectMap) -> Self {
        ObjectKind::Bucket(map)
    }

    pub fn len(&self) -> usize {
        match self {
            ObjectKind::Legacy(m) => m.len(),
            ObjectKind::Bucket(b) => b.len(),
            ObjectKind::Inline(v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        match self {
            ObjectKind::Legacy(m) => m.is_empty(),
            ObjectKind::Bucket(b) => b.is_empty(),
            ObjectKind::Inline(v) => v.is_empty(),
        }
    }

    pub fn legacy_get(&self, key: &str) -> Option<&Value> {
        match self {
            ObjectKind::Legacy(m) => m.get(key),
            ObjectKind::Bucket(_) | ObjectKind::Inline(_) => None,
        }
    }

    pub fn legacy_contains_key(&self, key: &str) -> bool {
        match self {
            ObjectKind::Legacy(m) => m.contains_key(key),
            ObjectKind::Bucket(_) | ObjectKind::Inline(_) => false,
        }
    }

    /// Lookup by full [`Value`] key (linear scan for [`ObjectKind::Inline`]).
    pub fn get_by_value_key(&self, key: &Value) -> Option<Value> {
        match self {
            ObjectKind::Legacy(m) => {
                if let Value::String(s) = key {
                    m.get(s.as_str()).cloned()
                } else {
                    None
                }
            }
            ObjectKind::Bucket(_) => None,
            ObjectKind::Inline(v) => v.iter().find(|(k, _)| k == key).map(|(_, val)| val.clone()),
        }
    }

    /// Empty plain dict (bucket-backed).
    pub fn empty_bucket() -> Self {
        ObjectKind::Bucket(ObjectMap::new())
    }

    #[inline]
    pub fn legacy_ref(&self) -> Option<&HashMap<String, Value>> {
        match self {
            ObjectKind::Legacy(m) => Some(m),
            ObjectKind::Bucket(_) | ObjectKind::Inline(_) => None,
        }
    }

    #[inline]
    pub fn legacy_mut(&mut self) -> Option<&mut HashMap<String, Value>> {
        match self {
            ObjectKind::Legacy(m) => Some(m),
            ObjectKind::Bucket(_) | ObjectKind::Inline(_) => None,
        }
    }

    /// String-key lookup (ORM / host metadata); includes [`ObjectKind::Inline`] string-key entries.
    pub fn str_key_get(&self, key: &str) -> Option<&Value> {
        match self {
            ObjectKind::Legacy(m) => m.get(key),
            ObjectKind::Inline(v) => v.iter().find_map(|(k, val)| {
                if let Value::String(sk) = k {
                    (sk.as_str() == key).then_some(val)
                } else {
                    None
                }
            }),
            ObjectKind::Bucket(_) => None,
        }
    }

    /// Upsert plain string-key field (legacy + inline-with-string-keys only).
    pub fn str_key_insert(&mut self, key: String, value: Value) -> Option<Option<Value>> {
        match self {
            ObjectKind::Legacy(m) => Some(m.insert(key, value)),
            ObjectKind::Inline(v) => {
                let idx = v
                    .iter()
                    .position(|(k, _)| matches!(k, Value::String(s) if *s == key));
                Some(if let Some(i) = idx {
                    Some(std::mem::replace(&mut v[i].1, value))
                } else {
                    v.push((Value::String(key), value));
                    None
                })
            }
            ObjectKind::Bucket(_) => None,
        }
    }

    #[inline]
    pub fn str_key_contains(&self, key: &str) -> bool {
        self.str_key_get(key).is_some()
    }

    /// Borrowed `(string_key, value)` pairs for iteration (ORM layouts).
    pub fn str_key_pairs(&self) -> Vec<(String, &Value)> {
        match self {
            ObjectKind::Legacy(m) => m.iter().map(|(k, v)| (k.clone(), v)).collect(),
            ObjectKind::Inline(v) => v
                .iter()
                .filter_map(|(k, val)| {
                    if let Value::String(sk) = k {
                        Some((sk.clone(), val))
                    } else {
                        None
                    }
                })
                .collect(),
            ObjectKind::Bucket(_) => vec![],
        }
    }

    /// Owned `(string_key, copy_of_value)` pairs (ORM iteration / INSERT builder).
    pub fn str_key_entries_cloned(&self) -> Vec<(String, Value)> {
        match self {
            ObjectKind::Legacy(m) => m.iter().map(|(k, v)| (k.clone(), v.clone())).collect(),
            ObjectKind::Inline(v) => v
                .iter()
                .filter_map(|(k, val)| match k {
                    Value::String(sk) => Some((sk.clone(), val.clone())),
                    _ => None,
                })
                .collect(),
            ObjectKind::Bucket(_) => vec![],
        }
    }
}

/// Callback reference for lazy iterators (avoids storing full [`Value`] in [`IterableInner`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CallableSlot {
    UserFunction(usize),
    NativeFunction(usize),
}

/// Backing for lazy [`IterableInner::Chunks`] (yield one row at a time in `for` / `next`).
#[derive(Debug, Clone)]
pub enum ChunkSource {
    Array(Rc<RefCell<Vec<Value>>>),
    ArrayView(ArrayViewData),
    Bytes(ByteBuffer),
}

/// Lazy iterator graph: arrays, views, `map`, `filter` (stacked via shared [`Rc`]).
#[derive(Debug)]
pub enum IterableInner {
    Array {
        array: Rc<RefCell<Vec<Value>>>,
        index: usize,
    },
    ArrayView {
        view: ArrayViewData,
        index: usize,
    },
    Map {
        source: Rc<RefCell<IterableInner>>,
        func: CallableSlot,
        fn_arity: u8,
        index: usize,
    },
    Filter {
        source: Rc<RefCell<IterableInner>>,
        pred: CallableSlot,
        fn_arity: u8,
        index: usize,
    },
    /// `enum(...)` / `for` over `Value::Enumerate`: yields `(index, element)` tuples like `get_enumerate`.
    Enumerate {
        data: Rc<RefCell<Vec<Value>>>,
        start: i64,
        index: usize,
    },
    /// `array.chunk(n)` / `view.chunk(n)`: yields each chunk as an owned array without building all chunks upfront.
    Chunks {
        source: ChunkSource,
        chunk_size: usize,
        chunk_index: usize,
    },
    /// Row-by-row iteration over [`Value::Table`] (each row as [`Value::Object`]).
    TableRows {
        table: Rc<RefCell<Table>>,
        index: usize,
    },
    /// `enum(table)` / lazy zip: yields `(start + n, element)` by wrapping any inner lazy iterator.
    EnumerateIter {
        source: Rc<RefCell<IterableInner>>,
        start: i64,
        next_index: usize,
    },
    /// `stream fn` / [`Value::Generator`]: один проход через [`crate::vm::generator::run_generator_next`].
    StreamGenerator {
        state: Rc<RefCell<GeneratorState>>,
    },
    /// `for x in obj.keys` / `obj.values`: yields loaded values from snapshot cell ids.
    ObjectFieldList {
        element_ids: Rc<Vec<ValueId>>,
        index: usize,
    },
    /// `for x in set`: yields elements; detects mutation via [`SetMap::generation`].
    Set {
        set: Rc<RefCell<SetMap>>,
        element_ids: Rc<Vec<ValueId>>,
        index: usize,
        start_generation: u64,
    },
    /// `for ch in str`: yields each Unicode code point as a single-character string.
    String {
        text: String,
        index: usize,
    },
    /// Lazy `range(start, end, step)` — O(1) memory; yields [`Value::Number`] per element.
    Range {
        current: i64,
        end: i64,
        step: i64,
    },
    /// Class instance with `@iter` / `@next` protocol.
    SpecialInstance {
        receiver_id: ValueId,
    },
}

impl Clone for IterableInner {
    fn clone(&self) -> Self {
        match self {
            Self::Array { array, .. } => Self::Array {
                array: array.clone(),
                index: 0,
            },
            Self::ArrayView { view, .. } => Self::ArrayView {
                view: view.clone(),
                index: 0,
            },
            Self::Map {
                source,
                func,
                fn_arity,
                ..
            } => Self::Map {
                source: Rc::new(RefCell::new(source.borrow().clone())),
                func: func.clone(),
                fn_arity: *fn_arity,
                index: 0,
            },
            Self::Filter {
                source,
                pred,
                fn_arity,
                ..
            } => Self::Filter {
                source: Rc::new(RefCell::new(source.borrow().clone())),
                pred: pred.clone(),
                fn_arity: *fn_arity,
                index: 0,
            },
            Self::Enumerate { data, start, .. } => Self::Enumerate {
                data: data.clone(),
                start: *start,
                index: 0,
            },
            Self::Chunks {
                source, chunk_size, ..
            } => Self::Chunks {
                source: source.clone(),
                chunk_size: *chunk_size,
                chunk_index: 0,
            },
            Self::TableRows { table, .. } => Self::TableRows {
                table: table.clone(),
                index: 0,
            },
            Self::EnumerateIter { source, start, .. } => Self::EnumerateIter {
                source: Rc::new(RefCell::new(source.borrow().clone())),
                start: *start,
                next_index: 0,
            },
            Self::StreamGenerator { state } => Self::StreamGenerator {
                state: Rc::clone(state),
            },
            Self::ObjectFieldList { element_ids, .. } => Self::ObjectFieldList {
                element_ids: Rc::clone(element_ids),
                index: 0,
            },
            Self::Set {
                set,
                element_ids,
                start_generation,
                ..
            } => Self::Set {
                set: Rc::clone(set),
                element_ids: Rc::clone(element_ids),
                index: 0,
                start_generation: *start_generation,
            },
            Self::String { text, .. } => Self::String {
                text: text.clone(),
                index: 0,
            },
            Self::Range {
                current,
                end,
                step,
            } => Self::Range {
                current: *current,
                end: *end,
                step: *step,
            },
            Self::SpecialInstance { receiver_id } => Self::SpecialInstance {
                receiver_id: *receiver_id,
            },
        }
    }
}

impl std::fmt::Debug for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Value::Object(map_rc) => {
                let kind = map_rc.borrow();
                match &*kind {
                    ObjectKind::Legacy(map) => {
                        if map
                            .get("__meta")
                            .and_then(|v| {
                                if let Value::Bool(b) = v {
                                    Some(*b)
                                } else {
                                    None
                                }
                            })
                            .unwrap_or(false)
                        {
                            let schema = map
                                .get("schema")
                                .map(|v| v.to_string())
                                .unwrap_or_else(|| "?".to_string());
                            return write!(f, "Object(<metadata: schema={}>)", schema);
                        }
                        if map
                            .get("__create_all")
                            .and_then(|v| {
                                if let Value::Bool(b) = v {
                                    Some(*b)
                                } else {
                                    None
                                }
                            })
                            .unwrap_or(false)
                        {
                            return write!(f, "Object(<create_all>)");
                        }
                        if map
                            .get("__column")
                            .and_then(|v| {
                                if let Value::Bool(b) = v {
                                    Some(*b)
                                } else {
                                    None
                                }
                            })
                            .unwrap_or(false)
                        {
                            return write!(f, "Object(<column>)");
                        }
                        if map_is_sqenum_member_legacy(map) {
                            let name = map
                                .get(crate::database_engine::sqenum::KEY_ENUM_NAME)
                                .map(|v| v.to_string())
                                .unwrap_or_else(|| "?".to_string());
                            return write!(f, "Object(<enum_member {}>)", name);
                        }
                        let sqenumish = map
                            .get(crate::database_engine::sqenum::KEY_SQENUM)
                            .and_then(|v| {
                                if let Value::Bool(b) = v {
                                    Some(*b)
                                } else {
                                    None
                                }
                            })
                            .unwrap_or(false)
                            || map
                                .get(crate::database_engine::sqenum::KEY_EXTENDS_SQENUM)
                                .and_then(|v| {
                                    if let Value::Bool(b) = v {
                                        Some(*b)
                                    } else {
                                        None
                                    }
                                })
                                .unwrap_or(false)
                            || map
                                .get(crate::database_engine::sqenum::KEY_BUILTIN_SQENUM)
                                .and_then(|v| {
                                    if let Value::Bool(b) = v {
                                        Some(*b)
                                    } else {
                                        None
                                    }
                                })
                                .unwrap_or(false);
                        if sqenumish {
                            let cn = map
                                .get("__class_name")
                                .map(|v| v.to_string())
                                .unwrap_or_else(|| "?".to_string());
                            return write!(f, "Object(<SQLEnum {}>)", cn);
                        }
                        f.debug_map().entries(map.iter()).finish()
                    }
                    ObjectKind::Bucket(b) => write!(
                        f,
                        "Object(<dict buckets={} entries={} frozen={}>)",
                        b.buckets_ref().len(),
                        b.len(),
                        b.is_frozen()
                    ),
                    ObjectKind::Inline(entries) => f
                        .debug_struct("Object")
                        .field("inline_entries", &entries.len())
                        .finish_non_exhaustive(),
                }
            }
            _ => match self {
                Value::Number(n) => write!(f, "Number({:?})", n),
                Value::Int(i) => write!(f, "{:?}", i),
                Value::Float(v) => write!(f, "{:?}", v),
                Value::Bool(b) => std::fmt::Debug::fmt(b, f),
                Value::String(s) => std::fmt::Debug::fmt(s, f),
                Value::Array(arr) => f.debug_tuple("Array").field(&arr.borrow()).finish(),
                Value::Tuple(tup) => f.debug_tuple("Tuple").field(&tup.borrow()).finish(),
                Value::Function(i) => f.debug_tuple("Function").field(i).finish(),
                Value::ModuleFunction {
                    module_uid,
                    local_index,
                } => f
                    .debug_struct("ModuleFunction")
                    .field("module_uid", module_uid)
                    .field("local_index", local_index)
                    .finish(),
                Value::NativeFunction(i) => f.debug_tuple("NativeFunction").field(i).finish(),
                Value::Path(p) => f.debug_tuple("Path").field(p).finish(),
                Value::Uuid(hi, lo) => f.debug_tuple("Uuid").field(hi).field(lo).finish(),
                Value::Date(d) => write!(f, "Date({})", d.to_rfc3339()),
                Value::Duration(d) => write!(
                    f,
                    "Duration({}.{:09}s)",
                    d.num_seconds(),
                    d.subsec_nanos()
                ),
                Value::Table(t) => f.debug_tuple("Table").field(&t.borrow()).finish(),
                Value::Set(s) => f
                    .debug_struct("Set")
                    .field("len", &s.borrow().len())
                    .finish(),
                Value::Object(_) => unreachable!(),
                Value::ColumnReference { table, column_name } => f
                    .debug_struct("ColumnReference")
                    .field("table", table)
                    .field("column_name", column_name)
                    .finish(),
                Value::PluginOpaque { tag, id } => f
                    .debug_struct("PluginOpaque")
                    .field("tag", tag)
                    .field("id", id)
                    .finish(),
                Value::Window(h) => f.debug_tuple("Window").field(h).finish(),
                Value::Image(img) => f.debug_tuple("Image").field(&img.borrow()).finish(),
                Value::Figure(fig) => f.debug_tuple("Figure").field(&fig.borrow()).finish(),
                Value::Axis(ax) => f.debug_tuple("Axis").field(&ax.borrow()).finish(),
                Value::DatabaseEngine(e) => {
                    f.debug_tuple("DatabaseEngine").field(&e.borrow()).finish()
                }
                Value::DatabaseCluster(c) => {
                    f.debug_tuple("DatabaseCluster").field(&c.borrow()).finish()
                }
                Value::Archive(a) => f.debug_tuple("Archive").field(&a.borrow()).finish(),
                Value::DataSource(_) => f.debug_tuple("DataSource").finish_non_exhaustive(),
                Value::DataSourceResponse(_) => {
                    f.debug_tuple("DataSourceResponse").finish_non_exhaustive()
                }
                Value::HttpResponse(_) => f.debug_tuple("HttpResponse").finish_non_exhaustive(),
                Value::WebPage(p) => f
                    .debug_struct("WebPage")
                    .field("id", &p.borrow().id)
                    .field("closed", &p.borrow().closed)
                    .finish(),
                Value::WebElement(e) => f
                    .debug_struct("WebElement")
                    .field("selector", &e.borrow().selector)
                    .finish(),
                Value::Enumerate { data, start } => f
                    .debug_struct("Enumerate")
                    .field("data", &data.borrow())
                    .field("start", start)
                    .finish(),
                Value::ArrayView(av) => f
                    .debug_struct("ArrayView")
                    .field("offset", &av.offset)
                    .field("length", &av.length)
                    .finish_non_exhaustive(),
                Value::Iterable(_) => write!(f, "Iterable(<lazy>)"),
                Value::Generator(g) => f.debug_tuple("Generator").field(&Rc::as_ptr(g)).finish(),
                Value::ByteBuffer(b) => f
                    .debug_struct("ByteBuffer")
                    .field("len", &b.len)
                    .finish_non_exhaustive(),
                Value::ObjectFieldList {
                    projection,
                    element_ids,
                    ..
                } => f
                    .debug_struct("ObjectFieldList")
                    .field("projection", projection)
                    .field("len", &element_ids.len())
                    .finish_non_exhaustive(),
                Value::Null => write!(f, "Null"),
                Value::Ellipsis => write!(f, "Ellipsis"),
            },
        }
    }
}

fn inline_dict_pairs_eq(a: &[(Value, Value)], b: &[(Value, Value)]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut used = vec![false; b.len()];
    for (ka, va) in a {
        let mut matched = false;
        for (j, (kb, vb)) in b.iter().enumerate() {
            if used[j] {
                continue;
            }
            if ka == kb && va == vb {
                used[j] = true;
                matched = true;
                break;
            }
        }
        if !matched {
            return false;
        }
    }
    true
}

fn map_is_sqenum_member_legacy(m: &HashMap<String, Value>) -> bool {
    m.get("__enum_member")
        .and_then(|v| {
            if let Value::Bool(b) = v {
                Some(*b)
            } else {
                None
            }
        })
        .unwrap_or(false)
}

/// Stored scalar (`__enum_value`) for SQLEnum members stored as [`ObjectKind::Legacy`] or [`ObjectKind::Inline`]
/// (`ObjectKind::Bucket` has no [`ObjectKind::str_key_get`]).
fn sqenum_member_stored_value_ref(kind: &ObjectKind) -> Option<&Value> {
    match kind.str_key_get(crate::database_engine::sqenum::KEY_ENUM_MEMBER)? {
        Value::Bool(true) => kind.str_key_get(crate::database_engine::sqenum::KEY_ENUM_VALUE),
        _ => None,
    }
}

fn enum_stored_eq_scalar(stored: &Value, other: &Value) -> bool {
    match (stored, other) {
        (Value::String(a), Value::String(b)) => a == b,
        (Value::Int(ai), Value::Int(bi)) => ai == bi,
        (Value::Float(af), Value::Float(bf)) => *af == *bf,
        (Value::Int(i), Value::Float(f)) | (Value::Float(f), Value::Int(i)) => {
            numeric::numeric_eq_int_float(*i, *f)
        }
        (Value::String(a), v) => a
            .parse::<f64>()
            .ok()
            .map(|x| value_from_lex_number(x) == *v || Value::Number(x) == *v)
            .unwrap_or(false),
        (v, Value::String(a)) => a
            .parse::<f64>()
            .ok()
            .map(|x| value_from_lex_number(x) == *v || Value::Number(x) == *v)
            .unwrap_or(false),
        _ => false,
    }
}

/// SQLEnum member vs member / vs scalar (`UserRole.ADMIN == "admin"`).
///
/// Handles [`ObjectKind::Legacy`] and [`ObjectKind::Inline`] (bucket materialization from VM store).
fn try_sqenum_member_eq(a: &Value, b: &Value) -> Option<bool> {
    match (a, b) {
        (Value::Object(oa), Value::Object(ob)) => {
            let ak = oa.borrow();
            let bk = ob.borrow();
            let a_sv = sqenum_member_stored_value_ref(&*ak);
            let b_sv = sqenum_member_stored_value_ref(&*bk);
            let a_mem = a_sv.is_some();
            let b_mem = b_sv.is_some();
            if let (Some(v1), Some(v2)) = (a_sv, b_sv) {
                let c1 = ak.str_key_get(crate::database_engine::sqenum::KEY_ENUM_CLASS).and_then(
                    |v| {
                        if let Value::Object(o) = v {
                            Some(o)
                        } else {
                            None
                        }
                    },
                );
                let c2 = bk.str_key_get(crate::database_engine::sqenum::KEY_ENUM_CLASS).and_then(
                    |v| {
                        if let Value::Object(o) = v {
                            Some(o)
                        } else {
                            None
                        }
                    },
                );
                if let (Some(c1), Some(c2)) = (c1, c2) {
                    return Some(Rc::ptr_eq(c1, c2) && v1 == v2);
                }
                return Some(false);
            }
            if a_mem || b_mem {
                return Some(false);
            }
            None
        }
        (Value::Object(oa), other) => {
            let ak = oa.borrow();
            if let Some(stored) = sqenum_member_stored_value_ref(&*ak) {
                return Some(enum_stored_eq_scalar(stored, other));
            }
            None
        }
        (other, Value::Object(ob)) => {
            let bk = ob.borrow();
            if let Some(stored) = sqenum_member_stored_value_ref(&*bk) {
                return Some(enum_stored_eq_scalar(stored, other));
            }
            None
        }
        _ => None,
    }
}

impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        if let (Value::Object(a), Value::Object(b)) = (self, other) {
            if Rc::ptr_eq(a, b) {
                return true;
            }
        }
        if let Some(r) = try_sqenum_member_eq(self, other) {
            return r;
        }
        match (self, other) {
            (Value::Number(a), Value::Number(b)) => a == b,
            (Value::Number(a), Value::Int(i)) | (Value::Int(i), Value::Number(a)) => {
                numeric::numeric_eq_int_float(*i, FloatValue::from_f64_for_stack(*a))
            }
            (Value::Number(a), Value::Float(f)) | (Value::Float(f), Value::Number(a)) => {
                FloatValue::from_f64_for_stack(*a) == *f
            }
            (Value::Int(a), Value::Int(b)) => a == b,
            (Value::Float(a), Value::Float(b)) => *a == *b,
            (Value::Int(i), Value::Float(f)) | (Value::Float(f), Value::Int(i)) => {
                numeric::numeric_eq_int_float(*i, *f)
            }
            (Value::Bool(a), Value::Bool(b)) => a == b,
            (Value::String(a), Value::String(b)) => a == b,
            (Value::Array(a), Value::Array(b)) => *a.borrow() == *b.borrow(),
            (Value::ArrayView(a), Value::ArrayView(b)) => a == b,
            (Value::Tuple(a), Value::Tuple(b)) => *a.borrow() == *b.borrow(),
            (Value::Function(a), Value::Function(b)) => a == b,
            (
                Value::ModuleFunction {
                    module_uid: a_uid,
                    local_index: a_li,
                },
                Value::ModuleFunction {
                    module_uid: b_uid,
                    local_index: b_li,
                },
            ) => a_uid == b_uid && a_li == b_li,
            (Value::NativeFunction(a), Value::NativeFunction(b)) => a == b,
            (Value::Path(a), Value::Path(b)) => a == b,
            (Value::Uuid(hi_a, lo_a), Value::Uuid(hi_b, lo_b)) => hi_a == hi_b && lo_a == lo_b,
            (Value::Date(a), Value::Date(b)) => a == b,
            (Value::Duration(a), Value::Duration(b)) => a == b,
            (Value::Table(a), Value::Table(b)) => *a.borrow() == *b.borrow(),
            (Value::Set(a), Value::Set(b)) => Rc::ptr_eq(a, b),
            (Value::Object(a), Value::Object(b)) => match (&*a.borrow(), &*b.borrow()) {
                (ObjectKind::Legacy(am), ObjectKind::Legacy(bm)) => {
                    let a_sqenum = am
                        .get("__sqenum")
                        .and_then(|v| {
                            if let Value::Bool(x) = v {
                                Some(*x)
                            } else {
                                None
                            }
                        })
                        .unwrap_or(false);
                    let b_sqenum = bm
                        .get("__sqenum")
                        .and_then(|v| {
                            if let Value::Bool(x) = v {
                                Some(*x)
                            } else {
                                None
                            }
                        })
                        .unwrap_or(false);
                    if a_sqenum && b_sqenum {
                        std::rc::Rc::ptr_eq(a, b)
                    } else {
                        let a_ext_sqenum = am
                            .get(crate::database_engine::sqenum::KEY_EXTENDS_SQENUM)
                            .and_then(|v| {
                                if let Value::Bool(x) = v {
                                    Some(*x)
                                } else {
                                    None
                                }
                            })
                            .unwrap_or(false);
                        let b_ext_sqenum = bm
                            .get(crate::database_engine::sqenum::KEY_EXTENDS_SQENUM)
                            .and_then(|v| {
                                if let Value::Bool(x) = v {
                                    Some(*x)
                                } else {
                                    None
                                }
                            })
                            .unwrap_or(false);
                        if a_ext_sqenum && b_ext_sqenum {
                            std::rc::Rc::ptr_eq(a, b)
                        } else {
                            let a_lookup = am
                                .get("__sqenum_by_value_lookup")
                                .and_then(|v| {
                                    if let Value::Bool(x) = v {
                                        Some(*x)
                                    } else {
                                        None
                                    }
                                })
                                .unwrap_or(false);
                            let b_lookup = bm
                                .get("__sqenum_by_value_lookup")
                                .and_then(|v| {
                                    if let Value::Bool(x) = v {
                                        Some(*x)
                                    } else {
                                        None
                                    }
                                })
                                .unwrap_or(false);
                            if a_lookup && b_lookup {
                                std::rc::Rc::ptr_eq(a, b)
                            } else {
                                let a_meta = am
                                    .get("__meta")
                                    .and_then(|v| {
                                        if let Value::Bool(x) = v {
                                            Some(*x)
                                        } else {
                                            None
                                        }
                                    })
                                    .unwrap_or(false);
                                let a_create_all = am
                                    .get("__create_all")
                                    .and_then(|v| {
                                        if let Value::Bool(x) = v {
                                            Some(*x)
                                        } else {
                                            None
                                        }
                                    })
                                    .unwrap_or(false);
                                let b_meta = bm
                                    .get("__meta")
                                    .and_then(|v| {
                                        if let Value::Bool(x) = v {
                                            Some(*x)
                                        } else {
                                            None
                                        }
                                    })
                                    .unwrap_or(false);
                                let b_create_all = bm
                                    .get("__create_all")
                                    .and_then(|v| {
                                        if let Value::Bool(x) = v {
                                            Some(*x)
                                        } else {
                                            None
                                        }
                                    })
                                    .unwrap_or(false);
                                if (a_meta || a_create_all) && (b_meta || b_create_all) {
                                    std::rc::Rc::ptr_eq(a, b)
                                } else {
                                    *am == *bm
                                }
                            }
                        }
                    }
                }
                (ObjectKind::Bucket(_), ObjectKind::Bucket(_)) => std::rc::Rc::ptr_eq(a, b),
                (ObjectKind::Inline(ai), ObjectKind::Inline(bi)) => inline_dict_pairs_eq(ai, bi),
                _ => false,
            },
            (
                Value::ColumnReference {
                    table: a,
                    column_name: col_a,
                },
                Value::ColumnReference {
                    table: b,
                    column_name: col_b,
                },
            ) => Rc::ptr_eq(a, b) && col_a == col_b,
            (Value::PluginOpaque { tag: ta, id: ia }, Value::PluginOpaque { tag: tb, id: ib }) => {
                ta == tb && ia == ib
            }
            (Value::Window(a), Value::Window(b)) => a.id == b.id,
            (Value::Image(a), Value::Image(b)) => Rc::ptr_eq(a, b),
            (Value::Figure(a), Value::Figure(b)) => Rc::ptr_eq(a, b),
            (Value::Axis(a), Value::Axis(b)) => Rc::ptr_eq(a, b),
            (Value::DatabaseEngine(a), Value::DatabaseEngine(b)) => Rc::ptr_eq(a, b),
            (Value::DatabaseCluster(a), Value::DatabaseCluster(b)) => Rc::ptr_eq(a, b),
            (Value::Archive(a), Value::Archive(b)) => Rc::ptr_eq(a, b),
            (Value::DataSource(a), Value::DataSource(b)) => Rc::ptr_eq(a, b),
            (Value::DataSourceResponse(a), Value::DataSourceResponse(b)) => Rc::ptr_eq(a, b),
            (Value::HttpResponse(a), Value::HttpResponse(b)) => Rc::ptr_eq(a, b),
            (Value::WebPage(a), Value::WebPage(b)) => Rc::ptr_eq(a, b),
            (Value::WebElement(a), Value::WebElement(b)) => Rc::ptr_eq(a, b),
            (Value::Enumerate { data: a, start: sa }, Value::Enumerate { data: b, start: sb }) => {
                Rc::ptr_eq(a, b) && sa == sb
            }
            (Value::Iterable(a), Value::Iterable(b)) => Rc::ptr_eq(a, b),
            (Value::Generator(a), Value::Generator(b)) => Rc::ptr_eq(a, b),
            (Value::ByteBuffer(a), Value::ByteBuffer(b)) => {
                a.len == b.len && a.bytes.as_ptr() == b.bytes.as_ptr() && a.offset == b.offset
            }
            (
                Value::ObjectFieldList {
                    source_object_id: sa,
                    projection: pa,
                    element_ids: ea,
                },
                Value::ObjectFieldList {
                    source_object_id: sb,
                    projection: pb,
                    element_ids: eb,
                },
            ) => sa == sb && pa == pb && ea.as_ref() == eb.as_ref(),
            (Value::Null, Value::Null) => true,
            (Value::Ellipsis, Value::Ellipsis) => true,
            _ => false,
        }
    }
}

impl Value {
    /// Back-compat constructor: same semantics as [`value_from_lex_number`].
    #[inline]
    pub fn number(n: f64) -> Self {
        Value::Number(n)
    }

    /// Extract a finite IEEE `f64` if this value represents a finite real number (int or float domain).
    #[inline]
    pub fn as_finite_f64(&self) -> Option<f64> {
        match self {
            Value::Number(n) => n.is_finite().then_some(*n),
            Value::Int(IntValue::Finite(n)) => Some(*n as f64),
            Value::Float(FloatValue::Finite(f)) => Some(*f),
            _ => None,
        }
    }

    /// Coerce numeric value to IEEE `f64` including ±inf; `NaN` only on float-domain NaN.
    #[inline]
    pub fn as_ieee_f64(&self) -> Option<f64> {
        match self {
            Value::Number(n) => Some(*n),
            Value::Int(i) => Some(match *i {
                IntValue::Finite(n) => n as f64,
                IntValue::PosInfinity => f64::INFINITY,
                IntValue::NegInfinity => f64::NEG_INFINITY,
            }),
            Value::Float(f) => Some((*f).as_raw_f64()),
            _ => None,
        }
    }

    /// String-key namespace map (modules, classes, host metadata).
    #[inline]
    pub fn legacy_object(map: HashMap<String, Value>) -> Self {
        Value::Object(Rc::new(RefCell::new(ObjectKind::Legacy(map))))
    }

    /// Hashable values for caches and stable keys (includes tuples of hashables; see [`crate::common::type_model`]).
    pub fn is_hashable(&self) -> bool {
        crate::common::type_model::is_hashable_value(self)
    }

    pub fn is_truthy(&self) -> bool {
        match self {
            Value::Null => false,
            Value::Bool(false) => false,
            Value::Number(n) => *n != 0.0,
            Value::Int(i) => match *i {
                IntValue::Finite(n) => n != 0,
                IntValue::PosInfinity | IntValue::NegInfinity => true,
            },
            Value::Float(f) => match *f {
                FloatValue::Finite(n) => n != 0.0,
                FloatValue::NaN | FloatValue::PosInfinity | FloatValue::NegInfinity => true,
            },
            Value::String(s) => !s.is_empty(), // Пустая строка = false
            Value::Array(arr) => !arr.borrow().is_empty(),
            Value::ArrayView(av) => av.length > 0,
            Value::Tuple(tuple) => !tuple.borrow().is_empty(),
            Value::Path(p) => !p.as_os_str().is_empty(), // Путь не пустой = true
            Value::Uuid(_, _) => true,                   // UUID всегда truthy
            Value::Date(_) => true,
            Value::Duration(d) => !d.is_zero(),
            Value::Table(table) => table.borrow().len() > 0, // Таблица не пустая = true
            Value::Object(map_rc) => !map_rc.borrow().is_empty(),
            Value::ColumnReference { table, column_name } => {
                if let Some(column) = table.borrow_mut().get_column(column_name) {
                    !column.is_empty()
                } else {
                    false
                }
            }
            Value::PluginOpaque { .. } => true,
            Value::Window(_) => true,
            Value::Image(_) => true,
            Value::Figure(_) => true,
            Value::Axis(_) => true,
            Value::DatabaseEngine(_) => true,
            Value::DatabaseCluster(c) => !c.borrow().connections.is_empty(),
            Value::Archive(_) => true,
            Value::DataSource(_) => true,
            Value::DataSourceResponse(r) => !r.borrow().body.is_empty(),
            Value::HttpResponse(r) => r.borrow().ok,
            Value::WebPage(p) => !p.borrow().closed,
            Value::WebElement(_) => true,
            Value::Enumerate { data, .. } => !data.borrow().is_empty(),
            Value::Iterable(_) => true,
            Value::ByteBuffer(b) => b.len > 0,
            Value::Set(s) => !s.borrow().is_empty(),
            Value::Ellipsis => true,
            _ => true,
        }
    }

    pub fn to_string(&self) -> String {
        match self {
            Value::Number(n) => {
                if n.fract() == 0.0 && n.is_finite() && n.abs() <= i64::MAX as f64 {
                    format!("{}", *n as i64)
                } else {
                    format!("{}", n)
                }
            }
            Value::Int(i) => i.to_display_string(),
            Value::Float(n) => n.to_display_string(),
            Value::Bool(b) => format!("{}", b),
            Value::String(s) => s.clone(),
            Value::Array(arr) => {
                let arr_ref = arr.borrow();
                let elements: Vec<String> = arr_ref.iter().map(|v| v.to_string()).collect();
                format!("[{}]", elements.join(", "))
            }
            Value::ArrayView(av) => {
                format!("<array view len={}>", av.length)
            }
            Value::ByteBuffer(b) if b.display_hex => b.hex_string(),
            Value::ByteBuffer(b) => {
                format!("<bytes len={}>", b.len)
            }
            Value::ObjectFieldList { element_ids, .. } => {
                format!("<dict field view len={}>", element_ids.len())
            }
            Value::Tuple(tuple) => {
                let tuple_ref = tuple.borrow();
                let elements: Vec<String> = tuple_ref.iter().map(|v| v.to_string()).collect();
                format!("({})", elements.join(", "))
            }
            Value::Function(_) | Value::ModuleFunction { .. } => "<function>".to_string(),
            Value::NativeFunction(_) => "<native function>".to_string(),
            Value::Path(p) => {
                if crate::dcp::dcp_vfs_active() {
                    if let Ok(key) = crate::dcp::normalize_vfs_path(p) {
                        return crate::dcp::format_logical_path(&key);
                    }
                }
                // В режиме --use-ve показываем относительные пути
                use crate::websocket::{get_use_ve, get_user_session_path};
                if get_use_ve() {
                    if let Some(session_path) = get_user_session_path() {
                        // Канонизируем оба пути для корректного сравнения
                        let canonical_session =
                            session_path.canonicalize().ok().unwrap_or(session_path);
                        let canonical_path = p.canonicalize().ok().unwrap_or(p.clone());

                        // Проверяем, начинается ли путь с пути сессии
                        if let Ok(stripped) = canonical_path.strip_prefix(&canonical_session) {
                            // Формируем относительный путь с префиксом ./
                            let relative = stripped.to_string_lossy().to_string();
                            if relative.is_empty() || relative == "." {
                                "./".to_string()
                            } else {
                                // Убираем начальные слеши и добавляем ./
                                let trimmed = relative.trim_start_matches(['/', '\\']);
                                if trimmed.is_empty() {
                                    "./".to_string()
                                } else {
                                    format!("./{}", trimmed)
                                }
                            }
                        } else {
                            // Путь вне сессии - возвращаем как есть (не канонизированный для сохранения оригинального формата)
                            p.to_string_lossy().to_string()
                        }
                    } else {
                        // Нет пути сессии - возвращаем как есть
                        p.to_string_lossy().to_string()
                    }
                } else {
                    // Не режим --use-ve - возвращаем полный путь
                    p.to_string_lossy().to_string()
                }
            }
            Value::Date(d) => d.to_rfc3339(),
            Value::Duration(d) => format!(
                "{}s",
                d.num_seconds() as f64 + d.subsec_nanos() as f64 * 1e-9
            ),
            Value::Table(table) => {
                let t = table.borrow();
                format!("<table: {} rows, {} columns>", t.len(), t.column_count())
            }
            Value::ColumnReference { table, column_name } => {
                let mut t = table.borrow_mut();
                let name: String = t
                    .name
                    .as_ref()
                    .map(|n| n.as_str())
                    .unwrap_or("table")
                    .to_string();
                if let Some(column) = t.get_column(column_name) {
                    format!(
                        "<column: {}.{} ({} values)>",
                        name,
                        column_name,
                        column.len()
                    )
                } else {
                    format!(
                        "<column: {}.{} (not found)>",
                        t.name.as_ref().map(|n| n.as_str()).unwrap_or("table"),
                        column_name
                    )
                }
            }
            Value::Set(s) => format!("set(<{} elements>)", s.borrow().len()),
            Value::Object(map_rc) => {
                if let Some(s) = crate::vm::special_methods::try_instance_string(self) {
                    return s;
                }
                let kind = map_rc.borrow();
                if let Some(v) = sqenum_member_stored_value_ref(&*kind) {
                    return v.to_string();
                }
                match &*kind {
                    ObjectKind::Legacy(map) => {
                        if map
                            .get("__meta")
                            .and_then(|v| {
                                if let Value::Bool(b) = v {
                                    Some(*b)
                                } else {
                                    None
                                }
                            })
                            .unwrap_or(false)
                        {
                            return format!(
                                "<metadata: schema={}>",
                                map.get("schema")
                                    .map(|v| v.to_string())
                                    .unwrap_or_else(|| "?".to_string())
                            );
                        }
                        let pairs: Vec<String> = map
                            .iter()
                            .map(|(k, v)| format!("\"{}\": {}", k, v.to_string()))
                            .collect();
                        format!("{{{}}}", pairs.join(", "))
                    }
                    ObjectKind::Bucket(b) => format!(
                        "<dict entries={} frozen={}>",
                        b.len(),
                        b.is_frozen()
                    ),
                    ObjectKind::Inline(entries) => {
                        let pairs: Vec<String> = entries
                            .iter()
                            .map(|(k, v)| format!("{}: {}", k.to_string(), v.to_string()))
                            .collect();
                        format!("{{{}}}", pairs.join(", "))
                    }
                }
            }
            Value::PluginOpaque { tag, id } => {
                format!("<plugin_opaque tag={} id={}>", tag, id)
            }
            Value::Window(handle) => {
                format!("<window: id={:?}>", handle.id)
            }
            Value::Image(image) => {
                let img = image.borrow();
                format!("<image: {}x{}>", img.width, img.height)
            }
            Value::Figure(figure) => {
                let fig = figure.borrow();
                format!(
                    "<figure: {}x{} axes, figsize=({}, {})>",
                    fig.axes.len(),
                    if !fig.axes.is_empty() {
                        fig.axes[0].len()
                    } else {
                        0
                    },
                    fig.figsize.0,
                    fig.figsize.1
                )
            }
            Value::Axis(_) => {
                format!("<axis>")
            }
            Value::Enumerate { .. } => "<enumerate>".to_string(),
            Value::Iterable(rc) => {
                let ptr = Rc::as_ptr(rc);
                let kind = match &*rc.borrow() {
                    IterableInner::Map { .. } => "map",
                    IterableInner::Filter { .. } => "filter",
                    IterableInner::Enumerate { .. } => "enumerate",
                    IterableInner::Chunks { .. } => "chunk",
                    IterableInner::TableRows { .. } => "table_rows",
                    IterableInner::EnumerateIter { .. } => "enumerate_iter",
                    IterableInner::Array { .. } | IterableInner::ArrayView { .. } => "iterable",
                    IterableInner::StreamGenerator { .. } => "stream_generator",
                    IterableInner::ObjectFieldList { .. } => "dict_field_view",
                    IterableInner::Set { .. } => "set_iter",
                    IterableInner::String { .. } => "str_iter",
                    IterableInner::Range { .. } => "range",
                    IterableInner::SpecialInstance { .. } => "special_instance",
                };
                format!("<{} object at {:p}>", kind, ptr)
            }
            Value::Generator(g) => {
                format!("<generator at {:p}>", Rc::as_ptr(g))
            }
            Value::DatabaseEngine(engine) => {
                let e = engine.borrow();
                format!("<database_engine: {}>", e.url)
            }
            Value::DatabaseCluster(c) => {
                let n = c.borrow().connections.len();
                format!("<database_cluster: {} connections>", n)
            }
            Value::Archive(a) => {
                let arch = a.borrow();
                format!(
                    "<archive: {} ({})>",
                    arch.path.display(),
                    arch.format.as_str()
                )
            }
            Value::DataSource(ds) => {
                let d = ds.borrow();
                format!("<datasource: {}>", d.connector_type())
            }
            Value::DataSourceResponse(r) => {
                let resp = r.borrow();
                format!("<response: {} {}>", resp.status, resp.url)
            }
            Value::HttpResponse(r) => {
                let resp = r.borrow();
                format!("<http_response: {} {}>", resp.status, resp.url)
            }
            Value::WebPage(p) => {
                let page = p.borrow();
                format!("<web_page: id={} closed={}>", page.id, page.closed)
            }
            Value::WebElement(e) => {
                let el = e.borrow();
                format!("<web_element: {}>", el.selector)
            }
            Value::Uuid(hi, lo) => {
                let hi_b = hi.to_be_bytes();
                let lo_b = lo.to_be_bytes();
                format!(
                    "{:08x}-{:04x}-{:04x}-{:02x}{:02x}-{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}",
                    u32::from_be_bytes([hi_b[0], hi_b[1], hi_b[2], hi_b[3]]),
                    u16::from_be_bytes([hi_b[4], hi_b[5]]),
                    u16::from_be_bytes([hi_b[6], hi_b[7]]),
                    lo_b[0],
                    lo_b[1],
                    lo_b[2],
                    lo_b[3],
                    lo_b[4],
                    lo_b[5],
                    lo_b[6],
                    lo_b[7]
                )
            }
            Value::Null => "null".to_string(),
            Value::Ellipsis => "...".to_string(),
        }
    }
}

// Реализуем Hash только для простых типов
// Для сложных типов Hash не реализован - они не могут быть ключами кэша
impl Hash for Value {
    fn hash<H: Hasher>(&self, state: &mut H) {
        match self {
            Value::Number(n) => {
                state.write_u8(0);
                state.write_u64(n.to_bits());
            }
            Value::Int(i) => {
                state.write_u8(11);
                state.write_u64(numeric::hash_int_value(*i));
            }
            Value::Float(n) => {
                state.write_u8(12);
                state.write_u64(numeric::hash_float_value(*n));
            }
            Value::Bool(b) => {
                state.write_u8(1); // Тег для Bool
                state.write_u8(if *b { 1 } else { 0 });
            }
            Value::String(s) => {
                state.write_u8(2); // Тег для String
                s.hash(state);
            }
            Value::Null => {
                state.write_u8(3); // Тег для Null
            }
            Value::Ellipsis => {
                state.write_u8(5); // Тег для Ellipsis
            }
            Value::Uuid(hi, lo) => {
                state.write_u8(4); // Тег для Uuid
                state.write_u64(*hi);
                state.write_u64(*lo);
            }
            Value::Date(d) => {
                state.write_u8(6); // Тег для Date
                state.write_i64(d.timestamp());
                state.write_u32(d.timestamp_subsec_nanos());
                state.write_i32(d.offset().local_minus_utc());
            }
            Value::Duration(d) => {
                state.write_u8(7); // Тег для Duration
                state.write_i64(d.num_seconds());
                state.write_u32(d.subsec_nanos() as u32);
            }
            Value::Path(p) => {
                state.write_u8(8);
                p.hash(state);
            }
            Value::Tuple(t) => {
                if !crate::common::type_model::is_hashable_value(self) {
                    panic!("Cannot hash tuple with mutable/non-hashable elements");
                }
                state.write_u8(9);
                let r = t.borrow();
                r.len().hash(state);
                for item in r.iter() {
                    item.hash(state);
                }
            }
            Value::Object(rc) => match &*rc.borrow() {
                ObjectKind::Bucket(m) if m.is_frozen() => {
                    if let Some(h) = crate::common::type_model::object_key_hash_value(self) {
                        state.write_u8(10);
                        h.hash(state);
                    } else {
                        panic!("Cannot hash frozen object map");
                    }
                }
                _ => {
                    panic!("Cannot hash complex types (Array, Tuple with mutable elems, Table, Object, Function, Iterable)");
                }
            },
            _ => {
                panic!("Cannot hash complex types (Array, Tuple with mutable elems, Table, Object, Function, Iterable)");
            }
        }
    }
}

impl Eq for Value {}

impl Clone for Value {
    fn clone(&self) -> Self {
        match self {
            Value::Int(i) => Value::Int(*i),
            Value::Float(n) => Value::Float(*n),
            Value::Number(n) => Value::Number(*n),
            Value::Bool(b) => Value::Bool(*b),
            Value::String(s) => Value::String(s.clone()),
            Value::Array(arr) => {
                // Создаем глубокую копию массива, рекурсивно клонируя все элементы
                let arr_ref = arr.borrow();
                let cloned_vec: Vec<Value> = arr_ref.iter().map(|v| v.clone()).collect();
                Value::Array(Rc::new(RefCell::new(cloned_vec)))
            }
            Value::Tuple(tuple) => {
                // Создаем глубокую копию кортежа, рекурсивно клонируя все элементы
                let tuple_ref = tuple.borrow();
                let cloned_vec: Vec<Value> = tuple_ref.iter().map(|v| v.clone()).collect();
                Value::Tuple(Rc::new(RefCell::new(cloned_vec)))
            }
            Value::Function(idx) => Value::Function(*idx),
            Value::ModuleFunction {
                module_uid,
                local_index,
            } => Value::ModuleFunction {
                module_uid: *module_uid,
                local_index: *local_index,
            },
            Value::NativeFunction(idx) => Value::NativeFunction(*idx),
            Value::Path(p) => Value::Path(p.clone()),
            Value::Uuid(hi, lo) => Value::Uuid(*hi, *lo),
            Value::Date(d) => Value::Date(*d),
            Value::Duration(d) => Value::Duration(*d),
            Value::Table(table) => {
                // Создаем новый Rc с глубокой копией таблицы
                Value::Table(Rc::new(RefCell::new(table.borrow().clone())))
            }
            Value::ColumnReference { table, column_name } => {
                // Для ColumnReference клонируем ссылку на таблицу и имя колонки
                Value::ColumnReference {
                    table: table.clone(),
                    column_name: column_name.clone(),
                }
            }
            Value::Object(map_rc) => {
                // Клонируем Rc (shallow copy), чтобы изменения сохранялись
                Value::Object(map_rc.clone())
            }
            Value::Set(s) => Value::Set(Rc::new(RefCell::new(s.borrow().clone()))),
            Value::PluginOpaque { tag, id } => Value::PluginOpaque { tag: *tag, id: *id },
            Value::Window(handle) => {
                // WindowHandle is Copy, so just copy it
                Value::Window(*handle)
            }
            Value::Image(image) => {
                // Клонируем Rc (shallow copy), чтобы изменения сохранялись
                Value::Image(image.clone())
            }
            Value::Figure(figure) => {
                // Клонируем Rc (shallow copy), чтобы изменения сохранялись
                Value::Figure(figure.clone())
            }
            Value::Axis(axis) => {
                // Клонируем Rc (shallow copy), чтобы изменения сохранялись
                Value::Axis(axis.clone())
            }
            Value::DatabaseEngine(engine) => Value::DatabaseEngine(engine.clone()),
            Value::DatabaseCluster(cluster) => Value::DatabaseCluster(cluster.clone()),
            Value::Archive(archive) => Value::Archive(archive.clone()),
            Value::DataSource(ds) => Value::DataSource(ds.clone()),
            Value::DataSourceResponse(r) => Value::DataSourceResponse(r.clone()),
            Value::HttpResponse(r) => Value::HttpResponse(r.clone()),
            Value::WebPage(p) => Value::WebPage(p.clone()),
            Value::WebElement(e) => Value::WebElement(e.clone()),
            Value::Enumerate { data, start } => Value::Enumerate {
                data: data.clone(),
                start: *start,
            },
            Value::ArrayView(av) => Value::ArrayView(av.clone()),
            // Share iterator state (Rc) — deep clone would reset IterableInner indices and break for-in / iterable_next.
            Value::Iterable(rc) => Value::Iterable(Rc::clone(rc)),
            Value::Generator(rc) => Value::Generator(Rc::clone(rc)),
            Value::ByteBuffer(b) => Value::ByteBuffer(b.clone()),
            Value::ObjectFieldList {
                source_object_id,
                projection,
                element_ids,
            } => Value::ObjectFieldList {
                source_object_id: *source_object_id,
                projection: *projection,
                element_ids: Rc::clone(element_ids),
            },
            Value::Null => Value::Null,
            Value::Ellipsis => Value::Ellipsis,
        }
    }
}

#[cfg(test)]
mod byte_buffer_tests {
    use super::ByteBuffer;

    #[test]
    fn hex_string_lowercase() {
        let b = ByteBuffer::from_vec_hex(vec![0x2c, 0xf2, 0x4d]);
        assert_eq!(b.hex_string(), "2cf24d");
    }

    #[test]
    fn slice_inherits_display_hex() {
        let b = ByteBuffer::from_vec_hex(vec![1, 2, 3, 4]);
        let slice = b.slice_range(1, 3).unwrap();
        assert!(slice.display_hex);
        assert_eq!(slice.hex_string(), "0203");
    }

    #[test]
    fn to_string_summary_without_hex_flag() {
        let v = super::Value::ByteBuffer(ByteBuffer::from_vec(vec![1, 2]));
        assert_eq!(v.to_string(), "<bytes len=2>");
    }

    #[test]
    fn to_string_hex_with_flag() {
        let v = super::Value::ByteBuffer(ByteBuffer::from_vec_hex(vec![0xab, 0xcd]));
        assert_eq!(v.to_string(), "abcd");
    }
}
