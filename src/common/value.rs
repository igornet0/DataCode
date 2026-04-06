// Единый тип значений для VM

use crate::common::table::Table;
use crate::common::value_store::ValueId;
use crate::common::TaggedValue;
use crate::database_engine::cluster::DatabaseCluster;
use crate::database_engine::engine::DatabaseEngine;
use crate::plot::{Axis, Figure, Image, PlotWindowHandle};
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

/// Dense byte payload (e.g. `read_file_bin`): one `Vec<u8>` shared by slice views, no per-byte `Value::Number`.
#[derive(Debug, Clone)]
pub struct ByteBuffer {
    pub bytes: Rc<Vec<u8>>,
    pub offset: usize,
    pub len: usize,
}

impl ByteBuffer {
    pub fn from_vec(v: Vec<u8>) -> Self {
        let len = v.len();
        Self {
            bytes: Rc::new(v),
            offset: 0,
            len,
        }
    }

    pub fn slice_range(&self, start: usize, end: usize) -> Option<Self> {
        if start > end || end > self.len {
            return None;
        }
        Some(Self {
            bytes: Rc::clone(&self.bytes),
            offset: self.offset + start,
            len: end - start,
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
    Table(Rc<RefCell<Table>>),
    Object(Rc<RefCell<HashMap<String, Value>>>), // Словарь/объект: ключ-значение (обернут в Rc<RefCell> для мутабельности)
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
    Enumerate {
        data: Rc<RefCell<Vec<Value>>>,
        start: i64,
    }, // enum(iterable): lazy (idx, element) wrapper
    /// Zero-copy view; see [`ArrayViewData`].
    ArrayView(ArrayViewData),
    /// Raw bytes from a file or similar; slice with `ByteBuffer::slice_range` / VM slice ops.
    ByteBuffer(ByteBuffer),
    /// Lazy functional pipeline (`map` / `filter`); single-pass iteration, no intermediate array.
    Iterable(Rc<RefCell<IterableInner>>),
    /// Результат вызова `stream fn`: ленивый генератор с фиксированным состоянием.
    Generator(Rc<RefCell<GeneratorState>>),
    Null,
    Ellipsis, // ... (e.g. Field(...) for required field)
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
    /// `stream fn` / [`Value::Generator`]: один проход через [`crate::vm::generator::run_generator_next`].
    StreamGenerator {
        state: Rc<RefCell<GeneratorState>>,
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
            Self::StreamGenerator { state } => Self::StreamGenerator {
                state: Rc::clone(state),
            },
        }
    }
}

impl std::fmt::Debug for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Value::Object(map_rc) => {
                let map = map_rc.borrow();
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
                f.debug_map().entries(map.iter()).finish()
            }
            _ => match self {
                Value::Number(n) => std::fmt::Debug::fmt(n, f),
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
                Value::Table(t) => f.debug_tuple("Table").field(&t.borrow()).finish(),
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
                Value::Null => write!(f, "Null"),
                Value::Ellipsis => write!(f, "Ellipsis"),
            },
        }
    }
}

impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Value::Number(a), Value::Number(b)) => a == b,
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
            (Value::Table(a), Value::Table(b)) => *a.borrow() == *b.borrow(),
            (Value::Object(a), Value::Object(b)) => {
                let am = a.borrow();
                let bm = b.borrow();
                // MetaData and create_all have circular refs; compare by pointer to avoid recursion
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
            (Value::Enumerate { data: a, start: sa }, Value::Enumerate { data: b, start: sb }) => {
                Rc::ptr_eq(a, b) && sa == sb
            }
            (Value::Iterable(a), Value::Iterable(b)) => Rc::ptr_eq(a, b),
            (Value::Generator(a), Value::Generator(b)) => Rc::ptr_eq(a, b),
            (Value::ByteBuffer(a), Value::ByteBuffer(b)) => {
                a.len == b.len && a.bytes.as_ptr() == b.bytes.as_ptr() && a.offset == b.offset
            }
            (Value::Null, Value::Null) => true,
            (Value::Ellipsis, Value::Ellipsis) => true,
            _ => false,
        }
    }
}

impl Value {
    /// Проверяет, можно ли использовать это значение как ключ кэша
    /// (только простые типы: Number, Bool, String, Null)
    pub fn is_hashable(&self) -> bool {
        matches!(
            self,
            Value::Number(_)
                | Value::Bool(_)
                | Value::String(_)
                | Value::Uuid(_, _)
                | Value::Null
                | Value::Ellipsis
        )
    }

    pub fn is_truthy(&self) -> bool {
        match self {
            Value::Null => false,
            Value::Bool(false) => false,
            Value::Number(n) => *n != 0.0,
            Value::String(s) => !s.is_empty(), // Пустая строка = false
            Value::Array(arr) => !arr.borrow().is_empty(),
            Value::ArrayView(av) => av.length > 0,
            Value::Tuple(tuple) => !tuple.borrow().is_empty(),
            Value::Path(p) => !p.as_os_str().is_empty(), // Путь не пустой = true
            Value::Uuid(_, _) => true,                   // UUID всегда truthy
            Value::Table(table) => table.borrow().len() > 0, // Таблица не пустая = true
            Value::Object(map_rc) => !map_rc.borrow().is_empty(), // Объект не пустой = true
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
            Value::Enumerate { data, .. } => !data.borrow().is_empty(),
            Value::Iterable(_) => true,
            Value::ByteBuffer(b) => b.len > 0,
            Value::Ellipsis => true,
            _ => true,
        }
    }

    pub fn to_string(&self) -> String {
        match self {
            Value::Number(n) => {
                if n.fract() == 0.0 {
                    format!("{}", *n as i64)
                } else {
                    format!("{}", n)
                }
            }
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
            Value::ByteBuffer(b) => {
                format!("<bytes len={}>", b.len)
            }
            Value::Tuple(tuple) => {
                let tuple_ref = tuple.borrow();
                let elements: Vec<String> = tuple_ref.iter().map(|v| v.to_string()).collect();
                format!("({})", elements.join(", "))
            }
            Value::Function(_) | Value::ModuleFunction { .. } => "<function>".to_string(),
            Value::NativeFunction(_) => "<native function>".to_string(),
            Value::Path(p) => {
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
            Value::Object(map_rc) => {
                let map = map_rc.borrow();
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
                    IterableInner::Array { .. } | IterableInner::ArrayView { .. } => "iterable",
                    IterableInner::StreamGenerator { .. } => "stream_generator",
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
                // Хешируем число как байты для точности
                state.write_u8(0); // Тег для Number
                state.write_u64(n.to_bits());
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
            // Для остальных типов не реализуем Hash - они не могут быть ключами кэша
            _ => {
                panic!("Cannot hash complex types (Array, Tuple, Table, Object, Function, Path, Iterable)");
            }
        }
    }
}

impl Eq for Value {}

impl Clone for Value {
    fn clone(&self) -> Self {
        match self {
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
            Value::Enumerate { data, start } => Value::Enumerate {
                data: data.clone(),
                start: *start,
            },
            Value::ArrayView(av) => Value::ArrayView(av.clone()),
            // Share iterator state (Rc) — deep clone would reset IterableInner indices and break for-in / iterable_next.
            Value::Iterable(rc) => Value::Iterable(Rc::clone(rc)),
            Value::Generator(rc) => Value::Generator(Rc::clone(rc)),
            Value::ByteBuffer(b) => Value::ByteBuffer(b.clone()),
            Value::Null => Value::Null,
            Value::Ellipsis => Value::Ellipsis,
        }
    }
}
