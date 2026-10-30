// Колонка таблицы: типизированный буфер или срез Value (Mixed).
// Фасад `values()` отдаёт `&[Value]` для любого варианта, поэтому читатели не меняются.

use crate::common::numeric::{FloatValue, IntValue};
use crate::common::value::Value;
use std::cell::OnceCell;
use crate::common::value_store::IdBuildHasher;
use std::collections::HashMap;
use std::rc::Rc;

/// One appended column of a Columnar table.
/// Typed variants build their `Value` cells lazily, once, on first `values()`.
#[derive(Debug, Clone)]
pub enum TypedColumn {
    F64 {
        data: Rc<[f64]>,
        /// `true` marks a null cell; None means no nulls.
        nulls: Option<Rc<[bool]>>,
        cells: Rc<OnceCell<Vec<Value>>>,
        /// Which `Value` the cells are: `Float` or `Number`.
        kind: F64Kind,
    },
    I64 {
        data: Rc<[i64]>,
        nulls: Option<Rc<[bool]>>,
        cells: Rc<OnceCell<Vec<Value>>>,
    },
    /// Low-cardinality string column: `dict[codes[i]]` is cell `i`. Entries are unique;
    /// `null_code` is the entry holding `Value::Null`, if any.
    Dict {
        codes: Rc<[u32]>,
        dict: Rc<[Value]>,
        null_code: Option<u32>,
        cells: Rc<OnceCell<Vec<Value>>>,
    },
    /// Any column; the current representation.
    Mixed(Rc<Vec<Value>>),
}

/// Cell type of an F64 column. `Number` (untyped literal result) and `Float` print and
/// combine differently, so one column never holds both.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum F64Kind {
    Float,
    Number,
}

impl F64Kind {
    #[inline]
    pub fn cell(self, x: f64) -> Value {
        match self {
            F64Kind::Float => Value::Float(FloatValue::classify_f64(x)),
            F64Kind::Number => Value::Number(x),
        }
    }
}

impl TypedColumn {
    pub fn mixed(values: Rc<Vec<Value>>) -> Self {
        TypedColumn::Mixed(values)
    }

    pub fn f64(data: Rc<[f64]>, nulls: Option<Rc<[bool]>>) -> Self {
        Self::f64_of(F64Kind::Float, data, nulls)
    }

    pub fn f64_of(kind: F64Kind, data: Rc<[f64]>, nulls: Option<Rc<[bool]>>) -> Self {
        debug_assert!(nulls.as_ref().map_or(true, |n| n.len() == data.len()));
        TypedColumn::F64 {
            data,
            nulls,
            cells: Rc::new(OnceCell::new()),
            kind,
        }
    }

    pub fn i64(data: Rc<[i64]>, nulls: Option<Rc<[bool]>>) -> Self {
        debug_assert!(nulls.as_ref().map_or(true, |n| n.len() == data.len()));
        TypedColumn::I64 {
            data,
            nulls,
            cells: Rc::new(OnceCell::new()),
        }
    }

    pub fn dict(codes: Rc<[u32]>, dict: Rc<[Value]>, null_code: Option<u32>) -> Self {
        debug_assert!(null_code.map_or(true, |c| matches!(dict[c as usize], Value::Null)));
        TypedColumn::Dict {
            codes,
            dict,
            null_code,
            cells: Rc::new(OnceCell::new()),
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        match self {
            TypedColumn::F64 { data, .. } => data.len(),
            TypedColumn::I64 { data, .. } => data.len(),
            TypedColumn::Dict { codes, .. } => codes.len(),
            TypedColumn::Mixed(v) => v.len(),
        }
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Cells as `Value`; typed variants materialize once and cache.
    pub fn values(&self) -> &[Value] {
        match self {
            TypedColumn::Mixed(v) => v.as_slice(),
            TypedColumn::F64 { data, nulls, cells, kind } => {
                cells.get_or_init(|| build_cells(data, nulls.as_deref(), |x| kind.cell(x)))
            }
            TypedColumn::I64 { data, nulls, cells } => cells.get_or_init(|| {
                build_cells(data, nulls.as_deref(), |x| Value::Int(IntValue::Finite(x)))
            }),
            TypedColumn::Dict { codes, dict, cells, .. } => {
                cells.get_or_init(|| codes.iter().map(|&c| dict[c as usize].clone()).collect())
            }
        }
    }

    /// Shared `Vec<Value>` buffer of a Mixed column (no copy).
    pub fn as_mixed(&self) -> Option<&Rc<Vec<Value>>> {
        match self {
            TypedColumn::Mixed(v) => Some(v),
            _ => None,
        }
    }

    /// Raw f64 data and null mask of a `Float` column.
    pub fn as_f64(&self) -> Option<(&[f64], Option<&[bool]>)> {
        self.as_f64_of(F64Kind::Float)
    }

    /// Raw f64 data and null mask of a `Number` column.
    pub fn as_number(&self) -> Option<(&[f64], Option<&[bool]>)> {
        self.as_f64_of(F64Kind::Number)
    }

    pub fn as_f64_of(&self, want: F64Kind) -> Option<(&[f64], Option<&[bool]>)> {
        match self {
            TypedColumn::F64 { data, nulls, kind, .. } if *kind == want => Some((data, nulls.as_deref())),
            _ => None,
        }
    }

    /// Raw i64 data and null mask of an I64 column.
    pub fn as_i64(&self) -> Option<(&[i64], Option<&[bool]>)> {
        match self {
            TypedColumn::I64 { data, nulls, .. } => Some((data, nulls.as_deref())),
            _ => None,
        }
    }

    /// Codes, dictionary and null entry of a Dict column.
    pub fn as_dict(&self) -> Option<(&[u32], &[Value], Option<u32>)> {
        match self {
            TypedColumn::Dict { codes, dict, null_code, .. } => Some((codes, dict, *null_code)),
            _ => None,
        }
    }

    /// False only when the column surely holds no null cell.
    pub fn may_have_nulls(&self) -> bool {
        match self {
            TypedColumn::F64 { nulls, .. } | TypedColumn::I64 { nulls, .. } => nulls.is_some(),
            TypedColumn::Dict { null_code, .. } => null_code.is_some(),
            TypedColumn::Mixed(_) => true,
        }
    }

    /// Shared `Vec<Value>` of this column: Mixed is shared, typed ones are built.
    pub fn to_shared_values(&self) -> Rc<Vec<Value>> {
        match self {
            TypedColumn::Mixed(v) => v.clone(),
            _ => Rc::new(self.to_values_vec()),
        }
    }

    /// Cells as an owned `Vec<Value>`. Uses the cell cache if it is already filled,
    /// but never fills it, so the column does not keep a second copy around.
    pub fn to_values_vec(&self) -> Vec<Value> {
        match self {
            TypedColumn::Mixed(v) => v.as_ref().clone(),
            TypedColumn::F64 { cells, .. }
            | TypedColumn::I64 { cells, .. }
            | TypedColumn::Dict { cells, .. } => match cells.get() {
                Some(v) => v.clone(),
                None => (0..self.len()).map(|i| self.cell(i)).collect(),
            },
        }
    }

    /// One cell as `Value` without filling the cell cache.
    #[inline]
    pub fn cell(&self, i: usize) -> Value {
        match self {
            TypedColumn::Mixed(v) => v[i].clone(),
            TypedColumn::F64 { data, nulls, kind, .. } => {
                if nulls.as_ref().is_some_and(|n| n[i]) {
                    Value::Null
                } else {
                    kind.cell(data[i])
                }
            }
            TypedColumn::I64 { data, nulls, .. } => {
                if nulls.as_ref().is_some_and(|n| n[i]) {
                    Value::Null
                } else {
                    Value::Int(IntValue::Finite(data[i]))
                }
            }
            TypedColumn::Dict { codes, dict, .. } => dict[codes[i] as usize].clone(),
        }
    }

    #[inline]
    pub fn is_null(&self, i: usize) -> bool {
        match self {
            TypedColumn::Mixed(v) => matches!(v[i], Value::Null),
            TypedColumn::F64 { nulls, .. } | TypedColumn::I64 { nulls, .. } => nulls.as_ref().is_some_and(|n| n[i]),
            TypedColumn::Dict { codes, null_code, .. } => *null_code == Some(codes[i]),
        }
    }

    /// Typed null mask; None for Mixed and Dict (use `is_null`) and for columns without nulls.
    pub fn null_mask(&self) -> Option<&[bool]> {
        match self {
            TypedColumn::F64 { nulls, .. } | TypedColumn::I64 { nulls, .. } => nulls.as_deref(),
            TypedColumn::Dict { .. } | TypedColumn::Mixed(_) => None,
        }
    }

    /// Rows at `indices`, keeping the variant.
    pub fn gather(&self, indices: &[usize]) -> TypedColumn {
        fn mask(nulls: &Option<Rc<[bool]>>, indices: &[usize]) -> Option<Rc<[bool]>> {
            let n = nulls.as_ref()?;
            let out: Vec<bool> = indices.iter().map(|&i| n[i]).collect();
            out.contains(&true).then(|| Rc::from(out))
        }
        match self {
            TypedColumn::Mixed(v) => TypedColumn::mixed(Rc::new(indices.iter().map(|&i| v[i].clone()).collect())),
            TypedColumn::F64 { data, nulls, kind, .. } => TypedColumn::f64_of(
                *kind,
                indices.iter().map(|&i| data[i]).collect(),
                mask(nulls, indices),
            ),
            TypedColumn::I64 { data, nulls, .. } => {
                TypedColumn::i64(indices.iter().map(|&i| data[i]).collect(), mask(nulls, indices))
            }
            TypedColumn::Dict { codes, dict, null_code, .. } => {
                TypedColumn::dict(indices.iter().map(|&i| codes[i]).collect(), dict.clone(), *null_code)
            }
        }
    }

    /// Like `gather`, but `None` yields a null cell (outer-join padding).
    pub fn gather_opt(&self, indices: &[Option<usize>]) -> TypedColumn {
        fn mask(nulls: &Option<Rc<[bool]>>, indices: &[Option<usize>]) -> Option<Rc<[bool]>> {
            let out: Vec<bool> = indices
                .iter()
                .map(|i| i.map_or(true, |i| nulls.as_ref().is_some_and(|n| n[i])))
                .collect();
            out.contains(&true).then(|| Rc::from(out))
        }
        match self {
            TypedColumn::Mixed(v) => TypedColumn::mixed(Rc::new(
                indices.iter().map(|i| i.map_or(Value::Null, |i| v[i].clone())).collect(),
            )),
            TypedColumn::F64 { data, nulls, kind, .. } => TypedColumn::f64_of(
                *kind,
                indices.iter().map(|i| i.map_or(0.0, |i| data[i])).collect(),
                mask(nulls, indices),
            ),
            TypedColumn::I64 { data, nulls, .. } => {
                TypedColumn::i64(indices.iter().map(|i| i.map_or(0, |i| data[i])).collect(), mask(nulls, indices))
            }
            TypedColumn::Dict { codes, dict, null_code, .. } => {
                let (dict, nc) = match null_code {
                    Some(nc) => (dict.clone(), *nc),
                    None if indices.iter().all(Option::is_some) => {
                        return TypedColumn::dict(indices.iter().map(|i| codes[i.unwrap()]).collect(), dict.clone(), None);
                    }
                    None => {
                        let mut d = dict.to_vec();
                        d.push(Value::Null);
                        (Rc::from(d), dict.len() as u32)
                    }
                };
                TypedColumn::dict(indices.iter().map(|i| i.map_or(nc, |i| codes[i])).collect(), dict, Some(nc))
            }
        }
    }

    /// Rows `start..start + len`, keeping the variant.
    pub fn slice(&self, start: usize, len: usize) -> TypedColumn {
        if start == 0 && len == self.len() {
            return self.clone();
        }
        let r = start..start + len;
        let mask = |nulls: &Option<Rc<[bool]>>| {
            nulls
                .as_ref()
                .filter(|n| n[r.clone()].contains(&true))
                .map(|n| Rc::from(&n[r.clone()]))
        };
        match self {
            TypedColumn::Mixed(v) => TypedColumn::mixed(Rc::new(v[r.clone()].to_vec())),
            TypedColumn::F64 { data, nulls, kind, .. } => {
                TypedColumn::f64_of(*kind, Rc::from(&data[r.clone()]), mask(nulls))
            }
            TypedColumn::I64 { data, nulls, .. } => TypedColumn::i64(Rc::from(&data[r.clone()]), mask(nulls)),
            TypedColumn::Dict { codes, dict, null_code, .. } => {
                TypedColumn::dict(Rc::from(&codes[r.clone()]), dict.clone(), *null_code)
            }
        }
    }

    /// Nulls replaced by `with`. Stays typed when `with` fits the variant; otherwise rebuilt
    /// through `ColumnBuilder`, so the result equals replacing cell by cell.
    pub fn replace_nulls(&self, with: &Value) -> TypedColumn {
        match (self, with) {
            (TypedColumn::F64 { nulls: None, .. } | TypedColumn::I64 { nulls: None, .. }, _) => self.clone(),
            (TypedColumn::F64 { data, nulls: Some(n), kind, .. }, _) => {
                let fill = match (kind, with) {
                    (F64Kind::Number, Value::Number(x)) => Some(*x),
                    (F64Kind::Float, Value::Float(f)) => float_bits(*f),
                    _ => None,
                };
                match fill {
                    Some(x) => TypedColumn::f64_of(
                        *kind,
                        data.iter().zip(n.iter()).map(|(&d, &null)| if null { x } else { d }).collect(),
                        None,
                    ),
                    None => self.replace_nulls_slow(with),
                }
            }
            (TypedColumn::I64 { data, nulls: Some(n), .. }, Value::Int(IntValue::Finite(x))) => TypedColumn::i64(
                data.iter().zip(n.iter()).map(|(&d, &null)| if null { *x } else { d }).collect(),
                None,
            ),
            (TypedColumn::Dict { null_code: None, .. }, _) | (TypedColumn::Dict { .. }, Value::Null) => self.clone(),
            (TypedColumn::Dict { codes, dict, null_code: Some(nc), .. }, Value::String(s)) => {
                let nc = *nc;
                match dict.iter().position(|d| matches!(d, Value::String(x) if x == s)) {
                    // `with` already has an entry: point the null code at it.
                    Some(k) => TypedColumn::dict(
                        codes.iter().map(|&c| if c == nc { k as u32 } else { c }).collect(),
                        dict.clone(),
                        None,
                    ),
                    // Otherwise the null entry becomes `with`; codes are shared.
                    None => {
                        let mut d = dict.to_vec();
                        d[nc as usize] = with.clone();
                        TypedColumn::dict(codes.clone(), Rc::from(d), None)
                    }
                }
            }
            (TypedColumn::Mixed(v), _) => {
                if !v.iter().any(|c| matches!(c, Value::Null)) {
                    return self.clone();
                }
                TypedColumn::mixed(Rc::new(
                    v.iter()
                        .map(|c| if matches!(c, Value::Null) { with.clone() } else { c.clone() })
                        .collect(),
                ))
            }
            _ => self.replace_nulls_slow(with),
        }
    }

    fn replace_nulls_slow(&self, with: &Value) -> TypedColumn {
        let len = self.len();
        let mut b = ColumnBuilder::with_capacity(len);
        for i in 0..len {
            b.push(if self.is_null(i) { with.clone() } else { self.cell(i) });
        }
        b.finish()
    }
}

/// Builds a column cell by cell and picks the narrowest variant that keeps semantics:
/// all `Float` → F64, all `Number` → F64 of kind Number, all `Int(Finite)` → I64,
/// strings (plus nulls) of low cardinality → Dict, anything else (mix, all-null) → Mixed.
pub struct ColumnBuilder {
    state: BuildState,
    cap: usize,
}

enum BuildState {
    /// Only nulls so far.
    Pending(usize),
    F64(Vec<f64>, Option<Vec<bool>>, F64Kind),
    I64(Vec<i64>, Option<Vec<bool>>),
    Dict {
        codes: Vec<u32>,
        dict: Vec<Value>,
        index: DictIndex,
        null_code: Option<u32>,
    },
    Mixed(Vec<Value>),
}

/// Past this many entries a finished Dict must stay below one entry per two rows.
pub(crate) const DICT_SPARSE: usize = 1 << 10;
/// Rows after which a column whose strings are (almost) all distinct goes Mixed
/// without hashing the rest: an id-like column never pays off as a Dict.
pub(crate) const DICT_PROBE: usize = 1 << 16;

const NO_CODE: u32 = u32::MAX;

/// String → code lookup that borrows the strings from `dict` instead of cloning them:
/// hash → first code, then a chain through `next` (indexed by code) for collisions.
struct DictIndex {
    heads: HashMap<u64, u32, IdBuildHasher>,
    next: Vec<u32>,
}

#[inline]
fn str_hash(s: &str) -> u64 {
    let b = s.as_bytes();
    let mut h = b.len() as u64;
    let mut chunks = b.chunks_exact(8);
    for c in &mut chunks {
        let w = u64::from_le_bytes(c.try_into().unwrap());
        h = (h.rotate_left(5) ^ w).wrapping_mul(0x517c_c1b7_2722_0a95);
    }
    let mut tail = 0u64;
    for (i, &x) in chunks.remainder().iter().enumerate() {
        tail |= (x as u64) << (8 * i);
    }
    h = (h.rotate_left(5) ^ tail).wrapping_mul(0x517c_c1b7_2722_0a95);
    h ^ (h >> 29)
}

/// `a == b` without a `memcmp` call for the short keys typical of categorical columns.
#[inline]
fn short_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    if a.len() > 16 {
        return a == b;
    }
    a.iter().zip(b).fold(0u8, |d, (x, y)| d | (x ^ y)) == 0
}

impl DictIndex {
    fn new() -> Self {
        DictIndex { heads: HashMap::default(), next: Vec::new() }
    }

    #[inline]
    fn find(&self, dict: &[Value], h: u64, s: &str) -> Option<u32> {
        let mut c = *self.heads.get(&h)?;
        while c != NO_CODE {
            if matches!(&dict[c as usize], Value::String(t) if short_eq(t.as_bytes(), s.as_bytes())) {
                return Some(c);
            }
            c = self.next[c as usize];
        }
        None
    }

    /// Records entry `c` (== `next.len()`); non-string entries pass `None`.
    #[inline]
    fn add(&mut self, c: u32, h: Option<u64>) {
        let prev = match h {
            Some(h) => self.heads.insert(h, c).unwrap_or(NO_CODE),
            None => NO_CODE,
        };
        self.next.push(prev);
    }
}

/// Moves each entry out of `dict` at its last use, cloning only repeats.
fn dict_to_cells(codes: &[u32], dict: Vec<Value>, out: &mut Vec<Value>) {
    let mut left = vec![0u32; dict.len()];
    for &c in codes {
        left[c as usize] += 1;
    }
    let mut dict: Vec<Option<Value>> = dict.into_iter().map(Some).collect();
    out.extend(codes.iter().map(|&c| {
        let c = c as usize;
        left[c] -= 1;
        if left[c] == 0 {
            dict[c].take().unwrap()
        } else {
            dict[c].clone().unwrap()
        }
    }));
}

/// f64 that rebuilds exactly this `FloatValue` through `classify_f64`.
#[inline]
fn float_bits(f: FloatValue) -> Option<f64> {
    match f {
        FloatValue::Finite(x) if x.is_finite() => Some(x),
        FloatValue::Finite(_) => None,
        FloatValue::NaN => Some(f64::NAN),
        FloatValue::PosInfinity => Some(f64::INFINITY),
        FloatValue::NegInfinity => Some(f64::NEG_INFINITY),
    }
}

#[inline]
fn push_typed<T: Default>(data: &mut Vec<T>, nulls: &mut Option<Vec<bool>>, x: Option<T>) {
    match x {
        Some(x) => {
            data.push(x);
            if let Some(n) = nulls {
                n.push(false);
            }
        }
        None => {
            let n = nulls.get_or_insert_with(|| vec![false; data.len()]);
            n.push(true);
            data.push(T::default());
        }
    }
}

impl ColumnBuilder {
    pub fn with_capacity(cap: usize) -> Self {
        ColumnBuilder {
            state: BuildState::Pending(0),
            cap,
        }
    }

    pub fn push(&mut self, v: Value) {
        match &mut self.state {
            BuildState::Mixed(out) => out.push(v),
            BuildState::F64(data, nulls, kind) => match (v, *kind) {
                (Value::Null, _) => push_typed(data, nulls, None),
                (Value::Float(f), F64Kind::Float) => match float_bits(f) {
                    Some(x) => push_typed(data, nulls, Some(x)),
                    None => self.to_mixed(Value::Float(f)),
                },
                (Value::Number(x), F64Kind::Number) => push_typed(data, nulls, Some(x)),
                (other, _) => self.to_mixed(other),
            },
            BuildState::I64(data, nulls) => match v {
                Value::Null => push_typed(data, nulls, None),
                Value::Int(IntValue::Finite(x)) => push_typed(data, nulls, Some(x)),
                other => self.to_mixed(other),
            },
            BuildState::Dict { codes, dict, index, null_code } => match v {
                Value::Null => {
                    let c = *null_code.get_or_insert_with(|| {
                        index.add(dict.len() as u32, None);
                        dict.push(Value::Null);
                        (dict.len() - 1) as u32
                    });
                    codes.push(c);
                }
                Value::String(s) => {
                    let h = str_hash(&s);
                    match index.find(dict, h, &s) {
                        Some(c) => codes.push(c),
                        None if codes.len() >= DICT_PROBE && dict.len() * 20 >= codes.len() * 19 => {
                            self.to_mixed(Value::String(s))
                        }
                        None => {
                            let c = dict.len() as u32;
                            index.add(c, Some(h));
                            dict.push(Value::String(s));
                            codes.push(c);
                        }
                    }
                }
                other => self.to_mixed(other),
            },
            BuildState::Pending(n) => {
                let n = *n;
                let lead = |len: usize| (n > 0).then(|| vec![true; len]);
                match v {
                    Value::Null => self.state = BuildState::Pending(n + 1),
                    Value::Float(f) if float_bits(f).is_some() => {
                        let mut data = Vec::with_capacity(self.cap);
                        data.resize(n, 0.0);
                        let mut nulls = lead(n);
                        push_typed(&mut data, &mut nulls, float_bits(f));
                        self.state = BuildState::F64(data, nulls, F64Kind::Float);
                    }
                    Value::Number(x) => {
                        let mut data = Vec::with_capacity(self.cap);
                        data.resize(n, 0.0);
                        let mut nulls = lead(n);
                        push_typed(&mut data, &mut nulls, Some(x));
                        self.state = BuildState::F64(data, nulls, F64Kind::Number);
                    }
                    Value::Int(IntValue::Finite(x)) => {
                        let mut data = Vec::with_capacity(self.cap);
                        data.resize(n, 0);
                        let mut nulls = lead(n);
                        push_typed(&mut data, &mut nulls, Some(x));
                        self.state = BuildState::I64(data, nulls);
                    }
                    Value::String(s) => {
                        let mut codes = Vec::with_capacity(self.cap);
                        let mut dict = Vec::new();
                        let null_code = (n > 0).then(|| {
                            dict.push(Value::Null);
                            codes.resize(n, 0);
                            0
                        });
                        let mut index = DictIndex::new();
                        if null_code.is_some() {
                            index.add(0, None);
                        }
                        index.add(dict.len() as u32, Some(str_hash(&s)));
                        codes.push(dict.len() as u32);
                        dict.push(Value::String(s));
                        self.state = BuildState::Dict { codes, dict, index, null_code };
                    }
                    other => self.to_mixed(other),
                }
            }
        }
    }

    /// Same as `push(Value::Number(x))`, without building the `Value` on the hot path.
    #[inline]
    pub fn push_number(&mut self, x: f64) {
        match &mut self.state {
            BuildState::F64(data, nulls, F64Kind::Number) => push_typed(data, nulls, Some(x)),
            _ => self.push(Value::Number(x)),
        }
    }

    /// Pushes `s` if it is already a Dict entry, without building a `Value`.
    /// Returns false (nothing pushed) otherwise. Dict keys only ever come from
    /// `Value::String`, so a hit pushes exactly what `push(Value::String(s))` would.
    #[inline]
    pub fn push_known_str(&mut self, s: &str) -> bool {
        if let BuildState::Dict { codes, dict, index, .. } = &mut self.state {
            if let Some(c) = index.find(dict, str_hash(s), s) {
                codes.push(c);
                return true;
            }
        }
        false
    }

    /// Rebuilds the cells pushed so far as `Value`s, then appends `v`.
    #[cold]
    fn to_mixed(&mut self, v: Value) {
        let mut out = Vec::with_capacity(self.cap);
        match std::mem::replace(&mut self.state, BuildState::Pending(0)) {
            BuildState::Pending(n) => out.resize(n, Value::Null),
            BuildState::F64(data, nulls, kind) => out.extend(build_cells(&data, nulls.as_deref(), |x| kind.cell(x))),
            BuildState::I64(data, nulls) => out.extend(build_cells(&data, nulls.as_deref(), |x| {
                Value::Int(IntValue::Finite(x))
            })),
            BuildState::Dict { codes, dict, .. } => dict_to_cells(&codes, dict, &mut out),
            BuildState::Mixed(m) => out = m,
        }
        out.push(v);
        self.state = BuildState::Mixed(out);
    }

    pub fn finish(self) -> TypedColumn {
        match self.state {
            BuildState::Pending(n) => TypedColumn::mixed(Rc::new(vec![Value::Null; n])),
            BuildState::F64(data, nulls, kind) => TypedColumn::f64_of(kind, Rc::from(data), nulls.map(Rc::from)),
            BuildState::I64(data, nulls) => TypedColumn::i64(Rc::from(data), nulls.map(Rc::from)),
            BuildState::Dict { codes, dict, null_code, index } => {
                drop(index);
                let strings = dict.len() - null_code.is_some() as usize;
                if strings > DICT_SPARSE && strings * 2 > codes.len() {
                    let mut out = Vec::with_capacity(codes.len());
                    dict_to_cells(&codes, dict, &mut out);
                    return TypedColumn::mixed(Rc::new(out));
                }
                TypedColumn::dict(Rc::from(codes), Rc::from(dict), null_code)
            }
            BuildState::Mixed(v) => TypedColumn::mixed(Rc::new(v)),
        }
    }

    pub fn from_values<'a>(values: impl IntoIterator<Item = &'a Value>, cap: usize) -> TypedColumn {
        let mut b = ColumnBuilder::with_capacity(cap);
        for v in values {
            b.push(v.clone());
        }
        b.finish()
    }
}

fn build_cells<T: Copy>(data: &[T], nulls: Option<&[bool]>, cell: impl Fn(T) -> Value) -> Vec<Value> {
    match nulls {
        None => data.iter().map(|&x| cell(x)).collect(),
        Some(nulls) => data
            .iter()
            .zip(nulls)
            .map(|(&x, &null)| if null { Value::Null } else { cell(x) })
            .collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mixed_shares_buffer() {
        let buf = Rc::new(vec![Value::Int(IntValue::Finite(1)), Value::Null]);
        let col = TypedColumn::mixed(buf.clone());
        assert_eq!(col.len(), 2);
        assert!(std::ptr::eq(col.values().as_ptr(), buf.as_ptr()));
        assert!(Rc::ptr_eq(&col.to_shared_values(), &buf));
    }

    #[test]
    fn f64_facade_with_nulls() {
        let col = TypedColumn::f64(Rc::from(vec![1.5, 0.0, f64::NAN]), Some(Rc::from(vec![false, true, false])));
        let v = col.values();
        assert_eq!(v.len(), 3);
        assert_eq!(v[0], Value::Float(FloatValue::Finite(1.5)));
        assert_eq!(v[1], Value::Null);
        assert!(matches!(v[2], Value::Float(FloatValue::NaN)));
        assert!(std::ptr::eq(col.values().as_ptr(), v.as_ptr()), "cells are cached");
        assert!(col.as_mixed().is_none());
    }

    #[test]
    fn i64_facade_without_nulls() {
        let col = TypedColumn::i64(Rc::from(vec![3, -4]), None);
        assert_eq!(
            col.values(),
            &[Value::Int(IntValue::Finite(3)), Value::Int(IntValue::Finite(-4))]
        );
        assert_eq!(col.to_shared_values().len(), 2);
    }

    fn build(vals: Vec<Value>) -> TypedColumn {
        ColumnBuilder::from_values(&vals, vals.len())
    }

    fn f(x: f64) -> Value {
        Value::Float(FloatValue::classify_f64(x))
    }

    fn i(x: i64) -> Value {
        Value::Int(IntValue::Finite(x))
    }

    #[test]
    fn builder_picks_variant() {
        assert!(build(vec![f(1.5), f(2.0)]).as_f64().is_some());
        assert!(build(vec![i(1), i(2)]).as_i64().is_some());
        assert!(build(vec![i(1), f(2.0)]).as_mixed().is_some(), "int/float mix stays Mixed");
        assert!(build(vec![f(1.0), i(2)]).as_mixed().is_some());
        assert!(build(vec![Value::Number(1.0)]).as_number().is_some());
        assert!(build(vec![Value::Number(1.0)]).as_f64().is_none(), "Number is not Float");
        assert!(build(vec![Value::Number(1.0), f(2.0)]).as_mixed().is_some(), "Number/Float mix stays Mixed");
        assert!(build(vec![f(1.0), Value::Number(2.0)]).as_mixed().is_some());
        assert!(build(vec![Value::Number(1.0), i(2)]).as_mixed().is_some());
        assert!(build(vec![Value::Null, Value::Null]).as_mixed().is_some(), "all-null stays Mixed");
        assert!(build(vec![]).as_mixed().is_some());
    }

    #[test]
    fn builder_round_trips_cells() {
        let cases = vec![
            vec![Value::Null, f(1.5), Value::Null, f(f64::NAN), f(f64::INFINITY), f(f64::NEG_INFINITY)],
            vec![Value::Null, Value::Null, i(7), Value::Null, i(i64::MIN)],
            vec![i(1), Value::Null, f(2.5), Value::Null],
            vec![Value::Null, i(1), Value::Number(2.0)],
            vec![Value::Null, Value::Number(2.5), Value::Number(f64::NAN), Value::Number(f64::INFINITY)],
            vec![Value::Number(1.0), Value::Null, f(2.0)],
            vec![f(1.0), Value::Float(FloatValue::Finite(f64::INFINITY))],
        ];
        for vals in cases {
            let col = build(vals.clone());
            let got = col.values();
            assert_eq!(got.len(), vals.len());
            for (g, v) in got.iter().zip(&vals) {
                assert_eq!(format!("{:?}", g), format!("{:?}", v));
            }
        }
    }

    #[test]
    fn builder_leading_nulls_masked() {
        let col = build(vec![Value::Null, f(2.0)]);
        let (data, nulls) = col.as_f64().unwrap();
        assert_eq!(data.len(), 2);
        assert_eq!(nulls, Some(&[true, false][..]));
        assert!(build(vec![f(2.0), f(3.0)]).as_f64().unwrap().1.is_none(), "no nulls, no mask");
    }

    fn s(x: &str) -> Value {
        Value::String(x.to_string())
    }

    #[test]
    fn builder_dict_round_trip() {
        let vals = vec![Value::Null, s("a"), s("b"), Value::Null, s("a")];
        let col = build(vals.clone());
        let (codes, dict, nc) = col.as_dict().expect("strings go Dict");
        assert_eq!(codes, &[0, 1, 2, 0, 1]);
        assert_eq!(dict, &[Value::Null, s("a"), s("b")]);
        assert_eq!(nc, Some(0));
        assert_eq!(col.values(), &vals[..]);
        assert!(col.is_null(3) && !col.is_null(1));
        assert_eq!(col.cell(4), s("a"));
        assert!(build(vec![s("a"), i(1)]).as_mixed().is_some(), "string/int mix stays Mixed");
        assert!(build(vec![i(1), s("a")]).as_mixed().is_some());
        assert_eq!(build(vec![s("a"), s("a")]).as_dict().unwrap().2, None);
    }

    #[test]
    fn builder_dict_high_cardinality_is_mixed() {
        let vals: Vec<Value> = (0..3000).map(|k| s(&k.to_string())).collect();
        let col = build(vals.clone());
        assert!(col.as_mixed().is_some());
        assert_eq!(col.values(), &vals[..]);
    }

    #[test]
    fn builder_dict_cardinality_decided_on_whole_column() {
        // 5000 distinct keys, but the first 5000 rows are all new: the prefix alone looks unique.
        let mut vals: Vec<Value> = (0..20_000).map(|k| s(&format!("k{}", k % 5000))).collect();
        vals[7] = Value::Null;
        vals[12_345] = Value::Null;
        let col = build(vals.clone());
        let (_, dict, null_code) = col.as_dict().expect("5000 keys over 20000 rows stay Dict");
        assert_eq!(dict.len(), 5001);
        assert!(null_code.is_some());
        assert_eq!(col.values(), &vals[..]);
        // Same key set, every row distinct past the probe window → Mixed, cells intact.
        let ids: Vec<Value> = (0..70_000).map(|k| s(&format!("id{}", k))).collect();
        let col = build(ids.clone());
        assert!(col.as_mixed().is_some());
        assert_eq!(col.values(), &ids[..]);
    }

    #[test]
    fn dict_gather_slice_share_dict() {
        let col = build(vec![s("x"), Value::Null, s("y"), s("x")]);
        let g = col.gather(&[3, 1, 2]);
        assert_eq!(g.values(), &[s("x"), Value::Null, s("y")]);
        let sl = col.slice(1, 2);
        assert_eq!(sl.values(), &[Value::Null, s("y")]);
        assert!(std::ptr::eq(col.as_dict().unwrap().1, g.as_dict().unwrap().1));
    }

    #[test]
    fn dict_replace_nulls() {
        let col = build(vec![s("x"), Value::Null, s("y")]);
        let known = col.replace_nulls(&s("y"));
        assert_eq!(known.values(), &[s("x"), s("y"), s("y")]);
        assert!(!known.may_have_nulls());
        let fresh = col.replace_nulls(&s("z"));
        assert_eq!(fresh.values(), &[s("x"), s("z"), s("y")]);
        assert!(fresh.as_dict().is_some());
        let other = col.replace_nulls(&i(0));
        assert_eq!(other.values(), &[s("x"), i(0), s("y")]);
        assert_eq!(col.replace_nulls(&Value::Null).values(), col.values());
    }

    #[test]
    fn builder_non_finite_finite_is_mixed() {
        // Float(Finite(inf)) does not survive classify_f64, so it must not go typed.
        let col = build(vec![Value::Float(FloatValue::Finite(f64::INFINITY))]);
        assert!(col.as_mixed().is_some());
    }
}
