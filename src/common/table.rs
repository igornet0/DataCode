// Структура данных для таблиц: flat storage (row-major), ColumnView для итерации без материализации.

use crate::common::typed_column::TypedColumn;
use crate::common::value::Value;
use crate::common::value_store::ValueId;
use std::cell::{OnceCell, RefCell};
use std::collections::HashMap;
use std::rc::Rc;

/// Clone-on-write: detach a unique table handle before mutation.
///
/// After VM `load_value`, a table referenced by one heap slot plus the native
/// argument has `strong_count == 2`. Extra aliases (`t2 = t1`) raise the count
/// and trigger a deep copy of `TableData`.
pub fn table_make_mut(rc: &mut Rc<RefCell<Table>>) {
    if Rc::strong_count(rc) > 2 {
        let cloned = rc.borrow().clone();
        *rc = Rc::new(RefCell::new(cloned));
    }
}

/// Table data: flat row-major storage for both View (ValueIds) and Owned (Values).
#[derive(Debug, Clone)]
pub enum TableData {
    /// View over ValueStore: one flat Vec<ValueId> of length rows*num_cols; no column materialization.
    View {
        flat_cell_ids: Vec<ValueId>,
        num_cols: usize,
        headers: Vec<String>,
    },
    /// Owned: one flat Vec<Value> row-major; get_row/get_column without store.
    Owned {
        flat: Vec<Value>,
        num_cols: usize,
        headers: Vec<String>,
        /// Lazy column cache for get_column(); built from flat on first access.
        column_cache: HashMap<String, Vec<Value>>,
    },
    /// Columnar: immutable shared row-major `base` plus appended column buffers.
    /// Adding a column shares every buffer by `Rc` instead of copying rows.
    /// Row readers build a row-major flat once via `materialized`.
    Columnar {
        base: Rc<SharedRows>,
        extra: Vec<TypedColumn>,
        headers: Vec<String>,
        materialized: Rc<OnceCell<Vec<Value>>>,
    },
}

/// Immutable row-major block shared between Columnar tables.
#[derive(Debug)]
pub struct SharedRows {
    pub flat: Vec<Value>,
    pub num_cols: usize,
}

/// Reference to owned rows as a view over flat storage (no copy).
#[derive(Debug)]
pub struct RowsRef<'a> {
    flat: &'a [Value],
    num_cols: usize,
    len: usize,
}

impl<'a> RowsRef<'a> {
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns row `i` as a slice into flat storage (no allocation).
    #[inline]
    pub fn row(&self, i: usize) -> Option<&'a [Value]> {
        if i >= self.len {
            return None;
        }
        let start = i * self.num_cols;
        let end = (i + 1) * self.num_cols;
        if end <= self.flat.len() {
            Some(&self.flat[start..end])
        } else {
            None
        }
    }

    /// Iterator over rows as slices (no allocation per row).
    pub fn iter(&'a self) -> RowsIter<'a> {
        RowsIter { rr: self, next: 0 }
    }

    /// Materialize all rows into Vec<Vec<Value>> (e.g. for clone/join).
    pub fn to_vec(&self) -> Vec<Vec<Value>> {
        self.iter().map(|s| s.to_vec()).collect()
    }
}

/// Iterator over row slices for RowsRef.
pub struct RowsIter<'a> {
    rr: &'a RowsRef<'a>,
    next: usize,
}

impl<'a> Iterator for RowsIter<'a> {
    type Item = &'a [Value];

    fn next(&mut self) -> Option<Self::Item> {
        if self.next >= self.rr.len {
            return None;
        }
        let row = self.rr.row(self.next);
        self.next += 1;
        row
    }
}

/// Lazy column view: access by index without materializing the whole column.
#[derive(Debug)]
pub enum ColumnView<'a> {
    Owned {
        flat: &'a [Value],
        num_cols: usize,
        col_index: usize,
        len: usize,
    },
    View {
        flat_cell_ids: &'a [ValueId],
        num_cols: usize,
        col_index: usize,
        len: usize,
    },
}

impl<'a> ColumnView<'a> {
    #[inline]
    pub fn len(&self) -> usize {
        match self {
            ColumnView::Owned { len, .. } => *len,
            ColumnView::View { len, .. } => *len,
        }
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get cell at row index (Owned: no store; View: caller must use table_ops::get_cell_value with store/heap).
    #[inline]
    pub fn get_owned(&self, row: usize) -> Option<Value> {
        match self {
            ColumnView::Owned {
                flat,
                num_cols,
                col_index,
                len,
            } => {
                if row >= *len {
                    return None;
                }
                let idx = row * num_cols + col_index;
                flat.get(idx).cloned()
            }
            ColumnView::View { .. } => None,
        }
    }

    /// Borrow cell at row index (Owned only; View returns None).
    #[inline]
    pub fn get_ref(&self, row: usize) -> Option<&'a Value> {
        match self {
            ColumnView::Owned {
                flat,
                num_cols,
                col_index,
                len,
            } => {
                if row >= *len {
                    return None;
                }
                flat.get(row * num_cols + col_index)
            }
            ColumnView::View { .. } => None,
        }
    }

    /// For View: index into flat_cell_ids for row (value still needs load_value in table_ops).
    #[inline]
    pub fn cell_id_at(&self, row: usize) -> Option<ValueId> {
        match self {
            ColumnView::View {
                flat_cell_ids,
                num_cols,
                col_index,
                len,
            } => {
                if row >= *len {
                    return None;
                }
                let idx = row * num_cols + col_index;
                flat_cell_ids.get(idx).copied()
            }
            ColumnView::Owned { .. } => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Table {
    pub data: TableData,
    pub name: Option<String>,
}

impl Table {
    pub fn new() -> Self {
        Self {
            data: TableData::Owned {
                flat: Vec::new(),
                num_cols: 0,
                headers: Vec::new(),
                column_cache: HashMap::new(),
            },
            name: None,
        }
    }

    pub fn set_name(&mut self, name: String) {
        self.name = Some(name);
    }

    /// Build table from rows (owned). Flattens into row-major storage.
    pub fn from_data(data: Vec<Vec<Value>>, headers: Option<Vec<String>>) -> Self {
        if data.is_empty() {
            let mut table = Self::new();
            if let Some(h) = headers {
                table.set_headers(h.clone());
                if let TableData::Owned { num_cols, .. } = &mut table.data {
                    *num_cols = h.len();
                }
            }
            return table;
        }

        let num_cols = data[0].len();
        let headers =
            headers.unwrap_or_else(|| (0..num_cols).map(|i| format!("Column_{}", i)).collect());

        let flat: Vec<Value> = data
            .into_iter()
            .flat_map(|row| {
                let mut r = row;
                if r.len() < num_cols {
                    r.resize(num_cols, Value::Null);
                } else if r.len() > num_cols {
                    r.truncate(num_cols);
                }
                r
            })
            .collect();

        Table {
            data: TableData::Owned {
                flat,
                num_cols,
                headers,
                column_cache: HashMap::new(),
            },
            name: None,
        }
    }

    /// Empty owned table with reserved flat capacity (`row_capacity * headers.len()`).
    pub fn with_capacity_owned(headers: Vec<String>, row_capacity: usize) -> Self {
        let num_cols = headers.len();
        Table {
            data: TableData::Owned {
                flat: Vec::with_capacity(row_capacity.saturating_mul(num_cols)),
                num_cols,
                headers,
                column_cache: HashMap::new(),
            },
            name: None,
        }
    }

    /// Build owned table from an already-flat row-major buffer.
    pub fn from_flat_owned(flat: Vec<Value>, num_cols: usize, headers: Vec<String>) -> Self {
        Table {
            data: TableData::Owned {
                flat,
                num_cols,
                headers,
                column_cache: HashMap::new(),
            },
            name: None,
        }
    }

    /// Build table as view over ValueStore using flat cell IDs (rows*num_cols).
    pub fn from_row_ids(row_ids: Vec<ValueId>, headers: Vec<String>) -> Self {
        Table {
            data: TableData::View {
                flat_cell_ids: row_ids,
                num_cols: headers.len().max(1),
                headers,
            },
            name: None,
        }
    }

    /// Build View table from pre-flattened cell IDs (length must be rows*num_cols).
    pub fn from_flat_view(
        flat_cell_ids: Vec<ValueId>,
        num_cols: usize,
        headers: Vec<String>,
    ) -> Self {
        Table {
            data: TableData::View {
                flat_cell_ids,
                num_cols,
                headers,
            },
            name: None,
        }
    }

    /// Build owned table with rows, headers, and optional column cache (e.g. for rename_columns).
    pub fn from_data_with_columns(
        rows: Vec<Vec<Value>>,
        headers: Vec<String>,
        columns: HashMap<String, Vec<Value>>,
    ) -> Self {
        if rows.is_empty() {
            return Table {
                data: TableData::Owned {
                    flat: Vec::new(),
                    num_cols: 0,
                    headers,
                    column_cache: columns,
                },
                name: None,
            };
        }
        let num_cols = headers.len().max(1);
        let flat: Vec<Value> = rows
            .into_iter()
            .flat_map(|row| {
                let mut r = row;
                if r.len() < num_cols {
                    r.resize(num_cols, Value::Null);
                } else if r.len() > num_cols {
                    r.truncate(num_cols);
                }
                r
            })
            .collect();
        Table {
            data: TableData::Owned {
                flat,
                num_cols,
                headers,
                column_cache: columns,
            },
            name: None,
        }
    }

    pub fn set_headers(&mut self, headers: Vec<String>) {
        match &mut self.data {
            TableData::View { headers: h, .. } => *h = headers,
            TableData::Owned { headers: h, .. } => *h = headers,
            TableData::Columnar { headers: h, .. } => *h = headers,
        }
    }

    /// Row-major cells of an Owned or Columnar table (Columnar materializes once).
    /// None for View tables.
    pub fn owned_parts(&self) -> Option<(&Vec<Value>, usize, &Vec<String>)> {
        match &self.data {
            TableData::Owned {
                flat,
                num_cols,
                headers,
                ..
            } => Some((flat, *num_cols, headers)),
            TableData::Columnar {
                base,
                extra,
                headers,
                materialized,
            } => {
                let flat = materialized.get_or_init(|| interleave(base, extra));
                Some((flat, headers.len(), headers))
            }
            TableData::View { .. } => None,
        }
    }

    /// Turn a Columnar table into Owned so mutators can work in place.
    /// Takes the materialized buffer when unique, otherwise builds a fresh one.
    pub(crate) fn columnar_to_owned(&mut self) {
        if !matches!(self.data, TableData::Columnar { .. }) {
            return;
        }
        let data = std::mem::replace(
            &mut self.data,
            TableData::Owned {
                flat: Vec::new(),
                num_cols: 0,
                headers: Vec::new(),
                column_cache: HashMap::new(),
            },
        );
        if let TableData::Columnar {
            base,
            extra,
            headers,
            materialized,
        } = data
        {
            let flat = match Rc::try_unwrap(materialized).ok().and_then(|c| c.into_inner()) {
                Some(flat) => flat,
                None => interleave(&base, &extra),
            };
            self.data = TableData::Owned {
                flat,
                num_cols: headers.len(),
                headers,
                column_cache: HashMap::new(),
            };
        }
    }

    /// Share this table's rows as an immutable base without copying them.
    /// Owned becomes Columnar with no extra columns; contents and headers stay the same.
    pub fn share_as_base(&mut self) -> Option<(Rc<SharedRows>, Vec<TypedColumn>)> {
        if let TableData::Owned { num_cols, headers, .. } = &self.data {
            if *num_cols != headers.len() || *num_cols == 0 {
                return None;
            }
            let data = std::mem::replace(
                &mut self.data,
                TableData::Owned {
                    flat: Vec::new(),
                    num_cols: 0,
                    headers: Vec::new(),
                    column_cache: HashMap::new(),
                },
            );
            if let TableData::Owned {
                flat,
                num_cols,
                headers,
                ..
            } = data
            {
                self.data = TableData::Columnar {
                    base: Rc::new(SharedRows { flat, num_cols }),
                    extra: Vec::new(),
                    headers,
                    materialized: Rc::new(OnceCell::new()),
                };
            }
        }
        match &self.data {
            TableData::Columnar { base, extra, .. } => Some((base.clone(), extra.clone())),
            _ => None,
        }
    }

    /// New table sharing this Columnar table's buffers plus one appended column.
    /// The caller checks the name and length.
    pub fn with_extra_column(&self, name: &str, values: Rc<Vec<Value>>) -> Option<Table> {
        self.with_extra_typed(name, TypedColumn::mixed(values))
    }

    /// Like [`Table::with_extra_column`] for an already built column.
    pub fn with_extra_typed(&self, name: &str, column: TypedColumn) -> Option<Table> {
        match &self.data {
            TableData::Columnar {
                base,
                extra,
                headers,
                ..
            } => {
                let mut extra = extra.clone();
                extra.push(column);
                let mut headers = headers.clone();
                headers.push(name.to_string());
                Some(Table {
                    data: TableData::Columnar {
                        base: base.clone(),
                        extra,
                        headers,
                        materialized: Rc::new(OnceCell::new()),
                    },
                    name: None,
                })
            }
            _ => None,
        }
    }

    /// Single-column table whose column buffer is shared, not copied.
    pub fn from_column(name: String, values: Rc<Vec<Value>>) -> Table {
        Table::from_typed_column(name, TypedColumn::mixed(values))
    }

    /// Single-column table over an already built column.
    pub fn from_typed_column(name: String, column: TypedColumn) -> Table {
        Table {
            data: TableData::Columnar {
                base: Rc::new(SharedRows {
                    flat: Vec::new(),
                    num_cols: 0,
                }),
                extra: vec![column],
                headers: vec![name],
                materialized: Rc::new(OnceCell::new()),
            },
            name: None,
        }
    }

    /// Table over already built columns, one per header, all of the same length.
    pub fn from_typed_columns(headers: Vec<String>, columns: Vec<TypedColumn>) -> Table {
        debug_assert_eq!(headers.len(), columns.len());
        Table {
            data: TableData::Columnar {
                base: Rc::new(SharedRows {
                    flat: Vec::new(),
                    num_cols: 0,
                }),
                extra: columns,
                headers,
                materialized: Rc::new(OnceCell::new()),
            },
            name: None,
        }
    }

    pub fn is_view(&self) -> bool {
        matches!(self.data, TableData::View { .. })
    }

    /// Identity for `relate` / `primary_key` flush: shared `Rc` handles compare equal by content.
    pub fn same_schema_binding(&self, other: &Table) -> bool {
        if self.headers() != other.headers() || self.len() != other.len() {
            return false;
        }
        match (&self.data, &other.data) {
            (
                TableData::Owned { flat: a, .. },
                TableData::Owned { flat: b, .. },
            ) => a == b,
            (
                TableData::View {
                    flat_cell_ids: a, ..
                },
                TableData::View {
                    flat_cell_ids: b, ..
                },
            ) => a == b,
            (TableData::View { .. }, _) | (_, TableData::View { .. }) => false,
            _ => match (self.owned_parts(), other.owned_parts()) {
                (Some((a, _, _)), Some((b, _, _))) => a == b,
                _ => false,
            },
        }
    }

    /// Convert a View table to Owned by loading each cell with the given callback.
    /// No-op for already Owned tables. Used so ML dataset() can call get_column on tables created by table() fast path.
    pub fn materialize_with<F>(&self, load: F) -> Self
    where
        F: Fn(ValueId) -> Value,
    {
        match &self.data {
            TableData::Owned { .. } | TableData::Columnar { .. } => self.clone(),
            TableData::View {
                flat_cell_ids,
                num_cols,
                headers,
            } => {
                let flat: Vec<Value> = flat_cell_ids.iter().map(|&id| load(id)).collect();
                Table {
                    data: TableData::Owned {
                        flat,
                        num_cols: *num_cols,
                        headers: headers.clone(),
                        column_cache: HashMap::new(),
                    },
                    name: self.name.clone(),
                }
            }
        }
    }

    pub fn len(&self) -> usize {
        match &self.data {
            TableData::View {
                flat_cell_ids,
                num_cols,
                ..
            } => {
                if *num_cols == 0 {
                    0
                } else {
                    flat_cell_ids.len() / num_cols
                }
            }
            TableData::Owned { flat, num_cols, .. } => {
                if *num_cols == 0 {
                    0
                } else {
                    flat.len() / num_cols
                }
            }
            TableData::Columnar { base, extra, .. } => columnar_len(base, extra),
        }
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn column_count(&self) -> usize {
        self.headers().len()
    }

    pub fn headers(&self) -> &Vec<String> {
        match &self.data {
            TableData::View { headers, .. } => headers,
            TableData::Owned { headers, .. } => headers,
            TableData::Columnar { headers, .. } => headers,
        }
    }

    #[inline]
    pub fn headers_ref(&self) -> &Vec<String> {
        self.headers()
    }

    pub fn has_column(&self, name: &str) -> bool {
        self.headers().iter().any(|h| h == name)
    }

    /// Reference to owned rows (flat storage view). None for View tables.
    pub fn rows_ref(&self) -> Option<RowsRef<'_>> {
        match &self.data {
            TableData::Owned { flat, num_cols, .. } => {
                let len = if *num_cols == 0 {
                    0
                } else {
                    flat.len() / num_cols
                };
                Some(RowsRef {
                    flat: flat.as_slice(),
                    num_cols: *num_cols,
                    len,
                })
            }
            TableData::Columnar { .. } => {
                let len = self.len();
                let (flat, num_cols, _) = self.owned_parts()?;
                Some(RowsRef {
                    flat: flat.as_slice(),
                    num_cols,
                    len,
                })
            }
            TableData::View { .. } => None,
        }
    }

    /// Column view for iteration without materializing (no Vec<Value>).
    pub fn column_view(&self, name: &str) -> Option<ColumnView<'_>> {
        let col_index = self.headers().iter().position(|h| h == name)?;
        let len = self.len();
        match &self.data {
            TableData::Owned { flat, num_cols, .. } => Some(ColumnView::Owned {
                flat: flat.as_slice(),
                num_cols: *num_cols,
                col_index,
                len,
            }),
            TableData::View {
                flat_cell_ids,
                num_cols,
                ..
            } => Some(ColumnView::View {
                flat_cell_ids: flat_cell_ids.as_slice(),
                num_cols: *num_cols,
                col_index,
                len,
            }),
            TableData::Columnar { base, extra, .. } => Some(if col_index < base.num_cols {
                ColumnView::Owned {
                    flat: base.flat.as_slice(),
                    num_cols: base.num_cols,
                    col_index,
                    len,
                }
            } else {
                ColumnView::Owned {
                    flat: extra[col_index - base.num_cols].values(),
                    num_cols: 1,
                    col_index: 0,
                    len,
                }
            }),
        }
    }

    /// Shared buffer of an appended Columnar column (no copy for Mixed). None otherwise.
    pub fn shared_column(&self, name: &str) -> Option<Rc<Vec<Value>>> {
        self.shared_typed_column(name).map(|c| c.to_shared_values())
    }

    /// Appended Columnar column as a cheap clone (buffers shared). None otherwise.
    pub fn shared_typed_column(&self, name: &str) -> Option<TypedColumn> {
        self.typed_column(name).cloned()
    }

    /// All columns by index when the table is fully typed (Columnar with an empty base).
    pub fn typed_columns(&self) -> Option<Vec<&TypedColumn>> {
        match &self.data {
            TableData::Columnar { base, extra, .. } if base.num_cols == 0 => Some(extra.iter().collect()),
            _ => None,
        }
    }

    /// Appended Columnar column, borrowed. None otherwise.
    pub fn typed_column(&self, name: &str) -> Option<&TypedColumn> {
        match &self.data {
            TableData::Columnar {
                base,
                extra,
                headers,
                ..
            } => {
                let i = headers.iter().position(|h| h == name)?;
                i.checked_sub(base.num_cols).map(|j| &extra[j])
            }
            _ => None,
        }
    }

    /// Returns reference to column if already cached (Owned only).
    pub fn get_column_cached(&self, name: &str) -> Option<&Vec<Value>> {
        match &self.data {
            TableData::Owned { column_cache, .. } => column_cache.get(name),
            TableData::View { .. } | TableData::Columnar { .. } => None,
        }
    }

    /// Returns column by name (Owned only). Builds and caches from flat on first access.
    pub fn get_column(&mut self, name: &str) -> Option<&Vec<Value>> {
        self.columnar_to_owned();
        match &mut self.data {
            TableData::Owned {
                flat,
                num_cols,
                headers,
                column_cache,
            } => {
                if column_cache.contains_key(name) {
                    return column_cache.get(name);
                }
                let col_idx = headers.iter().position(|h| h == name)?;
                let num_cols = *num_cols;
                let len = if num_cols == 0 {
                    0
                } else {
                    flat.len() / num_cols
                };
                let column: Vec<Value> = (0..len)
                    .map(|row| {
                        let idx = row * num_cols + col_idx;
                        flat.get(idx).cloned().unwrap_or(Value::Null)
                    })
                    .collect();
                column_cache.insert(name.to_string(), column);
                column_cache.get(name)
            }
            TableData::View { .. } | TableData::Columnar { .. } => None,
        }
    }

    /// Expected row width for `add_row`: `num_cols` if set, else `headers.len()` if non-empty.
    pub fn expected_row_width(&self) -> Option<usize> {
        match &self.data {
            TableData::View { num_cols, headers, .. } | TableData::Owned { num_cols, headers, .. } => {
                if *num_cols > 0 {
                    Some(*num_cols)
                } else if !headers.is_empty() {
                    Some(headers.len())
                } else {
                    None
                }
            }
            TableData::Columnar { headers, .. } => Some(headers.len()).filter(|&n| n > 0),
        }
    }

    /// Append one row in-place (Owned only). Row length must match [`Self::expected_row_width`].
    pub fn add_row(&mut self, row: Vec<Value>) -> Result<(), String> {
        let expected = self
            .expected_row_width()
            .ok_or_else(|| "ValueError: cannot add_row to table without columns".to_string())?;
        if row.len() != expected {
            return Err(format!(
                "ValueError: row length {} does not match table columns {}",
                row.len(),
                expected
            ));
        }
        self.columnar_to_owned();
        match &mut self.data {
            TableData::Owned {
                flat,
                num_cols,
                column_cache,
                ..
            } => {
                if *num_cols == 0 {
                    *num_cols = expected;
                }
                flat.extend(row);
                column_cache.clear();
                Ok(())
            }
            TableData::View { .. } | TableData::Columnar { .. } => Err(
                "ReadOnlyError: cannot add_row on table view; materialize first".to_string(),
            ),
        }
    }

    /// Returns row by index (Owned only) as slice into flat storage.
    pub fn get_row(&self, index: usize) -> Option<&[Value]> {
        match &self.data {
            TableData::Owned { flat, num_cols, .. } => {
                let len = if *num_cols == 0 {
                    0
                } else {
                    flat.len() / num_cols
                };
                if index >= len {
                    return None;
                }
                let start = index * num_cols;
                let end = (index + 1) * num_cols;
                if end <= flat.len() {
                    Some(&flat[start..end])
                } else {
                    None
                }
            }
            TableData::Columnar { .. } => {
                let (flat, num_cols, _) = self.owned_parts()?;
                flat.get(index * num_cols..(index + 1) * num_cols)
                    .filter(|_| num_cols > 0)
            }
            TableData::View { .. } => None,
        }
    }

    /// Convert View → Owned in place; no-op when already owned.
    pub fn ensure_owned<F>(&mut self, load: F)
    where
        F: Fn(ValueId) -> Value,
    {
        self.columnar_to_owned();
        if let TableData::View {
            flat_cell_ids,
            num_cols,
            headers,
        } = &self.data
        {
            let flat: Vec<Value> = flat_cell_ids.iter().map(|&id| load(id)).collect();
            self.data = TableData::Owned {
                flat,
                num_cols: *num_cols,
                headers: headers.clone(),
                column_cache: HashMap::new(),
            };
        }
    }

    pub fn clear_column_cache(&mut self) {
        if let TableData::Owned { column_cache, .. } = &mut self.data {
            column_cache.clear();
        }
    }

    /// Owned flat buffer (for bulk append fast paths).
    pub fn owned_flat(&self) -> Option<&[Value]> {
        match &self.data {
            TableData::Owned { flat, .. } => Some(flat.as_slice()),
            TableData::Columnar { .. } => self.owned_parts().map(|(f, _, _)| f.as_slice()),
            TableData::View { .. } => None,
        }
    }

    pub fn owned_num_cols(&self) -> Option<usize> {
        match &self.data {
            TableData::Owned { num_cols, .. } => Some(*num_cols),
            TableData::Columnar { headers, .. } => Some(headers.len()),
            TableData::View { .. } => None,
        }
    }

    /// Column values for an owned table (read-only, no cache mutation).
    pub fn column_values_owned(&self, name: &str) -> Option<Vec<Value>> {
        let col_idx = self.headers().iter().position(|h| h == name)?;
        match &self.data {
            TableData::Owned { flat, num_cols, .. } => {
                let len = self.len();
                let num_cols = *num_cols;
                Some(
                    (0..len)
                        .map(|row| flat[row * num_cols + col_idx].clone())
                        .collect(),
                )
            }
            TableData::Columnar { base, extra, .. } => {
                let len = self.len();
                Some(if col_idx < base.num_cols {
                    (0..len)
                        .map(|row| base.flat[row * base.num_cols + col_idx].clone())
                        .collect()
                } else {
                    extra[col_idx - base.num_cols].to_values_vec()
                })
            }
            TableData::View { .. } => None,
        }
    }

    /// One cell by row and column name, without materializing the column.
    /// None for a View table, an unknown column or a row out of range.
    pub fn cell_value(&self, row: usize, name: &str) -> Option<Value> {
        let col_idx = self.headers().iter().position(|h| h == name)?;
        if row >= self.len() {
            return None;
        }
        match &self.data {
            TableData::Owned { flat, num_cols, .. } => flat.get(row * num_cols + col_idx).cloned(),
            TableData::Columnar { base, extra, .. } => Some(if col_idx < base.num_cols {
                base.flat[row * base.num_cols + col_idx].clone()
            } else {
                extra[col_idx - base.num_cols].cell(row)
            }),
            TableData::View { .. } => None,
        }
    }

    /// Append new columns at the end; existing rows get `Null` in new columns.
    pub fn extend_columns(&mut self, names: &[String]) -> Result<(), String> {
        self.columnar_to_owned();
        match &mut self.data {
            TableData::Owned {
                flat,
                num_cols,
                headers,
                column_cache,
            } => {
                let new_names: Vec<String> = names
                    .iter()
                    .filter(|n| !headers.contains(n))
                    .cloned()
                    .collect();
                if new_names.is_empty() {
                    return Ok(());
                }
                let row_count = if *num_cols == 0 {
                    0
                } else {
                    flat.len() / *num_cols
                };
                let old_num_cols = *num_cols;
                let add = new_names.len();
                let new_num_cols = old_num_cols + add;
                if row_count > 0 {
                    let mut new_flat = Vec::with_capacity(row_count * new_num_cols);
                    for row_idx in 0..row_count {
                        let start = row_idx * old_num_cols;
                        let end = start + old_num_cols;
                        new_flat.extend_from_slice(&flat[start..end]);
                        new_flat.resize(new_flat.len() + add, Value::Null);
                    }
                    *flat = new_flat;
                }
                headers.extend(new_names);
                *num_cols = new_num_cols;
                column_cache.clear();
                Ok(())
            }
            TableData::View { .. } | TableData::Columnar { .. } => Err("table must be owned".to_string()),
        }
    }

    /// Append a contiguous row-major chunk; length must be a multiple of `num_cols`.
    pub fn append_flat_chunk(&mut self, chunk: &[Value]) -> Result<usize, String> {
        self.columnar_to_owned();
        match &mut self.data {
            TableData::Owned {
                flat,
                num_cols,
                column_cache,
                ..
            } => {
                if chunk.is_empty() {
                    return Ok(0);
                }
                if *num_cols == 0 {
                    return Err("table has no columns".to_string());
                }
                if chunk.len() % *num_cols != 0 {
                    return Err(format!(
                        "invalid flat chunk length {} for {} columns",
                        chunk.len(),
                        num_cols
                    ));
                }
                let rows = chunk.len() / *num_cols;
                flat.extend_from_slice(chunk);
                column_cache.clear();
                Ok(rows)
            }
            TableData::View { .. } | TableData::Columnar { .. } => Err("table must be owned".to_string()),
        }
    }

    /// Append rows; each row length must equal `num_cols`.
    pub fn append_rows(&mut self, rows: &[Vec<Value>]) -> Result<usize, String> {
        self.columnar_to_owned();
        match &mut self.data {
            TableData::Owned {
                flat,
                num_cols,
                column_cache,
                ..
            } => {
                if rows.is_empty() {
                    return Ok(0);
                }
                if *num_cols == 0 {
                    return Err("table has no columns".to_string());
                }
                for row in rows {
                    if row.len() != *num_cols {
                        return Err(format!(
                            "Invalid row length\nExpected {} columns\nGot {}",
                            num_cols,
                            row.len()
                        ));
                    }
                }
                for row in rows {
                    flat.extend_from_slice(row);
                }
                column_cache.clear();
                Ok(rows.len())
            }
            TableData::View { .. } | TableData::Columnar { .. } => Err("table must be owned".to_string()),
        }
    }

    /// Clone owned flat buffer with same headers.
    pub fn clone_flat_owned(&self) -> Option<Self> {
        self.clone_flat_with_headers(self.headers().clone())
    }

    /// Map one column on owned storage (single flat clone, in-place column update).
    pub fn map_column_owned<F>(&self, col_idx: usize, mut f: F) -> Option<Self>
    where
        F: FnMut(Value) -> Value,
    {
        match self.owned_parts() {
            Some((flat, num_cols, headers)) => {
                let num_cols = &num_cols;
                let nc = *num_cols;
                if nc == 0 || col_idx >= nc {
                    return None;
                }
                let mut out = flat.clone();
                let n_rows = out.len() / nc;
                for row in 0..n_rows {
                    let idx = row * nc + col_idx;
                    let cell = out[idx].clone();
                    out[idx] = f(cell);
                }
                Some(Table::from_flat_owned(out, nc, headers.clone()))
            }
            None => None,
        }
    }

    /// Replace `Null` cells in the given columns with `replacement` on owned storage.
    pub fn replace_nulls_owned(&self, col_indices: &[usize], replacement: &Value) -> Option<Self> {
        if let TableData::Columnar { base, extra, headers, .. } = &self.data {
            if base.num_cols == 0 && !extra.is_empty() {
                if col_indices.iter().any(|&c| c >= extra.len()) {
                    return None;
                }
                let mut cols = extra.to_vec();
                for &ci in col_indices {
                    cols[ci] = cols[ci].replace_nulls(replacement);
                }
                return Some(Table::from_typed_columns(headers.clone(), cols));
            }
        }
        match self.owned_parts() {
            Some((flat, num_cols, headers)) => {
                let num_cols = &num_cols;
                let nc = *num_cols;
                if col_indices.iter().any(|&c| c >= nc.max(1)) {
                    return None;
                }
                let mut out = flat.clone();
                if nc > 0 {
                    for row in out.chunks_exact_mut(nc) {
                        for &ci in col_indices {
                            if matches!(row[ci], Value::Null) {
                                row[ci] = replacement.clone();
                            }
                        }
                    }
                }
                Some(Table::from_flat_owned(out, nc, headers.clone()))
            }
            None => None,
        }
    }

    /// Append a column in place on owned storage, moving `values` into the buffer.
    /// Returns `values` back if the table is a view or the length does not match.
    pub fn push_column_owned(&mut self, name: &str, values: Vec<Value>) -> Result<(), Vec<Value>> {
        self.columnar_to_owned();
        match &mut self.data {
            TableData::Owned { flat, num_cols, headers, column_cache } => {
                let nc = *num_cols;
                let n_rows = if nc == 0 { 0 } else { flat.len() / nc };
                if values.len() != n_rows || nc == 0 {
                    return Err(values);
                }
                let old = std::mem::take(flat);
                let mut out = Vec::with_capacity(n_rows * (nc + 1));
                let mut cells = old.into_iter();
                for v in values {
                    out.extend(cells.by_ref().take(nc));
                    out.push(v);
                }
                *flat = out;
                *num_cols = nc + 1;
                headers.push(name.to_string());
                column_cache.remove(name);
                Ok(())
            }
            TableData::View { .. } | TableData::Columnar { .. } => Err(values),
        }
    }

    /// Append multiple columns on owned storage in one pass.
    pub fn append_columns_owned(
        &self,
        names: &[String],
        columns: &[Vec<Value>],
    ) -> Option<Self> {
        match self.owned_parts() {
            Some((flat, num_cols, headers)) => {
                let num_cols = &num_cols;
                let nc = *num_cols;
                let n_rows = if nc == 0 { 0 } else { flat.len() / nc };
                if names.len() != columns.len() {
                    return None;
                }
                for col in columns {
                    if col.len() != n_rows {
                        return None;
                    }
                }
                let n_new = names.len();
                let new_nc = if nc == 0 { n_new.max(1) } else { nc + n_new };
                let mut out = Vec::with_capacity(n_rows.saturating_mul(new_nc));
                if nc == 0 {
                    for col in columns {
                        out.extend_from_slice(col);
                    }
                } else {
                    for row in 0..n_rows {
                        let start = row * nc;
                        out.extend_from_slice(&flat[start..start + nc]);
                        for col in columns {
                            out.push(col[row].clone());
                        }
                    }
                }
                let mut new_headers = headers.clone();
                new_headers.extend(names.iter().cloned());
                Some(Table::from_flat_owned(out, new_nc, new_headers))
            }
            None => None,
        }
    }

    /// Clone owned flat buffer with new headers (header-only transforms).
    pub fn clone_flat_with_headers(&self, headers: Vec<String>) -> Option<Self> {
        match self.owned_parts() {
            Some((flat, num_cols, _)) => {
                let num_cols = &num_cols;
                Some(Table::from_flat_owned(flat.clone(), *num_cols, headers))
            }
            None => None,
        }
    }

    /// Contiguous row slice on owned storage.
    pub fn slice_rows_owned(&self, start_row: usize, row_count: usize) -> Option<Self> {
        if let TableData::Columnar { base, extra, headers, .. } = &self.data {
            if !extra.is_empty() {
                let len = columnar_len(base, extra);
                if start_row.saturating_add(row_count) > len {
                    return None;
                }
                return Some(columnar_map(
                    base,
                    extra,
                    headers,
                    |b| {
                        let nc = b.num_cols;
                        Rc::new(SharedRows {
                            flat: b.flat[start_row * nc..(start_row + row_count) * nc].to_vec(),
                            num_cols: nc,
                        })
                    },
                    |c| c.slice(start_row, row_count),
                ));
            }
        }
        match self.owned_parts() {
            Some((flat, num_cols, headers)) => {
                let num_cols = &num_cols;
                let nc = *num_cols;
                if nc == 0 {
                    return Some(Table::from_flat_owned(Vec::new(), 0, headers.clone()));
                }
                if row_count == 0 {
                    return Some(Table::from_flat_owned(Vec::new(), nc, headers.clone()));
                }
                let start = start_row.saturating_mul(nc);
                let end = start.saturating_add(row_count.saturating_mul(nc));
                if end > flat.len() {
                    return None;
                }
                Some(Table::from_flat_owned(
                    flat[start..end].to_vec(),
                    nc,
                    headers.clone(),
                ))
            }
            None => None,
        }
    }

    /// Gather rows by index list on owned storage.
    pub fn gather_rows_owned(&self, indices: &[usize]) -> Option<Self> {
        if let TableData::Columnar { base, extra, headers, .. } = &self.data {
            if !extra.is_empty() {
                // Out-of-range indices are skipped, as on owned storage.
                let len = columnar_len(base, extra);
                let kept;
                let indices = if indices.iter().all(|&r| r < len) {
                    indices
                } else {
                    kept = indices.iter().copied().filter(|&r| r < len).collect::<Vec<_>>();
                    &kept[..]
                };
                return Some(columnar_map(
                    base,
                    extra,
                    headers,
                    |b| {
                        let nc = b.num_cols;
                        let mut flat = Vec::with_capacity(indices.len() * nc);
                        for &r in indices {
                            flat.extend_from_slice(&b.flat[r * nc..(r + 1) * nc]);
                        }
                        Rc::new(SharedRows { flat, num_cols: nc })
                    },
                    |c| c.gather(indices),
                ));
            }
        }
        match self.owned_parts() {
            Some((flat, num_cols, headers)) => {
                let num_cols = &num_cols;
                let nc = *num_cols;
                if nc == 0 {
                    return Some(Table::from_flat_owned(Vec::new(), 0, headers.clone()));
                }
                let mut out = Vec::with_capacity(indices.len().saturating_mul(nc));
                for &row in indices {
                    let start = row.saturating_mul(nc);
                    let end = start + nc;
                    if end <= flat.len() {
                        out.extend_from_slice(&flat[start..end]);
                    }
                }
                Some(Table::from_flat_owned(out, nc, headers.clone()))
            }
            None => None,
        }
    }

    /// Project columns on owned storage.
    pub fn select_columns_owned(
        &self,
        col_indices: &[usize],
        new_headers: Vec<String>,
    ) -> Option<Self> {
        match self.owned_parts() {
            Some((flat, num_cols, _)) => {
                let num_cols = &num_cols;
                let nc = *num_cols;
                if nc == 0 {
                    return Some(Table::from_flat_owned(Vec::new(), 0, new_headers));
                }
                let n_rows = flat.len() / nc;
                let new_nc = col_indices.len().max(1);
                let mut out = Vec::with_capacity(n_rows.saturating_mul(new_nc));
                for row in 0..n_rows {
                    let base = row * nc;
                    for &ci in col_indices {
                        if ci >= nc {
                            return None;
                        }
                        out.push(flat[base + ci].clone());
                    }
                }
                Some(Table::from_flat_owned(out, new_nc, new_headers))
            }
            None => None,
        }
    }

    /// Append a column on owned storage without row-major Vec allocation.
    pub fn append_column_owned(&self, name: &str, values: &[Value]) -> Option<Self> {
        match self.owned_parts() {
            Some((flat, num_cols, headers)) => {
                let num_cols = &num_cols;
                let nc = *num_cols;
                let n_rows = if nc == 0 { 0 } else { flat.len() / nc };
                if values.len() != n_rows {
                    return None;
                }
                let new_nc = if nc == 0 { 1 } else { nc + 1 };
                let mut out = Vec::with_capacity(n_rows.saturating_mul(new_nc));
                if nc == 0 {
                    for v in values {
                        out.push(v.clone());
                    }
                } else {
                    for row in 0..n_rows {
                        let start = row * nc;
                        out.extend_from_slice(&flat[start..start + nc]);
                        out.push(values[row].clone());
                    }
                }
                let mut new_headers = headers.clone();
                new_headers.push(name.to_string());
                Some(Table::from_flat_owned(out, new_nc, new_headers))
            }
            None => None,
        }
    }
}

impl PartialEq for Table {
    fn eq(&self, other: &Self) -> bool {
        self.headers() == other.headers()
            && self.len() == other.len()
            && match (&self.data, &other.data) {
                (
                    TableData::Owned {
                        flat: a,
                        num_cols: nc_a,
                        ..
                    },
                    TableData::Owned {
                        flat: b,
                        num_cols: nc_b,
                        ..
                    },
                ) => nc_a == nc_b && a == b,
                _ => match (self.owned_parts(), other.owned_parts()) {
                    (Some((a, nc_a, _)), Some((b, nc_b, _))) => nc_a == nc_b && a == b,
                    _ => false,
                },
            }
    }
}

/// Row count of a Columnar table.
fn columnar_len(base: &SharedRows, extra: &[TypedColumn]) -> usize {
    if base.num_cols > 0 {
        base.flat.len() / base.num_cols
    } else {
        extra.first().map_or(0, |c| c.len())
    }
}

/// Row-major flat of a Columnar table: base row followed by each extra cell.
fn interleave(base: &SharedRows, extra: &[TypedColumn]) -> Vec<Value> {
    let len = columnar_len(base, extra);
    let width = base.num_cols + extra.len();
    let mut flat = Vec::with_capacity(len * width);
    for r in 0..len {
        if base.num_cols > 0 {
            flat.extend_from_slice(&base.flat[r * base.num_cols..(r + 1) * base.num_cols]);
        }
        // `cell` keeps typed columns from caching a second copy of their cells.
        for col in extra {
            flat.push(col.cell(r));
        }
    }
    flat
}

/// Columnar table built from per-column parts of `table`, keeping its base rows
/// through `base_rows` and transforming every appended column with `col`.
fn columnar_map(
    base: &Rc<SharedRows>,
    extra: &[TypedColumn],
    headers: &[String],
    base_rows: impl FnOnce(&SharedRows) -> Rc<SharedRows>,
    col: impl Fn(&TypedColumn) -> TypedColumn,
) -> Table {
    let base = if base.num_cols == 0 {
        base.clone()
    } else {
        base_rows(base)
    };
    Table {
        data: TableData::Columnar {
            base,
            extra: extra.iter().map(col).collect(),
            headers: headers.to_vec(),
            materialized: Rc::new(OnceCell::new()),
        },
        name: None,
    }
}
