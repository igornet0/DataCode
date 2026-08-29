// Структура данных для таблиц: flat storage (row-major), ColumnView для итерации без материализации.

use crate::common::value::Value;
use crate::common::value_store::ValueId;
use std::cell::RefCell;
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
            _ => false,
        }
    }

    /// Convert a View table to Owned by loading each cell with the given callback.
    /// No-op for already Owned tables. Used so ML dataset() can call get_column on tables created by table() fast path.
    pub fn materialize_with<F>(&self, load: F) -> Self
    where
        F: Fn(ValueId) -> Value,
    {
        match &self.data {
            TableData::Owned { .. } => self.clone(),
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
        }
    }

    /// Returns reference to column if already cached (Owned only).
    pub fn get_column_cached(&self, name: &str) -> Option<&Vec<Value>> {
        match &self.data {
            TableData::Owned { column_cache, .. } => column_cache.get(name),
            TableData::View { .. } => None,
        }
    }

    /// Returns column by name (Owned only). Builds and caches from flat on first access.
    pub fn get_column(&mut self, name: &str) -> Option<&Vec<Value>> {
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
            TableData::View { .. } => None,
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
            TableData::View { .. } => Err(
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
            TableData::View { .. } => None,
        }
    }

    /// Convert View → Owned in place; no-op when already owned.
    pub fn ensure_owned<F>(&mut self, load: F)
    where
        F: Fn(ValueId) -> Value,
    {
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
            TableData::View { .. } => None,
        }
    }

    pub fn owned_num_cols(&self) -> Option<usize> {
        match &self.data {
            TableData::Owned { num_cols, .. } => Some(*num_cols),
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
            TableData::View { .. } => None,
        }
    }

    /// Append new columns at the end; existing rows get `Null` in new columns.
    pub fn extend_columns(&mut self, names: &[String]) -> Result<(), String> {
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
            TableData::View { .. } => Err("table must be owned".to_string()),
        }
    }

    /// Append a contiguous row-major chunk; length must be a multiple of `num_cols`.
    pub fn append_flat_chunk(&mut self, chunk: &[Value]) -> Result<usize, String> {
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
            TableData::View { .. } => Err("table must be owned".to_string()),
        }
    }

    /// Append rows; each row length must equal `num_cols`.
    pub fn append_rows(&mut self, rows: &[Vec<Value>]) -> Result<usize, String> {
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
            TableData::View { .. } => Err("table must be owned".to_string()),
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
        match &self.data {
            TableData::Owned { flat, num_cols, headers, .. } => {
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
            TableData::View { .. } => None,
        }
    }

    /// Append multiple columns on owned storage in one pass.
    pub fn append_columns_owned(
        &self,
        names: &[String],
        columns: &[Vec<Value>],
    ) -> Option<Self> {
        match &self.data {
            TableData::Owned { flat, num_cols, headers, .. } => {
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
            TableData::View { .. } => None,
        }
    }

    /// Clone owned flat buffer with new headers (header-only transforms).
    pub fn clone_flat_with_headers(&self, headers: Vec<String>) -> Option<Self> {
        match &self.data {
            TableData::Owned { flat, num_cols, .. } => {
                Some(Table::from_flat_owned(flat.clone(), *num_cols, headers))
            }
            TableData::View { .. } => None,
        }
    }

    /// Contiguous row slice on owned storage.
    pub fn slice_rows_owned(&self, start_row: usize, row_count: usize) -> Option<Self> {
        match &self.data {
            TableData::Owned { flat, num_cols, headers, .. } => {
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
            TableData::View { .. } => None,
        }
    }

    /// Gather rows by index list on owned storage.
    pub fn gather_rows_owned(&self, indices: &[usize]) -> Option<Self> {
        match &self.data {
            TableData::Owned { flat, num_cols, headers, .. } => {
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
            TableData::View { .. } => None,
        }
    }

    /// Project columns on owned storage.
    pub fn select_columns_owned(
        &self,
        col_indices: &[usize],
        new_headers: Vec<String>,
    ) -> Option<Self> {
        match &self.data {
            TableData::Owned { flat, num_cols, .. } => {
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
            TableData::View { .. } => None,
        }
    }

    /// Append a column on owned storage without row-major Vec allocation.
    pub fn append_column_owned(&self, name: &str, values: &[Value]) -> Option<Self> {
        match &self.data {
            TableData::Owned { flat, num_cols, headers, .. } => {
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
            TableData::View { .. } => None,
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
                _ => false,
            }
    }
}
