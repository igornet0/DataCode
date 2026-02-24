// Структура данных для таблиц: flat storage (row-major), ColumnView для итерации без материализации.

use crate::common::value::Value;
use crate::common::value_store::ValueId;
use std::collections::HashMap;

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
        RowsIter {
            rr: self,
            next: 0,
        }
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
                table.set_headers(h);
            }
            return table;
        }

        let num_cols = data[0].len();
        let headers = headers.unwrap_or_else(|| {
            (0..num_cols)
                .map(|i| format!("Column_{}", i))
                .collect()
        });

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
    pub fn from_flat_view(flat_cell_ids: Vec<ValueId>, num_cols: usize, headers: Vec<String>) -> Self {
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

    fn set_headers(&mut self, headers: Vec<String>) {
        match &mut self.data {
            TableData::View { headers: h, .. } => *h = headers,
            TableData::Owned { headers: h, .. } => *h = headers,
        }
    }

    pub fn is_view(&self) -> bool {
        matches!(self.data, TableData::View { .. })
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
}

impl PartialEq for Table {
    fn eq(&self, other: &Self) -> bool {
        self.headers() == other.headers()
            && self.len() == other.len()
            && match (&self.data, &other.data) {
                (
                    TableData::Owned { flat: a, num_cols: nc_a, .. },
                    TableData::Owned { flat: b, num_cols: nc_b, .. },
                ) => nc_a == nc_b && a == b,
                _ => false,
            }
    }
}
