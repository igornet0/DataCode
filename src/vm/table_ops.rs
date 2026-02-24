// Table access for both View (flat_cell_ids) and Owned (flat). No column materialization for hot paths.

use crate::common::table::{Table, TableData};
use crate::common::value::Value;
use crate::common::value_store::{ValueStore, NULL_VALUE_ID};
use crate::vm::heavy_store::HeavyStore;
use crate::vm::store_convert::load_value;

/// Get one cell at (row_index, col_name). O(1) for flat storage.
pub fn get_cell_value(
    table: &Table,
    row_index: usize,
    col_name: &str,
    store: &ValueStore,
    heap: &HeavyStore,
) -> Option<Value> {
    match &table.data {
        TableData::View {
            flat_cell_ids,
            num_cols,
            headers,
        } => {
            let col_idx = headers.iter().position(|h| h == col_name)?;
            let idx = row_index * num_cols + col_idx;
            let cell_id = *flat_cell_ids.get(idx)?;
            Some(load_value(cell_id, store, heap))
        }
        TableData::Owned { flat, num_cols, headers, .. } => {
            let col_idx = headers.iter().position(|h| h == col_name)?;
            let idx = row_index * num_cols + col_idx;
            Some(flat.get(idx).cloned().unwrap_or(Value::Null))
        }
    }
}

#[inline]
pub fn column_len(table: &Table, col_name: &str) -> Option<usize> {
    if table.has_column(col_name) {
        Some(table.len())
    } else {
        None
    }
}

/// Get a single row as Vec<Value>. View: from flat_cell_ids; Owned: copy from flat slice.
pub fn get_row(
    table: &Table,
    index: usize,
    store: &ValueStore,
    heap: &HeavyStore,
) -> Option<Vec<Value>> {
    match &table.data {
        TableData::View {
            flat_cell_ids,
            num_cols,
            ..
        } => {
            let len = if *num_cols == 0 {
                0
            } else {
                flat_cell_ids.len() / num_cols
            };
            if index >= len {
                return None;
            }
            let start = index * num_cols;
            let end = (index + 1) * num_cols;
            let mut row = Vec::with_capacity(*num_cols);
            for &cid in &flat_cell_ids[start..end] {
                row.push(load_value(cid, store, heap));
            }
            Some(row)
        }
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
            Some(flat[start..end].to_vec())
        }
    }
}

/// Get a column as Vec<Value>. For View: iterates flat_cell_ids (no cache). For Owned: uses get_column.
pub fn get_column(
    table: &mut Table,
    name: &str,
    store: &ValueStore,
    heap: &HeavyStore,
) -> Option<Vec<Value>> {
    match &table.data {
        TableData::View {
            flat_cell_ids,
            num_cols,
            headers,
        } => {
            let col_idx = headers.iter().position(|h| h == name)?;
            let len = if *num_cols == 0 {
                0
            } else {
                flat_cell_ids.len() / num_cols
            };
            let column: Vec<Value> = (0..len)
                .map(|row| {
                    let idx = row * num_cols + col_idx;
                    let cid = flat_cell_ids.get(idx).copied().unwrap_or(NULL_VALUE_ID);
                    load_value(cid, store, heap)
                })
                .collect();
            Some(column)
        }
        TableData::Owned { .. } => table.get_column(name).map(|c| c.clone()),
    }
}

/// Materialize all rows. View: from flat_cell_ids; Owned: clone flat into rows.
pub fn materialize_rows(
    table: &Table,
    store: &ValueStore,
    heap: &HeavyStore,
) -> Vec<Vec<Value>> {
    if table.is_view() {
        (0..table.len())
            .map(|i| get_row(table, i, store, heap).unwrap_or_default())
            .collect()
    } else {
        table
            .rows_ref()
            .map(|r| r.iter().map(|s| s.to_vec()).collect())
            .unwrap_or_default()
    }
}

/// Get column if cached (Owned only). View has no column cache.
pub fn get_column_cached(
    table: &Table,
    name: &str,
    _store: &ValueStore,
    _heap: &HeavyStore,
) -> Option<Vec<Value>> {
    match &table.data {
        TableData::View { .. } => None,
        TableData::Owned { column_cache, .. } => column_cache.get(name).cloned(),
    }
}
