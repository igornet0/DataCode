// Table access for both View (flat_cell_ids) and Owned (flat). No column materialization for hot paths.

use crate::common::range_args::range_len;
use crate::common::TaggedValue;
use crate::common::table::{Table, TableData};
use crate::common::value::{IterableInner, Value};
use crate::common::value_store::{ValueCell, ValueId, ValueStore, NULL_VALUE_ID};
use crate::vm::heavy_store::HeavyStore;
use crate::vm::store_convert::{load_value, store_value, tagged_to_value_id};

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
        TableData::Owned {
            flat,
            num_cols,
            headers,
            ..
        } => {
            let col_idx = headers.iter().position(|h| h == col_name)?;
            let idx = row_index * num_cols + col_idx;
            Some(flat.get(idx).cloned().unwrap_or(Value::Null))
        }
    }
}

/// Cells from `col_names` at `row_index`, in that order. `None` if any column or cell is missing.
pub fn row_cells(
    table: &Table,
    row_index: usize,
    col_names: &[String],
    store: &ValueStore,
    heap: &HeavyStore,
) -> Option<Vec<Value>> {
    let mut out = Vec::with_capacity(col_names.len());
    for name in col_names {
        out.push(get_cell_value(table, row_index, name, store, heap)?);
    }
    Some(out)
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
pub fn materialize_rows(table: &Table, store: &ValueStore, heap: &HeavyStore) -> Vec<Vec<Value>> {
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

/// Build row-major flat cell IDs for `table(data, headers)` View fast path.
/// Supports row-oriented `[[r1],[r2],…]` and column-oriented `[col1, col2, …]`.
fn view_column_len(value: &Value) -> Option<usize> {
    match value {
        Value::Array(a) => Some(a.borrow().len()),
        Value::Iterable(rc) => match &*rc.borrow() {
            IterableInner::Range {
                current,
                end,
                step,
            } => Some(range_len(*current, *end, *step)),
            _ => None,
        },
        _ => None,
    }
}

fn view_column_get(value: &Value, row: usize) -> Value {
    match value {
        Value::Array(a) => a
            .borrow()
            .get(row)
            .cloned()
            .unwrap_or(Value::Null),
        Value::Iterable(rc) => match &*rc.borrow() {
            IterableInner::Range {
                current,
                end,
                step,
            } => {
                let val = current + row as i64 * step;
                if (*step > 0 && val >= *end) || (*step < 0 && val <= *end) {
                    Value::Null
                } else {
                    Value::Number(val as f64)
                }
            }
            _ => Value::Null,
        },
        _ => Value::Null,
    }
}

fn view_slot_array_len(
    row_tv: TaggedValue,
    value_store: &mut ValueStore,
    heap: &mut HeavyStore,
) -> Option<usize> {
    if !row_tv.is_heap() {
        return None;
    }
    let id = tagged_to_value_id(row_tv, value_store);
    view_column_len(&load_value(id, value_store, heap))
}

pub fn build_view_flat_from_row_slots(
    row_slots: &[TaggedValue],
    header_count: usize,
    value_store: &mut ValueStore,
    heap: &mut HeavyStore,
) -> Option<(Vec<ValueId>, usize)> {
    if row_slots.is_empty() || header_count == 0 {
        return None;
    }
    let first_len = view_slot_array_len(row_slots[0], value_store, heap)?;
    if first_len == 0 {
        return None;
    }

    let column_oriented =
        row_slots.len() == header_count && row_slots.len() != first_len;

    let mut flat_values: Vec<Value> = if column_oriented {
        let n_cols = header_count;
        let n_rows = first_len;
        for &row_tv in row_slots.iter() {
            if view_slot_array_len(row_tv, value_store, heap)? != n_rows {
                return None;
            }
        }
        let mut out = Vec::with_capacity(n_rows * n_cols);
        for r in 0..n_rows {
            for c in 0..n_cols {
                let col_id = tagged_to_value_id(row_slots[c], value_store);
                let col_val = load_value(col_id, value_store, heap);
                out.push(view_column_get(&col_val, r));
            }
        }
        out
    } else {
        let n_cols = first_len;
        let mut out = Vec::with_capacity(row_slots.len() * n_cols);
        for &row_tv in row_slots.iter() {
            let row_id = tagged_to_value_id(row_tv, value_store);
            let row = match load_value(row_id, value_store, heap) {
                Value::Array(a) => a.borrow().clone(),
                _ => return None,
            };
            if row.len() < n_cols {
                return None;
            }
            for i in 0..n_cols {
                out.push(row.get(i).cloned().unwrap_or(Value::Null));
            }
        }
        out
    };

    let num_cols = if column_oriented {
        header_count
    } else {
        first_len
    };
    if flat_values.len() % num_cols.max(1) != 0 {
        return None;
    }
    let flat_cell_ids: Vec<ValueId> = flat_values
        .into_iter()
        .map(|v| store_value(v, value_store, heap))
        .collect();
    Some((flat_cell_ids, num_cols.max(1)))
}
