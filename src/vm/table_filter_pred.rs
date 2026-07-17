//! Runtime evaluation of compound table filter predicates (and/or).

use crate::common::table::Table;
use crate::common::value::Value;
use crate::vm::membership::value_in_container;
use crate::vm::natives::table::compare_values;
use crate::vm::string_match::match_cell_string;
use std::cell::RefCell;
use std::cmp::Ordering;
use std::rc::Rc;

fn row_cell(table: &Rc<RefCell<Table>>, row_idx: usize, column: &str) -> Option<Value> {
    let table_ref = table.borrow();
    let col_idx = table_ref.headers().iter().position(|h| h == column)?;
    if table_ref.is_view() {
        drop(table_ref);
        crate::vm::vm::with_current_stores(|store, heap| {
            let mut t = table.borrow_mut();
            crate::vm::table_ops::get_column(&mut *t, column, store, heap)
                .and_then(|col| col.get(row_idx).cloned())
        })
    } else {
        table_ref
            .get_row(row_idx)
            .and_then(|row| row.get(col_idx).cloned())
    }
}

fn cell_matches(cell: &Value, operator: &str, filter_value: &Value) -> bool {
    match operator {
        ">" => compare_values(cell, filter_value) == Ordering::Greater,
        "<" => compare_values(cell, filter_value) == Ordering::Less,
        ">=" => {
            let cmp = compare_values(cell, filter_value);
            cmp == Ordering::Greater || cmp == Ordering::Equal
        }
        "<=" => {
            let cmp = compare_values(cell, filter_value);
            cmp == Ordering::Less || cmp == Ordering::Equal
        }
        "==" | "=" => compare_values(cell, filter_value) == Ordering::Equal,
        "!=" | "<>" => compare_values(cell, filter_value) != Ordering::Equal,
        _ => false,
    }
}

fn eval_pred_node(
    table: &Rc<RefCell<Table>>,
    row_idx: usize,
    node: &Value,
    values: &[Value],
) -> bool {
    let arr = match node {
        Value::Array(rc) => rc.borrow(),
        _ => return false,
    };
    if arr.is_empty() {
        return false;
    }
    let kind = match &arr[0] {
        Value::String(s) => s.as_str(),
        _ => return false,
    };
    match kind {
        "cmp" => {
            if arr.len() < 4 {
                return false;
            }
            let column = match &arr[1] {
                Value::String(s) => s.as_str(),
                _ => return false,
            };
            let op = match &arr[2] {
                Value::String(s) => s.as_str(),
                _ => return false,
            };
            let vi = match &arr[3] {
                Value::Number(n) => *n as usize,
                _ => return false,
            };
            let filter_value = values.get(vi).unwrap_or(&Value::Null);
            match row_cell(table, row_idx, column) {
                Some(cell) => cell_matches(&cell, op, filter_value),
                None => false,
            }
        }
        "member" => {
            if arr.len() < 4 {
                return false;
            }
            let column = match &arr[1] {
                Value::String(s) => s.as_str(),
                _ => return false,
            };
            let vi = match &arr[2] {
                Value::Number(n) => *n as usize,
                _ => return false,
            };
            let negate = matches!(&arr[3], Value::Number(n) if *n != 0.0);
            let container = values.get(vi).unwrap_or(&Value::Null);
            match row_cell(table, row_idx, column) {
                Some(cell) => match value_in_container(&cell, container, 0) {
                    Ok(found) => if negate { !found } else { found },
                    Err(_) => false,
                },
                None => false,
            }
        }
        "str" => {
            if arr.len() < 4 {
                return false;
            }
            let column = match &arr[1] {
                Value::String(s) => s.as_str(),
                _ => return false,
            };
            let op = match &arr[2] {
                Value::String(s) => s.as_str(),
                _ => return false,
            };
            let vi = match &arr[3] {
                Value::Number(n) => *n as usize,
                _ => return false,
            };
            let pattern = values.get(vi).unwrap_or(&Value::Null);
            match row_cell(table, row_idx, column) {
                Some(cell) => match_cell_string(&cell, op, pattern),
                None => false,
            }
        }
        "and" => {
            if arr.len() < 3 {
                return false;
            }
            eval_pred_node(table, row_idx, &arr[1], values)
                && eval_pred_node(table, row_idx, &arr[2], values)
        }
        "or" => {
            if arr.len() < 3 {
                return false;
            }
            eval_pred_node(table, row_idx, &arr[1], values)
                || eval_pred_node(table, row_idx, &arr[2], values)
        }
        _ => false,
    }
}

/// Filter `table` by compound predicate `pred` (array tree) with leaf values in `values`.
pub fn table_filter_pred_impl(
    table: &Rc<RefCell<Table>>,
    pred: &Value,
    values: &[Value],
) -> Value {
    let table_ref = table.borrow();
    let headers = table_ref.headers().clone();
    let n_rows = table_ref.len();
    let is_view = table_ref.is_view();
    drop(table_ref);

    let matching_indices: Vec<usize> = (0..n_rows)
        .filter(|&i| eval_pred_node(table, i, pred, values))
        .collect();

    let new_rows: Vec<Vec<Value>> = if is_view {
        crate::vm::vm::with_current_stores(|store, heap| {
            let t = table.borrow();
            matching_indices
                .iter()
                .filter_map(|&idx| crate::vm::table_ops::get_row(&*t, idx, store, heap))
                .collect()
        })
    } else {
        let table_ref = table.borrow();
        matching_indices
            .iter()
            .filter_map(|&idx| table_ref.get_row(idx).map(|r| r.to_vec()))
            .collect()
    };

    let new_table = Table::from_data(new_rows, Some(headers));
    Value::Table(Rc::new(RefCell::new(new_table)))
}
