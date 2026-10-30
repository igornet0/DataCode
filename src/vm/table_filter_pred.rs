//! Runtime evaluation of compound table filter predicates (and/or).

use crate::common::table::Table;
use crate::common::typed_column::TypedColumn;
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

/// `eval_pred_node` с заранее найденными индексами колонок (для owned-таблиц).
enum Compiled<'a> {
    False,
    Cmp(usize, String, &'a Value),
    Member(usize, &'a Value, bool),
    Str(usize, String, &'a Value),
    And(Box<Compiled<'a>>, Box<Compiled<'a>>),
    Or(Box<Compiled<'a>>, Box<Compiled<'a>>),
}

fn compile_pred<'a>(headers: &[String], node: &Value, values: &'a [Value]) -> Compiled<'a> {
    let arr = match node {
        Value::Array(rc) => rc.borrow(),
        _ => return Compiled::False,
    };
    let kind = match arr.first() {
        Some(Value::String(s)) => s.as_str(),
        _ => return Compiled::False,
    };
    let col = |v: &Value| match v {
        Value::String(s) => headers.iter().position(|h| h == s.as_str()),
        _ => None,
    };
    let val = |v: &Value| match v {
        Value::Number(n) => Some(values.get(*n as usize).unwrap_or(&Value::Null)),
        _ => None,
    };
    let op = |v: &Value| match v {
        Value::String(s) => Some(s.as_str().to_string()),
        _ => None,
    };
    match kind {
        "cmp" | "str" if arr.len() >= 4 => {
            let c = col(&arr[1]);
            let o = op(&arr[2]);
            let v = val(&arr[3]);
            match (o, v, c) {
                (Some(o), Some(v), Some(c)) if kind == "cmp" => Compiled::Cmp(c, o, v),
                (Some(o), Some(v), Some(c)) => Compiled::Str(c, o, v),
                _ => Compiled::False,
            }
        }
        "member" if arr.len() >= 4 => {
            let negate = matches!(&arr[3], Value::Number(n) if *n != 0.0);
            match (val(&arr[2]), col(&arr[1])) {
                (Some(v), Some(c)) => Compiled::Member(c, v, negate),
                _ => Compiled::False,
            }
        }
        "and" | "or" if arr.len() >= 3 => {
            let a = Box::new(compile_pred(headers, &arr[1], values));
            let b = Box::new(compile_pred(headers, &arr[2], values));
            if kind == "and" { Compiled::And(a, b) } else { Compiled::Or(a, b) }
        }
        _ => Compiled::False,
    }
}

fn eval_compiled(node: &Compiled, row: &[Value]) -> bool {
    match node {
        Compiled::False => false,
        Compiled::Cmp(c, op, v) => row.get(*c).is_some_and(|cell| cell_matches(cell, op, v)),
        Compiled::Member(c, container, negate) => row.get(*c).is_some_and(|cell| {
            match value_in_container(cell, container, 0) {
                Ok(found) => found != *negate,
                Err(_) => false,
            }
        }),
        Compiled::Str(c, op, p) => row.get(*c).is_some_and(|cell| match_cell_string(cell, op, p)),
        Compiled::And(a, b) => eval_compiled(a, row) && eval_compiled(b, row),
        Compiled::Or(a, b) => eval_compiled(a, row) || eval_compiled(b, row),
    }
}

/// Результат `cell_matches` по готовому `Ordering` (op уже проверен на допустимость).
#[inline]
fn ord_matches(op: &str, o: Ordering) -> bool {
    match op {
        ">" => o == Ordering::Greater,
        "<" => o == Ordering::Less,
        ">=" => o != Ordering::Less,
        "<=" => o != Ordering::Greater,
        "==" | "=" => o == Ordering::Equal,
        "!=" | "<>" => o != Ordering::Equal,
        _ => false,
    }
}

/// Порядок `(rank, x)` из `numeric_sort_key`: тот же, что у `cmp_numeric_values`.
#[inline]
fn key_cmp(a: (u8, f64), b: (u8, f64)) -> Ordering {
    a.0.cmp(&b.0)
        .then_with(|| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal))
}

#[inline]
fn f64_key(x: f64) -> (u8, f64) {
    if x.is_nan() {
        (3, 0.0)
    } else if x == f64::INFINITY {
        (2, 0.0)
    } else if x == f64::NEG_INFINITY {
        (0, 0.0)
    } else {
        (1, x)
    }
}

/// Dict-колонка: предикат считается один раз на элемент словаря, затем раскладывается по кодам.
fn dict_mask(codes: &[u32], dict: &[Value], f: impl Fn(&Value) -> bool) -> Vec<bool> {
    let per_entry: Vec<bool> = dict.iter().map(f).collect();
    codes.iter().map(|&c| per_entry[c as usize]).collect()
}

/// Сравнение одной колонки целиком; `out[i] = cell_matches(col[i], op, v)`.
fn cmp_column_mask(col: &TypedColumn, op: &str, v: &Value, out: &mut [bool]) {
    use crate::common::numeric::numeric_sort_key;
    let nulls = |n: &Option<Rc<[bool]>>, i: usize| n.as_ref().is_some_and(|m| m[i]);
    match (col, numeric_sort_key(v)) {
        // null < число (value_partial_cmp), поэтому null-ячейка даёт Ordering::Less.
        (TypedColumn::F64 { data, nulls: nm, .. }, Some(kv)) => {
            for (i, o) in out.iter_mut().enumerate() {
                let ord = if nulls(nm, i) { Ordering::Less } else { key_cmp(f64_key(data[i]), kv) };
                *o = ord_matches(op, ord);
            }
        }
        (TypedColumn::I64 { data, nulls: nm, .. }, Some(kv)) => {
            for (i, o) in out.iter_mut().enumerate() {
                let ord = if nulls(nm, i) { Ordering::Less } else { key_cmp((1, data[i] as f64), kv) };
                *o = ord_matches(op, ord);
            }
        }
        (TypedColumn::Mixed(cells), _) => {
            for (o, cell) in out.iter_mut().zip(cells.iter()) {
                *o = cell_matches(cell, op, v);
            }
        }
        (TypedColumn::Dict { codes, dict, .. }, _) => {
            out.copy_from_slice(&dict_mask(codes, dict, |cell| cell_matches(cell, op, v)));
        }
        _ => {
            for (i, o) in out.iter_mut().enumerate() {
                *o = cell_matches(&col.cell(i), op, v);
            }
        }
    }
}

/// `eval_compiled` по типизированным колонкам, поколоночно: маска на лист, затем and/or масок.
fn eval_typed_mask(node: &Compiled, cols: &[&TypedColumn], n: usize) -> Vec<bool> {
    match node {
        Compiled::False => vec![false; n],
        Compiled::Cmp(c, op, v) => {
            let mut out = vec![false; n];
            if let Some(col) = cols.get(*c) {
                cmp_column_mask(col, op, v, &mut out);
            }
            out
        }
        Compiled::Member(c, container, negate) => match cols.get(*c) {
            Some(TypedColumn::Dict { codes, dict, .. }) => dict_mask(codes, dict, |cell| {
                match value_in_container(cell, container, 0) {
                    Ok(found) => found != *negate,
                    Err(_) => false,
                }
            }),
            Some(col) => (0..n)
                .map(|i| match value_in_container(&col.cell(i), container, 0) {
                    Ok(found) => found != *negate,
                    Err(_) => false,
                })
                .collect(),
            None => vec![false; n],
        },
        Compiled::Str(c, op, p) => match cols.get(*c) {
            Some(TypedColumn::Mixed(cells)) => cells.iter().map(|cell| match_cell_string(cell, op, p)).collect(),
            Some(TypedColumn::Dict { codes, dict, .. }) => dict_mask(codes, dict, |cell| match_cell_string(cell, op, p)),
            Some(col) => (0..n).map(|i| match_cell_string(&col.cell(i), op, p)).collect(),
            None => vec![false; n],
        },
        Compiled::And(a, b) => {
            let mut m = eval_typed_mask(a, cols, n);
            for (x, y) in m.iter_mut().zip(eval_typed_mask(b, cols, n)) {
                *x &= y;
            }
            m
        }
        Compiled::Or(a, b) => {
            let mut m = eval_typed_mask(a, cols, n);
            for (x, y) in m.iter_mut().zip(eval_typed_mask(b, cols, n)) {
                *x |= y;
            }
            m
        }
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

    if !is_view {
        let table_ref = table.borrow();
        if let Some(cols) = table_ref.typed_columns() {
            let compiled = compile_pred(&headers, pred, values);
            let mask = eval_typed_mask(&compiled, &cols, n_rows);
            let matching_indices: Vec<usize> =
                mask.iter().enumerate().filter_map(|(i, &k)| k.then_some(i)).collect();
            if !matching_indices.is_empty() {
                if let Some(t) = table_ref.gather_rows_owned(&matching_indices) {
                    return Value::Table(Rc::new(RefCell::new(t)));
                }
            }
            return Value::Table(Rc::new(RefCell::new(Table::from_data(Vec::new(), Some(headers)))));
        }
        if let Some((flat, nc, _)) = table_ref.owned_parts() {
            // Колонки разрешаются один раз; ячейки читаются из плоского буфера без клонов.
            let compiled = compile_pred(&headers, pred, values);
            let matching_indices: Vec<usize> = (0..n_rows)
                .filter(|&i| {
                    let start = i * nc;
                    flat.get(start..start + nc)
                        .is_some_and(|row| eval_compiled(&compiled, row))
                })
                .collect();
            if !matching_indices.is_empty() {
                if let Some(t) = table_ref.gather_rows_owned(&matching_indices) {
                    return Value::Table(Rc::new(RefCell::new(t)));
                }
            }
            return Value::Table(Rc::new(RefCell::new(Table::from_data(Vec::new(), Some(headers)))));
        }
    }

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
