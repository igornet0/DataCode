// JOIN operations native functions

use crate::common::numeric::IntValue;
use crate::common::table::Table;
use crate::common::value::Value;
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::hash::Hash;
use std::rc::Rc;

/// If table is View, materialize so join can use rows_ref(). Owned tables are shared by Rc.
fn ensure_owned_rc(table: &Rc<RefCell<Table>>) -> Rc<RefCell<Table>> {
    if !table.borrow().is_view() {
        return Rc::clone(table);
    }
    let owned = crate::vm::vm::with_current_stores(|store, heap| {
        table
            .borrow()
            .materialize_with(|id| crate::vm::store_convert::load_value(id, store, heap))
    });
    Rc::new(RefCell::new(owned))
}

fn table_arg(value: &Value) -> Option<Rc<RefCell<Table>>> {
    match value {
        Value::Table(t) => Some(ensure_owned_rc(t)),
        _ => None,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum JoinType {
    Inner,
    Left,
    Right,
    Full,
    Cross,
    Semi,
    Anti,
}

#[derive(Debug, Clone)]
struct JoinKey {
    left_col: String,
    right_col: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct KeyHash {
    values: Vec<Value>,
}

/// Precomputed column indices for join keys.
struct JoinKeyPlan {
    left_indices: Vec<usize>,
    right_indices: Vec<usize>,
}

impl JoinKeyPlan {
    fn new(left_table: &Table, right_table: &Table, keys: &[JoinKey]) -> Option<Self> {
        let mut left_indices = Vec::with_capacity(keys.len());
        let mut right_indices = Vec::with_capacity(keys.len());
        for key in keys {
            let li = left_table.headers().iter().position(|h| h == &key.left_col)?;
            let ri = right_table.headers().iter().position(|h| h == &key.right_col)?;
            left_indices.push(li);
            right_indices.push(ri);
        }
        Some(Self {
            left_indices,
            right_indices,
        })
    }
}

const JOIN_NULL_KEY: u64 = u64::MAX;

fn value_to_join_bits(val: &Value, nulls_equal: bool) -> Option<u64> {
    match val {
        Value::Null if nulls_equal => Some(JOIN_NULL_KEY),
        Value::Null => None,
        Value::Number(n) => Some(n.to_bits()),
        Value::Int(IntValue::Finite(n)) => Some(*n as u64),
        _ => None,
    }
}

enum JoinIndex {
    Numeric(HashMap<u64, Vec<usize>>),
    General(HashMap<KeyHash, Vec<usize>>),
}

impl JoinIndex {
    fn contains(
        &self,
        row: &[Value],
        key_indices: &[usize],
        nulls_equal: bool,
    ) -> bool {
        match self {
            JoinIndex::Numeric(map) => extract_numeric_key(row, key_indices, nulls_equal)
                .map(|k| map.contains_key(&k))
                .unwrap_or(false),
            JoinIndex::General(map) => extract_general_key(row, key_indices)
                .map(|k| map.contains_key(&k))
                .unwrap_or(false),
        }
    }

    fn lookup<'a>(
        &'a self,
        row: &[Value],
        key_indices: &[usize],
        nulls_equal: bool,
    ) -> Option<&'a Vec<usize>> {
        match self {
            JoinIndex::Numeric(map) => {
                let key = extract_numeric_key(row, key_indices, nulls_equal)?;
                map.get(&key)
            }
            JoinIndex::General(map) => {
                let key = extract_general_key(row, key_indices)?;
                map.get(&key)
            }
        }
    }
}

fn extract_numeric_key(row: &[Value], key_indices: &[usize], nulls_equal: bool) -> Option<u64> {
    let idx = *key_indices.first()?;
    value_to_join_bits(row.get(idx)?, nulls_equal)
}

fn extract_general_key(row: &[Value], key_indices: &[usize]) -> Option<KeyHash> {
    let mut key_values = Vec::with_capacity(key_indices.len());
    for &idx in key_indices {
        key_values.push(row.get(idx)?.clone());
    }
    Some(KeyHash { values: key_values })
}

fn build_join_index(
    table: &Table,
    key_indices: &[usize],
    nulls_equal: bool,
) -> JoinIndex {
    let use_numeric = key_indices.len() == 1;
    let rows = table.rows_ref().unwrap();

    if use_numeric {
        let col_idx = key_indices[0];
        let mut map = HashMap::new();
        for (row_idx, row) in rows.iter().enumerate() {
            if let Some(bits) =
                value_to_join_bits(row.get(col_idx).unwrap_or(&Value::Null), nulls_equal)
            {
                map.entry(bits).or_insert_with(Vec::new).push(row_idx);
            }
        }
        JoinIndex::Numeric(map)
    } else {
        let mut map = HashMap::new();
        for (row_idx, row) in rows.iter().enumerate() {
            let mut key_values = Vec::with_capacity(key_indices.len());
            let mut valid_key = true;
            for &col_idx in key_indices {
                let val = row.get(col_idx).unwrap_or(&Value::Null);
                if matches!(val, Value::Null) && !nulls_equal {
                    valid_key = false;
                    break;
                }
                key_values.push(val.clone());
            }
            if valid_key {
                map.entry(KeyHash { values: key_values })
                    .or_insert_with(Vec::new)
                    .push(row_idx);
            }
        }
        JoinIndex::General(map)
    }
}

struct FlatJoinBuilder {
    flat: Vec<Value>,
    num_cols: usize,
}

impl FlatJoinBuilder {
    fn new(estimated_rows: usize, left_cols: usize, right_cols: usize) -> Self {
        let num_cols = left_cols + right_cols;
        Self {
            flat: Vec::with_capacity(estimated_rows.saturating_mul(num_cols)),
            num_cols,
        }
    }

    fn push_lr(&mut self, left_row: &[Value], right_row: &[Value]) {
        self.flat.extend_from_slice(left_row);
        self.flat.extend_from_slice(right_row);
    }

    fn push_left_null_right(&mut self, left_row: &[Value], null_right: &[Value]) {
        self.flat.extend_from_slice(left_row);
        self.flat.extend_from_slice(null_right);
    }

    fn push_null_left_right(&mut self, null_left: &[Value], right_row: &[Value]) {
        self.flat.extend_from_slice(null_left);
        self.flat.extend_from_slice(right_row);
    }

    fn into_table(self, headers: Vec<String>) -> Table {
        Table::from_flat_owned(self.flat, self.num_cols, headers)
    }
}

fn join_result_headers(
    left_table: &Table,
    right_table: &Table,
    left_alias: &str,
    right_alias: &str,
) -> Vec<String> {
    let left_headers_set: HashSet<String> = left_table.headers().iter().cloned().collect();
    let right_headers_set: HashSet<String> = right_table.headers().iter().cloned().collect();
    let mut headers = apply_column_aliases(&left_table.headers(), left_alias, &right_headers_set);
    headers.extend(apply_column_aliases(
        &right_table.headers(),
        right_alias,
        &left_headers_set,
    ));
    headers
}

fn join_table_value(builder: FlatJoinBuilder, headers: Vec<String>) -> Value {
    Value::Table(Rc::new(RefCell::new(builder.into_table(headers))))
}

fn asof_time_f64(v: &Value) -> Option<f64> {
    match v {
        Value::Number(n) => Some(*n),
        Value::Int(IntValue::Finite(n)) => Some(*n as f64),
        _ => None,
    }
}

/// Last index in sorted `times` with `times[i] <= target` (backward asof).
fn asof_backward_idx(times: &[f64], target: f64) -> Option<usize> {
    if times.is_empty() {
        return None;
    }
    let mut lo = 0usize;
    let mut hi = times.len();
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if times[mid] <= target {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    if lo == 0 { None } else { Some(lo - 1) }
}

/// First index in sorted `times` with `times[i] >= target` (forward asof).
fn asof_forward_idx(times: &[f64], target: f64) -> Option<usize> {
    if times.is_empty() {
        return None;
    }
    let mut lo = 0usize;
    let mut hi = times.len();
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if times[mid] < target {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    if lo >= times.len() { None } else { Some(lo) }
}

/// Nearest by absolute distance; ties prefer the earlier index.
fn asof_nearest_idx(times: &[f64], target: f64) -> Option<usize> {
    if times.is_empty() {
        return None;
    }
    let backward = asof_backward_idx(times, target);
    let forward = asof_forward_idx(times, target);
    match (backward, forward) {
        (Some(b), Some(f)) => {
            let db = (target - times[b]).abs();
            let df = (times[f] - target).abs();
            if db <= df { Some(b) } else { Some(f) }
        }
        (Some(b), None) => Some(b),
        (None, Some(f)) => Some(f),
        (None, None) => None,
    }
}

fn asof_pick_idx(times: &[f64], target: f64, direction: &str) -> Option<usize> {
    match direction {
        "forward" => asof_forward_idx(times, target),
        "nearest" => asof_nearest_idx(times, target),
        _ => asof_backward_idx(times, target),
    }
}

use super::table::compare_values;

// Парсинг ключей JOIN из Value
fn parse_join_keys(
    value: &Value,
    left_table: &Table,
    right_table: &Table,
) -> Result<Vec<JoinKey>, String> {
    match value {
        Value::String(col_name) => {
            if left_table.has_column(col_name) && right_table.has_column(col_name) {
                Ok(vec![JoinKey {
                    left_col: col_name.clone(),
                    right_col: col_name.clone(),
                }])
            } else {
                Err(format!("Column '{}' not found in both tables", col_name))
            }
        }
        Value::Array(arr) => {
            let arr_ref = arr.borrow();
            if arr_ref.is_empty() {
                return Err("Join keys array cannot be empty".to_string());
            }

            // Специальная обработка: если массив содержит ровно две строки,
            // интерпретируем их как пару [left_col, right_col]
            if arr_ref.len() == 2 {
                if let (Value::String(left_col), Value::String(right_col)) =
                    (&arr_ref[0], &arr_ref[1])
                {
                    if !left_table.has_column(left_col) {
                        return Err(format!("Column '{}' not found in left table", left_col));
                    }
                    if !right_table.has_column(right_col) {
                        return Err(format!("Column '{}' not found in right table", right_col));
                    }
                    return Ok(vec![JoinKey {
                        left_col: left_col.clone(),
                        right_col: right_col.clone(),
                    }]);
                }
            }

            let mut keys = Vec::new();
            for item in arr_ref.iter() {
                match item {
                    Value::Array(tuple) => {
                        let tuple_ref = tuple.borrow();
                        if tuple_ref.len() != 2 {
                            return Err("Join key tuple must have exactly 2 elements".to_string());
                        }
                        let left_col = match &tuple_ref[0] {
                            Value::String(s) => s.clone(),
                            _ => return Err("Join key must be a string".to_string()),
                        };
                        let right_col = match &tuple_ref[1] {
                            Value::String(s) => s.clone(),
                            _ => return Err("Join key must be a string".to_string()),
                        };

                        if !left_table.has_column(&left_col) {
                            return Err(format!("Column '{}' not found in left table", left_col));
                        }
                        if !right_table.has_column(&right_col) {
                            return Err(format!("Column '{}' not found in right table", right_col));
                        }

                        keys.push(JoinKey {
                            left_col,
                            right_col,
                        });
                    }
                    Value::String(col_name) => {
                        // Одиночная строка в массиве - автоматическое сопоставление
                        if left_table.has_column(col_name) && right_table.has_column(col_name) {
                            keys.push(JoinKey {
                                left_col: col_name.clone(),
                                right_col: col_name.clone(),
                            });
                        } else {
                            return Err(format!("Column '{}' not found in both tables", col_name));
                        }
                    }
                    _ => {
                        return Err(
                            "Join key must be a string or tuple [string, string]".to_string()
                        )
                    }
                }
            }
            Ok(keys)
        }
        _ => Err("Join keys must be a string or array of tuples".to_string()),
    }
}

// Применение алиасов таблиц к именам колонок для разрешения коллизий
fn apply_column_aliases(
    headers: &[String],
    alias: &str,
    existing_names: &HashSet<String>,
) -> Vec<String> {
    headers
        .iter()
        .map(|h| {
            if existing_names.contains(h) {
                format!("{}.{}", alias, h) // Конфликт: left.id
            } else {
                h.clone() // Нет конфликта: name
            }
        })
        .collect()
}

// Выполнение INNER JOIN
fn perform_inner_join(
    left_table: &Table,
    right_table: &Table,
    keys: &[JoinKey],
    left_alias: &str,
    right_alias: &str,
    nulls_equal: bool,
) -> Value {
    let Some(plan) = JoinKeyPlan::new(left_table, right_table, keys) else {
        return Value::Null;
    };
    let right_index = build_join_index(right_table, &plan.right_indices, nulls_equal);
    let result_headers = join_result_headers(left_table, right_table, left_alias, right_alias);
    let left_cols = left_table.headers().len();
    let right_cols = right_table.headers().len();
    let mut builder = FlatJoinBuilder::new(left_table.len(), left_cols, right_cols);

    let left_rr = left_table.rows_ref().unwrap();
    let right_rr = right_table.rows_ref().unwrap();
    for left_row in left_rr.iter() {
        if let Some(right_indices) = right_index.lookup(left_row, &plan.left_indices, nulls_equal) {
            for &right_idx in right_indices {
                if let Some(right_row) = right_rr.row(right_idx) {
                    builder.push_lr(left_row, right_row);
                }
            }
        }
    }

    join_table_value(builder, result_headers)
}

// Выполнение LEFT JOIN
fn perform_left_join(
    left_table: &Table,
    right_table: &Table,
    keys: &[JoinKey],
    left_alias: &str,
    right_alias: &str,
    nulls_equal: bool,
) -> Value {
    let Some(plan) = JoinKeyPlan::new(left_table, right_table, keys) else {
        return Value::Null;
    };
    let right_index = build_join_index(right_table, &plan.right_indices, nulls_equal);
    let result_headers = join_result_headers(left_table, right_table, left_alias, right_alias);
    let left_cols = left_table.headers().len();
    let right_cols = right_table.headers().len();
    let null_right_row: Vec<Value> = vec![Value::Null; right_cols];
    let mut builder = FlatJoinBuilder::new(left_table.len(), left_cols, right_cols);

    let left_rr = left_table.rows_ref().unwrap();
    let right_rr = right_table.rows_ref().unwrap();
    for left_row in left_rr.iter() {
        if let Some(right_indices) = right_index.lookup(left_row, &plan.left_indices, nulls_equal) {
            for &right_idx in right_indices {
                if let Some(right_row) = right_rr.row(right_idx) {
                    builder.push_lr(left_row, right_row);
                }
            }
        } else {
            builder.push_left_null_right(left_row, &null_right_row);
        }
    }

    join_table_value(builder, result_headers)
}

// Выполнение RIGHT JOIN
fn perform_right_join(
    left_table: &Table,
    right_table: &Table,
    keys: &[JoinKey],
    left_alias: &str,
    right_alias: &str,
    nulls_equal: bool,
) -> Value {
    let Some(plan) = JoinKeyPlan::new(left_table, right_table, keys) else {
        return Value::Null;
    };
    // Index left table, probe from right rows (right join semantics, left|right column order).
    let left_index = build_join_index(left_table, &plan.left_indices, nulls_equal);
    let result_headers = join_result_headers(left_table, right_table, left_alias, right_alias);
    let left_cols = left_table.headers().len();
    let right_cols = right_table.headers().len();
    let null_left_row: Vec<Value> = vec![Value::Null; left_cols];
    let mut builder = FlatJoinBuilder::new(right_table.len(), left_cols, right_cols);

    let left_rr = left_table.rows_ref().unwrap();
    let right_rr = right_table.rows_ref().unwrap();
    for right_row in right_rr.iter() {
        if let Some(left_indices) = left_index.lookup(right_row, &plan.right_indices, nulls_equal) {
            for &left_idx in left_indices {
                if let Some(left_row) = left_rr.row(left_idx) {
                    builder.push_lr(left_row, right_row);
                }
            }
        } else {
            builder.push_null_left_right(&null_left_row, right_row);
        }
    }

    join_table_value(builder, result_headers)
}

// Выполнение FULL JOIN
fn perform_full_join(
    left_table: &Table,
    right_table: &Table,
    keys: &[JoinKey],
    left_alias: &str,
    right_alias: &str,
    nulls_equal: bool,
) -> Value {
    let Some(plan) = JoinKeyPlan::new(left_table, right_table, keys) else {
        return Value::Null;
    };
    let right_index = build_join_index(right_table, &plan.right_indices, nulls_equal);
    let result_headers = join_result_headers(left_table, right_table, left_alias, right_alias);
    let left_cols = left_table.headers().len();
    let right_cols = right_table.headers().len();
    let null_left_row: Vec<Value> = vec![Value::Null; left_cols];
    let null_right_row: Vec<Value> = vec![Value::Null; right_cols];
    let mut builder = FlatJoinBuilder::new(left_table.len() + right_table.len(), left_cols, right_cols);
    let mut matched_right_indices = HashSet::new();

    let left_rr = left_table.rows_ref().unwrap();
    let right_rr = right_table.rows_ref().unwrap();
    for left_row in left_rr.iter() {
        if let Some(right_indices) = right_index.lookup(left_row, &plan.left_indices, nulls_equal) {
            for &right_idx in right_indices {
                matched_right_indices.insert(right_idx);
                if let Some(right_row) = right_rr.row(right_idx) {
                    builder.push_lr(left_row, right_row);
                }
            }
        } else {
            builder.push_left_null_right(left_row, &null_right_row);
        }
    }

    for (right_idx, right_row) in right_rr.iter().enumerate() {
        if !matched_right_indices.contains(&right_idx) {
            builder.push_null_left_right(&null_left_row, right_row);
        }
    }

    join_table_value(builder, result_headers)
}

// Выполнение SEMI JOIN (только строки left, колонки right не включаются)
fn perform_semi_join(
    left_table: &Table,
    right_table: &Table,
    keys: &[JoinKey],
    nulls_equal: bool,
) -> Value {
    let Some(plan) = JoinKeyPlan::new(left_table, right_table, keys) else {
        return Value::Null;
    };
    let right_index = build_join_index(right_table, &plan.right_indices, nulls_equal);
    let left_cols = left_table.headers().len();
    let mut flat = Vec::with_capacity(left_table.len().saturating_mul(left_cols));
    let left_rr = left_table.rows_ref().unwrap();
    for left_row in left_rr.iter() {
        if right_index.contains(left_row, &plan.left_indices, nulls_equal) {
            flat.extend_from_slice(left_row);
        }
    }
    let table = Table::from_flat_owned(flat, left_cols, left_table.headers().clone());
    Value::Table(Rc::new(RefCell::new(table)))
}

// Выполнение ANTI JOIN (строки left без совпадений в right)
fn perform_anti_join(
    left_table: &Table,
    right_table: &Table,
    keys: &[JoinKey],
    nulls_equal: bool,
) -> Value {
    let Some(plan) = JoinKeyPlan::new(left_table, right_table, keys) else {
        return Value::Null;
    };
    let right_index = build_join_index(right_table, &plan.right_indices, nulls_equal);
    let left_cols = left_table.headers().len();
    let mut flat = Vec::new();
    let left_rr = left_table.rows_ref().unwrap();
    for left_row in left_rr.iter() {
        let include = match right_index.lookup(left_row, &plan.left_indices, nulls_equal) {
            None => true,
            Some(_) => false,
        };
        if include {
            flat.extend_from_slice(left_row);
        }
    }
    let table = Table::from_flat_owned(flat, left_cols, left_table.headers().clone());
    Value::Table(Rc::new(RefCell::new(table)))
}

// Выполнение CROSS JOIN (декартово произведение)
fn perform_cross_join(
    left_table: &Table,
    right_table: &Table,
    left_alias: &str,
    right_alias: &str,
) -> Value {
    let result_headers = join_result_headers(left_table, right_table, left_alias, right_alias);
    let left_cols = left_table.headers().len();
    let right_cols = right_table.headers().len();
    let out_rows = left_table.len().saturating_mul(right_table.len());
    let mut builder = FlatJoinBuilder::new(out_rows, left_cols, right_cols);

    let left_rr = left_table.rows_ref().unwrap();
    let right_rr = right_table.rows_ref().unwrap();
    for left_row in left_rr.iter() {
        for right_row in right_rr.iter() {
            builder.push_lr(left_row, right_row);
        }
    }

    join_table_value(builder, result_headers)
}

// Универсальная функция JOIN для таблиц
pub fn native_table_join(args: &[Value]) -> Value {
    if args.len() < 3 {
        return Value::Null;
    }

    // Извлекаем left и right таблицы
    let left_rc = match table_arg(&args[0]) {
        Some(t) => t,
        None => return Value::Null,
    };
    let right_rc = match table_arg(&args[1]) {
        Some(t) => t,
        None => return Value::Null,
    };
    let left_table = left_rc.borrow();
    let right_table = right_rc.borrow();

    // Парсим тип JOIN (по умолчанию inner) - нужно проверить ДО парсинга ключей
    let join_type = if args.len() > 3 {
        match &args[3] {
            Value::String(s) => match s.as_str() {
                "inner" => JoinType::Inner,
                "left" => JoinType::Left,
                "right" => JoinType::Right,
                "full" => JoinType::Full,
                "cross" => JoinType::Cross,
                "semi" => JoinType::Semi,
                "anti" => JoinType::Anti,
                _ => JoinType::Inner,
            },
            _ => JoinType::Inner,
        }
    } else {
        JoinType::Inner
    };

    // Для CROSS JOIN не нужны ключи - пропускаем парсинг
    let keys = if join_type == JoinType::Cross {
        Vec::new()
    } else {
        // Парсим параметр on (ключи)
        match parse_join_keys(&args[2], &left_table, &right_table) {
            Ok(k) => k,
            Err(err_msg) => {
                use crate::websocket::set_native_error;
                set_native_error(err_msg);
                return Value::Null;
            }
        }
    };

    // Используем имена таблиц из table.name, если они установлены, иначе используем переданные алиасы или значения по умолчанию
    let left_alias = left_table
        .name
        .as_ref()
        .map(|n| n.clone())
        .unwrap_or_else(|| {
            if args.len() > 4 {
                match &args[4] {
                    Value::Array(arr) => {
                        let arr_ref = arr.borrow();
                        if arr_ref.len() >= 1 {
                            match &arr_ref[0] {
                                Value::String(s) => s.clone(),
                                _ => "left".to_string(),
                            }
                        } else {
                            "left".to_string()
                        }
                    }
                    _ => "left".to_string(),
                }
            } else {
                "left".to_string()
            }
        });

    let right_alias = right_table
        .name
        .as_ref()
        .map(|n| n.clone())
        .unwrap_or_else(|| {
            if args.len() > 4 {
                match &args[4] {
                    Value::Array(arr) => {
                        let arr_ref = arr.borrow();
                        if arr_ref.len() >= 2 {
                            match &arr_ref[1] {
                                Value::String(s) => s.clone(),
                                _ => "right".to_string(),
                            }
                        } else {
                            "right".to_string()
                        }
                    }
                    _ => "right".to_string(),
                }
            } else {
                "right".to_string()
            }
        });

    // Парсим nulls_equal (по умолчанию false)
    let nulls_equal = if args.len() > 5 {
        match &args[5] {
            Value::Bool(b) => *b,
            Value::Number(n) => *n != 0.0,
            _ => false,
        }
    } else {
        false
    };

    // CROSS JOIN - особый случай (декартово произведение)
    if join_type == JoinType::Cross {
        return perform_cross_join(&left_table, &right_table, &left_alias, &right_alias);
    }

    // Для остальных типов JOIN нужны ключи
    if keys.is_empty() {
        return Value::Null;
    }

    // Выполняем JOIN в зависимости от типа
    match join_type {
        JoinType::Inner => perform_inner_join(
            &left_table,
            &right_table,
            &keys,
            &left_alias,
            &right_alias,
            nulls_equal,
        ),
        JoinType::Left => perform_left_join(
            &left_table,
            &right_table,
            &keys,
            &left_alias,
            &right_alias,
            nulls_equal,
        ),
        JoinType::Right => perform_right_join(
            &left_table,
            &right_table,
            &keys,
            &left_alias,
            &right_alias,
            nulls_equal,
        ),
        JoinType::Full => perform_full_join(
            &left_table,
            &right_table,
            &keys,
            &left_alias,
            &right_alias,
            nulls_equal,
        ),
        JoinType::Semi => perform_semi_join(&left_table, &right_table, &keys, nulls_equal),
        JoinType::Anti => perform_anti_join(&left_table, &right_table, &keys, nulls_equal),
        JoinType::Cross => unreachable!(), // Уже обработано выше
    }
}

// Вспомогательная функция для выполнения ASOF join для одной группы
fn asof_join_single_group(
    left_rows: &[Vec<Value>],
    right_rows: &[Vec<Value>],
    left_time_idx: usize,
    right_time_idx: usize,
    direction: &str,
    null_right_row: &[Value],
) -> Vec<Vec<Value>> {
    let mut result = Vec::with_capacity(left_rows.len());
    if right_rows.is_empty() {
        for left_row in left_rows {
            let mut new_row = left_row.clone();
            new_row.extend_from_slice(null_right_row);
            result.push(new_row);
        }
        return result;
    }

    let mut right_indices: Vec<usize> = (0..right_rows.len()).collect();
    right_indices.sort_by(|&a, &b| {
        compare_values(
            &right_rows[a][right_time_idx],
            &right_rows[b][right_time_idx],
        )
    });
    let right_times: Option<Vec<f64>> = right_indices
        .iter()
        .map(|&i| asof_time_f64(&right_rows[i][right_time_idx]))
        .collect();
    let Some(right_times) = right_times else {
        return asof_join_single_group_linear(
            left_rows,
            right_rows,
            left_time_idx,
            right_time_idx,
            direction,
            null_right_row,
        );
    };

    for left_row in left_rows {
        let left_time = &left_row[left_time_idx];
        if let Some(target) = asof_time_f64(left_time) {
            if let Some(pos) = asof_pick_idx(&right_times, target, direction) {
                let right_row = &right_rows[right_indices[pos]];
                let mut new_row = left_row.clone();
                new_row.extend_from_slice(right_row);
                result.push(new_row);
                continue;
            }
        }
        let mut new_row = left_row.clone();
        new_row.extend_from_slice(null_right_row);
        result.push(new_row);
    }

    result
}

fn asof_join_single_group_linear(
    left_rows: &[Vec<Value>],
    right_rows: &[Vec<Value>],
    left_time_idx: usize,
    right_time_idx: usize,
    direction: &str,
    null_right_row: &[Value],
) -> Vec<Vec<Value>> {
    let mut result = Vec::with_capacity(left_rows.len());
    let mut right_indices: Vec<usize> = (0..right_rows.len()).collect();
    right_indices.sort_by(|&a, &b| {
        compare_values(
            &right_rows[a][right_time_idx],
            &right_rows[b][right_time_idx],
        )
    });

    for left_row in left_rows {
        let left_time = &left_row[left_time_idx];
        let mut best_match: Option<usize> = None;
        let mut best_diff: Option<f64> = None;

        for &right_idx in &right_indices {
            let right_row = &right_rows[right_idx];
            let right_time = &right_row[right_time_idx];
            let diff = match (left_time, right_time) {
                (Value::Number(l), Value::Number(r)) => {
                    let diff_val = match direction {
                        "forward" => *r - *l,
                        "nearest" => (*l - *r).abs(),
                        _ => *l - *r,
                    };
                    Some(diff_val)
                }
                _ => None,
            };

            if let Some(d) = diff {
                let matches_direction = match direction {
                    "backward" => d >= 0.0,
                    "forward" => d >= 0.0,
                    "nearest" => true,
                    _ => d >= 0.0,
                };

                if matches_direction {
                    let should_update = match best_diff {
                        None => true,
                        Some(bd) => d < bd,
                    };

                    if should_update {
                        best_match = Some(right_idx);
                        best_diff = Some(d);
                    }
                }
            }
        }

        if let Some(right_idx) = best_match {
            let mut new_row = left_row.clone();
            new_row.extend_from_slice(&right_rows[right_idx]);
            result.push(new_row);
        } else {
            let mut new_row = left_row.clone();
            new_row.extend_from_slice(null_right_row);
            result.push(new_row);
        }
    }

    result
}

// Specialized JOIN Functions (syntactic sugar over native_table_join)
pub fn native_inner_join(args: &[Value]) -> Value {
    if args.len() < 3 {
        return Value::Null;
    }
    // Вызываем native_table_join с type="inner"
    let mut new_args = args.to_vec();

    // Если переданы два отдельных строковых аргумента (left_col, right_col)
    if new_args.len() == 4 {
        if let (Value::String(_), Value::String(_)) = (&new_args[2], &new_args[3]) {
            // Создаем массив кортежей: [["left_col", "right_col"]]
            let tuple = Value::Array(Rc::new(RefCell::new(vec![
                new_args[2].clone(),
                new_args[3].clone(),
            ])));
            let keys_array = Value::Array(Rc::new(RefCell::new(vec![tuple])));
            new_args[2] = keys_array;
            new_args[3] = Value::String("inner".to_string());
        } else {
            // Иначе перезаписываем тип join
            new_args[3] = Value::String("inner".to_string());
        }
    } else if new_args.len() == 3 {
        new_args.push(Value::String("inner".to_string()));
    } else {
        new_args[3] = Value::String("inner".to_string());
    }
    native_table_join(&new_args)
}

pub fn native_left_join(args: &[Value]) -> Value {
    if args.len() < 3 {
        return Value::Null;
    }
    let mut new_args = args.to_vec();

    // Если переданы два отдельных строковых аргумента (left_col, right_col)
    if new_args.len() == 4 {
        if let (Value::String(_), Value::String(_)) = (&new_args[2], &new_args[3]) {
            // Создаем массив кортежей: [["left_col", "right_col"]]
            let tuple = Value::Array(Rc::new(RefCell::new(vec![
                new_args[2].clone(),
                new_args[3].clone(),
            ])));
            let keys_array = Value::Array(Rc::new(RefCell::new(vec![tuple])));
            new_args[2] = keys_array;
            new_args[3] = Value::String("left".to_string());
        } else {
            // Иначе перезаписываем тип join
            new_args[3] = Value::String("left".to_string());
        }
    } else if new_args.len() == 3 {
        new_args.push(Value::String("left".to_string()));
    } else {
        new_args[3] = Value::String("left".to_string());
    }
    native_table_join(&new_args)
}

pub fn native_right_join(args: &[Value]) -> Value {
    if args.len() < 3 {
        return Value::Null;
    }
    let mut new_args = args.to_vec();

    // Если переданы два отдельных строковых аргумента (left_col, right_col)
    if new_args.len() == 4 {
        if let (Value::String(_), Value::String(_)) = (&new_args[2], &new_args[3]) {
            // Создаем массив кортежей: [["left_col", "right_col"]]
            let tuple = Value::Array(Rc::new(RefCell::new(vec![
                new_args[2].clone(),
                new_args[3].clone(),
            ])));
            let keys_array = Value::Array(Rc::new(RefCell::new(vec![tuple])));
            new_args[2] = keys_array;
            new_args[3] = Value::String("right".to_string());
        } else {
            // Иначе перезаписываем тип join
            new_args[3] = Value::String("right".to_string());
        }
    } else if new_args.len() == 3 {
        new_args.push(Value::String("right".to_string()));
    } else {
        new_args[3] = Value::String("right".to_string());
    }
    native_table_join(&new_args)
}

pub fn native_full_join(args: &[Value]) -> Value {
    if args.len() < 3 {
        return Value::Null;
    }
    let mut new_args = args.to_vec();

    // Если переданы два отдельных строковых аргумента (left_col, right_col)
    if new_args.len() == 4 {
        if let (Value::String(_), Value::String(_)) = (&new_args[2], &new_args[3]) {
            // Создаем массив кортежей: [["left_col", "right_col"]]
            let tuple = Value::Array(Rc::new(RefCell::new(vec![
                new_args[2].clone(),
                new_args[3].clone(),
            ])));
            let keys_array = Value::Array(Rc::new(RefCell::new(vec![tuple])));
            new_args[2] = keys_array;
            new_args[3] = Value::String("full".to_string());
        } else {
            // Иначе перезаписываем тип join
            new_args[3] = Value::String("full".to_string());
        }
    } else if new_args.len() == 3 {
        new_args.push(Value::String("full".to_string()));
    } else {
        new_args[3] = Value::String("full".to_string());
    }
    native_table_join(&new_args)
}

pub fn native_cross_join(args: &[Value]) -> Value {
    if args.len() < 2 {
        return Value::Null;
    }
    // CROSS JOIN не требует ключей, используем пустой массив
    let mut new_args = vec![args[0].clone(), args[1].clone()];
    new_args.push(Value::Array(Rc::new(RefCell::new(Vec::new()))));
    new_args.push(Value::String("cross".to_string()));
    native_table_join(&new_args)
}

pub fn native_semi_join(args: &[Value]) -> Value {
    if args.len() < 3 {
        return Value::Null;
    }
    let mut new_args = args.to_vec();

    // Если переданы два отдельных строковых аргумента (left_col, right_col)
    if new_args.len() == 4 {
        if let (Value::String(_), Value::String(_)) = (&new_args[2], &new_args[3]) {
            // Создаем массив кортежей: [["left_col", "right_col"]]
            let tuple = Value::Array(Rc::new(RefCell::new(vec![
                new_args[2].clone(),
                new_args[3].clone(),
            ])));
            let keys_array = Value::Array(Rc::new(RefCell::new(vec![tuple])));
            new_args[2] = keys_array;
            new_args[3] = Value::String("semi".to_string());
        } else {
            // Иначе перезаписываем тип join
            new_args[3] = Value::String("semi".to_string());
        }
    } else if new_args.len() == 3 {
        new_args.push(Value::String("semi".to_string()));
    } else {
        new_args[3] = Value::String("semi".to_string());
    }
    native_table_join(&new_args)
}

pub fn native_anti_join(args: &[Value]) -> Value {
    if args.len() < 3 {
        return Value::Null;
    }
    let mut new_args = args.to_vec();

    // Если переданы два отдельных строковых аргумента (left_col, right_col)
    if new_args.len() == 4 {
        if let (Value::String(_), Value::String(_)) = (&new_args[2], &new_args[3]) {
            // Создаем массив кортежей: [["left_col", "right_col"]]
            let tuple = Value::Array(Rc::new(RefCell::new(vec![
                new_args[2].clone(),
                new_args[3].clone(),
            ])));
            let keys_array = Value::Array(Rc::new(RefCell::new(vec![tuple])));
            new_args[2] = keys_array;
            new_args[3] = Value::String("anti".to_string());
        } else {
            // Иначе перезаписываем тип join
            new_args[3] = Value::String("anti".to_string());
        }
    } else if new_args.len() == 3 {
        new_args.push(Value::String("anti".to_string()));
    } else {
        new_args[3] = Value::String("anti".to_string());
    }
    native_table_join(&new_args)
}

// ZIP JOIN - позиционное соединение по индексу строки
pub fn native_zip_join(args: &[Value]) -> Value {
    if args.len() < 2 {
        return Value::Null;
    }

    let left_rc = match table_arg(&args[0]) {
        Some(t) => t,
        None => return Value::Null,
    };
    let right_rc = match table_arg(&args[1]) {
        Some(t) => t,
        None => return Value::Null,
    };
    let left_table = left_rc.borrow();
    let right_table = right_rc.borrow();

    // Используем имена таблиц из table.name, если они установлены
    let left_alias = left_table
        .name
        .as_ref()
        .map(|n| n.clone())
        .unwrap_or_else(|| {
            if args.len() > 2 {
                match &args[2] {
                    Value::Array(arr) => {
                        let arr_ref = arr.borrow();
                        if arr_ref.len() >= 1 {
                            match &arr_ref[0] {
                                Value::String(s) => s.clone(),
                                _ => "left".to_string(),
                            }
                        } else {
                            "left".to_string()
                        }
                    }
                    _ => "left".to_string(),
                }
            } else {
                "left".to_string()
            }
        });

    let right_alias = right_table
        .name
        .as_ref()
        .map(|n| n.clone())
        .unwrap_or_else(|| {
            if args.len() > 2 {
                match &args[2] {
                    Value::Array(arr) => {
                        let arr_ref = arr.borrow();
                        if arr_ref.len() >= 2 {
                            match &arr_ref[1] {
                                Value::String(s) => s.clone(),
                                _ => "right".to_string(),
                            }
                        } else {
                            "right".to_string()
                        }
                    }
                    _ => "right".to_string(),
                }
            } else {
                "right".to_string()
            }
        });

    let mut result_rows = Vec::with_capacity(left_table.len());
    let mut result_headers = Vec::new();

    // Создаем заголовки с учетом алиасов таблиц
    let left_headers_set: HashSet<String> = left_table.headers().iter().cloned().collect();
    let right_headers_set: HashSet<String> = right_table.headers().iter().cloned().collect();

    let mut left_headers =
        apply_column_aliases(&left_table.headers(), &left_alias, &right_headers_set);
    let mut right_headers =
        apply_column_aliases(&right_table.headers(), &right_alias, &left_headers_set);

    result_headers.append(&mut left_headers);
    result_headers.append(&mut right_headers);

    // Соединяем строки по позиции (индексу)
    let left_rr = left_table.rows_ref().unwrap();
    let right_rr = right_table.rows_ref().unwrap();
    let min_len = std::cmp::min(left_rr.len(), right_rr.len());
    for i in 0..min_len {
        let mut new_row = left_rr.row(i).unwrap().to_vec();
        new_row.extend_from_slice(right_rr.row(i).unwrap());
        result_rows.push(new_row);
    }

    Value::Table(Rc::new(RefCell::new(Table::from_data(
        result_rows,
        Some(result_headers),
    ))))
}

// APPLY JOIN / LATERAL JOIN - для каждой строки left вызывает функцию
pub fn native_apply_join(args: &[Value]) -> Value {
    use super::utils::call_user_function;
    use crate::vm::vm::{current_vm_ptr, VmExecutionContext, VM_CALL_CONTEXT};

    if args.len() < 2 {
        return Value::Null;
    }

    let left_rc = match table_arg(&args[0]) {
        Some(t) => t,
        None => return Value::Null,
    };
    let left_table = left_rc.borrow();

    // Извлекаем функцию из аргументов
    let function_index = match &args[1] {
        Value::Function(idx) => *idx,
        _ => {
            // Если функция не передана, возвращаем Null
            return Value::Null;
        }
    };

    // Извлекаем тип JOIN (по умолчанию "inner"). Копируем в String, т.к. args может мутироваться при реентрантных вызовах VM.
    let join_type: String = args
        .iter()
        .find_map(|v| {
            if let Value::String(s) = v {
                let t = s.as_str();
                if t == "left" || t == "inner" || t == "right" {
                    Some(t.to_string())
                } else {
                    None
                }
            } else {
                None
            }
        })
        .unwrap_or_else(|| "inner".to_string());

    // Получаем доступ к VM через thread-local storage
    let vm_ptr = current_vm_ptr();

    let mut result_rows = Vec::with_capacity(left_table.len());
    let mut result_headers = Vec::new();

    // Создаем заголовки результата (начнем с заголовков левой таблицы)
    result_headers.extend_from_slice(left_table.headers());

    // Отслеживаем максимальное количество колонок правой таблицы для корректной обработки NULLs
    let mut max_right_columns = 0;

    // Для каждой строки левой таблицы вызываем функцию
    let left_rr = left_table.rows_ref().unwrap();
    for left_row in left_rr.iter() {
        // Восстанавливаем контекст перед каждым вызовом
        if let Some(vm_ptr) = vm_ptr {
            VM_CALL_CONTEXT.with(|ctx| {
                *ctx.borrow_mut() = Some(VmExecutionContext { vm: vm_ptr });
            });
        }

        // Вызываем функцию с аргументом - массив значений строки
        let row_array = Value::Array(Rc::new(RefCell::new(left_row.to_vec())));
        let function_result = match call_user_function(function_index, &[row_array]) {
            Ok(result) => result,
            Err(_e) => {
                // Если произошла ошибка при вызове функции, пропускаем строку для inner join
                // или добавляем с NULLs для left join
                if join_type == "left" {
                    result_rows.push(left_row.to_vec());
                }
                continue;
            }
        };

        let rows_before = result_rows.len();
        match function_result {
            Value::Table(right_table) => {
                let right_owned_rc = ensure_owned_rc(&right_table);
                let right_table_ref = right_owned_rc.borrow();

                // Обновляем максимальное количество колонок
                if right_table_ref.headers().len() > max_right_columns {
                    max_right_columns = right_table_ref.headers().len();
                }

                // Если заголовки результата еще не установлены полностью, добавляем заголовки правой таблицы
                if result_headers.len() == left_table.headers().len() {
                    // Проверяем конфликты имен колонок
                    let left_headers_set: HashSet<String> =
                        left_table.headers().iter().cloned().collect();
                    let mut right_headers = Vec::new();
                    for header in right_table_ref.headers() {
                        if left_headers_set.contains(header) {
                            right_headers.push(format!("right_{}", header));
                        } else {
                            right_headers.push(header.clone());
                        }
                    }
                    result_headers.extend_from_slice(&right_headers);
                }

                // Для каждой строки в результате функции добавляем комбинацию left_row + right_row
                let right_rr = right_table_ref.rows_ref().unwrap();
                for right_row in right_rr.iter() {
                    let mut new_row = left_row.to_vec();
                    new_row.extend_from_slice(right_row);
                    result_rows.push(new_row);
                }
                // Left join: если функция вернула пустую таблицу, всё равно сохраняем left row (NULLs добавятся ниже)
                if join_type == "left" && right_rr.is_empty() {
                    result_rows.push(left_row.to_vec());
                }
            }
            Value::Null => {
                // Если функция вернула Null и это left join, добавляем строку с NULLs
                if join_type == "left" {
                    result_rows.push(left_row.to_vec());
                }
            }
            _ => {
                // Если функция вернула что-то другое, игнорируем для inner join
                // или добавляем с NULLs для left join
                if join_type == "left" {
                    result_rows.push(left_row.to_vec());
                }
            }
        }
        // Left join: если за эту итерацию не добавили ни одной строки, сохраняем left row
        if join_type == "left" && result_rows.len() == rows_before {
            result_rows.push(left_row.to_vec());
        }
    }

    // Если это left join и были строки без правой части, добавляем NULL значения
    if join_type == "left" && max_right_columns > 0 {
        // Добавляем NULL значения для строк, которые не имеют правой части
        let expected_length = left_table.headers().len() + max_right_columns;
        for row in &mut result_rows {
            while row.len() < expected_length {
                row.push(Value::Null);
            }
        }
    }

    Value::Table(Rc::new(RefCell::new(Table::from_data(
        result_rows,
        Some(result_headers),
    ))))
}

// ASOF JOIN - временное соединение
pub fn native_asof_join(args: &[Value]) -> Value {
    if args.len() < 3 {
        return Value::Null;
    }

    let left_rc = match table_arg(&args[0]) {
        Some(t) => t,
        None => return Value::Null,
    };
    let right_rc = match table_arg(&args[1]) {
        Some(t) => t,
        None => return Value::Null,
    };
    let left_table = left_rc.borrow();
    let right_table = right_rc.borrow();

    // Парсим временную колонку
    let time_column = match &args[2] {
        Value::String(s) => s.clone(),
        _ => return Value::Null,
    };

    if !left_table.has_column(&time_column) || !right_table.has_column(&time_column) {
        return Value::Null;
    }

    // Парсим by (группирующие колонки) - опционально
    let by_columns: Vec<String> = if args.len() > 3 {
        match &args[3] {
            Value::String(s) => vec![s.clone()],
            Value::Array(arr) => {
                let arr_ref = arr.borrow();
                let mut cols = Vec::new();
                for val in arr_ref.iter() {
                    if let Value::String(s) = val {
                        cols.push(s.clone());
                    }
                }
                cols
            }
            _ => Vec::new(),
        }
    } else {
        Vec::new()
    };

    // Парсим direction (по умолчанию "backward")
    let direction = if args.len() > 4 {
        match &args[4] {
            Value::String(s) => s.as_str(),
            _ => "backward",
        }
    } else {
        "backward"
    };

    // Используем имена таблиц из table.name, если они установлены
    let left_alias = left_table
        .name
        .as_ref()
        .map(|n| n.clone())
        .unwrap_or_else(|| {
            if args.len() > 5 {
                match &args[5] {
                    Value::Array(arr) => {
                        let arr_ref = arr.borrow();
                        if arr_ref.len() >= 1 {
                            match &arr_ref[0] {
                                Value::String(s) => s.clone(),
                                _ => "left".to_string(),
                            }
                        } else {
                            "left".to_string()
                        }
                    }
                    _ => "left".to_string(),
                }
            } else {
                "left".to_string()
            }
        });

    let right_alias = right_table
        .name
        .as_ref()
        .map(|n| n.clone())
        .unwrap_or_else(|| {
            if args.len() > 5 {
                match &args[5] {
                    Value::Array(arr) => {
                        let arr_ref = arr.borrow();
                        if arr_ref.len() >= 2 {
                            match &arr_ref[1] {
                                Value::String(s) => s.clone(),
                                _ => "right".to_string(),
                            }
                        } else {
                            "right".to_string()
                        }
                    }
                    _ => "right".to_string(),
                }
            } else {
                "right".to_string()
            }
        });

    // Создаем индексы для временной колонки
    let left_time_idx = left_table
        .headers()
        .iter()
        .position(|h| h == &time_column)
        .unwrap();
    let right_time_idx = right_table
        .headers()
        .iter()
        .position(|h| h == &time_column)
        .unwrap();

    // Если есть by колонки, группируем данные
    let mut result_rows = Vec::with_capacity(left_table.len());
    let mut result_headers = Vec::new();

    // Создаем заголовки с учетом алиасов таблиц
    let left_headers_set: HashSet<String> = left_table.headers().iter().cloned().collect();
    let right_headers_set: HashSet<String> = right_table.headers().iter().cloned().collect();

    let mut left_headers =
        apply_column_aliases(&left_table.headers(), &left_alias, &right_headers_set);
    let mut right_headers =
        apply_column_aliases(&right_table.headers(), &right_alias, &left_headers_set);

    result_headers.append(&mut left_headers);
    result_headers.append(&mut right_headers);

    // Создаем NULL-строку для правой таблицы
    let null_right_row: Vec<Value> = (0..right_table.headers().len())
        .map(|_| Value::Null)
        .collect();

    if by_columns.is_empty() {
        let right_rr = right_table.rows_ref().unwrap();
        let mut right_indices: Vec<usize> = (0..right_rr.len()).collect();
        right_indices.sort_by(|&a, &b| {
            let time_a = &right_rr.row(a).unwrap()[right_time_idx];
            let time_b = &right_rr.row(b).unwrap()[right_time_idx];
            compare_values(time_a, time_b)
        });
        let right_times: Option<Vec<f64>> = right_indices
            .iter()
            .map(|&i| asof_time_f64(&right_rr.row(i).unwrap()[right_time_idx]))
            .collect();

        let left_rr = left_table.rows_ref().unwrap();
        if let Some(right_times) = right_times {
            for left_row in left_rr.iter() {
                if let Some(target) = asof_time_f64(&left_row[left_time_idx]) {
                    if let Some(pos) = asof_pick_idx(&right_times, target, direction) {
                        let right_row = right_rr.row(right_indices[pos]).unwrap();
                        let mut new_row = left_row.to_vec();
                        new_row.extend_from_slice(right_row);
                        result_rows.push(new_row);
                        continue;
                    }
                }
                let mut new_row = left_row.to_vec();
                new_row.extend_from_slice(&null_right_row);
                result_rows.push(new_row);
            }
        } else {
            let right_rows: Vec<Vec<Value>> = right_rr.iter().map(|r| r.to_vec()).collect();
            let left_rows: Vec<Vec<Value>> = left_rr.iter().map(|r| r.to_vec()).collect();
            result_rows.extend(asof_join_single_group_linear(
                &left_rows,
                &right_rows,
                left_time_idx,
                right_time_idx,
                direction,
                &null_right_row,
            ));
        }
    } else {
        // Есть группировка - группируем по by колонкам
        // Получаем индексы by колонок
        let mut by_indices_left = Vec::new();
        let mut by_indices_right = Vec::new();

        for by_col in &by_columns {
            if let Some(idx) = left_table.headers().iter().position(|h| h == by_col) {
                by_indices_left.push(idx);
            } else {
                return Value::Null; // Колонка не найдена
            }
            if let Some(idx) = right_table.headers().iter().position(|h| h == by_col) {
                by_indices_right.push(idx);
            } else {
                return Value::Null; // Колонка не найдена
            }
        }

        // Группируем левую таблицу по by колонкам
        let mut left_groups: HashMap<Vec<Value>, Vec<Vec<Value>>> = HashMap::new();
        let left_rr = left_table.rows_ref().unwrap();
        for row in left_rr.iter() {
            let key: Vec<Value> = by_indices_left
                .iter()
                .map(|&idx| row[idx].clone())
                .collect();
            left_groups
                .entry(key)
                .or_insert_with(Vec::new)
                .push(row.to_vec());
        }

        // Группируем правую таблицу по by колонкам
        let mut right_groups: HashMap<Vec<Value>, Vec<Vec<Value>>> = HashMap::new();
        let right_rr = right_table.rows_ref().unwrap();
        for row in right_rr.iter() {
            let key: Vec<Value> = by_indices_right
                .iter()
                .map(|&idx| row[idx].clone())
                .collect();
            right_groups
                .entry(key)
                .or_insert_with(Vec::new)
                .push(row.to_vec());
        }

        // Для каждой группы в левой таблице выполняем ASOF join
        for (group_key, left_group_rows) in left_groups {
            if let Some(right_group_rows) = right_groups.get(&group_key) {
                // Выполняем ASOF join для этой группы
                let group_results = asof_join_single_group(
                    &left_group_rows,
                    right_group_rows,
                    left_time_idx,
                    right_time_idx,
                    direction,
                    &null_right_row,
                );
                result_rows.extend(group_results);
            } else {
                // Группы нет в правой таблице - добавляем строки с NULLs (для left join семантики)
                for left_row in left_group_rows {
                    let mut new_row = left_row.clone();
                    new_row.extend_from_slice(&null_right_row);
                    result_rows.push(new_row);
                }
            }
        }
    }

    Value::Table(Rc::new(RefCell::new(Table::from_data(
        result_rows,
        Some(result_headers),
    ))))
}

// JOIN ON - non-equi join с произвольным условием
pub fn native_join_on(args: &[Value]) -> Value {
    if args.len() < 3 {
        return Value::Null;
    }

    let left_rc = match table_arg(&args[0]) {
        Some(t) => t,
        None => return Value::Null,
    };
    let right_rc = match table_arg(&args[1]) {
        Some(t) => t,
        None => return Value::Null,
    };
    let left_table = left_rc.borrow();
    let right_table = right_rc.borrow();

    // Парсим условие - пока упрощенная версия
    let condition = &args[2];

    // Парсим тип JOIN (по умолчанию inner)
    let join_type = if args.len() > 3 {
        match &args[3] {
            Value::String(s) => match s.as_str() {
                "inner" => JoinType::Inner,
                "left" => JoinType::Left,
                "right" => JoinType::Right,
                "full" => JoinType::Full,
                _ => JoinType::Inner,
            },
            _ => JoinType::Inner,
        }
    } else {
        JoinType::Inner
    };

    // Используем имена таблиц из table.name, если они установлены
    let left_alias = left_table
        .name
        .as_ref()
        .map(|n| n.clone())
        .unwrap_or_else(|| {
            if args.len() > 4 {
                match &args[4] {
                    Value::Array(arr) => {
                        let arr_ref = arr.borrow();
                        if arr_ref.len() >= 1 {
                            match &arr_ref[0] {
                                Value::String(s) => s.clone(),
                                _ => "left".to_string(),
                            }
                        } else {
                            "left".to_string()
                        }
                    }
                    _ => "left".to_string(),
                }
            } else {
                "left".to_string()
            }
        });

    let right_alias = right_table
        .name
        .as_ref()
        .map(|n| n.clone())
        .unwrap_or_else(|| {
            if args.len() > 4 {
                match &args[4] {
                    Value::Array(arr) => {
                        let arr_ref = arr.borrow();
                        if arr_ref.len() >= 2 {
                            match &arr_ref[1] {
                                Value::String(s) => s.clone(),
                                _ => "right".to_string(),
                            }
                        } else {
                            "right".to_string()
                        }
                    }
                    _ => "right".to_string(),
                }
            } else {
                "right".to_string()
            }
        });

    // Парсим условие - упрощенная версия
    let (left_col, op, right_col) = match condition {
        Value::String(s) => {
            // Парсим строку вида "left_col >= right_col"
            let parts: Vec<&str> = s.split_whitespace().collect();
            if parts.len() >= 3 {
                (
                    parts[0].to_string(),
                    parts[1].to_string(),
                    parts[2].to_string(),
                )
            } else {
                return Value::Null;
            }
        }
        Value::Array(arr) => {
            let arr_ref = arr.borrow();
            if arr_ref.len() >= 3 {
                let left = match &arr_ref[0] {
                    Value::String(s) => s.clone(),
                    _ => return Value::Null,
                };
                let op = match &arr_ref[1] {
                    Value::String(s) => s.clone(),
                    _ => return Value::Null,
                };
                let right = match &arr_ref[2] {
                    Value::String(s) => s.clone(),
                    _ => return Value::Null,
                };
                (left, op, right)
            } else {
                return Value::Null;
            }
        }
        _ => return Value::Null,
    };

    if !left_table.has_column(&left_col) || !right_table.has_column(&right_col) {
        return Value::Null;
    }

    let left_col_idx = left_table
        .headers()
        .iter()
        .position(|h| h == &left_col)
        .unwrap();
    let right_col_idx = right_table
        .headers()
        .iter()
        .position(|h| h == &right_col)
        .unwrap();

    let mut result_rows = Vec::with_capacity(left_table.len());
    let mut result_headers = Vec::new();

    // Создаем заголовки с учетом алиасов таблиц
    let left_headers_set: HashSet<String> = left_table.headers().iter().cloned().collect();
    let right_headers_set: HashSet<String> = right_table.headers().iter().cloned().collect();

    let mut left_headers =
        apply_column_aliases(&left_table.headers(), &left_alias, &right_headers_set);
    let mut right_headers =
        apply_column_aliases(&right_table.headers(), &right_alias, &left_headers_set);

    result_headers.append(&mut left_headers);
    result_headers.append(&mut right_headers);

    // Создаем NULL-строки
    let null_left_row: Vec<Value> = (0..left_table.headers().len())
        .map(|_| Value::Null)
        .collect();
    let null_right_row: Vec<Value> = (0..right_table.headers().len())
        .map(|_| Value::Null)
        .collect();

    // Nested loop join с проверкой условия
    let mut matched_right_indices = HashSet::new();
    let left_rr = left_table.rows_ref().unwrap();
    let right_rr = right_table.rows_ref().unwrap();

    for left_row in left_rr.iter() {
        let left_val = &left_row[left_col_idx];
        let mut found_match = false;

        for (right_idx, right_row) in right_rr.iter().enumerate() {
            let right_val = &right_row[right_col_idx];

            // Проверяем условие
            let condition_met = match op.as_str() {
                ">" => compare_values(left_val, right_val) == std::cmp::Ordering::Greater,
                "<" => compare_values(left_val, right_val) == std::cmp::Ordering::Less,
                ">=" => {
                    let cmp = compare_values(left_val, right_val);
                    cmp == std::cmp::Ordering::Greater || cmp == std::cmp::Ordering::Equal
                }
                "<=" => {
                    let cmp = compare_values(left_val, right_val);
                    cmp == std::cmp::Ordering::Less || cmp == std::cmp::Ordering::Equal
                }
                "==" | "=" => compare_values(left_val, right_val) == std::cmp::Ordering::Equal,
                "!=" | "<>" => compare_values(left_val, right_val) != std::cmp::Ordering::Equal,
                _ => false,
            };

            if condition_met {
                found_match = true;
                matched_right_indices.insert(right_idx);
                let mut new_row = left_row.to_vec();
                new_row.extend_from_slice(right_row);
                result_rows.push(new_row);
            }
        }

        // Для LEFT JOIN добавляем строки без совпадений
        if !found_match && (join_type == JoinType::Left || join_type == JoinType::Full) {
            let mut new_row = left_row.to_vec();
            new_row.extend_from_slice(&null_right_row);
            result_rows.push(new_row);
        }
    }

    // Для RIGHT и FULL JOIN добавляем несовпадающие строки справа
    if join_type == JoinType::Right || join_type == JoinType::Full {
        for (right_idx, right_row) in right_rr.iter().enumerate() {
            if !matched_right_indices.contains(&right_idx) {
                let mut new_row = null_left_row.clone();
                new_row.extend_from_slice(right_row);
                result_rows.push(new_row);
            }
        }
    }

    Value::Table(Rc::new(RefCell::new(Table::from_data(
        result_rows,
        Some(result_headers),
    ))))
}

fn suffix_header_name(
    header: &str,
    prefix_to_table: &HashMap<String, &str>,
    left_suffix: &str,
    right_suffix: &str,
) -> String {
    if let Some(dot_pos) = header.find('.') {
        let prefix = &header[..dot_pos];
        let base_name = &header[dot_pos + 1..];
        if let Some(table_side) = prefix_to_table.get(prefix) {
            if *table_side == "left" {
                return format!("{}{}", base_name, left_suffix);
            }
            return format!("{}{}", base_name, right_suffix);
        }
    }
    header.to_string()
}

// Применение суффиксов к колонкам таблицы после join
pub fn native_table_suffixes(args: &[Value]) -> Value {
    if args.len() < 3 {
        return Value::Null;
    }

    let table_rc = match table_arg(&args[0]) {
        Some(t) => t,
        None => return Value::Null,
    };

    let left_suffix = match &args[1] {
        Value::String(s) => s.clone(),
        _ => return Value::Null,
    };

    let right_suffix = match &args[2] {
        Value::String(s) => s.clone(),
        _ => return Value::Null,
    };

    let mut seen_prefixes = Vec::new();
    let mut prefix_to_table: HashMap<String, &str> = HashMap::new();
    {
        let table = table_rc.borrow();
        for header in table.headers() {
            if let Some(dot_pos) = header.find('.') {
                let prefix = &header[..dot_pos];
                if !seen_prefixes.iter().any(|p| p == prefix) {
                    seen_prefixes.push(prefix.to_string());
                }
            }
        }
    }
    for (i, prefix) in seen_prefixes.iter().enumerate() {
        let side = if i == 0 { "left" } else { "right" };
        prefix_to_table.insert(prefix.clone(), side);
    }
    prefix_to_table.insert("left".to_string(), "left");
    prefix_to_table.insert("right".to_string(), "right");

    let new_headers: Vec<String> = {
        let table = table_rc.borrow();
        table
            .headers()
            .iter()
            .map(|header| suffix_header_name(header, &prefix_to_table, &left_suffix, &right_suffix))
            .collect()
    };

    table_rc.borrow_mut().set_headers(new_headers);
    Value::Table(table_rc)
}
