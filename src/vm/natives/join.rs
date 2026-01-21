// JOIN operations native functions

use crate::common::value::Value;
use crate::common::table::Table;
use std::rc::Rc;
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::hash::Hash;

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

// Import compare_values from table module
use super::table::compare_values;

// Парсинг ключей JOIN из Value
fn parse_join_keys(value: &Value, left_table: &Table, right_table: &Table) -> Result<Vec<JoinKey>, String> {
    match value {
        Value::String(col_name) => {
            // Автоматическое сопоставление: ищем колонку с таким именем в обеих таблицах
            if left_table.get_column(col_name).is_some() && right_table.get_column(col_name).is_some() {
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
                if let (Value::String(left_col), Value::String(right_col)) = (&arr_ref[0], &arr_ref[1]) {
                    if left_table.get_column(left_col).is_none() {
                        return Err(format!("Column '{}' not found in left table", left_col));
                    }
                    if right_table.get_column(right_col).is_none() {
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
                        
                        if left_table.get_column(&left_col).is_none() {
                            return Err(format!("Column '{}' not found in left table", left_col));
                        }
                        if right_table.get_column(&right_col).is_none() {
                            return Err(format!("Column '{}' not found in right table", right_col));
                        }
                        
                        keys.push(JoinKey { left_col, right_col });
                    }
                    Value::String(col_name) => {
                        // Одиночная строка в массиве - автоматическое сопоставление
                        if left_table.get_column(col_name).is_some() && right_table.get_column(col_name).is_some() {
                            keys.push(JoinKey {
                                left_col: col_name.clone(),
                                right_col: col_name.clone(),
                            });
                        } else {
                            return Err(format!("Column '{}' not found in both tables", col_name));
                        }
                    }
                    _ => return Err("Join key must be a string or tuple [string, string]".to_string()),
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
                format!("{}.{}", alias, h)  // Конфликт: left.id
            } else {
                h.clone()  // Нет конфликта: name
            }
        })
        .collect()
}

// Построение хеш-таблицы для правой таблицы по ключам
fn build_right_hash_table(
    right_table: &Table,
    keys: &[JoinKey],
    nulls_equal: bool,
) -> HashMap<KeyHash, Vec<usize>> {
    let mut hash_map = HashMap::new();
    
    for (row_idx, row) in right_table.rows.iter().enumerate() {
        let mut key_values = Vec::new();
        let mut valid_key = true;
        
        for key in keys {
            if let Some(col_idx) = right_table.headers.iter().position(|h| h == &key.right_col) {
                if col_idx < row.len() {
                    let val = &row[col_idx];
                    // Если nulls_equal=false и значение NULL, пропускаем эту строку
                    if matches!(val, Value::Null) && !nulls_equal {
                        valid_key = false;
                        break;
                    }
                    key_values.push(val.clone());
                } else {
                    valid_key = false;
                    break;
                }
            } else {
                valid_key = false;
                break;
            }
        }
        
        if valid_key {
            let key_hash = KeyHash { values: key_values };
            hash_map.entry(key_hash).or_insert_with(Vec::new).push(row_idx);
        }
    }
    
    hash_map
}

// Извлечение ключа из строки левой таблицы
fn extract_left_key(left_row: &[Value], keys: &[JoinKey], left_table: &Table) -> Option<KeyHash> {
    let mut key_values = Vec::new();
    
    for key in keys {
        if let Some(col_idx) = left_table.headers.iter().position(|h| h == &key.left_col) {
            if col_idx < left_row.len() {
                key_values.push(left_row[col_idx].clone());
            } else {
                return None;
            }
        } else {
            return None;
        }
    }
    
    Some(KeyHash { values: key_values })
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
    let right_hash = build_right_hash_table(right_table, keys, nulls_equal);
    
    let mut result_rows = Vec::new();
    let mut result_headers = Vec::new();
    
    // Создаем заголовки с учетом алиасов таблиц
    let left_headers_set: HashSet<String> = left_table.headers.iter().cloned().collect();
    let right_headers_set: HashSet<String> = right_table.headers.iter().cloned().collect();
    
    let mut left_headers = apply_column_aliases(&left_table.headers, left_alias, &right_headers_set);
    let mut right_headers = apply_column_aliases(&right_table.headers, right_alias, &left_headers_set);
    
    result_headers.append(&mut left_headers);
    result_headers.append(&mut right_headers);
    
    // Проходим по левой таблице и ищем совпадения
    for left_row in left_table.rows.iter() {
        if let Some(left_key) = extract_left_key(left_row, keys, left_table) {
            if let Some(right_indices) = right_hash.get(&left_key) {
                for &right_idx in right_indices {
                    if let Some(right_row) = right_table.get_row(right_idx) {
                        let mut new_row = left_row.clone();
                        new_row.extend_from_slice(right_row);
                        result_rows.push(new_row);
                    }
                }
            }
        }
    }
    
    Value::Table(Rc::new(RefCell::new(Table::from_data(result_rows, Some(result_headers)))))
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
    let right_hash = build_right_hash_table(right_table, keys, nulls_equal);
    
    let mut result_rows = Vec::new();
    let mut result_headers = Vec::new();
    
    // Создаем заголовки с учетом алиасов таблиц
    let left_headers_set: HashSet<String> = left_table.headers.iter().cloned().collect();
    let right_headers_set: HashSet<String> = right_table.headers.iter().cloned().collect();
    
    let mut left_headers = apply_column_aliases(&left_table.headers, left_alias, &right_headers_set);
    let mut right_headers = apply_column_aliases(&right_table.headers, right_alias, &left_headers_set);
    
    result_headers.append(&mut left_headers);
    result_headers.append(&mut right_headers);
    
    // Создаем NULL-строку для правой таблицы
    let null_right_row: Vec<Value> = (0..right_table.headers.len()).map(|_| Value::Null).collect();
    
    // Проходим по левой таблице
    for left_row in left_table.rows.iter() {
        if let Some(left_key) = extract_left_key(left_row, keys, left_table) {
            if let Some(right_indices) = right_hash.get(&left_key) {
                // Есть совпадения - добавляем все совпадения
                for &right_idx in right_indices {
                    if let Some(right_row) = right_table.get_row(right_idx) {
                        let mut new_row = left_row.clone();
                        new_row.extend_from_slice(right_row);
                        result_rows.push(new_row);
                    }
                }
            } else {
                // Нет совпадений - добавляем строку с NULL справа
                let mut new_row = left_row.clone();
                new_row.extend_from_slice(&null_right_row);
                result_rows.push(new_row);
            }
        } else {
            // Невалидный ключ - добавляем строку с NULL справа
            let mut new_row = left_row.clone();
            new_row.extend_from_slice(&null_right_row);
            result_rows.push(new_row);
        }
    }
    
    Value::Table(Rc::new(RefCell::new(Table::from_data(result_rows, Some(result_headers)))))
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
    // RIGHT JOIN - это LEFT JOIN с переставленными таблицами
    let result = perform_left_join(right_table, left_table, 
        &keys.iter().map(|k| JoinKey {
            left_col: k.right_col.clone(),
            right_col: k.left_col.clone(),
        }).collect::<Vec<_>>(),
        right_alias, left_alias, nulls_equal);
    
    // Переставляем заголовки обратно
    if let Value::Table(t) = result {
        let table_ref = t.borrow();
        // Меняем порядок заголовков: сначала right, потом left
        let mut new_headers = Vec::new();
        let right_count = right_table.headers.len();
        let left_count = left_table.headers.len();
        
        // Сначала заголовки правой таблицы
        for i in 0..right_count {
            if i < table_ref.headers.len() {
                new_headers.push(table_ref.headers[i].clone());
            }
        }
        // Потом заголовки левой таблицы
        for i in right_count..(right_count + left_count) {
            if i < table_ref.headers.len() {
                new_headers.push(table_ref.headers[i].clone());
            }
        }
        
        // Переставляем данные в строках тоже
        let mut new_rows = Vec::new();
        for row in &table_ref.rows {
            let mut new_row = Vec::new();
            // Сначала данные правой таблицы
            for i in 0..right_count {
                if i < row.len() {
                    new_row.push(row[i].clone());
                }
            }
            // Потом данные левой таблицы
            for i in right_count..(right_count + left_count) {
                if i < row.len() {
                    new_row.push(row[i].clone());
                }
            }
            new_rows.push(new_row);
        }
        
        // Освобождаем заимствование перед созданием новой таблицы
        drop(table_ref);
        Value::Table(Rc::new(RefCell::new(Table::from_data(new_rows, Some(new_headers)))))
    } else {
        result
    }
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
    let right_hash = build_right_hash_table(right_table, keys, nulls_equal);
    
    let mut result_rows = Vec::new();
    let mut result_headers = Vec::new();
    
    // Создаем заголовки с учетом алиасов таблиц
    let left_headers_set: HashSet<String> = left_table.headers.iter().cloned().collect();
    let right_headers_set: HashSet<String> = right_table.headers.iter().cloned().collect();
    
    let mut left_headers = apply_column_aliases(&left_table.headers, left_alias, &right_headers_set);
    let mut right_headers = apply_column_aliases(&right_table.headers, right_alias, &left_headers_set);
    
    result_headers.append(&mut left_headers);
    result_headers.append(&mut right_headers);
    
    // Создаем NULL-строки
    let null_left_row: Vec<Value> = (0..left_table.headers.len()).map(|_| Value::Null).collect();
    let null_right_row: Vec<Value> = (0..right_table.headers.len()).map(|_| Value::Null).collect();
    
    let mut matched_right_indices = HashSet::new();
    
    // Проходим по левой таблице
    for left_row in &left_table.rows {
        if let Some(left_key) = extract_left_key(left_row, keys, left_table) {
            if let Some(right_indices) = right_hash.get(&left_key) {
                for &right_idx in right_indices {
                    matched_right_indices.insert(right_idx);
                    if let Some(right_row) = right_table.get_row(right_idx) {
                        let mut new_row = left_row.clone();
                        new_row.extend_from_slice(right_row);
                        result_rows.push(new_row);
                    }
                }
            } else {
                // Нет совпадений справа - добавляем строку с NULL справа
                let mut new_row = left_row.clone();
                new_row.extend_from_slice(&null_right_row);
                result_rows.push(new_row);
            }
        } else {
            // Невалидный ключ - добавляем строку с NULL справа
            let mut new_row = left_row.clone();
            new_row.extend_from_slice(&null_right_row);
            result_rows.push(new_row);
        }
    }
    
    // Проходим по правой таблице и добавляем несовпадающие строки
    for (right_idx, right_row) in right_table.rows.iter().enumerate() {
        if !matched_right_indices.contains(&right_idx) {
            let mut new_row = null_left_row.clone();
            new_row.extend_from_slice(right_row);
            result_rows.push(new_row);
        }
    }
    
    Value::Table(Rc::new(RefCell::new(Table::from_data(result_rows, Some(result_headers)))))
}

// Выполнение SEMI JOIN (только строки left, колонки right не включаются)
fn perform_semi_join(
    left_table: &Table,
    right_table: &Table,
    keys: &[JoinKey],
    nulls_equal: bool,
) -> Value {
    let right_hash = build_right_hash_table(right_table, keys, nulls_equal);
    
    let mut result_rows = Vec::new();
    
    // Проходим по левой таблице и проверяем наличие совпадений
    for left_row in &left_table.rows {
        if let Some(left_key) = extract_left_key(left_row, keys, left_table) {
            if right_hash.contains_key(&left_key) {
                // Есть совпадение - добавляем строку из left
                result_rows.push(left_row.clone());
            }
        }
    }
    
    Value::Table(Rc::new(RefCell::new(Table::from_data(result_rows, Some(left_table.headers.clone())))))
}

// Выполнение ANTI JOIN (строки left без совпадений в right)
fn perform_anti_join(
    left_table: &Table,
    right_table: &Table,
    keys: &[JoinKey],
    nulls_equal: bool,
) -> Value {
    let right_hash = build_right_hash_table(right_table, keys, nulls_equal);
    
    let mut result_rows = Vec::new();
    
    // Проходим по левой таблице и проверяем отсутствие совпадений
    for left_row in &left_table.rows {
        if let Some(left_key) = extract_left_key(left_row, keys, left_table) {
            if !right_hash.contains_key(&left_key) {
                // Нет совпадения - добавляем строку из left
                result_rows.push(left_row.clone());
            }
        } else {
            // Невалидный ключ - добавляем строку
            result_rows.push(left_row.clone());
        }
    }
    
    Value::Table(Rc::new(RefCell::new(Table::from_data(result_rows, Some(left_table.headers.clone())))))
}

// Выполнение CROSS JOIN (декартово произведение)
fn perform_cross_join(
    left_table: &Table,
    right_table: &Table,
    left_alias: &str,
    right_alias: &str,
) -> Value {
    let mut result_rows = Vec::new();
    let mut result_headers = Vec::new();
    
    // Создаем заголовки с учетом алиасов таблиц
    let left_headers_set: HashSet<String> = left_table.headers.iter().cloned().collect();
    let right_headers_set: HashSet<String> = right_table.headers.iter().cloned().collect();
    
    let mut left_headers = apply_column_aliases(&left_table.headers, left_alias, &right_headers_set);
    let mut right_headers = apply_column_aliases(&right_table.headers, right_alias, &left_headers_set);
    
    result_headers.append(&mut left_headers);
    result_headers.append(&mut right_headers);
    
    // Декартово произведение
    for left_row in &left_table.rows {
        for right_row in &right_table.rows {
            let mut new_row = left_row.clone();
            new_row.extend_from_slice(right_row);
            result_rows.push(new_row);
        }
    }
    
    Value::Table(Rc::new(RefCell::new(Table::from_data(result_rows, Some(result_headers)))))
}

// Универсальная функция JOIN для таблиц
pub fn native_table_join(args: &[Value]) -> Value {
    if args.len() < 3 {
        return Value::Null;
    }

    // Извлекаем left и right таблицы
    let left_table = match &args[0] {
        Value::Table(t) => t.borrow().clone(),
        _ => return Value::Null,
    };
    
    let right_table = match &args[1] {
        Value::Table(t) => t.borrow().clone(),
        _ => return Value::Null,
    };

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
    let left_alias = left_table.name.as_ref()
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
    
    let right_alias = right_table.name.as_ref()
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
        JoinType::Inner => perform_inner_join(&left_table, &right_table, &keys, &left_alias, &right_alias, nulls_equal),
        JoinType::Left => perform_left_join(&left_table, &right_table, &keys, &left_alias, &right_alias, nulls_equal),
        JoinType::Right => perform_right_join(&left_table, &right_table, &keys, &left_alias, &right_alias, nulls_equal),
        JoinType::Full => perform_full_join(&left_table, &right_table, &keys, &left_alias, &right_alias, nulls_equal),
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
    let mut result = Vec::new();
    
    // Сортируем правую таблицу по времени (для эффективного поиска)
    let mut right_indices: Vec<usize> = (0..right_rows.len()).collect();
    right_indices.sort_by(|&a, &b| {
        let time_a = &right_rows[a][right_time_idx];
        let time_b = &right_rows[b][right_time_idx];
        compare_values(time_a, time_b)
    });
    
    for left_row in left_rows {
        let left_time = &left_row[left_time_idx];
        
        // Ищем ближайшую строку в правой таблице
        let mut best_match: Option<usize> = None;
        let mut best_diff: Option<f64> = None;
        
        for &right_idx in &right_indices {
            let right_row = &right_rows[right_idx];
            let right_time = &right_row[right_time_idx];
            
            // Вычисляем разницу времени (упрощенная версия - только для чисел)
            let diff = match (left_time, right_time) {
                (Value::Number(l), Value::Number(r)) => {
                    let diff_val = match direction {
                        "backward" => *l - *r,  // left_time >= right_time
                        "forward" => *r - *l,   // right_time >= left_time
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
                        Some(bd) => match direction {
                            "nearest" => d < bd,
                            _ => d < bd,
                        },
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
            // Нет совпадения - добавляем строку с NULL справа (для left join семантики)
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

    let left_table = match &args[0] {
        Value::Table(t) => t.borrow().clone(),
        _ => return Value::Null,
    };
    
    let right_table = match &args[1] {
        Value::Table(t) => t.borrow().clone(),
        _ => return Value::Null,
    };

    // Используем имена таблиц из table.name, если они установлены
    let left_alias = left_table.name.as_ref()
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
    
    let right_alias = right_table.name.as_ref()
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

    let mut result_rows = Vec::new();
    let mut result_headers = Vec::new();
    
    // Создаем заголовки с учетом алиасов таблиц
    let left_headers_set: HashSet<String> = left_table.headers.iter().cloned().collect();
    let right_headers_set: HashSet<String> = right_table.headers.iter().cloned().collect();
    
    let mut left_headers = apply_column_aliases(&left_table.headers, &left_alias, &right_headers_set);
    let mut right_headers = apply_column_aliases(&right_table.headers, &right_alias, &left_headers_set);
    
    result_headers.append(&mut left_headers);
    result_headers.append(&mut right_headers);
    
    // Соединяем строки по позиции (индексу)
    let min_len = std::cmp::min(left_table.rows.len(), right_table.rows.len());
    for i in 0..min_len {
        let mut new_row = left_table.rows[i].clone();
        new_row.extend_from_slice(&right_table.rows[i]);
        result_rows.push(new_row);
    }
    
    Value::Table(Rc::new(RefCell::new(Table::from_data(result_rows, Some(result_headers)))))
}

// APPLY JOIN / LATERAL JOIN - для каждой строки left вызывает функцию
pub fn native_apply_join(args: &[Value]) -> Value {
    use super::utils::call_user_function;
    use crate::vm::vm::VM_CALL_CONTEXT;
    
    if args.len() < 2 {
        return Value::Null;
    }

    let left_table = match &args[0] {
        Value::Table(t) => t.borrow().clone(),
        _ => return Value::Null,
    };

    // Извлекаем функцию из аргументов
    let function_index = match &args[1] {
        Value::Function(idx) => *idx,
        _ => {
            // Если функция не передана, возвращаем Null
            return Value::Null;
        }
    };

    // Извлекаем тип JOIN (по умолчанию "inner")
    let join_type = if args.len() > 2 {
        match &args[2] {
            Value::String(s) => s.as_str(),
            _ => "inner",
        }
    } else {
        "inner"
    };

    // Получаем доступ к VM через thread-local storage
    let vm_ptr = VM_CALL_CONTEXT.with(|ctx| {
        let ctx_ref = ctx.borrow();
        *ctx_ref
    });
    
    let mut result_rows = Vec::new();
    let mut result_headers = Vec::new();

    // Создаем заголовки результата (начнем с заголовков левой таблицы)
    result_headers.extend_from_slice(&left_table.headers);

    // Отслеживаем максимальное количество колонок правой таблицы для корректной обработки NULLs
    let mut max_right_columns = 0;

    // Для каждой строки левой таблицы вызываем функцию
    for left_row in &left_table.rows {
        // Восстанавливаем контекст перед каждым вызовом
        if let Some(vm_ptr) = vm_ptr {
            VM_CALL_CONTEXT.with(|ctx| {
                *ctx.borrow_mut() = Some(vm_ptr);
            });
        }
        
        // Вызываем функцию с аргументом - массив значений строки
        let row_array = Value::Array(Rc::new(RefCell::new(left_row.clone())));
        let function_result = match call_user_function(function_index, &[row_array]) {
            Ok(result) => result,
            Err(_e) => {
                // Если произошла ошибка при вызове функции, пропускаем строку для inner join
                // или добавляем с NULLs для left join
                if join_type == "left" {
                    let new_row = left_row.clone();
                    result_rows.push(new_row);
                }
                continue;
            }
        };
        
        match function_result {
            Value::Table(right_table) => {
                let right_table_ref = right_table.borrow();
                
                // Обновляем максимальное количество колонок
                if right_table_ref.headers.len() > max_right_columns {
                    max_right_columns = right_table_ref.headers.len();
                }
                
                // Если заголовки результата еще не установлены полностью, добавляем заголовки правой таблицы
                if result_headers.len() == left_table.headers.len() {
                    // Проверяем конфликты имен колонок
                    let left_headers_set: HashSet<String> = 
                        left_table.headers.iter().cloned().collect();
                    let mut right_headers = Vec::new();
                    for header in &right_table_ref.headers {
                        if left_headers_set.contains(header) {
                            right_headers.push(format!("right_{}", header));
                        } else {
                            right_headers.push(header.clone());
                        }
                    }
                    result_headers.extend_from_slice(&right_headers);
                }

                // Для каждой строки в результате функции добавляем комбинацию left_row + right_row
                for right_row in &right_table_ref.rows {
                    let mut new_row = left_row.clone();
                    new_row.extend_from_slice(right_row);
                    result_rows.push(new_row);
                }
            }
            Value::Null => {
                // Если функция вернула Null и это left join, добавляем строку с NULLs
                if join_type == "left" {
                    let new_row = left_row.clone();
                    result_rows.push(new_row);
                }
            }
            _ => {
                // Если функция вернула что-то другое, игнорируем для inner join
                // или добавляем с NULLs для left join
                if join_type == "left" {
                    let new_row = left_row.clone();
                    result_rows.push(new_row);
                }
            }
        }
    }

    // Если это left join и были строки без правой части, добавляем NULL значения
    if join_type == "left" && max_right_columns > 0 {
        // Добавляем NULL значения для строк, которые не имеют правой части
        let expected_length = left_table.headers.len() + max_right_columns;
        for row in &mut result_rows {
            while row.len() < expected_length {
                row.push(Value::Null);
            }
        }
    }

    Value::Table(Rc::new(RefCell::new(Table::from_data(result_rows, Some(result_headers)))))
}

// ASOF JOIN - временное соединение
pub fn native_asof_join(args: &[Value]) -> Value {
    if args.len() < 3 {
        return Value::Null;
    }

    let left_table = match &args[0] {
        Value::Table(t) => t.borrow().clone(),
        _ => return Value::Null,
    };
    
    let right_table = match &args[1] {
        Value::Table(t) => t.borrow().clone(),
        _ => return Value::Null,
    };

    // Парсим временную колонку
    let time_column = match &args[2] {
        Value::String(s) => s.clone(),
        _ => return Value::Null,
    };

    // Проверяем наличие временной колонки в обеих таблицах
    if left_table.get_column(&time_column).is_none() {
        return Value::Null;
    }
    if right_table.get_column(&time_column).is_none() {
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
    let left_alias = left_table.name.as_ref()
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
    
    let right_alias = right_table.name.as_ref()
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
    let left_time_idx = left_table.headers.iter().position(|h| h == &time_column).unwrap();
    let right_time_idx = right_table.headers.iter().position(|h| h == &time_column).unwrap();

    // Если есть by колонки, группируем данные
    let mut result_rows = Vec::new();
    let mut result_headers = Vec::new();
    
    // Создаем заголовки с учетом алиасов таблиц
    let left_headers_set: HashSet<String> = left_table.headers.iter().cloned().collect();
    let right_headers_set: HashSet<String> = right_table.headers.iter().cloned().collect();
    
    let mut left_headers = apply_column_aliases(&left_table.headers, &left_alias, &right_headers_set);
    let mut right_headers = apply_column_aliases(&right_table.headers, &right_alias, &left_headers_set);
    
    result_headers.append(&mut left_headers);
    result_headers.append(&mut right_headers);

    // Создаем NULL-строку для правой таблицы
    let null_right_row: Vec<Value> = (0..right_table.headers.len()).map(|_| Value::Null).collect();

    if by_columns.is_empty() {
        // Нет группировки - простой ASOF join
        // Сортируем правую таблицу по времени (для бинарного поиска)
        let mut right_indices: Vec<usize> = (0..right_table.rows.len()).collect();
        right_indices.sort_by(|&a, &b| {
            let time_a = &right_table.rows[a][right_time_idx];
            let time_b = &right_table.rows[b][right_time_idx];
            compare_values(time_a, time_b)
        });

        for left_row in &left_table.rows {
            let left_time = &left_row[left_time_idx];
            
            // Ищем ближайшую строку в правой таблице
            let mut best_match: Option<usize> = None;
            let mut best_diff: Option<f64> = None;

            for &right_idx in &right_indices {
                let right_row = &right_table.rows[right_idx];
                let right_time = &right_row[right_time_idx];

                // Вычисляем разницу времени (упрощенная версия - только для чисел)
                let diff = match (left_time, right_time) {
                    (Value::Number(l), Value::Number(r)) => {
                        let diff_val = match direction {
                            "backward" => *l - *r,  // left_time >= right_time
                            "forward" => *r - *l,   // right_time >= left_time
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
                            Some(bd) => match direction {
                                "nearest" => d < bd,
                                _ => d < bd,
                            },
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
                new_row.extend_from_slice(&right_table.rows[right_idx]);
                result_rows.push(new_row);
            } else {
                // Нет совпадения - добавляем строку с NULL справа (для left join семантики)
                let mut new_row = left_row.clone();
                new_row.extend_from_slice(&null_right_row);
                result_rows.push(new_row);
            }
        }
    } else {
        // Есть группировка - группируем по by колонкам
        // Получаем индексы by колонок
        let mut by_indices_left = Vec::new();
        let mut by_indices_right = Vec::new();
        
        for by_col in &by_columns {
            if let Some(idx) = left_table.headers.iter().position(|h| h == by_col) {
                by_indices_left.push(idx);
            } else {
                return Value::Null; // Колонка не найдена
            }
            if let Some(idx) = right_table.headers.iter().position(|h| h == by_col) {
                by_indices_right.push(idx);
            } else {
                return Value::Null; // Колонка не найдена
            }
        }
        
        // Группируем левую таблицу по by колонкам
        let mut left_groups: HashMap<Vec<Value>, Vec<Vec<Value>>> = HashMap::new();
        for row in &left_table.rows {
            let key: Vec<Value> = by_indices_left.iter().map(|&idx| row[idx].clone()).collect();
            left_groups.entry(key).or_insert_with(Vec::new).push(row.clone());
        }
        
        // Группируем правую таблицу по by колонкам
        let mut right_groups: HashMap<Vec<Value>, Vec<Vec<Value>>> = HashMap::new();
        for row in &right_table.rows {
            let key: Vec<Value> = by_indices_right.iter().map(|&idx| row[idx].clone()).collect();
            right_groups.entry(key).or_insert_with(Vec::new).push(row.clone());
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

    Value::Table(Rc::new(RefCell::new(Table::from_data(result_rows, Some(result_headers)))))
}

// JOIN ON - non-equi join с произвольным условием
pub fn native_join_on(args: &[Value]) -> Value {
    if args.len() < 3 {
        return Value::Null;
    }

    let left_table = match &args[0] {
        Value::Table(t) => t.borrow().clone(),
        _ => return Value::Null,
    };
    
    let right_table = match &args[1] {
        Value::Table(t) => t.borrow().clone(),
        _ => return Value::Null,
    };

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
    let left_alias = left_table.name.as_ref()
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
    
    let right_alias = right_table.name.as_ref()
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
                (parts[0].to_string(), parts[1].to_string(), parts[2].to_string())
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

    // Проверяем наличие колонок
    if left_table.get_column(&left_col).is_none() {
        return Value::Null;
    }
    if right_table.get_column(&right_col).is_none() {
        return Value::Null;
    }

    let left_col_idx = left_table.headers.iter().position(|h| h == &left_col).unwrap();
    let right_col_idx = right_table.headers.iter().position(|h| h == &right_col).unwrap();

    let mut result_rows = Vec::new();
    let mut result_headers = Vec::new();
    
    // Создаем заголовки с учетом алиасов таблиц
    let left_headers_set: HashSet<String> = left_table.headers.iter().cloned().collect();
    let right_headers_set: HashSet<String> = right_table.headers.iter().cloned().collect();
    
    let mut left_headers = apply_column_aliases(&left_table.headers, &left_alias, &right_headers_set);
    let mut right_headers = apply_column_aliases(&right_table.headers, &right_alias, &left_headers_set);
    
    result_headers.append(&mut left_headers);
    result_headers.append(&mut right_headers);

    // Создаем NULL-строки
    let null_left_row: Vec<Value> = (0..left_table.headers.len()).map(|_| Value::Null).collect();
    let null_right_row: Vec<Value> = (0..right_table.headers.len()).map(|_| Value::Null).collect();

    // Nested loop join с проверкой условия
    let mut matched_right_indices = HashSet::new();

    for left_row in &left_table.rows {
        let left_val = &left_row[left_col_idx];
        let mut found_match = false;

        for (right_idx, right_row) in right_table.rows.iter().enumerate() {
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
                let mut new_row = left_row.clone();
                new_row.extend_from_slice(right_row);
                result_rows.push(new_row);
            }
        }

        // Для LEFT JOIN добавляем строки без совпадений
        if !found_match && (join_type == JoinType::Left || join_type == JoinType::Full) {
            let mut new_row = left_row.clone();
            new_row.extend_from_slice(&null_right_row);
            result_rows.push(new_row);
        }
    }

    // Для RIGHT и FULL JOIN добавляем несовпадающие строки справа
    if join_type == JoinType::Right || join_type == JoinType::Full {
        for (right_idx, right_row) in right_table.rows.iter().enumerate() {
            if !matched_right_indices.contains(&right_idx) {
                let mut new_row = null_left_row.clone();
                new_row.extend_from_slice(right_row);
                result_rows.push(new_row);
            }
        }
    }

    Value::Table(Rc::new(RefCell::new(Table::from_data(result_rows, Some(result_headers)))))
}

// Применение суффиксов к колонкам таблицы после join
pub fn native_table_suffixes(args: &[Value]) -> Value {
    if args.len() < 3 {
        return Value::Null;
    }

    // Аргументы приходят в порядке: [table, left_suffix, right_suffix]
    let table = match &args[0] {
        Value::Table(t) => t.borrow().clone(),
        _ => return Value::Null,
    };

    // Извлекаем суффиксы
    let left_suffix = match &args[1] {
        Value::String(s) => s.clone(),
        _ => return Value::Null,
    };

    let right_suffix = match &args[2] {
        Value::String(s) => s.clone(),
        _ => return Value::Null,
    };

    // Создаем новую таблицу с переименованными колонками
    let mut new_headers = Vec::new();
    let mut column_mapping = HashMap::new(); // старое имя -> новое имя

    // Определяем, какие колонки относятся к левой таблице, а какие к правой
    let mut seen_prefixes = Vec::new();
    let mut prefix_to_table = HashMap::new(); // prefix -> "left" или "right"
    
    // Сначала проходим по всем заголовкам и собираем уникальные префиксы
    for header in &table.headers {
        if let Some(dot_pos) = header.find('.') {
            let prefix = &header[..dot_pos];
            if !seen_prefixes.contains(&prefix.to_string()) {
                seen_prefixes.push(prefix.to_string());
            }
        }
    }
    
    // Первый уникальный префикс относится к левой таблице, остальные - к правой
    for (i, prefix) in seen_prefixes.iter().enumerate() {
        if i == 0 {
            prefix_to_table.insert(prefix.clone(), "left");
        } else {
            prefix_to_table.insert(prefix.clone(), "right");
        }
    }
    
    // Также добавляем стандартные префиксы
    prefix_to_table.insert("left".to_string(), "left");
    prefix_to_table.insert("right".to_string(), "right");
    
    // Проходим по всем заголовкам и переименовываем колонки с префиксами
    for header in &table.headers {
        let new_header = if let Some(dot_pos) = header.find('.') {
            let prefix = &header[..dot_pos];
            let base_name = &header[dot_pos + 1..];
            
            // Определяем, к какой таблице относится колонка
            if let Some(table_side) = prefix_to_table.get(prefix) {
                if *table_side == "left" {
                    // Колонка левой таблицы - применяем left_suffix
                    let new_name = format!("{}{}", base_name, left_suffix);
                    column_mapping.insert(header.clone(), new_name.clone());
                    new_name
                } else {
                    // Колонка правой таблицы - применяем right_suffix
                    let new_name = format!("{}{}", base_name, right_suffix);
                    column_mapping.insert(header.clone(), new_name.clone());
                    new_name
                }
            } else {
                // Неизвестный префикс - оставляем без изменений
                column_mapping.insert(header.clone(), header.clone());
                header.clone()
            }
        } else {
            // Колонка без префикса - оставляем без изменений
            column_mapping.insert(header.clone(), header.clone());
            header.clone()
        };
        new_headers.push(new_header);
    }

    // Создаем новую структуру колонок с переименованными ключами
    let mut new_columns = HashMap::new();
    for (old_header, new_header) in &column_mapping {
        if let Some(column_data) = table.columns.get(old_header) {
            new_columns.insert(new_header.clone(), column_data.clone());
        }
    }

    // Создаем новую таблицу
    let mut new_table = Table::new();
    new_table.headers = new_headers;
    new_table.columns = new_columns;
    new_table.rows = table.rows.clone();
    new_table.name = table.name.clone();

    Value::Table(Rc::new(RefCell::new(new_table)))
}
