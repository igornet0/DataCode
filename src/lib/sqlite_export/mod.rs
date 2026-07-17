// Модуль для экспорта таблиц DataCode в SQLite

use crate::common::table::{Table, TableData};
use crate::common::value::Value;
use crate::vm::Vm;
use chrono::Utc;
use rusqlite::types::{ToSqlOutput, ValueRef};
use rusqlite::{params, Connection, Result as SqliteResult};
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::rc::Rc;
use std::time::Instant;

/// Адаптер для привязки [`Value`] к параметрам SQLite без боксинга на каждую ячейку.
struct ValueParam<'a>(&'a Value);

impl rusqlite::ToSql for ValueParam<'_> {
    fn to_sql(&self) -> rusqlite::Result<ToSqlOutput<'_>> {
        Ok(match self.0 {
            Value::Number(n) if n.fract() == 0.0 => ToSqlOutput::Owned(rusqlite::types::Value::Integer(*n as i64)),
            Value::Number(n) => ToSqlOutput::Owned(rusqlite::types::Value::Real(*n)),
            Value::Bool(b) => ToSqlOutput::Owned(rusqlite::types::Value::Integer(if *b {
                1
            } else {
                0
            })),
            Value::String(s) => ToSqlOutput::Borrowed(ValueRef::Text(s.as_bytes())),
            Value::Date(d) => ToSqlOutput::Owned(rusqlite::types::Value::Text(d.to_rfc3339())),
            Value::Duration(d) => {
                let sec = d.num_seconds() as f64 + d.subsec_nanos() as f64 * 1e-9;
                ToSqlOutput::Owned(rusqlite::types::Value::Real(sec))
            }
            Value::Null => ToSqlOutput::Owned(rusqlite::types::Value::Null),
            other => ToSqlOutput::Owned(rusqlite::types::Value::Text(other.to_string())),
        })
    }
}

/// Структура для хранения информации о таблице для экспорта
struct TableInfo {
    name: String,
    table: Rc<RefCell<Table>>,
    sqlite_name: String,
}

/// Структура для информации о первичном ключе
struct PrimaryKeyInfo {
    table_name: String,
    column_name: String,
}

/// Структура для информации о внешнем ключе
#[derive(Clone)]
struct ForeignKeyInfo {
    table_name: String,
    column_name: String,
    referenced_table: String,
    referenced_column: String,
}

/// Главная функция экспорта в SQLite (глобальные таблицы VM).
pub fn export_to_sqlite(vm: &mut Vm, output_path: &str, debug_timings: bool) -> Result<(), String> {
    let tables_map = get_global_tables(vm)?;
    if tables_map.is_empty() {
        return Err("Нет таблиц для экспорта".to_string());
    }
    let tables: Vec<(String, Rc<RefCell<Table>>)> = tables_map.into_iter().collect();
    export_tables_to_sqlite(vm, &tables, output_path, &HashMap::new(), debug_timings)
}

/// Экспорт указанного набора таблиц в SQLite.
pub fn export_tables_to_sqlite(
    vm: &mut Vm,
    tables: &[(String, Rc<RefCell<Table>>)],
    output_path: &str,
    extra_variables: &HashMap<String, Value>,
    debug_timings: bool,
) -> Result<(), String> {
    let t_total = Instant::now();
    if tables.is_empty() {
        return Err("Нет таблиц для экспорта".to_string());
    }

    vm.flush_pending_schema_metadata();

    let mut conn =
        Connection::open(output_path).map_err(|e| format!("Ошибка создания базы данных: {}", e))?;

    conn.execute_batch(
        "
        PRAGMA journal_mode = MEMORY;
        PRAGMA synchronous = OFF;
        PRAGMA temp_store = MEMORY;
        PRAGMA locking_mode = EXCLUSIVE;
        PRAGMA cache_size = -65536;
        PRAGMA foreign_keys = OFF;
    ",
    )
    .map_err(|e| format!("Ошибка PRAGMA для быстрой загрузки: {}", e))?;

    let mut table_infos = Vec::new();
    for (var_name, table) in tables {
        let sqlite_name = sanitize_table_name(var_name);
        table_infos.push(TableInfo {
            name: var_name.clone(),
            table: table.clone(),
            sqlite_name,
        });
    }

    materialize_view_tables_for_sqlite(vm, &table_infos)?;

    let t = Instant::now();
    let primary_keys = detect_primary_keys(&table_infos, vm)?;
    if debug_timings {
        eprintln!("[sqlite_export] detect_primary_keys: {:?}", t.elapsed());
    }

    let explicit_foreign_keys = get_explicit_foreign_keys(&table_infos, vm);

    let t = Instant::now();
    let foreign_keys = detect_foreign_keys(&table_infos, &primary_keys, &explicit_foreign_keys)?;
    if debug_timings {
        eprintln!("[sqlite_export] detect_foreign_keys: {:?}", t.elapsed());
    }

    let sorted_indices: Vec<usize> = if foreign_keys.is_empty() {
        (0..table_infos.len()).collect()
    } else {
        topological_sort_tables(&table_infos, &foreign_keys)?
    };

    let mut fk_by_table: HashMap<String, Vec<&ForeignKeyInfo>> = HashMap::new();
    for fk in &foreign_keys {
        fk_by_table
            .entry(fk.table_name.clone())
            .or_insert_with(Vec::new)
            .push(fk);
    }

    let mut pk_by_table: HashMap<String, &PrimaryKeyInfo> = HashMap::new();
    for pk in &primary_keys {
        pk_by_table.insert(pk.table_name.clone(), pk);
    }

    let t = Instant::now();
    let tx = conn
        .transaction()
        .map_err(|e| format!("Ошибка начала транзакции: {}", e))?;

    for &table_idx in &sorted_indices {
        create_and_fill_table(&tx, &table_infos[table_idx], &pk_by_table, &fk_by_table)
            .map_err(|e| format!("Ошибка экспорта таблицы {}: {}", table_infos[table_idx].name, e))?;
    }

    create_indexes_impl(&tx, &foreign_keys)
        .map_err(|e| format!("Ошибка создания индексов: {}", e))?;

    tx.commit()
        .map_err(|e| format!("Ошибка коммита транзакции: {}", e))?;

    if debug_timings {
        eprintln!("[sqlite_export] create tables + insert + indexes: {:?}", t.elapsed());
    }

    conn.execute("PRAGMA foreign_keys = ON", [])
        .map_err(|e| format!("Ошибка включения FOREIGN KEY после загрузки: {}", e))?;

    let mut fk_check = conn
        .prepare("PRAGMA foreign_key_check")
        .map_err(|e| format!("Ошибка PRAGMA foreign_key_check: {}", e))?;
    let mut violations = fk_check
        .query([])
        .map_err(|e| format!("Ошибка выполнения foreign_key_check: {}", e))?;
    if violations
        .next()
        .map_err(|e| format!("Ошибка чтения foreign_key_check: {}", e))?
        .is_some()
    {
        return Err(
            "Ошибка целостности: foreign_key_check обнаружил нарушения после экспорта".to_string(),
        );
    }

    let t = Instant::now();
    let tables_map: HashMap<String, Rc<RefCell<Table>>> =
        tables.iter().map(|(n, t)| (n.clone(), t.clone())).collect();
    create_metadata_table(&conn, vm, &tables_map, extra_variables)
        .map_err(|e| format!("Ошибка создания таблицы метаданных: {}", e))?;
    if debug_timings {
        eprintln!("[sqlite_export] create_metadata_table: {:?}", t.elapsed());
        eprintln!("[sqlite_export] total: {:?}", t_total.elapsed());
    }

    println!("✅ Экспорт завершен: {}", output_path);
    Ok(())
}

/// Export a single materialized table to a new SQLite database file.
pub fn export_single_table(
    table: &Table,
    output_path: &Path,
    table_name: &str,
) -> Result<(), String> {
    let sqlite_name = sanitize_table_name(table_name);
    let table_rc = Rc::new(RefCell::new(table.clone()));
    let table_info = TableInfo {
        name: table_name.to_string(),
        table: table_rc,
        sqlite_name,
    };
    let mut conn = Connection::open(output_path)
        .map_err(|e| format!("Ошибка создания базы данных: {}", e))?;
    conn.execute_batch(
        "
        PRAGMA journal_mode = MEMORY;
        PRAGMA synchronous = OFF;
        PRAGMA foreign_keys = OFF;
    ",
    )
    .map_err(|e| format!("Ошибка PRAGMA: {}", e))?;
    let tx = conn
        .transaction()
        .map_err(|e| format!("Ошибка начала транзакции: {}", e))?;
    let empty_pk: HashMap<String, &PrimaryKeyInfo> = HashMap::new();
    let empty_fk: HashMap<String, Vec<&ForeignKeyInfo>> = HashMap::new();
    create_and_fill_table(&tx, &table_info, &empty_pk, &empty_fk)
        .map_err(|e| format!("Ошибка экспорта таблицы: {}", e))?;
    tx.commit()
        .map_err(|e| format!("Ошибка коммита транзакции: {}", e))?;
    Ok(())
}

/// Получить все таблицы из глобальных переменных VM (globals are GlobalSlot)
pub fn get_global_tables(
    vm: &mut crate::vm::vm::Vm,
) -> Result<HashMap<String, Rc<RefCell<Table>>>, String> {
    use crate::vm::store_convert::load_value;
    let mut tables = HashMap::new();
    let globals = vm.get_globals();
    let mut names = vm.get_global_names().clone();
    for (idx, name) in vm.get_explicit_global_names() {
        names.insert(*idx, name.clone());
    }

    let to_scan: Vec<(usize, String)> = globals
        .iter()
        .enumerate()
        .filter_map(|(index, _)| names.get(&index).map(|name| (index, name.clone())))
        .collect();
    for (index, var_name) in to_scan {
        let value_id = vm.resolve_global_to_value_id(index);
        let value = load_value(value_id, vm.value_store(), vm.heavy_store());
        if matches!(&value, Value::NativeFunction(_) | Value::Function(_)) {
            continue;
        }
        if let Value::Table(table) = &value {
            tables.insert(var_name, table.clone());
        }
    }
    Ok(tables)
}

#[cfg(test)]
mod tests {
    use super::{export_to_sqlite, get_global_tables};
    use rusqlite::Connection;

    #[test]
    fn get_global_tables_finds_script_local_tables() {
        let source = r#"
users = table([[1, "Alice"]], ["id", "name"])
orders = table([[1, 100]], ["user_id", "amount"])
"#;
        let (_v, mut vm) = crate::run_with_vm(source).expect("run_with_vm should succeed");
        let tables = get_global_tables(&mut vm).expect("get_global_tables should succeed");
        assert!(
            tables.contains_key("users") && tables.contains_key("orders"),
            "expected users and orders, got keys={:?}",
            tables.keys().collect::<Vec<_>>()
        );
    }

    #[test]
    fn get_global_tables_falls_back_to_global_names() {
        let source = r#"
global users = table([
    [1, "Алиса", "Разработка"],
    [2, "Боб", "Маркетинг"]
], ["id", "name", "department"])
print(len(users))
"#;

        let (_v, mut vm) = crate::run_with_vm(source).expect("run_with_vm should succeed");
        let tables = get_global_tables(&mut vm).expect("get_global_tables should succeed");
        assert!(
            tables.contains_key("users"),
            "expected users table in globals, got keys={:?}",
            tables.keys().collect::<Vec<_>>()
        );
    }

    #[test]
    fn sqlite_export_one_pass_and_foreign_key_list() {
        let source = r#"
global users = table([[1, "a"], [2, "b"]], ["id", "name"])
global orders = table([[1, 1, 100], [2, 2, 200]], ["id", "user_id", "amount"])
"#;
        let (_v, mut vm) = crate::run_with_vm(source).expect("run_with_vm should succeed");
        let dir = tempfile::tempdir().expect("tempdir");
        let db_path = dir.path().join("export_test.db");
        export_to_sqlite(&mut vm, db_path.to_str().unwrap(), false).expect("export");

        let conn = Connection::open(&db_path).expect("open db");
        conn.execute("PRAGMA foreign_keys = ON", []).unwrap();
        let mut stmt = conn
            .prepare("SELECT COUNT(*) FROM pragma_foreign_key_list('orders')")
            .unwrap();
        let n: i64 = stmt.query_row([], |r| r.get(0)).unwrap();
        assert!(n >= 1, "expected at least one FK on orders, got {}", n);

        let user_count: i64 = conn
            .query_row("SELECT COUNT(*) FROM users", [], |r| r.get(0))
            .unwrap();
        let order_count: i64 = conn
            .query_row("SELECT COUNT(*) FROM orders", [], |r| r.get(0))
            .unwrap();
        assert_eq!(user_count, 2);
        assert_eq!(order_count, 2);
    }
}

/// Таблицы в VM могут быть `TableData::View` (например `table([...])`); для экспорта нужен flat-owned ряд.
fn materialize_view_tables_for_sqlite(vm: &Vm, table_infos: &[TableInfo]) -> Result<(), String> {
    use crate::vm::store_convert::load_value;
    for ti in table_infos {
        let materialized = {
            let table = ti.table.borrow();
            if table.is_view() {
                let load = |id| load_value(id, vm.value_store(), vm.heavy_store());
                Some(table.materialize_with(load))
            } else {
                None
            }
        };
        if let Some(owned) = materialized {
            *ti.table.borrow_mut() = owned;
        }
    }
    Ok(())
}

/// Санитизация имени таблицы для SQLite
fn sanitize_table_name(name: &str) -> String {
    name.chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

/// Тип SQLite для колонки по данным (ранний выход при первой строке — TEXT).
fn infer_column_type_for_header(table: &Table, header: &str) -> String {
    let Some(col_idx) = table.headers().iter().position(|h| h == header) else {
        return "TEXT".to_string();
    };
    match &table.data {
        TableData::Owned {
            flat,
            num_cols,
            ..
        } => {
            if *num_cols == 0 {
                return "TEXT".to_string();
            }
            let nrows = flat.len() / num_cols;
            let mut has_integer = false;
            let mut has_float = false;
            let mut has_bool = false;

            for row in 0..nrows {
                let value = flat.get(row * num_cols + col_idx).unwrap_or(&Value::Null);
                match value {
                    Value::String(_) => return "TEXT".to_string(),
                    Value::Date(_) => return "TEXT".to_string(),
                    Value::Duration(_) => return "REAL".to_string(),
                    Value::Number(n) => {
                        if n.fract() == 0.0 {
                            has_integer = true;
                        } else {
                            has_float = true;
                        }
                    }
                    Value::Bool(_) => has_bool = true,
                    Value::Null => {}
                    _ => return "TEXT".to_string(),
                }
            }

            if has_float {
                "REAL".to_string()
            } else if has_integer || has_bool {
                "INTEGER".to_string()
            } else {
                "TEXT".to_string()
            }
        }
        TableData::View { .. } => "TEXT".to_string(),
    }
}

fn is_integer_column_for_header(table: &Table, header: &str) -> bool {
    let Some(col_idx) = table.headers().iter().position(|h| h == header) else {
        return false;
    };
    match &table.data {
        TableData::Owned {
            flat,
            num_cols,
            ..
        } => {
            if *num_cols == 0 {
                return false;
            }
            let nrows = flat.len() / num_cols;
            for row in 0..nrows {
                match flat.get(row * num_cols + col_idx).unwrap_or(&Value::Null) {
                    Value::Number(n) if n.fract() != 0.0 => return false,
                    Value::Null => {}
                    Value::Number(_) => {}
                    _ => return false,
                }
            }
            true
        }
        TableData::View { .. } => false,
    }
}

fn is_unique_column_for_header(table: &Table, header: &str) -> bool {
    let Some(col_idx) = table.headers().iter().position(|h| h == header) else {
        return false;
    };
    match &table.data {
        TableData::Owned {
            flat,
            num_cols,
            ..
        } => {
            if *num_cols == 0 {
                return true;
            }
            let nrows = flat.len() / num_cols;
            let mut seen = HashSet::new();
            for row in 0..nrows {
                let value = flat.get(row * num_cols + col_idx).unwrap_or(&Value::Null);
                if !matches!(value, Value::Null) && !seen.insert(value) {
                    return false;
                }
            }
            true
        }
        TableData::View { .. } => false,
    }
}

/// Создание таблицы с PK/FK и одна вставка данных (без промежуточного DROP).
fn create_and_fill_table(
    tx: &rusqlite::Transaction<'_>,
    table_info: &TableInfo,
    pk_by_table: &HashMap<String, &PrimaryKeyInfo>,
    fk_by_table: &HashMap<String, Vec<&ForeignKeyInfo>>,
) -> SqliteResult<()> {
    let table = table_info.table.borrow();
    if table.headers().is_empty() {
        return Ok(());
    }

    let headers: Vec<String> = table.headers().clone();
    let sanitized_headers: Vec<String> = headers.iter().map(|h| sanitize_column_name(h)).collect();
    let column_types: Vec<String> = headers
        .iter()
        .map(|h| infer_column_type_for_header(&table, h))
        .collect();

    let pk_column = pk_by_table.get(&table_info.sqlite_name);

    let mut columns_sql: Vec<String> = sanitized_headers
        .iter()
        .zip(column_types.iter())
        .map(|(san_name, sql_type)| {
            if let Some(pk) = pk_column {
                if pk.column_name == *san_name {
                    return format!("{} {} PRIMARY KEY", san_name, sql_type);
                }
            }
            format!("{} {}", san_name, sql_type)
        })
        .collect();

    if let Some(fks) = fk_by_table.get(&table_info.sqlite_name) {
        for fk in fks {
            columns_sql.push(format!(
                "FOREIGN KEY ({}) REFERENCES {}({})",
                fk.column_name, fk.referenced_table, fk.referenced_column
            ));
        }
    }

    let create_sql = format!(
        "CREATE TABLE {} ({})",
        table_info.sqlite_name,
        columns_sql.join(", ")
    );
    tx.execute(&create_sql, [])?;

    let Some(rr) = table.rows_ref() else {
        return Err(rusqlite::Error::InvalidParameterName(
            "SQLite export: таблица-представление (View) не поддерживается — материализуйте таблицу перед экспортом".into(),
        ));
    };

    if rr.is_empty() {
        return Ok(());
    }

    let placeholders = (0..headers.len()).map(|_| "?").collect::<Vec<_>>().join(", ");
    let insert_sql = format!(
        "INSERT INTO {} ({}) VALUES ({})",
        table_info.sqlite_name,
        sanitized_headers.join(", "),
        placeholders
    );

    let mut stmt = tx.prepare(&insert_sql)?;
    for row in rr.iter() {
        stmt.execute(rusqlite::params_from_iter(row.iter().map(ValueParam)))?;
    }

    Ok(())
}

/// Санитизация имени колонки для SQLite
fn sanitize_column_name(name: &str) -> String {
    let sanitized: String = name
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect();

    let upper = sanitized.to_uppercase();
    let is_keyword = matches!(
        upper.as_str(),
        "SELECT"
            | "FROM"
            | "WHERE"
            | "INSERT"
            | "UPDATE"
            | "DELETE"
            | "CREATE"
            | "DROP"
            | "TABLE"
            | "INDEX"
            | "PRIMARY"
            | "KEY"
            | "FOREIGN"
            | "REFERENCES"
            | "INTEGER"
            | "REAL"
            | "TEXT"
            | "BLOB"
            | "NULL"
            | "NOT"
            | "DEFAULT"
            | "UNIQUE"
            | "CHECK"
            | "AS"
            | "AND"
            | "OR"
            | "ORDER"
            | "BY"
            | "GROUP"
            | "HAVING"
            | "LIMIT"
            | "OFFSET"
            | "INNER"
            | "LEFT"
            | "RIGHT"
            | "JOIN"
            | "ON"
            | "UNION"
            | "ALL"
            | "DISTINCT"
            | "EXISTS"
            | "CASE"
            | "WHEN"
            | "THEN"
            | "ELSE"
            | "END"
            | "IS"
            | "LIKE"
            | "GLOB"
            | "REGEXP"
            | "MATCH"
            | "ESCAPE"
            | "CAST"
            | "COLLATE"
            | "ASC"
            | "DESC"
            | "INTO"
            | "VALUES"
            | "SET"
            | "BEGIN"
            | "COMMIT"
            | "ROLLBACK"
            | "TRANSACTION"
            | "SAVEPOINT"
            | "RELEASE"
            | "ATTACH"
            | "DETACH"
            | "DATABASE"
            | "ALTER"
            | "RENAME"
            | "TO"
            | "ADD"
            | "COLUMN"
            | "VACUUM"
            | "ANALYZE"
            | "EXPLAIN"
            | "PRAGMA"
            | "WITH"
            | "RECURSIVE"
            | "WINDOW"
            | "OVER"
            | "PARTITION"
            | "RANGE"
            | "ROWS"
            | "PRECEDING"
            | "FOLLOWING"
            | "CURRENT"
            | "ROW"
            | "UNBOUNDED"
            | "FILTER"
            | "EXCLUDE"
            | "GROUPS"
            | "TIES"
            | "NO"
            | "OTHERS"
    );

    if is_keyword {
        format!("\"{}\"", sanitized)
    } else {
        sanitized
    }
}

/// Получить явные первичные ключи из VM и преобразовать их в PrimaryKeyInfo
fn get_explicit_primary_keys(table_infos: &[TableInfo], vm: &Vm) -> Vec<PrimaryKeyInfo> {
    let explicit_primary_keys = vm.get_explicit_primary_keys();
    let mut primary_keys = Vec::new();

    let mut table_name_map: HashMap<&str, &TableInfo> = HashMap::new();
    for table_info in table_infos {
        table_name_map.insert(&table_info.name, table_info);
    }

    for explicit_pk in explicit_primary_keys {
        if let Some(table_info) = table_name_map.get(explicit_pk.table_name.as_str()) {
            primary_keys.push(PrimaryKeyInfo {
                table_name: table_info.sqlite_name.clone(),
                column_name: sanitize_column_name(&explicit_pk.column_name),
            });
        }
    }

    primary_keys
}

/// Определение первичных ключей
fn detect_primary_keys(table_infos: &[TableInfo], vm: &Vm) -> Result<Vec<PrimaryKeyInfo>, String> {
    let explicit_primary_keys = get_explicit_primary_keys(table_infos, vm);

    let mut tables_with_explicit_pk: HashSet<String> = HashSet::new();
    for pk in &explicit_primary_keys {
        tables_with_explicit_pk.insert(pk.table_name.clone());
    }

    let mut primary_keys = explicit_primary_keys;

    for table_info in table_infos {
        if tables_with_explicit_pk.contains(&table_info.sqlite_name) {
            continue;
        }
        let table = table_info.table.borrow();
        let headers: Vec<String> = table.headers().clone();
        for header in &headers {
            if header.to_lowercase() == "id" {
                if is_integer_column_for_header(&table, header) {
                    primary_keys.push(PrimaryKeyInfo {
                        table_name: table_info.sqlite_name.clone(),
                        column_name: sanitize_column_name(header),
                    });
                    break;
                }
            }

            if header.to_lowercase().ends_with("_id") && is_integer_column_for_header(&table, header)
            {
                if is_unique_column_for_header(&table, header) {
                    primary_keys.push(PrimaryKeyInfo {
                        table_name: table_info.sqlite_name.clone(),
                        column_name: sanitize_column_name(header),
                    });
                    break;
                }
            }

            let header_lower = header.to_lowercase();
            if header_lower.starts_with("pk_") || header_lower.starts_with("key_") {
                primary_keys.push(PrimaryKeyInfo {
                    table_name: table_info.sqlite_name.clone(),
                    column_name: sanitize_column_name(header),
                });
                break;
            }

            if is_unique_column_for_header(&table, header)
                && is_integer_column_for_header(&table, header)
            {
                primary_keys.push(PrimaryKeyInfo {
                    table_name: table_info.sqlite_name.clone(),
                    column_name: sanitize_column_name(header),
                });
                break;
            }
        }
    }

    Ok(primary_keys)
}

/// Получить явные связи из VM и преобразовать их в ForeignKeyInfo
fn get_explicit_foreign_keys(table_infos: &[TableInfo], vm: &Vm) -> Vec<ForeignKeyInfo> {
    let explicit_relations = vm.get_explicit_relations();
    let mut foreign_keys = Vec::new();

    let mut table_name_map: HashMap<&str, &TableInfo> = HashMap::new();
    for table_info in table_infos {
        table_name_map.insert(&table_info.name, table_info);
    }

    for relation in explicit_relations {
        if let (Some(source_table), Some(target_table)) = (
            table_name_map.get(relation.source_table_name.as_str()),
            table_name_map.get(relation.target_table_name.as_str()),
        ) {
            foreign_keys.push(ForeignKeyInfo {
                table_name: source_table.sqlite_name.clone(),
                column_name: sanitize_column_name(&relation.source_column_name),
                referenced_table: target_table.sqlite_name.clone(),
                referenced_column: sanitize_column_name(&relation.target_column_name),
            });
        }
    }

    foreign_keys
}

/// Определение внешних ключей
fn detect_foreign_keys(
    table_infos: &[TableInfo],
    primary_keys: &[PrimaryKeyInfo],
    explicit_foreign_keys: &[ForeignKeyInfo],
) -> Result<Vec<ForeignKeyInfo>, String> {
    let mut foreign_keys = explicit_foreign_keys.to_vec();

    let mut pk_index: HashMap<(String, String), &PrimaryKeyInfo> = HashMap::new();
    for pk in primary_keys {
        pk_index.insert((pk.table_name.clone(), pk.column_name.clone()), pk);
    }

    let mut explicit_fk_set: HashSet<(String, String)> = HashSet::new();
    for fk in explicit_foreign_keys {
        explicit_fk_set.insert((fk.table_name.clone(), fk.column_name.clone()));
    }

    for table_info in table_infos {
        let table = table_info.table.borrow();
        let headers: Vec<String> = table.headers().clone();
        for header in &headers {
            if !is_id_like_column(header) {
                continue;
            }

            if !is_integer_column_for_header(&table, header) {
                continue;
            }

            let col_sanitized = sanitize_column_name(header);
            if explicit_fk_set.contains(&(table_info.sqlite_name.clone(), col_sanitized.clone())) {
                continue;
            }

            if header.to_lowercase().ends_with("_id") {
                let base_name = header[..header.len() - 3].to_lowercase();

                for other_table_info in table_infos {
                    let other_table_name = other_table_info.sqlite_name.to_lowercase();
                    if other_table_name == base_name || other_table_name == format!("{}s", base_name)
                    {
                        if pk_index.contains_key(&(other_table_info.sqlite_name.clone(), "id".to_string()))
                        {
                            foreign_keys.push(ForeignKeyInfo {
                                table_name: table_info.sqlite_name.clone(),
                                column_name: col_sanitized.clone(),
                                referenced_table: other_table_info.sqlite_name.clone(),
                                referenced_column: "id".to_string(),
                            });
                            break;
                        }
                    }
                }
            }

            if header.to_lowercase() == "id" {
                continue;
            }
        }
    }

    Ok(foreign_keys)
}

/// Проверка, является ли колонка ID-подобной (строго: `id` или суффикс `_id`)
fn is_id_like_column(column_name: &str) -> bool {
    let lower = column_name.to_lowercase();
    lower == "id" || lower.ends_with("_id")
}

/// Топологическая сортировка таблиц по зависимостям FOREIGN KEY
fn topological_sort_tables(
    table_infos: &[TableInfo],
    foreign_keys: &[ForeignKeyInfo],
) -> Result<Vec<usize>, String> {
    let n = table_infos.len();

    let mut table_index_map: HashMap<String, usize> = HashMap::new();
    for (i, table_info) in table_infos.iter().enumerate() {
        table_index_map.insert(table_info.sqlite_name.clone(), i);
    }

    let mut dependencies: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut in_degree: Vec<usize> = vec![0; n];

    for fk in foreign_keys {
        if let (Some(&dependent_idx), Some(&referenced_idx)) = (
            table_index_map.get(&fk.table_name),
            table_index_map.get(&fk.referenced_table),
        ) {
            dependencies[referenced_idx].push(dependent_idx);
            in_degree[dependent_idx] += 1;
        }
    }

    let mut queue: Vec<usize> = Vec::new();
    let mut result: Vec<usize> = Vec::new();

    for (i, &degree) in in_degree.iter().enumerate() {
        if degree == 0 {
            queue.push(i);
        }
    }

    while let Some(current) = queue.pop() {
        result.push(current);

        for &dependent in &dependencies[current] {
            in_degree[dependent] -= 1;
            if in_degree[dependent] == 0 {
                queue.push(dependent);
            }
        }
    }

    if result.len() != n {
        let mut missing = Vec::new();
        for (i, &degree) in in_degree.iter().enumerate() {
            if degree > 0 {
                missing.push(table_infos[i].sqlite_name.clone());
            }
        }
        return Err(format!(
            "Обнаружены циклические зависимости или отсутствующие таблицы: {:?}",
            missing
        ));
    }

    Ok(result)
}

/// Индексы только для внешних ключей (PRIMARY KEY уже индексируется SQLite).
fn create_indexes_impl(
    conn: &rusqlite::Transaction,
    foreign_keys: &[ForeignKeyInfo],
) -> SqliteResult<()> {
    for fk in foreign_keys {
        let index_name = format!("idx_{}_{}", fk.table_name, fk.column_name);
        let sql = format!(
            "CREATE INDEX IF NOT EXISTS {} ON {} ({})",
            index_name, fk.table_name, fk.column_name
        );
        conn.execute(&sql, [])?;
    }

    Ok(())
}

/// Создание таблицы метаданных
fn create_metadata_table(
    conn: &Connection,
    vm: &mut crate::vm::vm::Vm,
    exported_tables: &HashMap<String, Rc<RefCell<Table>>>,
    extra_variables: &HashMap<String, Value>,
) -> SqliteResult<()> {
    conn.execute(
        "CREATE TABLE IF NOT EXISTS _datacode_variables (
            variable_name TEXT PRIMARY KEY,
            variable_type TEXT NOT NULL,
            table_name TEXT,
            row_count INTEGER,
            column_count INTEGER,
            created_at TEXT,
            description TEXT,
            value TEXT
        )",
        [],
    )?;

    let created_at = Utc::now().to_rfc3339();

    let mut stmt = conn.prepare(
        "INSERT OR REPLACE INTO _datacode_variables 
         (variable_name, variable_type, table_name, row_count, column_count, created_at, description, value)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
    )?;

    let mut inserted: HashSet<String> = HashSet::new();

    for (var_name, table_rc) in exported_tables {
        let table_ref = table_rc.borrow();
        let sqlite_name = sanitize_table_name(var_name);
        stmt.execute(params![
            var_name,
            "Table",
            sqlite_name,
            table_ref.len() as i64,
            table_ref.column_count() as i64,
            created_at,
            None::<String>,
            format!("Table[{}x{}]", table_ref.len(), table_ref.column_count())
        ])?;
        inserted.insert(var_name.clone());
    }

    let globals = vm.get_globals();
    let explicit_global_names = vm.get_explicit_global_names();
    let to_export: Vec<(usize, String)> = globals
        .iter()
        .enumerate()
        .filter_map(|(index, _)| {
            explicit_global_names
                .get(&index)
                .map(|name| (index, name.clone()))
        })
        .collect();

    for (index, var_name) in to_export {
        if inserted.contains(&var_name) {
            continue;
        }
        let value_id = vm.resolve_global_to_value_id(index);
        let value =
            crate::vm::store_convert::load_value(value_id, vm.value_store(), vm.heavy_store());
        if matches!(&value, Value::NativeFunction(_) | Value::Function(_)) {
            continue;
        }
        let var_type = get_value_type_name(&value);
        let (table_name, row_count, column_count, value_str) = if let Value::Table(table) = &value {
            let table_ref = table.borrow();
            let sqlite_name = sanitize_table_name(&var_name);
            (
                Some(sqlite_name),
                Some(table_ref.len() as i64),
                Some(table_ref.column_count() as i64),
                Some(format!(
                    "Table[{}x{}]",
                    table_ref.len(),
                    table_ref.column_count()
                )),
            )
        } else {
            (
                None,
                None,
                None,
                Some(value.to_string()),
            )
        };
        stmt.execute(params![
            var_name,
            var_type,
            table_name,
            row_count,
            column_count,
            created_at,
            None::<String>,
            value_str
        ])?;
        inserted.insert(var_name);
    }

    for (key, value) in extra_variables {
        if inserted.contains(key) {
            continue;
        }
        stmt.execute(params![
            key,
            get_value_type_name(value),
            None::<String>,
            None::<i64>,
            None::<i64>,
            created_at,
            None::<String>,
            value.to_string()
        ])?;
    }

    Ok(())
}

/// Получить имя типа значения
fn get_value_type_name(value: &Value) -> &str {
    match value {
        Value::Int(_) => "Int",
        Value::Float(_) => "Float",
        Value::Number(_) => "Number",
        Value::Bool(_) => "Bool",
        Value::String(_) => "String",
        Value::Date(_) => "Date",
        Value::Duration(_) => "Duration",
        Value::Array(_) | Value::ArrayView(_) | Value::ByteBuffer(_) | Value::ObjectFieldList { .. } => {
            "Array"
        }
        Value::Tuple(_) => "Tuple",
        Value::Table(_) => "Table",
        Value::Object(_) => "Object",
        Value::Set(_) => "Set",
        Value::Path(_) => "Path",
        Value::Uuid(_, _) => "UUID",
        Value::ColumnReference { .. } => "ColumnReference",
        Value::Function(_) | Value::ModuleFunction { .. } => "Function",
        Value::NativeFunction(_) => "NativeFunction",
        Value::PluginOpaque { .. } => "PluginOpaque",
        Value::Null => "Null",
        Value::Window(_) => "Window",
        Value::Image(_) => "Image",
        Value::Figure(_) => "Figure",
        Value::Axis(_) => "Axis",
        Value::DatabaseEngine(_) => "DatabaseEngine",
        Value::DatabaseCluster(_) => "DatabaseCluster",
        Value::Archive(_) => "Archive",
        Value::DataSource(_) => "DataSource",
        Value::DataSourceResponse(_) => "DataSourceResponse",
        Value::Enumerate { .. } => "Enumerate",
        Value::Iterable(_) => "Iterable",
        Value::Generator(_) => "Generator",
        Value::Ellipsis => "Ellipsis",
    }
}
