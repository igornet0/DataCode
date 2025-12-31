// Модуль для экспорта таблиц DataCode в SQLite

use crate::common::{value::Value, table::Table};
use crate::vm::Vm;
use rusqlite::{Connection, params, Result as SqliteResult};
use std::rc::Rc;
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use chrono::Utc;

/// Структура для хранения информации о таблице для экспорта
struct TableInfo {
    #[allow(dead_code)]
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
    #[allow(dead_code)]
    referenced_table: String,
    #[allow(dead_code)]
    referenced_column: String,
}

/// Главная функция экспорта в SQLite
pub fn export_to_sqlite(vm: &Vm, output_path: &str) -> Result<(), String> {
    // Получаем все таблицы из глобальных переменных
    let tables = get_global_tables(vm)?;
    
    if tables.is_empty() {
        return Err("Нет таблиц для экспорта".to_string());
    }

    // Создаем базу данных
    let mut conn = Connection::open(output_path)
        .map_err(|e| format!("Ошибка создания базы данных: {}", e))?;
    
    // Включаем поддержку FOREIGN KEY constraints
    conn.execute("PRAGMA foreign_keys = ON", [])
        .map_err(|e| format!("Ошибка включения FOREIGN KEY: {}", e))?;

    // Экспортируем каждую таблицу (сначала без FOREIGN KEY)
    let mut table_infos = Vec::new();
    for (var_name, table) in &tables {
        let sqlite_name = sanitize_table_name(var_name);
        export_table(&conn, var_name, &sqlite_name, table)
            .map_err(|e| format!("Ошибка экспорта таблицы {}: {}", var_name, e))?;
        table_infos.push(TableInfo {
            name: var_name.clone(),
            table: table.clone(),
            sqlite_name,
        });
    }

    // Определяем первичные ключи (явные имеют приоритет)
    let primary_keys = detect_primary_keys(&table_infos, vm)?;

    // Получаем явные связи из VM
    let explicit_foreign_keys = get_explicit_foreign_keys(&table_infos, vm);

    // Определяем внешние ключи (явные связи имеют приоритет, затем автоматическое определение)
    let foreign_keys = detect_foreign_keys(&table_infos, &primary_keys, &explicit_foreign_keys)?;

    // Пересоздаем таблицы с FOREIGN KEY constraints
    if !foreign_keys.is_empty() {
        recreate_tables_with_foreign_keys(&mut conn, &table_infos, &primary_keys, &foreign_keys)
            .map_err(|e| format!("Ошибка пересоздания таблиц с FOREIGN KEY: {}", e))?;
    } else {
        // Если нет foreign keys, просто создаем индексы
        create_indexes(&conn, &primary_keys, &foreign_keys)
            .map_err(|e| format!("Ошибка создания индексов: {}", e))?;
    }

    // Создаем таблицу метаданных
    create_metadata_table(&conn, vm, &tables)
        .map_err(|e| format!("Ошибка создания таблицы метаданных: {}", e))?;

    println!("✅ Экспорт завершен: {}", output_path);
    Ok(())
}

/// Получить все таблицы из глобальных переменных VM
fn get_global_tables(vm: &Vm) -> Result<HashMap<String, Rc<RefCell<Table>>>, String> {
    let mut tables = HashMap::new();
    let globals = vm.get_globals();
    let explicit_global_names = vm.get_explicit_global_names();

    // Итерируемся только по переменным, явно объявленным с ключевым словом 'global'
    for (index, value) in globals.iter().enumerate() {
        if let Some(_var_name) = explicit_global_names.get(&index) {
            // Пропускаем нативные функции
            if matches!(value, Value::NativeFunction(_) | Value::Function(_)) {
                continue;
            }

            // Проверяем, является ли значение таблицей
            if let Value::Table(table) = value {
                if let Some(var_name) = explicit_global_names.get(&index) {
                    tables.insert(var_name.clone(), table.clone());
                }
            }
        }
    }

    Ok(tables)
}

/// Санитизация имени таблицы для SQLite
fn sanitize_table_name(name: &str) -> String {
    // Заменяем недопустимые символы на подчеркивания
    name.chars()
        .map(|c| if c.is_alphanumeric() || c == '_' { c } else { '_' })
        .collect()
}

/// Экспорт одной таблицы в SQLite
fn export_table(
    conn: &Connection,
    _var_name: &str,
    sqlite_name: &str,
    table: &Rc<RefCell<Table>>,
) -> SqliteResult<()> {
    let table_ref = table.borrow();
    
    if table_ref.headers.is_empty() {
        return Ok(()); // Пустая таблица, пропускаем
    }

    // Определяем типы колонок
    let column_types: Vec<String> = table_ref.headers.iter()
        .map(|header| {
            let column = table_ref.columns.get(header).unwrap();
            infer_column_type(column)
        })
        .collect();

    // Создаем SQL для создания таблицы
    let columns_sql: Vec<String> = table_ref.headers.iter()
        .zip(column_types.iter())
        .map(|(header, sql_type)| {
            format!("{} {}", sanitize_column_name(header), sql_type)
        })
        .collect();

    let create_sql = format!(
        "CREATE TABLE IF NOT EXISTS {} ({})",
        sqlite_name,
        columns_sql.join(", ")
    );

    conn.execute(&create_sql, [])?;

    // Вставляем данные
    if !table_ref.rows.is_empty() {
        let placeholders: Vec<String> = (0..table_ref.headers.len())
            .map(|_| "?".to_string())
            .collect();
        let insert_sql = format!(
            "INSERT INTO {} ({}) VALUES ({})",
            sqlite_name,
            table_ref.headers.iter()
                .map(|h| sanitize_column_name(h))
                .collect::<Vec<_>>()
                .join(", "),
            placeholders.join(", ")
        );

        let mut stmt = conn.prepare(&insert_sql)?;
        
        for row in &table_ref.rows {
            // Преобразуем значения в параметры SQLite
            let mut params_vec: Vec<Box<dyn rusqlite::ToSql>> = Vec::new();
            for value in row {
                params_vec.push(match value {
                    Value::Number(n) => {
                        if n.fract() == 0.0 {
                            Box::new(*n as i64) as Box<dyn rusqlite::ToSql>
                        } else {
                            Box::new(*n) as Box<dyn rusqlite::ToSql>
                        }
                    }
                    Value::Bool(b) => {
                        Box::new(if *b { 1i64 } else { 0i64 }) as Box<dyn rusqlite::ToSql>
                    }
                    Value::String(s) => {
                        Box::new(s.clone()) as Box<dyn rusqlite::ToSql>
                    }
                    Value::Null => {
                        Box::new(Option::<String>::None) as Box<dyn rusqlite::ToSql>
                    }
                    _ => {
                        let s = value.to_string();
                        Box::new(s) as Box<dyn rusqlite::ToSql>
                    }
                });
            }
            
            // Преобразуем в срез параметров
            let params: Vec<&dyn rusqlite::ToSql> = params_vec.iter()
                .map(|v| v.as_ref())
                .collect();
            
            stmt.execute(params.as_slice())?;
        }
    }

    Ok(())
}

/// Определение типа колонки на основе данных
fn infer_column_type(column: &[Value]) -> String {
    if column.is_empty() {
        return "TEXT".to_string();
    }

    let mut has_integer = false;
    let mut has_float = false;
    let mut has_string = false;
    let mut has_bool = false;

    for value in column {
        match value {
            Value::Number(n) => {
                if n.fract() == 0.0 {
                    has_integer = true;
                } else {
                    has_float = true;
                }
            }
            Value::String(_) => has_string = true,
            Value::Bool(_) => has_bool = true,
            Value::Null => {}, // NULL значения не влияют на тип
            _ => has_string = true, // Остальные типы как текст
        }
    }

    // Определяем тип по приоритету
    if has_float {
        "REAL".to_string()
    } else if has_integer && !has_string {
        "INTEGER".to_string()
    } else if has_bool && !has_string && !has_integer {
        "INTEGER".to_string() // Bool как INTEGER (0/1)
    } else {
        "TEXT".to_string()
    }
}

/// Санитизация имени колонки
fn sanitize_column_name(name: &str) -> String {
    // Заменяем недопустимые символы
    let sanitized: String = name.chars()
        .map(|c| if c.is_alphanumeric() || c == '_' { c } else { '_' })
        .collect();
    
    // Проверяем на ключевые слова SQLite (основные)
    let upper = sanitized.to_uppercase();
    let is_keyword = matches!(upper.as_str(),
        "SELECT" | "FROM" | "WHERE" | "INSERT" | "UPDATE" | "DELETE" | 
        "CREATE" | "DROP" | "TABLE" | "INDEX" | "PRIMARY" | "KEY" |
        "FOREIGN" | "REFERENCES" | "INTEGER" | "REAL" | "TEXT" | "BLOB" |
        "NULL" | "NOT" | "DEFAULT" | "UNIQUE" | "CHECK" | "AS" | "AND" |
        "OR" | "ORDER" | "BY" | "GROUP" | "HAVING" | "LIMIT" | "OFFSET" |
        "INNER" | "LEFT" | "RIGHT" | "JOIN" | "ON" | "UNION" | "ALL" |
        "DISTINCT" | "EXISTS" | "CASE" | "WHEN" | "THEN" | "ELSE" | "END" |
        "IS" | "LIKE" | "GLOB" | "REGEXP" | "MATCH" | "ESCAPE" | "CAST" |
        "COLLATE" | "ASC" | "DESC" | "INTO" | "VALUES" | "SET" | "BEGIN" |
        "COMMIT" | "ROLLBACK" | "TRANSACTION" | "SAVEPOINT" | "RELEASE" |
        "ATTACH" | "DETACH" | "DATABASE" | "ALTER" | "RENAME" | "TO" |
        "ADD" | "COLUMN" | "VACUUM" | "ANALYZE" | "EXPLAIN" | "PRAGMA" |
        "WITH" | "RECURSIVE" | "WINDOW" | "OVER" | "PARTITION" | "RANGE" |
        "ROWS" | "PRECEDING" | "FOLLOWING" | "CURRENT" | "ROW" | "UNBOUNDED" |
        "FILTER" | "EXCLUDE" | "GROUPS" | "TIES" | "NO" | "OTHERS"
    );
    
    if is_keyword {
        format!("\"{}\"", sanitized)
    } else {
        sanitized
    }
}


/// Получить явные первичные ключи из VM и преобразовать их в PrimaryKeyInfo
fn get_explicit_primary_keys(
    table_infos: &[TableInfo],
    vm: &Vm,
) -> Vec<PrimaryKeyInfo> {
    let explicit_primary_keys = vm.get_explicit_primary_keys();
    let mut primary_keys = Vec::new();

    // Создаем индекс для быстрого поиска таблиц по имени переменной
    let mut table_name_map: HashMap<&str, &TableInfo> = HashMap::new();
    for table_info in table_infos {
        table_name_map.insert(&table_info.name, table_info);
    }

    for explicit_pk in explicit_primary_keys {
        // Ищем таблицу по имени переменной
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
    // Сначала получаем явные первичные ключи из VM
    let explicit_primary_keys = get_explicit_primary_keys(table_infos, vm);
    
    // Создаем множество таблиц, для которых уже указан явный первичный ключ
    let mut tables_with_explicit_pk: HashSet<String> = HashSet::new();
    for pk in &explicit_primary_keys {
        tables_with_explicit_pk.insert(pk.table_name.clone());
    }
    
    let mut primary_keys = explicit_primary_keys;

    // Для таблиц без явного первичного ключа применяем автоматическое определение
    for table_info in table_infos {
        // Пропускаем таблицы, для которых уже указан явный первичный ключ
        if tables_with_explicit_pk.contains(&table_info.sqlite_name) {
            continue;
        }
        
        let table = table_info.table.borrow();
        
        // Проверяем каждую колонку
        for header in &table.headers {
            let column = table.columns.get(header).ok_or("Колонка не найдена")?;
            
            // Правило 1: Колонка с именем "id" типа Integer
            if header.to_lowercase() == "id" {
                if is_integer_column(column) {
                    primary_keys.push(PrimaryKeyInfo {
                        table_name: table_info.sqlite_name.clone(),
                        column_name: sanitize_column_name(header),
                    });
                    break; // Нашли первичный ключ для этой таблицы, переходим к следующей
                }
            }

            // Правило 2: Колонки с именами "*_id" типа Integer
            if header.to_lowercase().ends_with("_id") && is_integer_column(column) {
                // Проверяем, что это не внешний ключ (проверяем уникальность)
                if is_unique_column(column) {
                    primary_keys.push(PrimaryKeyInfo {
                        table_name: table_info.sqlite_name.clone(),
                        column_name: sanitize_column_name(header),
                    });
                    break; // Нашли первичный ключ для этой таблицы, переходим к следующей
                }
            }

            // Правило 3: Колонки с префиксом "pk_" или "key_"
            let header_lower = header.to_lowercase();
            if header_lower.starts_with("pk_") || header_lower.starts_with("key_") {
                primary_keys.push(PrimaryKeyInfo {
                    table_name: table_info.sqlite_name.clone(),
                    column_name: sanitize_column_name(header),
                });
                break; // Нашли первичный ключ для этой таблицы, переходим к следующей
            }

            // Правило 4: Колонки, где все значения уникальны
            if is_unique_column(column) && is_integer_column(column) {
                primary_keys.push(PrimaryKeyInfo {
                    table_name: table_info.sqlite_name.clone(),
                    column_name: sanitize_column_name(header),
                });
                break; // Нашли первичный ключ для этой таблицы, переходим к следующей
            }
        }
    }

    Ok(primary_keys)
}

/// Получить явные связи из VM и преобразовать их в ForeignKeyInfo
fn get_explicit_foreign_keys(
    table_infos: &[TableInfo],
    vm: &Vm,
) -> Vec<ForeignKeyInfo> {
    let explicit_relations = vm.get_explicit_relations();
    let mut foreign_keys = Vec::new();

    // Создаем индекс для быстрого поиска таблиц по имени переменной
    let mut table_name_map: HashMap<&str, &TableInfo> = HashMap::new();
    for table_info in table_infos {
        table_name_map.insert(&table_info.name, table_info);
    }

    for relation in explicit_relations {
        // Ищем исходную и целевую таблицы
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

    // Создаем индекс первичных ключей по таблице и колонке
    let mut pk_index: HashMap<(String, String), &PrimaryKeyInfo> = HashMap::new();
    for pk in primary_keys {
        pk_index.insert((pk.table_name.clone(), pk.column_name.clone()), pk);
    }

    // Создаем индекс явных foreign keys, чтобы не дублировать их
    let mut explicit_fk_set: HashSet<(String, String)> = HashSet::new();
    for fk in explicit_foreign_keys {
        explicit_fk_set.insert((fk.table_name.clone(), fk.column_name.clone()));
    }

    for table_info in table_infos {
        let table = table_info.table.borrow();
        
        for header in &table.headers {
            let column = table.columns.get(header).ok_or("Колонка не найдена")?;
            
            // Проверяем, является ли колонка ID-подобной
            if !is_id_like_column(header) {
                continue;
            }

            // Проверяем, что это Integer колонка
            if !is_integer_column(column) {
                continue;
            }

            // Пропускаем, если эта связь уже определена явно
            if explicit_fk_set.contains(&(table_info.sqlite_name.clone(), sanitize_column_name(header).clone())) {
                continue;
            }

            // Ищем соответствующую таблицу с первичным ключом
            // Вариант 1: Имя колонки заканчивается на "_id", ищем таблицу без суффикса
            if header.to_lowercase().ends_with("_id") {
                let base_name = header[..header.len() - 3].to_lowercase();
                
                // Ищем таблицу, имя которой совпадает с base_name
                for other_table_info in table_infos {
                    let other_table_name = other_table_info.sqlite_name.to_lowercase();
                    if other_table_name == base_name || other_table_name == format!("{}s", base_name) {
                        // Проверяем, есть ли в этой таблице первичный ключ "id"
                        if pk_index.contains_key(&(other_table_info.sqlite_name.clone(), "id".to_string())) {
                            foreign_keys.push(ForeignKeyInfo {
                                table_name: table_info.sqlite_name.clone(),
                                column_name: sanitize_column_name(header),
                                referenced_table: other_table_info.sqlite_name.clone(),
                                referenced_column: "id".to_string(),
                            });
                            break;
                        }
                    }
                }
            }

            // Вариант 2: Колонка называется "id", ищем таблицы с первичным ключом с таким же именем
            if header.to_lowercase() == "id" {
                // Это обычно первичный ключ, не внешний
                continue;
            }
        }
    }

    Ok(foreign_keys)
}

/// Проверка, является ли колонка ID-подобной
fn is_id_like_column(column_name: &str) -> bool {
    let lower = column_name.to_lowercase();
    lower.ends_with("_id") || 
    lower == "id" || 
    lower.ends_with("id") ||
    lower.ends_with("_id") ||
    lower.contains("id")
}

/// Проверка, является ли колонка целочисленной
fn is_integer_column(column: &[Value]) -> bool {
    if column.is_empty() {
        return false;
    }

    for value in column {
        match value {
            Value::Number(n) => {
                if n.fract() != 0.0 {
                    return false;
                }
            }
            Value::Null => continue, // NULL значения пропускаем
            _ => return false,
        }
    }

    true
}

/// Проверка уникальности колонки
fn is_unique_column(column: &[Value]) -> bool {
    use std::collections::HashSet;
    let mut seen = HashSet::new();
    
    for value in column {
        if !matches!(value, Value::Null) {
            if !seen.insert(value) {
                return false;
            }
        }
    }
    
    true
}

/// Топологическая сортировка таблиц по зависимостям FOREIGN KEY
/// Таблицы, на которые ссылаются другие, должны создаваться первыми
fn topological_sort_tables(
    table_infos: &[TableInfo],
    foreign_keys: &[ForeignKeyInfo],
) -> Result<Vec<usize>, String> {
    let n = table_infos.len();
    
    // Создаем индекс таблиц по sqlite_name
    let mut table_index_map: HashMap<String, usize> = HashMap::new();
    for (i, table_info) in table_infos.iter().enumerate() {
        table_index_map.insert(table_info.sqlite_name.clone(), i);
    }
    
    // Строим граф зависимостей: для каждой таблицы список таблиц, от которых она зависит
    let mut dependencies: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut in_degree: Vec<usize> = vec![0; n];
    
    for fk in foreign_keys {
        // fk.table_name зависит от fk.referenced_table
        // Это означает, что fk.table_name должна быть создана ПОСЛЕ fk.referenced_table
        if let (Some(&dependent_idx), Some(&referenced_idx)) = (
            table_index_map.get(&fk.table_name),
            table_index_map.get(&fk.referenced_table),
        ) {
            // Таблица dependent_idx зависит от referenced_idx
            // referenced_idx должна быть создана перед dependent_idx
            // Поэтому referenced_idx "влияет" на dependent_idx
            dependencies[referenced_idx].push(dependent_idx);
            in_degree[dependent_idx] += 1;
        }
    }
    
    // Алгоритм Кана (Kahn's algorithm) для топологической сортировки
    let mut queue: Vec<usize> = Vec::new();
    let mut result: Vec<usize> = Vec::new();
    
    // Находим все таблицы без зависимостей (in_degree == 0)
    for (i, &degree) in in_degree.iter().enumerate() {
        if degree == 0 {
            queue.push(i);
        }
    }
    
    // Обрабатываем таблицы без зависимостей
    while let Some(current) = queue.pop() {
        result.push(current);
        
        // Уменьшаем in_degree для всех таблиц, которые зависят от current
        for &dependent in &dependencies[current] {
            in_degree[dependent] -= 1;
            if in_degree[dependent] == 0 {
                queue.push(dependent);
            }
        }
    }
    
    // Проверяем, все ли таблицы обработаны
    if result.len() != n {
        // Есть циклические зависимости или таблицы, на которые ссылаются, но которых нет
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

/// Пересоздание таблиц с FOREIGN KEY constraints
fn recreate_tables_with_foreign_keys(
    conn: &mut Connection,
    table_infos: &[TableInfo],
    primary_keys: &[PrimaryKeyInfo],
    foreign_keys: &[ForeignKeyInfo],
) -> SqliteResult<()> {

    // Создаем индекс foreign keys по имени таблицы
    let mut fk_by_table: HashMap<String, Vec<&ForeignKeyInfo>> = HashMap::new();
    for fk in foreign_keys {
        fk_by_table.entry(fk.table_name.clone()).or_insert_with(Vec::new).push(fk);
    }

    // Создаем индекс primary keys по имени таблицы
    let mut pk_by_table: HashMap<String, &PrimaryKeyInfo> = HashMap::new();
    for pk in primary_keys {
        pk_by_table.insert(pk.table_name.clone(), pk);
    }

    // Начинаем транзакцию
    let tx = conn.transaction()?;

    // Сохраняем данные всех таблиц
    let mut saved_data: HashMap<String, Vec<Vec<Value>>> = HashMap::new();
    for table_info in table_infos {
        let table = table_info.table.borrow();
        saved_data.insert(table_info.sqlite_name.clone(), table.rows.clone());
    }

    // Удаляем все таблицы
    for table_info in table_infos {
        tx.execute(&format!("DROP TABLE IF EXISTS {}", table_info.sqlite_name), [])?;
    }

    // Сортируем таблицы в топологическом порядке (таблицы без зависимостей создаются первыми)
    let sorted_indices = topological_sort_tables(table_infos, foreign_keys)
        .map_err(|e| rusqlite::Error::SqliteFailure(
            rusqlite::ffi::Error::new(rusqlite::ffi::SQLITE_CONSTRAINT),
            Some(e)
        ))?;

    // Пересоздаем таблицы с FOREIGN KEY constraints в правильном порядке
    for &table_idx in &sorted_indices {
        let table_info = &table_infos[table_idx];
        let table = table_info.table.borrow();
        
        if table.headers.is_empty() {
            continue; // Пустая таблица, пропускаем
        }

        // Определяем типы колонок
        let column_types: Vec<String> = table.headers.iter()
            .map(|header| {
                let column = table.columns.get(header).unwrap();
                infer_column_type(column)
            })
            .collect();

        // Определяем PRIMARY KEY для этой таблицы
        let pk_column = pk_by_table.get(&table_info.sqlite_name);

        // Создаем SQL для создания таблицы с FOREIGN KEY constraints
        let mut columns_sql: Vec<String> = table.headers.iter()
            .zip(column_types.iter())
            .map(|(header, sql_type)| {
                let col_name = sanitize_column_name(header);
                // Если это primary key, добавляем PRIMARY KEY
                if let Some(pk) = pk_column {
                    if pk.column_name == col_name {
                        format!("{} {} PRIMARY KEY", col_name, sql_type)
                    } else {
                        format!("{} {}", col_name, sql_type)
                    }
                } else {
                    format!("{} {}", col_name, sql_type)
                }
            })
            .collect();

        // Добавляем FOREIGN KEY constraints
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

        // Вставляем данные обратно
        if let Some(rows) = saved_data.get(&table_info.sqlite_name) {
            if !rows.is_empty() {
                let placeholders: Vec<String> = (0..table.headers.len())
                    .map(|_| "?".to_string())
                    .collect();
                let insert_sql = format!(
                    "INSERT INTO {} ({}) VALUES ({})",
                    table_info.sqlite_name,
                    table.headers.iter()
                        .map(|h| sanitize_column_name(h))
                        .collect::<Vec<_>>()
                        .join(", "),
                    placeholders.join(", ")
                );

                let mut stmt = tx.prepare(&insert_sql)?;
                
                for row in rows {
                    let mut params_vec: Vec<Box<dyn rusqlite::ToSql>> = Vec::new();
                    for value in row {
                        params_vec.push(match value {
                            Value::Number(n) => {
                                if n.fract() == 0.0 {
                                    Box::new(*n as i64) as Box<dyn rusqlite::ToSql>
                                } else {
                                    Box::new(*n) as Box<dyn rusqlite::ToSql>
                                }
                            }
                            Value::Bool(b) => {
                                Box::new(if *b { 1i64 } else { 0i64 }) as Box<dyn rusqlite::ToSql>
                            }
                            Value::String(s) => {
                                Box::new(s.clone()) as Box<dyn rusqlite::ToSql>
                            }
                            Value::Null => {
                                Box::new(Option::<String>::None) as Box<dyn rusqlite::ToSql>
                            }
                            _ => {
                                let s = value.to_string();
                                Box::new(s) as Box<dyn rusqlite::ToSql>
                            }
                        });
                    }
                    
                    let params: Vec<&dyn rusqlite::ToSql> = params_vec.iter()
                        .map(|v| v.as_ref())
                        .collect();
                    
                    stmt.execute(params.as_slice())?;
                }
            }
        }
    }

    // Создаем индексы для первичных и внешних ключей
    create_indexes_impl(&tx, primary_keys, foreign_keys)?;

    // Коммитим транзакцию
    tx.commit()?;

    Ok(())
}

/// Создание индексов для первичных и внешних ключей (внутренняя реализация)
fn create_indexes_impl(
    conn: &rusqlite::Transaction,
    primary_keys: &[PrimaryKeyInfo],
    foreign_keys: &[ForeignKeyInfo],
) -> SqliteResult<()> {
    // Создаем индексы для первичных ключей (если они еще не созданы как PRIMARY KEY)
    for pk in primary_keys {
        let index_name = format!("idx_{}_{}", pk.table_name, pk.column_name);
        let sql = format!(
            "CREATE INDEX IF NOT EXISTS {} ON {} ({})",
            index_name, pk.table_name, pk.column_name
        );
        conn.execute(&sql, [])?;
    }

    // Создаем индексы для внешних ключей
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

/// Создание индексов для первичных и внешних ключей
fn create_indexes(
    conn: &Connection,
    primary_keys: &[PrimaryKeyInfo],
    foreign_keys: &[ForeignKeyInfo],
) -> SqliteResult<()> {
    // Создаем индексы для первичных ключей
    for pk in primary_keys {
        let index_name = format!("idx_{}_{}", pk.table_name, pk.column_name);
        let sql = format!(
            "CREATE INDEX IF NOT EXISTS {} ON {} ({})",
            index_name, pk.table_name, pk.column_name
        );
        conn.execute(&sql, [])?;
    }

    // Создаем индексы для внешних ключей
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
    vm: &Vm,
    _tables: &HashMap<String, Rc<RefCell<Table>>>,
) -> SqliteResult<()> {
    // Создаем таблицу метаданных
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

    let globals = vm.get_globals();
    let explicit_global_names = vm.get_explicit_global_names();
    let created_at = Utc::now().to_rfc3339();

    let mut stmt = conn.prepare(
        "INSERT OR REPLACE INTO _datacode_variables 
         (variable_name, variable_type, table_name, row_count, column_count, created_at, description, value)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)"
    )?;

    // Добавляем информацию только о переменных, явно объявленных с ключевым словом 'global'
    for (index, value) in globals.iter().enumerate() {
        if let Some(var_name) = explicit_global_names.get(&index) {
            // Пропускаем нативные функции
            if matches!(value, Value::NativeFunction(_) | Value::Function(_)) {
                continue;
            }

            let var_type = get_value_type_name(value);
            let (table_name, row_count, column_count) = if let Value::Table(table) = value {
                let table_ref = table.borrow();
                let sqlite_name = sanitize_table_name(var_name);
                (Some(sqlite_name), Some(table_ref.len() as i64), Some(table_ref.column_count() as i64))
            } else {
                (None, None, None)
            };

            let value_str = value.to_string();
            let description = None::<String>; // Опциональное описание

            stmt.execute(params![
                var_name,
                var_type,
                table_name,
                row_count,
                column_count,
                created_at,
                description,
                value_str
            ])?;
        }
    }

    Ok(())
}

/// Получить имя типа значения
fn get_value_type_name(value: &Value) -> &str {
    match value {
        Value::Number(_) => "Number",
        Value::Bool(_) => "Bool",
        Value::String(_) => "String",
        Value::Array(_) => "Array",
        Value::Tuple(_) => "Tuple",
        Value::Table(_) => "Table",
        Value::Object(_) => "Object",
        Value::Path(_) => "Path",
        Value::ColumnReference { .. } => "ColumnReference",
        Value::Function(_) => "Function",
        Value::NativeFunction(_) => "NativeFunction",
        Value::Tensor(_) => "Tensor",
        Value::Graph(_) => "Graph",
        Value::LinearRegression(_) => "LinearRegression",
        Value::SGD(_) => "SGD",
        Value::Momentum(_) => "Momentum",
        Value::NAG(_) => "NAG",
        Value::Adagrad(_) => "Adagrad",
        Value::RMSprop(_) => "RMSprop",
        Value::Adam(_) => "Adam",
        Value::AdamW(_) => "AdamW",
        Value::Dataset(_) => "Dataset",
        Value::Null => "Null",
        Value::NeuralNetwork(_) => "NeuralNetwork",
        Value::Sequential(_) => "Sequential",
        Value::Layer(_) => "Layer",
        Value::Window(_) => "Window",
        Value::Image(_) => "Image",
        Value::Figure(_) => "Figure",
        Value::Axis(_) => "Axis",
    }
}

