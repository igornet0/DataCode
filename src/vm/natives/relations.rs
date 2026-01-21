// Relations and primary keys native functions

use crate::common::value::Value;
use crate::common::table::Table;
use std::rc::Rc;
use std::cell::RefCell;

// Thread-local storage для хранения временных связей, созданных через relate()
thread_local! {
    static RELATIONS: RefCell<Vec<(*const RefCell<Table>, String, *const RefCell<Table>, String)>> = RefCell::new(Vec::new());
}

// Thread-local storage для хранения временных первичных ключей, созданных через primary_key()
thread_local! {
    static PRIMARY_KEYS: RefCell<Vec<(*const RefCell<Table>, String)>> = RefCell::new(Vec::new());
}

/// Получить все временные связи (для использования в VM)
pub fn take_relations() -> Vec<(*const RefCell<Table>, String, *const RefCell<Table>, String)> {
    RELATIONS.with(|r| {
        let mut relations = r.borrow_mut();
        let result = relations.clone();
        relations.clear();
        result
    })
}

/// Получить все временные первичные ключи (для использования в VM)
pub fn take_primary_keys() -> Vec<(*const RefCell<Table>, String)> {
    PRIMARY_KEYS.with(|pk| {
        let mut primary_keys = pk.borrow_mut();
        let result = primary_keys.clone();
        primary_keys.clear();
        result
    })
}

/// Нативная функция для создания связи между колонками таблиц
pub fn native_relate(args: &[Value]) -> Value {
    if args.len() != 2 {
        return Value::Null;
    }

    let col1 = match &args[0] {
        Value::ColumnReference { table, column_name } => (table, column_name),
        _ => return Value::Null,
    };

    let col2 = match &args[1] {
        Value::ColumnReference { table, column_name } => (table, column_name),
        _ => return Value::Null,
    };

    // Проверяем совместимость типов колонок (оба должны быть одинакового типа)
    let table1_ref = col1.0.borrow();
    let table2_ref = col2.0.borrow();
    
    let col1_data = match table1_ref.get_column(col1.1) {
        Some(col) => col,
        None => return Value::Null,
    };
    
    let col2_data = match table2_ref.get_column(col2.1) {
        Some(col) => col,
        None => return Value::Null,
    };

    // Проверяем совместимость типов (простая проверка - оба должны иметь значения одного типа)
    // Это базовая проверка, более точная проверка будет сделана при экспорте
    if !col1_data.is_empty() && !col2_data.is_empty() {
        // Проверяем первый не-null элемент каждой колонки
        let type1 = &col1_data[0];
        let type2 = &col2_data[0];
        
        // Проверяем совместимость типов (Number <-> Number, String <-> String)
        match (type1, type2) {
            (Value::Number(_), Value::Number(_)) => {},
            (Value::String(_), Value::String(_)) => {},
            _ => {
                // Типы не совместимы, но все равно сохраняем связь
                // Более строгая проверка будет при экспорте
            }
        }
    }

    // Сохраняем связь в thread-local storage
    RELATIONS.with(|r| {
        let mut relations = r.borrow_mut();
        relations.push((
            Rc::as_ptr(col1.0),
            col1.1.clone(),
            Rc::as_ptr(col2.0),
            col2.1.clone(),
        ));
    });

    Value::Null
}

/// Нативная функция для указания первичного ключа таблицы
pub fn native_primary_key(args: &[Value]) -> Value {
    if args.len() != 1 {
        return Value::Null;
    }

    let col = match &args[0] {
        Value::ColumnReference { table, column_name } => (table, column_name),
        _ => return Value::Null,
    };

    // Проверяем, что колонка существует
    let table_ref = col.0.borrow();
    if table_ref.get_column(col.1).is_none() {
        return Value::Null;
    }

    // Сохраняем первичный ключ в thread-local storage
    PRIMARY_KEYS.with(|pk| {
        let mut primary_keys = pk.borrow_mut();
        primary_keys.push((
            Rc::as_ptr(col.0),
            col.1.clone(),
        ));
    });

    Value::Null
}

