// Relations and primary keys native functions
//
// When VM_CALL_CONTEXT is set, natives push to VM-owned pending_relations/pending_primary_keys
// (no RefCell on hot path). Fallback to thread-local RELATIONS/PRIMARY_KEYS when VM context
// is not set (e.g. tests). See docs/gil_bottlenecks.md.

use crate::common::value::Value;
use crate::common::table::Table;
use crate::vm::vm::VM_CALL_CONTEXT;
use std::rc::Rc;
use std::cell::RefCell;

// Thread-local fallback when VM_CALL_CONTEXT is not set (e.g. standalone tests).
thread_local! {
    static RELATIONS: RefCell<Vec<(Rc<RefCell<Table>>, String, Rc<RefCell<Table>>, String)>> = RefCell::new(Vec::new());
}
thread_local! {
    static PRIMARY_KEYS: RefCell<Vec<(Rc<RefCell<Table>>, String)>> = RefCell::new(Vec::new());
}

/// Take relations from thread-local (fallback when executor uses VM-owned pending; kept for tests).
pub fn take_relations() -> Vec<(Rc<RefCell<Table>>, String, Rc<RefCell<Table>>, String)> {
    RELATIONS.with(|r| {
        let mut relations = r.borrow_mut();
        let result = relations.clone();
        relations.clear();
        result
    })
}

/// Take primary keys from thread-local (fallback; kept for tests).
pub fn take_primary_keys() -> Vec<(Rc<RefCell<Table>>, String)> {
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

    let col1_data = match col1.0.borrow_mut().get_column(col1.1) {
        Some(col) => col.clone(),
        None => return Value::Null,
    };
    let col2_data = match col2.0.borrow_mut().get_column(col2.1) {
        Some(col) => col.clone(),
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

    let relation = (
        col1.0.clone(),
        col1.1.clone(),
        col2.0.clone(),
        col2.1.clone(),
    );
    VM_CALL_CONTEXT.with(|ctx| {
        if let Some(ptr) = *ctx.borrow() {
            unsafe {
                (*ptr).pending_relations.push(relation);
            }
        } else {
            RELATIONS.with(|r| r.borrow_mut().push(relation));
        }
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

    if col.0.borrow_mut().get_column(col.1).is_none() {
        return Value::Null;
    }

    let pk_entry = (col.0.clone(), col.1.clone());
    VM_CALL_CONTEXT.with(|ctx| {
        if let Some(ptr) = *ctx.borrow() {
            unsafe {
                (*ptr).pending_primary_keys.push(pk_entry);
            }
        } else {
            PRIMARY_KEYS.with(|pk| pk.borrow_mut().push(pk_entry));
        }
    });

    Value::Null
}

