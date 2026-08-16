// Relations and primary keys native functions
//
// When VM_CALL_CONTEXT is set, natives push to VM-owned pending_relations/pending_primary_keys
// (no RefCell on hot path). Fallback to thread-local RELATIONS/PRIMARY_KEYS when VM context
// is not set (e.g. tests). See docs/gil_bottlenecks.md.

use crate::common::table::Table;
use crate::common::value::Value;
use crate::vm::calls::get_type_name_value;
use crate::vm::vm::current_vm_ptr;
use std::cell::RefCell;
use std::rc::Rc;

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

type ColRef = (Rc<RefCell<Table>>, String);

fn as_column_ref(v: &Value) -> Result<ColRef, String> {
    match v {
        Value::ColumnReference { table, column_name } => {
            if !table.borrow().has_column(column_name) {
                return Err(format!(
                    "TypeError: relate() column '{}' not found in table",
                    column_name
                ));
            }
            Ok((Rc::clone(table), column_name.clone()))
        }
        other => Err(format!(
            "TypeError: relate() expected a column reference, got {}",
            get_type_name_value(other)
        )),
    }
}

/// Collect columns from `relate(col, ...)` or `relate([col, ...])`.
fn collect_relate_columns(args: &[Value]) -> Result<Vec<ColRef>, String> {
    if args.is_empty() {
        return Err("TypeError: relate() expected at least 2 column references".to_string());
    }

    // Single array argument: relate([col1, col2, ...])
    if args.len() == 1 {
        match &args[0] {
            Value::Array(a) => {
                let items = a.borrow();
                if items.len() < 2 {
                    return Err(
                        "TypeError: relate() expected at least 2 column references".to_string(),
                    );
                }
                let mut cols = Vec::with_capacity(items.len());
                for item in items.iter() {
                    cols.push(as_column_ref(item)?);
                }
                return Ok(cols);
            }
            _ => {
                // One non-array arg — not enough for a relation.
                let _ = as_column_ref(&args[0])?;
                return Err(
                    "TypeError: relate() expected at least 2 column references".to_string(),
                );
            }
        }
    }

    // Varargs: relate(col1, col2, ...) — reject mixing array with extra args.
    if args.iter().any(|a| matches!(a, Value::Array(_))) {
        return Err(
            "TypeError: relate() expects either column references or a single array of columns, not both"
                .to_string(),
        );
    }

    let mut cols = Vec::with_capacity(args.len());
    for arg in args {
        cols.push(as_column_ref(arg)?);
    }
    if cols.len() < 2 {
        return Err("TypeError: relate() expected at least 2 column references".to_string());
    }
    Ok(cols)
}

fn push_relation(pk: &ColRef, fk: &ColRef) {
    let relation = (Rc::clone(&pk.0), pk.1.clone(), Rc::clone(&fk.0), fk.1.clone());
    if let Some(ptr) = current_vm_ptr() {
        unsafe {
            (*ptr).pending_relations.push(relation);
        }
    } else {
        RELATIONS.with(|r| r.borrow_mut().push(relation));
    }
}

/// `relate(pk_col, fk_col, ...)` / `relate([pk_col, fk_col, ...])`
///
/// Star semantics: first column is the PK (target); each following column is a FK (source) on it.
pub fn native_relate(args: &[Value]) -> Value {
    let cols = match collect_relate_columns(args) {
        Ok(c) => c,
        Err(msg) => {
            crate::websocket::set_native_error(msg);
            return Value::Null;
        }
    };

    let pk = &cols[0];
    for fk in &cols[1..] {
        push_relation(pk, fk);
    }

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

    if !col.0.borrow().has_column(col.1) {
        return Value::Null;
    }

    let pk_entry = (col.0.clone(), col.1.clone());
    if let Some(ptr) = current_vm_ptr() {
        unsafe {
            (*ptr).pending_primary_keys.push(pk_entry);
        }
    } else {
        PRIMARY_KEYS.with(|pk| pk.borrow_mut().push(pk_entry));
    }

    Value::Null
}
