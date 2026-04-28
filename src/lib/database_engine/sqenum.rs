//! SQLEnum: declarative enum types for ORM columns (values + DDL helpers).

use crate::common::value::Value;
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::rc::Rc;

pub const KEY_ENUM_MEMBER: &str = "__enum_member";
pub const KEY_SQENUM: &str = "__sqenum";
pub const KEY_EXTENDS_SQENUM: &str = "__extends_sqenum";
pub const KEY_ENUM_CLASS: &str = "__enum_class";
pub const KEY_ENUM_NAME: &str = "__enum_name";
pub const KEY_ENUM_VALUE: &str = "__enum_value";
pub const KEY_ENUM_BY_VALUE: &str = "__enum_by_value";
pub const KEY_ENUM_MEMBERS: &str = "__enum_members";
pub const KEY_ENUM_PG_TYPE: &str = "__enum_pg_type_name";
pub const KEY_BUILTIN_SQENUM: &str = "__builtin_sqenum";
/// Marker on the `__enum_by_value` map object so `Value::Object` equality uses pointer identity only
/// (avoids deep `HashMap` comparison cycles with enum members ↔ class).
pub const KEY_ENUM_BY_VALUE_LOOKUP: &str = "__sqenum_by_value_lookup";

fn get_string(v: &Value) -> Option<String> {
    match v {
        Value::String(s) => Some(s.clone()),
        Value::Path(p) => Some(p.to_string_lossy().to_string()),
        _ => None,
    }
}

/// True if `v` is a finalized SQLEnum class object (`cls X(SQLEnum)` after finalize).
pub fn is_sqenum_class_object(v: &Value) -> bool {
    let Value::Object(rc) = v else {
        return false;
    };
    rc.borrow()
        .get(KEY_SQENUM)
        .and_then(|x| {
            if let Value::Bool(b) = x {
                Some(*b)
            } else {
                None
            }
        })
        .unwrap_or(false)
}

/// Marker object imported as `SQLEnum` (not a user-defined enum class).
pub fn is_sqenum_marker_object(v: &Value) -> bool {
    let Value::Object(rc) = v else {
        return false;
    };
    rc.borrow()
        .get(KEY_BUILTIN_SQENUM)
        .and_then(|x| {
            if let Value::Bool(b) = x {
                Some(*b)
            } else {
                None
            }
        })
        .unwrap_or(false)
}

pub fn is_enum_member_value(v: &Value) -> bool {
    let Value::Object(rc) = v else {
        return false;
    };
    rc.borrow()
        .get(KEY_ENUM_MEMBER)
        .and_then(|x| {
            if let Value::Bool(b) = x {
                Some(*b)
            } else {
                None
            }
        })
        .unwrap_or(false)
}

/// Stored scalar for SQL: string or integer (whole number).
pub fn enum_member_stored_value(member: &Value) -> Option<Value> {
    let Value::Object(rc) = member else {
        return None;
    };
    rc.borrow().get(KEY_ENUM_VALUE).cloned()
}

pub fn enum_class_of_member(member: &Value) -> Option<Rc<RefCell<HashMap<String, Value>>>> {
    let Value::Object(rc) = member else {
        return None;
    };
    let b = rc.borrow();
    b.get(KEY_ENUM_CLASS).and_then(|v| {
        if let Value::Object(c) = v {
            Some(Rc::clone(c))
        } else {
            None
        }
    })
}

/// `SQLEnum.add_member(class, name, value)` — value must be string or int literal.
pub fn native_sqenum_add_member(args: &[Value]) -> Value {
    if args.len() < 3 {
        crate::websocket::set_native_error(
            "SQLEnum.add_member requires (enum_class, name, value)".to_string(),
        );
        return Value::Null;
    }
    let Value::Object(class_rc) = &args[0] else {
        crate::websocket::set_native_error("SQLEnum.add_member: first argument must be a class object".to_string());
        return Value::Null;
    };
    let Some(name) = get_string(&args[1]) else {
        crate::websocket::set_native_error("SQLEnum.add_member: member name must be a string".to_string());
        return Value::Null;
    };
    if class_rc
        .borrow()
        .get(KEY_SQENUM)
        .and_then(|v| {
            if let Value::Bool(b) = v {
                Some(*b)
            } else {
                None
            }
        })
        .unwrap_or(false)
    {
        crate::websocket::set_native_error(format!(
            "SQLEnum.add_member: class '{}' is already finalized",
            class_rc
                .borrow()
                .get("__class_name")
                .and_then(|v| get_string(v))
                .unwrap_or_default()
        ));
        return Value::Null;
    }

    let stored: Value = match &args[2] {
        Value::String(s) => Value::String(s.clone()),
        Value::Number(n) => {
            if n.fract() != 0.0 || !n.is_finite() {
                crate::websocket::set_native_error(
                    "SQLEnum.add_member: numeric members must be whole integers".to_string(),
                );
                return Value::Null;
            }
            Value::Number(*n)
        }
        _ => {
            crate::websocket::set_native_error(
                "SQLEnum.add_member: value must be a string or integer literal".to_string(),
            );
            return Value::Null;
        }
    };

    let mut class_mut = class_rc.borrow_mut();
    let kind_slot = "__sqenum_value_kind";
    let kind = class_mut.get(kind_slot).and_then(|v| get_string(v));
    let new_kind = match &stored {
        Value::String(_) => "str",
        Value::Number(_) => "int",
        _ => unreachable!(),
    };
    match kind.as_deref() {
        None => {
            class_mut.insert(kind_slot.to_string(), Value::String(new_kind.to_string()));
        }
        Some(k) if k == new_kind => {}
        Some(_) => {
            drop(class_mut);
            crate::websocket::set_native_error(
                "SQLEnum.add_member: cannot mix string and integer values in one enum".to_string(),
            );
            return Value::Null;
        }
    }

    let seen_key = "__sqenum_seen_values";
    let seen_val = class_mut.entry(seen_key.to_string()).or_insert_with(|| {
        Value::Object(Rc::new(RefCell::new(HashMap::new())))
    });
    let seen_rc = if let Value::Object(r) = seen_val {
        Rc::clone(r)
    } else {
        let r = Rc::new(RefCell::new(HashMap::new()));
        *seen_val = Value::Object(Rc::clone(&r));
        r
    };
    let dedupe_key = value_dedupe_key(&stored);
    {
        let mut seen = seen_rc.borrow_mut();
        if seen.contains_key(&dedupe_key) {
            drop(class_mut);
            crate::websocket::set_native_error(format!(
                "SQLEnum.add_member: duplicate enum value {:?}",
                stored
            ));
            return Value::Null;
        }
        seen.insert(
            dedupe_key,
            Value::String(name.clone()),
        );
    }

    let member = {
        let mut m = HashMap::new();
        m.insert(KEY_ENUM_MEMBER.to_string(), Value::Bool(true));
        m.insert(KEY_ENUM_NAME.to_string(), Value::String(name.clone()));
        m.insert(KEY_ENUM_VALUE.to_string(), stored.clone());
        m.insert(KEY_ENUM_CLASS.to_string(), Value::Object(Rc::clone(class_rc)));
        Value::Object(Rc::new(RefCell::new(m)))
    };

    class_mut.insert(name.clone(), member.clone());

    let members_arr = class_mut
        .entry(KEY_ENUM_MEMBERS.to_string())
        .or_insert_with(|| Value::Array(Rc::new(RefCell::new(Vec::new()))));
    if let Value::Array(arr) = members_arr {
        arr.borrow_mut().push(member);
    }

    Value::Null
}

fn value_dedupe_key(v: &Value) -> String {
    match v {
        Value::String(s) => format!("s:{}", s),
        Value::Number(n) => format!("n:{}", *n as i64),
        _ => format!("o:{:?}", v),
    }
}

fn snake_case_type_name(class_name: &str) -> String {
    let mut out = String::new();
    for (i, c) in class_name.chars().enumerate() {
        if c.is_uppercase() && i > 0 {
            out.push('_');
        }
        out.push(c.to_lowercase().next().unwrap_or(c));
    }
    out
}

/// `SQLEnum.finalize(class)` — build reverse maps and mark class ready for ORM / DDL.
pub fn native_sqenum_finalize(args: &[Value]) -> Value {
    if args.is_empty() {
        crate::websocket::set_native_error("SQLEnum.finalize requires (enum_class)".to_string());
        return Value::Null;
    }
    let Value::Object(class_rc) = &args[0] else {
        crate::websocket::set_native_error(
            "SQLEnum.finalize: first argument must be a class object".to_string(),
        );
        return Value::Null;
    };
    let class_name = class_rc
        .borrow()
        .get("__class_name")
        .and_then(|v| get_string(v))
        .unwrap_or_else(|| "enum".to_string());

    let members: Vec<Value> = {
        let b = class_rc.borrow();
        b.get(KEY_ENUM_MEMBERS)
            .and_then(|v| {
                if let Value::Array(a) = v {
                    Some(a.borrow().clone())
                } else {
                    None
                }
            })
            .unwrap_or_default()
    };

    if members.is_empty() {
        crate::websocket::set_native_error(format!(
            "SQLEnum.finalize: enum class '{}' has no members",
            class_name
        ));
        return Value::Null;
    }

    let mut by_val: HashMap<String, Value> = HashMap::new();
    let mut sqlite_literals: Vec<String> = Vec::new();
    let kind = class_rc
        .borrow()
        .get("__sqenum_value_kind")
        .and_then(|v| get_string(v))
        .unwrap_or_else(|| "str".to_string());

    for m in &members {
        let Some(v) = enum_member_stored_value(m) else {
            continue;
        };
        let k = value_dedupe_key(&v);
        if by_val.insert(k.clone(), m.clone()).is_some() {
            crate::websocket::set_native_error(
                "SQLEnum.finalize: internal duplicate value".to_string(),
            );
            return Value::Null;
        }
        sqlite_literals.push(sqenum_sqlite_in_literal(&v, &kind));
    }

    let pg_name = snake_case_type_name(&class_name);

    let mut class_mut = class_rc.borrow_mut();
    let aff = if kind == "int" {
        "INTEGER"
    } else {
        "TEXT"
    };
    class_mut.insert(
        "__sqenum_sqlite_affinity".to_string(),
        Value::String(aff.to_string()),
    );
    let mut by_val_map: HashMap<String, Value> = by_val.into_iter().collect();
    by_val_map.insert(
        KEY_ENUM_BY_VALUE_LOOKUP.to_string(),
        Value::Bool(true),
    );
    class_mut.insert(
        KEY_ENUM_BY_VALUE.to_string(),
        Value::Object(Rc::new(RefCell::new(by_val_map))),
    );
    class_mut.insert(
        "__sqenum_sqlite_in_list".to_string(),
        Value::String(sqlite_literals.join(", ")),
    );
    class_mut.insert(KEY_ENUM_PG_TYPE.to_string(), Value::String(pg_name));
    class_mut.insert(KEY_SQENUM.to_string(), Value::Bool(true));
    class_mut.insert(KEY_EXTENDS_SQENUM.to_string(), Value::Bool(true));
    class_mut.remove("__sqenum_seen_values");
    class_mut.remove("__sqenum_value_kind");

    Value::Null
}

fn sqenum_sqlite_in_literal(v: &Value, kind: &str) -> String {
    match (v, kind) {
        (Value::String(s), _) => format!("'{}'", s.replace('\'', "''")),
        (Value::Number(n), _) if n.fract() == 0.0 => format!("{}", *n as i64),
        (Value::Number(n), _) => n.to_string(),
        _ => "'INVALID'".to_string(),
    }
}

/// SQLite fragment: `CHECK (col IN (...))` using precomputed list on class.
pub fn sqlite_check_in_clause_for_column(
    class_rc: &Rc<RefCell<HashMap<String, Value>>>,
    col_name: &str,
) -> Option<String> {
    let b = class_rc.borrow();
    let list = b.get("__sqenum_sqlite_in_list")?.clone();
    let list_s = get_string(&list)?;
    Some(format!("CHECK ({} IN ({}))", col_name, list_s))
}

/// Normalize assignable value to SQL bind parameter for a SQLEnum column.
pub fn normalize_sqenum_column_value(
    enum_class: &Rc<RefCell<HashMap<String, Value>>>,
    raw: &Value,
    col_name: &str,
) -> Result<Value, String> {
    if matches!(raw, Value::Null) {
        return Ok(Value::Null);
    }
    if is_enum_member_value(raw) {
        let ec = enum_class_of_member(raw);
        if ec.as_ref().map(|r| Rc::ptr_eq(r, enum_class)) != Some(true) {
            return Err(format!(
                "column '{}': enum member belongs to a different enum class",
                col_name
            ));
        }
        return enum_member_stored_value(raw).ok_or_else(|| {
            format!(
                "column '{}': invalid enum member object (missing value)",
                col_name
            )
        });
    }
    let b = enum_class.borrow();
    let by = b.get(KEY_ENUM_BY_VALUE).ok_or_else(|| {
        format!(
            "column '{}': enum class is not finalized (__enum_by_value missing)",
            col_name
        )
    })?;
    let Value::Object(by_rc) = by else {
        return Err(format!(
            "column '{}': corrupt enum class (expected __enum_by_value object)",
            col_name
        ));
    };
    let key = match raw {
        Value::String(s) => format!("s:{}", s),
        Value::Number(n) if n.fract() == 0.0 && n.is_finite() => format!("n:{}", *n as i64),
        Value::Bool(bo) => format!("s:{}", if *bo { "true" } else { "false" }),
        _ => {
            return Err(format!(
                "column '{}': value must be a string, integer, or enum member",
                col_name
            ));
        }
    };
    let map = by_rc.borrow();
    map.get(&key)
        .and_then(|m| enum_member_stored_value(m))
        .ok_or_else(|| {
            format!(
                "column '{}': value {:?} is not a valid variant for this enum",
                col_name, raw
            )
        })
}

/// Map a DB cell to enum member (strict).
pub fn hydrate_sqenum_cell(
    enum_class: &Rc<RefCell<HashMap<String, Value>>>,
    cell: &Value,
) -> Result<Value, String> {
    if matches!(cell, Value::Null) {
        return Ok(Value::Null);
    }
    let b = enum_class.borrow();
    let by = b.get(KEY_ENUM_BY_VALUE).ok_or_else(|| {
        "enum class is not finalized (__enum_by_value missing)".to_string()
    })?;
    let Value::Object(by_rc) = by else {
        return Err("corrupt enum class".to_string());
    };
    let key = match cell {
        Value::String(s) => format!("s:{}", s),
        Value::Number(n) if n.fract() == 0.0 && n.is_finite() => format!("n:{}", *n as i64),
        Value::Bool(bo) => format!("s:{}", if *bo { "true" } else { "false" }),
        _ => {
            return Err(format!(
                "cannot hydrate enum from DB value {:?}",
                cell
            ));
        }
    };
    let map = by_rc.borrow();
    map.get(&key)
        .cloned()
        .ok_or_else(|| format!("DB value {:?} is not in enum set", cell))
}

/// Collect ordered stored values for Postgres `CREATE TYPE ... AS ENUM`.
pub fn pg_enum_values_from_class(
    class_rc: &Rc<RefCell<HashMap<String, Value>>>,
) -> Option<Vec<String>> {
    let b = class_rc.borrow();
    let members = b.get(KEY_ENUM_MEMBERS)?;
    let Value::Array(arr) = members else {
        return None;
    };
    let mut out: Vec<String> = arr
        .borrow()
        .iter()
        .filter_map(|m| enum_member_stored_value(m))
        .map(|v| match v {
            Value::String(s) => s,
            Value::Number(n) if n.fract() == 0.0 => format!("{}", n as i64),
            Value::Number(n) => n.to_string(),
            _ => String::new(),
        })
        .filter(|s| !s.is_empty())
        .collect();
    let mut seen = HashSet::new();
    out.retain(|s| seen.insert(s.clone()));
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dedupe_key_string_int_distinct() {
        assert_ne!(value_dedupe_key(&Value::String("1".into())), value_dedupe_key(&Value::Number(1.0)));
    }
}
