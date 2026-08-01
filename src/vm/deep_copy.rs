//! Recursive deep copy for container [`Value`]s (used by `copy()`, `.clone()`, `set.copy()`).

use crate::common::object_map::ObjectMap;
use crate::common::set_map::SetMap;
use crate::common::table::{Table, TableData};
use crate::common::value::{ObjectKind, Value};
use crate::common::value_store::{ValueStore, NULL_VALUE_ID};
use crate::vm::calls::get_type_name_value;
use crate::vm::heavy_store::HeavyStore;
use crate::vm::set_ops::set_insert_material;
use crate::vm::store_convert::{load_value, object_map_upsert, store_value};
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

type ContainerPtr = *const ();

pub struct DeepCopyCtx<'a> {
    store: &'a mut ValueStore,
    heap: &'a mut HeavyStore,
    seen: HashMap<ContainerPtr, Value>,
}

pub fn deep_copy(
    value: &Value,
    store: &mut ValueStore,
    heap: &mut HeavyStore,
) -> Result<Value, String> {
    let mut ctx = DeepCopyCtx {
        store,
        heap,
        seen: HashMap::new(),
    };
    deep_copy_impl(value, &mut ctx)
}

fn deep_copy_impl(value: &Value, ctx: &mut DeepCopyCtx<'_>) -> Result<Value, String> {
    match value {
        Value::Int(_)
        | Value::Float(_)
        | Value::Number(_)
        | Value::Bool(_)
        | Value::String(_)
        | Value::Null
        | Value::Ellipsis
        | Value::Path(_)
        | Value::Uuid(_, _)
        | Value::Date(_)
        | Value::Duration(_)
        | Value::PluginOpaque { .. }
        | Value::Window(_) => Ok(value.clone()),

        Value::Array(arr) => {
            let ptr = Rc::as_ptr(arr) as ContainerPtr;
            if let Some(v) = ctx.seen.get(&ptr) {
                return Ok(v.clone());
            }
            let shell = Rc::new(RefCell::new(Vec::new()));
            ctx.seen
                .insert(ptr, Value::Array(Rc::clone(&shell)));
            let copied: Result<Vec<Value>, String> = arr
                .borrow()
                .iter()
                .map(|v| deep_copy_impl(v, ctx))
                .collect();
            *shell.borrow_mut() = copied?;
            Ok(Value::Array(shell))
        }

        Value::Tuple(tuple) => {
            let ptr = Rc::as_ptr(tuple) as ContainerPtr;
            if let Some(v) = ctx.seen.get(&ptr) {
                return Ok(v.clone());
            }
            let shell = Rc::new(RefCell::new(Vec::new()));
            ctx.seen
                .insert(ptr, Value::Tuple(Rc::clone(&shell)));
            let copied: Result<Vec<Value>, String> = tuple
                .borrow()
                .iter()
                .map(|v| deep_copy_impl(v, ctx))
                .collect();
            *shell.borrow_mut() = copied?;
            Ok(Value::Tuple(shell))
        }

        Value::Object(obj_rc) => deep_copy_object(obj_rc, ctx),

        Value::Set(set_rc) => {
            let ptr = Rc::as_ptr(set_rc) as ContainerPtr;
            if let Some(v) = ctx.seen.get(&ptr) {
                return Ok(v.clone());
            }
            let shell = Rc::new(RefCell::new(SetMap::with_capacity(set_rc.borrow().len())));
            ctx.seen.insert(ptr, Value::Set(Rc::clone(&shell)));
            for kid in crate::vm::set_ops::set_member_key_ids(&set_rc.borrow(), ctx.store) {
                let elem = load_value(kid, ctx.store, ctx.heap);
                let copied = deep_copy_impl(&elem, ctx)?;
                set_insert_material(&mut shell.borrow_mut(), &copied, ctx.store, ctx.heap)?;
            }
            Ok(Value::Set(shell))
        }

        Value::Table(table_rc) => {
            let ptr = Rc::as_ptr(table_rc) as ContainerPtr;
            if let Some(v) = ctx.seen.get(&ptr) {
                return Ok(v.clone());
            }
            let table_ref = table_rc.borrow();
            let materialized = table_ref.materialize_with(|id| load_value(id, ctx.store, ctx.heap));
            let shell = Rc::new(RefCell::new(Table {
                data: TableData::Owned {
                    flat: Vec::new(),
                    num_cols: 0,
                    headers: Vec::new(),
                    column_cache: HashMap::new(),
                },
                name: table_ref.name.clone(),
            }));
            ctx.seen.insert(ptr, Value::Table(Rc::clone(&shell)));
            match &materialized.data {
                TableData::Owned {
                    flat,
                    num_cols,
                    headers,
                    ..
                } => {
                    let new_flat: Result<Vec<Value>, String> = flat
                        .iter()
                        .map(|v| deep_copy_impl(v, ctx))
                        .collect();
                    shell.borrow_mut().data = TableData::Owned {
                        flat: new_flat?,
                        num_cols: *num_cols,
                        headers: headers.clone(),
                        column_cache: HashMap::new(),
                    };
                }
                TableData::View { .. } => {
                    return Err("copy() internal error: table materialization failed".to_string());
                }
            }
            Ok(Value::Table(shell))
        }

        Value::ArrayView(av) => Ok(Value::ArrayView(av.clone())),
        Value::ByteBuffer(b) => Ok(Value::ByteBuffer(b.clone())),
        Value::Image(img) => Ok(Value::Image(img.clone())),
        Value::Figure(fig) => Ok(Value::Figure(fig.clone())),
        Value::Axis(axis) => Ok(Value::Axis(axis.clone())),

        Value::Enumerate { data, start } => {
            let copied_data = match deep_copy_impl(&Value::Array(data.clone()), ctx)? {
                Value::Array(rc) => rc,
                _ => return Err("copy() internal error: enumerate data".to_string()),
            };
            Ok(Value::Enumerate {
                data: copied_data,
                start: *start,
            })
        }

        Value::ColumnReference { table, column_name } => {
            let copied_table = match deep_copy_impl(&Value::Table(table.clone()), ctx)? {
                Value::Table(rc) => rc,
                _ => return Err("copy() internal error: column reference table".to_string()),
            };
            Ok(Value::ColumnReference {
                table: copied_table,
                column_name: column_name.clone(),
            })
        }

        Value::Function(_)
        | Value::ModuleFunction { .. }
        | Value::NativeFunction(_)
        | Value::Iterable(_)
        | Value::Generator(_)
        | Value::DatabaseEngine(_)
        | Value::DatabaseCluster(_)
        | Value::Archive(_)
        | Value::DataSource(_)
        | Value::DataSourceResponse(_)
        | Value::HttpResponse(_)
        | Value::WebPage(_)
        | Value::WebElement(_)
        | Value::ObjectFieldList { .. } => Err(format!(
            "TypeError: copy() does not support type {}",
            get_type_name_value(value)
        )),
    }
}

fn deep_copy_object(
    obj_rc: &Rc<RefCell<ObjectKind>>,
    ctx: &mut DeepCopyCtx<'_>,
) -> Result<Value, String> {
    let ptr = Rc::as_ptr(obj_rc) as ContainerPtr;
    if let Some(v) = ctx.seen.get(&ptr) {
        return Ok(v.clone());
    }
    let shell = Rc::new(RefCell::new(ObjectKind::Inline(Vec::new())));
    ctx.seen.insert(ptr, Value::Object(Rc::clone(&shell)));

    let snap = obj_rc.borrow().clone();
    match snap {
        ObjectKind::Legacy(map) => {
            let mut out = HashMap::new();
            for (k, v) in map {
                out.insert(k, deep_copy_impl(&v, ctx)?);
            }
            *shell.borrow_mut() = ObjectKind::Legacy(out);
        }
        ObjectKind::Inline(pairs) => {
            let mut out = Vec::with_capacity(pairs.len());
            for (k, v) in pairs {
                out.push((deep_copy_impl(&k, ctx)?, deep_copy_impl(&v, ctx)?));
            }
            *shell.borrow_mut() = ObjectKind::Inline(out);
        }
        ObjectKind::Bucket(omap) => {
            let mut new_map = ObjectMap::with_capacity(omap.len());
            for (_, kid, vid) in omap.iter_entries() {
                if kid == NULL_VALUE_ID && vid == NULL_VALUE_ID {
                    continue;
                }
                let key_mat = load_value(kid, ctx.store, ctx.heap);
                let val_mat = load_value(vid, ctx.store, ctx.heap);
                let new_key = deep_copy_impl(&key_mat, ctx)?;
                let new_val = deep_copy_impl(&val_mat, ctx)?;
                let new_kid = store_value(new_key.clone(), ctx.store, ctx.heap);
                let new_vid = store_value(new_val, ctx.store, ctx.heap);
                object_map_upsert(
                    &mut new_map,
                    ctx.store,
                    ctx.heap,
                    &new_key,
                    new_kid,
                    new_vid,
                );
            }
            *shell.borrow_mut() = ObjectKind::Bucket(new_map);
        }
    }
    Ok(Value::Object(shell))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::value::Value;
    use crate::vm::heavy_store::HeavyStore;
    use std::cell::RefCell;
    use std::rc::Rc;

    #[test]
    fn deep_copy_nested_array_independence() {
        let mut store = ValueStore::new();
        let mut heap = HeavyStore::new();
        let inner = Rc::new(RefCell::new(vec![Value::Number(2.0)]));
        let outer = Rc::new(RefCell::new(vec![
            Value::Number(1.0),
            Value::Array(Rc::clone(&inner)),
        ]));
        let v = Value::Array(outer);
        let copied = deep_copy(&v, &mut store, &mut heap).unwrap();
        if let Value::Array(arr) = copied {
            arr.borrow_mut()[1] = Value::Array(Rc::new(RefCell::new(vec![Value::Number(99.0)])));
        }
        assert_eq!(inner.borrow()[0], Value::Number(2.0));
    }

    #[test]
    fn deep_copy_legacy_object() {
        let mut store = ValueStore::new();
        let mut heap = HeavyStore::new();
        let nested = Rc::new(RefCell::new(vec![Value::Number(1.0)]));
        let mut map = HashMap::new();
        map.insert("a".to_string(), Value::Array(Rc::clone(&nested)));
        let v = Value::Object(Rc::new(RefCell::new(ObjectKind::Legacy(map))));
        let copied = deep_copy(&v, &mut store, &mut heap).unwrap();
        if let Value::Object(obj) = copied {
            if let ObjectKind::Legacy(m) = &*obj.borrow() {
                if let Value::Array(arr) = &m["a"] {
                    arr.borrow_mut()[0] = Value::Number(42.0);
                }
            }
        }
        assert_eq!(nested.borrow()[0], Value::Number(1.0));
    }

    #[test]
    fn deep_copy_cycle_array() {
        let mut store = ValueStore::new();
        let mut heap = HeavyStore::new();
        let arr = Rc::new(RefCell::new(Vec::<Value>::new()));
        arr.borrow_mut().push(Value::Array(Rc::clone(&arr)));
        let copied = deep_copy(&Value::Array(Rc::clone(&arr)), &mut store, &mut heap).unwrap();
        if let Value::Array(out) = copied {
            assert_eq!(out.borrow().len(), 1);
            assert!(matches!(out.borrow()[0], Value::Array(_)));
        } else {
            panic!("expected array");
        }
    }
}
