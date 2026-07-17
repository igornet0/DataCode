//! `table.save_csv(path)` and `table.save_sqlite(path)` — export table to file.

use crate::common::table::Table;
use crate::common::table_csv_export::write_table_csv;
use crate::common::value::Value;
use crate::common::value_store::{ValueCell, ValueStore};
use crate::file_io::path_from_value;
use crate::sqlite_export::export_single_table;
use crate::vm::natives::file::resolve_path_in_session;
use crate::vm::permission_policy::PermissionPolicy;
use crate::vm::store_convert::load_value;
use crate::vm::vm::{current_vm_ptr, with_current_stores};
use std::cell::RefCell;
use std::path::{Path, PathBuf};
use std::rc::Rc;

fn save_error(msg: impl Into<String>) -> Value {
    crate::websocket::set_native_error(msg.into());
    Value::Null
}

fn check_fs_write() -> Result<(), String> {
    let Some(ptr) = current_vm_ptr() else {
        return Ok(());
    };
    let vm = unsafe { &*ptr };
    if vm.can_system_permission(PermissionPolicy::FS_WRITE) {
        Ok(())
    } else {
        Err(format!(
            "permission denied: {}",
            PermissionPolicy::FS_WRITE
        ))
    }
}

fn materialize_table(table_rc: &Rc<RefCell<Table>>) -> Table {
    with_current_stores(|store, heap| {
        let table_ref = table_rc.borrow();
        if table_ref.is_view() {
            table_ref.materialize_with(|id| load_value(id, store, heap))
        } else {
            table_ref.clone()
        }
    })
}

fn ensure_parent_dir(path: &Path) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() && parent != Path::new(".") && !parent.exists() {
            return Err(format!(
                "Directory does not exist: {}",
                parent.to_string_lossy()
            ));
        }
    }
    Ok(())
}

fn resolve_output_path(arg: &Value, default_ext: &str) -> Result<PathBuf, String> {
    let mut path = path_from_value(arg).map_err(|_| "path must be a string".to_string())?;
    path = resolve_path_in_session(&path)?;
    if path.extension().is_none() {
        path.set_extension(default_ext);
    }
    ensure_parent_dir(&path)?;
    Ok(path)
}

fn tables_same_content(a: &Table, b: &Table) -> bool {
    if a.headers() != b.headers() || a.len() != b.len() {
        return false;
    }
    for i in 0..a.len() {
        let Some(a_row) = a.get_row(i) else {
            return false;
        };
        let Some(b_row) = b.get_row(i) else {
            return false;
        };
        if a_row != b_row {
            return false;
        }
    }
    true
}

fn table_matches_binding(candidate: &Rc<RefCell<Table>>, target: &Rc<RefCell<Table>>) -> bool {
    Rc::ptr_eq(candidate, target)
        || tables_same_content(&candidate.borrow(), &target.borrow())
}

fn heavy_index_of_value_id(
    value_id: crate::common::value_store::ValueId,
    store: &ValueStore,
) -> Option<usize> {
    match store.get(value_id)? {
        ValueCell::Heavy(idx) => Some(*idx),
        _ => None,
    }
}

fn name_for_table_in_map(
    map: &std::collections::HashMap<String, Rc<RefCell<Table>>>,
    table_rc: &Rc<RefCell<Table>>,
) -> Option<String> {
    let mut matched = Vec::new();
    for (name, candidate) in map {
        if table_matches_binding(candidate, table_rc) {
            matched.push(name.clone());
        }
    }
    if matched.len() == 1 {
        return Some(matched[0].clone());
    }
    if !matched.is_empty() {
        return matched.into_iter().next();
    }
    // Array literals may hold materialized copies — match by column headers when unique.
    let headers: Vec<String> = table_rc.borrow().headers().to_vec();
    let mut header_matches: Vec<String> = map
        .iter()
        .filter(|(_, t)| t.borrow().headers().as_slice() == headers.as_slice())
        .map(|(n, _)| n.clone())
        .collect();
    if header_matches.len() == 1 {
        Some(header_matches.remove(0))
    } else {
        None
    }
}

fn binding_name_for_table(table_rc: &Rc<RefCell<Table>>) -> Option<String> {
    if let Some(name) = table_rc.borrow().name.clone().filter(|n| !n.is_empty()) {
        return Some(name);
    }
    let Some(ptr) = current_vm_ptr() else {
        return None;
    };
    let vm = unsafe { &mut *ptr };
    let global_len = vm.get_globals().len();
    let global_name_map: std::collections::BTreeMap<usize, String> = vm
        .get_global_names()
        .iter()
        .chain(vm.get_explicit_global_names().iter())
        .map(|(i, n)| (*i, n.clone()))
        .collect();

    let name_for_index =
        |index: usize, names: &std::collections::BTreeMap<usize, String>| -> Option<String> {
            names.get(&index).cloned()
        };

    let mut sole_table_name: Option<String> = None;
    let mut table_global_count = 0usize;

    for index in 0..global_len {
        let value_id = vm.resolve_global_to_value_id(index);
        let value = load_value(value_id, vm.value_store(), vm.heavy_store());
        if let Value::Table(candidate) = &value {
            table_global_count += 1;
            if let Some(name) = name_for_index(index, &global_name_map) {
                sole_table_name = Some(name.clone());
            }
            if table_matches_binding(candidate, table_rc) {
                if let Some(name) = name_for_index(index, &global_name_map) {
                    table_rc.borrow_mut().set_name(name.clone());
                    return Some(name);
                }
            }
        }
    }

    if table_global_count == 1 {
        if let Some(name) = sole_table_name {
            table_rc.borrow_mut().set_name(name.clone());
            return Some(name);
        }
    }

    // Heavy-index fallback: native args may be a clone Rc while globals still point at the heap slot.
    let arg_heavy = (0..vm.heavy_store().len()).find(|&i| {
        matches!(
            vm.heavy_store().get(i),
            Some(Value::Table(t)) if table_matches_binding(t, table_rc)
        )
    });
    if let Some(arg_h) = arg_heavy {
        for index in 0..global_len {
            let value_id = vm.resolve_global_to_value_id(index);
            if heavy_index_of_value_id(value_id, vm.value_store()) == Some(arg_h) {
                if let Some(name) = name_for_index(index, &global_name_map) {
                    table_rc.borrow_mut().set_name(name.clone());
                    return Some(name);
                }
            }
        }
    }

    None
}

pub fn native_table_save_csv(args: &[Value]) -> Value {
    if args.len() < 2 {
        return save_error("save_csv() expects a path argument");
    }
    if let Err(e) = check_fs_write() {
        return save_error(e);
    }
    let Value::Table(table_rc) = &args[0] else {
        return Value::Null;
    };
    let path = match resolve_output_path(&args[1], "csv") {
        Ok(p) => p,
        Err(e) => return save_error(e),
    };
    let table = materialize_table(table_rc);
    match write_table_csv(&table, &path) {
        Ok(()) => Value::String(path.to_string_lossy().to_string()),
        Err(e) => save_error(e),
    }
}

pub fn native_table_save_sqlite(args: &[Value]) -> Value {
    if args.len() < 2 {
        return save_error("save_sqlite() expects a path argument");
    }
    if let Err(e) = check_fs_write() {
        return save_error(e);
    }
    let Value::Table(table_rc) = &args[0] else {
        return Value::Null;
    };
    let table_name = match binding_name_for_table(table_rc) {
        Some(n) => n,
        None => {
            return save_error(
                "save_sqlite: table has no name; assign table to a variable first",
            );
        }
    };
    let path = match resolve_output_path(&args[1], "sqlite") {
        Ok(p) => p,
        Err(e) => return save_error(e),
    };
    let table = materialize_table(table_rc);
    match export_single_table(&table, &path, &table_name) {
        Ok(()) => Value::String(path.to_string_lossy().to_string()),
        Err(e) => save_error(e),
    }
}

fn kwargs_object_to_map(value: &Value) -> std::collections::HashMap<String, Value> {
    let mut out = std::collections::HashMap::new();
    let Value::Object(rc) = value else {
        return out;
    };
    for (k, v) in rc.borrow().str_key_entries_cloned() {
        out.insert(k, v);
    }
    out
}

/// `save_tables_sqlite(tables, filename="db", **kwargs)` — multi-table SQLite export.
pub fn native_save_tables_sqlite(args: &[Value]) -> Value {
    if args.is_empty() {
        return save_error("save_tables_sqlite() expects an array of tables");
    }
    if let Err(e) = check_fs_write() {
        return save_error(e);
    }
    let Value::Array(arr_rc) = &args[0] else {
        return save_error("save_tables_sqlite: first argument must be an array of tables");
    };
    let filename_arg = if args.len() > 1 && !matches!(args[1], Value::Null) {
        &args[1]
    } else {
        &Value::String("db".to_string())
    };
    let path = match resolve_output_path(filename_arg, "sqlite") {
        Ok(p) => p,
        Err(e) => return save_error(e),
    };
    let extra = if args.len() > 2 {
        kwargs_object_to_map(&args[2])
    } else {
        std::collections::HashMap::new()
    };

    let Some(ptr) = current_vm_ptr() else {
        return save_error("save_tables_sqlite: VM context not available");
    };
    let vm = unsafe { &mut *ptr };

    let arr = arr_rc.borrow();
    let global_tables = match crate::sqlite_export::get_global_tables(vm) {
        Ok(m) => m,
        Err(e) => return save_error(e),
    };
    let mut tables: Vec<(String, Rc<RefCell<Table>>)> = Vec::new();
    for item in arr.iter() {
        let Value::Table(table_rc) = item else {
            return save_error("save_tables_sqlite: array must contain only tables");
        };
        let name = name_for_table_in_map(&global_tables, table_rc)
            .or_else(|| binding_name_for_table(table_rc));
        let name = match name {
            Some(n) => n,
            None => {
                return save_error(
                    "save_tables_sqlite: table has no name; assign each table to a variable first",
                );
            }
        };
        tables.push((name, table_rc.clone()));
    }
    if tables.is_empty() {
        return save_error("save_tables_sqlite: tables array is empty");
    }

    let path_str = path.to_string_lossy().to_string();
    match crate::sqlite_export::export_tables_to_sqlite(
        vm,
        &tables,
        &path_str,
        &extra,
        false,
    ) {
        Ok(()) => Value::String(path_str),
        Err(e) => save_error(e),
    }
}
