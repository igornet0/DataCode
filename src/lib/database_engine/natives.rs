// Native functions for database module

use crate::common::value::Value;
use crate::database_engine::cluster::DatabaseCluster;
use crate::database_engine::engine::DatabaseEngine;
use crate::vm::globals;
use crate::vm::natives::utils::{call_user_function, resolve_global_by_name};
use crate::vm::vm::VM_CALL_CONTEXT;
use std::collections::{HashMap, HashSet};
use std::rc::Rc;
use std::cell::RefCell;

fn get_string(v: &Value) -> Option<String> {
    match v {
        Value::String(s) => Some(s.clone()),
        Value::Path(p) => Some(p.to_string_lossy().to_string()),
        _ => None,
    }
}

fn get_bool(v: &Value, default: bool) -> bool {
    match v {
        Value::Bool(b) => *b,
        _ => default,
    }
}

fn get_u32(v: &Value, default: u32) -> u32 {
    match v {
        Value::Number(n) if *n >= 0.0 && n.fract() == 0.0 => *n as u32,
        _ => default,
    }
}

fn get_f64_opt(v: &Value) -> Option<f64> {
    match v {
        Value::Number(n) => Some(*n),
        Value::Null => None,
        _ => None,
    }
}

fn extract_connect_args(v: &Value) -> HashMap<String, Value> {
    let mut map = HashMap::new();
    if let Value::Object(rc) = v {
        for (k, val) in rc.borrow().iter() {
            map.insert(k.clone(), val.clone());
        }
    }
    map
}

fn extract_params_array(v: &Value) -> Vec<Value> {
    match v {
        Value::Array(rc) => rc.borrow().clone(),
        _ => vec![],
    }
}

fn get_engine_mut(args: &[Value]) -> Option<std::cell::RefMut<'_, DatabaseEngine>> {
    if args.is_empty() {
        return None;
    }
    match &args[0] {
        Value::DatabaseEngine(rc) => Some(rc.borrow_mut()),
        _ => None,
    }
}

fn get_cluster_mut(args: &[Value]) -> Option<std::cell::RefMut<'_, DatabaseCluster>> {
    if args.is_empty() {
        return None;
    }
    match &args[0] {
        Value::DatabaseCluster(rc) => Some(rc.borrow_mut()),
        _ => None,
    }
}

/// engine(url, echo?, echo_pool?, pool_size?, max_overflow?, timeout?, connect_args?)
/// Creates a database engine. URL scheme determines backend (sqlite:// for SQLite).
pub fn native_engine(args: &[Value]) -> Value {
    if args.is_empty() {
        crate::websocket::set_native_error("database.engine requires url argument".to_string());
        return Value::Null;
    }
    let url = match get_string(&args[0]) {
        Some(s) => s,
        None => {
            crate::websocket::set_native_error("database.engine: url must be a string".to_string());
            return Value::Null;
        }
    };

    let echo = args.get(1).map(|v| get_bool(v, false)).unwrap_or(false);
    let echo_pool = args.get(2).map(|v| get_bool(v, false)).unwrap_or(false);
    let pool_size = args.get(3).map(|v| get_u32(v, 5)).unwrap_or(5);
    let max_overflow = args.get(4).map(|v| get_u32(v, 10)).unwrap_or(10);
    let timeout = args.get(5).and_then(get_f64_opt);
    let connect_args = args
        .get(6)
        .map(extract_connect_args)
        .unwrap_or_default();

    let url_lower = url.to_lowercase();
    if url_lower.starts_with("sqlite:") {
        match DatabaseEngine::new_sqlite(
            url,
            echo,
            echo_pool,
            pool_size,
            max_overflow,
            timeout,
            connect_args,
        ) {
            Ok(engine) => Value::DatabaseEngine(Rc::new(RefCell::new(engine))),
            Err(e) => {
                crate::websocket::set_native_error(format!("database.engine: {}", e));
                Value::Null
            }
        }
    } else {
        crate::websocket::set_native_error(format!(
            "database.engine: unsupported database URL scheme (only sqlite:// is supported in MVP)"
        ));
        Value::Null
    }
}

/// connect() - returns connection (for SQLite, same as engine)
pub fn native_engine_connect(args: &[Value]) -> Value {
    if args.is_empty() {
        crate::websocket::set_native_error("engine.connect requires engine".to_string());
        return Value::Null;
    }
    match &args[0] {
        Value::DatabaseEngine(rc) => Value::DatabaseEngine(Rc::clone(rc)),
        _ => {
            crate::websocket::set_native_error("engine.connect: first argument must be a database engine".to_string());
            Value::Null
        }
    }
}

/// execute(sql, params?) - execute SQL, return row count
pub fn native_engine_execute(args: &[Value]) -> Value {
    if args.len() < 2 {
        crate::websocket::set_native_error("engine.execute requires (engine, sql [, params])".to_string());
        return Value::Null;
    }
    let sql = match get_string(&args[1]) {
        Some(s) => s,
        None => {
            crate::websocket::set_native_error("engine.execute: sql must be a string".to_string());
            return Value::Null;
        }
    };
    let params = args.get(2).map(extract_params_array).unwrap_or_default();

    let mut engine_ref = match get_engine_mut(args) {
        Some(r) => r,
        None => {
            crate::websocket::set_native_error("engine.execute: first argument must be a database engine".to_string());
            return Value::Null;
        }
    };

    match engine_ref.execute(&sql, &params) {
        Ok(count) => Value::Number(count as f64),
        Err(e) => {
            crate::websocket::set_native_error(format!("engine.execute: {}", e));
            Value::Null
        }
    }
}

/// query(sql, params?) - execute SELECT, return Table
pub fn native_engine_query(args: &[Value]) -> Value {
    if args.len() < 2 {
        crate::websocket::set_native_error("engine.query requires (engine, sql [, params])".to_string());
        return Value::Null;
    }
    let sql = match get_string(&args[1]) {
        Some(s) => s,
        None => {
            crate::websocket::set_native_error("engine.query: sql must be a string".to_string());
            return Value::Null;
        }
    };
    let params = args.get(2).map(extract_params_array).unwrap_or_default();

    let mut engine_ref = match get_engine_mut(args) {
        Some(r) => r,
        None => {
            crate::websocket::set_native_error("engine.query: first argument must be a database engine".to_string());
            return Value::Null;
        }
    };

    match engine_ref.query(&sql, &params) {
        Ok(table) => Value::Table(Rc::new(RefCell::new(table))),
        Err(e) => {
            crate::websocket::set_native_error(format!("engine.query: {}", e));
            Value::Null
        }
    }
}

/// MetaData(schema?, quote_schema?, naming_convention?, info?) - creates metadata object for ORM
pub fn native_metadata(args: &[Value]) -> Value {
    let mut meta = HashMap::new();
    meta.insert("__meta".to_string(), Value::Bool(true));
    meta.insert("schema".to_string(), args.get(0).and_then(get_string).map(Value::String).unwrap_or(Value::String("public".to_string())));
    meta.insert("quote_schema".to_string(), args.get(1).cloned().unwrap_or(Value::Null));
    meta.insert("naming_convention".to_string(), args.get(2).cloned().unwrap_or(Value::Object(Rc::new(RefCell::new(HashMap::new())))));
    meta.insert("info".to_string(), args.get(3).cloned().unwrap_or(Value::Object(Rc::new(RefCell::new(HashMap::new())))));
    meta.insert("tables".to_string(), Value::Array(Rc::new(RefCell::new(Vec::new()))));
    meta.insert("classes".to_string(), Value::Object(Rc::new(RefCell::new(HashMap::new()))));
    let meta_rc = Rc::new(RefCell::new(meta));
    let mut create_all_obj = HashMap::new();
    create_all_obj.insert("__create_all".to_string(), Value::Bool(true));
    create_all_obj.insert("metadata".to_string(), Value::Object(Rc::clone(&meta_rc)));
    meta_rc.borrow_mut().insert("create_all".to_string(), Value::Object(Rc::new(RefCell::new(create_all_obj))));
    Value::Object(meta_rc)
}

/// Column(type?, primary_key?, autoincrement?, unique?, default?, nullable?, onupdate?) - column descriptor
pub fn native_column(args: &[Value]) -> Value {
    let mut col = HashMap::new();
    col.insert("__column".to_string(), Value::Bool(true));
    col.insert("type".to_string(), args.get(0).cloned().unwrap_or(Value::Null));
    col.insert("primary_key".to_string(), Value::Bool(args.get(1).map(|v| get_bool(v, false)).unwrap_or(false)));
    col.insert("autoincrement".to_string(), Value::Bool(args.get(2).map(|v| get_bool(v, false)).unwrap_or(false)));
    col.insert("unique".to_string(), Value::Bool(args.get(3).map(|v| get_bool(v, false)).unwrap_or(false)));
    col.insert("default".to_string(), args.get(4).cloned().unwrap_or(Value::Null));
    col.insert("nullable".to_string(), Value::Bool(args.get(5).map(|v| get_bool(v, false)).unwrap_or(false)));
    col.insert("onupdate".to_string(), args.get(6).cloned().unwrap_or(Value::Null));
    Value::Object(Rc::new(RefCell::new(col)))
}

/// now_call() - returns current datetime for default/onupdate (ISO string for SQLite)
pub fn native_now_call(_args: &[Value]) -> Value {
    use chrono::Utc;
    Value::String(Utc::now().format("%Y-%m-%dT%H:%M:%SZ").to_string())
}

/// select(model_class) - returns select statement object for conn.run(select(User))
pub fn native_select(args: &[Value]) -> Value {
    if args.is_empty() {
        crate::websocket::set_native_error("database.select requires model class argument".to_string());
        return Value::Null;
    }
    let mut sel = HashMap::new();
    sel.insert("__select".to_string(), Value::Bool(true));
    sel.insert("model".to_string(), args[0].clone());
    Value::Object(Rc::new(RefCell::new(sel)))
}

/// run(engine, arg) - dispatches: create_all -> DDL, model instance -> INSERT, select(Model) -> SELECT
pub fn native_engine_run(args: &[Value]) -> Value {
    if args.len() < 2 {
        crate::websocket::set_native_error("engine.run requires (engine, callable_or_instance)".to_string());
        return Value::Null;
    }
    let arg = args[1].clone();
    let mut engine_ref = match get_engine_mut(args) {
        Some(r) => r,
        None => {
            crate::websocket::set_native_error("engine.run: first argument must be a database engine".to_string());
            return Value::Null;
        }
    };

    // Resolve arg: if passed a model class (has __class_name, metadata.create_all) instead of create_all
    // (e.g. due to slot/global index mismatch in VM), use class.metadata.create_all.
    let arg_resolved = if let Value::Object(rc) = &arg {
        let obj = rc.borrow();
        let is_create_all = obj.get("__create_all").and_then(|v| if let Value::Bool(b) = v { Some(*b) } else { None }).unwrap_or(false);
        let has_class_name = obj.get("__class_name").is_some();
        let meta_opt = obj.get("metadata").cloned();
        let create_all_opt: Option<Value> = meta_opt.as_ref().and_then(|m| {
            if let Value::Object(mr) = m { mr.borrow().get("create_all").cloned() } else { None }
        });
        let create_all_has_marker = create_all_opt.as_ref().and_then(|c| {
            if let Value::Object(cr) = c { cr.borrow().get("__create_all").and_then(|v| if let Value::Bool(b) = v { Some(*b) } else { None }) } else { None }
        }).unwrap_or(false);
        drop(obj);
        if is_create_all {
            arg.clone()
        } else if has_class_name && create_all_has_marker {
            create_all_opt.unwrap()
        } else {
            arg.clone()
        }
    } else {
        arg.clone()
    };

    // Check metadata.create_all first so create_all object is never mistaken for a model instance
    if let Value::Object(rc) = &arg_resolved {
        let obj = rc.borrow();
        let is_create_all = obj.get("__create_all").and_then(|v| if let Value::Bool(b) = v { Some(*b) } else { None }).unwrap_or(false);
        let meta_opt = obj.get("metadata").cloned();
        drop(obj);
        if is_create_all {
            if let Some(Value::Object(meta_rc)) = meta_opt {
                if let Err(e) = run_create_all(&meta_rc, &mut engine_ref) {
                    crate::websocket::set_native_error(format!("engine.run create_all: {}", e));
                    return Value::Null;
                }
                return Value::Null;
            }
        }
    }

    // Check if arg is select(Model) result
    if let Value::Object(rc) = &arg_resolved {
        let obj = rc.borrow();
        if obj.get("__select").and_then(|v| if let Value::Bool(b) = v { Some(*b) } else { None }).unwrap_or(false) {
            if let Some(Value::Object(model_class)) = obj.get("model") {
                let classes = get_classes_from_class_chain(model_class);
                let table_name = get_table_name_from_class(&Value::Object(Rc::clone(model_class)), classes.as_ref());
                if let Some(name) = table_name {
                    let sql = format!("SELECT * FROM {}", name);
                    let model_class = Rc::clone(model_class);
                    let class_name = model_class.borrow().get("__class_name").and_then(get_string).unwrap_or_default();
                    drop(obj);
                    match engine_ref.query(&sql, &[]) {
                        Ok(table) => {
                            let headers = table.headers().clone();
                            let data_rows = match table.rows_ref() {
                                Some(rr) => {
                                    let len = rr.len();
                                    (0..len)
                                        .filter_map(|i| rr.row(i).map(|s| s.to_vec()))
                                        .collect::<Vec<_>>()
                                }
                                None => vec![],
                            };
                            let mut data_list = Vec::with_capacity(data_rows.len());
                            let mut rows_list = Vec::with_capacity(data_rows.len());
                            for row_vals in &data_rows {
                                let mut instance = HashMap::new();
                                instance.insert("__class_name".to_string(), Value::String(class_name.clone()));
                                instance.insert("__class".to_string(), Value::Object(Rc::clone(&model_class)));
                                for (col_idx, col_name) in headers.iter().enumerate() {
                                    let val = row_vals.get(col_idx).cloned().unwrap_or(Value::Null);
                                    instance.insert(col_name.clone(), val);
                                }
                                instance.insert("__state".to_string(), Value::String("loaded".to_string()));
                                data_list.push(Value::Object(Rc::new(RefCell::new(instance))));
                                rows_list.push(Value::Array(Rc::new(RefCell::new(row_vals.clone()))));
                            }
                            let mut result = HashMap::new();
                            result.insert("__result".to_string(), Value::Bool(true));
                            result.insert("data".to_string(), Value::Array(Rc::new(RefCell::new(data_list))));
                            result.insert("rows".to_string(), Value::Array(Rc::new(RefCell::new(rows_list))));
                            result.insert("model".to_string(), Value::Object(model_class));
                            result.insert("row_count".to_string(), Value::Number(data_rows.len() as f64));
                            return Value::Object(Rc::new(RefCell::new(result)));
                        }
                        Err(e) => {
                            crate::websocket::set_native_error(format!("engine.run select: {}", e));
                            return Value::Null;
                        }
                    }
                }
            }
        }
    }

    // Check if arg is model instance (object we can get table name from, e.g. __class_name) -> INSERT
    if let Value::Object(rc) = &arg_resolved {
        let table_name = get_table_name_from_class_object(&arg).or_else(|| {
            let obj = rc.borrow();
            obj.get("__class_name").and_then(|v| get_string(v))
        });
        if let Some(table_name) = table_name {
            let (sql, params) = build_insert_from_instance(&arg, &table_name);
            match engine_ref.execute(&sql, &params) {
                Ok(count) => return Value::Number(count as f64),
                Err(e) => {
                    crate::websocket::set_native_error(format!("engine.run insert: {}", e));
                    return Value::Null;
                }
            }
        }
    }

    crate::websocket::set_native_error("engine.run: argument must be select(Model), model instance, or metadata.create_all".to_string());
    Value::Null
}

/// Find __tablename__ method in class or ancestor chain (when classes is provided).
/// Uses build_class_chain to walk the full inheritance hierarchy; the chain yields
/// [root, ..., parent, current], so we iterate in reverse to prefer the most specific override.
fn find_tablename_method(
    class_rc: &Rc<RefCell<HashMap<String, Value>>>,
    classes: Option<&HashMap<String, Value>>,
) -> Option<Value> {
    let chain: Vec<Rc<RefCell<HashMap<String, Value>>>> = if let Some(classes) = classes {
        build_class_chain(class_rc, classes)
    } else {
        vec![Rc::clone(class_rc)]
    };
    for ancestor_rc in chain.iter().rev() {
        let obj = ancestor_rc.borrow();
        if let Some(m) = obj.get("__tablename__").cloned() {
            if matches!(&m, Value::Function(_) | Value::ModuleFunction { .. }) {
                return Some(m);
            }
        }
    }
    None
}

/// Resolve function index from Value::Function or Value::ModuleFunction.
fn resolve_function_index(method_val: &Value) -> Option<usize> {
    match method_val {
        Value::Function(idx) => Some(*idx),
        Value::ModuleFunction { module_id, local_index } => {
            VM_CALL_CONTEXT.with(|ctx| {
                let vm_ptr = ctx.borrow();
                vm_ptr.and_then(|ptr| unsafe { (*ptr).get_module_function_index(*module_id, *local_index) })
            })
        }
        _ => None,
    }
}

fn get_table_name_from_class(
    class_obj: &Value,
    classes: Option<&HashMap<String, Value>>,
) -> Option<String> {
    let Value::Object(rc) = class_obj else {
        return None;
    };
    let class_rc = Rc::clone(rc);
    let class_obj_val = Value::Object(class_rc.clone());

    // Try __tablename__ if present
    if let Some(method_val) = find_tablename_method(&class_rc, classes) {
        if let Some(fn_idx) = resolve_function_index(&method_val) {
            let args = [class_obj_val.clone(), class_obj_val];
            if let Ok(result) = call_user_function(fn_idx, &args) {
                if let Value::String(s) = result {
                    return Some(s);
                }
            }
        }
    }

    // Fallback: use __class_name as-is when __tablename__ call fails
    let obj = class_rc.borrow();
    obj.get("__class_name")
        .and_then(|v| if let Value::String(s) = v { Some(s.clone()) } else { None })
}

/// Walk __superclass chain via VM globals until we find a class with metadata; return metadata.classes.
/// Used when insert/select don't have metadata (unlike create_all) so we can build the class chain for __tablename__.
fn get_classes_from_class_chain(class_rc: &Rc<RefCell<HashMap<String, Value>>>) -> Option<HashMap<String, Value>> {
    let mut current_name = class_rc.borrow().get("__superclass").and_then(|v| {
        if let Value::String(s) = v { Some(s.clone()) } else { None }
    })?;
    let mut seen = HashSet::new();
    while seen.insert(current_name.clone()) {
        let parent_val = resolve_global_by_name(&current_name)?;
        let Value::Object(parent_rc) = parent_val else { return None };
        let parent = parent_rc.borrow();
        if let Some(Value::Object(meta_rc)) = parent.get("metadata") {
            if let Some(Value::Object(classes_rc)) = meta_rc.borrow().get("classes") {
                return Some(classes_rc.borrow().clone());
            }
        }
        current_name = match parent.get("__superclass") {
            Some(Value::String(s)) => s.clone(),
            _ => return None,
        };
    }
    None
}

fn get_table_name_from_class_object(instance: &Value) -> Option<String> {
    let Value::Object(rc) = instance else {
        return None;
    };
    let obj = rc.borrow();
    let class_opt = obj.get("__class").cloned();
    if let Some(Value::Object(class_rc)) = class_opt {
        drop(obj);
        let classes = get_classes_from_class_chain(&class_rc);
        return get_table_name_from_class(&Value::Object(class_rc), classes.as_ref());
    }
    obj.get("__class_name")
        .and_then(|v| if let Value::String(s) = v { Some(s.clone()) } else { None })
}

fn build_insert_from_instance(instance: &Value, table_name: &str) -> (String, Vec<Value>) {
    let Value::Object(rc) = instance else { return (String::new(), vec![]) };
    let obj = rc.borrow();
    // Must use __class.__col_names: order and set of columns come from the model, not instance.keys().
    let (col_names, class_opt): (Vec<String>, _) = obj
        .get("__class")
        .and_then(|v| {
            if let Value::Object(class_rc) = v {
                let names = class_rc
                    .borrow()
                    .get("__col_names")
                    .and_then(|a| {
                        if let Value::Array(arr) = a {
                            Some(
                                arr.borrow()
                                    .iter()
                                    .filter_map(|v| get_string(v))
                                    .collect(),
                            )
                        } else {
                            None
                        }
                    })
                    .unwrap_or_default();
                Some((names, Some(Rc::clone(class_rc))))
            } else {
                None
            }
        })
        .unwrap_or_else(|| {
            let class_only: std::collections::HashSet<&str> =
                ["parent", "full_name", "method_names"].into_iter().collect();
            let names: Vec<String> = obj
                .iter()
                .filter(|(k, _)| !k.starts_with("__") && !k.starts_with("new_") && !class_only.contains(k.as_str()))
                .map(|(k, _)| k.clone())
                .collect();
            (names, None)
        });
    drop(obj);
    let obj = rc.borrow();
    let mut cols = Vec::new();
    let mut vals = Vec::new();
    for k in &col_names {
        // Value from instance, else default from class __col_<name>, else Null.
        let val = obj.get(k).cloned().or_else(|| {
            class_opt.as_ref().and_then(|class_rc| {
                let desc_key = format!("__col_{}", k);
                let desc_val = class_rc.borrow().get(&desc_key).cloned();
                // Ref dropped here; then we can borrow desc_rc
                desc_val.and_then(|d| {
                    let Value::Object(desc_rc) = d else { return None };
                    let default_val = desc_rc.borrow().get("default").cloned();
                    default_val
                })
            })
        }).unwrap_or(Value::Null);
        cols.push(k.clone());
        vals.push(val);
    }
    drop(obj);
    let col_list: String = cols.join(", ");
    let placeholders: String = (0..cols.len()).map(|_| "?").collect::<Vec<_>>().join(", ");
    let sql = format!("INSERT INTO {} ({}) VALUES ({})", table_name, col_list, placeholders);
    (sql, vals)
}

/// Build chain of ancestor class objects from root to current (current last), using metadata.classes.
fn build_class_chain(
    class_rc: &Rc<RefCell<HashMap<String, Value>>>,
    classes: &HashMap<String, Value>,
) -> Vec<Rc<RefCell<HashMap<String, Value>>>> {
    let mut chain = Vec::new();
    let mut current: Option<Rc<RefCell<HashMap<String, Value>>>> = Some(Rc::clone(class_rc));
    while let Some(rc) = current.take() {
        let super_name = rc.borrow().get("__superclass").and_then(|v| {
            if let Value::String(s) = v { Some(s.clone()) } else { None }
        });
        if let Some(parent_name) = super_name {
            if let Some(Value::Object(parent_rc)) = classes.get(&parent_name) {
                chain.push(Rc::clone(parent_rc));
                current = Some(Rc::clone(parent_rc));
                continue;
            }
        }
        break;
    }
    chain.reverse();
    chain.push(Rc::clone(class_rc));
    chain
}

/// Collect (col_name, spec) for one class in declaration order if __col_names present, else arbitrary.
fn collect_column_specs_for_class(class_rc: &Rc<RefCell<HashMap<String, Value>>>) -> Vec<(String, String)> {
    let class_obj = class_rc.borrow();
    let mut out = Vec::new();
    let order_names: Option<Vec<String>> = class_obj.get("__col_names").and_then(|v| {
        if let Value::Array(rc) = v {
            Some(rc.borrow().iter().filter_map(|v| get_string(v)).collect())
        } else {
            None
        }
    });
    if let Some(ref names) = order_names {
        for name in names {
            let col_key = format!("__col_{}", name);
            if let Some(attr_val) = class_obj.get(&col_key) {
                if let Value::Object(col_rc) = attr_val {
                    let col = col_rc.borrow();
                    if !col.get("__column").and_then(|v| if let Value::Bool(b) = v { Some(*b) } else { None }).unwrap_or(false) {
                        continue;
                    }
                    let pk = col.get("primary_key").and_then(|v| if let Value::Bool(b) = v { Some(*b) } else { None }).unwrap_or(false);
                    let auto = col.get("autoincrement").and_then(|v| if let Value::Bool(b) = v { Some(*b) } else { None }).unwrap_or(false);
                    let sql_type = if pk && auto {
                        "INTEGER".to_string()
                    } else {
                        column_type_to_sql(attr_val)
                    };
                    let uq = col.get("unique").and_then(|v| if let Value::Bool(b) = v { Some(*b) } else { None }).unwrap_or(false);
                    let null = col.get("nullable").and_then(|v| if let Value::Bool(b) = v { Some(*b) } else { None }).unwrap_or(false);
                    let default_val = col.get("default").cloned();
                    let mut spec = format!("{} {}", name, sql_type);
                    if pk {
                        spec.push_str(" PRIMARY KEY");
                        if auto {
                            spec.push_str(" AUTOINCREMENT");
                        }
                    }
                    if uq {
                        spec.push_str(" UNIQUE");
                    }
                    if !null && !pk {
                        spec.push_str(" NOT NULL");
                    }
                    if let Some(ref dv) = default_val {
                        if let Some(default_sql) = default_value_to_sql(dv) {
                            spec.push_str(" DEFAULT ");
                            spec.push_str(&default_sql);
                        }
                    }
                    out.push((name.clone(), spec));
                }
            }
        }
    } else {
        for (attr_name, attr_val) in class_obj.iter() {
            let col_name = if attr_name.starts_with("__col_") {
                attr_name.strip_prefix("__col_").unwrap_or(attr_name).to_string()
            } else if attr_name.starts_with("__") {
                continue;
            } else {
                attr_name.clone()
            };
            if let Value::Object(col_rc) = attr_val {
                let col = col_rc.borrow();
                if !col.get("__column").and_then(|v| if let Value::Bool(b) = v { Some(*b) } else { None }).unwrap_or(false) {
                    continue;
                }
                let pk = col.get("primary_key").and_then(|v| if let Value::Bool(b) = v { Some(*b) } else { None }).unwrap_or(false);
                let auto = col.get("autoincrement").and_then(|v| if let Value::Bool(b) = v { Some(*b) } else { None }).unwrap_or(false);
                let sql_type = if pk && auto {
                    "INTEGER".to_string()
                } else {
                    column_type_to_sql(attr_val)
                };
                let uq = col.get("unique").and_then(|v| if let Value::Bool(b) = v { Some(*b) } else { None }).unwrap_or(false);
                let null = col.get("nullable").and_then(|v| if let Value::Bool(b) = v { Some(*b) } else { None }).unwrap_or(false);
                let default_val = col.get("default").cloned();
                let mut spec = format!("{} {}", col_name, sql_type);
                if pk {
                    spec.push_str(" PRIMARY KEY");
                    if auto {
                        spec.push_str(" AUTOINCREMENT");
                    }
                }
                if uq {
                    spec.push_str(" UNIQUE");
                }
                if !null && !pk {
                    spec.push_str(" NOT NULL");
                }
                if let Some(ref dv) = default_val {
                    if let Some(default_sql) = default_value_to_sql(dv) {
                        spec.push_str(" DEFAULT ");
                        spec.push_str(&default_sql);
                    }
                }
                out.push((col_name, spec));
            }
        }
    }
    out
}

fn run_create_all(meta_rc: &Rc<RefCell<HashMap<String, Value>>>, engine: &mut std::cell::RefMut<'_, DatabaseEngine>) -> Result<(), String> {
    let meta = meta_rc.borrow();
    let tables = match meta.get("tables") {
        Some(Value::Array(rc)) => rc.borrow().clone(),
        _ => vec![],
    };
    let classes = match meta.get("classes") {
        Some(Value::Object(rc)) => rc.borrow().clone(),
        _ => HashMap::new(),
    };
    drop(meta);
    // Prefer metadata.classes so that all registered models (including subclasses that were not pushed to tables) get a table.
    // Sort by __class_name for deterministic iteration (e.g. Base, RServoDrive, SDKDemo, User).
    let mut model_classes: Vec<Value> = if !classes.is_empty() {
        classes.values().cloned().collect()
    } else {
        tables
    };
    model_classes.sort_by(|a, b| {
        let an = match a {
            Value::Object(r) => r.borrow().get("__class_name").and_then(get_string),
            _ => None,
        };
        let bn = match b {
            Value::Object(r) => r.borrow().get("__class_name").and_then(get_string),
            _ => None,
        };
        an.as_deref().unwrap_or("").cmp(bn.as_deref().unwrap_or(""))
    });
    // Resolve all table names first (before any engine.execute) to avoid VM re-entrancy issues
    let mut to_create: Vec<(Rc<RefCell<HashMap<String, Value>>>, String)> = Vec::new();
    for model_class in model_classes {
        if let Value::Object(class_rc) = model_class {
            let is_abstract = {
                let class_map = class_rc.borrow();
                class_map.get("__abstract").and_then(|v| if let Value::Bool(b) = v { Some(*b) } else { None }).unwrap_or(false)
                    || class_map.get("is_abstract").and_then(|v| if let Value::Bool(b) = v { Some(*b) } else { None }).unwrap_or(false)
            };
            if is_abstract {
                continue;
            }
            let table_name = get_table_name_from_class(&Value::Object(Rc::clone(&class_rc)), Some(&classes));
            if let Some(name) = table_name {
                to_create.push((class_rc, name));
            }
        }
    }
    for (class_rc, name) in to_create {
        let chain = build_class_chain(&class_rc, &classes);
        let mut col_specs = Vec::new();
        let mut seen = HashSet::new();
        // Current class first (declaration order), then parent columns at end
        for ancestor_rc in chain.iter().rev() {
            for (col_name, spec) in collect_column_specs_for_class(ancestor_rc) {
                if seen.insert(col_name) {
                    col_specs.push(spec);
                }
            }
        }
        if col_specs.is_empty() {
            return Err(format!("no column specs for table '{}' (missing __col_* or __col_names on class)", name));
        }
        let cols = col_specs.join(", ");
        let sql = format!("CREATE TABLE IF NOT EXISTS {} ({})", name, cols);
        engine.execute(&sql, &[]).map_err(|e| e.to_string())?;
    }
    Ok(())
}

/// Convert Column default/onupdate value to SQL DEFAULT clause (e.g. 0 -> "0", "user" -> "'user'", now_call -> "CURRENT_TIMESTAMP").
fn default_value_to_sql(v: &Value) -> Option<String> {
    match v {
        Value::Null => None,
        Value::Number(n) => {
            if n.fract() == 0.0 {
                Some(format!("{}", *n as i64))
            } else {
                Some(n.to_string())
            }
        }
        Value::Bool(b) => Some(if *b { "1" } else { "0" }.to_string()),
        Value::String(s) => Some(format!("'{}'", s.replace('\'', "''"))),
        Value::NativeFunction(_) => {
            // now_call() is stored as native fn ref; treat as current timestamp for SQLite
            Some("CURRENT_TIMESTAMP".to_string())
        }
        _ => None,
    }
}

fn column_type_to_sql(col_val: &Value) -> String {
    let Value::Object(rc) = col_val else { return "TEXT".to_string() };
    let col = rc.borrow();
    let type_val = col.get("type").cloned().unwrap_or(Value::Null);
    drop(col);
    if let Value::Object(type_rc) = &type_val {
        let t = type_rc.borrow();
        if t.get("__type").and_then(|v| if let Value::String(s) = v { Some(s.as_str()) } else { None }) == Some("str") {
            let len = t.get("__length").and_then(|v| if let Value::Number(n) = v { Some(*n as i64) } else { None }).unwrap_or(0);
            return if len > 0 { format!("VARCHAR({})", len) } else { "TEXT".to_string() };
        }
        return "TEXT".to_string();
    }
    if let Value::NativeFunction(idx) = type_val {
        if let Some(name) = globals::builtin_global_name(idx) {
            let type_lower = name.to_lowercase();
            return match type_lower.as_str() {
                "int" | "integer" => "INTEGER".to_string(),
                "float" | "real" => "REAL".to_string(),
                "bool" | "boolean" => "INTEGER".to_string(),
                "date" => "DATETIME".to_string(),
                _ => "TEXT".to_string(),
            };
        }
        return "TEXT".to_string();
    }
    if let Value::Null = type_val {
        return "TEXT".to_string();
    }
    let type_str = type_val.to_string();
    let type_lower = type_str.to_lowercase();
    match type_lower.as_str() {
        "int" | "integer" => "INTEGER".to_string(),
        "float" | "real" => "REAL".to_string(),
        "bool" | "boolean" => "INTEGER".to_string(),
        "date" => "DATETIME".to_string(),
        _ => "TEXT".to_string(),
    }
}

/// DatabaseCluster() - creates an empty cluster of named connections.
pub fn native_cluster(_args: &[Value]) -> Value {
    Value::DatabaseCluster(Rc::new(RefCell::new(DatabaseCluster::new())))
}

/// cluster.add(name, engine) or cluster.add(engine) - add named connection. With one arg, name is engine URL or "default".
pub fn native_cluster_add(args: &[Value]) -> Value {
    if args.len() < 2 {
        crate::websocket::set_native_error("cluster.add requires (cluster, name, engine) or (cluster, engine)".to_string());
        return Value::Null;
    }
    let mut cluster_ref = match get_cluster_mut(args) {
        Some(r) => r,
        None => {
            crate::websocket::set_native_error("cluster.add: first argument must be a database cluster".to_string());
            return Value::Null;
        }
    };
    let (name, engine) = if args.len() >= 3 {
        let name = match get_string(&args[1]) {
            Some(s) => s,
            None => {
                crate::websocket::set_native_error("cluster.add: name must be a string".to_string());
                return Value::Null;
            }
        };
        let engine = match &args[2] {
            Value::DatabaseEngine(rc) => Rc::clone(rc),
            _ => {
                crate::websocket::set_native_error("cluster.add: third argument must be a database engine".to_string());
                return Value::Null;
            }
        };
        (name, engine)
    } else {
        let engine = match &args[1] {
            Value::DatabaseEngine(rc) => Rc::clone(rc),
            _ => {
                crate::websocket::set_native_error("cluster.add: second argument must be a database engine".to_string());
                return Value::Null;
            }
        };
        let name = engine.borrow().url.clone();
        (name, engine)
    };
    cluster_ref.add(name, engine);
    Value::Null
}

/// cluster.get(name) - returns engine by name or Null.
pub fn native_cluster_get(args: &[Value]) -> Value {
    if args.len() < 2 {
        crate::websocket::set_native_error("cluster.get requires (cluster, name)".to_string());
        return Value::Null;
    }
    let cluster = match &args[0] {
        Value::DatabaseCluster(rc) => rc.borrow(),
        _ => {
            crate::websocket::set_native_error("cluster.get: first argument must be a database cluster".to_string());
            return Value::Null;
        }
    };
    let name = match get_string(&args[1]) {
        Some(s) => s,
        None => {
            crate::websocket::set_native_error("cluster.get: name must be a string".to_string());
            return Value::Null;
        }
    };
    match cluster.get(&name) {
        Some(engine_rc) => Value::DatabaseEngine(engine_rc),
        None => Value::Null,
    }
}

/// cluster.names() - returns array of connection names.
pub fn native_cluster_names(args: &[Value]) -> Value {
    if args.is_empty() {
        crate::websocket::set_native_error("cluster.names requires cluster".to_string());
        return Value::Null;
    }
    let cluster = match &args[0] {
        Value::DatabaseCluster(rc) => rc.borrow(),
        _ => {
            crate::websocket::set_native_error("cluster.names: first argument must be a database cluster".to_string());
            return Value::Null;
        }
    };
    let names: Vec<Value> = cluster.names().into_iter().map(Value::String).collect();
    Value::Array(Rc::new(RefCell::new(names)))
}
