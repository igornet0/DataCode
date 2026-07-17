//! Runtime binding of positional/named/spread arguments to function parameters (*args / **kwargs).

use crate::bytecode::Function;
use crate::common::value::{ObjectKind, Value};
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

/// Number of fixed (non-variadic) parameters before `*args` or `**kwargs`.
pub fn fixed_param_count(function: &Function) -> usize {
    match (
        function.variadic_pos_index,
        function.variadic_kw_index,
    ) {
        (Some(i), _) | (None, Some(i)) => i,
        (None, None) => function.arity,
    }
}

fn empty_array() -> Value {
    Value::Array(Rc::new(RefCell::new(Vec::new())))
}

fn empty_object() -> Value {
    Value::Object(Rc::new(RefCell::new(ObjectKind::legacy(HashMap::new()))))
}

fn object_str_entries(obj: &Value) -> Result<Vec<(String, Value)>, String> {
    let Value::Object(rc) = obj else {
        return Err("expected object".to_string());
    };
    Ok(rc.borrow().str_key_entries_cloned())
}

fn merge_entries_into(target: &mut Value, entries: Vec<(String, Value)>) -> Result<(), String> {
    let Value::Object(rc) = target else {
        return Err("expected object".to_string());
    };
    let mut map = rc.borrow_mut();
    let Some(legacy) = map.legacy_mut() else {
        return Err("kwargs object must use string keys".to_string());
    };
    for (k, v) in entries {
        if legacy.contains_key(&k) {
            return Err(format!("got multiple values for argument '{}'", k));
        }
        legacy.insert(k, v);
    }
    Ok(())
}

/// Bind call-site arguments to `function.arity` slot values (including *args/**kwargs slots).
pub fn bind_function_args(
    function: &Function,
    mut positional: Vec<Value>,
    mut named: HashMap<String, Value>,
    star_objects: &[Value],
    starstar_objects: &[Value],
) -> Result<Vec<Value>, String> {
    for arr in star_objects {
        let Value::Array(rc) = arr else {
            return Err("* unpacking requires an array".to_string());
        };
        positional.extend(rc.borrow().iter().cloned());
    }

    for obj in starstar_objects {
        let entries = object_str_entries(obj).map_err(|_| {
            "** unpacking requires an object with string keys".to_string()
        })?;
        for (k, v) in entries {
            if named.contains_key(&k) {
                return Err(format!("got multiple values for argument '{}'", k));
            }
            named.insert(k, v);
        }
    }

    let fixed_n = fixed_param_count(function);
    let mut result = vec![Value::Null; function.arity];
    let mut filled = vec![false; function.arity];

    // Named → fixed parameters
    for (i, name) in function.param_names.iter().enumerate().take(fixed_n) {
        if let Some(v) = named.remove(name) {
            if filled[i] {
                return Err(format!(
                    "Function '{}' got multiple values for argument '{}'",
                    function.name, name
                ));
            }
            result[i] = v;
            filled[i] = true;
        }
    }

    if function.variadic_kw_index.is_none() && !named.is_empty() {
        let unknown = named.keys().next().cloned().unwrap_or_default();
        return Err(format!(
            "Function '{}' got an unexpected keyword argument '{}'",
            function.name, unknown
        ));
    }

    // Positional → unfilled fixed slots
    let mut pos_idx = 0;
    for i in 0..fixed_n {
        if filled[i] {
            continue;
        }
        if pos_idx < positional.len() {
            result[i] = positional[pos_idx].clone();
            filled[i] = true;
            pos_idx += 1;
        } else if let Some(default) = function.default_values.get(i).and_then(|d| d.as_ref()) {
            result[i] = default.clone();
            filled[i] = true;
        } else {
            return Err(format!(
                "Function '{}' missing required argument '{}'",
                function.name, function.param_names[i]
            ));
        }
    }

    let extra_pos: Vec<Value> = positional[pos_idx..].to_vec();

    if let Some(var_i) = function.variadic_pos_index {
        result[var_i] = if extra_pos.is_empty() {
            empty_array()
        } else {
            Value::Array(Rc::new(RefCell::new(extra_pos)))
        };
    } else if !extra_pos.is_empty() {
        return Err(format!(
            "Function '{}' takes at most {} positional arguments but {} were given",
            function.name,
            fixed_n,
            positional.len()
        ));
    }

    if let Some(kw_i) = function.variadic_kw_index {
        let mut kw_obj = empty_object();
        for (k, v) in named {
            if function.param_names[..fixed_n].contains(&k) {
                return Err(format!(
                    "Function '{}' got an unexpected keyword argument '{}'",
                    function.name, k
                ));
            }
            merge_entries_into(&mut kw_obj, vec![(k, v)])?;
        }
        result[kw_i] = kw_obj;
    } else if !named.is_empty() {
        let unknown = named.keys().next().cloned().unwrap_or_default();
        return Err(format!(
            "Function '{}' got an unexpected keyword argument '{}'",
            function.name, unknown
        ));
    }

    Ok(result)
}

/// Whether the function accepts variadic arguments at call site.
pub fn function_accepts_variadic(function: &Function) -> bool {
    function.variadic_pos_index.is_some() || function.variadic_kw_index.is_some()
}

/// Bind arguments for a native with optional `**kwargs` collector parameter.
pub fn bind_native_varkw_args(
    param_names: &[String],
    varkw_name: Option<&str>,
    positional: Vec<Value>,
    named: HashMap<String, Value>,
    star_objects: &[Value],
    starstar_objects: &[Value],
) -> Result<Vec<Value>, String> {
    let mut positional = positional;
    let mut named = named;

    for arr in star_objects {
        let Value::Array(rc) = arr else {
            return Err("* unpacking requires an array".to_string());
        };
        positional.extend(rc.borrow().iter().cloned());
    }
    for obj in starstar_objects {
        for (k, v) in object_str_entries(obj)? {
            if named.contains_key(&k) {
                return Err(format!("got multiple values for argument '{}'", k));
            }
            named.insert(k, v);
        }
    }

    let fixed_n = param_names.len();
    let mut result = vec![Value::Null; fixed_n + if varkw_name.is_some() { 1 } else { 0 }];

    for (i, name) in param_names.iter().enumerate() {
        if let Some(v) = named.remove(name) {
            if i < positional.len() {
                return Err(format!("got multiple values for argument '{}'", name));
            }
            result[i] = v;
        } else if i < positional.len() {
            result[i] = positional[i].clone();
        }
    }

    // Fill remaining fixed from positional
    for i in 0..fixed_n {
        if result[i] == Value::Null && i < positional.len() {
            result[i] = positional[i].clone();
        }
    }

    if varkw_name.is_some() {
        let mut kw = empty_object();
        for (k, v) in named {
            if param_names.contains(&k) {
                return Err(format!("got an unexpected keyword argument '{}'", k));
            }
            merge_entries_into(&mut kw, vec![(k, v)])?;
        }
        *result.last_mut().unwrap() = kw;
    } else if !named.is_empty() {
        let k = named.keys().next().unwrap();
        return Err(format!("got an unexpected keyword argument '{}'", k));
    }

    Ok(result)
}
