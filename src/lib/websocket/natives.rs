//! Built-in `websocket` module for ws_app.dc configuration.

use crate::common::value::Value;
use crate::websocket::config::with_bootstrap_config;
use crate::vm::PermissionPolicy;
use crate::websocket::set_native_error;

fn arg_object_first(args: &[Value], fn_name: &str) -> Option<std::rc::Rc<std::cell::RefCell<crate::common::value::ObjectKind>>> {
    match args.first() {
        Some(Value::Object(rc)) => Some(rc.clone()),
        _ => {
            set_native_error(format!(
                "TypeError: {}() requires an options object",
                fn_name
            ));
            None
        }
    }
}

fn arg_string(args: &[Value], fn_name: &str) -> Option<String> {
    match args.first() {
        Some(Value::String(s)) => Some(s.clone()),
        _ => {
            set_native_error(format!(
                "TypeError: {}() argument must be a string",
                fn_name
            ));
            None
        }
    }
}

/// `websocket.configure({ "execute_policy": "restricted" | "allow_all" })`
pub fn native_websocket_configure(args: &[Value]) -> Value {
    let Some(obj_rc) = arg_object_first(args, "configure") else {
        return Value::Null;
    };
    let obj = obj_rc.borrow();
    with_bootstrap_config(|config| {
        if let Some(policy_val) = obj.str_key_get("execute_policy") {
            let policy = match policy_val {
                Value::String(s) if s == "restricted" => PermissionPolicy::Restricted,
                Value::String(s) if s == "allow_all" => PermissionPolicy::AllowAll,
                Value::String(s) => {
                    set_native_error(format!(
                        "ValueError: unknown execute_policy '{}', use 'allow_all' or 'restricted'",
                        s
                    ));
                    return;
                }
                _ => {
                    set_native_error(
                        "TypeError: execute_policy must be a string".to_string(),
                    );
                    return;
                }
            };
            config.execute_permission_policy = policy;
        }
    });
    Value::Null
}

/// `websocket.disable_builtin("upload_file")`
pub fn native_websocket_disable_builtin(args: &[Value]) -> Value {
    let Some(name) = arg_string(args, "disable_builtin") else {
        return Value::Null;
    };
    with_bootstrap_config(|config| {
        config.disabled_builtins.insert(name);
    });
    Value::Null
}

/// `websocket.enable_builtin("upload_file")`
pub fn native_websocket_enable_builtin(args: &[Value]) -> Value {
    let Some(name) = arg_string(args, "enable_builtin") else {
        return Value::Null;
    };
    with_bootstrap_config(|config| {
        config.disabled_builtins.remove(&name);
    });
    Value::Null
}
