//! Registry and compile-time validation for class special methods (`@add`, `@string`, …).

use crate::common::error::LangError;
use crate::parser::ast::{Method, Param, TypePart};

/// Reserved special method base names (without `@`). `drop` is reserved but not wired in runtime yet.
pub const SPECIAL_METHOD_NAMES: &[&str] = &[
    "string", "len", "add", "sub", "mul", "div", "mod", "pow", "eq", "neq", "lt", "lte", "gt",
    "gte", "get", "set", "contains", "call", "iter", "next", "clone", "hash", "init", "drop",
];

pub fn special_method_key(base: &str) -> String {
    format!("@{base}")
}

pub fn is_reserved_special_base(base: &str) -> bool {
    SPECIAL_METHOD_NAMES.contains(&base)
}

pub fn is_special_method_name(name: &str) -> bool {
    name.starts_with('@')
        && name.len() > 1
        && is_reserved_special_base(&name[1..])
}

pub fn special_method_mangled_suffix(name: &str) -> Option<String> {
    if !is_special_method_name(name) {
        return None;
    }
    Some(name[1..].to_string())
}

fn user_params(method: &Method) -> Vec<&Param> {
    method
        .params
        .iter()
        .filter(|p| p.name != "@class")
        .collect()
}

fn type_parts_is(name: &str, ty: Option<&Vec<TypePart>>) -> bool {
    ty.is_some_and(|parts| {
        parts.len() == 1 && matches!(&parts[0], TypePart::TypeName(s) if s == name)
    })
}

fn type_parts_is_bool_or_logic(ty: Option<&Vec<TypePart>>) -> bool {
    type_parts_is("bool", ty) || type_parts_is("logic", ty)
}

fn validate_return_type(method: &Method, expected: &str, line: usize) -> Result<(), LangError> {
    if method.return_type.is_some() && !type_parts_is(expected, method.return_type.as_ref()) {
        return Err(LangError::ParseError {
            message: format!(
                "`{}` must return `{}` when a return type is specified",
                method.name, expected
            ),
            line,
            file: None,
        });
    }
    Ok(())
}

pub fn validate_special_method_signature(method: &Method, class_name: &str) -> Result<(), LangError> {
    let line = method.line;
    if !is_special_method_name(&method.name) {
        return Err(LangError::ParseError {
            message: format!(
                "Unknown special method `{}` in class `{}`",
                method.name, class_name
            ),
            line,
            file: None,
        });
    }

    let params = user_params(method);
    let base = &method.name[1..];

    match base {
        "string" => {
            if !params.is_empty() {
                return Err(LangError::ParseError {
                    message: "`@string` must take 0 arguments".to_string(),
                    line,
                    file: None,
                });
            }
            validate_return_type(method, "str", line)?;
        }
        "len" => {
            if !params.is_empty() {
                return Err(LangError::ParseError {
                    message: "`@len` must take 0 arguments".to_string(),
                    line,
                    file: None,
                });
            }
            validate_return_type(method, "int", line)?;
        }
        "add" | "sub" | "mul" | "div" | "mod" | "pow" => {
            if params.len() != 1 {
                return Err(LangError::ParseError {
                    message: format!("`{}` expects 1 parameter", method.name),
                    line,
                    file: None,
                });
            }
        }
        "eq" | "neq" | "lt" | "lte" | "gt" | "gte" | "contains" => {
            if params.len() != 1 {
                return Err(LangError::ParseError {
                    message: format!("`{}` expects 1 parameter", method.name),
                    line,
                    file: None,
                });
            }
            if method.return_type.is_some() && !type_parts_is_bool_or_logic(method.return_type.as_ref())
            {
                return Err(LangError::ParseError {
                    message: format!("`{}` must return `bool` or `logic`", method.name),
                    line,
                    file: None,
                });
            }
        }
        "get" => {
            if params.len() != 1 {
                return Err(LangError::ParseError {
                    message: "`@get` expects 1 parameter".to_string(),
                    line,
                    file: None,
                });
            }
        }
        "set" => {
            if params.len() != 2 {
                return Err(LangError::ParseError {
                    message: "`@set` expects 2 parameters".to_string(),
                    line,
                    file: None,
                });
            }
            if method.return_type.is_some() && !type_parts_is("null", method.return_type.as_ref()) {
                return Err(LangError::ParseError {
                    message: "`@set` must return `null` when a return type is specified".to_string(),
                    line,
                    file: None,
                });
            }
        }
        "iter" | "next" | "clone" | "hash" => {
            if !params.is_empty() {
                return Err(LangError::ParseError {
                    message: format!("`{}` must take 0 arguments", method.name),
                    line,
                    file: None,
                });
            }
            if base == "hash" {
                validate_return_type(method, "int", line)?;
            }
        }
        "call" => {
            // variadic — any arity OK
        }
        "init" => {
            // validated separately against each constructor
        }
        "drop" => {
            return Err(LangError::ParseError {
                message: "`@drop` is reserved but not implemented yet".to_string(),
                line,
                file: None,
            });
        }
        _ => {}
    }
    Ok(())
}

pub fn validate_init_matches_constructor(
    init: &Method,
    ctor_params: &[Param],
    line: usize,
) -> Result<(), LangError> {
    let init_user = user_params(init);
    if init_user.len() != ctor_params.len() {
        return Err(LangError::ParseError {
            message: format!(
                "`@init` parameter count ({}) must match constructor parameter count ({})",
                init_user.len(),
                ctor_params.len()
            ),
            line,
            file: None,
        });
    }
    for (i, (a, b)) in init_user.iter().zip(ctor_params.iter()).enumerate() {
        if a.name != b.name {
            return Err(LangError::ParseError {
                message: format!(
                    "`@init` parameter {} name `{}` must match constructor parameter `{}`",
                    i + 1,
                    a.name,
                    b.name
                ),
                line,
                file: None,
            });
        }
    }
    if init.return_type.is_some() && !type_parts_is("null", init.return_type.as_ref()) {
        return Err(LangError::ParseError {
            message: "`@init` must return `null` when a return type is specified".to_string(),
            line,
            file: None,
        });
    }
    Ok(())
}
