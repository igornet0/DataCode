//! Shared helpers for typed constructor overload resolution (compile-time and runtime).

/// Match an inferred primitive type name to a constructor suffix (`array` → `array[int]`).
pub fn ctor_suffix_matches_inferred(ctor_suffix: &str, inferred: &str) -> bool {
    if ctor_suffix == inferred {
        return true;
    }
    if inferred == "array" && (ctor_suffix == "array" || ctor_suffix.starts_with("array[")) {
        return true;
    }
    if inferred == "object"
        && (ctor_suffix == "object"
            || ctor_suffix == "dict"
            || ctor_suffix.starts_with("dict["))
    {
        return true;
    }
    false
}

/// Whether `ctor_name` is `Class::new_{arity}_{type...}` (typed overload, not bare `new_{arity}`).
pub fn typed_constructor_type_suffix(class_name: &str, ctor_name: &str, arity: usize) -> Option<String> {
    let prefix = format!("{}::new_{}_", class_name, arity);
    ctor_name.strip_prefix(&prefix).map(|rest| rest.to_string())
}

pub fn is_typed_constructor(class_name: &str, ctor_name: &str, arity: usize) -> bool {
    typed_constructor_type_suffix(class_name, ctor_name, arity).is_some()
}
