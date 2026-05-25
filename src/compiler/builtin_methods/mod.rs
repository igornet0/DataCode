//! Intrinsic zero-arg method dispatch for [`crate::compiler::expr::method_call::compile_module_method`].
//!
//! Some built-in "methods" are really property access via [`crate::bytecode::OpCode::GetArrayElement`]
//! (e.g. `d.year()`, `o.keys()`). They are grouped here by **receiver family** (not user `class` types).

mod date_fields;
mod plain_dict;

/// Built-in category aligned with VM/runtime semantics (documentation / future inference).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReceiverFamily {
    DateLike,
    PlainDictLike,
}

/// How to compile `receiver.method()` when `args` are empty.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ZeroArgDispatch {
    /// Same as `receiver.method` — `GetArrayElement` only, no `Call`.
    PropertyViaGetArrayElement(ReceiverFamily),
}

/// If `method` is a known zero-arg property name for some intrinsic family, return its dispatch.
/// Names are disjoint across families today; if they ever collide, add receiver inference.
pub fn zero_arg_dispatch_for_method(method: &str) -> Option<ZeroArgDispatch> {
    if date_fields::METHODS.contains(&method) {
        return Some(ZeroArgDispatch::PropertyViaGetArrayElement(
            ReceiverFamily::DateLike,
        ));
    }
    if plain_dict::METHODS.contains(&method) {
        return Some(ZeroArgDispatch::PropertyViaGetArrayElement(
            ReceiverFamily::PlainDictLike,
        ));
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn date_year_is_property_shortcut() {
        assert!(matches!(
            zero_arg_dispatch_for_method("year"),
            Some(ZeroArgDispatch::PropertyViaGetArrayElement(
                ReceiverFamily::DateLike
            ))
        ));
    }

    #[test]
    fn dict_keys_values_are_property_shortcut() {
        for m in ["keys", "values"] {
            assert!(matches!(
                zero_arg_dispatch_for_method(m),
                Some(ZeroArgDispatch::PropertyViaGetArrayElement(
                    ReceiverFamily::PlainDictLike
                ))
            ));
        }
    }

    #[test]
    fn unknown_method_not_shortcut() {
        assert!(zero_arg_dispatch_for_method("no_such_builtin_prop").is_none());
    }
}
