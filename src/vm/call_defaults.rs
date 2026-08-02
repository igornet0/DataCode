//! Runtime application of trailing default parameter values for under-arity Calls.

use crate::bytecode::Function;
use crate::common::value::Value;
use crate::common::value_store::ValueStore;
use crate::common::TaggedValue;
use crate::vm::heavy_store::HeavyStore;
use crate::vm::store_convert::store_value;

/// True when every parameter slot in `[provided, function.arity)` has a default value.
#[inline]
pub fn trailing_defaults_available(function: &Function, provided: usize) -> bool {
    if provided >= function.arity {
        return false;
    }
    (provided..function.arity).all(|i| {
        function
            .default_values
            .get(i)
            .and_then(|v| v.as_ref())
            .is_some()
    })
}

/// Append default values for parameters `[provided, function.arity)` onto `args` / `arg_tvs`.
/// Caller must ensure [`trailing_defaults_available`] is true.
pub fn append_trailing_defaults(
    function: &Function,
    provided: usize,
    args: &mut Vec<Value>,
    arg_tvs: &mut Vec<TaggedValue>,
    value_store: &mut ValueStore,
    heavy_store: &mut HeavyStore,
) {
    for i in provided..function.arity {
        if let Some(def) = function.default_values.get(i).and_then(|v| v.as_ref()) {
            let id = store_value(def.clone(), value_store, heavy_store);
            arg_tvs.push(TaggedValue::from_heap(id));
            args.push(def.clone());
        }
    }
}

/// Push default values for parameters `[call_arity, total_arity)` onto the VM stack
/// (constructor under-arity path before args are popped).
pub fn push_trailing_defaults_on_stack(
    stack: &mut Vec<TaggedValue>,
    func: &Function,
    call_arity: usize,
    total_arity: usize,
    value_store: &mut ValueStore,
    heavy_store: &mut HeavyStore,
) {
    for i in call_arity..total_arity {
        if let Some(def) = func.default_values.get(i).and_then(|v| v.as_ref()) {
            let id = store_value(def.clone(), value_store, heavy_store);
            crate::vm::stack::push(stack, TaggedValue::from_heap(id));
        }
    }
}
