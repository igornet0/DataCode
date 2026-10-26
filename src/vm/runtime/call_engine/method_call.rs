//! Method call argument preparation: inject @class and drop receiver when needed.

use crate::common::value::Value;
use crate::common::value_store::ValueStore;
use crate::common::TaggedValue;
use crate::vm::heavy_store::HeavyStore;
use crate::vm::store_convert::store_value;

/// Prepare arguments for a method or module-style call:
/// - Inject @class from args[0].__class when the second parameter is declared as @class.
/// - Drop the receiver when function.arity == 0 and the single arg is an Object (module-style call).
pub fn prepare_method_args(
    function_index: usize,
    function: &crate::bytecode::Function,
    args: &mut Vec<Value>,
    arg_tvs: &mut Vec<TaggedValue>,
    value_store: &mut ValueStore,
    heavy_store: &mut HeavyStore,
    vm_ptr: *mut crate::vm::vm::Vm,
) {
    // A `*args` / `**kwargs` method takes any number of arguments; the caller never passes @class.
    let variadic = crate::vm::variadic_bind::function_accepts_variadic(function);
    if function.param_names.get(1).map(|s| s.as_str()) == Some("@class")
        && !args.is_empty()
        && (variadic || args.len() < function.arity)
    {
        let after_inject = args.len() + 1;
        let rest_ok = variadic
            || (after_inject <= function.arity
                && (after_inject == function.arity
                    || crate::vm::call_defaults::trailing_defaults_available(function, after_inject)));
        if rest_ok {
            let this_val = &args[0];
            let class_val = match this_val {
                Value::Object(obj_rc) => obj_rc
                    .borrow()
                    .str_key_get("__class")
                    .cloned()
                    .unwrap_or(Value::Null),
                _ => Value::Null,
            };
            let class_id = store_value(class_val.clone(), value_store, heavy_store);
            let class_tv = TaggedValue::from_heap(class_id);
            args.insert(1, class_val);
            arg_tvs.insert(1, class_tv);
        }
    }

    // `m.f(x)` on an imported `.dc` module: the compiler passes the namespace as a receiver, but
    // module functions are plain functions and never take it. Only a namespace that owns the
    // called function is a receiver; `show(m)` passes a module as an ordinary argument.
    if args
        .first()
        .is_some_and(|a| namespace_owns_function(a, function_index, vm_ptr))
    {
        args.remove(0);
        arg_tvs.remove(0);
        return;
    }

    if function.arity == 0 && args.len() == 1 {
        if let Value::Object(_) = &args[0] {
            args.remove(0);
            arg_tvs.remove(0);
        }
    }
}

/// True when `receiver` is a `.dc` module namespace that exports the function at `function_index`
/// (as `Function(i)` or a `ModuleFunction` resolving to it).
pub(crate) fn namespace_owns_function(
    receiver: &Value,
    function_index: usize,
    vm_ptr: *mut crate::vm::vm::Vm,
) -> bool {
    if !crate::vm::module_object::is_module_namespace(receiver) {
        return false;
    }
    let Value::Object(rc) = receiver else {
        return false;
    };
    let ns = rc.borrow();
    ns.str_key_pairs().into_iter().any(|(_, v)| match v {
        Value::Function(i) => *i == function_index,
        Value::ModuleFunction {
            module_uid,
            local_index,
        } => {
            let resolved =
                unsafe { (*vm_ptr).get_module_function_index(*module_uid, *local_index) };
            resolved == Some(function_index)
        }
        _ => false,
    })
}
