//! Method call argument preparation: inject @class and drop receiver when needed.

use crate::common::value::Value;
use crate::common::value_store::ValueStore;
use crate::vm::store_convert::{store_value};
use crate::vm::heavy_store::HeavyStore;
use crate::common::TaggedValue;

/// Prepare arguments for a method or module-style call:
/// - Inject @class from args[0].__class when the second parameter is declared as @class.
/// - Drop the receiver when function.arity == 0 and the single arg is an Object (module-style call).
pub fn prepare_method_args(
    function: &crate::bytecode::Function,
    args: &mut Vec<Value>,
    arg_tvs: &mut Vec<TaggedValue>,
    value_store: &mut ValueStore,
    heavy_store: &mut HeavyStore,
) {
    if function.param_names.get(1).map(|s| s.as_str()) == Some("@class")
        && args.len() + 1 == function.arity
        && !args.is_empty()
    {
        let this_val = &args[0];
        let class_val = match this_val {
            Value::Object(obj_rc) => obj_rc
                .borrow()
                .get("__class")
                .cloned()
                .unwrap_or(Value::Null),
            _ => Value::Null,
        };
        let class_id = store_value(class_val.clone(), value_store, heavy_store);
        let class_tv = TaggedValue::from_heap(class_id);
        args.insert(1, class_val);
        arg_tvs.insert(1, class_tv);
    }

    if function.arity == 0 && args.len() == 1 {
        if let Value::Object(_) = &args[0] {
            args.remove(0);
            arg_tvs.remove(0);
        }
    }
}
