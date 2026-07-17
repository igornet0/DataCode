//! Shared membership (`in` / `not in`) logic for operator and table filters.

use crate::common::error::{ErrorType, LangError};
use crate::common::numeric::integer_value_as_i64_if_whole;
use crate::common::value::{ObjectKind, Value};
use crate::vm::iterable::membership_via_iterable;
use crate::vm::memory::object_map_try_lookup_by_key_id;
use crate::vm::set_ops::set_contains_value;
use crate::vm::special_methods::{class_has_special, dispatch_special, is_class_instance};
use crate::vm::store_convert::store_value;
use crate::vm::vm::{current_vm_ptr, with_current_stores};

fn special_contains_result(result: Value) -> bool {
    match result {
        Value::Bool(b) => b,
        other => other.is_truthy(),
    }
}

fn object_key_membership(member: &Value, obj: &ObjectKind) -> bool {
    if obj.get_by_value_key(member).is_some() {
        return true;
    }
    if let Value::String(s) = member {
        if obj.str_key_get(s).is_some() {
            return true;
        }
    }
    if let ObjectKind::Bucket(map) = obj {
        if let Some(vm_ptr) = current_vm_ptr() {
            unsafe {
                return (*vm_ptr).with_stores_mut(|store, heap| {
                    let key_id = store_value(member.clone(), store, heap);
                    object_map_try_lookup_by_key_id(map, key_id, store, heap).is_some()
                });
            }
        }
    }
    false
}

/// Returns whether `member` is contained in `container` (same semantics as the `in` operator).
pub fn value_in_container(
    member: &Value,
    container: &Value,
    line: usize,
) -> Result<bool, LangError> {
    let member_canonical = integer_value_as_i64_if_whole(member);

    match container {
        Value::Array(arr) => {
            let found = if let Some(c) = member_canonical {
                arr.borrow().iter().any(|item| {
                    integer_value_as_i64_if_whole(item) == Some(c) || item == member
                })
            } else {
                arr.borrow().iter().any(|item| item == member)
            };
            Ok(found)
        }
        Value::Set(s) => {
            let smap = s.borrow();
            if let Some(c) = member_canonical {
                if smap.contains_integral(c) {
                    return Ok(true);
                }
            }
            Ok(with_current_stores(|store, heap| {
                set_contains_value(&smap, member, store, heap)
            }))
        }
        Value::Object(obj) if is_class_instance(container) => {
            if class_has_special(container, "@contains") {
                if let Ok(Some(result)) =
                    dispatch_special(container, "@contains", &[member.clone()])
                {
                    return Ok(special_contains_result(result));
                }
            }
            Ok(false)
        }
        Value::Object(obj) => Ok(object_key_membership(member, &obj.borrow())),
        _ => {
            let vm_ptr = current_vm_ptr().ok_or_else(|| {
                LangError::runtime_error(
                    "operator 'in' on iterable: VM execution context unavailable".to_string(),
                    line,
                )
            })?;
            unsafe {
                let vm = &mut *vm_ptr;
                match membership_via_iterable(vm, container.clone(), member) {
                    Ok(Some(found)) => Ok(found),
                    Ok(None) => Err(LangError::runtime_error_with_type(
                        "TypeError: operator 'in' supports only arrays, sets, and iterables"
                            .to_string(),
                        line,
                        ErrorType::TypeError,
                    )),
                    Err(e) => Err(e),
                }
            }
        }
    }
}
