//! Dispatch arms for `OpCode::Call` after callee resolution: native, plugin opaque, invalid callee.

use crate::common::{
    error::LangError,
    value::Value,
    value_store::{ValueId, ValueStore, NULL_VALUE_ID},
    TaggedValue,
};
use crate::vm::exceptions::ExceptionHandler;
use crate::vm::frame::CallFrame;
use crate::vm::global_slot::GlobalSlot;
use crate::vm::heavy_store::HeavyStore;
use crate::vm::host::HostEntry;
use crate::vm::interpreter::helpers::pop_to_value_id;
use crate::vm::stack;
use crate::vm::store_convert::{load_value, store_value};
use crate::vm::types::VMStatus;
use crate::vm::types::{ExplicitPrimaryKey, ExplicitRelation};

#[allow(clippy::too_many_arguments)]
pub(super) fn dispatch_call_arms(
    actual_callee: Value,
    arity: usize,
    line: usize,
    stack: &mut Vec<TaggedValue>,
    frames: &mut Vec<CallFrame>,
    globals: &mut Vec<GlobalSlot>,
    explicit_global_names: &std::collections::BTreeMap<usize, String>,
    natives: &[HostEntry],
    exception_handlers: &mut Vec<ExceptionHandler>,
    explicit_relations: &mut Vec<ExplicitRelation>,
    explicit_primary_keys: &mut Vec<ExplicitPrimaryKey>,
    abi_natives: &mut Vec<crate::abi::NativeAbiFn>,
    value_store: &mut ValueStore,
    heavy_store: &mut HeavyStore,
    native_args_buffer: &mut Vec<Value>,
    reusable_native_arg_ids: &mut Vec<ValueId>,
    reusable_all_popped: &mut Vec<Value>,
    vm_ptr: *mut crate::vm::vm::Vm,
) -> Result<VMStatus, LangError> {
    match actual_callee {
        Value::NativeFunction(native_index) => {
            super::super::native_call::execute_native_call(
                native_index,
                arity,
                line,
                stack,
                frames,
                natives,
                exception_handlers,
                value_store,
                heavy_store,
                native_args_buffer,
                reusable_native_arg_ids,
                reusable_all_popped,
                abi_natives,
                explicit_relations,
                explicit_primary_keys,
                globals,
                explicit_global_names,
                vm_ptr,
            )
        }
        Value::PluginOpaque { .. } => {
            let Some(native_idx) = (unsafe { (*vm_ptr).plugin_call_native }) else {
                let error = ExceptionHandler::runtime_error(
                    &frames,
                    "Plugin opaque call requires a native module that exports native_plugin_call"
                        .to_string(),
                    line,
                );
                match ExceptionHandler::handle_exception(
                    stack,
                    frames,
                    exception_handlers,
                    error,
                    value_store,
                    heavy_store,
                ) {
                    Ok(()) => {
                        stack::push_id(stack, NULL_VALUE_ID);
                        return Ok(VMStatus::Continue);
                    }
                    Err(e) => return Err(e),
                }
            };
            let builtin_count = natives.len();
            if native_idx < builtin_count || native_idx >= builtin_count + abi_natives.len() {
                let error = ExceptionHandler::runtime_error(
                    &frames,
                    "native_plugin_call index is invalid (reload native module)".to_string(),
                    line,
                );
                match ExceptionHandler::handle_exception(
                    stack,
                    frames,
                    exception_handlers,
                    error,
                    value_store,
                    heavy_store,
                ) {
                    Ok(()) => {
                        stack::push_id(stack, NULL_VALUE_ID);
                        return Ok(VMStatus::Continue);
                    }
                    Err(e) => return Err(e),
                }
            }
            let frame = frames.last().unwrap();
            let available_args = crate::vm::stack::available_in_frame(stack, frame.stack_start);
            if available_args < arity {
                let error = ExceptionHandler::runtime_error(
                    &frames,
                    format!(
                        "Not enough arguments for plugin opaque call: expected {} but got {}",
                        arity, available_args
                    ),
                    line,
                );
                match ExceptionHandler::handle_exception(
                    stack,
                    frames,
                    exception_handlers,
                    error,
                    value_store,
                    heavy_store,
                ) {
                    Ok(()) => {
                        stack::push_id(stack, NULL_VALUE_ID);
                        return Ok(VMStatus::Continue);
                    }
                    Err(e) => return Err(e),
                }
            }
            let mut call_args: Vec<Value> = Vec::with_capacity(arity);
            for _ in 0..arity {
                let vid =
                    pop_to_value_id(stack, frames, exception_handlers, value_store, heavy_store)?;
                call_args.push(load_value(vid, value_store, heavy_store));
            }
            call_args.reverse();
            let mut args = Vec::with_capacity(1 + call_args.len());
            args.push(actual_callee.clone());
            args.extend(call_args);
            let result = crate::vm::native_loader::call_abi_native(
                abi_natives[native_idx - builtin_count],
                &args,
                Some((value_store, heavy_store)),
            );
            if let Some(abi_err) = crate::vm::native_loader::take_last_abi_error() {
                match ExceptionHandler::handle_exception(
                    stack,
                    frames,
                    exception_handlers,
                    abi_err,
                    value_store,
                    heavy_store,
                ) {
                    Ok(()) => {
                        stack::push_id(stack, NULL_VALUE_ID);
                        return Ok(VMStatus::Continue);
                    }
                    Err(e) => return Err(e),
                }
            }
            stack::push_id(stack, store_value(result, value_store, heavy_store));
            Ok(VMStatus::Continue)
        }
        _ => {
            let callee_type = match &actual_callee {
                Value::Null => "Null",
                Value::Array(_) => "Array",
                Value::Object(obj_rc) => {
                    let obj = obj_rc.borrow();
                    if obj.str_key_get("__class_name").is_some() {
                        "Object(class)"
                    } else {
                        "Object"
                    }
                }
                Value::Function(_) => "Function",
                Value::NativeFunction(_) => "NativeFunction",
                _ => "Other",
            };
            let frame_name = frames
                .last()
                .map(|f| f.function.name.as_str())
                .unwrap_or("?");
            if crate::common::debug::verbose_constructor_debug() {
                eprintln!(
                    "[Call error] callee type={}, line={}, function={}",
                    callee_type, line, frame_name
                );
                if matches!(&actual_callee, Value::Array(_)) {
                    if let Some(frame) = frames.last() {
                        let prev_ip = frame.ip.saturating_sub(2);
                        if let Some(crate::bytecode::OpCode::LoadGlobal(idx)) =
                            frame.function.chunk.code.get(prev_ip)
                        {
                            let name = frame
                                .function
                                .chunk
                                .global_names
                                .get(idx)
                                .map(|s| s.as_str())
                                .unwrap_or("?");
                            eprintln!("[Call error] previous instruction at IP {}: LoadGlobal({}) name={}", prev_ip, idx, name);
                        }
                    }
                }
            }
            let error_msg = match &actual_callee {
                Value::Null => {
                    "Cannot call null - function may not be imported or defined".to_string()
                }
                Value::Object(obj_rc) => {
                    let obj = obj_rc.borrow();
                    if let Some(Value::String(class_name)) = obj.str_key_get("__class_name") {
                        format!("Class '{}' cannot accept {} argument(s)", class_name, arity)
                    } else {
                        format!(
                            "Can only call functions, got: {:?}",
                            std::mem::discriminant(&actual_callee)
                        )
                    }
                }
                _ => format!(
                    "Can only call functions, got: {:?}",
                    std::mem::discriminant(&actual_callee)
                ),
            };
            let error = ExceptionHandler::runtime_error(&frames, error_msg, line);
            match ExceptionHandler::handle_exception(
                stack,
                frames,
                exception_handlers,
                error,
                value_store,
                heavy_store,
            ) {
                Ok(()) => {
                    stack::push_id(stack, NULL_VALUE_ID);
                    Ok(VMStatus::Continue)
                }
                Err(e) => Err(e),
            }
        }
    }
}
