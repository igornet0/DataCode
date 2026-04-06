//! `Call(arity)` opcode: user functions, natives, constructors, method dispatch.

use super::super::closure_call;
use super::dispatch_arms;
use super::resolve_callee::{resolve_call_callee, CalleeResolveOutcome};
use crate::common::{
    error::LangError,
    value::Value,
    value_store::{ValueId, ValueStore},
    TaggedValue,
};
use crate::vm::exceptions::ExceptionHandler;
use crate::vm::frame::CallFrame;
use crate::vm::global_slot::GlobalSlot;
use crate::vm::heavy_store::HeavyStore;
use crate::vm::host::HostEntry;
use crate::vm::stack;
use crate::vm::types::VMStatus;
use crate::vm::types::{ExplicitPrimaryKey, ExplicitRelation};

/// Execute Call(arity): user functions, natives, class constructors, method dispatch.
#[allow(clippy::too_many_arguments)]
pub(crate) fn execute_call(
    arity: usize,
    line: usize,
    stack: &mut Vec<TaggedValue>,
    frames: &mut Vec<CallFrame>,
    globals: &mut Vec<GlobalSlot>,
    global_names: &mut std::collections::BTreeMap<usize, String>,
    explicit_global_names: &std::collections::BTreeMap<usize, String>,
    functions: &mut Vec<crate::bytecode::Function>,
    natives: &[HostEntry],
    exception_handlers: &mut Vec<ExceptionHandler>,
    error_type_table: &mut Vec<String>,
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
    let callee_tv = stack::pop(stack, frames, exception_handlers, value_store, heavy_store)?;
    let resolved = resolve_call_callee(
        arity,
        line,
        callee_tv,
        stack,
        frames,
        globals,
        global_names,
        functions,
        exception_handlers,
        value_store,
        heavy_store,
        vm_ptr,
    )?;
    match resolved {
        CalleeResolveOutcome::EarlyReturn(status) => Ok(status),
        CalleeResolveOutcome::Resolved(r) => {
            if r.function_index_resolved.is_some() {
                return closure_call::execute_closure_call(
                    r.current_ip,
                    r.function_index_final,
                    r.constructing_class_opt.clone(),
                    arity,
                    line,
                    stack,
                    frames,
                    globals,
                    global_names,
                    functions,
                    exception_handlers,
                    error_type_table,
                    value_store,
                    heavy_store,
                    vm_ptr,
                );
            }
            dispatch_arms::dispatch_call_arms(
                r.actual_callee,
                arity,
                line,
                stack,
                frames,
                globals,
                explicit_global_names,
                natives,
                exception_handlers,
                explicit_relations,
                explicit_primary_keys,
                abi_natives,
                value_store,
                heavy_store,
                native_args_buffer,
                reusable_native_arg_ids,
                reusable_all_popped,
                vm_ptr,
            )
        }
    }
}
