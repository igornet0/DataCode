//! Execution of native (builtin and ABI) function calls.

use crate::common::error::ErrorType;
use crate::common::table::Table;
use crate::common::{
    error::LangError,
    value::{ObjectKind, Value},
    value_store::{ValueCell, ValueId, ValueStore, NULL_VALUE_ID},
    TaggedValue,
};
use crate::debug_println;
use crate::vm::exceptions::ExceptionHandler;
use crate::vm::frame::CallFrame;
use crate::vm::global_slot::GlobalSlot;
use crate::vm::native_indices::builtin;
use crate::vm::heavy_store::HeavyStore;
use crate::vm::host::HostEntry;
use crate::vm::stack;
use crate::vm::memory::{canonical_integral_from_key_id, push_stack_value_id, slot_to_value};
use crate::vm::store_convert::{
    load_value, store_value, tagged_to_value_id,
    update_cell_if_mutable,
};
use crate::vm::types::VMStatus;
use crate::vm::types::{ExplicitPrimaryKey, ExplicitRelation};
use crate::vm::vm::{VmExecutionContext, VM_CALL_CONTEXT};
use super::fast_paths;
use std::cell::RefCell;
use std::rc::Rc;

/// Restores VM_CALL_CONTEXT to its previous value on drop.
/// Allows nested native calls (e.g. __tablename__ calling enum()) without losing context.
struct RestoreVmContextGuard {
    previous: Option<VmExecutionContext>,
}

impl Drop for RestoreVmContextGuard {
    fn drop(&mut self) {
        VM_CALL_CONTEXT.with(|ctx| {
            *ctx.borrow_mut() = self.previous;
        });
    }
}

fn object_looks_like_module_receiver(map: &ObjectKind) -> bool {
    fn native_export_count(map: &ObjectKind) -> usize {
        let count_native = |v: &Value| matches!(v, Value::NativeFunction(_));
        match map {
            ObjectKind::Legacy(hm) => hm.values().filter(|v| count_native(v)).count(),
            ObjectKind::Inline(pairs) => pairs.iter().filter(|(_, v)| count_native(v)).count(),
            _ => 0,
        }
    }
    native_export_count(map) >= 1
}

/// Execute a native (builtin or ABI) call. Called from call_dispatch when callee is Value::NativeFunction(native_index).
#[allow(clippy::too_many_arguments)]
pub(crate) fn execute_native_call(
    native_index: usize,
    arity: usize,
    line: usize,
    stack: &mut Vec<TaggedValue>,
    frames: &mut Vec<CallFrame>,
    natives: &[HostEntry],
    exception_handlers: &mut Vec<ExceptionHandler>,
    value_store: &mut ValueStore,
    heavy_store: &mut HeavyStore,
    native_args_buffer: &mut Vec<Value>,
    reusable_native_arg_ids: &mut Vec<ValueId>,
    reusable_all_popped: &mut Vec<Value>,
    abi_natives: &mut Vec<crate::abi::NativeAbiFn>,
    explicit_relations: &mut Vec<ExplicitRelation>,
    explicit_primary_keys: &mut Vec<ExplicitPrimaryKey>,
    globals: &mut Vec<GlobalSlot>,
    explicit_global_names: &std::collections::BTreeMap<usize, String>,
    vm_ptr: *mut crate::vm::vm::Vm,
) -> Result<VMStatus, LangError> {
    let builtin_count = natives.len();
    if native_index >= builtin_count + abi_natives.len() {
        let error = ExceptionHandler::runtime_error(
            &frames,
            format!("Native function index {} out of bounds", native_index),
            line,
        );
        return ExceptionHandler::handle_exception_vm(
            stack,
            frames,
            exception_handlers,
            error,
            value_store,
            heavy_store,
        );
    }

    if let Some(s) = fast_paths::try_range_fast_path(
        native_index,
        arity,
        line,
        stack,
        frames,
        exception_handlers,
        value_store,
        heavy_store,
    )? {
        return Ok(s);
    }
    if let Some(s) = fast_paths::try_push_fast_path(
        native_index,
        arity,
        stack,
        frames,
        value_store,
    ) {
        return Ok(s);
    }
    if let Some(s) = fast_paths::try_pop_fast_path(
        native_index,
        arity,
        stack,
        frames,
        value_store,
    ) {
        return Ok(s);
    }
    if let Some(s) = fast_paths::try_len_fast_path(
        native_index,
        arity,
        stack,
        frames,
        value_store,
        heavy_store,
    ) {
        return Ok(s);
    }
    if let Some(s) = fast_paths::try_cast_typeof_fast_path(
        native_index,
        arity,
        stack,
        frames,
        value_store,
        heavy_store,
    ) {
        return Ok(s);
    }
    if let Some(s) = fast_paths::try_abs_fast_path(
        native_index,
        arity,
        stack,
        frames,
        value_store,
    ) {
        return Ok(s);
    }
    if let Some(s) = fast_paths::try_table_legacy_fast_path(
        native_index,
        arity,
        stack,
        frames,
        value_store,
        heavy_store,
    ) {
        return Ok(s);
    }

    let native_ptr = natives.get(native_index).and_then(HostEntry::as_fn_ptr);
    if let Some(r) = super::heapq_fast::try_heapq_fast_path(
        native_ptr,
        arity,
        line,
        stack,
        frames,
        exception_handlers,
        value_store,
        heavy_store,
    ) {
        return r;
    }

    if let Some(r) = super::object_get_fast::try_object_get_early(
        native_index,
        arity,
        line,
        stack,
        frames,
        exception_handlers,
        value_store,
        heavy_store,
    ) {
        return r;
    }

    if let Some(r) = super::object_clear_fast::try_object_clear_early(
        native_index,
        arity,
        line,
        stack,
        frames,
        exception_handlers,
        value_store,
        heavy_store,
    ) {
        return r;
    }

    if let Some(r) = super::set_fast::try_set_integral_mut_early(
        native_index,
        arity,
        line,
        stack,
        frames,
        exception_handlers,
        value_store,
        heavy_store,
    ) {
        return r;
    }

    use crate::database_engine::natives as db_natives;
    use crate::database_engine::sqenum;
    let is_sqenum_add_member = natives
        .get(native_index)
        .and_then(HostEntry::as_fn_ptr)
        == Some(sqenum::native_sqenum_add_member as *const ());
    let is_sqenum_finalize = natives
        .get(native_index)
        .and_then(HostEntry::as_fn_ptr)
        == Some(sqenum::native_sqenum_finalize as *const ());
    // Database module registers these as HostEntry::Extended after builtins; match by pointer, not index.
    let native_ptr = natives.get(native_index).and_then(HostEntry::as_fn_ptr);
    let is_db_connect = native_ptr == Some(db_natives::native_engine_connect as *const ());
    let is_db_execute = native_ptr == Some(db_natives::native_engine_execute as *const ());
    let is_db_query = native_ptr == Some(db_natives::native_engine_query as *const ());
    let is_db_run = native_ptr == Some(db_natives::native_engine_run as *const ());
    let is_db_cluster_add = native_ptr == Some(db_natives::native_cluster_add as *const ());
    let is_db_cluster_get = native_ptr == Some(db_natives::native_cluster_get as *const ());
    let is_db_cluster_names = native_ptr == Some(db_natives::native_cluster_names as *const ());
    let is_db_column = native_ptr == Some(db_natives::native_column as *const ());
    let is_db_engine_method = is_db_connect
        || is_db_execute
        || is_db_query
        || is_db_run
        || is_db_cluster_add
        || is_db_cluster_get
        || is_db_cluster_names;

    #[cfg(feature = "profile")]
    crate::vm::profile::record_native_call(
        crate::vm::native_indices::builtin_native_name(native_index),
    );

    native_args_buffer.clear();
    let mut native_arg_ids: Option<&mut Vec<ValueId>> = None;
    if is_db_engine_method {
        let frame = frames.last().unwrap();
        let available = stack::available_in_frame(stack, frame.stack_start);
        let to_pop_total = arity.min(available);
        reusable_all_popped.clear();
        reusable_all_popped.reserve(to_pop_total);
        for _ in 0..to_pop_total {
            let tv = stack::pop_direct(stack).unwrap_or(TaggedValue::null());
            let id = tagged_to_value_id(tv, value_store);
            reusable_all_popped.push(load_value(id, value_store, heavy_store));
        }
        reusable_all_popped.reverse();
        let receiver_predicate: fn(&Value) -> bool =
            if is_db_cluster_add || is_db_cluster_get || is_db_cluster_names {
                |v| matches!(v, Value::DatabaseCluster(_))
            } else {
                |v| matches!(v, Value::DatabaseEngine(_) | Value::DatabaseCluster(_))
            };
        if let Some(engine_idx) = reusable_all_popped.iter().position(receiver_predicate) {
            let receiver = reusable_all_popped.remove(engine_idx);
            native_args_buffer.push(receiver);
        }
        native_args_buffer.append(reusable_all_popped);
    }
    if !is_db_engine_method {
        let frame = frames.last().unwrap();
        // Must match saturating_sub elsewhere: if stack.len() < stack_start (bug or imbalance),
        // raw subtraction wraps and we may pop past the frame — corrupting the stack (SIGSEGV).
        let available_args = stack::available_in_frame(stack, frame.stack_start);
        if available_args < arity {
            let error = ExceptionHandler::runtime_error(
                &frames,
                format!(
                    "Not enough arguments on stack for native function: expected {} but got {}",
                    arity, available_args
                ),
                line,
            );
            return ExceptionHandler::handle_exception_vm(
                stack,
                frames,
                exception_handlers,
                error,
                value_store,
                heavy_store,
            );
        }
        // Fast path table(data, headers)
        if native_index == builtin::TABLE && arity == 2 {
            let headers_tv = stack::pop_direct(stack).unwrap_or(TaggedValue::null());
            let data_tv = stack::pop_direct(stack).unwrap_or(TaggedValue::null());
            let headers_id = tagged_to_value_id(headers_tv, value_store);
            let data_id = tagged_to_value_id(data_tv, value_store);
            let row_slots_opt2 = value_store.get(data_id).and_then(|c| {
                if let ValueCell::Array(s) = c {
                    Some(s.clone())
                } else {
                    None
                }
            });
            if let Some(row_slots) = row_slots_opt2 {
                let num_cols = row_slots
                    .first()
                    .and_then(|row_tv| {
                        if row_tv.is_heap() {
                            value_store.get(row_tv.get_heap_id()).and_then(|c| match c {
                                ValueCell::Array(slots) => Some(slots.len()),
                                _ => None,
                            })
                        } else {
                            None
                        }
                    })
                    .unwrap_or(0);
                if num_cols > 0 && !row_slots.is_empty() {
                    let mut flat_cell_ids = Vec::with_capacity(row_slots.len() * num_cols);
                    let mut ok = true;
                    for row_tv in row_slots.iter() {
                        if !row_tv.is_heap() {
                            ok = false;
                            break;
                        }
                        let row_id = row_tv.get_heap_id();
                        let cell_slots: Vec<TaggedValue> = value_store
                            .get(row_id)
                            .and_then(|c| {
                                if let ValueCell::Array(s) = c {
                                    Some(s.clone())
                                } else {
                                    None
                                }
                            })
                            .unwrap_or_default();
                        if cell_slots.len() >= num_cols {
                            for slot in cell_slots.iter().take(num_cols) {
                                flat_cell_ids.push(tagged_to_value_id(*slot, value_store));
                            }
                        } else {
                            ok = false;
                            break;
                        }
                    }
                    if ok && flat_cell_ids.len() == row_slots.len() * num_cols {
                        let headers_val = load_value(headers_id, value_store, heavy_store);
                        let header_strings: Vec<String> = match &headers_val {
                            Value::Array(rc) => rc.borrow().iter().map(|v| v.to_string()).collect(),
                            _ => Vec::new(),
                        };
                        if header_strings.len() >= num_cols {
                            let table =
                                Table::from_flat_view(flat_cell_ids, num_cols, header_strings);
                            let idx = heavy_store.push(Value::Table(Rc::new(RefCell::new(table))));
                            let result_id = value_store.allocate(ValueCell::Heavy(idx));
                            stack::push_id(stack, result_id);
                            return Ok(VMStatus::Continue);
                        }
                    }
                }
            }
            stack::push_id(stack, data_id);
            stack::push_id(stack, headers_id);
        }

        reusable_native_arg_ids.clear();
        reusable_native_arg_ids.reserve(arity);
        native_args_buffer.clear();
        native_args_buffer.reserve(arity);
        for _ in 0..arity {
            let arg_tv = stack::pop_direct(stack).unwrap_or(TaggedValue::null());
            let arg_id = if arg_tv.is_heap() {
                arg_tv.get_heap_id()
            } else {
                tagged_to_value_id(arg_tv, value_store)
            };
            reusable_native_arg_ids.push(arg_id);
            let arg_val = if arg_tv.is_heap() {
                load_value(arg_id, value_store, heavy_store)
            } else {
                slot_to_value(arg_tv, value_store, heavy_store)
            };
            native_args_buffer.push(arg_val);
        }
        reusable_native_arg_ids.reverse();
        native_args_buffer.reverse();
        native_arg_ids = Some(reusable_native_arg_ids);
    }

    let prev_ctx = VM_CALL_CONTEXT.with(|ctx| {
        let prev = *ctx.borrow();
        *ctx.borrow_mut() = Some(VmExecutionContext { vm: vm_ptr });
        prev
    });
    let _ctx_guard = RestoreVmContextGuard { previous: prev_ctx };

    if native_index == builtin::RANGE {
        if arity < 1 || arity > 3 {
            let error = ExceptionHandler::runtime_error(
                &frames,
                format!("range() expects 1, 2, or 3 arguments, got {}", arity),
                line,
            );
            return ExceptionHandler::handle_exception_vm(
                stack,
                frames,
                exception_handlers,
                error,
                value_store,
                heavy_store,
            );
        }
        match crate::common::range_args::range_spec_from_values(native_args_buffer) {
            Ok(_) => {}
            Err("zero_step") => {
                let error = ExceptionHandler::runtime_error(
                    &frames,
                    "range() step cannot be zero".to_string(),
                    line,
                );
                return ExceptionHandler::handle_exception_vm(
                    stack,
                    frames,
                    exception_handlers,
                    error,
                    value_store,
                    heavy_store,
                );
            }
            Err(_) => {
                let error = ExceptionHandler::runtime_error(
                    &frames,
                    "range() arguments must be integral numbers".to_string(),
                    line,
                );
                return ExceptionHandler::handle_exception_vm(
                    stack,
                    frames,
                    exception_handlers,
                    error,
                    value_store,
                    heavy_store,
                );
            }
        }
    }
    // Index matches legacy `execute.rs` (72 = primary_key in registry; message text unchanged).
    if native_index == builtin::PRIMARY_KEY {
        if arity != 1 {
            let error = ExceptionHandler::runtime_error(
                &frames,
                format!("enum() expects 1 argument, got {}", arity),
                line,
            );
            return ExceptionHandler::handle_exception_vm(
                stack,
                frames,
                exception_handlers,
                error,
                value_store,
                heavy_store,
            );
        }
    }

    if native_args_buffer.len() == 3 {
        if let Some(Value::Table(rc)) = native_args_buffer.get(0) {
            let t = rc.borrow();
            if t.is_view() {
                let owned = t.materialize_with(|id| load_value(id, value_store, heavy_store));
                drop(t);
                native_args_buffer[0] = Value::Table(Rc::new(RefCell::new(owned)));
            }
        }
    } else if native_args_buffer.len() == 4 {
        if let Some(Value::Table(rc)) = native_args_buffer.get(1) {
            let t = rc.borrow();
            if t.is_view() {
                let owned = t.materialize_with(|id| load_value(id, value_store, heavy_store));
                drop(t);
                native_args_buffer[1] = Value::Table(Rc::new(RefCell::new(owned)));
            }
        }
    }

    use crate::plot::natives as plot_natives;
    let is_plot_line_bar_pie_heatmap = native_index < builtin_count && {
        let ptr = natives[native_index].as_fn_ptr();
        ptr == Some(plot_natives::native_plot_line as *const ())
            || ptr == Some(plot_natives::native_plot_bar as *const ())
            || ptr == Some(plot_natives::native_plot_pie as *const ())
            || ptr == Some(plot_natives::native_plot_heatmap as *const ())
    };
    let second_is_class = native_args_buffer.len() >= 2
        && matches!(&native_args_buffer[1], Value::Object(rc) if rc.borrow().str_key_get("__class_name").is_some());
    let skip_drop = arity == 1
        || native_index == builtin::ISINSTANCE
        || native_index == builtin::OBJECT_GET
        || native_index == builtin::SAVE
        || native_index == builtin::COPY
        || second_is_class
        || (native_index == builtin::LEN && native_args_buffer.len() == 1)
        || is_db_column
        || is_sqenum_add_member
        || is_sqenum_finalize
        || is_plot_line_bar_pie_heatmap;
    if !skip_drop && !native_args_buffer.is_empty() {
        if native_args_buffer.len() > 1 {
            if let Value::Object(obj_rc) = &native_args_buffer[0] {
                if object_looks_like_module_receiver(&obj_rc.borrow()) {
                    native_args_buffer.remove(0);
                    if let Some(ids_ref) = native_arg_ids.as_mut() {
                        ids_ref.remove(0);
                    }
                }
            }
            if native_args_buffer.len() > 1 {
                if let Some(Value::Object(obj_rc)) = native_args_buffer.last() {
                    if object_looks_like_module_receiver(&obj_rc.borrow()) {
                        let last = native_args_buffer.len() - 1;
                        native_args_buffer.remove(last);
                        if let Some(ids_ref) = native_arg_ids.as_mut() {
                            ids_ref.remove(last);
                        }
                    }
                }
            }
        } else if let Value::Object(obj_rc) = &native_args_buffer[0] {
            if object_looks_like_module_receiver(&obj_rc.borrow()) {
                native_args_buffer.remove(0);
                if let Some(ids_ref) = native_arg_ids.as_mut() {
                    ids_ref.remove(0);
                }
            }
        }
    }

    if native_index == builtin::OBJECT_GET {
        if let Some(ids) = native_arg_ids.as_ref() {
            if (arity == 2 || arity == 3) && ids.len() == arity {
                let obj_id = ids[0];
                let key_id = ids[1];
                let default_id = ids.get(2).copied().unwrap_or(NULL_VALUE_ID);
                let out_id = if matches!(value_store.get(obj_id), Some(ValueCell::Object(_))) {
                    if canonical_integral_from_key_id(key_id, value_store).is_none() {
                        let key_ok = value_store
                            .get(key_id)
                            .map(crate::common::type_model::value_cell_is_hashable_key)
                            .unwrap_or(false);
                        if !key_ok {
                            let key_material = load_value(key_id, value_store, heavy_store);
                            if !crate::common::type_model::is_hashable_value(&key_material) {
                                let tn = crate::vm::calls::get_type_name_value(&key_material);
                                let error = ExceptionHandler::runtime_error_with_type(
                                    &frames,
                                    format!("unhashable type: {}", tn),
                                    line,
                                    ErrorType::TypeError,
                                );
                                return ExceptionHandler::handle_exception_vm(
                                    stack,
                                    frames,
                                    exception_handlers,
                                    error,
                                    value_store,
                                    heavy_store,
                                );
                            }
                        }
                    }
                    crate::vm::memory::object_cell_try_lookup_by_key_id(
                        obj_id,
                        key_id,
                        value_store,
                        heavy_store,
                    )
                    .unwrap_or(default_id)
                } else {
                    let error = ExceptionHandler::runtime_error_with_type(
                        &frames,
                        "TypeError: .get() expects a plain dict object".to_string(),
                        line,
                        ErrorType::TypeError,
                    );
                    return ExceptionHandler::handle_exception_vm(
                        stack,
                        frames,
                        exception_handlers,
                        error,
                        value_store,
                        heavy_store,
                    );
                };
                native_args_buffer.clear();
                push_stack_value_id(stack, value_store, out_id);
                return Ok(VMStatus::Continue);
            }
        }
        let error = ExceptionHandler::runtime_error_with_type(
            &frames,
            format!(
                "TypeError: .get() takes 1 or 2 arguments (plus optional default), got {}",
                arity
            ),
            line,
            ErrorType::TypeError,
        );
        return ExceptionHandler::handle_exception_vm(
            stack,
            frames,
            exception_handlers,
            error,
            value_store,
            heavy_store,
        );
    }

    let result = if native_index < builtin_count {
        match natives[native_index].invoke(&native_args_buffer) {
            Ok(v) => v,
            Err(e) => {
                return ExceptionHandler::handle_exception_vm(
                    stack,
                    frames,
                    exception_handlers,
                    e,
                    value_store,
                    heavy_store,
                );
            }
        }
    } else {
        let mut abi_args: Vec<Value> = native_args_buffer.clone();
        let vm = unsafe { &mut *vm_ptr };
        for v in &mut abi_args {
            let tmp = v.clone();
            match crate::vm::iterable::materialize_iterables_in_value(
                vm,
                &tmp,
                value_store,
                heavy_store,
            ) {
                Ok(x) => *v = x,
                Err(e) => {
                    return ExceptionHandler::handle_exception_vm(
                        stack,
                        frames,
                        exception_handlers,
                        e,
                        value_store,
                        heavy_store,
                    );
                }
            }
        }
        for v in &mut abi_args {
            if let Value::Table(rc) = v {
                let t = rc.borrow();
                if t.is_view() {
                    let owned = t.materialize_with(|id| load_value(id, value_store, heavy_store));
                    drop(t);
                    *v = Value::Table(Rc::new(RefCell::new(owned)));
                }
            }
        }
        crate::vm::native_loader::call_abi_native(
            abi_natives[native_index - builtin_count],
            &abi_args,
            Some((value_store, heavy_store)),
        )
    };

    if let Some(abi_err) = crate::vm::native_loader::take_last_abi_error() {
        return ExceptionHandler::handle_exception_vm(
            stack,
            frames,
            exception_handlers,
            abi_err,
            value_store,
            heavy_store,
        );
    }

    if native_index == builtin::ANTI_JOIN {
        let relations = unsafe { (*vm_ptr).take_pending_relations() };
        for (table1_rc, col1_name, table2_rc, col2_name) in relations {
            let mut found_table1_name = None;
            let mut found_table2_name = None;
            for (index, slot) in globals.iter_mut().enumerate() {
                let value_id = slot.resolve_to_value_id(value_store);
                let value = load_value(value_id, value_store, heavy_store);
                if let Value::Table(table) = &value {
                    if Rc::ptr_eq(table, &table1_rc) {
                        if let Some(var_name) = explicit_global_names.get(&index) {
                            found_table1_name = Some(var_name.clone());
                        }
                    }
                    if Rc::ptr_eq(table, &table2_rc) {
                        if let Some(var_name) = explicit_global_names.get(&index) {
                            found_table2_name = Some(var_name.clone());
                        }
                    }
                }
            }
            if let (Some(table1_name), Some(table2_name)) = (found_table1_name, found_table2_name) {
                explicit_relations.push(ExplicitRelation {
                    source_table_name: table2_name,
                    source_column_name: col2_name,
                    target_table_name: table1_name,
                    target_column_name: col1_name,
                });
            }
        }
    }

    // sort(...) возвращает отсортированный массив; синхронизируем arg0 перед write-back в store.
    if native_index == builtin::SORT && arity >= 1 && !native_args_buffer.is_empty() {
        if matches!(&result, Value::Array(_)) {
            native_args_buffer[0] = result.clone();
        }
    }

    if native_index == builtin::ZIP_JOIN {
        let primary_keys = unsafe { (*vm_ptr).take_pending_primary_keys() };
        for (table_rc, col_name) in primary_keys {
            let mut found_table_name = None;
            for (index, slot) in globals.iter_mut().enumerate() {
                let value_id = slot.resolve_to_value_id(value_store);
                let value = load_value(value_id, value_store, heavy_store);
                if let Value::Table(table) = &value {
                    if Rc::ptr_eq(table, &table_rc) {
                        if let Some(var_name) = explicit_global_names.get(&index) {
                            found_table_name = Some(var_name.clone());
                        }
                    }
                }
            }
            if let Some(table_name) = found_table_name {
                explicit_primary_keys.push(ExplicitPrimaryKey {
                    table_name,
                    column_name: col_name,
                });
            }
        }
    }

    use crate::websocket::take_native_error;
    if let Some(error_msg) = take_native_error() {
        if error_msg.contains("Falling back to CPU")
            || error_msg.contains("not available") && error_msg.contains("GPU")
        {
            debug_println!("⚠️  Предупреждение: {}", error_msg);
        } else {
            let error_type = if error_msg.starts_with("ReadOnlyError:") {
                ErrorType::ReadOnlyError
            } else if error_msg.starts_with("ZeroDivisionError:") {
                ErrorType::ZeroDivisionError
            } else if error_msg.starts_with("ValueError:") {
                ErrorType::ValueError
            } else if error_msg.starts_with("TypeError:") {
                ErrorType::TypeError
            } else if error_msg.starts_with("IndexError:") {
                ErrorType::IndexError
            } else if error_msg.starts_with("KeyError:") {
                ErrorType::KeyError
            } else if error_msg.starts_with("RuntimeError:") {
                ErrorType::RuntimeError
            } else if error_msg.contains("ShapeError")
                || error_msg.contains("Shape mismatch")
                || error_msg.starts_with("ShapeError:")
            {
                ErrorType::ValueError
            } else {
                ErrorType::IOError
            };
            let error =
                ExceptionHandler::runtime_error_with_type(&frames, error_msg, line, error_type);
            return ExceptionHandler::handle_exception_vm(
                stack,
                frames,
                exception_handlers,
                error,
                value_store,
                heavy_store,
            );
        }
    }

    if let Some(ref ids) = native_arg_ids {
        for (i, &id) in ids.iter().enumerate() {
            if i < native_args_buffer.len() {
                // native_push(arr, item): only write back the mutated array (arg 0). The pushed `item`
                // may be Value::Array (e.g. a slice); update_cell_if_mutable(item_id, &array) would
                // overwrite the ValueStore cell at item_id — which can alias another live array
                // (e.g. empty pixels_list) and corrupt it. Other natives still get full write-back.
                if native_index == builtin::PUSH && arity == 2 && i == 1 {
                    continue;
                }
                if native_index == builtin::TABLE_ADD_ROW && arity == 2 && i == 1 {
                    continue;
                }
                // ORM Column(...) does not mutate its arguments. Write-back would resynthesize cells
                // from Values and, for SQLEnum class objects, can recurse deeply on member graphs.
                if is_db_column {
                    continue;
                }
                update_cell_if_mutable(id, &native_args_buffer[i], value_store, heavy_store);
            }
        }
    }

    // Drop argument clones before materializing the return value (smaller peak stack at return).
    native_args_buffer.clear();

    stack::push_id(stack, store_value(result, value_store, heavy_store));
    Ok(VMStatus::Continue)
}
