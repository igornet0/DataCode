//! Call Engine: execution of Call and CallWithUnpack opcodes.
//! User functions, natives, class constructors, and method dispatch.

use super::{closure_call, native_call};
use crate::debug_println;
use crate::bytecode::OpCode;
use crate::common::{error::LangError, value::Value, value_store::{ValueCell, ValueId, ValueStore, NULL_VALUE_ID}, TaggedValue};
use crate::vm::types::VMStatus;
use crate::vm::frame::CallFrame;
use crate::vm::exceptions::ExceptionHandler;
use crate::vm::stack;
use crate::vm::global_slot::GlobalSlot;
use crate::vm::store_convert::{store_value, load_value, tagged_to_value_id};
use crate::vm::heavy_store::HeavyStore;
use crate::vm::host::HostEntry;
use crate::vm::types::{ExplicitRelation, ExplicitPrimaryKey};
use crate::vm::interpreter::helpers::pop_to_value_id;

/// Execute CallWithUnpack(unpack_arity): kwargs object unpacking into function call.
#[allow(clippy::too_many_arguments)]
pub fn execute_call_with_unpack(
    unpack_arity: usize,
    line: usize,
    stack: &mut Vec<TaggedValue>,
    frames: &mut Vec<CallFrame>,
    functions: &mut Vec<crate::bytecode::Function>,
    exception_handlers: &mut Vec<ExceptionHandler>,
    error_type_table: &mut Vec<String>,
    value_store: &mut ValueStore,
    heavy_store: &mut HeavyStore,
    vm_ptr: *mut crate::vm::vm::Vm,
) -> Result<VMStatus, LangError> {
    // Body extracted from executor OpCode::CallWithUnpack
    let callee_tv = stack::pop(stack, frames, exception_handlers, value_store, heavy_store)?;
    if unpack_arity != 1 {
        let error = ExceptionHandler::runtime_error(
            &frames,
            format!("CallWithUnpack expects 1 argument (kwargs object), got {}", unpack_arity),
            line,
        );
        match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error, value_store, heavy_store) {
            Ok(()) => return Ok(VMStatus::Continue),
            Err(e) => return Err(e),
        }
    }
    let kwargs_tv = stack::pop(stack, frames, exception_handlers, value_store, heavy_store)?;
    let kwargs_id = tagged_to_value_id(kwargs_tv, value_store);
    let kwargs_val = load_value(kwargs_id, value_store, heavy_store);
    let obj_map = match &kwargs_val {
        Value::Object(rc) => rc.borrow().clone(),
        _ => {
            let error = ExceptionHandler::runtime_error(
                &frames,
                "** unpacking in call requires an object (dict), not a value of another type".to_string(),
                line,
            );
            match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error, value_store, heavy_store) {
                Ok(()) => return Ok(VMStatus::Continue),
                Err(e) => return Err(e),
            }
        }
    };
    let callee_id = tagged_to_value_id(callee_tv, value_store);
    let callee_val = load_value(callee_id, value_store, heavy_store);
    let (_function_index, function) = match &callee_val {
        Value::Function(i) if *i < functions.len() => (*i, functions[*i].clone()),
        Value::ModuleFunction { module_id, local_index } => {
            match unsafe { (*vm_ptr).get_module_function_index(*module_id, *local_index) } {
                Some(real_idx) if real_idx < functions.len() => (real_idx, functions[real_idx].clone()),
                _ => {
                    let error = ExceptionHandler::runtime_error(
                        &frames,
                        "** unpacking is only supported for user-defined functions".to_string(),
                        line,
                    );
                    match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error, value_store, heavy_store) {
                        Ok(()) => return Ok(VMStatus::Continue),
                        Err(e) => return Err(e),
                    }
                }
            }
        }
        _ => {
            let error = ExceptionHandler::runtime_error(
                &frames,
                "** unpacking is only supported for user-defined functions".to_string(),
                line,
            );
            match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error, value_store, heavy_store) {
                Ok(()) => return Ok(VMStatus::Continue),
                Err(e) => return Err(e),
            }
        }
    };
    let param_names = &function.param_names;
    let obj_keys: std::collections::HashSet<&String> = obj_map.keys().collect();
    let param_set: std::collections::HashSet<&String> = param_names.iter().collect();
    if obj_keys != param_set {
        let expected: Vec<&str> = param_names.iter().map(|s| s.as_str()).collect();
        let got: Vec<&str> = obj_map.keys().map(|s| s.as_str()).collect();
        let error = ExceptionHandler::runtime_error(
            &frames,
            format!(
                "Object keys must match function parameters. Expected keys: [{}], got keys: [{}]",
                expected.join(", "),
                got.join(", ")
            ),
            line,
        );
        match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error, value_store, heavy_store) {
            Ok(()) => return Ok(VMStatus::Continue),
            Err(e) => return Err(e),
        }
    }
    let mut arg_tvs = Vec::with_capacity(param_names.len());
    for p in param_names {
        let v = obj_map.get(p).cloned().unwrap_or(Value::Null);
        let id = store_value(v, value_store, heavy_store);
        arg_tvs.push(TaggedValue::from_heap(id));
    }
    let stack_start = stack.len();
    let mut new_frame = if function.is_cached {
        CallFrame::new_with_cache(function.clone(), stack_start, arg_tvs.clone(), value_store, heavy_store)
    } else {
        CallFrame::new(function.clone(), stack_start, value_store, heavy_store)
    };
    if !function.chunk.error_type_table.is_empty() {
        *error_type_table = function.chunk.error_type_table.clone();
    }
    if !frames.is_empty() && !function.captured_vars.is_empty() {
        for captured_var in &function.captured_vars {
            if captured_var.local_slot_index >= new_frame.slots.len() {
                new_frame.slots.resize(captured_var.local_slot_index + 1, TaggedValue::null());
            }
            let ancestor_index = frames.len().saturating_sub(1 + captured_var.ancestor_depth);
            if ancestor_index < frames.len() {
                let ancestor_frame = &frames[ancestor_index];
                if captured_var.parent_slot_index < ancestor_frame.slots.len() {
                    new_frame.slots[captured_var.local_slot_index] = ancestor_frame.slots[captured_var.parent_slot_index];
                } else {
                    new_frame.slots[captured_var.local_slot_index] = TaggedValue::null();
                }
            } else {
                new_frame.slots[captured_var.local_slot_index] = TaggedValue::null();
            }
        }
    }
    let param_start_index = function.captured_vars.len();
    for (i, &arg_tv) in arg_tvs.iter().enumerate() {
        let slot_index = param_start_index + i;
        if slot_index >= new_frame.slots.len() {
            new_frame.slots.resize(slot_index + 1, TaggedValue::null());
        }
        new_frame.slots[slot_index] = arg_tv;
    }
    frames.push(new_frame);
    Ok(VMStatus::Continue)
}

/// Execute Call(arity): user functions, natives, class constructors, method dispatch.
#[allow(clippy::too_many_arguments)]
pub fn execute_call(
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
        let current_ip = frames.last().unwrap().ip - 1;
        let mut function_index_opt: Option<usize> = None;
        let mut constructing_class_opt: Option<Value> = None;
        {
            let frame = frames.last_mut().unwrap();
            if frame.call_cache_ip == Some(current_ip) && frame.call_cache_is_user_function && callee_tv.is_heap() {
                let id = callee_tv.get_heap_id();
                match value_store.get(id) {
                    Some(ValueCell::Function(i)) if *i < functions.len() => {
                        function_index_opt = Some(*i);
                    }
                    Some(ValueCell::ModuleFunction { module_id, local_index }) => {
                        if let Some(real_idx) = unsafe { (*vm_ptr).get_module_function_index(*module_id, *local_index) } {
                            function_index_opt = Some(real_idx);
                        } else {
                            frame.call_cache_is_user_function = false;
                        }
                    }
                    _ => {
                        frame.call_cache_is_user_function = false;
                    }
                }
            }
        }
        let mut actual_callee: Value = if function_index_opt.is_none() {
            let frame = frames.last().unwrap();
            debug_println!("[DEBUG executor OpCode::Call] Получен OpCode::Call({}) на строке {}, IP: {}", arity, line, current_ip);
            if let Some(OpCode::Call(recorded_arity)) = frame.function.chunk.code.get(current_ip) {
                if *recorded_arity != arity {
                    debug_println!("[ERROR executor OpCode::Call] КРИТИЧЕСКАЯ ОШИБКА: В байткоде на IP {} записано Call({}), но прочитано Call({})!",
                        current_ip, recorded_arity, arity);
                }
            }
            let function_value_id = tagged_to_value_id(callee_tv, value_store);
            let function_value = load_value(function_value_id, value_store, heavy_store);
            let function_type = match &function_value {
                Value::Null => "Null",
                Value::Function(_) => "Function",
                Value::NativeFunction(_) => "NativeFunction",
                _ => "Other",
            };
            debug_println!("[DEBUG executor OpCode::Call] Значение на стеке перед вызовом: тип = {}, значение = {:?}", function_type, function_value);
            if matches!(&function_value, Value::Null) {
                debug_println!("[DEBUG Call] Пытаемся вызвать Null с {} аргументами на строке {}", arity, line);
            }
            // If callee is a class Object (from import), resolve to constructor; if Object has __call__, use it (e.g. Settings)
            let ac: Value = {
            if let Value::Object(obj_rc) = &function_value {
                let (class_name_opt, is_abstract, method_new, call_opt) = {
                    let obj_ref = obj_rc.borrow();
                    let class_name = obj_ref.get("__class_name").cloned();
                    let abstract_val = obj_ref.get("__abstract").cloned();
                    let method_key = format!("new_{}", arity);
                    let method_new = obj_ref.get(&method_key).cloned();
                    let call_ = obj_ref.get("__call__").cloned();
                    (class_name, abstract_val, method_new, call_)
                };
                if let Some(Value::String(ref class_name)) = class_name_opt {
                    if matches!(is_abstract.as_ref(), Some(Value::Bool(true))) {
                        let error = ExceptionHandler::runtime_error(
                            &frames,
                            format!("Cannot instantiate abstract class '{}'", class_name),
                            line,
                        );
                        match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error, value_store, heavy_store) {
                            Ok(()) => return Ok(VMStatus::Continue),
                            Err(e) => return Err(e),
                        }
                    }
                    let constructor_name = format!("{}::new_{}", class_name, arity);
                    let constructor_value = global_names
                        .iter()
                        .find(|(_, n)| *n == &constructor_name)
                        .and_then(|(idx, _)| {
                            if *idx < globals.len() {
                                let id = globals[*idx].resolve_to_value_id(value_store);
                                Some(load_value(id, value_store, heavy_store))
                            } else {
                                None
                            }
                        });
                    if let Some(Value::Function(constructor_fn_idx)) = constructor_value.as_ref() {
                        debug_println!("[DEBUG executor OpCode::Call] Class object '{}' resolved to constructor '{}'", class_name, constructor_name);
                        constructing_class_opt = Some(function_value.clone());
                        Value::Function(*constructor_fn_idx)
                    } else if let Some(Value::ModuleFunction { module_id, local_index }) = constructor_value.as_ref() {
                        debug_println!("[DEBUG executor OpCode::Call] Class object '{}' resolved to module constructor '{}'", class_name, constructor_name);
                        constructing_class_opt = Some(function_value.clone());
                        Value::ModuleFunction { module_id: *module_id, local_index: *local_index }
                    } else if let Some(Value::Function(constructor_fn_idx)) = method_new {
                        debug_println!("[DEBUG executor OpCode::Call] Class object '{}' resolved to constructor from class key '{}'", class_name, format!("new_{}", arity));
                        constructing_class_opt = Some(function_value.clone());
                        Value::Function(constructor_fn_idx)
                    } else if let Some(Value::ModuleFunction { module_id, local_index }) = method_new {
                        debug_println!("[DEBUG executor OpCode::Call] Class object '{}' resolved to module constructor from class key '{}'", class_name, format!("new_{}", arity));
                        constructing_class_opt = Some(function_value.clone());
                        Value::ModuleFunction { module_id, local_index }
                    } else {
                        function_value
                    }
                } else if let Some(Value::Function(_)) | Some(Value::NativeFunction(_)) = call_opt.as_ref() {
                    call_opt.unwrap()
                } else {
                    function_value
                }
            } else {
                function_value
            }
            };
            if let Value::Function(i) = &ac {
                function_index_opt = Some(*i);
                let fr = frames.last_mut().unwrap();
                fr.call_cache_ip = Some(current_ip);
                fr.call_cache_is_user_function = true;
            } else if let Value::ModuleFunction { module_id, local_index } = &ac {
                if let Some(real_idx) = unsafe { (*vm_ptr).get_module_function_index(*module_id, *local_index) } {
                    function_index_opt = Some(real_idx);
                }
                let fr = frames.last_mut().unwrap();
                fr.call_cache_ip = Some(current_ip);
                fr.call_cache_is_user_function = true;
            } else {
                let fr = frames.last_mut().unwrap();
                fr.call_cache_ip = Some(current_ip);
                fr.call_cache_is_user_function = false;
            }
            ac
        } else {
            Value::Null
        };
        // Runtime fallback: callee is Array but previous instruction was LoadGlobal(idx) for a constructor.
        // 1) If chunk has name for idx, resolve by name. 2) Else (name=? from stale/cached chunk) resolve by arity.
        if matches!(&actual_callee, Value::Array(_)) {
            if let Some(frame) = frames.last() {
                let prev_ip = frame.ip.saturating_sub(2);
                let chunk = &frame.function.chunk;
                let resolved = if let Some(crate::bytecode::OpCode::LoadGlobal(_idx)) = chunk.code.get(prev_ip) {
                    // Try by name first (chunk has global_names[idx] = "DevSettings::new_0" etc.)
                    let by_name = chunk.global_names.get(_idx)
                        .filter(|n| n.contains("::new_"))
                        .and_then(|name| {
                            global_names
                                .iter()
                                .filter(|(_, n)| n.as_str() == name.as_str())
                                .find_map(|(i, _)| {
                                    if *i < globals.len() {
                                        let id = globals[*i].resolve_to_value_id(value_store);
                                        let v = load_value(id, value_store, heavy_store);
                                        if matches!(&v, Value::Function(_)) { Some(v) } else { None }
                                    } else { None }
                                })
                        });
                    if by_name.is_some() {
                        by_name
                    } else {
                        // No name in chunk (name=?): find constructor by arity (e.g. Call(0) -> *::new_0)
                        let suffix = format!("::new_{}", arity);
                        let candidates: Vec<Value> = chunk.global_names
                            .iter()
                            .filter(|(_, n)| n.ends_with(&suffix))
                            .filter_map(|(_, n)| {
                                global_names.iter()
                                    .find(|(_, nn)| nn.as_str() == n.as_str())
                                    .and_then(|(i, _)| {
                                        if *i < globals.len() {
                                            let id = globals[*i].resolve_to_value_id(value_store);
                                            let v = load_value(id, value_store, heavy_store);
                                            if matches!(&v, Value::Function(_)) { Some(v) } else { None }
                                        } else { None }
                                    })
                            })
                            .collect();
                        if candidates.len() == 1 {
                            candidates.into_iter().next()
                        } else {
                            None
                        }
                    }
                } else {
                    None
                };
                if let Some(Value::Function(fn_idx)) = resolved {
                    if fn_idx < functions.len() {
                        actual_callee = Value::Function(fn_idx);
                        function_index_opt = Some(fn_idx);
                        let fr = frames.last_mut().unwrap();
                        fr.call_cache_ip = Some(current_ip);
                        fr.call_cache_is_user_function = true;
                    }
                }
            }
        }
        // Resolve actual function index from callee when callee is a heap Function or ModuleFunction.
        let function_index_resolved = if callee_tv.is_heap() {
            let id = callee_tv.get_heap_id();
            match value_store.get(id) {
                Some(ValueCell::Function(i)) => {
                    if *i < functions.len() { Some(*i) } else { function_index_opt }
                }
                Some(ValueCell::ModuleFunction { module_id, local_index }) => {
                    unsafe { (*vm_ptr).get_module_function_index(*module_id, *local_index) }.or(function_index_opt)
                }
                _ => function_index_opt,
            }
        } else {
            function_index_opt
        };
        let function_index_final = if let Some(function_index) = function_index_resolved {
            if function_index < functions.len() {
                function_index
            } else {
                // Index out of bounds: namespace may have raw module indices (remap missed or wrong VM).
                // Resolve by name from previous LoadGlobal so merged module code can still call (e.g. DevSettings::new_0).
                let fallback_name = frames.last().and_then(|f| {
                    let prev_ip = current_ip.saturating_sub(1);
                    if let Some(crate::bytecode::OpCode::LoadGlobal(idx)) = f.function.chunk.code.get(prev_ip) {
                        f.function.chunk.global_names.get(idx).cloned()
                    } else {
                        None
                    }
                });
                let by_name = fallback_name.clone().and_then(|name| {
                    let constructor_name = format!("{}::new_{}", name, arity);
                    functions.iter().position(|f| f.name == constructor_name)
                });
                if let Some(correct_idx) = by_name {
                    debug_println!(
                        "[CALL] out of bounds: raw_index={}, functions.len()={}, resolved by name -> correct_idx={} ('{}')",
                        function_index,
                        functions.len(),
                        correct_idx,
                        functions.get(correct_idx).map(|f| f.name.as_str()).unwrap_or("?")
                    );
                    correct_idx
                } else {
                    let expected_by_name = fallback_name.map(|n| format!("{}::new_{}", n, arity));
                    debug_println!(
                        "[CALL] Function index {} out of bounds; functions.len()={}; expected by name: {:?}; DevSettings::new_0 would be at index {:?}",
                        function_index,
                        functions.len(),
                        expected_by_name,
                        functions.iter().position(|f| f.name == "DevSettings::new_0")
                    );
                    let error = ExceptionHandler::runtime_error(
                        &frames,
                        format!("Function index {} out of bounds", function_index),
                        line,
                    );
                    match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error, value_store, heavy_store) {
                        Ok(()) => return Ok(VMStatus::Continue),
                        Err(e) => return Err(e),
                    }
                }
            }
        } else if matches!(actual_callee, Value::NativeFunction(_)) {
            // Callee is a builtin; function_index_resolved is None (heap cell is NativeFunction, not Function).
            // Do not error: dispatch happens in match actual_callee below.
            0
        } else if let Value::Object(class_rc) = &actual_callee {
            // Callee is a class Object; constructor may be ModuleFunction but registry lookup failed. Try by name.
            let class_name = class_rc.borrow().get("__class_name").and_then(|v| if let Value::String(s) = v { Some(s.clone()) } else { None });
            let constructor_name = class_name.as_ref().map(|n| format!("{}::new_{}", n, arity));
            let by_name = constructor_name.as_ref().and_then(|name| functions.iter().position(|f| f.name.as_str() == name.as_str()));
            let from_module = if by_name.is_none() {
                // Class may come from current module or a submodule; try constructor by name in all loaded modules.
                let cname = constructor_name.clone();
                let modules = unsafe { (*vm_ptr).get_modules() };
                let found: Option<Value> = cname.as_ref().and_then(|name| {
                    for (_mod_key, rc) in modules.iter() {
                        if let Some(exp) = rc.borrow().get_export(name) {
                            return Some(exp);
                        }
                    }
                    None
                });
                drop(modules);
                found.and_then(|exp| match &exp {
                    Value::Function(i) if *i < functions.len() => Some(*i),
                    Value::ModuleFunction { module_id, local_index } => {
                        unsafe { (*vm_ptr).get_module_function_index(*module_id, *local_index) }
                    }
                    _ => None,
                })
            } else {
                None
            };
            match by_name.or_else(|| from_module) {
                Some(idx) => idx,
                None => {
                    // Runtime fallback (Variant B): Object may be module/namespace; if previous LoadGlobal
                    // was for a name that exists as callable in the object (e.g. engine from database_engine),
                    // use it. Fixes name collision when submodule "engine" shadows imported engine().
                    let load_global_name = frames.last().and_then(|f| {
                        let prev_ip = current_ip.saturating_sub(1);
                        f.function.chunk.code.get(prev_ip).and_then(|op| {
                            if let crate::bytecode::OpCode::LoadGlobal(idx) = op {
                                f.function.chunk.global_names.get(idx).cloned()
                            } else {
                                None
                            }
                        })
                    });
                    let fallback_ok = if let (Some(name), Value::Object(ref obj_rc)) = (load_global_name.as_ref(), &actual_callee) {
                        let obj = obj_rc.borrow();
                        let v_opt = obj.get(name).cloned();
                        drop(obj);
                        if let Some(v) = v_opt {
                            if matches!(&v, Value::NativeFunction(_) | Value::Function(_) | Value::ModuleFunction { .. }) {
                                debug_println!(
                                    "[DEBUG Call dispatch] Object fallback: resolved '{}' from namespace to callable",
                                    name,
                                );
                                actual_callee = v;
                                true
                            } else {
                                false
                            }
                        } else {
                            false
                        }
                    } else {
                        false
                    };
                    if fallback_ok {
                        0 // dummy; match actual_callee will dispatch to NativeFunction/closure
                    } else {
                    if crate::common::debug::is_debug_enabled() {
                        let has_class = class_name.is_some();
                        debug_println!(
                            "[DEBUG Call dispatch] Object callee: has __class_name={}, LoadGlobal name={:?}, frame={}",
                            has_class,
                            load_global_name,
                            frames.last().map(|f| f.function.name.as_str()).unwrap_or("?"),
                        );
                    }
                    let error = ExceptionHandler::runtime_error(
                        &frames,
                        "Can only call functions".to_string(),
                        line,
                    );
                    match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error, value_store, heavy_store) {
                        Ok(()) => return Ok(VMStatus::Continue),
                        Err(e) => return Err(e),
                    }
                    }
                }
            }
        } else {
            let error = ExceptionHandler::runtime_error(
                &frames,
                "Can only call functions".to_string(),
                line,
            );
            match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error, value_store, heavy_store) {
                Ok(()) => return Ok(VMStatus::Continue),
                Err(e) => return Err(e),
            }
        };
        if function_index_resolved.is_some() {
            return closure_call::execute_closure_call(
                current_ip,
                function_index_final,
                constructing_class_opt.clone(),
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
        match actual_callee {
            Value::NativeFunction(native_index) => return native_call::execute_native_call(
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
            ),
            Value::Layer(layer_id) => {
                // Layers can be called as functions: layer(input_tensor) -> output_tensor
                if arity != 1 {
                    let error = ExceptionHandler::runtime_error(
                &frames,
                        format!("Layer call expects 1 argument (input tensor), got {}", arity),
                        line,
                    );
                    match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error, value_store, heavy_store) {
                        Ok(()) => {
                            stack::push_id(stack, NULL_VALUE_ID);
                            return Ok(VMStatus::Continue);
                        }
                        Err(e) => return Err(e),
                    }
                }
                
                let input_value_id = pop_to_value_id(stack, frames, exception_handlers, value_store, heavy_store)?;
                let input_value = load_value(input_value_id, value_store, heavy_store);
                use crate::ml::natives;
                let args = vec![Value::Layer(layer_id), input_value];
                let result = natives::native_layer_call(&args);
                stack::push_id(stack, store_value(result, value_store, heavy_store));
                return Ok(VMStatus::Continue);
            }
            Value::NeuralNetwork(_) | Value::LinearRegression(_) => {
                // Models can be called as functions: model(input_tensor) -> output_tensor
                if arity != 1 {
                    let error = ExceptionHandler::runtime_error(
                &frames,
                        format!("Model call expects 1 argument (input tensor), got {}", arity),
                        line,
                    );
                    match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error, value_store, heavy_store) {
                        Ok(()) => {
                            stack::push_id(stack, NULL_VALUE_ID);
                            return Ok(VMStatus::Continue);
                        }
                        Err(e) => return Err(e),
                    }
                }
                
                let input_value_id = pop_to_value_id(stack, frames, exception_handlers, value_store, heavy_store)?;
                let input_value = load_value(input_value_id, value_store, heavy_store);
                use crate::ml::natives;
                let args = vec![actual_callee.clone(), input_value];
                let result = natives::native_nn_forward(&args);
                stack::push_id(stack, store_value(result, value_store, heavy_store));
                return Ok(VMStatus::Continue);
            }
            _ => {
                // Debug: callee type, line, function name (plan 1.1)
                let callee_type = match &actual_callee {
                    Value::Null => "Null",
                    Value::Array(_) => "Array",
                    Value::Object(obj_rc) => {
                        let obj = obj_rc.borrow();
                        if obj.get("__class_name").is_some() {
                            "Object(class)"
                        } else {
                            "Object"
                        }
                    }
                    Value::Function(_) => "Function",
                    Value::NativeFunction(_) => "NativeFunction",
                    _ => "Other",
                };
                let frame_name = frames.last().map(|f| f.function.name.as_str()).unwrap_or("?");
                if crate::common::debug::verbose_constructor_debug() {
                    eprintln!("[Call error] callee type={}, line={}, function={}", callee_type, line, frame_name);
                    if matches!(&actual_callee, Value::Array(_)) {
                        if let Some(frame) = frames.last() {
                            let prev_ip = frame.ip.saturating_sub(2);
                            if let Some(crate::bytecode::OpCode::LoadGlobal(idx)) = frame.function.chunk.code.get(prev_ip) {
                                let name = frame.function.chunk.global_names.get(idx).map(|s| s.as_str()).unwrap_or("?");
                                eprintln!("[Call error] previous instruction at IP {}: LoadGlobal({}) name={}", prev_ip, idx, name);
                            }
                        }
                    }
                }
                // Try to provide more helpful error message
                let error_msg = match &actual_callee {
                    Value::Null => "Cannot call null - function may not be imported or defined".to_string(),
                    Value::Object(obj_rc) => {
                        let obj = obj_rc.borrow();
                        if let Some(Value::String(class_name)) = obj.get("__class_name") {
                            format!("Class '{}' cannot accept {} argument(s)", class_name, arity)
                        } else {
                            format!("Can only call functions, got: {:?}", std::mem::discriminant(&actual_callee))
                        }
                    }
                    _ => format!("Can only call functions, got: {:?}", std::mem::discriminant(&actual_callee)),
                };
                let error = ExceptionHandler::runtime_error(
                &frames,error_msg, line);
                match ExceptionHandler::handle_exception(stack, frames, exception_handlers, error, value_store, heavy_store) {
                    Ok(()) => {
                        // Exception handled, but we need to push null to maintain stack consistency
                        // since the caller expects a return value
                        stack::push_id(stack, NULL_VALUE_ID);
                        return Ok(VMStatus::Continue);
                    }
                    Err(e) => return Err(e),
                }
            }
        }
}
