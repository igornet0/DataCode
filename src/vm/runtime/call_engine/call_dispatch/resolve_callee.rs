//! Resolution phase for `OpCode::Call`: callee value, constructor/class object, array fallback, index repair.

use crate::bytecode::OpCode;
use crate::common::{
    error::LangError,
    value::Value,
    value_store::{ValueCell, ValueStore},
    TaggedValue,
};
use crate::debug_println;
use crate::vm::exceptions::ExceptionHandler;
use crate::vm::frame::CallFrame;
use crate::vm::global_slot::GlobalSlot;
use crate::vm::heavy_store::HeavyStore;
use crate::vm::stack;
use crate::vm::store_convert::{load_value, tagged_to_value_id};
use crate::vm::types::VMStatus;

/// State after resolving callee and function index for `Call(arity)`.
pub(super) struct CalleeResolution {
    pub actual_callee: Value,
    pub function_index_resolved: Option<usize>,
    pub function_index_final: usize,
    pub constructing_class_opt: Option<Value>,
    pub current_ip: usize,
    /// When set, use this arity for frame setup (defaults pushed for class-object calls).
    pub effective_arity: usize,
}

pub(super) enum CalleeResolveOutcome {
    /// Stop `execute_call` with this status (exception handled or error path).
    EarlyReturn(VMStatus),
    /// Continue to closure dispatch or `match actual_callee`.
    Resolved(CalleeResolution),
}

fn constructor_arity_from_fn_name(class_name: &str, fn_name: &str) -> Option<usize> {
    let prefix = format!("{}::new_", class_name);
    let rest = fn_name.strip_prefix(&prefix)?;
    let digit_len = rest.chars().take_while(|c| c.is_ascii_digit()).count();
    if digit_len == 0 {
        return None;
    }
    rest[..digit_len].parse().ok()
}

fn find_constructor_with_default_params(
    class_name: &str,
    call_arity: usize,
    functions: &[crate::bytecode::Function],
) -> Option<(usize, usize)> {
    let prefix = format!("{}::new_", class_name);
    let mut best: Option<(usize, usize)> = None;
    for (idx, f) in functions.iter().enumerate() {
        if !f.name.starts_with(&prefix) {
            continue;
        }
        let Some(m) = constructor_arity_from_fn_name(class_name, &f.name) else {
            continue;
        };
        if m < call_arity {
            continue;
        }
        let ok = (call_arity..m).all(|i| {
            f.default_values
                .get(i)
                .and_then(|v| v.as_ref())
                .is_some()
        });
        if !ok {
            continue;
        }
        match best {
            None => best = Some((idx, m)),
            Some((_, bm)) if m < bm => best = Some((idx, m)),
            // Duplicate same-arity entries (nested package merge) are not ambiguous — keep first.
            Some((_, bm)) if m == bm => {}
            _ => {}
        }
    }
    best
}

/// Under-arity ctor from class object keys `new_M` (M >= call_arity). Prefers smallest M.
/// Returns `(callee, total_arity, host_function_index)` for padding defaults.
fn find_constructor_with_defaults_from_class_keys(
    class_obj: &crate::common::value::ObjectKind,
    call_arity: usize,
    functions: &[crate::bytecode::Function],
    vm_ptr: *mut crate::vm::vm::Vm,
) -> Option<(Value, usize, usize)> {
    let mut best: Option<(Value, usize, usize)> = None;
    for (key, val) in class_obj.str_key_pairs() {
        let Some(rest) = key.strip_prefix("new_") else {
            continue;
        };
        if rest.is_empty() || !rest.chars().all(|c| c.is_ascii_digit()) {
            continue;
        }
        let Ok(m) = rest.parse::<usize>() else {
            continue;
        };
        if m < call_arity {
            continue;
        }
        let host_idx = match val {
            Value::Function(i) if *i < functions.len() => *i,
            Value::ModuleFunction {
                module_uid,
                local_index,
            } => match unsafe { (*vm_ptr).get_module_function_index(*module_uid, *local_index) } {
                Some(i) => i,
                None => continue,
            },
            _ => continue,
        };
        let f = &functions[host_idx];
        if !ctor_supplies_defaults(f, call_arity) {
            continue;
        }
        let total = f.arity;
        if total < call_arity {
            continue;
        }
        match &best {
            None => best = Some((val.clone(), total, host_idx)),
            Some((_, bm, _)) if total < *bm => best = Some((val.clone(), total, host_idx)),
            _ => {}
        }
    }
    best
}

/// When multiple typed `Class::new_{arity}_{types}` overloads exist, pick by argument types on the stack.
fn find_typed_constructor_by_stack_args(
    class_name: &str,
    call_arity: usize,
    stack: &[TaggedValue],
    functions: &[crate::bytecode::Function],
    value_store: &mut ValueStore,
    heavy_store: &mut HeavyStore,
) -> Option<usize> {
    if call_arity == 0 || stack.len() < call_arity {
        return None;
    }
    let prefix = format!("{}::new_{}_", class_name, call_arity);
    let candidates: Vec<(usize, String)> = functions
        .iter()
        .enumerate()
        .filter_map(|(idx, f)| {
            f.name
                .strip_prefix(&prefix)
                .map(|suffix| (idx, suffix.to_string()))
        })
        .collect();
    if candidates.is_empty() {
        return None;
    }
    if candidates.len() == 1 {
        return Some(candidates[0].0);
    }
    let arg_start = if stack.len() > call_arity {
        stack.len() - call_arity - 1
    } else {
        stack.len().saturating_sub(call_arity)
    };
    let mut inferred = Vec::with_capacity(call_arity);
    for tv in &stack[arg_start..arg_start.saturating_add(call_arity).min(stack.len())] {
        let v = load_value(tagged_to_value_id(*tv, value_store), value_store, heavy_store);
        inferred.push(
            crate::vm::type_compat::primitive_display_value_type(&v).to_string(),
        );
    }
    if call_arity == 1 {
        let inferred = &inferred[0];
        let matched: Vec<_> = candidates
            .iter()
            .filter(|(_, suffix)| {
                crate::common::constructor_overload::ctor_suffix_matches_inferred(
                    suffix,
                    inferred,
                )
            })
            .collect();
        if let Some((idx, _)) = matched.first() {
            return Some(*idx);
        }
    }
    let combined = inferred.join("_");
    candidates
        .iter()
        .find(|(_, suffix)| *suffix == combined)
        .map(|(idx, _)| *idx)
}

fn push_constructor_default_args(
    stack: &mut Vec<TaggedValue>,
    func: &crate::bytecode::Function,
    call_arity: usize,
    total_arity: usize,
    value_store: &mut ValueStore,
    heavy_store: &mut HeavyStore,
) {
    crate::vm::call_defaults::push_trailing_defaults_on_stack(
        stack,
        func,
        call_arity,
        total_arity,
        value_store,
        heavy_store,
    );
}

fn ctor_supplies_defaults(func: &crate::bytecode::Function, call_arity: usize) -> bool {
    let m = func.param_names.len();
    if m < call_arity {
        return false;
    }
    (call_arity..m).all(|i| {
        func.default_values
            .get(i)
            .and_then(|v| v.as_ref())
            .is_some()
    })
}

fn load_global_name_before_call(frames: &[CallFrame], current_ip: usize) -> Option<String> {
    frames.last().and_then(|f| {
        let prev_ip = current_ip.saturating_sub(1);
        f.function.chunk.code.get(prev_ip).and_then(|op| {
            if let OpCode::LoadGlobal(idx) = op {
                f.function.chunk.global_names.get(idx).cloned()
            } else {
                None
            }
        })
    })
}

fn class_name_and_call_arity_from_ctor_global(name: &str) -> Option<(String, usize)> {
    let pos = name.find("::new_")?;
    let class_name = name[..pos].to_string();
    let rest = &name[pos + 6..];
    let digit_len = rest.chars().take_while(|c| c.is_ascii_digit()).count();
    if digit_len == 0 {
        return None;
    }
    let call_arity = rest[..digit_len].parse().ok()?;
    Some((class_name, call_arity))
}

#[allow(clippy::too_many_arguments)]
fn load_class_object_for_name(
    class_name: &str,
    globals: &mut [GlobalSlot],
    global_names: &std::collections::BTreeMap<usize, String>,
    value_store: &mut ValueStore,
    heavy_store: &HeavyStore,
    vm_ptr: *mut crate::vm::vm::Vm,
) -> Option<Value> {
    let indices: Vec<usize> = global_names
        .iter()
        .filter(|(_, n)| n.as_str() == class_name)
        .map(|(idx, _)| *idx)
        .collect();
    for i in indices {
        if i >= globals.len() {
            continue;
        }
        let id = globals[i].resolve_to_value_id(value_store);
        let v = load_value(id, value_store, heavy_store);
        if let Value::Object(rc) = &v {
            let o = rc.borrow();
            if matches!(o.str_key_get("__class_name"), Some(Value::String(s)) if s.as_str() == class_name)
            {
                return Some(v.clone());
            }
        }
    }
    for i in 0..globals.len() {
        let id = globals[i].resolve_to_value_id(value_store);
        let v = load_value(id, value_store, heavy_store);
        if let Value::Object(rc) = &v {
            let o = rc.borrow();
            if matches!(o.str_key_get("__class_name"), Some(Value::String(s)) if s.as_str() == class_name)
            {
                return Some(v.clone());
            }
        }
    }
    let modules = unsafe { (*vm_ptr).get_modules() };
    for (_mod_key, rc) in modules.iter() {
        if let Some(v) = rc.borrow().get_export(class_name) {
            if let Value::Object(obj_rc) = &v {
                let o = obj_rc.borrow();
                let name_ok = matches!(o.str_key_get("__class_name"), Some(Value::String(s)) if s.as_str() == class_name)
                    || o.str_key_contains("new_0")
                    || o.str_key_contains("new_1");
                if name_ok {
                    return Some(v.clone());
                }
            }
        }
    }
    None
}

#[allow(clippy::too_many_arguments)]
fn try_resolve_missing_constructor_call(
    call_arity: usize,
    load_global_name: &str,
    stack: &mut Vec<TaggedValue>,
    functions: &[crate::bytecode::Function],
    globals: &mut [GlobalSlot],
    global_names: &std::collections::BTreeMap<usize, String>,
    value_store: &mut ValueStore,
    heavy_store: &mut HeavyStore,
    vm_ptr: *mut crate::vm::vm::Vm,
) -> Option<(Value, usize, usize, Option<Value>)> {
    let (class_name, name_arity) = class_name_and_call_arity_from_ctor_global(load_global_name)?;
    if name_arity != call_arity {
        return None;
    }
    let (ctor_idx, m) = find_constructor_with_default_params(&class_name, call_arity, functions)?;
    if m > call_arity {
        push_constructor_default_args(
            stack,
            &functions[ctor_idx],
            call_arity,
            m,
            value_store,
            heavy_store,
        );
    }
    let constructing_class = load_class_object_for_name(
        &class_name,
        globals,
        global_names,
        value_store,
        heavy_store,
        vm_ptr,
    );
    Some((
        Value::Function(ctor_idx),
        m,
        ctor_idx,
        constructing_class,
    ))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn resolve_call_callee(
    arity: usize,
    line: usize,
    callee_tv: TaggedValue,
    stack: &mut Vec<TaggedValue>,
    frames: &mut Vec<CallFrame>,
    globals: &mut Vec<GlobalSlot>,
    global_names: &mut std::collections::BTreeMap<usize, String>,
    functions: &mut Vec<crate::bytecode::Function>,
    exception_handlers: &mut Vec<ExceptionHandler>,
    value_store: &mut ValueStore,
    heavy_store: &mut HeavyStore,
    vm_ptr: *mut crate::vm::vm::Vm,
) -> Result<CalleeResolveOutcome, LangError> {
    let current_ip = frames.last().unwrap().ip - 1;
    let mut effective_arity = arity;
    let mut function_index_opt: Option<usize> = None;
    let mut constructing_class_opt: Option<Value> = None;
    {
        let frame = frames.last_mut().unwrap();
        if frame.call_cache_ip == Some(current_ip)
            && frame.call_cache_is_user_function
            && callee_tv.is_heap()
        {
            let id = callee_tv.get_heap_id();
            match value_store.get(id) {
                Some(ValueCell::Function(i)) if *i < functions.len() => {
                    function_index_opt = Some(*i);
                }
                Some(ValueCell::ModuleFunction {
                    module_uid,
                    local_index,
                }) => {
                    if let Some(real_idx) =
                        unsafe { (*vm_ptr).get_module_function_index(*module_uid, *local_index) }
                    {
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
        debug_println!(
            "[DEBUG executor OpCode::Call] Получен OpCode::Call({}) на строке {}, IP: {}",
            arity,
            line,
            current_ip
        );
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
            debug_println!(
                "[DEBUG Call] Пытаемся вызвать Null с {} аргументами на строке {}",
                arity,
                line
            );
        }
        let ac: Value = {
            if let Value::Object(obj_rc) = &function_value {
                let (class_name_opt, is_abstract, method_new, call_opt) = {
                    let obj_ref = obj_rc.borrow();
                    let class_name = obj_ref.str_key_get("__class_name").cloned();
                    let abstract_val = obj_ref.str_key_get("__abstract").cloned();
                    let method_key = format!("new_{}", arity);
                    let method_new = obj_ref.str_key_get(method_key.as_str()).cloned();
                    let call_ = obj_ref.str_key_get("__call__").cloned();
                    (class_name, abstract_val, method_new, call_)
                };
                if let Some(Value::String(ref class_name)) = class_name_opt {
                    if !crate::vm::special_methods::is_class_instance(&function_value) {
                    if matches!(is_abstract.as_ref(), Some(Value::Bool(true))) {
                        let error = ExceptionHandler::runtime_error(
                            &frames,
                            format!("Cannot instantiate abstract class '{}'", class_name),
                            line,
                        );
                        return Ok(CalleeResolveOutcome::EarlyReturn(
                            ExceptionHandler::handle_exception_vm(
                                stack,
                                frames,
                                exception_handlers,
                                error,
                                value_store,
                                heavy_store,
                            )?,
                        ));
                    }
                    let typed_prefix = format!("{}::new_{}_", class_name, arity);
                    let typed_overload_count = functions
                        .iter()
                        .filter(|f| f.name.starts_with(&typed_prefix))
                        .count();
                    if typed_overload_count > 1 {
                        if let Some(typed_idx) = find_typed_constructor_by_stack_args(
                            class_name,
                            arity,
                            stack,
                            functions,
                            value_store,
                            heavy_store,
                        ) {
                            debug_println!(
                                "[DEBUG executor OpCode::Call] Class object '{}' resolved to typed constructor '{}'",
                                class_name,
                                functions[typed_idx].name
                            );
                            constructing_class_opt = Some(function_value.clone());
                            Value::Function(typed_idx)
                        } else {
                            let error = ExceptionHandler::runtime_error(
                                &frames,
                                format!(
                                    "Ambiguous constructor call for class '{}': cannot pick overload for {} argument(s) at runtime",
                                    class_name, arity
                                ),
                                line,
                            );
                            return Ok(CalleeResolveOutcome::EarlyReturn(
                                ExceptionHandler::handle_exception_vm(
                                    stack,
                                    frames,
                                    exception_handlers,
                                    error,
                                    value_store,
                                    heavy_store,
                                )?,
                            ));
                        }
                    } else {
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
                    } else if let Some(Value::ModuleFunction {
                        module_uid,
                        local_index,
                    }) = constructor_value.as_ref()
                    {
                        debug_println!("[DEBUG executor OpCode::Call] Class object '{}' resolved to module constructor '{}'", class_name, constructor_name);
                        constructing_class_opt = Some(function_value.clone());
                        Value::ModuleFunction {
                            module_uid: *module_uid,
                            local_index: *local_index,
                        }
                    } else if let Some(Value::Function(constructor_fn_idx)) = method_new {
                        debug_println!("[DEBUG executor OpCode::Call] Class object '{}' resolved to constructor from class key '{}'", class_name, format!("new_{}", arity));
                        constructing_class_opt = Some(function_value.clone());
                        Value::Function(constructor_fn_idx)
                    } else if let Some(Value::ModuleFunction {
                        module_uid,
                        local_index,
                    }) = method_new
                    {
                        debug_println!("[DEBUG executor OpCode::Call] Class object '{}' resolved to module constructor from class key '{}'", class_name, format!("new_{}", arity));
                        constructing_class_opt = Some(function_value.clone());
                        Value::ModuleFunction {
                            module_uid,
                            local_index,
                        }
                    } else if let Some((callee, m, host_idx)) = {
                        let obj_ref = obj_rc.borrow();
                        find_constructor_with_defaults_from_class_keys(
                            &obj_ref,
                            arity,
                            functions,
                            vm_ptr,
                        )
                    } {
                        debug_println!(
                            "[DEBUG executor OpCode::Call] Class object '{}' resolved to constructor via class key defaults (call arity {} -> {})",
                            class_name,
                            arity,
                            m
                        );
                        if m > arity {
                            push_constructor_default_args(
                                stack,
                                &functions[host_idx],
                                arity,
                                m,
                                value_store,
                                heavy_store,
                            );
                            effective_arity = m;
                        }
                        constructing_class_opt = Some(function_value.clone());
                        callee
                    } else if let Some((ctor_idx, m)) =
                        find_constructor_with_default_params(class_name, arity, functions)
                    {
                        debug_println!(
                            "[DEBUG executor OpCode::Call] Class object '{}' resolved to constructor '{}' via default parameters (call arity {} -> {})",
                            class_name,
                            functions[ctor_idx].name,
                            arity,
                            m
                        );
                        if m > arity {
                            push_constructor_default_args(
                                stack,
                                &functions[ctor_idx],
                                arity,
                                m,
                                value_store,
                                heavy_store,
                            );
                            effective_arity = m;
                        }
                        constructing_class_opt = Some(function_value.clone());
                        Value::Function(ctor_idx)
                    } else {
                        function_value
                    }
                    }
                    } else {
                        function_value.clone()
                    }
                } else if let Some(Value::Function(_)) | Some(Value::NativeFunction(_)) =
                    call_opt.as_ref()
                {
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
        } else if let Value::ModuleFunction {
            module_uid,
            local_index,
        } = &ac
        {
            if let Some(real_idx) =
                unsafe { (*vm_ptr).get_module_function_index(*module_uid, *local_index) }
            {
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
    if crate::vm::special_methods::is_class_instance(&actual_callee) {
        if let Some(call_fn) =
            crate::vm::special_methods::try_resolve_callable_instance(&actual_callee)
        {
            let frame = frames.last().unwrap();
            let receiver_id = tagged_to_value_id(callee_tv, value_store);
            stack::insert_at_frame_start(
                stack,
                frame.stack_start,
                TaggedValue::from_heap(receiver_id),
            );
            effective_arity += 1;
            actual_callee = call_fn;
            if let Value::Function(i) = &actual_callee {
                function_index_opt = Some(*i);
            }
        }
    }
    if let Some(load_name) = load_global_name_before_call(frames, current_ip) {
        if load_name.contains("::new_") {
            if matches!(&actual_callee, Value::Null) {
                if let Some((resolved, eff, fn_idx, class_opt)) =
                    try_resolve_missing_constructor_call(
                        arity,
                        &load_name,
                        stack,
                        functions,
                        globals,
                        global_names,
                        value_store,
                        heavy_store,
                        vm_ptr,
                    )
                {
                    debug_println!(
                        "[DEBUG executor OpCode::Call] Null constructor slot '{}' resolved to '{}' (call arity {} -> {})",
                        load_name,
                        functions.get(fn_idx).map(|f| f.name.as_str()).unwrap_or("?"),
                        arity,
                        eff
                    );
                    actual_callee = resolved;
                    effective_arity = eff;
                    function_index_opt = Some(fn_idx);
                    constructing_class_opt = class_opt.or(constructing_class_opt);
                }
            } else if let Value::Function(fn_idx) = &actual_callee {
                if *fn_idx < functions.len() {
                    let func = &functions[*fn_idx];
                    let m = func.param_names.len();
                    if m > effective_arity && ctor_supplies_defaults(func, effective_arity) {
                        push_constructor_default_args(
                            stack,
                            func,
                            effective_arity,
                            m,
                            value_store,
                            heavy_store,
                        );
                        effective_arity = m;
                        if constructing_class_opt.is_none() {
                            if let Some(class_name) = func.name.split("::").next() {
                                constructing_class_opt = load_class_object_for_name(
                                    class_name,
                                    globals,
                                    global_names,
                                    value_store,
                                    heavy_store,
                                    vm_ptr,
                                );
                            }
                        }
                    }
                }
            }
        }
    }
    if matches!(&actual_callee, Value::Array(_)) {
        if let Some(frame) = frames.last() {
            let prev_ip = frame.ip.saturating_sub(2);
            let chunk = &frame.function.chunk;
            let resolved =
                if let Some(crate::bytecode::OpCode::LoadGlobal(_idx)) = chunk.code.get(prev_ip) {
                    let by_name = chunk
                        .global_names
                        .get(_idx)
                        .filter(|n| n.contains("::new_"))
                        .and_then(|name| {
                            global_names
                                .iter()
                                .filter(|(_, n)| n.as_str() == name.as_str())
                                .find_map(|(i, _)| {
                                    if *i < globals.len() {
                                        let id = globals[*i].resolve_to_value_id(value_store);
                                        let v = load_value(id, value_store, heavy_store);
                                        if matches!(&v, Value::Function(_)) {
                                            Some(v)
                                        } else {
                                            None
                                        }
                                    } else {
                                        None
                                    }
                                })
                        });
                    if by_name.is_some() {
                        by_name
                    } else {
                        let suffix = format!("::new_{}", arity);
                        let candidates: Vec<Value> = chunk
                            .global_names
                            .iter()
                            .filter(|(_, n)| n.ends_with(&suffix))
                            .filter_map(|(_, n)| {
                                global_names
                                    .iter()
                                    .find(|(_, nn)| nn.as_str() == n.as_str())
                                    .and_then(|(i, _)| {
                                        if *i < globals.len() {
                                            let id = globals[*i].resolve_to_value_id(value_store);
                                            let v = load_value(id, value_store, heavy_store);
                                            if matches!(&v, Value::Function(_)) {
                                                Some(v)
                                            } else {
                                                None
                                            }
                                        } else {
                                            None
                                        }
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
    let mut function_index_resolved = if callee_tv.is_heap() {
        let id = callee_tv.get_heap_id();
        match value_store.get(id) {
            Some(ValueCell::Function(i)) => {
                if *i < functions.len() {
                    Some(*i)
                } else {
                    function_index_opt
                }
            }
            Some(ValueCell::ModuleFunction {
                module_uid,
                local_index,
            }) => unsafe { (*vm_ptr).get_module_function_index(*module_uid, *local_index) }
                .or(function_index_opt),
            _ => function_index_opt,
        }
    } else {
        function_index_opt
    };
    let function_index_final = if let Some(function_index) = function_index_resolved {
        if function_index < functions.len() {
            function_index
        } else {
            let fallback_name = frames.last().and_then(|f| {
                let prev_ip = current_ip.saturating_sub(1);
                if let Some(crate::bytecode::OpCode::LoadGlobal(idx)) =
                    f.function.chunk.code.get(prev_ip)
                {
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
                return Ok(CalleeResolveOutcome::EarlyReturn(
                    ExceptionHandler::handle_exception_vm(
                        stack,
                        frames,
                        exception_handlers,
                        error,
                        value_store,
                        heavy_store,
                    )?,
                ));
            }
        }
    } else if matches!(actual_callee, Value::NativeFunction(_)) {
        0
    } else if let Value::Object(class_rc) = &actual_callee {
        let class_name = class_rc.borrow().str_key_get("__class_name").and_then(|v| {
            if let Value::String(s) = v {
                Some(s.clone())
            } else {
                None
            }
        });
        let constructor_name = class_name.as_ref().map(|n| format!("{}::new_{}", n, arity));
        let by_name = constructor_name.as_ref().and_then(|name| {
            functions
                .iter()
                .position(|f| f.name.as_str() == name.as_str())
        });
        let from_module = if by_name.is_none() {
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
                Value::ModuleFunction {
                    module_uid,
                    local_index,
                } => unsafe { (*vm_ptr).get_module_function_index(*module_uid, *local_index) },
                _ => None,
            })
        } else {
            None
        };
        match by_name.or(from_module).or_else(|| {
            {
                let obj_ref = class_rc.borrow();
                find_constructor_with_defaults_from_class_keys(
                    &obj_ref,
                    arity,
                    functions,
                    vm_ptr,
                )
            }
            .map(|(_, m, host_idx)| {
                if m > arity {
                    push_constructor_default_args(
                        stack,
                        &functions[host_idx],
                        arity,
                        m,
                        value_store,
                        heavy_store,
                    );
                    effective_arity = m;
                }
                constructing_class_opt = Some(Value::Object(class_rc.clone()));
                host_idx
            })
            .or_else(|| {
                class_name.as_ref().and_then(|cn| {
                    find_constructor_with_default_params(cn, arity, functions).map(|(idx, m)| {
                        if m > arity {
                            push_constructor_default_args(
                                stack,
                                &functions[idx],
                                arity,
                                m,
                                value_store,
                                heavy_store,
                            );
                            effective_arity = m;
                        }
                        constructing_class_opt = Some(Value::Object(class_rc.clone()));
                        idx
                    })
                })
            })
        }) {
            Some(idx) => {
                actual_callee = Value::Function(idx);
                function_index_resolved = Some(idx);
                idx
            }
            None => {
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
                let fallback_ok = if let (Some(name), Value::Object(ref obj_rc)) =
                    (load_global_name.as_ref(), &actual_callee)
                {
                    let obj = obj_rc.borrow();
                    let is_callable = |v: &Value| {
                        matches!(
                            v,
                            Value::NativeFunction(_)
                                | Value::Function(_)
                                | Value::ModuleFunction { .. }
                        )
                    };
                    let v_opt = obj
                        .str_key_get("__call__")
                        .cloned()
                        .filter(|v| is_callable(v))
                        .or_else(|| obj.str_key_get(name).cloned().filter(|v| is_callable(v)));
                    drop(obj);
                    if let Some(v) = v_opt {
                        if matches!(
                            &v,
                            Value::NativeFunction(_)
                                | Value::Function(_)
                                | Value::ModuleFunction { .. }
                        ) {
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
                    0
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
                    let hint = load_global_name.as_deref().unwrap_or("?");
                    let error = ExceptionHandler::runtime_error(
                        &frames,
                        format!(
                            "Can only call functions (got Object when calling '{}')",
                            hint
                        ),
                        line,
                    );
                    return Ok(CalleeResolveOutcome::EarlyReturn(
                        ExceptionHandler::handle_exception_vm(
                            stack,
                            frames,
                            exception_handlers,
                            error,
                            value_store,
                            heavy_store,
                        )?,
                    ));
                }
            }
        }
    } else if matches!(&actual_callee, Value::PluginOpaque { .. }) {
        0
    } else {
        let msg = if matches!(&actual_callee, Value::Null) {
            "Cannot call null — the callee may be missing (e.g. wrong or absent method on a module object)".to_string()
        } else {
            "Can only call functions".to_string()
        };
        let error = ExceptionHandler::runtime_error(&frames, msg, line);
        return Ok(CalleeResolveOutcome::EarlyReturn(
            ExceptionHandler::handle_exception_vm(
                stack,
                frames,
                exception_handlers,
                error,
                value_store,
                heavy_store,
            )?,
        ));
    };
    Ok(CalleeResolveOutcome::Resolved(CalleeResolution {
        actual_callee,
        function_index_resolved,
        function_index_final,
        constructing_class_opt,
        current_ip,
        effective_arity,
    }))
}
