// Function call operations for VM (Stage 1: stack/slots as ValueId)

use crate::common::{
    error::{ErrorType, LangError},
    value::{GeneratorState, Value},
    value_store::{ValueId, ValueStore},
    TaggedValue,
};
use crate::debug_println;
use crate::parser::ast::TypePart;
use crate::vm::frame::CallFrame;
use crate::vm::global_slot::GlobalSlot;
use crate::vm::global_utils::get_superclass_chain;
use crate::vm::heavy_store::HeavyStore;
use crate::vm::store_convert::{load_value, store_value};
use crate::vm::type_compat;
use std::cell::RefCell;
use std::rc::Rc;

use crate::bytecode::function::CapturedVar;

/// Index of the frame whose `slots[parent_slot_index]` must be copied for a closure capture.
#[inline]
pub(crate) fn ancestor_frame_index_for_capture(
    frames: &[CallFrame],
    c: &CapturedVar,
) -> Option<usize> {
    if c.parent_function_index != usize::MAX {
        frames
            .iter()
            .rposition(|f| f.function_index == c.parent_function_index)
    } else {
        let i = frames.len().saturating_sub(1 + c.ancestor_depth);
        (i < frames.len()).then_some(i)
    }
}

/// Проверяет, соответствует ли значение хотя бы одному из типов (union на уровне параметра).
pub fn check_type_value(value: &Value, type_parts: &[TypePart]) -> bool {
    type_compat::value_matches_type_parts(value, type_parts, None)
}

/// Like [`check_type_value`], resolving user class supertypes via class objects in globals.
pub fn check_type_value_with_globals(
    value: &Value,
    type_parts: &[TypePart],
    globals: &mut [GlobalSlot],
    global_names: &std::collections::BTreeMap<usize, String>,
    store: &mut ValueStore,
    heap: &HeavyStore,
) -> bool {
    let chain = superclass_chain_for_instance(value, globals, global_names, store, heap);
    type_compat::value_matches_type_parts(value, type_parts, chain.as_deref())
}

/// Class hierarchy for a class instance (`[Self, Parent, …]`) from globals, if applicable.
pub fn superclass_chain_for_instance(
    value: &Value,
    globals: &mut [GlobalSlot],
    global_names: &std::collections::BTreeMap<usize, String>,
    store: &mut ValueStore,
    heap: &HeavyStore,
) -> Option<Vec<String>> {
    type_compat::instance_class_name(value)
        .map(|cn| get_superclass_chain(globals, global_names, &cn, store, heap))
}

/// Форматирует список типов для сообщения об ошибке (LiteralStr в кавычках)
pub fn format_type_parts(type_parts: &[TypePart]) -> String {
    type_parts
        .iter()
        .map(TypePart::format_display)
        .collect::<Vec<_>>()
        .join(" | ")
}

/// Возвращает имя типа значения (примитивы — статическая строка; классы — `"object"`).
/// For error messages prefer [`type_compat::display_value_type`].
pub fn get_type_name_value(value: &Value) -> &'static str {
    type_compat::primitive_display_value_type(value)
}

/// Setup a function call by creating a new call frame and setting up captured variables (Stage 1: stack/slots as ValueId).
/// Returns the cached result if available (Value), or None if execution is needed.
pub fn setup_function_call(
    function_index: usize,
    args: &[Value],
    functions: &[crate::bytecode::Function],
    stack: &mut Vec<crate::common::TaggedValue>,
    frames: &mut Vec<CallFrame>,
    error_type_table: &mut Vec<String>,
    store: &mut ValueStore,
    heap: &mut HeavyStore,
    globals: &mut [GlobalSlot],
    global_names: &std::collections::BTreeMap<usize, String>,
) -> Result<Option<Value>, LangError> {
    if function_index >= functions.len() {
        return Err(LangError::runtime_error(
            format!(
                "Function index {} out of bounds (functions.len() = {})",
                function_index,
                functions.len()
            ),
            0,
        ));
    }

    let function = functions[function_index].clone();

    // Module-style call: compiler passes receiver for obj.method(); if function has arity 0, treat single Object arg as receiver and drop it.
    let padded_args: Option<Vec<Value>> = if args.len() < function.arity
        && !(function.arity == 0 && args.len() == 1)
        && crate::vm::call_defaults::trailing_defaults_available(&function, args.len())
    {
        let mut owned = args.to_vec();
        for i in args.len()..function.arity {
            if let Some(def) = function.default_values.get(i).and_then(|v| v.as_ref()) {
                owned.push(def.clone());
            }
        }
        Some(owned)
    } else {
        None
    };
    let effective_args: &[Value] = if let Some(ref owned) = padded_args {
        owned.as_slice()
    } else if function.arity == 0 && args.len() == 1 {
        if let Value::Object(_) = &args[0] {
            &args[1..] // empty slice
        } else {
            args
        }
    } else {
        args
    };

    debug_println!("[DEBUG setup_function_call] Вызываем функцию с индексом {}, имя: '{}', arity: {}, получено аргументов: {} (всего функций в VM: {})", 
        function_index, function.name, function.arity, effective_args.len(), functions.len());

    // Проверяем количество аргументов
    if effective_args.len() != function.arity {
        return Err(LangError::runtime_error(
            format!(
                "Expected {} arguments but got {}",
                function.arity,
                args.len()
            ),
            0,
        ));
    }

    // Проверяем типы аргументов, если указаны аннотации типов
    for (i, (arg, expected_types)) in effective_args.iter().zip(&function.param_types).enumerate() {
        if let Some(type_names) = expected_types {
            if !check_type_value_with_globals(arg, type_names, globals, global_names, store, heap)
            {
                let param_name = function
                    .param_names
                    .get(i)
                    .map(|s| s.as_str())
                    .unwrap_or("unknown");
                return Err(LangError::runtime_error_with_type(
                    format!(
                        "Argument '{}' expected type '{}', got '{}'",
                        param_name,
                        format_type_parts(type_names),
                        type_compat::display_value_type(arg)
                    ),
                    0,
                    ErrorType::TypeError,
                ));
            }
        }
    }

    if function.is_stream {
        if !function.captured_vars.is_empty() {
            return Err(LangError::runtime_error(
                "stream fn with captured variables is not supported yet".to_string(),
                0,
            ));
        }
        let gen = GeneratorState {
            fn_index: function_index,
            ip: 0,
            slots: Vec::new(),
            finished: false,
            final_value: None,
            pending_args: Some(effective_args.to_vec()),
            waiting_for_input: false,
            pending_first_send: None,
            cold_send_first_yield: None,
            pending_deferred_yield: None,
        };
        return Ok(Some(Value::Generator(Rc::new(RefCell::new(gen)))));
    }

    // Проверяем кэш, если функция помечена как кэшируемая
    if function.is_cached {
        use crate::bytecode::function::CacheKey;

        if let Some(cache_key) = CacheKey::new(effective_args) {
            if let Some(cache_rc) = &function.cache {
                let cache = cache_rc.borrow();
                if let Some(cached_result) = cache.map.get(&cache_key) {
                    return Ok(Some(cached_result.clone()));
                }
                drop(cache);
            }
        }
    }

    let args_tvs: Vec<TaggedValue> = effective_args
        .iter()
        .map(|a| TaggedValue::from_heap(store_value(a.clone(), store, heap)))
        .collect();
    let stack_start = crate::vm::stack::truncate_to_current_sp(stack);
    let mut new_frame = if function.is_cached {
        CallFrame::new_with_cache(
            function.clone(),
            function_index,
            stack_start,
            args_tvs.clone(),
            store,
            heap,
        )
    } else {
        CallFrame::new(
            function.clone(),
            function_index,
            stack_start,
            store,
            heap,
        )
    };

    if !function.chunk.error_type_table.is_empty() {
        *error_type_table = function.chunk.error_type_table.clone();
    }

    if !frames.is_empty() && !function.captured_vars.is_empty() {
        for captured_var in &function.captured_vars {
            if captured_var.local_slot_index >= new_frame.slots.len() {
                new_frame
                    .slots
                    .resize(captured_var.local_slot_index + 1, TaggedValue::null());
            }
            if let Some(ancestor_index) = ancestor_frame_index_for_capture(frames, captured_var) {
                if ancestor_index < frames.len() {
                    let ancestor_frame = &frames[ancestor_index];
                    if captured_var.parent_slot_index < ancestor_frame.slots.len() {
                        new_frame.slots[captured_var.local_slot_index] =
                            ancestor_frame.slots[captured_var.parent_slot_index];
                        continue;
                    }
                }
            }
            new_frame.slots[captured_var.local_slot_index] = TaggedValue::null();
        }
    }

    let param_start_index = function.captured_vars.len();
    for (i, &arg_tv) in args_tvs.iter().enumerate() {
        let slot_index = param_start_index + i;
        if slot_index >= new_frame.slots.len() {
            new_frame.slots.resize(slot_index + 1, TaggedValue::null());
        }
        new_frame.slots[slot_index] = arg_tv;
    }

    frames.push(new_frame);
    store.enter_ephemeral();
    Ok(None)
}

/// Like [`setup_function_call`], but passes canonical [`ValueId`]s as arguments (no rematerialize/copy of `this`).
pub fn setup_function_call_with_arg_ids(
    function_index: usize,
    arg_ids: &[ValueId],
    functions: &[crate::bytecode::Function],
    stack: &mut Vec<crate::common::TaggedValue>,
    frames: &mut Vec<CallFrame>,
    error_type_table: &mut Vec<String>,
    store: &mut ValueStore,
    heap: &mut HeavyStore,
    globals: &mut [GlobalSlot],
    global_names: &std::collections::BTreeMap<usize, String>,
) -> Result<Option<Value>, LangError> {
    if function_index >= functions.len() {
        return Err(LangError::runtime_error(
            format!(
                "Function index {} out of bounds (functions.len() = {})",
                function_index,
                functions.len()
            ),
            0,
        ));
    }

    let function = functions[function_index].clone();

    let padded_ids: Option<Vec<ValueId>> = if arg_ids.len() < function.arity
        && crate::vm::call_defaults::trailing_defaults_available(&function, arg_ids.len())
    {
        let mut owned = arg_ids.to_vec();
        for i in arg_ids.len()..function.arity {
            if let Some(def) = function.default_values.get(i).and_then(|v| v.as_ref()) {
                owned.push(store_value(def.clone(), store, heap));
            }
        }
        Some(owned)
    } else {
        None
    };
    let arg_ids: &[ValueId] = padded_ids.as_deref().unwrap_or(arg_ids);

    if arg_ids.len() != function.arity {
        return Err(LangError::runtime_error(
            format!(
                "Expected {} arguments but got {}",
                function.arity,
                arg_ids.len()
            ),
            0,
        ));
    }

    for (i, (&arg_id, expected_types)) in arg_ids.iter().zip(&function.param_types).enumerate() {
        if let Some(type_names) = expected_types {
            let arg = load_value(arg_id, store, heap);
            if !check_type_value_with_globals(&arg, type_names, globals, global_names, store, heap)
            {
                let param_name = function
                    .param_names
                    .get(i)
                    .map(|s| s.as_str())
                    .unwrap_or("unknown");
                return Err(LangError::runtime_error_with_type(
                    format!(
                        "Argument '{}' expected type '{}', got '{}'",
                        param_name,
                        format_type_parts(type_names),
                        type_compat::display_value_type(&arg)
                    ),
                    0,
                    ErrorType::TypeError,
                ));
            }
        }
    }

    if function.is_stream {
        return Err(LangError::runtime_error(
            "stream fn with captured variables is not supported yet".to_string(),
            0,
        ));
    }

    if function.is_cached {
        use crate::bytecode::function::CacheKey;
        let args: Vec<Value> = arg_ids
            .iter()
            .map(|&id| load_value(id, store, heap))
            .collect();
        if let Some(cache_key) = CacheKey::new(&args) {
            if let Some(cache_rc) = &function.cache {
                let cache = cache_rc.borrow();
                if let Some(cached_result) = cache.map.get(&cache_key) {
                    return Ok(Some(cached_result.clone()));
                }
            }
        }
    }

    let args_tvs: Vec<TaggedValue> = arg_ids
        .iter()
        .map(|&id| TaggedValue::from_heap(id))
        .collect();
    let stack_start = crate::vm::stack::truncate_to_current_sp(stack);
    let mut new_frame = if function.is_cached {
        CallFrame::new_with_cache(
            function.clone(),
            function_index,
            stack_start,
            args_tvs.clone(),
            store,
            heap,
        )
    } else {
        CallFrame::new(
            function.clone(),
            function_index,
            stack_start,
            store,
            heap,
        )
    };

    if !function.chunk.error_type_table.is_empty() {
        *error_type_table = function.chunk.error_type_table.clone();
    }

    if !frames.is_empty() && !function.captured_vars.is_empty() {
        for captured_var in &function.captured_vars {
            if captured_var.local_slot_index >= new_frame.slots.len() {
                new_frame
                    .slots
                    .resize(captured_var.local_slot_index + 1, TaggedValue::null());
            }
            if let Some(ancestor_index) = ancestor_frame_index_for_capture(frames, captured_var) {
                if ancestor_index < frames.len() {
                    let ancestor_frame = &frames[ancestor_index];
                    if captured_var.parent_slot_index < ancestor_frame.slots.len() {
                        new_frame.slots[captured_var.local_slot_index] =
                            ancestor_frame.slots[captured_var.parent_slot_index];
                        continue;
                    }
                }
            }
            new_frame.slots[captured_var.local_slot_index] = TaggedValue::null();
        }
    }

    let param_start_index = function.captured_vars.len();
    for (i, &arg_tv) in args_tvs.iter().enumerate() {
        let slot_index = param_start_index + i;
        if slot_index >= new_frame.slots.len() {
            new_frame.slots.resize(slot_index + 1, TaggedValue::null());
        }
        new_frame.slots[slot_index] = arg_tv;
    }

    frames.push(new_frame);
    store.enter_ephemeral();
    Ok(None)
}
