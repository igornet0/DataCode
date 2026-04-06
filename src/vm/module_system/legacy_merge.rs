//! Legacy global merge between VMs (pre–module-isolation model).
//!
//! **When to use:** tests or embedders that still merge a submodule VM’s globals/functions/natives
//! into a host VM **without** the normal `ImportFrom` pipeline. Production code should load modules
//! via [`crate::vm::module_system::import_handler`].
//!
//! **Public API:** call [`crate::vm::Vm::merge_globals_from`] or [`crate::vm::Vm::merge_globals_from_into`]
//! only; do not depend on this module’s free functions from outside the `data-code` crate (they are
//! `pub(crate)`). Enable Cargo feature `legacy_vm_merge` if you need a future public re-export
//! (currently reserved; same behavior as default build).

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use crate::common::value::Value;
use crate::common::value_store::ValueStore;
use crate::debug_println;
use crate::vm::global_slot::{self, GlobalSlot};
use crate::vm::globals;
use crate::vm::heavy_store::HeavyStore;
use crate::vm::store_convert::{load_value, slot_to_value, store_value_arena};

/// Legacy: merge module VM into caller's buffers.
pub(crate) fn merge_globals_from_into(
    other_globals: &[GlobalSlot],
    other_global_names: &std::collections::BTreeMap<usize, String>,
    other_natives: &[crate::vm::host::HostEntry],
    other_functions: &[crate::bytecode::Function],
    other_value_store: &ValueStore,
    other_heavy_store: &HeavyStore,
    target_globals: &mut Vec<GlobalSlot>,
    target_global_names: &mut std::collections::BTreeMap<usize, String>,
    target_functions: &mut Vec<crate::bytecode::Function>,
    target_natives: &mut Vec<crate::vm::host::HostEntry>,
    store: &mut ValueStore,
    heap: &mut HeavyStore,
) {
    const BUILTIN_COUNT: usize = globals::BUILTIN_GLOBAL_COUNT;
    if other_natives.len() > BUILTIN_COUNT {
        target_natives.extend_from_slice(&other_natives[BUILTIN_COUNT..]);
        debug_println!(
            "[DEBUG merge_globals_from_into] Добавлено нативов из модуля: {}",
            other_natives.len() - BUILTIN_COUNT
        );
    }
    debug_println!(
        "[DEBUG merge_globals_from_into] Объединяем глобальные переменные из другого VM"
    );
    let start_function_index = target_functions.len();
    target_functions.extend(other_functions.iter().cloned());
    debug_println!(
        "[DEBUG merge_globals_from_into] start_function_index: {}, всего функций: {}",
        start_function_index,
        target_functions.len()
    );

    let mut pairs: Vec<_> = other_global_names
        .iter()
        .map(|(i, n)| (*i, n.clone()))
        .collect();
    pairs.sort_by(|a, b| a.1.cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
    let mut by_name: std::collections::HashMap<String, (usize, Value)> =
        std::collections::HashMap::new();
    for (index, name) in &pairs {
        if name == "argv" {
            continue;
        }
        if let Some(slot) = other_globals.get(*index) {
            let value = match slot {
                GlobalSlot::Inline(tv) => slot_to_value(*tv, other_value_store, other_heavy_store),
                GlobalSlot::Heap(id) => load_value(*id, other_value_store, other_heavy_store),
            };
            let prefer = match by_name.get(name) {
                Some((prev_index, existing)) => {
                    let existing_is_null = matches!(existing, Value::Null);
                    let value_is_null = matches!(&value, Value::Null);
                    if value_is_null && !existing_is_null {
                        false
                    } else if !value_is_null && existing_is_null {
                        true
                    } else {
                        let existing_is_obj = matches!(existing, Value::Object(_));
                        let value_is_obj = matches!(&value, Value::Object(_));
                        if value_is_obj && !existing_is_obj {
                            true
                        } else if !value_is_obj && existing_is_obj {
                            false
                        } else {
                            *index >= *prev_index
                        }
                    }
                }
                None => true,
            };
            if prefer {
                by_name.insert(name.clone(), (*index, value));
            }
        }
    }

    let mut names_sorted: Vec<_> = by_name.keys().cloned().collect();
    names_sorted.sort();
    for name in names_sorted {
        let (index, value) = by_name.get(&name).unwrap().clone();
        debug_println!(
            "[DEBUG merge_globals_from_into] Объединяем '{}' (index: {})",
            name,
            index
        );
        let value_to_store = match value {
            Value::Function(function_index) => {
                Value::Function(start_function_index + function_index)
            }
            Value::Object(obj_rc) => {
                let obj = obj_rc.borrow();
                let mut new_obj = HashMap::new();
                for (key, val) in obj.iter() {
                    let updated_val = match val {
                        Value::Function(function_index) => {
                            Value::Function(start_function_index + *function_index)
                        }
                        Value::NativeFunction(i) => {
                            if *i >= other_natives.len() {
                                val.clone()
                            } else if let Some(fn_ptr) = other_natives[*i].as_fn_ptr() {
                                let remapped = target_natives[BUILTIN_COUNT..]
                                    .iter()
                                    .position(|e| e.as_fn_ptr() == Some(fn_ptr))
                                    .map(|pos| BUILTIN_COUNT + pos)
                                    .unwrap_or_else(|| {
                                        target_natives.push(other_natives[*i].clone());
                                        target_natives.len() - 1
                                    });
                                Value::NativeFunction(remapped)
                            } else {
                                val.clone()
                            }
                        }
                        _ => val.clone(),
                    };
                    new_obj.insert(key.clone(), updated_val);
                }
                Value::Object(Rc::new(RefCell::new(new_obj)))
            }
            Value::NativeFunction(i) if i >= BUILTIN_COUNT => {
                if i >= other_natives.len() {
                    value.clone()
                } else if let Some(fn_ptr) = other_natives[i].as_fn_ptr() {
                    let remapped = target_natives[BUILTIN_COUNT..]
                        .iter()
                        .position(|e| e.as_fn_ptr() == Some(fn_ptr))
                        .map(|pos| BUILTIN_COUNT + pos)
                        .unwrap_or_else(|| {
                            target_natives.push(other_natives[i].clone());
                            target_natives.len() - 1
                        });
                    Value::NativeFunction(remapped)
                } else {
                    value.clone()
                }
            }
            _ => value.clone(),
        };

        let existing_indices: Vec<usize> = target_global_names
            .iter()
            .filter(|(_, n)| n.as_str() == name.as_str())
            .map(|(idx, _)| *idx)
            .collect();
        if let Some(&existing_index) = existing_indices.iter().min() {
            if existing_index < BUILTIN_COUNT {
                continue;
            }
            if existing_index < target_globals.len() {
                let existing_slot = &target_globals[existing_index];
                let existing_val = match existing_slot {
                    GlobalSlot::Inline(tv) => slot_to_value(*tv, store, heap),
                    GlobalSlot::Heap(id) => load_value(*id, store, heap),
                };
                if matches!(existing_val, Value::NativeFunction(_)) {
                    continue;
                }
            }
            if crate::vm::modules::is_known_module(name.as_str()) {
                continue;
            }
            if let Value::NativeFunction(i) = &value_to_store {
                if *i < BUILTIN_COUNT {
                    if let Some(canonical) = globals::builtin_global_name(*i) {
                        if canonical != name.as_str() {
                            continue;
                        }
                    }
                }
            }
            let id = store_value_arena(value_to_store.clone(), store, heap);
            if existing_index < target_globals.len() {
                target_globals[existing_index] = GlobalSlot::Heap(id);
            } else {
                target_globals.resize(existing_index + 1, global_slot::default_global_slot());
                target_globals[existing_index] = GlobalSlot::Heap(id);
            }
        } else {
            let any_existing: Option<usize> = target_global_names
                .iter()
                .find(|(_, n)| n.as_str() == name.as_str())
                .map(|(i, _)| *i);
            if let Some(canonical_slot) = any_existing {
                if canonical_slot >= BUILTIN_COUNT {
                    if canonical_slot >= target_globals.len() {
                        target_globals
                            .resize(canonical_slot + 1, global_slot::default_global_slot());
                    }
                    let id = store_value_arena(value_to_store.clone(), store, heap);
                    target_globals[canonical_slot] = GlobalSlot::Heap(id);
                }
                continue;
            }
            if let Value::NativeFunction(i) = &value_to_store {
                if *i < BUILTIN_COUNT {
                    if let Some(canonical) = globals::builtin_global_name(*i) {
                        if canonical != name.as_str() {
                            continue;
                        }
                    }
                }
            }
            let new_index = target_globals.len();
            target_globals.push(GlobalSlot::Heap(store_value_arena(
                value_to_store,
                store,
                heap,
            )));
            target_global_names.insert(new_index, name.clone());
        }
    }
    debug_println!(
        "[DEBUG merge_globals_from_into] Объединение завершено. Всего глобальных переменных: {}",
        target_global_names.len()
    );
}

/// Legacy: merges another VM's globals into target (self). Different prefer logic than merge_globals_from_into.
pub(crate) fn merge_globals_from(
    other_globals: &[GlobalSlot],
    other_global_names: &std::collections::BTreeMap<usize, String>,
    other_natives: &[crate::vm::host::HostEntry],
    other_functions: &[crate::bytecode::Function],
    other_value_store: &ValueStore,
    other_heavy_store: &HeavyStore,
    target_globals: &mut Vec<GlobalSlot>,
    target_global_names: &mut std::collections::BTreeMap<usize, String>,
    target_functions: &mut Vec<crate::bytecode::Function>,
    target_natives: &mut Vec<crate::vm::host::HostEntry>,
    store: &mut ValueStore,
    heap: &mut HeavyStore,
) {
    const BUILTIN_COUNT: usize = globals::BUILTIN_GLOBAL_COUNT;
    if other_natives.len() > BUILTIN_COUNT {
        target_natives.extend_from_slice(&other_natives[BUILTIN_COUNT..]);
        debug_println!(
            "[DEBUG merge_globals_from] Добавлено нативов из модуля: {}",
            other_natives.len() - BUILTIN_COUNT
        );
    }
    debug_println!("[DEBUG merge_globals_from] Объединяем глобальные переменные из другого VM");
    debug_println!(
        "[DEBUG merge_globals_from] Функций в текущем VM: {}",
        target_functions.len() - other_functions.len()
    );
    debug_println!(
        "[DEBUG merge_globals_from] Функций в другом VM: {}",
        other_functions.len()
    );
    debug_println!(
        "[DEBUG merge_globals_from] Глобальных переменных в другом VM: {}",
        other_global_names.len()
    );
    let start_function_index = target_functions.len();
    target_functions.extend(other_functions.iter().cloned());
    debug_println!(
        "[DEBUG merge_globals_from] Начальный индекс функций: {} (функций в текущем VM: {})",
        start_function_index,
        target_functions.len() - other_functions.len()
    );
    debug_println!(
        "[DEBUG merge_globals_from] Всего функций после объединения: {}",
        target_functions.len()
    );

    let mut pairs: Vec<_> = other_global_names
        .iter()
        .map(|(i, n)| (*i, n.clone()))
        .collect();
    pairs.sort_by(|a, b| a.1.cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
    let mut by_name: std::collections::HashMap<String, (usize, Value)> =
        std::collections::HashMap::new();
    for (index, name) in &pairs {
        if name == "argv" {
            continue;
        }
        if let Some(slot) = other_globals.get(*index) {
            let value = match slot {
                GlobalSlot::Inline(tv) => slot_to_value(*tv, other_value_store, other_heavy_store),
                GlobalSlot::Heap(id) => load_value(*id, other_value_store, other_heavy_store),
            };
            let prefer = match by_name.get(name) {
                Some((prev_index, existing)) => {
                    let existing_is_obj = matches!(existing, Value::Object(_));
                    let value_is_obj = matches!(&value, Value::Object(_));
                    if value_is_obj && !existing_is_obj {
                        true
                    } else if !value_is_obj && existing_is_obj {
                        false
                    } else {
                        *index >= *prev_index
                    }
                }
                None => true,
            };
            if prefer {
                by_name.insert(name.clone(), (*index, value));
            }
        }
    }

    let mut names_sorted: Vec<_> = by_name.keys().cloned().collect();
    names_sorted.sort();
    for name in names_sorted {
        let (index, value) = by_name.get(&name).unwrap().clone();
        debug_println!(
            "[DEBUG merge_globals_from] Объединяем '{}' (index: {})",
            name,
            index
        );
        let value_to_store = match value {
            Value::Function(function_index) => {
                let new_function_index = start_function_index + function_index;
                debug_println!(
                    "[DEBUG merge_globals_from] Обновляем индекс функции для '{}': {} -> {}",
                    name,
                    function_index,
                    new_function_index
                );
                Value::Function(new_function_index)
            }
            Value::Object(obj_rc) => {
                let obj = obj_rc.borrow();
                let mut new_obj = HashMap::new();
                for (key, val) in obj.iter() {
                    let updated_val = match val {
                        Value::Function(function_index) => {
                            Value::Function(start_function_index + *function_index)
                        }
                        Value::NativeFunction(i) => {
                            if *i >= other_natives.len() {
                                val.clone()
                            } else if let Some(fn_ptr) = other_natives[*i].as_fn_ptr() {
                                let remapped = if *i < target_natives.len()
                                    && target_natives[*i].as_fn_ptr() == Some(fn_ptr)
                                {
                                    *i
                                } else {
                                    target_natives[BUILTIN_COUNT..]
                                        .iter()
                                        .position(|e| e.as_fn_ptr() == Some(fn_ptr))
                                        .map(|pos| BUILTIN_COUNT + pos)
                                        .unwrap_or_else(|| {
                                            target_natives.push(other_natives[*i].clone());
                                            target_natives.len() - 1
                                        })
                                };
                                Value::NativeFunction(remapped)
                            } else {
                                val.clone()
                            }
                        }
                        _ => val.clone(),
                    };
                    new_obj.insert(key.clone(), updated_val);
                }
                Value::Object(Rc::new(RefCell::new(new_obj)))
            }
            Value::NativeFunction(i) if i >= BUILTIN_COUNT => {
                if i >= other_natives.len() {
                    value.clone()
                } else if let Some(fn_ptr) = other_natives[i].as_fn_ptr() {
                    let remapped = if i < target_natives.len()
                        && target_natives[i].as_fn_ptr() == Some(fn_ptr)
                    {
                        i
                    } else {
                        target_natives[BUILTIN_COUNT..]
                            .iter()
                            .position(|e| e.as_fn_ptr() == Some(fn_ptr))
                            .map(|pos| BUILTIN_COUNT + pos)
                            .unwrap_or_else(|| {
                                target_natives.push(other_natives[i].clone());
                                target_natives.len() - 1
                            })
                    };
                    Value::NativeFunction(remapped)
                } else {
                    value.clone()
                }
            }
            _ => value.clone(),
        };

        let existing_indices: Vec<usize> = target_global_names
            .iter()
            .filter(|(_, n)| n.as_str() == name.as_str())
            .map(|(idx, _)| *idx)
            .collect();
        if let Some(&existing_index) = existing_indices.iter().min() {
            if existing_index < BUILTIN_COUNT {
                debug_println!("[DEBUG merge_globals_from] Пропуск перезаписи '{}' (слот {} — канонический натив)", name, existing_index);
                continue;
            }
            if existing_index < target_globals.len() {
                let existing_slot = &target_globals[existing_index];
                let existing_val = match existing_slot {
                    GlobalSlot::Inline(tv) => slot_to_value(*tv, store, heap),
                    GlobalSlot::Heap(id) => load_value(*id, store, heap),
                };
                if matches!(existing_val, Value::NativeFunction(_)) {
                    debug_println!("[DEBUG merge_globals_from] Пропуск перезаписи '{}' (текущее значение — нативная функция)", name);
                    continue;
                }
            }
            if crate::vm::modules::is_known_module(name.as_str()) {
                debug_println!(
                    "[DEBUG merge_globals_from] Пропуск перезаписи '{}' (встроенный модуль)",
                    name
                );
                continue;
            }
            if let Value::NativeFunction(i) = &value_to_store {
                if *i < BUILTIN_COUNT {
                    if let Some(canonical) = globals::builtin_global_name(*i) {
                        if canonical != name.as_str() {
                            debug_println!("[DEBUG merge_globals_from] Пропуск перезаписи '{}' на встроенный {:?} (индекс {})", name, canonical, i);
                            continue;
                        }
                    }
                }
            }
            let val_type = match &value_to_store {
                Value::Object(_) => "Object",
                Value::Function(_) => "Function",
                _ => "Other",
            };
            debug_println!(
                "[DEBUG merge_globals_from] '{}' уже существует, перезаписываем globals[{}] ({})",
                name,
                existing_index,
                val_type
            );
            if name == "Config" || name == "DatabaseConfig" {
                debug_println!(
                    "[DEBUG merge_globals_from] Config/DatabaseConfig: '{}' -> слот {} ({})",
                    name,
                    existing_index,
                    val_type
                );
            }
            let id = store_value_arena(value_to_store.clone(), store, heap);
            if existing_index < target_globals.len() {
                target_globals[existing_index] = GlobalSlot::Heap(id);
            } else {
                target_globals.resize(existing_index + 1, global_slot::default_global_slot());
                target_globals[existing_index] = GlobalSlot::Heap(id);
            }
        } else {
            if let Value::NativeFunction(i) = &value_to_store {
                if *i < BUILTIN_COUNT {
                    if let Some(canonical) = globals::builtin_global_name(*i) {
                        if canonical != name.as_str() {
                            debug_println!("[DEBUG merge_globals_from] Пропуск создания глобальной '{}' с встроенным {:?} (индекс {})", name, canonical, i);
                            continue;
                        }
                    }
                }
            }
            let new_index = target_globals.len();
            let val_type = match &value_to_store {
                Value::Object(_) => "Object",
                Value::Function(_) => "Function",
                _ => "Other",
            };
            target_globals.push(GlobalSlot::Heap(store_value_arena(
                value_to_store,
                store,
                heap,
            )));
            target_global_names.insert(new_index, name.clone());
            debug_println!("[DEBUG merge_globals_from] Создана новая глобальная переменная '{}' в globals[{}] ({})", name, new_index, val_type);
            if name == "Config" || name == "DatabaseConfig" {
                debug_println!(
                    "[DEBUG merge_globals_from] Config/DatabaseConfig: '{}' -> новый слот {} ({})",
                    name,
                    new_index,
                    val_type
                );
            }
        }
    }
    debug_println!(
        "[DEBUG merge_globals_from] Объединение завершено. Всего глобальных переменных: {}",
        target_global_names.len()
    );
}
