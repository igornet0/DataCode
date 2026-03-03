//! Linker: ensure_globals_*, set_functions, merge_globals.
//! Extracted from vm.rs for Phase 7 (VM Facade).

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use crate::debug_println;
use crate::common::value::Value;
use crate::common::value_store::{ValueStore, ValueCell};
use crate::vm::global_slot::{self, GlobalSlot};
use crate::vm::globals;
use crate::vm::heavy_store::HeavyStore;
use crate::vm::store_convert::{load_value, slot_to_value, store_value_arena};

/// ValueError::new_1 native index (must match VM's VALUE_ERROR_NATIVE_INDEX).
const VALUE_ERROR_NATIVE_INDEX: usize = 75;

/// Добавляет в VM слоты для всех имён из chunk.global_names, которых ещё нет в VM.
pub fn ensure_globals_from_chunk(
    globals: &mut Vec<GlobalSlot>,
    global_names: &mut std::collections::BTreeMap<usize, String>,
    chunk: &crate::bytecode::Chunk,
) {
    const UNDEFINED_GLOBAL_SENTINEL: usize = usize::MAX;
    let mut entries: Vec<_> = chunk
        .global_names
        .iter()
        .filter(|(idx, _)| **idx != UNDEFINED_GLOBAL_SENTINEL)
        .filter(|(_, name)| name.as_str() != "argv")
        .filter(|(_, name)| !global_names.values().any(|n| n == name.as_str()))
        .map(|(idx, name)| (*idx, name.clone()))
        .collect();
    entries.sort_by_key(|(idx, _)| *idx);
    for (_old_idx, name) in entries {
        let new_idx = globals.len();
        globals.push(global_slot::default_global_slot());
        global_names.insert(new_idx, name.clone());
        debug_println!("[DEBUG ensure_globals_from_chunk] Добавлен слот для '{}' в globals[{}]", name, new_idx);
        if name == "Config" || name == "DatabaseConfig" {
            debug_println!("[DEBUG ensure_globals_from_chunk] Config/DatabaseConfig: '{}' -> слот {}", name, new_idx);
        }
    }
}

/// Как ensure_globals_from_chunk, но сохраняет индексы из chunk.
pub fn ensure_globals_from_chunk_preserve_indices(
    globals: &mut Vec<GlobalSlot>,
    global_names: &mut std::collections::BTreeMap<usize, String>,
    chunk: &crate::bytecode::Chunk,
) {
    const UNDEFINED_GLOBAL_SENTINEL: usize = usize::MAX;
    let mut entries: Vec<_> = chunk
        .global_names
        .iter()
        .filter(|(i, _)| **i != UNDEFINED_GLOBAL_SENTINEL)
        .map(|(i, n)| (*i, n.clone()))
        .collect();
    entries.sort_by_key(|(idx, _)| *idx);
    for (idx, name) in entries {
        let existing_name = global_names.get(&idx).map(|s| s.as_str());
        if std::env::var("DATACODE_DEBUG").is_ok() {
            eprintln!("ENSURE idx={} name={} existing_name={:?}", idx, name, existing_name);
        }
        if idx >= globals.len() {
            globals.resize(idx + 1, global_slot::default_global_slot());
        }
        global_names.insert(idx, name.clone());
        debug_println!("[DEBUG ensure_globals_from_chunk_preserve_indices] Добавлен слот для '{}' в globals[{}]", name, idx);
        if name == "Config" || name == "DatabaseConfig" {
            debug_println!("[DEBUG ensure_globals_from_chunk_preserve_indices] Config/DatabaseConfig: '{}' -> слот {}", name, idx);
        }
    }
    debug_println!(
        "[DEBUG ensure_globals_from_chunk_preserve_indices] после: global_names 75..80: {:?}",
        (75..80).filter_map(|i| global_names.get(&i).map(|n| (i, n.as_str()))).collect::<Vec<_>>()
    );
}

/// Fills global slots at index >= 75 whose name is a builtin.
pub fn ensure_builtin_globals_high_indices(
    globals: &mut Vec<GlobalSlot>,
    global_names: &std::collections::BTreeMap<usize, String>,
    value_store: &mut ValueStore,
    heavy_store: &crate::vm::heavy_store::HeavyStore,
) {
    const BUILTIN_END: usize = 75;
    for (idx, name) in global_names.iter() {
        if *idx < BUILTIN_END {
            continue;
        }
        if let Some(builtin_index) = globals::builtin_global_index(name) {
            if *idx >= globals.len() {
                continue;
            }
            let value_id = globals[*idx].resolve_to_value_id(value_store);
            let current = load_value(value_id, value_store, heavy_store);
            if matches!(current, Value::Null) {
                let id = value_store.allocate(ValueCell::NativeFunction(builtin_index));
                globals[*idx] = GlobalSlot::Heap(id);
                debug_println!("[DEBUG ensure_builtin_globals_high_indices] '{}' at globals[{}] -> builtin index {}", name, idx, builtin_index);
            }
        }
    }
}

/// Injects built-in exception constructors (e.g. ValueError::new_1) into slots that are still null.
pub fn ensure_exception_constructors(
    globals: &mut Vec<GlobalSlot>,
    global_names: &std::collections::BTreeMap<usize, String>,
    value_store: &mut ValueStore,
    heavy_store: &crate::vm::heavy_store::HeavyStore,
) {
    const NAME: &str = "ValueError::new_1";
    let idx = match global_names.iter().find(|(_, n)| n.as_str() == NAME).map(|(i, _)| *i) {
        Some(i) => i,
        None => return,
    };
    if idx >= globals.len() {
        return;
    }
    let value_id = globals[idx].resolve_to_value_id(value_store);
    let current = load_value(value_id, value_store, heavy_store);
    if matches!(current, Value::Null) {
        let id = value_store.allocate(ValueCell::NativeFunction(VALUE_ERROR_NATIVE_INDEX));
        globals[idx] = GlobalSlot::Heap(id);
        debug_println!("[DEBUG ensure_exception_constructors] Injected {} at globals[{}]", NAME, idx);
    }
}

fn find_function_index_by_name(functions: &[crate::bytecode::Function], function_name: &str) -> Option<usize> {
    functions.iter().position(|f| f.name == function_name)
}

/// Устанавливает функции в VM. Патчит LoadGlobal/StoreGlobal в main и function chunks, обновляет globals.
pub fn set_functions(
    globals: &mut Vec<GlobalSlot>,
    global_names: &mut std::collections::BTreeMap<usize, String>,
    functions: &mut Vec<crate::bytecode::Function>,
    value_store: &mut ValueStore,
    explicit_global_names: &mut std::collections::BTreeMap<usize, String>,
    argv_slot_index: Option<usize>,
    new_functions: Vec<crate::bytecode::Function>,
    main_chunk: Option<&mut crate::bytecode::Chunk>,
    main_old_idx_to_name: Option<std::collections::HashMap<usize, String>>,
) {
    use crate::vm::module_system::chunk_patcher::MODEL_CONFIG_CLASS_LOAD_INDEX;

    let mut explicit_global_names_to_add = std::collections::HashMap::new();

    let mut all_pairs: Vec<(usize, String)> = Vec::new();
    if let Some(mc) = main_chunk.as_ref() {
        for (idx, name) in &mc.global_names {
            all_pairs.push((*idx, name.clone()));
        }
    }
    for function in &new_functions {
        for (idx, name) in &function.chunk.global_names {
            all_pairs.push((*idx, name.clone()));
        }
    }
    let mut unique_names: Vec<String> = all_pairs.iter().map(|(_, n)| n.clone()).collect();
    unique_names.sort();
    unique_names.dedup();

    const UNDEFINED_GLOBAL_SENTINEL: usize = usize::MAX;
    let mut name_to_new_idx: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    let mut constructor_slots_used: std::collections::HashSet<usize> = std::collections::HashSet::new();
    for name in &unique_names {
        let only_sentinel = all_pairs.iter()
            .filter(|(_, n)| n == name)
            .all(|(idx, _)| *idx == UNDEFINED_GLOBAL_SENTINEL);
        if only_sentinel {
            continue;
        }
        let new_idx = if let Some(builtin_idx) = globals::builtin_global_index(name) {
            debug_println!("[DEBUG set_functions] Встроенное имя '{}' -> канонический индекс {}", name, builtin_idx);
            builtin_idx
        } else if name == "argv" && argv_slot_index.is_some() {
            let slot = argv_slot_index.unwrap();
            debug_println!("[DEBUG set_functions] argv -> слот {} (argv_slot_index)", slot);
            slot
        } else {
            let matching_indices: Vec<usize> = global_names.iter()
                .filter(|(_, n)| *n == name)
                .map(|(idx, _)| *idx)
                .collect();
            let candidate = matching_indices.iter().min().copied();
            let is_constructor = name.contains("::new_");
            let new_idx = if is_constructor {
                if let Some(real_idx) = candidate {
                    if constructor_slots_used.insert(real_idx) {
                        real_idx
                    } else {
                        let fresh = globals.len();
                        globals.push(global_slot::default_global_slot());
                        global_names.insert(fresh, name.clone());
                        constructor_slots_used.insert(fresh);
                        debug_println!("[DEBUG set_functions] Конструктор '{}' -> новый слот {} (конфликт)", name, fresh);
                        fresh
                    }
                } else {
                    let fresh = globals.len();
                    globals.push(global_slot::default_global_slot());
                    global_names.insert(fresh, name.clone());
                    constructor_slots_used.insert(fresh);
                    debug_println!("[DEBUG set_functions] Конструктор '{}' не найден в VM, слот {}", name, fresh);
                    fresh
                }
            } else if let Some(real_idx) = candidate {
                debug_println!("[DEBUG set_functions] Найдена переменная '{}' в VM с индексом {} (детерминированный min)", name, real_idx);
                if name == "Config" || name == "DatabaseConfig" {
                    debug_println!("[DEBUG set_functions] Config/DatabaseConfig: '{}' -> индекс {} (найден в self.global_names)", name, real_idx);
                }
                real_idx
            } else {
                let new_idx = globals.len();
                globals.push(global_slot::default_global_slot());
                global_names.insert(new_idx, name.clone());
                debug_println!("[DEBUG set_functions] Переменная '{}' не найдена в VM, создаем новый индекс {}", name, new_idx);
                if name == "Config" || name == "DatabaseConfig" {
                    debug_println!("[DEBUG set_functions] Config/DatabaseConfig: '{}' -> индекс {} (создан новый слот)", name, new_idx);
                }
                new_idx
            };
            new_idx
        };
        name_to_new_idx.insert(name.clone(), new_idx);
    }

    if main_chunk.is_none() {
        if let Some(main_function) = new_functions.first() {
            for (idx, name) in &main_function.chunk.explicit_global_names {
                explicit_global_names_to_add.insert(*idx, name.clone());
            }
        }
    }

    let main_old_idx_to_name: std::collections::HashMap<usize, String> = main_old_idx_to_name
        .unwrap_or_else(|| {
            main_chunk
                .as_ref()
                .map(|mc| mc.global_names.iter().map(|(i, n)| (*i, n.clone())).collect())
                .unwrap_or_default()
        });

    if let Some(mc) = main_chunk {
        let old_to_new: std::collections::HashMap<usize, usize> = mc
            .global_names
            .iter()
            .filter_map(|(old_idx, name)| {
                name_to_new_idx.get(name).and_then(|&new_idx| {
                    if *old_idx != new_idx { Some((*old_idx, new_idx)) } else { None }
                })
            })
            .collect();
        for opcode in &mut mc.code {
            match opcode {
                crate::bytecode::OpCode::LoadGlobal(idx) => {
                    if let Some(&new_idx) = old_to_new.get(idx) {
                        *opcode = crate::bytecode::OpCode::LoadGlobal(new_idx);
                    }
                }
                crate::bytecode::OpCode::StoreGlobal(idx) => {
                    if let Some(&new_idx) = old_to_new.get(idx) {
                        *opcode = crate::bytecode::OpCode::StoreGlobal(new_idx);
                    }
                }
                _ => {}
            }
        }
        let to_insert: Vec<_> = old_to_new
            .iter()
            .filter_map(|(old_idx, new_idx)| {
                mc.global_names.get(old_idx).map(|name| (*old_idx, *new_idx, name.clone()))
            })
            .collect();
        for (old_idx, _, _) in &to_insert {
            mc.global_names.remove(old_idx);
        }
        for (_, new_idx, name) in to_insert {
            mc.global_names.insert(new_idx, name);
        }
    }

    let existing_functions_count = functions.len();
    let mut updated_functions = new_functions;
    for function in &mut updated_functions {
        let mut old_to_new: std::collections::HashMap<usize, usize> = function
            .chunk
            .global_names
            .iter()
            .filter_map(|(old_idx, name)| {
                let new_idx = if *old_idx == MODEL_CONFIG_CLASS_LOAD_INDEX {
                    let matching: Vec<usize> = global_names.iter()
                        .filter(|(_, n)| *n == name)
                        .map(|(idx, _)| *idx)
                        .collect();
                    if let Some(&real_idx) = matching.iter().min() {
                        debug_println!(
                            "[DEBUG set_functions] sentinel функция '{}' класс '{}' -> globals[{}] (из VM global_names)",
                            function.name, name, real_idx
                        );
                        Some(real_idx)
                    } else {
                        let fallback = name_to_new_idx.get(name).copied();
                        debug_println!(
                            "[DEBUG set_functions] WARNING: sentinel функция '{}' класс '{}' не найден в self.global_names, fallback name_to_new_idx -> {:?}",
                            function.name, name, fallback
                        );
                        fallback
                    }
                } else {
                    name_to_new_idx.get(name).copied()
                };
                new_idx.and_then(|new_idx| {
                    if *old_idx != new_idx { Some((*old_idx, new_idx)) } else { None }
                })
            })
            .collect();
        for (_op_index, opcode) in function.chunk.code.iter().enumerate() {
            let idx = match opcode {
                crate::bytecode::OpCode::LoadGlobal(i) | crate::bytecode::OpCode::StoreGlobal(i) => *i,
                _ => continue,
            };
            if old_to_new.contains_key(&idx) {
                continue;
            }
            let name = function.chunk.global_names.get(&idx)
                .or_else(|| main_old_idx_to_name.get(&idx));
            if let Some(name) = name {
                if let Some(&new_idx) = name_to_new_idx.get(name) {
                    if idx != new_idx {
                        old_to_new.insert(idx, new_idx);
                    }
                }
            }
        }
        for opcode in &mut function.chunk.code {
            match opcode {
                crate::bytecode::OpCode::LoadGlobal(idx) => {
                    if let Some(&new_idx) = old_to_new.get(idx) {
                        *opcode = crate::bytecode::OpCode::LoadGlobal(new_idx);
                    }
                }
                crate::bytecode::OpCode::StoreGlobal(idx) => {
                    if let Some(&new_idx) = old_to_new.get(idx) {
                        *opcode = crate::bytecode::OpCode::StoreGlobal(new_idx);
                    }
                }
                _ => {}
            }
        }
        let to_insert: Vec<_> = old_to_new
            .iter()
            .filter_map(|(old_idx, new_idx)| {
                function.chunk.global_names.get(old_idx).map(|name| (*old_idx, *new_idx, name.clone()))
            })
            .collect();
        for (old_idx, _, _) in &to_insert {
            function.chunk.global_names.remove(old_idx);
        }
        for (_, new_idx, name) in to_insert {
            function.chunk.global_names.insert(new_idx, name);
        }
    }

    debug_println!("[DEBUG set_functions] Добавляем {} новых функций к существующим {} функциям", updated_functions.len(), existing_functions_count);
    for function in &updated_functions {
        debug_println!("[DEBUG set_functions] Проверка Call инструкций в функции '{}':", function.name);
        for (ip, opcode) in function.chunk.code.iter().enumerate() {
            if let crate::bytecode::OpCode::Call(arity) = opcode {
                debug_println!("[DEBUG set_functions]   IP {}: Call({})", ip, arity);
            }
        }
    }

    let mut names_from_chunks: std::collections::HashSet<String> = std::collections::HashSet::new();
    for function in &updated_functions {
        for (_, name) in &function.chunk.global_names {
            names_from_chunks.insert(name.clone());
        }
    }
    let mut names_sorted: Vec<String> = names_from_chunks.into_iter().collect();
    names_sorted.sort();
    let chunk_global_names: Vec<(usize, String)> = names_sorted
        .iter()
        .filter_map(|name| name_to_new_idx.get(name).map(|&idx| (idx, name.clone())))
        .collect();

    functions.extend(updated_functions);

    debug_println!("[DEBUG set_functions] Начинаем обновление индексов функций в globals. Всего глобальных переменных в VM: {}, имён из chunk: {}", global_names.len(), chunk_global_names.len());

    for (real_global_idx, name) in &chunk_global_names {
        debug_println!("[DEBUG set_functions] Обрабатываем '{}' из chunk -> слот {}", name, real_global_idx);
        if Some(*real_global_idx) == argv_slot_index {
            continue;
        }
        if *real_global_idx < globals::BUILTIN_GLOBAL_NAMES.len() {
            if let Some(canonical) = globals::builtin_global_name(*real_global_idx) {
                if canonical == name.as_str() {
                    if let Some(fn_idx) = find_function_index_by_name(functions, name) {
                        globals[*real_global_idx] = GlobalSlot::Heap(value_store.allocate_arena(ValueCell::Function(fn_idx)));
                        debug_println!("[DEBUG set_functions] Перезапись встроенного натива '{}' в слоте {} пользовательской функцией", name, real_global_idx);
                    } else {
                        debug_println!("[DEBUG set_functions] Пропуск перезаписи встроенного натива '{}' в слоте {}", name, real_global_idx);
                    }
                    continue;
                }
            }
        }
        if *real_global_idx < globals.len() {
            debug_println!("[DEBUG set_functions] Проверяем globals[{}] для '{}'", real_global_idx, name);
            let id = globals[*real_global_idx].resolve_to_value_id(value_store);
            let old_fn_idx_opt = value_store.get(id)
                .and_then(|c| if let ValueCell::Function(i) = c { Some(*i) } else { None });
            if let Some(old_fn_idx) = old_fn_idx_opt {
                debug_println!("[DEBUG set_functions] Найдена функция '{}' в globals[{}] (из chunk) с индексом функции {}", name, real_global_idx, old_fn_idx);
                if let Some(new_fn_idx) = find_function_index_by_name(functions, name) {
                    if old_fn_idx != new_fn_idx {
                        debug_println!("[DEBUG set_functions] Обновляем индекс функции для '{}' в globals[{}]: {} -> {}", name, real_global_idx, old_fn_idx, new_fn_idx);
                        globals[*real_global_idx] = GlobalSlot::Heap(value_store.allocate_arena(ValueCell::Function(new_fn_idx)));
                    } else {
                        debug_println!("[DEBUG set_functions] Индекс функции для '{}' уже правильный: {}", name, new_fn_idx);
                    }
                } else {
                    debug_println!("[DEBUG set_functions] WARNING: Функция '{}' не найдена в списке функций VM", name);
                }
            } else {
                if let Some(fn_idx) = find_function_index_by_name(functions, name) {
                    globals[*real_global_idx] = GlobalSlot::Heap(value_store.allocate_arena(ValueCell::Function(fn_idx)));
                    debug_println!("[DEBUG set_functions] Установлена функция '{}' в globals[{}] = Value::Function({}) (слот был Null/другой)", name, real_global_idx, fn_idx);
                } else {
                    debug_println!("[DEBUG set_functions] globals[{}] для '{}' не является функцией", real_global_idx, name);
                }
            }
        } else {
            debug_println!("[DEBUG set_functions] WARNING: Индекс {} выходит за границы globals (всего: {})", real_global_idx, globals.len());
        }
    }

    let start_fn_idx = existing_functions_count;
    for new_fn_idx in start_fn_idx..functions.len() {
        let function_name = &functions[new_fn_idx].name;
        debug_println!("[DEBUG set_functions] Проверяем функцию '{}' с индексом {} (только что добавлена, compiler_idx: {})", function_name, new_fn_idx, new_fn_idx - existing_functions_count);
        let compiler_fn_idx = new_fn_idx - existing_functions_count;
        let mut found_in_chunk = false;
        for function in &functions[start_fn_idx..] {
            if let Some(global_idx) = function.chunk.global_names.iter()
                .find(|(_, name)| *name == function_name)
                .map(|(idx, _)| *idx) {
                let real_global_idx = global_idx;
                if real_global_idx < globals::BUILTIN_GLOBAL_NAMES.len() {
                    if let Some(canonical) = globals::builtin_global_name(real_global_idx) {
                        if canonical == function_name.as_str() {
                            globals[real_global_idx] = GlobalSlot::Heap(value_store.allocate_arena(ValueCell::Function(new_fn_idx)));
                            debug_println!("[DEBUG set_functions] Перезапись встроенного натива '{}' в слоте {} пользовательской функцией (второй цикл)", function_name, real_global_idx);
                            found_in_chunk = true;
                            break;
                        }
                    }
                }
                if real_global_idx < globals.len() {
                    let rid = globals[real_global_idx].resolve_to_value_id(value_store);
                    let old_opt = value_store.get(rid)
                        .and_then(|c| if let ValueCell::Function(i) = c { Some(*i) } else { None });
                    if let Some(old_fn_idx) = old_opt {
                        if old_fn_idx != new_fn_idx {
                            debug_println!("[DEBUG set_functions] Найдена функция '{}' в globals[{}] через chunk global_names: {} -> {}", function_name, real_global_idx, old_fn_idx, new_fn_idx);
                            globals[real_global_idx] = GlobalSlot::Heap(value_store.allocate_arena(ValueCell::Function(new_fn_idx)));
                            found_in_chunk = true;
                            break;
                        }
                    }
                }
            }
        }

        if !found_in_chunk {
            debug_println!("[DEBUG set_functions] Ищем функцию '{}' во всех globals (compiler_idx: {})", function_name, compiler_fn_idx);
            for global_idx in 0..globals.len() {
                let gid = globals[global_idx].resolve_to_value_id(value_store);
                let old_fn_idx_opt = value_store.get(gid)
                    .and_then(|c| if let ValueCell::Function(i) = c { Some(*i) } else { None });
                if let Some(old_fn_idx) = old_fn_idx_opt {
                    let already_processed = global_names.get(&global_idx)
                        .map(|name| name == function_name)
                        .unwrap_or(false);
                    if !already_processed && old_fn_idx == compiler_fn_idx {
                        debug_println!("[DEBUG set_functions] Найден кандидат для '{}' в globals[{}] с old_fn_idx={}, compiler_idx={}", function_name, global_idx, old_fn_idx, compiler_fn_idx);
                    }
                    if !already_processed {
                        let should_update = if old_fn_idx == compiler_fn_idx {
                            if old_fn_idx < functions.len() {
                                let old_fn_name = &functions[old_fn_idx].name;
                                let was_overwritten = old_fn_name != function_name;
                                debug_println!("[DEBUG set_functions] Проверка для '{}': old_fn_idx={}, old_fn_name='{}', compiler_idx={}, was_overwritten={}", function_name, old_fn_idx, old_fn_name, compiler_fn_idx, was_overwritten);
                                was_overwritten
                            } else {
                                true
                            }
                        } else if old_fn_idx < functions.len() {
                            functions[old_fn_idx].name == *function_name && old_fn_idx != new_fn_idx
                        } else {
                            false
                        };
                        if should_update {
                            debug_println!("[DEBUG set_functions] Найдена функция '{}' в globals[{}] с индексом {} -> {} (compiler_idx: {}, old_fn_name: '{}')", function_name, global_idx, old_fn_idx, new_fn_idx, compiler_fn_idx, if old_fn_idx < functions.len() { &functions[old_fn_idx].name } else { "OUT_OF_BOUNDS" });
                            globals[global_idx] = GlobalSlot::Heap(value_store.allocate_arena(ValueCell::Function(new_fn_idx)));
                            break;
                        }
                    }
                }
            }
        }
    }

    for (global_idx, name) in global_names.clone().iter() {
        if Some(*global_idx) == argv_slot_index {
            continue;
        }
        if *global_idx < globals.len() {
            let gid = globals[*global_idx].resolve_to_value_id(value_store);
            let old_fn_idx_opt = value_store.get(gid)
                .and_then(|c| if let ValueCell::Function(i) = c { Some(*i) } else { None });
            if let Some(old_fn_idx) = old_fn_idx_opt {
                debug_println!("[DEBUG set_functions] Найдена функция '{}' в globals[{}] с индексом функции {}", name, global_idx, old_fn_idx);
                let old_fn_name_matches = if old_fn_idx < functions.len() {
                    let matches = functions[old_fn_idx].name == *name;
                    debug_println!("[DEBUG set_functions] Старый индекс функции {}: имя='{}', совпадает с глобальной переменной '{}': {}", old_fn_idx, functions[old_fn_idx].name, name, matches);
                    matches
                } else {
                    debug_println!("[DEBUG set_functions] Старый индекс функции {} выходит за границы (всего функций: {})", old_fn_idx, functions.len());
                    false
                };
                if let Some(new_fn_idx) = find_function_index_by_name(functions, name) {
                    debug_println!("[DEBUG set_functions] Найдена функция '{}' в VM с индексом {}", name, new_fn_idx);
                    if new_fn_idx < functions.len() && functions[new_fn_idx].name == *name {
                        if old_fn_idx != new_fn_idx {
                            debug_println!("[DEBUG set_functions] Обновляем индекс функции для '{}' в globals[{}]: {} -> {} (старое имя совпадает: {})", name, global_idx, old_fn_idx, new_fn_idx, old_fn_name_matches);
                            globals[*global_idx] = GlobalSlot::Heap(value_store.allocate_arena(ValueCell::Function(new_fn_idx)));
                        } else {
                            debug_println!("[DEBUG set_functions] Индекс функции для '{}' уже правильный: {}", name, new_fn_idx);
                        }
                    } else {
                        debug_println!("[DEBUG set_functions] WARNING: Имя функции в новом индексе {} не совпадает с именем глобальной переменной '{}' (имя функции: '{}')", new_fn_idx, name, if new_fn_idx < functions.len() { &functions[new_fn_idx].name } else { "OUT OF BOUNDS" });
                    }
                } else {
                    debug_println!("[DEBUG set_functions] WARNING: Функция '{}' не найдена в списке функций VM (старый индекс: {}, старое имя совпадает: {})", name, old_fn_idx, old_fn_name_matches);
                }
            }
        }
    }
    debug_println!("[DEBUG set_functions] Завершено обновление индексов функций в globals");

    let argv_slot = argv_slot_index;
    if let Some((&global_idx, _)) = global_names.iter().find(|(_, n)| *n == "__main__") {
        if let Some(fn_idx) = find_function_index_by_name(functions, "__main__") {
            if global_idx < globals.len() && Some(global_idx) != argv_slot {
                globals[global_idx] = GlobalSlot::Heap(value_store.allocate_arena(ValueCell::Function(fn_idx)));
                debug_println!("[DEBUG set_functions] Установлен слот __main__ в globals[{}] = Value::Function({})", global_idx, fn_idx);
            }
        }
    }
    if let Some((&global_idx, _)) = global_names.iter().find(|(_, n)| *n == "main") {
        if let Some(fn_idx) = find_function_index_by_name(functions, "main") {
            if global_idx < globals.len() && Some(global_idx) != argv_slot {
                globals[global_idx] = GlobalSlot::Heap(value_store.allocate_arena(ValueCell::Function(fn_idx)));
                debug_println!("[DEBUG set_functions] Установлен слот main в globals[{}] = Value::Function({})", global_idx, fn_idx);
            }
        }
    }

    for (idx, name) in global_names.iter() {
        if name.contains("::new_") {
            if let Some(slot) = globals.get_mut(*idx) {
                let id = slot.resolve_to_value_id(value_store);
                if let Some(ValueCell::Function(fn_idx)) = value_store.get(id) {
                    if *fn_idx < functions.len() {
                        let func = &functions[*fn_idx];
                        debug_println!("[DEBUG set_functions] Проверка: конструктор '{}' в globals[{}] имеет индекс функции {}, имя функции: '{}', arity: {}", name, idx, fn_idx, func.name, func.arity);
                    } else {
                        debug_println!("[DEBUG set_functions] ОШИБКА: конструктор '{}' в globals[{}] имеет индекс функции {} (выходит за границы, всего функций: {})", name, idx, fn_idx, functions.len());
                    }
                }
            }
        }
    }

    for (idx, name) in explicit_global_names_to_add {
        explicit_global_names.insert(idx, name);
    }
}

/// After merging a module into the caller, re-establish __main__ and main in the caller's globals.
pub fn ensure_entry_point_slots(
    target_globals: &mut [GlobalSlot],
    target_global_names: &std::collections::BTreeMap<usize, String>,
    target_functions: &[crate::bytecode::Function],
    store: &mut ValueStore,
) {
    for name in ["__main__", "main"] {
        if let Some(&slot) = target_global_names.iter().find(|(_, n)| n.as_str() == name).map(|(idx, _)| idx) {
            if let Some(fn_idx) = target_functions.iter().position(|f| f.name == name) {
                if slot < target_globals.len() {
                    target_globals[slot] = GlobalSlot::Heap(store.allocate_arena(ValueCell::Function(fn_idx)));
                    debug_println!("[DEBUG ensure_entry_point_slots] Установлен слот '{}' в globals[{}] = Function({})", name, slot, fn_idx);
                }
            }
        }
    }
}

/// Remap Function indices in a value from module export (and Object inner) for merge into caller.
pub fn remap_module_export_value(value: &Value, start_fn: usize) -> Value {
    match value {
        Value::Function(fn_idx) => Value::Function(start_fn + fn_idx),
        Value::ModuleFunction { module_id, local_index } => Value::ModuleFunction { module_id: *module_id, local_index: *local_index },
        Value::Object(obj_rc) => {
            let obj = obj_rc.borrow();
            let mut new_obj = HashMap::new();
            for (k, v) in obj.iter() {
                let inner = match v {
                    Value::Function(i) => Value::Function(start_fn + i),
                    Value::ModuleFunction { module_id, local_index } => Value::ModuleFunction { module_id: *module_id, local_index: *local_index },
                    _ => v.clone(),
                };
                new_obj.insert(k.clone(), inner);
            }
            Value::Object(Rc::new(RefCell::new(new_obj)))
        }
        _ => value.clone(),
    }
}

/// Merge module object exports into caller's globals.
pub fn merge_module_exports_into_globals_into(
    module_object: &Value,
    target_globals: &mut Vec<GlobalSlot>,
    target_global_names: &mut std::collections::BTreeMap<usize, String>,
    store: &mut ValueStore,
    heap: &mut HeavyStore,
) {
    const BUILTIN_COUNT: usize = 75;
    let obj = match module_object {
        Value::Object(rc) => rc.borrow(),
        _ => return,
    };
    let mut names_sorted: Vec<String> = obj.keys().cloned().collect();
    names_sorted.sort();
    for name in names_sorted {
        if name == "__start_function_index" || name == "argv" {
            continue;
        }
        let value = match obj.get(&name) {
            Some(v) => v.clone(),
            None => continue,
        };
        let mut existing_indices: Vec<usize> = target_global_names
            .iter()
            .filter(|(_, n)| n.as_str() == name.as_str())
            .map(|(idx, _)| *idx)
            .collect();
        existing_indices.sort_unstable();
        if let Some(&first_index) = existing_indices.first() {
            if first_index < BUILTIN_COUNT {
                continue;
            }
            if crate::vm::modules::is_known_module(name.as_str()) {
                continue;
            }
            if first_index < target_globals.len() {
                let existing_slot = &target_globals[first_index];
                let existing_val = match existing_slot {
                    GlobalSlot::Inline(tv) => slot_to_value(*tv, store, heap),
                    GlobalSlot::Heap(id) => load_value(*id, store, heap),
                };
                if matches!(existing_val, Value::Function(_) | Value::ModuleFunction { .. }) {
                    continue;
                }
                if matches!(existing_val, Value::Object(_)) {
                    continue;
                }
                if matches!(&value, Value::Null) && !matches!(existing_val, Value::Null) {
                    continue;
                }
            }
            let start_fn = obj.get("__start_function_index").and_then(|v| if let Value::Number(s) = v { Some(*s as usize) } else { None }).unwrap_or(0);
            let value_to_store = remap_module_export_value(&value, start_fn);
            let id = store_value_arena(value_to_store, store, heap);
            for &idx in &existing_indices {
                if idx < target_globals.len() {
                    target_globals[idx] = GlobalSlot::Heap(id);
                } else {
                    target_globals.resize(idx + 1, global_slot::default_global_slot());
                    target_globals[idx] = GlobalSlot::Heap(id);
                }
            }
        } else {
            let start_fn = obj.get("__start_function_index").and_then(|v| if let Value::Number(s) = v { Some(*s as usize) } else { None }).unwrap_or(0);
            let value_to_store = remap_module_export_value(&value, start_fn);
            if let Value::NativeFunction(native_idx) = &value_to_store {
                if *native_idx < BUILTIN_COUNT {
                    if let Some(canonical) = globals::builtin_global_name(*native_idx) {
                        if canonical != name.as_str() {
                            continue;
                        }
                    }
                }
            }
            let new_index = target_globals.len();
            target_globals.push(GlobalSlot::Heap(store_value_arena(value_to_store, store, heap)));
            target_global_names.insert(new_index, name);
        }
    }
}

/// Legacy: merge module VM into caller's buffers.
#[allow(dead_code)]
pub fn merge_globals_from_into(
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
    const BUILTIN_COUNT: usize = 75;
    if other_natives.len() > BUILTIN_COUNT {
        target_natives.extend_from_slice(&other_natives[BUILTIN_COUNT..]);
        debug_println!("[DEBUG merge_globals_from_into] Добавлено нативов из модуля: {}", other_natives.len() - BUILTIN_COUNT);
    }
    debug_println!("[DEBUG merge_globals_from_into] Объединяем глобальные переменные из другого VM");
    let start_function_index = target_functions.len();
    target_functions.extend(other_functions.iter().cloned());
    debug_println!("[DEBUG merge_globals_from_into] start_function_index: {}, всего функций: {}", start_function_index, target_functions.len());

    let mut pairs: Vec<_> = other_global_names.iter().map(|(i, n)| (*i, n.clone())).collect();
    pairs.sort_by(|a, b| a.1.cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
    let mut by_name: std::collections::HashMap<String, (usize, Value)> = std::collections::HashMap::new();
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
        debug_println!("[DEBUG merge_globals_from_into] Объединяем '{}' (index: {})", name, index);
        let value_to_store = match value {
            Value::Function(function_index) => Value::Function(start_function_index + function_index),
            Value::Object(obj_rc) => {
                let obj = obj_rc.borrow();
                let mut new_obj = HashMap::new();
                for (key, val) in obj.iter() {
                    let updated_val = match val {
                        Value::Function(function_index) => Value::Function(start_function_index + *function_index),
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
                        target_globals.resize(canonical_slot + 1, global_slot::default_global_slot());
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
            target_globals.push(GlobalSlot::Heap(store_value_arena(value_to_store, store, heap)));
            target_global_names.insert(new_index, name.clone());
        }
    }
    debug_println!("[DEBUG merge_globals_from_into] Объединение завершено. Всего глобальных переменных: {}", target_global_names.len());
}

/// Legacy: merges another VM's globals into target (self). Different prefer logic than merge_globals_from_into.
#[allow(dead_code)]
pub fn merge_globals_from(
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
    const BUILTIN_COUNT: usize = 75;
    if other_natives.len() > BUILTIN_COUNT {
        target_natives.extend_from_slice(&other_natives[BUILTIN_COUNT..]);
        debug_println!("[DEBUG merge_globals_from] Добавлено нативов из модуля: {}", other_natives.len() - BUILTIN_COUNT);
    }
    debug_println!("[DEBUG merge_globals_from] Объединяем глобальные переменные из другого VM");
    debug_println!("[DEBUG merge_globals_from] Функций в текущем VM: {}", target_functions.len() - other_functions.len());
    debug_println!("[DEBUG merge_globals_from] Функций в другом VM: {}", other_functions.len());
    debug_println!("[DEBUG merge_globals_from] Глобальных переменных в другом VM: {}", other_global_names.len());
    let start_function_index = target_functions.len();
    target_functions.extend(other_functions.iter().cloned());
    debug_println!("[DEBUG merge_globals_from] Начальный индекс функций: {} (функций в текущем VM: {})", start_function_index, target_functions.len() - other_functions.len());
    debug_println!("[DEBUG merge_globals_from] Всего функций после объединения: {}", target_functions.len());

    let mut pairs: Vec<_> = other_global_names.iter().map(|(i, n)| (*i, n.clone())).collect();
    pairs.sort_by(|a, b| a.1.cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
    let mut by_name: std::collections::HashMap<String, (usize, Value)> = std::collections::HashMap::new();
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
        debug_println!("[DEBUG merge_globals_from] Объединяем '{}' (index: {})", name, index);
        let value_to_store = match value {
            Value::Function(function_index) => {
                let new_function_index = start_function_index + function_index;
                debug_println!("[DEBUG merge_globals_from] Обновляем индекс функции для '{}': {} -> {}", name, function_index, new_function_index);
                Value::Function(new_function_index)
            }
            Value::Object(obj_rc) => {
                let obj = obj_rc.borrow();
                let mut new_obj = HashMap::new();
                for (key, val) in obj.iter() {
                    let updated_val = match val {
                        Value::Function(function_index) => Value::Function(start_function_index + *function_index),
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
                debug_println!("[DEBUG merge_globals_from] Пропуск перезаписи '{}' (встроенный модуль)", name);
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
            debug_println!("[DEBUG merge_globals_from] '{}' уже существует, перезаписываем globals[{}] ({})", name, existing_index, val_type);
            if name == "Config" || name == "DatabaseConfig" {
                debug_println!("[DEBUG merge_globals_from] Config/DatabaseConfig: '{}' -> слот {} ({})", name, existing_index, val_type);
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
            target_globals.push(GlobalSlot::Heap(store_value_arena(value_to_store, store, heap)));
            target_global_names.insert(new_index, name.clone());
            debug_println!("[DEBUG merge_globals_from] Создана новая глобальная переменная '{}' в globals[{}] ({})", name, new_index, val_type);
            if name == "Config" || name == "DatabaseConfig" {
                debug_println!("[DEBUG merge_globals_from] Config/DatabaseConfig: '{}' -> новый слот {} ({})", name, new_index, val_type);
            }
        }
    }
    debug_println!("[DEBUG merge_globals_from] Объединение завершено. Всего глобальных переменных: {}", target_global_names.len());
}
