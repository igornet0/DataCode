//! Chunk patching: update LoadGlobal/StoreGlobal indices to match VM's global_names.
//! Extracted from vm.rs for Phase 7 (VM Facade).

use crate::debug_println;
use crate::common::value::Value;
use crate::common::value_store::ValueStore;
use crate::vm::global_slot::GlobalSlot;
use crate::vm::store_convert::load_value;

/// Sentinel index for model_config class load in Settings subclass constructors (must match compiler).
pub const MODEL_CONFIG_CLASS_LOAD_INDEX: usize = 0x0FFF_FFFF;

/// Обновляет индексы глобалов в chunk по переданной карте имён.
/// globals_for_verify: Option<&mut [GlobalSlot]>; при проверке слот материализуется через resolve_to_value_id + load_value (с кэшем).
/// store/heap: когда None, проверка по слотам не выполняется (для вызовов без доступа к store/heap).
/// argv_slot_index: когда Some(slot), имя "argv" всегда мапится на этот слот (после ImportFrom слот 79 может стать load_settings).
/// resolve_undefined_sentinel: когда false (main chunk после ImportFrom), не резолвим UNDEFINED_GLOBAL_SENTINEL,
///   чтобы LoadGlobal(MAX) для неимпортированных имён (напр. "settings") оставался MAX и выдавал NameError.
#[allow(clippy::too_many_arguments)]
pub fn update_chunk_indices_from_names(
    chunk: &mut crate::bytecode::Chunk,
    global_names: &std::collections::BTreeMap<usize, String>,
    mut globals_for_verify: Option<&mut [GlobalSlot]>,
    mut store: Option<&mut ValueStore>,
    heap: Option<&crate::vm::heavy_store::HeavyStore>,
    argv_slot_index: Option<usize>,
    resolve_undefined_sentinel: bool,
) {
    if std::env::var("DATACODE_DEBUG").is_ok() {
        eprintln!("[update_chunk_indices_from_names] start mapping globals...");
        for (old_idx, name) in &chunk.global_names {
            let exists = global_names.values().any(|n| n == name);
            eprintln!(
                "  mapping old_idx={} name={} exists_in_caller_global_names={}",
                old_idx, name, exists
            );
        }
    }
    // Phase 1: Build old_idx → real_idx mapping (no bytecode changes).
    let mut old_to_real: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
    let mut name_index_pairs: Vec<_> = chunk.global_names.iter().map(|(i, n)| (*i, n.clone())).collect();
    name_index_pairs.sort_by(|a, b| a.1.cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
    let can_prefer_non_null = globals_for_verify.is_some() && store.is_some() && heap.is_some();
    for (old_idx, name) in &name_index_pairs {
        let real_idx: Option<usize> = if name == "argv" && argv_slot_index.is_some() {
            argv_slot_index
        } else {
            let matching_indices: Vec<usize> = global_names
                .iter()
                .filter(|(_, n)| *n == name)
                .map(|(idx, _)| *idx)
                .collect();
            if matching_indices.is_empty() {
                None
            } else if can_prefer_non_null {
                const BUILTIN_COUNT: usize = 75;
                if let (Some(globals), Some(store), Some(heap)) = (globals_for_verify.as_deref_mut(), store.as_deref_mut(), heap) {
                    let want_function = name.contains("::new_");
                    let non_null: Vec<usize> = matching_indices
                        .iter()
                        .filter(|&&idx| {
                            idx >= BUILTIN_COUNT
                                && idx < globals.len()
                                && !matches!(
                                    load_value(globals[idx].resolve_to_value_id(store), store, heap),
                                    Value::Null
                                )
                        })
                        .copied()
                        .collect();
                    let preferred: Option<usize> = if want_function {
                        non_null
                            .iter()
                            .filter(|&&idx| {
                                idx < globals.len()
                                    && matches!(
                                        load_value(globals[idx].resolve_to_value_id(store), store, heap),
                                        Value::Function(_) | Value::ModuleFunction { .. }
                                    )
                            })
                            .min()
                            .copied()
                    } else {
                        None
                    };
                    preferred
                        .or_else(|| non_null.into_iter().min())
                        .or_else(|| matching_indices.iter().filter(|&&i| i >= BUILTIN_COUNT).min().copied())
                        .or_else(|| matching_indices.iter().min().copied())
                } else {
                    matching_indices.iter().min().copied()
                }
            } else {
                matching_indices.iter().min().copied()
            }
        };
        if let Some(r) = real_idx {
            let r_final = if name == "argv" && argv_slot_index == Some(*old_idx) && r != *old_idx {
                *old_idx
            } else {
                r
            };
            if *old_idx != r_final {
                let would_overwrite = chunk.global_names.get(&r_final).map(|existing| {
                    existing != name
                        && global_names
                            .iter()
                            .filter(|(_, n)| *n == existing)
                            .map(|(idx, _)| *idx)
                            .all(|idx| idx == r_final)
                }).unwrap_or(false);
                let force_argv = name == "argv" && argv_slot_index == Some(r_final);
                if !would_overwrite || force_argv {
                    old_to_real.insert(*old_idx, r_final);
                    debug_println!("[DEBUG update_chunk_indices] Маппинг '{}': {} -> {} (argv forced={})", name, old_idx, r_final, name == "argv" && argv_slot_index.is_some());
                } else {
                    debug_println!("[DEBUG update_chunk_indices] Пропуск маппинга '{}': {} -> {} (слот {} занят другим именем)", name, old_idx, r_final, r_final);
                }
            }
        } else {
            if name == "argv" || crate::common::debug::verbose_constructor_debug() {
                eprintln!("[update_chunk_indices] no match for name '{}' (old_idx {}), chunk LoadGlobal will not be patched", name, old_idx);
            }
        }
    }
    if crate::common::debug::verbose_constructor_debug() {
        for (old_idx, name) in &name_index_pairs {
            if let Some(&real_idx) = old_to_real.get(old_idx) {
                eprintln!("[update_chunk_indices] '{}' old_idx {} -> real_idx {}", name, old_idx, real_idx);
            } else {
                eprintln!("[update_chunk_indices] '{}' old_idx {} -> no match (not in caller global_names)", name, old_idx);
            }
        }
    }
    const UNDEFINED_GLOBAL_SENTINEL: usize = usize::MAX;
    if resolve_undefined_sentinel {
        if let Some(name) = chunk.global_names.get(&UNDEFINED_GLOBAL_SENTINEL) {
            let matching: Vec<usize> = global_names
                .iter()
                .filter(|(_, n)| *n == name)
                .map(|(idx, _)| *idx)
                .collect();
            if let Some(&real_idx) = matching.iter().min() {
                old_to_real.insert(UNDEFINED_GLOBAL_SENTINEL, real_idx);
                debug_println!("[DEBUG update_chunk_indices] Маппинг sentinel '{}': MAX -> {}", name, real_idx);
            }
        }
    }
    let needs_sentinel = chunk.code.iter().any(|op| {
        matches!(op, crate::bytecode::OpCode::LoadGlobal(i) if *i == MODEL_CONFIG_CLASS_LOAD_INDEX)
    });
    if needs_sentinel {
        if let Some(name) = chunk.global_names.get(&MODEL_CONFIG_CLASS_LOAD_INDEX) {
            let matching_indices: Vec<usize> = global_names
                .iter()
                .filter(|(_, n)| *n == name)
                .map(|(idx, _)| *idx)
                .collect();
            let real_idx = if name.as_str() == "__constructing_class__" {
                matching_indices.into_iter().min()
            } else {
                let mut class_slots = Vec::new();
                if let (Some(globals), Some(store), Some(heap)) = (globals_for_verify.as_deref_mut(), store.as_deref_mut(), heap) {
                    for &idx in &matching_indices {
                        if idx < globals.len() {
                            let id = globals[idx].resolve_to_value_id(store);
                            let v = load_value(id, store, heap);
                            if let Value::Object(obj_rc) = &v {
                                let obj = obj_rc.borrow();
                                if obj.get("__class_name").is_some() && obj.get("model_config").is_some() {
                                    class_slots.push(idx);
                                }
                            }
                        }
                    }
                }
                class_slots.into_iter().max().or_else(|| matching_indices.into_iter().max())
            };
            if let Some(real_idx) = real_idx {
                old_to_real.insert(MODEL_CONFIG_CLASS_LOAD_INDEX, real_idx);
                debug_println!("[DEBUG update_chunk_indices] Маппинг model_config (sentinel) класс '{}': sentinel -> globals[{}]", name, real_idx);
                if let (Some(globals), Some(store), Some(heap)) = (globals_for_verify.as_deref_mut(), store.as_deref_mut(), heap) {
                    if real_idx < globals.len() {
                        let id = globals[real_idx].resolve_to_value_id(store);
                        let v = load_value(id, store, heap);
                        let slot_type = match &v {
                            Value::Object(_) => "Object",
                            Value::Function(_) | Value::ModuleFunction { .. } => "Function",
                            Value::Null => "Null",
                            _ => "Other",
                        };
                        debug_println!("[DEBUG update_chunk_indices] sentinel '{}' -> globals[{}], значение в слоте: {}", name, real_idx, slot_type);
                        if let Value::Object(obj_rc) = &v {
                            let obj = obj_rc.borrow();
                            if let Some(Value::String(actual_name)) = obj.get("__class_name") {
                                debug_println!("[DEBUG update_chunk_indices] globals[{}].__class_name = '{}'", real_idx, actual_name);
                                if actual_name != name {
                                    debug_println!(
                                        "[DEBUG update_chunk_indices] WARNING: ожидался класс '{}', в слоте {} — класс '{}'",
                                        name, real_idx, actual_name
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Phase 2: Single pass over bytecode
    let mut updated_count = 0usize;
    for opcode in &mut chunk.code {
        match opcode {
            crate::bytecode::OpCode::LoadGlobal(idx) => {
                if let Some(&real) = old_to_real.get(idx) {
                    *opcode = crate::bytecode::OpCode::LoadGlobal(real);
                    updated_count += 1;
                }
            }
            crate::bytecode::OpCode::StoreGlobal(idx) => {
                if let Some(&real) = old_to_real.get(idx) {
                    *opcode = crate::bytecode::OpCode::StoreGlobal(real);
                    updated_count += 1;
                }
            }
            _ => {}
        }
    }
    if !old_to_real.is_empty() {
        debug_println!("[DEBUG update_chunk_indices] Обновлено {} инструкций по {} маппингам", updated_count, old_to_real.len());
    }

    // Phase 3: Update chunk.global_names
    let to_insert: Vec<_> = old_to_real
        .iter()
        .filter_map(|(old_idx, real_idx)| {
            chunk.global_names.get(old_idx).map(|name| (*real_idx, name.clone()))
        })
        .collect();
    for old_idx in old_to_real.keys() {
        chunk.global_names.remove(old_idx);
    }
    for (real_idx, name) in to_insert {
        if let Some(existing) = chunk.global_names.get(&real_idx) {
            if existing != &name && !old_to_real.contains_key(&real_idx) {
                debug_println!("[DEBUG update_chunk_indices] Phase3: пропуск вставки (real_idx {} уже = '{}', не перезаписываем на '{}')", real_idx, existing, name);
                continue;
            }
        }
        chunk.global_names.insert(real_idx, name);
    }
    if std::env::var("DATACODE_DEBUG").is_ok() {
        eprintln!("[update_chunk_indices_from_names] AFTER mapping caller_global_names keys:");
        for (idx, name) in global_names.iter() {
            eprintln!("  idx={} name={}", idx, name);
        }
        if !global_names.values().any(|n| n == "create_all") {
            eprintln!("[WARNING] create_all STILL MISSING AFTER update_chunk_indices_from_names");
        }
    }
}
