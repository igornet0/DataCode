//! Run orchestration: setup context, execute main loop, return result.
//! Keeps vm.rs focused on VM state and accessors.

use crate::bytecode::Chunk;
use crate::common::{error::LangError, value::Value, value_store::{ValueId, NULL_VALUE_ID}};
use crate::vm::context_guards::{ClearScriptArgvGuard, MlContextGuard, PlotContextGuard, RunContextGuard};
use crate::vm::frame::CallFrame;
use crate::vm::store_convert::{load_value, tagged_to_value_id};
use crate::vm::types::VMStatus;
use crate::vm::global_slot::GlobalSlot;
use crate::vm::vm::{RestoreArgvIdGuard, Vm};
use std::path::PathBuf;

/// argv_patch: when Some((argv_slot_index, old_indices, argv_value_id)), the chunk is cloned and any LoadGlobal(old_idx)
/// in the clone is replaced with LoadGlobal(argv_slot_index). If argv_value_id is Some(id), that value is written
/// to globals[argv_slot_index] immediately before the execution loop.
pub fn execute_run(
    vm: &mut Vm,
    chunk: &Chunk,
    argv_patch: Option<(usize, &[usize], Option<ValueId>)>,
) -> Result<Value, LangError> {
    let saved_argv_id = vm.get_current_argv_value_id();
    let _argv_guard = RestoreArgvIdGuard(vm as *mut Vm, saved_argv_id);
    let argv_value_id_to_use = argv_patch.and_then(|(_, _, id)| id);
    vm.set_current_argv_value_id(argv_value_id_to_use);
    vm.set_argv_old_indices(argv_patch.map(|(_, o, _)| o.to_vec()));

    let run_ctx = crate::vm::run_context::RunContext {
        base_path: vm.get_base_path(),
        project_root: vm.get_project_root(),
        executing_lib: false,
        dpm_package_paths: crate::vm::file_import::get_dpm_package_paths(),
        smb_manager: crate::vm::file_ops::get_smb_manager(),
        argv_value_id: argv_value_id_to_use,
    };
    crate::vm::run_context::RunContext::set_current(run_ctx);

    if let Some(id) = argv_value_id_to_use {
        crate::vm::run_context::RunContext::set_script_argv_value_id(Some(id));
        let v = load_value(id, vm.value_store(), vm.heavy_store());
        let non_empty_argv = matches!(&v, Value::Array(a) if !a.borrow().is_empty());
        crate::vm::run_context::RunContext::set_restored_script_argv_after_import(
            if non_empty_argv { Some(id) } else { None },
        );
    }

    let base_path_ptr: *mut Option<PathBuf> = vm.get_base_path_mut_ptr();
    let _run_guard = RunContextGuard(base_path_ptr);
    let _script_argv_guard = ClearScriptArgvGuard(argv_value_id_to_use.is_some());
    crate::vm::file_import::set_base_path(vm.get_base_path());

    if let Some(ctx) = vm.take_ml_context() {
        crate::ml::MlContext::set_current(ctx);
    } else {
        let _ = crate::ml::MlContext::take_current();
    }
    let ml_ctx_ptr = vm.get_ml_context_mut_ptr();
    let _ml_guard = MlContextGuard(ml_ctx_ptr);

    let plot_ctx = vm.take_plot_context().unwrap_or_else(crate::plot::PlotContext::new);
    crate::plot::PlotContext::set_current(plot_ctx);
    let plot_ctx_ptr = vm.get_plot_context_mut_ptr();
    let _plot_guard = PlotContextGuard(plot_ctx_ptr);

    vm.merge_global_names_from_chunk(chunk);

    const CONSTRUCTING_CLASS_NAME: &str = "__constructing_class__";
    if !vm.global_names_contains(CONSTRUCTING_CLASS_NAME) {
        vm.ensure_global_slot(CONSTRUCTING_CLASS_NAME);
    }

    #[cfg(feature = "profile")]
    crate::vm::profile::set();

    let mut chunk_to_run = chunk.clone();
    if let Some((argv_slot_index, _old_indices, _)) = argv_patch.as_ref() {
        chunk_to_run.global_names.insert(*argv_slot_index, "argv".to_string());
        for opcode in &mut chunk_to_run.code {
            if let crate::bytecode::OpCode::LoadGlobal(idx) = opcode {
                let name_is_argv = chunk_to_run.global_names.get(idx).map(|n| n.as_str()) == Some("argv");
                if name_is_argv && *idx != *argv_slot_index {
                    *idx = *argv_slot_index;
                }
            }
        }
    }

    let function = crate::bytecode::Function::new("<main>".to_string(), 0);
    let mut function = function;
    function.chunk = chunk_to_run;
    let frame = vm.with_stores_mut(|store, heap| CallFrame::new(function, 0, store, heap));
    vm.push_frame(frame);

    if let Some((argv_slot_index, _old_indices, argv_value_id)) = argv_patch {
        vm.set_current_argv_value_id(argv_value_id);
        if let Some(argv_id) = argv_value_id {
            vm.ensure_globals_len(argv_slot_index + 1);
            vm.set_global_slot(argv_slot_index, GlobalSlot::Heap(argv_id));
        }
    } else {
        vm.set_current_argv_value_id(None);
    }

    loop {
        match vm.step()? {
            VMStatus::Continue => {}
            VMStatus::Return(id) => {
                #[cfg(feature = "profile")]
                if let Some(stats) = crate::vm::profile::take() {
                    crate::vm::profile::print_stats(&stats);
                }
                return Ok(load_value(id, vm.value_store(), vm.heavy_store()));
            }
            VMStatus::FrameEnded => break,
        }
    }

    #[cfg(feature = "profile")]
    if let Some(stats) = crate::vm::profile::take() {
        crate::vm::profile::print_stats(&stats);
    }

    if !vm.stack_is_empty() {
        let tv = vm.stack_pop().unwrap();
        let id = tagged_to_value_id(tv, vm.value_store_mut());
        Ok(load_value(id, vm.value_store(), vm.heavy_store()))
    } else {
        Ok(load_value(NULL_VALUE_ID, vm.value_store(), vm.heavy_store()))
    }
}
