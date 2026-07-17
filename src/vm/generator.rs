//! Выполнение `stream fn`: шаг генератора до yield или завершения.

use crate::bytecode::{Function, OpCode};
use crate::common::error::LangError;
use crate::common::value::{GeneratorState, Value};
use crate::common::TaggedValue;
use crate::vm::frame::CallFrame;
use crate::vm::store_convert::load_value;
use crate::vm::store_convert::store_value;
use crate::vm::types::{PendingGeneratorSend, VMStatus};
use crate::vm::vm::Vm;

fn build_stream_slots_from_args(
    function: &Function,
    args: &[Value],
    store: &mut crate::common::ValueStore,
    heap: &mut crate::vm::heavy_store::HeavyStore,
) -> Result<Vec<TaggedValue>, LangError> {
    if args.len() != function.arity {
        return Err(LangError::runtime_error(
            format!(
                "stream fn: expected {} arguments, got {}",
                function.arity,
                args.len()
            ),
            0,
        ));
    }
    let args_tvs: Vec<TaggedValue> = args
        .iter()
        .map(|a| TaggedValue::from_heap(store_value(a.clone(), store, heap)))
        .collect();
    let mut slots = Vec::new();
    for (i, &tv) in args_tvs.iter().enumerate() {
        if i >= slots.len() {
            slots.resize(i + 1, TaggedValue::null());
        }
        slots[i] = tv;
    }
    Ok(slots)
}

/// Первый в потоке `Yield` / `YieldAwaitInput` (пропускает `Constant` и т.д.).
fn first_yield_opcode_kind(code: &[OpCode]) -> Option<bool> {
    for op in code {
        match op {
            OpCode::Yield(_) => return Some(false),
            OpCode::YieldAwaitInput(_, _) => return Some(true),
            _ => {}
        }
    }
    None
}

#[derive(Clone)]
pub(crate) enum GeneratorResumeMode {
    /// Обычный `.next()` / итератор.
    Next,
    /// `gen.final()` — дожать с подстановкой RHS yield в слот yield-await.
    NextFinalDrain,
    Send(Value),
}

/// `for_iterable`: true — сообщение для `for x in gen`; false — для `.next()` / `.send()`.
pub(crate) fn run_generator_resume(
    vm: &mut Vm,
    gen: &mut GeneratorState,
    mode: GeneratorResumeMode,
    for_iterable: bool,
) -> Result<Option<Value>, LangError> {
    if gen.finished {
        return match mode {
            GeneratorResumeMode::Next | GeneratorResumeMode::NextFinalDrain => Ok(None),
            GeneratorResumeMode::Send(_) => Err(LangError::runtime_error(
                "generator.send() on finished generator".to_string(),
                0,
            )),
        };
    }

    if matches!(&mode, GeneratorResumeMode::Next) {
        if let Some(v) = gen.pending_deferred_yield.take() {
            return Ok(Some(v));
        }
    }

    match &mode {
        GeneratorResumeMode::Next => {
            if gen.waiting_for_input {
                if for_iterable {
                    return Err(LangError::runtime_error(
                        "generator requires input inside for-in; use send() outside the loop"
                            .to_string(),
                        0,
                    ));
                }
                // `.next()` without `.send()` at a yield-await: substitute the yielded RHS value into the
                // assign slot (same as Python `send(None)`). Using null here made `x`/`d` null after
                // `x = ireturn 40` / `d = return 50`, so `ereturn (x+d)*n` became 0 instead of 900.
                vm.pending_generator_send = Some(PendingGeneratorSend::NextDefaultRhs);
            } else {
                vm.pending_generator_send = None;
            }
        }
        GeneratorResumeMode::NextFinalDrain => {
            if gen.waiting_for_input {
                vm.pending_generator_send = Some(PendingGeneratorSend::NextDefaultRhs);
            } else {
                vm.pending_generator_send = None;
            }
        }
        GeneratorResumeMode::Send(v) => {
            let cold = gen.pending_args.is_some();
            if !gen.waiting_for_input && !cold {
                return Err(LangError::runtime_error(
                    "generator.send() is not valid when the generator is not at a yield-await point (use .next())".to_string(),
                    0,
                ));
            }
            if cold {
                gen.pending_first_send = Some(v.clone());
            } else {
                vm.pending_generator_send = Some(PendingGeneratorSend::Explicit(v.clone()));
            }
        }
    }

    // До push фрейма stream fn: обычно один фрейм вызывающего кода (main).
    let frames_base = vm.frame_len();

    if gen.pending_args.is_some() {
        let args = gen.pending_args.take().unwrap();
        let function = vm
            .get_functions()
            .get(gen.fn_index)
            .cloned()
            .ok_or_else(|| {
                LangError::runtime_error("generator: invalid function index".to_string(), 0)
            })?;
        if !function.is_stream {
            return Err(LangError::runtime_error(
                "generator: not a stream function".to_string(),
                0,
            ));
        }
        gen.slots = vm.with_stores_mut(|store, heap| {
            build_stream_slots_from_args(&function, &args, store, heap)
        })?;
        gen.ip = 0;
    }

    let function = vm
        .get_functions()
        .get(gen.fn_index)
        .cloned()
        .ok_or_else(|| {
            LangError::runtime_error("generator: invalid function index".to_string(), 0)
        })?;
    if !function.is_stream {
        return Err(LangError::runtime_error(
            "generator: not a stream function".to_string(),
            0,
        ));
    }

    if gen.pending_first_send.is_some() {
        if first_yield_opcode_kind(&function.chunk.code) == Some(false) {
            gen.pending_first_send = None;
            return Err(LangError::runtime_error(
                "generator.send() is not valid when the generator is not at a yield-await point (use .next())"
                    .to_string(),
                0,
            ));
        }
    }

    if let Some(s) = vm.pending_generator_send.take() {
        if gen.ip < function.chunk.code.len() {
            if matches!(&function.chunk.code[gen.ip], OpCode::Yield(_)) {
                vm.pending_generator_send = Some(s);
                return Err(LangError::runtime_error(
                    "generator.send() is not valid when the generator is not at a yield-await point (use .next())".to_string(),
                    0,
                ));
            }
        }
        vm.pending_generator_send = Some(s);
    }

    'resume: loop {
        let stack_start = vm.stack_len();
        let mut frame = vm.with_stores_mut(|store, heap| {
            CallFrame::new(function.clone(), gen.fn_index, stack_start, store, heap)
        });
        frame.slots = gen.slots.clone();
        frame.ip = gen.ip;
        vm.push_frame(frame);

        loop {
            match vm.step()? {
                VMStatus::Continue => {}
                VMStatus::GeneratorYield(id) => {
                    let v = load_value(id, vm.value_store(), vm.heavy_store());
                    let popped = vm.pop_last_frame_for_generator();
                    let Some(f) = popped else {
                        vm.truncate_frames_to(frames_base);
                        return Err(LangError::runtime_error(
                            "internal: stream fn yield with no frame on stack".to_string(),
                            0,
                        ));
                    };
                    gen.slots = f.slots;
                    gen.ip = f.ip;
                    gen.waiting_for_input = false;
                    vm.truncate_frames_to(frames_base);
                    if let Some(rhs) = vm.pending_send_rhs_return.take() {
                        gen.pending_deferred_yield = Some(v);
                        return Ok(Some(rhs));
                    }
                    if let Some(first) = gen.cold_send_first_yield.take() {
                        return Ok(Some(first));
                    }
                    return Ok(Some(v));
                }
                VMStatus::GeneratorYieldAwait(yield_id, _) => {
                    let popped = vm.pop_last_frame_for_generator();
                    let Some(f) = popped else {
                        vm.truncate_frames_to(frames_base);
                        return Err(LangError::runtime_error(
                            "internal: stream fn yield-await with no frame on stack".to_string(),
                            0,
                        ));
                    };
                    gen.slots = f.slots;
                    gen.ip = f.ip.saturating_sub(1);
                    gen.waiting_for_input = true;
                    let v = load_value(yield_id, vm.value_store(), vm.heavy_store());
                    vm.yield_await_resume_value = Some(v.clone());
                    vm.pending_generator_send = None;
                    if for_iterable {
                        return Err(LangError::runtime_error(
                            "generator requires input inside for-in; use send() outside the loop"
                                .to_string(),
                            0,
                        ));
                    }
                    if let Some(q) = gen.pending_first_send.take() {
                        gen.cold_send_first_yield = Some(v.clone());
                        vm.pending_generator_send = Some(PendingGeneratorSend::Explicit(q));
                        continue 'resume;
                    }
                    vm.truncate_frames_to(frames_base);
                    return Ok(Some(v));
                }
                VMStatus::GeneratorDone(final_id) => {
                    gen.finished = true;
                    gen.waiting_for_input = false;
                    gen.slots.clear();
                    gen.ip = 0;
                    vm.pending_send_rhs_return = None;
                    if let Some(id) = final_id {
                        gen.final_value = Some(load_value(id, vm.value_store(), vm.heavy_store()));
                    }
                    vm.truncate_frames_to(frames_base);
                    if let Some(first) = gen.cold_send_first_yield.take() {
                        return Ok(Some(first));
                    }
                    return Ok(None);
                }
                VMStatus::Return(_) => {
                    gen.finished = true;
                    gen.waiting_for_input = false;
                    let _ = vm.pop_last_frame_for_generator();
                    vm.truncate_frames_to(frames_base);
                    return Err(LangError::runtime_error(
                        "internal: Return in stream fn (expected Yield/GeneratorDone)".to_string(),
                        0,
                    ));
                }
                VMStatus::FrameEnded => {
                    gen.finished = true;
                    gen.waiting_for_input = false;
                    vm.pending_send_rhs_return = None;
                    vm.truncate_frames_to(frames_base);
                    return Ok(None);
                }
            }
        }
    }
}

/// Один шаг итератора: `Some(v)` при yield, `None` при завершении.
pub fn run_generator_next(
    vm: &mut Vm,
    gen: &mut GeneratorState,
) -> Result<Option<Value>, LangError> {
    run_generator_resume(vm, gen, GeneratorResumeMode::Next, true)
}
