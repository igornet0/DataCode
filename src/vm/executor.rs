// Opcode execution for VM (Stage 1: stack/globals as Vec<ValueId>; one borrow store per instruction)

use crate::bytecode::OpCode;
use crate::common::{error::LangError, value::Value, value_store::NULL_VALUE_ID, TaggedValue};
use crate::debug_println;
use crate::vm::exception;
use crate::vm::exceptions::ExceptionHandler;
use crate::vm::frame::CallFrame;
use crate::vm::global_slot::GlobalSlot;
use crate::vm::interpreter::{
    arithmetic, bitwise, comparison, control_flow, element_ops, for_iterable, grid_ops, memory,
    object, special_init, stack_ops,
};
use crate::vm::module_system::import_handler;
use crate::vm::runtime::call_engine;
use crate::vm::stack;
use crate::vm::store_convert::{load_value, slot_to_value, store_value, tagged_to_value_id};
use crate::vm::types::{PendingGeneratorSend, VMStatus};

// Re-export for backward compatibility (call_engine, import_handler, memory use executor::global_index_by_name)
pub(crate) use crate::vm::global_utils::{global_index_by_name, global_indices_by_name};

/// Execute one step of the VM - get next instruction and execute it
pub fn step(frames: &mut Vec<CallFrame>) -> Result<Option<(OpCode, usize)>, LangError> {
    loop {
        let frame = match frames.last_mut() {
            Some(f) => f,
            None => return Ok(None),
        };

        if frame.ip >= frame.function.chunk.code.len() {
            // Frame exhausted (e.g. empty method body); pop and continue with caller
            frames.pop();
            continue;
        }

        let ip = frame.ip;
        let instruction = frame.function.chunk.code[ip].clone();
        let line = frame.function.chunk.get_line(ip);
        frame.ip += 1;

        return Ok(Some((instruction, line)));
    }
}

/// Execute a single instruction
/// Returns VMStatus indicating what to do next. vm_ptr used for VM_CALL_CONTEXT and module loading.
pub(crate) fn execute_instruction(
    instruction: OpCode,
    line: usize,
    stack: &mut Vec<TaggedValue>,
    frames: &mut Vec<CallFrame>,
    globals: &mut Vec<GlobalSlot>,
    global_names: &mut std::collections::BTreeMap<usize, String>,
    explicit_global_names: &std::collections::BTreeMap<usize, String>,
    functions: &mut Vec<crate::bytecode::Function>,
    natives: &mut Vec<crate::vm::host::HostEntry>,
    exception_handlers: &mut Vec<ExceptionHandler>,
    error_type_table: &mut Vec<String>,
    explicit_relations: &mut Vec<crate::vm::types::ExplicitRelation>,
    explicit_primary_keys: &mut Vec<crate::vm::types::ExplicitPrimaryKey>,
    loaded_modules: &mut std::collections::HashSet<String>,
    abi_natives: &mut Vec<crate::abi::NativeAbiFn>,
    loaded_native_libraries: &mut Vec<libloading::Library>,
    value_store: &mut crate::common::ValueStore,
    heavy_store: &mut crate::vm::heavy_store::HeavyStore,
    native_args_buffer: &mut Vec<Value>,
    reusable_native_arg_ids: &mut Vec<crate::common::value_store::ValueId>,
    reusable_all_popped: &mut Vec<Value>,
    vm_ptr: *mut crate::vm::vm::Vm,
) -> Result<VMStatus, LangError> {
    #[cfg(feature = "profile")]
    crate::vm::profile::set_current_opcode(&instruction);

    let frame = frames.last_mut().unwrap();
    let current_ip = frame.ip - 1; // IP уже инкрементирован в step()

    // Hot-first dispatch for A* inner loops (`--features threaded_dispatch`).
    #[cfg(feature = "threaded_dispatch")]
    {
        match instruction {
            OpCode::LoadLocal(index) => {
                return stack_ops::op_load_local(index, current_ip, stack, frames)
            }
            OpCode::StoreLocal(index) => {
                return stack_ops::op_store_local(
                    index,
                    stack,
                    frames,
                    exception_handlers,
                    value_store,
                    heavy_store,
                )
            }
            OpCode::Less => {
                return comparison::op_less(
                    stack,
                    frames,
                    exception_handlers,
                    value_store,
                    heavy_store,
                )
            }
            OpCode::In => {
                return comparison::op_in(
                    line,
                    stack,
                    frames,
                    exception_handlers,
                    value_store,
                    heavy_store,
                )
            }
            OpCode::InIntegral => {
                return comparison::op_in_integral(
                    line,
                    stack,
                    frames,
                    exception_handlers,
                    value_store,
                    heavy_store,
                )
            }
            OpCode::NotInIntegral => {
                return comparison::op_not_in_integral(
                    line,
                    stack,
                    frames,
                    exception_handlers,
                    value_store,
                    heavy_store,
                )
            }
            OpCode::InGridBounds => {
                return comparison::op_in_grid_bounds(
                    stack,
                    frames,
                    exception_handlers,
                    value_store,
                    heavy_store,
                )
            }
            OpCode::InGridBoundsOut => {
                return comparison::op_in_grid_bounds_out(
                    stack,
                    frames,
                    exception_handlers,
                    value_store,
                    heavy_store,
                )
            }
            OpCode::FScoreStaleCheck => {
                return comparison::op_f_score_stale_check(
                    stack,
                    frames,
                    exception_handlers,
                    value_store,
                    heavy_store,
                )
            }
            OpCode::DictGetIntegralLt => {
                return comparison::op_dict_get_integral_lt(
                    stack,
                    frames,
                    exception_handlers,
                    value_store,
                    heavy_store,
                )
            }
            OpCode::DictIndexIntegralAddImm(addend) => {
                return comparison::op_dict_index_integral_add_imm(
                    addend,
                    stack,
                    frames,
                    exception_handlers,
                    value_store,
                    heavy_store,
                )
            }
            OpCode::NotEqual => {
                return comparison::op_not_equal(
                    stack,
                    frames,
                    exception_handlers,
                    value_store,
                    heavy_store,
                )
            }
            OpCode::Equal => {
                return comparison::op_equal(
                    stack,
                    frames,
                    exception_handlers,
                    value_store,
                    heavy_store,
                )
            }
            OpCode::ObjectIndexIntegral => {
                return element_ops::op_object_index_integral(
                    line,
                    stack,
                    frames,
                    globals,
                    global_names,
                    functions,
                    natives,
                    exception_handlers,
                    value_store,
                    heavy_store,
                    vm_ptr,
                )
            }
            OpCode::ObjectSetIntegral => {
                return element_ops::op_object_set_integral(
                    line,
                    stack,
                    frames,
                    globals,
                    global_names,
                    functions,
                    natives,
                    exception_handlers,
                    value_store,
                    heavy_store,
                )
            }
            OpCode::GetArrayElement => {
                return element_ops::op_get_array_element(
                    line,
                    stack,
                    frames,
                    globals,
                    global_names,
                    functions,
                    natives,
                    exception_handlers,
                    value_store,
                    heavy_store,
                    vm_ptr,
                )
            }
            OpCode::HeappopUnpack2(f_slot, n_slot) => {
                return stack_ops::op_heappop_unpack2(
                    f_slot,
                    n_slot,
                    line,
                    stack,
                    frames,
                    exception_handlers,
                    value_store,
                    heavy_store,
                )
            }
            OpCode::HeappushFlat => {
                return stack_ops::op_heappush_flat(
                    line,
                    stack,
                    frames,
                    exception_handlers,
                    value_store,
                    heavy_store,
                )
            }
            OpCode::DivmodUnpack2(q_slot, r_slot) => {
                return stack_ops::op_divmod_unpack2(
                    q_slot,
                    r_slot,
                    line,
                    stack,
                    frames,
                    exception_handlers,
                    value_store,
                    heavy_store,
                )
            }
            OpCode::ObjectGetIntegral => {
                return stack_ops::op_object_get_integral(
                    line,
                    stack,
                    frames,
                    exception_handlers,
                    value_store,
                    heavy_store,
                )
            }
            OpCode::HeappopFlat => {
                return stack_ops::op_heappop_flat(
                    line,
                    stack,
                    frames,
                    exception_handlers,
                    value_store,
                    heavy_store,
                )
            }
            OpCode::SetDiscardIntegral => {
                return stack_ops::op_set_discard_integral(
                    line,
                    stack,
                    frames,
                    exception_handlers,
                    value_store,
                    heavy_store,
                )
            }
            OpCode::SetAddIntegral => {
                return stack_ops::op_set_add_integral(
                    line,
                    stack,
                    frames,
                    exception_handlers,
                    value_store,
                    heavy_store,
                )
            }
            OpCode::ObjectClear => {
                return stack_ops::op_object_clear(
                    line,
                    stack,
                    frames,
                    exception_handlers,
                    value_store,
                    heavy_store,
                )
            }
            OpCode::AbsI32 => {
                return arithmetic::op_abs_i32(
                    stack,
                    frames,
                    exception_handlers,
                    value_store,
                    heavy_store,
                )
            }
            OpCode::GridGetI32(buf, idx) => {
                return grid_ops::op_grid_get_i32(
                    buf,
                    idx,
                    line,
                    stack,
                    frames,
                    exception_handlers,
                    value_store,
                    heavy_store,
                )
            }
            OpCode::GridSetI32(buf, idx, val) => {
                return grid_ops::op_grid_set_i32(
                    buf,
                    idx,
                    val,
                    line,
                    stack,
                    frames,
                    exception_handlers,
                    value_store,
                    heavy_store,
                )
            }
            OpCode::GridGetU8(buf, idx) => {
                return grid_ops::op_grid_get_u8(
                    buf,
                    idx,
                    line,
                    stack,
                    frames,
                    exception_handlers,
                    value_store,
                    heavy_store,
                )
            }
            OpCode::GridSetU8(buf, idx, val) => {
                return grid_ops::op_grid_set_u8(
                    buf,
                    idx,
                    val,
                    line,
                    stack,
                    frames,
                    exception_handlers,
                    value_store,
                    heavy_store,
                )
            }
            OpCode::GridTestBlocked(bitmap, idx) => {
                return grid_ops::op_grid_test_blocked(
                    bitmap,
                    idx,
                    line,
                    stack,
                    frames,
                    exception_handlers,
                    value_store,
                    heavy_store,
                )
            }
            OpCode::GridHeapPush(heap, node, f_buf) => {
                return grid_ops::op_grid_heap_push(
                    heap,
                    node,
                    f_buf,
                    line,
                    stack,
                    frames,
                    exception_handlers,
                    value_store,
                    heavy_store,
                )
            }
            OpCode::GridHeapPopUnpack2(f_slot, n_slot, heap) => {
                return grid_ops::op_grid_heap_pop_unpack2(
                    f_slot,
                    n_slot,
                    heap,
                    line,
                    stack,
                    frames,
                    exception_handlers,
                    value_store,
                    heavy_store,
                )
            }
            OpCode::GridHeapLen(heap) => {
                return grid_ops::op_grid_heap_len(
                    heap,
                    line,
                    stack,
                    frames,
                    exception_handlers,
                    value_store,
                    heavy_store,
                )
            }
            OpCode::InvokeSpecialInit(this_slot, param_count) => {
                return special_init::op_invoke_special_init(
                    this_slot,
                    param_count,
                    line,
                    stack,
                    frames,
                    exception_handlers,
                    value_store,
                    heavy_store,
                )
            }
            _ => {}
        }
    }

    // Логирование выполнения конструктора
    let is_constructor = frame.function.name.contains("::new_");

    if is_constructor && crate::common::debug::is_debug_enabled() {
        let is_return = matches!(instruction, OpCode::Return);
        debug_println!(
            "[DEBUG executor constructor] '{}' IP {} line {}: {:?} (stack len {})",
            frame.function.name,
            current_ip,
            line,
            instruction,
            stack.len()
        );
        if is_return && !stack.is_empty() {
            let return_tv = stack[stack.len() - 1];
            let return_id = tagged_to_value_id(return_tv, value_store);
            let return_value = load_value(return_id, value_store, heavy_store);
            let val_type = match &return_value {
                Value::Object(obj_rc) => {
                    let map = obj_rc.borrow();
                    let keys: Vec<String> = map.str_key_pairs().into_iter().map(|(k, _)| k).collect();
                    format!("Object с ключами: {:?}", keys)
                }
                _ => format!("{:?}", return_value),
            };
            debug_println!(
                "[DEBUG executor constructor] Возвращаемое значение: {}",
                val_type
            );
        }
    }

    match instruction {
        OpCode::Import(module_index) => {
            return import_handler::handle_import(
                module_index,
                line,
                stack,
                frames,
                globals,
                global_names,
                functions,
                natives,
                exception_handlers,
                loaded_modules,
                abi_natives,
                loaded_native_libraries,
                value_store,
                heavy_store,
                vm_ptr,
            );
        }
        OpCode::ImportFrom(module_index, items_index) => {
            return import_handler::handle_import_from(
                module_index,
                items_index,
                line,
                stack,
                frames,
                globals,
                global_names,
                functions,
                natives,
                exception_handlers,
                loaded_modules,
                abi_natives,
                loaded_native_libraries,
                value_store,
                heavy_store,
                vm_ptr,
            );
        }

        OpCode::Constant(index) => return stack_ops::op_constant(index, stack, frame),
        OpCode::HeappopUnpack2(f_slot, n_slot) => {
            return stack_ops::op_heappop_unpack2(
                f_slot,
                n_slot,
                line,
                stack,
                frames,
                exception_handlers,
                value_store,
                heavy_store,
            )
        }
        OpCode::HeappushFlat => {
            return stack_ops::op_heappush_flat(
                line,
                stack,
                frames,
                exception_handlers,
                value_store,
                heavy_store,
            )
        }
        OpCode::DivmodUnpack2(q_slot, r_slot) => {
            return stack_ops::op_divmod_unpack2(
                q_slot,
                r_slot,
                line,
                stack,
                frames,
                exception_handlers,
                value_store,
                heavy_store,
            )
        }
        OpCode::ObjectGetIntegral => {
            return stack_ops::op_object_get_integral(
                line,
                stack,
                frames,
                exception_handlers,
                value_store,
                heavy_store,
            )
        }
        OpCode::HeappopFlat => {
            return stack_ops::op_heappop_flat(
                line,
                stack,
                frames,
                exception_handlers,
                value_store,
                heavy_store,
            )
        }
        OpCode::SetDiscardIntegral => {
            return stack_ops::op_set_discard_integral(
                line,
                stack,
                frames,
                exception_handlers,
                value_store,
                heavy_store,
            )
        }
        OpCode::ObjectIndexIntegral => {
            return element_ops::op_object_index_integral(
                line,
                stack,
                frames,
                globals,
                global_names,
                functions,
                natives,
                exception_handlers,
                value_store,
                heavy_store,
                vm_ptr,
            )
        }
        OpCode::SetAddIntegral => {
            return stack_ops::op_set_add_integral(
                line,
                stack,
                frames,
                exception_handlers,
                value_store,
                heavy_store,
            )
        }
        OpCode::ObjectClear => {
            return stack_ops::op_object_clear(
                line,
                stack,
                frames,
                exception_handlers,
                value_store,
                heavy_store,
            )
        }
        OpCode::ObjectSetIntegral => {
            return element_ops::op_object_set_integral(
                line,
                stack,
                frames,
                globals,
                global_names,
                functions,
                natives,
                exception_handlers,
                value_store,
                heavy_store,
            )
        }
        OpCode::AbsI32 => {
            return arithmetic::op_abs_i32(
                stack,
                frames,
                exception_handlers,
                value_store,
                heavy_store,
            )
        }
        OpCode::GridGetI32(buf, idx) => {
            return grid_ops::op_grid_get_i32(
                buf,
                idx,
                line,
                stack,
                frames,
                exception_handlers,
                value_store,
                heavy_store,
            )
        }
        OpCode::GridSetI32(buf, idx, val) => {
            return grid_ops::op_grid_set_i32(
                buf,
                idx,
                val,
                line,
                stack,
                frames,
                exception_handlers,
                value_store,
                heavy_store,
            )
        }
        OpCode::GridGetU8(buf, idx) => {
            return grid_ops::op_grid_get_u8(
                buf,
                idx,
                line,
                stack,
                frames,
                exception_handlers,
                value_store,
                heavy_store,
            )
        }
        OpCode::GridSetU8(buf, idx, val) => {
            return grid_ops::op_grid_set_u8(
                buf,
                idx,
                val,
                line,
                stack,
                frames,
                exception_handlers,
                value_store,
                heavy_store,
            )
        }
        OpCode::GridTestBlocked(bitmap, idx) => {
            return grid_ops::op_grid_test_blocked(
                bitmap,
                idx,
                line,
                stack,
                frames,
                exception_handlers,
                value_store,
                heavy_store,
            )
        }
        OpCode::GridHeapPush(heap, node, f_buf) => {
            return grid_ops::op_grid_heap_push(
                heap,
                node,
                f_buf,
                line,
                stack,
                frames,
                exception_handlers,
                value_store,
                heavy_store,
            )
        }
        OpCode::GridHeapPopUnpack2(f_slot, n_slot, heap) => {
            return grid_ops::op_grid_heap_pop_unpack2(
                f_slot,
                n_slot,
                heap,
                line,
                stack,
                frames,
                exception_handlers,
                value_store,
                heavy_store,
            )
        }
        OpCode::GridHeapLen(heap) => {
            return grid_ops::op_grid_heap_len(
                heap,
                line,
                stack,
                frames,
                exception_handlers,
                value_store,
                heavy_store,
            )
        }
        OpCode::InvokeSpecialInit(this_slot, param_count) => {
            return special_init::op_invoke_special_init(
                this_slot,
                param_count,
                line,
                stack,
                frames,
                exception_handlers,
                value_store,
                heavy_store,
            )
        }
        OpCode::LoadLocal(index) => {
            return stack_ops::op_load_local(index, current_ip, stack, frames)
        }
        OpCode::StoreLocal(index) => {
            return stack_ops::op_store_local(
                index,
                stack,
                frames,
                exception_handlers,
                value_store,
                heavy_store,
            )
        }
        OpCode::LoadGlobal(index) => {
            return memory::op_load_global(
                index,
                line,
                stack,
                frames,
                globals,
                global_names,
                functions,
                exception_handlers,
                loaded_modules,
                value_store,
                heavy_store,
                vm_ptr,
            );
        }

        OpCode::StoreGlobal(index) => {
            return memory::op_store_global(
                index,
                line,
                stack,
                frames,
                globals,
                global_names,
                explicit_global_names,
                exception_handlers,
                value_store,
                heavy_store,
                vm_ptr,
            );
        }

        OpCode::Add => {
            return arithmetic::op_add(
                current_ip,
                stack,
                frames,
                exception_handlers,
                value_store,
                heavy_store,
            )
        }
        OpCode::FormatInterp(index) => {
            return stack_ops::op_format_interp(
                index,
                stack,
                frames,
                exception_handlers,
                value_store,
                heavy_store,
            )
        }
        OpCode::RegAdd(rd, r1, r2) => return arithmetic::op_reg_add(rd, r1, r2, frames),
        OpCode::Sub => {
            return arithmetic::op_sub(
                current_ip,
                stack,
                frames,
                exception_handlers,
                value_store,
                heavy_store,
            )
        }
        OpCode::Mul => {
            return arithmetic::op_mul(
                current_ip,
                stack,
                frames,
                exception_handlers,
                value_store,
                heavy_store,
            )
        }
        OpCode::MatMul => {
            return arithmetic::op_matmul(
                current_ip,
                stack,
                frames,
                exception_handlers,
                value_store,
                heavy_store,
            )
        }
        OpCode::BinaryOp(idx) => {
            return arithmetic::op_binary_op(
                idx,
                current_ip,
                stack,
                frames,
                exception_handlers,
                value_store,
                heavy_store,
            )
        }
        OpCode::Div => {
            return arithmetic::op_div(
                current_ip,
                stack,
                frames,
                exception_handlers,
                value_store,
                heavy_store,
            )
        }
        OpCode::IntDiv => {
            return arithmetic::op_int_div(
                current_ip,
                stack,
                frames,
                exception_handlers,
                value_store,
                heavy_store,
            )
        }
        OpCode::Mod => {
            return arithmetic::op_mod(
                current_ip,
                stack,
                frames,
                exception_handlers,
                value_store,
                heavy_store,
            )
        }
        OpCode::Pow => {
            return arithmetic::op_pow(stack, frames, exception_handlers, value_store, heavy_store)
        }
        OpCode::Negate => {
            return arithmetic::op_negate(
                stack,
                frames,
                exception_handlers,
                value_store,
                heavy_store,
            )
        }
        OpCode::BitAnd => {
            return bitwise::op_bit_and(
                current_ip,
                stack,
                frames,
                exception_handlers,
                value_store,
                heavy_store,
            )
        }
        OpCode::BitOr => {
            return bitwise::op_bit_or(
                current_ip,
                stack,
                frames,
                exception_handlers,
                value_store,
                heavy_store,
            )
        }
        OpCode::BitXor => {
            return bitwise::op_bit_xor(
                current_ip,
                stack,
                frames,
                exception_handlers,
                value_store,
                heavy_store,
            )
        }
        OpCode::ShiftLeft => {
            return bitwise::op_shift_left(
                current_ip,
                stack,
                frames,
                exception_handlers,
                value_store,
                heavy_store,
            )
        }
        OpCode::ShiftRight => {
            return bitwise::op_shift_right(
                current_ip,
                stack,
                frames,
                exception_handlers,
                value_store,
                heavy_store,
            )
        }
        OpCode::BitNot => {
            return bitwise::op_bit_not(
                current_ip,
                stack,
                frames,
                exception_handlers,
                value_store,
                heavy_store,
            )
        }
        OpCode::Not => {
            return arithmetic::op_not(stack, frames, exception_handlers, value_store, heavy_store)
        }
        OpCode::Or => {
            return arithmetic::op_or(stack, frames, exception_handlers, value_store, heavy_store)
        }
        OpCode::And => {
            return arithmetic::op_and(stack, frames, exception_handlers, value_store, heavy_store)
        }

        OpCode::Equal => {
            return comparison::op_equal(
                stack,
                frames,
                exception_handlers,
                value_store,
                heavy_store,
            )
        }
        OpCode::NotEqual => {
            return comparison::op_not_equal(
                stack,
                frames,
                exception_handlers,
                value_store,
                heavy_store,
            )
        }
        OpCode::Greater => {
            return comparison::op_greater(
                stack,
                frames,
                exception_handlers,
                value_store,
                heavy_store,
            )
        }
        OpCode::Less => {
            return comparison::op_less(stack, frames, exception_handlers, value_store, heavy_store)
        }
        OpCode::GreaterEqual => {
            return comparison::op_greater_equal(
                stack,
                frames,
                exception_handlers,
                value_store,
                heavy_store,
            )
        }
        OpCode::LessEqual => {
            return comparison::op_less_equal(
                stack,
                frames,
                exception_handlers,
                value_store,
                heavy_store,
            )
        }
        OpCode::In => {
            return comparison::op_in(
                line,
                stack,
                frames,
                exception_handlers,
                value_store,
                heavy_store,
            )
        }
        OpCode::InIntegral => {
            return comparison::op_in_integral(
                line,
                stack,
                frames,
                exception_handlers,
                value_store,
                heavy_store,
            )
        }
        OpCode::NotInIntegral => {
            return comparison::op_not_in_integral(
                line,
                stack,
                frames,
                exception_handlers,
                value_store,
                heavy_store,
            )
        }
        OpCode::InGridBounds => {
            return comparison::op_in_grid_bounds(
                stack,
                frames,
                exception_handlers,
                value_store,
                heavy_store,
            )
        }
        OpCode::InGridBoundsOut => {
            return comparison::op_in_grid_bounds_out(
                stack,
                frames,
                exception_handlers,
                value_store,
                heavy_store,
            )
        }
        OpCode::FScoreStaleCheck => {
            return comparison::op_f_score_stale_check(
                stack,
                frames,
                exception_handlers,
                value_store,
                heavy_store,
            )
        }
        OpCode::DictGetIntegralLt => {
            return comparison::op_dict_get_integral_lt(
                stack,
                frames,
                exception_handlers,
                value_store,
                heavy_store,
            )
        }
        OpCode::DictIndexIntegralAddImm(addend) => {
            return comparison::op_dict_index_integral_add_imm(
                addend,
                stack,
                frames,
                exception_handlers,
                value_store,
                heavy_store,
            )
        }

        OpCode::Jump8(offset) => return control_flow::op_jump8(offset, frames),
        OpCode::Jump16(offset) => return control_flow::op_jump16(offset, frames),
        OpCode::Jump32(offset) => return control_flow::op_jump32(offset, frames),
        OpCode::JumpIfFalse8(offset) => {
            return control_flow::op_jump_if_false8(
                offset,
                stack,
                frames,
                exception_handlers,
                value_store,
                heavy_store,
            )
        }
        OpCode::JumpIfFalse16(offset) => {
            return control_flow::op_jump_if_false16(
                offset,
                stack,
                frames,
                exception_handlers,
                value_store,
                heavy_store,
            )
        }
        OpCode::JumpIfFalse32(offset) => {
            return control_flow::op_jump_if_false32(
                offset,
                stack,
                frames,
                exception_handlers,
                value_store,
                heavy_store,
            )
        }
        OpCode::JumpIfLocalHeapArrayEmpty8(slot, offset) => {
            return control_flow::op_jump_if_local_heap_array_empty8(slot, offset, frames, value_store)
        }
        OpCode::JumpIfLocalHeapArrayEmpty16(slot, offset) => {
            return control_flow::op_jump_if_local_heap_array_empty16(slot, offset, frames, value_store)
        }
        OpCode::JumpIfLocalHeapArrayEmpty32(slot, offset) => {
            return control_flow::op_jump_if_local_heap_array_empty32(slot, offset, frames, value_store)
        }
        OpCode::JumpLabel(_) | OpCode::JumpIfFalseLabel(_) | OpCode::JumpIfLocalHeapArrayEmptyLabel(_, _) => {
            return control_flow::op_jump_label(line)
        }

        OpCode::ForRange(var_slot, start_const, end_const, step_const, end_offset) => {
            return control_flow::op_for_range(
                var_slot,
                start_const,
                end_const,
                step_const,
                end_offset,
                frames,
                value_store,
            )
        }
        OpCode::ForRangeNext(back_offset) => {
            return control_flow::op_for_range_next(back_offset, frames)
        }
        OpCode::PopForRange => return control_flow::op_pop_for_range(frames),
        OpCode::CoerceForInIterable(iter_local) => {
            return for_iterable::op_coerce_for_in_iterable(
                iter_local,
                line,
                stack,
                frames,
                exception_handlers,
                value_store,
                heavy_store,
            );
        }
        OpCode::ForIterableNext(iter_local) => {
            return for_iterable::op_for_iterable_next(
                iter_local,
                line,
                stack,
                frames,
                exception_handlers,
                value_store,
                heavy_store,
            );
        }
        OpCode::CallWithUnpack(unpack_arity) => {
            return call_engine::execute_call_with_unpack(
                unpack_arity,
                line,
                stack,
                frames,
                functions,
                exception_handlers,
                error_type_table,
                value_store,
                heavy_store,
                vm_ptr,
            );
        }
        OpCode::CallVariadic(packed) => {
            return call_engine::execute_call_variadic(
                packed,
                line,
                stack,
                frames,
                globals,
                global_names,
                explicit_global_names,
                functions,
                natives,
                exception_handlers,
                error_type_table,
                explicit_relations,
                explicit_primary_keys,
                abi_natives,
                value_store,
                heavy_store,
                native_args_buffer,
                reusable_native_arg_ids,
                reusable_all_popped,
                vm_ptr,
            );
        }
        OpCode::Call(arity) => {
            return call_engine::execute_call(
                arity,
                line,
                stack,
                frames,
                globals,
                global_names,
                explicit_global_names,
                functions,
                natives,
                exception_handlers,
                error_type_table,
                explicit_relations,
                explicit_primary_keys,
                abi_natives,
                value_store,
                heavy_store,
                native_args_buffer,
                reusable_native_arg_ids,
                reusable_all_popped,
                vm_ptr,
            );
        }

        OpCode::Return => {
            // Получаем возвращаемое значение (если есть)
            // Проверяем стек относительно stack_start текущего фрейма
            let frame = frames.last().unwrap();

            // Логирование для конструкторов
            let is_constructor = frame.function.name.contains("::new_");
            if is_constructor {
                debug_println!("[DEBUG executor Return] constructor '{}' line {} Return (stack len {}, stack_start {})", frame.function.name, line, stack.len(), frame.stack_start);
            }

            // Truncate at the returning frame's watermark (not the caller's): caller may keep
            // expression temps in [caller.stack_start .. callee.stack_start).
            let callee_stack_start = frame.stack_start;
            let return_tv = stack::pop_return_value(stack, callee_stack_start);
            let return_value_id = tagged_to_value_id(return_tv, value_store);
            if cfg!(debug_assertions) {
                if frame.function.name.contains("::new_") && return_tv.is_heap() {
                    let key_count = match value_store.get(return_tv.get_heap_id()) {
                        Some(crate::common::value_store::ValueCell::Object(omap)) => {
                            Some(omap.len())
                        }
                        Some(crate::common::value_store::ValueCell::Heavy(idx)) => {
                            heavy_store.get(*idx).and_then(|v| {
                                if let Value::Object(obj_rc) = v {
                                    Some(obj_rc.borrow().len())
                                } else {
                                    None
                                }
                            })
                        }
                        _ => None,
                    };
                    if let Some(key_count) = key_count {
                        debug_println!(
                            "[DEBUG Return] constructor '{}' line {} returns Object ({} keys)",
                            frame.function.name,
                            line,
                            key_count
                        );
                    }
                }
            }
            let frames_count = frames.len();
            if frames_count > 1 {
                if let Some(frame) = frames.last() {
                    if frame.function.is_cached {
                        if let Some(ref cached_args) = frame.cached_args {
                            use crate::bytecode::function::CacheKey;
                            let cached_vals: Vec<Value> = cached_args
                                .iter()
                                .map(|&tv| slot_to_value(tv, value_store, heavy_store))
                                .collect();
                            if let Some(cache_key) = CacheKey::new(&cached_vals) {
                                if let Some(cache_rc) = &frame.function.cache {
                                    let mut cache = cache_rc.borrow_mut();
                                    let result_val =
                                        load_value(return_value_id, value_store, heavy_store);
                                    cache.map.insert(cache_key, result_val);
                                }
                            }
                        }
                    }
                }
                frames.pop();
                let return_value_id = value_store.leave_ephemeral(return_value_id);
                let promoted_tv = if return_value_id == NULL_VALUE_ID {
                    TaggedValue::null()
                } else {
                    TaggedValue::from_heap(return_value_id)
                };
                if let Some(sp) = stack::active_sp_for(stack) {
                    let old_sp = *sp;
                    let vm = unsafe { &mut *vm_ptr };
                    crate::vm::stack_sweep_hook::maybe_sweep_dead_stack(
                        vm,
                        stack,
                        old_sp,
                        callee_stack_start,
                    );
                    stack::compact_caller_stack_after_return(
                        stack,
                        sp,
                        callee_stack_start,
                        promoted_tv,
                    );
                } else {
                    stack.truncate(callee_stack_start);
                    stack.push(promoted_tv);
                }
                return Ok(VMStatus::Continue);
            } else {
                let return_value_id = value_store.leave_ephemeral(return_value_id);
                if let Some(sp) = stack::active_sp_for(stack) {
                    stack::truncate_to(stack, sp, 0);
                } else {
                    stack.clear();
                }
                return Ok(VMStatus::Return(return_value_id));
            }
        }
        OpCode::Yield(_next_st) => {
            let frame = frames.last().unwrap();
            if !frame.function.is_stream {
                return Err(LangError::runtime_error(
                    "Yield is only valid in stream fn".to_string(),
                    line,
                ));
            }
            let vm = unsafe { &mut *vm_ptr };
            if vm.pending_generator_send.is_some() {
                vm.pending_generator_send = None;
                return Err(LangError::runtime_error(
                    "generator.send() is not valid when the generator is not at a yield-await point (use .next())"
                        .to_string(),
                    line,
                ));
            }
            let yield_value_id = if stack::available_in_frame(stack, frame.stack_start) > 0 {
                let tv = stack::pop_direct(stack).unwrap_or(TaggedValue::null());
                tagged_to_value_id(tv, value_store)
            } else {
                NULL_VALUE_ID
            };
            return Ok(VMStatus::GeneratorYield(yield_value_id));
        }
        OpCode::YieldAwaitInput(_st, assign_slot) => {
            let frame = frames.last_mut().unwrap();
            if !frame.function.is_stream {
                return Err(LangError::runtime_error(
                    "YieldAwaitInput is only valid in stream fn".to_string(),
                    line,
                ));
            }
            let yield_value_id = if stack::available_in_frame(stack, frame.stack_start) > 0 {
                let tv = stack::pop_direct(stack).unwrap_or(TaggedValue::null());
                tagged_to_value_id(tv, value_store)
            } else {
                NULL_VALUE_ID
            };
            let vm = unsafe { &mut *vm_ptr };
            if let Some(pending) = vm.pending_generator_send.take() {
                if assign_slot >= frame.slots.len() {
                    frame.slots.resize(assign_slot + 1, TaggedValue::null());
                }
                let val = match pending {
                    PendingGeneratorSend::NextDefaultRhs => {
                        if yield_value_id != NULL_VALUE_ID {
                            load_value(yield_value_id, value_store, heavy_store)
                        } else {
                            vm.yield_await_resume_value.take().unwrap_or(Value::Null)
                        }
                    }
                    PendingGeneratorSend::Explicit(sent) => {
                        vm.pending_send_rhs_return = vm.yield_await_resume_value.take();
                        sent
                    }
                };
                let tid = store_value(val, value_store, heavy_store);
                frame.slots[assign_slot] = TaggedValue::from_heap(tid);
                // Bypasses StoreLocal: slot changed — drop opcode inline caches (Add/Mul/LoadLocal/…).
                frame.invalidate_inline_caches();
                // Потребитель уже получил yield при первом suspend; подстановка в слот без второго yield.
                return Ok(VMStatus::Continue);
            }
            return Ok(VMStatus::GeneratorYieldAwait(yield_value_id, assign_slot));
        }
        OpCode::GeneratorDone => {
            let frame = frames.last().unwrap();
            if !frame.function.is_stream {
                return Err(LangError::runtime_error(
                    "GeneratorDone is only valid in stream fn".to_string(),
                    line,
                ));
            }
            frames.pop();
            return Ok(VMStatus::GeneratorDone(None));
        }
        OpCode::GeneratorDoneWithFinal => {
            let frame = frames.last().unwrap();
            if !frame.function.is_stream {
                return Err(LangError::runtime_error(
                    "GeneratorDoneWithFinal is only valid in stream fn".to_string(),
                    line,
                ));
            }
            let final_value_id = if stack::available_in_frame(stack, frame.stack_start) > 0 {
                let tv = stack::pop_direct(stack).unwrap_or(TaggedValue::null());
                tagged_to_value_id(tv, value_store)
            } else {
                NULL_VALUE_ID
            };
            frames.pop();
            return Ok(VMStatus::GeneratorDone(Some(final_value_id)));
        }
        OpCode::Pop => return stack_ops::op_pop(stack, frames),
        OpCode::Dup => {
            return stack_ops::op_dup(stack, frames, exception_handlers, value_store, heavy_store)
        }
        OpCode::MakeArray(count) => {
            return object::op_make_array(
                count,
                stack,
                frames,
                exception_handlers,
                value_store,
                heavy_store,
            )
        }
        OpCode::MakeTuple(count) => {
            return object::op_make_tuple(
                count,
                stack,
                frames,
                exception_handlers,
                value_store,
                heavy_store,
            )
        }
        OpCode::MakeObject(pair_count) => {
            return object::op_make_object(
                pair_count,
                line,
                stack,
                frames,
                exception_handlers,
                value_store,
                heavy_store,
            )
        }
        OpCode::UnpackObject(count_slot) => {
            return object::op_unpack_object(
                count_slot,
                line,
                stack,
                frames,
                exception_handlers,
                value_store,
                heavy_store,
            )
        }
        OpCode::MakeObjectDynamic => {
            return object::op_make_object_dynamic(
                line,
                stack,
                frames,
                exception_handlers,
                value_store,
                heavy_store,
            )
        }
        OpCode::MakeSet(count) => {
            return object::op_make_set(
                count,
                line,
                stack,
                frames,
                exception_handlers,
                value_store,
                heavy_store,
            )
        }
        OpCode::MakeSetDynamic => {
            return object::op_make_set_dynamic(
                line,
                stack,
                frames,
                exception_handlers,
                value_store,
                heavy_store,
            )
        }
        OpCode::MakeArrayDynamic => {
            return object::op_make_array_dynamic(
                line,
                stack,
                frames,
                exception_handlers,
                value_store,
                heavy_store,
            )
        }
        OpCode::GetArrayLength => {
            return object::op_get_array_length(
                line,
                stack,
                frames,
                exception_handlers,
                value_store,
                heavy_store,
                vm_ptr,
            )
        }
        OpCode::TableFilter => {
            return object::op_table_filter(
                line,
                stack,
                frames,
                exception_handlers,
                value_store,
                heavy_store,
                vm_ptr,
            )
        }
        OpCode::TableFilterPred(pred_index) => {
            return object::op_table_filter_pred(
                pred_index,
                line,
                stack,
                frames,
                exception_handlers,
                value_store,
                heavy_store,
                vm_ptr,
            )
        }
        OpCode::GetArrayElement => {
            return element_ops::op_get_array_element(
                line,
                stack,
                frames,
                globals,
                global_names,
                functions,
                natives,
                exception_handlers,
                value_store,
                heavy_store,
                vm_ptr,
            )
        }
        OpCode::GetArraySlice => {
            return element_ops::op_get_array_slice(
                line,
                stack,
                frames,
                exception_handlers,
                value_store,
                heavy_store,
            )
        }
        OpCode::SetArrayElement => {
            return element_ops::op_set_array_element(
                line,
                stack,
                frames,
                globals,
                global_names,
                functions,
                natives,
                exception_handlers,
                value_store,
                heavy_store,
            )
        }
        OpCode::SetArraySlice => {
            return element_ops::op_set_array_slice(
                line,
                stack,
                frames,
                exception_handlers,
                value_store,
                heavy_store,
            )
        }
        OpCode::Clone => {
            return object::op_clone(stack, frames, exception_handlers, value_store, heavy_store)
        }

        OpCode::BeginTry(handler_index) => {
            return exception::op_begin_try(
                handler_index,
                stack,
                frames,
                exception_handlers,
                error_type_table,
            )
        }
        OpCode::EndTry => return exception::op_end_try(frames, exception_handlers),
        OpCode::Catch(_) => return exception::op_catch(),
        OpCode::EndCatch => return exception::op_end_catch(),
        OpCode::Throw(_) => {
            return exception::op_throw(
                line,
                stack,
                frames,
                exception_handlers,
                value_store,
                heavy_store,
            )
        }
        OpCode::PopExceptionHandler => {
            return exception::op_pop_exception_handler(exception_handlers)
        }
    }
}
