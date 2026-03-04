// Opcode execution for VM (Stage 1: stack/globals as Vec<ValueId>; one borrow store per instruction)

use crate::debug_println;
use crate::bytecode::OpCode;
use crate::common::{error::LangError, value::Value, value_store::NULL_VALUE_ID, TaggedValue};
use crate::vm::types::VMStatus;
use crate::vm::frame::CallFrame;
use crate::vm::exceptions::ExceptionHandler;
use crate::vm::exception;
use crate::vm::interpreter::{arithmetic, comparison, control_flow, element_ops, memory, object, stack_ops};
use crate::vm::runtime::call_engine;
use crate::vm::module_system::import_handler;
use crate::vm::stack;
use crate::vm::global_slot::GlobalSlot;
use crate::vm::store_convert::{load_value, tagged_to_value_id, slot_to_value};

// Re-export for backward compatibility (call_engine, import_handler, memory use executor::global_index_by_name)
pub(crate) use crate::vm::global_utils::{global_index_by_name, global_indices_by_name};

/// Execute one step of the VM - get next instruction and execute it
pub fn step(
    frames: &mut Vec<CallFrame>,
) -> Result<Option<(OpCode, usize)>, LangError> {
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
pub fn execute_instruction(
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
    crate::vm::profile::record_opcode();
    #[cfg(feature = "profile")]
    crate::vm::profile::set_current_opcode(&instruction);

    let frame = frames.last_mut().unwrap();
    let current_ip = frame.ip - 1; // IP уже инкрементирован в step()

    // Логирование выполнения конструктора
    let is_constructor = frame.function.name.contains("::new_");
    
    if is_constructor && crate::common::debug::is_debug_enabled() {
        let is_return = matches!(instruction, OpCode::Return);
        debug_println!("[DEBUG executor constructor] '{}' IP {} line {}: {:?} (stack len {})",
            frame.function.name, current_ip, line, instruction, stack.len());
        if is_return && !stack.is_empty() {
            let return_tv = stack[stack.len() - 1];
            let return_id = tagged_to_value_id(return_tv, value_store);
            let return_value = load_value(return_id, value_store, heavy_store);
            let val_type = match &return_value {
                Value::Object(obj_rc) => {
                    let map = obj_rc.borrow();
                    let keys: Vec<String> = map.keys().cloned().collect();
                    format!("Object с ключами: {:?}", keys)
                },
                _ => format!("{:?}", return_value),
            };
            debug_println!("[DEBUG executor constructor] Возвращаемое значение: {}", val_type);
        }
    }
    
    match instruction {
        OpCode::Import(module_index) => {
            return import_handler::handle_import(
                module_index, line, stack, frames, globals, global_names,
                functions, natives, exception_handlers, loaded_modules,
                abi_natives, loaded_native_libraries, value_store, heavy_store, vm_ptr,
            );
        }
        OpCode::ImportFrom(module_index, items_index) => {
            return import_handler::handle_import_from(
                module_index, items_index, line, stack, frames, globals, global_names,
                functions, natives, exception_handlers, loaded_modules,
                abi_natives, loaded_native_libraries, value_store, heavy_store, vm_ptr,
            );
        }

        OpCode::Constant(index) => return stack_ops::op_constant(index, stack, frame),
        OpCode::LoadLocal(index) => return stack_ops::op_load_local(index, current_ip, stack, frames),
        OpCode::StoreLocal(index) => return stack_ops::op_store_local(index, stack, frames, exception_handlers, value_store, heavy_store),
        OpCode::LoadGlobal(index) => {
            return memory::op_load_global(
                index, line, stack, frames, globals, global_names,
                functions, exception_handlers, loaded_modules,
                value_store, heavy_store, vm_ptr,
            );
        }

        OpCode::StoreGlobal(index) => {
            return memory::op_store_global(
                index, line, stack, frames, globals, global_names,
                exception_handlers, value_store, heavy_store, vm_ptr,
            );
        }

        OpCode::Add => return arithmetic::op_add(current_ip, stack, frames, exception_handlers, value_store, heavy_store),
        OpCode::FormatInterp(index) => return stack_ops::op_format_interp(index, stack, frames, exception_handlers, value_store, heavy_store),
        OpCode::RegAdd(rd, r1, r2) => return arithmetic::op_reg_add(rd, r1, r2, frames),
        OpCode::Sub => return arithmetic::op_sub(current_ip, stack, frames, exception_handlers, value_store, heavy_store),
        OpCode::Mul => return arithmetic::op_mul(current_ip, stack, frames, exception_handlers, value_store, heavy_store),
        OpCode::Div => return arithmetic::op_div(current_ip, stack, frames, exception_handlers, value_store, heavy_store),
        OpCode::IntDiv => return arithmetic::op_int_div(current_ip, stack, frames, exception_handlers, value_store, heavy_store),
        OpCode::Mod => return arithmetic::op_mod(current_ip, stack, frames, exception_handlers, value_store, heavy_store),
        OpCode::Pow => return arithmetic::op_pow(stack, frames, exception_handlers, value_store, heavy_store),
        OpCode::Negate => return arithmetic::op_negate(stack, frames, exception_handlers, value_store, heavy_store),
        OpCode::Not => return arithmetic::op_not(stack, frames, exception_handlers, value_store, heavy_store),
        OpCode::Or => return arithmetic::op_or(stack, frames, exception_handlers, value_store, heavy_store),
        OpCode::And => return arithmetic::op_and(stack, frames, exception_handlers, value_store, heavy_store),

        OpCode::Equal => return comparison::op_equal(stack, frames, exception_handlers, value_store, heavy_store),
        OpCode::NotEqual => return comparison::op_not_equal(stack, frames, exception_handlers, value_store, heavy_store),
        OpCode::Greater => return comparison::op_greater(stack, frames, exception_handlers, value_store, heavy_store),
        OpCode::Less => return comparison::op_less(stack, frames, exception_handlers, value_store, heavy_store),
        OpCode::GreaterEqual => return comparison::op_greater_equal(stack, frames, exception_handlers, value_store, heavy_store),
        OpCode::LessEqual => return comparison::op_less_equal(stack, frames, exception_handlers, value_store, heavy_store),
        OpCode::In => return comparison::op_in(line, stack, frames, exception_handlers, value_store, heavy_store),

        OpCode::Jump8(offset) => return control_flow::op_jump8(offset, frames),
        OpCode::Jump16(offset) => return control_flow::op_jump16(offset, frames),
        OpCode::Jump32(offset) => return control_flow::op_jump32(offset, frames),
        OpCode::JumpIfFalse8(offset) => return control_flow::op_jump_if_false8(offset, stack, frames, exception_handlers, value_store, heavy_store),
        OpCode::JumpIfFalse16(offset) => return control_flow::op_jump_if_false16(offset, stack, frames, exception_handlers, value_store, heavy_store),
        OpCode::JumpIfFalse32(offset) => return control_flow::op_jump_if_false32(offset, stack, frames, exception_handlers, value_store, heavy_store),
        OpCode::JumpLabel(_) | OpCode::JumpIfFalseLabel(_) => return control_flow::op_jump_label(line),
        
        OpCode::ForRange(var_slot, start_const, end_const, step_const, end_offset) => return control_flow::op_for_range(var_slot, start_const, end_const, step_const, end_offset, frames, value_store),
        OpCode::ForRangeNext(back_offset) => return control_flow::op_for_range_next(back_offset, frames),
        OpCode::PopForRange => return control_flow::op_pop_for_range(frames),
        OpCode::CallWithUnpack(unpack_arity) => {
            return call_engine::execute_call_with_unpack(
                unpack_arity, line, stack, frames, functions,
                exception_handlers, error_type_table, value_store, heavy_store, vm_ptr,
            );
        }
        OpCode::Call(arity) => {
            return call_engine::execute_call(
                arity, line, stack, frames, globals, global_names, explicit_global_names,
                functions, natives, exception_handlers, error_type_table,
                explicit_relations, explicit_primary_keys, abi_natives, value_store, heavy_store,
                native_args_buffer, reusable_native_arg_ids, reusable_all_popped, vm_ptr,
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
            
            let return_value_id = if stack.len() > frame.stack_start {
                let tv = stack.pop().unwrap_or(TaggedValue::null());
                tagged_to_value_id(tv, value_store)
            } else {
                NULL_VALUE_ID
            };
            if cfg!(debug_assertions) {
                let ret_val = load_value(return_value_id, value_store, heavy_store);
                if let Value::Object(obj_rc) = &ret_val {
                    if frame.function.name.contains("::new_") {
                        let key_count = obj_rc.borrow().len();
                        debug_println!("[DEBUG Return] constructor '{}' line {} returns Object ({} keys)", frame.function.name, line, key_count);
                    }
                }
            }
            let frames_count = frames.len();
            if frames_count > 1 {
                if let Some(frame) = frames.last() {
                    if frame.function.is_cached {
                        if let Some(ref cached_args) = frame.cached_args {
                            use crate::bytecode::function::CacheKey;
                            let cached_vals: Vec<Value> = cached_args.iter().map(|&tv| slot_to_value(tv, value_store, heavy_store)).collect();
                            if let Some(cache_key) = CacheKey::new(&cached_vals) {
                                if let Some(cache_rc) = &frame.function.cache {
                                    let mut cache = cache_rc.borrow_mut();
                                    let result_val = load_value(return_value_id, value_store, heavy_store);
                                    cache.map.insert(cache_key, result_val);
                                }
                            }
                        }
                    }
                }
                frames.pop();
                stack::push_id(stack, return_value_id);
                return Ok(VMStatus::Continue);
            } else {
                return Ok(VMStatus::Return(return_value_id));
            }
        }
        OpCode::Pop => return stack_ops::op_pop(stack, frames),
        OpCode::Dup => return stack_ops::op_dup(stack, frames, exception_handlers, value_store, heavy_store),
        OpCode::MakeArray(count) => return object::op_make_array(count, stack, frames, exception_handlers, value_store, heavy_store),
        OpCode::MakeTuple(count) => return object::op_make_tuple(count, stack, frames, exception_handlers, value_store, heavy_store),
        OpCode::MakeObject(pair_count) => return object::op_make_object(pair_count, line, stack, frames, exception_handlers, value_store, heavy_store),
        OpCode::UnpackObject(count_slot) => return object::op_unpack_object(count_slot, line, stack, frames, exception_handlers, value_store, heavy_store),
        OpCode::MakeObjectDynamic => return object::op_make_object_dynamic(line, stack, frames, exception_handlers, value_store, heavy_store),
        OpCode::MakeArrayDynamic => return object::op_make_array_dynamic(line, stack, frames, exception_handlers, value_store, heavy_store),
        OpCode::GetArrayLength => return object::op_get_array_length(line, stack, frames, exception_handlers, value_store, heavy_store),
        OpCode::TableFilter => return object::op_table_filter(line, stack, frames, exception_handlers, value_store, heavy_store, vm_ptr),
        OpCode::GetArrayElement => return element_ops::op_get_array_element(line, stack, frames, globals, global_names, functions, natives, exception_handlers, value_store, heavy_store, vm_ptr),
        OpCode::SetArrayElement => return element_ops::op_set_array_element(line, stack, frames, globals, global_names, functions, natives, exception_handlers, value_store, heavy_store),
        OpCode::Clone => return object::op_clone(stack, frames, exception_handlers, value_store, heavy_store),
        
        OpCode::BeginTry(handler_index) => return exception::op_begin_try(handler_index, stack, frames, exception_handlers, error_type_table),
        OpCode::EndTry => return exception::op_end_try(frames, exception_handlers),
        OpCode::Catch(_) => return exception::op_catch(),
        OpCode::EndCatch => return exception::op_end_catch(),
        OpCode::Throw(_) => return exception::op_throw(line, stack, frames, exception_handlers, value_store, heavy_store),
        OpCode::PopExceptionHandler => return exception::op_pop_exception_handler(exception_handlers),
    }
}

