// CallFrame для виртуальной машины (slots as TaggedValue; constant_tagged for immediates on stack)

use crate::bytecode::Function;
use crate::common::value::Value;
use crate::common::value_store::{ValueId, ValueStore};
use crate::common::TaggedValue;
use crate::vm::heavy_store::HeavyStore;
use crate::vm::store_convert::store_value;

/// Состояние одного уровня специализированного цикла ForRange: (current, end, step, var_slot).
/// Стек нужен для вложенных циклов for i in range(...).
pub type ForRangeState = (i64, i64, i64, usize);

pub struct CallFrame {
    pub function: Function,
    pub ip: usize,           // Instruction pointer
    pub slots: Vec<TaggedValue>, // Локальные переменные (TaggedValue: immediates без store)
    pub stack_start: usize,  // Начало стека для этой функции в общем стеке VM
    /// Константы текущего chunk, загруженные в store при создании фрейма
    pub constant_ids: Vec<ValueId>,
    /// Tagged form for immediates (Number/Bool/Null) to push without store lookup; None => use constant_ids.
    pub constant_tagged: Vec<Option<TaggedValue>>,
    pub cached_args: Option<Vec<TaggedValue>>, // Аргументы для кэширования (только для кэшируемых функций)
    /// Стек состояний ForRange (один элемент на уровень вложенности)
    pub for_range_stack: Vec<ForRangeState>,
    /// Inline cache for Add: при совпадении IP и флага быстрый путь number+number.
    pub add_cache_ip: Option<usize>,
    pub add_cache_both_number: bool,
    /// Inline cache for Sub (number - number).
    pub sub_cache_ip: Option<usize>,
    pub sub_cache_both_number: bool,
    /// Inline cache for Mul (number * number).
    pub mul_cache_ip: Option<usize>,
    pub mul_cache_both_number: bool,
    /// Inline cache for Div (number / number, b != 0).
    pub div_cache_ip: Option<usize>,
    pub div_cache_both_number: bool,
    /// Inline cache for IntDiv (number // number, b != 0).
    pub intdiv_cache_ip: Option<usize>,
    pub intdiv_cache_both_number: bool,
    /// Inline cache for Mod (number % number, b != 0).
    pub mod_cache_ip: Option<usize>,
    pub mod_cache_both_number: bool,
    /// Inline cache for GetArrayElement: array+number index or object+string key.
    pub get_array_element_cache_ip: Option<usize>,
    pub get_array_element_cache_array_number: bool,
    pub get_array_element_cache_object_string: bool,
    /// Inline cache for Call: at this IP callee was a user function (ValueCell::Function).
    pub call_cache_ip: Option<usize>,
    pub call_cache_is_user_function: bool,
    /// Inline cache for LoadLocal: at (IP, slot) push cached TaggedValue without store lookup.
    pub load_local_cache_ip: Option<usize>,
    pub load_local_cache_slot: Option<usize>,
    pub load_local_cache_tagged: Option<TaggedValue>,
    /// Register bank for Register VM (п.4 этап 1). Используется опкодами RegAdd и др.; компилятор пока не эмитирует.
    pub regs: Vec<TaggedValue>,
    /// Module this frame's function belongs to (for LoadGlobal/StoreGlobal). None = use VM's unified globals (legacy).
    pub module_name: Option<String>,
}

impl CallFrame {
    pub fn new(
        function: Function,
        stack_start: usize,
        store: &mut ValueStore,
        heap: &mut HeavyStore,
    ) -> Self {
        let initial_slots = function.arity.max(8);
        let constants = &function.chunk.constants;
        let constant_ids: Vec<ValueId> = constants
            .iter()
            .map(|c| store_value(c.clone(), store, heap))
            .collect();
        let constant_tagged: Vec<Option<TaggedValue>> = constants
            .iter()
            .map(|c| match c {
                Value::Number(n) => Some(TaggedValue::from_f64(*n)),
                Value::Bool(b) => Some(TaggedValue::from_bool(*b)),
                Value::Null => Some(TaggedValue::null()),
                _ => None,
            })
            .collect();
        let module_name = function.module_name.clone();
        Self {
            slots: Vec::with_capacity(initial_slots + 64),
            ip: 0,
            function,
            stack_start,
            constant_ids,
            constant_tagged,
            cached_args: None,
            for_range_stack: Vec::new(),
            add_cache_ip: None,
            add_cache_both_number: false,
            sub_cache_ip: None,
            sub_cache_both_number: false,
            mul_cache_ip: None,
            mul_cache_both_number: false,
            div_cache_ip: None,
            div_cache_both_number: false,
            intdiv_cache_ip: None,
            intdiv_cache_both_number: false,
            mod_cache_ip: None,
            mod_cache_both_number: false,
            get_array_element_cache_ip: None,
            get_array_element_cache_array_number: false,
            get_array_element_cache_object_string: false,
            call_cache_ip: None,
            call_cache_is_user_function: false,
            load_local_cache_ip: None,
            load_local_cache_slot: None,
            load_local_cache_tagged: None,
            regs: Vec::with_capacity(32),
            module_name,
        }
    }

    pub fn new_with_cache(
        function: Function,
        stack_start: usize,
        args: Vec<TaggedValue>,
        store: &mut ValueStore,
        heap: &mut HeavyStore,
    ) -> Self {
        let mut frame = Self::new(function, stack_start, store, heap);
        frame.cached_args = Some(args);
        frame
    }
}

impl CallFrame {
    /// Гарантирует, что слоты вмещают по крайней мере slot_index+1.
    pub fn ensure_slot(&mut self, slot_index: usize) {
        while self.slots.len() <= slot_index {
            self.slots.push(TaggedValue::null());
        }
    }
}

