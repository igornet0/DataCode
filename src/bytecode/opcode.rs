// Байт-код инструкции для VM

#[derive(Debug, Clone, PartialEq)]
pub enum OpCode {
    // Константы
    Constant(usize), // Индекс в массиве констант

    // Локальные переменные
    LoadLocal(usize),  // Загрузить локальную переменную по индексу
    StoreLocal(usize), // Сохранить в локальную переменную по индексу

    // Глобальные переменные
    LoadGlobal(usize),  // Загрузить глобальную переменную по индексу
    StoreGlobal(usize), // Сохранить в глобальную переменную по индексу

    // Арифметические операции
    Add,
    Sub,
    Mul,
    /// **Deprecated for new emits:** use [`BinaryOp`] with logical name `"matmul"`. Kept for stable [`crate::vm::dcb::SerOpCode`] / `.dcb` compatibility.
    MatMul,
    /// User-registered or plugin infix op: constant pool index → logical op name (e.g. `"matmul"`).
    BinaryOp(usize),
    Div,
    IntDiv, // Целочисленное деление (//)
    Mod,    // Модуло (%)
    Pow,    // Возведение в степень (**)
    Negate, // Унарный минус

    // Побитовые операции (только int)
    BitAnd,
    BitOr,
    BitXor,
    ShiftLeft,
    ShiftRight,
    BitNot, // Унарный ~

    // Логические операции
    Not, // Унарный Bang (!)
    Or,  // Логическое ИЛИ (or)
    And, // Логическое И (and)

    // Операции сравнения
    Equal,
    Greater,
    Less,
    NotEqual,
    GreaterEqual,
    LessEqual,
    In, // Проверка наличия значения в массиве (value in array)

    // Управление потоком
    // Временные метки для этапа компиляции
    JumpLabel(usize),        // Временная метка для безусловного перехода (label_id)
    JumpIfFalseLabel(usize), // Временная метка для условного перехода (label_id)
    /// `while arr`: jump if local heap array is empty (compile-time label).
    JumpIfLocalHeapArrayEmptyLabel(usize, usize),

    // Специализированный цикл for i in range(...): без материализации диапазона
    /// (var_slot, start_const, end_const, step_const, end_offset). end_offset патчится при finalize.
    ForRange(usize, usize, usize, usize, i32),
    /// back_offset: на сколько инструкций откатить IP (к ForRange)
    ForRangeNext(i32),
    /// Снять одно состояние с for_range_stack (при break из for i in range(...))
    PopForRange,

    /// `for x in ...`: заменить значение в локале на [`Value::Iterable`] через [`crate::vm::iterable::prepare_for_in_iterable`].
    CoerceForInIterable(usize),
    /// Следующий элемент ленивого итератора: локал с `Value::Iterable`. Кладёт на стек сначала элемент, затем `true`;
    /// при исчерпании — только `false`. Следующая инструкция обычно `JumpIfFalse` → выход из цикла.
    ForIterableNext(usize),

    // Финальные инструкции с относительными смещениями
    Jump8(i8),          // Безусловный переход с 8-битным смещением [-128, +127]
    Jump16(i16),        // Безусловный переход с 16-битным смещением [-32768, +32767]
    Jump32(i32),        // Безусловный переход с 32-битным смещением [-2^31, +2^31-1]
    JumpIfFalse8(i8),   // Условный переход с 8-битным смещением [-128, +127]
    JumpIfFalse16(i16), // Условный переход с 16-битным смещением [-32768, +32767]
    JumpIfFalse32(i32), // Условный переход с 32-битным смещением [-2^31, +2^31-1]
    /// Jump if `frame.slots[slot]` is not a non-empty heap array (flat heap: len ≥ 2).
    JumpIfLocalHeapArrayEmpty8(usize, i8),
    JumpIfLocalHeapArrayEmpty16(usize, i16),
    JumpIfLocalHeapArrayEmpty32(usize, i32),

    // Функции
    Call(usize),           // Вызов функции с количеством аргументов
    CallWithUnpack(usize), // Вызов: один аргумент — объект для распаковки в kwargs; ключи должны совпадать с именами параметров
    /// Вызов user/native функции с * / ** / именованными аргументами.
    /// Операнд: `(n_pos << 24) | (n_star << 16) | (n_named << 8) | n_starstar`.
    CallVariadic(u32),
    Return,                // Возврат из функции
    /// `stream fn`: снять значение с вершины стека как yield; резюм с IP после инструкции. `i32` — номер состояния (отладка).
    Yield(i32),
    /// `stream fn`: `x = return expr` — yield значения expr; затем ждать `.send()` и записать в локальный слот `usize`.
    YieldAwaitInput(i32, usize),
    /// `stream fn`: завершить генератор (`ereturn` без expr / неявный конец).
    GeneratorDone,
    /// `stream fn`: `ereturn expr` — снять значение со стека как финальное (не yield), завершить.
    GeneratorDoneWithFinal,

    // Массивы
    MakeArray(usize), // Создать массив из N элементов со стека (compile-time размер)
    MakeArrayDynamic, // Создать массив из N элементов со стека (runtime размер: N на стеке, затем N элементов)
    GetArrayLength,   // Получить длину массива
    GetArrayElement,  // Получить элемент массива по индексу (индекс и массив на стеке)
    /// Срез: на стеке снизу массив, затем start, stop, step (Null = пропуск); вершина = step.
    GetArraySlice,
    SetArrayElement, // Установить элемент массива/объекта по индексу (значение, индекс, массив/объект на стеке)
    /// Присваивание срезу: value, start, stop, step (Null), container (вершина).
    SetArraySlice,
    TableFilter, // Фильтр таблицы: stack [table, column, op, value] → отфильтрованная таблица
    /// Составной фильтр: stack [table, values_array], operand = индекс константы предиката
    TableFilterPred(usize),
    Clone,       // Глубокое клонирование значения на стеке (для массивов и таблиц)

    // Кортежи
    MakeTuple(usize), // Создать кортеж из N элементов со стека (compile-time размер)

    // Объекты
    MakeObject(usize), // Создать объект из N пар (ключ, значение) со стека (compile-time количество пар)
    /// Распаковать объект со стека: положить пары (ключ, значение) на стек и увеличить счётчик в слоте на число пар.
    UnpackObject(usize), // индекс слота для счётчика пар
    /// Создать объект: со стека снять count, затем 2*count значений (value, key на пару), собрать объект.
    MakeObjectDynamic,

    // Множества
    MakeSet(usize), // Создать set из N элементов со стека
    /// Создать set: со стека снять count, затем count элементов.
    MakeSetDynamic,

    // Обработка исключений
    BeginTry(usize), // Начало try блока, аргумент - индекс обработчика в таблице обработчиков
    EndTry,          // Конец try блока
    Catch(Option<usize>), // Начало catch блока, Option<usize> - тип ошибки (индекс в таблице типов), None для catch всех
    EndCatch,             // Конец catch блока
    Throw(Option<usize>), // Выбрасывание исключения (для будущего использования), Option<usize> - тип ошибки
    PopExceptionHandler,  // Удаление обработчика исключений со стека

    // Стек
    Pop, // Удалить значение со стека
    Dup, // Дублировать вершину стека (для short-circuit or/and)
    /// Форматирование значения для интерполяции: pop value, format по константе (например .2f), push string.
    FormatInterp(usize),

    // Модули
    Import(usize),            // Импорт модуля (индекс имени модуля в константах)
    ImportFrom(usize, usize), // from-импорт: (индекс имени модуля, индекс массива элементов импорта в константах)

    // Register VM (этап 1): опкоды с регистрами; компилятор пока не эмитирует.
    /// Add reg[rd] = reg[r1] + reg[r2] (number+number); иначе fallback на store. Регистры — индексы в frame.regs.
    RegAdd(u8, u8, u8),

    /// A* peephole: stack `[heap]` → `f_slot`, `node_slot` without tuple temp (heap popped from stack).
    HeappopUnpack2(usize, usize),
    /// A* peephole: stack `[heap, a, b]` → flat `heappush` without native call or MakeTuple.
    HeappushFlat,
    /// A* peephole: stack `[a, b]` → `divmod` into two locals without tuple alloc.
    DivmodUnpack2(usize, usize),
    /// Plain dict `.get(key [, default])` — stack `[obj, key, default]` → value (no GetArrayElement + Call).
    ObjectGetIntegral,
    /// `heapq.heappop(heap)` — pop flat pair onto stack as two values (heap popped from stack).
    HeappopFlat,
    /// Plain set `.discard(int_key)` — stack `[set, key]` → set (no GetArrayElement + Call).
    SetDiscardIntegral,
    /// Plain set `.add(int_key)` — stack `[set, key]` → set (no GetArrayElement + Call).
    SetAddIntegral,
    /// Plain dict/array `obj[int_key]` — stack `[obj, key]` → value (no GetArrayElement dispatch overhead).
    ObjectIndexIntegral,
    /// Plain dict `.clear()` — stack `[obj]` → obj (in-place clear).
    ObjectClear,
    /// Plain dict `obj[int_key] = value` — stack `[value, key, obj]` (no SetArrayElement dispatch).
    ObjectSetIntegral,
    /// `member in set/dict` with integral member — stack `[member, container]`.
    InIntegral,
    /// Negated integral `in` for plain set/dict (compiler peephole for `!x in s`).
    NotInIntegral,
    /// `abs(int_expr)` without native Call — pop scalar int/whole number, push abs.
    AbsI32,
    /// Grid A*: `0 <= nr < rows and 0 <= nc < cols` — stack `[nr, nc, rows, cols]` → bool.
    InGridBounds,
    /// Negated grid bounds (out-of-grid): same stack as [`InGridBounds`], pushes `!in_bounds`.
    InGridBoundsOut,
    /// A* stale heap entry: stack `[current_f, dict, key]` → bool (`current_f != dict.get(key)`).
    FScoreStaleCheck,
    /// A* relax test: stack `[tentative_g, dict, key]` → bool (`tentative_g < dict.get(key, +inf)`).
    DictGetIntegralLt,
    /// Plain dict `dict[integral_key] + addend` — stack `[dict, key]` → numeric sum (missing/null → addend).
    DictIndexIntegralAddImm(i8),
    /// `grid.get_i32(buf_local, idx_local)` — push i32 without native Call.
    GridGetI32(usize, usize),
    /// `grid.set_i32(buf, idx, val)` locals.
    GridSetI32(usize, usize, usize),
    /// `grid.get_u8(buf, idx)`.
    GridGetU8(usize, usize),
    /// `grid.set_u8(buf, idx, val)`.
    GridSetU8(usize, usize, usize),
    /// `grid.test_blocked(bitmap, idx)`.
    GridTestBlocked(usize, usize),
    /// `grid.heap_push(heap, node, f_buf)` locals.
    GridHeapPush(usize, usize, usize),
    /// `grid.heap_pop(heap)` → unpack into f_slot, node_slot locals.
    GridHeapPopUnpack2(usize, usize, usize),
    /// `grid.heap_len(heap)`.
    GridHeapLen(usize),
    /// Constructor epilogue: `@init(this, constructor_params…)` via [`dispatch_special_by_id`].
    InvokeSpecialInit(usize, usize),
}

/// Количество слотов гистограммы профиля (по одному на вариант [`OpCode`] без параметров).
pub const PROFILE_OPCODE_SLOTS: usize = 111;

/// Индекс «неизвестного» опкода в гистограмме alloc/get (вне [`execute_instruction`]).
pub const PROFILE_OPCODE_UNKNOWN: u8 = 255;

impl OpCode {
    /// Плотный индекс варианта для `--features profile` (без аллокаций в hot path).
    pub fn profile_index(&self) -> u8 {
        match self {
            OpCode::Constant(_) => 0,
            OpCode::LoadLocal(_) => 1,
            OpCode::StoreLocal(_) => 2,
            OpCode::LoadGlobal(_) => 3,
            OpCode::StoreGlobal(_) => 4,
            OpCode::Add => 5,
            OpCode::Sub => 6,
            OpCode::Mul => 7,
            OpCode::MatMul => 8,
            OpCode::BinaryOp(_) => 9,
            OpCode::Div => 10,
            OpCode::IntDiv => 11,
            OpCode::Mod => 12,
            OpCode::Pow => 13,
            OpCode::Negate => 14,
            OpCode::BitAnd => 15,
            OpCode::BitOr => 16,
            OpCode::BitXor => 17,
            OpCode::ShiftLeft => 18,
            OpCode::ShiftRight => 19,
            OpCode::BitNot => 20,
            OpCode::Not => 21,
            OpCode::Or => 22,
            OpCode::And => 23,
            OpCode::Equal => 24,
            OpCode::Greater => 25,
            OpCode::Less => 26,
            OpCode::NotEqual => 27,
            OpCode::GreaterEqual => 28,
            OpCode::LessEqual => 29,
            OpCode::In => 30,
            OpCode::JumpLabel(_) => 31,
            OpCode::JumpIfFalseLabel(_) => 32,
            OpCode::JumpIfLocalHeapArrayEmptyLabel(_, _) => 33,
            OpCode::ForRange(_, _, _, _, _) => 34,
            OpCode::ForRangeNext(_) => 35,
            OpCode::PopForRange => 36,
            OpCode::CoerceForInIterable(_) => 37,
            OpCode::ForIterableNext(_) => 38,
            OpCode::Jump8(_) => 39,
            OpCode::Jump16(_) => 40,
            OpCode::Jump32(_) => 41,
            OpCode::JumpIfFalse8(_) => 42,
            OpCode::JumpIfFalse16(_) => 43,
            OpCode::JumpIfFalse32(_) => 44,
            OpCode::JumpIfLocalHeapArrayEmpty8(_, _) => 45,
            OpCode::JumpIfLocalHeapArrayEmpty16(_, _) => 46,
            OpCode::JumpIfLocalHeapArrayEmpty32(_, _) => 47,
            OpCode::Call(_) => 48,
            OpCode::CallWithUnpack(_) => 49,
            OpCode::CallVariadic(_) => 50,
            OpCode::Return => 51,
            OpCode::Yield(_) => 52,
            OpCode::YieldAwaitInput(_, _) => 53,
            OpCode::GeneratorDone => 54,
            OpCode::GeneratorDoneWithFinal => 55,
            OpCode::MakeArray(_) => 56,
            OpCode::MakeArrayDynamic => 57,
            OpCode::GetArrayLength => 58,
            OpCode::GetArrayElement => 59,
            OpCode::GetArraySlice => 60,
            OpCode::SetArrayElement => 61,
            OpCode::SetArraySlice => 62,
            OpCode::TableFilter => 63,
            OpCode::TableFilterPred(_) => 64,
            OpCode::Clone => 65,
            OpCode::MakeTuple(_) => 66,
            OpCode::MakeObject(_) => 67,
            OpCode::UnpackObject(_) => 68,
            OpCode::MakeObjectDynamic => 69,
            OpCode::MakeSet(_) => 70,
            OpCode::MakeSetDynamic => 71,
            OpCode::BeginTry(_) => 72,
            OpCode::EndTry => 73,
            OpCode::Catch(_) => 74,
            OpCode::EndCatch => 75,
            OpCode::Throw(_) => 76,
            OpCode::PopExceptionHandler => 77,
            OpCode::Pop => 78,
            OpCode::Dup => 79,
            OpCode::FormatInterp(_) => 80,
            OpCode::Import(_) => 81,
            OpCode::ImportFrom(_, _) => 82,
            OpCode::RegAdd(_, _, _) => 83,
            OpCode::HeappopUnpack2(_, _) => 84,
            OpCode::HeappushFlat => 85,
            OpCode::DivmodUnpack2(_, _) => 86,
            OpCode::ObjectGetIntegral => 87,
            OpCode::HeappopFlat => 88,
            OpCode::SetDiscardIntegral => 89,
            OpCode::SetAddIntegral => 90,
            OpCode::ObjectIndexIntegral => 91,
            OpCode::ObjectClear => 92,
            OpCode::ObjectSetIntegral => 93,
            OpCode::InIntegral => 94,
            OpCode::NotInIntegral => 95,
            OpCode::AbsI32 => 96,
            OpCode::InGridBounds => 97,
            OpCode::InGridBoundsOut => 98,
            OpCode::FScoreStaleCheck => 99,
            OpCode::DictGetIntegralLt => 100,
            OpCode::DictIndexIntegralAddImm(_) => 101,
            OpCode::GridGetI32(_, _) => 102,
            OpCode::GridSetI32(_, _, _) => 103,
            OpCode::GridGetU8(_, _) => 104,
            OpCode::GridSetU8(_, _, _) => 105,
            OpCode::GridTestBlocked(_, _) => 106,
            OpCode::GridHeapPush(_, _, _) => 107,
            OpCode::GridHeapPopUnpack2(_, _, _) => 108,
            OpCode::GridHeapLen(_) => 109,
            OpCode::InvokeSpecialInit(_, _) => 110,
        }
    }

    /// Имя варианта опкода без параметров для агрегации в профиле (MakeArray(8) и MakeArray(3) → "MakeArray").
    pub fn variant_name(&self) -> &'static str {
        Self::profile_name(self.profile_index())
    }

    pub fn profile_name(index: u8) -> &'static str {
        const NAMES: [&str; PROFILE_OPCODE_SLOTS] = [
            "Constant",
            "LoadLocal",
            "StoreLocal",
            "LoadGlobal",
            "StoreGlobal",
            "Add",
            "Sub",
            "Mul",
            "MatMul",
            "BinaryOp",
            "Div",
            "IntDiv",
            "Mod",
            "Pow",
            "Negate",
            "BitAnd",
            "BitOr",
            "BitXor",
            "ShiftLeft",
            "ShiftRight",
            "BitNot",
            "Not",
            "Or",
            "And",
            "Equal",
            "Greater",
            "Less",
            "NotEqual",
            "GreaterEqual",
            "LessEqual",
            "In",
            "JumpLabel",
            "JumpIfFalseLabel",
            "JumpIfLocalHeapArrayEmptyLabel",
            "ForRange",
            "ForRangeNext",
            "PopForRange",
            "CoerceForInIterable",
            "ForIterableNext",
            "Jump8",
            "Jump16",
            "Jump32",
            "JumpIfFalse8",
            "JumpIfFalse16",
            "JumpIfFalse32",
            "JumpIfLocalHeapArrayEmpty8",
            "JumpIfLocalHeapArrayEmpty16",
            "JumpIfLocalHeapArrayEmpty32",
            "Call",
            "CallWithUnpack",
            "CallVariadic",
            "Return",
            "Yield",
            "YieldAwaitInput",
            "GeneratorDone",
            "GeneratorDoneWithFinal",
            "MakeArray",
            "MakeArrayDynamic",
            "GetArrayLength",
            "GetArrayElement",
            "GetArraySlice",
            "SetArrayElement",
            "SetArraySlice",
            "TableFilter",
            "TableFilterPred",
            "Clone",
            "MakeTuple",
            "MakeObject",
            "UnpackObject",
            "MakeObjectDynamic",
            "MakeSet",
            "MakeSetDynamic",
            "BeginTry",
            "EndTry",
            "Catch",
            "EndCatch",
            "Throw",
            "PopExceptionHandler",
            "Pop",
            "Dup",
            "FormatInterp",
            "Import",
            "ImportFrom",
            "RegAdd",
            "HeappopUnpack2",
            "HeappushFlat",
            "DivmodUnpack2",
            "ObjectGetIntegral",
            "HeappopFlat",
            "SetDiscardIntegral",
            "SetAddIntegral",
            "ObjectIndexIntegral",
            "ObjectClear",
            "ObjectSetIntegral",
            "InIntegral",
            "NotInIntegral",
            "AbsI32",
            "InGridBounds",
            "InGridBoundsOut",
            "FScoreStaleCheck",
            "DictGetIntegralLt",
            "DictIndexIntegralAddImm",
            "GridGetI32",
            "GridSetI32",
            "GridGetU8",
            "GridSetU8",
            "GridTestBlocked",
            "GridHeapPush",
            "GridHeapPopUnpack2",
            "GridHeapLen",
            "InvokeSpecialInit",
        ];
        NAMES.get(index as usize).copied().unwrap_or("?")
    }
}
