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
    Div,
    IntDiv, // Целочисленное деление (//)
    Mod, // Модуло (%)
    Pow, // Возведение в степень (**)
    Negate, // Унарный минус
    
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
    
    // Специализированный цикл for i in range(...): без материализации диапазона
    /// (var_slot, start_const, end_const, step_const, end_offset). end_offset патчится при finalize.
    ForRange(usize, usize, usize, usize, i32),
    /// back_offset: на сколько инструкций откатить IP (к ForRange)
    ForRangeNext(i32),
    /// Снять одно состояние с for_range_stack (при break из for i in range(...))
    PopForRange,

    // Финальные инструкции с относительными смещениями
    Jump8(i8),          // Безусловный переход с 8-битным смещением [-128, +127]
    Jump16(i16),        // Безусловный переход с 16-битным смещением [-32768, +32767]
    Jump32(i32),        // Безусловный переход с 32-битным смещением [-2^31, +2^31-1]
    JumpIfFalse8(i8),   // Условный переход с 8-битным смещением [-128, +127]
    JumpIfFalse16(i16), // Условный переход с 16-битным смещением [-32768, +32767]
    JumpIfFalse32(i32), // Условный переход с 32-битным смещением [-2^31, +2^31-1]

    // Функции
    Call(usize),         // Вызов функции с количеством аргументов
    CallWithUnpack(usize), // Вызов: один аргумент — объект для распаковки в kwargs; ключи должны совпадать с именами параметров
    Return,              // Возврат из функции

    // Массивы
    MakeArray(usize), // Создать массив из N элементов со стека (compile-time размер)
    MakeArrayDynamic, // Создать массив из N элементов со стека (runtime размер: N на стеке, затем N элементов)
    GetArrayLength,   // Получить длину массива
    GetArrayElement,  // Получить элемент массива по индексу (индекс и массив на стеке)
    SetArrayElement,  // Установить элемент массива/объекта по индексу (значение, индекс, массив/объект на стеке)
    TableFilter,      // Фильтр таблицы: stack [table, column, op, value] → отфильтрованная таблица
    Clone,            // Глубокое клонирование значения на стеке (для массивов и таблиц)
    
    // Кортежи
    MakeTuple(usize), // Создать кортеж из N элементов со стека (compile-time размер)
    
    // Объекты
    MakeObject(usize), // Создать объект из N пар (ключ, значение) со стека (compile-time количество пар)
    /// Распаковать объект со стека: положить пары (ключ, значение) на стек и увеличить счётчик в слоте на число пар.
    UnpackObject(usize), // индекс слота для счётчика пар
    /// Создать объект: со стека снять count, затем 2*count значений (value, key на пару), собрать объект.
    MakeObjectDynamic,

    // Обработка исключений
    BeginTry(usize),        // Начало try блока, аргумент - индекс обработчика в таблице обработчиков
    EndTry,                 // Конец try блока
    Catch(Option<usize>),   // Начало catch блока, Option<usize> - тип ошибки (индекс в таблице типов), None для catch всех
    EndCatch,               // Конец catch блока
    Throw(Option<usize>),    // Выбрасывание исключения (для будущего использования), Option<usize> - тип ошибки
    PopExceptionHandler,     // Удаление обработчика исключений со стека

    // Стек
    Pop, // Удалить значение со стека
    Dup, // Дублировать вершину стека (для short-circuit or/and)

    // Модули
    Import(usize), // Импорт модуля (индекс имени модуля в константах)
    ImportFrom(usize, usize), // from-импорт: (индекс имени модуля, индекс массива элементов импорта в константах)

    // Register VM (этап 1): опкоды с регистрами; компилятор пока не эмитирует.
    /// Add reg[rd] = reg[r1] + reg[r2] (number+number); иначе fallback на store. Регистры — индексы в frame.regs.
    RegAdd(u8, u8, u8),
}

impl OpCode {
    /// Имя варианта опкода без параметров для агрегации в профиле (MakeArray(8) и MakeArray(3) → "MakeArray").
    pub fn variant_name(&self) -> &'static str {
        match self {
            OpCode::Constant(_) => "Constant",
            OpCode::LoadLocal(_) => "LoadLocal",
            OpCode::StoreLocal(_) => "StoreLocal",
            OpCode::LoadGlobal(_) => "LoadGlobal",
            OpCode::StoreGlobal(_) => "StoreGlobal",
            OpCode::Add => "Add",
            OpCode::Sub => "Sub",
            OpCode::Mul => "Mul",
            OpCode::Div => "Div",
            OpCode::IntDiv => "IntDiv",
            OpCode::Mod => "Mod",
            OpCode::Pow => "Pow",
            OpCode::Negate => "Negate",
            OpCode::Not => "Not",
            OpCode::Or => "Or",
            OpCode::And => "And",
            OpCode::Equal => "Equal",
            OpCode::Greater => "Greater",
            OpCode::Less => "Less",
            OpCode::NotEqual => "NotEqual",
            OpCode::GreaterEqual => "GreaterEqual",
            OpCode::LessEqual => "LessEqual",
            OpCode::In => "In",
            OpCode::JumpLabel(_) => "JumpLabel",
            OpCode::JumpIfFalseLabel(_) => "JumpIfFalseLabel",
            OpCode::ForRange(_, _, _, _, _) => "ForRange",
            OpCode::ForRangeNext(_) => "ForRangeNext",
            OpCode::PopForRange => "PopForRange",
            OpCode::Jump8(_) => "Jump8",
            OpCode::Jump16(_) => "Jump16",
            OpCode::Jump32(_) => "Jump32",
            OpCode::JumpIfFalse8(_) => "JumpIfFalse8",
            OpCode::JumpIfFalse16(_) => "JumpIfFalse16",
            OpCode::JumpIfFalse32(_) => "JumpIfFalse32",
            OpCode::Call(_) => "Call",
            OpCode::CallWithUnpack(_) => "CallWithUnpack",
            OpCode::Return => "Return",
            OpCode::MakeArray(_) => "MakeArray",
            OpCode::MakeArrayDynamic => "MakeArrayDynamic",
            OpCode::GetArrayLength => "GetArrayLength",
            OpCode::GetArrayElement => "GetArrayElement",
            OpCode::SetArrayElement => "SetArrayElement",
            OpCode::TableFilter => "TableFilter",
            OpCode::Clone => "Clone",
            OpCode::MakeTuple(_) => "MakeTuple",
            OpCode::MakeObject(_) => "MakeObject",
            OpCode::UnpackObject(_) => "UnpackObject",
            OpCode::MakeObjectDynamic => "MakeObjectDynamic",
            OpCode::BeginTry(_) => "BeginTry",
            OpCode::EndTry => "EndTry",
            OpCode::Catch(_) => "Catch",
            OpCode::EndCatch => "EndCatch",
            OpCode::Throw(_) => "Throw",
            OpCode::PopExceptionHandler => "PopExceptionHandler",
            OpCode::Pop => "Pop",
            OpCode::Dup => "Dup",
            OpCode::Import(_) => "Import",
            OpCode::ImportFrom(_, _) => "ImportFrom",
            OpCode::RegAdd(_, _, _) => "RegAdd",
        }
    }
}

