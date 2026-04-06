// Types and structures for VM

use crate::common::value::Value;
use crate::common::value_store::ValueId;

/// Вход для `OpCode::YieldAwaitInput`: от `.next()` / `final()` / `.send(v)`.
#[derive(Clone, Debug)]
pub(crate) enum PendingGeneratorSend {
    /// `.next()` после yield-await: в слот — RHS последнего yield (как `send(None)` в Python).
    NextDefaultRhs,
    /// Явное значение из `.send(v)`, в том числе `null`.
    Explicit(Value),
}

/// Структура для хранения явной связи между колонками таблиц
#[derive(Debug, Clone)]
pub struct ExplicitRelation {
    pub source_table_name: String,
    pub source_column_name: String,
    pub target_table_name: String,
    pub target_column_name: String,
}

/// Структура для хранения явного первичного ключа таблицы
#[derive(Debug, Clone)]
pub struct ExplicitPrimaryKey {
    pub table_name: String,
    pub column_name: String,
}

/// Info for a merged module: used to resolve Value::ModuleFunction { module_uid, local_index } at Call.
#[derive(Clone, Debug)]
pub struct ModuleInfo {
    pub name: String,
    pub function_offset: usize,
    pub function_count: usize,
}

/// Статус выполнения одного шага VM (Stage 1: Return carries ValueId)
#[derive(Debug)]
pub enum VMStatus {
    Continue,        // Продолжить выполнение
    Return(ValueId), // Возврат из функции (значение в store по id)
    FrameEnded,      // Фрейм завершился без return
    /// `stream fn`: yield значение; фрейм генератора остаётся на стеке.
    GeneratorYield(ValueId),
    /// `stream fn`: yield + ожидание значения из `.send()` (`x = return expr`), пока `pending_generator_send` пуст.
    GeneratorYieldAwait(ValueId, usize),
    /// `stream fn`: генератор завершён; `None` — без финального значения, `Some(id)` — `ereturn expr`.
    GeneratorDone(Option<ValueId>),
}

// Thread-local storage для хранения контекста VM во время вызова нативных функций
// Это позволяет нативным функциям вызывать пользовательские функции
// Определено в vm.rs для избежания циклических зависимостей
