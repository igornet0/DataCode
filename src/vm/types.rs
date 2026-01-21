// Types and structures for VM

use crate::common::value::Value;

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

/// Статус выполнения одного шага VM
#[derive(Debug)]
pub enum VMStatus {
    Continue,        // Продолжить выполнение
    Return(Value),   // Возврат из функции с значением
    FrameEnded,      // Фрейм завершился без return
}

// Thread-local storage для хранения контекста VM во время вызова нативных функций
// Это позволяет нативным функциям вызывать пользовательские функции
// Определено в vm.rs для избежания циклических зависимостей

