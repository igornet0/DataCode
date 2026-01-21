/// Разрешение переменных (локальных/глобальных)

use crate::bytecode::OpCode;
use crate::common::error::LangError;
use crate::compiler::context::CompilationContext;

pub struct VariableResolver;

impl VariableResolver {
    /// Разрешает переменную для присваивания и сохраняет значение
    pub fn resolve_and_store(
        ctx: &mut CompilationContext,
        name: &str,
        is_global: bool,
        line: usize,
    ) -> Result<(), LangError> {
        if is_global {
            // Явное объявление глобальной переменной
            let global_index = if let Some(&idx) = ctx.scope.globals.get(name) {
                idx
            } else {
                let idx = ctx.scope.globals.len();
                ctx.scope.globals.insert(name.to_string(), idx);
                idx
            };
            ctx.chunk.global_names.insert(global_index, name.to_string());
            ctx.chunk.explicit_global_names.insert(global_index, name.to_string());
            ctx.chunk.write_with_line(OpCode::StoreGlobal(global_index), line);
        } else {
            // Локальная переменная или глобальная на верхнем уровне
            if let Some(local_index) = ctx.scope.resolve_local(name) {
                // Локальная переменная уже объявлена - обновляем
                ctx.chunk.write_with_line(OpCode::StoreLocal(local_index), line);
            } else if ctx.current_function.is_some() {
                // Мы находимся внутри функции - объявляем новую локальную переменную
                let index = ctx.scope.declare_local(name);
                ctx.chunk.write_with_line(OpCode::StoreLocal(index), line);
            } else {
                // Переменная не найдена локально - проверяем, является ли она глобальной
                if let Some(&global_index) = ctx.scope.globals.get(name) {
                    // Глобальная переменная уже существует - обновляем
                    ctx.chunk.global_names.insert(global_index, name.to_string());
                    ctx.chunk.write_with_line(OpCode::StoreGlobal(global_index), line);
                } else {
                    // Новая глобальная переменная на верхнем уровне
                    let global_index = ctx.scope.globals.len();
                    ctx.scope.globals.insert(name.to_string(), global_index);
                    ctx.chunk.global_names.insert(global_index, name.to_string());
                    ctx.chunk.write_with_line(OpCode::StoreGlobal(global_index), line);
                }
            }
        }
        Ok(())
    }

    /// Разрешает переменную для загрузки значения
    pub fn resolve_and_load(
        ctx: &mut CompilationContext,
        name: &str,
        line: usize,
    ) -> Result<(), LangError> {
        if let Some(local_index) = ctx.scope.resolve_local(name) {
            ctx.chunk.write_with_line(OpCode::LoadLocal(local_index), line);
        } else if let Some(&global_index) = ctx.scope.globals.get(name) {
            ctx.chunk.write_with_line(OpCode::LoadGlobal(global_index), line);
        } else {
            // Переменная не найдена - создаем новый глобальный индекс
            let global_index = ctx.scope.globals.len();
            ctx.scope.globals.insert(name.to_string(), global_index);
            ctx.chunk.global_names.insert(global_index, name.to_string());
            ctx.chunk.write_with_line(OpCode::LoadGlobal(global_index), line);
        }
        Ok(())
    }

    /// Разрешает переменную для присваивания с оператором (например, +=)
    /// Возвращает true, если переменная была локальной, false если глобальной
    pub fn resolve_for_assign_op(
        ctx: &mut CompilationContext,
        name: &str,
        line: usize,
    ) -> Result<bool, LangError> {
        if let Some(local_index) = ctx.scope.resolve_local(name) {
            // Локальная переменная найдена
            ctx.chunk.write_with_line(OpCode::LoadLocal(local_index), line);
            Ok(true)
        } else if ctx.current_function.is_some() {
            // Мы находимся внутри функции - создаем локальную переменную
            let index = ctx.scope.declare_local(name);
            // Загружаем 0 как начальное значение
            let zero_index = ctx.chunk.add_constant(crate::common::value::Value::Number(0.0));
            ctx.chunk.write_with_line(OpCode::Constant(zero_index), line);
            ctx.chunk.write_with_line(OpCode::StoreLocal(index), line);
            ctx.chunk.write_with_line(OpCode::LoadLocal(index), line);
            Ok(true)
        } else if let Some(&global_index) = ctx.scope.globals.get(name) {
            // Глобальная переменная найдена
            ctx.chunk.write_with_line(OpCode::LoadGlobal(global_index), line);
            Ok(false)
        } else {
            // Переменная не найдена - создаем новую локальную переменную
            let index = ctx.scope.declare_local(name);
            let zero_index = ctx.chunk.add_constant(crate::common::value::Value::Number(0.0));
            ctx.chunk.write_with_line(OpCode::Constant(zero_index), line);
            ctx.chunk.write_with_line(OpCode::StoreLocal(index), line);
            ctx.chunk.write_with_line(OpCode::LoadLocal(index), line);
            Ok(true)
        }
    }

    /// Сохраняет результат обратно в переменную после операции
    pub fn store_after_operation(
        ctx: &mut CompilationContext,
        name: &str,
        is_local: bool,
        line: usize,
    ) -> Result<(), LangError> {
        if is_local {
            if let Some(local_index) = ctx.scope.resolve_local(name) {
                ctx.chunk.write_with_line(OpCode::StoreLocal(local_index), line);
                ctx.chunk.write_with_line(OpCode::LoadLocal(local_index), line);
            }
        } else {
            if let Some(&global_index) = ctx.scope.globals.get(name) {
                ctx.chunk.global_names.insert(global_index, name.to_string());
                ctx.chunk.write_with_line(OpCode::StoreGlobal(global_index), line);
                ctx.chunk.write_with_line(OpCode::LoadGlobal(global_index), line);
            }
        }
        Ok(())
    }
}

