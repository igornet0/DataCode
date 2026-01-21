/// Компиляция for statements

use crate::parser::ast::{Stmt, UnpackPattern};
use crate::bytecode::OpCode;
use crate::common::error::LangError;
use crate::common::value::Value;
use crate::compiler::context::{CompilationContext, LoopContext};
use crate::compiler::expr;
use crate::compiler::stmt;
use crate::compiler::unpack;

pub fn compile_for(ctx: &mut CompilationContext, stmt: &Stmt) -> Result<(), LangError> {
    if let Stmt::For { pattern, iterable, body, line } = stmt {
        *ctx.current_line = *line;
        
        // Начинаем новую область видимости для переменных цикла
        ctx.scope.begin_scope();
        
        // Компилируем итерируемое выражение (оно должно быть массивом)
        expr::compile_expr(ctx, iterable)?;
        
        // Сохраняем массив во временную переменную (локальную)
        let array_local = ctx.scope.declare_local("__array_iter");
        ctx.chunk.write_with_line(OpCode::StoreLocal(array_local), *line);
        
        // Создаем переменную для индекса
        let index_local = ctx.scope.declare_local("__index_iter");
        // Инициализируем индекс как 0
        let zero_index = ctx.chunk.add_constant(Value::Number(0.0));
        ctx.chunk.write_with_line(OpCode::Constant(zero_index), *line);
        ctx.chunk.write_with_line(OpCode::StoreLocal(index_local), *line);
        
        // Проверяем, является ли это простым случаем (одна переменная без распаковки)
        let is_simple_case = pattern.len() == 1 && matches!(pattern[0], UnpackPattern::Variable(_));
        
        // Подсчитываем количество переменных (не wildcard) для проверки
        let expected_count = if is_simple_case {
            0 // Не распаковываем, просто присваиваем
        } else {
            unpack::count_unpack_variables(pattern)
        };
        
        // Объявляем переменные из паттерна распаковки
        let var_locals = if is_simple_case {
            // Простой случай: одна переменная
            if let UnpackPattern::Variable(name) = &pattern[0] {
                let index = ctx.scope.declare_local(name);
                vec![Some(index)]
            } else {
                vec![None]
            }
        } else {
            unpack::declare_unpack_pattern_variables(pattern, ctx.scope, *line)?
        };
        
        // Создаем метки для цикла
        let loop_start_label = ctx.labels.create_label();
        let continue_label = ctx.labels.create_label();
        let loop_end_label = ctx.labels.create_label();
        
        // Помечаем начало цикла
        ctx.labels.mark_label(loop_start_label, ctx.chunk.code.len());
        
        // Проверяем условие: индекс < длина массива
        ctx.chunk.write_with_line(OpCode::LoadLocal(index_local), *line);
        ctx.chunk.write_with_line(OpCode::LoadLocal(array_local), *line);
        ctx.chunk.write_with_line(OpCode::GetArrayLength, *line);
        ctx.chunk.write_with_line(OpCode::Less, *line);
        
        // Если условие false, выходим из цикла
        // ВАЖНО: JumpIfFalse удаляет результат сравнения со стека перед переходом
        ctx.labels.emit_jump(ctx.chunk, *ctx.current_line, true, loop_end_label)?;
        
        // Загружаем элемент массива по индексу
        ctx.chunk.write_with_line(OpCode::LoadLocal(array_local), *line);
        ctx.chunk.write_with_line(OpCode::LoadLocal(index_local), *line);
        ctx.chunk.write_with_line(OpCode::GetArrayElement, *line);
        
        // Распаковываем элемент в переменные (или просто присваиваем для простого случая)
        if is_simple_case {
            // Простой случай: просто сохраняем элемент в переменную
            if let Some(local_index) = var_locals[0] {
                ctx.chunk.write_with_line(OpCode::StoreLocal(local_index), *line);
            }
        } else {
            unpack::compile_unpack_pattern(
                pattern,
                &var_locals,
                expected_count,
                ctx.chunk,
                ctx.scope,
                ctx.labels,
                *ctx.current_line,
                *line,
            )?;
        }
        
        // Создаем контекст цикла
        let loop_context = LoopContext {
            continue_label: continue_label, // continue переходит к инкременту
            break_label: loop_end_label,    // break переходит к концу цикла
        };
        ctx.loop_contexts.push(loop_context);
        
        // Компилируем тело цикла
        for stmt in body {
            stmt::compile_stmt(ctx, stmt, true)?;
        }
        
        // Помечаем метку continue (начало инкремента индекса)
        ctx.labels.mark_label(continue_label, ctx.chunk.code.len());
        
        // Инкрементируем индекс
        ctx.chunk.write_with_line(OpCode::LoadLocal(index_local), *line);
        let one_index = ctx.chunk.add_constant(Value::Number(1.0));
        ctx.chunk.write_with_line(OpCode::Constant(one_index), *line);
        ctx.chunk.write_with_line(OpCode::Add, *line);
        ctx.chunk.write_with_line(OpCode::StoreLocal(index_local), *line);
        
        // Переход к началу цикла
        ctx.labels.emit_loop(ctx.chunk, *ctx.current_line, loop_start_label)?;
        
        // Помечаем конец цикла (после JumpLabel, который был добавлен emit_loop)
        // ВАЖНО: loop_end_label должна быть размещена ПОСЛЕ всех инструкций цикла
        // и ПЕРЕД end_scope(), чтобы гарантировать, что после перехода к этой метке
        // стек чист и нет инструкций, ожидающих значений на стеке
        // Перезаписываем временное значение правильным IP
        let loop_end_ip = ctx.chunk.code.len();
        ctx.labels.mark_label(loop_end_label, loop_end_ip);
        
        // ВАЖНО: На этом этапе стек должен быть чистым.
        // JumpIfFalse уже удалил результат сравнения при выходе из цикла,
        // а StoreLocal удалил результат инкремента при продолжении цикла.
        // Если стек не чист, это может вызвать ошибку при возврате из функции.
        // Проверяем, что после loop_end_label нет инструкций, которые ожидают значения на стеке.
        
        // Заканчиваем область видимости
        ctx.scope.end_scope();
        ctx.loop_contexts.pop();
        
        Ok(())
    } else {
        Err(LangError::ParseError {
            message: "Expected For statement".to_string(),
            line: stmt.line(),
        })
    }
}

