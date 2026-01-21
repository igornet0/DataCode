/// Компиляция массивов, кортежей и индексации

use crate::parser::ast::Expr;
use crate::bytecode::OpCode;
use crate::common::error::LangError;
use crate::compiler::context::CompilationContext;
use crate::compiler::expr;

pub fn compile_array(ctx: &mut CompilationContext, expr: &Expr) -> Result<(), LangError> {
    match expr {
        Expr::ArrayLiteral { elements, line } => {
            *ctx.current_line = *line;
            // Компилируем каждый элемент массива
            for element in elements {
                expr::compile_expr(ctx, element)?;
            }
            // Создаем массив из элементов на стеке
            let arity = elements.len();
            ctx.chunk.write_with_line(OpCode::MakeArray(arity), *line);
            Ok(())
        }
        Expr::TupleLiteral { elements, line } => {
            *ctx.current_line = *line;
            // Компилируем каждый элемент кортежа
            for element in elements {
                expr::compile_expr(ctx, element)?;
            }
            // Создаем кортеж из элементов на стеке
            // Используем MakeArray, но в VM будем создавать Tuple
            let arity = elements.len();
            ctx.chunk.write_with_line(OpCode::MakeTuple(arity), *line);
            Ok(())
        }
        Expr::ObjectLiteral { pairs, line } => {
            *ctx.current_line = *line;
            // Компилируем пары (ключ, значение) в обратном порядке
            // На стеке будут: [key1, value1, key2, value2, ...]
            for (key, value) in pairs.iter().rev() {
                // Сначала добавляем ключ как строку
                use crate::common::value::Value;
                let key_index = ctx.chunk.add_constant(Value::String(key.clone()));
                ctx.chunk.write_with_line(OpCode::Constant(key_index), *line);
                // Затем компилируем значение
                expr::compile_expr(ctx, value)?;
            }
            // Создаем объект из пар на стеке
            let pair_count = pairs.len();
            ctx.chunk.write_with_line(OpCode::MakeObject(pair_count), *line);
            Ok(())
        }
        Expr::ArrayIndex { array, index, line } => {
            *ctx.current_line = *line;
            // Компилируем выражение массива (оно должно быть на стеке первым)
            expr::compile_expr(ctx, array)?;
            // Компилируем индексное выражение
            expr::compile_expr(ctx, index)?;
            // Получаем элемент массива по индексу
            ctx.chunk.write_with_line(OpCode::GetArrayElement, *line);
            Ok(())
        }
        _ => Err(LangError::ParseError {
            message: "Expected ArrayLiteral, TupleLiteral, ObjectLiteral, or ArrayIndex expression".to_string(),
            line: expr.line(),
        }),
    }
}

