/// Компиляция массивов, кортежей и индексации

use crate::parser::ast::Expr;
use crate::lexer::TokenKind;
use crate::bytecode::OpCode;
use crate::common::error::LangError;
use crate::common::value::Value;
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
            use crate::common::value::Value;
            use crate::parser::ast::ObjectPair;
            let has_spread = pairs.iter().any(|p| matches!(p, ObjectPair::Spread(_)));
            if !has_spread {
                for p in pairs.iter().rev() {
                    if let ObjectPair::KeyValue(key, value) = p {
                        let key_index = ctx.chunk.add_constant(Value::String(key.clone()));
                        ctx.chunk.write_with_line(OpCode::Constant(key_index), *line);
                        expr::compile_expr(ctx, value)?;
                    }
                }
                ctx.chunk.write_with_line(OpCode::MakeObject(pairs.len()), *line);
            } else {
                let count_slot = ctx.scope.declare_local("__object_pair_count");
                let zero_index = ctx.chunk.add_constant(Value::Number(0.0));
                ctx.chunk.write_with_line(OpCode::Constant(zero_index), *line);
                ctx.chunk.write_with_line(OpCode::StoreLocal(count_slot), *line);
                for p in pairs {
                    match p {
                        ObjectPair::KeyValue(key, value) => {
                            let key_index = ctx.chunk.add_constant(Value::String(key.clone()));
                            ctx.chunk.write_with_line(OpCode::Constant(key_index), *line);
                            expr::compile_expr(ctx, value)?;
                            ctx.chunk.write_with_line(OpCode::LoadLocal(count_slot), *line);
                            let one_index = ctx.chunk.add_constant(Value::Number(1.0));
                            ctx.chunk.write_with_line(OpCode::Constant(one_index), *line);
                            ctx.chunk.write_with_line(OpCode::Add, *line);
                            ctx.chunk.write_with_line(OpCode::StoreLocal(count_slot), *line);
                        }
                        ObjectPair::Spread(expr) => {
                            expr::compile_expr(ctx, expr)?;
                            ctx.chunk.write_with_line(OpCode::UnpackObject(count_slot), *line);
                        }
                    }
                }
                ctx.chunk.write_with_line(OpCode::LoadLocal(count_slot), *line);
                ctx.chunk.write_with_line(OpCode::MakeObjectDynamic, *line);
            }
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
        Expr::TableFilter { table, column, op, value, line } => {
            *ctx.current_line = *line;
            expr::compile_expr(ctx, table)?;
            let column_index = ctx.chunk.add_constant(Value::String(column.clone()));
            ctx.chunk.write_with_line(OpCode::Constant(column_index), *line);
            let op_str = match op {
                TokenKind::Equal => "=",
                TokenKind::EqualEqual => "==",
                TokenKind::BangEqual => "!=",
                TokenKind::Less => "<",
                TokenKind::Greater => ">",
                TokenKind::LessEqual => "<=",
                TokenKind::GreaterEqual => ">=",
                _ => "==",
            };
            let op_index = ctx.chunk.add_constant(Value::String(op_str.to_string()));
            ctx.chunk.write_with_line(OpCode::Constant(op_index), *line);
            expr::compile_expr(ctx, value)?;
            ctx.chunk.write_with_line(OpCode::TableFilter, *line);
            Ok(())
        }
        _ => Err(LangError::ParseError {
            message: "Expected ArrayLiteral, TupleLiteral, ObjectLiteral, ArrayIndex, or TableFilter expression".to_string(),
            line: expr.line(),
            file: None,
        }),
    }
}

