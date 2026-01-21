/// Компиляция let statements

use crate::parser::ast::{Stmt, Expr};
use crate::bytecode::OpCode;
use crate::common::error::LangError;
use crate::common::value::Value;
use crate::compiler::context::CompilationContext;
use crate::compiler::expr;
use crate::compiler::variable::VariableResolver;

pub fn compile_let(ctx: &mut CompilationContext, stmt: &Stmt, _pop_value: bool) -> Result<(), LangError> {
    if let Stmt::Let { name, value, is_global, line } = stmt {
        *ctx.current_line = *line;
        
        // Проверяем, является ли value UnpackAssign (распаковка кортежа)
        if let Expr::UnpackAssign { names, value: tuple_value, .. } = value {
            // Распаковка кортежа в let statement
            // tuple_value - это &Box<Expr> в match, нужно получить &Expr
            expr::compile_expr(ctx, &**tuple_value)?;
            
            // Сохраняем кортеж во временную переменную
            let tuple_temp = ctx.scope.declare_local(&format!("__tuple_temp_{}", line));
            ctx.chunk.write_with_line(OpCode::StoreLocal(tuple_temp), *line);
            
            // Для каждой переменной извлекаем элемент кортежа и сохраняем
            for (index, var_name) in names.iter().enumerate() {
                // Загружаем кортеж
                ctx.chunk.write_with_line(OpCode::LoadLocal(tuple_temp), *line);
                // Загружаем индекс
                let index_const = ctx.chunk.add_constant(Value::Number(index as f64));
                ctx.chunk.write_with_line(OpCode::Constant(index_const), *line);
                // Получаем элемент по индексу
                ctx.chunk.write_with_line(OpCode::GetArrayElement, *line);
                
                // Сохраняем в переменную
                if *is_global {
                    // Глобальная переменная
                    let global_index = if let Some(&idx) = ctx.scope.globals.get(var_name) {
                        idx
                    } else {
                        let idx = ctx.scope.globals.len();
                        ctx.scope.globals.insert(var_name.clone(), idx);
                        idx
                    };
                    ctx.chunk.global_names.insert(global_index, var_name.clone());
                    ctx.chunk.explicit_global_names.insert(global_index, var_name.clone());
                    ctx.chunk.write_with_line(OpCode::StoreGlobal(global_index), *line);
                } else {
                    // Локальная переменная
                    if let Some(local_index) = ctx.scope.resolve_local(var_name) {
                        ctx.chunk.write_with_line(OpCode::StoreLocal(local_index), *line);
                    } else if ctx.current_function.is_some() {
                        let var_index = ctx.scope.declare_local(var_name);
                        ctx.chunk.write_with_line(OpCode::StoreLocal(var_index), *line);
                    } else {
                        // На верхнем уровне - проверяем, есть ли глобальная переменная
                        if let Some(&global_index) = ctx.scope.globals.get(var_name) {
                            ctx.chunk.global_names.insert(global_index, var_name.clone());
                            ctx.chunk.write_with_line(OpCode::StoreGlobal(global_index), *line);
                        } else {
                            let global_index = ctx.scope.globals.len();
                            ctx.scope.globals.insert(var_name.clone(), global_index);
                            ctx.chunk.global_names.insert(global_index, var_name.clone());
                            ctx.chunk.write_with_line(OpCode::StoreGlobal(global_index), *line);
                        }
                    }
                }
            }
            
            // Загружаем последнее значение на стек
            if let Some(last_name) = names.last() {
                VariableResolver::resolve_and_load(ctx, last_name, *line)?;
            }
            return Ok(());
        }
        
        // Обычное присваивание
        expr::compile_expr(ctx, value)?;
        // Не клонируем автоматически - переменные должны разделять ссылки на массивы/таблицы/объекты
        VariableResolver::resolve_and_store(ctx, name, *is_global, *line)?;
        
        Ok(())
    } else {
        Err(LangError::ParseError {
            message: "Expected Let statement".to_string(),
            line: stmt.line(),
        })
    }
}

