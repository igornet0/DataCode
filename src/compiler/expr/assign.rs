/// Компиляция присваиваний (Assign, AssignOp, UnpackAssign)

use crate::parser::ast::Expr;
use crate::bytecode::OpCode;
use crate::common::error::LangError;
use crate::common::value::Value;
use crate::lexer::TokenKind;
use crate::compiler::context::CompilationContext;
use crate::compiler::expr;
use crate::compiler::variable::VariableResolver;

pub fn compile_assign(ctx: &mut CompilationContext, expr: &Expr) -> Result<(), LangError> {
    match expr {
        Expr::Assign { name, value, line } => {
            *ctx.current_line = *line;
            // Компилируем значение
            expr::compile_expr(ctx, value)?;
            // Не клонируем автоматически - переменные должны разделять ссылки на массивы/таблицы/объекты
            // Клонирование происходит только при явном вызове .clone()
            
            // Используем VariableResolver для разрешения переменной
            // Проверяем, является ли переменная локальной или глобальной
            if let Some(local_index) = ctx.scope.resolve_local(name) {
                // Локальная переменная найдена - обновляем
                ctx.chunk.write_with_line(OpCode::StoreLocal(local_index), *line);
                ctx.chunk.write_with_line(OpCode::LoadLocal(local_index), *line);
            } else if ctx.current_function.is_some() {
                // Мы находимся внутри функции - создаем локальную переменную, которая скрывает глобальную
                let index = ctx.scope.declare_local(name);
                ctx.chunk.write_with_line(OpCode::StoreLocal(index), *line);
                ctx.chunk.write_with_line(OpCode::LoadLocal(index), *line);
            } else if let Some(&global_index) = ctx.scope.globals.get(name) {
                // Мы в главной функции, глобальная переменная найдена - обновляем
                ctx.chunk.global_names.insert(global_index, name.clone());
                ctx.chunk.write_with_line(OpCode::StoreGlobal(global_index), *line);
                ctx.chunk.write_with_line(OpCode::LoadGlobal(global_index), *line);
            } else {
                // Переменная не найдена ни локально, ни глобально
                if ctx.current_function.is_some() {
                    // Мы находимся внутри функции - создаем локальную переменную
                    let index = ctx.scope.declare_local(name);
                    ctx.chunk.write_with_line(OpCode::StoreLocal(index), *line);
                    ctx.chunk.write_with_line(OpCode::LoadLocal(index), *line);
                } else {
                    // На верхнем уровне - создаем новую глобальную переменную
                    let global_index = ctx.scope.globals.len();
                    ctx.scope.globals.insert(name.clone(), global_index);
                    ctx.chunk.global_names.insert(global_index, name.clone());
                    ctx.chunk.write_with_line(OpCode::StoreGlobal(global_index), *line);
                    ctx.chunk.write_with_line(OpCode::LoadGlobal(global_index), *line);
                }
            }
            Ok(())
        }
        Expr::AssignOp { name, op, value, line } => {
            *ctx.current_line = *line;
            compile_assign_op(ctx, name, op, value, *line)
        }
        Expr::UnpackAssign { names, value, line } => {
            *ctx.current_line = *line;
            compile_unpack_assign(ctx, names, value, *line)
        }
        _ => Err(LangError::ParseError {
            message: "Expected Assign, AssignOp, or UnpackAssign expression".to_string(),
            line: expr.line(),
        }),
    }
}

fn compile_assign_op(
    ctx: &mut CompilationContext,
    name: &str,
    op: &TokenKind,
    value: &Expr,
    line: usize,
) -> Result<(), LangError> {
    // Оператор присваивания: a += b эквивалентно a = a + b
    // Используем VariableResolver для разрешения переменной
    let is_local = VariableResolver::resolve_for_assign_op(ctx, name, line)?;
    
    // Компилируем правую часть
    expr::compile_expr(ctx, value)?;
    
    // Выполняем операцию
    match op {
        TokenKind::PlusEqual => ctx.chunk.write_with_line(OpCode::Add, line),
        TokenKind::MinusEqual => ctx.chunk.write_with_line(OpCode::Sub, line),
        TokenKind::StarEqual => ctx.chunk.write_with_line(OpCode::Mul, line),
        TokenKind::StarStarEqual => ctx.chunk.write_with_line(OpCode::Pow, line),
        TokenKind::SlashEqual => ctx.chunk.write_with_line(OpCode::Div, line),
        TokenKind::SlashSlashEqual => ctx.chunk.write_with_line(OpCode::IntDiv, line),
        TokenKind::PercentEqual => ctx.chunk.write_with_line(OpCode::Mod, line),
        _ => {
            return Err(LangError::ParseError {
                message: format!("Unknown assignment operator: {:?}", op),
                line,
            });
        }
    }
    
    // Сохраняем результат обратно
    VariableResolver::store_after_operation(ctx, name, is_local, line)?;
    
    Ok(())
}

fn compile_unpack_assign(
    ctx: &mut CompilationContext,
    names: &[String],
    value: &Expr,
    line: usize,
) -> Result<(), LangError> {
    // Распаковка кортежа: a, b, c = tuple_expr
    // Компилируем правую часть (должна вернуть кортеж)
    expr::compile_expr(ctx, value)?;
    
    // Сохраняем кортеж во временную переменную, чтобы можно было извлекать элементы
    let tuple_temp = ctx.scope.declare_local(&format!("__tuple_temp_{}", line));
    ctx.chunk.write_with_line(OpCode::StoreLocal(tuple_temp), line);
    
    // Для каждой переменной извлекаем элемент кортежа и сохраняем
    for (index, name) in names.iter().enumerate() {
        // Загружаем кортеж
        ctx.chunk.write_with_line(OpCode::LoadLocal(tuple_temp), line);
        // Загружаем индекс
        let index_const = ctx.chunk.add_constant(Value::Number(index as f64));
        ctx.chunk.write_with_line(OpCode::Constant(index_const), line);
        // Получаем элемент по индексу
        ctx.chunk.write_with_line(OpCode::GetArrayElement, line);
        
        // Сохраняем в переменную
        if let Some(local_index) = ctx.scope.resolve_local(name) {
            // Локальная переменная найдена - обновляем
            ctx.chunk.write_with_line(OpCode::StoreLocal(local_index), line);
        } else if ctx.current_function.is_some() {
            // Мы находимся внутри функции - создаем локальную переменную
            let var_index = ctx.scope.declare_local(name);
            ctx.chunk.write_with_line(OpCode::StoreLocal(var_index), line);
        } else {
            // На верхнем уровне - проверяем, есть ли глобальная переменная
            if let Some(&global_index) = ctx.scope.globals.get(name) {
                // Глобальная переменная найдена - обновляем
                ctx.chunk.global_names.insert(global_index, name.clone());
                ctx.chunk.write_with_line(OpCode::StoreGlobal(global_index), line);
            } else {
                // Новая глобальная переменная на верхнем уровне
                let global_index = ctx.scope.globals.len();
                ctx.scope.globals.insert(name.clone(), global_index);
                ctx.chunk.global_names.insert(global_index, name.clone());
                ctx.chunk.write_with_line(OpCode::StoreGlobal(global_index), line);
            }
        }
    }
    
    // Загружаем последнее значение на стек (для возврата результата)
    if let Some(last_name) = names.last() {
        VariableResolver::resolve_and_load(ctx, last_name, line)?;
    }
    
    Ok(())
}

