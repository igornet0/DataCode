/// Компиляция переменных

use crate::parser::ast::Expr;
use crate::bytecode::OpCode;
use crate::common::error::LangError;
use crate::compiler::context::CompilationContext;

pub fn compile_variable(ctx: &mut CompilationContext, expr: &Expr) -> Result<(), LangError> {
    if let Expr::Variable { name, line } = expr {
        *ctx.current_line = *line;
        
        // Определяем, глобальная или локальная переменная
        if let Some(local_index) = ctx.scope.resolve_local(name) {
            // Локальная переменная
            ctx.chunk.write_with_line(OpCode::LoadLocal(local_index), *line);
        } else if let Some(&global_index) = ctx.scope.globals.get(name) {
            // Глобальная переменная или функция
            ctx.chunk.write_with_line(OpCode::LoadGlobal(global_index), *line);
        } else {
            // Переменная не найдена - создаем новый глобальный индекс
            // Это позволит проверить переменную во время выполнения
            let global_index = ctx.scope.globals.len();
            ctx.scope.globals.insert(name.clone(), global_index);
            ctx.chunk.global_names.insert(global_index, name.clone());
            ctx.chunk.write_with_line(OpCode::LoadGlobal(global_index), *line);
        }
        Ok(())
    } else {
        Err(LangError::ParseError {
            message: "Expected Variable expression".to_string(),
            line: expr.line(),
        })
    }
}

