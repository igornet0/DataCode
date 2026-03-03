/// Компиляция переменных

use crate::parser::ast::Expr;
use crate::bytecode::OpCode;
use crate::common::error::LangError;
use crate::compiler::context::CompilationContext;

/// Sentinel index for undefined variables. Always >= globals.len() at runtime, so LoadGlobal
/// will report "Undefined variable" instead of loading a wrong slot (e.g. a built-in module).
const UNDEFINED_GLOBAL_SENTINEL: usize = usize::MAX;

pub fn compile_variable(ctx: &mut CompilationContext, expr: &Expr) -> Result<(), LangError> {
    if let Expr::Variable { name, line } = expr {
        *ctx.current_line = *line;
        
        if let Some(local_index) = ctx.scope.resolve_local(name) {
            ctx.chunk.write_with_line(OpCode::LoadLocal(local_index), *line);
        } else if let Some(&global_index) = ctx.scope.globals.get(name) {
            ctx.chunk.global_names.insert(global_index, name.clone());
            ctx.chunk.write_with_line(OpCode::LoadGlobal(global_index), *line);
        } else {
            // Неизвестная переменная — откладываем до runtime (VM выбросит, try/catch перехватит)
            // Используем sentinel, чтобы LoadGlobal всегда вызывал ошибку (index >= globals.len())
            // Имя в chunk.global_names нужно для update_chunk_indices_from_names при merge модулей.
            ctx.scope.globals.insert(name.clone(), UNDEFINED_GLOBAL_SENTINEL);
            ctx.chunk.global_names.insert(UNDEFINED_GLOBAL_SENTINEL, name.clone());
            ctx.chunk.write_with_line(OpCode::LoadGlobal(UNDEFINED_GLOBAL_SENTINEL), *line);
        }
        Ok(())
    } else {
        Err(LangError::ParseError {
            message: "Expected Variable expression".to_string(),
            line: expr.line(),
            file: None,
        })
    }
}

