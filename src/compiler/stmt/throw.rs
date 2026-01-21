/// Компиляция throw statements

use crate::parser::ast::Stmt;
use crate::bytecode::OpCode;
use crate::common::error::LangError;
use crate::compiler::context::CompilationContext;
use crate::compiler::expr;

pub fn compile_throw(ctx: &mut CompilationContext, stmt: &Stmt) -> Result<(), LangError> {
    if let Stmt::Throw { value, line } = stmt {
        *ctx.current_line = *line;
        // Компилируем выражение (оно оставит значение на стеке)
        expr::compile_expr(ctx, value)?;
        // Генерируем Throw опкод (None означает RuntimeError без конкретного типа)
        ctx.chunk.write_with_line(OpCode::Throw(None), *line);
        Ok(())
    } else {
        Err(LangError::ParseError {
            message: "Expected Throw statement".to_string(),
            line: stmt.line(),
        })
    }
}

