/// Компиляция this выражения

use crate::parser::ast::Expr;
use crate::bytecode::OpCode;
use crate::common::error::LangError;
use crate::compiler::context::CompilationContext;

pub fn compile_this(ctx: &mut CompilationContext, expr: &Expr) -> Result<(), LangError> {
    if let Expr::This { line } = expr {
        *ctx.current_line = *line;
        
        // this может быть:
        // 1. В конструкторе: слот constructor_this_slot (arity), если задан
        // 2. Локальной переменной (конструктор, fallback)
        // 3. Первым параметром метода (слот 0)
        if let Some(slot) = ctx.constructor_this_slot {
            ctx.chunk.write_with_line(OpCode::LoadLocal(slot), *line);
        } else if let Some(local_index) = ctx.scope.resolve_local("this") {
            ctx.chunk.write_with_line(OpCode::LoadLocal(local_index), *line);
        } else {
            ctx.chunk.write_with_line(OpCode::LoadLocal(0), *line);
        }
        
        Ok(())
    } else {
        Err(LangError::ParseError {
            message: "Expected This expression".to_string(),
            line: expr.line(),
            file: None,
        })
    }
}
