/// Компиляция this выражения

use crate::parser::ast::Expr;
use crate::bytecode::OpCode;
use crate::common::error::LangError;
use crate::compiler::context::CompilationContext;

pub fn compile_this(ctx: &mut CompilationContext, expr: &Expr) -> Result<(), LangError> {
    if let Expr::This { line } = expr {
        *ctx.current_line = *line;
        
        // this может быть:
        // 1. Первым параметром метода (слот 0)
        // 2. Локальной переменной в конструкторе (созданной и сохраненной)
        // Проверяем, есть ли this как локальная переменная
        if let Some(local_index) = ctx.scope.resolve_local("this") {
            // this найден как локальная переменная (конструктор)
            ctx.chunk.write_with_line(OpCode::LoadLocal(local_index), *line);
        } else {
            // this не найден как локальная переменная, значит это первый параметр (метод)
            // В методах this всегда в слоте 0
            ctx.chunk.write_with_line(OpCode::LoadLocal(0), *line);
        }
        
        Ok(())
    } else {
        Err(LangError::ParseError {
            message: "Expected This expression".to_string(),
            line: expr.line(),
        })
    }
}
