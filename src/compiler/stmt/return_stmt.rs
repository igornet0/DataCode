/// Компиляция return statements

use crate::parser::ast::Stmt;
use crate::bytecode::OpCode;
use crate::common::error::LangError;
use crate::common::value::Value;
use crate::compiler::context::CompilationContext;
use crate::compiler::expr;

pub fn compile_return(ctx: &mut CompilationContext, stmt: &Stmt) -> Result<(), LangError> {
    if let Stmt::Return { value, line } = stmt {
        *ctx.current_line = *line;
        if let Some(expr) = value {
            expr::compile_expr(ctx, expr)?;
        } else {
            let const_index = ctx.chunk.add_constant(Value::Null);
            ctx.chunk.write_with_line(OpCode::Constant(const_index), *line);
        }
        ctx.chunk.write_with_line(OpCode::Return, *line);
        Ok(())
    } else {
        Err(LangError::ParseError {
            message: "Expected Return statement".to_string(),
            line: stmt.line(),
        })
    }
}

