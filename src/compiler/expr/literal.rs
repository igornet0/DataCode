/// Компиляция литералов

use crate::parser::ast::Expr;
use crate::bytecode::OpCode;
use crate::common::error::LangError;
use crate::compiler::context::CompilationContext;

pub fn compile_literal(ctx: &mut CompilationContext, expr: &Expr) -> Result<(), LangError> {
    if let Expr::Literal { value, line } = expr {
        *ctx.current_line = *line;
        let constant_index = ctx.chunk.add_constant(value.clone());
        ctx.chunk.write_with_line(OpCode::Constant(constant_index), *line);
        Ok(())
    } else {
        Err(LangError::ParseError {
            message: "Expected Literal expression".to_string(),
            line: expr.line(),
        })
    }
}

