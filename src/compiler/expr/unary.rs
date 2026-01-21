/// Компиляция унарных операторов

use crate::parser::ast::Expr;
use crate::bytecode::OpCode;
use crate::common::error::LangError;
use crate::lexer::TokenKind;
use crate::compiler::context::CompilationContext;
use crate::compiler::expr;

pub fn compile_unary(ctx: &mut CompilationContext, expr: &Expr) -> Result<(), LangError> {
    if let Expr::Unary { op, right, line } = expr {
        *ctx.current_line = *line;
        match op {
            TokenKind::Minus => {
                // Унарный минус: -value
                expr::compile_expr(ctx, right)?;
                ctx.chunk.write_with_line(OpCode::Negate, *line);
            }
            TokenKind::Bang => {
                // Унарный Bang: !value (логическое отрицание)
                expr::compile_expr(ctx, right)?;
                ctx.chunk.write_with_line(OpCode::Not, *line);
            }
            _ => {
                return Err(LangError::ParseError {
                    message: format!("Unknown unary operator: {:?}", op),
                    line: *line,
                });
            }
        }
        Ok(())
    } else {
        Err(LangError::ParseError {
            message: "Expected Unary expression".to_string(),
            line: expr.line(),
        })
    }
}

