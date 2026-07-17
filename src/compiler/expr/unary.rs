use crate::bytecode::OpCode;
use crate::common::error::LangError;
use crate::compiler::context::CompilationContext;
use crate::compiler::expr;
use crate::lexer::TokenKind;
/// Компиляция унарных операторов
use crate::parser::ast::Expr;

pub fn compile_unary(ctx: &mut CompilationContext, expr: &Expr) -> Result<(), LangError> {
    if let Expr::Unary { op, right, line } = expr {
        *ctx.current_line = *line;
        match op {
            TokenKind::Minus => {
                // Унарный минус: -value
                expr::compile_expr(ctx, right)?;
                ctx.chunk.write_with_line(OpCode::Negate, *line);
            }
            TokenKind::Tilde => {
                expr::compile_expr(ctx, right)?;
                ctx.chunk.write_with_line(OpCode::BitNot, *line);
            }
            TokenKind::Bang => {
                if crate::compiler::expr::integral_peephole::try_compile_negated_grid_bounds(
                    ctx, right, *line,
                )? {
                    return Ok(());
                }
                if crate::compiler::expr::integral_peephole::try_compile_not_in_integral(
                    ctx, right, *line,
                )? {
                    return Ok(());
                }
                expr::compile_expr(ctx, right)?;
                ctx.chunk.write_with_line(OpCode::Not, *line);
            }
            _ => {
                return Err(LangError::ParseError {
                    message: format!("Unknown unary operator: {:?}", op),
                    line: *line,
                    file: None,
                });
            }
        }
        Ok(())
    } else {
        Err(LangError::ParseError {
            message: "Expected Unary expression".to_string(),
            line: expr.line(),
            file: None,
        })
    }
}
