/// Компиляция бинарных операторов

use crate::parser::ast::Expr;
use crate::bytecode::OpCode;
use crate::common::error::LangError;
use crate::lexer::TokenKind;
use crate::compiler::context::CompilationContext;
use crate::compiler::expr;

pub fn compile_binary(ctx: &mut CompilationContext, expr: &Expr) -> Result<(), LangError> {
    if let Expr::Binary { left, op, right, line } = expr {
        *ctx.current_line = *line;
        
        // Специальная обработка логических операторов
        if *op == TokenKind::EqualEqual {
            expr::compile_expr(ctx, left)?;
            expr::compile_expr(ctx, right)?;
            ctx.chunk.write_with_line(OpCode::Equal, *line);
        } else if *op == TokenKind::BangEqual {
            expr::compile_expr(ctx, left)?;
            expr::compile_expr(ctx, right)?;
            ctx.chunk.write_with_line(OpCode::NotEqual, *line);
        } else {
            expr::compile_expr(ctx, left)?;
            expr::compile_expr(ctx, right)?;
            match op {
                TokenKind::Plus => ctx.chunk.write_with_line(OpCode::Add, *line),
                TokenKind::Minus => ctx.chunk.write_with_line(OpCode::Sub, *line),
                TokenKind::Star => ctx.chunk.write_with_line(OpCode::Mul, *line),
                TokenKind::StarStar => ctx.chunk.write_with_line(OpCode::Pow, *line),
                TokenKind::Slash => ctx.chunk.write_with_line(OpCode::Div, *line),
                TokenKind::SlashSlash => ctx.chunk.write_with_line(OpCode::IntDiv, *line),
                TokenKind::Percent => ctx.chunk.write_with_line(OpCode::Mod, *line),
                TokenKind::Greater => ctx.chunk.write_with_line(OpCode::Greater, *line),
                TokenKind::Less => ctx.chunk.write_with_line(OpCode::Less, *line),
                TokenKind::GreaterEqual => ctx.chunk.write_with_line(OpCode::GreaterEqual, *line),
                TokenKind::LessEqual => ctx.chunk.write_with_line(OpCode::LessEqual, *line),
                TokenKind::In => ctx.chunk.write_with_line(OpCode::In, *line),
                TokenKind::Or => ctx.chunk.write_with_line(OpCode::Or, *line),
                TokenKind::And => ctx.chunk.write_with_line(OpCode::And, *line),
                _ => {
                    return Err(LangError::ParseError {
                        message: format!("Unknown binary operator: {:?}", op),
                        line: *line,
                    });
                }
            }
        }
        Ok(())
    } else {
        Err(LangError::ParseError {
            message: "Expected Binary expression".to_string(),
            line: expr.line(),
        })
    }
}

