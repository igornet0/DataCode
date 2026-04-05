/// Компиляция бинарных операторов

use crate::parser::ast::{BinaryOpKind, Expr};
use crate::bytecode::OpCode;
use crate::common::error::LangError;
use crate::common::value::Value;
use crate::lexer::TokenKind;
use crate::compiler::context::CompilationContext;
use crate::compiler::expr;

pub fn compile_binary(ctx: &mut CompilationContext, expr: &Expr) -> Result<(), LangError> {
    if let Expr::Binary { left, op, right, line } = expr {
        *ctx.current_line = *line;

        match op {
            BinaryOpKind::Plugin { name, .. } => {
                expr::compile_expr(ctx, left)?;
                expr::compile_expr(ctx, right)?;
                let idx = ctx.chunk.add_constant(Value::String(name.clone()));
                ctx.chunk.write_with_line(OpCode::BinaryOp(idx), *line);
                Ok(())
            }
            BinaryOpKind::Builtin(TokenKind::Or) => {
                expr::compile_expr(ctx, left)?;
                ctx.chunk.write_with_line(OpCode::Dup, *line);
                let skip_right_label = ctx.labels.create_label();
                let end_label = ctx.labels.create_label();
                ctx.labels.emit_jump(ctx.chunk, *line, true, skip_right_label)?;
                ctx.labels.emit_jump(ctx.chunk, *line, false, end_label)?;
                ctx.labels.mark_label(skip_right_label, ctx.chunk.code.len());
                ctx.chunk.write_with_line(OpCode::Pop, *line);
                expr::compile_expr(ctx, right)?;
                ctx.labels.mark_label(end_label, ctx.chunk.code.len());
                Ok(())
            }
            BinaryOpKind::Builtin(TokenKind::And) => {
                expr::compile_expr(ctx, left)?;
                ctx.chunk.write_with_line(OpCode::Dup, *line);
                let end_label = ctx.labels.create_label();
                ctx.labels.emit_jump(ctx.chunk, *line, true, end_label)?;
                ctx.chunk.write_with_line(OpCode::Pop, *line);
                expr::compile_expr(ctx, right)?;
                ctx.labels.mark_label(end_label, ctx.chunk.code.len());
                Ok(())
            }
            BinaryOpKind::Builtin(TokenKind::EqualEqual) => {
                expr::compile_expr(ctx, left)?;
                expr::compile_expr(ctx, right)?;
                ctx.chunk.write_with_line(OpCode::Equal, *line);
                Ok(())
            }
            BinaryOpKind::Builtin(TokenKind::BangEqual) => {
                expr::compile_expr(ctx, left)?;
                expr::compile_expr(ctx, right)?;
                ctx.chunk.write_with_line(OpCode::NotEqual, *line);
                Ok(())
            }
            BinaryOpKind::Builtin(tok) => {
                expr::compile_expr(ctx, left)?;
                expr::compile_expr(ctx, right)?;
                match tok {
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
                    _ => {
                        return Err(LangError::ParseError {
                            message: format!("Unknown binary operator: {:?}", tok),
                            line: *line,
                            file: None,
                        });
                    }
                }
                Ok(())
            }
        }
    } else {
        Err(LangError::ParseError {
            message: "Expected Binary expression".to_string(),
            line: expr.line(),
            file: None,
        })
    }
}
