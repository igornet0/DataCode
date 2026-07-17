use crate::bytecode::OpCode;
use crate::common::error::LangError;
use crate::common::value::Value;
use crate::compiler::context::CompilationContext;
use crate::compiler::expr;
use crate::lexer::TokenKind;
/// Компиляция бинарных операторов
use crate::parser::ast::{BinaryOpKind, Expr};

pub fn compile_binary(ctx: &mut CompilationContext, expr: &Expr) -> Result<(), LangError> {
    if let Expr::Binary {
        left,
        op,
        right,
        line,
    } = expr
    {
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
                ctx.labels
                    .emit_jump(ctx.chunk, *line, true, skip_right_label)?;
                ctx.labels.emit_jump(ctx.chunk, *line, false, end_label)?;
                ctx.labels
                    .mark_label(skip_right_label, ctx.chunk.code.len());
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
                if crate::compiler::expr::integral_peephole::try_compile_f_score_stale_check(
                    ctx, left, right, *line,
                )? {
                    return Ok(());
                }
                expr::compile_expr(ctx, left)?;
                expr::compile_expr(ctx, right)?;
                ctx.chunk.write_with_line(OpCode::NotEqual, *line);
                Ok(())
            }
            BinaryOpKind::Builtin(TokenKind::Less) => {
                if crate::compiler::expr::integral_peephole::try_compile_dict_get_integral_lt(
                    ctx, left, right, *line,
                )? {
                    return Ok(());
                }
                expr::compile_expr(ctx, left)?;
                expr::compile_expr(ctx, right)?;
                ctx.chunk.write_with_line(OpCode::Less, *line);
                Ok(())
            }
            BinaryOpKind::Builtin(TokenKind::Plus) => {
                if crate::compiler::expr::integral_peephole::try_compile_dict_index_integral_add(
                    ctx, left, right, *line,
                )? {
                    return Ok(());
                }
                expr::compile_expr(ctx, left)?;
                expr::compile_expr(ctx, right)?;
                ctx.chunk.write_with_line(OpCode::Add, *line);
                Ok(())
            }
            BinaryOpKind::Builtin(tok) => {
                expr::compile_expr(ctx, left)?;
                expr::compile_expr(ctx, right)?;
                match tok {
                    TokenKind::Plus => unreachable!("Plus handled above"),
                    TokenKind::Minus => ctx.chunk.write_with_line(OpCode::Sub, *line),
                    TokenKind::Star => ctx.chunk.write_with_line(OpCode::Mul, *line),
                    TokenKind::StarStar => ctx.chunk.write_with_line(OpCode::Pow, *line),
                    TokenKind::Slash => ctx.chunk.write_with_line(OpCode::Div, *line),
                    TokenKind::SlashSlash => ctx.chunk.write_with_line(OpCode::IntDiv, *line),
                    TokenKind::Percent => ctx.chunk.write_with_line(OpCode::Mod, *line),
                    TokenKind::Greater => ctx.chunk.write_with_line(OpCode::Greater, *line),
                    TokenKind::Less => unreachable!("Less handled above"),
                    TokenKind::GreaterEqual => {
                        ctx.chunk.write_with_line(OpCode::GreaterEqual, *line)
                    }
                    TokenKind::LessEqual => ctx.chunk.write_with_line(OpCode::LessEqual, *line),
                    TokenKind::In => {
                        if crate::compiler::expr::integral_peephole::expr_may_be_integral(left) {
                            ctx.chunk.write_with_line(OpCode::InIntegral, *line);
                        } else {
                            ctx.chunk.write_with_line(OpCode::In, *line);
                        }
                    }
                    TokenKind::Amp => ctx.chunk.write_with_line(OpCode::BitAnd, *line),
                    TokenKind::Pipe => ctx.chunk.write_with_line(OpCode::BitOr, *line),
                    TokenKind::Caret => ctx.chunk.write_with_line(OpCode::BitXor, *line),
                    TokenKind::LessLess => ctx.chunk.write_with_line(OpCode::ShiftLeft, *line),
                    TokenKind::GreaterGreater => {
                        ctx.chunk.write_with_line(OpCode::ShiftRight, *line)
                    }
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
