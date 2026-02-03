/// Компиляция интерполированных строк "Hello ${name}!"

use crate::parser::ast::{Expr, InterpolatedSegment};
use crate::bytecode::OpCode;
use crate::common::error::LangError;
use crate::common::value::Value;
use crate::compiler::context::CompilationContext;
use crate::compiler::expr;

pub fn compile_interpolated_string(
    ctx: &mut CompilationContext,
    expr: &Expr,
) -> Result<(), LangError> {
    let Expr::InterpolatedString { segments, line } = expr else {
        return Err(LangError::ParseError {
            message: "Expected InterpolatedString expression".to_string(),
            line: expr.line(),
        });
    };
    *ctx.current_line = *line;
    for (i, seg) in segments.iter().enumerate() {
        match seg {
            InterpolatedSegment::Literal(s) => {
                let idx = ctx.chunk.add_constant(Value::String(s.clone()));
                ctx.chunk.write_with_line(OpCode::Constant(idx), *line);
            }
            InterpolatedSegment::Expr(e) => {
                expr::compile_expr(ctx, e)?;
            }
        }
        if i > 0 {
            ctx.chunk.write_with_line(OpCode::Add, *line);
        }
    }
    Ok(())
}
