/// Компиляция интерполированных строк "Hello ${name}!", "${n=}", "${n:.2f}", "${n=:.0f}"

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
            file: None,
        });
    };
    *ctx.current_line = *line;
    for (i, seg) in segments.iter().enumerate() {
        match seg {
            InterpolatedSegment::Literal(s) => {
                let idx = ctx.chunk.add_constant(Value::String(s.clone()));
                ctx.chunk.write_with_line(OpCode::Constant(idx), *line);
            }
            InterpolatedSegment::Expr {
                expr: e,
                include_name,
                display_name,
                format,
            } => {
                if *include_name {
                    let name = display_name.as_deref().unwrap_or("");
                    let name_idx = ctx.chunk.add_constant(Value::String(name.to_string()));
                    let eq_idx = ctx.chunk.add_constant(Value::String("=".to_string()));
                    ctx.chunk.write_with_line(OpCode::Constant(name_idx), *line);
                    ctx.chunk.write_with_line(OpCode::Constant(eq_idx), *line);
                    ctx.chunk.write_with_line(OpCode::Add, *line);
                }
                expr::compile_expr(ctx, e)?;
                if let Some(ref fmt) = format {
                    let fmt_idx = ctx.chunk.add_constant(Value::String(fmt.clone()));
                    ctx.chunk.write_with_line(OpCode::FormatInterp(fmt_idx), *line);
                }
                if *include_name {
                    ctx.chunk.write_with_line(OpCode::Add, *line);
                }
            }
        }
        if i > 0 {
            ctx.chunk.write_with_line(OpCode::Add, *line);
        }
    }
    Ok(())
}
