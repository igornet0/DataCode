/// Вызов выражения-калли: `(fn(x) => x)(1)`
use crate::bytecode::OpCode;
use crate::common::error::LangError;
use crate::compiler::context::CompilationContext;
use crate::compiler::expr;
use crate::parser::ast::{Arg, Expr};

pub fn compile_call_value(ctx: &mut CompilationContext, expr: &Expr) -> Result<(), LangError> {
    if let Expr::CallValue { callee, args, line } = expr {
        *ctx.current_line = *line;
        for arg in args {
            match arg {
                Arg::Positional(e) => expr::compile_expr(ctx, e)?,
                Arg::Named { value, .. } => expr::compile_expr(ctx, value)?,
                Arg::UnpackObject(e) | Arg::UnpackArray(e) => expr::compile_expr(ctx, e)?,
            }
        }
        expr::compile_expr(ctx, callee.as_ref())?;
        let arity = args.len();
        ctx.chunk.write_with_line(OpCode::Call(arity), *line);
        Ok(())
    } else {
        Err(LangError::ParseError {
            message: "Expected CallValue expression".to_string(),
            line: expr.line(),
            file: None,
        })
    }
}
