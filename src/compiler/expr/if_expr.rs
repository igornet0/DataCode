use crate::common::error::LangError;
use crate::compiler::context::CompilationContext;
use crate::compiler::expr;
use crate::compiler::stmt;
use crate::parser::ast::{Expr, IfBranch};

pub fn compile_if_expr(ctx: &mut CompilationContext, expr: &Expr) -> Result<(), LangError> {
    if let Expr::If {
        condition,
        then_branch,
        else_branch,
        line,
    } = expr
    {
        *ctx.current_line = *line;
        expr::compile_expr(ctx, condition)?;

        let else_label = ctx.labels.create_label();
        let end_label = ctx.labels.create_label();

        ctx.labels
            .emit_jump(ctx.chunk, *ctx.current_line, true, else_label)?;

        compile_if_branch(ctx, then_branch)?;

        ctx.labels
            .emit_jump(ctx.chunk, *ctx.current_line, false, end_label)?;

        ctx.labels.mark_label(else_label, ctx.chunk.code.len());
        compile_if_branch(ctx, else_branch)?;

        ctx.labels.mark_label(end_label, ctx.chunk.code.len());
        Ok(())
    } else {
        Err(LangError::ParseError {
            message: "Expected If expression".to_string(),
            line: expr.line(),
            file: ctx.source_name.map(|s| s.to_string()),
        })
    }
}

fn compile_if_branch(ctx: &mut CompilationContext, branch: &IfBranch) -> Result<(), LangError> {
    match branch {
        IfBranch::Expr(e) => expr::compile_expr(ctx, e),
        IfBranch::Block(stmts) => {
            for (i, stmt) in stmts.iter().enumerate() {
                let is_last = i == stmts.len() - 1;
                stmt::compile_stmt(ctx, stmt, !is_last)?;
            }
            Ok(())
        }
    }
}
