/// Компиляция if statements

use crate::parser::ast::Stmt;
use crate::common::error::LangError;
use crate::compiler::context::CompilationContext;
use crate::compiler::expr;
use crate::compiler::stmt;

pub fn compile_if(ctx: &mut CompilationContext, stmt: &Stmt, pop_value: bool) -> Result<(), LangError> {
    if let Stmt::If { condition, then_branch, else_branch, line } = stmt {
        *ctx.current_line = *line;
        expr::compile_expr(ctx, condition)?;
        
        // Создаем метки для else и end
        let else_label = ctx.labels.create_label();
        let end_label = ctx.labels.create_label();
        
        // Jump if false к else (или end, если else нет)
        let target_label = if else_branch.is_some() { else_label } else { end_label };
        ctx.labels.emit_jump(ctx.chunk, *ctx.current_line, true, target_label)?;
        
        // Компилируем then ветку (с новой областью видимости)
        ctx.scope.begin_scope();
        for (i, stmt) in then_branch.iter().enumerate() {
            let is_last = i == then_branch.len() - 1;
            stmt::compile_stmt(ctx, stmt, !is_last || pop_value)?;
        }
        ctx.scope.end_scope();
        
        // Jump к end после then
        ctx.labels.emit_jump(ctx.chunk, *ctx.current_line, false, end_label)?;
        
        // Помечаем метку else (если есть)
        if else_branch.is_some() {
            ctx.labels.mark_label(else_label, ctx.chunk.code.len());
            
            // Компилируем else ветку (с новой областью видимости)
            ctx.scope.begin_scope();
            for (i, stmt) in else_branch.as_ref().unwrap().iter().enumerate() {
                let is_last = i == else_branch.as_ref().unwrap().len() - 1;
                stmt::compile_stmt(ctx, stmt, !is_last || pop_value)?;
            }
            ctx.scope.end_scope();
        }
        
        // Помечаем метку end
        ctx.labels.mark_label(end_label, ctx.chunk.code.len());
        Ok(())
    } else {
        Err(LangError::ParseError {
            message: "Expected If statement".to_string(),
            line: stmt.line(),
        })
    }
}

