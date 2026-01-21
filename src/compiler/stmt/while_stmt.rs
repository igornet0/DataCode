/// Компиляция while statements

use crate::parser::ast::Stmt;
use crate::common::error::LangError;
use crate::compiler::context::{CompilationContext, LoopContext};
use crate::compiler::expr;
use crate::compiler::stmt;

pub fn compile_while(ctx: &mut CompilationContext, stmt: &Stmt) -> Result<(), LangError> {
    if let Stmt::While { condition, body, line } = stmt {
        *ctx.current_line = *line;
        
        // Создаем метки для начала и конца цикла
        let loop_start_label = ctx.labels.create_label();
        let loop_end_label = ctx.labels.create_label();
        
        // Помечаем начало цикла
        ctx.labels.mark_label(loop_start_label, ctx.chunk.code.len());
        
        expr::compile_expr(ctx, condition)?;
        // Jump if false к концу цикла
        ctx.labels.emit_jump(ctx.chunk, *ctx.current_line, true, loop_end_label)?;
        
        // Создаем контекст цикла
        let loop_context = LoopContext {
            continue_label: loop_start_label, // continue возвращается к началу цикла
            break_label: loop_end_label,      // break переходит к концу цикла
        };
        ctx.loop_contexts.push(loop_context);
        
        // Компилируем тело цикла (с новой областью видимости)
        ctx.scope.begin_scope();
        for stmt in body {
            stmt::compile_stmt(ctx, stmt, true)?;
        }
        ctx.scope.end_scope();
        
        // Jump к началу цикла
        ctx.labels.emit_loop(ctx.chunk, *ctx.current_line, loop_start_label)?;
        
        // Помечаем конец цикла
        ctx.labels.mark_label(loop_end_label, ctx.chunk.code.len());
        
        // Удаляем контекст цикла
        ctx.loop_contexts.pop();
        
        Ok(())
    } else {
        Err(LangError::ParseError {
            message: "Expected While statement".to_string(),
            line: stmt.line(),
        })
    }
}

