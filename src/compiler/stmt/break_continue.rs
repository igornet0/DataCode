/// Компиляция break и continue statements

use crate::bytecode::OpCode;
use crate::parser::ast::Stmt;
use crate::common::error::LangError;
use crate::compiler::context::CompilationContext;

pub fn compile_break(ctx: &mut CompilationContext, stmt: &Stmt) -> Result<(), LangError> {
    if let Stmt::Break { line } = stmt {
        *ctx.current_line = *line;
        if ctx.loop_contexts.is_empty() {
            return Err(LangError::ParseError {
                message: "break statement outside of loop".to_string(),
                line: *line,
                file: None,
            });
        }
        let loop_ctx = ctx.loop_contexts.last().unwrap();
        // При break из for i in range(...) снять состояние цикла с for_range_stack
        if loop_ctx.is_for_range {
            ctx.chunk.write_with_line(OpCode::PopForRange, *line);
        }
        // Jump к метке конца цикла
        let break_label = loop_ctx.break_label;
        ctx.labels.emit_jump(ctx.chunk, *ctx.current_line, false, break_label)?;
        Ok(())
    } else {
        Err(LangError::ParseError {
            message: "Expected Break statement".to_string(),
            line: stmt.line(),
            file: None,
        })
    }
}

pub fn compile_continue(ctx: &mut CompilationContext, stmt: &Stmt) -> Result<(), LangError> {
    if let Stmt::Continue { line } = stmt {
        *ctx.current_line = *line;
        if ctx.loop_contexts.is_empty() {
            return Err(LangError::ParseError {
                message: "continue statement outside of loop".to_string(),
                line: *line,
                file: None,
            });
        }
        // Jump к метке continue
        let continue_label = ctx.loop_contexts.last().unwrap().continue_label;
        ctx.labels.emit_jump(ctx.chunk, *ctx.current_line, false, continue_label)?;
        Ok(())
    } else {
        Err(LangError::ParseError {
            message: "Expected Continue statement".to_string(),
            line: stmt.line(),
            file: None,
        })
    }
}

