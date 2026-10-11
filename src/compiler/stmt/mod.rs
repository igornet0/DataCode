pub mod break_continue;
pub mod class;
pub mod for_stmt;
pub mod function;
pub mod if_stmt;
/// Модуль компиляции statements
pub mod import;
pub mod let_stmt;
pub mod return_stmt;
pub mod throw;
pub mod try_catch;
pub mod while_stmt;

use crate::common::error::LangError;
use crate::compiler::context::CompilationContext;
use crate::parser::ast::Stmt;

/// Диспетчеризация компиляции statements
pub fn compile_stmt(
    ctx: &mut CompilationContext,
    stmt: &Stmt,
    pop_value: bool,
) -> Result<(), LangError> {
    let stmt_line = stmt.line();
    *ctx.current_line = stmt_line;

    match stmt {
        Stmt::Import { .. } => import::compile_import(ctx, stmt),
        Stmt::Let { .. } => let_stmt::compile_let(ctx, stmt, pop_value),
        Stmt::Expr { .. } => {
            if let Stmt::Expr { expr, line } = stmt {
                crate::compiler::expr::compile_expr(ctx, expr)?;
                if pop_value && !drop_assignment_reload(ctx) {
                    ctx.chunk
                        .write_with_line(crate::bytecode::OpCode::Pop, *line);
                }
                Ok(())
            } else {
                unreachable!()
            }
        }
        Stmt::If { .. } => if_stmt::compile_if(ctx, stmt, pop_value),
        Stmt::While { .. } => while_stmt::compile_while(ctx, stmt),
        Stmt::For { .. } => for_stmt::compile_for(ctx, stmt),
        Stmt::Function { .. } => function::compile_function(ctx, stmt),
        Stmt::StreamFunction { .. } => function::compile_stream_function(ctx, stmt),
        Stmt::Return { .. } => return_stmt::compile_return(ctx, stmt),
        Stmt::EReturn { .. } => Err(LangError::ParseError {
            message: "'ereturn' may only appear inside a stream fn body".to_string(),
            line: stmt.line(),
            file: None,
        }),
        Stmt::Break { .. } => break_continue::compile_break(ctx, stmt),
        Stmt::Continue { .. } => break_continue::compile_continue(ctx, stmt),
        Stmt::Throw { .. } => throw::compile_throw(ctx, stmt),
        Stmt::Try { .. } => try_catch::compile_try_catch(ctx, stmt),
        Stmt::Class { .. } => class::compile_class(ctx, stmt, pop_value),
    }
}

/// Assignment statement peephole: `x = v` compiles to `Store x; Load x` so the assignment can be
/// used as an expression; as a statement the reloaded value is popped right away. Drop the
/// reload instead of emitting `Pop` (two fewer instructions per assignment). Jumps inside `v`
/// target at most the store, so removing the trailing load keeps every offset valid.
/// `StoreLocal` then also does the dead-slot truncation the dropped `Pop` did at the frame
/// boundary (some call paths still read the stack top as `stack.len()`).
fn drop_assignment_reload(ctx: &mut CompilationContext) -> bool {
    use crate::bytecode::OpCode;
    let code = &ctx.chunk.code;
    let n = code.len();
    if n < 2 {
        return false;
    }
    let redundant = match (&code[n - 2], &code[n - 1]) {
        (OpCode::StoreLocal(a), OpCode::LoadLocal(b)) => a == b,
        _ => false,
    };
    if redundant {
        ctx.chunk.code.pop();
        ctx.chunk.lines.pop();
    }
    redundant
}
