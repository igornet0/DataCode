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
                if pop_value {
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
