/// Модуль компиляции statements

pub mod import;
pub mod let_stmt;
pub mod if_stmt;
pub mod while_stmt;
pub mod for_stmt;
pub mod function;
pub mod return_stmt;
pub mod break_continue;
pub mod throw;
pub mod try_catch;

use crate::parser::ast::Stmt;
use crate::common::error::LangError;
use crate::compiler::context::CompilationContext;

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
                    ctx.chunk.write_with_line(crate::bytecode::OpCode::Pop, *line);
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
        Stmt::Return { .. } => return_stmt::compile_return(ctx, stmt),
        Stmt::Break { .. } => break_continue::compile_break(ctx, stmt),
        Stmt::Continue { .. } => break_continue::compile_continue(ctx, stmt),
        Stmt::Throw { .. } => throw::compile_throw(ctx, stmt),
        Stmt::Try { .. } => try_catch::compile_try_catch(ctx, stmt),
    }
}

