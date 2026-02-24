/// Компиляция super выражений

use crate::parser::ast::{Expr, Arg};
use crate::bytecode::OpCode;
use crate::common::error::LangError;
use crate::compiler::context::CompilationContext;
use crate::compiler::expr;

/// Compile standalone `super` - this is an error (super must be used as super() or super.method())
pub fn compile_super(ctx: &mut CompilationContext, expr: &Expr) -> Result<(), LangError> {
    if let Expr::Super { line } = expr {
        *ctx.current_line = *line;
        Err(LangError::ParseError {
            message: "super cannot be used as a value. Use super(...) to call parent constructor or super.method() to call parent method".to_string(),
            line: *line,
        })
    } else {
        Err(LangError::ParseError {
            message: "Expected Super expression".to_string(),
            line: expr.line(),
        })
    }
}

/// Compile `super(args)` - call to parent constructor
/// This is normally handled by class.rs for the first statement in constructor.
/// If it appears elsewhere, it's an error.
pub fn compile_super_call(ctx: &mut CompilationContext, expr: &Expr) -> Result<(), LangError> {
    if let Expr::SuperCall { line, .. } = expr {
        *ctx.current_line = *line;
        
        // super() outside constructor context is an error
        if !ctx.in_constructor {
            return Err(LangError::ParseError {
                message: "super() can only be called from within a constructor".to_string(),
                line: *line,
            });
        }
        
        // If we reach here in normal expr compilation, it means super() was not the first statement
        // (class.rs handles the first statement specially)
        Err(LangError::ParseError {
            message: "super() must be the first statement in constructor".to_string(),
            line: *line,
        })
    } else {
        Err(LangError::ParseError {
            message: "Expected SuperCall expression".to_string(),
            line: expr.line(),
        })
    }
}

/// Compile `super.method(args)` - call to parent method
pub fn compile_super_method_call(ctx: &mut CompilationContext, expr: &Expr) -> Result<(), LangError> {
    if let Expr::SuperMethodCall { method, args, line } = expr {
        *ctx.current_line = *line;
        
        // Check that we're inside a class that extends another class
        let superclass_name = match &ctx.current_superclass {
            Some(name) => name.clone(),
            None => {
                return Err(LangError::ParseError {
                    message: "super can only be used in a class that extends another class".to_string(),
                    line: *line,
                });
            }
        };
        
        // Look up parent method: Parent::method_<method_name>
        let parent_method_name = format!("{}::method_{}", superclass_name, method);
        let method_global_index = match ctx.scope.globals.get(&parent_method_name) {
            Some(&idx) => idx,
            None => {
                return Err(LangError::ParseError {
                    message: format!("Method '{}' not found in parent class '{}'", method, superclass_name),
                    line: *line,
                });
            }
        };
        
        // Load this (first argument for method call)
        // In methods, this is at slot 0; in constructors, use constructor_this_slot
        if let Some(slot) = ctx.constructor_this_slot {
            ctx.chunk.write_with_line(OpCode::LoadLocal(slot), *line);
        } else if let Some(local_index) = ctx.scope.resolve_local("this") {
            ctx.chunk.write_with_line(OpCode::LoadLocal(local_index), *line);
        } else {
            ctx.chunk.write_with_line(OpCode::LoadLocal(0), *line);
        }
        
        // Compile arguments
        for arg in args {
            match arg {
                Arg::Positional(arg_expr) => expr::compile_expr(ctx, arg_expr)?,
                Arg::Named { value, .. } => expr::compile_expr(ctx, value)?,
            }
        }
        
        // Load and call parent method
        ctx.chunk.global_names.insert(method_global_index, parent_method_name);
        ctx.chunk.write_with_line(OpCode::LoadGlobal(method_global_index), *line);
        ctx.chunk.write_with_line(OpCode::Call(args.len() + 1), *line); // +1 for this
        
        Ok(())
    } else {
        Err(LangError::ParseError {
            message: "Expected SuperMethodCall expression".to_string(),
            line: expr.line(),
        })
    }
}
