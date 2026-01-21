/// Модуль компиляции выражений

pub mod literal;
pub mod variable;
pub mod assign;
pub mod unary;
pub mod binary;
pub mod call;
pub mod array;
pub mod property;
pub mod method_call;

use crate::parser::ast::Expr;
use crate::common::error::LangError;
use crate::compiler::context::CompilationContext;

/// Трейт для компиляции выражений
pub trait ExprCompiler {
    fn compile_expr(&mut self, ctx: &mut CompilationContext, expr: &Expr) -> Result<(), LangError>;
}

/// Диспетчеризация компиляции выражений
pub fn compile_expr(
    ctx: &mut CompilationContext,
    expr: &Expr,
) -> Result<(), LangError> {
    match expr {
        Expr::Literal { .. } => literal::compile_literal(ctx, expr),
        Expr::Variable { .. } => variable::compile_variable(ctx, expr),
        Expr::Assign { .. } | Expr::AssignOp { .. } | Expr::UnpackAssign { .. } => {
            assign::compile_assign(ctx, expr)
        }
        Expr::Unary { .. } => unary::compile_unary(ctx, expr),
        Expr::Binary { .. } => binary::compile_binary(ctx, expr),
        Expr::Call { .. } => call::compile_call(ctx, expr),
        Expr::ArrayLiteral { .. } | Expr::TupleLiteral { .. } | Expr::ObjectLiteral { .. } | Expr::ArrayIndex { .. } => {
            array::compile_array(ctx, expr)
        }
        Expr::Property { .. } => property::compile_property(ctx, expr),
        Expr::MethodCall { .. } => method_call::compile_method_call(ctx, expr),
    }
}

