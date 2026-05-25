pub mod array;
pub mod assign;
pub mod binary;
pub mod call;
pub mod call_value;
pub mod dict_comp;
pub mod if_expr;
pub mod list_comp;
/// Модуль компиляции выражений
pub mod interpolated;
pub mod lambda;
pub mod literal;
pub mod method_call;
pub mod property;
pub mod super_expr;
pub mod this;
pub mod unary;
pub mod variable;

use crate::bytecode::OpCode;
use crate::common::error::LangError;
use crate::common::value::Value;
use crate::compiler::context::CompilationContext;
use crate::parser::ast::Expr;

/// Трейт для компиляции выражений
pub trait ExprCompiler {
    fn compile_expr(&mut self, ctx: &mut CompilationContext, expr: &Expr) -> Result<(), LangError>;
}

/// Диспетчеризация компиляции выражений
pub fn compile_expr(ctx: &mut CompilationContext, expr: &Expr) -> Result<(), LangError> {
    match expr {
        Expr::Literal { .. } => literal::compile_literal(ctx, expr),
        Expr::Ellipsis { line } => {
            *ctx.current_line = *line;
            let constant_index = ctx.chunk.add_constant(Value::Ellipsis);
            ctx.chunk
                .write_with_line(OpCode::Constant(constant_index), *line);
            Ok(())
        }
        Expr::Variable { .. } => variable::compile_variable(ctx, expr),
        Expr::Assign { .. }
        | Expr::AssignOp { .. }
        | Expr::AssignArray { .. }
        | Expr::AssignArrayOp { .. }
        | Expr::UnpackAssign { .. } => assign::compile_assign(ctx, expr),
        Expr::Unary { .. } => unary::compile_unary(ctx, expr),
        Expr::Binary { .. } => binary::compile_binary(ctx, expr),
        Expr::Call { .. } => call::compile_call(ctx, expr),
        Expr::CallValue { .. } => call_value::compile_call_value(ctx, expr),
        Expr::Lambda { .. } => lambda::compile_lambda(ctx, expr),
        Expr::DictComprehension {
            key_expr,
            value_expr,
            loop_var,
            iterable,
            condition,
            line,
        } => dict_comp::compile_dict_comprehension(
            ctx,
            key_expr,
            value_expr,
            loop_var,
            iterable,
            condition.as_deref(),
            *line,
        ),
        Expr::ListComprehension {
            elt,
            clauses,
            line,
        } => list_comp::compile_list_comprehension(ctx, elt.as_ref(), clauses, *line),
        Expr::ArrayLiteral { .. }
        | Expr::TupleLiteral { .. }
        | Expr::ObjectLiteral { .. }
        | Expr::ArrayIndex { .. }
        | Expr::TableFilter { .. } => array::compile_array(ctx, expr),
        Expr::Property { .. } => property::compile_property(ctx, expr),
        Expr::MethodCall { .. } => method_call::compile_method_call(ctx, expr),
        Expr::This { .. } => this::compile_this(ctx, expr),
        Expr::Super { .. } => super_expr::compile_super(ctx, expr),
        Expr::SuperCall { .. } => super_expr::compile_super_call(ctx, expr),
        Expr::SuperMethodCall { .. } => super_expr::compile_super_method_call(ctx, expr),
        Expr::InterpolatedString { .. } => interpolated::compile_interpolated_string(ctx, expr),
        Expr::If { .. } => if_expr::compile_if_expr(ctx, expr),
        Expr::ExprReturn { line, .. } => Err(LangError::ParseError {
            message:
                "`return` as an expression is only compiled inside stream fn (e.g. x = return ...)"
                    .to_string(),
            line: *line,
            file: ctx.source_name.map(|s| s.to_string()),
        }),
        Expr::Ireturn { line, .. } => Err(LangError::ParseError {
            message: "`ireturn` is only compiled inside stream fn body".to_string(),
            line: *line,
            file: ctx.source_name.map(|s| s.to_string()),
        }),
    }
}
