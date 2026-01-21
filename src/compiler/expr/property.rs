/// Компиляция доступа к свойствам объектов

use crate::parser::ast::Expr;
use crate::bytecode::OpCode;
use crate::common::value::Value;
use crate::common::error::LangError;
use crate::compiler::context::CompilationContext;
use crate::compiler::expr;

pub fn compile_property(ctx: &mut CompilationContext, expr: &Expr) -> Result<(), LangError> {
    if let Expr::Property { object, name, line } = expr {
        *ctx.current_line = *line;
        // Компилируем объект
        expr::compile_expr(ctx, object)?;
        // Для table.idx мы просто оставляем таблицу на стеке
        // Затем при индексации [i] это будет обработано как table[i]
        if name != "idx" {
            // Для других свойств создаем строку и используем индексацию
            let name_index = ctx.chunk.add_constant(Value::String(name.clone()));
            ctx.chunk.write_with_line(OpCode::Constant(name_index), *line);
            ctx.chunk.write_with_line(OpCode::GetArrayElement, *line);
        }
        // Для "idx" просто оставляем объект на стеке
        Ok(())
    } else {
        Err(LangError::ParseError {
            message: "Expected Property expression".to_string(),
            line: expr.line(),
        })
    }
}

