use crate::bytecode::OpCode;
use crate::common::error::LangError;
use crate::compiler::context::CompilationContext;
/// Компиляция this выражения
use crate::parser::ast::Expr;

pub fn compile_this(ctx: &mut CompilationContext, expr: &Expr) -> Result<(), LangError> {
    if let Expr::This { line } = expr {
        *ctx.current_line = *line;

        // this может быть:
        // 1. В конструкторе: слот constructor_this_slot (arity), только пока компилируем тело ctor
        // 2. Локальной переменной метода или fallback в ctor без слота (resolve_local("this"))
        // 3. Первым параметром метода (обычно слот 0)
        match ctx.this_local_slot_for_member_access() {
            Some(slot) => ctx.chunk.write_with_line(OpCode::LoadLocal(slot), *line),
            None => ctx.chunk.write_with_line(OpCode::LoadLocal(0), *line),
        }

        Ok(())
    } else {
        Err(LangError::ParseError {
            message: "Expected This expression".to_string(),
            line: expr.line(),
            file: None,
        })
    }
}
