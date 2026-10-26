use crate::bytecode::OpCode;
use crate::common::error::LangError;
use crate::compiler::context::CompilationContext;
/// Компиляция переменных
use crate::parser::ast::Expr;

pub fn compile_variable(ctx: &mut CompilationContext, expr: &Expr) -> Result<(), LangError> {
    if let Expr::Variable { name, line } = expr {
        *ctx.current_line = *line;

        if let Some(local_index) = ctx.scope.resolve_local(name) {
            ctx.chunk
                .write_with_line(OpCode::LoadLocal(local_index), *line);
        } else if let Some(&global_index) = ctx.scope.globals.get(name) {
            ctx.chunk.global_names.insert(global_index, name.clone());
            ctx.chunk
                .write_with_line(OpCode::LoadGlobal(global_index), *line);
        } else {
            // Неизвестная переменная — откладываем до runtime (VM разрешит по имени или выбросит
            // «Undefined variable», try/catch перехватит). Своя заглушка на каждое имя: имя хранится
            // в chunk.global_names (нужно и для update_chunk_indices_from_names при merge модулей).
            // В scope.globals заглушку не кладём: присваивание ниже (`B = 5` после функции, которая
            // читает B) должно получить настоящий индекс, а не StoreGlobal(заглушка).
            let sentinel = undefined_global_sentinel_for(ctx, name);
            ctx.chunk.global_names.insert(sentinel, name.clone());
            ctx.chunk
                .write_with_line(OpCode::LoadGlobal(sentinel), *line);
        }
        Ok(())
    } else {
        Err(LangError::ParseError {
            message: "Expected Variable expression".to_string(),
            line: expr.line(),
            file: None,
        })
    }
}

/// Placeholder index for an undefined name in the current chunk: reuse the one already assigned
/// to this name, otherwise take the next free slot counting down from `usize::MAX`.
fn undefined_global_sentinel_for(ctx: &CompilationContext, name: &str) -> usize {
    let mut lowest = usize::MAX;
    let mut any = false;
    for (&idx, n) in ctx.chunk.global_names.iter().rev() {
        if !crate::bytecode::is_undefined_global_sentinel(idx) {
            break;
        }
        if n == name {
            return idx;
        }
        lowest = idx;
        any = true;
    }
    if any {
        lowest - 1
    } else {
        usize::MAX
    }
}
