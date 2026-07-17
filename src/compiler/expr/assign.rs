use crate::bytecode::OpCode;
use crate::common::error::LangError;
use crate::common::value::Value;
use crate::compiler::context::CompilationContext;
use crate::compiler::defaults;
use crate::compiler::expr;
use crate::compiler::expr::array::emit_slice_bound;
use crate::compiler::stmt::for_stmt;
use crate::compiler::variable::VariableResolver;
use crate::lexer::TokenKind;
/// Компиляция присваиваний (Assign, AssignOp, UnpackAssign)
use crate::parser::ast::{Arg, AssignTarget, Expr, IndexExpr};

fn load_property_path_root(
    ctx: &mut CompilationContext,
    root_name: &str,
    line: usize,
) -> Result<(), LangError> {
    if root_name == "this" {
        let slot = ctx.this_local_slot_with_fallback_for_member_access();
        ctx.chunk.write_with_line(OpCode::LoadLocal(slot), line);
    } else if let Some(local_index) = ctx.scope.resolve_local(root_name) {
        ctx.chunk
            .write_with_line(OpCode::LoadLocal(local_index), line);
    } else if let Some(&global_index) = ctx.scope.globals.get(root_name) {
        ctx.chunk
            .write_with_line(OpCode::LoadGlobal(global_index), line);
    } else {
        return Err(LangError::ParseError {
            message: format!("Variable '{}' not found", root_name),
            line,
            file: None,
        });
    }
    Ok(())
}

fn store_property_path_root(
    ctx: &mut CompilationContext,
    root_name: &str,
    line: usize,
) -> Result<(), LangError> {
    if root_name == "this" {
        let slot = ctx.this_local_slot_with_fallback_for_member_access();
        ctx.chunk.write_with_line(OpCode::StoreLocal(slot), line);
        ctx.chunk.write_with_line(OpCode::LoadLocal(slot), line);
    } else if let Some(local_index) = ctx.scope.resolve_local(root_name) {
        ctx.chunk
            .write_with_line(OpCode::StoreLocal(local_index), line);
        ctx.chunk
            .write_with_line(OpCode::LoadLocal(local_index), line);
    } else if let Some(&global_index) = ctx.scope.globals.get(root_name) {
        ctx.chunk
            .global_names
            .insert(global_index, root_name.to_string());
        ctx.chunk
            .write_with_line(OpCode::StoreGlobal(global_index), line);
        ctx.chunk
            .write_with_line(OpCode::LoadGlobal(global_index), line);
    }
    Ok(())
}

/// Traverse `root.intermediate...` and leave the container for the final field on the stack.
/// Stack before: arbitrary prefix; stack after: `[..., container]`.
fn emit_load_property_container(
    ctx: &mut CompilationContext,
    parts: &[&str],
    line: usize,
) -> Result<(), LangError> {
    load_property_path_root(ctx, parts[0], line)?;
    for intermediate in &parts[1..parts.len() - 1] {
        let idx = ctx
            .chunk
            .add_constant(Value::String((*intermediate).to_string()));
        ctx.chunk.write_with_line(OpCode::Constant(idx), line);
        ctx.chunk.write_with_line(OpCode::GetArrayElement, line);
    }
    Ok(())
}

/// Assign to `a.b.c = value` where `value` is already on the stack.
fn emit_property_path_store(
    ctx: &mut CompilationContext,
    path: &str,
    line: usize,
) -> Result<(), LangError> {
    let parts: Vec<&str> = path.split('.').collect();
    if parts.len() < 2 {
        return Err(LangError::ParseError {
            message: format!("Invalid property path: {}", path),
            line,
            file: None,
        });
    }
    for seg in &parts[1..] {
        if seg.starts_with('@') {
            return Err(LangError::ParseError {
                message: format!(
                    "Special methods cannot be accessed directly (got path `{}`)",
                    path
                ),
                line,
                file: ctx.source_name.map(|s| s.to_string()),
            });
        }
    }
    let field_name = parts[parts.len() - 1];
    let field_name_index = ctx
        .chunk
        .add_constant(Value::String(field_name.to_string()));
    ctx.chunk
        .write_with_line(OpCode::Constant(field_name_index), line);
    emit_load_property_container(ctx, &parts, line)?;
    ctx.chunk.write_with_line(OpCode::SetArrayElement, line);
    if parts.len() == 2 {
        store_property_path_root(ctx, parts[0], line)?;
    }
    Ok(())
}

/// Load the current value at a property path (for `+=`, etc.).
fn emit_load_property_path_value(
    ctx: &mut CompilationContext,
    path: &str,
    line: usize,
) -> Result<(), LangError> {
    let parts: Vec<&str> = path.split('.').collect();
    if parts.len() < 2 {
        return Err(LangError::ParseError {
            message: format!("Invalid property path: {}", path),
            line,
            file: None,
        });
    }
    for seg in &parts[1..] {
        if seg.starts_with('@') {
            return Err(LangError::ParseError {
                message: format!(
                    "Special methods cannot be accessed directly (got path `{}`)",
                    path
                ),
                line,
                file: ctx.source_name.map(|s| s.to_string()),
            });
        }
    }
    emit_load_property_container(ctx, &parts, line)?;
    let field_name_index = ctx
        .chunk
        .add_constant(Value::String(parts[parts.len() - 1].to_string()));
    ctx.chunk
        .write_with_line(OpCode::Constant(field_name_index), line);
    ctx.chunk.write_with_line(OpCode::GetArrayElement, line);
    Ok(())
}

pub fn compile_assign(ctx: &mut CompilationContext, expr: &Expr) -> Result<(), LangError> {
    match expr {
        Expr::Assign { name, value, line } => {
            *ctx.current_line = *line;
            // Best-effort tracking for `set()` locals to safely emit SetAddIntegral/SetDiscardIntegral.
            // This avoids miscompiling class methods named `add`.
            if !name.contains('.') {
                let is_set_call = matches!(
                    value.as_ref(),
                    Expr::Call { name: callee, args, .. } if callee == "set" && args.is_empty()
                );
                if is_set_call {
                    ctx.known_set_vars.insert(name.clone());
                } else {
                    ctx.known_set_vars.remove(name);
                }
                let is_ctor_like_call = matches!(
                    value.as_ref(),
                    Expr::Call { name: callee, .. }
                        if callee.chars().next().is_some_and(|ch| ch.is_uppercase())
                );
                if is_ctor_like_call {
                    ctx.known_class_instance_vars.insert(name.clone());
                } else {
                    ctx.known_class_instance_vars.remove(name);
                }
                if let Some(pairs) = for_stmt::try_const_tuple_of_pairs(value) {
                    ctx.known_const_pair_tuples.insert(name.clone(), pairs);
                } else {
                    ctx.known_const_pair_tuples.remove(name);
                }
                if ctx.current_function.is_none() {
                    defaults::update_compile_time_binding(ctx.compile_time_bindings, name, value);
                }
            }
            // Компилируем значение
            expr::compile_expr(ctx, value)?;
            // Не клонируем автоматически - переменные должны разделять ссылки на массивы/таблицы/объекты
            // Клонирование происходит только при явном вызове .clone()

            // Присваивание к свойству: obj.field = value, node.prev.next = value, ...
            if name.contains('.') {
                emit_property_path_store(ctx, name, *line)?;
                return Ok(());
            }

            // Обычное присваивание переменной
            // Используем VariableResolver для разрешения переменной
            // Проверяем, является ли переменная локальной или глобальной
            if let Some(local_index) = ctx.scope.resolve_local(name) {
                // Локальная переменная найдена - обновляем
                ctx.chunk
                    .write_with_line(OpCode::StoreLocal(local_index), *line);
                ctx.chunk
                    .write_with_line(OpCode::LoadLocal(local_index), *line);
                ctx.record_bound_name(name);
            } else if ctx.current_function.is_some()
                && VariableResolver::try_store_explicit_global(ctx, name, *line, true)
            {
                ctx.record_bound_name(name);
            } else if ctx.current_function.is_some() {
                // Внутри функции — локальная переменная затеняет глобал (в т.ч. одноимённую функцию)
                let index = ctx.declare_local_for_binding(name);
                ctx.chunk.write_with_line(OpCode::StoreLocal(index), *line);
                ctx.chunk.write_with_line(OpCode::LoadLocal(index), *line);
                ctx.record_bound_name(name);
            } else if let Some(&global_index) = ctx.scope.globals.get(name) {
                // На верхнем уровне — обновляем существующий глобал
                ctx.chunk.global_names.insert(global_index, name.clone());
                ctx.chunk
                    .write_with_line(OpCode::StoreGlobal(global_index), *line);
                ctx.chunk
                    .write_with_line(OpCode::LoadGlobal(global_index), *line);
                ctx.record_bound_name(name);
            } else {
                // Новая глобальная переменная на верхнем уровне
                let global_index = ctx.scope.globals.len();
                ctx.scope.globals.insert(name.clone(), global_index);
                ctx.chunk.global_names.insert(global_index, name.clone());
                ctx.chunk
                    .write_with_line(OpCode::StoreGlobal(global_index), *line);
                ctx.chunk
                    .write_with_line(OpCode::LoadGlobal(global_index), *line);
                ctx.record_bound_name(name);
            }
            Ok(())
        }
        Expr::AssignOp {
            name,
            op,
            value,
            line,
        } => {
            *ctx.current_line = *line;
            compile_assign_op(ctx, name, op, value, *line)
        }
        Expr::AssignArray {
            array,
            index,
            value,
            line,
        } => {
            *ctx.current_line = *line;
            expr::compile_expr(ctx, value)?;
            match index {
                IndexExpr::Scalar(e) => {
                    expr::compile_expr(ctx, e)?;
                    expr::compile_expr(ctx, array)?;
                    if crate::compiler::expr::integral_peephole::expr_may_be_integral(e) {
                        ctx.chunk
                            .write_with_line(OpCode::ObjectSetIntegral, *line);
                    } else {
                        ctx.chunk.write_with_line(OpCode::SetArrayElement, *line);
                    }
                }
                IndexExpr::Slice {
                    start,
                    stop,
                    step,
                    line: sl,
                } => {
                    emit_slice_bound(ctx, start.as_deref(), *sl)?;
                    emit_slice_bound(ctx, stop.as_deref(), *sl)?;
                    emit_slice_bound(ctx, step.as_deref(), *sl)?;
                    expr::compile_expr(ctx, array)?;
                    ctx.chunk.write_with_line(OpCode::SetArraySlice, *line);
                }
            }
            Ok(())
        }
        Expr::AssignArrayOp {
            array,
            index,
            op,
            value,
            line,
        } => {
            *ctx.current_line = *line;
            let IndexExpr::Scalar(ie) = index else {
                return Err(LangError::ParseError {
                    message: "Augmented assignment on array slice is not supported".to_string(),
                    line: *line,
                    file: None,
                });
            };
            expr::compile_expr(ctx, array)?;
            expr::compile_expr(ctx, ie)?;
            ctx.chunk.write_with_line(OpCode::GetArrayElement, *line);
            expr::compile_expr(ctx, value)?;
            match op {
                TokenKind::PlusEqual => ctx.chunk.write_with_line(OpCode::Add, *line),
                TokenKind::MinusEqual => ctx.chunk.write_with_line(OpCode::Sub, *line),
                TokenKind::StarEqual => ctx.chunk.write_with_line(OpCode::Mul, *line),
                TokenKind::StarStarEqual => ctx.chunk.write_with_line(OpCode::Pow, *line),
                TokenKind::SlashEqual => ctx.chunk.write_with_line(OpCode::Div, *line),
                TokenKind::SlashSlashEqual => ctx.chunk.write_with_line(OpCode::IntDiv, *line),
                TokenKind::PercentEqual => ctx.chunk.write_with_line(OpCode::Mod, *line),
                _ => {
                    return Err(LangError::ParseError {
                        message: format!("Unknown assignment operator: {:?}", op),
                        line: *line,
                        file: None,
                    });
                }
            }
            expr::compile_expr(ctx, ie)?;
            expr::compile_expr(ctx, array)?;
            ctx.chunk.write_with_line(OpCode::SetArrayElement, *line);
            Ok(())
        }
        Expr::UnpackAssign { targets, value, line } => {
            *ctx.current_line = *line;
            compile_unpack_assign(ctx, targets, value, *line)
        }
        _ => Err(LangError::ParseError {
            message: "Expected Assign, AssignOp, or UnpackAssign expression".to_string(),
            line: expr.line(),
            file: None,
        }),
    }
}

fn compile_assign_op(
    ctx: &mut CompilationContext,
    name: &str,
    op: &TokenKind,
    value: &Expr,
    line: usize,
) -> Result<(), LangError> {
    // Оператор присваивания: a += b эквивалентно a = a + b
    // Проверяем, является ли это присваиванием к свойству объекта
    if name.contains('.') {
        let parts: Vec<&str> = name.split('.').collect();
        if parts.len() < 2 {
            return Err(LangError::ParseError {
                message: format!("Invalid property path: {}", name),
                line,
                file: None,
            });
        }

        emit_load_property_path_value(ctx, name, line)?;
        expr::compile_expr(ctx, value)?;

        match op {
            TokenKind::PlusEqual => ctx.chunk.write_with_line(OpCode::Add, line),
            TokenKind::MinusEqual => ctx.chunk.write_with_line(OpCode::Sub, line),
            TokenKind::StarEqual => ctx.chunk.write_with_line(OpCode::Mul, line),
            TokenKind::StarStarEqual => ctx.chunk.write_with_line(OpCode::Pow, line),
            TokenKind::SlashEqual => ctx.chunk.write_with_line(OpCode::Div, line),
            TokenKind::SlashSlashEqual => ctx.chunk.write_with_line(OpCode::IntDiv, line),
            TokenKind::PercentEqual => ctx.chunk.write_with_line(OpCode::Mod, line),
            _ => {
                return Err(LangError::ParseError {
                    message: format!("Unknown assignment operator: {:?}", op),
                    line,
                    file: None,
                });
            }
        }

        emit_property_path_store(ctx, name, line)?;
        return Ok(());
    }

    // Обычное присваивание переменной
    // Используем VariableResolver для разрешения переменной
    let is_local = VariableResolver::resolve_for_assign_op(ctx, name, line)?;

    // Компилируем правую часть
    expr::compile_expr(ctx, value)?;

    // Выполняем операцию
    match op {
        TokenKind::PlusEqual => ctx.chunk.write_with_line(OpCode::Add, line),
        TokenKind::MinusEqual => ctx.chunk.write_with_line(OpCode::Sub, line),
        TokenKind::StarEqual => ctx.chunk.write_with_line(OpCode::Mul, line),
        TokenKind::StarStarEqual => ctx.chunk.write_with_line(OpCode::Pow, line),
        TokenKind::SlashEqual => ctx.chunk.write_with_line(OpCode::Div, line),
        TokenKind::SlashSlashEqual => ctx.chunk.write_with_line(OpCode::IntDiv, line),
        TokenKind::PercentEqual => ctx.chunk.write_with_line(OpCode::Mod, line),
        _ => {
            return Err(LangError::ParseError {
                message: format!("Unknown assignment operator: {:?}", op),
                line,
                file: None,
            });
        }
    }

    // Сохраняем результат обратно
    VariableResolver::store_after_operation(ctx, name, is_local, line)?;

    Ok(())
}

/// `a, b = grid.heap_pop(heap_var)` — unpack without tuple temp.
fn try_compile_grid_heap_pop_unpack2(
    ctx: &mut CompilationContext,
    targets: &[AssignTarget],
    heap_name: &str,
    line: usize,
) -> Result<bool, LangError> {
    let (AssignTarget::Name(name0), AssignTarget::Name(name1)) = (&targets[0], &targets[1]) else {
        return Ok(false);
    };
    let f_slot = ctx
        .scope
        .resolve_local(name0)
        .unwrap_or_else(|| ctx.scope.declare_local(name0));
    let n_slot = ctx
        .scope
        .resolve_local(name1)
        .unwrap_or_else(|| ctx.scope.declare_local(name1));
    let heap_slot = ctx.scope.declare_local(&format!("__grid_heap_pop_{line}"));
    VariableResolver::resolve_and_load(ctx, heap_name, line)?;
    ctx.chunk
        .write_with_line(OpCode::StoreLocal(heap_slot), line);
    ctx.chunk.write_with_line(
        OpCode::GridHeapPopUnpack2(f_slot, n_slot, heap_slot),
        line,
    );
    if let Some(last_target) = targets.last() {
        compile_load_assign_target(ctx, last_target, line)?;
    }
    Ok(true)
}

/// `a, b = heapq.heappop(heap_var)` — unpack without tuple temp / GetArrayElement.
fn try_compile_heapq_heappop_unpack2(
    ctx: &mut CompilationContext,
    targets: &[AssignTarget],
    heap_name: &str,
    line: usize,
) -> Result<bool, LangError> {
    let (AssignTarget::Name(name0), AssignTarget::Name(name1)) = (&targets[0], &targets[1]) else {
        return Ok(false);
    };
    let f_slot = ctx
        .scope
        .resolve_local(name0)
        .unwrap_or_else(|| ctx.scope.declare_local(name0));
    let n_slot = ctx
        .scope
        .resolve_local(name1)
        .unwrap_or_else(|| ctx.scope.declare_local(name1));
    VariableResolver::resolve_and_load(ctx, heap_name, line)?;
    ctx.chunk
        .write_with_line(OpCode::HeappopUnpack2(f_slot, n_slot), line);
    if let Some(last_target) = targets.last() {
        compile_load_assign_target(ctx, last_target, line)?;
    }
    Ok(true)
}

fn compile_store_assign_target(
    ctx: &mut CompilationContext,
    target: &AssignTarget,
    line: usize,
) -> Result<(), LangError> {
    // На стеке: [value]
    match target {
        AssignTarget::Name(name) => {
            if let Some(local_index) = ctx.scope.resolve_local(name) {
                ctx.chunk
                    .write_with_line(OpCode::StoreLocal(local_index), line);
            } else if ctx.current_function.is_some()
                && VariableResolver::try_store_explicit_global(ctx, name, line, false)
            {
            } else if ctx.current_function.is_some() {
                let var_index = ctx.declare_local_for_binding(name);
                ctx.chunk
                    .write_with_line(OpCode::StoreLocal(var_index), line);
            } else if let Some(&global_index) = ctx.scope.globals.get(name) {
                ctx.chunk.global_names.insert(global_index, name.clone());
                ctx.chunk
                    .write_with_line(OpCode::StoreGlobal(global_index), line);
            } else {
                let global_index = ctx.scope.globals.len();
                ctx.scope.globals.insert(name.clone(), global_index);
                ctx.chunk.global_names.insert(global_index, name.clone());
                ctx.chunk
                    .write_with_line(OpCode::StoreGlobal(global_index), line);
            }
            ctx.record_bound_name(name);
        }
        AssignTarget::Index { array, index } => {
            expr::compile_expr(ctx, index)?;
            expr::compile_expr(ctx, array)?;
            ctx.chunk.write_with_line(OpCode::SetArrayElement, line);
        }
    }
    Ok(())
}

fn compile_load_assign_target(
    ctx: &mut CompilationContext,
    target: &AssignTarget,
    line: usize,
) -> Result<(), LangError> {
    match target {
        AssignTarget::Name(name) => VariableResolver::resolve_and_load(ctx, name, line),
        AssignTarget::Index { array, index } => {
            expr::compile_expr(ctx, array)?;
            expr::compile_expr(ctx, index)?;
            ctx.chunk.write_with_line(OpCode::GetArrayElement, line);
            Ok(())
        }
    }
}

fn compile_unpack_assign(
    ctx: &mut CompilationContext,
    targets: &[AssignTarget],
    value: &Expr,
    line: usize,
) -> Result<(), LangError> {
    if targets.len() == 2 {
        if let (AssignTarget::Name(q_name), AssignTarget::Name(r_name)) = (&targets[0], &targets[1])
        {
            if let Expr::Call { name, args, .. } = value {
                if name == "divmod" && args.len() == 2 {
                    if let (Arg::Positional(a_expr), Arg::Positional(b_expr)) = (&args[0], &args[1])
                    {
                        expr::compile_expr(ctx, a_expr)?;
                        expr::compile_expr(ctx, b_expr)?;
                        let q_slot = ctx
                            .scope
                            .resolve_local(q_name)
                            .unwrap_or_else(|| ctx.declare_local_for_binding(q_name));
                        let r_slot = ctx
                            .scope
                            .resolve_local(r_name)
                            .unwrap_or_else(|| ctx.declare_local_for_binding(r_name));
                        ctx.chunk.write_with_line(
                            OpCode::DivmodUnpack2(q_slot, r_slot),
                            line,
                        );
                        if let Some(last_target) = targets.last() {
                            compile_load_assign_target(ctx, last_target, line)?;
                        }
                        return Ok(());
                    }
                }
            }
        }
        if let (AssignTarget::Name(_), AssignTarget::Name(_)) = (&targets[0], &targets[1]) {
            if let Expr::CallValue { callee, args, .. } = value {
                if args.len() == 1 {
                    if let Arg::Positional(Expr::Variable { name: heap_name, .. }) = &args[0] {
                        if let Expr::Property {
                            object,
                            name: method,
                            ..
                        } = callee.as_ref()
                        {
                            if let Expr::Variable { name: mod_name, .. } = object.as_ref() {
                                if mod_name == "heapq"
                                    && method == "heappop"
                                    && try_compile_heapq_heappop_unpack2(
                                        ctx, targets, heap_name, line,
                                    )?
                                {
                                    return Ok(());
                                }
                                if mod_name == "grid"
                                    && method == "heap_pop"
                                    && try_compile_grid_heap_pop_unpack2(
                                        ctx, targets, heap_name, line,
                                    )?
                                {
                                    return Ok(());
                                }
                            }
                        }
                    }
                }
            }
            if let Expr::MethodCall {
                object,
                method,
                args,
                ..
            } = value
            {
                if method == "heappop" && args.len() == 1 {
                    if let (
                        Expr::Variable { name: mod_name, .. },
                        Arg::Positional(Expr::Variable { name: heap_name, .. }),
                    ) = (object.as_ref(), &args[0])
                    {
                        if mod_name == "heapq"
                            && try_compile_heapq_heappop_unpack2(ctx, targets, heap_name, line)?
                        {
                            return Ok(());
                        }
                    }
                }
                if method == "heap_pop" && args.len() == 1 {
                    if let (
                        Expr::Variable { name: mod_name, .. },
                        Arg::Positional(Expr::Variable { name: heap_name, .. }),
                    ) = (object.as_ref(), &args[0])
                    {
                        if mod_name == "grid"
                            && try_compile_grid_heap_pop_unpack2(ctx, targets, heap_name, line)?
                        {
                            return Ok(());
                        }
                    }
                }
            }
        }
    }

    // `a, b = expr0, expr1` (TupleLiteral) — evaluate all RHS before any store (parallel assign).
    if let Expr::TupleLiteral { elements, .. } = value {
        if elements.len() == targets.len() {
            let all_names = targets.iter().all(|t| matches!(t, AssignTarget::Name(_)));
            if all_names {
                let mut rhs_temps = Vec::with_capacity(elements.len());
                for (i, elem) in elements.iter().enumerate() {
                    expr::compile_expr(ctx, elem)?;
                    let temp = ctx
                        .scope
                        .declare_local(&format!("__unpack_rhs_{}_{}", line, i));
                    ctx.chunk
                        .write_with_line(OpCode::StoreLocal(temp), line);
                    rhs_temps.push(temp);
                }
                for (target, &temp) in targets.iter().zip(rhs_temps.iter()) {
                    ctx.chunk
                        .write_with_line(OpCode::LoadLocal(temp), line);
                    compile_store_assign_target(ctx, target, line)?;
                }
                if let Some(last_target) = targets.last() {
                    compile_load_assign_target(ctx, last_target, line)?;
                }
                return Ok(());
            }
        }
    }

    // Распаковка: targets = tuple_expr (RHS вычисляется до присваиваний)
    expr::compile_expr(ctx, value)?;

    let tuple_temp = ctx.scope.declare_local(&format!("__tuple_temp_{}", line));
    ctx.chunk
        .write_with_line(OpCode::StoreLocal(tuple_temp), line);

    for (index, target) in targets.iter().enumerate() {
        ctx.chunk
            .write_with_line(OpCode::LoadLocal(tuple_temp), line);
        let index_const = ctx.chunk.add_constant(Value::Number(index as f64));
        ctx.chunk
            .write_with_line(OpCode::Constant(index_const), line);
        ctx.chunk.write_with_line(OpCode::GetArrayElement, line);
        compile_store_assign_target(ctx, target, line)?;
    }

    if let Some(last_target) = targets.last() {
        compile_load_assign_target(ctx, last_target, line)?;
    }

    Ok(())
}
