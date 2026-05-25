//! Compiles method calls: special cases (clone, suffixes, joins, DB receivers) and generic dispatch.

use crate::bytecode::OpCode;
use crate::common::error::LangError;
use crate::common::value::{ObjectKind, Value};
use crate::compiler::args;
use crate::compiler::builtin_methods::{self, ReceiverFamily, ZeroArgDispatch};
use crate::compiler::context::CompilationContext;
use crate::compiler::expr;
use crate::compiler::variable::VariableResolver;
use crate::debug_println;
use crate::parser::ast::{Arg, Expr};

/// Receiver is syntactically a plain `{...}` dict literal (bucket map), not an instance/custom object.
fn expr_is_definitely_plain_dict_literal(e: &Expr) -> bool {
    match e {
        Expr::ObjectLiteral { .. } => true,
        Expr::Literal {
            value: Value::Object(rc),
            ..
        } => {
            let o = rc.borrow();
            object_kind_lacks_visibility_metadata(&o)
        }
        _ => false,
    }
}

#[inline]
fn object_kind_lacks_visibility_metadata(o: &ObjectKind) -> bool {
    !o.str_key_contains("__class_name")
        && !o.str_key_contains("__private_fields")
        && !o.str_key_contains("__protected_fields")
        && !o.str_key_contains("__private_methods")
        && !o.str_key_contains("__protected_methods")
}

/// `cluster.get("primary")` keeps the DB receiver path (args evaluated before receiver).
/// Plain dict `.get(int_var [, default])` uses `compile_module_method` → `ObjectGetIntegral`.
fn get_method_uses_db_receiver_path(args: &[Arg]) -> bool {
    let first = args.first();
    matches!(
        first,
        Some(Arg::Positional(Expr::Literal {
            value: Value::String(_),
            ..
        }))
            | Some(Arg::Named {
                value: Expr::Literal {
                    value: Value::String(_),
                    ..
                },
                ..
            })
    )
}

/// Compiles one call argument expression to the stack (`positional`, named value, or unpack).
#[inline]
fn compile_call_arg(ctx: &mut CompilationContext, arg: &Arg) -> Result<(), LangError> {
    match arg {
        Arg::Positional(expr) => expr::compile_expr(ctx, expr),
        Arg::Named { value, .. } => expr::compile_expr(ctx, value),
        Arg::UnpackObject(expr) => expr::compile_expr(ctx, expr),
    }
}

pub fn compile_method_call(ctx: &mut CompilationContext, expr: &Expr) -> Result<(), LangError> {
    if let Expr::MethodCall {
        object,
        method,
        args: call_args,
        line,
    } = expr
    {
        *ctx.current_line = *line;

        // Специальная обработка для метода clone()
        if method == "clone" {
            expr::compile_expr(ctx, object)?;
            return compile_clone_method(ctx, call_args, *line);
        }

        // Специальная обработка для метода suffixes
        if method == "suffixes" {
            expr::compile_expr(ctx, object)?;
            return compile_suffixes_method(ctx, call_args, *line);
        }

        // Специальная обработка для JOIN методов
        if matches!(
            method.as_str(),
            "inner_join"
                | "left_join"
                | "right_join"
                | "full_join"
                | "cross_join"
                | "semi_join"
                | "anti_join"
                | "zip_join"
                | "asof_join"
                | "apply_join"
                | "join_on"
        ) {
            expr::compile_expr(ctx, object)?;
            return compile_join_method(ctx, method, call_args, *line);
        }

        // NOTE: We intentionally do NOT use the direct Constant+Call path for class methods.
        // Private and protected methods must go through GetArrayElement so the VM can enforce
        // visibility (__private_methods, __protected_methods) at runtime. The direct path would
        // bypass that check. So we always use compile_generic_method -> compile_module_method for
        // instance method calls.
        //
        // Для методов engine/cluster (run, execute, query, ...) сначала компилируем аргументы в временные слоты,
        // затем receiver, чтобы StoreLocal(receiver) не перезаписывал слот переменной-аргумента (например create_all).
        let is_db_receiver = matches!(
            method.as_str(),
            "names" | "connect" | "execute" | "query" | "run"
        ) || (method == "get" && get_method_uses_db_receiver_path(call_args))
            || (method == "add"
                && (call_args.len() >= 2 || get_method_uses_db_receiver_path(call_args)));
        if is_db_receiver {
            let mut arg_slots = Vec::with_capacity(call_args.len());
            for (i, arg) in call_args.iter().enumerate() {
                compile_call_arg(ctx, arg)?;
                let slot = ctx.scope.declare_local(&format!("__arg_{}", i));
                ctx.chunk.write_with_line(OpCode::StoreLocal(slot), *line);
                arg_slots.push(slot);
            }
            expr::compile_expr(ctx, object)?;
            let temp_object_slot = ctx.scope.declare_local("__method_object");
            ctx.chunk
                .write_with_line(OpCode::StoreLocal(temp_object_slot), *line);
            compile_db_receiver_method_with_arg_slots(
                ctx,
                method,
                &arg_slots,
                temp_object_slot,
                *line,
            )
        } else {
            // Общий случай: компилируем объект и вызываем compile_generic_method
            debug_println!("[DEBUG compile_method_call] Метод '{}' не распознан как метод класса, используем compile_generic_method", method);
            expr::compile_expr(ctx, object)?;
            compile_generic_method(ctx, object, method, call_args, *line)
        }
    } else {
        Err(LangError::ParseError {
            message: "Expected MethodCall expression".to_string(),
            line: expr.line(),
            file: None,
        })
    }
}

fn compile_clone_method(
    ctx: &mut CompilationContext,
    args: &[Arg],
    line: usize,
) -> Result<(), LangError> {
    // Для clone() не нужны аргументы
    if !args.is_empty() {
        return Err(LangError::ParseError {
            message: "clone() method takes no arguments".to_string(),
            line,
            file: None,
        });
    }
    // Используем специальный opcode для клонирования
    ctx.chunk.write_with_line(OpCode::Clone, line);
    Ok(())
}

fn compile_suffixes_method(
    ctx: &mut CompilationContext,
    args: &[Arg],
    line: usize,
) -> Result<(), LangError> {
    // Метод suffixes для применения суффиксов к колонкам таблицы
    // Проверяем количество аргументов (должно быть 2)
    if args.len() != 2 {
        return Err(LangError::ParseError {
            message: format!(
                "suffixes() method expects 2 arguments (left_suffix, right_suffix), got {}",
                args.len()
            ),
            line,
            file: None,
        });
    }

    // Table is already on the stack; stack layout before Call: table, left_suffix, right_suffix, native_fn.
    let temp_object_slot = ctx.scope.declare_local("__method_object");
    ctx.chunk
        .write_with_line(OpCode::StoreLocal(temp_object_slot), line);

    let Some(&function_index) = ctx.scope.globals.get("table_suffixes") else {
        return Err(LangError::ParseError {
            message: "Function 'table_suffixes' not found".to_string(),
            line,
            file: None,
        });
    };

    ctx.chunk
        .write_with_line(OpCode::LoadLocal(temp_object_slot), line);
    for arg in args {
        compile_call_arg(ctx, arg)?;
    }
    ctx.chunk
        .write_with_line(OpCode::LoadGlobal(function_index), line);
    ctx.chunk.write_with_line(OpCode::Call(3), line);
    Ok(())
}

fn compile_join_method(
    ctx: &mut CompilationContext,
    method: &str,
    args: &[Arg],
    line: usize,
) -> Result<(), LangError> {
    let Some(&function_index) = ctx.scope.globals.get(method) else {
        return Err(LangError::ParseError {
            message: format!("Function '{}' not found", method),
            line,
            file: None,
        });
    };

    let temp_object_slot = ctx.scope.declare_local("__method_object");
    ctx.chunk
        .write_with_line(OpCode::StoreLocal(temp_object_slot), line);

    ctx.chunk
        .write_with_line(OpCode::LoadLocal(temp_object_slot), line);
    for arg in args {
        compile_call_arg(ctx, arg)?;
    }
    ctx.chunk
        .write_with_line(OpCode::LoadGlobal(function_index), line);
    ctx.chunk
        .write_with_line(OpCode::Call(args.len() + 1), line);
    Ok(())
}

/// Named arg matches the plugin export's param list (from preloaded `native_call_descriptor`).
fn is_kwarg_name_for_export(ctx: &CompilationContext, export_key: &str, name: &str) -> bool {
    ctx.native_call_param_registry
        .and_then(|r| r.get(export_key))
        .map(|params| params.iter().any(|p| p == name))
        .unwrap_or(false)
}

/// Heuristic: plugin/module method path vs builtin (e.g. `String.split` vs `dataset.split`) when the
/// plugin registered `__method__` → export_key for this method name.
fn ambiguous_plugin_method_use_module_path(
    ctx: &CompilationContext,
    args: &[Arg],
    export_key: &str,
) -> bool {
    if args.is_empty() {
        return true;
    }
    if args.len() >= 2 {
        return true;
    }
    if args.iter().any(|a| {
        matches!(a, Arg::Named { name, .. } if is_kwarg_name_for_export(ctx, export_key, name.as_str()))
    }) {
        return true;
    }
    if let Arg::Positional(expr) = &args[0] {
        return matches!(
            expr,
            Expr::Literal {
                value: Value::Number(_),
                ..
            }
        );
    }
    false
}

fn compile_generic_method(
    ctx: &mut CompilationContext,
    object: &Expr,
    method: &str,
    args: &[Arg],
    line: usize,
) -> Result<(), LangError> {
    // Сохраняем объект во временную переменную
    let temp_object_slot = ctx.scope.declare_local("__method_object");
    ctx.chunk
        .write_with_line(OpCode::StoreLocal(temp_object_slot), line);

    // Проверяем, является ли это методом объекта (например, axis.imshow)
    let is_axis_method = matches!(method, "imshow" | "set_title" | "axis");
    let is_string_method = matches!(
        method,
        "lower" | "upper" | "isupper" | "islower" | "trim" | "join" | "contains" | "split"
    );

    if is_axis_method {
        return compile_axis_method(ctx, method, args, temp_object_slot, line);
    }

    if let Some(reg) = ctx.native_call_param_registry {
        if let Some(export_key) = reg.export_for_method(method) {
            if ambiguous_plugin_method_use_module_path(ctx, args, export_key) {
                let param_owned = reg.get(export_key).map(|s| s.to_vec());
                let param_refs: Vec<&str> = param_owned
                    .as_ref()
                    .map(|v| v.iter().map(|s| s.as_str()).collect())
                    .unwrap_or_default();
                let override_native = if param_refs.is_empty() {
                    None
                } else {
                    Some(param_refs.as_slice())
                };
                return compile_module_method(
                    ctx,
                    method,
                    args,
                    temp_object_slot,
                    line,
                    override_native,
                    Some(object),
                );
            }
        }
    }

    if is_string_method {
        compile_string_method(ctx, method, args, temp_object_slot, line)
    } else {
        compile_module_method(ctx, method, args, temp_object_slot, line, None, Some(object))
    }
}

fn compile_axis_method(
    ctx: &mut CompilationContext,
    method: &str,
    args: &[Arg],
    temp_object_slot: usize,
    line: usize,
) -> Result<(), LangError> {
    ctx.chunk
        .write_with_line(OpCode::LoadLocal(temp_object_slot), line);
    for arg in args {
        compile_call_arg(ctx, arg)?;
    }

    ctx.chunk
        .write_with_line(OpCode::LoadLocal(temp_object_slot), line);
    let method_name_index = ctx.chunk.add_constant(Value::String(method.to_string()));
    ctx.chunk
        .write_with_line(OpCode::Constant(method_name_index), line);
    ctx.chunk.write_with_line(OpCode::GetArrayElement, line);

    ctx.chunk
        .write_with_line(OpCode::Call(args.len() + 1), line);
    Ok(())
}

/// DB engine/cluster: args were evaluated first into `arg_slots`; rebuild stack as
/// `receiver`, then positional args, then `receiver` again for `GetArrayElement`, then `Call`.
fn compile_db_receiver_method_with_arg_slots(
    ctx: &mut CompilationContext,
    method: &str,
    arg_slots: &[usize],
    temp_object_slot: usize,
    line: usize,
) -> Result<(), LangError> {
    ctx.chunk
        .write_with_line(OpCode::LoadLocal(temp_object_slot), line);
    for &slot in arg_slots {
        ctx.chunk.write_with_line(OpCode::LoadLocal(slot), line);
    }
    ctx.chunk
        .write_with_line(OpCode::LoadLocal(temp_object_slot), line);
    let method_name_index = ctx.chunk.add_constant(Value::String(method.to_string()));
    ctx.chunk
        .write_with_line(OpCode::Constant(method_name_index), line);
    ctx.chunk.write_with_line(OpCode::GetArrayElement, line);
    ctx.chunk
        .write_with_line(OpCode::Call(1 + arg_slots.len()), line);
    Ok(())
}

fn compile_string_method(
    ctx: &mut CompilationContext,
    method: &str,
    args: &[Arg],
    temp_object_slot: usize,
    line: usize,
) -> Result<(), LangError> {
    // String methods: native receives (receiver, ...args) except join which expects (array, delim).
    // For "".join(chars): receiver is delim, arg is array; native_join(array, delim).
    // So for join we push arg first then receiver; stack [arg, receiver, fn], Call(2) -> args after reverse = [receiver, arg] -> we need [arg, receiver], so push arg, receiver.
    let n = args.len();
    if method == "join" {
        if n != 1 {
            return Err(LangError::ParseError {
                message: "string.join() takes exactly 1 argument (array)".to_string(),
                line,
                file: None,
            });
        }
        for arg in args {
            compile_call_arg(ctx, arg)?;
        }
        ctx.chunk
            .write_with_line(OpCode::LoadLocal(temp_object_slot), line);
    } else {
        if matches!(method, "lower" | "upper" | "isupper" | "islower" | "trim") && n != 0 {
            return Err(LangError::ParseError {
                message: format!("string.{}() takes no arguments", method),
                line,
                file: None,
            });
        }
        if matches!(method, "split" | "contains") && n != 1 {
            return Err(LangError::ParseError {
                message: format!("string.{}() takes exactly 1 argument", method),
                line,
                file: None,
            });
        }
        ctx.chunk
            .write_with_line(OpCode::LoadLocal(temp_object_slot), line);
        for arg in args {
            compile_call_arg(ctx, arg)?;
        }
    }
    ctx.chunk
        .write_with_line(OpCode::LoadLocal(temp_object_slot), line);
    let method_name_index = ctx.chunk.add_constant(Value::String(method.to_string()));
    ctx.chunk
        .write_with_line(OpCode::Constant(method_name_index), line);
    ctx.chunk.write_with_line(OpCode::GetArrayElement, line);
    ctx.chunk.write_with_line(OpCode::Call(1 + n), line);
    Ok(())
}

fn compile_module_method(
    ctx: &mut CompilationContext,
    method: &str,
    args: &[Arg],
    temp_object_slot: usize,
    line: usize,
    override_native_param_names: Option<&[&str]>,
    method_receiver_ast: Option<&Expr>,
) -> Result<(), LangError> {
    // Generic method call (module functions or class instance methods): pass receiver as first arg so method receives (self, arg_1, ...).
    let resolved_args = match args::resolve_function_args(
        method,
        args,
        None,
        line,
        ctx.source_name,
        None,
        override_native_param_names,
    ) {
        Ok(resolved) => resolved,
        Err(e) => {
            // Проверяем, является ли это ошибкой "not supported"
            let error_msg = match &e {
                LangError::ParseError { message, .. } => message,
                _ => "",
            };

            if error_msg.contains("not supported")
                || error_msg.contains("Named arguments are not supported")
            {
                // Fallback: компилируем аргументы как есть
                args.iter()
                    .map(|a| match a {
                        Arg::Positional(e) => Arg::Positional(e.clone()),
                        Arg::Named { value, .. } => Arg::Positional(value.clone()),
                        Arg::UnpackObject(e) => Arg::Positional(e.clone()),
                    })
                    .collect()
            } else {
                return Err(e);
            }
        }
    };

    // Stack before Call: [receiver, arg_1, ..., arg_n, method]. Call(1 + n) so method receives (receiver, arg_1, ...).
    let start_ip = ctx.chunk.code.len();
    debug_println!("[DEBUG compile_module_method] Начало компиляции вызова метода '{}' на строке {}, начальный IP: {}", method, line, start_ip);

    let args_to_compile = if resolved_args.is_empty() {
        debug_println!(
            "[DEBUG compile_module_method] resolved_args пуст, используем исходные args"
        );
        args.iter()
            .map(|a| match a {
                Arg::Positional(e) => Arg::Positional(e.clone()),
                Arg::Named { value, .. } => Arg::Positional(value.clone()),
                Arg::UnpackObject(e) => Arg::Positional(e.clone()),
            })
            .collect::<Vec<_>>()
    } else {
        debug_println!("[DEBUG compile_module_method] используем resolved_args");
        resolved_args.clone()
    };
    debug_println!(
        "[DEBUG compile_module_method] args_to_compile.len() = {}",
        args_to_compile.len()
    );

    let use_intrinsic_prop_shortcut = args_to_compile.is_empty()
        && match builtin_methods::zero_arg_dispatch_for_method(method) {
            Some(ZeroArgDispatch::PropertyViaGetArrayElement(
                ReceiverFamily::PlainDictLike,
            )) => method_receiver_ast.is_some_and(expr_is_definitely_plain_dict_literal),
            Some(ZeroArgDispatch::PropertyViaGetArrayElement(ReceiverFamily::DateLike)) => true,
            None => false,
        };

    // Date fields / plain-dict keys & values: VM exposes these via GetArrayElement (not NativeFunction).
    // Compile zero-arg `d.year()`, `{...}.keys()` like property access — same as `.year` / `.keys` — no Call.
    // Class instances (`HashMap()`) must keep `recv.keys()` as a real call so methods win over dict projection.
    // See [`crate::compiler::builtin_methods`] for the intrinsic name registry.
    if use_intrinsic_prop_shortcut {
        ctx.chunk
            .write_with_line(OpCode::LoadLocal(temp_object_slot), line);
        let method_name_index = ctx.chunk.add_constant(Value::String(method.to_string()));
        ctx.chunk
            .write_with_line(OpCode::Constant(method_name_index), line);
        ctx.chunk.write_with_line(OpCode::GetArrayElement, line);
        return Ok(());
    }

    // Plain dict `.get(key [, default])` — no GetArrayElement + Call (A* g_score / f_score).
    if method == "get" && !args_to_compile.is_empty() && args_to_compile.len() <= 2 {
        ctx.chunk
            .write_with_line(OpCode::LoadLocal(temp_object_slot), line);
        for arg in &args_to_compile {
            compile_call_arg(ctx, arg)?;
        }
        if args_to_compile.len() == 1 {
            let null_idx = ctx.chunk.add_constant(Value::Null);
            ctx.chunk.write_with_line(OpCode::Constant(null_idx), line);
        }
        ctx.chunk
            .write_with_line(OpCode::ObjectGetIntegral, line);
        return Ok(());
    }

    // Set `.add` / `.discard` fast opcodes are NOT emitted here: any receiver with `.add(n)`
    // would be miscompiled (e.g. class method `Calc.add`). Runtime `try_set_integral_mut_early`
    // optimizes real set.add after GetArrayElement + Call resolves SET_ADD.

    // heapq.heappop(heap_var) as expression — two stack values, no native Call.
    if method == "heappop" && args_to_compile.len() == 1 {
        if let Arg::Positional(Expr::Variable { name: heap_name, .. }) = &args_to_compile[0] {
            VariableResolver::resolve_and_load(ctx, heap_name, line)?;
            ctx.chunk.write_with_line(OpCode::HeappopFlat, line);
            return Ok(());
        }
    }

    // A* peephole: heapq.heappush(open_heap, (f, node)) — stack `[heap, tuple]` → flat push without Call.
    // Skip tuples with string elements (lexicographic tie-break tests use nested heap via native).
    if method == "heappush" && args_to_compile.len() == 2 {
        if let (
            Arg::Positional(heap_expr),
            Arg::Positional(Expr::TupleLiteral { elements, .. }),
        ) = (&args_to_compile[0], &args_to_compile[1])
        {
            let flat_ok = elements.len() == 2
                && !elements.iter().any(|el| {
                    matches!(el, Expr::Literal { value: Value::String(_), .. })
                });
            if flat_ok {
                expr::compile_expr(ctx, heap_expr)?;
                for el in elements {
                    expr::compile_expr(ctx, el)?;
                }
                ctx.chunk
                    .write_with_line(OpCode::MakeTuple(2), line);
                ctx.chunk
                    .write_with_line(OpCode::HeappushFlat, line);
                return Ok(());
            }
        }
    }

    // 1. Push receiver first, then compile args → stack [receiver, arg_1, ..., arg_n]
    ctx.chunk
        .write_with_line(OpCode::LoadLocal(temp_object_slot), line);
    for arg in &args_to_compile {
        compile_call_arg(ctx, arg)?;
    }
    debug_println!("[DEBUG compile_module_method] После компиляции аргументов, IP: {}, стек: [receiver, arg_1, ..., arg_n]", ctx.chunk.code.len());

    // 2. Get method: LoadLocal object, Constant(method_name), GetArrayElement → stack [receiver, arg_1, ..., arg_n, method]
    ctx.chunk
        .write_with_line(OpCode::LoadLocal(temp_object_slot), line);
    let method_name_index = ctx.chunk.add_constant(Value::String(method.to_string()));
    ctx.chunk
        .write_with_line(OpCode::Constant(method_name_index), line);
    ctx.chunk.write_with_line(OpCode::GetArrayElement, line);
    debug_println!("[DEBUG compile_module_method] После GetArrayElement для метода '{}', IP: {}, стек: [receiver, arg_1, ..., arg_n, method_function]", method, ctx.chunk.code.len());

    // 3. Call(1 + n): VM pops method then 1+n args → method receives (receiver, arg_1, ..., arg_n)
    let call_arity = 1 + args_to_compile.len();
    let call_ip = ctx.chunk.code.len();
    ctx.chunk.write_with_line(OpCode::Call(call_arity), line);
    debug_println!(
        "[DEBUG compile_module_method] Сгенерирован Call({}) на IP {} для метода '{}'",
        call_arity,
        call_ip,
        method
    );

    // ВАЖНО: После вызова метода Call должен извлечь функцию и аргументы со стека
    // и вызвать метод. Если Call не выполняется (например, из-за ошибки или раннего возврата),
    // функция может остаться на стеке. Это может вызвать проблемы в следующей итерации цикла.
    // Однако, мы не можем добавить Pop здесь, так как Call должен вернуть результат метода.
    // Вместо этого, мы полагаемся на то, что Call правильно обработает стек.

    // Логируем все инструкции, которые были сгенерированы
    debug_println!(
        "[DEBUG compile_module_method] Сгенерированные инструкции для метода '{}' (IP {} - {}):",
        method,
        start_ip,
        ctx.chunk.code.len()
    );
    for i in start_ip..ctx.chunk.code.len() {
        debug_println!(
            "[DEBUG compile_module_method]   IP {}: {:?}",
            i,
            ctx.chunk.code.get(i)
        );
    }

    Ok(())
}
