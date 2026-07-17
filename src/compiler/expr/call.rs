use crate::bytecode::OpCode;
use crate::common::error::LangError;
use crate::common::value::Value;
use crate::compiler::args;
use crate::compiler::context::CompilationContext;
use crate::compiler::expr;
use crate::compiler::stmt::class::{CONSTRUCTING_CLASS_GLOBAL_NAME, MODEL_CONFIG_CLASS_LOAD_INDEX};
use crate::compiler::stmt::function;
/// Компиляция вызовов функций
use crate::debug_println;
use crate::parser::ast::{Arg, BinaryOpKind, Expr, TypePart};

/// Class constructor call: name starts with ASCII uppercase (`Foo`, `Config`). Not `__main__`, `get_marker`, etc.
fn is_class_style_constructor_name(name: &str) -> bool {
    name.chars()
        .next()
        .is_some_and(|c| c.is_ascii_uppercase())
}

/// PascalCase builtin exports that are [`Value::NativeFunction`] factories, not user classes.
fn is_builtin_pascal_native_callable(name: &str) -> bool {
    matches!(
        name,
        "Column" | "MetaData" | "DatabaseCluster" | "Config" | "Field"
    )
}

/// First type-name component for constructor suffix (matches `param_types_suffix` in class.rs).
fn type_suffix_from_param_annotation(param_types: &Option<Vec<TypePart>>) -> Option<String> {
    let parts = param_types.as_ref()?;
    let first = parts.first()?;
    Some(first.format_display())
}

/// Infer argument type from expression for constructor/function overload resolution.
fn infer_arg_type_from_expr(expr: &Expr, ctx: Option<&CompilationContext>) -> Option<String> {
    match expr {
        Expr::Literal { value, .. } => match value {
            Value::Number(n) => {
                if n.fract() == 0.0 && n.is_finite() {
                    Some("int".to_string())
                } else {
                    Some("float".to_string())
                }
            }
            Value::String(_) => Some("str".to_string()),
            Value::Bool(_) => Some("bool".to_string()),
            Value::Null => Some("null".to_string()),
            Value::Array(_) => Some("array".to_string()),
            Value::Object(_) => Some("object".to_string()),
            _ => None,
        },
        Expr::Variable { name, .. } => {
            let fn_idx = ctx?.current_function?;
            let func = ctx?.functions.get(fn_idx)?;
            let pos = func.param_names.iter().position(|p| p == name)?;
            type_suffix_from_param_annotation(func.param_types.get(pos)?)
        }
        Expr::InterpolatedString { .. } => Some("str".to_string()),
        Expr::ArrayLiteral { .. } => Some("array".to_string()),
        Expr::TupleLiteral { .. } => Some("tuple".to_string()),
        _ => None,
    }
}

fn infer_arg_type_from_arg(arg: &Arg, ctx: Option<&CompilationContext>) -> Option<String> {
    let expr = match arg {
        Arg::Positional(e) => e,
        Arg::Named { value, .. } => value,
        Arg::UnpackObject(_) | Arg::UnpackArray(_) => return None,
    };
    infer_arg_type_from_expr(expr, ctx)
}

/// Collect typed constructor names `{Class}::new_{arity}_{types...}` registered for a class.
fn typed_constructor_candidates(
    class_name: &str,
    arity: usize,
    function_names: &[String],
    globals: &std::collections::HashMap<String, usize>,
) -> Vec<String> {
    let prefix = format!("{}::new_{}_", class_name, arity);
    let mut candidates: Vec<String> = function_names
        .iter()
        .filter(|n| n.starts_with(&prefix))
        .cloned()
        .collect();
    for key in globals.keys() {
        if key.starts_with(&prefix) && !candidates.iter().any(|c| c == key) {
            candidates.push(key.clone());
        }
    }
    candidates.sort();
    candidates
}

fn find_typed_constructor_by_inferred_types(
    class_name: &str,
    arity: usize,
    inferred_types: &[String],
    function_names: &[String],
    globals: &std::collections::HashMap<String, usize>,
) -> Option<String> {
    let candidates = typed_constructor_candidates(class_name, arity, function_names, globals);
    if candidates.is_empty() {
        return None;
    }
    let prefix = format!("{}::new_{}_", class_name, arity);
    let combined = inferred_types.join("_");
    if let Some(exact) = candidates
        .iter()
        .find(|c| c.strip_prefix(&prefix).is_some_and(|s| s == combined))
    {
        return Some(exact.clone());
    }
    if arity == 1 && inferred_types.len() == 1 {
        let inferred = &inferred_types[0];
        let matched: Vec<_> = candidates
            .iter()
            .filter(|c| {
                c.strip_prefix(&prefix)
                    .is_some_and(|suffix| crate::common::constructor_overload::ctor_suffix_matches_inferred(suffix, inferred))
            })
            .collect();
        if matched.len() == 1 {
            return Some(matched[0].clone());
        }
    }
    None
}

fn infer_arg_types(call_args: &[Arg], ctx: Option<&CompilationContext>) -> Option<Vec<String>> {
    call_args
        .iter()
        .map(|a| infer_arg_type_from_arg(a, ctx))
        .collect()
}

fn runtime_typed_constructor_call(
    class_name: &str,
    call_args: &[Arg],
) -> ResolvedConstructor {
    ResolvedConstructor {
        ctor_name: class_name.to_string(),
        function_index: 0,
        resolved_args: call_args.to_vec(),
        via_class_object: true,
    }
}

/// When exactly one typed overload exists for this arity, use it (even if base `Class::new_N` is already in globals as a placeholder).
fn apply_single_typed_constructor_override(
    class_name: &str,
    arity: usize,
    constructor_name: &str,
    function_names: &[String],
    globals: &std::collections::HashMap<String, usize>,
) -> String {
    if crate::common::constructor_overload::is_typed_constructor(class_name, constructor_name, arity) {
        return constructor_name.to_string();
    }
    let candidates = typed_constructor_candidates(class_name, arity, function_names, globals);
    if candidates.len() == 1 {
        candidates[0].clone()
    } else {
        constructor_name.to_string()
    }
}

/// Build constructor name for overload resolution: try type-specific name first when args are inferrable.
fn resolve_constructor_name(
    name: &str,
    call_args: &[Arg],
    function_names: &[String],
    globals: &std::collections::HashMap<String, usize>,
    ctx: Option<&CompilationContext>,
) -> String {
    let arity = call_args.len();
    let base_name = format!("{}::new_{}", name, arity);
    // Try to infer types from positional args (all must be inferrable: literals or enclosing params).
    let types: Option<Vec<String>> = call_args
        .iter()
        .map(|a| infer_arg_type_from_arg(a, ctx))
        .collect();
    if let Some(types) = &types {
        if !types.is_empty() {
            let suffix = types.join("_");
            let typed_name = format!("{}::new_{}_{}", name, arity, suffix);
            if function_names.contains(&typed_name) || globals.contains_key(&typed_name) {
                return typed_name;
            }
            if let Some(found) =
                find_typed_constructor_by_inferred_types(name, arity, types, function_names, globals)
            {
                return found;
            }
            // Imported class: ctor lives in the module namespace under the typed name.
            if ctx
                .map(|c| c.imported_symbols.contains_key(name))
                .unwrap_or(false)
            {
                return typed_name;
            }
        }
    }
    if function_names.contains(&base_name) || globals.contains_key(&base_name) {
        return base_name;
    }
    apply_single_typed_constructor_override(name, arity, &base_name, function_names, globals)
}

/// Resolved class constructor call (arity may differ from syntactic call after default args).
struct ResolvedConstructor {
    ctor_name: String,
    #[allow(dead_code)]
    function_index: usize,
    resolved_args: Vec<Arg>,
    /// Imported class: emit `LoadGlobal(Class) + Call(arity)` so runtime picks `new_N` and default args.
    via_class_object: bool,
}

fn constructor_arity_from_name(class_name: &str, ctor_name: &str) -> Option<usize> {
    let prefix = format!("{}::new_", class_name);
    let rest = ctor_name.strip_prefix(&prefix)?;
    let digit_len = rest.chars().take_while(|c| c.is_ascii_digit()).count();
    if digit_len == 0 {
        return None;
    }
    rest[..digit_len].parse().ok()
}

fn list_constructor_names(
    class_name: &str,
    function_names: &[String],
    globals: &std::collections::HashMap<String, usize>,
) -> Vec<String> {
    let prefix = format!("{}::new_", class_name);
    let mut names: Vec<String> = function_names
        .iter()
        .filter(|n| n.starts_with(&prefix))
        .cloned()
        .collect();
    for key in globals.keys() {
        if key.starts_with(&prefix) && !names.iter().any(|n| n == key) {
            names.push(key.clone());
        }
    }
    names.sort();
    names
}

fn function_index_for_ctor(ctx: &CompilationContext, ctor_name: &str) -> Option<usize> {
    ctx.function_names.iter().position(|n| n == ctor_name)
}

fn ctor_supplies_defaults(func: &crate::bytecode::Function, call_arity: usize) -> bool {
    let m = func.param_names.len();
    if m < call_arity {
        return false;
    }
    (call_arity..m).all(|i| {
        func.default_values
            .get(i)
            .and_then(|v| v.as_ref())
            .is_some()
    })
}

/// Resolve `ClassName(...)` to a constructor function, applying default parameter values when needed.
fn resolve_constructor_call(
    ctx: &CompilationContext,
    class_name: &str,
    call_args: &[Arg],
    line: usize,
) -> Result<Option<ResolvedConstructor>, LangError> {
    let call_arity = call_args.len();

    if is_builtin_pascal_native_callable(class_name) {
        return Ok(None);
    }

    let try_resolve = |ctor_name: &str| -> Result<Option<ResolvedConstructor>, LangError> {
        let Some(function_index) = function_index_for_ctor(ctx, ctor_name) else {
            return Ok(None);
        };
        let func = &ctx.functions[function_index];
        if call_arity > func.param_names.len() {
            return Ok(None);
        }
        if !ctor_supplies_defaults(func, call_arity)
            && call_arity != func.param_names.len()
        {
            return Ok(None);
        }
        let resolved_args = args::resolve_function_args(
            ctor_name,
            call_args,
            Some((function_index, func)),
            line,
            ctx.source_name,
            None,
            None,
        )?;
        Ok(Some(ResolvedConstructor {
            ctor_name: ctor_name.to_string(),
            function_index,
            resolved_args,
            via_class_object: false,
        }))
    };

    let exact_name = resolve_constructor_name(
        class_name,
        call_args,
        ctx.function_names,
        &ctx.scope.globals,
        Some(ctx),
    );
    if function_index_for_ctor(ctx, &exact_name).is_some() {
        return try_resolve(&exact_name);
    }

    // Imported class without local function body: call via class object so runtime resolves
    // `new_N` (including constructors whose parameters have defaults, e.g. `HashMap()` → `new_1`).
    if ctx.imported_symbols.contains_key(class_name)
        && !is_builtin_pascal_native_callable(class_name)
        && function_index_for_ctor(ctx, &exact_name).is_none()
    {
        if let Some(m) = constructor_arity_from_name(class_name, &exact_name) {
            if call_arity == m {
                return Ok(Some(ResolvedConstructor {
                    ctor_name: class_name.to_string(),
                    function_index: 0,
                    resolved_args: call_args.to_vec(),
                    via_class_object: true,
                }));
            }
        }
    }

    // Pre-reserved global constructor slot (body in merged module at runtime, not in this compiler unit).
    if let Some(m) = constructor_arity_from_name(class_name, &exact_name) {
        let typed_overloads =
            typed_constructor_candidates(class_name, call_arity, ctx.function_names, &ctx.scope.globals);
        if typed_overloads.len() <= 1
            && call_arity == m
            && ctx.scope.globals.contains_key(&exact_name)
            && !ctx.imported_symbols.contains_key(class_name)
        {
            return Ok(Some(ResolvedConstructor {
                ctor_name: exact_name,
                function_index: 0,
                resolved_args: call_args.to_vec(),
                via_class_object: false,
            }));
        }
    }

    let mut matching: Vec<String> = Vec::new();
    let mut min_m: Option<usize> = None;
    for ctor_name in list_constructor_names(class_name, ctx.function_names, &ctx.scope.globals) {
        let Some(m) = constructor_arity_from_name(class_name, &ctor_name) else {
            continue;
        };
        if m < call_arity {
            continue;
        }
        let Some(function_index) = function_index_for_ctor(ctx, &ctor_name) else {
            continue;
        };
        let func = &ctx.functions[function_index];
        if !ctor_supplies_defaults(func, call_arity) {
            continue;
        }
        match min_m {
            None => {
                min_m = Some(m);
                matching.push(ctor_name);
            }
            Some(best_m) if m < best_m => {
                min_m = Some(m);
                matching.clear();
                matching.push(ctor_name);
            }
            Some(best_m) if m == best_m => matching.push(ctor_name),
            _ => {}
        }
    }

    if matching.len() == 1 {
        return try_resolve(&matching[0]);
    }

    if matching.len() > 1 {
        if let Some(types) = infer_arg_types(call_args, Some(ctx)) {
            if let Some(typed) = find_typed_constructor_by_inferred_types(
                class_name,
                call_arity,
                &types,
                ctx.function_names,
                &ctx.scope.globals,
            ) {
                return try_resolve(&typed);
            }
        }
        if matching.iter().all(|n| {
            crate::common::constructor_overload::is_typed_constructor(class_name, n, call_arity)
        }) {
            return Ok(Some(runtime_typed_constructor_call(
                class_name,
                call_args,
            )));
        }
        return Err(LangError::ParseError {
            message: format!(
                "Ambiguous constructor call for class '{}': multiple constructors match {} argument(s)",
                class_name, call_arity
            ),
            line,
            file: None,
        });
    }

    if let Some(resolved) = try_imported_or_none(ctx, class_name, call_args) {
        return Ok(Some(resolved));
    }

    Ok(None)
}

fn try_imported_or_none(
    ctx: &CompilationContext,
    class_name: &str,
    call_args: &[Arg],
) -> Option<ResolvedConstructor> {
    if ctx.imported_symbols.contains_key(class_name)
        && !is_builtin_pascal_native_callable(class_name)
    {
        return Some(runtime_typed_constructor_call(class_name, call_args));
    }
    None
}

fn ensure_constructor_global_slot(ctx: &mut CompilationContext, ctor_name: &str) -> usize {
    if let Some(&idx) = ctx.scope.globals.get(ctor_name) {
        return idx;
    }
    let idx = ctx.scope.globals.len();
    ctx.scope.globals.insert(ctor_name.to_string(), idx);
    ctx.chunk
        .global_names
        .insert(idx, ctor_name.to_string());
    idx
}

fn emit_resolved_constructor_call(
    ctx: &mut CompilationContext,
    resolved: &ResolvedConstructor,
    line: usize,
) -> Result<(), LangError> {
    for arg in &resolved.resolved_args {
        match arg {
            Arg::Positional(expr) => expr::compile_expr(ctx, expr)?,
            Arg::Named { value, .. } => expr::compile_expr(ctx, value)?,
            Arg::UnpackObject(expr) | Arg::UnpackArray(expr) => expr::compile_expr(ctx, expr)?,
        }
    }
    let arity = resolved.resolved_args.len();
    if resolved.via_class_object {
        let class_global_index = ensure_constructor_global_slot(ctx, &resolved.ctor_name);
        ctx.chunk
            .global_names
            .insert(class_global_index, resolved.ctor_name.clone());
        ctx.chunk
            .write_with_line(OpCode::LoadGlobal(class_global_index), line);
    } else if let Some(function_index) = function_index_for_ctor(ctx, &resolved.ctor_name) {
        let constant_index = ctx.chunk.add_constant(Value::Function(function_index));
        ctx.chunk
            .write_with_line(OpCode::Constant(constant_index), line);
    } else {
        let global_index = ensure_constructor_global_slot(ctx, &resolved.ctor_name);
        ctx.chunk
            .write_with_line(OpCode::LoadGlobal(global_index), line);
    }
    ctx.chunk.write_with_line(OpCode::Call(arity), line);
    Ok(())
}

/// Emit `: this(...)` delegation to another constructor overload in the same class (no superclass).
pub fn emit_same_class_delegate(
    ctx: &mut CompilationContext,
    class_name: &str,
    current_ctor_name: &str,
    class_global_index: usize,
    delegate_exprs: &[Expr],
    this_slot: usize,
    line: usize,
) -> Result<(), LangError> {
    let delegate_args: Vec<Arg> = delegate_exprs
        .iter()
        .map(|e| Arg::Positional(e.clone()))
        .collect();
    let arity = delegate_args.len();

    let (target_ctor, via_class_object) = resolve_same_class_delegate_ctor(
        class_name,
        current_ctor_name,
        &delegate_args,
        ctx,
        line,
    )?;

    if target_ctor == current_ctor_name {
        return Err(LangError::ParseError {
            message: format!(
                "Constructor ': this(...)' in class '{}' cannot delegate to itself ('{}')",
                class_name, current_ctor_name
            ),
            line,
            file: None,
        });
    }

    for arg in &delegate_args {
        match arg {
            Arg::Positional(expr) => expr::compile_expr(ctx, expr)?,
            Arg::Named { value, .. } => expr::compile_expr(ctx, value)?,
            Arg::UnpackObject(expr) | Arg::UnpackArray(expr) => expr::compile_expr(ctx, expr)?,
        }
    }

    if via_class_object {
        ctx.chunk
            .global_names
            .insert(class_global_index, class_name.to_string());
        ctx.chunk
            .write_with_line(OpCode::LoadGlobal(class_global_index), line);
    } else if let Some(function_index) = function_index_for_ctor(ctx, &target_ctor) {
        let constant_index = ctx.chunk.add_constant(Value::Function(function_index));
        ctx.chunk
            .write_with_line(OpCode::Constant(constant_index), line);
    } else if let Some(&global_index) = ctx.scope.globals.get(&target_ctor) {
        ctx.chunk
            .global_names
            .insert(global_index, target_ctor.clone());
        ctx.chunk
            .write_with_line(OpCode::LoadGlobal(global_index), line);
    } else {
        return Err(LangError::ParseError {
            message: format!(
                "Constructor ': this(...)' in class '{}' could not resolve target '{}'",
                class_name, target_ctor
            ),
            line,
            file: None,
        });
    }

    ctx.chunk.write_with_line(OpCode::Call(arity), line);
    ctx.chunk
        .write_with_line(OpCode::StoreLocal(this_slot), line);
    Ok(())
}

fn resolve_same_class_delegate_ctor(
    class_name: &str,
    current_ctor_name: &str,
    delegate_args: &[Arg],
    ctx: &CompilationContext,
    line: usize,
) -> Result<(String, bool), LangError> {
    let arity = delegate_args.len();
    let mut resolved = resolve_constructor_name(
        class_name,
        delegate_args,
        ctx.function_names,
        &ctx.scope.globals,
        Some(ctx),
    );
    resolved = apply_single_typed_constructor_override(
        class_name,
        arity,
        &resolved,
        ctx.function_names,
        &ctx.scope.globals,
    );

    if resolved != current_ctor_name
        && (function_index_for_ctor(ctx, &resolved).is_some()
            || ctx.scope.globals.contains_key(&resolved))
    {
        return Ok((resolved, false));
    }

    let candidates = typed_constructor_candidates(
        class_name,
        arity,
        ctx.function_names,
        &ctx.scope.globals,
    );
    let others: Vec<&String> = candidates
        .iter()
        .filter(|c| *c != current_ctor_name)
        .collect();

    match others.len() {
        0 => Err(LangError::ParseError {
            message: format!(
                "Constructor ': this(...)' in class '{}' has no other overload with {} argument(s) to delegate to",
                class_name, arity
            ),
            line,
            file: None,
        }),
        1 => Ok((others[0].clone(), false)),
        _ => {
            if let Some(types) = infer_arg_types(delegate_args, Some(ctx)) {
                if let Some(found) = find_typed_constructor_by_inferred_types(
                    class_name,
                    arity,
                    &types,
                    ctx.function_names,
                    &ctx.scope.globals,
                ) {
                    if found != current_ctor_name {
                        return Ok((found, false));
                    }
                }
            }
            Ok((class_name.to_string(), true))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn resolve_constructor_name_uses_unique_typed_overload_when_arg_type_unknown() {
        let function_names = vec!["Deque::new_1_int".to_string()];
        let globals = HashMap::new();
        let args = vec![Arg::Positional(Expr::Binary {
            left: Box::new(Expr::Call {
                name: "len".to_string(),
                args: vec![Arg::Positional(Expr::Variable {
                    name: "s".to_string(),
                    line: 1,
                })],
                line: 1,
            }),
            op: crate::parser::ast::BinaryOpKind::Builtin(crate::lexer::TokenKind::Plus),
            right: Box::new(Expr::Literal {
                value: Value::Number(4.0),
                line: 1,
            }),
            line: 1,
        })];
        assert_eq!(
            resolve_constructor_name("Deque", &args, &function_names, &globals, None),
            "Deque::new_1_int"
        );
    }

    #[test]
    fn resolve_constructor_name_prefers_inferred_literal_type() {
        let function_names = vec![
            "Deque::new_1".to_string(),
            "Deque::new_1_int".to_string(),
        ];
        let globals = HashMap::new();
        let args = vec![Arg::Positional(Expr::Literal {
            value: Value::Number(4.0),
            line: 1,
        })];
        assert_eq!(
            resolve_constructor_name("Deque", &args, &function_names, &globals, None),
            "Deque::new_1_int"
        );
    }

    #[test]
    fn constructor_arity_from_name_parses_typed_suffix() {
        assert_eq!(
            constructor_arity_from_name("HashSet", "HashSet::new_1"),
            Some(1)
        );
        assert_eq!(
            constructor_arity_from_name("Deque", "Deque::new_1_int"),
            Some(1)
        );
    }

    fn resolve_constructor_name_keeps_base_when_multiple_typed_overloads() {
        let function_names = vec![
            "Foo::new_1_int".to_string(),
            "Foo::new_1_str".to_string(),
        ];
        let globals = HashMap::new();
        let args = vec![Arg::Positional(Expr::Variable {
            name: "x".to_string(),
            line: 1,
        })];
        assert_eq!(
            resolve_constructor_name("Foo", &args, &function_names, &globals, None),
            "Foo::new_1"
        );
    }
}

/// True if class name is "Settings" or has Settings as an ancestor (used for 1-arg call expansion).
fn is_in_settings_chain(
    name: &str,
    class_superclass: &std::collections::HashMap<String, String>,
) -> bool {
    if name == "Settings" {
        return true;
    }
    class_superclass
        .get(name)
        .map(|parent| is_in_settings_chain(parent, class_superclass))
        .unwrap_or(false)
}

pub fn compile_call(ctx: &mut CompilationContext, expr: &Expr) -> Result<(), LangError> {
    if let Expr::Call {
        name,
        args: call_args,
        line,
    } = expr
    {
        *ctx.current_line = *line;

        if name.starts_with('@') {
            return Err(LangError::ParseError {
                message: format!(
                    "Special methods cannot be called directly (got `{}()`); they are invoked by operators and builtins",
                    name
                ),
                line: *line,
                file: ctx.source_name.map(|s| s.to_string()),
            });
        }

        if name == "abs" && call_args.len() == 1 {
            if let Arg::Positional(arg_expr) = &call_args[0] {
                if let Expr::Binary {
                    left,
                    op: BinaryOpKind::Builtin(crate::lexer::TokenKind::Minus),
                    right: sub_right,
                    ..
                } = arg_expr
                {
                    expr::compile_expr(ctx, left)?;
                    expr::compile_expr(ctx, sub_right)?;
                    ctx.chunk.write_with_line(OpCode::Sub, *line);
                    ctx.chunk.write_with_line(OpCode::AbsI32, *line);
                    return Ok(());
                }
                if crate::compiler::expr::integral_peephole::expr_may_be_integral(arg_expr) {
                    expr::compile_expr(ctx, arg_expr)?;
                    ctx.chunk.write_with_line(OpCode::AbsI32, *line);
                    return Ok(());
                }
            }
        }

        // array(map(labels, fn(x) => one_hot(x, K)[0])) → onehots(tensor(labels), K); requires `onehots` in scope (injected import).
        if name == "array" && call_args.len() == 1 {
            if let Some((labels_expr, k_expr)) =
                crate::compiler::array_map_onehot_fusion::try_match_array_map_onehot_fusion(expr)
            {
                if let Some(&onehots_idx) = ctx.scope.globals.get("onehots") {
                    ctx.chunk
                        .global_names
                        .insert(onehots_idx, "onehots".to_string());
                    let tensor_expr = Expr::Call {
                        name: "tensor".to_string(),
                        args: vec![Arg::Positional(labels_expr)],
                        line: *line,
                    };
                    expr::compile_expr(ctx, &tensor_expr)?;
                    expr::compile_expr(ctx, &k_expr)?;
                    ctx.chunk
                        .write_with_line(OpCode::LoadGlobal(onehots_idx), *line);
                    ctx.chunk.write_with_line(OpCode::Call(2), *line);
                    return Ok(());
                }
            }
        }

        // Constructor resolution only for PascalCase names (`Foo()`). `__main__`, `get_marker`, `_private` are functions.
        let is_class_style_ctor = is_class_style_constructor_name(name);

        if !is_class_style_ctor {
            // Обычная функция (нативная или пользовательская) — не конструктор класса
            debug_println!(
                "[DEBUG compile_call] Функция '{}' не PascalCase-конструктор, обрабатываем как обычный вызов",
                name
            );
        } else if name == "ValueError" {
            let resolved_args = args::resolve_function_args(
                name,
                call_args,
                None,
                *line,
                ctx.source_name,
                ctx.imported_symbols.get(name).map(|s| s.as_str()),
                None,
            )?;
            for arg in &resolved_args {
                match arg {
                    Arg::Positional(expr) => expr::compile_expr(ctx, expr)?,
                    Arg::Named { value, .. } => expr::compile_expr(ctx, value)?,
                    Arg::UnpackObject(expr) | Arg::UnpackArray(expr) => expr::compile_expr(ctx, expr)?,
                }
            }
            let ctor_name = format!("ValueError::new_{}", resolved_args.len());
            let global_index = ensure_constructor_global_slot(ctx, &ctor_name);
            ctx.chunk
                .global_names
                .insert(global_index, ctor_name.clone());
            ctx.chunk
                .write_with_line(OpCode::LoadGlobal(global_index), *line);
            ctx.chunk
                .write_with_line(OpCode::Call(resolved_args.len()), *line);
            return Ok(());
        } else {
            // Имя начинается с заглавной буквы - это может быть конструктор класса
            let skip_default_ctor_resolve = name == "Settings"
                || is_in_settings_chain(name, ctx.class_superclass)
                || ctx.abstract_classes.contains(name);
            if !skip_default_ctor_resolve {
                if let Some(resolved) = resolve_constructor_call(ctx, name, call_args, *line)? {
                    emit_resolved_constructor_call(ctx, &resolved, *line)?;
                    return Ok(());
                }
            }

            // Конструкторы: ClassName::new_<arity> или ClassName::new_<arity>_<type1>_<type2> для overloading by type
            let mut constructor_name = resolve_constructor_name(
                name,
                call_args,
                ctx.function_names,
                &ctx.scope.globals,
                Some(ctx),
            );
            constructor_name = apply_single_typed_constructor_override(
                name,
                call_args.len(),
                &constructor_name,
                ctx.function_names,
                &ctx.scope.globals,
            );
            debug_println!(
                "[DEBUG compile_call] Проверяем вызов '{}' с {} аргументами",
                name,
                call_args.len()
            );
            debug_println!(
                "[DEBUG compile_call] Ищем конструктор '{}'",
                constructor_name
            );

            // Settings() with no args: expand to Settings(__constructing_class__["model_config"]) so env file comes from model_config.
            if name == "Settings"
                && call_args.is_empty()
                && !ctx.function_names.iter().any(|n| n == "Settings::new_0")
            {
                let settings_slot = if let Some(&idx) = ctx.scope.globals.get("Settings") {
                    idx
                } else {
                    let idx = ctx.scope.globals.len();
                    ctx.scope.globals.insert("Settings".to_string(), idx);
                    idx
                };
                ctx.chunk.global_names.insert(
                    MODEL_CONFIG_CLASS_LOAD_INDEX,
                    CONSTRUCTING_CLASS_GLOBAL_NAME.to_string(),
                );
                ctx.chunk
                    .write_with_line(OpCode::LoadGlobal(MODEL_CONFIG_CLASS_LOAD_INDEX), *line);
                let model_config_key = ctx
                    .chunk
                    .add_constant(Value::String("model_config".to_string()));
                ctx.chunk
                    .write_with_line(OpCode::Constant(model_config_key), *line);
                ctx.chunk.write_with_line(OpCode::GetArrayElement, *line);
                ctx.chunk
                    .global_names
                    .insert(settings_slot, "Settings".to_string());
                ctx.chunk
                    .write_with_line(OpCode::LoadGlobal(settings_slot), *line);
                ctx.chunk.write_with_line(OpCode::Call(1), *line);
                return Ok(());
            }

            // Settings subclass with 0 args (e.g. ProdSettings()): expand to (path="", required_keys, model_config) and Call(3).
            // Set __constructing_class__ = class before pushing args so required_keys and model_config are always loaded from the class we're instantiating (fixes cross-module call e.g. load_settings() -> DevSettings()).
            let arg_count = call_args.len();
            if arg_count == 0 && is_in_settings_chain(name, ctx.class_superclass) {
                let ctor_1_name = format!("{}::new_1", name);
                let class_global_index_opt = ctx.scope.globals.get(name).copied();
                let required_keys_value = ctx.class_required_keys_value.get(name).cloned();
                // Same-file: use compiler's required_keys; imported class: load from class["__required_keys"] at runtime.
                let emit_0_arg_settings = |ctx: &mut CompilationContext,
                                           line: usize,
                                           use_class_required_keys: bool|
                 -> Result<(), LangError> {
                    let class_global_index =
                        class_global_index_opt.expect("Settings class in globals");
                    let new_idx = ctx.scope.globals.len();
                    let constructing_class_idx = *ctx
                        .scope
                        .globals
                        .entry(CONSTRUCTING_CLASS_GLOBAL_NAME.to_string())
                        .or_insert(new_idx);
                    ctx.chunk.global_names.insert(
                        constructing_class_idx,
                        CONSTRUCTING_CLASS_GLOBAL_NAME.to_string(),
                    );
                    ctx.chunk
                        .write_with_line(OpCode::LoadGlobal(class_global_index), line);
                    ctx.chunk
                        .write_with_line(OpCode::StoreGlobal(constructing_class_idx), line);
                    let path_empty = ctx.chunk.add_constant(Value::String(String::new()));
                    ctx.chunk
                        .write_with_line(OpCode::Constant(path_empty), line);
                    if use_class_required_keys {
                        let req_const = ctx
                            .chunk
                            .add_constant(required_keys_value.as_ref().unwrap().clone());
                        ctx.chunk.write_with_line(OpCode::Constant(req_const), line);
                    } else {
                        ctx.chunk
                            .write_with_line(OpCode::LoadGlobal(constructing_class_idx), line);
                        let req_key = ctx
                            .chunk
                            .add_constant(Value::String("__required_keys".to_string()));
                        ctx.chunk.write_with_line(OpCode::Constant(req_key), line);
                        ctx.chunk.write_with_line(OpCode::GetArrayElement, line);
                    }
                    ctx.chunk
                        .write_with_line(OpCode::LoadGlobal(constructing_class_idx), line);
                    let model_config_key = ctx
                        .chunk
                        .add_constant(Value::String("model_config".to_string()));
                    ctx.chunk
                        .write_with_line(OpCode::Constant(model_config_key), line);
                    ctx.chunk.write_with_line(OpCode::GetArrayElement, line);
                    let null_const = ctx.chunk.add_constant(Value::Null);
                    ctx.chunk
                        .write_with_line(OpCode::Constant(null_const), line);
                    Ok(())
                };
                if required_keys_value.is_some() {
                    if let Some(function_index) =
                        ctx.function_names.iter().position(|n| n == &ctor_1_name)
                    {
                        emit_0_arg_settings(ctx, *line, true)?;
                        let constant_index =
                            ctx.chunk.add_constant(Value::Function(function_index));
                        ctx.chunk
                            .write_with_line(OpCode::Constant(constant_index), *line);
                        ctx.chunk.write_with_line(OpCode::Call(4), *line);
                        return Ok(());
                    }
                    if ctx.scope.globals.contains_key(name)
                        && ctx.scope.globals.contains_key(&ctor_1_name)
                    {
                        let &global_index = ctx.scope.globals.get(&ctor_1_name).unwrap();
                        emit_0_arg_settings(ctx, *line, true)?;
                        ctx.chunk
                            .global_names
                            .insert(global_index, ctor_1_name.clone());
                        ctx.chunk
                            .write_with_line(OpCode::LoadGlobal(global_index), *line);
                        ctx.chunk.write_with_line(OpCode::Call(4), *line);
                        return Ok(());
                    }
                }
                // Imported Settings class: load required_keys from class["__required_keys"].
                if ctx.scope.globals.contains_key(name)
                    && ctx.scope.globals.contains_key(&ctor_1_name)
                {
                    let &global_index = ctx.scope.globals.get(&ctor_1_name).unwrap();
                    emit_0_arg_settings(ctx, *line, false)?;
                    ctx.chunk
                        .global_names
                        .insert(global_index, ctor_1_name.clone());
                    ctx.chunk
                        .write_with_line(OpCode::LoadGlobal(global_index), *line);
                    ctx.chunk.write_with_line(OpCode::Call(4), *line);
                    return Ok(());
                }
            }

            // Сначала проверяем в function_names (для конструкторов, определенных в текущем файле)
            if let Some(function_index) = ctx
                .function_names
                .iter()
                .position(|n| n == &constructor_name)
            {
                // Это вызов конструктора
                debug_println!("[DEBUG compile_call] Найден конструктор '{}' с индексом функции {} в function_names", constructor_name, function_index);
                let arg_count = call_args.len();
                debug_println!("[DEBUG compile_call] Сохранено количество аргументов: {} для конструктора '{}' (function_names)", arg_count, constructor_name);
                // 0-arg Settings subclass: set __constructing_class__ = class so new_0's body can load required_keys/model_config from it.
                if arg_count == 0 && is_in_settings_chain(name, ctx.class_superclass) {
                    if let Some(&class_global_index) = ctx.scope.globals.get(name) {
                        let new_idx = ctx.scope.globals.len();
                        let constructing_class_idx = *ctx
                            .scope
                            .globals
                            .entry(CONSTRUCTING_CLASS_GLOBAL_NAME.to_string())
                            .or_insert(new_idx);
                        ctx.chunk.global_names.insert(
                            constructing_class_idx,
                            CONSTRUCTING_CLASS_GLOBAL_NAME.to_string(),
                        );
                        ctx.chunk
                            .write_with_line(OpCode::LoadGlobal(class_global_index), *line);
                        ctx.chunk
                            .write_with_line(OpCode::StoreGlobal(constructing_class_idx), *line);
                    }
                }
                // Settings subclass with 1 arg: expand to (path, required_keys, model_config) and Call(3).
                if arg_count == 1 && is_in_settings_chain(name, ctx.class_superclass) {
                    let required_keys_value = ctx.class_required_keys_value.get(name).cloned();
                    if let Some(required_keys_value) = required_keys_value {
                        match &call_args[0] {
                            Arg::Positional(e) => expr::compile_expr(ctx, e)?,
                            Arg::Named { value, .. } => expr::compile_expr(ctx, value)?,
                            Arg::UnpackObject(e) | Arg::UnpackArray(e) => expr::compile_expr(ctx, e)?,
                        }
                        let req_const = ctx.chunk.add_constant(required_keys_value);
                        ctx.chunk
                            .write_with_line(OpCode::Constant(req_const), *line);
                        let class_global_index = *ctx
                            .scope
                            .globals
                            .get(name)
                            .expect("Settings class in globals");
                        ctx.chunk
                            .global_names
                            .insert(class_global_index, name.clone());
                        ctx.chunk
                            .write_with_line(OpCode::LoadGlobal(class_global_index), *line);
                        let model_config_key = ctx
                            .chunk
                            .add_constant(Value::String("model_config".to_string()));
                        ctx.chunk
                            .write_with_line(OpCode::Constant(model_config_key), *line);
                        ctx.chunk.write_with_line(OpCode::GetArrayElement, *line);
                        let null_const = ctx.chunk.add_constant(Value::Null);
                        ctx.chunk
                            .write_with_line(OpCode::Constant(null_const), *line);
                        let constant_index =
                            ctx.chunk.add_constant(Value::Function(function_index));
                        ctx.chunk
                            .write_with_line(OpCode::Constant(constant_index), *line);
                        ctx.chunk.write_with_line(OpCode::Call(4), *line);
                        return Ok(());
                    }
                }

                for (i, arg) in call_args.iter().enumerate() {
                    match arg {
                        Arg::Positional(expr) => {
                            debug_println!("[DEBUG compile_call] Компилируем позиционный аргумент {} из {} (function_names)", i + 1, arg_count);
                            expr::compile_expr(ctx, expr)?;
                        }
                        Arg::Named { value, .. } => {
                            debug_println!("[DEBUG compile_call] Компилируем именованный аргумент {} из {} (function_names)", i + 1, arg_count);
                            expr::compile_expr(ctx, value)?;
                        }
                        Arg::UnpackObject(expr) | Arg::UnpackArray(expr) => {
                            debug_println!("[DEBUG compile_call] Компилируем ** аргумент {} из {} (function_names)", i + 1, arg_count);
                            expr::compile_expr(ctx, expr)?;
                        }
                    }
                }

                if ctx.abstract_classes.contains(name) {
                    let class_global_index = *ctx
                        .scope
                        .globals
                        .get(name)
                        .expect("abstract class must be in globals");
                    ctx.chunk
                        .write_with_line(OpCode::LoadGlobal(class_global_index), *line);
                    ctx.chunk.write_with_line(OpCode::Call(arg_count), *line);
                    return Ok(());
                }

                let constant_index = ctx.chunk.add_constant(Value::Function(function_index));
                let constant_ip = ctx.chunk.code.len();
                ctx.chunk
                    .write_with_line(OpCode::Constant(constant_index), *line);
                let call_ip = ctx.chunk.code.len();
                ctx.chunk.write_with_line(OpCode::Call(arg_count), *line);
                debug_println!("[DEBUG compile_call] Сгенерирован OpCode::Constant({}) на IP {}, OpCode::Call({}) на IP {} для конструктора '{}' (function_names)", 
                constant_index, constant_ip, arg_count, call_ip, constructor_name);

                if let Some(OpCode::Call(recorded_arity)) = ctx.chunk.code.get(call_ip) {
                    if *recorded_arity != arg_count {
                        debug_println!("[ERROR compile_call] КРИТИЧЕСКАЯ ОШИБКА: Записано Call({}), но ожидалось Call({})!", recorded_arity, arg_count);
                    }
                }

                return Ok(());
            }

            // Если конструктор не найден в function_names, проверяем в глобальных переменных
            // Это нужно для конструкторов, импортированных из модулей.
            // Входим сюда только если конструктор уже зарегистрирован (класс скомпилирован);
            // иначе вызов Base(10) при обычной функции Base не должен создавать слот Base::new_1.
            if ctx.scope.globals.contains_key(name)
                && ctx.scope.globals.contains_key(&constructor_name)
            {
                let &global_index = ctx.scope.globals.get(&constructor_name).unwrap();
                debug_println!(
                    "[DEBUG compile_call] Найден конструктор '{}' в globals с индексом {}",
                    constructor_name,
                    global_index
                );
                let arg_count = call_args.len();
                debug_println!("[DEBUG compile_call] Сохранено количество аргументов: {} для конструктора '{}' (globals)", arg_count, constructor_name);
                // 0-arg Settings subclass: set __constructing_class__ = class so new_0's body can load required_keys/model_config from it.
                if arg_count == 0 && is_in_settings_chain(name, ctx.class_superclass) {
                    if let Some(&class_global_index) = ctx.scope.globals.get(name) {
                        let new_idx = ctx.scope.globals.len();
                        let constructing_class_idx = *ctx
                            .scope
                            .globals
                            .entry(CONSTRUCTING_CLASS_GLOBAL_NAME.to_string())
                            .or_insert(new_idx);
                        ctx.chunk.global_names.insert(
                            constructing_class_idx,
                            CONSTRUCTING_CLASS_GLOBAL_NAME.to_string(),
                        );
                        ctx.chunk
                            .write_with_line(OpCode::LoadGlobal(class_global_index), *line);
                        ctx.chunk
                            .write_with_line(OpCode::StoreGlobal(constructing_class_idx), *line);
                    }
                }
                // Settings subclass with 1 arg: expand to (path, required_keys, model_config) and Call(3).
                if arg_count == 1 && is_in_settings_chain(name, ctx.class_superclass) {
                    let required_keys_value = ctx.class_required_keys_value.get(name).cloned();
                    if let Some(required_keys_value) = required_keys_value {
                        match &call_args[0] {
                            Arg::Positional(e) => expr::compile_expr(ctx, e)?,
                            Arg::Named { value, .. } => expr::compile_expr(ctx, value)?,
                            Arg::UnpackObject(e) | Arg::UnpackArray(e) => expr::compile_expr(ctx, e)?,
                        }
                        let req_const = ctx.chunk.add_constant(required_keys_value);
                        ctx.chunk
                            .write_with_line(OpCode::Constant(req_const), *line);
                        let class_global_index = *ctx
                            .scope
                            .globals
                            .get(name)
                            .expect("Settings class in globals");
                        ctx.chunk
                            .global_names
                            .insert(class_global_index, name.clone());
                        ctx.chunk
                            .write_with_line(OpCode::LoadGlobal(class_global_index), *line);
                        let model_config_key = ctx
                            .chunk
                            .add_constant(Value::String("model_config".to_string()));
                        ctx.chunk
                            .write_with_line(OpCode::Constant(model_config_key), *line);
                        ctx.chunk.write_with_line(OpCode::GetArrayElement, *line);
                        let null_const = ctx.chunk.add_constant(Value::Null);
                        ctx.chunk
                            .write_with_line(OpCode::Constant(null_const), *line);
                        ctx.chunk
                            .global_names
                            .insert(global_index, constructor_name.clone());
                        ctx.chunk
                            .write_with_line(OpCode::LoadGlobal(global_index), *line);
                        ctx.chunk.write_with_line(OpCode::Call(4), *line);
                        return Ok(());
                    }
                }

                for (i, arg) in call_args.iter().enumerate() {
                    match arg {
                        Arg::Positional(expr) => {
                            debug_println!("[DEBUG compile_call] Компилируем позиционный аргумент {} из {} (globals)", i + 1, arg_count);
                            expr::compile_expr(ctx, expr)?;
                        }
                        Arg::Named { value, .. } => {
                            debug_println!("[DEBUG compile_call] Компилируем именованный аргумент {} из {} (globals)", i + 1, arg_count);
                            expr::compile_expr(ctx, value)?;
                        }
                        Arg::UnpackObject(expr) | Arg::UnpackArray(expr) => {
                            debug_println!(
                                "[DEBUG compile_call] Компилируем ** аргумент {} из {} (globals)",
                                i + 1,
                                arg_count
                            );
                            expr::compile_expr(ctx, expr)?;
                        }
                    }
                }

                if ctx.abstract_classes.contains(name) {
                    let class_global_index = *ctx.scope.globals.get(name).unwrap();
                    ctx.chunk
                        .write_with_line(OpCode::LoadGlobal(class_global_index), *line);
                    ctx.chunk.write_with_line(OpCode::Call(arg_count), *line);
                    return Ok(());
                }
                // Нужно для update_chunk_indices при merge модуля (патч LoadGlobal по имени).
                ctx.chunk
                    .global_names
                    .insert(global_index, constructor_name.clone());
                let load_global_ip = ctx.chunk.code.len();
                ctx.chunk
                    .write_with_line(OpCode::LoadGlobal(global_index), *line);
                let call_ip = ctx.chunk.code.len();
                ctx.chunk.write_with_line(OpCode::Call(arg_count), *line);
                debug_println!("[DEBUG compile_call] Сгенерирован OpCode::LoadGlobal({}) на IP {}, OpCode::Call({}) на IP {} для конструктора '{}' (globals)", 
                global_index, load_global_ip, arg_count, call_ip, constructor_name);

                if let Some(OpCode::Call(recorded_arity)) = ctx.chunk.code.get(call_ip) {
                    if *recorded_arity != arg_count {
                        debug_println!("[ERROR compile_call] КРИТИЧЕСКАЯ ОШИБКА: Записано Call({}), но ожидалось Call({})!", recorded_arity, arg_count);
                    }
                }

                return Ok(());
            }

            // Если конструктор не найден в globals, но класс найден, генерируем код для проверки во время выполнения.
            // Это нужно для конструкторов, импортированных из модулей через __lib__.dc.
            // Не входим сюда для обычной функции с заглавной буквы (например Base) — только для классов.
            let has_constructor = ctx.scope.globals.contains_key(&constructor_name)
                || ctx.function_names.iter().any(|n| n == &constructor_name);
            if ctx.scope.globals.contains_key(name)
                && has_constructor
                && !is_builtin_pascal_native_callable(name)
            {
                debug_println!("[DEBUG compile_call] Класс '{}' найден в globals, генерируем код для проверки конструктора во время выполнения", name);
                let arg_count = call_args.len();
                debug_println!("[DEBUG compile_call] Сохранено количество аргументов: {} для конструктора '{}' (класс в globals)", arg_count, constructor_name);
                // 0-arg call: set __constructing_class__ = class so Settings subclass new_0 can load required_keys/model_config (class_superclass may be empty when class is imported).
                if arg_count == 0 {
                    if let Some(&class_global_index) = ctx.scope.globals.get(name) {
                        let new_idx = ctx.scope.globals.len();
                        let constructing_class_idx = *ctx
                            .scope
                            .globals
                            .entry(CONSTRUCTING_CLASS_GLOBAL_NAME.to_string())
                            .or_insert(new_idx);
                        ctx.chunk.global_names.insert(
                            constructing_class_idx,
                            CONSTRUCTING_CLASS_GLOBAL_NAME.to_string(),
                        );
                        ctx.chunk
                            .write_with_line(OpCode::LoadGlobal(class_global_index), *line);
                        ctx.chunk
                            .write_with_line(OpCode::StoreGlobal(constructing_class_idx), *line);
                    }
                }
                for (i, arg) in call_args.iter().enumerate() {
                    match arg {
                        Arg::Positional(expr) => {
                            debug_println!("[DEBUG compile_call] Компилируем позиционный аргумент {} из {} (класс в globals)", i + 1, arg_count);
                            expr::compile_expr(ctx, expr)?;
                        }
                        Arg::Named { value, .. } => {
                            debug_println!("[DEBUG compile_call] Компилируем именованный аргумент {} из {} (класс в globals)", i + 1, arg_count);
                            expr::compile_expr(ctx, value)?;
                        }
                        Arg::UnpackObject(expr) | Arg::UnpackArray(expr) => {
                            debug_println!("[DEBUG compile_call] Компилируем ** аргумент {} из {} (класс в globals)", i + 1, arg_count);
                            expr::compile_expr(ctx, expr)?;
                        }
                    }
                }

                if ctx.abstract_classes.contains(name) {
                    let class_global_index = *ctx.scope.globals.get(name).unwrap();
                    ctx.chunk
                        .write_with_line(OpCode::LoadGlobal(class_global_index), *line);
                    ctx.chunk.write_with_line(OpCode::Call(arg_count), *line);
                    return Ok(());
                }

                let constructor_global_index =
                    if let Some(&idx) = ctx.scope.globals.get(&constructor_name) {
                        idx
                    } else {
                        let idx = ctx.scope.globals.len();
                        ctx.scope.globals.insert(constructor_name.clone(), idx);
                        idx
                    };
                ctx.chunk
                    .global_names
                    .insert(constructor_global_index, constructor_name.clone());
                let load_global_ip = ctx.chunk.code.len();
                ctx.chunk
                    .write_with_line(OpCode::LoadGlobal(constructor_global_index), *line);
                let call_ip = ctx.chunk.code.len();
                ctx.chunk.write_with_line(OpCode::Call(arg_count), *line);
                debug_println!("[DEBUG compile_call] Сгенерирован OpCode::LoadGlobal({}) на IP {}, OpCode::Call({}) на IP {} для конструктора '{}' (класс в globals)", 
                constructor_global_index, load_global_ip, arg_count, call_ip, constructor_name);

                if let Some(OpCode::Call(recorded_arity)) = ctx.chunk.code.get(call_ip) {
                    if *recorded_arity != arg_count {
                        debug_println!("[ERROR compile_call] КРИТИЧЕСКАЯ ОШИБКА: Записано Call({}), но ожидалось Call({})!", recorded_arity, arg_count);
                    }
                }

                return Ok(());
            }

            // Класс найден, но конструктор отсутствует — возможно, неявный конструктор пропущен
            // (суперкласс не принимает переданное число аргументов). Ошибка на строке вызова.
            if ctx.scope.globals.contains_key(name)
                && !ctx.scope.globals.contains_key(&constructor_name)
                && !ctx.function_names.iter().any(|n| n == &constructor_name)
            {
                // Только для пропущенного неявного конструктора наследника — compile-time ошибка.
                // Прямые вызовы с неверной арностью (Parent(20)) дадут runtime-ошибку, чтобы try-catch сработал.
                if let Some(superclass_name) = ctx.class_superclass.get(name) {
                    return Err(LangError::ParseError {
                        message: format!(
                            "Class '{}' cannot accept {} argument(s)",
                            superclass_name,
                            call_args.len()
                        ),
                        line: *line,
                        file: None,
                    });
                }
            }

            // Если класс не найден, но имя начинается с заглавной буквы (характерно для классов),
            // это может быть импортированный класс из __lib__.dc или другого модуля.
            // Не входим сюда, если имя уже в scope (например обычная функция Base) — тогда обрабатываем как вызов функции ниже.
            if name
                .chars()
                .next()
                .map(|c| c.is_uppercase())
                .unwrap_or(false)
                && !ctx.scope.globals.contains_key(name)
            {
                debug_println!("[DEBUG compile_call] Класс '{}' не найден в globals, но имя начинается с заглавной буквы - предполагаем, что это класс, импортированный из модуля", name);

                // Регистрируем класс в scope, если его там еще нет
                if !ctx.scope.globals.contains_key(name) {
                    let class_global_index = ctx.scope.globals.len();
                    ctx.scope.globals.insert(name.clone(), class_global_index);
                    // ВАЖНО: Не добавляем класс в chunk.global_names здесь, чтобы избежать конфликта с конструктором
                    // Класс будет добавлен в chunk.global_names только если он используется напрямую
                    debug_println!("[DEBUG compile_call] Зарегистрирован класс '{}' в globals с индексом {} (не добавляем в chunk.global_names)", name, class_global_index);
                }

                // Регистрируем конструктор в scope
                let constructor_global_index = if let Some(&idx) =
                    ctx.scope.globals.get(&constructor_name)
                {
                    idx
                } else {
                    let idx = ctx.scope.globals.len();
                    ctx.scope.globals.insert(constructor_name.clone(), idx);
                    // ВАЖНО: Добавляем конструктор в chunk.global_names с правильным именем
                    ctx.chunk.global_names.insert(idx, constructor_name.clone());
                    debug_println!("[DEBUG compile_call] Зарегистрирован конструктор '{}' в globals с индексом {} и в chunk.global_names", constructor_name, idx);
                    idx
                };

                // Всегда записываем имя конструктора в chunk для update_chunk_indices при merge (безусловно).
                ctx.chunk
                    .global_names
                    .insert(constructor_global_index, constructor_name.clone());

                // Сохраняем количество аргументов до компиляции (на случай, если call_args будет перемещено)
                let arg_count = call_args.len();
                debug_println!("[DEBUG compile_call] Сохранено количество аргументов: {} для конструктора '{}'", arg_count, constructor_name);

                // Компилируем аргументы ПЕРЕД загрузкой конструктора
                let args_start_ip = ctx.chunk.code.len();
                debug_println!(
                    "[DEBUG compile_call] Начало компиляции аргументов, IP: {}",
                    args_start_ip
                );
                for (i, arg) in call_args.iter().enumerate() {
                    match arg {
                        Arg::Positional(expr) => {
                            debug_println!(
                                "[DEBUG compile_call] Компилируем позиционный аргумент {} из {}",
                                i + 1,
                                arg_count
                            );
                            expr::compile_expr(ctx, expr)?;
                        }
                        Arg::Named { value, .. } => {
                            debug_println!(
                                "[DEBUG compile_call] Компилируем именованный аргумент {} из {}",
                                i + 1,
                                arg_count
                            );
                            expr::compile_expr(ctx, value)?;
                        }
                        Arg::UnpackObject(expr) | Arg::UnpackArray(expr) => {
                            debug_println!(
                                "[DEBUG compile_call] Компилируем ** аргумент {} из {}",
                                i + 1,
                                arg_count
                            );
                            expr::compile_expr(ctx, expr)?;
                        }
                    }
                }
                let args_end_ip = ctx.chunk.code.len();
                debug_println!("[DEBUG compile_call] Конец компиляции аргументов, IP: {}, сгенерировано инструкций: {}", args_end_ip, args_end_ip - args_start_ip);

                // Генерируем код для загрузки конструктора из глобальных переменных во время выполнения
                debug_println!(
                    "[DEBUG compile_call] Компилируем вызов конструктора '{}' с {} аргументами",
                    constructor_name,
                    arg_count
                );
                let load_global_ip = ctx.chunk.code.len();
                ctx.chunk
                    .write_with_line(OpCode::LoadGlobal(constructor_global_index), *line);
                let call_ip = ctx.chunk.code.len();
                ctx.chunk.write_with_line(OpCode::Call(arg_count), *line);
                debug_println!("[DEBUG compile_call] Сгенерирован OpCode::LoadGlobal({}) на IP {}, OpCode::Call({}) на IP {} для конструктора '{}'", 
                constructor_global_index, load_global_ip, arg_count, call_ip, constructor_name);

                // Проверяем, что инструкция действительно записана правильно
                if let Some(OpCode::Call(recorded_arity)) = ctx.chunk.code.get(call_ip) {
                    if *recorded_arity != arg_count {
                        debug_println!("[ERROR compile_call] КРИТИЧЕСКАЯ ОШИБКА: Записано Call({}), но ожидалось Call({})!", recorded_arity, arg_count);
                        // Показываем окружающие инструкции для отладки
                        let start = call_ip.saturating_sub(5);
                        let end = (call_ip + 5).min(ctx.chunk.code.len());
                        debug_println!(
                            "[ERROR compile_call] Окружающие инструкции (IP {} - {}):",
                            start,
                            end
                        );
                        for i in start..end {
                            let marker = if i == call_ip {
                                " <-- ТЕКУЩАЯ"
                            } else {
                                ""
                            };
                            debug_println!(
                                "[ERROR compile_call]   IP {}: {:?}{}",
                                i,
                                ctx.chunk.code.get(i),
                                marker
                            );
                        }
                    } else {
                        debug_println!("[DEBUG compile_call] Подтверждено: Call({}) записан правильно на IP {}", arg_count, call_ip);
                    }
                } else {
                    debug_println!("[ERROR compile_call] КРИТИЧЕСКАЯ ОШИБКА: На IP {} не найдена инструкция Call!", call_ip);
                    // Показываем, что там на самом деле
                    if let Some(opcode) = ctx.chunk.code.get(call_ip) {
                        debug_println!(
                            "[ERROR compile_call] На IP {} найдена инструкция: {:?}",
                            call_ip,
                            opcode
                        );
                    }
                }

                // Дополнительная проверка: убеждаемся, что Call инструкция не была перезаписана
                // Проверяем еще раз после небольшой задержки (если есть другие операции)
                let final_check_ip = ctx.chunk.code.len() - 1;
                if final_check_ip == call_ip {
                    if let Some(OpCode::Call(final_arity)) = ctx.chunk.code.get(final_check_ip) {
                        if *final_arity != arg_count {
                            debug_println!("[ERROR compile_call] КРИТИЧЕСКАЯ ОШИБКА: Call инструкция была изменена! Ожидалось Call({}), но найдено Call({}) на IP {}", 
                            arg_count, final_arity, final_check_ip);
                        }
                    }
                }

                return Ok(());
            }

            // Импортированный символ с заглавной буквы (from X import Config): не fallback в обычную функцию,
            // а вызов конструктора — слот Config::new_N заполнится при выполнении ImportFrom.
            // Только для файловых модулей: встроенные (settings_env, plot, uuid) не экспортируют конструкторы в globals так же, оставляем старый путь (LoadGlobal(name)+Call).
            fn is_builtin_module(name: &str) -> bool {
                matches!(
                    name,
                    "plot" | "settings_env" | "uuid" | "database_engine" | "system"
                )
            }
            // Builtin PascalCase natives (database_engine, settings_env) are not class constructors.
            // When re-exported from file modules, treat as direct call, not constructor.
            fn is_native_callable_uppercase(name: &str) -> bool {
                is_builtin_pascal_native_callable(name)
            }
            if name
                .chars()
                .next()
                .map(|c| c.is_uppercase())
                .unwrap_or(false)
                && !is_native_callable_uppercase(name)
                && ctx
                    .imported_symbols
                    .get(name)
                    .map_or(false, |m| !is_builtin_module(m))
            {
                // Регистрируем класс в scope, если его там еще нет (уже есть от import, но индекс нужен для консистентности)
                if !ctx.scope.globals.contains_key(name) {
                    let class_global_index = ctx.scope.globals.len();
                    ctx.scope.globals.insert(name.clone(), class_global_index);
                }
                // Регистрируем конструктор в scope и chunk.global_names (обязательно для update_chunk_indices при merge).
                let constructor_global_index =
                    if let Some(&idx) = ctx.scope.globals.get(&constructor_name) {
                        idx
                    } else {
                        let idx = ctx.scope.globals.len();
                        ctx.scope.globals.insert(constructor_name.clone(), idx);
                        idx
                    };
                ctx.chunk
                    .global_names
                    .insert(constructor_global_index, constructor_name.clone());
                let arg_count = call_args.len();
                // 0-arg call: set __constructing_class__ = class so Settings subclass new_0 can load required_keys/model_config.
                if arg_count == 0 {
                    if let Some(&class_global_index) = ctx.scope.globals.get(name) {
                        ctx.chunk
                            .global_names
                            .insert(class_global_index, name.clone());
                        let new_idx = ctx.scope.globals.len();
                        let constructing_class_idx = *ctx
                            .scope
                            .globals
                            .entry(CONSTRUCTING_CLASS_GLOBAL_NAME.to_string())
                            .or_insert(new_idx);
                        ctx.chunk.global_names.insert(
                            constructing_class_idx,
                            CONSTRUCTING_CLASS_GLOBAL_NAME.to_string(),
                        );
                        ctx.chunk
                            .write_with_line(OpCode::LoadGlobal(class_global_index), *line);
                        ctx.chunk
                            .write_with_line(OpCode::StoreGlobal(constructing_class_idx), *line);
                    }
                }
                for arg in call_args.iter() {
                    match arg {
                        Arg::Positional(expr) => expr::compile_expr(ctx, expr)?,
                        Arg::Named { value, .. } => expr::compile_expr(ctx, value)?,
                        Arg::UnpackObject(expr) | Arg::UnpackArray(expr) => expr::compile_expr(ctx, expr)?,
                    }
                }
                ctx.chunk
                    .write_with_line(OpCode::LoadGlobal(constructor_global_index), *line);
                ctx.chunk.write_with_line(OpCode::Call(arg_count), *line);
                return Ok(());
            }

            debug_println!("[DEBUG compile_call] Конструктор '{}' не найден ни в function_names, ни в globals, класс '{}' тоже не найден, проверяем как обычную функцию", constructor_name, name);
        }

        // If constructor not found by arity but we have named args: resolve via class_constructor (extends_table field-based constructor)
        if is_class_style_ctor {
            let has_named = call_args
                .iter()
                .any(|a| matches!(a, Arg::Named { .. } | Arg::UnpackObject(_) | Arg::UnpackArray(_) | Arg::UnpackArray(_)));
            if has_named {
                let ctor_info = ctx
                    .class_constructor
                    .get(name)
                    .map(|(c, i)| (c.clone(), *i));
                if let Some((ctor_name, function_index)) = ctor_info {
                    let function = &ctx.functions[function_index];
                    let function_info = (function_index, function);
                    let resolved_args = args::resolve_function_args(
                        name,
                        call_args,
                        Some(function_info),
                        *line,
                        ctx.source_name,
                        None,
                        None,
                    )?;
                    let arity = resolved_args.len();
                    for arg in &resolved_args {
                        match arg {
                            Arg::Positional(expr) => expr::compile_expr(ctx, expr)?,
                            Arg::Named { value, .. } => expr::compile_expr(ctx, value)?,
                            Arg::UnpackObject(expr) | Arg::UnpackArray(expr) => expr::compile_expr(ctx, expr)?,
                        }
                    }
                    if let Some(&global_index) = ctx.scope.globals.get(&ctor_name) {
                        ctx.chunk
                            .global_names
                            .insert(global_index, ctor_name.clone());
                        ctx.chunk
                            .write_with_line(OpCode::LoadGlobal(global_index), *line);
                        ctx.chunk.write_with_line(OpCode::Call(arity), *line);
                        return Ok(());
                    }
                    let constant_index = ctx.chunk.add_constant(Value::Function(function_index));
                    ctx.chunk
                        .write_with_line(OpCode::Constant(constant_index), *line);
                    ctx.chunk.write_with_line(OpCode::Call(arity), *line);
                    return Ok(());
                }
            }
        }

        // Обработка обычной функции (для функций, начинающихся с маленькой буквы, или если конструктор не найден)
        // Находим функцию для получения информации о параметрах. Сначала проверяем function_names,
        // чтобы пользовательские функции с default-параметрами получали подстановку аргументов.
        let function_info = if let Some(info) = function::user_function_info_for_call(ctx, name) {
            debug_println!(
                "[DEBUG compile_call] Найдена функция '{}' с индексом {}",
                name,
                info.0
            );
            Some(info)
        } else if !is_class_style_ctor && ctx.scope.globals.contains_key(name) {
            // Builtin — arity разрешается во время выполнения
            None
        } else {
            if is_class_style_ctor {
                debug_println!(
                    "[DEBUG compile_call] WARNING: Функция '{}' не найдена в function_names",
                    name
                );
                debug_println!(
                    "[DEBUG compile_call] Доступные функции: {:?}",
                    ctx.function_names.iter().take(20).collect::<Vec<_>>()
                );
            }
            None
        };

        // Разрешаем аргументы: именованные -> позиционные, применяем значения по умолчанию
        let imported_from = ctx.imported_symbols.get(name).map(|s| s.as_str());
        let resolved_args = args::resolve_function_args(
            name,
            call_args,
            function_info,
            *line,
            ctx.source_name,
            imported_from,
            None,
        )?;

        // Специальная обработка для isinstance: преобразуем идентификаторы типов в строки
        let processed_args = if name == "isinstance" && resolved_args.len() >= 2 {
            let mut new_args = resolved_args.clone();
            let type_arg_line = match &resolved_args[1] {
                Arg::Positional(e) => e.line(),
                Arg::Named { value, .. } => value.line(),
                Arg::UnpackObject(e) | Arg::UnpackArray(e) => e.line(),
            };
            let type_name_opt = match &resolved_args[1] {
                Arg::Positional(Expr::Variable { name, .. }) => Some(name.as_str()),
                _ => None,
            };
            if let Some(type_name) = type_name_opt {
                let is_type_literal = matches!(
                    type_name,
                    "int" | "integer" | "str" | "string" | "bool" | "boolean" | "array" | "list"
                        | "bytes" | "null" | "none" | "num" | "number" | "float" | "table"
                        | "tuple" | "set" | "object" | "dict" | "dictionary" | "path" | "uuid"
                        | "function" | "date" | "money" | "duration" | "enumerate"
                        | "iterable" | "generator" | "ellipsis" | "column" | "window" | "image"
                        | "figure" | "axis" | "plugin_opaque"
                ) || type_name == "Table"
                    || ctx.scope.globals.get(type_name).is_some_and(|&idx| {
                        idx < crate::vm::globals::BUILTIN_GLOBAL_COUNT
                            && matches!(
                                crate::vm::globals::builtin_global_name(idx),
                                Some(
                                    "int" | "float" | "bool" | "str" | "array" | "set" | "tuple"
                                        | "table" | "Table" | "path" | "date" | "money"
                                        | "duration" | "typeof" | "enum"
                                )
                            )
                    });
                if is_type_literal {
                    new_args[1] = Arg::Positional(Expr::Literal {
                        value: Value::String(type_name.to_string()),
                        line: type_arg_line,
                    });
                }
            }
            new_args
        } else {
            resolved_args
        };

        // reverse/sort: assign return value back so the slot tracks the (possibly new) array id.
        // push: do NOT assign back — native_call update_cell_if_mutable already syncs arg0's cell;
        // storing store_value(return) would replace the slot with a new ValueId and break aliases
        // (e.g. let b = a; push(a, x) would leave b on the old id while a gets a new id → len 4+5=9).
        let in_place_assign_back = ["reverse", "sort"];
        let should_assign_back = in_place_assign_back.contains(&name.as_str())
            && !processed_args.is_empty()
            && matches!(&processed_args[0], Arg::Positional(Expr::Variable { .. }));

        // Если нужно присвоить обратно, сохраняем имя переменной
        let var_name_to_assign = if should_assign_back {
            if let Arg::Positional(Expr::Variable { name: var_name, .. }) = &processed_args[0] {
                Some(var_name.clone())
            } else {
                None
            }
        } else {
            None
        };

        // Вызов с единственным **obj: эмитим CallWithUnpack(1), ключи объекта проверяются в VM.
        let is_single_unpack =
            processed_args.len() == 1 && matches!(&processed_args[0], Arg::UnpackObject(_));

        let user_has_variadic = function::user_function_info_for_call(ctx, name)
            .map(|(_, f)| f.variadic_pos_index.is_some() || f.variadic_kw_index.is_some())
            .unwrap_or(false);
        let needs_variadic_opcode = crate::compiler::natives::call_needs_variadic_opcode(&processed_args)
            || user_has_variadic;

        if needs_variadic_opcode {
            let mut pos_exprs: Vec<&Expr> = Vec::new();
            let mut star_exprs: Vec<&Expr> = Vec::new();
            let mut named_pairs: Vec<(&String, &Expr)> = Vec::new();
            let mut starstar_exprs: Vec<&Expr> = Vec::new();
            for arg in &processed_args {
                match arg {
                    Arg::Positional(expr) => pos_exprs.push(expr),
                    Arg::UnpackArray(expr) => star_exprs.push(expr),
                    Arg::Named { name, value } => named_pairs.push((name, value)),
                    Arg::UnpackObject(expr) => starstar_exprs.push(expr),
                }
            }
            let n_pos = pos_exprs.len();
            let n_star = star_exprs.len();
            let n_named = named_pairs.len();
            let n_starstar = starstar_exprs.len();
            // Stack layout (bottom → top): positional, *spread, named (key, val)…, **spread, callee
            for expr in &pos_exprs {
                expr::compile_expr(ctx, expr)?;
            }
            for expr in &star_exprs {
                expr::compile_expr(ctx, expr)?;
            }
            for (kw, value) in &named_pairs {
                let key_idx = ctx.chunk.add_constant(Value::String((*kw).clone()));
                ctx.chunk.write_with_line(OpCode::Constant(key_idx), *line);
                expr::compile_expr(ctx, value)?;
            }
            for expr in &starstar_exprs {
                expr::compile_expr(ctx, expr)?;
            }
            // load callee (same as below)
            if let Some(local_index) = ctx.scope.resolve_local(name) {
                ctx.chunk
                    .write_with_line(OpCode::LoadLocal(local_index), *ctx.current_line);
            } else if ctx.function_names.iter().any(|n| n == name) {
                if let Some(&global_index) = ctx.scope.globals.get(name) {
                    ctx.chunk.global_names.insert(global_index, name.clone());
                    ctx.chunk
                        .write_with_line(OpCode::LoadGlobal(global_index), *ctx.current_line);
                } else {
                    let function_index = ctx.function_names.iter().position(|n| n == name).unwrap();
                    let constant_index = ctx.chunk.add_constant(Value::Function(function_index));
                    ctx.chunk
                        .write_with_line(OpCode::Constant(constant_index), *ctx.current_line);
                }
            } else if !is_class_style_ctor && ctx.scope.globals.contains_key(name) {
                let &global_index = ctx.scope.globals.get(name).unwrap();
                ctx.chunk.global_names.insert(global_index, name.clone());
                ctx.chunk
                    .write_with_line(OpCode::LoadGlobal(global_index), *ctx.current_line);
            } else if let Some(&global_index) = ctx.scope.globals.get(name) {
                ctx.chunk.global_names.insert(global_index, name.clone());
                ctx.chunk
                    .write_with_line(OpCode::LoadGlobal(global_index), *ctx.current_line);
            } else {
                return Err(LangError::ParseError {
                    message: format!("Undefined function: {}", name),
                    line: *line,
                    file: None,
                });
            }
            let packed = crate::compiler::natives::pack_call_variadic_operand(
                n_pos, n_star, n_named, n_starstar,
            );
            ctx.chunk
                .write_with_line(OpCode::CallVariadic(packed), *line);
        } else {
        // Компилируем аргументы на стек
        for arg in &processed_args {
            match arg {
                Arg::Positional(expr) => {
                    expr::compile_expr(ctx, expr)?;
                }
                Arg::Named { value, .. } => {
                    expr::compile_expr(ctx, value)?;
                }
                Arg::UnpackObject(expr) | Arg::UnpackArray(expr) => {
                    expr::compile_expr(ctx, expr)?;
                }
            }
        }

        // Загружаем функцию: локальные → function_names (user-defined) → globals (builtins) → globals / новый слот
        if let Some(local_index) = ctx.scope.resolve_local(name) {
            // Локальная переменная содержит функцию
            ctx.chunk
                .write_with_line(OpCode::LoadLocal(local_index), *ctx.current_line);
        } else if ctx.function_names.iter().any(|n| n == name) {
            // Пользовательская функция найдена. Если у неё есть глобальный слот (main, __main__ и т.д.),
            // используем LoadGlobal, чтобы брать значение из слота, установленного set_functions, и не
            // полагаться на константный пул (избегаем путаницы с argv или другими глобалами).
            if let Some(&global_index) = ctx.scope.globals.get(name) {
                ctx.chunk.global_names.insert(global_index, name.clone());
                ctx.chunk
                    .write_with_line(OpCode::LoadGlobal(global_index), *ctx.current_line);
            } else {
                let function_index = ctx.function_names.iter().position(|n| n == name).unwrap();
                let constant_index = ctx.chunk.add_constant(Value::Function(function_index));
                ctx.chunk
                    .write_with_line(OpCode::Constant(constant_index), *ctx.current_line);
            }
        } else if !is_class_style_ctor && ctx.scope.globals.contains_key(name) {
            // Для имён не PascalCase без пользовательского переопределения — встроенные
            let &global_index = ctx.scope.globals.get(name).unwrap();
            ctx.chunk.global_names.insert(global_index, name.clone());
            ctx.chunk
                .write_with_line(OpCode::LoadGlobal(global_index), *ctx.current_line);
        } else if let Some(&global_index) = ctx.scope.globals.get(name) {
            // Глобальная переменная содержит функцию (встроенная или импорт)
            ctx.chunk.global_names.insert(global_index, name.clone());
            ctx.chunk
                .write_with_line(OpCode::LoadGlobal(global_index), *ctx.current_line);
        } else {
            // Функция не найдена — ошибка на этапе компиляции
            return Err(LangError::ParseError {
                message: format!("Undefined function: {}", name),
                line: *line,
                file: None,
            });
        }

        // Вызываем функцию: при единственном **obj — CallWithUnpack(1), иначе Call(n)
        if is_single_unpack {
            ctx.chunk.write_with_line(OpCode::CallWithUnpack(1), *line);
        } else {
            ctx.chunk
                .write_with_line(OpCode::Call(processed_args.len()), *line);
        }
        }

        // Если нужно присвоить результат обратно в переменную
        if let Some(var_name) = var_name_to_assign {
            // Определяем, глобальная или локальная переменная
            let is_local =
                !ctx.scope.locals.is_empty() && ctx.scope.resolve_local(&var_name).is_some();

            if is_local {
                // Локальная переменная
                if let Some(local_index) = ctx.scope.resolve_local(&var_name) {
                    ctx.chunk
                        .write_with_line(OpCode::StoreLocal(local_index), *line);
                    ctx.chunk
                        .write_with_line(OpCode::LoadLocal(local_index), *line);
                }
            } else {
                // Глобальная переменная
                let global_index = if let Some(&idx) = ctx.scope.globals.get(&var_name) {
                    idx
                } else {
                    let idx = ctx.scope.globals.len();
                    ctx.scope.globals.insert(var_name.clone(), idx);
                    idx
                };
                ctx.chunk
                    .global_names
                    .insert(global_index, var_name.clone());
                ctx.chunk
                    .write_with_line(OpCode::StoreGlobal(global_index), *line);
                ctx.chunk
                    .write_with_line(OpCode::LoadGlobal(global_index), *line);
            }
        }

        Ok(())
    } else {
        Err(LangError::ParseError {
            message: "Expected Call expression".to_string(),
            line: expr.line(),
            file: None,
        })
    }
}
