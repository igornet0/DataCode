//! Compiler fusion: `array(map(labels, fn(x) => one_hot(x, K)[0]))` → `onehots(tensor(labels), K)`.
//! `from ml import …` is amended in-memory (adds `onehots`) so the source file need not change.

use crate::parser::ast::{Arg, Expr, ImportItem, ImportStmt, IndexExpr, Stmt};

/// `array(map(collection, fn(x) => one_hot(x, num_classes_expr)[0]))`
pub fn try_match_array_map_onehot_fusion(expr: &Expr) -> Option<(Expr, Expr)> {
    let Expr::Call {
        name: array_name,
        args: array_args,
        ..
    } = expr
    else {
        return None;
    };
    if array_name != "array" || array_args.len() != 1 {
        return None;
    }
    let Arg::Positional(map_expr) = &array_args[0] else {
        return None;
    };
    let Expr::Call {
        name: map_name,
        args: map_args,
        ..
    } = map_expr
    else {
        return None;
    };
    if map_name != "map" || map_args.len() != 2 {
        return None;
    }
    let Arg::Positional(labels_expr) = &map_args[0] else {
        return None;
    };
    let Arg::Positional(lambda_expr) = &map_args[1] else {
        return None;
    };
    let Expr::Lambda { params, body, .. } = lambda_expr else {
        return None;
    };
    if params.len() != 1 {
        return None;
    }
    let param_name = &params[0].name;

    let Expr::ArrayIndex {
        array: inner,
        index,
        ..
    } = body.as_ref()
    else {
        return None;
    };
    let IndexExpr::Scalar(idx_expr) = index else {
        return None;
    };
    match idx_expr.as_ref() {
        Expr::Literal {
            value: crate::common::value::Value::Number(n),
            ..
        } if *n == 0.0 => {}
        _ => return None,
    }

    let Expr::Call {
        name: oh_name,
        args: oh_args,
        ..
    } = inner.as_ref()
    else {
        return None;
    };
    if oh_name != "one_hot" || oh_args.len() != 2 {
        return None;
    }
    let Arg::Positional(class_arg) = &oh_args[0] else {
        return None;
    };
    let Arg::Positional(num_classes_arg) = &oh_args[1] else {
        return None;
    };
    let Expr::Variable { name: vname, .. } = class_arg else {
        return None;
    };
    if vname != param_name {
        return None;
    }

    Some((labels_expr.clone(), num_classes_arg.clone()))
}

fn expr_needs_array_map_onehot_fusion(expr: &Expr) -> bool {
    try_match_array_map_onehot_fusion(expr).is_some()
}

fn walk_expr(expr: &Expr, f: &mut impl FnMut(&Expr)) {
    f(expr);
    match expr {
        Expr::Literal { .. }
        | Expr::Variable { .. }
        | Expr::Ellipsis { .. }
        | Expr::This { .. }
        | Expr::Super { .. } => {}
        Expr::Assign { value, .. } => walk_expr(value, f),
        Expr::AssignOp { value, .. } => walk_expr(value, f),
        Expr::UnpackAssign { value, .. } => walk_expr(value, f),
        Expr::Binary { left, right, .. } => {
            walk_expr(left, f);
            walk_expr(right, f);
        }
        Expr::Unary { right, .. } => walk_expr(right, f),
        Expr::Call { args, .. } => {
            for a in args {
                match a {
                    Arg::Positional(e) => walk_expr(e, f),
                    Arg::Named { value, .. } => walk_expr(value, f),
                    Arg::UnpackObject(e) => walk_expr(e, f),
                }
            }
        }
        Expr::CallValue { callee, args, .. } => {
            walk_expr(callee, f);
            for a in args {
                match a {
                    Arg::Positional(e) => walk_expr(e, f),
                    Arg::Named { value, .. } => walk_expr(value, f),
                    Arg::UnpackObject(e) => walk_expr(e, f),
                }
            }
        }
        Expr::Lambda { body, .. } => walk_expr(body, f),
        Expr::ArrayLiteral { elements, .. } => {
            for e in elements {
                walk_expr(e, f);
            }
        }
        Expr::ObjectLiteral { pairs, .. } => {
            for p in pairs {
                if let crate::parser::ast::ObjectPair::KeyValue(_, e) = p {
                    walk_expr(e, f);
                } else if let crate::parser::ast::ObjectPair::Spread(e) = p {
                    walk_expr(e, f);
                }
            }
        }
        Expr::TupleLiteral { elements, .. } => {
            for e in elements {
                walk_expr(e, f);
            }
        }
        Expr::ArrayIndex { array, index, .. } => {
            walk_expr(array, f);
            match index {
                IndexExpr::Scalar(e) => walk_expr(e, f),
                IndexExpr::Slice {
                    start, stop, step, ..
                } => {
                    if let Some(e) = start {
                        walk_expr(e, f);
                    }
                    if let Some(e) = stop {
                        walk_expr(e, f);
                    }
                    if let Some(e) = step {
                        walk_expr(e, f);
                    }
                }
            }
        }
        Expr::AssignArray {
            array,
            index,
            value,
            ..
        } => {
            walk_expr(array, f);
            match index {
                IndexExpr::Scalar(e) => walk_expr(e, f),
                IndexExpr::Slice {
                    start, stop, step, ..
                } => {
                    if let Some(e) = start {
                        walk_expr(e, f);
                    }
                    if let Some(e) = stop {
                        walk_expr(e, f);
                    }
                    if let Some(e) = step {
                        walk_expr(e, f);
                    }
                }
            }
            walk_expr(value, f);
        }
        Expr::AssignArrayOp {
            array,
            index,
            value,
            ..
        } => {
            walk_expr(array, f);
            match index {
                IndexExpr::Scalar(e) => walk_expr(e, f),
                IndexExpr::Slice {
                    start, stop, step, ..
                } => {
                    if let Some(e) = start {
                        walk_expr(e, f);
                    }
                    if let Some(e) = stop {
                        walk_expr(e, f);
                    }
                    if let Some(e) = step {
                        walk_expr(e, f);
                    }
                }
            }
            walk_expr(value, f);
        }
        Expr::TableFilter { .. } => {}
        Expr::Property { object, .. } => walk_expr(object, f),
        Expr::MethodCall { object, args, .. } => {
            walk_expr(object, f);
            for a in args {
                match a {
                    Arg::Positional(e) => walk_expr(e, f),
                    Arg::Named { value, .. } => walk_expr(value, f),
                    Arg::UnpackObject(e) => walk_expr(e, f),
                }
            }
        }
        Expr::SuperCall { args, .. } | Expr::SuperMethodCall { args, .. } => {
            for a in args {
                match a {
                    Arg::Positional(e) => walk_expr(e, f),
                    Arg::Named { value, .. } => walk_expr(value, f),
                    Arg::UnpackObject(e) => walk_expr(e, f),
                }
            }
        }
        Expr::InterpolatedString { segments, .. } => {
            for s in segments {
                if let crate::parser::ast::InterpolatedSegment::Expr { expr, .. } = s {
                    walk_expr(expr, f);
                }
            }
        }
        Expr::ExprReturn { value, .. } | Expr::Ireturn { value, .. } => {
            if let Some(e) = value {
                walk_expr(e, f);
            }
        }
    }
}

fn walk_stmt(stmt: &Stmt, f: &mut impl FnMut(&Expr)) {
    match stmt {
        Stmt::Let { value, .. } => walk_expr(value, f),
        Stmt::Expr { expr, .. } => walk_expr(expr, f),
        Stmt::If {
            condition,
            then_branch,
            else_branch,
            ..
        } => {
            walk_expr(condition, f);
            for s in then_branch {
                walk_stmt(s, f);
            }
            if let Some(else_b) = else_branch {
                for s in else_b {
                    walk_stmt(s, f);
                }
            }
        }
        Stmt::While {
            condition, body, ..
        } => {
            walk_expr(condition, f);
            for s in body {
                walk_stmt(s, f);
            }
        }
        Stmt::For { iterable, body, .. } => {
            walk_expr(iterable, f);
            for s in body {
                walk_stmt(s, f);
            }
        }
        Stmt::Function { body, .. } | Stmt::StreamFunction { body, .. } => {
            for s in body {
                walk_stmt(s, f);
            }
        }
        Stmt::Return { value, .. } | Stmt::EReturn { value, .. } => {
            if let Some(e) = value {
                walk_expr(e, f);
            }
        }
        Stmt::Import { .. } => {}
        Stmt::Try {
            try_block,
            catch_blocks,
            else_block,
            finally_block,
            ..
        } => {
            for s in try_block {
                walk_stmt(s, f);
            }
            for c in catch_blocks {
                for s in &c.body {
                    walk_stmt(s, f);
                }
            }
            if let Some(eb) = else_block {
                for s in eb {
                    walk_stmt(s, f);
                }
            }
            if let Some(fin) = finally_block {
                for s in fin {
                    walk_stmt(s, f);
                }
            }
        }
        Stmt::Throw { value, .. } => walk_expr(value, f),
        Stmt::Class {
            constructors,
            methods,
            ..
        } => {
            for c in constructors {
                for s in &c.body {
                    walk_stmt(s, f);
                }
            }
            for m in methods {
                for s in &m.body {
                    walk_stmt(s, f);
                }
            }
        }
        Stmt::Break { .. } | Stmt::Continue { .. } => {}
    }
}

fn program_needs_ml_onehots(ast: &[Stmt]) -> bool {
    let mut found = false;
    for stmt in ast {
        walk_stmt(stmt, &mut |e| {
            if expr_needs_array_map_onehot_fusion(e) {
                found = true;
            }
        });
    }
    found
}

/// Adds `onehots` to `from ml import …` when the program contains a fusable `array(map(...one_hot...))`.
pub fn inject_ml_onehots_import(ast: &mut [Stmt]) {
    if !program_needs_ml_onehots(ast) {
        return;
    }
    for stmt in ast.iter_mut() {
        if let Stmt::Import {
            import_stmt: ImportStmt::From { module, items },
            ..
        } = stmt
        {
            if module != "ml" {
                continue;
            }
            let has = items.iter().any(|i| match i {
                ImportItem::Named(n) => n == "onehots",
                ImportItem::Aliased { name, .. } => name == "onehots",
                ImportItem::All => false,
            });
            if !has {
                items.push(ImportItem::Named("onehots".to_string()));
            }
        }
    }
}
