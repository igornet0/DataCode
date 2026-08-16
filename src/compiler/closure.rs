/// Работа с замыканиями: поиск захваченных переменных
use crate::common::error::LangError;
use crate::parser::ast::{AssignTarget, Expr, IndexExpr, InterpolatedSegment, ListComprehensionClause, Stmt, TableFilterPred, UnpackPattern};

fn walk_table_filter_pred_assign_checks(
    pred: &TableFilterPred,
    ancestor_bindings: &std::collections::HashSet<String>,
    locals: &mut std::collections::HashSet<String>,
) -> Result<(), LangError> {
    match pred {
        TableFilterPred::Compare { value, .. }
        | TableFilterPred::Membership { container: value, .. }
        | TableFilterPred::StringMatch { pattern: value, .. } => {
            walk_expr_assign_checks(value, ancestor_bindings, locals)
        }
        TableFilterPred::And(l, r) | TableFilterPred::Or(l, r) => {
            walk_table_filter_pred_assign_checks(l, ancestor_bindings, locals)?;
            walk_table_filter_pred_assign_checks(r, ancestor_bindings, locals)
        }
    }
}

/// Собирает имена переменных из паттерна распаковки
pub fn collect_unpack_pattern_variables(
    pattern: &[UnpackPattern],
    vars: &mut std::collections::HashSet<String>,
) {
    for pat in pattern {
        match pat {
            UnpackPattern::Variable(name) => {
                vars.insert(name.clone());
            }
            UnpackPattern::Wildcard => {}
            UnpackPattern::Variadic(name) => {
                vars.insert(name.clone());
            }
            UnpackPattern::VariadicWildcard => {}
            UnpackPattern::Nested(nested) => {
                collect_unpack_pattern_variables(nested, vars);
            }
        }
    }
}

/// Находит все переменные, используемые в выражениях
pub fn find_used_variables_in_expr(expr: &Expr) -> std::collections::HashSet<String> {
    let mut vars = std::collections::HashSet::new();
    match expr {
        Expr::Variable { name, .. } => {
            vars.insert(name.clone());
        }
        Expr::Assign { name, value, .. } => {
            vars.insert(name.clone());
            vars.extend(find_used_variables_in_expr(value));
        }
        Expr::UnpackAssign { targets, value, .. } => {
            for target in targets {
                match target {
                    AssignTarget::Name(name) => {
                        vars.insert(name.clone());
                    }
                    AssignTarget::Index { array, index } => {
                        vars.extend(find_used_variables_in_expr(array));
                        vars.extend(find_used_variables_in_expr(index));
                    }
                }
            }
            vars.extend(find_used_variables_in_expr(value));
        }
        Expr::Binary { left, right, .. } => {
            vars.extend(find_used_variables_in_expr(left));
            vars.extend(find_used_variables_in_expr(right));
        }
        Expr::Unary { right, .. } => {
            vars.extend(find_used_variables_in_expr(right));
        }
        Expr::Call { args, .. } => {
            use crate::parser::ast::Arg;
            for arg in args {
                match arg {
                    Arg::Positional(expr) => {
                        vars.extend(find_used_variables_in_expr(expr));
                    }
                    Arg::Named { value, .. } => {
                        vars.extend(find_used_variables_in_expr(value));
                    }
                    Arg::UnpackObject(expr) | Arg::UnpackArray(expr) => {
                        vars.extend(find_used_variables_in_expr(expr));
                    }
                }
            }
        }
        Expr::ArrayLiteral { elements, .. } => {
            for elem in elements {
                vars.extend(find_used_variables_in_expr(elem));
            }
        }
        Expr::ObjectLiteral { pairs, .. } => {
            for p in pairs {
                match p {
                    crate::parser::ast::ObjectPair::KeyValue(_, value) => {
                        vars.extend(find_used_variables_in_expr(value));
                    }
                    crate::parser::ast::ObjectPair::KeyValueExpr(key, value) => {
                        vars.extend(find_used_variables_in_expr(key));
                        vars.extend(find_used_variables_in_expr(value));
                    }
                    crate::parser::ast::ObjectPair::Spread(expr) => {
                        vars.extend(find_used_variables_in_expr(expr));
                    }
                }
            }
        }
        Expr::DictComprehension {
            key_expr,
            value_expr,
            loop_var,
            iterable,
            condition,
            ..
        } => {
            vars.extend(find_used_variables_in_expr(iterable));
            vars.extend(find_used_variables_in_expr(key_expr));
            vars.extend(find_used_variables_in_expr(value_expr));
            if let Some(c) = condition {
                vars.extend(find_used_variables_in_expr(c));
            }
            vars.remove(loop_var);
        }
        Expr::ListComprehension { elt, clauses, .. } => {
            let mut bound = std::collections::HashSet::new();
            for cl in clauses {
                if let ListComprehensionClause::For { pattern, .. } = cl {
                    collect_unpack_pattern_variables(pattern, &mut bound);
                }
            }
            for cl in clauses {
                match cl {
                    ListComprehensionClause::For { iterable, .. } => {
                        vars.extend(find_used_variables_in_expr(iterable));
                    }
                    ListComprehensionClause::If { condition } => {
                        vars.extend(find_used_variables_in_expr(condition));
                    }
                }
            }
            vars.extend(find_used_variables_in_expr(elt));
            for name in bound {
                vars.remove(&name);
            }
        }
        Expr::TupleLiteral { elements, .. } => {
            for elem in elements {
                vars.extend(find_used_variables_in_expr(elem));
            }
        }
        Expr::ArrayIndex { array, index, .. } => {
            vars.extend(find_used_variables_in_expr(array));
            vars.extend(find_used_variables_in_index_expr(index));
        }
        Expr::AssignArray {
            array,
            index,
            value,
            ..
        } => {
            vars.extend(find_used_variables_in_expr(array));
            vars.extend(find_used_variables_in_index_expr(index));
            vars.extend(find_used_variables_in_expr(value));
        }
        Expr::AssignArrayOp {
            array,
            index,
            value,
            ..
        } => {
            vars.extend(find_used_variables_in_expr(array));
            vars.extend(find_used_variables_in_index_expr(index));
            vars.extend(find_used_variables_in_expr(value));
        }
        Expr::TableFilter { table, predicate, .. } => {
            vars.extend(find_used_variables_in_expr(table));
            predicate.for_each_value_expr(&mut |e| vars.extend(find_used_variables_in_expr(e)));
        }
        Expr::Property { object, .. } => {
            vars.extend(find_used_variables_in_expr(object));
        }
        Expr::MethodCall { object, args, .. } => {
            vars.extend(find_used_variables_in_expr(object));
            use crate::parser::ast::Arg;
            for arg in args {
                match arg {
                    Arg::Positional(expr) => {
                        vars.extend(find_used_variables_in_expr(expr));
                    }
                    Arg::Named { value, .. } => {
                        vars.extend(find_used_variables_in_expr(value));
                    }
                    Arg::UnpackObject(expr) | Arg::UnpackArray(expr) => {
                        vars.extend(find_used_variables_in_expr(expr));
                    }
                }
            }
        }
        Expr::InterpolatedString { segments, .. } => {
            for seg in segments {
                if let InterpolatedSegment::Expr { expr: e, .. } = seg {
                    vars.extend(find_used_variables_in_expr(e));
                }
            }
        }
        Expr::CallValue { callee, args, .. } => {
            vars.extend(find_used_variables_in_expr(callee));
            for arg in args {
                match arg {
                    crate::parser::ast::Arg::Positional(expr) => {
                        vars.extend(find_used_variables_in_expr(expr));
                    }
                    crate::parser::ast::Arg::Named { value, .. } => {
                        vars.extend(find_used_variables_in_expr(value));
                    }
                    crate::parser::ast::Arg::UnpackObject(expr)
                    | crate::parser::ast::Arg::UnpackArray(expr) => {
                        vars.extend(find_used_variables_in_expr(expr));
                    }
                }
            }
        }
        Expr::Lambda { params, body, .. } => {
            for p in params {
                if let Some(ref d) = p.default_value {
                    vars.extend(find_used_variables_in_expr(d));
                }
            }
            let mut inner = find_used_variables_in_expr(body);
            for p in params {
                inner.remove(&p.name);
            }
            vars.extend(inner);
        }
        Expr::AssignOp { name, value, .. } => {
            vars.insert(name.clone());
            vars.extend(find_used_variables_in_expr(value));
        }
        Expr::ExprReturn { value, .. } | Expr::Ireturn { value, .. } => {
            if let Some(e) = value {
                vars.extend(find_used_variables_in_expr(e));
            }
        }
        Expr::If {
            condition,
            then_branch,
            else_branch,
            ..
        } => {
            vars.extend(find_used_variables_in_expr(condition));
            match then_branch {
                crate::parser::ast::IfBranch::Expr(e) => {
                    vars.extend(find_used_variables_in_expr(e));
                }
                crate::parser::ast::IfBranch::Block(stmts) => {
                    for s in stmts {
                        vars.extend(find_used_variables_in_stmt(s));
                    }
                }
            }
            match else_branch {
                crate::parser::ast::IfBranch::Expr(e) => {
                    vars.extend(find_used_variables_in_expr(e));
                }
                crate::parser::ast::IfBranch::Block(stmts) => {
                    for s in stmts {
                        vars.extend(find_used_variables_in_stmt(s));
                    }
                }
            }
        }
        _ => {}
    }
    vars
}

fn find_used_variables_in_index_expr(index: &IndexExpr) -> std::collections::HashSet<String> {
    let mut vars = std::collections::HashSet::new();
    match index {
        IndexExpr::Scalar(e) => {
            vars.extend(find_used_variables_in_expr(e));
        }
        IndexExpr::Slice {
            start, stop, step, ..
        } => {
            for e in [start, stop, step].into_iter().flatten() {
                vars.extend(find_used_variables_in_expr(e));
            }
        }
    }
    vars
}

/// Переменные, которые читаются для побочных эффектов и RHS (левая часть простого `=` / распаковки не считается чтением для захвата).
pub fn find_read_variables_in_expr(expr: &Expr) -> std::collections::HashSet<String> {
    let mut vars = std::collections::HashSet::new();
    match expr {
        Expr::Variable { name, .. } => {
            vars.insert(name.clone());
        }
        Expr::Assign { value, .. } => {
            vars.extend(find_read_variables_in_expr(value));
        }
        Expr::UnpackAssign { value, .. } => {
            vars.extend(find_read_variables_in_expr(value));
        }
        Expr::AssignOp { name, value, .. } => {
            vars.insert(name.clone());
            vars.extend(find_read_variables_in_expr(value));
        }
        Expr::Binary { left, right, .. } => {
            vars.extend(find_read_variables_in_expr(left));
            vars.extend(find_read_variables_in_expr(right));
        }
        Expr::Unary { right, .. } => {
            vars.extend(find_read_variables_in_expr(right));
        }
        Expr::Call { args, .. } => {
            use crate::parser::ast::Arg;
            for arg in args {
                match arg {
                    Arg::Positional(expr) => {
                        vars.extend(find_read_variables_in_expr(expr));
                    }
                    Arg::Named { value, .. } => {
                        vars.extend(find_read_variables_in_expr(value));
                    }
                    Arg::UnpackObject(expr) | Arg::UnpackArray(expr) => {
                        vars.extend(find_read_variables_in_expr(expr));
                    }
                }
            }
        }
        Expr::ArrayLiteral { elements, .. } => {
            for elem in elements {
                vars.extend(find_read_variables_in_expr(elem));
            }
        }
        Expr::ObjectLiteral { pairs, .. } => {
            for p in pairs {
                match p {
                    crate::parser::ast::ObjectPair::KeyValue(_, value) => {
                        vars.extend(find_read_variables_in_expr(value));
                    }
                    crate::parser::ast::ObjectPair::KeyValueExpr(key, value) => {
                        vars.extend(find_read_variables_in_expr(key));
                        vars.extend(find_read_variables_in_expr(value));
                    }
                    crate::parser::ast::ObjectPair::Spread(expr) => {
                        vars.extend(find_read_variables_in_expr(expr));
                    }
                }
            }
        }
        Expr::DictComprehension {
            key_expr,
            value_expr,
            loop_var,
            iterable,
            condition,
            ..
        } => {
            vars.extend(find_read_variables_in_expr(iterable));
            vars.extend(find_read_variables_in_expr(key_expr));
            vars.extend(find_read_variables_in_expr(value_expr));
            if let Some(c) = condition {
                vars.extend(find_read_variables_in_expr(c));
            }
            vars.remove(loop_var);
        }
        Expr::ListComprehension { elt, clauses, .. } => {
            let mut bound = std::collections::HashSet::new();
            for cl in clauses {
                if let ListComprehensionClause::For { pattern, .. } = cl {
                    collect_unpack_pattern_variables(pattern, &mut bound);
                }
            }
            for cl in clauses {
                match cl {
                    ListComprehensionClause::For { iterable, .. } => {
                        vars.extend(find_read_variables_in_expr(iterable));
                    }
                    ListComprehensionClause::If { condition } => {
                        vars.extend(find_read_variables_in_expr(condition));
                    }
                }
            }
            vars.extend(find_read_variables_in_expr(elt));
            for name in bound {
                vars.remove(&name);
            }
        }
        Expr::TupleLiteral { elements, .. } => {
            for elem in elements {
                vars.extend(find_read_variables_in_expr(elem));
            }
        }
        Expr::ArrayIndex { array, index, .. } => {
            vars.extend(find_read_variables_in_expr(array));
            vars.extend(find_read_variables_in_index_expr(index));
        }
        Expr::AssignArray {
            array,
            index,
            value,
            ..
        } => {
            vars.extend(find_read_variables_in_expr(array));
            vars.extend(find_read_variables_in_index_expr(index));
            vars.extend(find_read_variables_in_expr(value));
        }
        Expr::AssignArrayOp {
            array,
            index,
            value,
            ..
        } => {
            vars.extend(find_read_variables_in_expr(array));
            vars.extend(find_read_variables_in_index_expr(index));
            vars.extend(find_read_variables_in_expr(value));
        }
        Expr::TableFilter { table, predicate, .. } => {
            vars.extend(find_read_variables_in_expr(table));
            predicate.for_each_value_expr(&mut |e| vars.extend(find_read_variables_in_expr(e)));
        }
        Expr::Property { object, .. } => {
            vars.extend(find_read_variables_in_expr(object));
        }
        Expr::MethodCall { object, args, .. } => {
            vars.extend(find_read_variables_in_expr(object));
            use crate::parser::ast::Arg;
            for arg in args {
                match arg {
                    Arg::Positional(expr) => {
                        vars.extend(find_read_variables_in_expr(expr));
                    }
                    Arg::Named { value, .. } => {
                        vars.extend(find_read_variables_in_expr(value));
                    }
                    Arg::UnpackObject(expr) | Arg::UnpackArray(expr) => {
                        vars.extend(find_read_variables_in_expr(expr));
                    }
                }
            }
        }
        Expr::InterpolatedString { segments, .. } => {
            for seg in segments {
                if let InterpolatedSegment::Expr { expr: e, .. } = seg {
                    vars.extend(find_read_variables_in_expr(e));
                }
            }
        }
        Expr::CallValue { callee, args, .. } => {
            vars.extend(find_read_variables_in_expr(callee));
            for arg in args {
                match arg {
                    crate::parser::ast::Arg::Positional(expr) => {
                        vars.extend(find_read_variables_in_expr(expr));
                    }
                    crate::parser::ast::Arg::Named { value, .. } => {
                        vars.extend(find_read_variables_in_expr(value));
                    }
                    crate::parser::ast::Arg::UnpackObject(expr)
                    | crate::parser::ast::Arg::UnpackArray(expr) => {
                        vars.extend(find_read_variables_in_expr(expr));
                    }
                }
            }
        }
        Expr::Lambda { params, body, .. } => {
            for p in params {
                if let Some(ref d) = p.default_value {
                    vars.extend(find_read_variables_in_expr(d));
                }
            }
            let mut inner = find_read_variables_in_expr(body);
            for p in params {
                inner.remove(&p.name);
            }
            vars.extend(inner);
        }
        Expr::ExprReturn { value, .. } | Expr::Ireturn { value, .. } => {
            if let Some(e) = value {
                vars.extend(find_read_variables_in_expr(e));
            }
        }
        Expr::If {
            condition,
            then_branch,
            else_branch,
            ..
        } => {
            vars.extend(find_read_variables_in_expr(condition));
            match then_branch {
                crate::parser::ast::IfBranch::Expr(e) => {
                    vars.extend(find_read_variables_in_expr(e));
                }
                crate::parser::ast::IfBranch::Block(stmts) => {
                    for s in stmts {
                        vars.extend(find_read_variables_in_stmt(s));
                    }
                }
            }
            match else_branch {
                crate::parser::ast::IfBranch::Expr(e) => {
                    vars.extend(find_read_variables_in_expr(e));
                }
                crate::parser::ast::IfBranch::Block(stmts) => {
                    for s in stmts {
                        vars.extend(find_read_variables_in_stmt(s));
                    }
                }
            }
        }
        _ => {}
    }
    vars
}

fn find_read_variables_in_index_expr(index: &IndexExpr) -> std::collections::HashSet<String> {
    let mut vars = std::collections::HashSet::new();
    match index {
        IndexExpr::Scalar(e) => {
            vars.extend(find_read_variables_in_expr(e));
        }
        IndexExpr::Slice {
            start, stop, step, ..
        } => {
            for e in [start, stop, step].into_iter().flatten() {
                vars.extend(find_read_variables_in_expr(e));
            }
        }
    }
    vars
}


/// То же для statements (чтение для захвата).
pub fn find_read_variables_in_stmt(stmt: &Stmt) -> std::collections::HashSet<String> {
    let mut vars = std::collections::HashSet::new();
    match stmt {
        Stmt::Import { .. } => {}
        Stmt::Let { value, .. } => {
            vars.extend(find_read_variables_in_expr(value));
        }
        Stmt::Expr { expr, .. } => {
            vars.extend(find_read_variables_in_expr(expr));
        }
        Stmt::If {
            condition,
            then_branch,
            else_branch,
            ..
        } => {
            vars.extend(find_read_variables_in_expr(condition));
            for stmt in then_branch {
                vars.extend(find_read_variables_in_stmt(stmt));
            }
            if let Some(else_branch) = else_branch {
                for stmt in else_branch {
                    vars.extend(find_read_variables_in_stmt(stmt));
                }
            }
        }
        Stmt::While {
            condition, body, ..
        } => {
            vars.extend(find_read_variables_in_expr(condition));
            for stmt in body {
                vars.extend(find_read_variables_in_stmt(stmt));
            }
        }
        Stmt::For { iterable, body, .. } => {
            vars.extend(find_read_variables_in_expr(iterable));
            for stmt in body {
                vars.extend(find_read_variables_in_stmt(stmt));
            }
        }
        Stmt::Function { body, .. } | Stmt::StreamFunction { body, .. } => {
            for stmt in body {
                vars.extend(find_read_variables_in_stmt(stmt));
            }
        }
        Stmt::Return { value, .. } | Stmt::EReturn { value, .. } => {
            if let Some(expr) = value {
                vars.extend(find_read_variables_in_expr(expr));
            }
        }
        Stmt::Break { .. } => {}
        Stmt::Continue { .. } => {}
        Stmt::Try {
            try_block,
            catch_blocks,
            else_block,
            ..
        } => {
            for stmt in try_block {
                vars.extend(find_read_variables_in_stmt(stmt));
            }
            for catch_block in catch_blocks {
                for stmt in &catch_block.body {
                    vars.extend(find_read_variables_in_stmt(stmt));
                }
            }
            if let Some(else_block) = else_block {
                for stmt in else_block {
                    vars.extend(find_read_variables_in_stmt(stmt));
                }
            }
        }
        Stmt::Throw { value, .. } => {
            vars.extend(find_read_variables_in_expr(value));
        }
        Stmt::Class {
            private_fields,
            protected_fields,
            public_fields,
            private_variables,
            protected_variables,
            public_variables,
            constructors,
            methods,
            ..
        } => {
            for field in private_fields
                .iter()
                .chain(protected_fields.iter())
                .chain(public_fields.iter())
            {
                if let Some(ref default_expr) = field.default_value {
                    vars.extend(find_read_variables_in_expr(default_expr));
                }
            }
            for var in private_variables
                .iter()
                .chain(protected_variables.iter())
                .chain(public_variables.iter())
            {
                vars.extend(find_read_variables_in_expr(&var.value));
            }
            for constructor in constructors {
                for param in &constructor.params {
                    if let Some(ref default_expr) = param.default_value {
                        vars.extend(find_read_variables_in_expr(default_expr));
                    }
                }
                for stmt in &constructor.body {
                    vars.extend(find_read_variables_in_stmt(stmt));
                }
            }
            for method in methods {
                for param in &method.params {
                    if let Some(ref default_expr) = param.default_value {
                        vars.extend(find_read_variables_in_expr(default_expr));
                    }
                }
                for stmt in &method.body {
                    vars.extend(find_read_variables_in_stmt(stmt));
                }
            }
        }
    }
    vars
}


/// Находит все переменные, используемые в statements
pub fn find_used_variables_in_stmt(stmt: &Stmt) -> std::collections::HashSet<String> {
    let mut vars = std::collections::HashSet::new();
    match stmt {
        Stmt::Import { .. } => {
            // Import statements don't use variables
        }
        Stmt::Let { value, .. } => {
            vars.extend(find_used_variables_in_expr(value));
        }
        Stmt::Expr { expr, .. } => {
            vars.extend(find_used_variables_in_expr(expr));
        }
        Stmt::If {
            condition,
            then_branch,
            else_branch,
            ..
        } => {
            vars.extend(find_used_variables_in_expr(condition));
            for stmt in then_branch {
                vars.extend(find_used_variables_in_stmt(stmt));
            }
            if let Some(else_branch) = else_branch {
                for stmt in else_branch {
                    vars.extend(find_used_variables_in_stmt(stmt));
                }
            }
        }
        Stmt::While {
            condition, body, ..
        } => {
            vars.extend(find_used_variables_in_expr(condition));
            for stmt in body {
                vars.extend(find_used_variables_in_stmt(stmt));
            }
        }
        Stmt::For { iterable, body, .. } => {
            vars.extend(find_used_variables_in_expr(iterable));
            for stmt in body {
                vars.extend(find_used_variables_in_stmt(stmt));
            }
        }
        Stmt::Function { body, .. } | Stmt::StreamFunction { body, .. } => {
            for stmt in body {
                vars.extend(find_used_variables_in_stmt(stmt));
            }
        }
        Stmt::Return { value, .. } | Stmt::EReturn { value, .. } => {
            if let Some(expr) = value {
                vars.extend(find_used_variables_in_expr(expr));
            }
        }
        Stmt::Break { .. } => {
            // break не использует переменные
        }
        Stmt::Continue { .. } => {
            // continue не использует переменные
        }
        Stmt::Try {
            try_block,
            catch_blocks,
            else_block,
            ..
        } => {
            // Находим переменные в try блоке
            for stmt in try_block {
                vars.extend(find_used_variables_in_stmt(stmt));
            }
            // Находим переменные в catch блоках
            for catch_block in catch_blocks {
                for stmt in &catch_block.body {
                    vars.extend(find_used_variables_in_stmt(stmt));
                }
            }
            // Находим переменные в else блоке (если есть)
            if let Some(else_block) = else_block {
                for stmt in else_block {
                    vars.extend(find_used_variables_in_stmt(stmt));
                }
            }
        }
        Stmt::Throw { value, .. } => {
            // Находим переменные в выражении throw
            vars.extend(find_used_variables_in_expr(value));
        }
        Stmt::Class {
            private_fields,
            protected_fields,
            public_fields,
            private_variables,
            protected_variables,
            public_variables,
            constructors,
            methods,
            ..
        } => {
            // Находим переменные в значениях по умолчанию полей
            for field in private_fields
                .iter()
                .chain(protected_fields.iter())
                .chain(public_fields.iter())
            {
                if let Some(ref default_expr) = field.default_value {
                    vars.extend(find_used_variables_in_expr(default_expr));
                }
            }
            // Находим переменные в выражениях переменных уровня класса
            for var in private_variables
                .iter()
                .chain(protected_variables.iter())
                .chain(public_variables.iter())
            {
                vars.extend(find_used_variables_in_expr(&var.value));
            }
            // Находим переменные в конструкторах и методах
            for constructor in constructors {
                for param in &constructor.params {
                    if let Some(ref default_expr) = param.default_value {
                        vars.extend(find_used_variables_in_expr(default_expr));
                    }
                }
                for stmt in &constructor.body {
                    vars.extend(find_used_variables_in_stmt(stmt));
                }
            }
            for method in methods {
                for param in &method.params {
                    if let Some(ref default_expr) = param.default_value {
                        vars.extend(find_used_variables_in_expr(default_expr));
                    }
                }
                for stmt in &method.body {
                    vars.extend(find_used_variables_in_stmt(stmt));
                }
            }
        }
    }
    vars
}

/// Находит все переменные, объявленные локально в теле функции
/// (через let и for, рекурсивно проверяя вложенные блоки)
pub fn find_locally_declared_variables(body: &[Stmt]) -> std::collections::HashSet<String> {
    let mut declared_vars = std::collections::HashSet::new();

    for stmt in body {
        match stmt {
            Stmt::Let {
                name, is_global, ..
            } => {
                // Добавляем только локальные переменные (не глобальные)
                if !is_global {
                    declared_vars.insert(name.clone());
                }
            }
            Stmt::Expr { expr, .. } => {
                collect_assign_declarations_from_expr(expr, &mut declared_vars);
            }
            Stmt::For { pattern, body, .. } => {
                // Переменные цикла for объявляются локально
                collect_unpack_pattern_variables(pattern, &mut declared_vars);
                // Рекурсивно проверяем тело цикла
                declared_vars.extend(find_locally_declared_variables(body));
            }
            Stmt::If {
                then_branch,
                else_branch,
                ..
            } => {
                // Рекурсивно проверяем ветки if
                declared_vars.extend(find_locally_declared_variables(then_branch));
                if let Some(else_branch) = else_branch {
                    declared_vars.extend(find_locally_declared_variables(else_branch));
                }
            }
            Stmt::While { body, .. } => {
                // Рекурсивно проверяем тело while
                declared_vars.extend(find_locally_declared_variables(body));
            }
            Stmt::Function { .. } | Stmt::StreamFunction { .. } => {
                // Вложенная функция имеет собственную область; не смешиваем её let/for с родителем
            }
            Stmt::Try {
                try_block,
                catch_blocks,
                else_block,
                ..
            } => {
                // Рекурсивно проверяем try блок
                declared_vars.extend(find_locally_declared_variables(try_block));
                // Рекурсивно проверяем catch блоки
                for catch_block in catch_blocks {
                    if let Some(ev) = &catch_block.error_var {
                        declared_vars.insert(ev.clone());
                    }
                    declared_vars.extend(find_locally_declared_variables(&catch_block.body));
                }
                // Рекурсивно проверяем else блок (если есть)
                if let Some(else_block) = else_block {
                    declared_vars.extend(find_locally_declared_variables(else_block));
                }
            }
            Stmt::Class { .. } => {
                // Тело класса (поля/методы) не объявляет локали текущей функции
            }
            _ => {
                // Expr, Return, Break, Continue не объявляют переменные
            }
        }
    }

    declared_vars
}

fn collect_assign_declarations_from_expr(
    expr: &Expr,
    declared_vars: &mut std::collections::HashSet<String>,
) {
    match expr {
        Expr::Assign { name, .. } => {
            declared_vars.insert(name.clone());
        }
        Expr::UnpackAssign { targets, .. } => {
            for target in targets {
                if let AssignTarget::Name(name) = target {
                    declared_vars.insert(name.clone());
                }
            }
        }
        _ => {}
    }
}

/// Находит переменные, которые используются в теле функции, но не объявлены в ней
pub fn find_captured_variables(
    body: &[Stmt],
    parent_locals: &[std::collections::HashMap<String, usize>],
    params: &[String],
    current_scope_locals: &std::collections::HashMap<String, usize>,
) -> Vec<String> {
    let mut used_vars = std::collections::HashSet::new();
    for stmt in body {
        used_vars.extend(find_read_variables_in_stmt(stmt));
    }

    // Исключаем параметры функции
    let param_set: std::collections::HashSet<String> = params.iter().cloned().collect();
    used_vars.retain(|v| !param_set.contains(v));

    // Находим все переменные, объявленные локально в теле функции
    let locally_declared = find_locally_declared_variables(body);

    // Исключаем локально объявленные переменные из проверки захвата
    // Они локальные, не требуют захвата из родительских областей
    used_vars.retain(|v| !locally_declared.contains(v));

    // Ищем переменные, которые используются, но не найдены в текущих областях видимости
    // но найдены в родительских областях видимости
    let mut captured = Vec::new();

    for var_name in &used_vars {
        // Проверяем, найдена ли переменная в текущей области видимости функции
        // (только в последней области, которая была создана для этой функции)
        // НЕ проверяем в родительских областях, которые все еще в self.scope.locals
        let found_in_current_scope = current_scope_locals.contains_key(var_name);

        if !found_in_current_scope {
            // Проверяем, найдена ли переменная в родительских областях видимости
            let found_in_parent = parent_locals
                .iter()
                .any(|scope| scope.contains_key(var_name));

            if found_in_parent {
                captured.push(var_name.clone());
            }
        }
    }

    captured.sort();
    captured
}

/// Захват для лямбды: тело — выражение, параметры с опциональными default.
pub fn find_captured_variables_lambda(
    body: &Expr,
    params: &[crate::parser::ast::Param],
    parent_locals: &[std::collections::HashMap<String, usize>],
    current_scope_locals: &std::collections::HashMap<String, usize>,
) -> Vec<String> {
    let mut used_vars = find_read_variables_in_expr(body);
    for p in params {
        if let Some(ref d) = p.default_value {
            used_vars.extend(find_read_variables_in_expr(d));
        }
    }

    let param_set: std::collections::HashSet<String> =
        params.iter().map(|p| p.name.clone()).collect();
    used_vars.retain(|v| !param_set.contains(v));

    let locally_declared: std::collections::HashSet<String> = std::collections::HashSet::new();
    used_vars.retain(|v| !locally_declared.contains(v));

    let mut captured = Vec::new();
    for var_name in &used_vars {
        let found_in_current_scope = current_scope_locals.contains_key(var_name);
        if !found_in_current_scope {
            let found_in_parent = parent_locals
                .iter()
                .any(|scope| scope.contains_key(var_name));
            if found_in_parent {
                captured.push(var_name.clone());
            }
        }
    }
    captured.sort();
    captured
}

/// Имена всех родительских локальных слотов (плоско по стеку областей компилятора).
pub fn flatten_parent_binding_names(
    parent_locals_snapshot: &[std::collections::HashMap<String, usize>],
) -> std::collections::HashSet<String> {
    let mut names = std::collections::HashSet::new();
    for scope in parent_locals_snapshot {
        names.extend(scope.keys().cloned());
    }
    names
}

/// Bindings that a nested function must not assign without `let` shadowing.
/// Top-level `fn` / `stream fn` in the script chunk are not nested inside another function,
/// so script-scope names (e.g. `for x in …`) are globals — not closure parents.
pub fn ancestor_bindings_for_function_check(
    parent_locals_snapshot: &[std::collections::HashMap<String, usize>],
    compiling_inside_function: bool,
) -> std::collections::HashSet<String> {
    if compiling_inside_function {
        flatten_parent_binding_names(parent_locals_snapshot)
    } else {
        std::collections::HashSet::new()
    }
}

fn outer_assignment_error(name: &str, line: usize) -> LangError {
    LangError::ParseError {
        message: format!(
            "cannot assign to outer variable '{}' from nested function; pass state explicitly or use 'let {}' to shadow",
            name, name
        ),
        line,
        file: None,
    }
}

fn record_simple_assign_target(
    name: &str,
    line: usize,
    ancestor_bindings: &std::collections::HashSet<String>,
    locals: &mut std::collections::HashSet<String>,
) -> Result<(), LangError> {
    if name.contains('.') {
        return Ok(());
    }
    if ancestor_bindings.contains(name) && !locals.contains(name) {
        return Err(outer_assignment_error(name, line));
    }
    locals.insert(name.to_string());
    Ok(())
}

fn walk_args_assign_checks(
    args: &[crate::parser::ast::Arg],
    ancestor_bindings: &std::collections::HashSet<String>,
    locals: &mut std::collections::HashSet<String>,
) -> Result<(), LangError> {
    use crate::parser::ast::Arg;
    for arg in args {
        match arg {
            Arg::Positional(expr) => {
                walk_expr_assign_checks(expr, ancestor_bindings, locals)?;
            }
            Arg::Named { value, .. } => {
                walk_expr_assign_checks(value, ancestor_bindings, locals)?;
            }
            Arg::UnpackObject(expr) | Arg::UnpackArray(expr) => {
                walk_expr_assign_checks(expr, ancestor_bindings, locals)?;
            }
        }
    }
    Ok(())
}

fn walk_index_assign_checks(
    index: &IndexExpr,
    ancestor_bindings: &std::collections::HashSet<String>,
    locals: &mut std::collections::HashSet<String>,
) -> Result<(), LangError> {
    match index {
        IndexExpr::Scalar(e) => walk_expr_assign_checks(e, ancestor_bindings, locals),
        IndexExpr::Slice {
            start, stop, step, ..
        } => {
            for e in [start, stop, step].into_iter().flatten() {
                walk_expr_assign_checks(e, ancestor_bindings, locals)?;
            }
            Ok(())
        }
    }
}

fn check_lambda_assign_inner(
    body: &Expr,
    params: &[crate::parser::ast::Param],
    enclosing_bindings: &std::collections::HashSet<String>,
) -> Result<(), LangError> {
    let mut inner_locals: std::collections::HashSet<String> =
        params.iter().map(|p| p.name.clone()).collect();
    walk_expr_assign_checks(body, enclosing_bindings, &mut inner_locals)?;
    Ok(())
}

fn walk_expr_assign_checks(
    expr: &Expr,
    ancestor_bindings: &std::collections::HashSet<String>,
    locals: &mut std::collections::HashSet<String>,
) -> Result<(), LangError> {
    use crate::parser::ast::ObjectPair;
    match expr {
        Expr::Assign { name, value, line } => {
            walk_expr_assign_checks(value, ancestor_bindings, locals)?;
            record_simple_assign_target(name, *line, ancestor_bindings, locals)?;
        }
        Expr::AssignOp { name, value, line, .. } => {
            walk_expr_assign_checks(value, ancestor_bindings, locals)?;
            record_simple_assign_target(name, *line, ancestor_bindings, locals)?;
        }
        Expr::UnpackAssign { targets, value, line } => {
            walk_expr_assign_checks(value, ancestor_bindings, locals)?;
            for target in targets {
                match target {
                    AssignTarget::Name(n) => {
                        record_simple_assign_target(n, *line, ancestor_bindings, locals)?;
                    }
                    AssignTarget::Index { array, index } => {
                        walk_expr_assign_checks(array, ancestor_bindings, locals)?;
                        walk_expr_assign_checks(index, ancestor_bindings, locals)?;
                    }
                }
            }
        }
        Expr::Lambda {
            params,
            body,
            ..
        } => {
            let mut enclosing = ancestor_bindings.clone();
            enclosing.extend(locals.iter().cloned());
            for p in params {
                if let Some(ref d) = p.default_value {
                    let mut tmp = locals.clone();
                    walk_expr_assign_checks(d, &enclosing, &mut tmp)?;
                }
            }
            check_lambda_assign_inner(body.as_ref(), params, &enclosing)?;
        }
        Expr::DictComprehension {
            key_expr,
            value_expr,
            loop_var,
            iterable,
            condition,
            ..
        } => {
            walk_expr_assign_checks(iterable, ancestor_bindings, locals)?;
            let mut inner = locals.clone();
            inner.insert(loop_var.clone());
            walk_expr_assign_checks(key_expr, ancestor_bindings, &mut inner)?;
            walk_expr_assign_checks(value_expr, ancestor_bindings, &mut inner)?;
            if let Some(c) = condition {
                walk_expr_assign_checks(c.as_ref(), ancestor_bindings, &mut inner)?;
            }
        }
        Expr::ListComprehension { elt, clauses, .. } => {
            let mut inner = locals.clone();
            for cl in clauses {
                match cl {
                    ListComprehensionClause::For { pattern, iterable } => {
                        walk_expr_assign_checks(iterable, ancestor_bindings, &mut inner)?;
                        collect_unpack_pattern_variables(pattern, &mut inner);
                    }
                    ListComprehensionClause::If { condition } => {
                        walk_expr_assign_checks(condition.as_ref(), ancestor_bindings, &mut inner)?;
                    }
                }
            }
            walk_expr_assign_checks(elt, ancestor_bindings, &mut inner)?;
        }
        Expr::Binary { left, right, .. } => {
            walk_expr_assign_checks(left, ancestor_bindings, locals)?;
            walk_expr_assign_checks(right, ancestor_bindings, locals)?;
        }
        Expr::Unary { right, .. } => {
            walk_expr_assign_checks(right, ancestor_bindings, locals)?;
        }
        Expr::Call { args, .. } => {
            walk_args_assign_checks(args, ancestor_bindings, locals)?;
        }
        Expr::CallValue { callee, args, .. } => {
            walk_expr_assign_checks(callee, ancestor_bindings, locals)?;
            walk_args_assign_checks(args, ancestor_bindings, locals)?;
        }
        Expr::ArrayLiteral { elements, .. } => {
            for e in elements {
                walk_expr_assign_checks(e, ancestor_bindings, locals)?;
            }
        }
        Expr::ObjectLiteral { pairs, .. } => {
            for p in pairs {
                match p {
                    ObjectPair::KeyValue(_, value) => {
                        walk_expr_assign_checks(value, ancestor_bindings, locals)?;
                    }
                    ObjectPair::KeyValueExpr(key, value) => {
                        walk_expr_assign_checks(key, ancestor_bindings, locals)?;
                        walk_expr_assign_checks(value, ancestor_bindings, locals)?;
                    }
                    ObjectPair::Spread(e) => {
                        walk_expr_assign_checks(e, ancestor_bindings, locals)?;
                    }
                }
            }
        }
        Expr::TupleLiteral { elements, .. } => {
            for e in elements {
                walk_expr_assign_checks(e, ancestor_bindings, locals)?;
            }
        }
        Expr::ArrayIndex { array, index, .. } => {
            walk_expr_assign_checks(array, ancestor_bindings, locals)?;
            walk_index_assign_checks(index, ancestor_bindings, locals)?;
        }
        Expr::AssignArray {
            array,
            index,
            value,
            ..
        } => {
            walk_expr_assign_checks(array, ancestor_bindings, locals)?;
            walk_index_assign_checks(index, ancestor_bindings, locals)?;
            walk_expr_assign_checks(value, ancestor_bindings, locals)?;
        }
        Expr::AssignArrayOp {
            array,
            index,
            value,
            ..
        } => {
            walk_expr_assign_checks(array, ancestor_bindings, locals)?;
            walk_index_assign_checks(index, ancestor_bindings, locals)?;
            walk_expr_assign_checks(value, ancestor_bindings, locals)?;
        }
        Expr::TableFilter {
            table,
            predicate,
            ..
        } => {
            walk_expr_assign_checks(table, ancestor_bindings, locals)?;
            walk_table_filter_pred_assign_checks(predicate, ancestor_bindings, locals)?;
        }
        Expr::Property { object, .. } => {
            walk_expr_assign_checks(object, ancestor_bindings, locals)?;
        }
        Expr::MethodCall { object, args, .. } => {
            walk_expr_assign_checks(object, ancestor_bindings, locals)?;
            walk_args_assign_checks(args, ancestor_bindings, locals)?;
        }
        Expr::InterpolatedString { segments, .. } => {
            for seg in segments {
                if let InterpolatedSegment::Expr { expr: e, .. } = seg {
                    walk_expr_assign_checks(e, ancestor_bindings, locals)?;
                }
            }
        }
        Expr::SuperCall { args, .. } => {
            walk_args_assign_checks(args, ancestor_bindings, locals)?;
        }
        Expr::SuperMethodCall { args, .. } => {
            walk_args_assign_checks(args, ancestor_bindings, locals)?;
        }
        Expr::ExprReturn { value, .. } | Expr::Ireturn { value, .. } => {
            if let Some(e) = value {
                walk_expr_assign_checks(e.as_ref(), ancestor_bindings, locals)?;
            }
        }
        Expr::If {
            condition,
            then_branch,
            else_branch,
            ..
        } => {
            walk_expr_assign_checks(condition, ancestor_bindings, locals)?;
            match then_branch {
                crate::parser::ast::IfBranch::Expr(e) => {
                    walk_expr_assign_checks(e, ancestor_bindings, locals)?;
                }
                crate::parser::ast::IfBranch::Block(stmts) => {
                    walk_stmts_assign_checks(stmts, ancestor_bindings, locals)?;
                }
            }
            match else_branch {
                crate::parser::ast::IfBranch::Expr(e) => {
                    walk_expr_assign_checks(e, ancestor_bindings, locals)?;
                }
                crate::parser::ast::IfBranch::Block(stmts) => {
                    walk_stmts_assign_checks(stmts, ancestor_bindings, locals)?;
                }
            }
        }
        Expr::Variable { .. }
        | Expr::Literal { .. }
        | Expr::This { .. }
        | Expr::Super { .. }
        | Expr::Ellipsis { .. }
        | Expr::TableColumnWrite { .. }
        | Expr::AssignTableColumn { .. } => {}
    }
    Ok(())
}

fn walk_stmts_assign_checks(
    body: &[Stmt],
    ancestor_bindings: &std::collections::HashSet<String>,
    locals: &mut std::collections::HashSet<String>,
) -> Result<(), LangError> {
    for stmt in body {
        walk_stmt_assign_checks(stmt, ancestor_bindings, locals)?;
    }
    Ok(())
}

fn walk_stmt_assign_checks(
    stmt: &Stmt,
    ancestor_bindings: &std::collections::HashSet<String>,
    locals: &mut std::collections::HashSet<String>,
) -> Result<(), LangError> {
    match stmt {
        Stmt::Import { .. } => Ok(()),
        Stmt::Let {
            name,
            value,
            is_global,
            ..
        } => {
            walk_expr_assign_checks(value, ancestor_bindings, locals)?;
            if !is_global {
                locals.insert(name.clone());
            }
            Ok(())
        }
        Stmt::Expr { expr, .. } => walk_expr_assign_checks(expr, ancestor_bindings, locals),
        Stmt::If {
            condition,
            then_branch,
            else_branch,
            ..
        } => {
            walk_expr_assign_checks(condition, ancestor_bindings, locals)?;
            let base = locals.clone();
            let mut then_l = base.clone();
            walk_stmts_assign_checks(then_branch, ancestor_bindings, &mut then_l)?;
            locals.extend(then_l.iter().cloned());
            if let Some(eb) = else_branch {
                let mut else_l = base.clone();
                walk_stmts_assign_checks(eb, ancestor_bindings, &mut else_l)?;
                locals.extend(else_l.iter().cloned());
            }
            Ok(())
        }
        Stmt::While {
            condition, body, ..
        } => {
            walk_expr_assign_checks(condition, ancestor_bindings, locals)?;
            let base = locals.clone();
            let mut inner = base.clone();
            walk_stmts_assign_checks(body, ancestor_bindings, &mut inner)?;
            locals.extend(inner.iter().cloned());
            Ok(())
        }
        Stmt::For {
            pattern,
            iterable,
            body,
            ..
        } => {
            walk_expr_assign_checks(iterable, ancestor_bindings, locals)?;
            let mut pattern_names = std::collections::HashSet::new();
            collect_unpack_pattern_variables(pattern, &mut pattern_names);
            let mut inner = locals.clone();
            collect_unpack_pattern_variables(pattern, &mut inner);
            walk_stmts_assign_checks(body, ancestor_bindings, &mut inner)?;
            for name in inner {
                if !pattern_names.contains(&name) {
                    locals.insert(name);
                }
            }
            Ok(())
        }
        Stmt::Function { .. } | Stmt::StreamFunction { .. } | Stmt::Class { .. } => Ok(()),
        Stmt::Return { value, .. } | Stmt::EReturn { value, .. } => {
            if let Some(expr) = value {
                walk_expr_assign_checks(expr, ancestor_bindings, locals)?;
            }
            Ok(())
        }
        Stmt::Break { .. } | Stmt::Continue { .. } => Ok(()),
        Stmt::Try {
            try_block,
            catch_blocks,
            else_block,
            ..
        } => {
            walk_stmts_assign_checks(try_block, ancestor_bindings, locals)?;
            for catch_block in catch_blocks {
                let mut catch_locals = locals.clone();
                if let Some(ev) = &catch_block.error_var {
                    catch_locals.insert(ev.clone());
                }
                walk_stmts_assign_checks(&catch_block.body, ancestor_bindings, &mut catch_locals)?;
            }
            if let Some(eb) = else_block {
                let mut else_locals = locals.clone();
                walk_stmts_assign_checks(eb, ancestor_bindings, &mut else_locals)?;
            }
            Ok(())
        }
        Stmt::Throw { value, .. } => walk_expr_assign_checks(value, ancestor_bindings, locals),
    }
}

/// Запрет присваивания переменным родительских функций без `let`-тени (правило immutable closures).
pub fn check_illegal_outer_assignments_in_function_body(
    body: &[Stmt],
    ancestor_bindings: &std::collections::HashSet<String>,
    param_names: &[String],
) -> Result<(), LangError> {
    let mut locals: std::collections::HashSet<String> = param_names.iter().cloned().collect();
    walk_stmts_assign_checks(body, ancestor_bindings, &mut locals)?;
    Ok(())
}

pub fn check_illegal_outer_assignments_in_lambda_expr(
    body: &Expr,
    ancestor_bindings: &std::collections::HashSet<String>,
    param_names: &[String],
) -> Result<(), LangError> {
    let mut locals: std::collections::HashSet<String> = param_names.iter().cloned().collect();
    walk_expr_assign_checks(body, ancestor_bindings, &mut locals)?;
    Ok(())
}