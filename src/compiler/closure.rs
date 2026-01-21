/// Работа с замыканиями: поиск захваченных переменных

use crate::parser::ast::{Expr, Stmt, UnpackPattern};

/// Собирает имена переменных из паттерна распаковки
pub fn collect_unpack_pattern_variables(pattern: &[UnpackPattern], vars: &mut std::collections::HashSet<String>) {
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
        Expr::UnpackAssign { names, value, .. } => {
            for name in names {
                vars.insert(name.clone());
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
                }
            }
        }
        Expr::ArrayLiteral { elements, .. } => {
            for elem in elements {
                vars.extend(find_used_variables_in_expr(elem));
            }
        }
        Expr::ObjectLiteral { pairs, .. } => {
            for (_, value) in pairs {
                vars.extend(find_used_variables_in_expr(value));
            }
        }
        Expr::TupleLiteral { elements, .. } => {
            for elem in elements {
                vars.extend(find_used_variables_in_expr(elem));
            }
        }
        Expr::ArrayIndex { array, index, .. } => {
            vars.extend(find_used_variables_in_expr(array));
            vars.extend(find_used_variables_in_expr(index));
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
                }
            }
        }
        _ => {}
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
        Stmt::If { condition, then_branch, else_branch, .. } => {
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
        Stmt::While { condition, body, .. } => {
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
        Stmt::Function { body, .. } => {
            for stmt in body {
                vars.extend(find_used_variables_in_stmt(stmt));
            }
        }
        Stmt::Return { value, .. } => {
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
        Stmt::Try { try_block, catch_blocks, else_block, .. } => {
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
    }
    vars
}

/// Находит все переменные, объявленные локально в теле функции
/// (через let и for, рекурсивно проверяя вложенные блоки)
pub fn find_locally_declared_variables(body: &[Stmt]) -> std::collections::HashSet<String> {
    let mut declared_vars = std::collections::HashSet::new();
    
    for stmt in body {
        match stmt {
            Stmt::Let { name, is_global, .. } => {
                // Добавляем только локальные переменные (не глобальные)
                if !is_global {
                    declared_vars.insert(name.clone());
                }
            }
            Stmt::For { pattern, body, .. } => {
                // Переменные цикла for объявляются локально
                collect_unpack_pattern_variables(pattern, &mut declared_vars);
                // Рекурсивно проверяем тело цикла
                declared_vars.extend(find_locally_declared_variables(body));
            }
            Stmt::If { then_branch, else_branch, .. } => {
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
            Stmt::Function { body, .. } => {
                // Рекурсивно проверяем тело вложенной функции
                declared_vars.extend(find_locally_declared_variables(body));
            }
            Stmt::Try { try_block, catch_blocks, else_block, .. } => {
                // Рекурсивно проверяем try блок
                declared_vars.extend(find_locally_declared_variables(try_block));
                // Рекурсивно проверяем catch блоки
                for catch_block in catch_blocks {
                    declared_vars.extend(find_locally_declared_variables(&catch_block.body));
                }
                // Рекурсивно проверяем else блок (если есть)
                if let Some(else_block) = else_block {
                    declared_vars.extend(find_locally_declared_variables(else_block));
                }
            }
            _ => {
                // Expr, Return, Break, Continue не объявляют переменные
            }
        }
    }
    
    declared_vars
}

/// Находит переменные, которые используются в теле функции, но не объявлены в ней
/// (т.е. захваченные из родительских функций)
pub fn find_captured_variables(
    body: &[Stmt],
    parent_locals: &[std::collections::HashMap<String, usize>],
    params: &[String],
    current_scope_locals: &std::collections::HashMap<String, usize>,
) -> Vec<String> {
    let mut used_vars = std::collections::HashSet::new();
    for stmt in body {
        used_vars.extend(find_used_variables_in_stmt(stmt));
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
            let found_in_parent = parent_locals.iter().any(|scope| scope.contains_key(var_name));
        
            if found_in_parent {
                captured.push(var_name.clone());
            }
        }
    }
    
    captured
}


