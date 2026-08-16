// Разрешение переменных и подготовка к компиляции

use crate::common::error::LangError;
use crate::parser::ast::{
    Arg, AssignTarget, Expr, IndexExpr, ListComprehensionClause, Param, Stmt, UnpackPattern,
};
use crate::semantic::scope::Scope;

pub struct Resolver {
    scopes: Vec<Scope>,
    current_function: FunctionType,
    source_name: Option<String>,
}

#[derive(Clone, Copy, PartialEq)]
enum FunctionType {
    None,
    Function,
    /// `stream fn` — `return` это yield, допустим `ereturn`.
    Stream,
}

impl Resolver {
    pub fn new() -> Self {
        Self::new_with_source_name(None)
    }

    pub fn new_with_source_name(source_name: Option<&str>) -> Self {
        Self {
            scopes: Vec::new(),
            current_function: FunctionType::None,
            source_name: source_name.map(String::from),
        }
    }

    pub fn resolve(&mut self, statements: &[Stmt]) -> Result<(), LangError> {
        for stmt in statements {
            self.resolve_stmt(stmt)?;
        }
        Ok(())
    }

    fn resolve_stmt(&mut self, stmt: &Stmt) -> Result<(), LangError> {
        match stmt {
            Stmt::Import { .. } => {
                // Import statements don't need variable resolution
                // They are handled at runtime by the VM
            }
            Stmt::Let {
                name,
                value,
                is_global,
                ..
            } => {
                self.resolve_expr(value)?;
                // Глобальные переменные не добавляются в локальные области видимости
                if !is_global {
                    self.declare(name);
                    self.define(name);
                }
            }
            Stmt::Expr { expr, .. } => {
                self.resolve_expr(expr)?;
            }
            Stmt::Function {
                name, params, body, ..
            } => {
                self.declare(name);
                self.define(name);
                // Разрешаем значения по умолчанию параметров
                for param in params {
                    if let Some(ref default_expr) = param.default_value {
                        self.resolve_expr(default_expr)?;
                    }
                }
                self.resolve_function(params, body, FunctionType::Function)?;
            }
            Stmt::StreamFunction {
                name, params, body, ..
            } => {
                self.declare(name);
                self.define(name);
                for param in params {
                    if let Some(ref default_expr) = param.default_value {
                        self.resolve_expr(default_expr)?;
                    }
                }
                self.resolve_function(params, body, FunctionType::Stream)?;
            }
            Stmt::If {
                condition,
                then_branch,
                else_branch,
                ..
            } => {
                self.resolve_expr(condition)?;
                // Same as Python: then/else bindings use the enclosing scope.
                self.resolve_stmt_block(then_branch)?;
                if let Some(else_branch) = else_branch {
                    self.resolve_stmt_block(else_branch)?;
                }
            }
            Stmt::While {
                condition, body, ..
            } => {
                self.resolve_expr(condition)?;
                self.resolve_stmt_block(body)?;
            }
            Stmt::Return { value, line } => {
                if self.current_function == FunctionType::None {
                    return Err(LangError::SemanticError {
                        message: "Cannot return from top-level code".to_string(),
                        line: *line,
                        file: self.source_name.clone(),
                    });
                }
                if let Some(expr) = value {
                    self.resolve_expr(expr)?;
                }
            }
            Stmt::EReturn { value, line } => {
                if self.current_function != FunctionType::Stream {
                    return Err(LangError::SemanticError {
                        message: "'ereturn' is only allowed inside a stream fn".to_string(),
                        line: *line,
                        file: self.source_name.clone(),
                    });
                }
                if let Some(expr) = value {
                    self.resolve_expr(expr)?;
                }
            }
            Stmt::Break { .. } => {
                // break не требует разрешения переменных
            }
            Stmt::Continue { .. } => {
                // continue не требует разрешения переменных
            }
            Stmt::For {
                pattern,
                iterable,
                body,
                ..
            } => {
                // Iterator and body bindings use the enclosing scope (Python for-loop semantics).
                self.declare_unpack_pattern(pattern);
                self.resolve_expr(iterable)?;
                self.resolve_stmt_block(body)?;
            }
            Stmt::Try {
                try_block,
                catch_blocks,
                else_block,
                finally_block,
                ..
            } => {
                // Разрешаем try блок
                self.resolve_stmt_block(try_block)?;

                // Разрешаем catch блоки
                for catch_block in catch_blocks {
                    self.begin_scope();
                    // Если есть переменная ошибки, объявляем её
                    if let Some(ref error_var) = catch_block.error_var {
                        self.declare(error_var);
                        self.define(error_var);
                    }
                    self.resolve_stmt_block(&catch_block.body)?;
                    self.end_scope();
                }

                // Разрешаем else блок (если есть)
                if let Some(ref else_block) = else_block {
                    self.resolve_stmt_block(else_block)?;
                }

                // Разрешаем finally блок (если есть)
                if let Some(ref finally_block) = finally_block {
                    self.resolve_stmt_block(finally_block)?;
                }
            }
            Stmt::Throw { value, .. } => {
                // Разрешаем выражение в throw
                self.resolve_expr(value)?;
            }
            Stmt::Class {
                name,
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
                // Объявляем класс
                self.declare(name);
                self.define(name);

                // Разрешаем значения по умолчанию полей
                for field in private_fields
                    .iter()
                    .chain(protected_fields.iter())
                    .chain(public_fields.iter())
                {
                    if let Some(ref default_expr) = field.default_value {
                        self.resolve_expr(default_expr)?;
                    }
                }

                // Разрешаем выражения переменных уровня класса
                for var in private_variables
                    .iter()
                    .chain(protected_variables.iter())
                    .chain(public_variables.iter())
                {
                    self.resolve_expr(&var.value)?;
                }

                // Разрешаем конструкторы и методы
                for constructor in constructors {
                    // Разрешаем значения по умолчанию параметров
                    for param in &constructor.params {
                        if let Some(ref default_expr) = param.default_value {
                            self.resolve_expr(default_expr)?;
                        }
                    }
                    self.resolve_function(
                        &constructor.params,
                        &constructor.body,
                        FunctionType::Function,
                    )?;
                }

                for method in methods {
                    // Разрешаем значения по умолчанию параметров
                    for param in &method.params {
                        if let Some(ref default_expr) = param.default_value {
                            self.resolve_expr(default_expr)?;
                        }
                    }
                    self.resolve_function(&method.params, &method.body, FunctionType::Function)?;
                }
            }
        }
        Ok(())
    }

    fn resolve_stmt_block(&mut self, statements: &[Stmt]) -> Result<(), LangError> {
        self.begin_scope();
        for stmt in statements {
            self.resolve_stmt(stmt)?;
        }
        self.end_scope();
        Ok(())
    }

    fn resolve_table_filter_pred(&mut self, pred: &crate::parser::ast::TableFilterPred) -> Result<(), LangError> {
        match pred {
            crate::parser::ast::TableFilterPred::Compare { value, .. }
            | crate::parser::ast::TableFilterPred::Membership { container: value, .. }
            | crate::parser::ast::TableFilterPred::StringMatch { pattern: value, .. } => {
                self.resolve_expr(value)
            }
            crate::parser::ast::TableFilterPred::And(l, r) | crate::parser::ast::TableFilterPred::Or(l, r) => {
                self.resolve_table_filter_pred(l)?;
                self.resolve_table_filter_pred(r)
            }
        }
    }

    fn resolve_expr(&mut self, expr: &Expr) -> Result<(), LangError> {
        match expr {
            Expr::Variable { name, .. } => {
                // TDZ (temporal dead zone) for `let x = x` would require tracking declare vs define;
                // `declare`/`define` currently share one map — see `Scope::locals`.
                self.resolve_local(expr, name);
            }
            Expr::Assign { name, value, .. } => {
                self.resolve_expr(value)?;
                self.resolve_local(expr, name);
            }
            Expr::AssignOp { name, value, .. } => {
                self.resolve_expr(value)?;
                self.resolve_local(expr, name);
            }
            Expr::Literal { .. } => {}
            Expr::Binary { left, right, .. } => {
                self.resolve_expr(left)?;
                self.resolve_expr(right)?;
            }
            Expr::Call { name: _, args, .. } => {
                for arg in args {
                    match arg {
                        Arg::Positional(expr) => {
                            self.resolve_expr(expr)?;
                        }
                        Arg::Named { value, .. } => {
                            self.resolve_expr(value)?;
                        }
                        Arg::UnpackObject(expr) | Arg::UnpackArray(expr) => {
                            self.resolve_expr(expr)?;
                        }
                    }
                }
            }
            Expr::CallValue { callee, args, .. } => {
                self.resolve_expr(callee)?;
                for arg in args {
                    match arg {
                        Arg::Positional(expr) => {
                            self.resolve_expr(expr)?;
                        }
                        Arg::Named { value, .. } => {
                            self.resolve_expr(value)?;
                        }
                        Arg::UnpackObject(expr) | Arg::UnpackArray(expr) => {
                            self.resolve_expr(expr)?;
                        }
                    }
                }
            }
            Expr::Lambda { params, body, .. } => {
                for param in params {
                    if let Some(ref default_expr) = param.default_value {
                        self.resolve_expr(default_expr)?;
                    }
                }
                self.begin_scope();
                for param in params {
                    self.declare(&param.name);
                    self.define(&param.name);
                }
                self.resolve_expr(body)?;
                self.end_scope();
            }
            Expr::Unary { right, .. } => {
                self.resolve_expr(right)?;
            }
            Expr::ArrayLiteral { elements, .. } => {
                for element in elements {
                    self.resolve_expr(element)?;
                }
            }
            Expr::ObjectLiteral { pairs, .. } => {
                for p in pairs {
                    match p {
                        crate::parser::ast::ObjectPair::KeyValue(_, value) => {
                            self.resolve_expr(value)?;
                        }
                        crate::parser::ast::ObjectPair::KeyValueExpr(key, value) => {
                            self.resolve_expr(key)?;
                            self.resolve_expr(value)?;
                        }
                        crate::parser::ast::ObjectPair::Spread(expr) => {
                            self.resolve_expr(expr)?;
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
                self.begin_scope();
                self.declare(loop_var);
                self.define(loop_var);
                self.resolve_expr(iterable)?;
                self.resolve_expr(key_expr)?;
                self.resolve_expr(value_expr)?;
                if let Some(c) = condition {
                    self.resolve_expr(c)?;
                }
                self.end_scope();
            }
            Expr::ListComprehension { elt, clauses, .. } => {
                self.begin_scope();
                for clause in clauses {
                    match clause {
                        ListComprehensionClause::For { pattern, iterable } => {
                            self.declare_unpack_pattern(pattern);
                            self.resolve_expr(iterable)?;
                        }
                        ListComprehensionClause::If { condition } => {
                            self.resolve_expr(condition)?;
                        }
                    }
                }
                self.resolve_expr(elt)?;
                self.end_scope();
            }
            Expr::TupleLiteral { elements, .. } => {
                for element in elements {
                    self.resolve_expr(element)?;
                }
            }
            Expr::UnpackAssign { targets, value, .. } => {
                self.resolve_expr(value)?;
                for target in targets {
                    match target {
                        AssignTarget::Name(name) => {
                            self.resolve_local(expr, name);
                        }
                        AssignTarget::Index { array, index } => {
                            self.resolve_expr(array)?;
                            self.resolve_expr(index)?;
                        }
                    }
                }
            }
            Expr::ArrayIndex { array, index, .. } => {
                self.resolve_expr(array)?;
                self.resolve_index_expr(index)?;
            }
            Expr::AssignArray {
                array,
                index,
                value,
                ..
            } => {
                self.resolve_expr(array)?;
                self.resolve_index_expr(index)?;
                self.resolve_expr(value)?;
            }
            Expr::AssignArrayOp {
                array,
                index,
                value,
                ..
            } => {
                self.resolve_expr(array)?;
                self.resolve_index_expr(index)?;
                self.resolve_expr(value)?;
            }
            Expr::TableFilter { table, predicate, .. } => {
                self.resolve_expr(table)?;
                self.resolve_table_filter_pred(predicate)?;
            }
            Expr::TableColumnWrite { inner, .. } => {
                self.resolve_expr(inner)?;
            }
            Expr::AssignTableColumn {
                table,
                column,
                value,
                ..
            } => {
                self.resolve_expr(table)?;
                self.resolve_expr(column)?;
                self.resolve_expr(value)?;
            }
            Expr::Property { object, .. } => {
                self.resolve_expr(object)?;
            }
            Expr::MethodCall { object, args, .. } => {
                self.resolve_expr(object)?;
                for arg in args {
                    match arg {
                        Arg::Positional(expr) => {
                            self.resolve_expr(expr)?;
                        }
                        Arg::Named { value, .. } => {
                            self.resolve_expr(value)?;
                        }
                        Arg::UnpackObject(expr) | Arg::UnpackArray(expr) => {
                            self.resolve_expr(expr)?;
                        }
                    }
                }
            }
            Expr::This { .. } => {
                // this разрешается во время компиляции как первый параметр функции
                // Проверка, что this используется только в методах/конструкторах, будет в компиляторе
            }
            Expr::Super { .. } => {
                // super разрешается во время компиляции
            }
            Expr::SuperCall { args, .. } => {
                for arg in args {
                    match arg {
                        Arg::Positional(expr) => {
                            self.resolve_expr(expr)?;
                        }
                        Arg::Named { value, .. } => {
                            self.resolve_expr(value)?;
                        }
                        Arg::UnpackObject(expr) | Arg::UnpackArray(expr) => {
                            self.resolve_expr(expr)?;
                        }
                    }
                }
            }
            Expr::SuperMethodCall { args, .. } => {
                for arg in args {
                    match arg {
                        Arg::Positional(expr) => {
                            self.resolve_expr(expr)?;
                        }
                        Arg::Named { value, .. } => {
                            self.resolve_expr(value)?;
                        }
                        Arg::UnpackObject(expr) | Arg::UnpackArray(expr) => {
                            self.resolve_expr(expr)?;
                        }
                    }
                }
            }
            Expr::Ellipsis { .. } => {}
            Expr::ExprReturn { value, line } => {
                if self.current_function != FunctionType::Stream {
                    return Err(LangError::SemanticError {
                        message:
                            "'return' as an expression is only allowed inside a stream fn body"
                                .to_string(),
                        line: *line,
                        file: self.source_name.clone(),
                    });
                }
                if let Some(e) = value {
                    self.resolve_expr(e)?;
                }
            }
            Expr::Ireturn { value, line } => {
                if self.current_function != FunctionType::Stream {
                    return Err(LangError::SemanticError {
                        message: "'ireturn' is only allowed inside a stream fn body".to_string(),
                        line: *line,
                        file: self.source_name.clone(),
                    });
                }
                if let Some(e) = value {
                    self.resolve_expr(e)?;
                }
            }
            Expr::InterpolatedString { segments, .. } => {
                use crate::parser::ast::InterpolatedSegment;
                for seg in segments {
                    if let InterpolatedSegment::Expr { expr: e, .. } = seg {
                        self.resolve_expr(e)?;
                    }
                }
            }
            Expr::If {
                condition,
                then_branch,
                else_branch,
                ..
            } => {
                self.resolve_expr(condition)?;
                match then_branch {
                    crate::parser::ast::IfBranch::Expr(e) => self.resolve_expr(e)?,
                    crate::parser::ast::IfBranch::Block(stmts) => {
                        self.resolve_stmt_block(stmts)?;
                    }
                }
                match else_branch {
                    crate::parser::ast::IfBranch::Expr(e) => self.resolve_expr(e)?,
                    crate::parser::ast::IfBranch::Block(stmts) => {
                        self.resolve_stmt_block(stmts)?;
                    }
                }
            }
        }
        Ok(())
    }

    fn resolve_index_expr(&mut self, index: &IndexExpr) -> Result<(), LangError> {
        match index {
            IndexExpr::Scalar(e) => self.resolve_expr(e),
            IndexExpr::Slice {
                start, stop, step, ..
            } => {
                if let Some(e) = start {
                    self.resolve_expr(e)?;
                }
                if let Some(e) = stop {
                    self.resolve_expr(e)?;
                }
                if let Some(e) = step {
                    self.resolve_expr(e)?;
                }
                Ok(())
            }
        }
    }

    fn resolve_function(
        &mut self,
        params: &[Param],
        body: &[Stmt],
        function_type: FunctionType,
    ) -> Result<(), LangError> {
        let enclosing_function = self.current_function;
        self.current_function = function_type;

        self.begin_scope();
        for param in params {
            self.declare(&param.name);
            self.define(&param.name);
        }
        self.resolve_stmt_block(body)?;
        self.end_scope();

        self.current_function = enclosing_function;
        Ok(())
    }

    fn begin_scope(&mut self) {
        let parent = self.scopes.last().map(|last| Box::new(last.clone()));
        let scope = if let Some(parent) = parent {
            Scope::with_parent(parent)
        } else {
            Scope::new()
        };
        self.scopes.push(scope);
    }

    fn end_scope(&mut self) {
        self.scopes.pop();
    }

    fn declare(&mut self, name: &str) {
        if let Some(scope) = self.scopes.last_mut() {
            if scope.locals.contains_key(name) {
                // Переменная уже объявлена в этой области
            }
            // Пока просто добавляем без индекса, индекс будет назначен при компиляции
            scope.locals.insert(name.to_string(), 0);
        }
    }

    fn define(&mut self, _name: &str) {
        // Переменная определена и готова к использованию
        // Индекс будет установлен компилятором
    }

    fn resolve_local(&mut self, _expr: &Expr, _name: &str) {
        // Разрешение локальной переменной
        // Реальная логика будет в компиляторе
        // Переменная будет найдена компилятором при генерации байт-кода
    }

    fn declare_unpack_pattern(&mut self, pattern: &[UnpackPattern]) {
        for pat in pattern {
            match pat {
                UnpackPattern::Variable(name) => {
                    self.declare(name);
                    self.define(name);
                }
                UnpackPattern::Wildcard => {
                    // Wildcard не создает переменную
                }
                UnpackPattern::Variadic(name) => {
                    // Variadic переменная создает переменную
                    self.declare(name);
                    self.define(name);
                }
                UnpackPattern::VariadicWildcard => {
                    // Variadic wildcard не создает переменную
                }
                UnpackPattern::Nested(nested) => {
                    // Рекурсивно обрабатываем вложенные паттерны
                    self.declare_unpack_pattern(nested);
                }
            }
        }
    }
}
