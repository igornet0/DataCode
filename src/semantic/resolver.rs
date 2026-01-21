// Разрешение переменных и подготовка к компиляции

use crate::parser::ast::{Expr, Stmt, Param, Arg, UnpackPattern};
use crate::common::error::LangError;
use crate::semantic::scope::Scope;

pub struct Resolver {
    scopes: Vec<Scope>,
    current_function: FunctionType,
}

#[derive(Clone, Copy, PartialEq)]
enum FunctionType {
    None,
    Function,
}

impl Resolver {
    pub fn new() -> Self {
        Self {
            scopes: Vec::new(),
            current_function: FunctionType::None,
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
            Stmt::Let { name, value, is_global, .. } => {
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
            Stmt::Function { name, params, body, .. } => {
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
            Stmt::If { condition, then_branch, else_branch, .. } => {
                self.resolve_expr(condition)?;
                self.resolve_stmt_block(then_branch)?;
                if let Some(else_branch) = else_branch {
                    self.resolve_stmt_block(else_branch)?;
                }
            }
            Stmt::While { condition, body, .. } => {
                self.resolve_expr(condition)?;
                self.resolve_stmt_block(body)?;
            }
            Stmt::Return { value, line } => {
                if self.current_function == FunctionType::None {
                    return Err(LangError::SemanticError {
                        message: "Cannot return from top-level code".to_string(),
                        line: *line,
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
            Stmt::For { pattern, iterable, body, .. } => {
                // Начинаем новую область видимости для цикла for
                self.begin_scope();
                
                // Объявляем переменные из паттерна распаковки
                self.declare_unpack_pattern(pattern);
                
                // Разрешаем итерируемое выражение
                self.resolve_expr(iterable)?;
                
                // Разрешаем тело цикла
                self.resolve_stmt_block(body)?;
                
                self.end_scope();
            }
            Stmt::Try { try_block, catch_blocks, else_block, finally_block, .. } => {
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

    fn resolve_expr(&mut self, expr: &Expr) -> Result<(), LangError> {
        match expr {
            Expr::Variable { name, line } => {
                if !self.scopes.is_empty() {
                    if let Some(scope) = self.scopes.last() {
                        if scope.locals.contains_key(name) && !scope.locals.contains_key(name) {
                            // Переменная объявлена, но еще не определена
                            return Err(LangError::SemanticError {
                                message: format!("Cannot read local variable '{}' in its own initializer", name),
                                line: *line,
                            });
                        }
                    }
                }
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
                    }
                }
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
                for (_, value) in pairs {
                    self.resolve_expr(value)?;
                }
            }
            Expr::TupleLiteral { elements, .. } => {
                for element in elements {
                    self.resolve_expr(element)?;
                }
            }
            Expr::UnpackAssign { names, value, .. } => {
                self.resolve_expr(value)?;
                for name in names {
                    self.resolve_local(expr, name);
                }
            }
            Expr::ArrayIndex { array, index, .. } => {
                self.resolve_expr(array)?;
                self.resolve_expr(index)?;
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
                    }
                }
            }
        }
        Ok(())
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
        let parent = if let Some(last) = self.scopes.last() {
            Some(Box::new(last.clone()))
        } else {
            None
        };
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

