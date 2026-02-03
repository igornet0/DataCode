// Recursive Descent Parser

use crate::lexer::{Token, TokenKind};
use crate::parser::ast::{Expr, InterpolatedSegment, Stmt, Param, Arg, UnpackPattern, ImportItem, ImportStmt, ClassVariable};
use crate::common::error::LangError;
use crate::common::value::Value;
use std::rc::Rc;
use std::cell::RefCell;

pub struct Parser {
    tokens: Vec<Token>,
    current: usize,
}

impl Parser {
    pub fn new(tokens: Vec<Token>) -> Self {
        Self { tokens, current: 0 }
    }

    pub fn parse(&mut self) -> Result<Vec<Stmt>, LangError> {
        let mut statements = Vec::new();
        while !self.is_at_end() {
            statements.push(self.declaration()?);
        }
        Ok(statements)
    }

    fn declaration(&mut self) -> Result<Stmt, LangError> {
        if self.check(TokenKind::From) {
            // from ml import ...
            self.from_import_declaration()
        } else if self.match_token(TokenKind::Import) {
            // import ml, plot
            self.simple_import_declaration()
        } else if self.match_token(TokenKind::Global) {
            // global a = 5
            let global_line = self.previous().line;
            let name = self.consume(TokenKind::Identifier, "Expect variable name after 'global'")?.lexeme.clone();
            self.consume(TokenKind::Equal, "Expect '=' after variable name")?;
            let value = self.expression()?;
            Ok(Stmt::Let { name, value, is_global: true, line: global_line })
        } else if self.match_token(TokenKind::Let) {
            self.variable_declaration()
        } else if self.check(TokenKind::At) {
            self.advance(); // consume @
            // @class in expression (e.g. inside method body: @class.name) — parse as statement, not decorator
            if self.check(TokenKind::Identifier) && self.peek().lexeme == "class" {
                self.current = self.current.saturating_sub(1); // put @ back so statement() parses @class...
                return self.statement();
            }
            // @Abstract cls ... or @cache / @route fn ...
            if self.check(TokenKind::Abstract) {
                self.advance(); // consume Abstract
                self.consume(TokenKind::Cls, "Expect 'cls' after @Abstract")?;
                self.class_declaration(true)
            } else {
                let (is_cached, route) = self.parse_function_decorators_after_at()?;
                self.consume(TokenKind::Fn, "Expect 'fn' after decorators")?;
                self.function_declaration_body(is_cached, route)
            }
        } else if self.match_token(TokenKind::Cls) {
            self.class_declaration(false)
        } else if self.check(TokenKind::Fn) {
            self.function_declaration()
        } else {
            self.statement()
        }
    }

    /// Parses a dotted module name: identifier ( . identifier )*
    fn parse_dotted_module_name(&mut self, context: &str) -> Result<String, LangError> {
        let mut parts = vec![self.consume(TokenKind::Identifier, context)?.lexeme.clone()];
        while self.match_token(TokenKind::Dot) {
            parts.push(self.consume(TokenKind::Identifier, "Expect identifier after '.' in module name")?.lexeme.clone());
        }
        Ok(parts.join("."))
    }

    fn from_import_declaration(&mut self) -> Result<Stmt, LangError> {
        let import_line = self.peek().line;
        self.consume(TokenKind::From, "Expect 'from'")?;
        let module = self.parse_dotted_module_name("Expect module name after 'from'")?;
        self.consume(TokenKind::Import, "Expect 'import' after module name")?;
        
        let items = self.parse_import_items()?;
        
        Ok(Stmt::Import {
            import_stmt: ImportStmt::From { module, items },
            line: import_line,
        })
    }

    fn simple_import_declaration(&mut self) -> Result<Stmt, LangError> {
        let import_line = self.previous().line; // 'import' был уже потреблен
        let mut modules = Vec::new();
        loop {
            let module = self.parse_dotted_module_name("Expect module name after 'import'")?;
            modules.push(module);
            
            if !self.match_token(TokenKind::Comma) {
                break;
            }
        }
        
        Ok(Stmt::Import {
            import_stmt: ImportStmt::Modules(modules),
            line: import_line,
        })
    }

    fn parse_import_items(&mut self) -> Result<Vec<ImportItem>, LangError> {
        let mut items = Vec::new();
        
        loop {
            if self.match_token(TokenKind::Star) {
                // import *
                items.push(ImportItem::All);
            } else if self.check(TokenKind::Identifier) {
                let name = self.advance().lexeme.clone();
                
                // Проверяем, есть ли 'as' для алиаса
                if self.match_token(TokenKind::As) {
                    let alias = self.consume(TokenKind::Identifier, "Expect alias name after 'as'")?.lexeme.clone();
                    items.push(ImportItem::Aliased { name, alias });
                } else {
                    items.push(ImportItem::Named(name));
                }
            } else {
                return Err(LangError::ParseError {
                    message: "Expect identifier or '*' in import list".to_string(),
                    line: self.peek().line,
                });
            }
            
            if !self.match_token(TokenKind::Comma) {
                break;
            }
        }
        
        Ok(items)
    }

    fn variable_declaration(&mut self) -> Result<Stmt, LangError> {
        let let_line = self.previous().line;
        let name = self.consume(TokenKind::Identifier, "Expect variable name")?.lexeme.clone();
        
        // Проверяем, есть ли запятая после имени - это распаковка кортежа
        if self.match_token(TokenKind::Comma) {
            // Распаковка: let a, b, c = ...
            let mut names = vec![name];
            loop {
                if self.match_token(TokenKind::Identifier) {
                    names.push(self.previous().lexeme.clone());
                } else {
                    return Err(LangError::ParseError {
                        message: "Expect variable name in unpack declaration".to_string(),
                        line: self.peek().line,
                    });
                }
                if !self.match_token(TokenKind::Comma) {
                    break;
                }
            }
            self.consume(TokenKind::Equal, "Expect '=' after variable list")?;
            let value = self.expression()?;
            // Для let statements с распаковкой создаем UnpackAssign выражение
            // Но нам нужно сохранить это как Stmt::Let с особым значением
            // Пока что создадим временное решение - используем первую переменную как имя
            // и создадим UnpackAssign в value
            return Ok(Stmt::Let {
                name: names[0].clone(),
                value: Expr::UnpackAssign {
                    names,
                    value: Box::new(value),
                    line: let_line,
                },
                is_global: false,
                line: let_line,
            });
        }
        
        self.consume(TokenKind::Equal, "Expect '=' after variable name")?;
        let value = self.expression()?;
        Ok(Stmt::Let { name, value, is_global: false, line: let_line })
    }

    /// Parse function decorators when we have already consumed '@' and current token is the first decorator name.
    fn parse_function_decorators_after_at(&mut self) -> Result<(bool, Option<(String, String)>), LangError> {
        let mut is_cached = false;
        let mut route: Option<(String, String)> = None;
        loop {
            if self.match_token(TokenKind::Cache) {
                is_cached = true;
            } else if self.check(TokenKind::Identifier) && self.peek().lexeme == "route" {
                self.advance(); // consume "route"
                self.consume(TokenKind::LParen, "Expect '(' after @route")?;
                let method_expr = self.expression()?;
                let method = Self::expr_to_string(&method_expr).ok_or_else(|| LangError::ParseError {
                    message: "@route first argument must be a string literal (e.g. \"GET\")".to_string(),
                    line: self.previous().line,
                })?;
                self.consume(TokenKind::Comma, "Expect ',' in @route(...)")?;
                let path_expr = self.expression()?;
                let path = Self::expr_to_string(&path_expr).ok_or_else(|| LangError::ParseError {
                    message: "@route second argument must be a string literal (e.g. \"/\")".to_string(),
                    line: self.previous().line,
                })?;
                self.consume(TokenKind::RParen, "Expect ')' after @route(...)")?;
                route = Some((method, path));
            } else {
                let got = if self.check(TokenKind::Identifier) {
                    self.peek().lexeme.clone()
                } else {
                    "non-identifier".to_string()
                };
                return Err(LangError::ParseError {
                    message: format!("Expect 'cache' or 'route(...)' after '@', got '{}'", got),
                    line: self.peek().line,
                });
            }
            if !self.match_token(TokenKind::At) {
                break;
            }
        }
        Ok((is_cached, route))
    }

    fn function_declaration(&mut self) -> Result<Stmt, LangError> {
        // Parse decorators: @cache and/or @route("METHOD", "/path")
        let mut is_cached = false;
        let mut route: Option<(String, String)> = None;
        while self.match_token(TokenKind::At) {
            if self.match_token(TokenKind::Cache) {
                is_cached = true;
            } else if self.check(TokenKind::Identifier) && self.peek().lexeme == "route" {
                self.advance();
                self.consume(TokenKind::LParen, "Expect '(' after @route")?;
                let method_expr = self.expression()?;
                let method = Self::expr_to_string(&method_expr).ok_or_else(|| LangError::ParseError {
                    message: "@route first argument must be a string literal (e.g. \"GET\")".to_string(),
                    line: self.previous().line,
                })?;
                self.consume(TokenKind::Comma, "Expect ',' in @route(...)")?;
                let path_expr = self.expression()?;
                let path = Self::expr_to_string(&path_expr).ok_or_else(|| LangError::ParseError {
                    message: "@route second argument must be a string literal (e.g. \"/\")".to_string(),
                    line: self.previous().line,
                })?;
                self.consume(TokenKind::RParen, "Expect ')' after @route(...)")?;
                route = Some((method, path));
            } else {
                let dec_name = if self.check(TokenKind::Identifier) {
                    self.peek().lexeme.clone()
                } else {
                    "".to_string()
                };
                return Err(LangError::ParseError {
                    message: format!("Expect 'cache' or 'route(...)' after '@', got '{}'", dec_name),
                    line: self.peek().line,
                });
            }
        }

        self.consume(TokenKind::Fn, "Expect 'fn'")?;
        self.function_declaration_body(is_cached, route)
    }

    /// Parse function name, params, return type and body. Caller must have consumed 'fn' so previous() is 'fn'.
    fn function_declaration_body(&mut self, is_cached: bool, route: Option<(String, String)>) -> Result<Stmt, LangError> {
        let fn_line = self.previous().line;
        let name = self.consume(TokenKind::Identifier, "Expect function name")?.lexeme.clone();
        self.consume(TokenKind::LParen, "Expect '(' after function name")?;

        let mut params = Vec::new();
        let mut has_default = false;
        if !self.check(TokenKind::RParen) {
            loop {
                if params.len() >= 255 {
                    return Err(LangError::ParseError {
                        message: "Cannot have more than 255 parameters".to_string(),
                        line: self.previous().line,
                    });
                }
                
                let param_name = self.consume(TokenKind::Identifier, "Expect parameter name")?.lexeme.clone();
                let param_line = self.previous().line;
                
                // Проверяем, есть ли аннотация типа (param: type)
                let type_annotation = if self.match_token(TokenKind::Colon) {
                    Some(self.parse_type_name()?)
                } else {
                    None
                };
                
                // Проверяем, есть ли значение по умолчанию
                let default_value = if self.match_token(TokenKind::Equal) {
                    has_default = true;
                    Some(self.expression()?)
                } else {
                    // Проверяем порядок: обязательный параметр не может идти после параметра с default
                    if has_default {
                        return Err(LangError::ParseError {
                            message: "Non-default argument follows default argument".to_string(),
                            line: param_line,
                        });
                    }
                    None
                };
                
                params.push(Param {
                    name: param_name,
                    type_annotation,
                    default_value,
                });
                
                if !self.match_token(TokenKind::Comma) {
                    break;
                }
            }
        }
        self.consume(TokenKind::RParen, "Expect ')' after parameters")?;
        
        // Проверяем, есть ли аннотация возвращаемого типа (-> type)
        let return_type = if self.match_token(TokenKind::Arrow) {
            Some(self.parse_type_name()?)
        } else {
            None
        };
        
        self.consume(TokenKind::LBrace, "Expect '{' before function body")?;

        let body = self.block()?;

        Ok(Stmt::Function { name, params, return_type, body, is_cached, route, line: fn_line })
    }

    /// Extract string value from a string literal expression (for @route("GET", "/")).
    fn expr_to_string(expr: &Expr) -> Option<String> {
        match expr {
            Expr::Literal { value: Value::String(s), .. } => Some(s.clone()),
            Expr::InterpolatedString { .. } => None, // interpolated string is not a literal
            _ => None,
        }
    }

    fn class_declaration(&mut self, is_abstract: bool) -> Result<Stmt, LangError> {
        let cls_line = self.previous().line;
        let name = self.consume(TokenKind::Identifier, "Expect class name after 'cls'")?.lexeme.clone();
        let superclass = if self.match_token(TokenKind::LParen) {
            let super_name = self.consume(TokenKind::Identifier, "Expect superclass name after '('")?.lexeme.clone();
            self.consume(TokenKind::RParen, "Expect ')' after superclass name")?;
            Some(super_name)
        } else {
            None
        };
        self.consume(TokenKind::LBrace, "Expect '{' after class name")?;
        
        let mut private_fields = Vec::new();
        let mut protected_fields = Vec::new();
        let mut public_fields = Vec::new();
        let mut private_variables = Vec::new();
        let mut protected_variables = Vec::new();
        let mut public_variables = Vec::new();
        let mut constructors = Vec::new();
        let mut methods = Vec::new();
        // None = private, Some(false) = protected, Some(true) = public
        let mut current_section_public: Option<bool> = None;
        
        while !self.check(TokenKind::RBrace) && !self.is_at_end() {
            if self.match_token(TokenKind::Private) {
                self.consume(TokenKind::Colon, "Expect ':' after 'private'")?;
                current_section_public = None;
                loop {
                    if self.is_at_end() {
                        break;
                    }
                    if self.check(TokenKind::Protected) || self.check(TokenKind::Public) || self.check(TokenKind::RBrace) || self.check(TokenKind::Fn) {
                        break;
                    }
                    if self.check(TokenKind::Identifier) && self.peek().lexeme == "new" {
                        break;
                    }
                    let saved_pos = self.current;
                    if self.check(TokenKind::Identifier) {
                        self.advance();
                        if self.check(TokenKind::LParen) {
                            self.current = saved_pos;
                            break;
                        }
                        self.current = saved_pos;
                    } else {
                        break;
                    }
                    if self.check_next(TokenKind::Colon) {
                        let field = self.parse_class_field()?;
                        private_fields.push(field);
                    } else if self.check_next(TokenKind::Equal) {
                        let var = self.parse_class_variable_assignment()?;
                        private_variables.push(var);
                    } else {
                        break;
                    }
                }
            } else if self.match_token(TokenKind::Protected) {
                self.consume(TokenKind::Colon, "Expect ':' after 'protected'")?;
                current_section_public = Some(false);
                loop {
                    if self.is_at_end() {
                        break;
                    }
                    if self.check(TokenKind::Private) || self.check(TokenKind::Public) || self.check(TokenKind::RBrace) || self.check(TokenKind::Fn) {
                        break;
                    }
                    if self.check(TokenKind::Identifier) && self.peek().lexeme == "new" {
                        break;
                    }
                    let saved_pos = self.current;
                    if self.check(TokenKind::Identifier) {
                        self.advance();
                        if self.check(TokenKind::LParen) {
                            self.current = saved_pos;
                            break;
                        }
                        self.current = saved_pos;
                    } else {
                        break;
                    }
                    if self.check_next(TokenKind::Colon) {
                        let field = self.parse_class_field()?;
                        protected_fields.push(field);
                    } else if self.check_next(TokenKind::Equal) {
                        let var = self.parse_class_variable_assignment()?;
                        protected_variables.push(var);
                    } else {
                        break;
                    }
                }
            } else if self.match_token(TokenKind::Public) {
                self.consume(TokenKind::Colon, "Expect ':' after 'public'")?;
                current_section_public = Some(true);
                loop {
                    if self.is_at_end() {
                        break;
                    }
                    if self.check(TokenKind::Private) || self.check(TokenKind::Protected) || self.check(TokenKind::RBrace) || self.check(TokenKind::Fn) {
                        break;
                    }
                    if self.check(TokenKind::Identifier) && self.peek().lexeme == "new" {
                        break;
                    }
                    let saved_pos = self.current;
                    if self.check(TokenKind::Identifier) {
                        self.advance();
                        if self.check(TokenKind::LParen) {
                            self.current = saved_pos;
                            break;
                        }
                        self.current = saved_pos;
                    } else {
                        break;
                    }
                    if self.check_next(TokenKind::Colon) {
                        let field = self.parse_class_field()?;
                        public_fields.push(field);
                    } else if self.check_next(TokenKind::Equal) {
                        let var = self.parse_class_variable_assignment()?;
                        public_variables.push(var);
                    } else {
                        break;
                    }
                }
            } else if self.check(TokenKind::Identifier) && self.check_next(TokenKind::Colon) {
                let field = self.parse_class_field()?;
                match current_section_public {
                    None => private_fields.push(field),
                    Some(false) => protected_fields.push(field),
                    Some(true) => public_fields.push(field),
                }
            } else if self.check(TokenKind::Identifier) && self.check_next(TokenKind::Equal) {
                let var = self.parse_class_variable_assignment()?;
                match current_section_public {
                    None => private_variables.push(var),
                    Some(false) => protected_variables.push(var),
                    Some(true) => public_variables.push(var),
                }
            } else if self.check(TokenKind::Identifier) && self.peek().lexeme == "new" {
                let constructor = self.parse_constructor(&name)?;
                constructors.push(constructor);
            } else if self.check(TokenKind::Fn) {
                let method = self.parse_method()?;
                methods.push(method);
            } else {
                return Err(LangError::ParseError {
                    message: "Expect 'private:', 'protected:', 'public:', field (name : type), class variable (name = expr), constructor, or method in class body".to_string(),
                    line: self.peek().line,
                });
            }
        }
        
        self.consume(TokenKind::RBrace, "Expect '}' after class body")?;
        
        Ok(Stmt::Class {
            name,
            superclass,
            is_abstract,
            private_fields,
            protected_fields,
            public_fields,
            private_variables,
            protected_variables,
            public_variables,
            constructors,
            methods,
            line: cls_line,
        })
    }

    fn parse_class_field(&mut self) -> Result<crate::parser::ast::ClassField, LangError> {
        let field_name = self.consume(TokenKind::Identifier, "Expect field name")?.lexeme.clone();
        let _field_line = self.previous().line;
        
        // Проверяем, есть ли аннотация типа
        let type_annotation = if self.match_token(TokenKind::Colon) {
            Some(self.parse_type_name()?)
        } else {
            None
        };
        
        // Проверяем, есть ли значение по умолчанию
        let default_value = if self.match_token(TokenKind::Equal) {
            Some(self.expression()?)
        } else {
            None
        };
        
        Ok(crate::parser::ast::ClassField {
            name: field_name,
            type_annotation,
            default_value,
        })
    }

    fn parse_class_variable_assignment(&mut self) -> Result<ClassVariable, LangError> {
        let name = self.consume(TokenKind::Identifier, "Expect variable name")?.lexeme.clone();
        self.consume(TokenKind::Equal, "Expect '=' after variable name in class variable")?;
        let value = self.expression()?;
        Ok(ClassVariable { name, value })
    }

    fn parse_constructor(&mut self, class_name: &str) -> Result<crate::parser::ast::Constructor, LangError> {
        // Парсим: new ClassName(...) { ... } или new ClassName(...) : this(...) {}
        let new_line = self.consume(TokenKind::Identifier, "Expect 'new'")?.line;
        // Проверяем, что это действительно 'new'
        if self.previous().lexeme != "new" {
            return Err(LangError::ParseError {
                message: "Expect 'new' for constructor".to_string(),
                line: new_line,
            });
        }
        
        // Парсим имя класса (должно совпадать с именем класса)
        let constructor_class_name = self.consume(TokenKind::Identifier, "Expect class name after 'new'")?.lexeme.clone();
        if constructor_class_name != class_name {
            return Err(LangError::ParseError {
                message: format!("Constructor class name '{}' must match class name '{}'", constructor_class_name, class_name),
                line: self.previous().line,
            });
        }
        
        self.consume(TokenKind::LParen, "Expect '(' after class name in constructor")?;
        
        let mut params = Vec::new();
        let mut has_default = false;
        if !self.check(TokenKind::RParen) {
            loop {
                if params.len() >= 255 {
                    return Err(LangError::ParseError {
                        message: "Cannot have more than 255 parameters".to_string(),
                        line: self.previous().line,
                    });
                }
                
                let param_name = self.consume(TokenKind::Identifier, "Expect parameter name")?.lexeme.clone();
                let param_line = self.previous().line;
                
                // Проверяем, есть ли аннотация типа
                let type_annotation = if self.match_token(TokenKind::Colon) {
                    Some(self.parse_type_name()?)
                } else {
                    None
                };
                
                // Проверяем, есть ли значение по умолчанию
                let default_value = if self.match_token(TokenKind::Equal) {
                    has_default = true;
                    Some(self.expression()?)
                } else {
                    if has_default {
                        return Err(LangError::ParseError {
                            message: "Non-default argument follows default argument".to_string(),
                            line: param_line,
                        });
                    }
                    None
                };
                
                params.push(crate::parser::ast::Param {
                    name: param_name,
                    type_annotation,
                    default_value,
                });
                
                if !self.match_token(TokenKind::Comma) {
                    break;
                }
            }
        }
        self.consume(TokenKind::RParen, "Expect ')' after parameters")?;
        
        // Проверяем, есть ли делегирующий конструктор: new ClassName(...) : this(...) {}
        let (body, delegate_args) = if self.match_token(TokenKind::Colon) {
            // Делегирующий конструктор
            if !self.match_token(TokenKind::This) {
                return Err(LangError::ParseError {
                    message: "Expect 'this' after ':' in delegating constructor".to_string(),
                    line: self.peek().line,
                });
            }
            self.consume(TokenKind::LParen, "Expect '(' after 'this'")?;
            let mut delegate_args = Vec::new();
            if !self.check(TokenKind::RParen) {
                loop {
                    delegate_args.push(self.expression()?);
                    if !self.match_token(TokenKind::Comma) {
                        break;
                    }
                }
            }
            self.consume(TokenKind::RParen, "Expect ')' after delegate arguments")?;
            self.consume(TokenKind::LBrace, "Expect '{' after delegating constructor")?;
            self.consume(TokenKind::RBrace, "Expect '}' after delegating constructor body")?;
            (Vec::new(), Some(delegate_args))
        } else {
            // Обычный конструктор
            self.consume(TokenKind::LBrace, "Expect '{' before constructor body")?;
            (self.block()?, None)
        };
        
        Ok(crate::parser::ast::Constructor {
            params,
            body,
            delegate_args,
            line: new_line,
        })
    }

    fn parse_method(&mut self) -> Result<crate::parser::ast::Method, LangError> {
        // Парсим: fn methodName(...) -> type? { ... }
        self.consume(TokenKind::Fn, "Expect 'fn'")?;
        let method_line = self.previous().line;
        let name = self.consume(TokenKind::Identifier, "Expect method name")?.lexeme.clone();
        self.consume(TokenKind::LParen, "Expect '(' after method name")?;
        
        let mut params = Vec::new();
        let mut has_default = false;
        if !self.check(TokenKind::RParen) {
            loop {
                if params.len() >= 255 {
                    return Err(LangError::ParseError {
                        message: "Cannot have more than 255 parameters".to_string(),
                        line: self.previous().line,
                    });
                }
                
                let param_name = if self.match_token(TokenKind::At) {
                    self.consume(TokenKind::Identifier, "Expect 'class' after '@'")?;
                    if self.previous().lexeme != "class" {
                        return Err(LangError::ParseError {
                            message: "After '@' only 'class' is allowed as parameter name".to_string(),
                            line: self.previous().line,
                        });
                    }
                    if !params.is_empty() {
                        return Err(LangError::ParseError {
                            message: "@class can only be the first parameter of a method".to_string(),
                            line: self.previous().line,
                        });
                    }
                    "@class".to_string()
                } else {
                    self.consume(TokenKind::Identifier, "Expect parameter name")?.lexeme.clone()
                };
                let param_line = self.previous().line;
                
                // Проверяем, есть ли аннотация типа
                let type_annotation = if self.match_token(TokenKind::Colon) {
                    Some(self.parse_type_name()?)
                } else {
                    None
                };
                
                // Проверяем, есть ли значение по умолчанию
                let default_value = if self.match_token(TokenKind::Equal) {
                    has_default = true;
                    Some(self.expression()?)
                } else {
                    if has_default {
                        return Err(LangError::ParseError {
                            message: "Non-default argument follows default argument".to_string(),
                            line: param_line,
                        });
                    }
                    None
                };
                
                params.push(crate::parser::ast::Param {
                    name: param_name,
                    type_annotation,
                    default_value,
                });
                
                if !self.match_token(TokenKind::Comma) {
                    break;
                }
            }
        }
        self.consume(TokenKind::RParen, "Expect ')' after parameters")?;
        
        // Проверяем, есть ли аннотация возвращаемого типа
        let return_type = if self.match_token(TokenKind::Arrow) {
            Some(self.parse_type_name()?)
        } else {
            None
        };
        
        self.consume(TokenKind::LBrace, "Expect '{' before method body")?;
        let body = self.block()?;
        
        Ok(crate::parser::ast::Method {
            name,
            params,
            return_type,
            body,
            line: method_line,
        })
    }

    fn statement(&mut self) -> Result<Stmt, LangError> {
        if self.match_token(TokenKind::If) {
            self.if_statement()
        } else if self.match_token(TokenKind::While) {
            self.while_statement()
        } else if self.match_token(TokenKind::For) {
            self.for_statement()
        } else if self.match_token(TokenKind::Return) {
            self.return_statement()
        } else if self.match_token(TokenKind::Break) {
            self.break_statement()
        } else if self.match_token(TokenKind::Continue) {
            self.continue_statement()
        } else if self.match_token(TokenKind::Throw) {
            self.throw_statement()
        } else if self.match_token(TokenKind::Try) {
            self.try_statement()
        } else {
            self.expression_statement()
        }
    }
    

    fn if_statement(&mut self) -> Result<Stmt, LangError> {
        let if_line = self.previous().line;
        // Скобки опциональны - парсим условие как выражение
        // Если есть скобки, они будут частью выражения, а не синтаксиса if
        let condition = self.expression()?;
        // Проверяем, есть ли скобки вокруг условия (опциональные)
        // Если следующая лексема - это '{', значит скобок не было
        // Если следующая лексема - это ')', значит были скобки, пропускаем их
        if self.match_token(TokenKind::RParen) {
            // Были скобки, пропустили закрывающую
        }
        self.consume(TokenKind::LBrace, "Expect '{' after condition")?;
        let then_branch = self.block()?;
        
        let else_branch = if self.match_token(TokenKind::Else) {
            // Проверяем, является ли следующий токен 'if' (else if)
            if self.check(TokenKind::If) {
                // Потребляем 'if' и рекурсивно парсим if_statement для else if
                self.advance(); // Потребляем токен 'if'
                Some(vec![self.if_statement()?])
            } else {
                // Обычный else блок
                self.consume(TokenKind::LBrace, "Expect '{' after 'else'")?;
                Some(self.block()?)
            }
        } else {
            None
        };

        Ok(Stmt::If {
            condition,
            then_branch,
            else_branch,
            line: if_line,
        })
    }

    fn while_statement(&mut self) -> Result<Stmt, LangError> {
        let while_line = self.previous().line;
        // Скобки опциональны - парсим условие как выражение
        let condition = self.expression()?;
        // Проверяем, есть ли скобки вокруг условия (опциональные)
        if self.match_token(TokenKind::RParen) {
            // Были скобки, пропустили закрывающую
        }
        self.consume(TokenKind::LBrace, "Expect '{' after condition")?;
        let body = self.block()?;
        Ok(Stmt::While { condition, body, line: while_line })
    }

    fn for_statement(&mut self) -> Result<Stmt, LangError> {
        let for_line = self.previous().line;
        
        // Парсим паттерн распаковки: for pattern in iterable { body }
        // Поддерживаем: for x in, for x, y in, for (x, y) in, for [x, y] in, for x, _, y in
        let pattern = self.parse_unpack_pattern()?;
        self.consume(TokenKind::In, "Expect 'in' after unpack pattern")?;
        let iterable = self.expression()?;
        self.consume(TokenKind::LBrace, "Expect '{' before loop body")?;
        let body = self.block()?;

        Ok(Stmt::For {
            pattern,
            iterable,
            body,
            line: for_line,
        })
    }

    fn parse_unpack_pattern(&mut self) -> Result<Vec<UnpackPattern>, LangError> {
        // Проверяем, есть ли группировка (скобки или квадратные скобки)
        if self.match_token(TokenKind::LParen) {
            // for (x, y) in или for (x, (y, z)) in
            let pattern = self.parse_unpack_pattern_list()?;
            self.consume(TokenKind::RParen, "Expect ')' after unpack pattern")?;
            Ok(pattern)
        } else if self.match_token(TokenKind::LBracket) {
            // for [x, y] in
            let pattern = self.parse_unpack_pattern_list()?;
            self.consume(TokenKind::RBracket, "Expect ']' after unpack pattern")?;
            Ok(pattern)
        } else {
            // for x, y, z in или for x in (обратная совместимость)
            self.parse_unpack_pattern_list()
        }
    }

    fn parse_unpack_pattern_list(&mut self) -> Result<Vec<UnpackPattern>, LangError> {
        let mut patterns = Vec::new();
        let mut has_variadic = false;
        
        loop {
            // Проверяем, есть ли звездочка для variadic
            let is_variadic = self.match_token(TokenKind::Star);
            
            // Парсим один элемент паттерна
            if self.match_token(TokenKind::Identifier) {
                let name = self.previous().lexeme.clone();
                
                if is_variadic {
                    // Variadic переменная или wildcard
                    if has_variadic {
                        return Err(LangError::ParseError {
                            message: "Only one variadic variable (*) allowed in unpack pattern".to_string(),
                            line: self.previous().line,
                        });
                    }
                    has_variadic = true;
                    
                    if name == "_" {
                        // Variadic wildcard (*_)
                        patterns.push(UnpackPattern::VariadicWildcard);
                    } else {
                        // Variadic переменная (*y)
                        patterns.push(UnpackPattern::Variadic(name));
                    }
                } else {
                    // Обычная переменная или wildcard
                    if name == "_" {
                        patterns.push(UnpackPattern::Wildcard);
                    } else {
                        patterns.push(UnpackPattern::Variable(name));
                    }
                }
            } else if self.match_token(TokenKind::LParen) {
                // Вложенная распаковка: (x, y)
                // Variadic не поддерживается во вложенных паттернах на первом этапе
                if is_variadic {
                    return Err(LangError::ParseError {
                        message: "Variadic unpacking (*) not supported in nested patterns".to_string(),
                        line: self.previous().line,
                    });
                }
                let nested = self.parse_unpack_pattern_list()?;
                self.consume(TokenKind::RParen, "Expect ')' after nested unpack pattern")?;
                patterns.push(UnpackPattern::Nested(nested));
            } else if self.match_token(TokenKind::LBracket) {
                // Вложенная распаковка: [x, y]
                // Variadic не поддерживается во вложенных паттернах на первом этапе
                if is_variadic {
                    return Err(LangError::ParseError {
                        message: "Variadic unpacking (*) not supported in nested patterns".to_string(),
                        line: self.previous().line,
                    });
                }
                let nested = self.parse_unpack_pattern_list()?;
                self.consume(TokenKind::RBracket, "Expect ']' after nested unpack pattern")?;
                patterns.push(UnpackPattern::Nested(nested));
            } else {
                // Ошибка: ожидается переменная, wildcard или вложенный паттерн
                if is_variadic {
                    return Err(LangError::ParseError {
                        message: "Expect variable name after '*' in unpack pattern".to_string(),
                        line: self.peek().line,
                    });
                }
                return Err(LangError::ParseError {
                    message: "Expect variable name, '_', '*', or nested pattern in unpack pattern".to_string(),
                    line: self.peek().line,
                });
            }
            
            // Проверяем, есть ли еще элементы (запятая)
            if !self.match_token(TokenKind::Comma) {
                break;
            }
            
            // Если уже есть variadic, нельзя добавлять больше элементов после него
            if has_variadic {
                return Err(LangError::ParseError {
                    message: "Variadic variable (*) must be the last element in unpack pattern".to_string(),
                    line: self.previous().line,
                });
            }
        }
        
        if patterns.is_empty() {
            return Err(LangError::ParseError {
                message: "Unpack pattern cannot be empty".to_string(),
                line: self.peek().line,
            });
        }
        
        Ok(patterns)
    }

    fn return_statement(&mut self) -> Result<Stmt, LangError> {
        let return_line = self.previous().line;
        let value = if !self.check(TokenKind::Semicolon) && !self.check(TokenKind::RBrace) {
            // Парсим первое выражение
            let first_expr = self.expression()?;
            
            // Проверяем, есть ли запятая - это означает множественный возврат
            if self.match_token(TokenKind::Comma) {
                // Множественный возврат: return a, b, c
                let mut elements = vec![first_expr];
                loop {
                    // Проверяем, не конец ли выражения (semicolon или RBrace)
                    if self.check(TokenKind::Semicolon) || self.check(TokenKind::RBrace) {
                        break;
                    }
                    elements.push(self.expression()?);
                    if !self.match_token(TokenKind::Comma) {
                        break;
                    }
                }
                Some(Expr::TupleLiteral { elements, line: return_line })
            } else {
                // Одиночный возврат
                Some(first_expr)
            }
        } else {
            None
        };
        // Семиколон опционален для return
        self.match_token(TokenKind::Semicolon);
        Ok(Stmt::Return { value, line: return_line })
    }

    fn break_statement(&mut self) -> Result<Stmt, LangError> {
        let break_line = self.previous().line;
        // Семиколон опционален для break
        self.match_token(TokenKind::Semicolon);
        Ok(Stmt::Break { line: break_line })
    }

    fn continue_statement(&mut self) -> Result<Stmt, LangError> {
        let continue_line = self.previous().line;
        // Семиколон опционален для continue
        self.match_token(TokenKind::Semicolon);
        Ok(Stmt::Continue { line: continue_line })
    }

    fn throw_statement(&mut self) -> Result<Stmt, LangError> {
        let throw_line = self.previous().line;
        // Парсим выражение (значение ошибки)
        let value = self.expression()?;
        // Семиколон опционален для throw
        self.match_token(TokenKind::Semicolon);
        Ok(Stmt::Throw { value, line: throw_line })
    }

    fn try_statement(&mut self) -> Result<Stmt, LangError> {
        use crate::parser::ast::CatchBlock;
        
        let try_line = self.previous().line;
        
        // Парсим try блок
        self.consume(TokenKind::LBrace, "Expect '{' after 'try'")?;
        let try_block = self.block()?;
        
        // Парсим catch блоки (опционально, но должен быть хотя бы один catch или finally)
        let mut catch_blocks = Vec::new();
        
        while self.match_token(TokenKind::Catch) {
            let catch_line = self.previous().line;
            
            // Парсим тип ошибки (опционально)
            let error_type = if self.check(TokenKind::Identifier) {
                let error_type_name = self.peek().lexeme.clone();
                // Проверяем, является ли это типом ошибки
                if crate::common::error::ErrorType::from_name(&error_type_name).is_some() {
                    self.advance();
                    Some(error_type_name)
                } else {
                    None
                }
            } else {
                None
            };
            
            // Парсим переменную ошибки (опционально): "as e" или просто "e"
            let error_var = if self.match_token(TokenKind::As) {
                Some(self.consume(TokenKind::Identifier, "Expect variable name after 'as'")?.lexeme.clone())
            } else if self.match_token(TokenKind::Identifier) {
                Some(self.previous().lexeme.clone())
            } else {
                None
            };
            
            // Парсим тело catch блока
            self.consume(TokenKind::LBrace, "Expect '{' after 'catch'")?;
            let catch_body = self.block()?;
            
            catch_blocks.push(CatchBlock {
                error_type,
                error_var,
                body: catch_body,
                line: catch_line,
            });
        }
        
        // Парсим else блок (опционально)
        let else_block = if self.match_token(TokenKind::Else) {
            self.consume(TokenKind::LBrace, "Expect '{' after 'else'")?;
            Some(self.block()?)
        } else {
            None
        };
        
        // Парсим finally блок (опционально)
        let finally_block = if self.match_token(TokenKind::Finally) {
            self.consume(TokenKind::LBrace, "Expect '{' after 'finally'")?;
            Some(self.block()?)
        } else {
            None
        };
        
        // Проверяем, что есть хотя бы один catch блок или finally блок
        if catch_blocks.is_empty() && finally_block.is_none() {
            return Err(LangError::ParseError {
                message: "try statement must have at least one catch block or finally block".to_string(),
                line: try_line,
            });
        }
        
        Ok(Stmt::Try {
            try_block,
            catch_blocks,
            else_block,
            finally_block,
            line: try_line,
        })
    }

    fn expression_statement(&mut self) -> Result<Stmt, LangError> {
        let expr = self.expression()?;
        let line = expr.line();
        // Семиколон опционален для выражений
        self.match_token(TokenKind::Semicolon);
        Ok(Stmt::Expr { expr, line })
    }

    fn block(&mut self) -> Result<Vec<Stmt>, LangError> {
        let mut statements = Vec::new();
        while !self.check(TokenKind::RBrace) && !self.is_at_end() {
            statements.push(self.declaration()?);
        }
        self.consume(TokenKind::RBrace, "Expect '}' after block")?;
        Ok(statements)
    }

    fn expression(&mut self) -> Result<Expr, LangError> {
        self.assignment()
    }

    /// Parse a single expression from the current token stream; expects only that expression then Eof.
    pub fn parse_single_expression(&mut self) -> Result<Expr, LangError> {
        let expr = self.expression()?;
        if !self.is_at_end() {
            return Err(LangError::ParseError {
                message: format!("Expected end of expression in interpolation, found {:?}", self.peek().kind),
                line: self.peek().line,
            });
        }
        Ok(expr)
    }

    /// Parse one expression from source string (used for "${...}" contents).
    fn parse_expression_from_source(source: &str, _line: usize) -> Result<Expr, LangError> {
        let mut lexer = crate::lexer::Lexer::new(source);
        let tokens = lexer.tokenize()?;
        let mut sub_parser = Parser::new(tokens);
        sub_parser.parse_single_expression()
    }

    /// Unescape literal segments: lexer \$ pushes placeholder \u{E000}; we replace it with "$" (the "{" is already in the string)
    fn unescape_literal(s: &str) -> String {
        s.replace('\u{E000}', "$")
    }

    /// Split string content into interpolation segments; returns segments or error if unclosed "${".
    fn parse_interpolated_segments(raw: &str, line: usize) -> Result<Vec<InterpolatedSegment>, LangError> {
        let mut segments = Vec::new();
        let bytes = raw.as_bytes();
        let mut literal_start = 0;
        loop {
            match raw[literal_start..].find("${") {
                None => {
                    let lit = Self::unescape_literal(&raw[literal_start..]);
                    if !lit.is_empty() {
                        segments.push(InterpolatedSegment::Literal(lit));
                    }
                    break;
                }
                Some(rel) => {
                    let pos = literal_start + rel;
                    if pos > 0 && bytes[pos - 1] == b'\\' {
                        literal_start = pos + 1;
                        continue;
                    }
                    let lit = Self::unescape_literal(&raw[literal_start..pos]);
                    if !lit.is_empty() {
                        segments.push(InterpolatedSegment::Literal(lit));
                    }
                    let mut depth: i32 = 1;
                    let mut end_byte = None;
                    for (byte_off, c) in raw[pos + 2..].char_indices() {
                        let abs_byte = pos + 2 + byte_off;
                        match c {
                            '{' => depth += 1,
                            '}' => {
                                depth -= 1;
                                if depth == 0 {
                                    end_byte = Some(abs_byte);
                                    break;
                                }
                            }
                            _ => {}
                        }
                    }
                    let end_byte = end_byte.ok_or_else(|| LangError::ParseError {
                        message: format!("Unclosed interpolation at line {}", line),
                        line,
                    })?;
                    let expr_source = raw[pos + 2..end_byte].trim();
                    let expr = Self::parse_expression_from_source(expr_source, line)?;
                    segments.push(InterpolatedSegment::Expr(Box::new(expr)));
                    literal_start = end_byte + 1;
                }
            }
        }
        Ok(segments)
    }

    fn assignment(&mut self) -> Result<Expr, LangError> {
        // Проверяем, не является ли это распаковкой кортежа БЕЗ let: x, y = ...
        // Это нужно проверить ДО вызова or_expression(), чтобы избежать ошибки при парсинге запятой
        // Используем peek() чтобы не потреблять токен
        if !self.is_at_end() {
            let token0 = self.peek();
            if token0.kind == TokenKind::Identifier {
                let saved_position = self.current;
                if saved_position + 3 < self.tokens.len() {
                    if self.tokens[saved_position + 1].kind == TokenKind::Comma
                        && self.tokens[saved_position + 2].kind == TokenKind::Identifier
                        && self.tokens[saved_position + 3].kind == TokenKind::Equal
                    {
                        // Проверяем, не находимся ли мы внутри вызова функции (внутри скобок)
                        let mut paren_count = 0;
                        let mut found_lparen = false;
                        for i in (0..saved_position).rev() {
                            match self.tokens[i].kind {
                                TokenKind::RParen => paren_count += 1,
                                TokenKind::LParen => {
                                    if paren_count == 0 {
                                        found_lparen = true;
                                        break;
                                    }
                                    if paren_count > 0 {
                                        paren_count -= 1;
                                    }
                                }
                                _ => {}
                            }
                        }
                        
                        // Если мы внутри скобок, не проверяем распаковку
                        if !found_lparen {
                            // Проверяем, не является ли это именованным аргументом функции
                            // Именованный аргумент: identifier, identifier = identifier
                            let is_named_arg = saved_position + 4 < self.tokens.len()
                                && self.tokens[saved_position + 4].kind == TokenKind::Identifier;
                            
                            if !is_named_arg {
                                // Это распаковка - обрабатываем ее напрямую
                                let line = token0.line;
                                let mut names = Vec::new();
                                
                                // Собираем список переменных
                                loop {
                                    let name = self.consume(TokenKind::Identifier, "Expect variable name")?.lexeme.clone();
                                    names.push(name);
                                    
                                    if !self.match_token(TokenKind::Comma) {
                                        break;
                                    }
                                }
                                
                                // Потребляем =
                                self.consume(TokenKind::Equal, "Expect '=' after variable list")?;
                                
                                // Парсим правую часть
                                let value = self.assignment()?;
                                
                                return Ok(Expr::UnpackAssign {
                                    names,
                                    value: Box::new(value),
                                    line,
                                });
                            }
                        }
                    }
                }
            }
        }
        
        let expr = self.or_expression()?;
        
        // Проверяем операторы присваивания (+=, -=, *=, /=, //=, %=, **=)
        if self.match_token(TokenKind::PlusEqual)
            || self.match_token(TokenKind::MinusEqual)
            || self.match_token(TokenKind::StarEqual)
            || self.match_token(TokenKind::StarStarEqual)
            || self.match_token(TokenKind::SlashEqual)
            || self.match_token(TokenKind::SlashSlashEqual)
            || self.match_token(TokenKind::PercentEqual)
        {
            let op_line = self.previous().line;
            let op_kind = self.previous().kind.clone();
            if let Expr::Variable { name, .. } = expr {
                let value = self.assignment()?;
                return Ok(Expr::AssignOp {
                    name,
                    op: op_kind,
                    value: Box::new(value),
                    line: op_line,
                });
            } else if let Expr::Property { object, name, .. } = expr {
                // Присваивание к свойству объекта: obj.field += value
                let value = self.assignment()?;
                let property_path = match &*object {
                    Expr::Variable { name: var_name, .. } => format!("{}.{}", var_name, name),
                    Expr::This { .. } => format!("this.{}", name),
                    Expr::Property { object, name: prop_name, .. } => {
                        // Рекурсивно строим путь к свойству
                        let base = match &**object {
                            Expr::Variable { name, .. } => name.clone(),
                            Expr::This { .. } => "this".to_string(),
                            _ => return Err(LangError::ParseError {
                                message: "Invalid assignment target".to_string(),
                                line: op_line,
                            }),
                        };
                        format!("{}.{}.{}", base, prop_name, name)
                    },
                    _ => return Err(LangError::ParseError {
                        message: "Invalid assignment target".to_string(),
                        line: op_line,
                    }),
                };
                return Ok(Expr::AssignOp {
                    name: property_path,
                    op: op_kind,
                    value: Box::new(value),
                    line: op_line,
                });
            }
            return Err(LangError::ParseError {
                message: "Invalid assignment target".to_string(),
                line: op_line,
            });
        }
        
        // Проверяем, является ли это распаковкой: a, b, c = ...
        // Это должно быть обработано ДО проверки на =
        if let Expr::Variable { name, .. } = &expr {
            if self.check(TokenKind::Comma) {
                // Сохраняем позицию для возможного отката
                let saved_position = self.current;
                
                // Проверяем, не находимся ли мы внутри вызова функции (внутри скобок)
                // Если мы внутри скобок, то это не распаковка, а часть списка аргументов
                // Ищем открывающую скобку перед текущей позицией
                let mut paren_count = 0;
                let mut found_lparen = false;
                for i in (0..saved_position).rev() {
                    match self.tokens[i].kind {
                        TokenKind::RParen => {
                            paren_count += 1;
                        }
                        TokenKind::LParen => {
                            if paren_count == 0 {
                                // Нашли открывающую скобку - мы внутри скобок
                                found_lparen = true;
                                break;
                            }
                            if paren_count > 0 {
                                paren_count -= 1;
                            }
                        }
                        _ => {}
                    }
                }
                
                // Если мы внутри скобок, не проверяем распаковку
                // Это часть списка аргументов функции или группировки выражений
                if found_lparen {
                    return Ok(expr);
                }
                
                // Если мы уже проверили, что мы внутри скобок, то дальше проверять не нужно
                // Но если мы не внутри скобок, проверяем, не является ли это именованным аргументом функции
                // Паттерн: identifier, identifier = identifier (где после = идет не вызов функции)
                // Это именованные аргументы функции, а не распаковка
                // Именованный аргумент: func(x=1, y=2) - после = идет значение (число, строка, идентификатор без скобок)
                // Распаковка: x, y = swap(x, y) - после = идет вызов функции (идентификатор со скобками)
                if saved_position + 3 < self.tokens.len() {
                    let has_identifier_after_comma = self.tokens[saved_position + 1].kind == TokenKind::Identifier;
                    let has_equal = self.tokens[saved_position + 2].kind == TokenKind::Equal;
                    
                    if has_identifier_after_comma && has_equal {
                        // Если после запятой идет идентификатор и =, это может быть именованный аргумент или распаковка
                        // Проверяем, является ли это именованным аргументом (после = идет значение, не вызов функции)
                        // Именованный аргумент: после = идет значение без скобок
                        // Распаковка: после = идет вызов функции (идентификатор со скобками)
                        if saved_position + 3 < self.tokens.len() {
                            let after_equal = &self.tokens[saved_position + 3];
                            // Если после = идет НЕ идентификатор или идентификатор без скобок, это именованный аргумент
                            let is_named_arg = after_equal.kind != TokenKind::Identifier
                                || (saved_position + 4 >= self.tokens.len() || self.tokens[saved_position + 4].kind != TokenKind::LParen);
                            
                            if is_named_arg {
                                // Это именованный аргумент функции, не распаковка - просто возвращаем выражение
                                return Ok(expr);
                            }
                        }
                    }
                }
                
                // Это потенциальная распаковка: a, b, c = ...
                // Упрощенная проверка: если после запятой идет идентификатор, а затем =, то это распаковка
                // НО только если после = идет вызов функции (идентификатор со скобками)
                if saved_position + 2 < self.tokens.len() {
                    let has_identifier_after_comma = self.tokens[saved_position + 1].kind == TokenKind::Identifier;
                    let has_equal_after_identifier = self.tokens[saved_position + 2].kind == TokenKind::Equal;
                    
                    if has_identifier_after_comma && has_equal_after_identifier {
                        // Проверяем, что после = идет вызов функции (идентификатор со скобками)
                        // Это отличает распаковку от именованного аргумента
                        if saved_position + 3 < self.tokens.len() {
                            let after_equal = &self.tokens[saved_position + 3];
                            if after_equal.kind == TokenKind::Identifier {
                                // Проверяем, есть ли после идентификатора скобка
                                if saved_position + 4 < self.tokens.len() 
                                    && self.tokens[saved_position + 4].kind == TokenKind::LParen {
                                    // Это распаковка - обрабатываем ее
                                    let mut names = vec![name.clone()];
                                    self.advance(); // consume comma
                                    
                                    // Собираем список переменных
                                    loop {
                                        if self.match_token(TokenKind::Identifier) {
                                            names.push(self.previous().lexeme.clone());
                                        } else {
                                            // Это не распаковка - восстанавливаем позицию
                                            self.current = saved_position;
                                            break;
                                        }
                                        if !self.match_token(TokenKind::Comma) {
                                            break;
                                        }
                                    }
                                    
                                    // Проверяем, что после списка переменных идет =
                                    if self.check(TokenKind::Equal) {
                                        // Это действительно распаковка
                                        self.consume(TokenKind::Equal, "Expect '=' after variable list")?;
                                        let value = self.assignment()?;
                                        return Ok(Expr::UnpackAssign {
                                            names,
                                            value: Box::new(value),
                                            line: expr.line(),
                                        });
                                    } else {
                                        // Это не распаковка - восстанавливаем позицию и возвращаем исходное выражение
                                        self.current = saved_position;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        
        // Обычное присваивание (=)
        if self.match_token(TokenKind::Equal) {
            let equal_line = self.previous().line;
            if let Expr::Variable { name, .. } = expr {
                // Обычное присваивание одной переменной
                let value = self.assignment()?;
                return Ok(Expr::Assign {
                    name,
                    value: Box::new(value),
                    line: equal_line,
                });
            } else if let Expr::Property { object, name, .. } = expr {
                // Присваивание к свойству объекта: obj.field = value или this.field = value
                let value = self.assignment()?;
                // Для присваивания к свойству создаем специальное выражение
                // Используем формат "object.field" для имени, где object может быть "this" или именем переменной
                let property_path = match &*object {
                    Expr::Variable { name: var_name, .. } => format!("{}.{}", var_name, name),
                    Expr::This { .. } => format!("this.{}", name),
                    Expr::Property { object, name: prop_name, .. } => {
                        // Рекурсивно строим путь к свойству
                        let base = match &**object {
                            Expr::Variable { name, .. } => name.clone(),
                            Expr::This { .. } => "this".to_string(),
                            _ => return Err(LangError::ParseError {
                                message: "Invalid assignment target".to_string(),
                                line: equal_line,
                            }),
                        };
                        format!("{}.{}.{}", base, prop_name, name)
                    },
                    _ => return Err(LangError::ParseError {
                        message: "Invalid assignment target".to_string(),
                        line: equal_line,
                    }),
                };
                return Ok(Expr::Assign {
                    name: property_path,
                    value: Box::new(value),
                    line: equal_line,
                });
            }
            return Err(LangError::ParseError {
                message: "Invalid assignment target".to_string(),
                line: equal_line,
            });
        }
        
        Ok(expr)
    }

    fn or_expression(&mut self) -> Result<Expr, LangError> {
        let mut expr = self.and_expression()?;
        while self.match_token(TokenKind::Or) {
            let op_line = self.previous().line;
            let op_kind = self.previous().kind.clone();
            let right = self.and_expression()?;
            expr = Expr::Binary {
                left: Box::new(expr),
                op: op_kind,
                right: Box::new(right),
                line: op_line,
            };
        }
        Ok(expr)
    }

    fn and_expression(&mut self) -> Result<Expr, LangError> {
        let mut expr = self.equality()?;
        while self.match_token(TokenKind::And) {
            let op_line = self.previous().line;
            let op_kind = self.previous().kind.clone();
            let right = self.equality()?;
            expr = Expr::Binary {
                left: Box::new(expr),
                op: op_kind,
                right: Box::new(right),
                line: op_line,
            };
        }
        Ok(expr)
    }

    fn equality(&mut self) -> Result<Expr, LangError> {
        let mut expr = self.comparison()?;
        while self.match_token(TokenKind::BangEqual) || self.match_token(TokenKind::EqualEqual) {
            let op_line = self.previous().line;
            let op_kind = self.previous().kind.clone();
            let right = self.comparison()?;
            expr = Expr::Binary {
                left: Box::new(expr),
                op: op_kind,
                right: Box::new(right),
                line: op_line,
            };
        }
        Ok(expr)
    }

    fn comparison(&mut self) -> Result<Expr, LangError> {
        let mut expr = self.term()?;
        while self.match_token(TokenKind::Greater)
            || self.match_token(TokenKind::GreaterEqual)
            || self.match_token(TokenKind::Less)
            || self.match_token(TokenKind::LessEqual)
            || self.match_token(TokenKind::In)
        {
            let op_line = self.previous().line;
            let op_kind = self.previous().kind.clone();
            let right = self.term()?;
            expr = Expr::Binary {
                left: Box::new(expr),
                op: op_kind,
                right: Box::new(right),
                line: op_line,
            };
        }
        Ok(expr)
    }

    fn term(&mut self) -> Result<Expr, LangError> {
        let mut expr = self.factor()?;
        while self.match_token(TokenKind::Minus) || self.match_token(TokenKind::Plus) {
            let op_line = self.previous().line;
            let op_kind = self.previous().kind.clone();
            let right = self.factor()?;
            expr = Expr::Binary {
                left: Box::new(expr),
                op: op_kind,
                right: Box::new(right),
                line: op_line,
            };
        }
        Ok(expr)
    }

    fn factor(&mut self) -> Result<Expr, LangError> {
        let mut expr = self.exponent()?;
        while self.match_token(TokenKind::Slash) || self.match_token(TokenKind::SlashSlash) || self.match_token(TokenKind::Star) || self.match_token(TokenKind::Percent) {
            let op_line = self.previous().line;
            let op_kind = self.previous().kind.clone();
            let right = self.exponent()?;
            expr = Expr::Binary {
                left: Box::new(expr),
                op: op_kind,
                right: Box::new(right),
                line: op_line,
            };
        }
        Ok(expr)
    }

    fn exponent(&mut self) -> Result<Expr, LangError> {
        let mut expr = self.unary()?;
        // Exponentiation is right-associative: 2 ** 3 ** 2 = 2 ** (3 ** 2)
        // Don't match ** if the next token is **= (to avoid consuming **=)
        while self.check(TokenKind::StarStar) && !self.check_next(TokenKind::StarStarEqual) {
            self.advance(); // Consume StarStar
            let op_line = self.previous().line;
            let right = self.exponent()?; // Recursive call for right-associativity
            expr = Expr::Binary {
                left: Box::new(expr),
                op: TokenKind::StarStar,
                right: Box::new(right),
                line: op_line,
            };
        }
        Ok(expr)
    }

    fn unary(&mut self) -> Result<Expr, LangError> {
        if self.match_token(TokenKind::Bang) || self.match_token(TokenKind::Minus) {
            let op_line = self.previous().line;
            let op_kind = self.previous().kind.clone();
            let right = self.unary()?;
            return Ok(Expr::Unary {
                op: op_kind,
                right: Box::new(right),
                line: op_line,
            });
        }
        self.call()
    }

    fn call(&mut self) -> Result<Expr, LangError> {
        let mut expr = self.primary()?;
        loop {
            // Обрабатываем вызовы функций (круглые скобки)
            // Может быть вызовом переменной или метода (Property)
            // Но не группировкой выражений - проверяем тип выражения перед обработкой
            if self.check(TokenKind::LParen) {
                // Проверяем, что это действительно вызов функции/метода
                // (переменная, Property или Super), а не просто группировка
                match &expr {
                    Expr::Variable { .. } | Expr::Property { .. } | Expr::Super { .. } => {
                        self.advance(); // Съедаем LParen
                        expr = self.finish_call(expr)?;
                        continue;
                    }
                    _ => {
                        // Это не вызов функции - не обрабатываем скобки здесь
                        // Позволим более высокому уровню обработать это
                        break;
                    }
                }
            }
            
            // Обрабатываем индексацию массивов (квадратные скобки)
            // Массивом может быть любое выражение, не только переменная
            if self.match_token(TokenKind::LBracket) {
                expr = self.finish_array_index(expr)?;
                continue;
            }
            
            // Обрабатываем доступ к свойствам (точка)
            if self.match_token(TokenKind::Dot) {
                let name = self.consume(TokenKind::Identifier, "Expect property name after '.'")?.lexeme.clone();
                let line = self.previous().line;
                expr = Expr::Property {
                    object: Box::new(expr),
                    name,
                    line,
                };
                continue;
            }
            
            // Если ни вызов функции, ни индексация, ни свойство - выходим из цикла
            break;
        }
        Ok(expr)
    }

    fn finish_call(&mut self, callee: Expr) -> Result<Expr, LangError> {
        let call_line = self.previous().line; // Номер строки открывающей скобки (LParen)
        let mut args = Vec::new();
        let mut has_named = false;
        if !self.check(TokenKind::RParen) {
            loop {
                if args.len() >= 255 {
                    return Err(LangError::ParseError {
                        message: "Cannot have more than 255 arguments".to_string(),
                        line: self.peek().line,
                    });
                }
                
                
                // Проверяем, является ли это именованным аргументом (identifier = expression)
                // Сохраняем текущую позицию для проверки
                let arg = if self.check(TokenKind::Identifier) {
                    // Проверяем, является ли следующий токен '='
                    // Сохраняем текущую позицию
                    let saved_current = self.current;
                    // Временно продвигаемся вперед для проверки
                    let is_named = if saved_current + 1 < self.tokens.len() {
                        self.tokens[saved_current + 1].kind == TokenKind::Equal
                    } else {
                        false
                    };
                    
                    if is_named {
                        // Именованный аргумент: name = value
                        let name_token = self.advance();
                        let name = name_token.lexeme.clone();
                        self.consume(TokenKind::Equal, "Expect '=' after parameter name in named argument")?;
                        has_named = true;
                        let value = self.expression()?;
                        Arg::Named {
                            name,
                            value,
                        }
                    } else {
                        // Позиционный аргумент
                        // Проверяем, что после именованного аргумента не идет позиционный
                        if has_named {
                            return Err(LangError::ParseError {
                                message: "Positional argument follows named argument".to_string(),
                                line: self.peek().line,
                            });
                        }
                        let expr = self.expression()?;
                        Arg::Positional(expr)
                    }
                } else {
                    // Позиционный аргумент (не идентификатор)
                    // Проверяем, что после именованного аргумента не идет позиционный
                    if has_named {
                        return Err(LangError::ParseError {
                            message: "Positional argument follows named argument".to_string(),
                            line: self.peek().line,
                        });
                    }
                    Arg::Positional(self.expression()?)
                };
                
                args.push(arg);
                if !self.match_token(TokenKind::Comma) {
                    break;
                }
            }
        }
        let paren = self.consume(TokenKind::RParen, "Expect ')' after arguments")?;
        
        // Извлекаем имя функции из callee
        // Может быть переменной, методом (Property), super() или super.method()
        match callee {
            Expr::Variable { name, .. } => {
                Ok(Expr::Call { name, args, line: call_line })
            }
            Expr::Super { .. } => {
                // super(...) - вызов конструктора родителя
                Ok(Expr::SuperCall { args, line: call_line })
            }
            Expr::Property { object, name, .. } => {
                // Проверяем, является ли object Super - тогда это super.method()
                if matches!(object.as_ref(), Expr::Super { .. }) {
                    Ok(Expr::SuperMethodCall {
                        method: name,
                        args,
                        line: call_line,
                    })
                } else {
                    // Это обычный вызов метода - создаем MethodCall
                    Ok(Expr::MethodCall {
                        object,
                        method: name,
                        args,
                        line: call_line,
                    })
                }
            }
            _ => {
                // Для сложных выражений пока не поддерживаем вызовы
                Err(LangError::ParseError {
                    message: "Can only call functions, variables, methods, and super".to_string(),
                    line: paren.line,
                })
            }
        }
    }

    fn finish_array_index(&mut self, array: Expr) -> Result<Expr, LangError> {
        let index_line = self.previous().line; // Номер строки открывающей скобки (LBracket)
        let index = self.expression()?;
        self.consume(TokenKind::RBracket, "Expect ']' after array index")?;
        
        Ok(Expr::ArrayIndex {
            array: Box::new(array),
            index: Box::new(index),
            line: index_line,
        })
    }

    fn primary(&mut self) -> Result<Expr, LangError> {
        if self.match_token(TokenKind::False) {
            let line = self.previous().line;
            return Ok(Expr::Literal { value: Value::Bool(false), line });
        }
        if self.match_token(TokenKind::True) {
            let line = self.previous().line;
            return Ok(Expr::Literal { value: Value::Bool(true), line });
        }
        if self.match_token(TokenKind::Null) {
            let line = self.previous().line;
            return Ok(Expr::Literal { value: Value::Null, line });
        }
        if self.match_token(TokenKind::This) {
            let line = self.previous().line;
            return Ok(Expr::This { line });
        }
        if self.match_token(TokenKind::Super) {
            let line = self.previous().line;
            return Ok(Expr::Super { line });
        }
        if self.match_token(TokenKind::Ellipsis) {
            let line = self.previous().line;
            return Ok(Expr::Ellipsis { line });
        }
        if self.match_token(TokenKind::Number) {
            let line = self.previous().line;
            let lexeme = self.previous().lexeme.clone();
            let value = lexeme.parse::<f64>()
                .map_err(|_| LangError::ParseError {
                    message: "Invalid number".to_string(),
                    line,
                })?;
            return Ok(Expr::Literal { value: Value::Number(value), line });
        }
        if self.match_token(TokenKind::String) {
            let line = self.previous().line;
            let lexeme = self.previous().lexeme.clone();
            let raw = lexeme[1..lexeme.len() - 1].to_string(); // Убираем кавычки
            if raw.contains("${") {
                let mut has_interpolation = false;
                let bytes = raw.as_bytes();
                let mut start = 0;
                while let Some(rel) = raw[start..].find("${") {
                    let pos = start + rel;
                    if pos == 0 || bytes[pos - 1] != b'\\' {
                        has_interpolation = true;
                        break;
                    }
                    start = pos + 1;
                }
                if has_interpolation {
                    let segments = Self::parse_interpolated_segments(&raw, line)?;
                    return Ok(Expr::InterpolatedString { segments, line });
                }
            }
            return Ok(Expr::Literal { value: Value::String(Self::unescape_literal(&raw)), line });
        }
        if self.match_token(TokenKind::At) {
            let line = self.previous().line;
            self.consume(TokenKind::Identifier, "Expect 'class' after '@' in expression")?;
            if self.previous().lexeme != "class" {
                return Err(LangError::ParseError {
                    message: "After '@' only 'class' is allowed (e.g. @class.name)".to_string(),
                    line: self.previous().line,
                });
            }
            return Ok(Expr::Variable { name: "@class".to_string(), line });
        }
        if self.match_token(TokenKind::Identifier) {
            let line = self.previous().line;
            let name = self.previous().lexeme.clone();
            // Если после идентификатора идет запятая, а затем еще один идентификатор и =,
            // то это распаковка. Но мы не можем обработать ее здесь, так как primary() возвращает только выражение.
            // Проверка будет в assignment() после того, как мы вернем переменную.
            // Однако, если мы здесь, значит идентификатор уже был потреблен, и self.current указывает на следующий токен.
            // Поэтому проверка в assignment() должна работать, так как self.current указывает на запятую.
            return Ok(Expr::Variable { name, line });
        }
        if self.match_token(TokenKind::LParen) {
            let paren_line = self.previous().line;
            // Проверяем, является ли это кортежем или группировкой
            // Если сразу закрывающая скобка - пустой кортеж
            if self.check(TokenKind::RParen) {
                self.consume(TokenKind::RParen, "Expect ')' after '('")?;
                return Ok(Expr::TupleLiteral { elements: vec![], line: paren_line });
            }
            
            // Парсим первое выражение
            let first_expr = self.expression()?;
            
            if self.match_token(TokenKind::Comma) {
                // Это кортеж: (expr1, expr2, ...) или (expr1,)
                let mut elements = vec![first_expr];
                loop {
                    // Проверяем, не закрывающая ли скобка (для последнего элемента)
                    if self.check(TokenKind::RParen) {
                        break;
                    }
                    elements.push(self.expression()?);
                    if !self.match_token(TokenKind::Comma) {
                        break;
                    }
                }
                self.consume(TokenKind::RParen, "Expect ')' after tuple elements")?;
                return Ok(Expr::TupleLiteral { elements, line: paren_line });
            } else if self.check(TokenKind::RParen) {
                // Это группировка: (expr)
                self.consume(TokenKind::RParen, "Expect ')' after expression")?;
                return Ok(first_expr);
            } else {
                // Ошибка: ожидается либо запятая (кортеж), либо закрывающая скобка (группировка)
                return Err(LangError::ParseError {
                    message: "Expect ',' or ')' after expression in parentheses".to_string(),
                    line: self.peek().line,
                });
            }
        }
        if self.match_token(TokenKind::LBracket) {
            return self.array_literal();
        }
        if self.match_token(TokenKind::LBrace) {
            return self.object_literal();
        }

        let token = self.peek();
        Err(LangError::ParseError {
            message: format!("Expect expression, found {:?} '{}' at line {}", token.kind, token.lexeme, token.line),
            line: token.line,
        })
    }

    fn match_token(&mut self, kind: TokenKind) -> bool {
        if self.check(kind) {
            self.advance();
            true
        } else {
            false
        }
    }

    fn check(&self, kind: TokenKind) -> bool {
        if self.is_at_end() {
            false
        } else {
            self.peek().kind == kind
        }
    }

    fn check_next(&self, kind: TokenKind) -> bool {
        if self.is_at_end() || self.current + 1 >= self.tokens.len() {
            false
        } else {
            self.tokens[self.current + 1].kind == kind
        }
    }

    fn advance(&mut self) -> &Token {
        if !self.is_at_end() {
            self.current += 1;
        }
        self.previous()
    }

    fn is_at_end(&self) -> bool {
        self.peek().kind == TokenKind::Eof
    }

    fn peek(&self) -> &Token {
        &self.tokens[self.current]
    }

    fn previous(&self) -> &Token {
        &self.tokens[self.current - 1]
    }

    fn array_literal(&mut self) -> Result<Expr, LangError> {
        let line = self.previous().line;
        let mut elements = Vec::new();

        if !self.check(TokenKind::RBracket) {
            loop {
                elements.push(self.expression()?);
                if !self.match_token(TokenKind::Comma) {
                    break;
                }
            }
        }

        self.consume(TokenKind::RBracket, "Expect ']' after array elements")?;

        // Если все элементы - литералы, создаем Value::Array напрямую
        // Иначе создаем ArrayLiteral для компиляции во время выполнения
        let mut all_literals = true;
        let mut values = Vec::new();
        
        for expr in &elements {
            match expr {
                Expr::Literal { value, .. } => {
                    values.push(value.clone());
                }
                _ => {
                    all_literals = false;
                    break;
                }
            }
        }

        if all_literals {
            Ok(Expr::Literal {
                value: Value::Array(Rc::new(RefCell::new(values))),
                line,
            })
        } else {
            Ok(Expr::ArrayLiteral {
                elements,
                line,
            })
        }
    }

    fn object_literal(&mut self) -> Result<Expr, LangError> {
        let line = self.previous().line;
        let mut pairs = Vec::new();

        if !self.check(TokenKind::RBrace) {
            loop {
                // Ключ должен быть строкой
                let key_token = self.consume(TokenKind::String, "Expect string key in object literal")?;
                let key = key_token.lexeme[1..key_token.lexeme.len() - 1].to_string(); // Убираем кавычки
                
                // Потребляем двоеточие
                self.consume(TokenKind::Colon, "Expect ':' after key in object literal")?;
                
                // Парсим значение
                let value = self.expression()?;
                
                pairs.push((key, value));
                
                if !self.match_token(TokenKind::Comma) {
                    break;
                }
            }
        }

        self.consume(TokenKind::RBrace, "Expect '}' after object pairs")?;

        // Если все значения - литералы, создаем Value::Object напрямую
        // Иначе создаем ObjectLiteral для компиляции во время выполнения
        let mut all_literals = true;
        let mut object_map = std::collections::HashMap::new();
        
        for (key, expr) in &pairs {
            match expr {
                Expr::Literal { value, .. } => {
                    object_map.insert(key.clone(), value.clone());
                }
                _ => {
                    all_literals = false;
                    break;
                }
            }
        }

        if all_literals {
            Ok(Expr::Literal {
                value: Value::Object(Rc::new(RefCell::new(object_map))),
                line,
            })
        } else {
            Ok(Expr::ObjectLiteral {
                pairs,
                line,
            })
        }
    }

    /// Парсит один тип - может быть идентификатором, null, или с подстрочным аргументом (Column[date], str[50])
    fn parse_single_type(&mut self) -> Result<String, LangError> {
        if self.check(TokenKind::Null) {
            self.advance();
            Ok("null".to_string())
        } else if self.check(TokenKind::Identifier) {
            let base = self.consume(TokenKind::Identifier, "Expect type name")?.lexeme.clone();
            if self.match_token(TokenKind::LBracket) {
                let inner = if self.check(TokenKind::Number) {
                    self.consume(TokenKind::Number, "Expect number in type subscript")?.lexeme.clone()
                } else {
                    let inner_types = self.parse_type_name()?;
                    inner_types.join(" | ")
                };
                self.consume(TokenKind::RBracket, "Expect ']' after type parameter")?;
                Ok(format!("{}[{}]", base, inner))
            } else {
                Ok(base)
            }
        } else {
            Err(LangError::ParseError {
                message: "Expect type name (identifier or 'null')".to_string(),
                line: self.peek().line,
            })
        }
    }

    /// Парсит union типы - может быть один тип или несколько через |
    /// Например: "int", "str | int", "null | str | int"
    fn parse_type_name(&mut self) -> Result<Vec<String>, LangError> {
        let mut types = Vec::new();
        
        // Парсим первый тип
        types.push(self.parse_single_type()?);
        
        // Парсим дополнительные типы через |
        while self.match_token(TokenKind::Pipe) {
            types.push(self.parse_single_type()?);
        }
        
        Ok(types)
    }

    fn consume(&mut self, kind: TokenKind, message: &str) -> Result<&Token, LangError> {
        if self.check(kind) {
            Ok(self.advance())
        } else {
            Err(LangError::ParseError {
                message: message.to_string(),
                line: self.peek().line,
            })
        }
    }
}

