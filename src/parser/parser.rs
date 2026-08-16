// Recursive Descent Parser

use crate::common::error::LangError;
use crate::common::numeric::FloatValue;
use crate::common::value::{ObjectKind, Value};
use crate::lexer::{Token, TokenKind};
use crate::parser::ast::{
    Arg, AssignTarget, BinaryOpKind, ClassVariable, Expr, IfBranch, ImportItem, ImportStmt,
    IndexExpr, InterpolatedSegment, ListComprehensionClause, ObjectLiteralKey, ObjectPair, Param,
    ParamKind, Stmt, TypePart, UnpackPattern,
};
use crate::vm::operator_registry::{
    binding_power, token_kind_to_symbol, OperatorRegistry, SharedOperatorRegistry,
};
use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;

pub struct Parser {
    tokens: Vec<Token>,
    current: usize,
    source_name: Option<String>,
    /// Lazily loaded from [`Self::source_name`] for Python `:` indented blocks.
    source_lines: Option<Vec<String>>,
    operator_registry: SharedOperatorRegistry,
}

impl Parser {
    pub fn new(tokens: Vec<Token>) -> Self {
        Self::new_with_source_name(tokens, None)
    }

    pub fn new_with_source_name(tokens: Vec<Token>, source_name: Option<&str>) -> Self {
        Self::new_with_source_name_and_registry(
            tokens,
            source_name,
            Arc::new(OperatorRegistry::with_builtins()),
        )
    }

    pub fn new_with_source_name_and_registry(
        tokens: Vec<Token>,
        source_name: Option<&str>,
        operator_registry: SharedOperatorRegistry,
    ) -> Self {
        Self {
            tokens,
            current: 0,
            source_name: source_name.map(String::from),
            source_lines: None,
            operator_registry,
        }
    }

    fn ensure_source_lines(&mut self) {
        if self.source_lines.is_some() {
            return;
        }
        if let Some(path) = &self.source_name {
            if let Ok(text) = std::fs::read_to_string(path) {
                self.source_lines = Some(text.lines().map(|s| s.to_string()).collect());
            }
        }
    }

    fn source_line_indent(&mut self, line: usize) -> usize {
        self.ensure_source_lines();
        self.source_lines
            .as_ref()
            .and_then(|lines| lines.get(line.saturating_sub(1)))
            .map(|l| l.chars().take_while(|c| *c == ' ' || *c == '\t').count())
            .unwrap_or(0)
    }

    /// Body after `if`/`while` `:` when the next statement is on a later line (Python-style indent).
    fn parse_colon_indented_block(&mut self, header_line: usize) -> Result<Vec<Stmt>, LangError> {
        let header_indent = self.source_line_indent(header_line);
        let mut body = Vec::new();
        while !self.is_at_end() && !self.check(TokenKind::RBrace) {
            let next_indent = self.source_line_indent(self.peek().line);
            if next_indent <= header_indent {
                break;
            }
            body.push(self.statement()?);
        }
        if body.is_empty() {
            return Err(LangError::ParseError {
                message: "Expect indented block after ':'".to_string(),
                line: self.peek().line,
                file: self.source_name.clone(),
            });
        }
        Ok(body)
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
            // from ... import ...
            self.from_import_declaration()
        } else if self.match_token(TokenKind::Import) {
            // import ..., plot
            self.simple_import_declaration()
        } else if self.match_token(TokenKind::Global) {
            // global a = 5
            let global_line = self.previous().line;
            let name = self
                .consume(TokenKind::Identifier, "Expect variable name after 'global'")?
                .lexeme
                .clone();
            self.consume(TokenKind::Equal, "Expect '=' after variable name")?;
            let value = self.expression()?;
            Ok(Stmt::Let {
                name,
                value,
                is_global: true,
                line: global_line,
            })
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
                let (is_cached, route, ws_route) = self.parse_function_decorators_after_at()?;
                self.consume(TokenKind::Fn, "Expect 'fn' after decorators")?;
                self.function_declaration_body(is_cached, route, ws_route)
            }
        } else if self.match_token(TokenKind::Cls) {
            self.class_declaration(false)
        } else if self.check(TokenKind::Stream) && self.check_next(TokenKind::Fn) {
            self.stream_function_declaration()
        } else if self.check(TokenKind::Fn) {
            // `fn (` — анонимная функция (лямбда); `fn name` — объявление функции
            if self.check_next(TokenKind::LParen) {
                let fn_line = self.peek().line;
                self.advance(); // fn
                let expr = self.parse_lambda_after_fn()?;
                self.match_token(TokenKind::Semicolon);
                Ok(Stmt::Expr {
                    expr,
                    line: fn_line,
                })
            } else {
                self.function_declaration()
            }
        } else {
            self.statement()
        }
    }

    /// Parses a dotted module name: identifier ( . identifier )*
    fn parse_dotted_module_name(&mut self, context: &str) -> Result<String, LangError> {
        let mut parts = vec![self.consume(TokenKind::Identifier, context)?.lexeme.clone()];
        while self.match_token(TokenKind::Dot) {
            parts.push(
                self.consume(
                    TokenKind::Identifier,
                    "Expect identifier after '.' in module name",
                )?
                .lexeme
                .clone(),
            );
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
        let parenthesized = self.match_token(TokenKind::LParen);

        loop {
            if parenthesized && self.check(TokenKind::RParen) {
                break;
            }

            if self.match_token(TokenKind::Star) {
                items.push(ImportItem::All);
            } else if self.check(TokenKind::Identifier) {
                let name = self.advance().lexeme.clone();

                if self.match_token(TokenKind::As) {
                    let alias = self
                        .consume(TokenKind::Identifier, "Expect alias name after 'as'")?
                        .lexeme
                        .clone();
                    items.push(ImportItem::Aliased { name, alias });
                } else {
                    items.push(ImportItem::Named(name));
                }
            } else {
                return Err(LangError::ParseError {
                    message: if parenthesized {
                        "Expect identifier, '*', or ')' in import list".to_string()
                    } else {
                        "Expect identifier or '*' in import list".to_string()
                    },
                    line: self.peek().line,
                    file: self.source_name.clone(),
                });
            }

            if parenthesized {
                if self.match_token(TokenKind::Comma) {
                    continue;
                }
                if self.check(TokenKind::RParen) {
                    break;
                }
                return Err(LangError::ParseError {
                    message: "Expect ',' or ')' after import item".to_string(),
                    line: self.peek().line,
                    file: self.source_name.clone(),
                });
            } else if !self.match_token(TokenKind::Comma) {
                break;
            }
        }

        if parenthesized {
            self.consume(TokenKind::RParen, "Expect ')' after import list")?;
        }

        Ok(items)
    }

    fn variable_declaration(&mut self) -> Result<Stmt, LangError> {
        let let_line = self.previous().line;
        let name = self
            .consume(TokenKind::Identifier, "Expect variable name")?
            .lexeme
            .clone();

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
                        file: self.source_name.clone(),
                    });
                }
                if !self.match_token(TokenKind::Comma) {
                    break;
                }
            }
            self.consume(TokenKind::Equal, "Expect '=' after variable list")?;
            let value = self.parse_unpack_rhs()?;
            // Для let statements с распаковкой создаем UnpackAssign выражение
            // Но нам нужно сохранить это как Stmt::Let с особым значением
            // Пока что создадим временное решение - используем первую переменную как имя
            // и создадим UnpackAssign в value
            return Ok(Stmt::Let {
                name: names[0].clone(),
                value: Expr::UnpackAssign {
                    targets: names.into_iter().map(AssignTarget::Name).collect(),
                    value: Box::new(value),
                    line: let_line,
                },
                is_global: false,
                line: let_line,
            });
        }

        self.consume(TokenKind::Equal, "Expect '=' after variable name")?;
        let value = self.expression()?;
        Ok(Stmt::Let {
            name,
            value,
            is_global: false,
            line: let_line,
        })
    }

    /// Parse function decorators when we have already consumed '@' and current token is the first decorator name.
    fn parse_function_decorators_after_at(
        &mut self,
    ) -> Result<(bool, Option<(String, String)>, Option<String>), LangError> {
        let mut is_cached = false;
        let mut route: Option<(String, String)> = None;
        let mut ws_route: Option<String> = None;
        loop {
            if self.match_decorator_cache() {
                is_cached = true;
            } else if self.check(TokenKind::Identifier) && self.peek().lexeme == "route" {
                self.advance(); // consume "route"
                self.consume(TokenKind::LParen, "Expect '(' after @route")?;
                let method_expr = self.expression()?;
                let method =
                    Self::expr_to_string(&method_expr).ok_or_else(|| LangError::ParseError {
                        message: "@route first argument must be a string literal (e.g. \"GET\")"
                            .to_string(),
                        line: self.previous().line,
                        file: self.source_name.clone(),
                    })?;
                self.consume(TokenKind::Comma, "Expect ',' in @route(...)")?;
                let path_expr = self.expression()?;
                let path =
                    Self::expr_to_string(&path_expr).ok_or_else(|| LangError::ParseError {
                        message: "@route second argument must be a string literal (e.g. \"/\")"
                            .to_string(),
                        line: self.previous().line,
                        file: self.source_name.clone(),
                    })?;
                self.consume(TokenKind::RParen, "Expect ')' after @route(...)")?;
                route = Some((method, path));
            } else if self.check(TokenKind::Identifier) && self.peek().lexeme == "ws_route" {
                ws_route = Some(self.parse_ws_route_type_arg()?);
            } else {
                let got = if self.check(TokenKind::Identifier) {
                    self.peek().lexeme.clone()
                } else {
                    "non-identifier".to_string()
                };
                return Err(LangError::ParseError {
                    message: format!(
                        "Expect 'cache', 'route(...)' or 'ws_route(...)' after '@', got '{}'",
                        got
                    ),
                    line: self.peek().line,
                    file: self.source_name.clone(),
                });
            }
            if !self.match_token(TokenKind::At) {
                break;
            }
        }
        Ok((is_cached, route, ws_route))
    }

    fn parse_ws_route_type_arg(&mut self) -> Result<String, LangError> {
        self.advance(); // consume "ws_route"
        self.consume(TokenKind::LParen, "Expect '(' after @ws_route")?;
        let type_expr = self.expression()?;
        let msg_type =
            Self::expr_to_string(&type_expr).ok_or_else(|| LangError::ParseError {
                message: "@ws_route argument must be a string literal (e.g. \"execute\")"
                    .to_string(),
                line: self.previous().line,
                file: self.source_name.clone(),
            })?;
        self.consume(TokenKind::RParen, "Expect ')' after @ws_route(...)")?;
        Ok(msg_type)
    }

    fn function_declaration(&mut self) -> Result<Stmt, LangError> {
        // Parse decorators: @cache and/or @route("METHOD", "/path") and/or @ws_route("type")
        let mut is_cached = false;
        let mut route: Option<(String, String)> = None;
        let mut ws_route: Option<String> = None;
        while self.match_token(TokenKind::At) {
            if self.match_decorator_cache() {
                is_cached = true;
            } else if self.check(TokenKind::Identifier) && self.peek().lexeme == "route" {
                self.advance();
                self.consume(TokenKind::LParen, "Expect '(' after @route")?;
                let method_expr = self.expression()?;
                let method =
                    Self::expr_to_string(&method_expr).ok_or_else(|| LangError::ParseError {
                        message: "@route first argument must be a string literal (e.g. \"GET\")"
                            .to_string(),
                        line: self.previous().line,
                        file: self.source_name.clone(),
                    })?;
                self.consume(TokenKind::Comma, "Expect ',' in @route(...)")?;
                let path_expr = self.expression()?;
                let path =
                    Self::expr_to_string(&path_expr).ok_or_else(|| LangError::ParseError {
                        message: "@route second argument must be a string literal (e.g. \"/\")"
                            .to_string(),
                        line: self.previous().line,
                        file: self.source_name.clone(),
                    })?;
                self.consume(TokenKind::RParen, "Expect ')' after @route(...)")?;
                route = Some((method, path));
            } else if self.check(TokenKind::Identifier) && self.peek().lexeme == "ws_route" {
                ws_route = Some(self.parse_ws_route_type_arg()?);
            } else {
                let dec_name = if self.check(TokenKind::Identifier) {
                    self.peek().lexeme.clone()
                } else {
                    "".to_string()
                };
                return Err(LangError::ParseError {
                    message: format!(
                        "Expect 'cache', 'route(...)' or 'ws_route(...)' after '@', got '{}'",
                        dec_name
                    ),
                    line: self.peek().line,
                    file: self.source_name.clone(),
                });
            }
        }

        self.consume(TokenKind::Fn, "Expect 'fn'")?;
        self.function_declaration_body(is_cached, route, ws_route)
    }

    /// `stream fn` с теми же декораторами `@cache` / `@route`, что и обычная функция.
    fn stream_function_declaration(&mut self) -> Result<Stmt, LangError> {
        let mut is_cached = false;
        let mut route: Option<(String, String)> = None;
        let mut ws_route: Option<String> = None;
        while self.match_token(TokenKind::At) {
            if self.match_decorator_cache() {
                is_cached = true;
            } else if self.check(TokenKind::Identifier) && self.peek().lexeme == "route" {
                self.advance();
                self.consume(TokenKind::LParen, "Expect '(' after @route")?;
                let method_expr = self.expression()?;
                let method =
                    Self::expr_to_string(&method_expr).ok_or_else(|| LangError::ParseError {
                        message: "@route first argument must be a string literal (e.g. \"GET\")"
                            .to_string(),
                        line: self.previous().line,
                        file: self.source_name.clone(),
                    })?;
                self.consume(TokenKind::Comma, "Expect ',' in @route(...)")?;
                let path_expr = self.expression()?;
                let path =
                    Self::expr_to_string(&path_expr).ok_or_else(|| LangError::ParseError {
                        message: "@route second argument must be a string literal (e.g. \"/\")"
                            .to_string(),
                        line: self.previous().line,
                        file: self.source_name.clone(),
                    })?;
                self.consume(TokenKind::RParen, "Expect ')' after @route(...)")?;
                route = Some((method, path));
            } else if self.check(TokenKind::Identifier) && self.peek().lexeme == "ws_route" {
                ws_route = Some(self.parse_ws_route_type_arg()?);
            } else {
                let dec_name = if self.check(TokenKind::Identifier) {
                    self.peek().lexeme.clone()
                } else {
                    "".to_string()
                };
                return Err(LangError::ParseError {
                    message: format!(
                        "Expect 'cache', 'route(...)' or 'ws_route(...)' after '@', got '{}'",
                        dec_name
                    ),
                    line: self.peek().line,
                    file: self.source_name.clone(),
                });
            }
        }

        self.consume(TokenKind::Stream, "Expect 'stream'")?;
        self.consume(TokenKind::Fn, "Expect 'fn' after 'stream'")?;
        self.stream_function_declaration_body(is_cached, route, ws_route)
    }

    /// Parse stream function name, params, return type and body. Caller must have consumed `stream` and `fn`.
    fn stream_function_declaration_body(
        &mut self,
        is_cached: bool,
        route: Option<(String, String)>,
        ws_route: Option<String>,
    ) -> Result<Stmt, LangError> {
        let fn_line = self.previous().line;
        let name = self
            .consume(TokenKind::Identifier, "Expect function name")?
            .lexeme
            .clone();
        self.consume(TokenKind::LParen, "Expect '(' after function name")?;
        let params = self.parse_parameter_list_until_rparen()?;

        let return_type = if self.match_token(TokenKind::Arrow) {
            Some(self.parse_type_name()?)
        } else {
            None
        };

        self.consume(TokenKind::LBrace, "Expect '{' before function body")?;

        let body = self.block()?;

        Ok(Stmt::StreamFunction {
            name,
            params,
            return_type,
            body,
            is_cached,
            route,
            ws_route,
            line: fn_line,
        })
    }

    /// Parse function name, params, return type and body. Caller must have consumed 'fn' so previous() is 'fn'.
    fn function_declaration_body(
        &mut self,
        is_cached: bool,
        route: Option<(String, String)>,
        ws_route: Option<String>,
    ) -> Result<Stmt, LangError> {
        let fn_line = self.previous().line;
        let name = self
            .consume(TokenKind::Identifier, "Expect function name")?
            .lexeme
            .clone();
        self.consume(TokenKind::LParen, "Expect '(' after function name")?;
        let params = self.parse_parameter_list_until_rparen()?;

        // Проверяем, есть ли аннотация возвращаемого типа (-> type)
        let return_type = if self.match_token(TokenKind::Arrow) {
            Some(self.parse_type_name()?)
        } else {
            None
        };

        self.consume(TokenKind::LBrace, "Expect '{' before function body")?;

        let body = self.block()?;

        Ok(Stmt::Function {
            name,
            params,
            return_type,
            body,
            is_cached,
            route,
            ws_route,
            line: fn_line,
        })
    }

    /// После `(` — список параметров до `)` (как у именованной функции).
    fn parse_parameter_list_until_rparen(&mut self) -> Result<Vec<Param>, LangError> {
        let mut params = Vec::new();
        let mut has_default = false;
        let mut has_var_pos = false;
        let mut has_var_kw = false;
        if !self.check(TokenKind::RParen) {
            loop {
                if params.len() >= 255 {
                    return Err(LangError::ParseError {
                        message: "Cannot have more than 255 parameters".to_string(),
                        line: self.previous().line,
                        file: self.source_name.clone(),
                    });
                }

                let param_line = self.peek().line;

                // *args или **kwargs
                let (param_name, kind) = if self.check(TokenKind::StarStar) {
                    self.advance();
                    if has_var_kw {
                        return Err(LangError::ParseError {
                            message: "Only one **kwargs parameter allowed".to_string(),
                            line: param_line,
                            file: self.source_name.clone(),
                        });
                    }
                    if !params.is_empty() && self.check(TokenKind::Comma) {
                        return Err(LangError::ParseError {
                            message: "**kwargs must be the last parameter".to_string(),
                            line: param_line,
                            file: self.source_name.clone(),
                        });
                    }
                    let name = self
                        .consume(TokenKind::Identifier, "Expect parameter name after **")?
                        .lexeme
                        .clone();
                    has_var_kw = true;
                    (name, ParamKind::VariadicKeyword)
                } else if self.check(TokenKind::Star)
                    && self.current + 1 < self.tokens.len()
                    && self.tokens[self.current + 1].kind != TokenKind::Star
                {
                    self.advance();
                    if has_var_pos {
                        return Err(LangError::ParseError {
                            message: "Only one *args parameter allowed".to_string(),
                            line: param_line,
                            file: self.source_name.clone(),
                        });
                    }
                    if has_var_kw {
                        return Err(LangError::ParseError {
                            message: "*args must appear before **kwargs".to_string(),
                            line: param_line,
                            file: self.source_name.clone(),
                        });
                    }
                    let name = self
                        .consume(TokenKind::Identifier, "Expect parameter name after *")?
                        .lexeme
                        .clone();
                    has_var_pos = true;
                    (name, ParamKind::VariadicPositional)
                } else {
                    let name = self
                        .consume(TokenKind::Identifier, "Expect parameter name")?
                        .lexeme
                        .clone();
                    if has_var_pos || has_var_kw {
                        return Err(LangError::ParseError {
                            message: "Regular parameters cannot follow *args or **kwargs".to_string(),
                            line: param_line,
                            file: self.source_name.clone(),
                        });
                    }
                    (name, ParamKind::Regular)
                };

                let type_annotation = if self.match_token(TokenKind::Colon) {
                    Some(self.parse_type_name()?)
                } else {
                    None
                };

                let default_value = if kind == ParamKind::Regular && self.match_token(TokenKind::Equal) {
                    has_default = true;
                    Some(self.expression()?)
                } else {
                    if has_default && kind == ParamKind::Regular {
                        return Err(LangError::ParseError {
                            message: "Non-default argument follows default argument".to_string(),
                            line: param_line,
                            file: self.source_name.clone(),
                        });
                    }
                    if kind != ParamKind::Regular && self.check(TokenKind::Equal) {
                        return Err(LangError::ParseError {
                            message: "*args and **kwargs cannot have default values".to_string(),
                            line: param_line,
                            file: self.source_name.clone(),
                        });
                    }
                    None
                };

                if has_default && (kind == ParamKind::VariadicPositional || kind == ParamKind::VariadicKeyword) {
                    return Err(LangError::ParseError {
                        message: "Non-default argument follows default argument".to_string(),
                        line: param_line,
                        file: self.source_name.clone(),
                    });
                }

                params.push(Param {
                    name: param_name,
                    kind,
                    type_annotation,
                    default_value,
                });

                if !self.match_token(TokenKind::Comma) {
                    break;
                }
                if has_var_kw {
                    return Err(LangError::ParseError {
                        message: "**kwargs must be the last parameter".to_string(),
                        line: self.peek().line,
                        file: self.source_name.clone(),
                    });
                }
            }
        }
        self.consume(TokenKind::RParen, "Expect ')' after parameters")?;
        Ok(params)
    }

    /// Уже съеден `fn`; дальше `(params)` [`->` type] `=>` body.
    fn parse_lambda_after_fn(&mut self) -> Result<Expr, LangError> {
        let fn_line = self.previous().line;
        self.consume(TokenKind::LParen, "Expect '(' after 'fn'")?;
        let params = self.parse_parameter_list_until_rparen()?;
        let return_type = if self.match_token(TokenKind::Arrow) {
            Some(self.parse_type_name()?)
        } else {
            None
        };
        self.consume(TokenKind::FatArrow, "Expect '=>' before lambda body")?;
        let body = self.expression()?;
        Ok(Expr::Lambda {
            params,
            return_type,
            body: Box::new(body),
            line: fn_line,
        })
    }

    /// Extract string value from a string literal expression (for @route("GET", "/")).
    fn expr_to_string(expr: &Expr) -> Option<String> {
        match expr {
            Expr::Literal {
                value: Value::String(s),
                ..
            } => Some(s.clone()),
            Expr::InterpolatedString { .. } => None, // interpolated string is not a literal
            _ => None,
        }
    }

    fn class_declaration(&mut self, is_abstract: bool) -> Result<Stmt, LangError> {
        let cls_line = self.previous().line;
        let name = self
            .consume(TokenKind::Identifier, "Expect class name after 'cls'")?
            .lexeme
            .clone();
        let superclass = if self.match_token(TokenKind::LParen) {
            let super_name = self
                .consume(TokenKind::Identifier, "Expect superclass name after '('")?
                .lexeme
                .clone();
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
        let mut methods: Vec<crate::parser::ast::Method> = Vec::new();
        // Some(true) = public (default), None = private, Some(false) = protected
        let mut current_section_public: Option<bool> = Some(true);

        while !self.check(TokenKind::RBrace) && !self.is_at_end() {
            if self.match_token(TokenKind::Private) {
                self.consume(TokenKind::Colon, "Expect ':' after 'private'")?;
                current_section_public = None;
                loop {
                    if self.is_at_end() {
                        break;
                    }
                    if self.check(TokenKind::Protected)
                        || self.check(TokenKind::Public)
                        || self.check(TokenKind::RBrace)
                        || self.check(TokenKind::Fn)
                    {
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
                    if self.check(TokenKind::Private)
                        || self.check(TokenKind::Public)
                        || self.check(TokenKind::RBrace)
                        || self.check(TokenKind::Fn)
                    {
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
                    if self.check(TokenKind::Private)
                        || self.check(TokenKind::Protected)
                        || self.check(TokenKind::RBrace)
                        || self.check(TokenKind::Fn)
                    {
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
                let method = self.parse_method(current_section_public)?;
                if methods.iter().any(|m| m.name == method.name) {
                    return Err(LangError::ParseError {
                        message: format!("Duplicate method `{}` in class", method.name),
                        line: method.line,
                        file: self.source_name.clone(),
                    });
                }
                methods.push(method);
            } else {
                return Err(LangError::ParseError {
                    message: "Expect 'private:', 'protected:', 'public:', field (name : type), class variable (name = expr), constructor, or method in class body".to_string(),
                    line: self.peek().line,
                    file: self.source_name.clone(),
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
        let field_name = self
            .consume(TokenKind::Identifier, "Expect field name")?
            .lexeme
            .clone();
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
        let name = self
            .consume(TokenKind::Identifier, "Expect variable name")?
            .lexeme
            .clone();
        self.consume(
            TokenKind::Equal,
            "Expect '=' after variable name in class variable",
        )?;
        let value = self.expression()?;
        Ok(ClassVariable { name, value })
    }

    fn parse_constructor(
        &mut self,
        class_name: &str,
    ) -> Result<crate::parser::ast::Constructor, LangError> {
        // Парсим: new ClassName(...) { ... } или new ClassName(...) : this(...) {}
        let new_line = self.consume(TokenKind::Identifier, "Expect 'new'")?.line;
        // Проверяем, что это действительно 'new'
        if self.previous().lexeme != "new" {
            return Err(LangError::ParseError {
                message: "Expect 'new' for constructor".to_string(),
                line: new_line,
                file: self.source_name.clone(),
            });
        }

        // Парсим имя класса (должно совпадать с именем класса)
        let constructor_class_name = self
            .consume(TokenKind::Identifier, "Expect class name after 'new'")?
            .lexeme
            .clone();
        if constructor_class_name != class_name {
            return Err(LangError::ParseError {
                message: format!(
                    "Constructor class name '{}' must match class name '{}'",
                    constructor_class_name, class_name
                ),
                line: self.previous().line,
                file: self.source_name.clone(),
            });
        }

        self.consume(
            TokenKind::LParen,
            "Expect '(' after class name in constructor",
        )?;

        let mut params = Vec::new();
        let mut has_default = false;
        if !self.check(TokenKind::RParen) {
            loop {
                if params.len() >= 255 {
                    return Err(LangError::ParseError {
                        message: "Cannot have more than 255 parameters".to_string(),
                        line: self.previous().line,
                        file: self.source_name.clone(),
                    });
                }

                let param_name = self
                    .consume(TokenKind::Identifier, "Expect parameter name")?
                    .lexeme
                    .clone();
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
                            file: self.source_name.clone(),
                        });
                    }
                    None
                };

                params.push(crate::parser::ast::Param {
                    name: param_name,
                    kind: ParamKind::Regular,
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
                    file: self.source_name.clone(),
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
            let body = self.block()?;
            (body, Some(delegate_args))
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

    fn parse_method(
        &mut self,
        visibility: Option<bool>,
    ) -> Result<crate::parser::ast::Method, LangError> {
        // Парсим: fn methodName(...) -> type? { ... }
        self.consume(TokenKind::Fn, "Expect 'fn'")?;
        let method_line = self.previous().line;
        let (name, is_special) = if self.match_token(TokenKind::At) {
            let id = self
                .consume(TokenKind::Identifier, "Expect special method name after '@'")?;
            let base = id.lexeme.as_str();
            if !crate::compiler::special_methods::is_reserved_special_base(base) {
                return Err(LangError::ParseError {
                    message: format!("Unknown special method '@{}'", base),
                    line: id.line,
                    file: self.source_name.clone(),
                });
            }
            (format!("@{base}"), true)
        } else {
            let id = self
                .consume(TokenKind::Identifier, "Expect method name")?;
            (id.lexeme.clone(), false)
        };
        self.consume(TokenKind::LParen, "Expect '(' after method name")?;

        let mut params = Vec::new();
        let mut has_default = false;
        if !self.check(TokenKind::RParen) {
            loop {
                if params.len() >= 255 {
                    return Err(LangError::ParseError {
                        message: "Cannot have more than 255 parameters".to_string(),
                        line: self.previous().line,
                        file: self.source_name.clone(),
                    });
                }

                let param_name = if self.match_token(TokenKind::At) {
                    self.consume(TokenKind::Identifier, "Expect 'class' after '@'")?;
                    if self.previous().lexeme != "class" {
                        return Err(LangError::ParseError {
                            message: "After '@' only 'class' is allowed as parameter name"
                                .to_string(),
                            line: self.previous().line,
                            file: self.source_name.clone(),
                        });
                    }
                    if !params.is_empty() {
                        return Err(LangError::ParseError {
                            message: "@class can only be the first parameter of a method"
                                .to_string(),
                            line: self.previous().line,
                            file: self.source_name.clone(),
                        });
                    }
                    "@class".to_string()
                } else {
                    self.consume(TokenKind::Identifier, "Expect parameter name")?
                        .lexeme
                        .clone()
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
                            file: self.source_name.clone(),
                        });
                    }
                    None
                };

                params.push(crate::parser::ast::Param {
                    name: param_name,
                    kind: ParamKind::Regular,
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
            visibility,
            is_special,
        })
    }

    fn statement(&mut self) -> Result<Stmt, LangError> {
        if self.match_token(TokenKind::If) {
            self.if_statement()
        } else if self.match_token(TokenKind::While) {
            self.while_statement()
        } else if self.match_token(TokenKind::For) {
            self.for_statement()
        } else if self.match_token(TokenKind::Ereturn) {
            self.ereturn_statement()
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

        // `if <condition>: <statement>` or Python indented block after `:`
        if self.match_token(TokenKind::Colon) {
            if self.is_at_end()
                || self.check(TokenKind::RBrace)
                || self.check(TokenKind::Semicolon)
            {
                return Err(LangError::ParseError {
                    message: "Expect statement after ':' in if".to_string(),
                    line: self.peek().line,
                    file: self.source_name.clone(),
                });
            }
            let colon_line = self.previous().line;
            let then_branch = if self.peek().line > colon_line {
                self.parse_colon_indented_block(if_line)?
            } else {
                vec![self.statement()?]
            };
            return Ok(Stmt::If {
                condition,
                then_branch,
                else_branch: None,
                line: if_line,
            });
        }

        self.consume(TokenKind::LBrace, "Expect '{' or ':' after condition")?;
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

        // `while <condition>: <statement>` or Python indented block after `:`
        if self.match_token(TokenKind::Colon) {
            if self.is_at_end()
                || self.check(TokenKind::RBrace)
                || self.check(TokenKind::Semicolon)
            {
                return Err(LangError::ParseError {
                    message: "Expect statement after ':' in while".to_string(),
                    line: self.peek().line,
                    file: self.source_name.clone(),
                });
            }
            let colon_line = self.previous().line;
            let body = if self.peek().line > colon_line {
                self.parse_colon_indented_block(while_line)?
            } else {
                vec![self.statement()?]
            };
            return Ok(Stmt::While {
                condition,
                body,
                line: while_line,
            });
        }

        self.consume(TokenKind::LBrace, "Expect '{' or ':' after condition")?;
        let body = self.block()?;
        Ok(Stmt::While {
            condition,
            body,
            line: while_line,
        })
    }

    fn for_statement(&mut self) -> Result<Stmt, LangError> {
        let for_line = self.previous().line;

        // Парсим паттерн распаковки: for pattern in iterable { body }
        // Поддерживаем: for x in, for x, y in, for (x, y) in, for [x, y] in, for x, _, y in
        let pattern = self.parse_unpack_pattern()?;
        self.consume(TokenKind::In, "Expect 'in' after unpack pattern")?;
        let iterable = self.expression()?;

        // Single-line for: `for <target> in <iterable>: <statement>`
        if self.match_token(TokenKind::Colon) {
            if self.is_at_end()
                || self.check(TokenKind::RBrace)
                || self.check(TokenKind::Semicolon)
            {
                return Err(LangError::ParseError {
                    message: "Expect statement after ':' in for".to_string(),
                    line: self.peek().line,
                    file: self.source_name.clone(),
                });
            }
            let body = vec![self.statement()?];
            return Ok(Stmt::For {
                pattern,
                iterable,
                body,
                line: for_line,
            });
        }

        self.consume(TokenKind::LBrace, "Expect '{' or ':' before loop body")?;
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

    /// Левая часть распаковки: имя или `arr[i]` (без среза).
    fn expr_to_assign_target(&self, expr: &Expr) -> Result<AssignTarget, LangError> {
        match expr {
            Expr::Variable { name, .. } => Ok(AssignTarget::Name(name.clone())),
            Expr::ArrayIndex { array, index, line } => {
                let IndexExpr::Scalar(index_expr) = index else {
                    return Err(LangError::ParseError {
                        message: "Slice targets are not supported in unpack assignment".to_string(),
                        line: *line,
                        file: self.source_name.clone(),
                    });
                };
                Ok(AssignTarget::Index {
                    array: array.clone(),
                    index: index_expr.clone(),
                })
            }
            _ => Err(LangError::ParseError {
                message: "Invalid unpack assignment target".to_string(),
                line: expr.line(),
                file: self.source_name.clone(),
            }),
        }
    }

    /// После первого выражения: `target, target, ... = rhs` (не внутри `(...)`).
    fn try_parse_unpack_assign_after_first(
        &mut self,
        first_expr: &Expr,
    ) -> Result<Option<Expr>, LangError> {
        if self.is_inside_parentheses_at(self.current) || !self.check(TokenKind::Comma) {
            return Ok(None);
        }
        let first_target = match self.expr_to_assign_target(first_expr) {
            Ok(t) => t,
            Err(_) => return Ok(None),
        };
        let saved_position = self.current;
        self.advance(); // consume comma
        let mut targets = vec![first_target];
        loop {
            let next_expr = self.conditional()?;
            match self.expr_to_assign_target(&next_expr) {
                Ok(t) => targets.push(t),
                Err(_) => {
                    self.current = saved_position;
                    return Ok(None);
                }
            }
            if !self.match_token(TokenKind::Comma) {
                break;
            }
        }
        if !self.check(TokenKind::Equal) {
            self.current = saved_position;
            return Ok(None);
        }
        let line = first_expr.line();
        self.consume(TokenKind::Equal, "Expect '=' after unpack target list")?;
        let value = self.parse_unpack_rhs()?;
        Ok(Some(Expr::UnpackAssign {
            targets,
            value: Box::new(value),
            line,
        }))
    }

    /// RHS of unpack assign: `a, b = 1, 2` → `(1, 2)`; `a, b = f(1, 2)` stays a single call.
    /// Uses `conditional()` (not `assignment()`) so indexed elements like `a[i], a[j]` are not
    /// mistaken for nested unpack assignment on the LHS.
    fn parse_unpack_rhs(&mut self) -> Result<Expr, LangError> {
        let line = self.peek().line;
        let first = self.conditional()?;
        if !self.match_token(TokenKind::Comma) {
            return Ok(first);
        }
        let mut elements = vec![first];
        loop {
            elements.push(self.conditional()?);
            if !self.match_token(TokenKind::Comma) {
                break;
            }
        }
        Ok(Expr::TupleLiteral { elements, line })
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
                            message: "Only one variadic variable (*) allowed in unpack pattern"
                                .to_string(),
                            line: self.previous().line,
                            file: self.source_name.clone(),
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
                        message: "Variadic unpacking (*) not supported in nested patterns"
                            .to_string(),
                        line: self.previous().line,
                        file: self.source_name.clone(),
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
                        message: "Variadic unpacking (*) not supported in nested patterns"
                            .to_string(),
                        line: self.previous().line,
                        file: self.source_name.clone(),
                    });
                }
                let nested = self.parse_unpack_pattern_list()?;
                self.consume(
                    TokenKind::RBracket,
                    "Expect ']' after nested unpack pattern",
                )?;
                patterns.push(UnpackPattern::Nested(nested));
            } else {
                // Ошибка: ожидается переменная, wildcard или вложенный паттерн
                if is_variadic {
                    return Err(LangError::ParseError {
                        message: "Expect variable name after '*' in unpack pattern".to_string(),
                        line: self.peek().line,
                        file: self.source_name.clone(),
                    });
                }
                return Err(LangError::ParseError {
                    message: "Expect variable name, '_', '*', or nested pattern in unpack pattern"
                        .to_string(),
                    line: self.peek().line,
                    file: self.source_name.clone(),
                });
            }

            // Проверяем, есть ли еще элементы (запятая)
            if !self.match_token(TokenKind::Comma) {
                break;
            }

            // Если уже есть variadic, нельзя добавлять больше элементов после него
            if has_variadic {
                return Err(LangError::ParseError {
                    message: "Variadic variable (*) must be the last element in unpack pattern"
                        .to_string(),
                    line: self.previous().line,
                    file: self.source_name.clone(),
                });
            }
        }

        if patterns.is_empty() {
            return Err(LangError::ParseError {
                message: "Unpack pattern cannot be empty".to_string(),
                line: self.peek().line,
                file: self.source_name.clone(),
            });
        }

        Ok(patterns)
    }

    fn return_statement(&mut self) -> Result<Stmt, LangError> {
        let return_line = self.previous().line;
        // Bare `return` at EOL must not consume the next statement as a return value
        // (e.g. `if x: return` followed by `y = 1` on the next line).
        let value = if !self.check(TokenKind::Semicolon)
            && !self.check(TokenKind::RBrace)
            && self.peek().line <= return_line
        {
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
                Some(Expr::TupleLiteral {
                    elements,
                    line: return_line,
                })
            } else {
                // Одиночный возврат
                Some(first_expr)
            }
        } else {
            None
        };
        // Семиколон опционален для return
        self.match_token(TokenKind::Semicolon);
        Ok(Stmt::Return {
            value,
            line: return_line,
        })
    }

    fn ereturn_statement(&mut self) -> Result<Stmt, LangError> {
        let line = self.previous().line;
        let value = if !self.check(TokenKind::Semicolon)
            && !self.check(TokenKind::RBrace)
            && self.peek().line <= line
        {
            Some(self.expression()?)
        } else {
            None
        };
        self.match_token(TokenKind::Semicolon);
        Ok(Stmt::EReturn { value, line })
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
        Ok(Stmt::Continue {
            line: continue_line,
        })
    }

    fn throw_statement(&mut self) -> Result<Stmt, LangError> {
        let throw_line = self.previous().line;
        // Парсим выражение (значение ошибки)
        let value = self.expression()?;
        // Семиколон опционален для throw
        self.match_token(TokenKind::Semicolon);
        Ok(Stmt::Throw {
            value,
            line: throw_line,
        })
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
                Some(
                    self.consume(TokenKind::Identifier, "Expect variable name after 'as'")?
                        .lexeme
                        .clone(),
                )
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
                message: "try statement must have at least one catch block or finally block"
                    .to_string(),
                line: try_line,
                file: self.source_name.clone(),
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
                message: format!(
                    "Expected end of expression in interpolation, found {:?}",
                    self.peek().kind
                ),
                line: self.peek().line,
                file: self.source_name.clone(),
            });
        }
        Ok(expr)
    }

    /// Parse one expression from source string (used for "${...}" contents).
    fn parse_expression_from_source(
        &mut self,
        lexer: &mut crate::lexer::Lexer,
        source: &str,
        _line: usize,
    ) -> Result<Expr, LangError> {
        lexer.reset_source(source);
        let tokens = lexer.tokenize()?;
        let mut sub_parser =
            Parser::new_with_source_name_and_registry(tokens, None, self.operator_registry.clone());
        sub_parser.parse_single_expression()
    }

    /// Split interpolation content into expression part, optional "=" suffix, and optional ":format" suffix.
    /// E.g. "n=:.0f" → ("n", true, Some(".0f")); "a+b:.2f" → ("a+b", false, Some(".2f")).
    /// Colons inside `[...]` (slices/subscripts) are not format separators: "path[:5]" stays one expression.
    fn split_interpolation_suffix(content: &str) -> (String, bool, Option<String>) {
        let content = content.trim();
        let mut bracket_depth: i32 = 0;
        let mut format_colon: Option<usize> = None;
        for (i, c) in content.char_indices() {
            match c {
                '[' => bracket_depth += 1,
                ']' if bracket_depth > 0 => bracket_depth -= 1,
                ':' if bracket_depth == 0 => format_colon = Some(i),
                _ => {}
            }
        }
        let (middle, format_spec) = match format_colon {
            Some(colon_pos) => {
                let spec = content[colon_pos + 1..].trim();
                if Self::looks_like_format_spec(spec) {
                    (
                        content[..colon_pos].trim_end(),
                        Some(content[colon_pos + 1..].to_string()),
                    )
                } else {
                    (content, None)
                }
            }
            None => (content, None),
        };
        let (expr_content, include_name) = if middle.ends_with('=') {
            (middle[..middle.len() - 1].trim_end().to_string(), true)
        } else {
            (middle.to_string(), false)
        };
        (expr_content, include_name, format_spec)
    }

    /// True when text after a top-level `:` looks like a printf-style format, not slice syntax.
    fn looks_like_format_spec(spec: &str) -> bool {
        let s = spec.trim();
        if s.is_empty() || s.ends_with(']') {
            return false;
        }
        let first = s.chars().next().unwrap();
        matches!(first, '.' | ',' | '+' | '#' | '0')
            || first.is_ascii_digit()
            || "fdeFgGeExXos%".contains(first)
    }

    /// Unescape literal segments: lexer \$ pushes placeholder \u{E000}; we replace it with "$" (the "{" is already in the string)
    fn unescape_literal(s: &str) -> String {
        s.replace('\u{E000}', "$")
    }

    /// Inner text of a string token: `"""..."""` or `'...'` / `"..."`.
    fn string_lexeme_inner(lexeme: &str) -> String {
        if lexeme.len() >= 6 && lexeme.starts_with("\"\"\"") && lexeme.ends_with("\"\"\"") {
            lexeme[3..lexeme.len() - 3].to_string()
        } else if lexeme.len() >= 2 {
            let q = lexeme.chars().next().unwrap();
            if lexeme.ends_with(q) {
                lexeme[1..lexeme.len() - 1].to_string()
            } else {
                lexeme.to_string()
            }
        } else {
            lexeme.to_string()
        }
    }

    /// Split string content into interpolation segments; returns segments or error if unclosed "${".
    fn parse_interpolated_segments(
        &mut self,
        raw: &str,
        line: usize,
    ) -> Result<Vec<InterpolatedSegment>, LangError> {
        let mut segments = Vec::new();
        let mut lexer = crate::lexer::Lexer::new("");
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
                        file: None,
                    })?;
                    let expr_source = raw[pos + 2..end_byte].trim();
                    let (expr_content, include_name, format_spec) =
                        Self::split_interpolation_suffix(expr_source);
                    let expr = self.parse_expression_from_source(&mut lexer, &expr_content, line)?;
                    segments.push(InterpolatedSegment::Expr {
                        expr: Box::new(expr),
                        include_name,
                        display_name: if include_name {
                            Some(expr_content.clone())
                        } else {
                            None
                        },
                        format: format_spec,
                    });
                    literal_start = end_byte + 1;
                }
            }
        }
        Ok(segments)
    }

    /// If tokens at `start` match `ident (',' ident)+ '='`, returns index of `=`.
    fn scan_unpack_assign_equal_index(&self, start: usize) -> Option<usize> {
        if start >= self.tokens.len() || self.tokens[start].kind != TokenKind::Identifier {
            return None;
        }
        let mut i = start + 1;
        let mut saw_comma = false;
        while i < self.tokens.len() && self.tokens[i].kind == TokenKind::Comma {
            saw_comma = true;
            i += 1;
            if i >= self.tokens.len() || self.tokens[i].kind != TokenKind::Identifier {
                return None;
            }
            i += 1;
        }
        if !saw_comma {
            return None;
        }
        if i >= self.tokens.len() || self.tokens[i].kind != TokenKind::Equal {
            return None;
        }
        Some(i)
    }

    /// True when `pos` is inside `(...)` of a call or grouping, not at block/statement level
    /// where bare `a, b = rhs` is unpack assignment.
    fn is_inside_parentheses_at(&self, pos: usize) -> bool {
        if pos == 0 {
            return false;
        }
        if !self.tokens[..pos]
            .iter()
            .any(|t| matches!(t.kind, TokenKind::LParen | TokenKind::RParen))
        {
            return false;
        }

        let mut paren_count = 0;
        for i in (0..pos).rev() {
            match self.tokens[i].kind {
                TokenKind::Semicolon | TokenKind::LBrace | TokenKind::RBrace | TokenKind::Fn => {
                    if paren_count == 0 {
                        break;
                    }
                }
                TokenKind::RParen => paren_count += 1,
                TokenKind::LParen => {
                    if paren_count == 0 {
                        return true;
                    }
                    if paren_count > 0 {
                        paren_count -= 1;
                    }
                }
                _ => {}
            }
        }
        false
    }

    /// Build dotted path for property assignment targets (`a.b.c = ...`).
    fn property_assignment_path(
        &self,
        object: &Expr,
        field: &str,
        line: usize,
    ) -> Result<String, LangError> {
        match object {
            Expr::Variable { name, .. } => Ok(format!("{}.{}", name, field)),
            Expr::This { .. } => Ok(format!("this.{}", field)),
            Expr::Property {
                object: inner,
                name: prop,
                ..
            } => {
                let prefix = self.property_assignment_path(inner, prop, line)?;
                Ok(format!("{}.{}", prefix, field))
            }
            _ => Err(LangError::ParseError {
                message: "Invalid assignment target".to_string(),
                line,
                file: self.source_name.clone(),
            }),
        }
    }

    fn assignment(&mut self) -> Result<Expr, LangError> {
        // Проверяем, не является ли это распаковкой кортежа БЕЗ let: x, y = ... / x, y, z = ...
        // Это нужно проверить ДО вызова pratt_parse(), чтобы избежать ошибки при парсинге запятой
        if !self.is_at_end() {
            let token0 = self.peek();
            if token0.kind == TokenKind::Identifier {
                let saved_position = self.current;
                if self.scan_unpack_assign_equal_index(saved_position).is_some() {
                    if !self.is_inside_parentheses_at(saved_position) {
                        let line = token0.line;
                        let mut targets = Vec::new();

                        loop {
                            let name = self
                                .consume(TokenKind::Identifier, "Expect variable name")?
                                .lexeme
                                .clone();
                            targets.push(AssignTarget::Name(name));

                            if !self.match_token(TokenKind::Comma) {
                                break;
                            }
                        }

                        self.consume(TokenKind::Equal, "Expect '=' after variable list")?;
                        let value = self.parse_unpack_rhs()?;

                        return Ok(Expr::UnpackAssign {
                            targets,
                            value: Box::new(value),
                            line,
                        });
                    }
                }
            }
        }

        let expr = self.conditional()?;

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
                    Expr::Property {
                        object,
                        name: prop_name,
                        ..
                    } => {
                        // Рекурсивно строим путь к свойству
                        let base = match &**object {
                            Expr::Variable { name, .. } => name.clone(),
                            Expr::This { .. } => "this".to_string(),
                            _ => {
                                return Err(LangError::ParseError {
                                    message: "Invalid assignment target".to_string(),
                                    line: op_line,
                                    file: self.source_name.clone(),
                                })
                            }
                        };
                        format!("{}.{}.{}", base, prop_name, name)
                    }
                    _ => {
                        return Err(LangError::ParseError {
                            message: "Invalid assignment target".to_string(),
                            line: op_line,
                            file: self.source_name.clone(),
                        })
                    }
                };
                return Ok(Expr::AssignOp {
                    name: property_path,
                    op: op_kind,
                    value: Box::new(value),
                    line: op_line,
                });
            } else if let Expr::ArrayIndex { array, index, .. } = expr {
                if matches!(&index, IndexExpr::Slice { .. }) {
                    return Err(LangError::ParseError {
                        message: "Augmented assignment is not supported for array slices"
                            .to_string(),
                        line: op_line,
                        file: self.source_name.clone(),
                    });
                }
                let value = self.assignment()?;
                return Ok(Expr::AssignArrayOp {
                    array,
                    index,
                    op: op_kind,
                    value: Box::new(value),
                    line: op_line,
                });
            }
            return Err(LangError::ParseError {
                message: "Invalid assignment target".to_string(),
                line: op_line,
                file: self.source_name.clone(),
            });
        }

        // Распаковка: `a, b = ...` или `a[i], a[j] = ...`
        if let Some(unpack) = self.try_parse_unpack_assign_after_first(&expr)? {
            return Ok(unpack);
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
            } else if let Expr::Property {
                object,
                name,
                line: prop_line,
                ..
            } = &expr
            {
                if let Expr::TableColumnWrite { inner, .. } = object.as_ref() {
                    let value = self.assignment()?;
                    return Ok(Expr::AssignTableColumn {
                        table: inner.clone(),
                        column: Box::new(Expr::Literal {
                            value: Value::String(name.clone()),
                            line: *prop_line,
                        }),
                        value: Box::new(value),
                        line: equal_line,
                    });
                }
                // Присваивание к свойству: obj.field = value, node.prev.next = value, ...
                let value = self.assignment()?;
                let property_path =
                    self.property_assignment_path(object, name, equal_line)?;
                return Ok(Expr::Assign {
                    name: property_path,
                    value: Box::new(value),
                    line: equal_line,
                });
            } else if let Expr::ArrayIndex { array, index, .. } = &expr {
                if let Expr::TableColumnWrite { inner, .. } = array.as_ref() {
                    let IndexExpr::Scalar(column) = index else {
                        return Err(LangError::ParseError {
                            message: "Slice assignment is not supported for table column write"
                                .to_string(),
                            line: equal_line,
                            file: self.source_name.clone(),
                        });
                    };
                    let value = self.assignment()?;
                    return Ok(Expr::AssignTableColumn {
                        table: inner.clone(),
                        column: column.clone(),
                        value: Box::new(value),
                        line: equal_line,
                    });
                }
                let value = self.assignment()?;
                return Ok(Expr::AssignArray {
                    array: array.clone(),
                    index: index.clone(),
                    value: Box::new(value),
                    line: equal_line,
                });
            }
            return Err(LangError::ParseError {
                message: "Invalid assignment target".to_string(),
                line: equal_line,
                file: self.source_name.clone(),
            });
        }

        Ok(expr)
    }

    /// Whether the next tokens are Python `is` / `is not` (identity → `==` / `!=`) with precedence allowed by `min_bp`.
    fn peek_python_is(&self, min_bp: u8) -> bool {
        if !(self.check(TokenKind::Identifier) && self.peek().lexeme == "is") {
            return false;
        }
        self.operator_registry
            .get("==")
            .map(|info| binding_power(info).0 >= min_bp)
            .unwrap_or(false)
    }

    /// Python `x is y` / `x is not y` → `x == y` / `!(x == y)`.
    fn parse_python_is(&mut self, lhs: Expr) -> Result<Expr, LangError> {
        let info = self
            .operator_registry
            .get("==")
            .expect("peek_python_is");
        let (_, r_bp) = binding_power(info);
        let op_line = self.peek().line;
        self.advance(); // is
        let is_not = self.check(TokenKind::Identifier) && self.peek().lexeme == "not";
        if is_not {
            self.advance(); // not
        }
        let rhs = self.pratt_parse(r_bp)?;
        let eq = Expr::Binary {
            left: Box::new(lhs),
            op: BinaryOpKind::Builtin(TokenKind::EqualEqual),
            right: Box::new(rhs),
            line: op_line,
        };
        if is_not {
            Ok(Expr::Unary {
                op: TokenKind::Bang,
                right: Box::new(eq),
                line: op_line,
            })
        } else {
            Ok(eq)
        }
    }

    /// True when `not` starts an expression but is not `not in`, a call `not(...)`, or a bare variable `not`.
    fn looks_like_python_not_keyword(&self) -> bool {
        if !(self.check(TokenKind::Identifier) && self.peek().lexeme == "not") {
            return false;
        }
        if self.current + 1 >= self.tokens.len() {
            return false;
        }
        let next = &self.tokens[self.current + 1].kind;
        !matches!(
            next,
            TokenKind::In
                | TokenKind::LParen
                | TokenKind::LBrace
                | TokenKind::Colon
                | TokenKind::RParen
                | TokenKind::Semicolon
                | TokenKind::RBrace
                | TokenKind::Comma
                | TokenKind::Eof
        )
    }

    /// Whether the next tokens are Python `not in` with precedence allowed by `min_bp`.
    fn peek_python_not_in(&self, min_bp: u8) -> bool {
        if !(self.check(TokenKind::Identifier) && self.peek().lexeme == "not")
            || self.current + 1 >= self.tokens.len()
            || self.tokens[self.current + 1].kind != TokenKind::In
        {
            return false;
        }
        self.operator_registry
            .get("in")
            .map(|info| binding_power(info).0 >= min_bp)
            .unwrap_or(false)
    }

    /// Python `x not in y` → `!(x in y)`; caller must have verified [`Self::peek_python_not_in`].
    fn parse_python_not_in(&mut self, lhs: Expr) -> Result<Expr, LangError> {
        let info = self
            .operator_registry
            .get("in")
            .expect("peek_python_not_in");
        let (_, r_bp) = binding_power(info);
        let op_line = self.peek().line;
        self.advance(); // not
        self.advance(); // in
        let rhs = self.pratt_parse(r_bp)?;
        Ok(Expr::Unary {
            op: TokenKind::Bang,
            right: Box::new(Expr::Binary {
                left: Box::new(lhs),
                op: BinaryOpKind::Builtin(TokenKind::In),
                right: Box::new(rhs),
                line: op_line,
            }),
            line: op_line,
        })
    }

    fn is_comparison_binary_op(op: &BinaryOpKind) -> bool {
        matches!(
            op,
            BinaryOpKind::Builtin(
                TokenKind::Equal
                    | TokenKind::EqualEqual
                    | TokenKind::BangEqual
                    | TokenKind::Less
                    | TokenKind::Greater
                    | TokenKind::LessEqual
                    | TokenKind::GreaterEqual
            )
        )
    }

    /// Python `a < b < c` → `(a < b) and (b < c)`.
    fn fold_chained_comparisons(
        parts: Vec<Expr>,
        ops: Vec<BinaryOpKind>,
        line: usize,
    ) -> Expr {
        let mut acc = Expr::Binary {
            left: Box::new(parts[0].clone()),
            op: ops[0].clone(),
            right: Box::new(parts[1].clone()),
            line,
        };
        for i in 1..ops.len() {
            let next_cmp = Expr::Binary {
                left: Box::new(parts[i].clone()),
                op: ops[i].clone(),
                right: Box::new(parts[i + 1].clone()),
                line,
            };
            acc = Expr::Binary {
                left: Box::new(acc),
                op: BinaryOpKind::Builtin(TokenKind::And),
                right: Box::new(next_cmp),
                line,
            };
        }
        acc
    }

    /// Python-style ternary: `a if cond else b` (right-associative, low precedence).
    /// C-style ternary: `cond ? a : b` (same precedence, right-associative).
    /// The `if` in Python form must appear on the same line as the end of `then_expr`.
    fn conditional(&mut self) -> Result<Expr, LangError> {
        let mut expr = self.pratt_parse(0)?;
        if self.match_token(TokenKind::Question) {
            let line = self.previous().line;
            let then_branch = self.conditional()?;
            self.consume(
                TokenKind::Colon,
                "Expect ':' after '?' in conditional expression",
            )?;
            let else_branch = self.conditional()?;
            expr = Expr::If {
                condition: Box::new(expr),
                then_branch: IfBranch::Expr(Box::new(then_branch)),
                else_branch: IfBranch::Expr(Box::new(else_branch)),
                line,
            };
        } else if self.check(TokenKind::If) && self.peek().line == self.previous().line {
            self.advance();
            let line = self.previous().line;
            let condition = self.conditional()?;
            self.consume(
                TokenKind::Else,
                "Expect 'else' in conditional expression",
            )?;
            let else_branch = self.conditional()?;
            expr = Expr::If {
                condition: Box::new(condition),
                then_branch: IfBranch::Expr(Box::new(expr)),
                else_branch: IfBranch::Expr(Box::new(else_branch)),
                line,
            };
        }
        Ok(expr)
    }

    /// Block if-expression: `if cond { ... } else { ... }` in expression position.
    fn parse_if_expression(&mut self) -> Result<Expr, LangError> {
        let line = self.previous().line;
        let condition = self.conditional()?;
        self.consume(TokenKind::LBrace, "Expect '{' after condition in if expression")?;
        let then_branch = self.block()?;
        self.consume(TokenKind::Else, "Expect 'else' in if expression")?;
        self.consume(TokenKind::LBrace, "Expect '{' after 'else' in if expression")?;
        let else_branch = self.block()?;
        Ok(Expr::If {
            condition: Box::new(condition),
            then_branch: IfBranch::Block(then_branch),
            else_branch: IfBranch::Block(else_branch),
            line,
        })
    }

    /// Pratt / precedence climbing using [`OperatorRegistry`] (built-ins + preloaded plugin ops).
    fn pratt_parse(&mut self, min_bp: u8) -> Result<Expr, LangError> {
        let mut lhs = self.unary()?;
        loop {
            if self.peek_python_is(min_bp) {
                lhs = self.parse_python_is(lhs)?;
                continue;
            }
            if self.peek_python_not_in(min_bp) {
                lhs = self.parse_python_not_in(lhs)?;
                continue;
            }
            let Some((op_line, op_kind, l_bp, r_bp)) = self.peek_infix_binary_op()? else {
                break;
            };
            if l_bp < min_bp {
                break;
            }
            self.advance();
            let rhs = self.pratt_parse(r_bp)?;
            if Self::is_comparison_binary_op(&op_kind) {
                let mut parts = vec![lhs, rhs];
                let mut ops = vec![op_kind];
                let mut line = op_line;
                while let Some((next_line, next_op, next_l_bp, next_r_bp)) =
                    self.peek_infix_binary_op()?
                {
                    if !Self::is_comparison_binary_op(&next_op) || next_l_bp < min_bp {
                        break;
                    }
                    self.advance();
                    parts.push(self.pratt_parse(next_r_bp)?);
                    ops.push(next_op);
                    line = next_line;
                }
                lhs = if ops.len() == 1 {
                    Expr::Binary {
                        left: Box::new(parts.remove(0)),
                        op: ops.remove(0),
                        right: Box::new(parts.remove(0)),
                        line,
                    }
                } else {
                    Self::fold_chained_comparisons(parts, ops, line)
                };
            } else {
                lhs = Expr::Binary {
                    left: Box::new(lhs),
                    op: op_kind,
                    right: Box::new(rhs),
                    line: op_line,
                };
            }
        }
        Ok(lhs)
    }

    fn peek_infix_binary_op(&self) -> Result<Option<(usize, BinaryOpKind, u8, u8)>, LangError> {
        if self.is_at_end() {
            return Ok(None);
        }
        if self.check(TokenKind::StarStar) && self.check_next(TokenKind::StarStarEqual) {
            return Ok(None);
        }
        let t = self.peek();
        let kind = t.kind.clone();
        if kind == TokenKind::At {
            // Statement-level decorators must not be parsed as infix matmul after a full expression
            // (e.g. `print("...")` @cache fn ..., `{...}` @Abstract cls ...).
            if self.current + 2 < self.tokens.len() {
                let t1 = &self.tokens[self.current + 1];
                let t2 = &self.tokens[self.current + 2];
                if (t1.kind == TokenKind::Abstract && t2.kind == TokenKind::Cls)
                    || (t1.kind == TokenKind::Identifier
                        && t1.lexeme == "cache"
                        && t2.kind == TokenKind::Fn)
                    || (t1.kind == TokenKind::Identifier
                        && t1.lexeme == "ws_route"
                        && t2.kind == TokenKind::LParen)
                {
                    return Ok(None);
                }
            }
            if self.current + 1 < self.tokens.len() {
                let t1 = &self.tokens[self.current + 1];
                if t1.kind == TokenKind::Identifier && (t1.lexeme == "route" || t1.lexeme == "ws_route") {
                    return Ok(None);
                }
            }
            let Some(info) = self.operator_registry.get("@") else {
                return Err(LangError::ParseError {
                    message: "Unregistered infix operator '@' (import a module that registers it, e.g. import ml)"
                        .to_string(),
                    line: t.line,
                    file: self.source_name.clone(),
                });
            };
            let (l_bp, r_bp) = binding_power(info);
            return Ok(Some((
                t.line,
                BinaryOpKind::Plugin {
                    symbol: "@".to_string(),
                    name: info.name.clone(),
                },
                l_bp,
                r_bp,
            )));
        }
        let Some(sym) = token_kind_to_symbol(&kind) else {
            return Ok(None);
        };
        let Some(info) = self.operator_registry.get(&sym) else {
            return Ok(None);
        };
        let (l_bp, r_bp) = binding_power(info);
        let op_kind = BinaryOpKind::Builtin(kind);
        Ok(Some((t.line, op_kind, l_bp, r_bp)))
    }

    fn unary(&mut self) -> Result<Expr, LangError> {
        if self.looks_like_python_not_keyword() {
            return Err(LangError::ParseError {
                message: "Use `!` for logical negation instead of `not` (only `is not` and `not in` support `not`)"
                    .to_string(),
                line: self.peek().line,
                file: self.source_name.clone(),
            });
        }
        // `!` binds looser than `in` so `!x in arr` parses as `!(x in arr)`.
        // `min_bp` 41 stops before `&&` / `||` (tier 20) but still consumes `in` (tier 40).
        if self.match_token(TokenKind::Bang) {
            let op_line = self.previous().line;
            let op_kind = self.previous().kind.clone();
            let right = self.pratt_parse(41)?;
            return Ok(Expr::Unary {
                op: op_kind,
                right: Box::new(right),
                line: op_line,
            });
        }
        if self.match_token(TokenKind::Minus) {
            let op_line = self.previous().line;
            let op_kind = self.previous().kind.clone();
            let right = self.unary()?;
            return Ok(Expr::Unary {
                op: op_kind,
                right: Box::new(right),
                line: op_line,
            });
        }
        if self.match_token(TokenKind::Tilde) {
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
                    Expr::Variable { .. }
                    | Expr::Property { .. }
                    | Expr::Super { .. }
                    | Expr::Lambda { .. }
                    | Expr::Call { .. }
                    | Expr::CallValue { .. }
                    | Expr::MethodCall { .. } => {
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

            // Postfix `!` — запись колонки таблицы: orders!["col"] / orders!.col
            if self.check(TokenKind::Bang) {
                if self.current + 1 < self.tokens.len() {
                    let next = self.tokens[self.current + 1].kind.clone();
                    if next == TokenKind::LBracket || next == TokenKind::Dot {
                        let line = self.peek().line;
                        self.advance();
                        expr = Expr::TableColumnWrite {
                            inner: Box::new(expr),
                            line,
                        };
                        continue;
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
                let name = if self.match_token(TokenKind::At) {
                    let ident = self
                        .consume(TokenKind::Identifier, "Expect special method name after '@'")?
                        .lexeme
                        .clone();
                    format!("@{ident}")
                } else {
                    self
                        .consume(TokenKind::Identifier, "Expect property name after '.'")?
                        .lexeme
                        .clone()
                };
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
        let mut has_star_unpack = false;
        if !self.check(TokenKind::RParen) {
            loop {
                if args.len() >= 255 {
                    return Err(LangError::ParseError {
                        message: "Cannot have more than 255 arguments".to_string(),
                        line: self.peek().line,
                        file: self.source_name.clone(),
                    });
                }

                // Распаковка **expr (kwargs) или *expr (args)
                let arg = if self.check(TokenKind::StarStar) {
                    self.advance();
                    has_named = true;
                    Arg::UnpackObject(self.expression()?)
                } else if self.check(TokenKind::Star)
                    && self.current + 1 < self.tokens.len()
                    && self.tokens[self.current + 1].kind == TokenKind::Star
                {
                    self.advance();
                    self.advance();
                    has_named = true;
                    Arg::UnpackObject(self.expression()?)
                } else if self.check(TokenKind::Star) {
                    self.advance();
                    if has_named {
                        return Err(LangError::ParseError {
                            message: "* unpacking must appear before named arguments".to_string(),
                            line: self.peek().line,
                            file: self.source_name.clone(),
                        });
                    }
                    has_star_unpack = true;
                    Arg::UnpackArray(self.expression()?)
                } else if self.check(TokenKind::Identifier) {
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
                        self.consume(
                            TokenKind::Equal,
                            "Expect '=' after parameter name in named argument",
                        )?;
                        has_named = true;
                        let value = self.expression()?;
                        Arg::Named { name, value }
                    } else {
                        // Позиционный аргумент
                        // Проверяем, что после именованного аргумента не идет позиционный
                        if has_named {
                            return Err(LangError::ParseError {
                                message: "Positional argument follows named argument".to_string(),
                                line: self.peek().line,
                                file: self.source_name.clone(),
                            });
                        }
                        if has_star_unpack {
                            return Err(LangError::ParseError {
                                message: "Positional argument follows * unpacking".to_string(),
                                line: self.peek().line,
                                file: self.source_name.clone(),
                            });
                        }
                        let expr = self.expression()?;
                        Arg::Positional(expr)
                    }
                } else {
                    // Позиционный аргумент (не идентификатор, не * / **)
                    if has_named {
                        return Err(LangError::ParseError {
                            message: "Positional argument follows named argument".to_string(),
                            line: self.peek().line,
                            file: self.source_name.clone(),
                        });
                    }
                    if has_star_unpack {
                        return Err(LangError::ParseError {
                            message: "Positional argument follows * unpacking".to_string(),
                            line: self.peek().line,
                            file: self.source_name.clone(),
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
            Expr::Variable { name, .. } => Ok(Expr::Call {
                name,
                args,
                line: call_line,
            }),
            Expr::Super { .. } => {
                // super(...) - вызов конструктора родителя
                Ok(Expr::SuperCall {
                    args,
                    line: call_line,
                })
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
            Expr::Lambda { .. }
            | Expr::Call { .. }
            | Expr::CallValue { .. }
            | Expr::MethodCall { .. } => Ok(Expr::CallValue {
                callee: Box::new(callee),
                args,
                line: call_line,
            }),
            _ => Err(LangError::ParseError {
                message: "Can only call functions, variables, methods, and super".to_string(),
                line: paren.line,
                file: self.source_name.clone(),
            }),
        }
    }

    /// Парсит выражение внутри [] для индекса/фильтра. Не допускает присваивание,
    /// чтобы "age" = 28 разбиралось как сравнение (Binary), а не как Assign.
    fn parse_index_expression(&mut self) -> Result<Expr, LangError> {
        let left = self.pratt_parse(0)?;
        if self.match_token(TokenKind::Equal) {
            let op_line = self.previous().line;
            let right = self.pratt_parse(0)?;
            return Ok(Expr::Binary {
                left: Box::new(left),
                op: BinaryOpKind::Builtin(TokenKind::Equal),
                right: Box::new(right),
                line: op_line,
            });
        }
        if self.match_token(TokenKind::EqualEqual)
            || self.match_token(TokenKind::BangEqual)
            || self.match_token(TokenKind::Less)
            || self.match_token(TokenKind::Greater)
            || self.match_token(TokenKind::LessEqual)
            || self.match_token(TokenKind::GreaterEqual)
        {
            let op_line = self.previous().line;
            let op = self.previous().kind.clone();
            let right = self.pratt_parse(0)?;
            return Ok(Expr::Binary {
                left: Box::new(left),
                op: BinaryOpKind::Builtin(op),
                right: Box::new(right),
                line: op_line,
            });
        }
        Ok(left)
    }

    /// Часть среза после первого `:` (start уже съеден или None для `[:...`).
    fn parse_slice_after_start(
        &mut self,
        start: Option<Box<Expr>>,
    ) -> Result<IndexExpr, LangError> {
        let line = self.previous().line;
        let stop = if self.check(TokenKind::RBracket) {
            None
        } else if self.check(TokenKind::Colon) {
            self.advance();
            None
        } else {
            Some(Box::new(self.pratt_parse(0)?))
        };
        if self.match_token(TokenKind::Colon) {
            let step = if self.check(TokenKind::RBracket) {
                None
            } else {
                Some(Box::new(self.pratt_parse(0)?))
            };
            self.consume(TokenKind::RBracket, "Expect ']' after slice")?;
            return Ok(IndexExpr::Slice {
                start,
                stop,
                step,
                line,
            });
        }
        // `[::step]` после пустого stop: осталось `step]` (без третьего `:`)
        if stop.is_none() && !self.check(TokenKind::RBracket) {
            let step = Some(Box::new(self.pratt_parse(0)?));
            self.consume(TokenKind::RBracket, "Expect ']' after slice")?;
            return Ok(IndexExpr::Slice {
                start,
                stop,
                step,
                line,
            });
        }
        self.consume(TokenKind::RBracket, "Expect ']' after slice")?;
        Ok(IndexExpr::Slice {
            start,
            stop,
            step: None,
            line,
        })
    }

    /// Скаляр, срез или выражение для TableFilter внутри `[]`.
    fn parse_bracket_index_or_slice(&mut self) -> Result<IndexExpr, LangError> {
        if self.match_token(TokenKind::Colon) {
            return self.parse_slice_after_start(None);
        }
        let left = self.parse_index_expression()?;
        if self.check(TokenKind::RBracket) {
            self.advance();
            return Ok(IndexExpr::Scalar(Box::new(left)));
        }
        if self.match_token(TokenKind::Colon) {
            return self.parse_slice_after_start(Some(Box::new(left)));
        }
        Err(LangError::ParseError {
            message: "Expect ']' or ':' after array index expression".to_string(),
            line: self.peek().line,
            file: self.source_name.clone(),
        })
    }

    fn finish_array_index(&mut self, array: Expr) -> Result<Expr, LangError> {
        let index_line = self.previous().line; // Номер строки открывающей скобки (LBracket)
        let index = self.parse_bracket_index_or_slice()?;

        // Распознаём table["col" op value] и составные and/or как TableFilter
        if let IndexExpr::Scalar(inner) = &index {
            if let Some(predicate) = crate::parser::table_filter::try_extract_table_filter_pred(inner) {
                return Ok(Expr::TableFilter {
                    table: Box::new(array),
                    predicate,
                    line: index_line,
                });
            }
        }

        Ok(Expr::ArrayIndex {
            array: Box::new(array),
            index,
            line: index_line,
        })
    }

    fn primary(&mut self) -> Result<Expr, LangError> {
        if self.match_token(TokenKind::If) {
            return self.parse_if_expression();
        }
        if self.match_token(TokenKind::False) {
            let line = self.previous().line;
            return Ok(Expr::Literal {
                value: Value::Bool(false),
                line,
            });
        }
        if self.match_token(TokenKind::True) {
            let line = self.previous().line;
            return Ok(Expr::Literal {
                value: Value::Bool(true),
                line,
            });
        }
        if self.match_token(TokenKind::Null) {
            let line = self.previous().line;
            return Ok(Expr::Literal {
                value: Value::Null,
                line,
            });
        }
        if self.match_token(TokenKind::Inf) {
            let line = self.previous().line;
            return Ok(Expr::Literal {
                value: Value::Float(FloatValue::PosInfinity),
                line,
            });
        }
        if self.match_token(TokenKind::Nan) {
            let line = self.previous().line;
            return Ok(Expr::Literal {
                value: Value::Float(FloatValue::NaN),
                line,
            });
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
        if self.match_token(TokenKind::Return) {
            let line = self.previous().line;
            let value = if !self.check(TokenKind::Semicolon)
                && !self.check(TokenKind::RBrace)
                && !self.check(TokenKind::RParen)
                && !self.check(TokenKind::Comma)
                && !(self.check(TokenKind::Return) && self.peek().line > line)
            {
                Some(Box::new(self.expression()?))
            } else {
                None
            };
            return Ok(Expr::ExprReturn { value, line });
        }
        if self.match_token(TokenKind::Ireturn) {
            let line = self.previous().line;
            // Иначе `x = ireturn` и на следующей строке `return …` сливаются в `ireturn return …`
            // (пробелы/переводы строк между токенами игнорируются).
            let value = if !self.check(TokenKind::Semicolon)
                && !self.check(TokenKind::RBrace)
                && !self.check(TokenKind::RParen)
                && !self.check(TokenKind::Comma)
                && !(self.check(TokenKind::Return) && self.peek().line > line)
            {
                Some(Box::new(self.expression()?))
            } else {
                None
            };
            return Ok(Expr::Ireturn { value, line });
        }
        if self.match_token(TokenKind::Fn) {
            return self.parse_lambda_after_fn();
        }
        if self.match_token(TokenKind::Number) {
            let line = self.previous().line;
            let lexeme = self.previous().lexeme.clone();
            let value = crate::common::numeric::parse_number_lexeme(&lexeme).map_err(|_| {
                LangError::ParseError {
                    message: "Invalid number".to_string(),
                    line,
                    file: self.source_name.clone(),
                }
            })?;
            let lit = if lexeme.contains('.') {
                Value::Float(FloatValue::Finite(value))
            } else {
                Value::Number(value)
            };
            return Ok(Expr::Literal {
                value: lit,
                line,
            });
        }
        if self.match_token(TokenKind::String) {
            let line = self.previous().line;
            let lexeme = self.previous().lexeme.clone();
            let raw = Self::string_lexeme_inner(&lexeme);
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
                    let segments = self.parse_interpolated_segments(&raw, line)?;
                    return Ok(Expr::InterpolatedString { segments, line });
                }
            }
            return Ok(Expr::Literal {
                value: Value::String(Self::unescape_literal(&raw)),
                line,
            });
        }
        if self.match_token(TokenKind::At) {
            let line = self.previous().line;
            self.consume(
                TokenKind::Identifier,
                "Expect 'class' after '@' in expression",
            )?;
            if self.previous().lexeme != "class" {
                return Err(LangError::ParseError {
                    message: "After '@' only 'class' is allowed (e.g. @class.name)".to_string(),
                    line: self.previous().line,
                    file: self.source_name.clone(),
                });
            }
            return Ok(Expr::Variable {
                name: "@class".to_string(),
                line,
            });
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
                return Ok(Expr::TupleLiteral {
                    elements: vec![],
                    line: paren_line,
                });
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
                return Ok(Expr::TupleLiteral {
                    elements,
                    line: paren_line,
                });
            } else if self.check(TokenKind::RParen) {
                // Это группировка: (expr)
                self.consume(TokenKind::RParen, "Expect ')' after expression")?;
                return Ok(first_expr);
            } else {
                // Ошибка: ожидается либо запятая (кортеж), либо закрывающая скобка (группировка)
                return Err(LangError::ParseError {
                    message: "Expect ',' or ')' after expression in parentheses".to_string(),
                    line: self.peek().line,
                    file: self.source_name.clone(),
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
            message: format!(
                "Expect expression, found {:?} '{}' at line {}",
                token.kind, token.lexeme, token.line
            ),
            line: token.line,
            file: self.source_name.clone(),
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

    /// `@cache`: after `@` the word `cache` is lexed as a normal identifier.
    fn match_decorator_cache(&mut self) -> bool {
        if self.check(TokenKind::Identifier) && self.peek().lexeme == "cache" {
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

    /// Expression in comprehension `in` / `if` clauses — no Python `a if b else c` ternary.
    fn parse_comprehension_clause_expr(&mut self) -> Result<Expr, LangError> {
        self.pratt_parse(0)
    }

    /// `for` already consumed; parses `pat in iter (for ... | if ...)*`.
    fn parse_for_in_comprehension_clauses(
        &mut self,
        end_kind: TokenKind,
        end_msg: &str,
        mid_err: &str,
    ) -> Result<Vec<ListComprehensionClause>, LangError> {
        let mut clauses: Vec<ListComprehensionClause> = Vec::new();
        let pattern = self.parse_unpack_pattern()?;
        self.consume(
            TokenKind::In,
            "Expect 'in' after 'for' pattern in comprehension",
        )?;
        let iterable = Box::new(self.parse_comprehension_clause_expr()?);
        clauses.push(ListComprehensionClause::For { pattern, iterable });

        loop {
            if self.peek().kind == end_kind {
                break;
            }
            if self.match_token(TokenKind::For) {
                let pattern = self.parse_unpack_pattern()?;
                self.consume(
                    TokenKind::In,
                    "Expect 'in' after 'for' pattern in comprehension",
                )?;
                let iterable = Box::new(self.parse_comprehension_clause_expr()?);
                clauses.push(ListComprehensionClause::For { pattern, iterable });
            } else if self.match_token(TokenKind::If) {
                clauses.push(ListComprehensionClause::If {
                    condition: Box::new(self.parse_comprehension_clause_expr()?),
                });
            } else {
                let file = self.source_name.clone();
                return Err(LangError::ParseError {
                    message: mid_err.to_string(),
                    line: self.peek().line,
                    file,
                });
            }
        }

        self.consume(end_kind, end_msg)?;
        Ok(clauses)
    }

    fn array_literal(&mut self) -> Result<Expr, LangError> {
        let line = self.previous().line;

        if self.check(TokenKind::RBracket) {
            self.consume(TokenKind::RBracket, "Expect ']' after '['")?;
            return Ok(Expr::ArrayLiteral {
                elements: Vec::new(),
                line,
            });
        }

        let first = self.expression()?;

        if self.match_token(TokenKind::For) {
            let clauses = self.parse_for_in_comprehension_clauses(
                TokenKind::RBracket,
                "Expect ']' after list comprehension",
                "Expect ']', 'for', or 'if' in list comprehension",
            )?;
            return Ok(Expr::ListComprehension {
                elt: Box::new(first),
                clauses,
                line,
            });
        }

        let mut elements = vec![first];
        while self.match_token(TokenKind::Comma) {
            if self.check(TokenKind::RBracket) {
                break;
            }
            elements.push(self.expression()?);
        }

        self.consume(TokenKind::RBracket, "Expect ']' after array elements")?;

        // Arrays are mutable (push, etc.). Do not fold to Expr::Literal(Value::Array): chunk constant
        // deduplication would reuse one heap id for every `[]` / `[1,2]` site, breaking distinct locals.
        Ok(Expr::ArrayLiteral { elements, line })
    }

    fn object_pair_from_exprs(
        key_expr: Expr,
        value_expr: Expr,
    ) -> Result<ObjectPair, LangError> {
        use crate::common::value::Value;
        match key_expr {
            Expr::Variable { .. } => Ok(ObjectPair::KeyValueExpr(
                Box::new(key_expr),
                Box::new(value_expr),
            )),
            Expr::Literal {
                value: Value::Number(n),
                ..
            } => Ok(ObjectPair::KeyValue(ObjectLiteralKey::Number(n), value_expr)),
            Expr::Literal {
                value: Value::String(s),
                ..
            } => Ok(ObjectPair::KeyValue(ObjectLiteralKey::String(s), value_expr)),
            other => Ok(ObjectPair::KeyValueExpr(
                Box::new(other),
                Box::new(value_expr),
            )),
        }
    }

    fn object_literal(&mut self) -> Result<Expr, LangError> {
        let line = self.previous().line;
        let mut pairs: Vec<ObjectPair> = Vec::new();

        if !self.check(TokenKind::RBrace) {
            loop {
                if self.check(TokenKind::StarStar) {
                    self.advance();
                    let value = self.expression()?;
                    pairs.push(ObjectPair::Spread(value));
                } else if self.check(TokenKind::Star)
                    && self.current + 1 < self.tokens.len()
                    && self.tokens[self.current + 1].kind == TokenKind::Star
                {
                    self.advance();
                    self.advance();
                    let value = self.expression()?;
                    pairs.push(ObjectPair::Spread(value));
                } else {
                    let key_expr = self.expression()?;
                    // Python set literal: `{ a }` or `{ a, b, ... }` (no `:` / `=` before `,` or `}`).
                    if self.check(TokenKind::Comma) || self.check(TokenKind::RBrace) {
                        if !pairs.is_empty() {
                            let file = self.source_name.clone();
                            return Err(LangError::ParseError {
                                message: "Set literal must be the only content of `{ }` (cannot mix with spreads or other keys)".to_string(),
                                line,
                                file,
                            });
                        }
                        let mut elements = vec![key_expr];
                        while self.match_token(TokenKind::Comma) {
                            if self.check(TokenKind::RBrace) {
                                break;
                            }
                            elements.push(self.expression()?);
                        }
                        self.consume(TokenKind::RBrace, "Expect '}' after set literal")?;
                        return Ok(Expr::Call {
                            name: "set".to_string(),
                            args: vec![Arg::Positional(Expr::ArrayLiteral { elements, line })],
                            line,
                        });
                    }
                    // Python set comprehension: `{ elt for pat in iter }` (no `:` before `for`).
                    if self.match_token(TokenKind::For) {
                        if !pairs.is_empty() {
                            let file = self.source_name.clone();
                            return Err(LangError::ParseError {
                                message: "Set comprehension must be the only content of `{ }` (cannot mix with spreads or other keys)".to_string(),
                                line,
                                file,
                            });
                        }
                        let clauses = self.parse_for_in_comprehension_clauses(
                            TokenKind::RBrace,
                            "Expect '}' after set comprehension",
                            "Expect '}', 'for', or 'if' in set comprehension",
                        )?;
                        return Ok(Expr::Call {
                            name: "set".to_string(),
                            args: vec![Arg::Positional(Expr::ListComprehension {
                                elt: Box::new(key_expr),
                                clauses,
                                line,
                            })],
                            line,
                        });
                    }
                    let used_eq = if self.match_token(TokenKind::Equal) {
                        true
                    } else {
                        self.consume(
                            TokenKind::Colon,
                            "Expect ':' in dict comprehension or after object key (use ':' before 'for')",
                        )?;
                        false
                    };
                    let value_expr = self.expression()?;

                    if !used_eq && self.match_token(TokenKind::For) {
                        if !pairs.is_empty() {
                            let file = self.source_name.clone();
                            return Err(LangError::ParseError {
                                message: "Dict comprehension must be the only content of `{ }` (cannot mix with spreads or other keys)".to_string(),
                                line,
                                file,
                            });
                        }
                        let loop_var = self
                            .consume(
                                TokenKind::Identifier,
                                "Expect identifier after 'for' in dict comprehension",
                            )?
                            .lexeme
                            .clone();
                        self.consume(
                            TokenKind::In,
                            "Expect 'in' after loop variable in dict comprehension",
                        )?;
                        let iterable = Box::new(self.parse_comprehension_clause_expr()?);
                        let condition = if self.match_token(TokenKind::If) {
                            Some(Box::new(self.parse_comprehension_clause_expr()?))
                        } else {
                            None
                        };
                        self.consume(
                            TokenKind::RBrace,
                            "Expect '}' after dict comprehension",
                        )?;
                        return Ok(Expr::DictComprehension {
                            key_expr: Box::new(key_expr),
                            value_expr: Box::new(value_expr),
                            loop_var,
                            iterable,
                            condition,
                            line,
                        });
                    }

                    pairs.push(Self::object_pair_from_exprs(key_expr, value_expr)?);
                }
                if !self.match_token(TokenKind::Comma) {
                    break;
                }
                if self.check(TokenKind::RBrace) {
                    break;
                }
            }
        }

        self.consume(TokenKind::RBrace, "Expect '}' after object pairs")?;

        let has_spread = pairs.iter().any(|p| matches!(p, ObjectPair::Spread(_)));
        if has_spread {
            return Ok(Expr::ObjectLiteral { pairs, line });
        }
        let mut all_literals = true;
        let mut all_string_keys = true;
        for p in &pairs {
            match p {
                ObjectPair::KeyValue(key, expr) => {
                    if matches!(key, ObjectLiteralKey::Number(_)) {
                        all_string_keys = false;
                    }
                    if !matches!(expr, Expr::Literal { .. }) {
                        all_literals = false;
                        break;
                    }
                }
                ObjectPair::KeyValueExpr(_, _) => {
                    all_literals = false;
                    all_string_keys = false;
                    break;
                }
                ObjectPair::Spread(_) => {}
            }
        }
        if all_literals && all_string_keys {
            let mut object_map = std::collections::HashMap::new();
            for p in &pairs {
                if let ObjectPair::KeyValue(key, Expr::Literal { value, .. }) = p {
                    let sk = match key {
                        ObjectLiteralKey::Ident(s) | ObjectLiteralKey::String(s) => s.clone(),
                        ObjectLiteralKey::Number(_) => unreachable!(),
                    };
                    object_map.insert(sk, value.clone());
                }
            }
            Ok(Expr::Literal {
                value: Value::Object(Rc::new(RefCell::new(ObjectKind::Legacy(object_map)))),
                line,
            })
        } else {
            Ok(Expr::ObjectLiteral { pairs, line })
        }
    }

    /// Одна альтернатива типа после `|` в аннотации (null, литерал, скобочная группа, либо `ident` с опциональным `[...]`).
    fn parse_type_atom_union_branch(&mut self) -> Result<TypePart, LangError> {
        if self.match_token(TokenKind::LParen) {
            return self.parse_type_paren_union();
        }
        self.parse_type_leaf_plain()
    }

    /// `(a | b | c)` внутри аннотации — возвращает `Union` если альтернатив больше одной.
    fn parse_type_paren_union(&mut self) -> Result<TypePart, LangError> {
        let mut parts = Vec::new();
        parts.push(self.parse_type_atom_union_branch()?);
        while self.match_token(TokenKind::Pipe) {
            parts.push(self.parse_type_atom_union_branch()?);
        }
        self.consume(TokenKind::RParen, "Expect ')' after type group")?;
        Ok(if parts.len() == 1 {
            parts.pop().expect("one element")
        } else {
            TypePart::Union(parts)
        })
    }

    /// null, строковый литерал типа или `identifier` без ведущей `(`; для `[` — число (`str[N]`) либо generic-аргументы.
    fn parse_type_leaf_plain(&mut self) -> Result<TypePart, LangError> {
        if self.check(TokenKind::Null) {
            self.advance();
            return Ok(TypePart::TypeName("null".to_string()));
        }
        if self.check(TokenKind::String) {
            let tok = self.advance();
            let inner = Self::string_lexeme_inner(&tok.lexeme);
            return Ok(TypePart::LiteralStr(inner));
        }
        let base = self
            .consume(TokenKind::Identifier, "Expect type name")?
            .lexeme
            .clone();
        if self.match_token(TokenKind::LBracket) {
            if self.check(TokenKind::Number) {
                let len = self
                    .consume(TokenKind::Number, "Expect number in type subscript")?
                    .lexeme
                    .clone();
                self.consume(TokenKind::RBracket, "Expect ']' after type parameter")?;
                return Ok(TypePart::TypeName(format!("{}[{}]", base, len)));
            }
            let mut args = Vec::new();
            args.push(self.parse_type_atom_union_branch()?);
            while self.match_token(TokenKind::Comma) {
                args.push(self.parse_type_atom_union_branch()?);
            }
            self.consume(TokenKind::RBracket, "Expect ']' after type arguments")?;
            return Ok(TypePart::Generic { base, args });
        }
        Ok(TypePart::TypeName(base))
    }

    /// Парсит union типы для параметров/полей/`->`; верхний уровень: `alt | alt`; литералы `["a","b"]`; скобочная группировка `(a | b)`.
    fn parse_type_name(&mut self) -> Result<Vec<TypePart>, LangError> {
        // Массив литералов: ["dev", "prod"] — union перечисленных значений
        if self.match_token(TokenKind::LBracket) {
            let mut types = Vec::new();
            types.push(self.parse_type_leaf_plain()?);
            while self.match_token(TokenKind::Comma) {
                types.push(self.parse_type_leaf_plain()?);
            }
            self.consume(TokenKind::RBracket, "Expect ']' after array type")?;
            return Ok(types);
        }

        let had_paren = self.match_token(TokenKind::LParen);
        let mut types = Vec::new();
        types.push(self.parse_type_atom_union_branch()?);
        while self.match_token(TokenKind::Pipe) {
            types.push(self.parse_type_atom_union_branch()?);
        }
        if had_paren {
            self.consume(TokenKind::RParen, "Expect ')' after type")?;
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
                file: self.source_name.clone(),
            })
        }
    }
}
