//! User-defined infix operators (stage 0 constraints).
//!
//! - Operators must be registered **before** expressions that use them are parsed (see preload in `lib.rs`).
//! - One symbol → one operator at parse time (no type-based overloading in the grammar).
//! - Precedence and associativity come **only** from registration, not from types.
//! - VM keeps a single opaque binary hook: `opaque_binop` dispatches by logical `op_name` string.
//!
//! **Conflict policy:** registering the same `symbol` twice (via [`OperatorRegistry::register`] or
//! [`OperatorRegistry::merge_from_descriptor_value`]) is always an error — deterministic, no silent skip.

use std::collections::HashMap;
use std::fmt::Write;
use std::sync::Arc;

use crate::common::error::LangError;
use crate::common::value::Value;
use crate::lexer::TokenKind;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Associativity {
    Left,
    Right,
}

fn assoc_label(a: Associativity) -> &'static str {
    match a {
        Associativity::Left => "left",
        Associativity::Right => "right",
    }
}

/// Registered infix operator: `symbol` is source text (e.g. `"@"`), `name` is VM/plugin key (e.g. `"matmul"`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OperatorInfo {
    pub symbol: String,
    pub name: String,
    pub precedence: u8,
    pub associativity: Associativity,
    /// Origin of this binding (`"builtin"`, dylib module name, etc.) for debug and error messages.
    pub source_module: Option<String>,
}

fn builtin_op(symbol: &str, name: &str, precedence: u8, assoc: Associativity) -> OperatorInfo {
    OperatorInfo {
        symbol: symbol.to_string(),
        name: name.to_string(),
        precedence,
        associativity: assoc,
        source_module: Some("builtin".to_string()),
    }
}

/// Table keyed by operator symbol string (`"+"`, `"@"`, `"or"`, …).
#[derive(Debug, Clone, Default)]
pub struct OperatorRegistry {
    by_symbol: HashMap<String, OperatorInfo>,
}

impl OperatorRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Built-in binary operators and their precedence tier (higher binds tighter).
    /// Tiers: or=10, and=20, eq=30, cmp=40, add=50, mul=60, pow=70 (right-assoc).
    pub fn with_builtins() -> Self {
        let mut r = Self::new();
        let _ = r.register(builtin_op("or", "or", 10, Associativity::Left));
        let _ = r.register(builtin_op("and", "and", 20, Associativity::Left));
        for (sym, name, _prec) in [("==", "eq", 30), ("!=", "ne", 30)] {
            let _ = r.register(builtin_op(sym, name, 30, Associativity::Left));
        }
        for (sym, name, prec) in [
            ("<", "lt", 40),
            (">", "gt", 40),
            ("<=", "le", 40),
            (">=", "ge", 40),
            ("in", "in", 40),
        ] {
            let _ = r.register(builtin_op(sym, name, prec, Associativity::Left));
        }
        for sym in ["|", "^", "&"] {
            let name = match sym {
                "|" => "bor",
                "^" => "bxor",
                "&" => "band",
                _ => "band",
            };
            let prec = match sym {
                "|" => 42,
                "^" => 43,
                "&" => 44,
                _ => 44,
            };
            let _ = r.register(builtin_op(sym, name, prec, Associativity::Left));
        }
        for sym in ["<<", ">>"] {
            let name = if sym == "<<" { "shl" } else { "shr" };
            let _ = r.register(builtin_op(sym, name, 45, Associativity::Left));
        }
        for sym in ["+", "-"] {
            let _ = r.register(builtin_op(
                sym,
                if sym == "+" { "add" } else { "sub" },
                50,
                Associativity::Left,
            ));
        }
        for sym in ["*", "/", "//", "%"] {
            let name = match sym {
                "*" => "mul",
                "/" => "div",
                "//" => "idiv",
                "%" => "mod",
                _ => "mul",
            };
            let _ = r.register(builtin_op(sym, name, 60, Associativity::Left));
        }
        let _ = r.register(builtin_op("**", "pow", 70, Associativity::Right));
        r
    }

    pub fn register(&mut self, info: OperatorInfo) -> Result<(), LangError> {
        if let Some(existing) = self.by_symbol.get(&info.symbol) {
            return Err(LangError::ParseError {
                message: format!(
                    "Operator '{}' already registered (existing source: {:?}, new: {:?})",
                    info.symbol, existing.source_module, info.source_module
                ),
                line: 0,
                file: None,
            });
        }
        self.by_symbol.insert(info.symbol.clone(), info);
        Ok(())
    }

    /// Merge rows from a native module's `operator_descriptor` return value.
    /// `source_module` labels the dylib (e.g. `"ml"`) for errors and [`OperatorInfo::source_module`].
    pub fn merge_from_descriptor_value(
        &mut self,
        v: &Value,
        source_module: &str,
    ) -> Result<(), LangError> {
        let Value::Array(rows) = v else {
            return Err(LangError::runtime_error(
                "operator_descriptor must return an array".to_string(),
                0,
            ));
        };
        for row in rows.borrow().iter() {
            let Value::Array(cols) = row else {
                return Err(LangError::runtime_error(
                    "operator_descriptor: each row must be an array".to_string(),
                    0,
                ));
            };
            let c = cols.borrow();
            if c.len() < 4 {
                return Err(LangError::runtime_error(
                    "operator_descriptor row must be [symbol, name, precedence, assoc]".to_string(),
                    0,
                ));
            }
            let symbol = match &c[0] {
                Value::String(s) => s.clone(),
                _ => {
                    return Err(LangError::runtime_error(
                        "operator_descriptor: symbol must be a string".to_string(),
                        0,
                    ))
                }
            };
            let name = match &c[1] {
                Value::String(s) => s.clone(),
                _ => {
                    return Err(LangError::runtime_error(
                        "operator_descriptor: name must be a string".to_string(),
                        0,
                    ))
                }
            };
            let prec = match &c[2] {
                Value::Number(n) => *n as u8,
                _ => {
                    return Err(LangError::runtime_error(
                        "operator_descriptor: precedence must be a number".to_string(),
                        0,
                    ))
                }
            };
            let assoc = match &c[3] {
                Value::String(s) if s.eq_ignore_ascii_case("left") => Associativity::Left,
                Value::String(s) if s.eq_ignore_ascii_case("right") => Associativity::Right,
                Value::Number(n) => {
                    if *n == 0.0 {
                        Associativity::Left
                    } else {
                        Associativity::Right
                    }
                }
                _ => {
                    return Err(LangError::runtime_error(
                        "operator_descriptor: assoc must be \"left\", \"right\", 0, or 1"
                            .to_string(),
                        0,
                    ))
                }
            };
            if let Some(existing) = self.by_symbol.get(&symbol) {
                return Err(LangError::runtime_error(
                    format!(
                        "Operator '{}' already registered (existing source: {:?}, attempted module: {})",
                        symbol, existing.source_module, source_module
                    ),
                    0,
                ));
            }
            self.by_symbol.insert(
                symbol.clone(),
                OperatorInfo {
                    symbol,
                    name,
                    precedence: prec,
                    associativity: assoc,
                    source_module: Some(source_module.to_string()),
                },
            );
        }
        Ok(())
    }

    /// Human-readable table for runtime introspection (`debug.operators()`).
    pub fn format_debug_text(&self) -> String {
        let mut items: Vec<&OperatorInfo> = self.by_symbol.values().collect();
        items.sort_by(|a, b| a.symbol.cmp(&b.symbol));
        let mut out = String::new();
        for info in items {
            let src = info.source_module.as_deref().unwrap_or("(unknown)");
            let _ = writeln!(
                out,
                "symbol={}\tname={}\tprecedence={}\tassociativity={}\tsource={}",
                info.symbol,
                info.name,
                info.precedence,
                assoc_label(info.associativity),
                src
            );
        }
        out
    }

    pub fn get(&self, symbol: &str) -> Option<&OperatorInfo> {
        self.by_symbol.get(symbol)
    }

    pub fn contains_symbol(&self, symbol: &str) -> bool {
        self.by_symbol.contains_key(symbol)
    }
}

/// `Arc` wrapper used by the parser; clone is cheap.
pub type SharedOperatorRegistry = Arc<OperatorRegistry>;

/// Map a token to the registry key (symbol string).
pub fn token_kind_to_symbol(kind: &TokenKind) -> Option<String> {
    Some(match kind {
        TokenKind::Or => "or".to_string(),
        TokenKind::And => "and".to_string(),
        TokenKind::EqualEqual => "==".to_string(),
        TokenKind::BangEqual => "!=".to_string(),
        TokenKind::Less => "<".to_string(),
        TokenKind::Greater => ">".to_string(),
        TokenKind::LessEqual => "<=".to_string(),
        TokenKind::GreaterEqual => ">=".to_string(),
        TokenKind::In => "in".to_string(),
        TokenKind::Plus => "+".to_string(),
        TokenKind::Minus => "-".to_string(),
        TokenKind::Star => "*".to_string(),
        TokenKind::StarStar => "**".to_string(),
        TokenKind::Slash => "/".to_string(),
        TokenKind::SlashSlash => "//".to_string(),
        TokenKind::Percent => "%".to_string(),
        TokenKind::LessLess => "<<".to_string(),
        TokenKind::GreaterGreater => ">>".to_string(),
        TokenKind::Amp => "&".to_string(),
        TokenKind::Pipe => "|".to_string(),
        TokenKind::Caret => "^".to_string(),
        TokenKind::At => "@".to_string(),
        _ => return None,
    })
}

/// Left/right binding power for Pratt parsing from tier `precedence` (0–255) and associativity.
pub fn binding_power(info: &OperatorInfo) -> (u8, u8) {
    let p = info.precedence.saturating_mul(2);
    match info.associativity {
        Associativity::Left => (p, p.saturating_add(1)),
        Associativity::Right => (p.saturating_add(1), p),
    }
}
