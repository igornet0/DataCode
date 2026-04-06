// Лексер без использования regex

use super::token::{Token, TokenKind};
use crate::common::error::LangError;

pub struct Lexer {
    source: Vec<char>,
    current: usize,
    line: usize,
    source_name: Option<String>,
}

impl Lexer {
    pub fn new(source: &str) -> Self {
        Self::new_with_source_name(source, None)
    }

    pub fn new_with_source_name(source: &str, source_name: Option<&str>) -> Self {
        Self {
            source: source.chars().collect(),
            current: 0,
            line: 1,
            source_name: source_name.map(String::from),
        }
    }

    /// Reuse this lexer for another source slice (e.g. successive `"${...}"` expression fragments).
    pub fn reset_source(&mut self, source: &str) {
        self.source = source.chars().collect();
        self.current = 0;
        self.line = 1;
    }

    pub fn tokenize(&mut self) -> Result<Vec<Token>, LangError> {
        let mut tokens = Vec::new();
        loop {
            let token = self.next_token()?;
            let is_eof = matches!(token.kind, TokenKind::Eof);
            tokens.push(token);
            if is_eof {
                break;
            }
        }
        Ok(tokens)
    }

    pub fn next_token(&mut self) -> Result<Token, LangError> {
        self.skip_whitespace();

        if self.is_at_end() {
            return Ok(self.make_token(TokenKind::Eof));
        }

        let start = self.current;
        let c = self.advance();

        let token = match c {
            '(' => {
                let token = self.make_token(TokenKind::LParen);
                return Ok(token);
            }
            ')' => {
                let token = self.make_token(TokenKind::RParen);
                return Ok(token);
            }
            '{' => {
                let token = self.make_token(TokenKind::LBrace);
                return Ok(token);
            }
            '}' => {
                let token = self.make_token(TokenKind::RBrace);
                return Ok(token);
            }
            '[' => {
                let token = self.make_token(TokenKind::LBracket);
                return Ok(token);
            }
            ']' => {
                let token = self.make_token(TokenKind::RBracket);
                return Ok(token);
            }
            ',' => {
                let token = self.make_token(TokenKind::Comma);
                return Ok(token);
            }
            ';' => {
                let token = self.make_token(TokenKind::Semicolon);
                return Ok(token);
            }
            '.' => {
                if self.match_char('.') && self.match_char('.') {
                    let token = self.make_token(TokenKind::Ellipsis);
                    return Ok(token);
                }
                let token = self.make_token(TokenKind::Dot);
                return Ok(token);
            }
            ':' => {
                let token = self.make_token(TokenKind::Colon);
                return Ok(token);
            }
            '|' => {
                let token = self.make_token(TokenKind::Pipe);
                return Ok(token);
            }
            '@' => {
                let token = self.make_token(TokenKind::At);
                return Ok(token);
            }
            '+' => {
                if self.match_char('=') {
                    // Оператор +=
                    let token = self.make_token(TokenKind::PlusEqual);
                    return Ok(token);
                } else {
                    let token = self.make_token(TokenKind::Plus);
                    return Ok(token);
                }
            }
            '-' => {
                if self.match_char('>') {
                    // Оператор ->
                    let token = self.make_token(TokenKind::Arrow);
                    return Ok(token);
                } else if self.match_char('=') {
                    // Оператор -=
                    let token = self.make_token(TokenKind::MinusEqual);
                    return Ok(token);
                } else {
                    let token = self.make_token(TokenKind::Minus);
                    return Ok(token);
                }
            }
            '*' => {
                if self.match_char('*') {
                    // Проверяем, это ** или **=
                    if self.match_char('=') {
                        // Оператор **=
                        let token = self.make_token(TokenKind::StarStarEqual);
                        return Ok(token);
                    } else {
                        // Оператор **
                        let token = self.make_token(TokenKind::StarStar);
                        return Ok(token);
                    }
                } else if self.match_char('=') {
                    // Оператор *=
                    let token = self.make_token(TokenKind::StarEqual);
                    return Ok(token);
                } else {
                    let token = self.make_token(TokenKind::Star);
                    return Ok(token);
                }
            }
            '/' => {
                if self.match_char('/') {
                    // Оператор // (целочисленное деление) или //=
                    if self.match_char('=') {
                        // Оператор //=
                        let token = self.make_token(TokenKind::SlashSlashEqual);
                        return Ok(token);
                    } else {
                        // Оператор // (целочисленное деление)
                        let token = self.make_token(TokenKind::SlashSlash);
                        return Ok(token);
                    }
                } else if self.match_char('=') {
                    // Оператор /=
                    let token = self.make_token(TokenKind::SlashEqual);
                    return Ok(token);
                } else {
                    let token = self.make_token(TokenKind::Slash);
                    return Ok(token);
                }
            }
            '%' => {
                if self.match_char('=') {
                    // Оператор %=
                    let token = self.make_token(TokenKind::PercentEqual);
                    return Ok(token);
                } else {
                    let token = self.make_token(TokenKind::Percent);
                    return Ok(token);
                }
            }
            '#' => {
                // Комментарий до конца строки (стиль #)
                while self.peek() != '\n' && !self.is_at_end() {
                    self.advance();
                }
                // Продолжаем поиск следующего токена
                return self.next_token();
            }
            '!' => {
                let kind = if self.match_char('=') {
                    TokenKind::BangEqual
                } else {
                    TokenKind::Bang
                };
                let token = self.make_token(kind);
                return Ok(token);
            }
            '=' => {
                let kind = if self.match_char('=') {
                    TokenKind::EqualEqual
                } else if self.match_char('>') {
                    TokenKind::FatArrow
                } else {
                    TokenKind::Equal
                };
                let token = self.make_token(kind);
                return Ok(token);
            }
            '<' => {
                let kind = if self.match_char('=') {
                    TokenKind::LessEqual
                } else {
                    TokenKind::Less
                };
                let token = self.make_token(kind);
                return Ok(token);
            }
            '>' => {
                let kind = if self.match_char('=') {
                    TokenKind::GreaterEqual
                } else {
                    TokenKind::Greater
                };
                let token = self.make_token(kind);
                return Ok(token);
            }
            '"' => {
                if self.peek() == '"' && self.peek_next() == '"' {
                    self.advance(); // вторая "
                    self.advance(); // третья "
                    self.triple_quoted_string()?
                } else {
                    self.string('"')?
                }
            }
            '\'' => self.string('\'')?,
            c if c.is_ascii_digit() => {
                self.current = start;
                self.number()
            }
            c if c.is_alphabetic() || c == '_' => {
                self.current = start;
                self.identifier()
            }
            _ => {
                return Err(LangError::LexError {
                    message: format!("Unexpected character: {}", c),
                    line: self.line,
                    file: self.source_name.clone(),
                });
            }
        };

        Ok(token)
    }

    fn string(&mut self, delimiter: char) -> Result<Token, LangError> {
        let start_line = self.line;
        let mut value = String::new();

        while self.peek() != delimiter && !self.is_at_end() {
            if self.peek() == '\\' {
                // Обработка escape-последовательностей
                self.advance(); // Пропускаем обратный слэш
                if self.is_at_end() {
                    return Err(LangError::LexError {
                        message: "Unterminated string".to_string(),
                        line: start_line,
                        file: self.source_name.clone(),
                    });
                }
                let escaped = self.advance();
                match escaped {
                    'n' => value.push('\n'),
                    't' => value.push('\t'),
                    'r' => value.push('\r'),
                    '\\' => value.push('\\'),
                    '"' => value.push('"'),
                    '\'' => value.push('\''),
                    '$' => value.push('\u{E000}'), // \$ → placeholder for literal "${" (parser replaces with "${")
                    _ => {
                        // Неизвестная escape-последовательность, оставляем как есть
                        value.push('\\');
                        value.push(escaped);
                    }
                }
            } else {
                if self.peek() == '\n' {
                    self.line += 1;
                }
                value.push(self.advance());
            }
        }

        if self.is_at_end() {
            return Err(LangError::LexError {
                message: "Unterminated string".to_string(),
                line: start_line,
                file: self.source_name.clone(),
            });
        }

        // Пропускаем закрывающую кавычку
        self.advance();

        let lexeme = format!("{}{}{}", delimiter, value, delimiter);
        Ok(Token::new(TokenKind::String, lexeme, start_line))
    }

    /// Содержимое между `"""` и `"""`, с теми же escape-последовательностями, что и в `string()`.
    fn triple_quoted_string(&mut self) -> Result<Token, LangError> {
        let start_line = self.line;
        let mut value = String::new();

        while !self.is_at_end() {
            if self.peek() == '\\' {
                self.advance();
                if self.is_at_end() {
                    return Err(LangError::LexError {
                        message: "Unterminated triple-quoted string".to_string(),
                        line: start_line,
                        file: self.source_name.clone(),
                    });
                }
                let escaped = self.advance();
                match escaped {
                    'n' => value.push('\n'),
                    't' => value.push('\t'),
                    'r' => value.push('\r'),
                    '\\' => value.push('\\'),
                    '"' => value.push('"'),
                    '\'' => value.push('\''),
                    '$' => value.push('\u{E000}'),
                    _ => {
                        value.push('\\');
                        value.push(escaped);
                    }
                }
                continue;
            }

            if self.peek() == '"'
                && self.current + 1 < self.source.len()
                && self.source[self.current + 1] == '"'
                && self.current + 2 < self.source.len()
                && self.source[self.current + 2] == '"'
            {
                self.advance();
                self.advance();
                self.advance();
                let lexeme = format!("\"\"\"{}\"\"\"", value);
                return Ok(Token::new(TokenKind::String, lexeme, start_line));
            }

            if self.peek() == '\n' {
                self.line += 1;
            }
            value.push(self.advance());
        }

        Err(LangError::LexError {
            message: "Unterminated triple-quoted string (missing closing \"\"\")".to_string(),
            line: start_line,
            file: self.source_name.clone(),
        })
    }

    fn number(&mut self) -> Token {
        let start_line = self.line;
        let start = self.current;

        while self.peek().is_ascii_digit() {
            self.advance();
        }

        // Дробная часть
        if self.peek() == '.' && self.peek_next().is_ascii_digit() {
            self.advance(); // Пропускаем точку
            while self.peek().is_ascii_digit() {
                self.advance();
            }
        }

        let lexeme: String = self.source[start..self.current].iter().collect();
        Token::new(TokenKind::Number, lexeme, start_line)
    }

    fn identifier(&mut self) -> Token {
        let start_line = self.line;
        let start = self.current;

        while self.peek().is_alphanumeric() || self.peek() == '_' {
            self.advance();
        }

        let lexeme: String = self.source[start..self.current].iter().collect();
        let kind = self.identifier_type(&lexeme);
        Token::new(kind, lexeme, start_line)
    }

    fn identifier_type(&self, lexeme: &str) -> TokenKind {
        match lexeme {
            "let" => TokenKind::Let,
            "global" => TokenKind::Global,
            "fn" => TokenKind::Fn,
            "stream" => TokenKind::Stream,
            "if" => TokenKind::If,
            "else" => TokenKind::Else,
            "while" => TokenKind::While,
            "for" => TokenKind::For,
            "return" => TokenKind::Return,
            "ereturn" => TokenKind::Ereturn,
            "ireturn" => TokenKind::Ireturn,
            "break" => TokenKind::Break,
            "continue" => TokenKind::Continue,
            "true" => TokenKind::True,
            "false" => TokenKind::False,
            "null" => TokenKind::Null,
            "in" => TokenKind::In,
            "or" => TokenKind::Or,
            "and" => TokenKind::And,
            "try" => TokenKind::Try,
            "catch" => TokenKind::Catch,
            "throw" => TokenKind::Throw,
            "finally" => TokenKind::Finally,
            "cache" => TokenKind::Cache,
            "import" => TokenKind::Import,
            "from" => TokenKind::From,
            "as" => TokenKind::As,
            "Abstract" => TokenKind::Abstract,
            "cls" => TokenKind::Cls,
            "this" => TokenKind::This,
            "super" => TokenKind::Super,
            "private" => TokenKind::Private,
            "protected" => TokenKind::Protected,
            "public" => TokenKind::Public,
            _ => TokenKind::Identifier,
        }
    }

    fn skip_whitespace(&mut self) {
        loop {
            match self.peek() {
                ' ' | '\r' | '\t' => {
                    self.advance();
                }
                '\n' => {
                    self.line += 1;
                    self.advance();
                }
                _ => break,
            }
        }
    }

    fn peek(&self) -> char {
        if self.is_at_end() {
            '\0'
        } else {
            self.source[self.current]
        }
    }

    fn peek_next(&self) -> char {
        if self.current + 1 >= self.source.len() {
            '\0'
        } else {
            self.source[self.current + 1]
        }
    }

    fn match_char(&mut self, expected: char) -> bool {
        if self.is_at_end() || self.source[self.current] != expected {
            false
        } else {
            self.current += 1;
            true
        }
    }

    fn advance(&mut self) -> char {
        let c = self.source[self.current];
        self.current += 1;
        c
    }

    fn is_at_end(&self) -> bool {
        self.current >= self.source.len()
    }

    fn make_token(&mut self, kind: TokenKind) -> Token {
        let lexeme = match kind {
            TokenKind::Eof => "".to_string(),
            TokenKind::EqualEqual => "==".to_string(),
            TokenKind::BangEqual => "!=".to_string(),
            TokenKind::LessEqual => "<=".to_string(),
            TokenKind::GreaterEqual => ">=".to_string(),
            TokenKind::PlusEqual => "+=".to_string(),
            TokenKind::MinusEqual => "-=".to_string(),
            TokenKind::Arrow => "->".to_string(),
            TokenKind::FatArrow => "=>".to_string(),
            TokenKind::StarEqual => "*=".to_string(),
            TokenKind::StarStar => "**".to_string(),
            TokenKind::StarStarEqual => "**=".to_string(),
            TokenKind::SlashEqual => "/=".to_string(),
            TokenKind::SlashSlash => "//".to_string(),
            TokenKind::SlashSlashEqual => "//=".to_string(),
            TokenKind::PercentEqual => "%=".to_string(),
            TokenKind::Ellipsis => "...".to_string(),
            _ => {
                let start = if self.current > 0 {
                    self.current - 1
                } else {
                    0
                };
                self.source[start..self.current].iter().collect()
            }
        };
        Token::new(kind, lexeme, self.line)
    }
}
