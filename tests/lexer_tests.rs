// Тесты для лексера
#[cfg(test)]
mod tests {
    use data_code::lexer::{Lexer, TokenKind};

    #[test]
    fn test_lexer_creation() {
        let _lexer = Lexer::new("test");
        // Просто проверяем, что лексер создается
    }

    fn tokenize(source: &str) -> Vec<TokenKind> {
        let mut lexer = Lexer::new(source);
        let mut tokens = Vec::new();
        loop {
            let token = lexer.next_token().unwrap();
            match token.kind {
                TokenKind::Eof => break,
                kind => tokens.push(kind),
            }
        }
        tokens
    }

    #[test]
    fn test_basic_tokens() {
        let source = "let x = 10";
        let tokens = tokenize(source);
        assert_eq!(tokens, vec![
            TokenKind::Let,
            TokenKind::Identifier,
            TokenKind::Equal,
            TokenKind::Number,
        ]);
    }

    #[test]
    fn test_operators() {
        let source = "+ - * / == != < >";
        let tokens = tokenize(source);
        assert_eq!(tokens, vec![
            TokenKind::Plus,
            TokenKind::Minus,
            TokenKind::Star,
            TokenKind::Slash,
            TokenKind::EqualEqual,
            TokenKind::BangEqual,
            TokenKind::Less,
            TokenKind::Greater,
        ]);
    }

    #[test]
    fn test_function_declaration() {
        let source = "fn sum(a, b) { return a + b }";
        let tokens = tokenize(source);
        assert_eq!(tokens, vec![
            TokenKind::Fn,
            TokenKind::Identifier,
            TokenKind::LParen,
            TokenKind::Identifier,
            TokenKind::Comma,
            TokenKind::Identifier,
            TokenKind::RParen,
            TokenKind::LBrace,
            TokenKind::Return,
            TokenKind::Identifier,
            TokenKind::Plus,
            TokenKind::Identifier,
            TokenKind::RBrace,
        ]);
    }

    #[test]
    fn test_string_literal() {
        let source = r#"let s = "hello world""#;
        let tokens = tokenize(source);
        assert_eq!(tokens, vec![
            TokenKind::Let,
            TokenKind::Identifier,
            TokenKind::Equal,
            TokenKind::String,
        ]);
    }

    #[test]
    fn test_numbers() {
        let source = "123 456.789 0 42";
        let tokens = tokenize(source);
        assert_eq!(tokens.len(), 4);
        assert!(tokens.iter().all(|t| matches!(t, TokenKind::Number)));
    }

    #[test]
    fn test_comments() {
        let source = "let x = 10 # это комментарий\nlet y = 20";
        let tokens = tokenize(source);
        assert_eq!(tokens, vec![
            TokenKind::Let,
            TokenKind::Identifier,
            TokenKind::Equal,
            TokenKind::Number,
            TokenKind::Let,
            TokenKind::Identifier,
            TokenKind::Equal,
            TokenKind::Number,
        ]);
    }

    #[test]
    fn test_multiline_comment_basic() {
        let source = r#"let x = 10
"""
This is a multiline comment
It can span multiple lines
"""
let y = 20"#;
        let tokens = tokenize(source);
        assert_eq!(tokens, vec![
            TokenKind::Let,
            TokenKind::Identifier,
            TokenKind::Equal,
            TokenKind::Number,
            TokenKind::Let,
            TokenKind::Identifier,
            TokenKind::Equal,
            TokenKind::Number,
        ]);
    }

    #[test]
    fn test_multiline_comment_in_function() {
        let source = r#"fn test() {
    """
    Function documentation
    """
    return 42
}"#;
        let tokens = tokenize(source);
        assert_eq!(tokens, vec![
            TokenKind::Fn,
            TokenKind::Identifier,
            TokenKind::LParen,
            TokenKind::RParen,
            TokenKind::LBrace,
            TokenKind::Return,
            TokenKind::Number,
            TokenKind::RBrace,
        ]);
    }

    #[test]
    fn test_multiline_comment_with_quotes_inside() {
        let source = r#"let x = 10
"""
This comment has "quotes" inside
And 'single quotes' too
"""
let y = 20"#;
        let tokens = tokenize(source);
        assert_eq!(tokens, vec![
            TokenKind::Let,
            TokenKind::Identifier,
            TokenKind::Equal,
            TokenKind::Number,
            TokenKind::Let,
            TokenKind::Identifier,
            TokenKind::Equal,
            TokenKind::Number,
        ]);
    }

    #[test]
    fn test_unterminated_multiline_comment() {
        let source = r#"let x = 10
"""
This comment is not closed
let y = 20"#;
        let mut lexer = Lexer::new(source);
        let result = lexer.tokenize();
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(format!("{:?}", err).contains("Unterminated multiline comment"));
    }

    #[test]
    fn test_string_vs_multiline_comment() {
        // Обычная строка не должна интерпретироваться как комментарий
        let source = r#"let s = "hello""#;
        let tokens = tokenize(source);
        assert_eq!(tokens, vec![
            TokenKind::Let,
            TokenKind::Identifier,
            TokenKind::Equal,
            TokenKind::String,
        ]);
    }
}

