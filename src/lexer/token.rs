// Токены для лексера

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TokenKind {
    // Ключевые слова
    Let,
    Global,
    Fn,
    If,
    Else,
    While,
    For,
    Return,
    Break,
    Continue,
    True,
    False,
    Null,
    Import,
    From,
    As,

    // Литералы
    Identifier,
    Number,
    String,

    // Операторы
    Plus,    // +
    Minus,   // -
    Star,    // *
    StarStar, // **
    Slash,   // /
    SlashSlash, // // (целочисленное деление)
    Percent, // %
    Equal,   // =
    EqualEqual, // ==
    PlusEqual,   // +=
    MinusEqual,  // -=
    StarEqual,   // *=
    StarStarEqual, // **=
    SlashEqual,  // /=
    SlashSlashEqual, // //=
    PercentEqual, // %=
    Bang,    // !
    BangEqual, // !=
    Less,    // <
    Greater, // >
    LessEqual,  // <=
    GreaterEqual, // >=
    Or,         // or
    And,        // and

    // Разделители
    LParen,   // (
    RParen,   // )
    LBrace,   // {
    RBrace,   // }
    LBracket, // [
    RBracket, // ]
    Comma,    // ,
    Semicolon, // ;
    Dot,      // .

    // Ключевые слова для циклов
    In,       // in

    // Ключевые слова для обработки исключений
    Try,      // try
    Catch,    // catch
    Throw,    // throw 

    // Аннотации
    At,       // @
    Cache,    // cache

    // Конец файла
    Eof,
}

#[derive(Debug, Clone)]
pub struct Token {
    pub kind: TokenKind,
    pub lexeme: String,
    pub line: usize,
}

impl Token {
    pub fn new(kind: TokenKind, lexeme: String, line: usize) -> Self {
        Self { kind, lexeme, line }
    }
}

