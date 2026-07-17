// Токены для лексера

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TokenKind {
    // Ключевые слова
    Let,
    Global,
    Fn,
    /// `stream` — модификатор перед `fn` для генераторов (`stream fn`).
    Stream,
    If,
    Else,
    While,
    For,
    Return,
    /// `ereturn` — досрочное завершение stream fn (не yield).
    Ereturn,
    /// `ireturn` / `ireturn expr` — yield через `.next()` (только в `stream fn`).
    Ireturn,
    Break,
    Continue,
    True,
    False,
    Null,
    /// Литерал IEEE +∞ (`inf`); `int(inf)` / `float(inf)` → доменные ±∞.
    Inf,
    /// Литерал IEEE NaN (`nan`); `float(nan)` → [`FloatValue::NaN`].
    Nan,
    Import,
    From,
    As,

    // Литералы
    Identifier,
    Number,
    String,

    // Операторы
    Plus,            // +
    Minus,           // -
    Star,            // *
    StarStar,        // **
    Slash,           // /
    SlashSlash,      // // (целочисленное деление)
    Percent,         // %
    Equal,           // =
    EqualEqual,      // ==
    PlusEqual,       // +=
    MinusEqual,      // -=
    StarEqual,       // *=
    StarStarEqual,   // **=
    SlashEqual,      // /=
    SlashSlashEqual, // //=
    PercentEqual,    // %=
    Bang,            // !
    BangEqual,       // !=
    Less,            // <
    Greater,         // >
    LessEqual,       // <=
    GreaterEqual,    // >=
    LessLess,        // <<
    GreaterGreater,  // >>
    Amp,             // &
    Caret,           // ^
    Tilde,           // ~
    Or,              // or
    And,             // and

    // Разделители
    LParen,    // (
    RParen,    // )
    LBrace,    // {
    RBrace,    // }
    LBracket,  // [
    RBracket,  // ]
    Comma,     // ,
    Semicolon, // ;
    Dot,       // .
    Ellipsis,  // ...
    Colon,     // :
    Question,  // ? (C-style ternary; future: peek for ?. / ??)
    Arrow,     // ->
    FatArrow,  // =>
    Pipe,      // |

    // Ключевые слова для циклов
    In, // in

    // Ключевые слова для обработки исключений
    Try,     // try
    Catch,   // catch
    Throw,   // throw
    Finally, // finally

    // Аннотации
    At, // @

    // Ключевые слова для классов
    Abstract,  // @Abstract for abstract class
    Cls,       // cls
    This,      // this
    Super,     // super
    Private,   // private
    Protected, // protected
    Public,    // public

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
