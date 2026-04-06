// Позиции в исходном коде (для сообщений об ошибках)

#[derive(Debug, Clone, Copy)]
pub struct Span {
    pub line: usize,
    pub column: usize,
}

impl Span {
    pub fn new(line: usize, column: usize) -> Self {
        Self { line, column }
    }

    pub fn start() -> Self {
        Self { line: 1, column: 1 }
    }
}
