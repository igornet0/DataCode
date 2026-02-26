// Единый формат ошибок компиляции и рантайма

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ErrorType {
    // RuntimeError и его подтипы
    RuntimeError,
    ProtectError,
    ValueError,
    TypeError,
    IndexError,
    KeyError,
    StateError,
    OverflowError,
    // IOError и его подтипы
    IOError,
    FileNotFoundError,
    PermissionError,
    DirectoryError,
    ReadOnlyError,
    // ParseError и его подтипы
    ParseError,
    SyntaxError,
    TokenError,
    // DataError и его подтипы
    DataError,
    SchemaError,
    ColumnNotFoundError,
    DataFormatError,
}

impl ErrorType {
    pub fn name(&self) -> &'static str {
        match self {
            ErrorType::RuntimeError => "RuntimeError",
            ErrorType::ProtectError => "ProtectError",
            ErrorType::ValueError => "ValueError",
            ErrorType::TypeError => "TypeError",
            ErrorType::IndexError => "IndexError",
            ErrorType::KeyError => "KeyError",
            ErrorType::StateError => "StateError",
            ErrorType::OverflowError => "OverflowError",
            ErrorType::IOError => "IOError",
            ErrorType::FileNotFoundError => "FileNotFoundError",
            ErrorType::PermissionError => "PermissionError",
            ErrorType::DirectoryError => "DirectoryError",
            ErrorType::ReadOnlyError => "ReadOnlyError",
            ErrorType::ParseError => "ParseError",
            ErrorType::SyntaxError => "SyntaxError",
            ErrorType::TokenError => "TokenError",
            ErrorType::DataError => "DataError",
            ErrorType::SchemaError => "SchemaError",
            ErrorType::ColumnNotFoundError => "ColumnNotFoundError",
            ErrorType::DataFormatError => "DataFormatError",
        }
    }

    /// Проверяет, является ли данный тип ошибки или его предком указанный тип
    pub fn is_instance_of(&self, other: &ErrorType) -> bool {
        if self == other {
            return true;
        }
        
        // Проверка иерархии
        match (self, other) {
            // ValueError является RuntimeError
            (ErrorType::ValueError, ErrorType::RuntimeError) => true,
            // TypeError является RuntimeError
            (ErrorType::TypeError, ErrorType::RuntimeError) => true,
            // IndexError является RuntimeError
            (ErrorType::IndexError, ErrorType::RuntimeError) => true,
            // KeyError является RuntimeError
            (ErrorType::KeyError, ErrorType::RuntimeError) => true,
            // StateError является RuntimeError
            (ErrorType::StateError, ErrorType::RuntimeError) => true,
            // OverflowError является RuntimeError
            (ErrorType::OverflowError, ErrorType::RuntimeError) => true,
            // ProtectError является RuntimeError
            (ErrorType::ProtectError, ErrorType::RuntimeError) => true,

            // FileNotFoundError является IOError
            (ErrorType::FileNotFoundError, ErrorType::IOError) => true,
            // PermissionError является IOError
            (ErrorType::PermissionError, ErrorType::IOError) => true,
            // DirectoryError является IOError
            (ErrorType::DirectoryError, ErrorType::IOError) => true,
            // ReadOnlyError является IOError
            (ErrorType::ReadOnlyError, ErrorType::IOError) => true,
            
            // SyntaxError является ParseError
            (ErrorType::SyntaxError, ErrorType::ParseError) => true,
            // TokenError является ParseError
            (ErrorType::TokenError, ErrorType::ParseError) => true,
            
            // SchemaError является DataError
            (ErrorType::SchemaError, ErrorType::DataError) => true,
            // ColumnNotFoundError является DataError
            (ErrorType::ColumnNotFoundError, ErrorType::DataError) => true,
            // DataFormatError является DataError
            (ErrorType::DataFormatError, ErrorType::DataError) => true,
            
            _ => false,
        }
    }

    /// Парсит имя типа ошибки из строки
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "RuntimeError" => Some(ErrorType::RuntimeError),
            "ProtectError" => Some(ErrorType::ProtectError),
            "ValueError" => Some(ErrorType::ValueError),
            "TypeError" => Some(ErrorType::TypeError),
            "IndexError" => Some(ErrorType::IndexError),
            "KeyError" => Some(ErrorType::KeyError),
            "StateError" => Some(ErrorType::StateError),
            "OverflowError" => Some(ErrorType::OverflowError),
            "IOError" => Some(ErrorType::IOError),
            "FileNotFoundError" => Some(ErrorType::FileNotFoundError),
            "PermissionError" => Some(ErrorType::PermissionError),
            "DirectoryError" => Some(ErrorType::DirectoryError),
            "ReadOnlyError" => Some(ErrorType::ReadOnlyError),
            "ParseError" => Some(ErrorType::ParseError),
            "SyntaxError" => Some(ErrorType::SyntaxError),
            "TokenError" => Some(ErrorType::TokenError),
            "DataError" => Some(ErrorType::DataError),
            "SchemaError" => Some(ErrorType::SchemaError),
            "ColumnNotFoundError" => Some(ErrorType::ColumnNotFoundError),
            "DataFormatError" => Some(ErrorType::DataFormatError),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct StackTraceEntry {
    pub function_name: String,
    pub line: usize,
    pub file: Option<String>,
}

#[derive(Debug, Clone)]
pub enum LangError {
    LexError { message: String, line: usize, file: Option<String> },
    ParseError { message: String, line: usize, file: Option<String> },
    SemanticError { message: String, line: usize, file: Option<String> },
    RuntimeError { 
        message: String, 
        line: usize,
        file: Option<String>,
        stack_trace: Vec<StackTraceEntry>,
        error_type: Option<ErrorType>,
        source: Option<Box<LangError>>,
    },
}

impl LangError {
    pub fn runtime_error(message: String, line: usize) -> Self {
        LangError::runtime_error_with_file(message, line, None)
    }

    pub fn runtime_error_with_file(message: String, line: usize, file: Option<&str>) -> Self {
        LangError::RuntimeError {
            message,
            line,
            file: file.map(String::from),
            stack_trace: Vec::new(),
            error_type: None,
            source: None,
        }
    }

    pub fn runtime_error_with_trace(message: String, line: usize, stack_trace: Vec<StackTraceEntry>) -> Self {
        LangError::runtime_error_with_trace_and_file(message, line, None, stack_trace)
    }

    pub fn runtime_error_with_trace_and_file(
        message: String,
        line: usize,
        file: Option<&str>,
        stack_trace: Vec<StackTraceEntry>,
    ) -> Self {
        LangError::RuntimeError {
            message,
            line,
            file: file.map(String::from),
            stack_trace,
            error_type: None,
            source: None,
        }
    }

    pub fn runtime_error_with_type(message: String, line: usize, error_type: ErrorType) -> Self {
        LangError::runtime_error_with_type_and_file(message, line, error_type, None)
    }

    pub fn runtime_error_with_type_and_file(
        message: String,
        line: usize,
        error_type: ErrorType,
        file: Option<&str>,
    ) -> Self {
        LangError::RuntimeError {
            message,
            line,
            file: file.map(String::from),
            stack_trace: Vec::new(),
            error_type: Some(error_type),
            source: None,
        }
    }

    pub fn runtime_error_with_type_and_trace(
        message: String,
        line: usize,
        error_type: ErrorType,
        stack_trace: Vec<StackTraceEntry>,
    ) -> Self {
        LangError::runtime_error_with_type_trace_and_file(message, line, error_type, None, stack_trace)
    }

    pub fn runtime_error_with_type_trace_and_file(
        message: String,
        line: usize,
        error_type: ErrorType,
        file: Option<&str>,
        stack_trace: Vec<StackTraceEntry>,
    ) -> Self {
        LangError::RuntimeError {
            message,
            line,
            file: file.map(String::from),
            stack_trace,
            error_type: Some(error_type),
            source: None,
        }
    }

    /// Получить тип ошибки (если есть)
    pub fn error_type(&self) -> Option<&ErrorType> {
        match self {
            LangError::RuntimeError { error_type, .. } => error_type.as_ref(),
            _ => None,
        }
    }

    /// Проверяет, является ли ошибка указанного типа или его подтипом
    pub fn is_instance_of(&self, error_type: &ErrorType) -> bool {
        match self {
            LangError::RuntimeError { error_type: Some(et), .. } => et.is_instance_of(error_type),
            LangError::RuntimeError { error_type: None, .. } => {
                // Если тип не указан, считаем RuntimeError
                error_type == &ErrorType::RuntimeError
            }
            _ => false,
        }
    }

    /// Локация корневой причины (рекурсивно по цепочке source).
    pub fn root_location(err: &LangError) -> (Option<String>, usize) {
        match err {
            LangError::RuntimeError { source: Some(inner), .. } => Self::root_location(inner),
            LangError::LexError { file, line, .. }
            | LangError::ParseError { file, line, .. }
            | LangError::SemanticError { file, line, .. }
            | LangError::RuntimeError { file, line, .. } => (file.clone(), *line),
        }
    }

    /// Ошибка-обёртка с цепочкой причин; file/line берутся из корневой причины (source).
    pub fn runtime_error_with_source(message: impl Into<String>, source: LangError) -> Self {
        let (file, line) = Self::root_location(&source);
        LangError::RuntimeError {
            message: message.into(),
            line,
            file,
            stack_trace: Vec::new(),
            error_type: None,
            source: Some(Box::new(source)),
        }
    }

    /// То же, но с заданным stack_trace (для ExceptionHandler).
    pub fn runtime_error_with_source_and_trace(
        message: impl Into<String>,
        source: LangError,
        stack_trace: Vec<StackTraceEntry>,
    ) -> Self {
        let (file, line) = Self::root_location(&source);
        LangError::RuntimeError {
            message: message.into(),
            line,
            file,
            stack_trace,
            error_type: None,
            source: Some(Box::new(source)),
        }
    }
}

/// Error codes in Rust style (E01xx = lex, E02xx = parse, E03xx = semantic, E04xx = runtime).
fn error_code_and_label(err: &LangError) -> (&'static str, &'static str) {
    match err {
        LangError::LexError { .. } => ("E0100", "Lexer Error"),
        LangError::ParseError { .. } => ("E0200", "Parse Error"),
        LangError::SemanticError { .. } => ("E0300", "Semantic Error"),
        LangError::RuntimeError { error_type, .. } => {
            if let Some(et) = error_type {
                ("E0400", et.name())
            } else {
                ("E0400", "Runtime Error")
            }
        }
    }
}

fn fmt_location(f: &mut std::fmt::Formatter, file: &Option<String>, line: usize) -> std::fmt::Result {
    // Rust-style: "   --> path:line" or "   --> line N"
    write!(f, "   --> ")?;
    if let Some(path) = file {
        write!(f, "{}:{}", path, line)
    } else {
        write!(f, "line {}", line)
    }
}

fn fmt_cause_chain(f: &mut std::fmt::Formatter, err: &LangError) -> std::fmt::Result {
    let (code, _) = error_code_and_label(err);
    let (message, line, file) = match err {
        LangError::LexError { message, line, file } => (message.as_str(), *line, file),
        LangError::ParseError { message, line, file } => (message.as_str(), *line, file),
        LangError::SemanticError { message, line, file } => (message.as_str(), *line, file),
        LangError::RuntimeError { message, line, file, .. } => (message.as_str(), *line, file),
    };
    write!(f, "  error[{}]: {} (at ", code, message)?;
    if let Some(path) = file {
        write!(f, "{}:{})", path, line)?;
    } else {
        write!(f, "line {})", line)?;
    }
    writeln!(f)?;
    if let LangError::RuntimeError { source: Some(inner), .. } = err {
        fmt_cause_chain(f, inner)?;
    }
    Ok(())
}

impl std::fmt::Display for LangError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let (code, _label) = error_code_and_label(self);
        let (message, line, file) = match self {
            LangError::LexError { message, line, file } => (message, *line, file),
            LangError::ParseError { message, line, file } => (message, *line, file),
            LangError::SemanticError { message, line, file } => (message, *line, file),
            LangError::RuntimeError { message, line, file, .. } => (message, *line, file),
        };
        // First line: error[CODE]: message
        writeln!(f, "error[{}]: {}", code, message)?;
        // Second line:    --> file:line
        fmt_location(f, file, line)?;
        writeln!(f)?;
        // Empty bar like Rust when there's no source snippet
        writeln!(f, "    |")?;
        if let LangError::RuntimeError { stack_trace, .. } = self {
            if !stack_trace.is_empty() {
                writeln!(f)?;
                writeln!(f, "stack trace:")?;
                for entry in stack_trace {
                    write!(f, "   --> ")?;
                    if let Some(path) = &entry.file {
                        write!(f, "{}:{}", path, entry.line)?;
                    } else {
                        write!(f, "line {}", entry.line)?;
                    }
                    writeln!(f, " in {}", entry.function_name)?;
                }
            }
        }
        if let LangError::RuntimeError { source: Some(inner), .. } = self {
            writeln!(f)?;
            writeln!(f, "Caused by:")?;
            fmt_cause_chain(f, inner)?;
        }
        Ok(())
    }
}

impl std::error::Error for LangError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            LangError::RuntimeError { source: Some(inner), .. } => Some(inner.as_ref()),
            _ => None,
        }
    }
}

