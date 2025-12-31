// Единый формат ошибок компиляции и рантайма

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ErrorType {
    // RuntimeError и его подтипы
    RuntimeError,
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
}

#[derive(Debug, Clone)]
pub enum LangError {
    LexError { message: String, line: usize },
    ParseError { message: String, line: usize },
    SemanticError { message: String, line: usize },
    RuntimeError { 
        message: String, 
        line: usize,
        stack_trace: Vec<StackTraceEntry>,
        error_type: Option<ErrorType>,
    },
}

impl LangError {
    pub fn runtime_error(message: String, line: usize) -> Self {
        LangError::RuntimeError {
            message,
            line,
            stack_trace: Vec::new(),
            error_type: None,
        }
    }

    pub fn runtime_error_with_trace(message: String, line: usize, stack_trace: Vec<StackTraceEntry>) -> Self {
        LangError::RuntimeError {
            message,
            line,
            stack_trace,
            error_type: None,
        }
    }

    pub fn runtime_error_with_type(message: String, line: usize, error_type: ErrorType) -> Self {
        LangError::RuntimeError {
            message,
            line,
            stack_trace: Vec::new(),
            error_type: Some(error_type),
        }
    }

    pub fn runtime_error_with_type_and_trace(
        message: String,
        line: usize,
        error_type: ErrorType,
        stack_trace: Vec<StackTraceEntry>,
    ) -> Self {
        LangError::RuntimeError {
            message,
            line,
            stack_trace,
            error_type: Some(error_type),
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
}

impl std::fmt::Display for LangError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            LangError::LexError { message, line } => {
                write!(f, "[Lexer Error] Line {}: {}", line, message)
            }
            LangError::ParseError { message, line } => {
                write!(f, "[Parse Error] Line {}: {}", line, message)
            }
            LangError::SemanticError { message, line } => {
                write!(f, "[Semantic Error] Line {}: {}", line, message)
            }
            LangError::RuntimeError { message, line, stack_trace: _, error_type } => {
                if let Some(et) = error_type {
                    write!(f, "[{}] Line {}: {}", et.name(), line, message)?;
                } else {
                    write!(f, "[Runtime Error] Line {}: {}", line, message)?;
                }
                Ok(())
            }
        }
    }
}

impl std::error::Error for LangError {}

