// AST - выражения и инструкции

use crate::common::value::Value;
use crate::lexer::TokenKind;

#[derive(Debug, Clone)]
pub struct CatchBlock {
    pub error_type: Option<String>, // None для catch всех, Some("ValueError") для типизированного
    pub error_var: Option<String>,   // None для catch без переменной
    pub body: Vec<Stmt>,
    pub line: usize,
}

/// Параметр функции с опциональным значением по умолчанию и типом
#[derive(Debug, Clone)]
pub struct Param {
    pub name: String,
    pub type_annotation: Option<Vec<String>>, // Типы параметра: ["int"], ["str", "int"] для union типов
    pub default_value: Option<Expr>, // None для обязательных параметров
}

/// Аргумент при вызове функции - позиционный или именованный
#[derive(Debug, Clone)]
pub enum Arg {
    Positional(Expr),           // Позиционный аргумент
    Named { name: String, value: Expr }, // Именованный аргумент
}

/// Паттерн распаковки для циклов for
#[derive(Debug, Clone)]
pub enum UnpackPattern {
    Variable(String),           // Обычная переменная (x)
    Wildcard,                   // Пропуск значения (_)
    Variadic(String),           // Variadic переменная (*y) - получает остаток элементов
    VariadicWildcard,           // Variadic wildcard (*_) - пропуск остатка
    Nested(Vec<UnpackPattern>), // Вложенная распаковка ((x, y), [x, y])
}

/// Элемент импорта в from-import
#[derive(Debug, Clone)]
pub enum ImportItem {
    Named(String),              // load_mnist
    Aliased { name: String, alias: String }, // window as wd
    All,                       // *
}

/// Тип импорта
#[derive(Debug, Clone)]
pub enum ImportStmt {
    Modules(Vec<String>),      // import ml, plot
    From {                     // from ml import load_mnist, *
        module: String,
        items: Vec<ImportItem>,
    },
}

/// Поле класса с типом и значением по умолчанию
#[derive(Debug, Clone)]
pub struct ClassField {
    pub name: String,
    pub type_annotation: Option<Vec<String>>, // Типы поля: ["int"], ["str", "int"] для union типов
    pub default_value: Option<Expr>, // None для полей без значения по умолчанию
}

/// Переменная уровня класса (присваивание без аннотации типа): name = expression
#[derive(Debug, Clone)]
pub struct ClassVariable {
    pub name: String,
    pub value: Expr,
}

/// Конструктор класса
#[derive(Debug, Clone)]
pub struct Constructor {
    pub params: Vec<Param>,
    pub body: Vec<Stmt>,
    /// Аргументы вызова родителя в синтаксисе `: this(...)`
    pub delegate_args: Option<Vec<Expr>>,
    pub line: usize,
}

/// Метод класса
#[derive(Debug, Clone)]
pub struct Method {
    pub name: String,
    pub params: Vec<Param>,
    pub return_type: Option<Vec<String>>, // Тип возвращаемого значения
    pub body: Vec<Stmt>,
    pub line: usize,
}

#[derive(Debug, Clone)]
pub enum Expr {
    Literal {
        value: Value,
        line: usize,
    },
    Variable {
        name: String,
        line: usize,
    },
    Assign {
        name: String,
        value: Box<Expr>,
        line: usize,
    },
    AssignOp {
        name: String,
        op: TokenKind,
        value: Box<Expr>,
        line: usize,
    },
    UnpackAssign {
        names: Vec<String>,
        value: Box<Expr>,
        line: usize,
    },
    Binary {
        left: Box<Expr>,
        op: TokenKind,
        right: Box<Expr>,
        line: usize,
    },
    Unary {
        op: TokenKind,
        right: Box<Expr>,
        line: usize,
    },
    Call {
        name: String,
        args: Vec<Arg>,
        line: usize,
    },
    ArrayLiteral {
        elements: Vec<Expr>,
        line: usize,
    },
    ObjectLiteral {
        pairs: Vec<(String, Expr)>,
        line: usize,
    },
    TupleLiteral {
        elements: Vec<Expr>,
        line: usize,
    },
    ArrayIndex {
        array: Box<Expr>,
        index: Box<Expr>,
        line: usize,
    },
    Property {
        object: Box<Expr>,
        name: String,
        line: usize,
    },
    MethodCall {
        object: Box<Expr>,
        method: String,
        args: Vec<Arg>,
        line: usize,
    },
    This {
        line: usize,
    },
    /// super keyword - base for SuperCall and SuperMethodCall
    Super {
        line: usize,
    },
    /// super(args) - call to parent constructor
    SuperCall {
        args: Vec<Arg>,
        line: usize,
    },
    /// super.method(args) - call to parent method
    SuperMethodCall {
        method: String,
        args: Vec<Arg>,
        line: usize,
    },
    Ellipsis {
        line: usize,
    },
}

impl Expr {
    pub fn line(&self) -> usize {
        match self {
            Expr::Literal { line, .. } => *line,
            Expr::Variable { line, .. } => *line,
            Expr::Assign { line, .. } => *line,
            Expr::AssignOp { line, .. } => *line,
            Expr::UnpackAssign { line, .. } => *line,
            Expr::Binary { line, .. } => *line,
            Expr::Unary { line, .. } => *line,
            Expr::Call { line, .. } => *line,
            Expr::ArrayLiteral { line, .. } => *line,
            Expr::ObjectLiteral { line, .. } => *line,
            Expr::TupleLiteral { line, .. } => *line,
            Expr::ArrayIndex { line, .. } => *line,
            Expr::Property { line, .. } => *line,
            Expr::MethodCall { line, .. } => *line,
            Expr::This { line, .. } => *line,
            Expr::Super { line, .. } => *line,
            Expr::SuperCall { line, .. } => *line,
            Expr::SuperMethodCall { line, .. } => *line,
            Expr::Ellipsis { line, .. } => *line,
        }
    }
}

#[derive(Debug, Clone)]
pub enum Stmt {
    Let {
        name: String,
        value: Expr,
        is_global: bool,
        line: usize,
    },
    Expr {
        expr: Expr,
        line: usize,
    },
    If {
        condition: Expr,
        then_branch: Vec<Stmt>,
        else_branch: Option<Vec<Stmt>>,
        line: usize,
    },
    While {
        condition: Expr,
        body: Vec<Stmt>,
        line: usize,
    },
    For {
        pattern: Vec<UnpackPattern>, // Паттерн распаковки (может быть один элемент для обратной совместимости)
        iterable: Expr,              // Выражение-итерируемое (array или переменная)
        body: Vec<Stmt>,
        line: usize,
    },
    Function {
        name: String,
        params: Vec<Param>,
        return_type: Option<Vec<String>>, // Тип возвращаемого значения (может быть union типом)
        body: Vec<Stmt>,
        is_cached: bool,
        line: usize,
    },
    Return {
        value: Option<Expr>,
        line: usize,
    },
    Break {
        line: usize,
    },
    Continue {
        line: usize,
    },
    Try {
        try_block: Vec<Stmt>,
        catch_blocks: Vec<CatchBlock>,
        else_block: Option<Vec<Stmt>>,
        finally_block: Option<Vec<Stmt>>,
        line: usize,
    },
    Throw {
        value: Expr,
        line: usize,
    },
    Import {
        import_stmt: ImportStmt,
        line: usize,
    },
    Class {
        name: String,
        superclass: Option<String>,
        private_fields: Vec<ClassField>,
        protected_fields: Vec<ClassField>,
        public_fields: Vec<ClassField>,
        private_variables: Vec<ClassVariable>,
        protected_variables: Vec<ClassVariable>,
        public_variables: Vec<ClassVariable>,
        constructors: Vec<Constructor>,
        methods: Vec<Method>,
        line: usize,
    },
}

impl Stmt {
    pub fn line(&self) -> usize {
        match self {
            Stmt::Let { line, .. } => *line,
            Stmt::Expr { line, .. } => *line,
            Stmt::If { line, .. } => *line,
            Stmt::While { line, .. } => *line,
            Stmt::For { line, .. } => *line,
            Stmt::Function { line, .. } => *line,
            Stmt::Return { line, .. } => *line,
            Stmt::Break { line, .. } => *line,
            Stmt::Continue { line, .. } => *line,
            Stmt::Try { line, .. } => *line,
            Stmt::Throw { line, .. } => *line,
            Stmt::Import { line, .. } => *line,
            Stmt::Class { line, .. } => *line,
        }
    }
}

