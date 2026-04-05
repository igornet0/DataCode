// AST - выражения и инструкции

use crate::common::value::Value;
use crate::lexer::TokenKind;
use serde::{Deserialize, Serialize};

/// Infix operator: built-in (`TokenKind`) or user-registered plugin (`symbol` + logical `name` for VM).
#[derive(Debug, Clone, PartialEq)]
pub enum BinaryOpKind {
    Builtin(TokenKind),
    Plugin { symbol: String, name: String },
}

/// Компонент аннотации типа: имя типа (str, int, …) или строковый литерал ("dev", "prod").
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TypePart {
    /// Имя типа: str, int, null, bool, array, …
    TypeName(String),
    /// Конкретное строковое значение (литеральный тип): "dev", "prod"
    LiteralStr(String),
}

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
    pub type_annotation: Option<Vec<TypePart>>, // Типы параметра: union (TypeName + LiteralStr)
    pub default_value: Option<Expr>, // None для обязательных параметров
}

/// Аргумент при вызове функции - позиционный, именованный или распаковка объекта
#[derive(Debug, Clone)]
pub enum Arg {
    Positional(Expr),           // Позиционный аргумент
    Named { name: String, value: Expr }, // Именованный аргумент
    UnpackObject(Expr),         // **expr — распаковка объекта в kwargs
}

/// Элемент объектного литерала: пара ключ-значение или spread **expr
#[derive(Debug, Clone)]
pub enum ObjectPair {
    KeyValue(String, Expr),
    Spread(Expr),
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
    Modules(Vec<String>),      // import plot
    From {                     // from ... import load_mnist, *
        module: String,
        items: Vec<ImportItem>,
    },
}

/// Поле класса с типом и значением по умолчанию
#[derive(Debug, Clone)]
pub struct ClassField {
    pub name: String,
    pub type_annotation: Option<Vec<TypePart>>, // Типы поля: union (TypeName + LiteralStr)
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
/// visibility: None = private, Some(false) = protected, Some(true) = public
#[derive(Debug, Clone)]
pub struct Method {
    pub name: String,
    pub params: Vec<Param>,
    pub return_type: Option<Vec<TypePart>>, // Тип возвращаемого значения
    pub body: Vec<Stmt>,
    pub line: usize,
    /// None = private, Some(false) = protected, Some(true) = public
    pub visibility: Option<bool>,
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
        op: BinaryOpKind,
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
    /// Вызов значения-функции: `(fn(x) => x)(1)` или `f()(2)` когда callee — выражение.
    CallValue {
        callee: Box<Expr>,
        args: Vec<Arg>,
        line: usize,
    },
    /// Анонимная функция: `fn(x, i) => x + i`
    Lambda {
        params: Vec<Param>,
        return_type: Option<Vec<TypePart>>,
        body: Box<Expr>,
        line: usize,
    },
    ArrayLiteral {
        elements: Vec<Expr>,
        line: usize,
    },
    ObjectLiteral {
        pairs: Vec<ObjectPair>,
        line: usize,
    },
    TupleLiteral {
        elements: Vec<Expr>,
        line: usize,
    },
    /// Скалярный индекс или срез `start:stop:step` внутри `[]`.
    ArrayIndex {
        array: Box<Expr>,
        index: IndexExpr,
        line: usize,
    },
    /// Присваивание в элемент/диапазон массива: `arr[i] = v`, `arr[a:b] = rhs`.
    AssignArray {
        array: Box<Expr>,
        index: IndexExpr,
        value: Box<Expr>,
        line: usize,
    },
    /// Составное присваивание: `arr[i] += v`, `arr[a:b] += rhs` (rhs применяется поэлементно только для скалярного индекса; для среза — ошибка или не поддерживать).
    AssignArrayOp {
        array: Box<Expr>,
        index: IndexExpr,
        op: TokenKind,
        value: Box<Expr>,
        line: usize,
    },
    /// Фильтр таблицы: table["col" op value] → только строки, где col op value
    TableFilter {
        table: Box<Expr>,
        column: String,
        op: TokenKind,
        value: Box<Expr>,
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
    /// Выражение `return expr` в позиции RHS (только в `stream fn`): `x = return 10` — yield-await (`YieldAwaitInput`).
    ExprReturn {
        value: Option<Box<Expr>>,
        line: usize,
    },
    /// `ireturn` / `ireturn expr` — то же lowering, что `return` в RHS (yield-await).
    Ireturn {
        value: Option<Box<Expr>>,
        line: usize,
    },
    /// String interpolation: "Hello ${name}" → segments of literals and expressions
    InterpolatedString {
        segments: Vec<InterpolatedSegment>,
        line: usize,
    },
}

/// Выражение внутри квадратных скобок: один индекс или срез.
#[derive(Debug, Clone)]
pub enum IndexExpr {
    Scalar(Box<Expr>),
    Slice {
        start: Option<Box<Expr>>,
        stop: Option<Box<Expr>>,
        step: Option<Box<Expr>>,
        line: usize,
    },
}

/// Сегмент интерполированной строки: литерал или выражение.
#[derive(Debug, Clone)]
pub enum InterpolatedSegment {
    Literal(String),   // обычный текст (после замены "\\${" → "${" в литералах)
    /// Выражение с опциональным префиксом "name=" и/или форматом (например .2f).
    Expr {
        expr: Box<Expr>,
        /// При true выводить как "name=value" (display_name — источник, например имя переменной).
        include_name: bool,
        display_name: Option<String>,
        /// Спецификация формата числа, например ".2f", ".0f".
        format: Option<String>,
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
            Expr::CallValue { line, .. } => *line,
            Expr::Lambda { line, .. } => *line,
            Expr::ArrayLiteral { line, .. } => *line,
            Expr::ObjectLiteral { line, .. } => *line,
            Expr::TupleLiteral { line, .. } => *line,
            Expr::ArrayIndex { line, .. } => *line,
            Expr::AssignArray { line, .. } => *line,
            Expr::AssignArrayOp { line, .. } => *line,
            Expr::TableFilter { line, .. } => *line,
            Expr::Property { line, .. } => *line,
            Expr::MethodCall { line, .. } => *line,
            Expr::This { line, .. } => *line,
            Expr::Super { line, .. } => *line,
            Expr::SuperCall { line, .. } => *line,
            Expr::SuperMethodCall { line, .. } => *line,
            Expr::Ellipsis { line, .. } => *line,
            Expr::ExprReturn { line, .. } => *line,
            Expr::Ireturn { line, .. } => *line,
            Expr::InterpolatedString { line, .. } => *line,
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
        return_type: Option<Vec<TypePart>>, // Тип возвращаемого значения (union)
        body: Vec<Stmt>,
        is_cached: bool,
        /// Web route: (method, path) e.g. ("GET", "/") from @route("GET", "/")
        route: Option<(String, String)>,
        line: usize,
    },
    /// Генератор: `stream fn name(...) { ... }` — `return` даёт yield, `ereturn` завершает.
    StreamFunction {
        name: String,
        params: Vec<Param>,
        return_type: Option<Vec<TypePart>>,
        body: Vec<Stmt>,
        is_cached: bool,
        route: Option<(String, String)>,
        line: usize,
    },
    Return {
        value: Option<Expr>,
        line: usize,
    },
    /// Завершение генератора без yield (`ereturn` / `ereturn expr`).
    EReturn {
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
        is_abstract: bool,
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
            Stmt::StreamFunction { line, .. } => *line,
            Stmt::Return { line, .. } => *line,
            Stmt::EReturn { line, .. } => *line,
            Stmt::Break { line, .. } => *line,
            Stmt::Continue { line, .. } => *line,
            Stmt::Try { line, .. } => *line,
            Stmt::Throw { line, .. } => *line,
            Stmt::Import { line, .. } => *line,
            Stmt::Class { line, .. } => *line,
        }
    }
}

/// Collects module names from top-level import statements (for dependency graph).
pub fn import_module_names_from_stmts(stmts: &[Stmt]) -> Vec<String> {
    let mut names = Vec::new();
    for stmt in stmts {
        if let Stmt::Import { import_stmt, .. } = stmt {
            match import_stmt {
                ImportStmt::Modules(modules) => names.extend(modules.clone()),
                ImportStmt::From { module, .. } => names.push(module.clone()),
            }
        }
    }
    names
}

