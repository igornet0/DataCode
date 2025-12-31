// Публичный API языка DataCode (новая архитектура Bytecode + VM)

pub mod common;
pub mod lexer;
pub mod parser;
pub mod semantic;
pub mod bytecode;
pub mod compiler;
pub mod vm;

#[path = "lib/websocket/mod.rs"]
pub mod websocket;
#[path = "lib/sqlite_export/mod.rs"]
pub mod sqlite_export;
#[path = "lib/ml/mod.rs"]
pub mod ml;
#[path = "lib/plot/mod.rs"]
pub mod plot;

// Публичный API для запуска интерпретатора
pub use common::{error::LangError, value::Value};
pub use bytecode::Chunk;
pub use vm::Vm;

pub fn run(source: &str) -> Result<Value, LangError> {
    use lexer::Lexer;
    use parser::Parser;
    use semantic::resolver::Resolver;
    use compiler::Compiler;
    use vm::Vm;

    // 1. Лексический анализ
    let mut lexer = Lexer::new(source);
    let tokens = lexer.tokenize()?;

    // 2. Парсинг
    let mut parser = Parser::new(tokens);
    let ast = parser.parse()?;

    // 3. Семантический анализ
    let mut resolver = Resolver::new();
    resolver.resolve(&ast)?;

    // 4. Компиляция в байт-код
    let mut compiler = Compiler::new();
    let chunk = compiler.compile(&ast)?;
    let functions = compiler.get_functions();

    // 5. Выполнение на VM
    let mut vm = Vm::new();
    vm.set_functions(functions);
    vm.register_native_globals();
    let result = vm.run(&chunk)?;

    Ok(result)
}

/// Выполняет код и возвращает VM для доступа к глобальным переменным
pub fn run_with_vm(source: &str) -> Result<(Value, Vm), LangError> {
    use lexer::Lexer;
    use parser::Parser;
    use semantic::resolver::Resolver;
    use compiler::Compiler;
    use vm::Vm;

    // 1. Лексический анализ
    let mut lexer = Lexer::new(source);
    let tokens = lexer.tokenize()?;

    // 2. Парсинг
    let mut parser = Parser::new(tokens);
    let ast = parser.parse()?;

    // 3. Семантический анализ
    let mut resolver = Resolver::new();
    resolver.resolve(&ast)?;

    // 4. Компиляция в байт-код
    let mut compiler = Compiler::new();
    let chunk = compiler.compile(&ast)?;
    let functions = compiler.get_functions();

    // 5. Выполнение на VM
    let mut vm = Vm::new();
    vm.set_functions(functions);
    vm.register_native_globals();
    let result = vm.run(&chunk)?;

    Ok((result, vm))
}

/// Компилирует код в байт-код без выполнения (для отладки)
pub fn compile(source: &str) -> Result<(Chunk, Vec<bytecode::Function>), LangError> {
    use lexer::Lexer;
    use parser::Parser;
    use semantic::resolver::Resolver;
    use compiler::Compiler;

    // 1. Лексический анализ
    let mut lexer = Lexer::new(source);
    let tokens = lexer.tokenize()?;

    // 2. Парсинг
    let mut parser = Parser::new(tokens);
    let ast = parser.parse()?;

    // 3. Семантический анализ
    let mut resolver = Resolver::new();
    resolver.resolve(&ast)?;

    // 4. Компиляция в байт-код
    let mut compiler = Compiler::new();
    let chunk = compiler.compile(&ast)?;
    let functions = compiler.get_functions();

    Ok((chunk, functions))
}

/// Выполняет код с включенным debug mode (выводит байт-код)
pub fn run_debug(source: &str) -> Result<Value, LangError> {
    use lexer::Lexer;
    use parser::Parser;
    use semantic::resolver::Resolver;
    use compiler::Compiler;
    use vm::Vm;

    // 1. Лексический анализ
    let mut lexer = Lexer::new(source);
    let tokens = lexer.tokenize()?;

    // 2. Парсинг
    let mut parser = Parser::new(tokens);
    let ast = parser.parse()?;

    // 3. Семантический анализ
    let mut resolver = Resolver::new();
    resolver.resolve(&ast)?;

    // 4. Компиляция в байт-код
    let mut compiler = Compiler::new();
    let chunk = compiler.compile(&ast)?;
    let functions = compiler.get_functions();

    // Debug: выводим байт-код
    println!("{}", chunk.disassemble("<main>"));
    for function in &functions {
        println!("{}", function.chunk.disassemble(&function.name));
    }

    // 5. Выполнение на VM
    let mut vm = Vm::new();
    vm.set_functions(functions);
    vm.register_native_globals();
    let result = vm.run(&chunk)?;

    Ok(result)
}

