#[test]
fn dump() {
    use data_code::bytecode::OpCode;
    for (label, src) in [
        ("bare", "from core.config import get_settings\nget_settings()"),
        ("assign", "from core.config import get_settings\ns = get_settings()\n1"),
    ] {
        let mut lexer = data_code::lexer::Lexer::new(src);
        let tokens = lexer.tokenize().unwrap();
        // private preload - use Parser without registry if possible
        eprintln!("=== {} === tokens={}", label, tokens.len());
    }
}
