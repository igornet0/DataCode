// Tests for single-line for/while: `for x in arr: stmt`, `while cond: stmt`

#[cfg(test)]
mod tests {
    use data_code::lexer::Lexer;
    use data_code::parser::{Parser, Stmt};
    use data_code::{run, Value};

    fn parse(source: &str) -> Vec<Stmt> {
        let mut lexer = Lexer::new(source);
        let tokens = lexer.tokenize().unwrap();
        let mut parser = Parser::new(tokens);
        parser.parse().unwrap()
    }

    fn assert_parse_error(source: &str) {
        let mut lexer = Lexer::new(source);
        let tokens = lexer.tokenize().unwrap();
        let mut parser = Parser::new(tokens);
        assert!(parser.parse().is_err(), "Expected parse error for: {}", source);
    }

    fn assert_number(source: &str, expected: f64) {
        match run(source) {
            Ok(Value::Number(n)) => assert!((n - expected).abs() < 1e-10, "got {}", n),
            Ok(v) => panic!("Expected Number({}), got {:?}", expected, v),
            Err(e) => panic!("Error: {:?}", e),
        }
    }

    fn assert_bool(source: &str, expected: bool) {
        match run(source) {
            Ok(Value::Bool(b)) => assert_eq!(b, expected),
            Ok(v) => panic!("Expected Bool({}), got {:?}", expected, v),
            Err(e) => panic!("Error: {:?}", e),
        }
    }

    // ========== Parser ==========

    #[test]
    fn test_parse_single_line_for() {
        let stmts = parse("for x in arr: print(x)");
        assert_eq!(stmts.len(), 1);
        let Stmt::For { body, .. } = &stmts[0] else {
            panic!("Expected For");
        };
        assert_eq!(body.len(), 1);
        assert!(matches!(body[0], Stmt::Expr { .. }));
    }

    #[test]
    fn test_parse_single_line_while() {
        let stmts = parse("while x > 0: x -= 1");
        assert_eq!(stmts.len(), 1);
        let Stmt::While { body, .. } = &stmts[0] else {
            panic!("Expected While");
        };
        assert_eq!(body.len(), 1);
    }

    #[test]
    fn test_parse_block_for_still_works() {
        let stmts = parse("for x in arr { print(x) }");
        assert!(matches!(stmts[0], Stmt::For { .. }));
    }

    #[test]
    fn test_parse_block_while_still_works() {
        let stmts = parse("while x > 0 { x -= 1 }");
        assert!(matches!(stmts[0], Stmt::While { .. }));
    }

    #[test]
    fn test_parse_error_for_missing_statement_after_colon() {
        assert_parse_error("for x in arr:");
    }

    #[test]
    fn test_parse_error_while_missing_condition() {
        assert_parse_error("while : print(1)");
    }

    #[test]
    fn test_parse_error_for_missing_in() {
        assert_parse_error("for in arr: print(1)");
    }

    #[test]
    fn test_parse_error_while_missing_colon_and_brace() {
        assert_parse_error("while x > 0 print(x)");
    }

    // ========== Runtime ==========

    #[test]
    fn test_single_line_for_iterates() {
        let source = r#"
            out = []
            for x in [1, 2, 3]: out = push(out, x)
            out
        "#;
        match run(source) {
            Ok(Value::Array(a)) => {
                let v = a.borrow();
                assert_eq!(v.len(), 3);
                assert_eq!(v[0], Value::Number(1.0));
                assert_eq!(v[1], Value::Number(2.0));
                assert_eq!(v[2], Value::Number(3.0));
            }
            Ok(v) => panic!("Expected Array, got {:?}", v),
            Err(e) => panic!("Error: {:?}", e),
        }
    }

    #[test]
    fn test_single_line_for_plus_equals() {
        let source = r#"
            sum = 0
            for i in range(5): sum += i
            sum
        "#;
        assert_number(source, 10.0);
    }

    #[test]
    fn test_single_line_while_minus_equals() {
        let source = r#"
            x = 5
            while x > 0: x -= 1
            x
        "#;
        assert_number(source, 0.0);
    }

    #[test]
    fn test_block_for_with_single_line_if_continue() {
        let source = r#"
            out = []
            for x in [1, 2, 3] {
                if x == 2: continue
                out = push(out, x)
            }
            out
        "#;
        match run(source) {
            Ok(Value::Array(a)) => {
                let v = a.borrow();
                assert_eq!(v.len(), 2);
                assert_eq!(v[0], Value::Number(1.0));
                assert_eq!(v[1], Value::Number(3.0));
            }
            Ok(v) => panic!("Expected Array, got {:?}", v),
            Err(e) => panic!("Error: {:?}", e),
        }
    }

    #[test]
    fn test_single_line_while_break() {
        let source = r#"
            done = false
            while true: break
            done = true
            done
        "#;
        match run(source) {
            Ok(Value::Bool(true)) => {}
            Ok(v) => panic!("Expected Bool(true), got {:?}", v),
            Err(e) => panic!("Error: {:?}", e),
        }
    }

    #[test]
    fn test_single_line_for_range_many_iterations() {
        let source = r#"
            blocked_cells = set()
            for _ in range(10000): blocked_cells.add(123)
            len(blocked_cells)
        "#;
        assert_number(source, 1.0);
    }

    #[test]
    fn test_single_line_while_walk_linked_nodes() {
        let source = r#"
            n3 = {"next": null}
            n2 = {"next": n3}
            n1 = {"next": n2}
            node = n1
            while node != null: node = node.next
            node == null
        "#;
        assert_bool(source, true);
    }

    #[test]
    fn test_single_line_while_nested_calls() {
        let source = r#"
            queue = [1, 2, 3]
            out = []
            while len(queue) > 0: out = push(out, pop(queue))
            len(out)
        "#;
        assert_number(source, 3.0);
    }

    #[test]
    fn test_block_and_single_line_loops_coexist() {
        let source = r#"
            x = 10
            while x > 5 {
                x -= 1
            }
            while x > 0: x -= 1
            x
        "#;
        assert_number(source, 0.0);
    }
}
