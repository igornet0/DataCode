// Tests for single-line if: `if <condition>: <statement>`

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
    fn test_parse_single_line_if() {
        let stmts = parse("if x > 0: return 1");
        assert_eq!(stmts.len(), 1);
        let Stmt::If { then_branch, else_branch, .. } = &stmts[0] else {
            panic!("Expected If");
        };
        assert_eq!(then_branch.len(), 1);
        assert!(matches!(then_branch[0], Stmt::Return { .. }));
        assert!(else_branch.is_none());
    }

    #[test]
    fn test_parse_block_if_still_works() {
        let stmts = parse("if x > 5 { let y = 10 }");
        assert_eq!(stmts.len(), 1);
        assert!(matches!(stmts[0], Stmt::If { .. }));
    }

    #[test]
    fn test_parse_single_line_if_assignment() {
        let stmts = parse("if x < 0: x = 0");
        let Stmt::If { then_branch, .. } = &stmts[0] else {
            panic!("Expected If");
        };
        assert!(matches!(then_branch[0], Stmt::Expr { .. }));
    }

    #[test]
    fn test_parse_error_missing_statement_after_colon() {
        assert_parse_error("if x > 0:");
    }

    #[test]
    fn test_parse_error_missing_condition() {
        assert_parse_error("if : return 1");
    }

    #[test]
    fn test_parse_error_missing_colon_and_brace() {
        assert_parse_error("if x > 0 return 1");
    }

    // ========== Runtime (tests 1–7 from spec) ==========

    #[test]
    fn test_single_line_if_true_return() {
        let source = r#"
            fn test() {
                if true: return 1
            }
            test()
        "#;
        assert_number(source, 1.0);
    }

    #[test]
    fn test_single_line_if_false_return_skipped() {
        let source = r#"
            fn test() {
                if false: return 1
                return 2
            }
            test()
        "#;
        assert_number(source, 2.0);
    }

    #[test]
    fn test_single_line_if_assignment() {
        let source = r#"
            x = 0
            if x == 0: x = 10
            x
        "#;
        assert_number(source, 10.0);
    }

    #[test]
    fn test_single_line_if_in_operator() {
        let source = r#"
            fn test() {
                if 5 in [1, 2, 3, 4, 5]: return true
            }
            test()
        "#;
        assert_bool(source, true);
    }

    #[test]
    fn test_single_line_if_or_short_circuit() {
        let source = r#"
            fn test() {
                let blocked_ids = [10, 20]
                let start_id = 10
                let goal_id = 5
                if start_id in blocked_ids or goal_id in blocked_ids: return null
                return true
            }
            test()
        "#;
        match run(source) {
            Ok(Value::Null) => {}
            Ok(v) => panic!("Expected Null, got {:?}", v),
            Err(e) => panic!("Error: {:?}", e),
        }
    }

    #[test]
    fn test_single_line_if_continue_in_loop() {
        let source = r#"
            fn test() {
                let out = []
                for i in [1, 2, 3] {
                    if i == 2: continue
                    out = out + [i]
                }
                return out
            }
            test()
        "#;
        match run(source) {
            Ok(Value::Array(a)) => {
                assert_eq!(a.borrow().len(), 2);
            }
            Ok(v) => panic!("Expected Array, got {:?}", v),
            Err(e) => panic!("Error: {:?}", e),
        }
    }

    #[test]
    fn test_single_line_if_break_in_loop() {
        let source = r#"
            fn test() {
                let out = []
                for i in [1, 2, 3] {
                    if i == 2: break
                    out = out + [i]
                }
                return out
            }
            test()
        "#;
        match run(source) {
            Ok(Value::Array(a)) => {
                assert_eq!(a.borrow().len(), 1);
            }
            Ok(v) => panic!("Expected Array, got {:?}", v),
            Err(e) => panic!("Error: {:?}", e),
        }
    }

    #[test]
    fn test_chained_comparison_bounds() {
        assert_bool("0 <= 3 and 3 < 5", true);
        assert_bool("0 <= 5 and 5 < 5", false);
        assert_bool("!(0 <= 6 and 6 < 5)", true);
    }

    /// Pattern from `a_start_adv_2.dc`: `if !(0 <= nr < rows and 0 <= nc < cols): continue`
    #[test]
    fn test_complex_single_line_if_continue_grid_bounds() {
        let source = r#"
            fn inside(rows, cols, r, c, dr, dc) {
                let nr = r + dr
                let nc = c + dc
                if !(0 <= nr < rows and 0 <= nc < cols): return false
                return true
            }
            inside(5, 5, 1, 1, 0, 0)
        "#;
        assert_bool(source, true);
    }

    #[test]
    fn test_complex_single_line_if_continue_skips_out_of_bounds() {
        let source = r#"
            fn walk(rows, cols) {
                let visited = []
                for dr in [0, 1] {
                    for dc in [0, 1] {
                        let nr = dr
                        let nc = dc
                        if !(0 <= nr < rows and 0 <= nc < cols): continue
                        visited = visited + [nr * cols + nc]
                    }
                }
                return visited
            }
            walk(2, 2)
        "#;
        match run(source) {
            Ok(Value::Array(a)) => {
                let n = a.borrow().len();
                assert_eq!(n, 4, "expected 4 in-bounds cells, got {}", n);
            }
            Ok(v) => panic!("Expected Array, got {:?}", v),
            Err(e) => panic!("Error: {:?}", e),
        }
    }

    #[test]
    fn test_parse_complex_if_condition_with_continue() {
        let stmts = parse(
            "fn f(rows, cols, nr, nc) { if !(0 <= nr < rows and 0 <= nc < cols): continue }",
        );
        assert_eq!(stmts.len(), 1);
        let Stmt::Function { body, .. } = &stmts[0] else {
            panic!("expected fn");
        };
        assert!(body.iter().any(|s| matches!(s, Stmt::If { .. })));
    }

    #[test]
    fn test_block_and_single_line_if_coexist() {
        let source = r#"
            fn test(x) {
                if x > 0 {
                    return x * 2
                }
                if x < 0: return 0
                return -1
            }
            test(3)
        "#;
        assert_number(source, 6.0);
    }
}
