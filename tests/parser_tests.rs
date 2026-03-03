// Тесты для парсера
#[cfg(test)]
mod tests {
    use data_code::parser::{Parser, Stmt};
    use data_code::lexer::Lexer;
    use data_code::parser::ast::{Expr, Arg};

    fn parse(source: &str) -> Vec<Stmt> {
        let mut lexer = Lexer::new(source);
        let tokens = lexer.tokenize().unwrap();
        let mut parser = Parser::new(tokens);
        parser.parse().unwrap()
    }

    #[test]
    fn test_let_statement() {
        let source = "let x = 10";
        let stmts = parse(source);
        assert_eq!(stmts.len(), 1);
        if let Stmt::Let { name, .. } = &stmts[0] {
            assert_eq!(name, "x");
        } else {
            panic!("Expected Let statement");
        }
    }

    #[test]
    fn test_binary_expression() {
        let source = "let result = 5 + 3 * 2";
        let stmts = parse(source);
        assert_eq!(stmts.len(), 1);
    }

    #[test]
    fn test_if_statement() {
        let source = "if x > 5 { let y = 10 }";
        let stmts = parse(source);
        assert_eq!(stmts.len(), 1);
        if let Stmt::If { .. } = &stmts[0] {
            // OK
        } else {
            panic!("Expected If statement");
        }
    }

    #[test]
    fn test_function_declaration() {
        let source = "fn sum(a, b) { return a + b }";
        let stmts = parse(source);
        assert_eq!(stmts.len(), 1);
        if let Stmt::Function { name, params, .. } = &stmts[0] {
            assert_eq!(name, "sum");
            assert_eq!(params.len(), 2);
            assert_eq!(params[0].name, "a");
            assert_eq!(params[1].name, "b");
        } else {
            panic!("Expected Function statement");
        }
    }

    #[test]
    fn test_while_loop() {
        let source = "while x > 0 { x = x - 1 }";
        let stmts = parse(source);
        assert_eq!(stmts.len(), 1);
        if let Stmt::While { .. } = &stmts[0] {
            // OK
        } else {
            panic!("Expected While statement");
        }
    }

    #[test]
    fn test_global_variable_declaration() {
        let source = "global x = 10";
        let stmts = parse(source);
        assert_eq!(stmts.len(), 1);
        if let Stmt::Let { name, is_global, .. } = &stmts[0] {
            assert_eq!(name, "x");
            assert_eq!(*is_global, true);
        } else {
            panic!("Expected Let statement with is_global=true");
        }
    }

    #[test]
    fn test_local_variable_declaration() {
        let source = "let x = 10";
        let stmts = parse(source);
        assert_eq!(stmts.len(), 1);
        if let Stmt::Let { name, is_global, .. } = &stmts[0] {
            assert_eq!(name, "x");
            assert_eq!(*is_global, false);
        } else {
            panic!("Expected Let statement with is_global=false");
        }
    }

    #[test]
    fn test_named_argument_parsing() {
        let source = "let result = pow(base=2, exp=3)";
        let stmts = parse(source);
        assert_eq!(stmts.len(), 1);
        if let Stmt::Let { value, .. } = &stmts[0] {
            if let Expr::Call { name, args, .. } = value {
                assert_eq!(name, "pow");
                assert_eq!(args.len(), 2);
                // Проверяем, что оба аргумента именованные
                match &args[0] {
                    Arg::Named { name, .. } => assert_eq!(name, "base"),
                    _ => panic!("Expected named argument 'base'"),
                }
                match &args[1] {
                    Arg::Named { name, .. } => assert_eq!(name, "exp"),
                    _ => panic!("Expected named argument 'exp'"),
                }
            } else {
                panic!("Expected Call expression");
            }
        } else {
            panic!("Expected Let statement");
        }
    }

    #[test]
    fn test_mixed_positional_and_named_arguments() {
        let source = "let result = pow(2, exp=3)";
        let stmts = parse(source);
        assert_eq!(stmts.len(), 1);
        if let Stmt::Let { value, .. } = &stmts[0] {
            if let Expr::Call { name, args, .. } = value {
                assert_eq!(name, "pow");
                assert_eq!(args.len(), 2);
                // Первый аргумент должен быть позиционным
                match &args[0] {
                    Arg::Positional(_) => {},
                    _ => panic!("Expected positional argument"),
                }
                // Второй аргумент должен быть именованным
                match &args[1] {
                    Arg::Named { name, .. } => assert_eq!(name, "exp"),
                    _ => panic!("Expected named argument 'exp'"),
                }
            } else {
                panic!("Expected Call expression");
            }
        } else {
            panic!("Expected Let statement");
        }
    }

    #[test]
    fn test_only_named_arguments() {
        let source = "let result = pow(base=2, exp=3)";
        let stmts = parse(source);
        assert_eq!(stmts.len(), 1);
        if let Stmt::Let { value, .. } = &stmts[0] {
            if let Expr::Call { name, args, .. } = value {
                assert_eq!(name, "pow");
                assert_eq!(args.len(), 2);
                // Оба аргумента должны быть именованными
                for arg in args {
                    match arg {
                        Arg::Named { .. } => {},
                        _ => panic!("Expected all named arguments"),
                    }
                }
            } else {
                panic!("Expected Call expression");
            }
        } else {
            panic!("Expected Let statement");
        }
    }

    #[test]
    fn test_named_argument_with_function_call() {
        let source = "let result = read_file(path(\"data.csv\"), header_row=2)";
        let stmts = parse(source);
        assert_eq!(stmts.len(), 1);
        if let Stmt::Let { value, .. } = &stmts[0] {
            if let Expr::Call { name, args, .. } = value {
                assert_eq!(name, "read_file");
                assert_eq!(args.len(), 2);
                // Первый аргумент должен быть позиционным (вызов функции path)
                match &args[0] {
                    Arg::Positional(_) => {},
                    _ => panic!("Expected positional argument"),
                }
                // Второй аргумент должен быть именованным
                match &args[1] {
                    Arg::Named { name, .. } => assert_eq!(name, "header_row"),
                    _ => panic!("Expected named argument 'header_row'"),
                }
            } else {
                panic!("Expected Call expression");
            }
        } else {
            panic!("Expected Let statement");
        }
    }

    #[test]
    #[should_panic]
    fn test_positional_after_named_argument_error() {
        // Позиционный аргумент после именованного должен вызывать ошибку парсинга
        let source = "let result = pow(base=2, 3)";
        // parse() использует unwrap(), поэтому ошибка парсинга приведет к panic
        let _stmts = parse(source);
    }

    /// Call arguments must parse full expression including member chain (e.g. User.metadata.create_all).
    /// If the parser only parses primary(), run(User.metadata.create_all) would get just User as arg.
    #[test]
    fn test_call_argument_parses_member_chain() {
        let source = "e.run(User.metadata.create_all)";
        let stmts = parse(source);
        assert_eq!(stmts.len(), 1, "expected one statement");
        let Stmt::Expr { expr, .. } = &stmts[0] else {
            panic!("expected Expr statement, got {:?}", stmts[0]);
        };
        let Expr::MethodCall { method, args, .. } = expr else {
            panic!("expected MethodCall expression, got {:?}", expr);
        };
        assert_eq!(method, "run");
        assert_eq!(args.len(), 1, "expected one argument");
        let Arg::Positional(arg_expr) = &args[0] else {
            panic!("expected positional argument");
        };
        // Argument must be Property(Property(User, "metadata"), "create_all"), not just Variable(User)
        let Expr::Property { name, object, .. } = arg_expr else {
            panic!("expected argument to be Property (member chain), got {:?}", arg_expr);
        };
        assert_eq!(name, "create_all");
        let Expr::Property { name: inner_name, object: inner_obj, .. } = object.as_ref() else {
            panic!("expected inner Property (metadata), got {:?}", object);
        };
        assert_eq!(inner_name, "metadata");
        let Expr::Variable { name: var_name, .. } = inner_obj.as_ref() else {
            panic!("expected Variable(User), got {:?}", inner_obj);
        };
        assert_eq!(var_name, "User");
    }
}

