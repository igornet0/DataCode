//! Парсинг generic-аннотаций типов: tuple[...,], set[...,], Optional[...], вложение.

#[cfg(test)]
mod tests {
    use data_code::lexer::Lexer;
    use data_code::parser::ast::{Param, Stmt, TypePart};
    use data_code::parser::Parser;

    fn parse_stmts(source: &str) -> Vec<Stmt> {
        let mut lexer = Lexer::new(source);
        let tokens = lexer.tokenize().unwrap();
        let mut parser = Parser::new(tokens);
        parser.parse().unwrap()
    }

    fn first_fn_params(source: &str) -> Vec<Param> {
        let stmts = parse_stmts(source);
        let Stmt::Function { params, .. } = &stmts[0] else {
            panic!("expected first stmt to be Function");
        };
        params.clone()
    }

    fn first_fn_return(source: &str) -> Option<Vec<TypePart>> {
        let stmts = parse_stmts(source);
        let Stmt::Function { return_type, .. } = &stmts[0] else {
            panic!("expected first stmt to be Function");
        };
        return_type.clone()
    }

    #[test]
    fn parse_tuple_two_ints() {
        let p = &first_fn_params("fn f(x: tuple[int, int]) { }")[0];
        let Some(ann) = &p.type_annotation else {
            panic!("expected annotation");
        };
        assert_eq!(
            ann.as_slice(),
            &[TypePart::Generic {
                base: "tuple".to_string(),
                args: vec![
                    TypePart::TypeName("int".to_string()),
                    TypePart::TypeName("int".to_string()),
                ]
            }]
        );
    }

    #[test]
    fn parse_set_of_tuple_nested() {
        let p = &first_fn_params("fn g(s: set[tuple[int, int]]) { }")[0];
        let ann = p.type_annotation.as_ref().unwrap();
        assert_eq!(
            ann[0],
            TypePart::Generic {
                base: "set".to_string(),
                args: vec![TypePart::Generic {
                    base: "tuple".to_string(),
                    args: vec![
                        TypePart::TypeName("int".to_string()),
                        TypePart::TypeName("int".to_string()),
                    ]
                }]
            }
        );
    }

    #[test]
    fn parse_optional_array_nested() {
        let ret = first_fn_return("fn h() -> Optional[array[tuple[int, int]]] { return null }")
            .expect("return type");
        assert_eq!(
            ret.as_slice(),
            &[TypePart::Generic {
                base: "Optional".to_string(),
                args: vec![TypePart::Generic {
                    base: "array".to_string(),
                    args: vec![TypePart::Generic {
                        base: "tuple".to_string(),
                        args: vec![
                            TypePart::TypeName("int".to_string()),
                            TypePart::TypeName("int".to_string()),
                        ]
                    }]
                }]
            }]
        );
    }

    #[test]
    fn parse_str_numeric_subscript_stays_flat_typename() {
        let p = &first_fn_params("fn q(x: str[50]) { }")[0];
        assert_eq!(
            p.type_annotation.as_ref().unwrap()[0],
            TypePart::TypeName("str[50]".to_string())
        );
    }

    #[test]
    fn parse_column_int_generic() {
        let p = &first_fn_params("fn c(x: Column[int]) { }")[0];
        assert_eq!(
            p.type_annotation.as_ref().unwrap()[0],
            TypePart::Generic {
                base: "Column".to_string(),
                args: vec![TypePart::TypeName("int".to_string())],
            }
        );
    }

    #[test]
    fn parse_union_int_float() {
        let p = &first_fn_params("fn u(x: int | float) { }")[0];
        assert_eq!(
            p.type_annotation.as_ref().unwrap().as_slice(),
            &[
                TypePart::TypeName("int".to_string()),
                TypePart::TypeName("float".to_string())
            ]
        );
    }

    #[test]
    fn parse_paren_union_inside_tuple() {
        let p = &first_fn_params("fn t(x: tuple[(int | float), str]) { }")[0];
        assert_eq!(
            p.type_annotation.as_ref().unwrap()[0],
            TypePart::Generic {
                base: "tuple".to_string(),
                args: vec![
                    TypePart::Union(vec![
                        TypePart::TypeName("int".to_string()),
                        TypePart::TypeName("float".to_string()),
                    ]),
                    TypePart::TypeName("str".to_string()),
                ]
            }
        );
    }

    #[test]
    fn parse_error_unclosed_generic() {
        let mut lexer = Lexer::new("fn bad(x: tuple[int,) { }");
        let tokens = lexer.tokenize().unwrap();
        let mut parser = Parser::new(tokens);
        assert!(parser.parse().is_err());
    }

    /// Сигнатура из a_start_adv.dc (тело опущено — полный файл может требовать preload/другой грамматики).
    #[test]
    fn smoke_parse_a_star_grid_signature() {
        let source = r#"fn a_star_grid(rows: int, cols: int,
                start: tuple[int, int], goal: tuple[int, int],
                blocked: set[tuple[int, int]]) -> Optional[array[tuple[int, int]]] {
            return null
        }"#;
        let stmts = parse_stmts(source);
        let a_star = stmts
            .iter()
            .find(|s| matches!(s, Stmt::Function { name, .. } if name == "a_star_grid"))
            .expect("a_star_grid fn");
        let Stmt::Function {
            name,
            params,
            return_type,
            ..
        } = a_star
        else {
            unreachable!()
        };
        assert_eq!(name, "a_star_grid");
        assert_eq!(params.len(), 5);

        assert!(return_type.is_some());
        let rt = return_type.as_ref().unwrap();
        assert_eq!(rt.len(), 1);
        match &rt[0] {
            TypePart::Generic { base, .. } => assert_eq!(base, "Optional"),
            other => panic!("expected Optional generic, got {:?}", other),
        }

        let blocked = params
            .iter()
            .find(|p| p.name == "blocked")
            .expect("blocked param");
        let ann = blocked.type_annotation.as_ref().unwrap();
        assert!(matches!(
            &ann[0],
            TypePart::Generic { base, .. } if base == "set"
        ));
    }

    #[test]
    fn serde_roundtrip_typepart_generic_json() {
        let t = TypePart::Generic {
            base: "tuple".to_string(),
            args: vec![
                TypePart::TypeName("int".to_string()),
                TypePart::TypeName("int".to_string()),
            ],
        };
        let s = serde_json::to_string(&t).unwrap();
        let back: TypePart = serde_json::from_str(&s).unwrap();
        assert_eq!(t, back);

        let legacy_json = serde_json::to_string(&TypePart::TypeName("int".into())).unwrap();
        let lb: TypePart = serde_json::from_str(&legacy_json).unwrap();
        assert!(matches!(lb, TypePart::TypeName(ref s) if s == "int"));
    }

}
