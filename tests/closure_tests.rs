// Тесты для замыканий (closures): захват переменных из внешней области видимости.
// Проверяем, что вложенные функции корректно захватывают локальные переменные родителя.

#[cfg(test)]
mod tests {
    use data_code::{run, Value};

    fn assert_number_result(source: &str, expected: f64) {
        let result = run(source);
        match result {
            Ok(Value::Number(n)) => {
                assert_eq!(n, expected, "Expected {}, got {}", expected, n);
            }
            Ok(v) => panic!("Expected Number({}), got {:?}", expected, v),
            Err(e) => panic!("Error: {:?}", e),
        }
    }

    // ========== Захват переменной при чтении ==========

    #[test]
    fn test_closure_captures_local_variable() {
        // inner захватывает x из outer (локальная переменная outer, не глобальная)
        let source = r#"
            fn outer() {
                let x = 42
                fn inner() {
                    return x
                }
                return inner()
            }
            outer()
        "#;
        assert_number_result(source, 42.0);
    }

    #[test]
    fn test_closure_captures_from_multiple_levels() {
        // inner захватывает x из middle, middle захватывает x из outer
        let source = r#"
            fn outer() {
                let x = 10
                fn middle() {
                    fn inner() {
                        return x
                    }
                    return inner()
                }
                return middle()
            }
            outer()
        "#;
        assert_number_result(source, 10.0);
    }

    #[test]
    fn test_closure_sees_updated_value() {
        // Замыкание видит актуальное значение на момент вызова (если передаётся по ссылке/захвату)
        let source = r#"
            fn outer() {
                x = 1
                fn get_x() { return x }
                a = get_x()
                x = 2
                b = get_x()
                return a + b
            }
            outer()
        "#;
        // a=1 (при первом вызове), b=2 (при втором — x уже 2)
        assert_number_result(source, 3.0);
    }

    #[test]
    fn test_closure_captures_parameter() {
        // inner захватывает параметр a функции outer
        let source = r#"
            fn outer(a) {
                fn inner() {
                    return a + 10
                }
                return inner()
            }
            outer(5)
        "#;
        assert_number_result(source, 15.0);
    }

    #[test]
    fn test_closure_called_before_return() {
        // inner вызывается внутри outer и захватывает x
        let source = r#"
            fn make_and_apply(x) {
                fn add(y) {
                    return x + y
                }
                return add(3)
            }
            make_and_apply(5)
        "#;
        assert_number_result(source, 8.0);
    }

    #[test]
    fn test_closure_multiple_captures() {
        let source = r#"
            fn outer() {
                let a = 1
                let b = 2
                fn sum() {
                    return a + b
                }
                return sum()
            }
            outer()
        "#;
        assert_number_result(source, 3.0);
    }

    #[test]
    fn test_closure_shadowing_parameter() {
        // inner имеет свой параметр x, не захватывает outer's x
        let source = r#"
            fn outer() {
                let x = 100
                fn inner(x) {
                    return x
                }
                return inner(5)
            }
            outer()
        "#;
        assert_number_result(source, 5.0);
    }

    #[test]
    fn test_closure_and_global_distinction() {
        // inner использует глобальную g и захваченную локальную x
        let source = r#"
            let g = 10
            fn outer() {
                let x = 20
                fn inner() {
                    return g + x
                }
                return inner()
            }
            outer()
        "#;
        assert_number_result(source, 30.0);
    }
}
