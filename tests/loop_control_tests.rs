// Тесты для break и continue в циклах

#[cfg(test)]
mod tests {
    use data_code::{run, Value};

    // Вспомогательная функция для проверки числового результата
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

    // Вспомогательная функция для проверки ошибки
    fn assert_error(source: &str) {
        let result = run(source);
        assert!(result.is_err(), "Expected error, got {:?}", result);
    }

    // ========== Тесты для while циклов ==========

    #[test]
    fn test_while_break() {
        let source = r#"
            let i = 0
            let sum = 0
            while i < 10 {
                if i >= 5 {
                    break
                }
                sum += i
                i += 1
            }
            sum
        "#;
        // sum = 0 + 1 + 2 + 3 + 4 = 10
        assert_number_result(source, 10.0);
    }

    #[test]
    fn test_while_continue() {
        let source = r#"
            let i = 0
            let sum = 0
            while i < 10 {
                i += 1
                if i % 2 == 0 {
                    continue
                }
                sum += i
            }
            sum
        "#;
        // sum = 1 + 3 + 5 + 7 + 9 = 25 (только нечетные)
        assert_number_result(source, 25.0);
    }

    #[test]
    fn test_while_break_early() {
        let source = r#"
            let i = 0
            while i < 10 {
                break
                i += 1
            }
            i
        "#;
        // i остается 0, так как break выполняется сразу
        assert_number_result(source, 0.0);
    }

    #[test]
    fn test_while_continue_skip_all() {
        let source = r#"
            let i = 0
            let count = 0
            while i < 5 {
                i += 1
                continue
                count += 1
            }
            count
        "#;
        // count остается 0, так как continue пропускает остаток
        assert_number_result(source, 0.0);
    }

    // ========== Тесты для for циклов ==========

    #[test]
    fn test_for_break() {
        let source = r#"
            let sum = 0
            for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] {
                if i > 5 {
                    break
                }
                sum += i
            }
            sum
        "#;
        // sum = 1 + 2 + 3 + 4 + 5 = 15
        assert_number_result(source, 15.0);
    }

    #[test]
    fn test_for_continue() {
        let source = r#"
            let sum = 0
            for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] {
                if i % 2 == 0 {
                    continue
                }
                sum += i
            }
            sum
        "#;
        // sum = 1 + 3 + 5 + 7 + 9 = 25 (только нечетные)
        assert_number_result(source, 25.0);
    }

    #[test]
    fn test_for_break_early() {
        let source = r#"
            let count = 0
            for i in [1, 2, 3, 4, 5] {
                break
                count += 1
            }
            count
        "#;
        // count остается 0, так как break выполняется на первой итерации
        assert_number_result(source, 0.0);
    }

    #[test]
    fn test_for_continue_skip_elements() {
        let source = r#"
            let sum = 0
            for i in [1, 2, 3, 4, 5] {
                if i == 3 {
                    continue
                }
                sum += i
            }
            sum
        "#;
        // sum = 1 + 2 + 4 + 5 = 12 (пропущен 3)
        assert_number_result(source, 12.0);
    }

    // ========== Тесты для вложенных циклов ==========

    #[test]
    fn test_while_nested_break() {
        let source = r#"
            let sum = 0
            let i = 0
            while i < 3 {
                let j = 0
                while j < 5 {
                    if j >= 2 {
                        break
                    }
                    sum += j
                    j += 1
                }
                i += 1
            }
            sum
        "#;
        // sum = (0 + 1) * 3 = 3 (внутренний break прерывает только внутренний цикл)
        // Для каждой итерации внешнего цикла (i=0,1,2): j=0 добавляет 0, j=1 добавляет 1, j=2 делает break
        assert_number_result(source, 3.0);
    }

    #[test]
    fn test_while_nested_continue() {
        let source = r#"
            let sum = 0
            let i = 0
            while i < 3 {
                let j = 0
                while j < 3 {
                    j += 1
                    if j == 2 {
                        continue
                    }
                    sum += j
                }
                i += 1
            }
            sum
        "#;
        // sum = (1 + 3) * 3 = 12 (continue пропускает j=2)
        assert_number_result(source, 12.0);
    }

    #[test]
    fn test_for_nested_break() {
        let source = r#"
            let sum = 0
            for i in [1, 2, 3] {
                for j in [1, 2, 3, 4, 5] {
                    if j > 2 {
                        break
                    }
                    sum += i * j
                }
            }
            sum
        "#;
        // sum = (1*1 + 1*2) + (2*1 + 2*2) + (3*1 + 3*2) = 3 + 6 + 9 = 18
        assert_number_result(source, 18.0);
    }

    #[test]
    fn test_for_nested_continue() {
        let source = r#"
            let sum = 0
            for i in [1, 2, 3] {
                for j in [1, 2, 3] {
                    if j == 2 {
                        continue
                    }
                    sum += i * j
                }
            }
            sum
        "#;
        // sum = (1*1 + 1*3) + (2*1 + 2*3) + (3*1 + 3*3) = 4 + 8 + 12 = 24
        assert_number_result(source, 24.0);
    }

    #[test]
    fn test_break_in_nested_while_for() {
        let source = r#"
            let sum = 0
            let i = 0
            while i < 3 {
                for j in [1, 2, 3, 4, 5] {
                    if j > 3 {
                        break
                    }
                    sum += j
                }
                i += 1
            }
            sum
        "#;
        // sum = (1 + 2 + 3) * 3 = 18
        assert_number_result(source, 18.0);
    }

    #[test]
    fn test_continue_in_nested_for_while() {
        let source = r#"
            let sum = 0
            for i in [1, 2, 3] {
                let j = 0
                while j < 3 {
                    j += 1
                    if j == 2 {
                        continue
                    }
                    sum += i * j
                }
            }
            sum
        "#;
        // sum = (1*1 + 1*3) + (2*1 + 2*3) + (3*1 + 3*3) = 4 + 8 + 12 = 24
        assert_number_result(source, 24.0);
    }

    // ========== Комбинированные тесты ==========

    #[test]
    fn test_break_continue_mixed() {
        let source = r#"
            let sum = 0
            for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] {
                if i > 7 {
                    break
                }
                if i % 2 == 0 {
                    continue
                }
                sum += i
            }
            sum
        "#;
        // sum = 1 + 3 + 5 + 7 = 16 (нечетные до 7 включительно)
        assert_number_result(source, 16.0);
    }

    #[test]
    fn test_multiple_break_continue() {
        let source = r#"
            let count = 0
            let i = 0
            while i < 10 {
                i += 1
                if i < 3 {
                    continue
                }
                if i > 7 {
                    break
                }
                count += 1
            }
            count
        "#;
        // count = 5 (итерации 3, 4, 5, 6, 7)
        assert_number_result(source, 5.0);
    }

    // ========== Тесты для ошибок ==========

    #[test]
    fn test_break_outside_loop() {
        let source = "break";
        assert_error(source);
    }

    #[test]
    fn test_continue_outside_loop() {
        let source = "continue";
        assert_error(source);
    }

    #[test]
    fn test_break_in_if() {
        let source = r#"
            if true {
                break
            }
        "#;
        assert_error(source);
    }

    #[test]
    fn test_continue_in_if() {
        let source = r#"
            if true {
                continue
            }
        "#;
        assert_error(source);
    }

    // ========== break/continue внутри try в цикле ==========

    #[test]
    fn test_break_inside_try_in_loop() {
        let source = r#"
            let sum = 0
            let i = 0
            while i < 10 {
                i += 1
                try {
                    if i >= 5 {
                        break
                    }
                    sum += i
                } catch {
                    sum += 100
                }
            }
            sum
        "#;
        // sum = 1 + 2 + 3 + 4 = 10, потом break
        assert_number_result(source, 10.0);
    }

    #[test]
    fn test_continue_inside_try_in_loop() {
        let source = r#"
            let sum = 0
            for i in [1, 2, 3, 4, 5] {
                try {
                    if i == 3 {
                        continue
                    }
                    sum += i
                } catch {
                    sum += 100
                }
            }
            sum
        "#;
        // sum = 1 + 2 + 4 + 5 = 12 (пропущен 3)
        assert_number_result(source, 12.0);
    }

    #[test]
    fn test_break_inside_try_catch_in_for_loop() {
        let source = r#"
            let count = 0
            for x in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] {
                try {
                    if x > 6 {
                        break
                    }
                    count += 1
                } catch {
                    count += 100
                }
            }
            count
        "#;
        // count = 6 (итерации 1..6, затем break)
        assert_number_result(source, 6.0);
    }
}

