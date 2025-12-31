// Дополнительные тесты для интерпретации
// Тестируем граничные случаи, сложные сценарии и обработку ошибок

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

    // Вспомогательная функция для проверки строкового результата
    fn assert_string_result(source: &str, expected: &str) {
        let result = run(source);
        match result {
            Ok(Value::String(s)) => {
                assert_eq!(s, expected, "Expected '{}', got '{}'", expected, s);
            }
            Ok(v) => panic!("Expected String('{}'), got {:?}", expected, v),
            Err(e) => panic!("Error: {:?}", e),
        }
    }

    // ========== Тесты для граничных случаев ==========

    #[test]
    fn test_empty_program() {
        let source = "";
        let result = run(source);
        // Пустая программа должна вернуть null
        match result {
            Ok(Value::Null) => {}
            Ok(v) => panic!("Expected Null, got {:?}", v),
            Err(e) => panic!("Error: {:?}", e),
        }
    }

    #[test]
    fn test_single_number() {
        assert_number_result("42", 42.0);
    }

    #[test]
    fn test_single_string() {
        assert_string_result(r#""hello""#, "hello");
    }

    #[test]
    fn test_single_boolean() {
        let result = run("true");
        match result {
            Ok(Value::Bool(true)) => {}
            Ok(v) => panic!("Expected Bool(true), got {:?}", v),
            Err(e) => panic!("Error: {:?}", e),
        }
    }

    #[test]
    fn test_null_literal() {
        let result = run("null");
        match result {
            Ok(Value::Null) => {}
            Ok(v) => panic!("Expected Null, got {:?}", v),
            Err(e) => panic!("Error: {:?}", e),
        }
    }

    // ========== Тесты для сложных выражений ==========

    #[test]
    fn test_nested_parentheses() {
        // ((1 + 2) * (3 + 4)) / 2 = 21 / 2 = 10.5
        assert_number_result("((1 + 2) * (3 + 4)) / 2", 10.5);
    }

    #[test]
    fn test_deep_nesting() {
        // ((((1 + 2) + 3) + 4) + 5) = 15
        assert_number_result("((((1 + 2) + 3) + 4) + 5)", 15.0);
    }

    #[test]
    fn test_complex_arithmetic() {
        // 10 * 2 + 3 * 4 - 5 = 20 + 12 - 5 = 27
        assert_number_result("10 * 2 + 3 * 4 - 5", 27.0);
    }

    #[test]
    fn test_negative_numbers() {
        assert_number_result("-5", -5.0);
        assert_number_result("10 + -5", 5.0);
        assert_number_result("10 - -5", 15.0);
    }

    // ========== Тесты для функций ==========

    #[test]
    fn test_function_without_parameters() {
        let source = r#"
            fn get_five() {
                return 5
            }
            get_five()
        "#;
        assert_number_result(source, 5.0);
    }

    #[test]
    fn test_function_with_zero_return() {
        let source = r#"
            fn zero() {
                return 0
            }
            zero()
        "#;
        assert_number_result(source, 0.0);
    }

    #[test]
    fn test_function_returns_function_result() {
        let source = r#"
            fn add(a, b) {
                return a + b
            }
            fn multiply(a, b) {
                return a * b
            }
            multiply(add(2, 3), add(4, 1))
        "#;
        assert_number_result(source, 25.0);
    }

    #[test]
    fn test_mutual_recursion() {
        let source = r#"
            fn is_even(n) {
                if n == 0 {
                    return true
                }
                return is_odd(n - 1)
            }
            fn is_odd(n) {
                if n == 0 {
                    return false
                }
                return is_even(n - 1)
            }
            if is_even(4) {
                1
            } else {
                0
            }
        "#;
        assert_number_result(source, 1.0);
    }

    #[test]
    fn test_function_with_multiple_returns() {
        let source = r#"
            fn abs(n) {
                if n >= 0 {
                    return n
                }
                return -n
            }
            abs(-10)
        "#;
        assert_number_result(source, 10.0);
    }

    // ========== Тесты для циклов ==========

    #[test]
    fn test_empty_while_loop() {
        let source = r#"
            let x = 5
            while x > 10 {
                x = x + 1
            }
            x
        "#;
        // Цикл не должен выполняться
        assert_number_result(source, 5.0);
    }

    #[test]
    fn test_while_loop_with_break_condition() {
        let source = r#"
            let x = 0
            while x < 5 {
                x = x + 1
            }
            x
        "#;
        assert_number_result(source, 5.0);
    }

    #[test]
    fn test_nested_while_loops() {
        let source = r#"
            let sum = 0
            let i = 1
            while i <= 2 {
                let j = 1
                while j <= 2 {
                    sum = sum + i * j
                    j = j + 1
                }
                i = i + 1
            }
            sum
        "#;
        // sum = (1*1 + 1*2) + (2*1 + 2*2) = 3 + 6 = 9
        assert_number_result(source, 9.0);
    }

    // ========== Тесты для условных конструкций ==========

    #[test]
    fn test_if_without_else() {
        let source = r#"
            let x = 10
            if x > 5 {
                x = 20
            }
            x
        "#;
        assert_number_result(source, 20.0);
    }

    #[test]
    fn test_if_without_else_false() {
        let source = r#"
            let x = 3
            if x > 5 {
                x = 20
            }
            x
        "#;
        assert_number_result(source, 3.0);
    }

    #[test]
    fn test_nested_conditionals() {
        let source = r#"
            let x = 10
            let y = 5
            let result = 0
            if x > 5 {
                if y > 3 {
                    result = 1
                } else {
                    result = 2
                }
            } else {
                result = 3
            }
            result
        "#;
        assert_number_result(source, 1.0);
    }

    #[test]
    fn test_ternary_like_expression() {
        let source = r#"
            let x = 10
            let result = 0
            if x > 5 {
                result = 100
            } else {
                result = 200
            }
            result
        "#;
        assert_number_result(source, 100.0);
    }

    // ========== Тесты для строк ==========

    #[test]
    fn test_empty_string() {
        assert_string_result(r#""""#, "");
    }

    #[test]
    fn test_string_with_special_chars() {
        let source = r#"
            "hello\nworld\t!"
        "#;
        let result = run(source);
        match result {
            Ok(Value::String(s)) => {
                assert_eq!(s, "hello\nworld\t!", "Expected special chars");
            }
            Ok(v) => panic!("Expected String, got {:?}", v),
            Err(e) => panic!("Error: {:?}", e),
        }
    }

    #[test]
    fn test_multiple_string_concatenations() {
        let source = r#"
            "a" + "b" + "c" + "d"
        "#;
        assert_string_result(source, "abcd");
    }

    #[test]
    fn test_string_number_mixed_concatenation() {
        let source = r#"
            "Value: " + 42 + " is the answer"
        "#;
        assert_string_result(source, "Value: 42 is the answer");
    }

    // ========== Тесты для обработки ошибок ==========

    #[test]
    fn test_division_by_zero() {
        assert_error("10 / 0");
    }

    #[test]
    fn test_undefined_variable() {
        assert_error("x");
    }

    #[test]
    fn test_undefined_function() {
        assert_error("undefined_func()");
    }

    #[test]
    fn test_wrong_argument_count_more() {
        let source = r#"
            fn add(a, b) {
                return a + b
            }
            add(1, 2, 3)
        "#;
        assert_error(source);
    }

    #[test]
    fn test_wrong_argument_count_fewer() {
        let source = r#"
            fn add(a, b) {
                return a + b
            }
            add(1)
        "#;
        assert_error(source);
    }

    #[test]
    fn test_type_error_string_minus_number() {
        assert_error(r#""hello" - 5"#);
    }

    #[test]
    fn test_string_multiply_number() {
        assert_string_result(r#""-" * 10"#, "----------");
        assert_string_result(r#""hello" * 3"#, "hellohellohello");
        assert_string_result(r#""ab" * 5"#, "ababababab");
    }

    #[test]
    fn test_number_multiply_string() {
        assert_string_result(r#"5 * "-""#, "-----");
        assert_string_result(r#"3 * "hello""#, "hellohellohello");
        assert_string_result(r#"2 * "test""#, "testtest");
    }

    #[test]
    fn test_string_multiply_zero_or_negative() {
        assert_string_result(r#""hello" * 0"#, "");
        assert_string_result(r#""hello" * -5"#, "");
        assert_string_result(r#"0 * "hello""#, "");
        assert_string_result(r#"-3 * "hello""#, "");
    }

    #[test]
    fn test_type_error_string_divide_number() {
        assert_error(r#""hello" / 5"#);
    }

    #[test]
    fn test_type_error_string_compare_number() {
        assert_error(r#""hello" > 5"#);
    }

    // ========== Тесты для сложных сценариев ==========

    #[test]
    fn test_factorial_large() {
        let source = r#"
            fn factorial(n) {
                if n <= 1 {
                    return 1
                }
                return n * factorial(n - 1)
            }
            factorial(6)
        "#;
        assert_number_result(source, 720.0);
    }

    #[test]
    fn test_fibonacci_large() {
        let source = r#"
            fn fib(n) {
                if n <= 1 {
                    return n
                }
                return fib(n - 1) + fib(n - 2)
            }
            fib(8)
        "#;
        assert_number_result(source, 21.0);
    }

    #[test]
    fn test_complex_calculation() {
        let source = r#"
            fn power(base, exp) {
                if exp == 0 {
                    return 1
                }
                return base * power(base, exp - 1)
            }
            power(2, 8)
        "#;
        assert_number_result(source, 256.0);
    }

    #[test]
    fn test_gcd() {
        // gcd(48, 18) = gcd(18, 48 - 2*18) = gcd(18, 12) = gcd(12, 18 - 1*12) = gcd(12, 6) = gcd(6, 12 - 2*6) = gcd(6, 0) = 6
        // Но у нас нет оператора %, поэтому упростим
        let source = r#"
            fn gcd(a, b) {
                if b == 0 {
                    return a
                }
                if a > b {
                    return gcd(a - b, b)
                }
                return gcd(a, b - a)
            }
            gcd(48, 18)
        "#;
        assert_number_result(source, 6.0);
    }

    #[test]
    fn test_variable_shadowing() {
        let source = r#"
            let x = 100
            fn test() {
                let x = 10
                return x
            }
            test()
        "#;
        // Локальная переменная должна затенять глобальную
        assert_number_result(source, 10.0);
    }

    #[test]
    fn test_global_after_local() {
        let source = r#"
            let x = 100
            fn test() {
                let y = 10
                return x
            }
            test()
        "#;
        // Функция должна видеть глобальную переменную
        assert_number_result(source, 100.0);
    }

    #[test]
    fn test_multiple_global_variables() {
        let source = r#"
            let a = 10
            let b = 20
            let c = 30
            a + b + c
        "#;
        assert_number_result(source, 60.0);
    }

    #[test]
    fn test_function_as_value() {
        let source = r#"
            fn add(a, b) {
                return a + b
            }
            func = add
            func(5, 3)
        "#;
        // Функции должны быть доступны как значения
        assert_number_result(source, 8.0);
    }

    #[test]
    fn test_global_function_as_value() {
        let source = r#"
            fn add(a, b) {
                return a + b
            }
            global func = add
            func(5, 3)
        "#;
        // Функции должны быть доступны как значения
        assert_number_result(source, 8.0);
    }

    #[test]
    fn test_complex_nested_expressions() {
        let source = r#"
            let a = 10
            let b = 20
            let c = 30
            (a + b) * c - (a * b) / c
        "#;
        // (10 + 20) * 30 - (10 * 20) / 30 = 30 * 30 - 200 / 30 = 900 - 6.666... = 893.333...
        assert_number_result(source, 893.3333333333334);
    }

    #[test]
    fn test_boolean_logic() {
        let source = r#"
            let x = 10
            let y = 5
            if x > 5 and y < 10 {
                1
            } else {
                0
            }
        "#;
        // // У нас нет оператора &&, поэтому упростим
        // let source = r#"
        //     let x = 10
        //     let y = 5
        //     if x > 5 {
        //         if y < 10 {
        //             1
        //         } else {
        //             0
        //         }
        //     } else {
        //         0
        //     }
        // "#;
        assert_number_result(source, 1.0);
    }

    #[test]
    fn test_loop_with_conditional() {
        let source = r#"
            let sum = 0
            let i = 1
            while i <= 10 {
                if i > 5 {
                    sum = sum + i
                }
                i = i + 1
            }
            sum
        "#;
        // sum чисел > 5 = 6 + 7 + 8 + 9 + 10 = 40
        assert_number_result(source, 40.0);
    }

    #[test]
    fn test_recursive_sum() {
        let source = r#"
            fn sum(n) {
                if n <= 0 {
                    return 0
                }
                return n + sum(n - 1)
            }
            sum(100)
        "#;
        // sum(100) = 100 + 99 + ... + 1 = 5050
        assert_number_result(source, 5050.0);
    }

    // ========== Тесты для унарных операторов ==========

    #[test]
    fn test_unary_minus() {
        assert_number_result("-10", -10.0);
        assert_number_result("--5", 5.0); // Двойной минус
        assert_number_result("-(-5)", 5.0);
    }

    #[test]
    fn test_unary_bang() {
        let result = run("!true");
        match result {
            Ok(Value::Bool(false)) => {}
            Ok(v) => panic!("Expected Bool(false), got {:?}", v),
            Err(e) => panic!("Error: {:?}", e),
        }
        
        let result = run("!false");
        match result {
            Ok(Value::Bool(true)) => {}
            Ok(v) => panic!("Expected Bool(true), got {:?}", v),
            Err(e) => panic!("Error: {:?}", e),
        }
    }

    // ========== Тесты для логических операций ==========

    #[test]
    fn test_logical_and_equivalent() {
        // Эмулируем && через вложенные if
        let source = r#"
            let x = 10
            let y = 5
            if x > 5 {
                if y < 10 {
                    1
                } else {
                    0
                }
            } else {
                0
            }
        "#;
        assert_number_result(source, 1.0);
    }

    #[test]
    fn test_logical_or_equivalent() {
        // Эмулируем || через if-else
        let source = r#"
            let x = 3
            let result = 0
            if x > 5 {
                result = 1
            } else {
                if x < 2 {
                    result = 1
                } else {
                    result = 0
                }
            }
            result
        "#;
        assert_number_result(source, 0.0);
    }

    // ========== Тесты для операторов сравнения ==========

    #[test]
    fn test_comparison_operators() {
        assert_bool_result("10 > 5", true);
        assert_bool_result("5 > 10", false);
        assert_bool_result("10 < 5", false);
        assert_bool_result("5 < 10", true);
        assert_bool_result("10 >= 10", true);
        assert_bool_result("10 >= 5", true);
        assert_bool_result("5 >= 10", false);
        assert_bool_result("10 <= 10", true);
        assert_bool_result("5 <= 10", true);
        assert_bool_result("10 <= 5", false);
    }

    fn assert_bool_result(source: &str, expected: bool) {
        let result = run(source);
        match result {
            Ok(Value::Bool(b)) => {
                assert_eq!(b, expected, "Expected {}, got {}", expected, b);
            }
            Ok(v) => panic!("Expected Bool({}), got {:?}", expected, v),
            Err(e) => panic!("Error: {:?}", e),
        }
    }

    // ========== Тесты для сложных выражений с приоритетом ==========

    #[test]
    fn test_operator_precedence_complex() {
        // 2 + 3 * 4 - 5 / 2 = 2 + 12 - 2.5 = 11.5
        assert_number_result("2 + 3 * 4 - 5 / 2", 11.5);
    }

    #[test]
    fn test_operator_precedence_with_comparison() {
        // (2 + 3) * 4 > 10 = 20 > 10 = true
        let source = r#"
            if (2 + 3) * 4 > 10 {
                1
            } else {
                0
            }
        "#;
        assert_number_result(source, 1.0);
    }

    // ========== Тесты для функций с побочными эффектами ==========

    #[test]
    fn test_function_with_side_effects() {
        let source = r#"
            global counter = 0
            fn increment() {
                global counter = counter + 1
                return counter
            }
            increment()
            increment()
            increment()
        "#;
        assert_number_result(source, 3.0);
    }

    #[test]
    fn test_function_modifying_global() {
        let source = r#"
            let x = 10
            fn double() {
                x = x * 2
                return x
            }
            double()
        "#;
        assert_number_result(source, 20.0);
    }

    // ========== Тесты для вложенных функций ==========

    #[test]
    fn test_nested_function_calls() {
        let source = r#"
            fn add(a, b) {
                return a + b
            }
            fn multiply(a, b) {
                return a * b
            }
            fn calculate(x, y, z) {
                return multiply(add(x, y), z)
            }
            calculate(2, 3, 4)
        "#;
        // (2 + 3) * 4 = 20
        assert_number_result(source, 20.0);
    }

    // ========== Тесты для рекурсии с большими числами ==========

    #[test]
    fn test_deep_recursion() {
        let source = r#"
            fn countdown(n) {
                if n <= 0 {
                    return 0
                }
                return 1 + countdown(n - 1)
            }
            countdown(50)
        "#;
        assert_number_result(source, 50.0);
    }

    // ========== Тесты для циклов с условиями ==========

    #[test]
    fn test_while_with_break_logic() {
        let source = r#"
            let x = 0
            let found = false
            while x < 10 {
                if x == 5 {
                    found = true
                }
                x = x + 1
            }
            if found {
                1
            } else {
                0
            }
        "#;
        assert_number_result(source, 1.0);
    }

    // ========== Тесты для строковых операций ==========

    #[test]
    fn test_string_comparison() {
        let source = r#"
            if "hello" == "hello" {
                1
            } else {
                0
            }
        "#;
        assert_number_result(source, 1.0);
        
        let source = r#"
            if "hello" != "world" {
                1
            } else {
                0
            }
        "#;
        assert_number_result(source, 1.0);
    }

    #[test]
    fn test_string_concatenation_multiple() {
        let source = r#"
            "a" + "b" + "c" + "d" + "e"
        "#;
        assert_string_result(source, "abcde");
    }

    // ========== Тесты для null ==========

    #[test]
    fn test_null_comparison() {
        let source = r#"
            let x = null
            if x == null {
                1
            } else {
                0
            }
        "#;
        assert_number_result(source, 1.0);
    }

    #[test]
    fn test_null_in_function() {
        let source = r#"
            fn get_null() {
                return null
            }
            get_null()
        "#;
        let result = run(source);
        match result {
            Ok(Value::Null) => {}
            Ok(v) => panic!("Expected Null, got {:?}", v),
            Err(e) => panic!("Error: {:?}", e),
        }
    }

    // ========== Тесты для граничных случаев с числами ==========

    #[test]
    fn test_floating_point_arithmetic() {
        assert_number_result("0.1 + 0.2", 0.30000000000000004);
        assert_number_result("1.5 * 2", 3.0);
        assert_number_result("10.5 / 2", 5.25);
    }

    #[test]
    fn test_zero_operations() {
        assert_number_result("0 + 0", 0.0);
        assert_number_result("0 * 100", 0.0);
        assert_number_result("100 - 100", 0.0);
    }

    // ========== Тесты для сложных сценариев с переменными ==========

    #[test]
    fn test_variable_reassignment_in_loop() {
        let source = r#"
            let x = 0
            while x < 5 {
                x = x + 1
            }
            x
        "#;
        assert_number_result(source, 5.0);
    }

    #[test]
    fn test_multiple_variable_reassignments() {
        let source = r#"
            let a = 1
            let b = 2
            let c = 3
            a = b + c
            b = a + c
            c = a + b
            a + b + c
        "#;
        // a = 2 + 3 = 5
        // b = 5 + 3 = 8
        // c = 5 + 8 = 13
        // a + b + c = 5 + 8 + 13 = 26
        assert_number_result(source, 26.0);
    }

    // ========== Тесты для функций без параметров ==========

    #[test]
    fn test_function_no_params_with_globals() {
        let source = r#"
            let x = 100
            fn get_x() {
                return x
            }
            get_x()
        "#;
        assert_number_result(source, 100.0);
    }

    // ========== Тесты для сложных вложенных структур ==========

    #[test]
    fn test_deeply_nested_conditionals() {
        let source = r#"
            let x = 10
            let y = 5
            let z = 3
            let result = 0
            if x > 5 {
                if y > 3 {
                    if z > 2 {
                        result = 1
                    } else {
                        result = 2
                    }
                } else {
                    result = 3
                }
            } else {
                result = 4
            }
            result
        "#;
        assert_number_result(source, 1.0);
    }

    #[test]
    fn test_nested_loops_with_conditionals() {
        let source = r#"
            let sum = 0
            let i = 1
            while i <= 3 {
                let j = 1
                while j <= 2 {
                    if i * j > 2 {
                        sum = sum + i * j
                    }
                    j = j + 1
                }
                i = i + 1
            }
            sum
        "#;
        // i=1: j=1 (1*1=1 <=2), j=2 (1*2=2 <=2) -> нет
        // i=2: j=1 (2*1=2 <=2), j=2 (2*2=4 >2) -> +4
        // i=3: j=1 (3*1=3 >2) -> +3, j=2 (3*2=6 >2) -> +6
        // sum = 4 + 3 + 6 = 13
        assert_number_result(source, 13.0);
    }

    // ========== Тесты для цикла for ==========

    #[test]
    fn test_for_loop_basic() {
        let source = r#"
            let sum = 0
            for i in [1, 2, 3, 4, 5] {
                sum += i
            }
            sum
        "#;
        // sum = 1 + 2 + 3 + 4 + 5 = 15
        assert_number_result(source, 15.0);
    }

    #[test]
    fn test_for_loop_nested() {
        let source = r#"
            let sum = 0
            for i in [1, 2] {
                for j in [1, 2] {
                    sum += i * j
                }
            }
            sum
        "#;
        // sum = (1*1 + 1*2) + (2*1 + 2*2) = 3 + 6 = 9
        assert_number_result(source, 9.0);
    }

    #[test]
    fn test_for_loop_with_conditional() {
        let source = r#"
            let sum = 0
            for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] {
                if i > 5 {
                    sum += i
                }
            }
            sum
        "#;
        // sum = 6 + 7 + 8 + 9 + 10 = 40
        assert_number_result(source, 40.0);
    }
    
    #[test]
    fn test_for_loop_empty_array() {
        let source = r#"
            let count = 0
            for i in [] {
                count = count + 1
            }
            count
        "#;
        assert_number_result(source, 0.0);
    }

    #[test]
    fn test_for_loop_string_array() {
        // Проверяем, что цикл работает со строками
        let source = r#"
            let count = 0
            for item in ["hello", "world"] {
                count = count + 1
            }
            count
        "#;
        assert_number_result(source, 2.0);
    }

    #[test]
    fn test_for_loop_mixed_array() {
        let source = r#"
            let count = 0
            for item in [1, "hello", true] {
                count = count + 1
            }
            count
        "#;
        assert_number_result(source, 3.0);
    }

    #[test]
    fn test_for_loop_variable_array() {
        let source = r#"
            let arr = [1, 2, 3]
            let sum = 0
            for i in arr {
                sum = sum + i
            }
            sum
        "#;
        // sum = 1 + 2 + 3 = 6
        assert_number_result(source, 6.0);
    }

    #[test]
    fn test_for_loop_complex() {
        let source = r#"
            let result = 0
            for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] {
                if i > 5 {
                    result = result + i
                }
            }
            result
        "#;
        // result = 6 + 7 + 8 + 9 + 10 = 40
        assert_number_result(source, 40.0);
    }

    #[test]
    fn test_for_loop_decrement() {
        let source = r#"
            let sum = 0
            for i in [5, 4, 3, 2, 1] {
                sum = sum + i
            }
            sum
        "#;
        // sum = 5 + 4 + 3 + 2 + 1 = 15
        assert_number_result(source, 15.0);
    }

    #[test]
    fn test_for_loop_multiple_variables() {
        let source = r#"
            let sum = 0
            for i in [1, 2, 3] {
                let j = i * 2
                sum = sum + j
            }
            sum
        "#;
        // sum = 2 + 4 + 6 = 12
        assert_number_result(source, 12.0);
    }

    // ========== Дополнительные тесты для операторов сравнения ==========

    #[test]
    fn test_comparison_operators_edge_cases() {
        // Тесты для граничных случаев операторов сравнения
        assert_bool_result("0 >= 0", true);
        assert_bool_result("0 <= 0", true);
        assert_bool_result("-5 >= -10", true);
        assert_bool_result("-5 <= -10", false);
        assert_bool_result("3.14 > 3.13", true);
        assert_bool_result("3.14 < 3.15", true);
    }

    #[test]
    fn test_string_comparison_operators() {
        // Сравнение строк
        let source = r#"
            if "abc" < "def" {
                1
            } else {
                0
            }
        "#;
        assert_number_result(source, 1.0);
        
        let source = r#"
            if "abc" > "def" {
                1
            } else {
                0
            }
        "#;
        assert_number_result(source, 0.0);
    }

    #[test]
    fn test_comparison_with_expressions() {
        // Сравнение с выражениями
        let source = r#"
            if (10 + 5) >= (20 - 5) {
                1
            } else {
                0
            }
        "#;
        assert_number_result(source, 1.0);
    }

    // ========== Тесты для унарных операторов ==========

    #[test]
    fn test_unary_operators_complex() {
        // Сложные случаи унарных операторов
        assert_number_result("--10", 10.0);
        assert_number_result("---5", -5.0);
        
        let result = run("!!true");
        match result {
            Ok(Value::Bool(true)) => {}
            Ok(v) => panic!("Expected Bool(true), got {:?}", v),
            Err(e) => panic!("Error: {:?}", e),
        }
    }

    #[test]
    fn test_unary_with_expressions() {
        // Унарные операторы с выражениями
        assert_number_result("-(10 + 5)", -15.0);
        let source = r#"
            if !(10 > 5) {
                1
            } else {
                0
            }
        "#;
        assert_number_result(source, 0.0);
    }

    // ========== Тесты для встроенных функций ==========

    #[test]
    fn test_native_print_function() {
        // Проверяем, что print доступна как функция
        let source = r#"
            let result = 0
            print(42)
            result
        "#;
        // print должен работать без ошибок
        let result = run(source);
        assert!(result.is_ok(), "print should work without errors");
    }

    #[test]
    fn test_native_len_function() {
        // Проверяем функцию len
        let source = r#"
            len("hello")
        "#;
        assert_number_result(source, 5.0);
        
        let source = r#"
            len("")
        "#;
        assert_number_result(source, 0.0);
    }

    // ========== Тесты для сложных сценариев с for и while ==========

    #[test]
    fn test_for_and_while_nested() {
        // Вложенные циклы for и while
        let source = r#"
            let sum = 0
            for i in [1, 2] {
                let j = 1
                while j <= 2 {
                    sum = sum + i * j
                    j = j + 1
                }
            }
            sum
        "#;
        // sum = (1*1 + 1*2) + (2*1 + 2*2) = 3 + 6 = 9
        assert_number_result(source, 9.0);
    }

    #[test]
    fn test_for_loop_with_function_call() {
        // Цикл for с вызовом функции
        let source = r#"
            fn double(x) {
                return x * 2
            }
            let sum = 0
            for i in [1, 2, 3] {
                sum = sum + double(i)
            }
            sum
        "#;
        // sum = 2 + 4 + 6 = 12
        assert_number_result(source, 12.0);
    }

    #[test]
    fn test_for_loop_with_recursive_function() {
        // Цикл for с рекурсивной функцией
        let source = r#"
            fn factorial(n) {
                if n <= 1 {
                    return 1
                }
                return n * factorial(n - 1)
            }
            let sum = 0
            for i in [1, 2, 3] {
                sum = sum + factorial(i)
            }
            sum
        "#;
        // sum = 1! + 2! + 3! = 1 + 2 + 6 = 9
        assert_number_result(source, 9.0);
    }

    // ========== Тесты для обработки ошибок в циклах ==========
    // Эти тесты больше не применимы, так как новый синтаксис не использует условия в цикле

    // ========== Тесты для встроенной функции range ==========

    #[test]
    fn test_range_basic() {
        // range(1, 5) должен вернуть [1, 2, 3, 4]
        let source = r#"
            let arr = range(1, 5)
            len(arr)
        "#;
        assert_number_result(source, 4.0);
    }

    #[test]
    fn test_range_in_for_loop() {
        // Использование range в цикле for
        let source = r#"
            let sum = 0
            for i in range(1, 6) {
                sum = sum + i
            }
            sum
        "#;
        // sum = 1 + 2 + 3 + 4 + 5 = 15
        assert_number_result(source, 15.0);
    }

    #[test]
    fn test_range_in_for_loop_complex() {
        // Как в примере complex.dc
        let source = r#"
            let count = 0
            for i in range(1, 11) {
                count = count + 1
            }
            count
        "#;
        // count = 10 (от 1 до 10 включительно)
        assert_number_result(source, 10.0);
    }

    #[test]
    fn test_range_in_function() {
        // Использование range внутри функции
        let source = r#"
            fn sum_range(start, end) {
                let sum = 0
                for i in range(start, end) {
                    sum = sum + i
                }
                return sum
            }
            sum_range(1, 6)
        "#;
        // sum = 1 + 2 + 3 + 4 + 5 = 15
        assert_number_result(source, 15.0);
    }

    #[test]
    fn test_range_without_loop() {
        // Использование range без цикла - просто получение массива и проверка длины
        let source = r#"
            let arr = range(0, 3)
            len(arr)
        "#;
        // Длина массива [0, 1, 2] = 3
        assert_number_result(source, 3.0);
    }

    #[test]
    fn test_range_single_element() {
        // range(5, 6) должен вернуть [5]
        let source = r#"
            let arr = range(5, 6)
            len(arr)
        "#;
        assert_number_result(source, 1.0);
    }

    #[test]
    fn test_range_empty() {
        // range(5, 5) должен вернуть пустой массив
        let source = r#"
            let arr = range(5, 5)
            len(arr)
        "#;
        assert_number_result(source, 0.0);
    }

    #[test]
    fn test_range_zero_start() {
        // range(0, 3) должен вернуть [0, 1, 2]
        let source = r#"
            let arr = range(0, 3)
            let sum = 0
            for i in arr {
                sum = sum + i
            }
            sum
        "#;
        // 0 + 1 + 2 = 3
        assert_number_result(source, 3.0);
    }

    #[test]
    fn test_range_negative_numbers() {
        // range(-2, 2) должен вернуть [-2, -1, 0, 1]
        let source = r#"
            let arr = range(-2, 2)
            len(arr)
        "#;
        assert_number_result(source, 4.0);
    }

    #[test]
    fn test_range_nested_loops() {
        // Вложенные циклы с range
        let source = r#"
            let sum = 0
            for i in range(1, 4) {
                for j in range(1, 3) {
                    sum = sum + i * j
                }
            }
            sum
        "#;
        // i=1: j=1 (1*1=1), j=2 (1*2=2) -> 3
        // i=2: j=1 (2*1=2), j=2 (2*2=4) -> 6
        // i=3: j=1 (3*1=3), j=2 (3*2=6) -> 9
        // sum = 3 + 6 + 9 = 18
        assert_number_result(source, 18.0);
    }

    #[test]
    fn test_range_with_conditional() {
        // range в цикле с условием
        let source = r#"
            let sum = 0
            for i in range(1, 11) {
                if i > 5 {
                    sum = sum + i
                }
            }
            sum
        "#;
        // sum = 6 + 7 + 8 + 9 + 10 = 40
        assert_number_result(source, 40.0);
    }

    #[test]
    fn test_range_large_range() {
        // Большой диапазон
        let source = r#"
            let sum = 0
            for i in range(1, 101) {
                sum = sum + i
            }
            sum
        "#;
        // sum = 1 + 2 + ... + 100 = 5050
        assert_number_result(source, 5050.0);
    }

    #[test]
    fn test_range_as_variable() {
        // Сохранение результата range в переменную
        let source = r#"
            let r = range(1, 4)
            let sum = 0
            for i in r {
                sum = sum + i
            }
            sum
        "#;
        // sum = 1 + 2 + 3 = 6
        assert_number_result(source, 6.0);
    }

    #[test]
    fn test_range_in_recursive_function() {
        // Использование range в рекурсивной функции
        let source = r#"
            fn process_range(n) {
                if n <= 0 {
                    return 0
                }
                let sum = 0
                for i in range(1, n + 1) {
                    sum = sum + i
                }
                return sum + process_range(n - 1)
            }
            process_range(3)
        "#;
        // process_range(3) = (1+2+3) + process_range(2)
        // process_range(2) = (1+2) + process_range(1)
        // process_range(1) = (1) + process_range(0)
        // process_range(0) = 0
        // Итого: 6 + 3 + 1 = 10
        assert_number_result(source, 10.0);
    }

    #[test]
    fn test_range_error_wrong_args() {
        // range без аргументов должна вызвать ошибку
        assert_error("range()");
    }

    #[test]
    fn test_range_one_argument() {
        // range(10) должен вернуть [0, 1, 2, ..., 9]
        let source = r#"
            let arr = range(10)
            len(arr)
        "#;
        assert_number_result(source, 10.0);
    }

    #[test]
    fn test_range_one_argument_values() {
        // Проверка значений range(10)
        let source = r#"
            let arr = range(10)
            let sum = 0
            for i in arr {
                sum = sum + i
            }
            sum
        "#;
        // 0 + 1 + 2 + ... + 9 = 45
        assert_number_result(source, 45.0);
    }

    #[test]
    fn test_range_three_arguments() {
        // range(1, 10, 2) должен вернуть [1, 3, 5, 7, 9]
        let source = r#"
            let arr = range(1, 10, 2)
            len(arr)
        "#;
        assert_number_result(source, 5.0);
    }

    #[test]
    fn test_range_three_arguments_values() {
        // Проверка значений range(1, 10, 2)
        let source = r#"
            let arr = range(1, 10, 2)
            let sum = 0
            for i in arr {
                sum = sum + i
            }
            sum
        "#;
        // 1 + 3 + 5 + 7 + 9 = 25
        assert_number_result(source, 25.0);
    }

    #[test]
    fn test_range_negative_step() {
        // range(10, 0, -1) должен вернуть [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
        let source = r#"
            let arr = range(10, 0, -1)
            len(arr)
        "#;
        assert_number_result(source, 10.0);
    }

    #[test]
    fn test_range_negative_step_values() {
        // Проверка значений range(10, 0, -1)
        let source = r#"
            let arr = range(10, 0, -1)
            let sum = 0
            for i in arr {
                sum = sum + i
            }
            sum
        "#;
        // 10 + 9 + 8 + 7 + 6 + 5 + 4 + 3 + 2 + 1 = 55
        assert_number_result(source, 55.0);
    }

    #[test]
    fn test_range_zero_step() {
        // range с шагом 0 должна вызвать ошибку
        assert_error("range(1, 10, 0)");
    }

    #[test]
    fn test_range_error_non_number() {
        // range с нечисловыми аргументами должна вызвать ошибку
        assert_error(r#"range("hello", "world")"#);
    }

    #[test]
    fn test_range_error_four_args() {
        // range с 4 аргументами должна вызвать ошибку
        assert_error("range(1, 2, 3, 4)");
    }

    // ========== Тесты для мемоизации @cache ==========

    #[test]
    fn test_cache_fibonacci() {
        // Тест мемоизации для функции fibonacci
        let source = r#"
        @cache
        fn fibonacci(n) {
            if n <= 1 {
                return n
            }
            return fibonacci(n - 1) + fibonacci(n - 2)
        }
        fibonacci(10)
        "#;
        let result = run(source);
        match result {
            Ok(Value::Number(n)) => {
                assert_eq!(n, 55.0, "fibonacci(10) should be 55");
            }
            Ok(v) => panic!("Expected Number(55), got {:?}", v),
            Err(e) => panic!("Error: {:?}", e),
        }
    }

    #[test]
    fn test_cache_simple_function() {
        // Простой тест мемоизации
        let source = r#"
        @cache
        fn add(a, b) {
            return a + b
        }
        let result1 = add(1, 2)
        let result2 = add(1, 2)
        result2
        "#;
        let result = run(source);
        match result {
            Ok(Value::Number(n)) => {
                assert_eq!(n, 3.0, "add(1, 2) should be 3");
            }
            Ok(v) => panic!("Expected Number(3), got {:?}", v),
            Err(e) => panic!("Error: {:?}", e),
        }
    }

    #[test]
    fn test_cache_with_strings() {
        // Тест мемоизации с строковыми аргументами
        let source = r#"
        @cache
        fn concat(a, b) {
            return a + b
        }
        concat("hello", "world")
        "#;
        let result = run(source);
        match result {
            Ok(Value::String(s)) => {
                assert_eq!(s, "helloworld", "concat('hello', 'world') should be 'helloworld'");
            }
            Ok(v) => panic!("Expected String('helloworld'), got {:?}", v),
            Err(e) => panic!("Error: {:?}", e),
        }
    }

    #[test]
    fn test_cache_without_annotation() {
        // Функция без @cache должна работать нормально
        let source = r#"
        fn add(a, b) {
            return a + b
        }
        add(1, 2)
        "#;
        let result = run(source);
        match result {
            Ok(Value::Number(n)) => {
                assert_eq!(n, 3.0, "add(1, 2) should be 3");
            }
            Ok(v) => panic!("Expected Number(3), got {:?}", v),
            Err(e) => panic!("Error: {:?}", e),
        }
    }

    #[test]
    fn test_cache_recursive() {
        // Тест рекурсивной функции с кэшем
        let source = r#"
        @cache
        fn factorial(n) {
            if n <= 1 {
                return 1
            }
            return n * factorial(n - 1)
        }
        factorial(5)
        "#;
        let result = run(source);
        match result {
            Ok(Value::Number(n)) => {
                assert_eq!(n, 120.0, "factorial(5) should be 120");
            }
            Ok(v) => panic!("Expected Number(120), got {:?}", v),
            Err(e) => panic!("Error: {:?}", e),
        }
    }

    // ========== Тесты для независимости массивов после присваивания ==========

    #[test]
    fn test_array_independence_separate_empty_arrays() {
        // Два отдельных пустых массива должны быть независимыми
        let source = r#"
        let list = []
        let list1 = []
        push(list, 1)
        push(list, 2)
        len(list1)
        "#;
        let result = run(source);
        match result {
            Ok(Value::Number(n)) => {
                assert_eq!(n, 0.0, "list1 should remain empty (length 0)");
            }
            Ok(v) => panic!("Expected Number(0), got {:?}", v),
            Err(e) => panic!("Error: {:?}", e),
        }
    }

    #[test]
    fn test_array_independence_reverse_case() {
        // Проверяем независимость в обратном направлении
        let source = r#"
        let list = [1, 2]
        let list1 = list.clone()
        push(list1, 3)
        len(list)
        "#;
        let result = run(source);
        match result {
            Ok(Value::Number(n)) => {
                assert_eq!(n, 2.0, "list should have length 2 (not affected by push to list1)");
            }
            Ok(v) => panic!("Expected Number(2), got {:?}", v),
            Err(e) => panic!("Error: {:?}", e),
        }
    }

    #[test]
    fn test_array_independence_with_modifications() {
        // Проверяем, что изменения в одном массиве влияют на другой
        let source = r#"
        let list = [1, 2, 3]
        let list1 = list
        push(list, 4)
        push(list1, 5)
        len(list) + len(list1)
        "#;
        let result = run(source);
        match result {
            Ok(Value::Number(n)) => {
                // list должен иметь длину 5 (1,2,3,4, 5), list1 должен иметь длину 5 (1,2,3,4,5)
                assert_eq!(n, 10.0, "Sum of lengths should be 10 (5 + 5)");
            }
            Ok(v) => panic!("Expected Number(10), got {:?}", v),
            Err(e) => panic!("Error: {:?}", e),
        }
    }

    #[test]
    fn test_array_independence_nested_arrays() {
        // Проверяем зависимость для вложенных массивов
        let source = r#"
        let list = [[1, 2], [3, 4]]
        let list1 = list
        push(list[0], 5)
        len(list1[0])
        "#;
        let result = run(source);
        match result {
            Ok(Value::Number(n)) => {
                assert_eq!(n, 3.0, "list1[0] should have length 3");
            }
            Ok(v) => panic!("Expected Number(3), got {:?}", v),
            Err(e) => panic!("Error: {:?}", e),
        }
    }

    #[test]
    fn test_array_independence_nested_arrays_clone() {
        // Проверяем зависимость для вложенных массивов
        let source = r#"
        let list = [[1, 2], [3, 4]]
        let list1 = list.clone()
        push(list[0], 5)
        len(list1[0])
        "#;
        let result = run(source);
        match result {
            Ok(Value::Number(n)) => {
                assert_eq!(n, 2.0, "list1[0] should have length 2 (not affected by push to list[0])");
            }
            Ok(v) => panic!("Expected Number(2), got {:?}", v),
            Err(e) => panic!("Error: {:?}", e),
        }
    }

    // ========== Тесты для оператора возведения в степень (**) ==========

    #[test]
    fn test_exponentiation_basic() {
        assert_number_result("2 ** 3", 8.0);
    }

    #[test]
    fn test_exponentiation_zero_power() {
        assert_number_result("5 ** 0", 1.0);
    }

    #[test]
    fn test_exponentiation_one_power() {
        assert_number_result("5 ** 1", 5.0);
    }

    #[test]
    fn test_exponentiation_negative_base() {
        assert_number_result("(-2) ** 3", -8.0);
    }

    #[test]
    fn test_exponentiation_fractional_power() {
        assert_number_result("4 ** 0.5", 2.0);
    }

    #[test]
    fn test_exponentiation_right_associative() {
        // 2 ** 3 ** 2 = 2 ** (3 ** 2) = 2 ** 9 = 512
        assert_number_result("2 ** 3 ** 2", 512.0);
    }

    #[test]
    fn test_exponentiation_precedence() {
        // 2 * 3 ** 2 = 2 * 9 = 18
        assert_number_result("2 * 3 ** 2", 18.0);
        // 2 ** 3 * 2 = 8 * 2 = 16
        assert_number_result("2 ** 3 * 2", 16.0);
    }

    #[test]
    fn test_exponentiation_with_parentheses() {
        // (2 ** 3) ** 2 = 8 ** 2 = 64
        assert_number_result("(2 ** 3) ** 2", 64.0);
    }

    #[test]
    fn test_exponentiation_complex_expression() {
        // 2 ** 3 + 1 = 8 + 1 = 9
        assert_number_result("2 ** 3 + 1", 9.0);
        // 1 + 2 ** 3 = 1 + 8 = 9
        assert_number_result("1 + 2 ** 3", 9.0);
    }

    #[test]
    fn test_exponentiation_type_error() {
        assert_error("2 ** \"3\"");
        assert_error("\"2\" ** 3");
    }

    // ========== Тесты для оператора **= ==========

    #[test]
    fn test_exponentiation_assignment_local() {
        let source = r#"
        x = 2
        x **= 3
        x
        "#;
        assert_number_result(source, 8.0);
    }

    #[test]
    fn test_exponentiation_assignment_global() {
        let source = r#"
        x = 2
        x **= 3
        x
        "#;
        assert_number_result(source, 8.0);
    }

    #[test]
    fn test_exponentiation_assignment_returns_value() {
        let source = r#"
        x = 2
        y = x **= 3
        y
        "#;
        assert_number_result(source, 8.0);
    }

    #[test]
    fn test_exponentiation_assignment_multiple() {
        let source = r#"
        x = 2
        x **= 2
        x **= 2
        x
        "#;
        assert_number_result(source, 16.0);
    }

    #[test]
    fn test_exponentiation_assignment_in_function() {
        let source = r#"
        fn test() {
            x = 3
            x **= 2
            return x
        }
        test()
        "#;
        assert_number_result(source, 9.0);
    }

    #[test]
    fn test_exponentiation_assignment_complex() {
        let source = r#"
        let x = 2
        x **= 3
        x **= 0.5
        x
        "#;
        assert_number_result(source, 2.8284271247461903); // sqrt(8) ≈ 2.828
    }
}

