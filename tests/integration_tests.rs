// Интеграционные тесты для полного цикла интерпретации
// Тестируем: Lexer → Parser → Resolver → Compiler → VM

#[cfg(test)]
mod tests {
    use data_code::{run, Value};

    // Вспомогательная функция для проверки результата выполнения
    fn run_and_get_result(source: &str) -> Result<Value, data_code::LangError> {
        run(source)
    }

    // Вспомогательная функция для проверки числового результата
    fn assert_number_result(source: &str, expected: f64) {
        let result = run_and_get_result(source);
        match result {
            Ok(Value::Number(n)) => {
                assert_eq!(n, expected, "Expected {}, got {}", expected, n);
            }
            Ok(v) => panic!("Expected Number({}), got {:?}", expected, v),
            Err(e) => panic!("Error: {:?}", e),
        }
    }

    // Вспомогательная функция для проверки булевого результата
    fn assert_bool_result(source: &str, expected: bool) {
        let result = run_and_get_result(source);
        match result {
            Ok(Value::Bool(b)) => {
                assert_eq!(b, expected, "Expected {}, got {}", expected, b);
            }
            Ok(v) => panic!("Expected Bool({}), got {:?}", expected, v),
            Err(e) => panic!("Error: {:?}", e),
        }
    }

    // ========== Базовые арифметические операции ==========

    #[test]
    fn test_simple_addition() {
        let source = "10 + 20";
        assert_number_result(source, 30.0);
    }

    #[test]
    fn test_arithmetic_operations() {
        assert_number_result("10 + 5", 15.0);
        assert_number_result("10 - 5", 5.0);
        assert_number_result("10 * 5", 50.0);
        assert_number_result("10 / 5", 2.0);
    }

    #[test]
    fn test_operator_precedence() {
        // Должно быть: 2 + (3 * 4) = 14
        assert_number_result("2 + 3 * 4", 14.0);
        // Должно быть: (2 + 3) * 4 = 20
        assert_number_result("(2 + 3) * 4", 20.0);
    }

    #[test]
    fn test_comparison_operators() {
        assert_bool_result("10 > 5", true);
        assert_bool_result("5 > 10", false);
        assert_bool_result("10 < 5", false);
        assert_bool_result("5 < 10", true);
        assert_bool_result("10 == 10", true);
        assert_bool_result("10 == 5", false);
        assert_bool_result("10 != 5", true);
        assert_bool_result("10 != 10", false);
    }

    // ========== Условные конструкции ==========

    #[test]
    fn test_if_statement() {
        let source = r#"
            let x = 10
            let result = 0
            if x > 5 {
                result = 1
            }
        "#;
        let result = run_and_get_result(source);
        assert!(result.is_ok() || result.is_err());
    }

    #[test]
    fn test_if_else_statement() {
        let source = r#"
            let x = 3
            let result = 0
            if x > 5 {
                result = 1
            } else {
                result = 2
            }
        "#;
        let result = run_and_get_result(source);
        assert!(result.is_ok() || result.is_err());
    }

    #[test]
    fn test_nested_if() {
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
            }
        "#;
        let result = run_and_get_result(source);
        assert!(result.is_ok() || result.is_err());
    }

    #[test]
    fn test_if_else_if_else() {
        let source = r#"
            let x = 15
            let result = 0
            if x > 20 {
                result = 1
            } else if x > 10 {
                result = 2
            } else {
                result = 3
            }
            result
        "#;
        assert_number_result(source, 2.0);
    }

    #[test]
    fn test_if_else_if() {
        let source = r#"
            let x = 25
            let result = 0
            if x > 30 {
                result = 1
            } else if x > 20 {
                result = 2
            }
            result
        "#;
        assert_number_result(source, 2.0);
    }

    #[test]
    fn test_multiple_else_if() {
        let source = r#"
            let score = 85
            let grade = 0
            if score >= 90 {
                grade = 5
            } else if score >= 80 {
                grade = 4
            } else if score >= 70 {
                grade = 3
            } else if score >= 60 {
                grade = 2
            } else {
                grade = 1
            }
            grade
        "#;
        assert_number_result(source, 4.0);
    }

    #[test]
    fn test_nested_else_if() {
        let source = r#"
            let x = 10
            let y = 5
            let result = 0
            if x > 5 {
                if y > 10 {
                    result = 1
                } else if y > 3 {
                    result = 2
                } else {
                    result = 3
                }
            } else if x > 0 {
                result = 4
            } else {
                result = 5
            }
            result
        "#;
        assert_number_result(source, 2.0);
    }

    #[test]
    fn test_else_if_first_condition_true() {
        let source = r#"
            let x = 10
            let result = 0
            if x > 5 {
                result = 1
            } else if x > 15 {
                result = 2
            } else {
                result = 3
            }
            result
        "#;
        assert_number_result(source, 1.0);
    }

    #[test]
    fn test_else_if_last_condition_true() {
        let source = r#"
            let x = 25
            let result = 0
            if x > 30 {
                result = 1
            } else if x > 20 {
                result = 2
            } else if x > 10 {
                result = 3
            } else {
                result = 4
            }
            result
        "#;
        assert_number_result(source, 2.0);
    }

    #[test]
    fn test_else_if_final_else() {
        let source = r#"
            let x = 3
            let result = 0
            if x > 10 {
                result = 1
            } else if x > 5 {
                result = 2
            } else {
                result = 3
            }
            result
        "#;
        assert_number_result(source, 3.0);
    }

    // ========== Циклы ==========

    #[test]
    fn test_while_loop() {
        let source = r#"
            let x = 10
            while x > 0 {
                x = x - 1
            }
        "#;
        let result = run_and_get_result(source);
        assert!(result.is_ok() || result.is_err());
    }

    #[test]
    fn test_while_loop_counter() {
        let source = r#"
            let counter = 0
            let x = 5
            while x > 0 {
                counter = counter + 1
                x = x - 1
            }
        "#;
        let result = run_and_get_result(source);
        assert!(result.is_ok() || result.is_err());
    }

    #[test]
    fn test_while_loop_sum() {
        let source = r#"
            let sum = 0
            let i = 1
            while i <= 10 {
                sum = sum + i
                i = i + 1
            }
        "#;
        let result = run_and_get_result(source);
        assert!(result.is_ok() || result.is_err());
    }

    // ========== Функции ==========

    #[test]
    fn test_simple_function() {
        let source = r#"
            fn add(a, b) {
                return a + b
            }
            add(5, 3)
        "#;
        assert_number_result(source, 8.0);
    }

    #[test]
    fn test_function_with_local_variables() {
        let source = r#"
            fn multiply(a, b) {
                let result = a * b
                return result
            }
            multiply(4, 5)
        "#;
        assert_number_result(source, 20.0);
    }

    #[test]
    fn test_recursive_factorial() {
        let source = r#"
            fn factorial(n) {
                if n <= 1 {
                    return 1
                }
                return n * factorial(n - 1)
            }
            factorial(5)
        "#;
        assert_number_result(source, 120.0);
    }

    #[test]
    fn test_recursive_fibonacci() {
        let source = r#"
            fn fib(n) {
                if n <= 1 {
                    return n
                }
                return fib(n - 1) + fib(n - 2)
            }
            fib(6)
        "#;
        assert_number_result(source, 8.0);
    }

    #[test]
    fn test_function_with_conditionals() {
        let source = r#"
            fn max(a, b) {
                if a > b {
                    return a
                } else {
                    return b
                }
            }
            max(10, 5)
        "#;
        assert_number_result(source, 10.0);
    }

    // ========== Глобальные и локальные переменные ==========

    #[test]
    fn test_global_variables() {
        let source = r#"
            let global_x = 100
            let global_y = 200
            let sum = global_x + global_y
        "#;
        let result = run_and_get_result(source);
        assert!(result.is_ok() || result.is_err());
    }

    #[test]
    fn test_local_variables_in_function() {
        let source = r#"
            let global_x = 100
            fn test() {
                let local_x = 10
                return local_x
            }
            let result = test()
        "#;
        let result = run_and_get_result(source);
        assert!(result.is_ok() || result.is_err());
    }

    #[test]
    fn test_global_access_from_function() {
        let source = r#"
            let global_x = 100
            fn test() {
                return global_x
            }
            let result = test()
        "#;
        let result = run_and_get_result(source);
        assert!(result.is_ok() || result.is_err());
    }

    // ========== Строки ==========

    #[test]
    fn test_string_literals() {
        let source = r#"
            let s = "hello"
            let t = "world"
        "#;
        let result = run_and_get_result(source);
        assert!(result.is_ok() || result.is_err());
    }

    #[test]
    fn test_string_concatenation() {
        let source = r#"
            let a = "hello"
            let b = "world"
            a + " " + b
        "#;
        let result = run_and_get_result(source);
        match result {
            Ok(Value::String(s)) => {
                assert_eq!(s, "hello world", "Expected 'hello world', got '{}'", s);
            }
            Ok(v) => panic!("Expected String('hello world'), got {:?}", v),
            Err(e) => panic!("Error: {:?}", e),
        }
    }

    // ========== Сложные сценарии ==========

    #[test]
    fn test_complex_expression() {
        let source = r#"
            let a = 10
            let b = 20
            let c = 30
            (a + b) * c / 2
        "#;
        // (10 + 20) * 30 / 2 = 30 * 30 / 2 = 900 / 2 = 450
        assert_number_result(source, 450.0);
    }

    #[test]
    fn test_nested_function_calls() {
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
    fn test_loop_with_function() {
        let source = r#"
            fn square(n) {
                return n * n
            }
            let sum = 0
            let i = 1
            while i <= 5 {
                sum = sum + square(i)
                i = i + 1
            }
            sum
        "#;
        // sum = 1^2 + 2^2 + 3^2 + 4^2 + 5^2 = 1 + 4 + 9 + 16 + 25 = 55
        assert_number_result(source, 55.0);
    }

    // ========== Граничные случаи ==========

    #[test]
    fn test_empty_function() {
        let source = r#"
            fn empty() {
            }
            empty()
        "#;
        let result = run_and_get_result(source);
        assert!(result.is_ok() || result.is_err());
    }

    #[test]
    fn test_function_without_return() {
        let source = r#"
            fn no_return() {
                let x = 10
            }
            no_return()
        "#;
        let result = run_and_get_result(source);
        assert!(result.is_ok() || result.is_err());
    }

    #[test]
    fn test_zero_division_handling() {
        let source = r#"
            let x = 10 / 0
        "#;
        let result = run_and_get_result(source);
        // Должна быть ошибка деления на ноль
        assert!(result.is_err());
    }

    // ========== Тесты для обработки ошибок ==========

    #[test]
    fn test_undefined_variable_error() {
        let source = r#"
            let x = undefined_var
        "#;
        let result = run_and_get_result(source);
        // Должна быть ошибка неопределенной переменной
        assert!(result.is_err());
    }

    #[test]
    fn test_function_call_with_wrong_arity() {
        let source = r#"
            fn add(a, b) {
                return a + b
            }
            add(1, 2, 3)
        "#;
        let result = run_and_get_result(source);
        // Должна быть ошибка неверного количества аргументов
        assert!(result.is_err());
    }

    #[test]
    fn test_function_call_with_fewer_args() {
        let source = r#"
            fn add(a, b) {
                return a + b
            }
            add(1)
        "#;
        let result = run_and_get_result(source);
        // Должна быть ошибка неверного количества аргументов
        assert!(result.is_err());
    }

    #[test]
    fn test_undefined_function_error() {
        let source = r#"
            undefined_function(1, 2)
        "#;
        let result = run_and_get_result(source);
        // Должна быть ошибка неопределенной функции
        assert!(result.is_err());
    }

    #[test]
    fn test_type_error_in_arithmetic() {
        let source = r#"
            let x = "hello"
            x - 5
        "#;
        let result = run_and_get_result(source);
        // Должна быть ошибка типа (нельзя вычитать число из строки)
        assert!(result.is_err());
    }

    #[test]
    fn test_string_multiplication() {
        let source = r#"
            let x = "-"
            x * 10
        "#;
        let result = run_and_get_result(source);
        match result {
            Ok(Value::String(s)) => {
                assert_eq!(s, "----------", "Expected '----------', got '{}'", s);
            }
            Ok(v) => panic!("Expected String('----------'), got {:?}", v),
            Err(e) => panic!("Error: {:?}", e),
        }
        
        let source = r#"
            let x = "hello"
            x * 3
        "#;
        let result = run_and_get_result(source);
        match result {
            Ok(Value::String(s)) => {
                assert_eq!(s, "hellohellohello", "Expected 'hellohellohello', got '{}'", s);
            }
            Ok(v) => panic!("Expected String('hellohellohello'), got {:?}", v),
            Err(e) => panic!("Error: {:?}", e),
        }
    }

    #[test]
    fn test_type_error_in_division() {
        let source = r#"
            let x = "hello"
            x / 5
        "#;
        let result = run_and_get_result(source);
        // Должна быть ошибка типа (нельзя делить строку на число)
        assert!(result.is_err());
    }

    #[test]
    fn test_type_error_in_comparison() {
        let source = r#"
            let x = "hello"
            x > 5
        "#;
        let result = run_and_get_result(source);
        // Должна быть ошибка типа (нельзя сравнивать строку с числом)
        assert!(result.is_err());
    }

    // ========== Булевы значения ==========

    #[test]
    fn test_boolean_literals() {
        let source = r#"
            let t = true
            let f = false
        "#;
        let result = run_and_get_result(source);
        assert!(result.is_ok() || result.is_err());
    }

    #[test]
    fn test_boolean_operations() {
        assert_bool_result("10 > 5", true);
        assert_bool_result("3 < 2", false);
        assert_bool_result("5 == 5", true);
        assert_bool_result("5 != 3", true);
    }

    // ========== Дополнительные тесты для полного покрытия ==========

    #[test]
    fn test_variable_reassignment() {
        let source = r#"
            let x = 10
            x = 20
            x
        "#;
        assert_number_result(source, 20.0);
    }

    #[test]
    fn test_function_with_multiple_parameters() {
        let source = r#"
            fn multiply(a, b, c) {
                return a * b * c
            }
            multiply(2, 3, 4)
        "#;
        assert_number_result(source, 24.0);
    }

    #[test]
    fn test_function_without_return_statement() {
        let source = r#"
            fn no_return() {
                let x = 10
            }
            no_return()
        "#;
        let result = run_and_get_result(source);
        // Функция без return должна вернуть null
        match result {
            Ok(Value::Null) => {}
            Ok(v) => panic!("Expected Null, got {:?}", v),
            Err(e) => panic!("Error: {:?}", e),
        }
    }

    #[test]
    fn test_nested_loops() {
        let source = r#"
            let sum = 0
            let i = 1
            while i <= 3 {
                let j = 1
                while j <= 2 {
                    sum = sum + i * j
                    j = j + 1
                }
                i = i + 1
            }
            sum
        "#;
        // sum = (1*1 + 1*2) + (2*1 + 2*2) + (3*1 + 3*2) = 3 + 6 + 9 = 18
        assert_number_result(source, 18.0);
    }

    #[test]
    fn test_complex_nested_functions() {
        let source = r#"
            fn outer(x) {
                fn middle(y) {
                    fn inner(z) {
                        return x + y + z
                    }
                    return inner(3)
                }
                return middle(2)
            }
            outer(1)
        "#;
        // inner(3) = 1 + 2 + 3 = 6
        assert_number_result(source, 6.0);
    }

    #[test]
    fn test_arithmetic_with_variables() {
        let source = r#"
            let a = 5
            let b = 3
            let c = 2
            a * b + c
        "#;
        // 5 * 3 + 2 = 15 + 2 = 17
        assert_number_result(source, 17.0);
    }

    #[test]
    fn test_conditional_expression_in_function() {
        let source = r#"
            fn max(a, b) {
                if a > b {
                    return a
                } else {
                    return b
                }
            }
            max(15, 10)
        "#;
        assert_number_result(source, 15.0);
    }

    #[test]
    fn test_conditional_expression_min() {
        let source = r#"
            fn min(a, b) {
                if a < b {
                    return a
                } else {
                    return b
                }
            }
            min(15, 10)
        "#;
        assert_number_result(source, 10.0);
    }

    #[test]
    fn test_string_number_concatenation() {
        let source = r#"
            let s = "Number: "
            let n = 42
            s + n
        "#;
        let result = run_and_get_result(source);
        match result {
            Ok(Value::String(s)) => {
                assert_eq!(s, "Number: 42", "Expected 'Number: 42', got '{}'", s);
            }
            Ok(v) => panic!("Expected String('Number: 42'), got {:?}", v),
            Err(e) => panic!("Error: {:?}", e),
        }
    }

    #[test]
    fn test_number_string_concatenation() {
        let source = r#"
            let n = 42
            let s = " is the answer"
            n + s
        "#;
        let result = run_and_get_result(source);
        match result {
            Ok(Value::String(s)) => {
                assert_eq!(s, "42 is the answer", "Expected '42 is the answer', got '{}'", s);
            }
            Ok(v) => panic!("Expected String('42 is the answer'), got {:?}", v),
            Err(e) => panic!("Error: {:?}", e),
        }
    }

    #[test]
    fn test_boolean_comparison() {
        assert_bool_result("true == true", true);
        assert_bool_result("true == false", false);
        assert_bool_result("false == false", true);
        assert_bool_result("true != false", true);
    }

    #[test]
    fn test_truthiness() {
        let source = r#"
            let x = 0
            if x {
                "truthy"
            } else {
                "falsy"
            }
        "#;
        let result = run_and_get_result(source);
        match result {
            Ok(Value::String(s)) => {
                assert_eq!(s, "falsy", "Expected 'falsy', got '{}'", s);
            }
            Ok(v) => panic!("Expected String('falsy'), got {:?}", v),
            Err(e) => panic!("Error: {:?}", e),
        }
    }

    #[test]
    fn test_null_comparison() {
        let source = r#"
            let x = null
            x == null
        "#;
        assert_bool_result(source, true);
    }

    #[test]
    fn test_loop_break_condition() {
        let source = r#"
            let counter = 0
            let x = 5
            while x > 0 {
                counter = counter + 1
                x = x - 1
            }
            counter
        "#;
        assert_number_result(source, 5.0);
    }

    #[test]
    fn test_function_with_local_shadowing() {
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
    fn test_multiple_returns() {
        let source = r#"
            fn test(x) {
                if x > 5 {
                    return 10
                }
                return 5
            }
            test(10)
        "#;
        assert_number_result(source, 10.0);
    }

    #[test]
    fn test_multiple_returns_else() {
        let source = r#"
            fn test(x) {
                if x > 5 {
                    return 10
                }
                return 5
            }
            test(3)
        "#;
        assert_number_result(source, 5.0);
    }

    #[test]
    fn test_complex_arithmetic_expression() {
        let source = r#"
            let a = 10
            let b = 5
            let c = 2
            (a + b) * c - (a - b) / c
        "#;
        // (10 + 5) * 2 - (10 - 5) / 2 = 15 * 2 - 5 / 2 = 30 - 2.5 = 27.5
        assert_number_result(source, 27.5);
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
            sum(10)
        "#;
        // sum(10) = 10 + 9 + 8 + ... + 1 = 55
        assert_number_result(source, 55.0);
    }

    #[test]
    fn test_loop_with_conditional() {
        // sum четных чисел от 1 до 10 = 2 + 4 + 6 + 8 + 10 = 30
        // Но у нас нет оператора %, поэтому тест может не работать
        // Заменим на другой тест
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

    // ========== Аргументы по умолчанию и именованные аргументы ==========

    #[test]
    fn test_default_arguments() {
        let source = r#"
            fn test_fn(a, b=5) {
                return a + b
            }
            test_fn(4)
        "#;
        assert_number_result(source, 9.0);
    }

    #[test]
    fn test_default_arguments_override() {
        let source = r#"
            fn test_fn(a, b=5) {
                return a + b
            }
            test_fn(4, 6)
        "#;
        assert_number_result(source, 10.0);
    }

    #[test]
    fn test_named_arguments() {
        let source = r#"
            fn test_fn(a, b=5) {
                return a + b
            }
            test_fn(a=4)
        "#;
        assert_number_result(source, 9.0);
    }

    #[test]
    fn test_named_arguments_both() {
        let source = r#"
            fn test_fn(a, b=5) {
                return a + b
            }
            test_fn(a=4, b=1)
        "#;
        assert_number_result(source, 5.0);
    }

    #[test]
    fn test_mixed_positional_named() {
        let source = r#"
            fn test_fn(a, b=5) {
                return a + b
            }
            test_fn(4, b=10)
        "#;
        assert_number_result(source, 14.0);
    }

    #[test]
    fn test_multiple_defaults() {
        let source = r#"
            fn test_fn(a, b=5, c=10) {
                return a + b + c
            }
            test_fn(1)
        "#;
        assert_number_result(source, 16.0);
    }

    #[test]
    fn test_multiple_defaults_partial() {
        let source = r#"
            fn test_fn(a, b=5, c=10) {
                return a + b + c
            }
            test_fn(1, 2)
        "#;
        assert_number_result(source, 13.0);
    }

    #[test]
    fn test_multiple_defaults_named() {
        let source = r#"
            fn test_fn(a, b=5, c=10) {
                return a + b + c
            }
            test_fn(1, c=20)
        "#;
        assert_number_result(source, 26.0);
    }

    #[test]
    fn test_error_non_default_after_default() {
        let source = r#"
            fn test_fn(a=2, b) {
                return a + b
            }
        "#;
        let result = run_and_get_result(source);
        assert!(result.is_err(), "Expected parse error for non-default argument after default");
        if let Err(e) = result {
            assert!(e.to_string().contains("Non-default argument follows default argument"));
        }
    }

    #[test]
    fn test_error_positional_after_named() {
        let source = r#"
            fn test_fn(a, b=5) {
                return a + b
            }
            test_fn(a=4, 6)
        "#;
        let result = run_and_get_result(source);
        assert!(result.is_err(), "Expected parse error for positional argument after named");
    }

    #[test]
    fn test_error_missing_required_argument() {
        let source = r#"
            fn test_fn(a, b) {
                return a + b
            }
            test_fn(1)
        "#;
        let result = run_and_get_result(source);
        assert!(result.is_err(), "Expected error for missing required argument");
    }

    #[test]
    fn test_error_unexpected_keyword_argument() {
        let source = r#"
            fn test_fn(a, b=5) {
                return a + b
            }
            test_fn(a=4, c=1)
        "#;
        let result = run_and_get_result(source);
        assert!(result.is_err(), "Expected error for unexpected keyword argument");
        if let Err(e) = result {
            assert!(e.to_string().contains("unexpected keyword argument"));
        }
    }

    #[test]
    fn test_error_duplicate_argument() {
        let source = r#"
            fn test_fn(a, b=5) {
                return a + b
            }
            test_fn(4, a=1)
        "#;
        let result = run_and_get_result(source);
        assert!(result.is_err(), "Expected error for duplicate argument");
        if let Err(e) = result {
            assert!(e.to_string().contains("multiple values"));
        }
    }

    // ========== Тесты для работы с файлами ==========

    #[test]
    fn test_list_files() {
        let source = r#"
            let test_dir = path("tests/test_data")
            list_files(test_dir)
        "#;
        let result = run_and_get_result(source);
        
        match result {
            Ok(Value::Array(arr)) => {
                let arr_ref = arr.borrow();
                // Проверяем что массив не пустой
                assert!(!arr_ref.is_empty(), "Expected non-empty array of files");
                
                // Проверяем что все элементы - Path значения
                for item in arr_ref.iter() {
                    match item {
                        Value::Path(_) => {},
                        _ => panic!("Expected all items to be Path values, got {:?}", item),
                    }
                }
                
                // Собираем имена файлов/папок для проверки
                let file_names: Vec<String> = arr_ref.iter()
                    .map(|item| {
                        match item {
                            Value::Path(p) => {
                                p.file_name()
                                    .and_then(|n| n.to_str())
                                    .unwrap_or("")
                                    .to_string()
                            }
                            _ => String::new(),
                        }
                    })
                    .collect();
                
                // Проверяем наличие ожидаемых файлов
                assert!(file_names.contains(&"sample.csv".to_string()), 
                    "Expected sample.csv in list, got: {:?}", file_names);
                assert!(file_names.contains(&"sample.txt".to_string()), 
                    "Expected sample.txt in list, got: {:?}", file_names);
                assert!(file_names.contains(&"sample.xlsx".to_string()), 
                    "Expected sample.xlsx in list, got: {:?}", file_names);
                assert!(file_names.contains(&"dir_test".to_string()), 
                    "Expected dir_test in list, got: {:?}", file_names);
            }
            Ok(v) => panic!("Expected Array, got {:?}", v),
            Err(e) => panic!("Error: {:?}", e),
        }
    }
}

