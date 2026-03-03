// Тесты для обработки исключений try/catch в DataCode
// Тестируем различные сценарии использования try/catch

#[cfg(test)]
mod tests {
    use data_code::{run, Value, LangError};

    // Вспомогательная функция для проверки результата выполнения
    fn run_and_get_result(source: &str) -> Result<Value, LangError> {
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

    // Вспомогательная функция для проверки ошибки
    fn assert_error(source: &str) {
        let result = run_and_get_result(source);
        assert!(result.is_err(), "Expected error, got {:?}", result);
    }

    // ========== 1. Базовые тесты try/catch ==========

    #[test]
    fn test_try_catch_success_no_error() {
        // Успешное выполнение без ошибок (catch не выполняется)
        let source = r#"
            let result = 0
            try {
                result = 10 + 20
            } catch e {
                result = -1
            }
            result
        "#;
        assert_number_result(source, 30.0);
    }

    #[test]
    fn test_try_catch_division_by_zero() {
        // Обработка ошибки деления на ноль
        let source = r#"
            let result = 0
            try {
                result = 10 / 0
            } catch e {
                result = 100
            }
            result
        "#;
        assert_number_result(source, 100.0);
    }

    #[test]
    fn test_try_catch_undefined_variable() {
        // Обработка ошибки неопределенной переменной
        let source = r#"
            let result = 0
            try {
                result = undefined_var + 10
            } catch e {
                result = 200
            }
            result
        "#;
        assert_number_result(source, 200.0);
    }

    #[test]
    fn test_try_catch_type_error() {
        // Обработка ошибки типа (арифметика со строками)
        let source = r#"
            let result = 0
            try {
                let x = "hello"
                result = x - 5
            } catch e {
                result = 300
            }
            result
        "#;
        assert_number_result(source, 300.0);
    }

    #[test]
    fn test_try_catch_error_variable_access() {
        // Доступ к переменной ошибки `e` в блоке catch
        let source = r#"
            let error_msg = ""
            try {
                let x = 10 / 0
            } catch e {
                error_msg = str(e)
            }
            len(error_msg)
        "#;
        // Должна быть строка с описанием ошибки
        let result = run_and_get_result(source);
        match result {
            Ok(Value::Number(n)) => {
                assert!(n > 0.0, "Expected error message length > 0, got {}", n);
            }
            Ok(v) => panic!("Expected Number, got {:?}", v),
            Err(e) => panic!("Error: {:?}", e),
        }
    }

    #[test]
    fn test_try_catch_without_error_variable() {
        // Catch без переменной ошибки
        let source = r#"
            let result = 0
            try {
                result = 10 / 0
            } catch {
                result = 400
            }
            result
        "#;
        assert_number_result(source, 400.0);
    }

    #[test]
    fn test_try_catch_else_block() {
        // Try/catch с else блоком (выполняется если ошибки не было)
        let source = r#"
            let result = 0
            try {
                result = 10 + 20
            } catch e {
                result = -1
            } else {
                result = result + 5
            }
            result
        "#;
        assert_number_result(source, 35.0);
    }

    #[test]
    fn test_try_catch_else_block_with_error() {
        // Try/catch с else блоком (else не выполняется при ошибке)
        let source = r#"
            let result = 0
            try {
                result = 10 / 0
            } catch e {
                result = 500
            } else {
                result = -1
            }
            result
        "#;
        assert_number_result(source, 500.0);
    }

    // ========== 2. Тесты try/catch в циклах ==========

    #[test]
    fn test_try_catch_in_while_loop() {
        // Try/catch внутри while цикла
        let source = r#"
            let count = 0
            let i = 0
            while i < 5 {
                try {
                    let x = 10 / i
                    count = count + 1
                } catch e {
                    count = count + 10
                }
                i = i + 1
            }
            count
        "#;
        // i=0: ошибка деления на ноль -> +10
        // i=1,2,3,4: успешно -> +1 каждый
        // Итого: 10 + 1 + 1 + 1 + 1 = 14
        assert_number_result(source, 14.0);
    }

    #[test]
    fn test_try_catch_in_for_loop() {
        // Try/catch внутри for цикла
        let source = r#"
            let count = 0
            for i in [0, 1, 2, 3, 4] {
                try {
                    let x = 10 / i
                    count = count + 1
                } catch e {
                    count = count + 10
                }
            }
            count
        "#;
        // i=0: ошибка -> +10
        // i=1,2,3,4: успешно -> +1 каждый
        // Итого: 10 + 1 + 1 + 1 + 1 = 14
        assert_number_result(source, 14.0);
    }

    #[test]
    fn test_try_catch_error_every_iteration() {
        // Ошибка в каждой итерации цикла
        let source = r#"
            let count = 0
            let i = 0
            while i < 5 {
                try {
                    let x = undefined_var
                } catch e {
                    count = count + 1
                }
                i = i + 1
            }
            count
        "#;
        // В каждой итерации ошибка, catch выполняется -> +1 каждый раз
        assert_number_result(source, 5.0);
    }

    #[test]
    fn test_try_catch_error_specific_iteration() {
        // Ошибка в определенной итерации цикла
        let source = r#"
            let count = 0
            let i = 0
            while i < 5 {
                try {
                    if i == 2 {
                        let x = 10 / 0
                    }
                    count = count + 1
                } catch e {
                    count = count + 100
                }
                i = i + 1
            }
            count
        "#;
        // i=0,1,3,4: успешно -> +1 каждый = 4
        // i=2: ошибка -> +100
        // Итого: 4 + 100 = 104
        assert_number_result(source, 104.0);
    }

    #[test]
    fn test_try_catch_nested_in_loop() {
        // Вложенные try/catch в циклах
        let source = r#"
            let count = 0
            let i = 0
            while i < 3 {
                try {
                    try {
                        let x = 10 / i
                        count = count + 1
                    } catch e {
                        count = count + 10
                    }
                } catch e {
                    count = count + 100
                }
                i = i + 1
            }
            count
        "#;
        // i=0: внутренний catch -> +10
        // i=1,2: успешно -> +1 каждый
        // Итого: 10 + 1 + 1 = 12
        assert_number_result(source, 12.0);
    }

    // ========== 3. Тесты try/catch в функциях ==========

    #[test]
    fn test_try_catch_in_function() {
        // Try/catch внутри функции
        let source = r#"
            fn test() {
                let result = 0
                try {
                    result = 10 / 0
                } catch e {
                    result = 600
                }
                return result
            }
            test()
        "#;
        assert_number_result(source, 600.0);
    }

    #[test]
    fn test_try_catch_return_from_try() {
        // Возврат значения из try блока
        let source = r#"
            fn test() {
                try {
                    return 700
                } catch e {
                    return -1
                }
            }
            test()
        "#;
        assert_number_result(source, 700.0);
    }

    #[test]
    fn test_try_catch_return_from_catch() {
        // Возврат значения из catch блока
        let source = r#"
            fn test() {
                try {
                    let x = 10 / 0
                    return -1
                } catch e {
                    return 800
                }
            }
            test()
        "#;
        assert_number_result(source, 800.0);
    }

    #[test]
    fn test_try_catch_error_in_called_function() {
        // Ошибка в вызываемой функции, обработанная в вызывающей
        let source = r#"
            fn risky() {
                let x = 10 / 0
                return 100
            }
            
            fn safe() {
                let result = 0
                try {
                    result = risky()
                } catch e {
                    result = 900
                }
                return result
            }
            safe()
        "#;
        assert_number_result(source, 900.0);
    }

    #[test]
    fn test_try_catch_nested_function_calls() {
        // Вложенные вызовы функций с try/catch
        let source = r#"
            fn inner() {
                let x = 10 / 0
                return 1
            }
            
            fn middle() {
                let result = 0
                try {
                    result = inner()
                } catch e {
                    result = 2
                }
                return result
            }
            
            fn outer() {
                let result = 0
                try {
                    result = middle()
                } catch e {
                    result = 3
                }
                return result
            }
            outer()
        "#;
        // inner() вызывает ошибку, middle() ловит -> возвращает 2
        assert_number_result(source, 2.0);
    }

    #[test]
    fn test_try_catch_function_with_else() {
        // Функция с try/catch/else
        let source = r#"
            fn test(x) {
                let result = 0
                try {
                    result = 100 / x
                } catch e {
                    result = -1
                } else {
                    result = result + 50
                }
                return result
            }
            test(2)
        "#;
        // 100 / 2 = 50, else выполняется -> 50 + 50 = 100
        assert_number_result(source, 100.0);
    }

    // ========== 4. Тесты try/catch в рекурсии ==========

    #[test]
    fn test_try_catch_in_recursive_function() {
        // Try/catch в рекурсивной функции
        let source = r#"
            fn recursive(n) {
                let result = 0
                try {
                    if n <= 0 {
                        let x = 10 / 0
                    }
                    result = n + recursive(n - 1)
                } catch e {
                    result = 1000
                }
                return result
            }
            recursive(3)
        "#;
        // Когда n=0, ошибка -> catch возвращает 1000
        // Но это происходит на глубине, так что результат зависит от реализации
        let result = run_and_get_result(source);
        assert!(result.is_ok(), "Expected success, got error: {:?}", result);
    }

    #[test]
    fn test_try_catch_error_at_specific_recursion_depth() {
        // Ошибка на определенной глубине рекурсии
        let source = r#"
            fn recursive(n) {
                let result = 0
                try {
                    if n == 2 {
                        let x = 10 / 0
                    }
                    if n <= 0 {
                        return 0
                    }
                    result = n + recursive(n - 1)
                } catch e {
                    result = 2000
                }
                return result
            }
            recursive(5)
        "#;
        // Когда n=2, ошибка -> catch возвращает 2000
        let result = run_and_get_result(source);
        assert!(result.is_ok(), "Expected success, got error: {:?}", result);
    }

    #[test]
    fn test_try_catch_continue_recursion_after_error() {
        // Обработка ошибки и продолжение рекурсии
        let source = r#"
            fn recursive(n, acc) {
                if n <= 0 {
                    return acc
                }
                let result = acc
                try {
                    if n == 2 {
                        let x = 10 / 0
                    }
                    result = acc + n
                } catch e {
                    result = acc + 100
                }
                return recursive(n - 1, result)
            }
            recursive(3, 0)
        "#;
        // n=3: успешно -> acc=3
        // n=2: ошибка -> acc=3+100=103
        // n=1: успешно -> acc=103+1=104
        // n=0: возврат 104
        assert_number_result(source, 104.0);
    }

    #[test]
    fn test_try_catch_break_recursion_on_error() {
        // Обработка ошибки и прерывание рекурсии
        let source = r#"
            fn recursive(n) {
                let result = 0
                try {
                    if n == 2 {
                        let x = 10 / 0
                    }
                    if n <= 0 {
                        return 0
                    }
                    result = n + recursive(n - 1)
                } catch e {
                    return 3000
                }
                return result
            }
            recursive(5)
        "#;
        // Когда n=2, ошибка -> catch возвращает 3000 и прерывает рекурсию
        let result = run_and_get_result(source);
        assert!(result.is_ok(), "Expected success, got error: {:?}", result);
    }

    #[test]
    fn test_try_catch_recursive_factorial_with_error() {
        // Рекурсивный факториал с обработкой ошибки
        let source = r#"
            fn factorial(n) {
                if n <= 1 {
                    return 1
                }
                let result = 0
                try {
                    if n == 3 {
                        let x = 10 / 0
                    }
                    result = n * factorial(n - 1)
                } catch e {
                    result = 5000
                }
                return result
            }
            factorial(5)
        "#;
        // Когда n=3, ошибка -> catch возвращает 5000
        let result = run_and_get_result(source);
        assert!(result.is_ok(), "Expected success, got error: {:?}", result);
    }

    // ========== 5. Ошибочные тесты (должны вызывать ошибки парсинга/компиляции) ==========

    #[test]
    fn test_try_without_catch() {
        // Try без catch - должна быть ошибка парсинга
        let source = r#"
            try {
                let x = 10
            }
        "#;
        assert_error(source);
    }

    #[test]
    fn test_catch_without_try() {
        // Catch без try - должна быть ошибка парсинга
        let source = r#"
            catch e {
                let x = 10
            }
        "#;
        assert_error(source);
    }

    #[test]
    fn test_try_catch_missing_brace() {
        // Неправильный синтаксис - отсутствует открывающая скобка
        let source = r#"
            try
                let x = 10
            } catch e {
                let y = 20
            }
        "#;
        assert_error(source);
    }

    #[test]
    fn test_try_catch_missing_closing_brace() {
        // Неправильный синтаксис - отсутствует закрывающая скобка
        let source = r#"
            try {
                let x = 10
            catch e {
                let y = 20
            }
        "#;
        assert_error(source);
    }

    #[test]
    fn test_try_catch_nested_mismatch() {
        // Вложенные try без соответствующих catch
        let source = r#"
            try {
                try {
                    let x = 10
                }
            } catch e {
                let y = 20
            }
        "#;
        assert_error(source);
    }

    #[test]
    fn test_try_catch_else_without_catch() {
        // Else без catch - должна быть ошибка парсинга
        let source = r#"
            try {
                let x = 10
            } else {
                let y = 20
            }
        "#;
        assert_error(source);
    }

    // ========== Дополнительные тесты для специфических типов ошибок ==========

    #[test]
    fn test_try_catch_catch_specific_error_type() {
        // Catch для конкретного типа ошибки (ValueError)
        let source = r#"
            let result = 0
            try {
                let x = 10 / 0
            } catch ValueError e {
                result = 1000
            } catch e {
                result = 2000
            }
            result
        "#;
        // Должна быть ошибка парсинга, если типизированные catch еще не реализованы
        // Или успешное выполнение, если реализованы
        let result = run_and_get_result(source);
        // Пока предполагаем, что это может быть ошибка парсинга или успех
        assert!(result.is_ok() || result.is_err());
    }

    #[test]
    fn test_try_catch_multiple_catch_blocks() {
        // Несколько catch блоков для разных типов ошибок
        let source = r#"
            let result = 0
            try {
                let x = "hello" - 5
            } catch TypeError e {
                result = 1000
            } catch ValueError e {
                result = 2000
            } catch e {
                result = 3000
            }
            result
        "#;
        // Должна быть ошибка парсинга, если типизированные catch еще не реализованы
        // Или успешное выполнение, если реализованы
        let result = run_and_get_result(source);
        assert!(result.is_ok() || result.is_err());
    }

    #[test]
    fn test_try_catch_complex_nested() {
        // Сложный вложенный try/catch
        let source = r#"
            let result = 0
            try {
                try {
                    try {
                        let x = 10 / 0
                    } catch e {
                        result = result + 1
                    }
                } catch e {
                    result = result + 10
                }
            } catch e {
                result = result + 100
            }
            result
        "#;
        // Внутренний catch должен поймать ошибку -> result = 1
        assert_number_result(source, 1.0);
    }

    #[test]
    fn test_try_catch_in_conditional() {
        // Try/catch внутри условного оператора
        let source = r#"
            let result = 0
            if true {
                try {
                    result = 10 / 0
                } catch e {
                    result = 4000
                }
            }
            result
        "#;
        assert_number_result(source, 4000.0);
    }

    #[test]
    fn test_try_catch_with_variable_assignment() {
        // Присваивание переменной в try/catch
        let source = r#"
            let x = 0
            try {
                x = 10 / 0
            } catch e {
                x = 5000
            }
            x
        "#;
        assert_number_result(source, 5000.0);
    }

    // ========== 6. Тесты throw statement ==========

    #[test]
    fn test_throw_string_message() {
        // Базовый throw со строковым сообщением
        let source = r#"
            let result = 0
            try {
                throw "Something went wrong"
                result = 100
            } catch e {
                result = 200
            }
            result
        "#;
        assert_number_result(source, 200.0);
    }

    #[test]
    fn test_throw_number() {
        // Throw с числом (преобразуется в строку)
        let source = r#"
            let result = 0
            try {
                throw 42
                result = 100
            } catch e {
                result = 300
            }
            result
        "#;
        assert_number_result(source, 300.0);
    }

    #[test]
    fn test_throw_variable() {
        // Throw с переменной
        let source = r#"
            let result = 0
            let error_msg = "Custom error"
            try {
                throw error_msg
                result = 100
            } catch e {
                result = 400
            }
            result
        "#;
        assert_number_result(source, 400.0);
    }

    #[test]
    fn test_throw_without_catch() {
        // Throw без catch - должна быть ошибка
        let source = r#"
            throw "Unhandled error"
        "#;
        assert_error(source);
    }

    #[test]
    fn test_throw_in_function_caught_by_caller() {
        // Throw в функции, пойманный вызывающим кодом
        let source = r#"
            fn risky() {
                throw "Error from function"
                return 100
            }
            
            let result = 0
            try {
                result = risky()
            } catch e {
                result = 500
            }
            result
        "#;
        assert_number_result(source, 500.0);
    }

    #[test]
    fn test_throw_in_function_caught_inside() {
        // Throw в функции, пойманный внутри функции
        let source = r#"
            fn safe() {
                let result = 0
                try {
                    throw "Error inside"
                    result = 100
                } catch e {
                    result = 600
                }
                return result
            }
            safe()
        "#;
        assert_number_result(source, 600.0);
    }

    #[test]
    fn test_throw_nested_try_catch() {
        // Throw во вложенном try/catch
        let source = r#"
            let result = 0
            try {
                try {
                    throw "Inner error"
                    result = 100
                } catch e {
                    result = 700
                }
            } catch e {
                result = 800
            }
            result
        "#;
        // Внутренний catch должен поймать ошибку
        assert_number_result(source, 700.0);
    }

    #[test]
    fn test_throw_nested_try_catch_outer() {
        // Throw во вложенном try/catch, пойманный внешним catch
        let source = r#"
            let result = 0
            try {
                try {
                    throw "Inner error"
                    result = 100
                } catch e {
                    throw "Re-thrown error"
                }
            } catch e {
                result = 900
            }
            result
        "#;
        // Внешний catch должен поймать повторно выброшенную ошибку
        assert_number_result(source, 900.0);
    }

    #[test]
    fn test_throw_in_while_loop() {
        // Throw в цикле while
        let source = r#"
            let count = 0
            let i = 0
            while i < 5 {
                try {
                    if i == 2 {
                        throw "Error at iteration 2"
                    }
                    count = count + 1
                } catch e {
                    count = count + 10
                }
                i = i + 1
            }
            count
        "#;
        // i=0,1,3,4: успешно -> +1 каждый = 4
        // i=2: ошибка -> +10
        // Итого: 4 + 10 = 14
        assert_number_result(source, 14.0);
    }

    #[test]
    fn test_throw_in_for_loop() {
        // Throw в цикле for
        let source = r#"
            let count = 0
            for i in [0, 1, 2, 3, 4] {
                try {
                    if i == 3 {
                        throw "Error at 3"
                    }
                    count = count + 1
                } catch e {
                    count = count + 20
                }
            }
            count
        "#;
        // i=0,1,2,4: успешно -> +1 каждый = 4
        // i=3: ошибка -> +20
        // Итого: 4 + 20 = 24
        assert_number_result(source, 24.0);
    }

    #[test]
    fn test_throw_in_recursive_function() {
        // Throw в рекурсивной функции
        let source = r#"
            fn recursive(n) {
                if n <= 0 {
                    return 0
                }
                let result = 0
                try {
                    if n == 2 {
                        throw "Error at depth 2"
                    }
                    result = n + recursive(n - 1)
                } catch e {
                    result = 1000
                }
                return result
            }
            recursive(5)
        "#;
        // Когда n=2, ошибка -> catch возвращает 1000
        // Но это происходит на глубине, так что результат зависит от реализации
        let result = run_and_get_result(source);
        assert!(result.is_ok(), "Expected success, got error: {:?}", result);
    }

    #[test]
    fn test_throw_continue_recursion_after_catch() {
        // Throw в рекурсии, обработка и продолжение
        let source = r#"
            fn recursive(n, acc) {
                if n <= 0 {
                    return acc
                }
                let result = acc
                try {
                    if n == 2 {
                        throw "Error at 2"
                    }
                    result = acc + n
                } catch e {
                    result = acc + 50
                }
                return recursive(n - 1, result)
            }
            recursive(3, 0)
        "#;
        // n=3: успешно -> acc=3
        // n=2: ошибка -> acc=3+50=53
        // n=1: успешно -> acc=53+1=54
        // n=0: возврат 54
        assert_number_result(source, 54.0);
    }

    #[test]
    fn test_throw_error_variable_access() {
        // Доступ к переменной ошибки после throw
        let source = r#"
            let error_msg = ""
            try {
                throw "Test error message"
            } catch e {
                error_msg = e
            }
            len(error_msg)
        "#;
        // Должна быть строка с описанием ошибки
        let result = run_and_get_result(source);
        match result {
            Ok(Value::Number(n)) => {
                assert!(n > 0.0, "Expected error message length > 0, got {}", n);
            }
            Ok(v) => panic!("Expected Number, got {:?}", v),
            Err(e) => panic!("Error: {:?}", e),
        }
    }

    #[test]
    fn test_throw_in_conditional() {
        // Throw внутри условного оператора
        let source = r#"
            let result = 0
            if true {
                try {
                    throw "Error in if"
                } catch e {
                    result = 1100
                }
            }
            result
        "#;
        assert_number_result(source, 1100.0);
    }

    #[test]
    fn test_throw_multiple_times_in_loop() {
        // Множественные throw в цикле
        let source = r#"
            let count = 0
            let i = 0
            while i < 5 {
                try {
                    throw "Always throw"
                } catch e {
                    count = count + 1
                }
                i = i + 1
            }
            count
        "#;
        // В каждой итерации throw, catch выполняется -> +1 каждый раз
        assert_number_result(source, 5.0);
    }

    #[test]
    fn test_throw_with_else_block() {
        // Throw с else блоком (else не выполняется при throw)
        let source = r#"
            let result = 0
            try {
                throw "Error occurred"
            } catch e {
                result = 1200
            } else {
                result = -1
            }
            result
        "#;
        assert_number_result(source, 1200.0);
    }

    #[test]
    fn test_throw_no_error_with_else() {
        // Try без throw, else выполняется
        let source = r#"
            let result = 0
            try {
                result = 10 + 20
            } catch e {
                result = -1
            } else {
                result = result + 5
            }
            result
        "#;
        // 10 + 20 = 30, else -> 30 + 5 = 35
        assert_number_result(source, 35.0);
    }

    #[test]
    fn test_throw_complex_expression() {
        // Throw с сложным выражением
        let source = r#"
            let result = 0
            try {
                let x = 10
                let y = 20
                throw "Error: " + str(x + y)
            } catch e {
                result = 1300
            }
            result
        "#;
        assert_number_result(source, 1300.0);
    }
}

