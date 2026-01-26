// Тесты для типизации функций в DataCode
// Проверяем runtime проверку типов параметров и возвращаемых значений

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

    // Вспомогательная функция для проверки строкового результата
    fn assert_string_result(source: &str, expected: &str) {
        let result = run_and_get_result(source);
        match result {
            Ok(Value::String(s)) => {
                assert_eq!(s, expected, "Expected '{}', got '{}'", expected, s);
            }
            Ok(v) => panic!("Expected String('{}'), got {:?}", expected, v),
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

    // Вспомогательная функция для проверки TypeError
    fn assert_type_error(source: &str) {
        let result = run_and_get_result(source);
        match result {
            Err(LangError::RuntimeError { error_type: Some(error_type), message, .. }) => {
                use data_code::common::error::ErrorType;
                assert_eq!(error_type, ErrorType::TypeError, "Expected TypeError, got {:?}", error_type);
                // Также проверяем, что сообщение содержит информацию о типе
                assert!(message.contains("expected type") || message.contains("got"), 
                    "Error message should contain type information: {}", message);
            }
            Err(e) => panic!("Expected TypeError, got {:?}", e),
            Ok(v) => panic!("Expected TypeError, got success: {:?}", v),
        }
    }

    // ========== 1. Тесты функций с типизированными параметрами int ==========

    #[test]
    fn test_int_function_valid_args() {
        let source = r#"
            fn int_add(a: int, b: int) -> int {
                return a + b
            }
            int_add(1, 2)
        "#;
        assert_number_result(source, 3.0);
    }

    #[test]
    fn test_int_function_float_arg_error() {
        assert_type_error(r#"
            fn int_add(a: int, b: int) -> int {
                return a + b
            }
            int_add(1, 2.5)
        "#);
    }

    #[test]
    fn test_int_function_string_arg_error() {
        assert_type_error(r#"
            fn int_add(a: int, b: int) -> int {
                return a + b
            }
            int_add(1, "2")
        "#);
    }

    #[test]
    fn test_int_function_first_arg_error() {
        assert_type_error(r#"
            fn int_add(a: int, b: int) -> int {
                return a + b
            }
            int_add(1.5, 2)
        "#);
    }

    // ========== 2. Тесты функций с типизированными параметрами float ==========

    #[test]
    fn test_float_function_valid_int_arg() {
        // int может быть передан как float
        let source = r#"
            fn float_add(a: float, b: float) -> float {
                return a + b
            }
            float_add(1, 2)
        "#;
        assert_number_result(source, 3.0);
    }

    #[test]
    fn test_float_function_valid_float_arg() {
        let source = r#"
            fn float_add(a: float, b: float) -> float {
                return a + b
            }
            float_add(1.5, 2.5)
        "#;
        assert_number_result(source, 4.0);
    }

    #[test]
    fn test_float_function_string_arg_error() {
        assert_type_error(r#"
            fn float_add(a: float, b: float) -> float {
                return a + b
            }
            float_add(1.5, "2.5")
        "#);
    }

    // ========== 3. Тесты функций с типизированными параметрами str ==========

    #[test]
    fn test_str_function_valid_args() {
        let source = r#"
            fn str_concat(a: str, b: str) -> str {
                return a + b
            }
            str_concat("hello", "world")
        "#;
        assert_string_result(source, "helloworld");
    }

    #[test]
    fn test_str_function_int_arg_error() {
        assert_type_error(r#"
            fn str_concat(a: str, b: str) -> str {
                return a + b
            }
            str_concat("hello", 123)
        "#);
    }

    #[test]
    fn test_str_function_first_arg_error() {
        assert_type_error(r#"
            fn str_concat(a: str, b: str) -> str {
                return a + b
            }
            str_concat(123, "world")
        "#);
    }

    // ========== 4. Тесты функций с типизированными параметрами bool ==========

    #[test]
    fn test_bool_function_valid_args() {
        let source = r#"
            fn bool_and(a: bool, b: bool) -> bool {
                return a and b
            }
            bool_and(true, false)
        "#;
        assert_bool_result(source, false);
    }

    #[test]
    fn test_bool_function_int_arg_error() {
        assert_type_error(r#"
            fn bool_and(a: bool, b: bool) -> bool {
                return a and b
            }
            bool_and(true, 1)
        "#);
    }

    // ========== 5. Тесты функций с типизированными параметрами array ==========

    #[test]
    fn test_array_function_valid_args() {
        let source = r#"
            fn array_len(arr: array) -> int {
                return len(arr)
            }
            array_len([1, 2, 3])
        "#;
        assert_number_result(source, 3.0);
    }

    #[test]
    fn test_array_function_string_arg_error() {
        assert_type_error(r#"
            fn array_len(arr: array) -> int {
                return len(arr)
            }
            array_len("not an array")
        "#);
    }

    // ========== 6. Тесты функций без типизации (должны работать как раньше) ==========

    #[test]
    fn test_untyped_function_works() {
        let source = r#"
            fn add(a, b) {
                return a + b
            }
            add(1, 2)
        "#;
        assert_number_result(source, 3.0);
    }

    #[test]
    fn test_untyped_function_mixed_types() {
        let source = r#"
            fn add(a, b) {
                return a + b
            }
            add(1, 2.5)
        "#;
        assert_number_result(source, 3.5);
    }

    // ========== 7. Тесты частичной типизации ==========

    #[test]
    fn test_partial_typing_first_param() {
        let source = r#"
            fn add(a: int, b) {
                return a + b
            }
            add(1, 2)
        "#;
        assert_number_result(source, 3.0);
    }

    #[test]
    fn test_partial_typing_second_param() {
        let source = r#"
            fn add(a, b: int) {
                return a + b
            }
            add(1, 2)
        "#;
        assert_number_result(source, 3.0);
    }

    #[test]
    fn test_partial_typing_first_param_error() {
        assert_type_error(r#"
            fn add(a: int, b) {
                return a + b
            }
            add(1.5, 2)
        "#);
    }

    #[test]
    fn test_partial_typing_second_param_error() {
        assert_type_error(r#"
            fn add(a, b: int) {
                return a + b
            }
            add(1, 2.5)
        "#);
    }

    // ========== 8. Тесты функций с возвращаемым типом ==========

    #[test]
    fn test_return_type_int() {
        let source = r#"
            fn get_int() -> int {
                return 42
            }
            get_int()
        "#;
        assert_number_result(source, 42.0);
    }

    #[test]
    fn test_return_type_str() {
        let source = r#"
            fn get_str() -> str {
                return "hello"
            }
            get_str()
        "#;
        assert_string_result(source, "hello");
    }

    #[test]
    fn test_return_type_bool() {
        let source = r#"
            fn get_bool() -> bool {
                return true
            }
            get_bool()
        "#;
        assert_bool_result(source, true);
    }

    // ========== 9. Тесты обработки TypeError в try-catch ==========

    #[test]
    fn test_type_error_caught_by_typeerror() {
        let source = r#"
            fn int_add(a: int, b: int) -> int {
                return a + b
            }
            let result = 0
            try {
                int_add(1, 2.5)
            } catch TypeError {
                result = 999
            }
            result
        "#;
        assert_number_result(source, 999.0);
    }

    #[test]
    fn test_type_error_caught_by_runtimeerror() {
        let source = r#"
            fn int_add(a: int, b: int) -> int {
                return a + b
            }
            let result = 0
            try {
                int_add(1, 2.5)
            } catch RuntimeError {
                result = 888
            }
            result
        "#;
        assert_number_result(source, 888.0);
    }

    #[test]
    fn test_type_error_not_caught_by_other_error() {
        assert_type_error(r#"
            fn int_add(a: int, b: int) -> int {
                return a + b
            }
            try {
                int_add(1, 2.5)
            } catch ValueError {
                print("wrong catch")
            }
        "#);
    }

    #[test]
    fn test_type_error_caught_with_variable() {
        let source = r#"
            fn int_add(a: int, b: int) -> int {
                return a + b
            }
            let result = 0
            try {
                int_add(1, 2.5)
            } catch TypeError e {
                result = 777
            }
            result
        "#;
        assert_number_result(source, 777.0);
    }

    // ========== 10. Тесты множественных параметров ==========

    #[test]
    fn test_multiple_typed_params_all_valid() {
        let source = r#"
            fn process(a: int, b: str, c: bool) -> str {
                if c {
                    return b + str(a)
                } else {
                    return b
                }
            }
            process(42, "num: ", true)
        "#;
        assert_string_result(source, "num: 42");
    }

    #[test]
    fn test_multiple_typed_params_first_error() {
        assert_type_error(r#"
            fn process(a: int, b: str, c: bool) -> str {
                return b
            }
            process(42.5, "num: ", true)
        "#);
    }

    #[test]
    fn test_multiple_typed_params_second_error() {
        assert_type_error(r#"
            fn process(a: int, b: str, c: bool) -> str {
                return b
            }
            process(42, 123, true)
        "#);
    }

    #[test]
    fn test_multiple_typed_params_third_error() {
        assert_type_error(r#"
            fn process(a: int, b: str, c: bool) -> str {
                return b
            }
            process(42, "num: ", 1)
        "#);
    }

    // ========== 11. Тесты с параметрами по умолчанию ==========

    #[test]
    fn test_typed_param_with_default_valid() {
        let source = r#"
            fn add(a: int, b: int = 10) -> int {
                return a + b
            }
            add(5)
        "#;
        assert_number_result(source, 15.0);
    }

    #[test]
    fn test_typed_param_with_default_override_valid() {
        let source = r#"
            fn add(a: int, b: int = 10) -> int {
                return a + b
            }
            add(5, 20)
        "#;
        assert_number_result(source, 25.0);
    }

    #[test]
    fn test_typed_param_with_default_override_error() {
        assert_type_error(r#"
            fn add(a: int, b: int = 10) -> int {
                return a + b
            }
            add(5, 20.5)
        "#);
    }

    // ========== 12. Тесты с null типом ==========

    #[test]
    fn test_null_type_parameter() {
        let source = r#"
            fn process(value: null) -> str {
                return "null received"
            }
            process(null)
        "#;
        assert_string_result(source, "null received");
    }

    #[test]
    fn test_null_type_parameter_error() {
        assert_type_error(r#"
            fn process(value: null) -> str {
                return "null received"
            }
            process(123)
        "#);
    }

    // ========== 13. Тесты с object типом ==========

    #[test]
    fn test_object_type_parameter() {
        let source = r#"
            fn get_value(obj: object) -> str {
                return obj["key"]
            }
            get_value({"key": "value"})
        "#;
        assert_string_result(source, "value");
    }

    #[test]
    fn test_object_type_parameter_error() {
        assert_type_error(r#"
            fn get_value(obj: object) -> str {
                return obj["key"]
            }
            get_value("not an object")
        "#);
    }

    // ========== 14. Тесты с tuple типом ==========

    #[test]
    fn test_tuple_type_parameter() {
        let source = r#"
            fn get_first(t: tuple) -> int {
                return t[0]
            }
            get_first((10, 20, 30))
        "#;
        assert_number_result(source, 10.0);
    }

    #[test]
    fn test_tuple_type_parameter_error() {
        assert_type_error(r#"
            fn get_first(t: tuple) -> int {
                return t[0]
            }
            get_first([10, 20, 30])
        "#);
    }

    // ========== 15. Тесты вложенных функций с типизацией ==========

    #[test]
    fn test_nested_typed_functions() {
        let source = r#"
            fn outer(a: int) -> int {
                fn inner(b: int) -> int {
                    return b * 2
                }
                return inner(a)
            }
            outer(5)
        "#;
        assert_number_result(source, 10.0);
    }

    #[test]
    fn test_nested_typed_functions_error() {
        assert_type_error(r#"
            fn outer(a: int) -> int {
                fn inner(b: int) -> int {
                    return b * 2
                }
                return inner(a)
            }
            outer(5.5)
        "#);
    }

    // ========== 16. Тесты рекурсивных функций с типизацией ==========

    #[test]
    fn test_recursive_typed_function() {
        let source = r#"
            fn factorial(n: int) -> int {
                if n <= 1 {
                    return 1
                } else {
                    return n * factorial(n - 1)
                }
            }
            factorial(5)
        "#;
        assert_number_result(source, 120.0);
    }

    #[test]
    fn test_recursive_typed_function_error() {
        assert_type_error(r#"
            fn factorial(n: int) -> int {
                if n <= 1 {
                    return 1
                } else {
                    return n * factorial(n - 1)
                }
            }
            factorial(5.5)
        "#);
    }

    // ========== 17. Тесты сообщений об ошибках ==========

    #[test]
    fn test_type_error_message_contains_param_name() {
        let result = run_and_get_result(r#"
            fn test(a: int, b: str) -> int {
                return a
            }
            test(1, 2)
        "#);
        match result {
            Err(LangError::RuntimeError { message, error_type, .. }) => {
                use data_code::common::error::ErrorType;
                assert_eq!(error_type, Some(ErrorType::TypeError), "Expected TypeError");
                assert!(message.contains("b"), "Error message should contain parameter name 'b'");
                assert!(message.contains("str"), "Error message should contain expected type 'str'");
            }
            _ => panic!("Expected TypeError with message"),
        }
    }

    // ========== 18. Тесты совместимости типов (int/float) ==========

    #[test]
    fn test_int_accepts_only_integers() {
        // int должен принимать только целые числа
        assert_type_error(r#"
            fn test(n: int) -> int {
                return n
            }
            test(1.5)
        "#);
    }

    #[test]
    fn test_float_accepts_integers() {
        // float должен принимать и целые числа
        let source = r#"
            fn test(n: float) -> float {
                return n
            }
            test(1)
        "#;
        assert_number_result(source, 1.0);
    }

    #[test]
    fn test_float_accepts_floats() {
        let source = r#"
            fn test(n: float) -> float {
                return n
            }
            test(1.5)
        "#;
        assert_number_result(source, 1.5);
    }

    // ========== 19. Тесты union типов (str | int) ==========

    #[test]
    fn test_union_type_str_or_int_with_str() {
        let source = r#"
            fn process(value: str | int) -> str {
                return str(value)
            }
            process("hello")
        "#;
        assert_string_result(source, "hello");
    }

    #[test]
    fn test_union_type_str_or_int_with_int() {
        let source = r#"
            fn process(value: str | int) -> str {
                return str(value)
            }
            process(42)
        "#;
        assert_string_result(source, "42");
    }

    #[test]
    fn test_union_type_str_or_int_with_float_error() {
        assert_type_error(r#"
            fn process(value: str | int) -> str {
                return str(value)
            }
            process(3.14)
        "#);
    }

    #[test]
    fn test_union_type_three_types() {
        let source = r#"
            fn process(value: null | str | int) -> str {
                if value == null {
                    return "null"
                } else {
                    return str(value)
                }
            }
            process(null)
        "#;
        assert_string_result(source, "null");
    }

    #[test]
    fn test_union_type_three_types_with_str() {
        let source = r#"
            fn process(value: null | str | int) -> str {
                if value == null {
                    return "null"
                } else {
                    return str(value)
                }
            }
            process("test")
        "#;
        assert_string_result(source, "test");
    }

    #[test]
    fn test_union_type_three_types_with_int() {
        let source = r#"
            fn process(value: null | str | int) -> str {
                if value == null {
                    return "null"
                } else {
                    return str(value)
                }
            }
            process(123)
        "#;
        assert_string_result(source, "123");
    }

    #[test]
    fn test_union_type_three_types_with_bool_error() {
        assert_type_error(r#"
            fn process(value: null | str | int) -> str {
                return str(value)
            }
            process(true)
        "#);
    }

    #[test]
    fn test_union_type_with_default_value() {
        let source = r#"
            fn process(value: null | str | int = null) -> str {
                if value == null {
                    return "default"
                } else {
                    return str(value)
                }
            }
            process()
        "#;
        assert_string_result(source, "default");
    }

    #[test]
    fn test_union_type_with_default_value_override() {
        let source = r#"
            fn process(value: null | str | int = null) -> str {
                if value == null {
                    return "default"
                } else {
                    return str(value)
                }
            }
            process("custom")
        "#;
        assert_string_result(source, "custom");
    }

    #[test]
    fn test_union_type_with_default_value_override_error() {
        assert_type_error(r#"
            fn process(value: null | str | int = null) -> str {
                return str(value)
            }
            process(true)
        "#);
    }

    #[test]
    fn test_union_type_int_or_float() {
        let source = r#"
            fn add(a: int | float, b: int | float) -> float {
                return a + b
            }
            add(1, 2)
        "#;
        assert_number_result(source, 3.0);
    }

    #[test]
    fn test_union_type_int_or_float_with_float() {
        let source = r#"
            fn add(a: int | float, b: int | float) -> float {
                return a + b
            }
            add(1.5, 2.5)
        "#;
        assert_number_result(source, 4.0);
    }

    #[test]
    fn test_union_type_int_or_float_mixed() {
        let source = r#"
            fn add(a: int | float, b: int | float) -> float {
                return a + b
            }
            add(1, 2.5)
        "#;
        assert_number_result(source, 3.5);
    }

    #[test]
    fn test_union_type_int_or_float_with_str_error() {
        assert_type_error(r#"
            fn add(a: int | float, b: int | float) -> float {
                return a + b
            }
            add(1, "2")
        "#);
    }

    #[test]
    fn test_union_type_array_or_string() {
        let source = r#"
            fn get_length(value: array | str) -> int {
                return len(value)
            }
            get_length("hello")
        "#;
        assert_number_result(source, 5.0);
    }

    #[test]
    fn test_union_type_array_or_string_with_array() {
        let source = r#"
            fn get_length(value: array | str) -> int {
                return len(value)
            }
            get_length([1, 2, 3])
        "#;
        assert_number_result(source, 3.0);
    }

    #[test]
    fn test_union_type_array_or_string_with_int_error() {
        assert_type_error(r#"
            fn get_length(value: array | str) -> int {
                return len(value)
            }
            get_length(123)
        "#);
    }

    #[test]
    fn test_union_type_return_type() {
        let source = r#"
            fn get_value(flag: bool) -> str | int {
                if flag {
                    return "string"
                } else {
                    return 42
                }
            }
            get_value(true)
        "#;
        assert_string_result(source, "string");
    }

    #[test]
    fn test_union_type_return_type_int() {
        let source = r#"
            fn get_value(flag: bool) -> str | int {
                if flag {
                    return "string"
                } else {
                    return 42
                }
            }
            get_value(false)
        "#;
        assert_number_result(source, 42.0);
    }

    #[test]
    fn test_union_type_multiple_params() {
        let source = r#"
            fn process(a: str | int, b: str | int) -> str {
                return str(a) + " " + str(b)
            }
            process("hello", 42)
        "#;
        assert_string_result(source, "hello 42");
    }

    #[test]
    fn test_union_type_multiple_params_first_error() {
        assert_type_error(r#"
            fn process(a: str | int, b: str | int) -> str {
                return str(a) + str(b)
            }
            process(true, 42)
        "#);
    }

    #[test]
    fn test_union_type_multiple_params_second_error() {
        assert_type_error(r#"
            fn process(a: str | int, b: str | int) -> str {
                return str(a) + str(b)
            }
            process("hello", true)
        "#);
    }
}
