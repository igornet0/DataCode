// Тесты для проверки номеров строк в ошибках

#[cfg(test)]
mod tests {
    use data_code::{run, LangError};

    #[test]
    fn test_undefined_variable_line_number() {
        let source = r#"
            let x = 10
            let y = undefined_var
        "#;
        let result = run(source);
        assert!(result.is_err());
        // Неопределённая переменная обнаруживается при выполнении (RuntimeError)
        if let Err(LangError::RuntimeError { line, message, .. }) = result {
            assert!(line > 0, "Expected line number > 0, got {}", line);
            assert!(message.contains("Undefined variable"), "Expected undefined variable message, got {}", message);
        } else {
            panic!("Expected RuntimeError for undefined variable, got {:?}", result);
        }
    }

    #[test]
    fn test_division_by_zero_line_number() {
        let source = r#"
            let x = 10
            let y = x / 0
        "#;
        let result = run(source);
        assert!(result.is_err());
        if let Err(LangError::RuntimeError { line, .. }) = result {
            // Номер строки должен быть больше 0
            assert!(line > 0, "Expected line number > 0, got {}", line);
        } else {
            panic!("Expected RuntimeError, got {:?}", result);
        }
    }

    #[test]
    fn test_type_error_line_number() {
        let source = r#"
            let x = "hello"
            let y = x - 5
        "#;
        let result = run(source);
        assert!(result.is_err());
        if let Err(LangError::RuntimeError { line, .. }) = result {
            // Номер строки должен быть больше 0
            assert!(line > 0, "Expected line number > 0, got {}", line);
        } else {
            panic!("Expected RuntimeError, got {:?}", result);
        }
    }

    #[test]
    fn test_function_call_error_line_number() {
        let source = r#"
            fn add(a, b) {
                return a + b
            }
            add(1, 2, 3)
        "#;
        let result = run(source);
        assert!(result.is_err());
        if let Err(LangError::ParseError { line, .. }) = result {
            // Номер строки должен быть больше 0 (строка с вызовом add(1, 2, 3))
            assert!(line > 0, "Expected line number > 0, got {}", line);
        } else {
            panic!("Expected ParseError, got {:?}", result);
        }
    }

    #[test]
    fn test_stack_trace_has_line_numbers() {
        let source = r#"
            fn inner() {
                let x = 10 / 0
                return x
            }
            fn outer() {
                return inner()
            }
            outer()
        "#;
        let result = run(source);
        assert!(result.is_err());
        if let Err(LangError::RuntimeError { stack_trace, .. }) = result {
            // Стек вызовов должен содержать номера строк
            assert!(!stack_trace.is_empty(), "Expected non-empty stack trace");
            for entry in &stack_trace {
                // Каждая запись должна иметь номер строки (может быть 0, если не удалось определить)
                // Note: line is usize, so it's always >= 0, but we check it's reasonable
                assert!(entry.line > 0, "Expected line number > 0, got {}", entry.line);
            }
        } else {
            panic!("Expected RuntimeError with stack trace, got {:?}", result);
        }
    }
}

