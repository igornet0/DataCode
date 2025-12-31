// Тесты для распаковки параметров в цикле for

#[cfg(test)]
mod tests {
    use data_code::{run, LangError, Value};

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

    fn assert_error(source: &str, expected_error_contains: &str) {
        let result = run(source);
        match result {
            Err(LangError::RuntimeError { message, .. }) => {
                assert!(
                    message.contains(expected_error_contains),
                    "Expected error to contain '{}', but got: {}",
                    expected_error_contains,
                    message
                );
            }
            Ok(value) => {
                panic!("Expected error, but got value: {:?}", value);
            }
            Err(e) => {
                panic!("Expected RuntimeError, but got: {:?}", e);
            }
        }
    }

#[test]
fn test_basic_unpack() {
    let source = r#"
        let sum = 0
        for x, y in [[1, 2], [3, 4], [5, 6]] {
            sum = sum + x + y
        }
        sum
    "#;
    // sum = (1+2) + (3+4) + (5+6) = 3 + 7 + 11 = 21
    assert_number_result(source, 21.0);
}

#[test]
fn test_unpack_single_variable() {
    // Обратная совместимость: for x in array
    let source = r#"
        let sum = 0
        for x in [1, 2, 3] {
            sum = sum + x
        }
        sum
    "#;
    assert_number_result(source, 6.0);
}

#[test]
fn test_unpack_three_variables() {
    let source = r#"
        let sum = 0
        for x, y, z in [[1, 2, 3], [4, 5, 6]] {
            sum = sum + x + y + z
        }
        sum
    "#;
    // sum = (1+2+3) + (4+5+6) = 6 + 15 = 21
    assert_number_result(source, 21.0);
}

#[test]
fn test_unpack_wildcard() {
    let source = r#"
        let sum = 0
        for x, _, y in [[1, 2, 3], [4, 5, 6]] {
            sum = sum + x + y
        }
        sum
    "#;
    // sum = (1+3) + (4+6) = 4 + 10 = 14
    assert_number_result(source, 14.0);
}

#[test]
fn test_unpack_parentheses_syntax() {
    let source = r#"
        let sum = 0
        for (x, y) in [[1, 2], [3, 4]] {
            sum = sum + x + y
        }
        sum
    "#;
    // sum = (1+2) + (3+4) = 3 + 7 = 10
    assert_number_result(source, 10.0);
}

#[test]
fn test_unpack_brackets_syntax() {
    let source = r#"
        let sum = 0
        for [x, y] in [[1, 2], [3, 4]] {
            sum += x + y
        }
        sum
    "#;
    // sum = (1+2) + (3+4) = 3 + 7 = 10
    assert_number_result(source, 10.0);
}

#[test]
fn test_unpack_nested() {
    let source = r#"
        let sum = 0
        for (x, (y, z)) in [[1, [2, 3]], [4, [5, 6]]] {
            sum +=x  + y + z
        }
        sum
    "#;
    // sum = (1+2+3) + (4+5+6) = 6 + 15 = 21
    assert_number_result(source, 21.0);
}

    #[test]
    fn test_unpack_count_mismatch() {
        let source = r#"
        for x, y in [[1]] {
            x + y
        }
    "#;
        // Должна быть ошибка: ожидается 2 значения, получено 1
        // Теперь проверка происходит раньше, при проверке минимальной длины
        assert_error(source, "requires at least");
    }

#[test]
fn test_unpack_not_iterable() {
    let source = r#"
        for x, y in 123 {
            x + y
        }
    "#;
    // Должна быть ошибка: объект не итерируемый
    assert_error(source, "Expected array");
}

#[test]
fn test_unpack_not_unpackable() {
    let source = r#"
        for x, y in [[1, 2, 3], 123] {
            x + y
        }
    "#;
    // Должна быть ошибка при попытке распаковать число
    assert_error(source, "Expected array");
}

#[test]
fn test_unpack_null() {
    let source = r#"
        for (x, y) in [null] {
            x + y
        }
    "#;
    // Должна быть ошибка: null не распаковываем
    assert_error(source, "Expected array");
}

#[test]
fn test_unpack_empty_array() {
    let source = r#"
        let count = 0
        for x, y in [] {
            count = count + 1
        }
        count
    "#;
    // Пустой массив - цикл не выполнится
    assert_number_result(source, 0.0);
}

#[test]
fn test_unpack_multiple_wildcards() {
    let source = r#"
        let sum = 0
        for x, _, _, y in [[1, 2, 3, 4], [5, 6, 7, 8]] {
            sum = sum + x + y
        }
        sum
    "#;
    // sum = (1+4) + (5+8) = 5 + 13 = 18
    assert_number_result(source, 18.0);
}

#[test]
fn test_unpack_all_wildcards() {
    let source = r#"
        let count = 0
        for _, _ in [[1, 2], [3, 4]] {
            count = count + 1
        }
        count
    "#;
    // Все значения пропущены, но цикл выполняется
    assert_number_result(source, 2.0);
}

    #[test]
    fn test_unpack_nested_with_wildcard() {
        let source = r#"
        let sum = 0
        for (x, (_, z)) in [[1, [2, 3]], [4, [5, 6]]] {
            sum = sum + x + z
        }
        sum
    "#;
    // sum = (1+3) + (4+6) = 4 + 10 = 14
    assert_number_result(source, 14.0);
}

    // ========== Тесты для variadic unpacking ==========

    #[test]
    fn test_variadic_basic() {
        let source = r#"
        let sum = 0
        for x, *y in [[1], [1, 2], [1, 2, 3]] {
            sum = sum + x
            let y_len = len(y)
            sum = sum + y_len
        }
        sum
    "#;
        // Проверяем, что код выполняется без ошибок
        // x=1, y=[] (len=0) -> sum=1+0=1
        // x=1, y=[2] (len=1) -> sum=1+1+1=3
        // x=1, y=[2,3] (len=2) -> sum=3+1+2=6
        // Итого: 6
        assert_number_result(source, 6.0);
    }

    #[test]
    fn test_variadic_empty_rest() {
        let source = r#"
        let sum = 0
        for x, *y in [[1]] {
            sum = x
            if y == [] {
                sum = sum + 100
            }
        }
        sum
    "#;
        // x=1, y=[], sum должен быть 101
        assert_number_result(source, 101.0);
    }

    #[test]
    fn test_variadic_multiple() {
        let source = r#"
        let sum = 0
        for x, y, *z in [[1, 2, 3, 4]] {
            sum = x + y
            # Проверяем, что z не пустой (должен содержать [3, 4])
            let z_len = len(z)
            if z_len == 2 {
                sum += 100
            }
        }
        sum
    "#;
        // x=1, y=2, z=[3,4] (длина 2), sum должен быть 103
        assert_number_result(source, 103.0);
    }

    #[test]
    fn test_variadic_wildcard() {
        let source = r#"
        let sum = 0
        for x, *_ in [[1, 2, 3], [4, 5, 6]] {
            sum = sum + x
        }
        sum
    "#;
        // sum = 1 + 4 = 5
        assert_number_result(source, 5.0);
    }

    #[test]
    fn test_variadic_min_length_error() {
        let source = r#"
        for x, *y in [[]] {
            x + y
        }
    "#;
        // Должна быть ошибка: длина меньше минимально требуемой
        assert_error(source, "requires at least");
    }

    #[test]
    fn test_variadic_not_iterable() {
        let source = r#"
        for x, *y in 123 {
            x + y
        }
    "#;
        // Должна быть ошибка: объект не итерируемый
        assert_error(source, "Expected array");
    }

    #[test]
    fn test_variadic_not_unpackable() {
        let source = r#"
        for x, *y in [[1, 2], 123] {
            x + y
        }
    "#;
        // Должна быть ошибка при попытке распаковать число
        // Ошибка возникает при попытке получить длину не-массива
        // Проверяем, что ошибка связана с массивом или операцией
        let result = run(source);
        match result {
            Err(LangError::RuntimeError { message, .. }) => {
                assert!(
                    message.contains("array") || message.contains("Array") || message.contains("Operands"),
                    "Expected error to contain 'array', 'Array', or 'Operands', but got: {}",
                    message
                );
            }
            Ok(value) => {
                panic!("Expected error, but got value: {:?}", value);
            }
            Err(e) => {
                panic!("Expected RuntimeError, but got: {:?}", e);
            }
        }
    }

}

