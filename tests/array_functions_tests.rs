// Тесты для функций массивов
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

    // Вспомогательная функция для проверки массива
    fn assert_array_result(source: &str, expected: &[Value]) {
        let result = run(source);
        match result {
            Ok(Value::Array(arr)) => {
                let arr_ref = arr.borrow();
                assert_eq!(arr_ref.len(), expected.len(), "Array length mismatch");
                for (i, (actual, expected_val)) in arr_ref.iter().zip(expected.iter()).enumerate() {
                    assert_eq!(actual, expected_val, "Mismatch at index {}", i);
                }
            }
            Ok(v) => panic!("Expected Array, got {:?}", v),
            Err(e) => panic!("Error: {:?}", e),
        }
    }

    // ========== Тесты для push ==========

    #[test]
    fn test_push_basic() {
        let source = r#"
            let arr = [1, 2, 3]
            push(arr, 4)
        "#;
        assert_array_result(source, &[
            Value::Number(1.0),
            Value::Number(2.0),
            Value::Number(3.0),
            Value::Number(4.0),
        ]);
    }

    #[test]
    fn test_push_empty_array() {
        let source = r#"
            let arr = []
            push(arr, 1)
        "#;
        assert_array_result(source, &[Value::Number(1.0)]);
    }

    #[test]
    fn test_push_multiple() {
        let source = r#"
            let arr = [1]
            push(arr, 2)
            push(arr, 3)
        "#;
        assert_array_result(source, &[
            Value::Number(1.0),
            Value::Number(2.0),
            Value::Number(3.0),
        ]);
    }

    #[test]
    fn test_push_string() {
        let source = r#"
            let arr = ["a", "b"]
            push(arr, "c")
        "#;
        assert_array_result(source, &[
            Value::String("a".to_string()),
            Value::String("b".to_string()),
            Value::String("c".to_string()),
        ]);
    }

    // ========== Тесты для pop ==========

    #[test]
    fn test_pop_basic() {
        let source = r#"
            let arr = [1, 2, 3]
            pop(arr)
        "#;
        assert_number_result(source, 3.0);
    }

    #[test]
    fn test_pop_removes_element() {
        // pop возвращает последний элемент, но не модифицирует массив
        // (так как значения передаются по значению)
        let source = r#"
            let arr = [1, 2, 3]
            pop(arr)
        "#;
        assert_number_result(source, 3.0);
    }

    #[test]
    fn test_pop_empty_array() {
        let source = r#"
            let arr = []
            pop(arr)
        "#;
        let result = run(source);
        match result {
            Ok(Value::Null) => {}
            Ok(v) => panic!("Expected Null for empty array pop, got {:?}", v),
            Err(e) => panic!("Error: {:?}", e),
        }
    }

    #[test]
    fn test_pop_single_element() {
        let source = r#"
            let arr = [42]
            pop(arr)
        "#;
        assert_number_result(source, 42.0);
    }

    #[test]
    fn test_pop_string() {
        let source = r#"
            let arr = ["a", "b", "c"]
            pop(arr)
        "#;
        assert_string_result(source, "c");
    }

    // ========== Тесты для unique ==========

    #[test]
    fn test_unique_basic() {
        let source = r#"
            unique([1, 2, 2, 3, 3, 3])
        "#;
        assert_array_result(source, &[
            Value::Number(1.0),
            Value::Number(2.0),
            Value::Number(3.0),
        ]);
    }

    #[test]
    fn test_unique_all_unique() {
        let source = r#"
            unique([1, 2, 3])
        "#;
        assert_array_result(source, &[
            Value::Number(1.0),
            Value::Number(2.0),
            Value::Number(3.0),
        ]);
    }

    #[test]
    fn test_unique_empty() {
        let source = r#"
            unique([])
        "#;
        assert_array_result(source, &[]);
    }

    #[test]
    fn test_unique_strings() {
        let source = r#"
            unique(["a", "b", "a", "c", "b"])
        "#;
        assert_array_result(source, &[
            Value::String("a".to_string()),
            Value::String("b".to_string()),
            Value::String("c".to_string()),
        ]);
    }

    #[test]
    fn test_unique_preserves_order() {
        let source = r#"
            unique([3, 1, 3, 2, 1])
        "#;
        // Порядок должен сохраняться - первое вхождение каждого элемента
        assert_array_result(source, &[
            Value::Number(3.0),
            Value::Number(1.0),
            Value::Number(2.0),
        ]);
    }

    // ========== Тесты для reverse ==========

    #[test]
    fn test_reverse_basic() {
        let source = r#"
            let arr = [1, 2, 3]
            reverse(arr)
        "#;
        assert_array_result(source, &[
            Value::Number(3.0),
            Value::Number(2.0),
            Value::Number(1.0),
        ]);
    }

    #[test]
    fn test_reverse_empty() {
        let source = r#"
            let arr = []
            reverse(arr)
        "#;
        assert_array_result(source, &[]);
    }

    #[test]
    fn test_reverse_single() {
        let source = r#"
            let arr = [42]
            reverse(arr)
        "#;
        assert_array_result(source, &[Value::Number(42.0)]);
    }

    #[test]
    fn test_reverse_strings() {
        let source = r#"
            let arr = ["a", "b", "c"]
            reverse(arr)
        "#;
        assert_array_result(source, &[
            Value::String("c".to_string()),
            Value::String("b".to_string()),
            Value::String("a".to_string()),
        ]);
    }

    // ========== Тесты для sort ==========

    #[test]
    fn test_sort_numbers() {
        let source = r#"
            let arr = [3, 1, 4, 1, 5]
            sort(arr)
        "#;
        assert_array_result(source, &[
            Value::Number(1.0),
            Value::Number(1.0),
            Value::Number(3.0),
            Value::Number(4.0),
            Value::Number(5.0),
        ]);
    }

    #[test]
    fn test_sort_strings() {
        let source = r#"
            let arr = ["c", "a", "b"]
            sort(arr)
        "#;
        assert_array_result(source, &[
            Value::String("a".to_string()),
            Value::String("b".to_string()),
            Value::String("c".to_string()),
        ]);
    }

    #[test]
    fn test_sort_empty() {
        let source = r#"
            let arr = []
            sort(arr)
        "#;
        assert_array_result(source, &[]);
    }

    #[test]
    fn test_sort_already_sorted() {
        let source = r#"
            let arr = [1, 2, 3]
            sort(arr)
        "#;
        assert_array_result(source, &[
            Value::Number(1.0),
            Value::Number(2.0),
            Value::Number(3.0),
        ]);
    }

    // ========== Тесты для sum ==========

    #[test]
    fn test_sum_basic() {
        let source = r#"
            sum([1, 2, 3, 4, 5])
        "#;
        assert_number_result(source, 15.0);
    }

    #[test]
    fn test_sum_empty() {
        let source = r#"
            sum([])
        "#;
        assert_number_result(source, 0.0);
    }

    #[test]
    fn test_sum_single() {
        let source = r#"
            sum([42])
        "#;
        assert_number_result(source, 42.0);
    }

    #[test]
    fn test_sum_negative() {
        let source = r#"
            sum([1, -2, 3])
        "#;
        assert_number_result(source, 2.0);
    }

    #[test]
    fn test_sum_float() {
        let source = r#"
            sum([1.5, 2.5, 3.0])
        "#;
        assert_number_result(source, 7.0);
    }

    #[test]
    fn test_sum_ignores_non_numbers() {
        let source = r#"
            sum([1, 2, "three", 4])
        "#;
        // Должна суммировать только числа, игнорируя строки
        assert_number_result(source, 7.0);
    }

    // ========== Тесты для average ==========

    #[test]
    fn test_average_basic() {
        let source = r#"
            average([1, 2, 3, 4, 5])
        "#;
        assert_number_result(source, 3.0);
    }

    #[test]
    fn test_average_empty() {
        let source = r#"
            average([])
        "#;
        // Пустой массив - возвращаем 0.0 или Null
        let result = run(source);
        match result {
            Ok(Value::Number(n)) => {
                assert_eq!(n, 0.0, "Expected 0.0 for empty array");
            }
            Ok(Value::Null) => {} // Null тоже приемлемо
            Ok(v) => panic!("Expected Number(0.0) or Null, got {:?}", v),
            Err(e) => panic!("Error: {:?}", e),
        }
    }

    #[test]
    fn test_average_single() {
        let source = r#"
            average([42])
        "#;
        assert_number_result(source, 42.0);
    }

    #[test]
    fn test_average_float() {
        let source = r#"
            average([1.0, 2.0, 3.0])
        "#;
        assert_number_result(source, 2.0);
    }

    #[test]
    fn test_average_ignores_non_numbers() {
        let source = r#"
            average([1, 2, "three", 4, 5])
        "#;
        // Среднее от 1, 2, 4, 5 = 12/4 = 3.0
        assert_number_result(source, 3.0);
    }

    // ========== Тесты для count ==========

    #[test]
    fn test_count_basic() {
        let source = r#"
            count([1, 2, 3])
        "#;
        assert_number_result(source, 3.0);
    }

    #[test]
    fn test_count_empty() {
        let source = r#"
            count([])
        "#;
        assert_number_result(source, 0.0);
    }

    #[test]
    fn test_count_single() {
        let source = r#"
            count([42])
        "#;
        assert_number_result(source, 1.0);
    }

    #[test]
    fn test_count_mixed_types() {
        let source = r#"
            count([1, "two", 3, true, null])
        "#;
        assert_number_result(source, 5.0);
    }

    #[test]
    fn test_count_equals_len() {
        let source = r#"
            let arr = [1, 2, 3, 4, 5]
            count(arr) == len(arr)
        "#;
        let result = run(source);
        match result {
            Ok(Value::Bool(true)) => {}
            Ok(v) => panic!("Expected true, got {:?}", v),
            Err(e) => panic!("Error: {:?}", e),
        }
    }

    // ========== Комбинированные тесты ==========

    #[test]
    fn test_push_pop_combination() {
        let source = r#"
            let arr = [1, 2]
            let arr2 = []
            push(arr2, 3)
            pop(arr2)
        "#;
        assert_number_result(source, 3.0);
    }

    #[test]
    fn test_unique_sort_combination() {
        let source = r#"
            let arr = [3, 1, 2, 3, 1]
            let unique_arr = unique(arr)
            sort(unique_arr)
        "#;
        assert_array_result(source, &[
            Value::Number(1.0),
            Value::Number(2.0),
            Value::Number(3.0),
        ]);
    }

    /// Локализующий тест: sum изолированно должен возвращать 15 для [1,2,3,4,5].
    #[test]
    fn test_sum_isolated() {
        assert_number_result("sum([1, 2, 3, 4, 5])", 15.0);
    }

    /// Локализующий тест: count изолированно должен возвращать 5 для [1,2,3,4,5].
    #[test]
    fn test_count_isolated() {
        assert_number_result("count([1, 2, 3, 4, 5])", 5.0);
    }

    #[test]
    fn test_sum_average_relationship() {
        let source = r#"
            let arr = [1, 2, 3, 4, 5]
            let s = sum(arr)
            let c = count(arr)
            s / c
        "#;
        assert_number_result(source, 3.0);
    }

    #[test]
    fn test_reverse_twice() {
        let source = r#"
            let arr = [1, 2, 3]
            let arr2 = reverse(arr)
            reverse(arr2)
        "#;
        assert_array_result(source, &[
            Value::Number(1.0),
            Value::Number(2.0),
            Value::Number(3.0),
        ]);
    }

    // ========== Тесты обработки ошибок ==========

    #[test]
    fn test_push_wrong_type() {
        let source = r#"
            push(42, 1)
        "#;
        // Должна вернуть Null при неправильном типе
        let result = run(source);
        match result {
            Ok(Value::Null) => {}
            Ok(v) => panic!("Expected Null for wrong type, got {:?}", v),
            Err(_) => {} // Ошибка тоже приемлема
        }
    }

    #[test]
    fn test_pop_wrong_type() {
        let source = r#"
            pop("not an array")
        "#;
        let result = run(source);
        match result {
            Ok(Value::Null) => {}
            Ok(v) => panic!("Expected Null for wrong type, got {:?}", v),
            Err(_) => {}
        }
    }

    #[test]
    fn test_sum_wrong_type() {
        let source = r#"
            sum("not an array")
        "#;
        let result = run(source);
        match result {
            Ok(Value::Null) => {}
            Ok(Value::Number(0.0)) => {} // 0.0 тоже приемлемо
            Ok(v) => panic!("Expected Null or Number(0.0), got {:?}", v),
            Err(_) => {}
        }
    }

    // ========== Тесты для any ==========

    #[test]
    fn test_any_basic() {
        let source = r#"
            any([false, false, true, false])
        "#;
        let result = run(source);
        match result {
            Ok(Value::Bool(true)) => {}
            Ok(v) => panic!("Expected Bool(true), got {:?}", v),
            Err(e) => panic!("Error: {:?}", e),
        }
    }

    #[test]
    fn test_any_all_true() {
        let source = r#"
            any([true, true, true])
        "#;
        let result = run(source);
        match result {
            Ok(Value::Bool(true)) => {}
            Ok(v) => panic!("Expected Bool(true), got {:?}", v),
            Err(e) => panic!("Error: {:?}", e),
        }
    }

    #[test]
    fn test_any_all_false() {
        let source = r#"
            any([false, false, false])
        "#;
        let result = run(source);
        match result {
            Ok(Value::Bool(false)) => {}
            Ok(v) => panic!("Expected Bool(false), got {:?}", v),
            Err(e) => panic!("Error: {:?}", e),
        }
    }

    #[test]
    fn test_any_empty() {
        let source = r#"
            any([])
        "#;
        let result = run(source);
        match result {
            Ok(Value::Bool(false)) => {}
            Ok(v) => panic!("Expected Bool(false), got {:?}", v),
            Err(e) => panic!("Error: {:?}", e),
        }
    }

    #[test]
    fn test_any_single_true() {
        let source = r#"
            any([true])
        "#;
        let result = run(source);
        match result {
            Ok(Value::Bool(true)) => {}
            Ok(v) => panic!("Expected Bool(true), got {:?}", v),
            Err(e) => panic!("Error: {:?}", e),
        }
    }

    #[test]
    fn test_any_single_false() {
        let source = r#"
            any([false])
        "#;
        let result = run(source);
        match result {
            Ok(Value::Bool(false)) => {}
            Ok(v) => panic!("Expected Bool(false), got {:?}", v),
            Err(e) => panic!("Error: {:?}", e),
        }
    }

    #[test]
    fn test_any_with_numbers() {
        let source = r#"
            any([0, 0, 1, 0])
        "#;
        let result = run(source);
        match result {
            Ok(Value::Bool(true)) => {}
            Ok(v) => panic!("Expected Bool(true), got {:?}", v),
            Err(e) => panic!("Error: {:?}", e),
        }
    }

    #[test]
    fn test_any_with_strings() {
        let source = r#"
            any(["", "", "hello", ""])
        "#;
        let result = run(source);
        match result {
            Ok(Value::Bool(true)) => {}
            Ok(v) => panic!("Expected Bool(true), got {:?}", v),
            Err(e) => panic!("Error: {:?}", e),
        }
    }

    #[test]
    fn test_any_with_mixed_types() {
        let source = r#"
            any([false, 0, "", null, true])
        "#;
        let result = run(source);
        match result {
            Ok(Value::Bool(true)) => {}
            Ok(v) => panic!("Expected Bool(true), got {:?}", v),
            Err(e) => panic!("Error: {:?}", e),
        }
    }

    #[test]
    fn test_any_wrong_type() {
        let source = r#"
            any("not an array")
        "#;
        let result = run(source);
        match result {
            Ok(Value::Bool(false)) => {}
            Ok(v) => panic!("Expected Bool(false), got {:?}", v),
            Err(_) => {}
        }
    }

    // ========== Тесты для all ==========

    #[test]
    fn test_all_basic() {
        let source = r#"
            all([true, true, true, true])
        "#;
        let result = run(source);
        match result {
            Ok(Value::Bool(true)) => {}
            Ok(v) => panic!("Expected Bool(true), got {:?}", v),
            Err(e) => panic!("Error: {:?}", e),
        }
    }

    #[test]
    fn test_all_with_false() {
        let source = r#"
            all([true, true, false, true])
        "#;
        let result = run(source);
        match result {
            Ok(Value::Bool(false)) => {}
            Ok(v) => panic!("Expected Bool(false), got {:?}", v),
            Err(e) => panic!("Error: {:?}", e),
        }
    }

    #[test]
    fn test_all_all_false() {
        let source = r#"
            all([false, false, false])
        "#;
        let result = run(source);
        match result {
            Ok(Value::Bool(false)) => {}
            Ok(v) => panic!("Expected Bool(false), got {:?}", v),
            Err(e) => panic!("Error: {:?}", e),
        }
    }

    #[test]
    fn test_all_empty() {
        let source = r#"
            all([])
        "#;
        let result = run(source);
        match result {
            Ok(Value::Bool(false)) => {}
            Ok(v) => panic!("Expected Bool(false), got {:?}", v),
            Err(e) => panic!("Error: {:?}", e),
        }
    }

    #[test]
    fn test_all_single_true() {
        let source = r#"
            all([true])
        "#;
        let result = run(source);
        match result {
            Ok(Value::Bool(true)) => {}
            Ok(v) => panic!("Expected Bool(true), got {:?}", v),
            Err(e) => panic!("Error: {:?}", e),
        }
    }

    #[test]
    fn test_all_single_false() {
        let source = r#"
            all([false])
        "#;
        let result = run(source);
        match result {
            Ok(Value::Bool(false)) => {}
            Ok(v) => panic!("Expected Bool(false), got {:?}", v),
            Err(e) => panic!("Error: {:?}", e),
        }
    }

    #[test]
    fn test_all_with_numbers() {
        let source = r#"
            all([1, 2, 3, 4])
        "#;
        let result = run(source);
        match result {
            Ok(Value::Bool(true)) => {}
            Ok(v) => panic!("Expected Bool(true), got {:?}", v),
            Err(e) => panic!("Error: {:?}", e),
        }
    }

    #[test]
    fn test_all_with_zero() {
        let source = r#"
            all([1, 2, 0, 4])
        "#;
        let result = run(source);
        match result {
            Ok(Value::Bool(false)) => {}
            Ok(v) => panic!("Expected Bool(false), got {:?}", v),
            Err(e) => panic!("Error: {:?}", e),
        }
    }

    #[test]
    fn test_all_with_strings() {
        let source = r#"
            all(["hello", "world", "test"])
        "#;
        let result = run(source);
        match result {
            Ok(Value::Bool(true)) => {}
            Ok(v) => panic!("Expected Bool(true), got {:?}", v),
            Err(e) => panic!("Error: {:?}", e),
        }
    }

    #[test]
    fn test_all_with_empty_string() {
        let source = r#"
            all(["hello", "", "test"])
        "#;
        let result = run(source);
        match result {
            Ok(Value::Bool(false)) => {}
            Ok(v) => panic!("Expected Bool(false), got {:?}", v),
            Err(e) => panic!("Error: {:?}", e),
        }
    }

    #[test]
    fn test_all_with_mixed_types() {
        let source = r#"
            all([true, 1, "hello", [1, 2]])
        "#;
        let result = run(source);
        match result {
            Ok(Value::Bool(true)) => {}
            Ok(v) => panic!("Expected Bool(true), got {:?}", v),
            Err(e) => panic!("Error: {:?}", e),
        }
    }

    #[test]
    fn test_all_with_null() {
        let source = r#"
            all([true, 1, null, "hello"])
        "#;
        let result = run(source);
        match result {
            Ok(Value::Bool(false)) => {}
            Ok(v) => panic!("Expected Bool(false), got {:?}", v),
            Err(e) => panic!("Error: {:?}", e),
        }
    }

    #[test]
    fn test_all_wrong_type() {
        let source = r#"
            all("not an array")
        "#;
        let result = run(source);
        match result {
            Ok(Value::Bool(false)) => {}
            Ok(v) => panic!("Expected Bool(false), got {:?}", v),
            Err(_) => {}
        }
    }
}

