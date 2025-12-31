// Tests for ML Tensor module

use data_code::{run, Value};

// Вспомогательная функция для проверки массива чисел
fn assert_array_equals(actual: &Value, expected: &[f64]) {
    match actual {
        Value::Array(arr) => {
            let arr_ref = arr.borrow();
            assert_eq!(arr_ref.len(), expected.len(), "Array length mismatch");
            for (i, (actual_val, &expected_val)) in arr_ref.iter().zip(expected.iter()).enumerate() {
                match actual_val {
                    Value::Number(n) => {
                        assert!((n - expected_val).abs() < 1e-6, 
                            "Array[{}]: expected {}, got {}", i, expected_val, n);
                    }
                    _ => panic!("Array[{}]: expected Number, got {:?}", i, actual_val),
                }
            }
        }
        _ => panic!("Expected Array, got {:?}", actual),
    }
}

#[test]
fn test_tensor_creation() {
    // Проверяем, что тензор создается и можно получить его данные
    let code = r#"
        import ml
        let data = [1.0, 2.0, 3.0, 4.0]
        let shape = [4]
        let t = ml.tensor(data, shape)
        ml.data(t)
    "#;
    let result = run(code);
    assert!(result.is_ok(), "Test failed with error: {:?}", result);
    let tensor_data = result.unwrap();
    assert_array_equals(&tensor_data, &[1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn test_tensor_shape() {
    let code = r#"
        import ml
        let data = [1.0, 2.0, 3.0, 4.0]
        let shape = [4]
        let t = ml.tensor(data, shape)
        ml.shape(t)
    "#;
    let result = run(code);
    assert!(result.is_ok(), "Test failed with error: {:?}", result);
    let shape = result.unwrap();
    assert_array_equals(&shape, &[4.0]);
}

#[test]
fn test_tensor_data() {
    let code = r#"
        import ml
        let data = [1.0, 2.0, 3.0]
        let shape = [3]
        let t = ml.tensor(data, shape)
        ml.data(t)
    "#;
    let result = run(code);
    assert!(result.is_ok(), "Test failed with error: {:?}", result);
    let data = result.unwrap();
    assert_array_equals(&data, &[1.0, 2.0, 3.0]);
}

#[test]
fn test_tensor_add() {
    let code = r#"
        import ml
        let data1 = [1.0, 2.0, 3.0]
        let data2 = [4.0, 5.0, 6.0]
        let shape = [3]
        let t1 = ml.tensor(data1, shape)
        let t2 = ml.tensor(data2, shape)
        let result = ml.add(t1, t2)
        ml.data(result)
    "#;
    let result = run(code);
    assert!(result.is_ok(), "Test failed with error: {:?}", result);
    let result_data = result.unwrap();
    assert_array_equals(&result_data, &[5.0, 7.0, 9.0]);
}

#[test]
fn test_tensor_sub() {
    let code = r#"
        import ml
        let data1 = [5.0, 6.0, 7.0]
        let data2 = [1.0, 2.0, 3.0]
        let shape = [3]
        let t1 = ml.tensor(data1, shape)
        let t2 = ml.tensor(data2, shape)
        let result = ml.sub(t1, t2)
        ml.data(result)
    "#;
    let result = run(code);
    assert!(result.is_ok(), "Test failed with error: {:?}", result);
    let result_data = result.unwrap();
    assert_array_equals(&result_data, &[4.0, 4.0, 4.0]);
}

#[test]
fn test_tensor_mul() {
    let code = r#"
        import ml
        let data1 = [2.0, 3.0, 4.0]
        let data2 = [5.0, 6.0, 7.0]
        let shape = [3]
        let t1 = ml.tensor(data1, shape)
        let t2 = ml.tensor(data2, shape)
        let result = ml.mul(t1, t2)
        ml.data(result)
    "#;
    let result = run(code);
    assert!(result.is_ok(), "Test failed with error: {:?}", result);
    let result_data = result.unwrap();
    assert_array_equals(&result_data, &[10.0, 18.0, 28.0]);
}

#[test]
fn test_tensor_matmul() {
    let code = r#"
        import ml
        let data1 = [1.0, 2.0, 3.0, 4.0]
        let shape1 = [2, 2]
        let data2 = [5.0, 6.0, 7.0, 8.0]
        let shape2 = [2, 2]
        let t1 = ml.tensor(data1, shape1)
        let t2 = ml.tensor(data2, shape2)
        let result = ml.matmul(t1, t2)
        let result_shape = ml.shape(result)
        let result_data = ml.data(result)
        result_shape
    "#;
    let result = run(code);
    assert!(result.is_ok(), "Test failed with error: {:?}", result);
    let shape = result.unwrap();
    assert_array_equals(&shape, &[2.0, 2.0]);
    
    // Проверяем данные матричного умножения
    // [[1, 2], [3, 4]] @ [[5, 6], [7, 8]] = [[19, 22], [43, 50]]
    let data_code = r#"
        import ml
        let data1 = [1.0, 2.0, 3.0, 4.0]
        let shape1 = [2, 2]
        let data2 = [5.0, 6.0, 7.0, 8.0]
        let shape2 = [2, 2]
        let t1 = ml.tensor(data1, shape1)
        let t2 = ml.tensor(data2, shape2)
        let result = ml.matmul(t1, t2)
        ml.data(result)
    "#;
    let data_result = run(data_code);
    assert!(data_result.is_ok(), "Test failed with error: {:?}", data_result);
    let data = data_result.unwrap();
    assert_array_equals(&data, &[19.0, 22.0, 43.0, 50.0]);
}

#[test]
fn test_tensor_transpose() {
    // Проверяем форму после транспонирования
    let code = r#"
        import ml
        let data = [1.0, 2.0, 3.0, 4.0]
        let shape = [2, 2]
        let t = ml.tensor(data, shape)
        let transposed = ml.transpose(t)
        ml.shape(transposed)
    "#;
    let result = run(code);
    assert!(result.is_ok(), "Test failed with error: {:?}", result);
    let shape = result.unwrap();
    assert_array_equals(&shape, &[2.0, 2.0]);
    
    // Проверяем данные после транспонирования
    // [[1, 2], [3, 4]]^T = [[1, 3], [2, 4]]
    let data_code = r#"
        import ml
        let data = [1.0, 2.0, 3.0, 4.0]
        let shape = [2, 2]
        let t = ml.tensor(data, shape)
        let transposed = ml.transpose(t)
        ml.data(transposed)
    "#;
    let data_result = run(data_code);
    assert!(data_result.is_ok(), "Test failed with error: {:?}", data_result);
    let data = data_result.unwrap();
    assert_array_equals(&data, &[1.0, 3.0, 2.0, 4.0]);
}

#[test]
fn test_tensor_sum() {
    let code = r#"
        import ml
        let data = [1.0, 2.0, 3.0, 4.0]
        let shape = [4]
        let t = ml.tensor(data, shape)
        ml.sum(t)
    "#;
    let result = run(code);
    assert!(result.is_ok(), "Test failed with error: {:?}", result);
    match result.unwrap() {
        Value::Number(n) => {
            assert!((n - 10.0).abs() < 1e-6, "Expected sum 10.0, got {}", n);
        }
        v => panic!("Expected Number, got {:?}", v),
    }
}

#[test]
fn test_tensor_mean() {
    let code = r#"
        import ml
        let data = [2.0, 4.0, 6.0, 8.0]
        let shape = [4]
        let t = ml.tensor(data, shape)
        ml.mean(t)
    "#;
    let result = run(code);
    assert!(result.is_ok(), "Test failed with error: {:?}", result);
    match result.unwrap() {
        Value::Number(n) => {
            assert!((n - 5.0).abs() < 1e-6, "Expected mean 5.0, got {}", n);
        }
        v => panic!("Expected Number, got {:?}", v),
    }
}

#[test]
fn test_tensor_2d_operations() {
    // Проверяем форму после операции
    let code = r#"
        import ml
        let data1 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        let shape1 = [2, 3]
        let data2 = [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
        let shape2 = [2, 3]
        let t1 = ml.tensor(data1, shape1)
        let t2 = ml.tensor(data2, shape2)
        let result = ml.add(t1, t2)
        ml.shape(result)
    "#;
    let result = run(code);
    assert!(result.is_ok(), "Test failed with error: {:?}", result);
    let shape = result.unwrap();
    assert_array_equals(&shape, &[2.0, 3.0]);
    
    // Проверяем данные после сложения
    let data_code = r#"
        import ml
        let data1 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        let shape1 = [2, 3]
        let data2 = [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
        let shape2 = [2, 3]
        let t1 = ml.tensor(data1, shape1)
        let t2 = ml.tensor(data2, shape2)
        let result = ml.add(t1, t2)
        ml.data(result)
    "#;
    let data_result = run(data_code);
    assert!(data_result.is_ok(), "Test failed with error: {:?}", data_result);
    let data = data_result.unwrap();
    assert_array_equals(&data, &[8.0, 10.0, 12.0, 14.0, 16.0, 18.0]);
}

#[test]
fn test_multiple_imports() {
    let code = r#"
        import ml
        import ml
        let data = [1.0, 2.0, 3.0]
        let shape = [3]
        let t = ml.tensor(data, shape)
        print("Multiple imports work")
    "#;
    let result = run(code);
    assert!(result.is_ok());
}

#[test]
fn test_tensor_chain_operations() {
    // Цепочка: (t1 + t2) * t1, затем sum
    // t1 = [1, 2], t2 = [3, 4]
    // added = [4, 6]
    // multiplied = [4, 6] * [1, 2] = [4, 12]
    // sum = 16
    let code = r#"
        import ml
        let data1 = [1.0, 2.0]
        let data2 = [3.0, 4.0]
        let shape = [2]
        let t1 = ml.tensor(data1, shape)
        let t2 = ml.tensor(data2, shape)
        let added = ml.add(t1, t2)
        let multiplied = ml.mul(added, t1)
        ml.sum(multiplied)
    "#;
    let result = run(code);
    assert!(result.is_ok(), "Test failed with error: {:?}", result);
    match result.unwrap() {
        Value::Number(n) => {
            assert!((n - 16.0).abs() < 1e-6, "Expected sum 16.0, got {}", n);
        }
        v => panic!("Expected Number, got {:?}", v),
    }
}

