// Tests for ML Graph module

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
fn test_graph_creation() {
    let code = r#"
        import ml
        let g = ml.graph()
        print("Graph created")
    "#;
    let result = run(code);
    assert!(result.is_ok());
}

#[test]
fn test_graph_add_input() {
    let code = r#"
        import ml
        let g = ml.graph()
        let input1 = ml.graph_add_input(g)
        let input2 = ml.graph_add_input(g)
        input2
    "#;
    let result = run(code);
    assert!(result.is_ok());
    match result.unwrap() {
        Value::Number(n) => {
            assert!((n - 1.0).abs() < 1e-6, "Expected node ID 1 (second input), got {}", n);
        }
        v => panic!("Expected Number (node ID), got {:?}", v),
    }
}

#[test]
fn test_graph_simple_forward() {
    // Simple forward pass: input -> output (no operations)
    let code = r#"
        import ml
        let g = ml.graph()
        let input1 = ml.graph_add_input(g)
        let data = [1.0, 2.0, 3.0]
        let shape = [3]
        let t = ml.tensor(data, shape)
        ml.graph_forward(g, [t])
        let result = ml.graph_get_output(g, input1)
        ml.data(result)
    "#;
    let result = run(code);
    assert!(result.is_ok(), "Test failed with error: {:?}", result);
    let data = result.unwrap();
    assert_array_equals(&data, &[1.0, 2.0, 3.0]);
}

#[test]
fn test_graph_add_operation() {
    // Forward pass with Add operation
    let code = r#"
        import ml
        let g = ml.graph()
        let input1 = ml.graph_add_input(g)
        let input2 = ml.graph_add_input(g)
        let add_node = ml.graph_add_op(g, "add", [input1, input2])
        let data1 = [1.0, 2.0, 3.0]
        let data2 = [4.0, 5.0, 6.0]
        let shape = [3]
        let t1 = ml.tensor(data1, shape)
        let t2 = ml.tensor(data2, shape)
        ml.graph_forward(g, [t1, t2])
        let result = ml.graph_get_output(g, add_node)
        ml.data(result)
    "#;
    let result = run(code);
    assert!(result.is_ok(), "Test failed with error: {:?}", result);
    let data = result.unwrap();
    assert_array_equals(&data, &[5.0, 7.0, 9.0]);
}

#[test]
fn test_graph_matmul_forward() {
    // Forward pass with MatMul operation
    let code = r#"
        import ml
        let g = ml.graph()
        let input1 = ml.graph_add_input(g)
        let input2 = ml.graph_add_input(g)
        let matmul_node = ml.graph_add_op(g, "matmul", [input1, input2])
        let data1 = [1.0, 2.0, 3.0, 4.0]
        let shape1 = [2, 2]
        let data2 = [5.0, 6.0, 7.0, 8.0]
        let shape2 = [2, 2]
        let t1 = ml.tensor(data1, shape1)
        let t2 = ml.tensor(data2, shape2)
        ml.graph_forward(g, [t1, t2])
        let result = ml.graph_get_output(g, matmul_node)
        ml.data(result)
    "#;
    let result = run(code);
    assert!(result.is_ok(), "Test failed with error: {:?}", result);
    let data = result.unwrap();
    // [[1, 2], [3, 4]] @ [[5, 6], [7, 8]] = [[19, 22], [43, 50]]
    assert_array_equals(&data, &[19.0, 22.0, 43.0, 50.0]);
}

#[test]
fn test_graph_chain_operations() {
    // Chain of operations: Add -> Mul
    let code = r#"
        import ml
        let g = ml.graph()
        let input1 = ml.graph_add_input(g)
        let input2 = ml.graph_add_input(g)
        let add_node = ml.graph_add_op(g, "add", [input1, input2])
        let mul_node = ml.graph_add_op(g, "mul", [add_node, input1])
        let data1 = [1.0, 2.0]
        let data2 = [3.0, 4.0]
        let shape = [2]
        let t1 = ml.tensor(data1, shape)
        let t2 = ml.tensor(data2, shape)
        ml.graph_forward(g, [t1, t2])
        let result = ml.graph_get_output(g, mul_node)
        ml.data(result)
    "#;
    let result = run(code);
    assert!(result.is_ok(), "Test failed with error: {:?}", result);
    let data = result.unwrap();
    // (1+3)*1 = 4, (2+4)*2 = 12
    assert_array_equals(&data, &[4.0, 12.0]);
}

#[test]
fn test_graph_multiple_inputs() {
    // Multiple inputs with different operations
    let code = r#"
        import ml
        let g = ml.graph()
        let input1 = ml.graph_add_input(g)
        let input2 = ml.graph_add_input(g)
        let input3 = ml.graph_add_input(g)
        let add_node = ml.graph_add_op(g, "add", [input1, input2])
        let mul_node = ml.graph_add_op(g, "mul", [add_node, input3])
        let data1 = [2.0, 3.0]
        let data2 = [1.0, 1.0]
        let data3 = [5.0, 5.0]
        let shape = [2]
        let t1 = ml.tensor(data1, shape)
        let t2 = ml.tensor(data2, shape)
        let t3 = ml.tensor(data3, shape)
        ml.graph_forward(g, [t1, t2, t3])
        let result = ml.graph_get_output(g, mul_node)
        ml.data(result)
    "#;
    let result = run(code);
    assert!(result.is_ok(), "Test failed with error: {:?}", result);
    let data = result.unwrap();
    // (2+1)*5 = 15, (3+1)*5 = 20
    assert_array_equals(&data, &[15.0, 20.0]);
}

#[test]
fn test_graph_sub_operation() {
    // Forward pass with Sub operation
    let code = r#"
        import ml
        let g = ml.graph()
        let input1 = ml.graph_add_input(g)
        let input2 = ml.graph_add_input(g)
        let sub_node = ml.graph_add_op(g, "sub", [input1, input2])
        let data1 = [5.0, 6.0, 7.0]
        let data2 = [1.0, 2.0, 3.0]
        let shape = [3]
        let t1 = ml.tensor(data1, shape)
        let t2 = ml.tensor(data2, shape)
        ml.graph_forward(g, [t1, t2])
        let result = ml.graph_get_output(g, sub_node)
        ml.data(result)
    "#;
    let result = run(code);
    assert!(result.is_ok(), "Test failed with error: {:?}", result);
    let data = result.unwrap();
    assert_array_equals(&data, &[4.0, 4.0, 4.0]);
}

#[test]
fn test_graph_transpose_operation() {
    // Forward pass with Transpose operation
    let code = r#"
        import ml
        let g = ml.graph()
        let input1 = ml.graph_add_input(g)
        let transpose_node = ml.graph_add_op(g, "transpose", [input1])
        let data = [1.0, 2.0, 3.0, 4.0]
        let shape = [2, 2]
        let t = ml.tensor(data, shape)
        ml.graph_forward(g, [t])
        let result = ml.graph_get_output(g, transpose_node)
        ml.data(result)
    "#;
    let result = run(code);
    assert!(result.is_ok(), "Test failed with error: {:?}", result);
    let data = result.unwrap();
    // [[1, 2], [3, 4]]^T = [[1, 3], [2, 4]]
    assert_array_equals(&data, &[1.0, 3.0, 2.0, 4.0]);
}

#[test]
fn test_graph_sum_operation() {
    // Forward pass with Sum operation
    let code = r#"
        import ml
        let g = ml.graph()
        let input1 = ml.graph_add_input(g)
        let sum_node = ml.graph_add_op(g, "sum", [input1])
        let data = [1.0, 2.0, 3.0, 4.0]
        let shape = [4]
        let t = ml.tensor(data, shape)
        ml.graph_forward(g, [t])
        let result = ml.graph_get_output(g, sum_node)
        ml.data(result)
    "#;
    let result = run(code);
    assert!(result.is_ok(), "Test failed with error: {:?}", result);
    let data = result.unwrap();
    // Sum wraps result in tensor with shape [1]
    assert_array_equals(&data, &[10.0]);
}

#[test]
fn test_graph_mean_operation() {
    // Forward pass with Mean operation
    let code = r#"
        import ml
        let g = ml.graph()
        let input1 = ml.graph_add_input(g)
        let mean_node = ml.graph_add_op(g, "mean", [input1])
        let data = [2.0, 4.0, 6.0, 8.0]
        let shape = [4]
        let t = ml.tensor(data, shape)
        ml.graph_forward(g, [t])
        let result = ml.graph_get_output(g, mean_node)
        ml.data(result)
    "#;
    let result = run(code);
    assert!(result.is_ok(), "Test failed with error: {:?}", result);
    let data = result.unwrap();
    // Mean wraps result in tensor with shape [1]
    assert_array_equals(&data, &[5.0]);
}

#[test]
fn test_graph_complex_chain() {
    // Complex chain: input1 + input2 -> transpose -> matmul with input1
    let code = r#"
        import ml
        let g = ml.graph()
        let input1 = ml.graph_add_input(g)
        let input2 = ml.graph_add_input(g)
        let add_node = ml.graph_add_op(g, "add", [input1, input2])
        let transpose_node = ml.graph_add_op(g, "transpose", [add_node])
        let matmul_node = ml.graph_add_op(g, "matmul", [transpose_node, input1])
        let data1 = [1.0, 2.0, 3.0, 4.0]
        let shape1 = [2, 2]
        let data2 = [1.0, 1.0, 1.0, 1.0]
        let shape2 = [2, 2]
        let t1 = ml.tensor(data1, shape1)
        let t2 = ml.tensor(data2, shape2)
        ml.graph_forward(g, [t1, t2])
        let result = ml.graph_get_output(g, matmul_node)
        ml.shape(result)
    "#;
    let result = run(code);
    assert!(result.is_ok(), "Test failed with error: {:?}", result);
    let shape = result.unwrap();
    // Result should be 2x2 matrix
    assert_array_equals(&shape, &[2.0, 2.0]);
}

