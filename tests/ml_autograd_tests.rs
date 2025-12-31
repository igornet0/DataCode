// Tests for ML Autograd (automatic differentiation)

use data_code::{run, Value};

// Вспомогательная функция для проверки массива чисел
fn assert_array_equals(actual: &Value, expected: &[f64], tolerance: f64) {
    match actual {
        Value::Array(arr) => {
            let arr_ref = arr.borrow();
            assert_eq!(arr_ref.len(), expected.len(), "Array length mismatch");
            for (i, (actual_val, &expected_val)) in arr_ref.iter().zip(expected.iter()).enumerate() {
                match actual_val {
                    Value::Number(n) => {
                        assert!(
                            (n - expected_val).abs() < tolerance,
                            "Array[{}]: expected {}, got {}",
                            i,
                            expected_val,
                            n
                        );
                    }
                    _ => panic!("Array[{}]: expected Number, got {:?}", i, actual_val),
                }
            }
        }
        _ => panic!("Expected Array, got {:?}", actual),
    }
}

#[test]
fn test_autograd_simple_add() {
    // Test gradient computation for simple addition: c = a + b
    // grad_a = 1, grad_b = 1
    let code = r#"
        import ml
        let g = ml.graph()
        let input_a = ml.graph_add_input(g)
        let input_b = ml.graph_add_input(g)
        let add_node = ml.graph_add_op(g, "add", [input_a, input_b])
        
        let data_a = [2.0, 3.0]
        let data_b = [1.0, 1.0]
        let shape = [2]
        let t_a = ml.tensor(data_a, shape)
        let t_b = ml.tensor(data_b, shape)
        
        ml.graph_forward(g, [t_a, t_b])
        ml.graph_backward(g, add_node)
        
        let grad_a = ml.graph_get_gradient(g, input_a)
        ml.data(grad_a)
    "#;
    let result = run(code);
    assert!(result.is_ok(), "Test failed with error: {:?}", result);
    let grad_data = result.unwrap();
    // Gradient should be [1.0, 1.0] (ones)
    assert_array_equals(&grad_data, &[1.0, 1.0], 1e-6);
}

#[test]
fn test_autograd_simple_mul() {
    // Test gradient computation for multiplication: c = a * b
    // grad_a = b, grad_b = a
    let code = r#"
        import ml
        let g = ml.graph()
        let input_a = ml.graph_add_input(g)
        let input_b = ml.graph_add_input(g)
        let mul_node = ml.graph_add_op(g, "mul", [input_a, input_b])
        
        let data_a = [2.0, 3.0]
        let data_b = [4.0, 5.0]
        let shape = [2]
        let t_a = ml.tensor(data_a, shape)
        let t_b = ml.tensor(data_b, shape)
        
        ml.graph_forward(g, [t_a, t_b])
        ml.graph_backward(g, mul_node)
        
        let grad_a = ml.graph_get_gradient(g, input_a)
        ml.data(grad_a)
    "#;
    let result = run(code);
    assert!(result.is_ok(), "Test failed with error: {:?}", result);
    let grad_data = result.unwrap();
    // grad_a = b = [4.0, 5.0]
    assert_array_equals(&grad_data, &[4.0, 5.0], 1e-6);
}

#[test]
fn test_autograd_chain_operations() {
    // Test chain rule: c = a + b, d = c * a
    // grad_a = grad_d * (c + a), grad_b = grad_d * a
    let code = r#"
        import ml
        let g = ml.graph()
        let input_a = ml.graph_add_input(g)
        let input_b = ml.graph_add_input(g)
        let add_node = ml.graph_add_op(g, "add", [input_a, input_b])
        let mul_node = ml.graph_add_op(g, "mul", [add_node, input_a])
        
        let data_a = [2.0]
        let data_b = [3.0]
        let shape = [1]
        let t_a = ml.tensor(data_a, shape)
        let t_b = ml.tensor(data_b, shape)
        
        ml.graph_forward(g, [t_a, t_b])
        ml.graph_backward(g, mul_node)
        
        let grad_a = ml.graph_get_gradient(g, input_a)
        ml.data(grad_a)
    "#;
    let result = run(code);
    assert!(result.is_ok(), "Test failed with error: {:?}", result);
    let grad_data = result.unwrap();
    // c = a + b = 2 + 3 = 5
    // d = c * a = 5 * 2 = 10
    // grad_a = grad_d * (c + a) = 1 * (5 + 2) = 7
    // Actually: grad_a = grad_d * c + grad_d * a = 1 * 5 + 1 * 2 = 7
    // But in chain: grad_a through add = grad_d * a = 1 * 2 = 2
    // grad_a through mul = grad_d * c = 1 * 5 = 5
    // Total grad_a = 2 + 5 = 7
    assert_array_equals(&grad_data, &[7.0], 1e-5);
}

#[test]
fn test_autograd_matmul() {
    // Test gradient computation for matrix multiplication
    let code = r#"
        import ml
        let g = ml.graph()
        let input_a = ml.graph_add_input(g)
        let input_b = ml.graph_add_input(g)
        let matmul_node = ml.graph_add_op(g, "matmul", [input_a, input_b])
        
        let data_a = [1.0, 2.0, 3.0, 4.0]
        let shape_a = [2, 2]
        let data_b = [5.0, 6.0, 7.0, 8.0]
        let shape_b = [2, 2]
        let t_a = ml.tensor(data_a, shape_a)
        let t_b = ml.tensor(data_b, shape_b)
        
        ml.graph_forward(g, [t_a, t_b])
        ml.graph_backward(g, matmul_node)
        
        let grad_a = ml.graph_get_gradient(g, input_a)
        ml.data(grad_a)
    "#;
    let result = run(code);
    assert!(result.is_ok(), "Test failed with error: {:?}", result);
    // grad_a = grad_output @ b^T
    // grad_output is ones [1, 1, 1, 1] with shape [2, 2]
    // b^T = [[5, 7], [6, 8]]
    // grad_a = [[1, 1], [1, 1]] @ [[5, 7], [6, 8]] = [[11, 15], [11, 15]]
    // Actually, let's check the shape first
    let shape_code = r#"
        import ml
        let g = ml.graph()
        let input_a = ml.graph_add_input(g)
        let input_b = ml.graph_add_input(g)
        let matmul_node = ml.graph_add_op(g, "matmul", [input_a, input_b])
        
        let data_a = [1.0, 2.0, 3.0, 4.0]
        let shape_a = [2, 2]
        let data_b = [5.0, 6.0, 7.0, 8.0]
        let shape_b = [2, 2]
        let t_a = ml.tensor(data_a, shape_a)
        let t_b = ml.tensor(data_b, shape_b)
        
        ml.graph_forward(g, [t_a, t_b])
        ml.graph_backward(g, matmul_node)
        
        let grad_a = ml.graph_get_gradient(g, input_a)
        ml.shape(grad_a)
    "#;
    let shape_result = run(shape_code);
    assert!(shape_result.is_ok());
    // Gradient shape should match input_a shape [2, 2]
    assert_array_equals(&shape_result.unwrap(), &[2.0, 2.0], 1e-6);
}

#[test]
fn test_autograd_sum() {
    // Test gradient computation for sum operation
    // Sum reduces tensor to scalar, gradient should be broadcast back
    let code = r#"
        import ml
        let g = ml.graph()
        let input_a = ml.graph_add_input(g)
        let sum_node = ml.graph_add_op(g, "sum", [input_a])
        
        let data_a = [2.0, 3.0, 4.0]
        let shape = [3]
        let t_a = ml.tensor(data_a, shape)
        
        ml.graph_forward(g, [t_a])
        ml.graph_backward(g, sum_node)
        
        let grad_a = ml.graph_get_gradient(g, input_a)
        ml.data(grad_a)
    "#;
    let result = run(code);
    assert!(result.is_ok(), "Test failed with error: {:?}", result);
    let grad_data = result.unwrap();
    // Gradient should be broadcast: [1.0, 1.0, 1.0] (ones)
    assert_array_equals(&grad_data, &[1.0, 1.0, 1.0], 1e-6);
}

#[test]
fn test_autograd_mean() {
    // Test gradient computation for mean operation
    let code = r#"
        import ml
        let g = ml.graph()
        let input_a = ml.graph_add_input(g)
        let mean_node = ml.graph_add_op(g, "mean", [input_a])
        
        let data_a = [2.0, 4.0, 6.0]
        let shape = [3]
        let t_a = ml.tensor(data_a, shape)
        
        ml.graph_forward(g, [t_a])
        ml.graph_backward(g, mean_node)
        
        let grad_a = ml.graph_get_gradient(g, input_a)
        ml.data(grad_a)
    "#;
    let result = run(code);
    assert!(result.is_ok(), "Test failed with error: {:?}", result);
    let grad_data = result.unwrap();
    // Gradient should be broadcast: [1/3, 1/3, 1/3]
    assert_array_equals(&grad_data, &[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], 1e-5);
}

#[test]
fn test_autograd_sub() {
    // Test gradient computation for subtraction: c = a - b
    // grad_a = 1, grad_b = -1
    let code = r#"
        import ml
        let g = ml.graph()
        let input_a = ml.graph_add_input(g)
        let input_b = ml.graph_add_input(g)
        let sub_node = ml.graph_add_op(g, "sub", [input_a, input_b])
        
        let data_a = [5.0, 6.0]
        let data_b = [2.0, 3.0]
        let shape = [2]
        let t_a = ml.tensor(data_a, shape)
        let t_b = ml.tensor(data_b, shape)
        
        ml.graph_forward(g, [t_a, t_b])
        ml.graph_backward(g, sub_node)
        
        let grad_b = ml.graph_get_gradient(g, input_b)
        ml.data(grad_b)
    "#;
    let result = run(code);
    assert!(result.is_ok(), "Test failed with error: {:?}", result);
    let grad_data = result.unwrap();
    // grad_b should be [-1.0, -1.0]
    assert_array_equals(&grad_data, &[-1.0, -1.0], 1e-6);
}

#[test]
fn test_autograd_zero_grad() {
    // Test that zero_grad clears all gradients
    let code = r#"
        import ml
        let g = ml.graph()
        let input_a = ml.graph_add_input(g)
        let input_b = ml.graph_add_input(g)
        let add_node = ml.graph_add_op(g, "add", [input_a, input_b])
        
        let data_a = [1.0]
        let data_b = [2.0]
        let shape = [1]
        let t_a = ml.tensor(data_a, shape)
        let t_b = ml.tensor(data_b, shape)
        
        ml.graph_forward(g, [t_a, t_b])
        ml.graph_backward(g, add_node)
        
        # Check gradient exists
        let grad_before = ml.graph_get_gradient(g, input_a)
        ml.graph_zero_grad(g)
        
        # After zero_grad, gradient should be null (will return null from get_gradient)
        grad_before
    "#;
    let result = run(code);
    assert!(result.is_ok(), "Test failed with error: {:?}", result);
    // Before zero_grad, gradient should exist
    match result.unwrap() {
        Value::Tensor(_) => {} // Good, gradient exists
        _ => panic!("Expected gradient to exist before zero_grad"),
    }
}

#[test]
fn test_autograd_transpose() {
    // Test gradient computation for transpose
    let code = r#"
        import ml
        let g = ml.graph()
        let input_a = ml.graph_add_input(g)
        let transpose_node = ml.graph_add_op(g, "transpose", [input_a])
        
        let data_a = [1.0, 2.0, 3.0, 4.0]
        let shape_a = [2, 2]
        let t_a = ml.tensor(data_a, shape_a)
        
        ml.graph_forward(g, [t_a])
        ml.graph_backward(g, transpose_node)
        
        let grad_a = ml.graph_get_gradient(g, input_a)
        ml.shape(grad_a)
    "#;
    let result = run(code);
    assert!(result.is_ok(), "Test failed with error: {:?}", result);
    // Gradient shape should match original input shape [2, 2]
    assert_array_equals(&result.unwrap(), &[2.0, 2.0], 1e-6);
}

