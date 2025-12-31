// Tests for ML Loss Functions

use data_code::{run, Value};

#[test]
fn test_mse_loss_perfect_match() {
    // Test MSE loss when predictions match targets exactly
    let code = r#"
        import ml
        let y_pred_data = [1.0, 2.0, 3.0]
        let y_pred_shape = [3, 1]
        let y_pred = ml.tensor(y_pred_data, y_pred_shape)
        
        let y_true_data = [1.0, 2.0, 3.0]
        let y_true_shape = [3, 1]
        let y_true = ml.tensor(y_true_data, y_true_shape)
        
        let loss = ml.mse_loss(y_pred, y_true)
        ml.data(loss)[0]
    "#;
    let result = run(code);
    assert!(result.is_ok(), "Test failed with error: {:?}", result);
    match result.unwrap() {
        Value::Number(n) => {
            assert!((n - 0.0).abs() < 1e-6, "MSE should be 0.0 for perfect match, got {}", n);
        }
        _ => panic!("Expected Number for loss value"),
    }
}

#[test]
fn test_mse_loss_difference() {
    // Test MSE loss with difference
    // y_pred = [2, 3, 4], y_true = [1, 2, 3]
    // MSE = mean((1^2, 1^2, 1^2)) = mean(1, 1, 1) = 1.0
    let code = r#"
        import ml
        let y_pred_data = [2.0, 3.0, 4.0]
        let y_pred_shape = [3, 1]
        let y_pred = ml.tensor(y_pred_data, y_pred_shape)
        
        let y_true_data = [1.0, 2.0, 3.0]
        let y_true_shape = [3, 1]
        let y_true = ml.tensor(y_true_data, y_true_shape)
        
        let loss = ml.mse_loss(y_pred, y_true)
        ml.data(loss)[0]
    "#;
    let result = run(code);
    assert!(result.is_ok(), "Test failed with error: {:?}", result);
    match result.unwrap() {
        Value::Number(n) => {
            assert!((n - 1.0).abs() < 1e-5, "MSE should be approximately 1.0, got {}", n);
        }
        _ => panic!("Expected Number for loss value"),
    }
}

#[test]
fn test_binary_cross_entropy_loss() {
    // Test binary cross entropy loss
    let code = r#"
        import ml
        let y_pred_data = [0.7, 0.3, 0.9]
        let y_pred_shape = [3, 1]
        let y_pred = ml.tensor(y_pred_data, y_pred_shape)
        
        let y_true_data = [1.0, 0.0, 1.0]
        let y_true_shape = [3, 1]
        let y_true = ml.tensor(y_true_data, y_true_shape)
        
        let loss = ml.binary_cross_entropy_loss(y_pred, y_true)
        ml.data(loss)[0]
    "#;
    let result = run(code);
    assert!(result.is_ok(), "Test failed with error: {:?}", result);
    match result.unwrap() {
        Value::Number(n) => {
            assert!(n > 0.0, "BCE loss should be positive, got {}", n);
            assert!(n < 10.0, "BCE loss should be reasonable, got {}", n);
        }
        _ => panic!("Expected Number for loss value"),
    }
}

#[test]
fn test_cross_entropy_loss() {
    // Test cross entropy loss for multi-class classification
    // y_pred: logits [batch=2, classes=3]
    // y_true: one-hot [batch=2, classes=3]
    let code = r#"
        import ml
        let y_pred_data = [0.5, 0.3, 0.2, 0.1, 0.7, 0.2]
        let y_pred_shape = [2, 3]
        let y_pred = ml.tensor(y_pred_data, y_pred_shape)
        
        let y_true_data = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        let y_true_shape = [2, 3]
        let y_true = ml.tensor(y_true_data, y_true_shape)
        
        let loss = ml.categorical_cross_entropy_loss(y_pred, y_true)
        ml.data(loss)[0]
    "#;
    let result = run(code);
    assert!(result.is_ok(), "Test failed with error: {:?}", result);
    match result.unwrap() {
        Value::Number(n) => {
            assert!(n > 0.0, "Cross entropy loss should be positive, got {}", n);
        }
        _ => panic!("Expected Number for loss value"),
    }
}

#[test]
fn test_mae_loss_perfect_match() {
    // Test MAE loss when predictions match targets exactly
    let code = r#"
        import ml
        let y_pred_data = [1.0, 2.0, 3.0]
        let y_pred_shape = [3, 1]
        let y_pred = ml.tensor(y_pred_data, y_pred_shape)
        
        let y_true_data = [1.0, 2.0, 3.0]
        let y_true_shape = [3, 1]
        let y_true = ml.tensor(y_true_data, y_true_shape)
        
        let loss = ml.mae_loss(y_pred, y_true)
        ml.data(loss)[0]
    "#;
    let result = run(code);
    assert!(result.is_ok(), "Test failed with error: {:?}", result);
    match result.unwrap() {
        Value::Number(n) => {
            assert!((n - 0.0).abs() < 1e-6, "MAE should be 0.0 for perfect match, got {}", n);
        }
        _ => panic!("Expected Number for loss value"),
    }
}

#[test]
fn test_mae_loss_difference() {
    // Test MAE loss with difference
    // y_pred = [2, 3, 4], y_true = [1, 2, 3]
    // MAE = mean(|1|, |1|, |1|) = mean(1, 1, 1) = 1.0
    let code = r#"
        import ml
        let y_pred_data = [2.0, 3.0, 4.0]
        let y_pred_shape = [3, 1]
        let y_pred = ml.tensor(y_pred_data, y_pred_shape)
        
        let y_true_data = [1.0, 2.0, 3.0]
        let y_true_shape = [3, 1]
        let y_true = ml.tensor(y_true_data, y_true_shape)
        
        let loss = ml.mae_loss(y_pred, y_true)
        ml.data(loss)[0]
    "#;
    let result = run(code);
    assert!(result.is_ok(), "Test failed with error: {:?}", result);
    match result.unwrap() {
        Value::Number(n) => {
            assert!((n - 1.0).abs() < 1e-5, "MAE should be approximately 1.0, got {}", n);
        }
        _ => panic!("Expected Number for loss value"),
    }
}

#[test]
fn test_huber_loss_small_errors() {
    // Test Huber loss with small errors (L2 region)
    // delta = 1.0, errors = [0.5, 0.3, 0.2] (all < delta)
    // Should use quadratic form: 0.5 * error^2
    let code = r#"
        import ml
        let y_pred_data = [1.5, 2.3, 3.2]
        let y_pred_shape = [3, 1]
        let y_pred = ml.tensor(y_pred_data, y_pred_shape)
        
        let y_true_data = [1.0, 2.0, 3.0]
        let y_true_shape = [3, 1]
        let y_true = ml.tensor(y_true_data, y_true_shape)
        
        let loss = ml.huber_loss(y_pred, y_true, 1.0)
        ml.data(loss)[0]
    "#;
    let result = run(code);
    assert!(result.is_ok(), "Test failed with error: {:?}", result);
    match result.unwrap() {
        Value::Number(n) => {
            assert!(n > 0.0, "Huber loss should be positive, got {}", n);
            assert!(n < 1.0, "Huber loss for small errors should be small, got {}", n);
        }
        _ => panic!("Expected Number for loss value"),
    }
}

#[test]
fn test_huber_loss_large_errors() {
    // Test Huber loss with large errors (L1 region)
    // delta = 1.0, errors = [2.0, 3.0, 4.0] (all > delta)
    // Should use linear form: delta * |error| - 0.5 * delta^2
    let code = r#"
        import ml
        let y_pred_data = [3.0, 5.0, 7.0]
        let y_pred_shape = [3, 1]
        let y_pred = ml.tensor(y_pred_data, y_pred_shape)
        
        let y_true_data = [1.0, 2.0, 3.0]
        let y_true_shape = [3, 1]
        let y_true = ml.tensor(y_true_data, y_true_shape)
        
        let loss = ml.huber_loss(y_pred, y_true, 1.0)
        ml.data(loss)[0]
    "#;
    let result = run(code);
    assert!(result.is_ok(), "Test failed with error: {:?}", result);
    match result.unwrap() {
        Value::Number(n) => {
            assert!(n > 0.0, "Huber loss should be positive, got {}", n);
            // For large errors, loss should be approximately mean(1*2-0.5, 1*3-0.5, 1*4-0.5) = mean(1.5, 2.5, 3.5) = 2.5
            assert!(n > 1.0, "Huber loss for large errors should be larger, got {}", n);
        }
        _ => panic!("Expected Number for loss value"),
    }
}

#[test]
fn test_huber_loss_default_delta() {
    // Test Huber loss with default delta (should default to 1.0)
    let code = r#"
        import ml
        let y_pred_data = [1.5, 2.3, 3.2]
        let y_pred_shape = [3, 1]
        let y_pred = ml.tensor(y_pred_data, y_pred_shape)
        
        let y_true_data = [1.0, 2.0, 3.0]
        let y_true_shape = [3, 1]
        let y_true = ml.tensor(y_true_data, y_true_shape)
        
        let loss = ml.huber_loss(y_pred, y_true)
        ml.data(loss)[0]
    "#;
    let result = run(code);
    assert!(result.is_ok(), "Test failed with error: {:?}", result);
    match result.unwrap() {
        Value::Number(n) => {
            assert!(n > 0.0, "Huber loss should be positive, got {}", n);
        }
        _ => panic!("Expected Number for loss value"),
    }
}

#[test]
fn test_hinge_loss_correct_predictions() {
    // Test Hinge loss with correct predictions
    // y_true = [1, -1, 1], y_pred = [2, -2, 1.5]
    // margin = 1 - y_true * y_pred = [1-2, 1-(-2*-1), 1-1.5] = [-1, -1, -0.5]
    // max(0, margin) = [0, 0, 0] -> loss = 0
    let code = r#"
        import ml
        let y_pred_data = [2.0, -2.0, 1.5]
        let y_pred_shape = [3, 1]
        let y_pred = ml.tensor(y_pred_data, y_pred_shape)
        
        let y_true_data = [1.0, -1.0, 1.0]
        let y_true_shape = [3, 1]
        let y_true = ml.tensor(y_true_data, y_true_shape)
        
        let loss = ml.hinge_loss(y_pred, y_true)
        ml.data(loss)[0]
    "#;
    let result = run(code);
    assert!(result.is_ok(), "Test failed with error: {:?}", result);
    match result.unwrap() {
        Value::Number(n) => {
            assert!((n - 0.0).abs() < 1e-5, "Hinge loss should be 0 for correct predictions, got {}", n);
        }
        _ => panic!("Expected Number for loss value"),
    }
}

#[test]
fn test_hinge_loss_incorrect_predictions() {
    // Test Hinge loss with incorrect predictions
    // y_true = [1, -1, 1], y_pred = [-1, 1, -0.5]
    // margin = 1 - y_true * y_pred = [1-(-1), 1-(-1*1), 1-(-0.5)] = [2, 2, 1.5]
    // max(0, margin) = [2, 2, 1.5] -> loss = mean(2, 2, 1.5) = 1.833...
    let code = r#"
        import ml
        let y_pred_data = [-1.0, 1.0, -0.5]
        let y_pred_shape = [3, 1]
        let y_pred = ml.tensor(y_pred_data, y_pred_shape)
        
        let y_true_data = [1.0, -1.0, 1.0]
        let y_true_shape = [3, 1]
        let y_true = ml.tensor(y_true_data, y_true_shape)
        
        let loss = ml.hinge_loss(y_pred, y_true)
        ml.data(loss)[0]
    "#;
    let result = run(code);
    assert!(result.is_ok(), "Test failed with error: {:?}", result);
    match result.unwrap() {
        Value::Number(n) => {
            assert!(n > 0.0, "Hinge loss should be positive for incorrect predictions, got {}", n);
            assert!((n - 1.833333).abs() < 0.1, "Hinge loss should be approximately 1.833, got {}", n);
        }
        _ => panic!("Expected Number for loss value"),
    }
}

#[test]
fn test_kl_divergence_identical() {
    // Test KL divergence with identical distributions (should be 0)
    let code = r#"
        import ml
        let y_pred_data = [0.5, 0.3, 0.2]
        let y_pred_shape = [3, 1]
        let y_pred = ml.tensor(y_pred_data, y_pred_shape)
        
        let y_true_data = [0.5, 0.3, 0.2]
        let y_true_shape = [3, 1]
        let y_true = ml.tensor(y_true_data, y_true_shape)
        
        let loss = ml.kl_divergence(y_pred, y_true)
        ml.data(loss)[0]
    "#;
    let result = run(code);
    assert!(result.is_ok(), "Test failed with error: {:?}", result);
    match result.unwrap() {
        Value::Number(n) => {
            assert!((n - 0.0).abs() < 1e-5, "KL divergence should be 0 for identical distributions, got {}", n);
        }
        _ => panic!("Expected Number for loss value"),
    }
}

#[test]
fn test_kl_divergence_different() {
    // Test KL divergence with different distributions
    let code = r#"
        import ml
        let y_pred_data = [0.7, 0.2, 0.1]
        let y_pred_shape = [3, 1]
        let y_pred = ml.tensor(y_pred_data, y_pred_shape)
        
        let y_true_data = [0.5, 0.3, 0.2]
        let y_true_shape = [3, 1]
        let y_true = ml.tensor(y_true_data, y_true_shape)
        
        let loss = ml.kl_divergence(y_pred, y_true)
        ml.data(loss)[0]
    "#;
    let result = run(code);
    assert!(result.is_ok(), "Test failed with error: {:?}", result);
    match result.unwrap() {
        Value::Number(n) => {
            assert!(n > 0.0, "KL divergence should be positive for different distributions, got {}", n);
        }
        _ => panic!("Expected Number for loss value"),
    }
}

#[test]
fn test_smooth_l1_loss_small_differences() {
    // Test Smooth L1 loss with small differences (quadratic region)
    // differences = [0.5, 0.3, 0.2] (all < 1.0)
    // loss = mean(0.5 * 0.5^2, 0.5 * 0.3^2, 0.5 * 0.2^2) = mean(0.125, 0.045, 0.02) ≈ 0.063
    let code = r#"
        import ml
        let y_pred_data = [1.5, 2.3, 3.2]
        let y_pred_shape = [3, 1]
        let y_pred = ml.tensor(y_pred_data, y_pred_shape)
        
        let y_true_data = [1.0, 2.0, 3.0]
        let y_true_shape = [3, 1]
        let y_true = ml.tensor(y_true_data, y_true_shape)
        
        let loss = ml.smooth_l1_loss(y_pred, y_true)
        ml.data(loss)[0]
    "#;
    let result = run(code);
    assert!(result.is_ok(), "Test failed with error: {:?}", result);
    match result.unwrap() {
        Value::Number(n) => {
            assert!(n > 0.0, "Smooth L1 loss should be positive, got {}", n);
            assert!(n < 0.1, "Smooth L1 loss for small differences should be small, got {}", n);
        }
        _ => panic!("Expected Number for loss value"),
    }
}

#[test]
fn test_smooth_l1_loss_large_differences() {
    // Test Smooth L1 loss with large differences (linear region)
    // differences = [2.0, 3.0, 4.0] (all >= 1.0)
    // loss = mean(2.0 - 0.5, 3.0 - 0.5, 4.0 - 0.5) = mean(1.5, 2.5, 3.5) = 2.5
    let code = r#"
        import ml
        let y_pred_data = [3.0, 5.0, 7.0]
        let y_pred_shape = [3, 1]
        let y_pred = ml.tensor(y_pred_data, y_pred_shape)
        
        let y_true_data = [1.0, 2.0, 3.0]
        let y_true_shape = [3, 1]
        let y_true = ml.tensor(y_true_data, y_true_shape)
        
        let loss = ml.smooth_l1_loss(y_pred, y_true)
        ml.data(loss)[0]
    "#;
    let result = run(code);
    assert!(result.is_ok(), "Test failed with error: {:?}", result);
    match result.unwrap() {
        Value::Number(n) => {
            assert!(n > 0.0, "Smooth L1 loss should be positive, got {}", n);
            assert!((n - 2.5).abs() < 0.1, "Smooth L1 loss should be approximately 2.5, got {}", n);
        }
        _ => panic!("Expected Number for loss value"),
    }
}

#[test]
fn test_smooth_l1_loss_boundary() {
    // Test Smooth L1 loss at boundary (|x| = 1)
    // differences = [1.0, -1.0, 0.5, -0.5]
    // For |x| = 1: loss = 1.0 - 0.5 = 0.5
    // For |x| < 1: loss = 0.5 * x^2
    let code = r#"
        import ml
        let y_pred_data = [2.0, 0.0, 1.5, 0.5]
        let y_pred_shape = [4, 1]
        let y_pred = ml.tensor(y_pred_data, y_pred_shape)
        
        let y_true_data = [1.0, 1.0, 1.0, 1.0]
        let y_true_shape = [4, 1]
        let y_true = ml.tensor(y_true_data, y_true_shape)
        
        let loss = ml.smooth_l1_loss(y_pred, y_true)
        ml.data(loss)[0]
    "#;
    let result = run(code);
    assert!(result.is_ok(), "Test failed with error: {:?}", result);
    match result.unwrap() {
        Value::Number(n) => {
            assert!(n > 0.0, "Smooth L1 loss should be positive, got {}", n);
        }
        _ => panic!("Expected Number for loss value"),
    }
}

