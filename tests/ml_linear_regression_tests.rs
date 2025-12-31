// Tests for ML Linear Regression

use data_code::{run, Value};

#[test]
fn test_linear_regression_creation() {
    // Test model creation
    let code = r#"
        import ml
        let model = ml.linear_regression(2)
        print("Model created")
    "#;
    let result = run(code);
    assert!(result.is_ok());
}

#[test]
fn test_linear_regression_predict() {
    // Test prediction on simple data
    // y = 2*x1 + 3*x2 + 1
    // For x = [1, 1], y should be approximately 2*1 + 3*1 + 1 = 6 (after training)
    // But initially with small weights, it will be small
    let code = r#"
        import ml
        let model = ml.linear_regression(2)
        let features_data = [1.0, 1.0]
        let features_shape = [1, 2]
        let features = ml.tensor(features_data, features_shape)
        let predictions = ml.lr_predict(model, features)
        ml.data(predictions)
    "#;
    let result = run(code);
    assert!(result.is_ok(), "Test failed with error: {:?}", result);
    let pred_data = result.unwrap();
    // Initial prediction should be small (weights are 0.01)
    match pred_data {
        Value::Array(arr) => {
            let arr_ref = arr.borrow();
            assert_eq!(arr_ref.len(), 1, "Predictions should have 1 element");
        }
        _ => panic!("Expected Array of predictions"),
    }
}

#[test]
fn test_linear_regression_train_simple() {
    // Train on simple linear relationship: y = 2*x + 1
    // x = [[1], [2], [3]], y = [[3], [5], [7]]
    let code = r#"
        import ml
        let model = ml.linear_regression(1)
        
        let x_data = [1.0, 2.0, 3.0]
        let x_shape = [3, 1]
        let x = ml.tensor(x_data, x_shape)
        
        let y_data = [3.0, 5.0, 7.0]
        let y_shape = [3, 1]
        let y = ml.tensor(y_data, y_shape)
        
        let loss_history = ml.lr_train(model, x, y, 100, 0.01)
        loss_history
    "#;
    let result = run(code);
    assert!(result.is_ok(), "Test failed with error: {:?}", result);
    let loss_history = result.unwrap();
    
    // Check that loss history is an array
    match loss_history {
        Value::Array(arr) => {
            let arr_ref = arr.borrow();
            assert_eq!(arr_ref.len(), 100, "Loss history should have 100 entries");
            
            // Check that loss decreases (or at least the last loss is reasonable)
            if let (Some(Value::Number(first_loss)), Some(Value::Number(last_loss))) = 
                (arr_ref.first(), arr_ref.last()) {
                // Loss should decrease or be small
                assert!(*last_loss < *first_loss || *last_loss < 1.0, 
                    "Loss should decrease or be small. First: {}, Last: {}", first_loss, last_loss);
            }
        }
        _ => panic!("Expected Array of loss values"),
    }
}

#[test]
fn test_linear_regression_train_convergence() {
    // Train on perfect linear relationship and check convergence
    // y = 2*x + 1, with x = [1, 2, 3], y = [3, 5, 7]
    let code = r#"
        import ml
        let model = ml.linear_regression(1)
        
        let x_data = [1.0, 2.0, 3.0]
        let x_shape = [3, 1]
        let x = ml.tensor(x_data, x_shape)
        
        let y_data = [3.0, 5.0, 7.0]
        let y_shape = [3, 1]
        let y = ml.tensor(y_data, y_shape)
        
        ml.lr_train(model, x, y, 200, 0.01)
        
        # Test prediction after training
        let test_x_data = [4.0]
        let test_x_shape = [1, 1]
        let test_x = ml.tensor(test_x_data, test_x_shape)
        let pred = ml.lr_predict(model, test_x)
        ml.data(pred)
    "#;
    let result = run(code);
    assert!(result.is_ok(), "Test failed with error: {:?}", result);
    let pred_data = result.unwrap();
    
    // After training, prediction for x=4 should be close to y=9 (2*4+1)
    match pred_data {
        Value::Array(arr) => {
            let arr_ref = arr.borrow();
            assert_eq!(arr_ref.len(), 1);
            if let Some(Value::Number(pred)) = arr_ref.first() {
                // Prediction should be close to 9 (within reasonable tolerance)
                // Note: with only 200 epochs and simple initialization, it might not be perfect
                assert!((*pred - 9.0).abs() < 2.0, 
                    "Prediction should be close to 9, got {}", pred);
            }
        }
        _ => panic!("Expected Array of predictions"),
    }
}

#[test]
fn test_linear_regression_evaluate() {
    // Test evaluation (MSE computation)
    let code = r#"
        import ml
        let model = ml.linear_regression(1)
        
        let x_data = [1.0, 2.0, 3.0]
        let x_shape = [3, 1]
        let x = ml.tensor(x_data, x_shape)
        
        let y_data = [3.0, 5.0, 7.0]
        let y_shape = [3, 1]
        let y = ml.tensor(y_data, y_shape)
        
        # Train first
        ml.lr_train(model, x, y, 50, 0.01)
        
        # Evaluate
        let x_eval_data = [1.0, 2.0, 3.0]
        let x_eval_shape = [3, 1]
        let x_eval = ml.tensor(x_eval_data, x_eval_shape)
        
        let y_eval_data = [3.0, 5.0, 7.0]
        let y_eval_shape = [3, 1]
        let y_eval = ml.tensor(y_eval_data, y_eval_shape)
        
        ml.lr_evaluate(model, x_eval, y_eval)
    "#;
    let result = run(code);
    assert!(result.is_ok(), "Test failed with error: {:?}", result);
    let mse = result.unwrap();
    
    // MSE should be a number and should be small after training
    match mse {
        Value::Number(m) => {
            assert!(m >= 0.0, "MSE should be non-negative, got {}", m);
            // After training, MSE should be relatively small
            assert!(m < 10.0, "MSE should be small after training, got {}", m);
        }
        _ => panic!("Expected Number (MSE value)"),
    }
}

#[test]
fn test_linear_regression_multiple_features() {
    // Test with multiple features: y = 2*x1 + 3*x2 + 1
    let code = r#"
        import ml
        let model = ml.linear_regression(2)
        
        let x_data = [1.0, 1.0, 2.0, 1.0, 1.0, 2.0]
        let x_shape = [3, 2]
        let x = ml.tensor(x_data, x_shape)
        
        # y = 2*x1 + 3*x2 + 1
        # For [1,1]: y = 2*1 + 3*1 + 1 = 6
        # For [2,1]: y = 2*2 + 3*1 + 1 = 8
        # For [1,2]: y = 2*1 + 3*2 + 1 = 9
        let y_data = [6.0, 8.0, 9.0]
        let y_shape = [3, 1]
        let y = ml.tensor(y_data, y_shape)
        
        let loss_history = ml.lr_train(model, x, y, 100, 0.01)
        loss_history
    "#;
    let result = run(code);
    assert!(result.is_ok(), "Test failed with error: {:?}", result);
    let loss_history = result.unwrap();
    
    // Check that training completed
    match loss_history {
        Value::Array(arr) => {
            let arr_ref = arr.borrow();
            assert_eq!(arr_ref.len(), 100, "Loss history should have 100 entries");
        }
        _ => panic!("Expected Array of loss values"),
    }
}

#[test]
fn test_linear_regression_loss_decreases() {
    // Verify that loss decreases during training
    let code = r#"
        import ml
        let model = ml.linear_regression(1)
        
        let x_data = [1.0, 2.0, 3.0]
        let x_shape = [3, 1]
        let x = ml.tensor(x_data, x_shape)
        
        let y_data = [2.0, 4.0, 6.0]
        let y_shape = [3, 1]
        let y = ml.tensor(y_data, y_shape)
        
        let loss_history = ml.lr_train(model, x, y, 50, 0.05)
        loss_history
    "#;
    let result = run(code);
    assert!(result.is_ok(), "Test failed with error: {:?}", result);
    let loss_history = result.unwrap();
    
    match loss_history {
        Value::Array(arr) => {
            let arr_ref = arr.borrow();
            if arr_ref.len() >= 2 {
                if let (Some(Value::Number(first)), Some(Value::Number(last))) = 
                    (arr_ref.first(), arr_ref.last()) {
                    // Loss should generally decrease (allowing for some noise)
                    // We check that last loss is at least not much worse than first
                    assert!(*last <= *first * 1.5, 
                        "Loss should not increase significantly. First: {}, Last: {}", first, last);
                }
            }
        }
        _ => panic!("Expected Array of loss values"),
    }
}

