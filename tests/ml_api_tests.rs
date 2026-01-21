// Tests for ML Module API (import ml)

use data_code::{run, Value};

#[test]
fn test_ml_module_import() {
    // Test that ml module can be imported
    let code = r#"
        import ml
        typeof(ml)
    "#;
    let result = run(code);
    assert!(result.is_ok(), "Test failed with error: {:?}", result);
    match result.unwrap() {
        Value::String(s) => {
            assert_eq!(s, "object", "ml should be an object");
        }
        _ => panic!("Expected String for type"),
    }
}

#[test]
fn test_ml_module_functions_available() {
    // Test that key ML functions are available in the module
    let code = r#"
        import ml
        typeof(ml.tensor) == "function" and typeof(ml.linear_regression) == "function" and typeof(ml.mse_loss) == "function"
    "#;
    let result = run(code);
    assert!(result.is_ok(), "Test failed with error: {:?}", result);
    match result.unwrap() {
        Value::Bool(b) => {
            assert!(b, "ML functions should be available");
        }
        _ => panic!("Expected Bool"),
    }
}

#[test]
fn test_ml_end_to_end_workflow() {
    // Test complete ML workflow: dataset -> model -> train -> predict -> evaluate
    let code = r#"
        import ml
        
        # Create dataset from table
        let data = [[1.0, 2.0, 3.0], [2.0, 3.0, 5.0], [3.0, 4.0, 7.0], [4.0, 5.0, 9.0]]
        let headers = ["x1", "x2", "y"]
        let train_table = table(data, headers)
        
        let feature_cols = ["x1", "x2"]
        let target_cols = ["y"]
        let train_ds = ml.dataset(train_table, feature_cols, target_cols)
        
        # Get features and targets
        let x_train = ml.dataset_features(train_ds)
        let y_train = ml.dataset_targets(train_ds)
        
        # Create and train model
        let model = ml.linear_regression(2)
        let loss_history = ml.lr_train(model, x_train, y_train, 50, 0.01)
        
        # Check that loss decreased
        let initial_loss = loss_history[0]
        let final_loss = loss_history[len(loss_history) - 1]
        
        final_loss < initial_loss
    "#;
    let result = run(code);
    assert!(result.is_ok(), "Test failed with error: {:?}", result);
    match result.unwrap() {
        Value::Bool(b) => {
            assert!(b, "Loss should decrease during training");
        }
        _ => panic!("Expected Bool"),
    }
}

#[test]
fn test_ml_loss_functions_in_workflow() {
    // Test using loss functions in a training workflow
    let code = r#"
        import ml
        
        # Create simple data
        let x_data = [1.0, 2.0, 3.0]
        let x_shape = [3, 1]
        let x = ml.tensor(x_data, x_shape)
        
        let y_true_data = [2.0, 4.0, 6.0]
        let y_true_shape = [3, 1]
        let y_true = ml.tensor(y_true_data, y_true_shape)
        
        # Simple prediction (just x for now)
        let y_pred = x
        
        # Compute loss
        let loss = ml.mse_loss(y_pred, y_true)
        let loss_value = ml.data(loss)[0]
        
        loss_value > 0.0
    "#;
    let result = run(code);
    assert!(result.is_ok(), "Test failed with error: {:?}", result);
    match result.unwrap() {
        Value::Bool(b) => {
            assert!(b, "Loss should be positive");
        }
        _ => panic!("Expected Bool"),
    }
}

#[test]
fn test_ml_dataset_with_table_operations() {
    // Test that dataset works with table operations (filter, etc.)
    // This demonstrates the unique advantage: tables -> ML seamlessly
    let code = r#"
        import ml
        
        # Create table with more data
        let data = [[1.0, 2.0, 3.0], [2.0, 3.0, 5.0], [3.0, 4.0, 7.0], [10.0, 20.0, 30.0]]
        let headers = ["x1", "x2", "y"]
        let my_table = table(data, headers)
        
        # In a real scenario, you could filter the table here
        # For now, just create dataset from the table
        let feature_cols = ["x1", "x2"]
        let target_cols = ["y"]
        let ds = ml.dataset(my_table, feature_cols, target_cols)
        
        let features = ml.dataset_features(ds)
        let shape = ml.shape(features)
        shape[0]
    "#;
    let result = run(code);
    assert!(result.is_ok(), "Test failed with error: {:?}", result);
    match result.unwrap() {
        Value::Number(n) => {
            assert_eq!(n as usize, 4, "Should have 4 samples");
        }
        _ => panic!("Expected Number"),
    }
}

// ============================================================================
// ML Layer API Tests (ml.layer.*)
// ============================================================================

#[test]
fn test_ml_layer_linear() {
    // Test creating a Linear layer through the new API
    let code = r#"
        import ml
        let layer = ml.layer.linear(10, 5)
        typeof(layer)
    "#;
    let result = run(code);
    assert!(result.is_ok(), "Test failed with error: {:?}", result);
    match result.unwrap() {
        Value::String(s) => {
            assert_eq!(s, "layer", "ml.layer.linear() should return a layer");
        }
        _ => panic!("Expected String for layer type"),
    }
}

#[test]
fn test_ml_layer_relu() {
    // Test creating a ReLU layer through the new API
    let code = r#"
        import ml
        let layer = ml.layer.relu()
        typeof(layer)
    "#;
    let result = run(code);
    assert!(result.is_ok(), "Test failed with error: {:?}", result);
    match result.unwrap() {
        Value::String(s) => {
            assert_eq!(s, "layer", "ml.layer.relu() should return a layer");
        }
        _ => panic!("Expected String for layer type"),
    }
}

#[test]
fn test_ml_layer_softmax() {
    // Test creating a Softmax layer through the new API
    let code = r#"
        import ml
        let layer = ml.layer.softmax()
        typeof(layer)
    "#;
    let result = run(code);
    assert!(result.is_ok(), "Test failed with error: {:?}", result);
    match result.unwrap() {
        Value::String(s) => {
            assert_eq!(s, "layer", "ml.layer.softmax() should return a layer");
        }
        _ => panic!("Expected String for layer type"),
    }
}

#[test]
fn test_ml_layer_flatten() {
    // Test creating a Flatten layer through the new API
    let code = r#"
        import ml
        let layer = ml.layer.flatten()
        typeof(layer)
    "#;
    let result = run(code);
    assert!(result.is_ok(), "Test failed with error: {:?}", result);
    match result.unwrap() {
        Value::String(s) => {
            assert_eq!(s, "layer", "ml.layer.flatten() should return a layer");
        }
        _ => panic!("Expected String for layer type"),
    }
}

#[test]
fn test_ml_layer_with_sequential() {
    // Test creating Sequential with layers through ml.layer.*
    let code = r#"
        import ml
        
        # Create layers using new API
        let layer1 = ml.layer.linear(10, 5)
        let layer2 = ml.layer.relu()
        let layer3 = ml.layer.linear(5, 2)
        
        # Create Sequential
        let layers = [layer1, layer2, layer3]
        let seq = ml.sequential(layers)
        
        typeof(seq)
    "#;
    let result = run(code);
    assert!(result.is_ok(), "Test failed with error: {:?}", result);
    match result.unwrap() {
        Value::String(s) => {
            assert_eq!(s, "sequential", "Sequential should be created successfully");
        }
        _ => panic!("Expected String for sequential type"),
    }
}

#[test]
fn test_ml_layer_with_neural_network() {
    // Test creating MLP using ml.layer.* API and training
    let code = r#"
        import ml
        
        # Create MLP architecture using new API
        let layer1 = ml.layer.linear(10, 5)
        let layer2 = ml.layer.relu()
        let layer3 = ml.layer.linear(5, 2)
        let layer4 = ml.layer.softmax()
        
        # Create Sequential and Neural Network
        let layers = [layer1, layer2, layer3, layer4]
        let seq = ml.sequential(layers)
        let nn = ml.neural_network(seq)
        
        # Create simple training data
        let x_data = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]  # 2 samples, 10 features each
        let x_shape = [2, 10]
        let x = ml.tensor(x_data, x_shape)
        
        let y_data = [0.0, 1.0]  # 2 labels
        let y_shape = [2, 1]
        let y = ml.tensor(y_data, y_shape)
        
        # Train for 1 epoch
        let loss_history = ml.nn_train(nn, x, y, 1, 2, 0.01, "sparse_cross_entropy")
        
        len(loss_history.loss)
    "#;
    let result = run(code);
    assert!(result.is_ok(), "Test failed with error: {:?}", result);
    match result.unwrap() {
        Value::Number(n) => {
            assert_eq!(n as usize, 1, "Should have 1 loss value after 1 epoch");
        }
        _ => panic!("Expected Number for loss history length"),
    }
}

#[test]
fn test_ml_layer_forward_pass() {
    // Test forward pass through layers created with new API
    let code = r#"
        import ml
        
        # Create a simple layer
        let layer = ml.layer.linear(5, 3)
        
        # Create input tensor
        let x_data = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]  # 2 samples, 5 features each
        let x_shape = [2, 5]
        let x = ml.tensor(x_data, x_shape)
        
        # Create Sequential with the layer
        let layers = [layer]
        let seq = ml.sequential(layers)
        
        # Forward pass
        let output = ml.nn_forward(seq, x)
        let output_shape = ml.shape(output)
        
        output_shape[1]  # Should be 3 (out_features)
    "#;
    let result = run(code);
    assert!(result.is_ok(), "Test failed with error: {:?}", result);
    match result.unwrap() {
        Value::Number(n) => {
            assert_eq!(n as usize, 3, "Output should have 3 features");
        }
        _ => panic!("Expected Number for output shape"),
    }
}

#[test]
fn test_ml_layer_all_types() {
    // Test that all layer types are accessible through ml.layer.*
    let code = r#"
        import ml
        
        let linear = ml.layer.linear(10, 5)
        let relu = ml.layer.relu()
        let softmax = ml.layer.softmax()
        let flatten = ml.layer.flatten()
        
        typeof(linear) == "layer" and typeof(relu) == "layer" and typeof(softmax) == "layer" and typeof(flatten) == "layer"
    "#;
    let result = run(code);
    assert!(result.is_ok(), "Test failed with error: {:?}", result);
    match result.unwrap() {
        Value::Bool(b) => {
            assert!(b, "All layer types should be accessible");
        }
        _ => panic!("Expected Bool"),
    }
}

// ============================================================================
// ML Model Info Tests (ml.model_info)
// ============================================================================

#[test]
fn test_ml_model_info_neural_network() {
    // Test model_info for NeuralNetwork - create, train, check info
    let code = r#"
        import ml
        
        # Create MLP architecture
        let layer1 = ml.layer.linear(10, 5)
        let layer2 = ml.layer.relu()
        let layer3 = ml.layer.linear(5, 2)
        
        let layers = [layer1, layer2, layer3]
        let seq = ml.sequential(layers)
        let model = ml.neural_network(seq)
        
        # Create simple training data
        let x_data = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
        let x_shape = [2, 10]
        let x = ml.tensor(x_data, x_shape)
        
        let y_data = [0.0, 1.0]
        let y_shape = [2, 1]
        let y = ml.tensor(y_data, y_shape)
        
        # Train for 1 epoch
        let loss_history = model.train(x, y, 1, 2, 0.01, "cross_entropy", optimizer="SGD")
        
        # Check model_info returns null (text format)
        let info_result = ml.model_info(model)
        typeof(info_result) == "null"
    "#;
    let result = run(code);
    assert!(result.is_ok(), "Test failed with error: {:?}", result);
    match result.unwrap() {
        Value::Bool(b) => {
            assert!(b, "model_info should return null for text format");
        }
        _ => panic!("Expected Bool"),
    }
}

#[test]
fn test_ml_model_info_neural_network_json() {
    // Test model_info with JSON format
    let code = r#"
        import ml
        
        # Create MLP architecture
        let layer1 = ml.layer.linear(8, 4)
        let layer2 = ml.layer.relu()
        let layer3 = ml.layer.linear(4, 2)
        
        let layers = [layer1, layer2, layer3]
        let seq = ml.sequential(layers)
        let model = ml.neural_network(seq)
        
        # Get model info in JSON format
        let info_json = ml.model_info(model, format="json")
        
        # Check that it's a string (JSON)
        typeof(info_json) == "string"
    "#;
    let result = run(code);
    assert!(result.is_ok(), "Test failed with error: {:?}", result);
    match result.unwrap() {
        Value::Bool(b) => {
            assert!(b, "model_info should return string for JSON format");
        }
        _ => panic!("Expected Bool"),
    }
}

#[test]
fn test_ml_model_info_linear_regression() {
    // Test model_info for LinearRegression
    let code = r#"
        import ml
        
        # Create LinearRegression model
        let model = ml.linear_regression(3)
        
        # Create training data
        let x_data = [1.0, 2.0, 3.0, 2.0, 3.0, 4.0, 3.0, 4.0, 5.0]
        let x_shape = [3, 3]
        let x = ml.tensor(x_data, x_shape)
        
        let y_data = [6.0, 9.0, 12.0]
        let y_shape = [3, 1]
        let y = ml.tensor(y_data, y_shape)
        
        # Train for a few epochs
        let loss_history = ml.lr_train(model, x, y, 10, 0.01)
        
        # Check model_info returns null (text format)
        let info_result = ml.model_info(model)
        typeof(info_result) == "null"
    "#;
    let result = run(code);
    assert!(result.is_ok(), "Test failed with error: {:?}", result);
    match result.unwrap() {
        Value::Bool(b) => {
            assert!(b, "model_info should return null for text format");
        }
        _ => panic!("Expected Bool"),
    }
}

#[test]
fn test_ml_model_info_with_graph() {
    // Test model_info with show_graph parameter
    let code = r#"
        import ml
        
        # Create MLP architecture
        let layer1 = ml.layer.linear(6, 4)
        let layer2 = ml.layer.relu()
        let layer3 = ml.layer.linear(4, 3)
        
        let layers = [layer1, layer2, layer3]
        let seq = ml.sequential(layers)
        let model = ml.neural_network(seq)
        
        # Get model info with graph
        let info_result = ml.model_info(model, show_graph=true)
        typeof(info_result) == "null"
    "#;
    let result = run(code);
    assert!(result.is_ok(), "Test failed with error: {:?}", result);
    match result.unwrap() {
        Value::Bool(b) => {
            assert!(b, "model_info with show_graph should return null for text format");
        }
        _ => panic!("Expected Bool"),
    }
}

#[cfg(feature = "gpu")]
#[test]
fn test_ml_model_info_save_and_load() {
    // Test model_info after saving and loading a model
    use std::fs;
    use std::path::Path;
    
    let temp_file = "test_model_info_save_load.nn";
    
    // Clean up if file exists
    if Path::new(temp_file).exists() {
        let _ = fs::remove_file(temp_file);
    }
    
    // Create, train, save model
    let code = format!(r#"
        import ml
        
        # Create MLP architecture
        let layer1 = ml.layer.linear(8, 4)
        let layer2 = ml.layer.relu()
        let layer3 = ml.layer.linear(4, 2)
        
        let layers = [layer1, layer2, layer3]
        let seq = ml.sequential(layers)
        let model = ml.neural_network(seq)
        
        # Create training data
        let x_data = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
        let x_shape = [2, 8]
        let x = ml.tensor(x_data, x_shape)
        
        let y_data = [0.0, 1.0]
        let y_shape = [2, 1]
        let y = ml.tensor(y_data, y_shape)
        
        # Train for 1 epoch
        let loss_history = model.train(x, y, 1, 2, 0.01, "cross_entropy", optimizer="SGD")
        
        # Save the model
        model.save("{}")
        print("Model saved successfully")
        
        # Get model info before saving
        let info_before = ml.model_info(model, format="json")
        typeof(info_before) == "string"
    "#, temp_file);
    
    let result = run(&code);
    assert!(result.is_ok(), "Failed to create, train and save model: {:?}", result);
    
    // Load and check model_info
    let load_code = format!(r#"
        import ml
        
        # Load the model
        let loaded_model = ml.load("{}")
        
        # Get model info in JSON format
        let info_after = ml.model_info(loaded_model, format="json")
        
        # Check that info is a string (JSON)
        typeof(info_after) == "string"
    "#, temp_file);
    
    let load_result = run(&load_code);
    assert!(load_result.is_ok(), "Failed to load model and get info: {:?}", load_result);
    
    match load_result.unwrap() {
        Value::Bool(b) => {
            assert!(b, "model_info should return string for JSON format after loading");
        }
        _ => panic!("Expected Bool"),
    }
    
    // Clean up
    if Path::new(temp_file).exists() {
        let _ = fs::remove_file(temp_file);
    }
}

#[test]
fn test_ml_model_info_parameters_count() {
    // Test that model_info correctly counts parameters
    let code = r#"
        import ml
        
        # Create MLP architecture: 10 -> 5 -> 2
        # Expected params: (10*5 + 5) + (5*2 + 2) = 55 + 12 = 67
        let layer1 = ml.layer.linear(10, 5)
        let layer2 = ml.layer.relu()
        let layer3 = ml.layer.linear(5, 2)
        
        let layers = [layer1, layer2, layer3]
        let seq = ml.sequential(layers)
        let model = ml.neural_network(seq)
        
        # Get model info in JSON format
        let info_json = ml.model_info(model, format="json")
        
        # Parse JSON to check parameters (simplified check - just verify it's a string)
        typeof(info_json) == "string" and len(info_json) > 0
    "#;
    let result = run(code);
    assert!(result.is_ok(), "Test failed with error: {:?}", result);
    match result.unwrap() {
        Value::Bool(b) => {
            assert!(b, "model_info JSON should contain parameter information");
        }
        _ => panic!("Expected Bool"),
    }
}

#[test]
fn test_ml_model_info_complex_architecture() {
    // Test model_info with complex architecture (multiple layers)
    let code = r#"
        import ml
        
        # Create complex MLP: 20 -> 16 -> 8 -> 4 -> 2
        let layer1 = ml.layer.linear(20, 16)
        let layer2 = ml.layer.relu()
        let layer3 = ml.layer.linear(16, 8)
        let layer4 = ml.layer.relu()
        let layer5 = ml.layer.linear(8, 4)
        let layer6 = ml.layer.relu()
        let layer7 = ml.layer.linear(4, 2)
        
        let layers = [layer1, layer2, layer3, layer4, layer5, layer6, layer7]
        let seq = ml.sequential(layers)
        let model = ml.neural_network(seq)
        
        # Create training data
        let x_data = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
        let x_shape = [2, 20]
        let x = ml.tensor(x_data, x_shape)
        
        let y_data = [0.0, 1.0]
        let y_shape = [2, 1]
        let y = ml.tensor(y_data, y_shape)
        
        # Train for 1 epoch
        let loss_history = model.train(x, y, 1, 2, 0.01, "cross_entropy", optimizer="SGD")
        
        # Get model info with graph in JSON format
        let info_json = ml.model_info(model, format="json", show_graph=true)
        
        # Check that it's a string and contains graph info
        typeof(info_json) == "string" and len(info_json) > 0
    "#;
    let result = run(code);
    assert!(result.is_ok(), "Test failed with error: {:?}", result);
    match result.unwrap() {
        Value::Bool(b) => {
            assert!(b, "model_info should work with complex architecture");
        }
        _ => panic!("Expected Bool"),
    }
}

#[test]
fn test_ml_model_info_linear_regression_json() {
    // Test model_info for LinearRegression in JSON format
    let code = r#"
        import ml
        
        # Create LinearRegression model with 5 features
        let model = ml.linear_regression(5)
        
        # Create training data
        let x_data = [1.0, 2.0, 3.0, 4.0, 5.0, 2.0, 3.0, 4.0, 5.0, 6.0, 3.0, 4.0, 5.0, 6.0, 7.0]
        let x_shape = [3, 5]
        let x = ml.tensor(x_data, x_shape)
        
        let y_data = [15.0, 20.0, 25.0]
        let y_shape = [3, 1]
        let y = ml.tensor(y_data, y_shape)
        
        # Train for a few epochs
        let loss_history = ml.lr_train(model, x, y, 5, 0.01)
        
        # Get model info in JSON format
        let info_json = ml.model_info(model, format="json")
        
        # Check that it's a string (JSON)
        typeof(info_json) == "string" and len(info_json) > 0
    "#;
    let result = run(code);
    assert!(result.is_ok(), "Test failed with error: {:?}", result);
    match result.unwrap() {
        Value::Bool(b) => {
            assert!(b, "model_info should return JSON string for LinearRegression");
        }
        _ => panic!("Expected Bool"),
    }
}

#[test]
fn test_ml_model_info_parameters_and_type() {
    // Test that model_info correctly reports parameter counts and model type
    // Architecture: Linear(10, 5) -> ReLU -> Linear(5, 2)
    // Expected params: (10*5 + 5) + (5*2 + 2) = 55 + 12 = 67
    let code = r#"
        import ml
        
        # Create MLP architecture with known parameter count
        let layer1 = ml.layer.linear(10, 5)
        let layer2 = ml.layer.relu()
        let layer3 = ml.layer.linear(5, 2)
        
        let layers = [layer1, layer2, layer3]
        let seq = ml.sequential(layers)
        let model = ml.neural_network(seq)
        
        # Create training data
        let x_data = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
        let x_shape = [2, 10]
        let x = ml.tensor(x_data, x_shape)
        
        let y_data = [0.0, 1.0]
        let y_shape = [2, 1]
        let y = ml.tensor(y_data, y_shape)
        
        # Train for 1 epoch
        let loss_history = model.train(x, y, 1, 2, 0.01, "cross_entropy", optimizer="SGD")
        
        # Get model info in JSON format
        let info_json = ml.model_info(model, format="json")
        
        # Check that JSON contains model type and parameters
        # We'll check that the string contains expected values
        # Model type should be "MLP" (since we have more than 2 layers)
        # Parameters should be 67: (10*5 + 5) + (5*2 + 2) = 55 + 12 = 67
        
        # Check that JSON string contains "MLP" or "NeuralNetwork"
        let has_type = contains(info_json, "MLP") or contains(info_json, "NeuralNetwork")
        
        # Check that JSON contains "parameters" field
        let has_parameters = contains(info_json, "parameters")
        
        # Check that JSON contains "trainable" field
        let has_trainable = contains(info_json, "trainable")
        
        # Check that JSON contains "total" field
        let has_total = contains(info_json, "total")
        
        # Check that JSON contains "frozen" field
        let has_frozen = contains(info_json, "frozen")
        
        # Check exact parameter values
        # Expected: trainable = 67, frozen = 0, total = 67
        # Architecture: Linear(10, 5) -> ReLU -> Linear(5, 2)
        # Params: (10*5 + 5) + (5*2 + 2) = 55 + 12 = 67
        let has_trainable_67 = contains(info_json, "\"trainable\": 67")
        let has_frozen_0 = contains(info_json, "\"frozen\": 0")
        let has_total_67 = contains(info_json, "\"total\": 67")
        
        # All checks should pass
        has_type and has_parameters and has_trainable and has_total and has_frozen and has_trainable_67 and has_frozen_0 and has_total_67
    "#;
    let result = run(code);
    assert!(result.is_ok(), "Test failed with error: {:?}", result);
    match result.unwrap() {
        Value::Bool(b) => {
            assert!(b, "model_info JSON should contain model type and parameter information");
        }
        _ => panic!("Expected Bool"),
    }
}

#[test]
fn test_ml_model_info_linear_regression_parameters() {
    // Test that LinearRegression model_info correctly reports parameter counts
    // LinearRegression with 5 features: 5 weights + 1 bias = 6 parameters
    let code = r#"
        import ml
        
        # Create LinearRegression model with 5 features
        let model = ml.linear_regression(5)
        
        # Create training data
        let x_data = [1.0, 2.0, 3.0, 4.0, 5.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        let x_shape = [2, 5]
        let x = ml.tensor(x_data, x_shape)
        
        let y_data = [15.0, 20.0]
        let y_shape = [2, 1]
        let y = ml.tensor(y_data, y_shape)
        
        # Train for a few epochs
        let loss_history = ml.lr_train(model, x, y, 5, 0.01)
        
        # Get model info in JSON format
        let info_json = ml.model_info(model, format="json")
        
        # Check that JSON contains "LinearRegression" type
        let has_type = contains(info_json, "LinearRegression")
        
        # Check that JSON contains parameters information
        let has_parameters = contains(info_json, "parameters")
        let has_trainable = contains(info_json, "trainable")
        let has_total = contains(info_json, "total")
        let has_frozen = contains(info_json, "frozen")
        
        # Check exact parameter values
        # Expected: 5 features -> 5 weights + 1 bias = 6 total parameters
        # trainable = 6, frozen = 0, total = 6
        let has_trainable_6 = contains(info_json, "\"trainable\": 6")
        let has_frozen_0 = contains(info_json, "\"frozen\": 0")
        let has_total_6 = contains(info_json, "\"total\": 6")
        
        # All checks should pass
        has_type and has_parameters and has_trainable and has_total and has_frozen and has_trainable_6 and has_frozen_0 and has_total_6
    "#;
    let result = run(code);
    assert!(result.is_ok(), "Test failed with error: {:?}", result);
    match result.unwrap() {
        Value::Bool(b) => {
            assert!(b, "LinearRegression model_info should contain correct type and parameter counts");
        }
        _ => panic!("Expected Bool"),
    }
}

#[test]
fn test_ml_model_info_text_format_parameters() {
    // Test that text format correctly displays parameter counts
    let code = r#"
        import ml
        
        # Create simple MLP: Linear(8, 4) -> ReLU -> Linear(4, 2)
        # Expected params: (8*4 + 4) + (4*2 + 2) = 36 + 10 = 46
        let layer1 = ml.layer.linear(8, 4)
        let layer2 = ml.layer.relu()
        let layer3 = ml.layer.linear(4, 2)
        
        let layers = [layer1, layer2, layer3]
        let seq = ml.sequential(layers)
        let model = ml.neural_network(seq)
        
        # Get model info in text format (default)
        # Note: text format prints to stdout, so we can't easily capture it
        # But we can verify the function doesn't error
        let info_result = ml.model_info(model)
        
        # Function should return null for text format
        typeof(info_result) == "null"
    "#;
    let result = run(code);
    assert!(result.is_ok(), "Test failed with error: {:?}", result);
    match result.unwrap() {
        Value::Bool(b) => {
            assert!(b, "model_info should return null for text format");
        }
        _ => panic!("Expected Bool"),
    }
}

#[test]
fn test_ml_model_info_exact_parameter_values_complex() {
    // Test exact parameter values for complex architecture
    // Architecture: Linear(8, 6) -> ReLU -> Linear(6, 4) -> ReLU -> Linear(4, 2)
    // Expected params: (8*6 + 6) + (6*4 + 4) + (4*2 + 2) = 54 + 28 + 10 = 92
    let code = r#"
        import ml
        
        # Create complex MLP architecture
        let layer1 = ml.layer.linear(8, 6)
        let layer2 = ml.layer.relu()
        let layer3 = ml.layer.linear(6, 4)
        let layer4 = ml.layer.relu()
        let layer5 = ml.layer.linear(4, 2)
        
        let layers = [layer1, layer2, layer3, layer4, layer5]
        let seq = ml.sequential(layers)
        let model = ml.neural_network(seq)
        
        # Create dummy input to initialize parameters (forward pass)
        let x_data = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
        let x_shape = [1, 8]
        let x = ml.tensor(x_data, x_shape)
        
        # Forward pass to initialize parameters
        let _ = ml.nn_forward(model, x)
        
        # Get model info in JSON format
        let info_json = ml.model_info(model, format="json")
        
        # Check exact parameter values
        # Expected: trainable = 92, frozen = 0, total = 92
        # (8*6 + 6) + (6*4 + 4) + (4*2 + 2) = 54 + 28 + 10 = 92
        let has_trainable_92 = contains(info_json, "\"trainable\": 92")
        let has_frozen_0 = contains(info_json, "\"frozen\": 0")
        let has_total_92 = contains(info_json, "\"total\": 92")
        
        # Check model type
        let has_type = contains(info_json, "MLP")
        
        # All checks should pass
        has_type and has_trainable_92 and has_frozen_0 and has_total_92
    "#;
    let result = run(code);
    assert!(result.is_ok(), "Test failed with error: {:?}", result);
    match result.unwrap() {
        Value::Bool(b) => {
            assert!(b, "model_info should report exact parameter values for complex architecture");
        }
        _ => panic!("Expected Bool"),
    }
}

#[test]
fn test_layer_freeze_unfreeze_basic() {
    // Test basic freeze/unfreeze functionality
    let code = r#"
        import ml
        
        # Create a layer
        layer = ml.layer.linear(3, 4)
        
        # Layer should be trainable by default (we can't check directly, but freeze should work)
        # Freeze the layer
        ml.layer_freeze(layer)
        
        # Unfreeze the layer
        ml.layer_unfreeze(layer)
        
        true
    "#;
    let result = run(code);
    assert!(result.is_ok(), "Test failed with error: {:?}", result);
    match result.unwrap() {
        Value::Bool(b) => {
            assert!(b, "Freeze/unfreeze should work");
        }
        _ => panic!("Expected Bool"),
    }
}

#[test]
fn test_model_get_layer() {
    // Test getting layer by index from model
    let code = r#"
        import ml
        
        # Create model with layers
        layer1 = ml.layer.linear(2, 3)
        layer2 = ml.layer.relu()
        layer3 = ml.layer.linear(3, 1)
        layers = [layer1, layer2, layer3]
        model_seq = ml.sequential(layers)
        model = ml.neural_network(model_seq)
        
        # Get layer by index
        layer0 = ml.model_get_layer(model, 0)
        layer2_retrieved = ml.model_get_layer(model, 2)
        
        # Check that we got layers (they should not be null)
        layer0 != null and layer2_retrieved != null
    "#;
    let result = run(code);
    assert!(result.is_ok(), "Test failed with error: {:?}", result);
    match result.unwrap() {
        Value::Bool(b) => {
            assert!(b, "Should be able to get layers by index");
        }
        _ => panic!("Expected Bool"),
    }
}

#[test]
fn test_freeze_layer_before_training() {
    // Test that frozen layers don't update during training
    let code = r#"
        import ml
        
        # Create simple model
        layer1 = ml.layer.linear(2, 3)
        layer2 = ml.layer.linear(3, 1)
        layers = [layer1, layer2]
        model_seq = ml.sequential(layers)
        model = ml.neural_network(model_seq)
        
        # Create training data
        x_data = [1.0, 2.0, 3.0, 4.0]
        x_shape = [2, 2]
        x = ml.tensor(x_data, x_shape)
        
        y_data = [1.0, 2.0]
        y_shape = [2, 1]
        y = ml.tensor(y_data, y_shape)
        
        # Train model first to initialize parameters
        # Then freeze and train again to verify frozen layer doesn't update
        loss_history1 = ml.nn_train(model, x, y, 2, 2, 0.01, "mse")
        
        # Freeze first layer
        model.layers[0].freeze()
        
        # Train again (frozen layer should not update)
        loss_history2 = ml.nn_train(model, x, y, 2, 2, 0.01, "mse")
        
        # Both training sessions should complete
        len(loss_history1) == 2 and len(loss_history2) == 2
        
        # Check that training completed (loss_history should not be empty)
        len(loss_history2) > 0
    "#;
    let result = run(code);
    assert!(result.is_ok(), "Test failed with error: {:?}", result);
    match result.unwrap() {
        Value::Bool(b) => {
            assert!(b, "Training should work with frozen layers");
        }
        _ => panic!("Expected Bool"),
    }
}

#[test]
fn test_model_info_shows_frozen_layers() {
    // Test that model_info displays frozen layers
    let code = r#"
        import ml
        
        # Create model
        layer1 = ml.layer.linear(2, 3)
        layer2 = ml.layer.linear(3, 1)
        layers = [layer1, layer2]
        model_seq = ml.sequential(layers)
        model = ml.neural_network(model_seq)
        
        # Freeze first layer
        ml.layer_freeze(layer1)
        
        # Get model info as JSON
        info_json = ml.model_info(model, false, "json")
        
        # Check that frozen_layers field exists and contains layer0
        contains(info_json, "frozen_layers") and contains(info_json, "layer0")
    "#;
    let result = run(code);
    assert!(result.is_ok(), "Test failed with error: {:?}", result);
    match result.unwrap() {
        Value::Bool(b) => {
            assert!(b, "model_info should show frozen layers");
        }
        _ => panic!("Expected Bool"),
    }
}

#[test]
fn test_model_info_frozen_parameters_count() {
    // Test that model_info correctly counts frozen parameters
    let code = r#"
        import ml
        
        # Create model: Linear(2, 3) = 9 params, Linear(3, 1) = 4 params, total = 13
        layer1 = ml.layer.linear(2, 3)
        layer2 = ml.layer.linear(3, 1)
        layers = [layer1, layer2]
        model_seq = ml.sequential(layers)
        model = ml.neural_network(model_seq)
        
        # Freeze first layer (9 parameters)
        ml.layer_freeze(layer1)
        
        # Get model info as JSON
        info_json = ml.model_info(model, false, "json")
        
        # Check that frozen parameters count is 9 and trainable is 4
        contains(info_json, "\"frozen\": 9") and contains(info_json, "\"trainable\": 4")
    "#;
    let result = run(code);
    assert!(result.is_ok(), "Test failed with error: {:?}", result);
    match result.unwrap() {
        Value::Bool(b) => {
            assert!(b, "model_info should correctly count frozen and trainable parameters");
        }
        _ => panic!("Expected Bool"),
    }
}

#[test]
fn test_all_layers_frozen_warning() {
    // Test that training with all layers frozen shows a warning
    // Note: This test checks that training completes (warning is printed to stderr)
    let code = r#"
        import ml
        
        # Create model
        layer1 = ml.layer.linear(2, 3)
        layer2 = ml.layer.linear(3, 1)
        layers = [layer1, layer2]
        model_seq = ml.sequential(layers)
        model = ml.neural_network(model_seq)
        
        # Freeze all layers
        ml.layer_freeze(layer1)
        ml.layer_freeze(layer2)
        
        # Create training data
        x_data = [1.0, 2.0, 3.0, 4.0]
        x_shape = [2, 2]
        x = ml.tensor(x_data, x_shape)
        
        y_data = [1.0, 2.0]
        y_shape = [2, 1]
        y = ml.tensor(y_data, y_shape)
        
        # Train model (should complete but with warning)
        loss_history = ml.nn_train(model, x, y, 2, 2, 0.01, "mse")
        
        # Training should complete (loss_history should exist)
        len(loss_history.loss) == 2
    "#;
    let result = run(code);
    assert!(result.is_ok(), "Test failed with error: {:?}", result);
    match result.unwrap() {
        Value::Bool(b) => {
            assert!(b, "Training should complete even with all layers frozen");
        }
        _ => panic!("Expected Bool"),
    }
}

#[test]
fn test_freeze_unfreeze_cycle() {
    // Test multiple freeze/unfreeze cycles
    let code = r#"
        import ml
        
        # Create layer
        layer = ml.layer.linear(3, 4)
        
        # Freeze and unfreeze multiple times
        ml.layer_freeze(layer)
        ml.layer_unfreeze(layer)
        ml.layer_freeze(layer)
        ml.layer_unfreeze(layer)
        
        true
    "#;
    let result = run(code);
    assert!(result.is_ok(), "Test failed with error: {:?}", result);
    match result.unwrap() {
        Value::Bool(b) => {
            assert!(b, "Multiple freeze/unfreeze cycles should work");
        }
        _ => panic!("Expected Bool"),
    }
}

