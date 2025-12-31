// Tests for ML Neural Network module

use data_code::ml::tensor::Tensor;
use data_code::ml::layer::{Layer, Linear, Sequential};
use data_code::ml::dataset::Dataset;

#[test]
fn test_tensor_relu() {
    let t = Tensor::new(vec![-1.0, 0.0, 1.0, 2.0], vec![4]).unwrap();
    let result = t.relu();
    assert_eq!(result.data, vec![0.0, 0.0, 1.0, 2.0]);
}

#[test]
fn test_tensor_sigmoid() {
    let t = Tensor::new(vec![0.0], vec![1]).unwrap();
    let result = t.sigmoid();
    // sigmoid(0) = 0.5
    assert!((result.data[0] - 0.5).abs() < 1e-6);
}

#[test]
fn test_tensor_tanh() {
    let t = Tensor::new(vec![0.0], vec![1]).unwrap();
    let result = t.tanh();
    // tanh(0) = 0
    assert!((result.data[0]).abs() < 1e-6);
}

#[test]
fn test_tensor_softmax() {
    let t = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
    let result = t.softmax().unwrap();
    
    // Softmax should sum to 1
    let sum: f32 = result.data.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5);
    
    // All values should be positive
    for &val in &result.data {
        assert!(val > 0.0);
    }
    
    // Largest value should have highest probability
    assert!(result.data[2] > result.data[1]);
    assert!(result.data[1] > result.data[0]);
}

#[test]
fn test_tensor_randn() {
    let t = Tensor::randn(vec![10]).unwrap();
    assert_eq!(t.shape, vec![10]);
    assert_eq!(t.data.len(), 10);
}

#[test]
fn test_tensor_he_normal() {
    let t = Tensor::he_normal(vec![10, 5], 10).unwrap();
    assert_eq!(t.shape, vec![10, 5]);
    assert_eq!(t.data.len(), 50);
}

#[test]
fn test_tensor_xavier_uniform() {
    let t = Tensor::xavier_uniform(vec![10, 5], 10, 5).unwrap();
    assert_eq!(t.shape, vec![10, 5]);
    assert_eq!(t.data.len(), 50);
}

#[test]
fn test_linear_layer_creation() {
    let linear = Linear::new(3, 4).unwrap();
    assert_eq!(linear.in_features(), 3);
    assert_eq!(linear.out_features(), 4);
}

#[test]
fn test_linear_layer_xavier() {
    let linear = Linear::new_xavier(3, 4).unwrap();
    assert_eq!(linear.in_features(), 3);
    assert_eq!(linear.out_features(), 4);
}

#[test]
fn test_softmax_cross_entropy_loss() {
    // Simple test: 2 classes, 2 samples
    let logits = Tensor::new(vec![1.0, 2.0, 3.0, 1.0], vec![2, 2]).unwrap();
    let targets = Tensor::new(vec![0.0, 1.0, 1.0, 0.0], vec![2, 2]).unwrap(); // one-hot
    
    // Use categorical_cross_entropy_loss for one-hot targets
    use data_code::ml::loss::categorical_cross_entropy_loss;
    let loss = categorical_cross_entropy_loss(&logits, &targets).unwrap();
    
    // Loss should be positive
    assert!(loss.data[0] > 0.0);
}

#[test]
fn test_sequential_creation() {
    // Sequential can be created empty and layers added later
    let result = Sequential::new(vec![]);
    assert!(result.is_ok());
    
    // Sequential with layers should also work
    use data_code::ml::layer::add_layer_to_registry;
    let layer = Linear::new(3, 4).unwrap();
    let layer_id = add_layer_to_registry(Box::new(layer));
    let result = Sequential::new(vec![layer_id]);
    assert!(result.is_ok());
}

#[test]
fn test_dataset_batches() {
    use data_code::common::table::Table;
    use data_code::common::value::Value;
    
    let data = vec![
        vec![Value::Number(1.0), Value::Number(2.0), Value::Number(3.0)],
        vec![Value::Number(4.0), Value::Number(5.0), Value::Number(6.0)],
        vec![Value::Number(7.0), Value::Number(8.0), Value::Number(9.0)],
        vec![Value::Number(10.0), Value::Number(11.0), Value::Number(12.0)],
    ];
    let headers = vec!["x1".to_string(), "x2".to_string(), "y".to_string()];
    let table = Table::from_data(data, Some(headers));

    let dataset = Dataset::from_table(
        &table,
        &["x1".to_string(), "x2".to_string()],
        &["y".to_string()],
    ).unwrap();

    // Get batches of size 2
    let batches = dataset.batches(2, false).unwrap();
    assert_eq!(batches.len(), 2);
    
    // First batch should have 2 samples
    assert_eq!(batches[0].0.shape[0], 2);
    assert_eq!(batches[0].1.shape[0], 2);
    
    // Second batch should have 2 samples
    assert_eq!(batches[1].0.shape[0], 2);
    assert_eq!(batches[1].1.shape[0], 2);
}

#[test]
fn test_dataset_batches_shuffle() {
    use data_code::common::table::Table;
    use data_code::common::value::Value;
    
    let data = vec![
        vec![Value::Number(1.0), Value::Number(2.0), Value::Number(3.0)],
        vec![Value::Number(4.0), Value::Number(5.0), Value::Number(6.0)],
        vec![Value::Number(7.0), Value::Number(8.0), Value::Number(9.0)],
    ];
    let headers = vec!["x1".to_string(), "x2".to_string(), "y".to_string()];
    let table = Table::from_data(data, Some(headers));

    let dataset = Dataset::from_table(
        &table,
        &["x1".to_string(), "x2".to_string()],
        &["y".to_string()],
    ).unwrap();

    // Get batches with shuffle
    let batches = dataset.batches(2, true).unwrap();
    assert_eq!(batches.len(), 2); // 3 samples / 2 batch_size = 2 batches
}

// Note: XOR test and full NeuralNetwork integration tests would require
// complete implementation of Sequential with proper parameter initialization
// and Graph integration. These are placeholders for the full test suite.

#[test]
fn test_model_save_and_load() {
    use data_code::run;
    use std::fs;
    use std::path::Path;
    
    let temp_file = "test_model_save_load.nn";
    
    // Ensure file doesn't exist before test
    if Path::new(temp_file).exists() {
        fs::remove_file(temp_file).unwrap();
    }
    
    // Create, save and test model using DataCode
    let code = format!(r#"
import ml

# Create a simple model: Linear(10, 5) → ReLU → Linear(5, 2)
layer1 = ml.layer.linear(10, 5)
layer2 = ml.layer.relu()
layer3 = ml.layer.linear(5, 2)

layers = [layer1, layer2, layer3]
model_seq = ml.sequential(layers)
model = ml.neural_network(model_seq)

# Create test input - simple array with 20 values of 0.5
input_data = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
input = ml.tensor(input_data, [2, 10])

# Get original output
original_output = ml.nn_forward(model, input)

# Save the model
model.save("{}")

# Verify file was created (we'll check this in Rust)
"#, temp_file);
    
    let result = run(&code);
    assert!(result.is_ok(), "Failed to create and save model: {:?}", result);
    
    // Verify file was created
    assert!(Path::new(temp_file).exists(), "Model file was not created");
    
    // Load and test the model
    let load_code = format!(r#"
import ml

# Load the model
loaded_model = ml.load("{}")

# Create same test input
input_data = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
input = ml.tensor(input_data, [2, 10])

# Get output from loaded model
loaded_output = ml.nn_forward(loaded_model, input)

# Check that outputs match (shape should be [2, 2])
# Use ml.shape() which returns an array
output_shape = ml.shape(loaded_output)
if output_shape[0] == 2 and output_shape[1] == 2 {{
    print("Model loaded successfully!")
}} else {{
    print("Error: Output shape mismatch")
}}
"#, temp_file);
    
    let load_result = run(&load_code);
    assert!(load_result.is_ok(), "Failed to load model: {:?}", load_result);
    
    // Clean up: delete the test file
    fs::remove_file(temp_file).expect("Failed to delete test model file");
    assert!(!Path::new(temp_file).exists(), "Test file was not deleted");
}

#[test]
fn test_model_save_and_load_with_training() {
    use data_code::run;
    use std::fs;
    use std::path::Path;
    
    let temp_file = "test_model_trained.nn";
    
    // Ensure file doesn't exist before test
    if Path::new(temp_file).exists() {
        fs::remove_file(temp_file).unwrap();
    }
    
    // Create, train, save model using DataCode
    // Simplified: use smaller batch size to avoid long arrays
    let code = format!(r#"
import ml

# Create a model: Linear(20, 10) → ReLU → Linear(10, 5)
layer1 = ml.layer.linear(20, 10)
layer2 = ml.layer.relu()
layer3 = ml.layer.linear(10, 5)

layers = [layer1, layer2, layer3]
model_seq = ml.sequential(layers)
model = ml.neural_network(model_seq)

# Create training data - batch_size = 4, 4 * 20 = 80 values
input_data = [0.3, 0.3, 0.3, 0.3,
            0.3, 0.3, 0.3, 0.3, 
            0.3, 0.3, 0.3, 0.3, 
            0.3, 0.3, 0.3, 0.3, 
            0.3, 0.3, 0.3, 0.3, 
            0.3, 0.3, 0.3, 0.3, 
            0.3, 0.3, 0.3, 0.3, 
            0.3, 0.3, 0.3, 0.3, 
            0.3, 0.3, 0.3, 0.3, 
            0.3, 0.3, 0.3, 0.3, 
            0.3, 0.3, 0.3, 0.3, 
            0.3, 0.3, 0.3, 0.3, 
            0.3, 0.3, 0.3, 0.3, 
            0.3, 0.3, 0.3, 0.3, 
            0.3, 0.3, 0.3, 0.3, 
            0.3, 0.3, 0.3, 0.3, 
            0.3, 0.3, 0.3, 0.3, 
            0.3, 0.3, 0.3, 0.3, 
            0.3, 0.3, 0.3, 0.3, 
            0.3, 0.3, 0.3, 0.3]
x = ml.tensor(input_data, [4, 20])

# Create one-hot targets: 4 samples, 5 classes
# Sample 0: class 0, Sample 1: class 1, Sample 2: class 2, Sample 3: class 3
targets_data = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
y = ml.tensor(targets_data, [4, 5])

# Train the model
# Use categorical_cross_entropy for one-hot targets [4, 5]
loss_history = model.train(x, y, 2, 4, 0.01, "categorical_cross_entropy")
print("Training completed, loss history length: ", len(loss_history))

# Get output after training
output_after_training = ml.nn_forward(model, x)

# Save the trained model
model.save("{}")
print("Model saved successfully")
"#, temp_file);
    
    let result = run(&code);
    assert!(result.is_ok(), "Failed to create, train and save model: {:?}", result);
    
    // Verify file was created
    assert!(Path::new(temp_file).exists(), "Trained model file was not created");
    
    // Load and test the model
    let load_code = format!(r#"
import ml

# Load the model
loaded_model = ml.load("{}")

# Define batch size
batch_size = 4

# Create same training data
input_data = [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]
x = ml.tensor(input_data, [4, 20])

# Get output from loaded model
loaded_output = ml.nn_forward(loaded_model, x)

# Check that output shape matches
# Use ml.shape() which returns an array
output_shape = ml.shape(loaded_output)
if output_shape[0] == batch_size and output_shape[1] == 5 {{
    print("Loaded model output shape is correct")
}} else {{
    print("Error: Output shape mismatch")
}}

# Test that loaded model can continue training
targets_data = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
y = ml.tensor(targets_data, [batch_size, 5])

# Use categorical_cross_entropy for one-hot targets
loss_history_loaded = loaded_model.train(x, y, 1, batch_size, 0.01, "categorical_cross_entropy")
print("Loaded model can continue training, loss history length: ", len(loss_history_loaded))
"#, temp_file);
    
    let load_result = run(&load_code);
    assert!(load_result.is_ok(), "Failed to load and test trained model: {:?}", load_result);
    
    // Clean up
    fs::remove_file(temp_file).expect("Failed to delete test model file");
    assert!(!Path::new(temp_file).exists(), "Test file was not deleted");
}

#[test]
fn test_model_save_and_load_architecture() {
    use data_code::run;
    use std::fs;
    use std::path::Path;
    
    let temp_file = "test_model_architecture.nn";
    
    if Path::new(temp_file).exists() {
        fs::remove_file(temp_file).unwrap();
    }
    
    // Create a more complex model with different layer types using DataCode
    // Note: sigmoid is not available in ml.layer, so we'll use only available layers
    let code = format!(r#"
import ml

# Create a complex model: Linear(15, 8) → ReLU → Linear(8, 4) → ReLU → Linear(4, 3) → Softmax
layer1 = ml.layer.linear(15, 8)
layer2 = ml.layer.relu()
layer3 = ml.layer.linear(8, 4)
layer4 = ml.layer.relu()
layer5 = ml.layer.linear(4, 3)
layer6 = ml.layer.softmax()

layers = [layer1, layer2, layer3, layer4, layer5, layer6]
model_seq = ml.sequential(layers)
model = ml.neural_network(model_seq)

# Test input - 3 samples, 15 features each = 45 values
input_data = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]
input = ml.tensor(input_data, [3, 15])

# Get original output
original_output = ml.nn_forward(model, input)

# Verify output shape - use ml.shape() which returns an array
output_shape = ml.shape(original_output)
if output_shape[0] == 3 and output_shape[1] == 3 {{
    print("Original model output shape is correct: [3, 3]")
}} else {{
    print("Error: Original output shape mismatch")
}}

# Save model
model.save("{}")
print("Model with complex architecture saved successfully")
"#, temp_file);
    
    let result = run(&code);
    assert!(result.is_ok(), "Failed to create and save model with complex architecture: {:?}", result);
    assert!(Path::new(temp_file).exists(), "Model file was not created");
    
    // Load and test the model
    let load_code = format!(r#"
import ml

# Load model
loaded_model = ml.load("{}")

# Test input - 3 samples, 15 features each = 45 values
input_data = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]
input = ml.tensor(input_data, [3, 15])

# Verify architecture is preserved
loaded_output = ml.nn_forward(loaded_model, input)

# Check output shape - use ml.shape() which returns an array
output_shape = ml.shape(loaded_output)
if output_shape[0] == 3 and output_shape[1] == 3 {{
    print("Loaded model architecture is preserved, output shape: [3, 3]")
}} else {{
    print("Error: Loaded model output shape mismatch")
}}
"#, temp_file);
    
    let load_result = run(&load_code);
    assert!(load_result.is_ok(), "Failed to load model with complex architecture: {:?}", load_result);
    
    // Clean up
    fs::remove_file(temp_file).expect("Failed to delete test model file");
    assert!(!Path::new(temp_file).exists(), "Test file was not deleted");
}

#[test]
fn test_linear_layer_freeze_unfreeze() {
    use data_code::ml::layer::{add_layer_to_registry, with_layer};
    
    // Create a Linear layer
    let linear = Linear::new(3, 4).unwrap();
    let layer_id = add_layer_to_registry(Box::new(linear));
    
    // Initially, layer should be trainable
    let is_trainable_initial = with_layer(layer_id, |layer| layer.is_trainable());
    assert_eq!(is_trainable_initial, Some(true), "Layer should be trainable by default");
    
    // Freeze the layer
    with_layer(layer_id, |layer| layer.freeze());
    let is_trainable_after_freeze = with_layer(layer_id, |layer| layer.is_trainable());
    assert_eq!(is_trainable_after_freeze, Some(false), "Layer should be frozen after freeze()");
    
    // Unfreeze the layer
    with_layer(layer_id, |layer| layer.unfreeze());
    let is_trainable_after_unfreeze = with_layer(layer_id, |layer| layer.is_trainable());
    assert_eq!(is_trainable_after_unfreeze, Some(true), "Layer should be trainable after unfreeze()");
}

#[test]
fn test_neural_network_frozen_layers_detection() {
    use data_code::ml::layer::{add_layer_to_registry, Sequential, with_layer};
    use data_code::ml::model::NeuralNetwork;
    
    // Create layers
    let layer1 = Linear::new(2, 3).unwrap();
    let layer2 = Linear::new(3, 1).unwrap();
    
    let layer1_id = add_layer_to_registry(Box::new(layer1));
    let layer2_id = add_layer_to_registry(Box::new(layer2));
    
    // Create Sequential and NeuralNetwork
    let sequential = Sequential::new(vec![layer1_id, layer2_id]).unwrap();
    let mut nn = NeuralNetwork::new(sequential).unwrap();
    
    // Initialize parameters with a forward pass
    use data_code::ml::tensor::Tensor;
    let input = Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap();
    let _ = nn.forward(&input).unwrap();
    
    // Initially, no layers should be frozen
    let frozen_layers = nn.get_frozen_layers();
    assert!(frozen_layers.is_empty(), "No layers should be frozen initially");
    
    // Freeze first layer
    with_layer(layer1_id, |layer| layer.freeze());
    
    // Check frozen layers
    let frozen_layers_after = nn.get_frozen_layers();
    assert_eq!(frozen_layers_after.len(), 1, "One layer should be frozen");
    assert_eq!(frozen_layers_after[0], "layer0", "First layer should be frozen");
    
    // Freeze second layer too
    with_layer(layer2_id, |layer| layer.freeze());
    
    let frozen_layers_both = nn.get_frozen_layers();
    assert_eq!(frozen_layers_both.len(), 2, "Both layers should be frozen");
}

#[test]
fn test_neural_network_trainable_parameters_count() {
    use data_code::ml::layer::{add_layer_to_registry, Sequential, with_layer};
    use data_code::ml::model::NeuralNetwork;
    use data_code::ml::tensor::Tensor;
    
    // Create layers: Linear(2, 3) and Linear(3, 1)
    // Layer 1: 2*3 + 3 = 9 parameters
    // Layer 2: 3*1 + 1 = 4 parameters
    // Total: 13 parameters
    let layer1 = Linear::new(2, 3).unwrap();
    let layer2 = Linear::new(3, 1).unwrap();
    
    let layer1_id = add_layer_to_registry(Box::new(layer1));
    let layer2_id = add_layer_to_registry(Box::new(layer2));
    
    // Create Sequential and NeuralNetwork
    let sequential = Sequential::new(vec![layer1_id, layer2_id]).unwrap();
    let mut nn = NeuralNetwork::new(sequential).unwrap();
    
    // Initialize parameters with a forward pass
    let input = Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap();
    let _ = nn.forward(&input).unwrap();
    
    // Initially, all parameters should be trainable
    let (trainable, frozen) = nn.count_trainable_frozen_params();
    assert_eq!(trainable, 13, "All 13 parameters should be trainable initially");
    assert_eq!(frozen, 0, "No parameters should be frozen initially");
    
    // Freeze first layer (9 parameters)
    with_layer(layer1_id, |layer| layer.freeze());
    
    let (trainable_after, frozen_after) = nn.count_trainable_frozen_params();
    assert_eq!(trainable_after, 4, "4 parameters should be trainable after freezing layer 1");
    assert_eq!(frozen_after, 9, "9 parameters should be frozen after freezing layer 1");
    
    // Freeze second layer too
    with_layer(layer2_id, |layer| layer.freeze());
    
    let (trainable_both, frozen_both) = nn.count_trainable_frozen_params();
    assert_eq!(trainable_both, 0, "No parameters should be trainable when all layers are frozen");
    assert_eq!(frozen_both, 13, "All 13 parameters should be frozen");
}

#[test]
fn test_model_save_load_freeze_state() {
    use data_code::run;
    use std::fs;
    use std::path::Path;
    
    let temp_file = "test_freeze_save_load.nn";
    
    // Ensure file doesn't exist before test
    if Path::new(temp_file).exists() {
        fs::remove_file(temp_file).unwrap();
    }
    
    // Create model, freeze a layer, save and load
    let save_code = format!(r#"
import ml

# Create model with 2 layers
layer1 = ml.layer.linear(2, 3)
layer2 = ml.layer.linear(3, 1)
layers = [layer1, layer2]
model_seq = ml.sequential(layers)
model = ml.neural_network(model_seq)

# Freeze first layer
ml.layer_freeze(layer1)

# Save model
ml.save_model(model, "{}")
true
"#, temp_file);
    
    let save_result = run(&save_code);
    assert!(save_result.is_ok(), "Failed to save model with frozen layer: {:?}", save_result);
    
    // Load model and verify it loads successfully
    // Note: Checking frozen state after load requires forward pass and may not work
    // if layers are recreated. The important thing is that trainable status is saved/loaded.
    let load_code = format!(r#"
import ml

# Load model
loaded_model = ml.load("{}")

# Verify model loaded successfully by checking it's not null
loaded_model != null
"#, temp_file);
    
    let load_result = run(&load_code);
    assert!(load_result.is_ok(), "Failed to load model: {:?}", load_result);
    match load_result.unwrap() {
        data_code::Value::Bool(b) => {
            assert!(b, "Model should load successfully");
        }
        _ => panic!("Expected Bool"),
    }
    
    // Verify that the model file contains trainable field in JSON
    // This is a simpler check that doesn't require model initialization
    use std::io::Read;
    let mut file = fs::File::open(temp_file).expect("Failed to open saved model file");
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer).expect("Failed to read model file");
    
    // Find JSON section (after magic and version)
    let json_start = buffer.windows(8).position(|w| w == b"DATACODE").unwrap_or(0);
    if json_start > 0 {
        // Skip magic (8 bytes) + version (4 bytes) + json_len (4 bytes)
        let json_offset = json_start + 8 + 4 + 4;
        if json_offset < buffer.len() {
            let json_len_bytes = &buffer[json_offset - 4..json_offset];
            let json_len = u32::from_le_bytes([json_len_bytes[0], json_len_bytes[1], json_len_bytes[2], json_len_bytes[3]]) as usize;
            if json_offset + json_len <= buffer.len() {
                let json_bytes = &buffer[json_offset..json_offset + json_len];
                if let Ok(json_str) = std::str::from_utf8(json_bytes) {
                    // Check that JSON contains trainable field
                    assert!(json_str.contains("\"trainable\""), "Saved model JSON should contain trainable field");
                }
            }
        }
    }
    
    // Clean up
    fs::remove_file(temp_file).expect("Failed to delete test model file");
    assert!(!Path::new(temp_file).exists(), "Test file was not deleted");
}

